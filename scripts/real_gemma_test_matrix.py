from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from thought_tokens.datasets import collate_for_causal_lm
from thought_tokens.gemma_wrapper import GemmaThoughtWrapper
from thought_tokens.hierarchy import ThoughtNode, unit_embedding, unit_per_layer_embedding
from thought_tokens.losses import LossWeights, total_loss
from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderConfig, ThoughtBuilderOutput
from thought_tokens.training import gemma_load_config_from_dict, infer_model_hidden_size, load_config
from thought_tokens.world_generator import CzechWorldGenerator


def make_builder(kind: str, hidden_size: int, dtype: torch.dtype, device: torch.device) -> ThoughtBuilder | None:
    if kind == "baseline":
        return None
    cfg = ThoughtBuilderConfig(
        hidden_size=hidden_size,
        mode="replacement",
        grouping="fixed" if kind in {"fixed", "recursive"} else kind,
        fixed_window=4,
        max_levels=2 if kind == "recursive" else 1,
        min_group_size=2,
        max_group_size=8,
        gumbel_temperature=0.7,
    )
    builder = ThoughtBuilder(cfg).to(device=device, dtype=dtype)
    builder.eval()
    return builder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_level.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--out", default="reports/real_gemma_test_matrix.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_name:
        cfg["model"]["model_name"] = args.model_name
    hidden_size = infer_model_hidden_size(
        cfg["model"]["model_name"],
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
    )
    rows = [ex.to_json() for ex in CzechWorldGenerator(7).generate("train", args.n)]

    start_load = perf_counter()
    wrapper = GemmaThoughtWrapper.from_pretrained(gemma_load_config_from_dict(cfg["model"]), thought_builder=None)
    wrapper.eval()
    param = next(wrapper.model.parameters())
    report: dict[str, object] = {
        "model_name": cfg["model"]["model_name"],
        "model_class": type(wrapper.model).__name__,
        "dtype": str(param.dtype),
        "device": str(param.device),
        "cuda": torch.cuda.is_available(),
        "hidden_size": hidden_size,
        "n_examples": args.n,
        "max_length": args.max_length,
        "load_seconds": perf_counter() - start_load,
        "variants": [],
    }
    batch = collate_for_causal_lm(rows, wrapper.tokenizer, args.max_length)
    batch = {k: v.to(param.device) for k, v in batch.items()}

    variants: list[dict[str, object]] = []
    for kind in ["baseline", "fixed", "soft", "hard", "recursive"]:
        wrapper.thought_builder = make_builder(kind, hidden_size, param.dtype, param.device)
        with torch.no_grad():
            start = perf_counter()
            out = wrapper(**batch, use_thoughts=wrapper.thought_builder is not None)
            elapsed = perf_counter() - start
            model_output = out["model_output"]
            thought_output = out.get("thought_output")
            losses = {"answer": float(model_output.loss.detach().cpu())}
            if thought_output is not None:
                all_losses = total_loss(
                    model_output,
                    LossWeights(alpha_distillation=0.1, delta_compression=0.02),
                    teacher_output=out.get("teacher_output"),
                    thought_output=thought_output,
                )
                losses = {k: float(v.detach().cpu()) for k, v in all_losses.items()}
            item: dict[str, object] = {
                "variant": kind,
                "seconds": elapsed,
                "losses": losses,
                "thought_tokens": 0 if thought_output is None else len(thought_output.all_thoughts),
                "compression_ratios": [] if thought_output is None else thought_output.compression_ratios,
                "output_shape": list(model_output.logits.shape),
            }
            if thought_output is not None and thought_output.all_thoughts:
                item["interventions"] = run_interventions(wrapper, thought_output, model_output.logits)
            variants.append(item)

    report["variants"] = variants
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_interventions(
    wrapper: GemmaThoughtWrapper,
    thought_output: ThoughtBuilderOutput,
    original_logits: torch.Tensor,
) -> dict[str, object]:
    node = select_final_intervention_node(thought_output)
    results: dict[str, float] = {}
    for name, embeds, per_layer in [
        ("drop", *materialize_intervention(thought_output, drop_node=node)),
        ("random_replace", *materialize_intervention(thought_output, replace_node=node)),
        ("expand", *materialize_intervention(thought_output, expand_node=node)),
    ]:
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        kwargs = {"inputs_embeds": embeds, "attention_mask": mask, "use_cache": False}
        if per_layer is not None:
            kwargs["per_layer_inputs"] = per_layer
        logits = wrapper.model(**kwargs).logits
        min_len = min(logits.shape[1], original_logits.shape[1])
        results[name] = float((logits[:, :min_len] - original_logits[:, :min_len]).abs().mean().cpu())
    return {
        "target_level": node.level,
        "target_source_positions": node.source_positions,
        "deltas": results,
    }


def select_final_intervention_node(thought_output: ThoughtBuilderOutput) -> ThoughtNode:
    final_nodes = [
        unit
        for sample in thought_output.units
        for unit in sample
        if isinstance(unit, ThoughtNode)
    ]
    if final_nodes:
        return max(final_nodes, key=lambda node: (node.level, len(node.source_positions)))
    return max(thought_output.all_thoughts, key=lambda node: (node.level, len(node.source_positions)))


def materialize_intervention(
    thought_output: ThoughtBuilderOutput,
    drop_node: ThoughtNode | None = None,
    replace_node: ThoughtNode | None = None,
    expand_node: ThoughtNode | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    embed_rows: list[torch.Tensor] = []
    per_layer_rows: list[torch.Tensor] = []
    has_per_layer = thought_output.per_layer_inputs is not None
    for units in thought_output.units:
        embeds: list[torch.Tensor] = []
        per_layers: list[torch.Tensor] = []
        for unit in units:
            if unit is drop_node:
                continue
            if unit is expand_node:
                for child in unit.children:
                    embeds.append(unit_embedding(child))
                    if has_per_layer:
                        per_layers.append(unit_per_layer_embedding(child))  # type: ignore[arg-type]
                continue
            if unit is replace_node:
                generator = torch.Generator(device=unit.embedding.device).manual_seed(123)
                embeds.append(torch.randn(unit.embedding.shape, generator=generator, device=unit.embedding.device, dtype=unit.embedding.dtype))
                if has_per_layer:
                    ple = unit_per_layer_embedding(unit)
                    per_layers.append(torch.randn(ple.shape, generator=generator, device=ple.device, dtype=ple.dtype))  # type: ignore[union-attr]
                continue
            embeds.append(unit_embedding(unit))
            if has_per_layer:
                per_layers.append(unit_per_layer_embedding(unit))  # type: ignore[arg-type]
        embed_rows.append(torch.stack(embeds, dim=0))
        if has_per_layer:
            per_layer_rows.append(torch.stack(per_layers, dim=0))
    return pad(embed_rows), pad(per_layer_rows) if has_per_layer else None


def pad(rows: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(row.shape[0] for row in rows)
    out = rows[0].new_zeros(len(rows), max_len, *rows[0].shape[1:])
    for i, row in enumerate(rows):
        out[i, : row.shape[0]] = row
    return out


if __name__ == "__main__":
    main()
