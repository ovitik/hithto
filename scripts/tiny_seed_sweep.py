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
from thought_tokens.losses import LossWeights, total_loss
from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderConfig
from thought_tokens.training import gemma_load_config_from_dict, infer_model_hidden_size, load_config
from thought_tokens.world_generator import CzechWorldGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_level.yaml")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--variants", nargs="+", default=["fixed", "recursive"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fixed-window", type=int, default=4)
    parser.add_argument("--recursive-window", type=int, default=2)
    parser.add_argument("--out", default="reports/tiny_seed_sweep.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["model_name"] = args.model_name
    cfg["model"]["dtype"] = args.dtype
    cfg["model"]["freeze_backbone"] = True
    cfg["model"]["unfreeze_top_layers"] = 0
    hidden_size = infer_model_hidden_size(args.model_name, bool(cfg["model"].get("trust_remote_code", True)))

    load_start = perf_counter()
    wrapper = GemmaThoughtWrapper.from_pretrained(gemma_load_config_from_dict(cfg["model"]), thought_builder=None)
    wrapper.model.eval()
    param = next(wrapper.model.parameters())

    runs: list[dict[str, object]] = []
    for variant in args.variants:
        for seed in args.seeds:
            torch.manual_seed(seed)
            builder = make_builder(
                variant,
                hidden_size,
                fixed_window=args.recursive_window if variant == "recursive" else args.fixed_window,
            ).to(device=param.device, dtype=param.dtype)
            builder.train()
            wrapper.thought_builder = builder
            optimizer = torch.optim.AdamW(builder.parameters(), lr=args.lr)
            rows = [ex.to_json() for ex in CzechWorldGenerator(seed).generate("train", args.n)]
            batch = collate_for_causal_lm(rows, wrapper.tokenizer, args.max_length)
            batch = {k: v.to(param.device) for k, v in batch.items()}
            history: list[dict[str, float | int]] = []
            for step in range(args.steps):
                start = perf_counter()
                out = wrapper(**batch, use_thoughts=True)
                losses = total_loss(
                    out["model_output"],
                    LossWeights(alpha_distillation=0.02, delta_compression=0.005),
                    teacher_output=out.get("teacher_output"),
                    thought_output=out.get("thought_output"),
                )
                losses["total"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(builder.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                thought_output = out["thought_output"]
                history.append(
                    {
                        "step": step,
                        "seconds": perf_counter() - start,
                        "total": float(losses["total"].detach().cpu()),
                        "answer": float(losses["answer"].detach().cpu()),
                        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                        "thought_tokens": len(thought_output.all_thoughts),
                        "final_tokens": int(thought_output.attention_mask.sum().item()),
                        "compression_ratio": float(thought_output.compression_ratios[-1]),
                    }
                )
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "first_total": history[0]["total"],
                    "last_total": history[-1]["total"],
                    "delta_total": history[-1]["total"] - history[0]["total"],
                    "first_answer": history[0]["answer"],
                    "last_answer": history[-1]["answer"],
                    "delta_answer": history[-1]["answer"] - history[0]["answer"],
                    "compression_ratio": history[-1]["compression_ratio"],
                    "final_tokens": history[-1]["final_tokens"],
                    "history": history,
                }
            )

    summary = summarize(runs)
    report = {
        "model_name": args.model_name,
        "dtype": str(param.dtype),
        "device": str(param.device),
        "hidden_size": hidden_size,
        "load_seconds": perf_counter() - load_start,
        "steps": args.steps,
        "n_examples": args.n,
        "fixed_window": args.fixed_window,
        "recursive_window": args.recursive_window,
        "runs": runs,
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_builder(variant: str, hidden_size: int, fixed_window: int) -> ThoughtBuilder:
    grouping = "fixed" if variant == "recursive" else variant
    return ThoughtBuilder(
        ThoughtBuilderConfig(
            hidden_size=hidden_size,
            mode="replacement",
            grouping=grouping,
            fixed_window=fixed_window,
            max_levels=2 if variant == "recursive" else 1,
            min_group_size=2,
            max_group_size=8,
            gumbel_temperature=0.7,
        )
    )


def summarize(runs: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    variants = sorted({str(run["variant"]) for run in runs})
    for variant in variants:
        subset = [run for run in runs if run["variant"] == variant]
        for key in ["delta_total", "delta_answer", "compression_ratio", "final_tokens"]:
            values = torch.tensor([float(run[key]) for run in subset])
            out.setdefault(variant, {})[f"mean_{key}"] = float(values.mean())
            out[variant][f"std_{key}"] = float(values.std(unbiased=False))
    return out


if __name__ == "__main__":
    main()
