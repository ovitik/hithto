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
    parser.add_argument("--variant", choices=["fixed", "soft", "hard", "recursive"], default="fixed")
    parser.add_argument("--mode", choices=["replacement", "additive"], default="replacement")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["model_name"] = args.model_name
    cfg["model"]["dtype"] = args.dtype
    cfg["model"]["freeze_backbone"] = True
    cfg["model"]["unfreeze_top_layers"] = 0
    hidden_size = infer_model_hidden_size(args.model_name, bool(cfg["model"].get("trust_remote_code", True)))

    builder = make_builder(args.variant, args.mode, hidden_size)
    start_load = perf_counter()
    wrapper = GemmaThoughtWrapper.from_pretrained(gemma_load_config_from_dict(cfg["model"]), thought_builder=builder)
    wrapper.model.eval()
    wrapper.thought_builder.train()
    param = next(wrapper.model.parameters())

    rows = [ex.to_json() for ex in CzechWorldGenerator(11).generate("train", args.n)]
    batch = collate_for_causal_lm(rows, wrapper.tokenizer, args.max_length)
    batch = {k: v.to(param.device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(wrapper.thought_builder.parameters(), lr=args.lr)
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
        grad_norm = torch.nn.utils.clip_grad_norm_(wrapper.thought_builder.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        thought_output = out["thought_output"]
        history.append(
            {
                "step": step,
                "seconds": perf_counter() - start,
                "total": float(losses["total"].detach().cpu()),
                "answer": float(losses["answer"].detach().cpu()),
                "distillation": float(losses.get("distillation", torch.tensor(0.0)).detach().cpu()),
                "compression": float(losses.get("compression", torch.tensor(0.0)).detach().cpu()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "thought_tokens": len(thought_output.all_thoughts),
                "final_tokens": int(thought_output.attention_mask.sum().item()),
                "compression_ratio": float(thought_output.compression_ratios[-1]),
            }
        )

    report: dict[str, object] = {
        "model_name": args.model_name,
        "variant": args.variant,
        "mode": args.mode,
        "dtype": str(param.dtype),
        "device": str(param.device),
        "hidden_size": hidden_size,
        "load_seconds": perf_counter() - start_load,
        "steps": args.steps,
        "n_examples": args.n,
        "history": history,
        "loss_delta_total": history[-1]["total"] - history[0]["total"] if len(history) > 1 else 0.0,
        "loss_delta_answer": history[-1]["answer"] - history[0]["answer"] if len(history) > 1 else 0.0,
    }
    out_path = Path(
        args.out
        or f"reports/tiny_train_{args.model_name.replace('/', '_')}_{args.variant}_{args.mode}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_builder(variant: str, mode: str, hidden_size: int) -> ThoughtBuilder:
    if variant == "recursive":
        grouping = "fixed"
        max_levels = 2
    else:
        grouping = variant
        max_levels = 1
    return ThoughtBuilder(
        ThoughtBuilderConfig(
            hidden_size=hidden_size,
            mode=mode,
            grouping=grouping,
            fixed_window=4,
            max_levels=max_levels,
            min_group_size=2,
            max_group_size=8,
            gumbel_temperature=0.7,
        )
    )


if __name__ == "__main__":
    main()
