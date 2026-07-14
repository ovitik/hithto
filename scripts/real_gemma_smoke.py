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
from thought_tokens.interventions import drop_thought, replace_thought_with_random
from thought_tokens.losses import LossWeights, total_loss
from thought_tokens.training import (
    gemma_load_config_from_dict,
    load_config,
    make_thought_builder,
    sync_thought_hidden_size,
)
from thought_tokens.world_generator import CzechWorldGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_level.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", default=None, choices=["bf16", "fp32"])
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_name:
        cfg["model"]["model_name"] = args.model_name
    if args.dtype:
        cfg["model"]["dtype"] = args.dtype
    sync_thought_hidden_size(cfg)
    cfg["thought_builder"]["max_levels"] = min(1, int(cfg["thought_builder"].get("max_levels", 1)))

    report: dict[str, object] = {
        "model_name": cfg["model"]["model_name"],
        "dtype": cfg["model"]["dtype"],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "max_length": args.max_length,
    }

    start_load = perf_counter()
    builder = make_thought_builder(cfg)
    wrapper = GemmaThoughtWrapper.from_pretrained(gemma_load_config_from_dict(cfg["model"]), thought_builder=builder)
    wrapper.eval()
    report["load_seconds"] = perf_counter() - start_load
    report["device"] = str(next(wrapper.model.parameters()).device)
    report["model_class"] = type(wrapper.model).__name__

    rows = [ex.to_json() for ex in CzechWorldGenerator(args.seed).generate("train", args.n)]
    batch = collate_for_causal_lm(rows, wrapper.tokenizer, args.max_length)
    batch = {k: v.to(next(wrapper.model.parameters()).device) for k, v in batch.items()}

    with torch.no_grad():
        start = perf_counter()
        baseline = wrapper(**batch, use_thoughts=False)
        report["baseline_forward_seconds"] = perf_counter() - start
        report["baseline_loss"] = float(baseline["model_output"].loss.detach().cpu())

        start = perf_counter()
        thought = wrapper(**batch, use_thoughts=True)
        report["thought_forward_seconds"] = perf_counter() - start
        losses = total_loss(
            thought["model_output"],
            LossWeights(alpha_distillation=0.1, delta_compression=0.02),
            teacher_output=thought.get("teacher_output"),
            thought_output=thought.get("thought_output"),
        )
        report["thought_total_loss"] = float(losses["total"].detach().cpu())
        thought_output = thought["thought_output"]
        report["thought_tokens"] = len(thought_output.all_thoughts)
        report["compression_ratios"] = thought_output.compression_ratios
        report["thought_embedding_shape"] = list(thought_output.embeddings.shape)

        if thought_output.all_thoughts:
            node = thought_output.all_thoughts[0]
            report["drop_shape"] = list(drop_thought(thought_output, node).shape)
            report["random_replace_shape"] = list(replace_thought_with_random(thought_output, node).shape)

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "real_gemma_smoke.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
