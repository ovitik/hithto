from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import qwen_hacot_pilot as pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out-dir", default="/workspace/hacot_runs/qwen_hacot_pilot")
    parser.add_argument("--dev-n", type=int, default=120)
    parser.add_argument("--eval-n", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", default="flat,hacot")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    ns = argparse.Namespace(
        model_name=args.model_name,
        out_dir=args.out_dir,
        train_n=0,
        dev_n=args.dev_n,
        steps=0,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        eval_n=args.eval_n,
        lr=2e-4,
        seed=args.seed,
        log_every=25,
        variants=args.variants,
        lora=True,
        qlora_4bit=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=False,
        save_adapters=True,
        dry_run=False,
    )
    pilot.set_seed(args.seed)
    dev = pilot.generate_examples(args.dev_n, "dev", args.seed + 10_000)
    out_dir = Path(args.out_dir)
    summary = {
        "args": vars(ns),
        "variants": {},
        "eval_only": True,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    rows = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        print(json.dumps({"event": "eval_start", "variant": variant}), flush=True)
        metrics = pilot.evaluate_variant(ns, variant, dev, out_dir)
        summary["variants"][variant] = {
            "accuracy": metrics["accuracy"],
            "eval_seconds": metrics["eval_seconds"],
        }
        rows.extend(metrics["rows"])
        pilot.write_json(out_dir / "summary_eval_only_partial.json", summary)
    summary["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pilot.write_json(out_dir / "summary_eval_only.json", summary)
    pilot.write_json(out_dir / "per_example_results_eval_only.json", rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
