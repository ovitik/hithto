from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import qwen_hacot_pilot as pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out-dir", default="/workspace/hacot_runs/qwen_hacot_diagnostic")
    parser.add_argument("--train-n", type=int, default=64)
    parser.add_argument("--id-n", type=int, default=64)
    parser.add_argument("--eval-split", choices=["train", "dev"], default="dev")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--eval-n", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", default="direct")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--qlora-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ns = argparse.Namespace(
        model_name=args.model_name,
        out_dir=args.out_dir,
        train_n=args.train_n,
        dev_n=args.id_n,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        eval_n=args.eval_n,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        seed=args.seed,
        log_every=20,
        variants=args.variants,
        lora=True,
        qlora_4bit=args.qlora_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=True,
        save_adapters=True,
        dry_run=False,
    )

    pilot.set_seed(args.seed)
    train = pilot.generate_examples(args.train_n, "train", args.seed)
    id_dev = pilot.generate_examples(args.id_n, args.eval_split, args.seed + 12345)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot.write_json(
        out_dir / "diagnostic_data_manifest.json",
        {
            "train_n": len(train),
            "id_n": len(id_dev),
            "train_depths": sorted({x.difficulty for x in train}),
            "eval_split": args.eval_split,
            "id_depths": sorted({x.difficulty for x in id_dev}),
            "families": sorted({x.task_family for x in train + id_dev}),
        },
    )

    summary = {
        "args": vars(ns),
        "diagnostic": True,
        "variants": {},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    all_rows = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        print(json.dumps({"event": "train_start", "variant": variant}), flush=True)
        train_metrics = pilot.train_variant(ns, variant, train, out_dir)
        print(json.dumps({"event": "eval_train_start", "variant": variant}), flush=True)
        train_eval = pilot.evaluate_variant(ns, variant, train, out_dir)
        print(json.dumps({"event": "eval_id_start", "variant": variant}), flush=True)
        id_eval = pilot.evaluate_variant(ns, variant, id_dev, out_dir)
        summary["variants"][variant] = {
            **train_metrics,
            "train_accuracy": train_eval["accuracy"],
            "train_eval_seconds": train_eval["eval_seconds"],
            "id_accuracy": id_eval["accuracy"],
            "id_eval_seconds": id_eval["eval_seconds"],
        }
        for row in train_eval["rows"]:
            row["eval_split"] = "train_overfit"
        for row in id_eval["rows"]:
            row["eval_split"] = "id_depth_1_6"
        all_rows.extend(train_eval["rows"])
        all_rows.extend(id_eval["rows"])
        pilot.write_json(out_dir / "diagnostic_summary_partial.json", summary)
    summary["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pilot.write_json(out_dir / "diagnostic_summary.json", summary)
    pilot.write_json(out_dir / "diagnostic_per_example.json", all_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
