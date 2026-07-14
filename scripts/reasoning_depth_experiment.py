from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from thought_tokens.datasets import format_prompt
from thought_tokens.gemma_wrapper import GemmaThoughtWrapper, freeze_model
from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderConfig
from thought_tokens.training import gemma_load_config_from_dict, infer_model_hidden_size, load_config
from thought_tokens.world_generator import CzechWorldGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_level.yaml")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--variants", nargs="+", default=["baseline", "fixed", "recursive", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--train-n", type=int, default=6)
    parser.add_argument("--eval-n", type=int, default=6)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--mode", choices=["replacement", "additive"], default="replacement")
    parser.add_argument("--max-prompt-length", type=int, default=112)
    parser.add_argument("--max-answer-length", type=int, default=16)
    parser.add_argument("--fixed-window", type=int, default=4)
    parser.add_argument("--recursive-window", type=int, default=2)
    parser.add_argument("--random-latents", type=int, default=42)
    parser.add_argument("--out", default="reports/reasoning_depth_experiment.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["model_name"] = args.model_name
    cfg["model"]["dtype"] = args.dtype
    cfg["model"]["freeze_backbone"] = True
    hidden_size = infer_model_hidden_size(args.model_name, bool(cfg["model"].get("trust_remote_code", True)))

    load_start = perf_counter()
    wrapper = GemmaThoughtWrapper.from_pretrained(gemma_load_config_from_dict(cfg["model"]), thought_builder=None)
    wrapper.model.eval()
    freeze_model(wrapper.model)
    param = next(wrapper.model.parameters())

    report: dict[str, Any] = {
        "model_name": args.model_name,
        "dtype": str(param.dtype),
        "device": str(param.device),
        "hidden_size": hidden_size,
        "load_seconds": perf_counter() - load_start,
        "train_n": args.train_n,
        "eval_n": args.eval_n,
        "steps": args.steps,
        "mode": args.mode,
        "variants": args.variants,
        "seeds": args.seeds,
        "runs": [],
    }

    for seed in args.seeds:
        train_rows = generate_rows("train", args.train_n, seed)
        iid_rows = generate_rows("validation", args.eval_n, seed + 100)
        ood_rows = generate_rows("ood_depth", args.eval_n, seed + 200)
        for variant in args.variants:
            torch.manual_seed(seed)
            builder = make_builder(variant, hidden_size, args.fixed_window, args.recursive_window, args.mode)
            if builder is not None:
                builder.to(device=param.device, dtype=param.dtype)
            optimizer = None if builder is None else torch.optim.AdamW(builder.parameters(), lr=args.lr)
            history: list[dict[str, float]] = []
            if optimizer is not None:
                builder.train()
                for step in range(args.steps):
                    start = perf_counter()
                    loss, stats = batch_answer_loss(
                        wrapper,
                        train_rows,
                        builder,
                        args.max_prompt_length,
                        args.max_answer_length,
                    )
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(builder.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    history.append(
                        {
                            "step": step,
                            "seconds": perf_counter() - start,
                            "loss": float(loss.detach().cpu()),
                            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                            **stats,
                        }
                    )
            if builder is not None:
                builder.eval()
            iid = evaluate_forced_choice(
                wrapper,
                iid_rows,
                builder,
                args.max_prompt_length,
                args.max_answer_length,
                random_latents=args.random_latents if variant == "random" else 0,
                seed=seed,
            )
            ood = evaluate_forced_choice(
                wrapper,
                ood_rows,
                builder,
                args.max_prompt_length,
                args.max_answer_length,
                random_latents=args.random_latents if variant == "random" else 0,
                seed=seed,
            )
            report["runs"].append(
                {
                    "seed": seed,
                    "variant": variant,
                    "train_history": history,
                    "iid": iid,
                    "ood_depth": ood,
                }
            )

    report["summary"] = summarize(report["runs"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_builder(
    variant: str,
    hidden_size: int,
    fixed_window: int,
    recursive_window: int,
    mode: str,
) -> ThoughtBuilder | None:
    if variant in {"baseline", "random"}:
        return None
    if variant == "fixed":
        grouping = "fixed"
        window = fixed_window
        levels = 1
    elif variant == "recursive":
        grouping = "fixed"
        window = recursive_window
        levels = 2
    else:
        grouping = variant
        window = fixed_window
        levels = 1
    return ThoughtBuilder(
        ThoughtBuilderConfig(
            hidden_size=hidden_size,
            mode=mode,
            grouping=grouping,
            fixed_window=window,
            max_levels=levels,
            min_group_size=2,
            max_group_size=8,
            gumbel_temperature=0.7,
        )
    )


def generate_rows(split: str, n: int, seed: int) -> list[dict[str, Any]]:
    generator = CzechWorldGenerator(seed)
    rows: list[dict[str, Any]] = []
    # Keep the first pass focused on the two implemented compositional task families.
    for ex in generator.generate(split, n * 4):  # type: ignore[arg-type]
        if ex.task_type in {"nesting", "temporal"}:
            rows.append(ex.to_json())
        if len(rows) >= n:
            break
    return rows


def answer_candidates(row: dict[str, Any]) -> list[str]:
    if row["task_type"] == "nesting":
        generator = CzechWorldGenerator(0)
        values = generator.containers + generator.places
    elif row["task_type"] == "temporal":
        values = [
            "zapnul počítač",
            "odeslal zprávu",
            "zavolal na nádraží",
            "vyzvedl balík",
            "zamkl dveře",
            "odešel do práce",
            "zkontroloval kalendář",
            "připravil snídani",
        ]
    else:
        values = [row["answer"], "ano", "ne"]
    out = [row["answer"]]
    for value in values:
        if value not in out:
            out.append(value)
    return out


def batch_answer_loss(
    wrapper: GemmaThoughtWrapper,
    rows: list[dict[str, Any]],
    builder: ThoughtBuilder | None,
    max_prompt_length: int,
    max_answer_length: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    stats: list[dict[str, float]] = []
    for row in rows:
        loss, stat = score_answer(
            wrapper,
            row,
            row["answer"],
            builder,
            max_prompt_length,
            max_answer_length,
        )
        losses.append(loss)
        stats.append(stat)
    merged = {key: float(sum(s[key] for s in stats) / len(stats)) for key in stats[0]}
    return torch.stack(losses).mean(), merged


def evaluate_forced_choice(
    wrapper: GemmaThoughtWrapper,
    rows: list[dict[str, Any]],
    builder: ThoughtBuilder | None,
    max_prompt_length: int,
    max_answer_length: int,
    random_latents: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    correct = 0
    total_loss = 0.0
    by_depth: dict[str, list[int]] = {}
    predictions = []
    with torch.no_grad():
        for row in rows:
            candidates = answer_candidates(row)
            scores = []
            for candidate in candidates:
                loss, _ = score_answer(
                    wrapper,
                    row,
                    candidate,
                    builder,
                    max_prompt_length,
                    max_answer_length,
                    random_latents=random_latents,
                    seed=seed,
                )
                scores.append(float(loss.cpu()))
            best_idx = min(range(len(scores)), key=scores.__getitem__)
            pred = candidates[best_idx]
            ok = int(pred == row["answer"])
            correct += ok
            total_loss += scores[0]
            by_depth.setdefault(str(row["depth"]), []).append(ok)
            predictions.append(
                {
                    "id": row["id"],
                    "task_type": row["task_type"],
                    "depth": row["depth"],
                    "answer": row["answer"],
                    "prediction": pred,
                    "correct": bool(ok),
                    "gold_loss": scores[0],
                }
            )
    return {
        "accuracy": correct / max(1, len(rows)),
        "gold_loss": total_loss / max(1, len(rows)),
        "by_depth": {depth: sum(vals) / len(vals) for depth, vals in by_depth.items()},
        "predictions": predictions,
    }


def score_answer(
    wrapper: GemmaThoughtWrapper,
    row: dict[str, Any],
    answer: str,
    builder: ThoughtBuilder | None,
    max_prompt_length: int,
    max_answer_length: int,
    random_latents: int = 0,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(wrapper.model.parameters()).device
    tokenizer = wrapper.tokenizer
    prompt = format_prompt(row)
    prompt_batch = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)
    answer_batch = tokenizer(
        " " + answer,
        return_tensors="pt",
        truncation=True,
        max_length=max_answer_length,
        add_special_tokens=False,
    ).to(device)
    prompt_embeds = wrapper.embed_tokens(prompt_batch["input_ids"])
    answer_embeds = wrapper.embed_tokens(answer_batch["input_ids"])
    stats = {"thought_tokens": 0.0, "compression_ratio": 1.0, "context_tokens": float(prompt_embeds.shape[1])}

    if builder is not None:
        teacher = wrapper.model(
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        thought = builder(
            teacher.hidden_states[-1],
            attention_mask=prompt_batch["attention_mask"],
            token_ids=prompt_batch["input_ids"],
        )
        context_embeds = thought.embeddings
        context_mask = thought.attention_mask
        stats = {
            "thought_tokens": float(len(thought.all_thoughts)),
            "compression_ratio": float(thought.compression_ratios[-1]),
            "context_tokens": float(context_mask.sum().item()),
        }
    else:
        context_embeds = prompt_embeds
        context_mask = prompt_batch["attention_mask"]
        if random_latents > 0:
            gen = torch.Generator(device=device).manual_seed(seed)
            latent = torch.randn(
                context_embeds.shape[0],
                random_latents,
                context_embeds.shape[-1],
                generator=gen,
                device=device,
                dtype=context_embeds.dtype,
            ) * context_embeds.std().clamp_min(1e-6)
            context_embeds = torch.cat([context_embeds, latent], dim=1)
            context_mask = torch.cat(
                [context_mask, torch.ones(context_mask.shape[0], random_latents, dtype=context_mask.dtype, device=device)],
                dim=1,
            )
            stats["context_tokens"] = float(context_mask.sum().item())

    inputs_embeds = torch.cat([context_embeds, answer_embeds], dim=1)
    answer_mask = torch.ones(answer_batch["input_ids"].shape, dtype=context_mask.dtype, device=device)
    attention_mask = torch.cat([context_mask, answer_mask], dim=1)
    labels = torch.full(attention_mask.shape, -100, dtype=torch.long, device=device)
    labels[:, context_embeds.shape[1] :] = answer_batch["input_ids"]
    out = wrapper.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = out.loss
    if not torch.isfinite(loss):
        logits = out.logits[:, :-1]
        shifted_labels = labels[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )
    return loss, stats


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in sorted({run["variant"] for run in runs}):
        subset = [run for run in runs if run["variant"] == variant]
        for split in ["iid", "ood_depth"]:
            acc = torch.tensor([float(run[split]["accuracy"]) for run in subset])
            loss = torch.tensor([float(run[split]["gold_loss"]) for run in subset])
            summary.setdefault(variant, {})[f"{split}_accuracy_mean"] = float(acc.mean())
            summary[variant][f"{split}_accuracy_std"] = float(acc.std(unbiased=False))
            summary[variant][f"{split}_gold_loss_mean"] = float(loss.mean())
            summary[variant][f"{split}_gold_loss_std"] = float(loss.std(unbiased=False))
    return summary


if __name__ == "__main__":
    main()
