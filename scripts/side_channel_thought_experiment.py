from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn

from reasoning_depth_experiment import answer_candidates, generate_rows
from thought_tokens.datasets import format_prompt
from thought_tokens.gemma_wrapper import GemmaThoughtWrapper, freeze_model
from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderConfig, ThoughtBuilderOutput
from thought_tokens.training import gemma_load_config_from_dict, infer_model_hidden_size, load_config


class ThoughtMemoryAdapter(nn.Module):
    def __init__(self, hidden_size: int, n_memory_tokens: int = 4) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_memory_tokens, hidden_size) * 0.02)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, thought_embeddings: torch.Tensor) -> torch.Tensor:
        # thought_embeddings: [batch, n_thoughts, hidden]
        q = self.queries.to(dtype=thought_embeddings.dtype, device=thought_embeddings.device)
        q = q.unsqueeze(0).expand(thought_embeddings.shape[0], -1, -1)
        k = self.key(thought_embeddings)
        v = self.value(thought_embeddings)
        scale = thought_embeddings.shape[-1] ** -0.5
        weights = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * scale, dim=-1)
        memory = torch.matmul(weights, v)
        return self.out(memory)


class RandomMemory(nn.Module):
    def __init__(self, hidden_size: int, n_memory_tokens: int = 4) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.randn(n_memory_tokens, hidden_size) * 0.02)

    def forward(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.memory.to(dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_level.yaml")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--variants", nargs="+", default=["baseline", "random", "fixed", "recursive"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-n", type=int, default=4)
    parser.add_argument("--eval-n", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--memory-tokens", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=112)
    parser.add_argument("--max-answer-length", type=int, default=12)
    parser.add_argument("--fixed-window", type=int, default=4)
    parser.add_argument("--recursive-window", type=int, default=2)
    parser.add_argument("--out", default="reports/side_channel_thought_experiment.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["model_name"] = args.model_name
    cfg["model"]["dtype"] = args.dtype
    cfg["model"]["freeze_backbone"] = True
    hidden_size = infer_model_hidden_size(args.model_name, bool(cfg["model"].get("trust_remote_code", True)))

    torch.manual_seed(args.seed)
    load_start = perf_counter()
    wrapper = GemmaThoughtWrapper.from_pretrained(gemma_load_config_from_dict(cfg["model"]), thought_builder=None)
    wrapper.model.eval()
    freeze_model(wrapper.model)
    param = next(wrapper.model.parameters())
    load_seconds = perf_counter() - load_start
    train_rows = generate_rows("train", args.train_n, args.seed)
    iid_rows = generate_rows("validation", args.eval_n, args.seed + 100)
    ood_rows = generate_rows("ood_depth", args.eval_n, args.seed + 200)

    runs: list[dict[str, Any]] = []
    for variant in args.variants:
        torch.manual_seed(args.seed)
        builder = make_builder(variant, hidden_size, args.fixed_window, args.recursive_window)
        adapter: ThoughtMemoryAdapter | RandomMemory | None
        if variant == "baseline":
            adapter = None
        elif variant == "random":
            adapter = RandomMemory(hidden_size, args.memory_tokens).to(device=param.device, dtype=param.dtype)
        else:
            builder = builder.to(device=param.device, dtype=param.dtype)
            adapter = ThoughtMemoryAdapter(hidden_size, args.memory_tokens).to(device=param.device, dtype=param.dtype)

        trainable = []
        if builder is not None:
            trainable.extend(builder.parameters())
        if adapter is not None:
            trainable.extend(adapter.parameters())
        optimizer = None if not trainable else torch.optim.AdamW(trainable, lr=args.lr)

        history: list[dict[str, float]] = []
        if optimizer is not None:
            for step in range(args.steps):
                start = perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, stats = batch_loss(wrapper, train_rows, builder, adapter, args.max_prompt_length, args.max_answer_length)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                history.append(
                    {
                        "step": step,
                        "seconds": perf_counter() - start,
                        "loss": float(loss.detach().cpu()),
                        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                        **stats,
                    }
                )

        iid = evaluate(wrapper, iid_rows, builder, adapter, args.max_prompt_length, args.max_answer_length)
        ood = evaluate(wrapper, ood_rows, builder, adapter, args.max_prompt_length, args.max_answer_length)
        runs.append({"variant": variant, "train_history": history, "iid": iid, "ood_depth": ood})

    report = {
        "model_name": args.model_name,
        "dtype": str(param.dtype),
        "device": str(param.device),
        "hidden_size": hidden_size,
        "load_seconds": load_seconds,
        "seed": args.seed,
        "train_n": args.train_n,
        "eval_n": args.eval_n,
        "steps": args.steps,
        "memory_tokens": args.memory_tokens,
        "runs": runs,
        "summary": summarize(runs),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_builder(variant: str, hidden_size: int, fixed_window: int, recursive_window: int) -> ThoughtBuilder | None:
    if variant in {"baseline", "random"}:
        return None
    return ThoughtBuilder(
        ThoughtBuilderConfig(
            hidden_size=hidden_size,
            mode="replacement",
            grouping="fixed",
            fixed_window=recursive_window if variant == "recursive" else fixed_window,
            max_levels=2 if variant == "recursive" else 1,
            min_group_size=2,
            max_group_size=8,
        )
    )


def batch_loss(
    wrapper: GemmaThoughtWrapper,
    rows: list[dict[str, Any]],
    builder: ThoughtBuilder | None,
    adapter: ThoughtMemoryAdapter | RandomMemory | None,
    max_prompt_length: int,
    max_answer_length: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    stats = []
    for row in rows:
        loss, stat = score_answer(wrapper, row, row["answer"], builder, adapter, max_prompt_length, max_answer_length)
        losses.append(loss)
        stats.append(stat)
    merged = {key: float(sum(s[key] for s in stats) / len(stats)) for key in stats[0]}
    return torch.stack(losses).mean(), merged


@torch.no_grad()
def evaluate(
    wrapper: GemmaThoughtWrapper,
    rows: list[dict[str, Any]],
    builder: ThoughtBuilder | None,
    adapter: ThoughtMemoryAdapter | RandomMemory | None,
    max_prompt_length: int,
    max_answer_length: int,
) -> dict[str, Any]:
    correct = 0
    gold_loss = 0.0
    predictions = []
    by_depth: dict[str, list[int]] = {}
    for row in rows:
        candidates = answer_candidates(row)
        scores = []
        for candidate in candidates:
            loss, _ = score_answer(wrapper, row, candidate, builder, adapter, max_prompt_length, max_answer_length)
            scores.append(float(loss.cpu()))
        best = min(range(len(scores)), key=scores.__getitem__)
        pred = candidates[best]
        ok = int(pred == row["answer"])
        correct += ok
        gold_loss += scores[0]
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
        "gold_loss": gold_loss / max(1, len(rows)),
        "by_depth": {depth: sum(vals) / len(vals) for depth, vals in by_depth.items()},
        "predictions": predictions,
    }


def score_answer(
    wrapper: GemmaThoughtWrapper,
    row: dict[str, Any],
    answer: str,
    builder: ThoughtBuilder | None,
    adapter: ThoughtMemoryAdapter | RandomMemory | None,
    max_prompt_length: int,
    max_answer_length: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(wrapper.model.parameters()).device
    tokenizer = wrapper.tokenizer
    prompt_batch = tokenizer(format_prompt(row), return_tensors="pt", truncation=True, max_length=max_prompt_length).to(device)
    answer_batch = tokenizer(
        " " + answer,
        return_tensors="pt",
        truncation=True,
        max_length=max_answer_length,
        add_special_tokens=False,
    ).to(device)
    prompt_embeds = wrapper.embed_tokens(prompt_batch["input_ids"])
    context_embeds = prompt_embeds
    context_mask = prompt_batch["attention_mask"]
    stats = {"thought_tokens": 0.0, "context_tokens": float(context_mask.sum().item()), "compression_ratio": 1.0}

    if isinstance(adapter, RandomMemory):
        memory = adapter(prompt_embeds.shape[0], prompt_embeds.dtype, prompt_embeds.device)
        context_embeds = torch.cat([context_embeds, memory], dim=1)
        context_mask = append_ones(context_mask, memory.shape[1])
        stats["context_tokens"] = float(context_mask.sum().item())
    elif builder is not None and isinstance(adapter, ThoughtMemoryAdapter):
        with torch.no_grad():
            teacher = wrapper.model(
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        thought = builder(teacher.hidden_states[-1].detach(), prompt_batch["attention_mask"], prompt_batch["input_ids"])
        memory_source = thought.embeddings
        memory = adapter(memory_source)
        context_embeds = torch.cat([context_embeds, memory], dim=1)
        context_mask = append_ones(context_mask, memory.shape[1])
        stats = {
            "thought_tokens": float(len(thought.all_thoughts)),
            "context_tokens": float(context_mask.sum().item()),
            "compression_ratio": float(thought.compression_ratios[-1]),
        }

    answer_embeds = wrapper.embed_tokens(answer_batch["input_ids"])
    inputs_embeds = torch.cat([context_embeds, answer_embeds], dim=1)
    answer_mask = torch.ones(answer_batch["input_ids"].shape, dtype=context_mask.dtype, device=device)
    attention_mask = torch.cat([context_mask, answer_mask], dim=1)
    labels = torch.full(attention_mask.shape, -100, dtype=torch.long, device=device)
    labels[:, context_embeds.shape[1] :] = answer_batch["input_ids"]
    out = wrapper.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
    return out.loss, stats


def append_ones(mask: torch.Tensor, count: int) -> torch.Tensor:
    return torch.cat([mask, torch.ones(mask.shape[0], count, dtype=mask.dtype, device=mask.device)], dim=1)


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        run["variant"]: {
            "train_first": None if not run["train_history"] else run["train_history"][0]["loss"],
            "train_last": None if not run["train_history"] else run["train_history"][-1]["loss"],
            "iid_accuracy": run["iid"]["accuracy"],
            "iid_gold_loss": run["iid"]["gold_loss"],
            "ood_accuracy": run["ood_depth"]["accuracy"],
            "ood_gold_loss": run["ood_depth"]["gold_loss"],
        }
        for run in runs
    }


if __name__ == "__main__":
    main()
