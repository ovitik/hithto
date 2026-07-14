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
import torch.nn.functional as F

from reasoning_depth_experiment import generate_rows
from semantic_thought_experiment import (
    encode_prompt,
    evaluate_semantics,
    make_semantic_cases,
    row_for,
    semantic_vector,
)
from side_channel_thought_experiment import (
    ThoughtMemoryAdapter,
    batch_loss,
    evaluate,
    make_builder,
)
from thought_tokens.gemma_wrapper import GemmaThoughtWrapper, freeze_model
from thought_tokens.losses import paraphrase_contrastive_loss, reconstruction_loss
from thought_tokens.training import gemma_load_config_from_dict, infer_model_hidden_size, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_level.yaml")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--variants", nargs="+", default=["fixed", "recursive"])
    parser.add_argument("--conditions", nargs="+", default=["scratch", "semantic_pretrain"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--semantic-cases", type=int, default=5)
    parser.add_argument("--semantic-train-cases", type=int, default=3)
    parser.add_argument("--semantic-steps", type=int, default=3)
    parser.add_argument("--side-steps", type=int, default=2)
    parser.add_argument("--train-n", type=int, default=4)
    parser.add_argument("--eval-n", type=int, default=3)
    parser.add_argument("--semantic-lr", type=float, default=3e-4)
    parser.add_argument("--side-lr", type=float, default=3e-4)
    parser.add_argument("--memory-tokens", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=112)
    parser.add_argument("--max-answer-length", type=int, default=12)
    parser.add_argument("--fixed-window", type=int, default=4)
    parser.add_argument("--recursive-window", type=int, default=2)
    parser.add_argument("--out", default="reports/semantic_pretrain_side_channel_experiment.json")
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

    semantic_cases = make_semantic_cases(args.semantic_cases, seed=31 + args.seed)
    semantic_train = semantic_cases[: args.semantic_train_cases]
    train_rows = generate_rows("train", args.train_n, args.seed)
    iid_rows = generate_rows("validation", args.eval_n, args.seed + 100)
    ood_rows = generate_rows("ood_depth", args.eval_n, args.seed + 200)

    runs: list[dict[str, Any]] = []
    for variant in args.variants:
        for condition in args.conditions:
            torch.manual_seed(args.seed)
            builder = make_builder(variant, hidden_size, args.fixed_window, args.recursive_window)
            if builder is None:
                continue
            builder.to(device=param.device, dtype=param.dtype)
            adapter = ThoughtMemoryAdapter(hidden_size, args.memory_tokens).to(device=param.device, dtype=param.dtype)

            semantic_before = evaluate_semantics(
                wrapper,
                builder,
                semantic_cases,
                args.max_prompt_length,
                args.max_answer_length,
            )
            semantic_history: list[dict[str, float]] = []
            if condition == "semantic_pretrain":
                semantic_history = semantic_pretrain(
                    wrapper,
                    builder,
                    semantic_train,
                    args.semantic_steps,
                    args.semantic_lr,
                    args.max_prompt_length,
                )
            semantic_after = evaluate_semantics(
                wrapper,
                builder,
                semantic_cases,
                args.max_prompt_length,
                args.max_answer_length,
            )
            side_history = side_channel_train(
                wrapper,
                builder,
                adapter,
                train_rows,
                args.side_steps,
                args.side_lr,
                args.max_prompt_length,
                args.max_answer_length,
            )
            iid = evaluate(wrapper, iid_rows, builder, adapter, args.max_prompt_length, args.max_answer_length)
            ood = evaluate(wrapper, ood_rows, builder, adapter, args.max_prompt_length, args.max_answer_length)
            runs.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "semantic_before": semantic_before,
                    "semantic_after_pretrain": semantic_after,
                    "semantic_history": semantic_history,
                    "side_history": side_history,
                    "iid": iid,
                    "ood_depth": ood,
                }
            )

    report = {
        "model_name": args.model_name,
        "dtype": str(param.dtype),
        "device": str(param.device),
        "hidden_size": hidden_size,
        "load_seconds": load_seconds,
        "seed": args.seed,
        "semantic_steps": args.semantic_steps,
        "side_steps": args.side_steps,
        "train_n": args.train_n,
        "eval_n": args.eval_n,
        "memory_tokens": args.memory_tokens,
        "runs": runs,
        "summary": summarize(runs),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def semantic_pretrain(
    wrapper: GemmaThoughtWrapper,
    builder,
    cases,
    steps: int,
    lr: float,
    max_prompt_length: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(builder.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    builder.train()
    for step in range(steps):
        start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        anchors = []
        positives = []
        recon_terms = []
        cf_terms = []
        for case in cases:
            out_a = encode_prompt(wrapper, builder, row_for(case.text_a, case.question, case.answer), max_prompt_length)
            out_b = encode_prompt(wrapper, builder, row_for(case.text_b, case.question, case.answer), max_prompt_length)
            out_cf = encode_prompt(
                wrapper,
                builder,
                row_for(case.counterfactual_text, case.question, case.counterfactual_answer),
                max_prompt_length,
            )
            vec_a = semantic_vector(out_a)
            vec_b = semantic_vector(out_b)
            vec_cf = semantic_vector(out_cf)
            anchors.append(vec_a)
            positives.append(vec_b)
            cf_terms.append(F.relu(F.cosine_similarity(vec_a, vec_cf, dim=0) - 0.2))
            recon_terms.append(reconstruction_loss(out_a, builder.compressor.reconstruct_components))
            recon_terms.append(reconstruction_loss(out_b, builder.compressor.reconstruct_components))
        contrastive = paraphrase_contrastive_loss(torch.stack(anchors), torch.stack(positives), temperature=0.2)
        reconstruction = torch.stack(recon_terms).mean()
        counterfactual = torch.stack(cf_terms).mean()
        loss = contrastive + 0.1 * reconstruction + 0.5 * counterfactual
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(builder.parameters(), 1.0)
        optimizer.step()
        history.append(
            {
                "step": step,
                "seconds": perf_counter() - start,
                "loss": float(loss.detach().cpu()),
                "contrastive": float(contrastive.detach().cpu()),
                "reconstruction": float(reconstruction.detach().cpu()),
                "counterfactual": float(counterfactual.detach().cpu()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            }
        )
    return history


def side_channel_train(
    wrapper: GemmaThoughtWrapper,
    builder,
    adapter: ThoughtMemoryAdapter,
    rows: list[dict[str, Any]],
    steps: int,
    lr: float,
    max_prompt_length: int,
    max_answer_length: int,
) -> list[dict[str, float]]:
    params = list(builder.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    history: list[dict[str, float]] = []
    for step in range(steps):
        start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, stats = batch_loss(wrapper, rows, builder, adapter, max_prompt_length, max_answer_length)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
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
    return history


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run in runs:
        key = f"{run['variant']}:{run['condition']}"
        out[key] = {
            "semantic_margin_negative_before": run["semantic_before"]["paraphrase_margin_vs_negative"],
            "semantic_margin_negative_after": run["semantic_after_pretrain"]["paraphrase_margin_vs_negative"],
            "semantic_margin_counterfactual_before": run["semantic_before"]["paraphrase_margin_vs_counterfactual"],
            "semantic_margin_counterfactual_after": run["semantic_after_pretrain"]["paraphrase_margin_vs_counterfactual"],
            "side_train_first": None if not run["side_history"] else run["side_history"][0]["loss"],
            "side_train_last": None if not run["side_history"] else run["side_history"][-1]["loss"],
            "iid_accuracy": run["iid"]["accuracy"],
            "iid_gold_loss": run["iid"]["gold_loss"],
            "ood_accuracy": run["ood_depth"]["accuracy"],
            "ood_gold_loss": run["ood_depth"]["gold_loss"],
        }
    return out


if __name__ == "__main__":
    main()
