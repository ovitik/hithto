from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from thought_tokens.datasets import format_prompt
from thought_tokens.gemma_wrapper import GemmaThoughtWrapper, freeze_model
from thought_tokens.losses import paraphrase_contrastive_loss, reconstruction_loss
from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderConfig, ThoughtBuilderOutput
from thought_tokens.training import gemma_load_config_from_dict, infer_model_hidden_size, load_config
from thought_tokens.world_generator import CzechWorldGenerator


@dataclass
class SemanticCase:
    world_id: str
    text_a: str
    text_b: str
    counterfactual_text: str
    question: str
    answer: str
    counterfactual_answer: str
    depth: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_level.yaml")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--variant", choices=["fixed", "recursive", "soft"], default="recursive")
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--train-cases", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-prompt-length", type=int, default=112)
    parser.add_argument("--max-answer-length", type=int, default=12)
    parser.add_argument("--fixed-window", type=int, default=4)
    parser.add_argument("--recursive-window", type=int, default=2)
    parser.add_argument("--out", default="reports/semantic_thought_experiment.json")
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
    load_seconds = perf_counter() - load_start

    builder = make_builder(args.variant, hidden_size, args.fixed_window, args.recursive_window)
    builder.to(device=param.device, dtype=param.dtype)
    optimizer = torch.optim.AdamW(builder.parameters(), lr=args.lr)
    cases = make_semantic_cases(args.cases, seed=13)
    train_cases = cases[: args.train_cases]
    eval_cases = cases[args.train_cases :]

    before = evaluate_semantics(
        wrapper,
        builder,
        cases,
        args.max_prompt_length,
        args.max_answer_length,
    )

    history: list[dict[str, float]] = []
    builder.train()
    for step in range(args.steps):
        start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        anchors = []
        positives = []
        recon_terms = []
        cf_terms = []
        for case in train_cases:
            out_a = encode_prompt(wrapper, builder, row_for(case.text_a, case.question, case.answer), args.max_prompt_length)
            out_b = encode_prompt(wrapper, builder, row_for(case.text_b, case.question, case.answer), args.max_prompt_length)
            out_cf = encode_prompt(
                wrapper,
                builder,
                row_for(case.counterfactual_text, case.question, case.counterfactual_answer),
                args.max_prompt_length,
            )
            vec_a = semantic_vector(out_a)
            vec_b = semantic_vector(out_b)
            vec_cf = semantic_vector(out_cf)
            anchors.append(vec_a)
            positives.append(vec_b)
            cf_terms.append(F.relu(F.cosine_similarity(vec_a, vec_cf, dim=0) - 0.2))
            recon_terms.append(reconstruction_loss(out_a, builder.compressor.reconstruct_components))
            recon_terms.append(reconstruction_loss(out_b, builder.compressor.reconstruct_components))

        anchor_tensor = torch.stack(anchors, dim=0)
        positive_tensor = torch.stack(positives, dim=0)
        contrastive = paraphrase_contrastive_loss(anchor_tensor, positive_tensor, temperature=0.2)
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

    builder.eval()
    after = evaluate_semantics(
        wrapper,
        builder,
        cases,
        args.max_prompt_length,
        args.max_answer_length,
    )

    report: dict[str, Any] = {
        "model_name": args.model_name,
        "variant": args.variant,
        "dtype": str(param.dtype),
        "device": str(param.device),
        "hidden_size": hidden_size,
        "load_seconds": load_seconds,
        "cases": [asdict(case) for case in cases],
        "train_case_count": len(train_cases),
        "eval_case_count": len(eval_cases),
        "history": history,
        "before": before,
        "after": after,
        "delta": {
            key: after[key] - before[key]
            for key in before
            if isinstance(before[key], float) and isinstance(after.get(key), float)
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_builder(variant: str, hidden_size: int, fixed_window: int, recursive_window: int) -> ThoughtBuilder:
    if variant == "recursive":
        grouping = "fixed"
        window = recursive_window
        levels = 2
    elif variant == "fixed":
        grouping = "fixed"
        window = fixed_window
        levels = 1
    else:
        grouping = "soft"
        window = fixed_window
        levels = 1
    return ThoughtBuilder(
        ThoughtBuilderConfig(
            hidden_size=hidden_size,
            mode="replacement",
            grouping=grouping,
            fixed_window=window,
            max_levels=levels,
            min_group_size=2,
            max_group_size=8,
        )
    )


def make_semantic_cases(n: int, seed: int) -> list[SemanticCase]:
    rng = random.Random(seed)
    base = CzechWorldGenerator(seed)
    cases: list[SemanticCase] = []
    for i in range(n):
        obj = rng.choice(base.objects)
        depth = 2 + (i % 3)
        chain = rng.sample(base.containers + base.places, k=depth + 1)
        cf_final = rng.choice([x for x in base.containers + base.places if x != chain[-1]])
        cf_chain = list(chain)
        cf_chain[-1] = cf_final
        cases.append(
            SemanticCase(
                world_id=f"semantic-{i}",
                text_a=chain_text(obj, chain, style=0),
                text_b=chain_text(obj, chain, style=1),
                counterfactual_text=chain_text(obj, cf_chain, style=2),
                question=f"Kde je nakonec {obj}?",
                answer=chain[-1],
                counterfactual_answer=cf_final,
                depth=depth,
            )
        )
    return cases


def chain_text(obj: str, chain: list[str], style: int) -> str:
    current = obj
    parts = []
    for idx, nxt in enumerate(chain):
        if style == 0:
            parts.append(f"{current} je v {nxt}.")
        elif style == 1:
            parts.append(f"Uvnitř {nxt} se nachází {current}.")
        else:
            parts.append(f"Do {nxt} byl přesunut {current}.")
        current = nxt
    if style == 1:
        return " ".join(reversed(parts))
    return " ".join(parts)


def row_for(text: str, question: str, answer: str) -> dict[str, Any]:
    return {"text": text, "question": question, "answer": answer, "task_type": "nesting", "depth": 0, "id": "semantic"}


def encode_prompt(
    wrapper: GemmaThoughtWrapper,
    builder: ThoughtBuilder,
    row: dict[str, Any],
    max_prompt_length: int,
) -> ThoughtBuilderOutput:
    device = next(wrapper.model.parameters()).device
    batch = wrapper.tokenizer(
        format_prompt(row),
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)
    with torch.no_grad():
        teacher = wrapper.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    return builder(
        teacher.hidden_states[-1].detach(),
        attention_mask=batch["attention_mask"],
        token_ids=batch["input_ids"],
    )


def semantic_vector(out: ThoughtBuilderOutput) -> torch.Tensor:
    final_nodes = [
        unit
        for sample in out.units
        for unit in sample
        if hasattr(unit, "level")
    ]
    if final_nodes:
        max_level = max(node.level for node in final_nodes)
        selected = [node.embedding for node in final_nodes if node.level == max_level]
        return torch.stack(selected, dim=0).mean(dim=0)
    return out.embeddings.mean(dim=(0, 1))


@torch.no_grad()
def evaluate_semantics(
    wrapper: GemmaThoughtWrapper,
    builder: ThoughtBuilder,
    cases: list[SemanticCase],
    max_prompt_length: int,
    max_answer_length: int,
) -> dict[str, Any]:
    vectors_a = []
    vectors_b = []
    vectors_cf = []
    same = []
    cf = []
    hierarchy_sizes = []
    compressed_losses = []
    original_losses = []
    for case in cases:
        row_a = row_for(case.text_a, case.question, case.answer)
        row_b = row_for(case.text_b, case.question, case.answer)
        row_cf = row_for(case.counterfactual_text, case.question, case.counterfactual_answer)
        out_a = encode_prompt(wrapper, builder, row_a, max_prompt_length)
        out_b = encode_prompt(wrapper, builder, row_b, max_prompt_length)
        out_cf = encode_prompt(wrapper, builder, row_cf, max_prompt_length)
        vec_a = F.normalize(semantic_vector(out_a).float(), dim=0)
        vec_b = F.normalize(semantic_vector(out_b).float(), dim=0)
        vec_cf = F.normalize(semantic_vector(out_cf).float(), dim=0)
        vectors_a.append(vec_a)
        vectors_b.append(vec_b)
        vectors_cf.append(vec_cf)
        same.append(float(torch.dot(vec_a, vec_b).cpu()))
        cf.append(float(torch.dot(vec_a, vec_cf).cpu()))
        hierarchy_sizes.append(float(len(out_a.all_thoughts)))
        compressed_losses.append(
            float(score_answer_from_context(wrapper, out_a.embeddings, out_a.attention_mask, case.answer, max_answer_length).cpu())
        )
        original_losses.append(float(score_original_prompt(wrapper, row_a, case.answer, max_prompt_length, max_answer_length).cpu()))

    negative = []
    if len(vectors_a) > 1:
        for i, vec_a in enumerate(vectors_a):
            j = (i + 1) % len(vectors_b)
            negative.append(float(torch.dot(vec_a, vectors_b[j]).cpu()))
    else:
        negative.append(0.0)

    return {
        "same_paraphrase_cosine": mean(same),
        "negative_world_cosine": mean(negative),
        "counterfactual_cosine": mean(cf),
        "paraphrase_margin_vs_negative": mean(same) - mean(negative),
        "paraphrase_margin_vs_counterfactual": mean(same) - mean(cf),
        "mean_thought_count": mean(hierarchy_sizes),
        "compressed_gold_loss": mean(compressed_losses),
        "original_gold_loss": mean(original_losses),
        "compression_loss_gap": mean(compressed_losses) - mean(original_losses),
    }


def score_original_prompt(
    wrapper: GemmaThoughtWrapper,
    row: dict[str, Any],
    answer: str,
    max_prompt_length: int,
    max_answer_length: int,
) -> torch.Tensor:
    device = next(wrapper.model.parameters()).device
    tokenizer = wrapper.tokenizer
    prompt = tokenizer(format_prompt(row), return_tensors="pt", truncation=True, max_length=max_prompt_length).to(device)
    answer_batch = tokenizer(
        " " + answer,
        return_tensors="pt",
        truncation=True,
        max_length=max_answer_length,
        add_special_tokens=False,
    ).to(device)
    prompt_embeds = wrapper.embed_tokens(prompt["input_ids"])
    prompt_mask = prompt["attention_mask"]
    return score_answer_from_context(wrapper, prompt_embeds, prompt_mask, answer, max_answer_length)


def score_answer_from_context(
    wrapper: GemmaThoughtWrapper,
    context_embeds: torch.Tensor,
    context_mask: torch.Tensor,
    answer: str,
    max_answer_length: int,
) -> torch.Tensor:
    device = context_embeds.device
    answer_batch = wrapper.tokenizer(
        " " + answer,
        return_tensors="pt",
        truncation=True,
        max_length=max_answer_length,
        add_special_tokens=False,
    ).to(device)
    answer_embeds = wrapper.embed_tokens(answer_batch["input_ids"])
    inputs_embeds = torch.cat([context_embeds, answer_embeds], dim=1)
    answer_mask = torch.ones(answer_batch["input_ids"].shape, dtype=context_mask.dtype, device=device)
    attention_mask = torch.cat([context_mask, answer_mask], dim=1)
    labels = torch.full(attention_mask.shape, -100, dtype=torch.long, device=device)
    labels[:, context_embeds.shape[1] :] = answer_batch["input_ids"]
    out = wrapper.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
    return out.loss


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


if __name__ == "__main__":
    main()
