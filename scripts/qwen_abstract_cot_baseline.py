"""Scaled Abstract-CoT warm-up baseline for Qwen.

This follows the warm-up stage of Ramji et al. (2026), "Thinking Without
Words": random guided bottleneck SFT followed by prompt-only self-distillation.
It intentionally does not implement GRPO. The purpose is to establish a valid,
causally testable flat Abstract-CoT checkpoint before introducing hierarchy.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import qwen_hacot_pilot as tasks


@dataclass(frozen=True)
class Codebook:
    begin: str
    end: str
    tokens: tuple[str, ...]

    @property
    def all_tokens(self) -> tuple[str, ...]:
        return (self.begin, *self.tokens, self.end)


def build_codebook(size: int) -> Codebook:
    if not 2 <= size <= 64:
        raise ValueError("codebook size must be in [2, 64]")
    return Codebook(
        begin="<AC_BEGIN>",
        end="<AC_END>",
        tokens=tuple(f"<AC_{index:02d}>" for index in range(size)),
    )


def render_prompt(ex: tasks.Example) -> str:
    return f"Question: {ex.prompt}\n"


def render_teacher_cot(ex: tasks.Example) -> str:
    return f"Reasoning: {ex.verbal_cot}\n"


def render_abstract(codebook: Codebook, trace: list[str]) -> str:
    return f"{codebook.begin} {' '.join(trace)} {codebook.end}\n"


def render_answer(ex: tasks.Example) -> str:
    return f"Answer: {ex.answer}"


def random_trace(ex: tasks.Example, codebook: Codebook, max_tokens: int, rng: random.Random) -> list[str]:
    """Allocate 1-2 uniformly sampled codes per teacher reasoning step."""
    steps = max(1, len(ex.verbal_cot.split(";")))
    trace: list[str] = []
    for _ in range(steps):
        for _ in range(rng.randint(1, 2)):
            if len(trace) >= max_tokens:
                return trace
            trace.append(rng.choice(codebook.tokens))
    return trace or [rng.choice(codebook.tokens)]


def encode_piece(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


class BottleneckDataset(Dataset):
    """[prompt ; verbal CoT ; abstract trace ; answer] with answer-to-CoT edges removed."""

    def __init__(
        self,
        examples: list[tasks.Example],
        traces: list[list[str]],
        tokenizer,
        codebook: Codebook,
        max_length: int,
    ) -> None:
        self.rows = []
        for ex, trace in zip(examples, traces):
            prompt = encode_piece(tokenizer, render_prompt(ex))
            cot = encode_piece(tokenizer, render_teacher_cot(ex))
            abstract = encode_piece(tokenizer, render_abstract(codebook, trace))
            answer = encode_piece(tokenizer, render_answer(ex))
            full = torch.cat([prompt, cot, abstract, answer])[:max_length]
            answer_start = min(len(full), len(prompt) + len(cot) + len(abstract))
            labels = torch.full_like(full, -100)
            # Random code targets have deliberately weak supervision. The answer loss is
            # what gives their hidden states an information-bearing role.
            labels[len(prompt) + len(cot) : answer_start] = full[len(prompt) + len(cot) : answer_start]
            labels[answer_start:] = full[answer_start:]
            self.rows.append(
                {
                    "input_ids": full,
                    "labels": labels,
                    "cot_start": len(prompt),
                    "cot_end": min(len(full), len(prompt) + len(cot)),
                    "answer_start": answer_start,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class DistillationDataset(Dataset):
    """[prompt ; prompt-only abstract trace ; answer] for standard causal SFT."""

    def __init__(
        self,
        examples: list[tasks.Example],
        traces: list[list[str]],
        tokenizer,
        codebook: Codebook,
        max_length: int,
    ) -> None:
        self.rows = []
        for ex, trace in zip(examples, traces):
            prompt = encode_piece(tokenizer, render_prompt(ex))
            abstract = encode_piece(tokenizer, render_abstract(codebook, trace))
            answer = encode_piece(tokenizer, render_answer(ex))
            full = torch.cat([prompt, abstract, answer])[:max_length]
            labels = torch.full_like(full, -100)
            labels[len(prompt) :] = full[len(prompt) :]
            self.rows.append({"input_ids": full, "labels": labels})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def build_bottleneck_mask(rows: list[dict], pad_token_id: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Build the paper's block-causal mask, with answer queries unable to see CoT keys."""
    input_ids = pad_sequence([row["input_ids"] for row in rows], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([row["labels"] for row in rows], batch_first=True, padding_value=-100)
    batch, width = input_ids.shape
    min_value = torch.finfo(dtype).min
    mask = torch.full((batch, 1, width, width), min_value, dtype=dtype)
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        allowed = torch.tril(torch.ones((length, length), dtype=torch.bool))
        allowed[row["answer_start"] :, row["cot_start"] : row["cot_end"]] = False
        mask[index, 0, :length, :length] = torch.where(
            allowed,
            torch.zeros((), dtype=dtype),
            torch.full((), min_value, dtype=dtype),
        )
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}


def standard_collate(rows: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": pad_sequence([row["input_ids"] for row in rows], batch_first=True, padding_value=pad_token_id),
        "labels": pad_sequence([row["labels"] for row in rows], batch_first=True, padding_value=-100),
        "attention_mask": pad_sequence(
            [torch.ones_like(row["input_ids"]) for row in rows], batch_first=True, padding_value=0
        ),
    }


def load_model_and_tokenizer(args, codebook: Codebook):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": list(codebook.all_tokens)})
    code_ids = tokenizer.convert_tokens_to_ids(list(codebook.all_tokens))
    if any(token_id == tokenizer.unk_token_id for token_id in code_ids):
        raise RuntimeError("abstract codebook was not added as atomic tokens")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    # Match the original embedding distribution but keep independent random rows.
    with torch.no_grad():
        input_weights = model.get_input_embeddings().weight
        mean = float(input_weights[:original_vocab_size].mean())
        std = float(input_weights[:original_vocab_size].std())
        input_weights[code_ids].normal_(mean=mean, std=std)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            output_weights = output_embeddings.weight
            if output_weights.shape[0] >= len(tokenizer):
                output_weights[code_ids].normal_(mean=mean, std=std)
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # This is the crucial difference from the failed HACoT smoke: PEFT keeps
        # trainable deltas for only the new embedding/output rows.
        trainable_token_indices={"embed_tokens": code_ids, "lm_head": code_ids},
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    return model, tokenizer, code_ids


def train_steps(model, loader: DataLoader, steps: int, grad_accum: int, lr: float, bottleneck: bool) -> list[float]:
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=lr)
    model.train()
    losses: list[float] = []
    step = 0
    while step < steps:
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            out = model(**batch, use_cache=False)
            (out.loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach().cpu()))
            step += 1
            if step >= steps:
                break
    return losses


def generate_trace(model, tokenizer, codebook: Codebook, code_ids: list[int], prompt: str, max_codes: int) -> list[str]:
    from transformers import LogitsProcessor

    begin_id = tokenizer.convert_tokens_to_ids(codebook.begin)
    end_id = tokenizer.convert_tokens_to_ids(codebook.end)
    prefix = tokenizer(prompt + codebook.begin, add_special_tokens=False, return_tensors="pt").to(model.device)
    prefix_length = prefix["input_ids"].shape[1]

    class CodebookOnly(LogitsProcessor):
        def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
            generated = input_ids.shape[1] - prefix_length
            allowed = [end_id] if generated >= max_codes else [*code_ids[1:-1], end_id]
            blocked = torch.full_like(scores, torch.finfo(scores.dtype).min)
            blocked[:, allowed] = scores[:, allowed]
            return blocked

    model.eval()
    generated = model.generate(
        **prefix,
        max_new_tokens=max_codes + 1,
        do_sample=False,
        eos_token_id=end_id,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=[CodebookOnly()],
    )
    new_ids = generated[0, prefix_length:].tolist()
    if not new_ids or new_ids[-1] != end_id:
        new_ids.append(end_id)
    return tokenizer.convert_ids_to_tokens(new_ids[:-1])


@torch.inference_mode()
def generate_answer(model, tokenizer, context: str) -> tuple[str, str]:
    enc = tokenizer(context, add_special_tokens=False, return_tensors="pt").to(model.device)
    output = model.generate(
        **enc,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0, enc["input_ids"].shape[1] :], skip_special_tokens=False)
    return tasks.extract_answer(text), text


@torch.inference_mode()
def evaluate(model, tokenizer, codebook: Codebook, code_ids: list[int], examples: Iterable[tasks.Example], max_codes: int) -> dict:
    rows = []
    rng = random.Random(9_999)
    for ex in examples:
        trace = generate_trace(model, tokenizer, codebook, code_ids, render_prompt(ex), max_codes)
        context = render_prompt(ex) + render_abstract(codebook, trace)
        pred, text = generate_answer(model, tokenizer, context)
        permuted_trace = trace[:]
        rng.shuffle(permuted_trace)
        permuted_pred, _ = generate_answer(
            model, tokenizer, render_prompt(ex) + render_abstract(codebook, permuted_trace)
        )
        rows.append(
            {
                "prompt": ex.prompt,
                "gold": ex.answer,
                "pred": pred,
                "correct": tasks.norm_answer(pred) == tasks.norm_answer(ex.answer),
                "trace": trace,
                "permuted_trace": permuted_trace,
                "permuted_pred": permuted_pred,
                "permuted_correct": tasks.norm_answer(permuted_pred) == tasks.norm_answer(ex.answer),
                "raw_generation": text,
            }
        )
    counts = collections.Counter(token for row in rows for token in row["trace"])
    total = sum(counts.values())
    entropy = -sum((count / total) * torch.log2(torch.tensor(count / total)).item() for count in counts.values()) if total else 0.0
    return {
        "accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "permuted_accuracy": sum(row["permuted_correct"] for row in rows) / max(1, len(rows)),
        "unique_trace_count": len({tuple(row["trace"]) for row in rows}),
        "codebook_tokens_used": len(counts),
        "codebook_entropy_bits": entropy,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaled policy-iteration warm-up for flat Abstract-CoT.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out-dir", default="runs/qwen_abstract_cot")
    parser.add_argument("--train-n", type=int, default=384)
    parser.add_argument("--dev-n", type=int, default=96)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--max-codes", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--teacher-steps", type=int, default=480)
    parser.add_argument("--distill-steps", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    codebook = build_codebook(args.codebook_size)
    tasks.set_seed(args.seed)
    train = tasks.generate_examples(args.train_n, "train", args.seed)
    dev = tasks.generate_examples(args.dev_n, "dev", args.seed + 10_000)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks.write_json(
        out_dir / "manifest.json",
        {
            "args": vars(args),
            "method": "policy_iteration_warmup_only",
            "codebook": list(codebook.all_tokens),
            "train_n": len(train),
            "dev_n": len(dev),
        },
    )
    if args.dry_run:
        print(json.dumps({"ok": True, "codebook_size": args.codebook_size, "rounds": args.rounds}))
        return

    model, tokenizer, code_ids = load_model_and_tokenizer(args, codebook)
    summary = {"args": vars(args), "rounds": [], "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    rng = random.Random(args.seed)
    guided_traces = [random_trace(ex, codebook, args.max_codes, rng) for ex in train]
    for round_index in range(args.rounds):
        bottleneck_data = BottleneckDataset(train, guided_traces, tokenizer, codebook, args.max_length)
        bottleneck_loader = DataLoader(
            bottleneck_data,
            batch_size=args.batch_size,
            shuffle=True,
            # Qwen's QLoRA attention path keeps query states in fp32, so SDPA
            # requires the additive 4D mask in fp32 as well.
            collate_fn=lambda rows: build_bottleneck_mask(rows, tokenizer.pad_token_id, torch.float32),
        )
        teacher_losses = train_steps(
            model, bottleneck_loader, args.teacher_steps, args.grad_accum, args.lr, bottleneck=True
        )
        prompt_traces = [generate_trace(model, tokenizer, codebook, code_ids, render_prompt(ex), args.max_codes) for ex in train]
        distill_data = DistillationDataset(train, prompt_traces, tokenizer, codebook, args.max_length)
        distill_loader = DataLoader(
            distill_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda rows: standard_collate(rows, tokenizer.pad_token_id),
        )
        distill_losses = train_steps(model, distill_loader, args.distill_steps, args.grad_accum, args.lr, bottleneck=False)
        evaluation = evaluate(model, tokenizer, codebook, code_ids, dev, args.max_codes)
        summary["rounds"].append(
            {
                "round": round_index + 1,
                "teacher_loss_last": teacher_losses[-1],
                "distill_loss_last": distill_losses[-1],
                "accuracy": evaluation["accuracy"],
                "permuted_accuracy": evaluation["permuted_accuracy"],
                "unique_trace_count": evaluation["unique_trace_count"],
                "codebook_tokens_used": evaluation["codebook_tokens_used"],
                "codebook_entropy_bits": evaluation["codebook_entropy_bits"],
            }
        )
        tasks.write_json(out_dir / f"eval_round_{round_index + 1}.json", evaluation)
        tasks.write_json(out_dir / "summary_partial.json", summary)
        # Later policy-iteration rounds generate guided traces with the teacher CoT present.
        guided_traces = [
            generate_trace(model, tokenizer, codebook, code_ids, render_prompt(ex) + render_teacher_cot(ex), args.max_codes)
            for ex in train
        ]
    summary["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tasks.write_json(out_dir / "summary.json", summary)
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
