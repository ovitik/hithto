from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thought_tokens.hacot_grammar import HACoTGrammar, HA_TOKENS


Variant = Literal["direct", "cot", "flat", "hacot"]


# The abstract vocabulary is deliberately small and fixed.  A tree describes the
# program in postfix form: leaves are values and C2 tokens are binary operators.
# This gives HACoT structural meaning without placing natural-language traces in
# its target sequence.
ARITHMETIC_OPERATOR_TOKENS = {"+": "<C2_00>", "-": "<C2_01>", "*": "<C2_02>"}
BOOLEAN_OPERATOR_TOKENS = {"AND": "<C2_03>", "OR": "<C2_04>", "XOR": "<C2_05>"}
FLAT_OPERATOR_TOKENS = {
    "<C2_00>": "<A20>",
    "<C2_01>": "<A21>",
    "<C2_02>": "<A22>",
    "<C2_03>": "<A23>",
    "<C2_04>": "<A24>",
    "<C2_05>": "<A25>",
}


def value_token(value: int) -> str:
    """Encode a value that is visible in the question, never an intermediate result."""
    return f"<A{value % 10:02d}>"


def semantic_tree(operands: list[int], operators: list[str], op_tokens: dict[str, str]) -> list[str]:
    """Build a left-associated computation tree for a sequence of binary steps."""
    if len(operands) != len(operators) + 1:
        raise ValueError("binary program needs one more operand than operators")
    tokens = [HA_TOKENS[0], value_token(operands[0])]
    for operand, operator in zip(operands[1:], operators):
        tokens.extend([value_token(operand), op_tokens[operator]])
    tokens.extend(["<HA_ROOT>", "<HA_END>"])
    HACoTGrammar(max_depth=16).parse(tokens)
    return tokens


def flat_tokens_for_semantic_tree(tree_tokens: list[str]) -> list[str]:
    """Keep identical program symbols but remove the tree's compositional edges."""
    return [
        "<HA_START>",
        *[
            FLAT_OPERATOR_TOKENS.get(token, token)
            for token in tree_tokens
            if token not in {"<HA_START>", "<HA_ROOT>", "<HA_END>"}
        ],
        "<HA_END>",
    ]


@dataclass
class Example:
    prompt: str
    answer: str
    verbal_cot: str
    gold_tree: list[str]
    task_family: str
    difficulty: int
    split: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gen_arithmetic(rng: random.Random, difficulty: int, split: str) -> Example:
    value = rng.randint(1, 9)
    expr = str(value)
    trace = [f"start {value}"]
    operands = [value]
    operators = []
    for step in range(difficulty):
        op = rng.choice(["+", "-", "*"])
        x = rng.randint(1, 9)
        operands.append(x)
        operators.append(op)
        if op == "+":
            value = (value + x) % 10
        elif op == "-":
            value = (value - x) % 10
        else:
            value = (value * x) % 10
        expr = f"({expr} {op} {x})"
        trace.append(f"{step + 1}: {op}{x} -> {value}")
    return Example(
        prompt=f"Compute the value of {expr} modulo 10. Return only an integer from 0 to 9.",
        answer=str(value),
        verbal_cot="; ".join(trace),
        gold_tree=semantic_tree(operands, operators, ARITHMETIC_OPERATOR_TOKENS),
        task_family="arithmetic",
        difficulty=difficulty,
        split=split,
    )


def gen_boolean(rng: random.Random, difficulty: int, split: str) -> Example:
    values = {k: bool(rng.getrandbits(1)) for k in ["p", "q", "r", "s"]}
    expr = rng.choice(list(values))
    value = values[expr]
    trace = [f"{expr}={int(value)}"]
    operands = [int(value)]
    operators = []
    for _ in range(difficulty):
        var = rng.choice(list(values))
        op = rng.choice(["AND", "OR", "XOR"])
        rhs = values[var]
        operands.append(int(rhs))
        operators.append(op)
        if op == "AND":
            value = value and rhs
        elif op == "OR":
            value = value or rhs
        else:
            value = bool(value) ^ bool(rhs)
        expr = f"({expr} {op} {var})"
        trace.append(f"{op} {var}={int(rhs)} -> {int(value)}")
    prompt = (
        f"Given p={int(values['p'])}, q={int(values['q'])}, r={int(values['r'])}, "
        f"s={int(values['s'])}, evaluate {expr}. Return 0 or 1."
    )
    return Example(
        prompt,
        str(int(value)),
        "; ".join(trace),
        semantic_tree(operands, operators, BOOLEAN_OPERATOR_TOKENS),
        "boolean",
        difficulty,
        split,
    )


GENERATORS = [gen_arithmetic, gen_boolean]


def generate_examples(n: int, split: str, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples = []
    seen = set()
    while len(examples) < n:
        difficulty = rng.choice([1, 2, 3, 4, 5, 6]) if split == "train" else rng.randint(6, 12)
        ex = rng.choice(GENERATORS)(rng, difficulty, split)
        key = ex.prompt + "\n" + ex.answer
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)
    return examples


def render_target(ex: Example, variant: Variant) -> str:
    if variant == "direct":
        return f"Answer: {ex.answer}"
    if variant == "cot":
        return f"Reasoning: {ex.verbal_cot}\nAnswer: {ex.answer}"
    if variant == "flat":
        return " ".join(flat_tokens_for_semantic_tree(ex.gold_tree)) + f"\nAnswer: {ex.answer}"
    if variant == "hacot":
        return " ".join(ex.gold_tree) + f"\nAnswer: {ex.answer}"
    raise ValueError(variant)


def render_prompt(ex: Example) -> str:
    return f"Question: {ex.prompt}\nRespond with the final answer after any required abstract reasoning.\n"


def render_chat_text(tokenizer, prompt: str, target: str | None = None) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise solver. Follow the requested output format exactly. "
                "When an answer is requested, write a line starting with 'Answer:'."
            ),
        },
        {"role": "user", "content": prompt + "\n/no_think"},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        if target is None:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
        )
    if target is None:
        return prompt
    return prompt + target


class PromptDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, variant: Variant, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.variant = variant
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ex = self.examples[index]
        prompt = render_prompt(ex)
        target = render_target(ex, self.variant)
        full = render_chat_text(self.tokenizer, prompt, target)
        enc = self.tokenizer(full, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = enc["input_ids"].clone()
        prompt_text = render_chat_text(self.tokenizer, prompt, None)
        prompt_ids = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length, return_tensors="pt")["input_ids"]
        labels[:, : min(prompt_ids.shape[1], labels.shape[1])] = -100
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": labels[0],
        }


def collate(rows: list[dict[str, torch.Tensor]], tokenizer) -> dict[str, torch.Tensor]:
    keys = rows[0].keys()
    max_len = max(row["input_ids"].shape[0] for row in rows)
    out = {}
    for key in keys:
        pad = -100 if key == "labels" else (tokenizer.pad_token_id if key == "input_ids" else 0)
        out[key] = torch.stack(
            [
                torch.nn.functional.pad(row[key], (0, max_len - row[key].shape[0]), value=pad)
                for row in rows
            ]
        )
    return out


def extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    raw = match.group(1) if match else text
    raw = raw.strip().splitlines()[0].strip()
    raw = re.sub(r"<\|[^>]+?\|>", "", raw)
    raw = raw.replace("</s>", "").strip()
    return raw.strip().rstrip(".")


def norm_answer(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = text.replace("</s>", "")
    return re.sub(r"\s+", " ", text.strip().lower())


def load_model_and_tokenizer(args, attach_lora: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = None
    if args.qlora_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens({"additional_special_tokens": HA_TOKENS})
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if not args.qlora_4bit else None,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = not attach_lora
    if attach_lora and args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if added:
        model.resize_token_embeddings(len(tokenizer))
    if args.lora and attach_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.qlora_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    return model, tokenizer


def train_variant(args, variant: Variant, examples: list[Example], out_dir: Path) -> dict[str, float]:
    model, tokenizer = load_model_and_tokenizer(args, attach_lora=True)
    ds = PromptDataset(examples, tokenizer, variant, args.max_length)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate(rows, tokenizer),
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    t0 = time.time()
    losses = []
    step = 0
    while step < args.steps:
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch, use_cache=False)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach().cpu()))
            step += 1
            if step % args.log_every == 0:
                print(json.dumps({"variant": variant, "step": step, "loss": losses[-1]}), flush=True)
            if step >= args.steps:
                break
    elapsed = time.time() - t0
    save_dir = out_dir / "checkpoints" / variant
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_adapters and hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    del model
    torch.cuda.empty_cache()
    return {
        "train_loss_last": losses[-1],
        "train_loss_mean_last_20": float(np.mean(losses[-20:])),
        "train_seconds": elapsed,
        "steps": step,
    }


@torch.inference_mode()
def evaluate_variant(args, variant: Variant, examples: list[Example], out_dir: Path) -> dict[str, object]:
    model, tokenizer = load_model_and_tokenizer(args, attach_lora=False)
    adapter_dir = out_dir / "checkpoints" / variant
    if args.lora and adapter_dir.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    rows = []
    t0 = time.time()
    tokenizer.padding_side = "left"
    batch_size = getattr(args, "eval_batch_size", 8)
    for start in range(0, min(args.eval_n, len(examples)), batch_size):
        batch_examples = examples[start : start + batch_size]
        prompt_texts = [render_chat_text(tokenizer, render_prompt(ex), None) for ex in batch_examples]
        enc = tokenizer(prompt_texts, padding=True, return_tensors="pt").to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = gen[:, enc["input_ids"].shape[1] :]
        for ex, tokens in zip(batch_examples, generated):
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            pred = extract_answer(text)
            ok = norm_answer(pred) == norm_answer(ex.answer)
            rows.append({
                "variant": variant,
                "task_family": ex.task_family,
                "difficulty": ex.difficulty,
                "prompt": ex.prompt,
                "gold": ex.answer,
                "pred": pred,
                "correct": ok,
                "raw_generation": text,
            })
    elapsed = time.time() - t0
    acc = sum(r["correct"] for r in rows) / max(1, len(rows))
    del model
    torch.cuda.empty_cache()
    return {"accuracy": acc, "eval_seconds": elapsed, "rows": rows}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out-dir", default="runs/qwen_hacot_pilot")
    parser.add_argument("--train-n", type=int, default=1800)
    parser.add_argument("--dev-n", type=int, default=240)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--eval-n", type=int, default=120)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--variants", default="flat,hacot")
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qlora-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-adapters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = generate_examples(args.train_n, "train", args.seed)
    dev = generate_examples(args.dev_n, "dev", args.seed + 10_000)
    write_json(out_dir / "data_manifest.json", {
        "train_n": len(train),
        "dev_n": len(dev),
        "train_depths": sorted({x.difficulty for x in train}),
        "dev_depths": sorted({x.difficulty for x in dev}),
        "families": sorted({x.task_family for x in train + dev}),
    })
    write_json(out_dir / "sample_examples.json", [dataclasses.asdict(x) for x in train[:5]])

    if args.dry_run:
        print(json.dumps({"ok": True, "out_dir": str(out_dir), "variants": args.variants}))
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    summary = {"args": vars(args), "variants": {}, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    all_rows = []
    for variant in variants:
        if variant not in {"direct", "cot", "flat", "hacot"}:
            raise ValueError(f"unknown variant: {variant}")
        train_metrics = train_variant(args, variant, train, out_dir)
        eval_metrics = evaluate_variant(args, variant, dev, out_dir)
        summary["variants"][variant] = {
            **train_metrics,
            "accuracy": eval_metrics["accuracy"],
            "eval_seconds": eval_metrics["eval_seconds"],
        }
        all_rows.extend(eval_metrics["rows"])
        write_json(out_dir / "summary_partial.json", summary)
    summary["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "per_example_results.json", all_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
