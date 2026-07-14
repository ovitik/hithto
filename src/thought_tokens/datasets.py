from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from thought_tokens.world_generator import CzechWorldGenerator, WorldExample


class JsonlReasoningDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def format_prompt(row: dict[str, Any]) -> str:
    return f"{row['text']}\nOtázka: {row['question']}\nOdpověď:"


def format_target(row: dict[str, Any]) -> str:
    return f" {row['answer']}"


def collate_for_causal_lm(rows: list[dict[str, Any]], tokenizer: Any, max_length: int = 512) -> dict[str, torch.Tensor]:
    texts = [format_prompt(row) + format_target(row) for row in rows]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    for i, row in enumerate(rows):
        prompt_ids = tokenizer(format_prompt(row), truncation=True, max_length=max_length)["input_ids"]
        labels[i, : min(len(prompt_ids), labels.shape[1])] = -100
    encoded["labels"] = labels
    return encoded


def generate_jsonl(path: str | Path, split: str, n: int, seed: int = 0) -> None:
    generator = CzechWorldGenerator(seed)
    examples: list[WorldExample] = generator.generate(split=split, n=n)  # type: ignore[arg-type]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")
