from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch


@dataclass
class EvaluationMetrics:
    accuracy: float
    mean_latency_s: float
    mean_thought_tokens: float
    mean_compression_ratio: float


def exact_match(prediction: str, answer: str) -> bool:
    return prediction.strip().lower().strip(".") == answer.strip().lower().strip(".")


@torch.no_grad()
def evaluate_rows(wrapper: Any, rows: list[dict[str, Any]], tokenizer: Any, max_new_tokens: int = 16) -> EvaluationMetrics:
    correct = 0
    latencies: list[float] = []
    thought_counts: list[int] = []
    ratios: list[float] = []
    for row in rows:
        prompt = f"{row['text']}\nOtázka: {row['question']}\nOdpověď:"
        enc = tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        start = perf_counter()
        out = wrapper(**enc, use_thoughts=wrapper.thought_builder is not None)
        logits = out["model_output"].logits[:, -1]
        first = torch.argmax(logits, dim=-1, keepdim=True)
        prediction = tokenizer.decode(first[0], skip_special_tokens=True)
        latencies.append(perf_counter() - start)
        correct += int(exact_match(prediction, row["answer"]))
        thought = out.get("thought_output")
        thought_counts.append(0 if thought is None else len(thought.all_thoughts))
        ratios.append(1.0 if thought is None or not thought.compression_ratios else thought.compression_ratios[-1])
    n = max(1, len(rows))
    return EvaluationMetrics(
        accuracy=correct / n,
        mean_latency_s=sum(latencies) / n,
        mean_thought_tokens=sum(thought_counts) / n,
        mean_compression_ratio=sum(ratios) / n,
    )
