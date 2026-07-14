from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from thought_tokens.hierarchy import SourceToken, ThoughtNode
from thought_tokens.thought_builder import ThoughtBuilderOutput


def thought_tree_to_dict(node: ThoughtNode) -> dict[str, Any]:
    return {
        "level": node.level,
        "source_positions": node.source_positions,
        "confidence": float(node.confidence.detach().cpu()),
        "utility_score": None
        if node.utility_score is None
        else float(node.utility_score.detach().cpu()),
        "children": [
            {
                "source_position": child.position,
                "token_id": child.token_id,
                "text": child.text,
            }
            if isinstance(child, SourceToken)
            else thought_tree_to_dict(child)
            for child in node.children
        ],
    }


def example_report(
    row: dict[str, Any],
    tokens: list[str],
    thought_output: ThoughtBuilderOutput,
    answer: str,
    intervention_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "text": row["text"],
        "question": row["question"],
        "tokens": tokens,
        "thoughts_by_level": [
            [thought_tree_to_dict(node) for node in sample]
            for level in thought_output.thoughts_by_level
            for sample in level
        ],
        "compression_ratios": thought_output.compression_ratios,
        "answer": answer,
        "interventions": intervention_results or [],
    }


def save_example_report(path: str | Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_accuracy_by_depth(rows: list[dict[str, Any]], out: str | Path) -> None:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["depth"]), []).append(float(row["correct"]))
    depths = sorted(grouped)
    acc = [sum(grouped[d]) / len(grouped[d]) for d in depths]
    plt.figure(figsize=(6, 4))
    plt.plot(depths, acc, marker="o")
    plt.xlabel("Reasoning depth")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_distribution(values: list[float], title: str, out: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
