from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from thought_tokens.hierarchy import ThoughtNode, Unit, unit_embedding
from thought_tokens.thought_builder import ThoughtBuilderOutput


@dataclass
class InterventionResult:
    name: str
    logits_delta: float
    changed_tokens: int


def drop_thought(thought_output: ThoughtBuilderOutput, node: ThoughtNode) -> torch.Tensor:
    return _replace_node_embedding(thought_output, node, replacement=None)


def replace_thought_with_random(
    thought_output: ThoughtBuilderOutput,
    node: ThoughtNode,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device=node.embedding.device).manual_seed(seed)
    replacement = torch.randn(
        node.embedding.shape,
        generator=generator,
        device=node.embedding.device,
        dtype=node.embedding.dtype,
    ) * node.embedding.std().clamp_min(1e-6)
    return _replace_node_embedding(thought_output, node, replacement=replacement)


def shuffle_thoughts_between(a: ThoughtBuilderOutput, b: ThoughtBuilderOutput) -> tuple[torch.Tensor, torch.Tensor]:
    a_nodes = a.all_thoughts
    b_nodes = b.all_thoughts
    if not a_nodes or not b_nodes:
        return a.embeddings, b.embeddings
    return (
        _replace_node_embedding(a, a_nodes[0], b_nodes[0].embedding),
        _replace_node_embedding(b, b_nodes[0], a_nodes[0].embedding),
    )


def expand_thought_components(thought_output: ThoughtBuilderOutput, node: ThoughtNode) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for units in thought_output.units:
        sample: list[torch.Tensor] = []
        for unit in units:
            if unit is node:
                sample.extend([unit_embedding(child) for child in node.children])
            else:
                sample.append(unit_embedding(unit))
        rows.append(torch.stack(sample, dim=0))
    return _pad(rows)


def level_ablation(thought_output: ThoughtBuilderOutput, level: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for units in thought_output.units:
        sample = [
            unit_embedding(unit)
            for unit in units
            if not (isinstance(unit, ThoughtNode) and unit.level == level)
        ]
        rows.append(torch.stack(sample or [unit_embedding(units[0])], dim=0))
    return _pad(rows)


@torch.no_grad()
def logits_delta(
    run_with_embeds: Callable[[torch.Tensor], torch.Tensor],
    original_embeds: torch.Tensor,
    intervened_embeds: torch.Tensor,
) -> float:
    original = run_with_embeds(original_embeds)
    intervened = run_with_embeds(intervened_embeds)
    min_len = min(original.shape[1], intervened.shape[1])
    return float(torch.mean(torch.abs(original[:, :min_len] - intervened[:, :min_len])).cpu())


def _replace_node_embedding(
    thought_output: ThoughtBuilderOutput,
    node: ThoughtNode,
    replacement: torch.Tensor | None,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for units in thought_output.units:
        sample: list[torch.Tensor] = []
        for unit in units:
            if unit is node:
                if replacement is not None:
                    sample.append(replacement)
            else:
                sample.append(unit_embedding(unit))
        rows.append(torch.stack(sample or [thought_output.embeddings.new_zeros(thought_output.embeddings.shape[-1])], dim=0))
    return _pad(rows)


def _pad(rows: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(row.shape[0] for row in rows)
    hidden = rows[0].shape[-1]
    out = rows[0].new_zeros(len(rows), max_len, hidden)
    for i, row in enumerate(rows):
        out[i, : row.shape[0]] = row
    return out
