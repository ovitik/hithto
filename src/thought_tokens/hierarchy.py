from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import torch


@dataclass
class SourceToken:
    embedding: torch.Tensor
    position: int
    token_id: int | None = None
    text: str | None = None
    per_layer_embedding: torch.Tensor | None = None

    @property
    def source_positions(self) -> list[int]:
        return [self.position]


ThoughtChild: TypeAlias = "ThoughtNode | SourceToken"


@dataclass
class ThoughtNode:
    embedding: torch.Tensor
    level: int
    children: list[ThoughtChild]
    source_positions: list[int]
    confidence: torch.Tensor
    utility_score: torch.Tensor | None = None
    per_layer_embedding: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)

    def leaves(self) -> list[SourceToken]:
        out: list[SourceToken] = []
        for child in self.children:
            if isinstance(child, SourceToken):
                out.append(child)
            else:
                out.extend(child.leaves())
        return out

    def descendants(self) -> list["ThoughtNode"]:
        out: list[ThoughtNode] = []
        for child in self.children:
            if isinstance(child, ThoughtNode):
                out.append(child)
                out.extend(child.descendants())
        return out


Unit: TypeAlias = ThoughtNode | SourceToken


def unit_embedding(unit: Unit) -> torch.Tensor:
    return unit.embedding


def unit_per_layer_embedding(unit: Unit) -> torch.Tensor | None:
    return unit.per_layer_embedding


def unit_source_positions(unit: Unit) -> list[int]:
    if isinstance(unit, SourceToken):
        return [unit.position]
    return list(unit.source_positions)


def make_source_units(
    embeddings: torch.Tensor,
    token_ids: list[int] | None = None,
    texts: list[str] | None = None,
    per_layer_embeddings: torch.Tensor | None = None,
) -> list[SourceToken]:
    return [
        SourceToken(
            embedding=embeddings[i],
            position=i,
            token_id=None if token_ids is None else token_ids[i],
            text=None if texts is None else texts[i],
            per_layer_embedding=None if per_layer_embeddings is None else per_layer_embeddings[i],
        )
        for i in range(embeddings.shape[0])
    ]


def assert_acyclic(nodes: list[ThoughtNode]) -> None:
    visiting: set[int] = set()
    seen: set[int] = set()

    def visit(node: ThoughtNode) -> None:
        ident = id(node)
        if ident in visiting:
            raise ValueError("Thought hierarchy contains a cycle.")
        if ident in seen:
            return
        visiting.add(ident)
        for child in node.children:
            if isinstance(child, ThoughtNode):
                visit(child)
        visiting.remove(ident)
        seen.add(ident)

    for node in nodes:
        visit(node)


def flatten_thoughts(nodes: list[ThoughtNode]) -> list[ThoughtNode]:
    out: list[ThoughtNode] = []
    for node in nodes:
        out.append(node)
        out.extend(node.descendants())
    return out
