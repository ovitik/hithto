from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from thought_tokens.hierarchy import ThoughtNode


@dataclass
class ThoughtBankQuery:
    embedding: torch.Tensor
    k: int = 4
    min_similarity: float = 0.0


@dataclass
class ThoughtBankMatch:
    node: ThoughtNode
    similarity: torch.Tensor
    metadata: dict


class ThoughtBank(Protocol):
    """Future persistent memory interface.

    The first prototype intentionally does not keep a cross-task dictionary of thoughts. This
    protocol marks the integration point for later experiments without coupling the current
    end-to-end trainable path to retrieval.
    """

    def add(self, node: ThoughtNode, metadata: dict | None = None) -> str:
        ...

    def search(self, query: ThoughtBankQuery) -> list[ThoughtBankMatch]:
        ...

    def clear(self) -> None:
        ...
