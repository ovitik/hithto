from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AttentionCompressor(nn.Module):
    """Compress a contiguous span into one latent embedding using learned attention."""

    def __init__(self, hidden_size: int, max_group_size: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_group_size = max_group_size
        self.position = nn.Parameter(torch.zeros(max_group_size, hidden_size))
        nn.init.normal_(self.position, std=0.02)
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output = nn.Linear(hidden_size * 2, hidden_size)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, span: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if span.dim() != 2:
            raise ValueError("span must have shape [span_len, hidden_size]")
        length = span.shape[0]
        if length > self.max_group_size:
            raise ValueError(f"span length {length} exceeds max_group_size={self.max_group_size}")
        enriched = span + self.position[:length].to(dtype=span.dtype, device=span.device)
        weights = torch.softmax(self.score(enriched).squeeze(-1), dim=0)
        pooled = torch.sum(weights[:, None] * self.mix(enriched), dim=0)
        context = torch.sum(weights[:, None] * span, dim=0)
        token = self.output(torch.cat([pooled, context], dim=-1))
        return token, weights

    def reconstruct_components(self, thought_embeddings: torch.Tensor, target_len: int) -> torch.Tensor:
        base = self.reconstruction_head(thought_embeddings)
        if base.dim() == 1:
            base = base.unsqueeze(0)
        return base.unsqueeze(1).expand(base.shape[0], target_len, base.shape[-1])


class ProjectionToEmbedding(nn.Module):
    def __init__(self, hidden_size: int, embedding_size: int | None = None) -> None:
        super().__init__()
        embedding_size = hidden_size if embedding_size is None else embedding_size
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)
