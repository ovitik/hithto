from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Segment:
    start: int
    end: int
    confidence: torch.Tensor
    score: torch.Tensor

    @property
    def length(self) -> int:
        return self.end - self.start


class BaseGrouper(nn.Module):
    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> list[list[Segment]]:
        raise NotImplementedError


class FixedWindowGrouper(BaseGrouper):
    def __init__(self, window: int = 4, min_group_size: int = 2) -> None:
        super().__init__()
        self.window = window
        self.min_group_size = min_group_size

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> list[list[Segment]]:
        bsz, seq_len, _ = hidden.shape
        device = hidden.device
        segments: list[list[Segment]] = []
        for b in range(bsz):
            valid = int(attention_mask[b].sum().item()) if attention_mask is not None else seq_len
            sample: list[Segment] = []
            for start in range(0, valid, self.window):
                end = min(valid, start + self.window)
                if end - start >= self.min_group_size:
                    score = hidden.new_tensor(1.0, device=device)
                    sample.append(Segment(start, end, score, score))
            segments.append(sample)
        return segments


class BoundaryGrouper(BaseGrouper):
    """Learn contiguous groups from adjacent hidden states.

    A high boundary probability means "split here"; groups are spans between boundaries. The
    confidence of a segment is the average probability of not splitting inside that span.
    """

    def __init__(
        self,
        hidden_size: int,
        min_group_size: int = 2,
        max_group_size: int = 8,
        hard: bool = False,
        temperature: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.hard = hard
        self.temperature = temperature
        self.threshold = threshold
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def boundary_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        left = hidden[:, :-1]
        right = hidden[:, 1:]
        feats = torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)
        return self.scorer(feats).squeeze(-1)

    def _boundary_values(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.hard:
            return torch.sigmoid(logits)
        if self.training:
            y_soft = torch.sigmoid((logits + self._logistic_noise(logits)) / self.temperature)
        else:
            y_soft = torch.sigmoid(logits / self.temperature)
        y_hard = (y_soft >= self.threshold).to(y_soft.dtype)
        return y_hard.detach() - y_soft.detach() + y_soft

    @staticmethod
    def _logistic_noise(reference: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(reference.dtype).eps
        u = torch.rand_like(reference).clamp(eps, 1.0 - eps)
        return torch.log(u) - torch.log1p(-u)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> list[list[Segment]]:
        bsz, seq_len, _ = hidden.shape
        if seq_len < self.min_group_size:
            return [[] for _ in range(bsz)]
        logits = self.boundary_logits(hidden)
        boundary = self._boundary_values(logits)
        segments: list[list[Segment]] = []
        for b in range(bsz):
            valid = int(attention_mask[b].sum().item()) if attention_mask is not None else seq_len
            sample: list[Segment] = []
            start = 0
            for pos in range(valid - 1):
                must_split = (pos + 1 - start) >= self.max_group_size
                split = bool(boundary[b, pos].detach().item() >= self.threshold) or must_split
                if split:
                    self._append_if_valid(sample, start, pos + 1, boundary[b])
                    start = pos + 1
            self._append_if_valid(sample, start, valid, boundary[b])
            if not sample and valid >= self.min_group_size:
                end = min(valid, self.max_group_size)
                self._append_if_valid(sample, 0, end, boundary[b])
            segments.append(sample)
        return segments

    def _append_if_valid(
        self,
        sample: list[Segment],
        start: int,
        end: int,
        boundary: torch.Tensor,
    ) -> None:
        if end - start < self.min_group_size:
            return
        internal = boundary[start : end - 1]
        if internal.numel() == 0:
            confidence = boundary.new_tensor(1.0)
        else:
            confidence = (1.0 - internal).mean()
        score = confidence
        sample.append(Segment(start, end, confidence, score))


class SoftBoundaryGrouper(BoundaryGrouper):
    def __init__(self, hidden_size: int, min_group_size: int = 2, max_group_size: int = 8) -> None:
        super().__init__(hidden_size, min_group_size, max_group_size, hard=False)


class HardBoundaryGrouper(BoundaryGrouper):
    def __init__(
        self,
        hidden_size: int,
        min_group_size: int = 2,
        max_group_size: int = 8,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            hidden_size,
            min_group_size,
            max_group_size,
            hard=True,
            temperature=temperature,
        )
