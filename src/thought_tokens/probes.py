from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ProbeMetrics:
    accuracy: float
    loss: float


class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def train_probe_step(probe: LinearProbe, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> ProbeMetrics:
    logits = probe(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    pred = logits.argmax(dim=-1)
    return ProbeMetrics(accuracy=float((pred == y).float().mean()), loss=float(loss.detach()))
