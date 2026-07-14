from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


BaselineName = Literal[
    "direct",
    "text_cot",
    "soft_tokens",
    "coconut",
    "fixed_pooling",
    "learned_one_level",
    "recursive",
    "random_latent_control",
]


@dataclass
class BaselineResult:
    name: str
    logits: torch.Tensor
    active_tokens: int
    extra_forwards: int


def add_text_cot_prompt(prompt: str) -> str:
    return prompt.replace("Odpověď:", "Postupně uvažuj a potom odpověz stručně.\nOdpověď:")


def random_latent_control(inputs_embeds: torch.Tensor, n_tokens: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device=inputs_embeds.device).manual_seed(seed)
    latent = torch.randn(
        inputs_embeds.shape[0],
        n_tokens,
        inputs_embeds.shape[-1],
        generator=generator,
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    ) * inputs_embeds.std().clamp_min(1e-6)
    return torch.cat([inputs_embeds, latent], dim=1)


def coconut_step(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, -1:, :]
