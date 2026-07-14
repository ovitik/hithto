from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from thought_tokens.compressor import AttentionCompressor, ProjectionToEmbedding
from thought_tokens.grouping import (
    BaseGrouper,
    FixedWindowGrouper,
    HardBoundaryGrouper,
    Segment,
    SoftBoundaryGrouper,
)
from thought_tokens.hierarchy import (
    ThoughtNode,
    Unit,
    assert_acyclic,
    make_source_units,
    unit_embedding,
    unit_per_layer_embedding,
    unit_source_positions,
)


@dataclass
class ThoughtBuilderConfig:
    hidden_size: int
    mode: Literal["additive", "replacement"] = "replacement"
    grouping: Literal["fixed", "soft", "hard"] = "fixed"
    fixed_window: int = 4
    max_levels: int = 1
    min_group_size: int = 2
    max_group_size: int = 8
    gumbel_temperature: float = 1.0
    adaptive_stopping: bool = False


@dataclass
class ThoughtLevelOutput:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    units: list[list[Unit]]
    thoughts: list[list[ThoughtNode]]
    segments: list[list[Segment]]
    per_layer_inputs: torch.Tensor | None = None


@dataclass
class ThoughtBuilderOutput:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    units: list[list[Unit]]
    thoughts_by_level: list[list[list[ThoughtNode]]]
    compression_ratios: list[float]
    per_layer_inputs: torch.Tensor | None = None

    @property
    def all_thoughts(self) -> list[ThoughtNode]:
        out: list[ThoughtNode] = []
        for level in self.thoughts_by_level:
            for sample in level:
                out.extend(sample)
        return out


class ThoughtBuilder(nn.Module):
    def __init__(self, config: ThoughtBuilderConfig) -> None:
        super().__init__()
        self.config = config
        self.grouper = self._make_grouper(config)
        self.compressor = AttentionCompressor(config.hidden_size, max_group_size=config.max_group_size)
        self.to_embedding = ProjectionToEmbedding(config.hidden_size)
        self.utility_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1),
        )

    @staticmethod
    def _make_grouper(config: ThoughtBuilderConfig) -> BaseGrouper:
        if config.grouping == "fixed":
            return FixedWindowGrouper(config.fixed_window, config.min_group_size)
        if config.grouping == "soft":
            return SoftBoundaryGrouper(config.hidden_size, config.min_group_size, config.max_group_size)
        if config.grouping == "hard":
            return HardBoundaryGrouper(
                config.hidden_size,
                config.min_group_size,
                config.max_group_size,
                config.gumbel_temperature,
            )
        raise ValueError(f"Unknown grouping: {config.grouping}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        max_levels: int | None = None,
    ) -> ThoughtBuilderOutput:
        if attention_mask is None:
            attention_mask = hidden_states.new_ones(hidden_states.shape[:2], dtype=torch.long)
        max_levels = self.config.max_levels if max_levels is None else max_levels
        if max_levels <= 0:
            units = self._source_units_for_batch(hidden_states, attention_mask, token_ids, per_layer_inputs)
            return ThoughtBuilderOutput(hidden_states, attention_mask, units, [], [1.0], per_layer_inputs)

        current_embeddings = hidden_states
        current_mask = attention_mask
        current_per_layer = per_layer_inputs
        current_units = self._source_units_for_batch(hidden_states, attention_mask, token_ids, per_layer_inputs)
        thoughts_by_level: list[list[list[ThoughtNode]]] = []
        ratios: list[float] = []
        original_active = int(attention_mask.sum().item())

        for level in range(1, max_levels + 1):
            level_out = self.build_level(current_embeddings, current_mask, current_units, level)
            thoughts_by_level.append(level_out.thoughts)
            current_embeddings = level_out.embeddings
            current_mask = level_out.attention_mask
            current_per_layer = level_out.per_layer_inputs
            current_units = level_out.units
            active = max(1, int(current_mask.sum().item()))
            ratios.append(original_active / active)
            if self.config.adaptive_stopping and sum(len(x) for x in level_out.thoughts) == 0:
                break

        assert_acyclic([node for sample in current_units for node in sample if isinstance(node, ThoughtNode)])
        return ThoughtBuilderOutput(
            embeddings=current_embeddings,
            attention_mask=current_mask,
            units=current_units,
            thoughts_by_level=thoughts_by_level,
            compression_ratios=ratios,
            per_layer_inputs=current_per_layer,
        )

    def build_level(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        units: list[list[Unit]],
        level: int,
    ) -> ThoughtLevelOutput:
        segments = self.grouper(hidden_states, attention_mask)
        new_units_batch: list[list[Unit]] = []
        thoughts_batch: list[list[ThoughtNode]] = []
        embeddings_batch: list[torch.Tensor] = []
        per_layer_batch: list[torch.Tensor] = []
        has_per_layer = any(
            unit_per_layer_embedding(unit) is not None for sample in units for unit in sample
        )

        for b, sample_segments in enumerate(segments):
            sample_units = units[b]
            thoughts: list[ThoughtNode] = []
            new_units: list[Unit] = []
            cursor = 0

            for segment in sample_segments:
                while cursor < segment.start:
                    new_units.append(sample_units[cursor])
                    cursor += 1
                thought = self._make_thought(hidden_states[b, segment.start : segment.end], sample_units, segment, level)
                thoughts.append(thought)
                if self.config.mode == "replacement":
                    new_units.append(thought)
                elif self.config.mode == "additive":
                    new_units.extend(sample_units[segment.start : segment.end])
                    new_units.append(thought)
                else:
                    raise ValueError(f"Unknown mode: {self.config.mode}")
                cursor = segment.end

            valid = int(attention_mask[b].sum().item())
            while cursor < valid:
                new_units.append(sample_units[cursor])
                cursor += 1

            if not new_units:
                new_units = sample_units[:valid]
            new_units_batch.append(new_units)
            thoughts_batch.append(thoughts)
            embeddings_batch.append(torch.stack([unit_embedding(unit) for unit in new_units], dim=0))
            if has_per_layer:
                per_layer_values = [unit_per_layer_embedding(unit) for unit in new_units]
                if any(value is None for value in per_layer_values):
                    raise ValueError("Mixed units with and without per_layer_embedding are not supported.")
                per_layer_batch.append(torch.stack(per_layer_values, dim=0))  # type: ignore[arg-type]

        padded, new_mask = self._pad(embeddings_batch, hidden_states.shape[-1])
        padded_per_layer = self._pad_per_layer(per_layer_batch) if has_per_layer else None
        return ThoughtLevelOutput(
            padded,
            new_mask,
            new_units_batch,
            thoughts_batch,
            segments,
            padded_per_layer,
        )

    def _make_thought(
        self,
        span_hidden: torch.Tensor,
        sample_units: list[Unit],
        segment: Segment,
        level: int,
    ) -> ThoughtNode:
        token_hidden, weights = self.compressor(span_hidden)
        confidence = segment.confidence * weights.max()
        embedding = self.to_embedding(token_hidden) * confidence.clamp_min(0.05)
        children = sample_units[segment.start : segment.end]
        per_layer_embedding = self._compress_per_layer(children, weights)
        positions: list[int] = []
        for child in children:
            positions.extend(unit_source_positions(child))
        utility = torch.sigmoid(self.utility_head(token_hidden).squeeze(-1))
        return ThoughtNode(
            embedding=embedding,
            level=level,
            children=children,
            source_positions=sorted(set(positions)),
            confidence=confidence,
            utility_score=utility,
            per_layer_embedding=per_layer_embedding,
            metadata={"span": (segment.start, segment.end), "attention_weights": weights},
        )

    @staticmethod
    def _compress_per_layer(children: list[Unit], weights: torch.Tensor) -> torch.Tensor | None:
        values = [unit_per_layer_embedding(child) for child in children]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("Cannot compress mixed per-layer and non-per-layer children.")
        stacked = torch.stack(values, dim=0)  # type: ignore[arg-type]
        w = weights.to(dtype=stacked.dtype, device=stacked.device)
        return torch.sum(w[:, None, None] * stacked, dim=0)

    @staticmethod
    def _source_units_for_batch(
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        token_ids: torch.Tensor | None,
        per_layer_inputs: torch.Tensor | None,
    ) -> list[list[Unit]]:
        out: list[list[Unit]] = []
        for b in range(embeddings.shape[0]):
            valid = int(attention_mask[b].sum().item())
            ids = None if token_ids is None else token_ids[b, :valid].detach().cpu().tolist()
            pli = None if per_layer_inputs is None else per_layer_inputs[b, :valid]
            out.append(make_source_units(embeddings[b, :valid], ids, per_layer_embeddings=pli))
        return out

    @staticmethod
    def _pad(samples: list[torch.Tensor], hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(sample.shape[0] for sample in samples)
        device = samples[0].device
        dtype = samples[0].dtype
        padded = torch.zeros(len(samples), max_len, hidden_size, device=device, dtype=dtype)
        mask = torch.zeros(len(samples), max_len, device=device, dtype=torch.long)
        for i, sample in enumerate(samples):
            padded[i, : sample.shape[0]] = sample
            mask[i, : sample.shape[0]] = 1
        return padded, mask

    @staticmethod
    def _pad_per_layer(samples: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(sample.shape[0] for sample in samples)
        shape = samples[0].shape[1:]
        device = samples[0].device
        dtype = samples[0].dtype
        padded = torch.zeros(len(samples), max_len, *shape, device=device, dtype=dtype)
        for i, sample in enumerate(samples):
            padded[i, : sample.shape[0]] = sample
        return padded
