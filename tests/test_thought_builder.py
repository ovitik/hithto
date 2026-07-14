from __future__ import annotations

import torch

from thought_tokens.hierarchy import ThoughtNode, assert_acyclic
from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderConfig


def test_replacement_shapes_and_source_positions() -> None:
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(hidden_size=16, grouping="fixed", fixed_window=3, max_levels=1, max_group_size=4)
    )
    hidden = torch.randn(2, 7, 16, requires_grad=True)
    mask = torch.ones(2, 7, dtype=torch.long)
    out = builder(hidden, mask)
    assert out.embeddings.shape == (2, 3, 16)
    assert out.attention_mask.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert out.thoughts_by_level[0][0][0].source_positions == [0, 1, 2]


def test_additive_mode_keeps_originals_and_adds_thoughts() -> None:
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(
            hidden_size=8,
            grouping="fixed",
            fixed_window=2,
            max_levels=1,
            max_group_size=4,
            mode="additive",
        )
    )
    out = builder(torch.randn(1, 4, 8))
    assert out.embeddings.shape[1] == 6
    assert len(out.all_thoughts) == 2


def test_gradient_flows_through_compressor_and_grouper() -> None:
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(hidden_size=12, grouping="soft", max_levels=1, min_group_size=2, max_group_size=4)
    )
    hidden = torch.randn(1, 8, 12, requires_grad=True)
    out = builder(hidden)
    loss = out.embeddings.pow(2).mean()
    loss.backward()
    compressor_grads = [p.grad for p in builder.compressor.parameters() if p.requires_grad]
    grouper_grads = [p.grad for p in builder.grouper.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in compressor_grads)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grouper_grads)


def test_max_levels_zero_is_identity_baseline() -> None:
    builder = ThoughtBuilder(ThoughtBuilderConfig(hidden_size=8, max_levels=0))
    hidden = torch.randn(1, 5, 8)
    out = builder(hidden)
    assert torch.equal(out.embeddings, hidden)
    assert out.thoughts_by_level == []


def test_hierarchy_is_acyclic_and_recursive_uses_previous_thoughts() -> None:
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(hidden_size=10, grouping="fixed", fixed_window=3, max_levels=2, max_group_size=4)
    )
    out = builder(torch.randn(1, 10, 10))
    assert_acyclic(out.all_thoughts)
    level_two = [node for node in out.all_thoughts if node.level == 2]
    assert level_two
    assert any(isinstance(child, ThoughtNode) and child.level == 1 for child in level_two[0].children)


def test_hard_grouping_deterministic_in_eval() -> None:
    torch.manual_seed(123)
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(hidden_size=8, grouping="hard", max_levels=1, min_group_size=2, max_group_size=4)
    )
    builder.eval()
    hidden = torch.randn(1, 9, 8)
    out1 = builder(hidden)
    out2 = builder(hidden)
    spans1 = [node.metadata["span"] for node in out1.all_thoughts]
    spans2 = [node.metadata["span"] for node in out2.all_thoughts]
    assert spans1 == spans2


def test_bf16_forward_has_no_nan() -> None:
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(hidden_size=8, grouping="fixed", fixed_window=2, max_levels=1, max_group_size=4)
    ).to(dtype=torch.bfloat16)
    hidden = torch.randn(1, 6, 8, dtype=torch.bfloat16)
    out = builder(hidden)
    assert torch.isfinite(out.embeddings.float()).all()
