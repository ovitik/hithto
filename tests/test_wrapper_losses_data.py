from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from thought_tokens.datasets import generate_jsonl, JsonlReasoningDataset
from thought_tokens.gemma_wrapper import GemmaThoughtWrapper, align_labels_after_compression
from thought_tokens.losses import LossWeights, total_loss
from thought_tokens.training import load_thought_checkpoint, save_checkpoint
from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderConfig


class FakeLM(nn.Module):
    def __init__(self, vocab: int = 20, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab)
        self.device = torch.device("cpu")

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        **_: object,
    ) -> SimpleNamespace:
        hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=[hidden])


def test_wrapper_without_thoughts_matches_backbone() -> None:
    model = FakeLM()
    wrapper = GemmaThoughtWrapper(model)
    input_ids = torch.tensor([[1, 2, 3]])
    direct = model(input_ids=input_ids, output_hidden_states=True).logits
    wrapped = wrapper(input_ids=input_ids, use_thoughts=False)["model_output"].logits
    assert torch.allclose(direct, wrapped)


def test_label_alignment_after_replacement() -> None:
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(hidden_size=8, grouping="fixed", fixed_window=2, max_levels=1, max_group_size=4)
    )
    out = builder(torch.randn(1, 5, 8), token_ids=torch.tensor([[1, 2, 3, 4, 5]]))
    labels = torch.tensor([[10, 11, 12, 13, 14]])
    aligned = align_labels_after_compression(labels, out)
    assert aligned.shape[1] == out.embeddings.shape[1]
    assert aligned.tolist()[0] == [11, 13, 14]


def test_total_loss_with_thought_wrapper_is_finite() -> None:
    model = FakeLM()
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(hidden_size=8, grouping="fixed", fixed_window=2, max_levels=1, max_group_size=4)
    )
    wrapper = GemmaThoughtWrapper(model, builder)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = input_ids.clone()
    result = wrapper(input_ids=input_ids, labels=labels)
    losses = total_loss(
        result["model_output"],
        LossWeights(alpha_distillation=0.1, delta_compression=0.1),
        teacher_output=result["teacher_output"],
        thought_output=result["thought_output"],
    )
    assert torch.isfinite(losses["total"])


def test_data_generator_splits(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    generate_jsonl(path, "ood_depth", 12, seed=5)
    ds = JsonlReasoningDataset(path)
    assert len(ds) == 12
    assert max(row["depth"] for row in ds.rows) >= 4
    assert {"text", "question", "answer", "symbolic"} <= set(ds.rows[0])


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = FakeLM()
    builder = ThoughtBuilder(ThoughtBuilderConfig(hidden_size=8, grouping="fixed", max_levels=1))
    wrapper = GemmaThoughtWrapper(model, builder)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, wrapper, {"thought_builder": {"hidden_size": 8}})
    reloaded = ThoughtBuilder(ThoughtBuilderConfig(hidden_size=8, grouping="fixed", max_levels=1))
    payload = load_thought_checkpoint(path, reloaded)
    assert payload["config"]["thought_builder"]["hidden_size"] == 8


def test_fake_model_can_overfit_tiny_batch() -> None:
    torch.manual_seed(0)
    model = FakeLM(vocab=10, hidden=8)
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(hidden_size=8, grouping="fixed", fixed_window=2, max_levels=1, max_group_size=4)
    )
    wrapper = GemmaThoughtWrapper(model, builder)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=0.05)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = input_ids.clone()
    first_loss = None
    last_loss = None
    for _ in range(25):
        result = wrapper(input_ids=input_ids, labels=labels)
        loss = result["model_output"].loss
        first_loss = float(loss.detach()) if first_loss is None else first_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_loss = float(loss.detach())
    assert last_loss is not None and first_loss is not None
    assert last_loss < first_loss
