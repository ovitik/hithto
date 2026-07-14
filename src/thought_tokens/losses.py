from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from thought_tokens.thought_builder import ThoughtBuilderOutput


@dataclass
class LossWeights:
    alpha_distillation: float = 0.0
    beta_reconstruction: float = 0.0
    gamma_paraphrase: float = 0.0
    delta_compression: float = 0.0
    epsilon_stability: float = 0.0


def answer_loss_from_output(model_output: object) -> torch.Tensor:
    loss = getattr(model_output, "loss", None)
    if loss is None:
        raise ValueError("Model output does not include a loss; pass labels.")
    return loss


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    min_len = min(student_logits.shape[1], teacher_logits.shape[1])
    student = student_logits[:, :min_len] / temperature
    teacher = teacher_logits[:, :min_len] / temperature
    return F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher.detach(), dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def reconstruction_loss(
    thought_output: ThoughtBuilderOutput,
    decoder: nn.Module,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for level in thought_output.thoughts_by_level:
        for sample in level:
            for node in sample:
                target = torch.stack([child.embedding.detach() for child in node.children], dim=0)
                pred = decoder(node.embedding, target.shape[0]).squeeze(0)
                losses.append(F.mse_loss(pred, target))
    if not losses:
        return thought_output.embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def compression_loss(thought_output: ThoughtBuilderOutput, target_ratio: float = 1.5) -> torch.Tensor:
    if not thought_output.compression_ratios:
        return thought_output.embeddings.sum() * 0.0
    ratio = thought_output.embeddings.new_tensor(thought_output.compression_ratios[-1])
    too_small = F.relu(target_ratio - ratio)
    too_large = F.relu(ratio - target_ratio * 4.0)
    return too_small + 0.25 * too_large


def paraphrase_contrastive_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if anchors.numel() == 0 or positives.numel() == 0:
        return anchors.sum() * 0.0
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    logits = anchors @ positives.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def hierarchy_stability_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return (a.sum() + b.sum()) * 0.0
    min_len = min(a.shape[0], b.shape[0])
    return 1.0 - F.cosine_similarity(a[:min_len], b[:min_len], dim=-1).mean()


def total_loss(
    model_output: object,
    weights: LossWeights,
    teacher_output: object | None = None,
    thought_output: ThoughtBuilderOutput | None = None,
    reconstruction_decoder: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {"answer": answer_loss_from_output(model_output)}
    total = losses["answer"]
    if weights.alpha_distillation and teacher_output is not None:
        losses["distillation"] = distillation_loss(model_output.logits, teacher_output.logits)
        total = total + weights.alpha_distillation * losses["distillation"]
    if weights.beta_reconstruction and thought_output is not None and reconstruction_decoder is not None:
        losses["reconstruction"] = reconstruction_loss(thought_output, reconstruction_decoder)
        total = total + weights.beta_reconstruction * losses["reconstruction"]
    if weights.delta_compression and thought_output is not None:
        losses["compression"] = compression_loss(thought_output)
        total = total + weights.delta_compression * losses["compression"]
    losses["total"] = total
    return losses
