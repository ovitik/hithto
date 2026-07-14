from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from thought_tokens.datasets import JsonlReasoningDataset, collate_for_causal_lm
from thought_tokens.gemma_wrapper import GemmaLoadConfig, GemmaThoughtWrapper
from thought_tokens.losses import LossWeights, total_loss
from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_thought_builder(config: dict[str, Any]) -> ThoughtBuilder | None:
    tb = config.get("thought_builder", {})
    if not tb.get("enabled", True):
        return None
    return ThoughtBuilder(
        ThoughtBuilderConfig(
            hidden_size=int(tb["hidden_size"]),
            mode=tb.get("mode", "replacement"),
            grouping=tb.get("grouping", "fixed"),
            fixed_window=int(tb.get("fixed_window", 4)),
            max_levels=int(tb.get("max_levels", 1)),
            min_group_size=int(tb.get("min_group_size", 2)),
            max_group_size=int(tb.get("max_group_size", 8)),
            gumbel_temperature=float(tb.get("gumbel_temperature", 1.0)),
            adaptive_stopping=bool(tb.get("adaptive_stopping", False)),
        )
    )


def gemma_load_config_from_dict(config: dict[str, Any]) -> GemmaLoadConfig:
    allowed = {field.name for field in fields(GemmaLoadConfig)}
    return GemmaLoadConfig(**{k: v for k, v in config.items() if k in allowed})


def infer_model_hidden_size(model_name: str, trust_remote_code: bool = True) -> int:
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    text_config = getattr(cfg, "text_config", None)
    if text_config is not None and hasattr(text_config, "hidden_size"):
        return int(text_config.hidden_size)
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    raise ValueError(f"Could not infer hidden size for {model_name}.")


def sync_thought_hidden_size(config: dict[str, Any]) -> None:
    if not config.get("thought_builder", {}).get("enabled", True):
        return
    model_cfg = config.get("model", {})
    config["thought_builder"]["hidden_size"] = infer_model_hidden_size(
        model_cfg["model_name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )


def train_from_config(config_path: str | Path, train_path: str | Path | None = None, dry_run: bool = False) -> None:
    cfg = load_config(config_path)
    sync_thought_hidden_size(cfg)
    torch.manual_seed(int(cfg["train"].get("seed", 1)))
    builder = make_thought_builder(cfg)
    if dry_run:
        print(json.dumps({"ok": True, "thought_builder": builder is not None}, ensure_ascii=False))
        return

    wrapper = GemmaThoughtWrapper.from_pretrained(
        gemma_load_config_from_dict(cfg["model"]),
        thought_builder=builder,
    )
    if train_path is None:
        raise ValueError("--train-data is required unless --dry-run is used.")
    dataset = JsonlReasoningDataset(train_path)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 1)),
        shuffle=True,
        collate_fn=lambda rows: collate_for_causal_lm(rows, wrapper.tokenizer, int(cfg["train"].get("max_length", 512))),
    )
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["train"].get("lr", 2e-4)))
    weights = LossWeights(**cfg.get("loss", {}))
    log_path = Path(cfg["train"].get("log_jsonl", "runs/train/log.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    wrapper.train()
    while step < int(cfg["train"].get("steps", 100)):
        for batch in loader:
            batch = {k: v.to(wrapper.model.device) for k, v in batch.items()}
            out = wrapper(**batch)
            losses = total_loss(
                out["model_output"],
                weights,
                teacher_output=out.get("teacher_output"),
                thought_output=out.get("thought_output"),
                reconstruction_decoder=None if builder is None else builder.compressor.reconstruct_components,
            )
            losses["total"].backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({k: float(v.detach().cpu()) for k, v in losses.items()}) + "\n")
            step += 1
            if step >= int(cfg["train"].get("steps", 100)):
                break


def save_checkpoint(path: str | Path, wrapper: GemmaThoughtWrapper, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "thought_builder": None if wrapper.thought_builder is None else wrapper.thought_builder.state_dict(),
            "config": config,
        },
        path,
    )


def load_thought_checkpoint(path: str | Path, builder: ThoughtBuilder) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("thought_builder") is not None:
        builder.load_state_dict(payload["thought_builder"])
    return payload
