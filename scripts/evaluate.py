from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thought_tokens.datasets import JsonlReasoningDataset
from thought_tokens.gemma_wrapper import GemmaThoughtWrapper
from thought_tokens.training import (
    gemma_load_config_from_dict,
    load_config,
    load_thought_checkpoint,
    make_thought_builder,
    sync_thought_hidden_size,
)
from thought_tokens.evaluation import evaluate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    sync_thought_hidden_size(cfg)
    builder = make_thought_builder(cfg)
    if args.checkpoint and builder is not None:
        load_thought_checkpoint(args.checkpoint, builder)
    wrapper = GemmaThoughtWrapper.from_pretrained(gemma_load_config_from_dict(cfg["model"]), thought_builder=builder)
    rows = JsonlReasoningDataset(args.data).rows
    metrics = evaluate_rows(wrapper, rows, wrapper.tokenizer)
    print(json.dumps(metrics.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
