from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from thought_tokens.interventions import drop_thought, replace_thought_with_random, level_ablation
from thought_tokens.thought_builder import ThoughtBuilder, ThoughtBuilderConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=12)
    args = parser.parse_args()
    builder = ThoughtBuilder(
        ThoughtBuilderConfig(
            hidden_size=args.hidden_size,
            grouping="fixed",
            fixed_window=3,
            max_levels=2,
            max_group_size=4,
        )
    )
    hidden = torch.randn(1, args.seq_len, args.hidden_size)
    out = builder(hidden)
    node = out.all_thoughts[0]
    result = {
        "thought_tokens": len(out.all_thoughts),
        "drop_shape": list(drop_thought(out, node).shape),
        "random_replace_shape": list(replace_thought_with_random(out, node).shape),
        "level_ablation_shape": list(level_ablation(out, 1).shape),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
