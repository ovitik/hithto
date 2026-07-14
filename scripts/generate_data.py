from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thought_tokens.datasets import generate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_jsonl(args.out, args.split, args.n, args.seed)
    print(f"wrote {args.n} examples to {args.out}")


if __name__ == "__main__":
    main()
