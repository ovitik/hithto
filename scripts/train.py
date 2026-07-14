from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thought_tokens.training import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_from_config(args.config, train_path=args.train_data, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
