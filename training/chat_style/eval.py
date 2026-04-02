from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chat-style rewrite dataset quality.")
    parser.add_argument("--file", default="training/chat_style/data/chat_style_train.jsonl")
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    total = 0
    with path.open("r", encoding="utf-8") as f:
        for total, _ in enumerate(f, start=1):
            pass

    print(f"[info] total examples: {total}")
    print("[info] sample examples:")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.head:
                break
            ex = json.loads(line)
            print(f"\n--- sample {i + 1} ---")
            print("instruction:", ex.get("instruction", "")[:120])
            print("input:", ex.get("input", "")[:180])
            print("output:", ex.get("output", "")[:180])


if __name__ == "__main__":
    main()
