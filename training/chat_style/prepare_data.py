from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_csv_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _pair_from_row(row: dict) -> tuple[str, str] | None:
    # Support common dataset column naming variations.
    user = (
        row.get("user")
        or row.get("prompt")
        or row.get("question")
        or row.get("input")
        or row.get("message")
        or row.get("human")
        or row.get("instruction")
    )
    bot = (
        row.get("assistant")
        or row.get("response")
        or row.get("answer")
        or row.get("output")
        or row.get("bot")
        or row.get("gpt")
    )
    if not user or not bot:
        return None
    return user.strip(), bot.strip()


def _to_training_example(user_text: str, bot_text: str) -> dict:
    return {
        "instruction": "Rewrite in friendly conversational tone without changing facts.",
        "input": f"User: {user_text}\nFactual assistant reply: {bot_text}",
        "output": bot_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chat-style JSONL training data.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSV paths (conversation datasets).",
    )
    parser.add_argument(
        "--output",
        default="training/chat_style/data/chat_style_train.jsonl",
        help="Output JSONL file path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept = 0
    with output_path.open("w", encoding="utf-8") as out:
        for inp in args.inputs:
            path = Path(inp)
            if not path.exists():
                print(f"[warn] skipping missing file: {path}")
                continue
            rows = _read_csv_rows(path)
            total_rows += len(rows)
            for row in rows:
                pair = _pair_from_row(row)
                if not pair:
                    continue
                user_text, bot_text = pair
                if len(user_text) < 3 or len(bot_text) < 3:
                    continue
                ex = _to_training_example(user_text, bot_text)
                out.write(json.dumps(ex, ensure_ascii=True) + "\n")
                kept += 1

    print(f"[ok] wrote {kept} examples to {output_path} (from {total_rows} rows)")


if __name__ == "__main__":
    main()
