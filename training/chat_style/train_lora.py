from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA trainer scaffold for chat-style rewrite.")
    parser.add_argument("--train-file", default="training/chat_style/data/chat_style_train.jsonl")
    parser.add_argument("--base-model", default="google/flan-t5-base")
    parser.add_argument("--output-dir", default="training/chat_style/artifacts/lora_adapter")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    train_path = Path(args.train_file)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}. Run prepare_data.py first."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intentionally scaffold-only to keep the main app lean.
    # You can implement full LoRA training with transformers+peft in this file.
    print("=== Chat Style LoRA Trainer (Scaffold) ===")
    print(f"Train file : {train_path}")
    print(f"Base model : {args.base_model}")
    print(f"Output dir : {out_dir}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print("")
    print("Next step:")
    print("- Install: pip install transformers datasets peft accelerate bitsandbytes")
    print("- Implement trainer loop in this script for your chosen model.")
    print("- Save adapter weights to output dir and wire provider='lora'.")


if __name__ == "__main__":
    main()
