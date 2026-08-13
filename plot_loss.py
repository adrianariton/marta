"""
plot_loss.py
------------
Plots train + eval loss from a transformers trainer_state.json.

Usage:
    python plot_loss.py --state checkpoints/nemotron-guard-ft/checkpoint-837/trainer_state.json
    python plot_loss.py --state checkpoints/nemotron-guard-ft/checkpoint-837/trainer_state.json --out loss.png
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, help="Path to trainer_state.json")
    parser.add_argument("--out", default=None, help="Save plot to file (optional)")
    return parser.parse_args()


def main():
    args = parse_args()
    state = json.loads(Path(args.state).read_text())

    log_history = state.get("log_history", [])
    if not log_history:
        print("No log_history found in trainer_state.json")
        return

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in log_history:
        step = entry.get("step")
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])

    print(
        f"Train steps: {len(train_steps)}  first={train_steps[0] if train_steps else '-'}  last={train_steps[-1] if train_steps else '-'}"
    )
    print(
        f"Eval  steps: {len(eval_steps)}   first={eval_steps[0]  if eval_steps  else '-'}  last={eval_steps[-1]  if eval_steps  else '-'}"
    )
    if train_loss:
        print(
            f"Train loss:  min={min(train_loss):.4f}  max={max(train_loss):.4f}  final={train_loss[-1]:.4f}"
        )
    if eval_loss:
        print(
            f"Eval  loss:  min={min(eval_loss):.4f}   max={max(eval_loss):.4f}   final={eval_loss[-1]:.4f}"
        )
        best_step = eval_steps[eval_loss.index(min(eval_loss))]
        print(f"Best eval loss at step {best_step}: {min(eval_loss):.4f}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        if train_steps:
            ax.plot(train_steps, train_loss, label="train loss", linewidth=1.5, alpha=0.8)
        if eval_steps:
            ax.plot(eval_steps, eval_loss, label="eval loss", linewidth=2, marker="o", markersize=4)
            best_idx = eval_loss.index(min(eval_loss))
            ax.axvline(
                eval_steps[best_idx],
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"best eval @ step {eval_steps[best_idx]}",
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if args.out:
            plt.savefig(args.out, dpi=150)
            print(f"Saved to {args.out}")
        else:
            plt.savefig("loss.png", dpi=150)
            print("Saved to loss.png")

    except ImportError:
        print("matplotlib not available — printed stats above are all you get")


if __name__ == "__main__":
    main()
