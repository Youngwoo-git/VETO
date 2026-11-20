#!/usr/bin/env python3
"""Multiple-choice accuracy with the full input and from the text alone.

The prediction is the option letter with the largest logit at the answer position.

    python tools/evaluate.py --ckpt out/best.pt --data test.jsonl --conditions full,textonly --out result.json
    python tools/evaluate.py --base --data test.jsonl --out base.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

# The repository root holds the veto package, so the scripts run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veto.data import MCDataset
from veto.eval import evaluate
from veto.model import load_base_model, load_model_for_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--base", action="store_true", help="evaluate the base model without LoRA")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--conditions", default="full,textonly")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.base != bool(args.ckpt), "give exactly one of --ckpt and --base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = MCDataset(args.data).items
    if args.base:
        model, processor = load_base_model(args.model_name)
        #################### Evaluation: full input and text-only input ####################
        results = {"ckpt": "base"}
    else:
        model, processor, ckpt = load_model_for_eval(args.model_name, args.ckpt)
        results = {"ckpt": args.ckpt, "epoch": ckpt.get("epoch"), "val_accuracy": ckpt.get("val_accuracy")}

    for cond in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        assert cond in ("full", "textonly"), cond
        print(f"\n=== {cond} ===", flush=True)
        results[cond] = evaluate(model, processor, items, args.n_frames, device, textonly=(cond == "textonly"))
        print(f"  {cond}: {100 * results[cond]['accuracy']:.2f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
