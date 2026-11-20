#!/usr/bin/env python3
"""Accuracy with the full input and from the text alone for the other VideoLLMs.

    python tools/evaluate_backbones.py --family qwen3vl --model_path Qwen/Qwen3-VL-8B-Instruct \
        --ckpt out/qwen3vl8b_veto_s<seed>/best.pt --data test.jsonl --out result.json
Without --ckpt the base model is evaluated.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

# The repository root holds the veto package, so the scripts run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veto.backbones import LORA_SPEC, load_backbone
from veto.data import MCDataset
from veto.eval import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(LORA_SPEC))
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--conditions", default="full,textonly")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = MCDataset(args.data).items
    model, processor = load_backbone(args.family, args.model_path)
    ###################### Evaluation: full input and text-only input ######################
    results = {"family": args.family, "model_path": args.model_path, "ckpt": args.ckpt or "base"}
    if args.ckpt:
        from peft import LoraConfig, get_peft_model
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = get_peft_model(model, LoraConfig(
            r=ck["config"]["lora_rank"], lora_alpha=ck["config"]["lora_alpha"],
            target_modules=ck["lora_target"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        res = model.load_state_dict(ck["lora_state"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:3]
        model.eval()
        results.update(epoch=ck.get("epoch"), val_accuracy=ck.get("val_accuracy"))

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
