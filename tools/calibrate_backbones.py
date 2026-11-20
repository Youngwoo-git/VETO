#!/usr/bin/env python3
"""Calibration for Qwen3-VL, LLaVA-OneVision, Molmo2, and Gemma 4.

The calibration items are those drawn once by calibrate.py (`--save_calib`), so every
VideoLLM is calibrated on the same 400 prompts.

    python tools/calibrate_backbones.py --family qwen3vl --model_path Qwen/Qwen3-VL-8B-Instruct \
        --calib_data calib400.jsonl --out projectors_qwen3vl8b.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# The repository root holds the veto package, so the scripts run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate import run_textonly
from veto.backbones import LORA_SPEC, adapted_linears, load_backbone
from veto.data import MCDataset
from veto.subspace import SecondMomentCollector, save_projectors, subspaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(LORA_SPEC))
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--calib_data", nargs="+", required=True)
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cov_device", default=None)
    args = ap.parse_args()

    ########################### Calibration items and base model ###########################
    items = MCDataset(args.calib_data).items
    model, processor = load_backbone(args.family, args.model_path)
    mods = adapted_linears(model, args.family)
    print(f"[{args.family}] {len(mods)} adapted modules, {len(items)} calibration items", flush=True)

    ########################## Identifying the shortcut subspace ###########################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    col = SecondMomentCollector(mods, args.cov_device or device)
    run_textonly(model, processor, items, device, "calib")
    Us = subspaces(col.second_moment(), args.rho)
    col.remove()
    save_projectors(args.out, Us)
    ranks = np.array([u.shape[1] for u in Us.values()])
    print(f"rank mean {ranks.mean():.1f} (min {ranks.min()}, max {ranks.max()}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
