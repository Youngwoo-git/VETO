#!/usr/bin/env python3
"""Calibration: estimate the text-only subspace of the base model, once per VideoLLM.

Draws `--n` calibration items from the training pool, stratified by the `dim` field,
runs the base model on their text-only input, and keeps for every adapted module the
leading eigenvectors of the uncentered second moment of its input that carry a fraction
`--rho` of the spectrum. The result is an npz with one orthonormal basis per module.

    python tools/calibrate.py --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --pool data/train_calibpool.jsonl --n 400 --seed <seed> --rho 0.95 --out projectors.npz
"""
import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# The repository root holds the veto package, so the scripts run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veto.data import MCDataset, build_textonly_inputs
from veto.model import find_lora_modules, load_model_for_training
from veto.subspace import SecondMomentCollector, coverage, save_projectors, subspaces


################ Calibration items: stratified sample of the training pool #################
def stratified_sample(items, n, seed):
    """n items with an equal share per `dim`. A family with too few items gives all it
    has, and the shortfall is spread over the others."""
    by = collections.defaultdict(list)
    for it in items:
        by[it["dim"]].append(it)
    dims = sorted(by)
    rng = random.Random(seed)
    base, extra = divmod(n, len(dims))
    extra_dims = set(rng.sample(dims, extra))
    want = {d: base + (1 if d in extra_dims else 0) for d in dims}
    deficit = 0
    for d in dims:
        if len(by[d]) < want[d]:
            deficit += want[d] - len(by[d])
            want[d] = len(by[d])
    while deficit > 0:
        room = [d for d in dims if len(by[d]) > want[d]]
        assert room, "the pool is smaller than n"
        for d in room:
            if deficit == 0:
                break
            want[d] += 1
            deficit -= 1
    out = []
    for d in dims:
        out += rng.sample(by[d], want[d])
    rng.shuffle(out)
    return out


######################## Text-only forward passes of the base model ########################
@torch.no_grad()
def run_textonly(model, processor, items, device, tag):
    t0, ok = time.time(), 0
    for i, it in enumerate(items):
        try:
            inputs = build_textonly_inputs(processor, it, device)
            model(**{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)})
            ok += 1
        except Exception as e:
            print(f"  [{tag}] skip {it.get('id', '?')}: {e}", flush=True)
            torch.cuda.empty_cache()
        if (i + 1) % 200 == 0:
            print(f"  [{tag}] {i + 1}/{len(items)} ok={ok} {time.time() - t0:.0f}s", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--pool", nargs="+", help="training-pool manifests to sample from")
    ap.add_argument("--calib_data", nargs="+", help="use these items as they are instead of sampling")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--out", required=True, help="projector npz")
    ap.add_argument("--save_calib", default=None, help="write the sampled calibration manifest here")
    ap.add_argument("--heldout", nargs="+", default=None, help="manifests for held-out coverage")
    ap.add_argument("--heldout_n", type=int, default=1000)
    ap.add_argument("--cov_device", default=None, help="where the second moments are kept. cpu for very large models")
    args = ap.parse_args()
    assert bool(args.pool) != bool(args.calib_data), "give exactly one of --pool and --calib_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ################################## Calibration items ###################################
    if args.calib_data:
        items = MCDataset(args.calib_data).items
    else:
        items = stratified_sample(MCDataset(args.pool).items, args.n, args.seed)
    if args.save_calib:
        with open(args.save_calib, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    ##################### Base model with the adapted modules located ######################
    # The LoRA factors B start at zero, so the wrapped model is the base model.
    model, processor = load_model_for_training(args.model_name)
    model.eval()
    model.gradient_checkpointing_disable()
    lora_mods = find_lora_modules(model)
    print(f"{len(lora_mods)} adapted modules, {len(items)} calibration items", flush=True)

    ########################## Identifying the shortcut subspace ###########################
    args.cov_device = args.cov_device or device
    col = SecondMomentCollector(lora_mods, args.cov_device)
    run_textonly(model, processor, items, device, "calib")
    Us = subspaces(col.second_moment(), args.rho)
    col.remove()
    del col
    torch.cuda.empty_cache()
    save_projectors(args.out, Us)
    ranks = np.array([u.shape[1] for u in Us.values()])
    print(f"rank mean {ranks.mean():.1f} (min {ranks.min()}, max {ranks.max()}) -> {args.out}", flush=True)

    ##################### Held-out coverage of the subspace (optional) #####################
    if args.heldout:
        held_all = MCDataset(args.heldout).items
        assert not ({x["id"] for x in items} & {x["id"] for x in held_all}), "held-out overlaps calibration"
        held = random.Random(0).sample(held_all, min(args.heldout_n, len(held_all)))
        col = SecondMomentCollector(lora_mods, args.cov_device)
        run_textonly(model, processor, held, device, "heldout")
        cov = coverage(Us, col.second_moment())
        col.remove()
        print(f"held-out coverage {np.mean(list(cov.values())):.4f} on {len(held)} items", flush=True)


if __name__ == "__main__":
    main()
