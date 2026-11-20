#!/usr/bin/env python3
"""Build the disjoint splits from the blind-filtered training and evaluation manifests.

    python tools/prepare_splits.py --train blindfilt_ssv2_train_hard.jsonl --val blindfilt_ssv2_val_hard.jsonl --out_dir data

Writes to --out_dir:
    val_select.jsonl       200 items of the evaluation split, for checkpoint selection
    test.jsonl             the rest of the evaluation split
    train_calibpool.jsonl  the pool the calibration items are drawn from
    calib400.jsonl         400 calibration items, stratified by family
    train.jsonl            the training set, without the calibration items
"""
import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

# The repository root holds the veto package, so the scripts run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate import stratified_sample

SEED = 0


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def dump(path, items):
    with open(path, "w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{path}: {len(items)}")


######### Splits: selection, test, calibration pool, calibration set, training set #########
def build(train, val, n_calib=400, n_select=200, per_family=625):
    idx = list(range(len(val)))
    random.Random(SEED).shuffle(idx)
    select = [val[i] for i in idx[:n_select]]
    select_ids = {x["id"] for x in select}
    test = [x for x in val if x["id"] not in select_ids]

    # The calibration pool is the training set without a fixed, family-balanced subset
    # (per_family items of each family), drawn after setting aside 200 stratified items.
    # The paper's calibration set is reproduced by exactly this procedure.
    first = {x["id"] for x in stratified_sample(train, 200, SEED)}
    byfam = collections.defaultdict(list)
    for x in train:
        if x["id"] not in first:
            byfam[x["dim"]].append(x)
    held = set()
    for fam in sorted(byfam):
        items = byfam[fam]
        random.Random(f"{SEED}:{fam}").shuffle(items)
        held |= {x["id"] for x in items[:per_family]}
    pool = [x for x in train if x["id"] not in held]

    calib = stratified_sample(pool, n_calib, SEED)
    calib_ids = {x["id"] for x in calib}
    train_final = [x for x in train if x["id"] not in calib_ids]
    return dict(val_select=select, test=test, train_calibpool=pool, calib400=calib, train=train_final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="blind-filtered training manifest")
    ap.add_argument("--val", required=True, help="blind-filtered evaluation manifest")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, items in build(load(args.train), load(args.val)).items():
        dump(os.path.join(args.out_dir, f"{name}.jsonl"), items)


if __name__ == "__main__":
    main()
