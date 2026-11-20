#!/usr/bin/env python3
"""LoRA fine-tuning of Qwen2.5-VL on multiple-choice video QA, as SFT or with VETO.

With `--projectors`, the LoRA input factor A of every adapted module is projected off the
text-only subspace U after initialization and after every optimizer step,

    A <- A - (A U) U^T,

so the update B A never acts on the directions that the text-only input occupies.
Without `--projectors` the run is standard SFT.

    python tools/train.py --train_data train.jsonl --val_data val_select.jsonl \
        --projectors projectors.npz --out_dir out/veto_s<seed> --seed <seed> --micro_batch 4
"""
import argparse
import glob
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# The repository root holds the veto package, so the scripts run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veto.data import MCDataset, build_train_inputs, extract_frames
from veto.eval import evaluate
from veto.model import LORA_TARGET, find_lora_modules, load_model_for_training, lora_state
from veto.subspace import load_projectors, project_lora, subspace_fraction

VAL_ITEMS = 200   # checkpoints are selected on the first 200 items of --val_data


def epoch_of(path):
    return int(path.split("_")[-1].split(".")[0])


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.grad_accum % args.micro_batch == 0, "grad_accum must be a multiple of micro_batch"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ######################################### Data #########################################
    train_ds = MCDataset(args.train_data)
    val_items = MCDataset(args.val_data).items[:VAL_ITEMS]

    #################################### Model and LoRA ####################################
    model, processor = load_model_for_training(args.model_name, args.lora_rank, args.lora_alpha)
    lora_mods = find_lora_modules(model)
    print(f"adapted modules: {len(lora_mods)}")
    os.makedirs(args.out_dir, exist_ok=True)

    ########### VETO: load the shortcut subspace and project A at initialization ###########
    projectors = {}
    if args.projectors:
        projectors = load_projectors(args.projectors)
        missing = [k for k in lora_mods if k not in projectors]
        assert not missing, f"projectors missing for {missing[:5]}"
        project_lora(model, projectors, lora_mods)
        print(f"VETO: {len(projectors)} projectors from {args.projectors}, "
              f"subspace fraction after init {subspace_fraction(model, projectors, lora_mods):.6f}")
    else:
        print("SFT: no constraint")

    def constrain():
        if projectors:
            project_lora(model, projectors, lora_mods)

    baseline = evaluate(model, processor, val_items, args.n_frames, device)
    print(f"baseline val accuracy: {baseline['accuracy']:.3f}")

    ################################ Optimizer and schedule ################################
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_ds) * args.epochs) // args.grad_accum
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_acc, patience, global_step, start_epoch = -1, 0, 0, 0
    print(f"train {len(train_ds)}, val {len(val_items)}, epochs {args.epochs}, "
          f"items per update {args.grad_accum}, total updates {total_steps}")

    ######################################## Resume ########################################
    # --- resume from the last epoch checkpoint, and from a partial one inside the next epoch.
    #     The LoRA weights and the schedule are restored. The optimizer moments start afresh.
    partial_path = os.path.join(args.out_dir, "partial.pt")
    skip_items = 0
    if args.resume:
        eps = sorted(glob.glob(os.path.join(args.out_dir, "epoch_*.pt")), key=epoch_of)
        if eps:
            hist = [(epoch_of(p), torch.load(p, map_location="cpu")) for p in eps]
            last_ep, last_ck = hist[-1]
            res = model.load_state_dict(last_ck["lora_state"], strict=False)
            assert not res.unexpected_keys, res.unexpected_keys[:3]
            best_val_acc = max(ck["val_accuracy"] for _, ck in hist)
            best_ep = min(ep for ep, ck in hist if ck["val_accuracy"] == best_val_acc)
            start_epoch = last_ep
            patience = last_ep - best_ep
            global_step = start_epoch * (len(train_ds) // args.grad_accum)
            for _ in range(global_step):
                scheduler.step()
            constrain()
            print(f"resume from epoch {start_epoch}: best {best_val_acc:.3f} (epoch {best_ep}), "
                  f"patience {patience}/{args.early_stop}", flush=True)
            if patience >= args.early_stop:
                print("early stop was already reached")
                return
        if os.path.exists(partial_path):
            pc = torch.load(partial_path, map_location="cpu")
            if (pc.get("epoch") == start_epoch + 1 and pc["items_done"] % args.grad_accum == 0
                    and pc["items_done"] < len(train_ds)):
                res = model.load_state_dict(pc["lora_state"], strict=False)
                assert not res.unexpected_keys, res.unexpected_keys[:3]
                skip_items = pc["items_done"]
                k = skip_items // args.grad_accum
                global_step += k
                for _ in range(k):
                    scheduler.step()
                constrain()
                print(f"partial resume: epoch {start_epoch + 1}, {skip_items} items done", flush=True)

    #################################### Training loop #####################################
    B = args.micro_batch
    stop = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # the order of an epoch depends on the seed and the epoch alone, so a resumed run
        # sees the same order as an uninterrupted one
        indices = list(range(len(train_ds)))
        random.seed(args.seed + epoch)
        random.shuffle(indices)
        epoch_loss, epoch_n = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        # frames are decoded ahead of the GPU by background threads
        futs, pool = {}, None
        if args.prefetch:
            pool = ThreadPoolExecutor(max_workers=args.prefetch)

            def submit_upto(pos):
                for j in range(pos, min(pos + args.prefetch * 4, len(indices))):
                    jdx = indices[j]
                    if jdx not in futs:
                        it = train_ds[jdx]
                        futs[jdx] = pool.submit(extract_frames, it["video"], args.n_frames,
                                                it.get("oracle_frames"))

        for i, idx in enumerate(indices):
            if i < skip_items:
                continue
            if pool:
                submit_upto(i)
            if not (i % B == B - 1 or i == len(indices) - 1):
                continue
            chunk = indices[i - (i % B): i + 1]
            ############## Forward and backward on the full video-text input ###############
            try:
                frames_list = [futs.pop(j).result() for j in chunk] if pool else None
                items = [train_ds[j] for j in chunk]
                inputs, labels = build_train_inputs(processor, items, args.n_frames, device, frames_list)
                out = model(**{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)},
                            labels=labels)
                loss = out.loss * len(chunk)
                (loss / args.grad_accum).backward()
                epoch_loss += loss.item()
                epoch_n += len(chunk)
            except Exception as e:
                print(f"  batch skip {[train_ds[j].get('id', '?') for j in chunk]}: "
                      f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                torch.cuda.empty_cache()
                continue

            ################## Optimizer step, then VETO projection of A ###################
            if (i + 1) % args.grad_accum == 0 or i == len(indices) - 1:
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                constrain()
                global_step += 1

                if args.partial_every and global_step % args.partial_every == 0 and i < len(indices) - 1:
                    tmp = partial_path + ".tmp"
                    torch.save({"epoch": epoch + 1, "items_done": i + 1, "global_step": global_step,
                                "lora_state": lora_state(model)}, tmp)
                    os.replace(tmp, partial_path)
                if global_step % args.log_every == 0:
                    extra = (f" | subspace fraction {subspace_fraction(model, projectors, lora_mods):.4f}"
                             if projectors else "")
                    print(f"  step {global_step}/{total_steps} | loss {epoch_loss / max(epoch_n, 1):.4f} | "
                          f"lr {scheduler.get_last_lr()[0]:.2e}{extra} | {time.time() - t0:.0f}s", flush=True)

        if pool:
            pool.shutdown(wait=False)

        ##################### Validation, checkpoints, early stopping ######################
        val = evaluate(model, processor, val_items, args.n_frames, device)
        print(f"\n--- epoch {epoch + 1}/{args.epochs}: train loss {epoch_loss / max(epoch_n, 1):.4f}, "
              f"val accuracy {val['accuracy']:.3f} ({val['correct']}/{val['total']})", flush=True)
        state = lora_state(model)
        if val["accuracy"] > best_val_acc:
            best_val_acc, patience = val["accuracy"], 0
            torch.save({"epoch": epoch + 1, "lora_state": state, "val_accuracy": best_val_acc,
                        "lora_target": LORA_TARGET, "projectors": args.projectors,
                        "baseline_acc": baseline["accuracy"], "args": vars(args)},
                       os.path.join(args.out_dir, "best.pt"))
            print(f"  new best: {best_val_acc:.3f}")
        else:
            patience += 1
            print(f"  patience: {patience}/{args.early_stop}")
            stop = patience >= args.early_stop
        torch.save({"epoch": epoch + 1, "lora_state": state, "val_accuracy": val["accuracy"]},
                   os.path.join(args.out_dir, f"epoch_{epoch + 1}.pt"))
        if stop:
            print(f"  early stop at epoch {epoch + 1}")
            break
        skip_items = 0
        if os.path.exists(partial_path):
            os.remove(partial_path)

    print(f"\ndone. best val accuracy {best_val_acc:.3f} (baseline {baseline['accuracy']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", nargs="+", required=True)
    ap.add_argument("--val_data", nargs="+", required=True, help="selection set, disjoint from the test set")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--projectors", default=None, help="npz from calibrate.py. Omit for SFT")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--early_stop", type=int, default=5, help="patience in epochs")
    ap.add_argument("--grad_accum", type=int, default=8, help="items per update")
    ap.add_argument("--micro_batch", type=int, default=4, help="items per forward pass")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefetch", type=int, default=8, help="frame-decoding threads, 0 to decode inline")
    ap.add_argument("--partial_every", type=int, default=500, help="updates between partial checkpoints, 0 for none")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoints in --out_dir")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
