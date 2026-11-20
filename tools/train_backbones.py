#!/usr/bin/env python3
"""SFT and VETO fine-tuning for Qwen3-VL, LLaVA-OneVision, Molmo2, and Gemma 4.

The constraint is the one of train.py: with `--projectors`, the LoRA input factor A of
every adapted module is projected off the text-only subspace after initialization and
after every optimizer step. This trainer uses a constant learning rate, no gradient
clipping, and no LoRA dropout.

    python tools/train_backbones.py --family qwen3vl --model_path Qwen/Qwen3-VL-8B-Instruct \
        --train_data train.jsonl --val_data val_select.jsonl \
        --projectors projectors_qwen3vl8b.npz --out_dir out/qwen3vl8b_veto_s<seed> --micro_batch 4
"""
import argparse
import glob
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# The repository root holds the veto package, so the scripts run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veto.backbones import (LORA_SPEC, MICRO_BATCH_FAMILIES, build_train_inputs,
                            find_lora_modules, load_backbone, lora_target_names)
from veto.data import MCDataset
from veto.eval import evaluate
from veto.subspace import load_projectors, project_lora, subspace_fraction

LORA_RANK, LORA_ALPHA = 16, 32


def epoch_of(path):
    return int(path.split("_")[-1].split(".")[0])


def lora_state(model):
    return {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_ckpt(model, path, epoch, val_acc, targets):
    torch.save({"epoch": epoch, "lora_state": lora_state(model), "lora_target": targets,
                "config": {"lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA},
                "val_accuracy": val_acc}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(LORA_SPEC))
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_data", nargs="+", required=True)
    ap.add_argument("--val_data", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--projectors", default=None, help="npz from calibrate_backbones.py. Omit for SFT")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--early_stop", type=int, default=5, help="patience in epochs")
    ap.add_argument("--grad_accum", type=int, default=8, help="items per update")
    ap.add_argument("--micro_batch", type=int, default=1,
                    help="items per forward pass. Molmo2 always uses 1")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_items", type=int, default=200)
    ap.add_argument("--partial_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert args.grad_accum % max(args.micro_batch, 1) == 0, "grad_accum must be a multiple of micro_batch"
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MB = args.micro_batch if args.family in MICRO_BATCH_FAMILIES else 1

    ######################################### Data #########################################
    train_items = MCDataset(args.train_data).items
    val_items = MCDataset(args.val_data).items[:args.val_items]

    #################################### Model and LoRA ####################################
    model, processor = load_backbone(args.family, args.model_path)
    targets = lora_target_names(model, args.family)
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=targets, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM"))
    print(f"[{args.family}] {len(targets)} adapted modules, train {len(train_items)}, val {len(val_items)}")
    if args.family == "molmo2":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    ########### VETO: load the shortcut subspace and project A at initialization ###########
    projectors, lora_mods = None, None
    if args.projectors:
        projectors = load_projectors(args.projectors)
        lora_mods = find_lora_modules(model, args.family)
        missing = [k for k in lora_mods if k not in projectors]
        assert not missing, f"projectors missing for {missing[:5]}"
        project_lora(model, projectors, lora_mods)
        print(f"VETO: subspace fraction after init {subspace_fraction(model, projectors, lora_mods):.6f}", flush=True)

    def constrain():
        if projectors is not None:
            project_lora(model, projectors, lora_mods)

    def validate():
        res = evaluate(model, processor, val_items, args.n_frames, device)
        model.train()
        return res

    ######################################## Resume ########################################
    best_val, patience, start_epoch, skip_items = -1.0, 0, 0, 0
    partial_path = os.path.join(args.out_dir, "partial.pt")
    if args.resume:
        eps = sorted(glob.glob(os.path.join(args.out_dir, "epoch_*.pt")), key=epoch_of)
        if eps:
            hist = [torch.load(p, map_location="cpu", weights_only=False) for p in eps]
            res = model.load_state_dict(hist[-1]["lora_state"], strict=False)
            assert not res.unexpected_keys
            start_epoch = hist[-1]["epoch"]
            best_val = max(c["val_accuracy"] for c in hist)
            best_ep = min(c["epoch"] for c in hist if c["val_accuracy"] == best_val)
            patience = start_epoch - best_ep
            print(f"resume from epoch {start_epoch}: best {best_val:.3f} (epoch {best_ep}), "
                  f"patience {patience}/{args.early_stop}", flush=True)
        if os.path.exists(partial_path):
            pc = torch.load(partial_path, map_location="cpu", weights_only=False)
            if (pc.get("epoch") == start_epoch + 1 and pc["items_done"] % args.grad_accum == 0
                    and pc["items_done"] < len(train_items)):
                res = model.load_state_dict(pc["lora_state"], strict=False)
                assert not res.unexpected_keys
                skip_items = pc["items_done"]
                constrain()
                print(f"partial resume: epoch {start_epoch + 1}, {skip_items} items done", flush=True)

    if start_epoch == 0:
        print(f"baseline val accuracy: {validate()['accuracy']:.3f}", flush=True)

    #################################### Training loop #####################################
    gstep = 0
    for epoch in range(start_epoch, args.epochs):
        indices = list(range(len(train_items)))
        random.seed(args.seed + epoch)
        random.shuffle(indices)
        ep_loss, ep_n, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, idx in enumerate(indices):
            if i < skip_items:
                continue
            if not (i % MB == MB - 1 or i == len(indices) - 1):
                continue
            chunk = [train_items[j] for j in indices[i - (i % MB): i + 1]]
            ############## Forward and backward on the full video-text input ###############
            try:
                inputs, labels = build_train_inputs(processor, chunk, args.n_frames, device, pad_right=MB > 1)
                tens = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                try:
                    loss = model(**tens, labels=labels, use_cache=False).loss
                except TypeError:
                    loss = model(**tens, labels=labels).loss
                loss = loss * len(chunk)
                (loss / args.grad_accum).backward()
                ep_loss += loss.item()
                ep_n += len(chunk)
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"  batch skip ({type(e).__name__}: {str(e)[:100]})", flush=True)
            ################## Optimizer step, then VETO projection of A ###################
            if (i + 1) % args.grad_accum == 0 or i == len(indices) - 1:
                opt.step()
                opt.zero_grad(set_to_none=True)
                constrain()
                gstep += 1
                if args.partial_every and gstep % args.partial_every == 0 and i < len(indices) - 1:
                    tmp = partial_path + ".tmp"
                    torch.save({"epoch": epoch + 1, "items_done": i + 1, "lora_state": lora_state(model)}, tmp)
                    os.replace(tmp, partial_path)
            if (i + 1) % 200 == 0:
                print(f"  item {i + 1}/{len(indices)} loss {ep_loss / max(ep_n, 1):.4f} "
                      f"({(time.time() - t0) / (i + 1):.2f}s/item)", flush=True)

        ##################### Validation, checkpoints, early stopping ######################
        val = validate()
        extra = (f" | subspace fraction {subspace_fraction(model, projectors, lora_mods):.6f}"
                 if projectors is not None else "")
        print(f"--- epoch {epoch + 1}/{args.epochs}: train loss {ep_loss / max(ep_n, 1):.4f}, "
              f"val accuracy {val['accuracy']:.3f}{extra} | {time.time() - t0:.0f}s", flush=True)
        save_ckpt(model, os.path.join(args.out_dir, f"epoch_{epoch + 1}.pt"), epoch + 1, val["accuracy"], targets)
        skip_items = 0
        if os.path.exists(partial_path):
            os.remove(partial_path)
        if val["accuracy"] > best_val:
            best_val, patience = val["accuracy"], 0
            save_ckpt(model, os.path.join(args.out_dir, "best.pt"), epoch + 1, best_val, targets)
            print(f"  new best: {best_val:.3f}", flush=True)
        else:
            patience += 1
            print(f"  patience: {patience}/{args.early_stop}", flush=True)
            if patience >= args.early_stop:
                print(f"  early stop at epoch {epoch + 1}")
                break
    print(f"done. best val accuracy {best_val:.3f}")


if __name__ == "__main__":
    main()
