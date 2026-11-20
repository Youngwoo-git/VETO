#!/usr/bin/env python3
"""Blind filtering: remove the items the base model answers from the text alone.

Each item is queried `--n_perm` times under the text-only input, once in the original
option order and then with the options shuffled. An item is removed only if every query
is answered correctly.

    python tools/blind_filter.py --data train.jsonl val.jsonl --n_perm 3 --out_prefix blindfilt

Writes <out_prefix>_<name>_hard.jsonl for each input and <out_prefix>_report.json.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch


def load_model(model_name):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
    model.eval()
    return model, AutoProcessor.from_pretrained(model_name)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


######################## Text-only answer of the frozen base model #########################
@torch.no_grad()
def blind_answer(model, processor, question, options, device):
    opts_text = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    prompt = f"{question}\n{opts_text}\nAnswer with the letter only."
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    logits = model(**inputs).logits[0, -1, :].float().cpu()
    letters = "ABCDEFG"[:len(options)]
    opt_ids = [processor.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
    return letters[logits[opt_ids].argmax().item()]


##################### Filter: drop items answered from the text alone ######################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n_perm", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42, help="seed of the shuffled orders")
    ap.add_argument("--out_prefix", default="blindfilt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    model, processor = load_model(args.model_name)
    report = {"n_perm": args.n_perm, "files": {}}

    for data_path in args.data:
        items = load_jsonl(data_path)
        name = Path(data_path).stem
        print(f"\n=== {name}: {len(items)} items ===")
        dim_total, dim_blind = Counter(), Counter()
        hard_items, solvable_ids = [], []

        for i, item in enumerate(items):
            options = item.get("options", [])
            answer = item.get("answer", "").strip().upper()[:1]
            if not options or not answer:
                hard_items.append(item)
                continue
            dim = item.get("dim", "?")
            dim_total[dim] += 1

            # Ask the frozen base model from the text alone n_perm times: first in the
            # option order of the manifest, then in shuffled orders. The item is removed
            # only if every order is answered correctly, so a lucky guess under one order
            # does not remove it, and the order written in the manifest does not decide it.
            all_correct = True
            for p in range(args.n_perm):
                perm = list(range(len(options)))
                if p > 0:
                    rng.shuffle(perm)
                shuffled = [options[j] for j in perm]
                new_answer = chr(65 + perm.index(ord(answer) - ord("A")))
                if blind_answer(model, processor, item["question"], shuffled, device) != new_answer:
                    all_correct = False
                    break

            if all_correct:
                dim_blind[dim] += 1
                solvable_ids.append(item.get("id", "?"))
            else:
                hard_items.append(item)
            if (i + 1) % 100 == 0:
                so_far = sum(dim_blind.values())
                print(f"  [{i + 1}/{len(items)}] removed so far: {so_far} ({so_far / (i + 1):.1%})")
            torch.cuda.empty_cache()

        n_blind, n_total = sum(dim_blind.values()), sum(dim_total.values())
        print(f"  removed {n_blind}/{n_total} ({n_blind / max(n_total, 1):.1%}), kept {len(hard_items)}")
        report["files"][name] = {
            "total": n_total, "blind_solvable": n_blind, "hard_remaining": len(hard_items),
            "per_dim": {d: {"total": dim_total[d], "blind": dim_blind[d]} for d in dim_total},
            "solvable_ids": solvable_ids,
        }
        out_path = f"{args.out_prefix}_{name}_hard.jsonl"
        with open(out_path, "w") as f:
            for it in hard_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  saved: {out_path}")
        with open(f"{args.out_prefix}_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport: {args.out_prefix}_report.json")


if __name__ == "__main__":
    main()
