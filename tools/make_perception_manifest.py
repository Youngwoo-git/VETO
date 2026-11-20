#!/usr/bin/env python3
"""Evaluation manifest for Perception Test, in the format of the SSv2 manifests
{id, video, question, options, answer (letter), task}. Items whose video is not found
under --root are left out.

    python tools/make_perception_manifest.py --root perception_test --out_dir data

--root is the downloaded Perception Test, with mc_question_train.json and videos/<video_id>.mp4.
"""
import json, sys, os, re, glob, ast, argparse
OUT = None
def dump(name, rows):
    ok = [r for r in rows if os.path.exists(r["video"])]; miss = len(rows) - len(ok)
    p = f"{OUT}/{name}.jsonl"; open(p, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ok))
    from collections import Counter; print(f"{name}: {len(ok)} items ({miss} missing video) → {p}"); print("  by task:", dict(Counter(r.get("task") for r in ok)))
def letter(i): return chr(65 + i)


def perception(D):
    """Perception Test train MC: every item with a video, and a 2,000-item sample (seed 42)."""
    P=f"{D}/mc_question_train.json"; V=f"{D}/videos"; rows=[]
    for vk, v in json.load(open(P)).items():
        for q in v.get("mc_question", []):
            opts=q["options"]; opts=ast.literal_eval(opts) if isinstance(opts,str) else list(opts); a=int(q["answer_id"])
            if a>=len(opts): continue
            rows.append({"id": f"ptest_{vk}_{q['id']}", "video": f"{V}/{vk}.mp4", "question": q["question"].strip(), "options": [str(o) for o in opts], "answer": letter(a), "task": q.get("area"), "reasoning": q.get("reasoning")})
    dump("perception_full", rows)
    import random; ok=[r for r in rows if os.path.exists(r["video"])]; random.Random(42).shuffle(ok); dump("perception", ok[:2000])

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True, help="downloaded Perception Test directory")
ap.add_argument("--out_dir", default="data"); a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True); OUT = a.out_dir; perception(a.root)
