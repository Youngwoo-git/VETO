"""Manifests, frame extraction, and input construction for multiple-choice video QA.

A manifest is a JSONL file with one item per line:
    {"id": str, "video": path, "question": str, "options": [str, ...], "answer": "A", "dim": str}
`video` is a video file or a directory of pre-extracted frames named f*.png.
`dim` is the family label used to stratify the calibration sample.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

MAX_FRAME_SIDE = 448   # frames are downscaled so that the longer side is at most this
IM_START_ID = 151644   # <|im_start|> in the Qwen tokenizer


################################# Multiple-choice manifest #################################
class MCDataset:
    def __init__(self, jsonl_paths):
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        self.items = []
        for p in jsonl_paths:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.items.append(json.loads(line))
        print(f"MCDataset: {len(self.items)} items from {len(jsonl_paths)} files")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


##################################### Frame extraction #####################################
def _resize_cap(img):
    w, h = img.size
    m = max(w, h)
    if m <= MAX_FRAME_SIDE:
        return img
    scale = MAX_FRAME_SIDE / m
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))))


def _blocklisted(video_path):
    """Videos listed in the file named by $DECORD_BLOCKLIST skip decord and go to PyAV."""
    bl = os.environ.get("DECORD_BLOCKLIST")
    if bl and os.path.exists(bl):
        with open(bl) as f:
            return video_path in {l.strip() for l in f if l.strip()}
    return False


def extract_frames(video_path, n_frames=16, frame_indices=None):
    """Return a list of PIL images, sampled uniformly unless `frame_indices` is given."""
    from PIL import Image

    if os.path.isdir(video_path):
        fs = sorted(glob.glob(os.path.join(video_path, "f*.png")))
        if not fs:
            raise RuntimeError(f"no frames in dir: {video_path}")
        return [Image.open(f).convert("RGB") for f in fs]

    try:
        if _blocklisted(video_path):
            raise RuntimeError("blocklisted for decord")
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        if frame_indices is not None:
            indices = [min(f, total - 1) for f in frame_indices]
        else:
            indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
        frames_np = vr.get_batch(indices).asnumpy()
        return [_resize_cap(Image.fromarray(f)) for f in frames_np]
    except Exception:
        # Streaming fallback with PyAV, which never holds the whole video in memory.
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        total = int(stream.frames or 0)
        if total <= 0:
            total = sum(1 for _ in container.decode(video=0))
            container.close()
            container = av.open(video_path)
        if total == 0:
            raise RuntimeError(f"no frames decoded: {video_path}")
        if frame_indices is not None:
            indices = [min(f, total - 1) for f in frame_indices]
        else:
            indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
        want, last, got = set(indices), max(indices), {}
        for i, f in enumerate(container.decode(video=0)):
            if i in want:
                got[i] = _resize_cap(f.to_image())
            if i >= last:
                break
        container.close()
        if not got:
            raise RuntimeError(f"no frames decoded: {video_path}")
        fallback = got[max(got)]
        return [got.get(i, fallback) for i in indices]


def item_frames(item, n_frames):
    return extract_frames(item["video"], n_frames, item.get("oracle_frames"))


################################# Prompt and model inputs ##################################
def build_prompt(item):
    options = item.get("options", [])
    if not options:
        return item["question"]
    opts_text = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    return f"{item['question']}\n{opts_text}\nAnswer with the letter only."


def _to_device(inputs, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def build_eval_inputs(processor, item, device, frames=None):
    """Inputs up to the generation prompt. `frames=None` gives the text-only input,
    which is the full input with the video removed."""
    prompt = build_prompt(item)
    if frames is None:
        content = [{"type": "text", "text": prompt}]
    else:
        content = [{"type": "video", "video": frames}, {"type": "text", "text": prompt}]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    if frames is None:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
    return _to_device(inputs, device)


################# Text-only input: question and options without the video ##################
def build_textonly_inputs(processor, item, device):
    return build_eval_inputs(processor, item, device, frames=None)


############# Training input: full video-text input, loss on the answer letter #############
def build_train_inputs(processor, items, n_frames, device, frames_list=None):
    """Inputs and labels for a micro-batch. The loss is taken on the answer only."""
    if frames_list is None:
        frames_list = [item_frames(it, n_frames) for it in items]
    texts = []
    for it, frames in zip(items, frames_list):
        messages = [
            {"role": "user", "content": [{"type": "video", "video": frames},
                                         {"type": "text", "text": build_prompt(it)}]},
            {"role": "assistant", "content": it.get("answer", "")},
        ]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False))
    tok = processor.tokenizer
    prev_side = tok.padding_side
    tok.padding_side = "right"
    try:
        inputs = processor(text=texts, videos=frames_list, return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev_side
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    for r in range(input_ids.shape[0]):
        row = input_ids[r].tolist()
        positions = [i for i, x in enumerate(row) if x == IM_START_ID]
        # the answer starts after the last <|im_start|>, "assistant", and the newline
        cut = positions[-1] + 2 if positions else int(len(row) * 0.8)
        labels[r, :cut] = -100
    labels[inputs["attention_mask"] == 0] = -100
    return _to_device(inputs, device), labels.to(device)
