"""Loaders and LoRA targets for the VideoLLMs other than Qwen2.5-VL.

Qwen2.5-VL (3B, 7B, 32B) is handled by veto/model.py and the scripts without a suffix.
"""
from __future__ import annotations

import os

import torch

################################ LoRA targets per VideoLLM #################################
# family -> names of the adapted linear layers and the part of the model they are taken from
LORA_SPEC = {
    "qwen3vl": dict(suffixes=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"),
                    scope="language_model"),
    "llavaov": dict(suffixes=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"),
                    scope="language_model"),
    "gemma4": dict(suffixes=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"),
                   scope="language_model"),
    "molmo2": dict(suffixes=("att_proj", "attn_out", "ff_proj"), scope=None),
}
# families whose processor accepts several videos in one call
MICRO_BATCH_FAMILIES = ("qwen3vl", "llavaov", "gemma4")
_CONTAINERS = ("layers", "blocks")


################################## Processors and loaders ##################################
class _FixedFramesProcessor:
    """Passes already sampled frames to a processor that would otherwise resample them."""

    as_array = False

    def __init__(self, proc):
        self._p = proc

    def __getattr__(self, k):
        return getattr(self._p, k)

    def __call__(self, text=None, videos=None, **kw):
        if videos is None:
            return self._p(text=text, **kw)
        import numpy as np
        metas = [{"fps": 2.0, "total_num_frames": len(v), "frames_indices": list(range(len(v))),
                  "duration": len(v) / 2.0} for v in videos]
        if self.as_array:
            videos = [np.stack([np.asarray(f.convert("RGB")) for f in v]) for v in videos]
        kw.setdefault("do_sample_frames", False)
        kw.setdefault("video_metadata", metas)
        return self._p(text=text, videos=videos, **kw)


class _Molmo2Processor(_FixedFramesProcessor):
    as_array = True


def _load_hf(path, trust=False):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    kw = dict(dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust)
    attn = os.environ.get("ATTN_IMPL", "sdpa")
    if attn:
        kw["attn_implementation"] = attn
    model = AutoModelForImageTextToText.from_pretrained(path, **kw).eval()
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust)
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


def load_backbone(family, path):
    assert family in LORA_SPEC, family
    if family == "molmo2":
        model, proc = _load_hf(path, trust=True)
        return model, _Molmo2Processor(proc)
    if family == "gemma4":   # needs transformers >= 5.5
        model, proc = _load_hf(path)
        return model, _FixedFramesProcessor(proc)
    return _load_hf(path)


############# Adapted modules of the language model, the ones VETO constrains ##############
def _layer_index(parts):
    for i, p in enumerate(parts):
        if p in _CONTAINERS and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _in_spec(name, family):
    spec = LORA_SPEC[family]
    if name.split(".")[-1] not in spec["suffixes"]:
        return False
    return not spec["scope"] or spec["scope"] in name


def lora_target_names(model, family):
    return [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and _in_spec(n, family)]


def adapted_linears(model, family):
    """ "L<layer>_<name>" -> the linear layer of the base model, for calibration."""
    mods = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not _in_spec(name, family):
            continue
        layer = _layer_index(name.split("."))
        if layer is not None:
            mods[f"L{layer}_{name.split('.')[-1]}"] = {"module": module}
    assert mods, f"{family}: no adapted module found"
    return mods


def find_lora_modules(model, family):
    """ "L<layer>_<name>" -> the LoRA-wrapped module of the PEFT model."""
    mods = {}
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "base_layer")):
            continue
        if not _in_spec(name, family):
            continue
        layer = _layer_index(name.split("."))
        if layer is not None:
            mods[f"L{layer}_{name.split('.')[-1]}"] = {
                "module": module, "param_A_name": f"{name}.lora_A.default.weight"}
    assert mods, f"{family}: no LoRA module found"
    return mods


############# Training input: full video-text input, loss on the answer letter #############
def build_train_inputs(processor, items, n_frames, device, pad_right):
    """Inputs and labels for one or more items. The loss is taken on the answer letter,
    located as the part of the tokenized conversation that follows the prompt."""
    from veto.data import build_prompt, item_frames

    tok = processor.tokenizer
    prev_side = tok.padding_side
    if pad_right:
        tok.padding_side = "right"
    frames_list, texts_full, n_prompts = [], [], []
    try:
        for item in items:
            frames = item_frames(item, n_frames)
            answer = item.get("answer", "").strip().upper()[:1]
            user = {"role": "user", "content": [{"type": "video", "video": frames},
                                                {"type": "text", "text": build_prompt(item)}]}
            text_prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
            text_full = processor.apply_chat_template(
                [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                tokenize=False, add_generation_prompt=False)
            ip = processor(text=[text_prompt], videos=[frames], return_tensors="pt")["input_ids"][0]
            iff = processor(text=[text_full], videos=[frames], return_tensors="pt")["input_ids"][0]
            L = min(len(ip), len(iff))
            diverge = (ip[:L] != iff[:L]).nonzero()
            n_prompt = int(diverge[0]) if len(diverge) else L
            assert answer in tok.decode(iff[n_prompt:].tolist()), f"answer not in the labels ({item.get('id')})"
            frames_list.append(frames)
            texts_full.append(text_full)
            n_prompts.append(n_prompt)
        inputs = processor(text=texts_full, videos=frames_list, return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev_side
    labels = inputs["input_ids"].clone()
    for r, n_prompt in enumerate(n_prompts):
        labels[r, :n_prompt] = -100
    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = -100
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs, labels.to(device)
