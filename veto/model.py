"""Model loading and LoRA module lookup for Qwen2.5-VL."""
from __future__ import annotations

import os

import torch

####################################### LoRA targets #######################################
# down_proj is left untouched. With gate_proj and up_proj constrained, its input at the
# text positions is unchanged, and its 18,944-dimensional input makes calibration costly.
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
# These names are matched by suffix, so gate_proj and up_proj of the vision encoder MLP are
# adapted as well. The constraint applies to the modules of the language model only.
MAX_PIXELS = 360 * 420


######################### Loading Qwen2.5-VL with and without LoRA #########################
def _load_backbone(model_name):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("ATTN_IMPL", "sdpa"), device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name, max_pixels=MAX_PIXELS)
    try:
        if hasattr(processor, "video_processor"):
            processor.video_processor.max_pixels = MAX_PIXELS
        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = MAX_PIXELS
    except Exception:
        pass
    return model, processor


def load_base_model(model_name):
    model, processor = _load_backbone(model_name)
    model.eval()
    return model, processor


def load_model_for_training(model_name, lora_rank=16, lora_alpha=32, lora_dropout=0.05):
    from peft import LoraConfig, get_peft_model

    model, processor = _load_backbone(model_name)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=LORA_TARGET,
                     lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)")
    return model, processor


def load_model_for_eval(model_name, ckpt_path, lora_rank=16, lora_alpha=32):
    from peft import LoraConfig, get_peft_model

    model, processor = _load_backbone(model_name)
    for p in model.parameters():
        p.requires_grad = False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    target = ckpt.get("lora_target") or LORA_TARGET
    cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target,
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    result = model.load_state_dict(ckpt["lora_state"], strict=False)
    assert not result.unexpected_keys, result.unexpected_keys[:3]
    print(f"LoRA state loaded: {len(ckpt['lora_state'])} tensors "
          f"(epoch {ckpt.get('epoch', '?')}, val acc {ckpt.get('val_accuracy', '?')})")
    model.eval()
    return model, processor, ckpt


############# Adapted modules of the language model, the ones VETO constrains ##############
def find_lora_modules(model):
    """Map "L<layer>_<module type>" to each LoRA-wrapped module of the language model."""
    modules = {}
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "base_layer")):
            continue
        parts = name.split(".")
        mod_type = parts[-1]
        if mod_type not in LORA_TARGET:
            continue
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is None:
            continue
        modules[f"L{layer_idx}_{mod_type}"] = {
            "module": module,
            "param_A_name": f"{name}.lora_A.default.weight",
            "layer": layer_idx,
            "type": mod_type,
        }
    return modules


def lora_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
