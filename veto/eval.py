"""Letter-logit multiple-choice evaluation."""
import torch

from veto.data import build_eval_inputs, item_frames

LETTERS = "ABCDEFG"


################# Multiple-choice prediction from the option-letter logits #################
def option_token_ids(tokenizer):
    ids = []
    for letter in LETTERS:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if toks:
            ids.append(toks[0])
    return ids


@torch.no_grad()
def predict(model, processor, item, n_frames, device, textonly=False):
    frames = None if textonly else item_frames(item, n_frames)
    inputs = build_eval_inputs(processor, item, device, frames)
    logits = model(**{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}).logits
    n_opt = len(item.get("options", []))
    option_logits = logits[0, -1, option_token_ids(processor.tokenizer)[:n_opt]]
    return chr(65 + option_logits.argmax().item())


################### Accuracy with the full input or from the text alone ####################
@torch.no_grad()
def evaluate(model, processor, items, n_frames, device, textonly=False, log_every=50):
    """Items that fail to decode are skipped and recorded as None."""
    model.eval()
    correct, total, shown = 0, 0, 0
    item_correct, item_pred = [], []
    for i, item in enumerate(items):
        try:
            pred = predict(model, processor, item, n_frames, device, textonly)
            ok = int(pred == item.get("answer", "").strip().upper()[:1])
            correct += ok
            total += 1
            item_correct.append(ok)
            item_pred.append(pred)
        except Exception as e:
            if shown < 3:
                print(f"    skip {item.get('id', '?')}: {type(e).__name__}: {str(e)[:150]}", flush=True)
                shown += 1
            torch.cuda.empty_cache()
            item_correct.append(None)
            item_pred.append(None)
        if log_every and (i + 1) % log_every == 0:
            print(f"    {i + 1}/{len(items)} acc {correct / max(total, 1):.3f}", flush=True)
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total,
            "item_id": [it.get("id") for it in items],
            "item_pred": item_pred, "item_correct": item_correct}
