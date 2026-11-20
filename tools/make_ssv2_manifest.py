#!/usr/bin/env python3
"""Multiple-choice manifests from the Something-Something V2 labels.

An item is kept when its template belongs to one of the 16 families below. Its options are
the templates of that family, filled with the objects of the video and written in lower
case. The correct option is the template of the video. The options are shuffled per item
in a random order. The blind filter (tools/blind_filter.py) then queries every item under
several orders, so its result does not depend on this order.

    python tools/make_ssv2_manifest.py --labels ssv2/labels/train.json \
        --video_dir ssv2/videos --out ssv2_train.jsonl
    python tools/make_ssv2_manifest.py --labels ssv2/labels/validation.json \
        --video_dir ssv2/videos --out ssv2_val.jsonl

`--labels` is `train.json` or `validation.json` of the SSv2 download, a list of
{"id", "label", "template", "placeholders"}.
"""
import argparse
import json
import os
import random

QUESTION = "Which description correctly matches what happens in the video?"

########### Families: the options of an item are the templates of its family ###########
# The first templates of each family are its own, the ones a video can carry. The others
# are distractors taken from other SSv2 templates on the same objects.
FAMILIES = {
    "camera_horizontal": ["Approaching [something] with your camera", "Moving away from [something] with your camera",
                          "Turning the camera left while filming [something]", "Turning the camera right while filming [something]"],
    "camera_vertical": ["Turning the camera downwards while filming [something]", "Turning the camera upwards while filming [something]",
                        "Moving [something] away from the camera", "Moving [something] towards the camera"],
    "vertical_direction": ["Lifting [something] up completely without letting it drop down",
                           "Lifting [something] up completely, then letting it drop down",
                           "Moving [something] down", "Moving [something] up"],
    "horizontal_direction": ["Pulling [something] from left to right", "Pulling [something] from right to left",
                             "Pushing [something] from left to right", "Pushing [something] from right to left"],
    "object_relative": ["Moving [something] away from [something]", "Moving [something] closer to [something]",
                        "Moving [something] away from the camera", "Moving [something] towards the camera"],
    "relative_motion": ["Moving [something] and [something] away from each other",
                        "Moving [something] and [something] closer to each other",
                        "Moving [something] and [something] so they collide with each other",
                        "Moving [something] and [something] so they pass each other"],
    "push_outcome": ["Pushing [something] so that it almost falls off but doesn't",
                     "Pushing [something] so that it falls off the table", "Pushing [something] so that it slightly moves"],
    "poke_outcome": ["Poking [something] so it slightly moves", "Poking [something] so lightly that it doesn't or almost doesn't move",
                     "Poking [something] so that it falls over", "Poking [something] so that it spins around"],
    "spin_outcome": ["Spinning [something] so it continues spinning", "Spinning [something] that quickly stops spinning",
                     "Letting [something] roll down a slanted surface", "Rolling [something] on a flat surface"],
    "fall_style": ["[Something] falling like a feather or paper", "[Something] falling like a rock",
                   "Throwing [something] in the air and catching it", "Throwing [something] in the air and letting it fall"],
    "lift_outcome": ["Lifting up one end of [something] without letting it drop down",
                     "Lifting up one end of [something], then letting it drop down",
                     "Lifting [something] up completely without letting it drop down",
                     "Lifting [something] up completely, then letting it drop down"],
    "slant_outcome": ["Putting [something] that can't roll onto a slanted surface, so it slides down",
                      "Putting [something] that can't roll onto a slanted surface, so it stays where it is",
                      "Letting [something] roll down a slanted surface",
                      "Letting [something] roll up a slanted surface, so it rolls back down"],
    "containment": ["Pulling [something] out of [something]", "Putting [something] into [something]",
                    "Stuffing [something] into [something]", "Taking [something] out of [something]"],
    "plug_outcome": ["Plugging [something] into [something]",
                     "Plugging [something] into [something] but pulling it right out as you remove your hand",
                     "Putting [something] into [something]", "Taking [something] out of [something]"],
    "open_close": ["Closing [something]", "Opening [something]",
                   "Covering [something] with [something]", "Uncovering [something]"],
    "cover": ["Covering [something] with [something]", "Uncovering [something]",
              "Putting [something] into [something]", "Taking [something] out of [something]"],
}
# Number of own templates at the head of each list. Every template is the own template of
# one family at most, and may be a distractor in others.
OWN = {
    "camera_horizontal": 4, "camera_vertical": 4, "vertical_direction": 4, "horizontal_direction": 4,
    "object_relative": 2, "relative_motion": 4, "push_outcome": 3, "poke_outcome": 4, "spin_outcome": 2,
    "fall_style": 2, "lift_outcome": 2, "slant_outcome": 2, "containment": 4, "plug_outcome": 2,
    "open_close": 2, "cover": 2,
}
FAMILY_OF = {t: d for d, ts in FAMILIES.items() for t in ts[:OWN[d]]}


######################## Options: templates filled with the objects ########################
def fill(template, objects):
    """Fill the [something] slots in order. A slot without an object of its own reads
    'something'. The option is written in lower case."""
    out, i = template, 0
    while "[something]" in out.lower():
        k = out.lower().index("[something]")
        out = out[:k] + (objects[i] if i < len(objects) else "something") + out[k + len("[something]"):]
        i += 1
    return out.lower()


def to_item(label, video_dir, rng):
    family = FAMILY_OF[label["template"]]
    options = [fill(t, label["placeholders"]) for t in FAMILIES[family]]
    answer = options[FAMILIES[family].index(label["template"])]
    rng.shuffle(options)
    return {"id": f"ssv2_{label['id']}", "video": os.path.join(video_dir, f"{label['id']}.webm"),
            "question": QUESTION, "options": options, "answer": "ABCD"[options.index(answer)],
            "dim": family, "template": label["template"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="train.json or validation.json of SSv2")
    ap.add_argument("--video_dir", required=True, help="directory of the <id>.webm videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None, help="fix the option order (random by default)")
    args = ap.parse_args()
    own = [t for d, ts in FAMILIES.items() for t in ts[:OWN[d]]]
    assert len(own) == len(set(own)) == 47, "every own template belongs to exactly one family"
    rng = random.Random(args.seed)
    labels = json.load(open(args.labels))
    items = [to_item(x, args.video_dir, rng) for x in labels if x["template"] in FAMILY_OF]
    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(items)} of {len(labels)} videos, {len(FAMILIES)} families")


if __name__ == "__main__":
    main()
