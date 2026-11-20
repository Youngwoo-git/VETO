#!/bin/bash
# End-to-end example on one GPU: calibration, VETO and SFT fine-tuning, evaluation.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh <seed>
set -e
cd "$(dirname "$0")/.."
SEED=${1:-42}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
CALIB=${CALIB:-data/calib400.jsonl}       # calibration items from prepare_splits.py
TRAIN=${TRAIN:-data/train.jsonl}           # training set, without the calibration items
VAL=${VAL:-data/val_select.jsonl}          # 200 items for checkpoint selection
TEST=${TEST:-data/test.jsonl}              # evaluation split, without the selection items
OUT=${OUT:-out}
export ATTN_IMPL=${ATTN_IMPL:-sdpa} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT

# 1) calibration, once per VideoLLM
[ -f $OUT/projectors.npz ] || python tools/calibrate.py --model_name $MODEL --calib_data $CALIB \
    --rho 0.95 --out $OUT/projectors.npz

# 2) fine-tuning
python tools/train.py --model_name $MODEL --train_data $TRAIN --val_data $VAL --seed $SEED --resume \
    --projectors $OUT/projectors.npz --out_dir $OUT/veto_s$SEED
python tools/train.py --model_name $MODEL --train_data $TRAIN --val_data $VAL --seed $SEED --resume \
    --out_dir $OUT/sft_s$SEED

# 3) accuracy with the full input and from the text alone
python tools/evaluate.py --model_name $MODEL --base --data $TEST --out $OUT/base.json
for M in veto sft; do
  python tools/evaluate.py --model_name $MODEL --ckpt $OUT/${M}_s$SEED/best.pt --data $TEST --out $OUT/${M}_s$SEED.json
done
