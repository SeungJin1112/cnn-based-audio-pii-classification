#!/bin/bash
# 인자: $1 = 모델명, $2 = GPU 번호
M=$1; G=$2
PY=/data/hye0n/paper/venv/bin/python
for S in 42 43 44; do
  CUDA_VISIBLE_DEVICES=$G $PY seq_head_exp.py --model $M --seed $S --mode indist  --epochs 40
  CUDA_VISIBLE_DEVICES=$G $PY seq_head_exp.py --model $M --seed $S --mode heldcat --cat serial --epochs 40
done
echo "DONE $M"
