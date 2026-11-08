#!/bin/bash
# usage: bash run_tp_amp.sh <gpu> <coop|kgcoop> <seed> <dataset> [dataset ...]   (mixed precision)
gpu=$1; learner=$2; seed=$3; shift 3
for ds in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu python run_textprompt.py --root ~/data_compact --dataset $ds \
    --learner $learner --seed $seed --amp > logs/tp_${learner}_s${seed}_$ds.log 2>&1
done
