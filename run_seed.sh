#!/bin/bash
# usage: bash run_seed.sh <gpu> <seed> <dataset> [dataset ...]
gpu=$1; seed=$2; shift 2
for ds in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu python dump_logits.py --root ~/data_compact --dataset $ds --seed $seed \
    --out runs/logits_s$seed/$ds > logs/dump_s${seed}_$ds.log 2>&1
  CUDA_VISIBLE_DEVICES=$gpu python calibrate_mixed.py --root ~/data_compact --dataset $ds --seed $seed \
    --out runs/calib_s$seed/$ds > logs/calib_s${seed}_$ds.log 2>&1
done
