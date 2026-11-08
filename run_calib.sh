#!/bin/bash
# usage: bash run_calib.sh <gpu> <dataset> [dataset ...]
gpu=$1; shift
for ds in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu python calibrate_mixed.py --root ~/data_compact \
    --dataset $ds --out runs/calib/$ds > logs/calib_$ds.log 2>&1
done
