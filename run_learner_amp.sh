#!/bin/bash
# usage: bash run_learner_amp.sh <gpu> <learner> <seed> <dataset> [dataset ...]   (mixed precision)
# dumps test logits, then calibrates; skips any step whose output already exists
gpu=$1; learner=$2; seed=$3; shift 3
for ds in "$@"; do
  out=runs/$learner/logits_s$seed/$ds
  if [ ! -f $out/logits.npz ]; then
    CUDA_VISIBLE_DEVICES=$gpu python dump_logits.py --root ~/data_compact --dataset $ds --amp \
      --learner $learner --seed $seed --out $out > logs/${learner}_dump_s${seed}_$ds.log 2>&1
  fi
  if [ ! -f runs/$learner/calib_s$seed/$ds/points.npz ]; then
    CUDA_VISIBLE_DEVICES=$gpu python calibrate_mixed.py --root ~/data_compact --dataset $ds --amp \
      --learner $learner --seed $seed --out runs/$learner/calib_s$seed/$ds \
      > logs/${learner}_calib_s${seed}_$ds.log 2>&1
  fi
done
