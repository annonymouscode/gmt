#!/bin/bash
# usage: bash run_variant.sh <gpu> <maple|coop> <tag> <seed> "<extra args>" <dataset> [dataset ...]
# e.g.   bash run_variant.sh 0 maple maple_k1 1 "--shots 1" caltech101 dtd ...
# dump + 3-split calibration for a shots/backbone variant; skips outputs that already exist
gpu=$1; learner=$2; tag=$3; seed=$4; extra=$5; shift 5
for ds in "$@"; do
  if [ "$learner" = "coop" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python run_textprompt.py --root ~/data_compact --dataset $ds --learner coop \
      --tag $tag --seed $seed --amp $extra > logs/${tag}_s${seed}_$ds.log 2>&1
  else
    out=runs/$tag/logits_s$seed/$ds
    [ -f $out/logits.npz ] || CUDA_VISIBLE_DEVICES=$gpu python dump_logits.py --root ~/data_compact --dataset $ds \
      --learner $learner --seed $seed --out $out $extra > logs/${tag}_dump_s${seed}_$ds.log 2>&1
    [ -f runs/$tag/calib_s$seed/$ds/points.npz ] || CUDA_VISIBLE_DEVICES=$gpu python calibrate_mixed.py --root ~/data_compact \
      --dataset $ds --learner $learner --seed $seed --out runs/$tag/calib_s$seed/$ds $extra > logs/${tag}_calib_s${seed}_$ds.log 2>&1
  fi
done
