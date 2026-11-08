#!/bin/bash
# usage: bash run_newdata.sh <gpu> <learner> <seed> <dataset> [dataset ...]
# Test-logit dump only (no calibration) for the extra datasets; skips outputs that already exist.
# Learners: coop kgcoop cocoop (cached features) | maple ivlp promptsrc (full model)
gpu=$1; learner=$2; seed=$3; shift 3
ROOT=${DATA_ROOT:-~/data_compact}
mkdir -p logs
for ds in "$@"; do
  case $learner in
    coop|cocoop)  CUDA_VISIBLE_DEVICES=$gpu python run_textprompt.py --root $ROOT --dataset $ds --learner $learner \
                    --seed $seed --amp --no-calib > logs/new_${learner}_s${seed}_$ds.log 2>&1 ;;
    kgcoop)       CUDA_VISIBLE_DEVICES=$gpu python run_textprompt.py --root $ROOT --dataset $ds --learner kgcoop \
                    --seed $seed --no-calib > logs/new_${learner}_s${seed}_$ds.log 2>&1 ;;
    maple|ivlp|promptsrc)
      if [ "$learner" = "maple" ]; then out=runs/logits$([ $seed = 1 ] || echo _s$seed)/$ds; else out=runs/$learner/logits_s$seed/$ds; fi
      amp=$([ "$learner" = "promptsrc" ] && echo --amp)
      [ -f $out/logits.npz ] || CUDA_VISIBLE_DEVICES=$gpu python dump_logits.py --root $ROOT --dataset $ds \
        --learner $learner --seed $seed --out $out $amp > logs/new_${learner}_s${seed}_$ds.log 2>&1 ;;
  esac
done
