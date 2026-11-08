#!/bin/bash
# usage: bash post_learner.sh <learner>
# after a learner's runs finish: calib.json per seed, baselines (incl. GMT-E), gate ablation, analysis
L=$1
for s in 1 2 3; do
  python calibrate_mixed.py --summarize runs/$L/calib_s$s/*/points.npz --tol 0.0 \
    --out runs/$L/calib_s$s/calib.json > /dev/null
done
R=""; for s in 1 2 3; do R="$R --run runs/$L/logits_s$s:runs/$L/calib_s$s"; done
python baselines_cs.py $R --out runs/baselines3_$L.json > baselines3_$L.txt 2>&1
python ablate_gate.py  $R --out runs/ablate_$L.json    > ablate_$L.txt    2>&1
python analysis_suite.py --learners $L --out runs/analysis_$L.json > analysis_$L.txt 2>&1
echo "POST_DONE $L"
