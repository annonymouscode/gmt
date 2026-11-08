#!/bin/bash
set -euo pipefail
D=~/datasets; I=$D/imagenet
[ -f $I/READY ] && { echo "already prepared"; exit 0; }
grep -q IMAGENET_DOWNLOADED $D/dl_imagenet.log
SRC=$I/ILSVRC/Data/CLS-LOC; CSV=$I/LOC_val_solution.csv
[ -n "$SRC" ] && [ -f "$CSV" ] || { echo "CLS-LOC or LOC_val_solution.csv not found"; exit 1; }
mkdir -p $I/images
[ -d $I/images/train ] || mv "$SRC/train" $I/images/train
python - "$SRC/val" "$CSV" "$I/images/val" <<'PY'
import csv, os, sys, shutil
src, csvf, dst = sys.argv[1:]; n = 0
for r in csv.DictReader(open(csvf)):
    w = r["PredictionString"].split()[0]; os.makedirs(f"{dst}/{w}", exist_ok=True)
    s = f"{src}/{r['ImageId']}.JPEG"
    if os.path.exists(s): shutil.move(s, f"{dst}/{w}/{r['ImageId']}.JPEG"); n += 1
print("moved", n)
PY
nt=$(ls $I/images/train | wc -l); nv=$(ls $I/images/val | wc -l); ni=$(find $I/images/val -name "*.JPEG" | wc -l)
echo "train $nt, val classes $nv, val images $ni"
[ "$nt" -eq 1000 ] && [ "$nv" -eq 1000 ] && [ "$ni" -eq 50000 ] || { echo "COUNT MISMATCH"; exit 1; }
[ "$(wc -l < $I/classnames.txt)" -ge 1000 ] || { echo "classnames.txt has fewer than 1000 lines"; exit 1; }
touch $I/READY; echo IMAGENET_READY
