"""Check that each dataset under ROOT loads with the dataset loaders and that sample images open.
   python scripts/check_data.py /data"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from trustprompt.datasets import load_dataset, classnames_of, base_novel_split

NAMES = ["dtd", "eurosat", "oxford_flowers", "caltech101", "oxford_pets", "fgvc_aircraft", "food101",
         "sun397", "stanford_cars", "ucf101", "imagenet"]
EXPECTED = {"dtd": 47, "eurosat": 10, "oxford_flowers": 102, "caltech101": 100, "oxford_pets": 37,
            "fgvc_aircraft": 100, "food101": 101, "sun397": 397, "stanford_cars": 196, "ucf101": 101,
            "imagenet": 1000}  # class counts from CoOp (IJCV 2022) Table 6

root = sys.argv[1]
for n in NAMES:
    try:
        d = load_dataset(root, n)
        k = len(classnames_of(d["test"]))
        base, novel = base_novel_split(d)
        for part in ("train", "test"):
            Image.open(d[part][0][0]).convert("RGB")
        flag = "ok " if k == EXPECTED[n] else "?? "
        print(f"{flag}{n:15s} classes {k:4d} (expected {EXPECTED[n]})  train {len(d['train']):6d}  "
              f"test {len(d['test']):6d}  base/novel {len(base)}/{len(novel)}")
    except Exception as e:
        print(f"-- {n:15s} not ready: {type(e).__name__}: {e}")
