"""Shared helpers for building compact datasets (few-shot pool + full test, resized JPEGs).

Compact layout (read by trustprompt/datasets.py automatically):
  <dst_root>/<dataset>/images/<relative path>.jpg
  <dst_root>/<dataset>/split.json   {"train": [[rel, label, classname], ...], "val": [...], "test": [...]}
"""
import io
import json
import os
import random
from collections import defaultdict

from PIL import Image


def choose_items(split, train_pool=64, val_pool=8, seed=0):
    """Keep all test items, a fixed per-class pool of train items and a small val pool."""
    out = {"test": list(split["test"])}
    for part, k in (("train", train_pool), ("val", val_pool)):
        by = defaultdict(list)
        for it in split[part]:
            by[it[1]].append(it)
        keep = []
        for lab in sorted(by):
            items = sorted(by[lab], key=lambda t: t[0])
            rng = random.Random(seed * 100003 + int(lab))
            keep += rng.sample(items, min(k, len(items)))
        out[part] = keep
    return out


def resize_bytes(data, size=256, quality=90):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def out_rel(rel):
    return os.path.splitext(rel)[0] + ".jpg"


def write_split(dst_dir, chosen):
    split = {p: [[out_rel(r), int(l), c] for r, l, c in items] for p, items in chosen.items()}
    with open(os.path.join(dst_dir, "split.json"), "w") as f:
        json.dump(split, f)
    return split


def save_image(dst_dir, rel, data, size, quality):
    path = os.path.join(dst_dir, "images", out_rel(rel))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(resize_bytes(data, size, quality))
