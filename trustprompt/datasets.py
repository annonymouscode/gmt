"""Datasets, few-shot sampling and base/novel splits.

Follows the CoOp / CoCoOp data layout (see CoOp's DATASETS.md):
  <root>/<dataset_dir>/<image_dir>/...  and  <root>/<dataset_dir>/split_zhou_<Name>.json
Each json has "train", "val", "test" lists of [relative_image_path, label, classname].
FGVCAircraft and ImageNet use their own layouts (see loaders below).
Verify the directory names against CoOp's DATASETS.md before running.
"""
import json
import math
import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# name -> (dataset_dir, image_dir, split_json)
JSON_DATASETS = {
    "oxford_pets": ("oxford_pets", "images", "split_zhou_OxfordPets.json"),
    "caltech101": ("caltech-101", "101_ObjectCategories", "split_zhou_Caltech101.json"),
    "stanford_cars": ("stanford_cars", "", "split_zhou_StanfordCars.json"),
    "oxford_flowers": ("oxford_flowers", "jpg", "split_zhou_OxfordFlowers.json"),
    "food101": ("food-101", "images", "split_zhou_Food101.json"),
    "sun397": ("sun397", "SUN397", "split_zhou_SUN397.json"),
    "dtd": ("dtd", "images", "split_zhou_DescribableTextures.json"),
    "eurosat": ("eurosat", "2750", "split_zhou_EuroSAT.json"),
    "ucf101": ("ucf101", "UCF-101-midframes", "split_zhou_UCF101.json"),
}

# Hand-crafted zero-shot templates (CoOp IJCV 2022, Table 6).
TEMPLATES = {
    "imagenet": "a photo of a {}.",
    "caltech101": "a photo of a {}.",
    "oxford_pets": "a photo of a {}, a type of pet.",
    "stanford_cars": "a photo of a {}.",
    "oxford_flowers": "a photo of a {}, a type of flower.",
    "food101": "a photo of {}, a type of food.",
    "fgvc_aircraft": "a photo of a {}, a type of aircraft.",
    "sun397": "a photo of a {}.",
    "dtd": "{} texture.",
    "eurosat": "a centered satellite photo of {}.",
    "ucf101": "a photo of a person doing {}.",
}

# CoOp renames EuroSAT classes; applied only if the raw names appear.
EUROSAT_NAMES = {
    "AnnualCrop": "Annual Crop Land", "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land", "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings", "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land", "Residential": "Residential Buildings",
    "River": "River", "SeaLake": "Sea or Lake",
}


def _clean(name):
    return name.replace("_", " ")


def _load_json_dataset(root, name):
    ddir, idir, sjson = JSON_DATASETS[name]
    with open(os.path.join(root, ddir, sjson)) as f:
        split = json.load(f)
    img_root = os.path.join(root, ddir, idir)
    out = {}
    for part in ("train", "val", "test"):
        items = []
        for impath, label, cname in split[part]:
            if name == "eurosat":
                cname = EUROSAT_NAMES.get(cname, cname)
            items.append((os.path.join(img_root, impath), int(label), _clean(cname)))
        out[part] = items
    return out


def _load_fgvc(root):
    base = os.path.join(root, "fgvc_aircraft")
    with open(os.path.join(base, "variants.txt")) as f:
        names = [l.strip() for l in f if l.strip()]
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for part, fn in (("train", "images_variant_train.txt"), ("val", "images_variant_val.txt"),
                     ("test", "images_variant_test.txt")):
        items = []
        with open(os.path.join(base, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                imid, cname = line.split(" ", 1)
                items.append((os.path.join(base, "images", imid + ".jpg"), idx[cname], cname))
        out[part] = items
    return out


def _load_imagenet(root):
    base = os.path.join(root, "imagenet")
    wnids, names = [], []
    with open(os.path.join(base, "classnames.txt")) as f:
        for line in f:
            wnid, cname = line.strip().split(" ", 1)
            wnids.append(wnid)
            names.append(cname)

    def scan(folder):
        items = []
        for lab, wnid in enumerate(wnids):
            d = os.path.join(base, "images", folder, wnid)
            for fn in sorted(os.listdir(d)):
                items.append((os.path.join(d, fn), lab, names[lab]))
        return items

    train = scan("train")
    test = scan("val")  # CoOp evaluates on the ImageNet val set
    return {"train": train, "val": test, "test": test}


def _load_compact(root, name):
    d = os.path.join(root, name)
    with open(os.path.join(d, "split.json")) as f:
        split = json.load(f)
    return {part: [(os.path.join(d, "images", r), int(l), _clean(c)) for r, l, c in split[part]]
            for part in ("train", "val", "test")}


def load_dataset(root, name):
    """Compact layout (<root>/<name>/split.json, built by scripts/make_compact.py or
    scripts/stream_*.py) takes priority; otherwise the original CoOp layout is used."""
    if os.path.exists(os.path.join(root, name, "split.json")):
        return _load_compact(root, name)
    if name in JSON_DATASETS:
        return _load_json_dataset(root, name)
    if name == "fgvc_aircraft":
        return _load_fgvc(root)
    if name == "imagenet":
        return _load_imagenet(root)
    raise ValueError(f"unknown dataset {name}")


def few_shot(items, k, seed):
    by_label = defaultdict(list)
    for it in items:
        by_label[it[1]].append(it)
    rng = random.Random(seed)
    out = []
    for lab in sorted(by_label):
        pool = by_label[lab]
        out += rng.sample(pool, min(k, len(pool)))
    return out


def classnames_of(items):
    names = {}
    for _, lab, cname in items:
        names[lab] = cname
    return [names[i] for i in sorted(names)]


def base_novel_split(all_items_by_part, seed=None):
    """CoCoOp protocol: sort labels, first ceil(n/2) are base, rest novel; relabel from 0."""
    labels = sorted({it[1] for it in all_items_by_part["test"]})
    m = math.ceil(len(labels) / 2)
    base, novel = labels[:m], labels[m:]
    return base, novel


def subset(items, keep_labels, relabel=True):
    keep = {l: i for i, l in enumerate(keep_labels)}
    return [(p, keep[l] if relabel else l, c) for p, l, c in items if l in keep]


def build_transforms(train, size=224):
    norm = T.Normalize(CLIP_MEAN, CLIP_STD)
    if train:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                          T.RandomHorizontalFlip(), T.ToTensor(), norm])
    return T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size),
                      T.ToTensor(), norm])


class ImageList(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lab, _ = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), lab
