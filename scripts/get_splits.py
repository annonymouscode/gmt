"""Download CoOp split files (split_zhou_*.json), ImageNet classnames.txt, Flowers cat_to_name.json
and (optionally) UCF-101 middle frames, using the Google Drive links in CoOp's DATASETS.md.

  pip install gdown
  python scripts/get_splits.py ~/datasets                 # all split files
  python scripts/get_splits.py ~/datasets --only caltech eurosat food
  python scripts/get_splits.py ~/datasets --ucf_frames    # also the UCF-101 frames zip (~2 GB)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/KaiyangZhou/CoOp/main/DATASETS.md"
# file name in DATASETS.md -> (target folder, short name)
TARGETS = {
    "split_zhou_Caltech101.json": ("caltech-101", "caltech"),
    "split_zhou_OxfordPets.json": ("oxford_pets", "pets"),
    "split_zhou_StanfordCars.json": ("stanford_cars", "cars"),
    "split_zhou_OxfordFlowers.json": ("oxford_flowers", "flowers"),
    "cat_to_name.json": ("oxford_flowers", "flowers"),
    "split_zhou_Food101.json": ("food-101", "food"),
    "split_zhou_SUN397.json": ("sun397", "sun"),
    "split_zhou_DescribableTextures.json": ("dtd", "dtd"),
    "split_zhou_EuroSAT.json": ("eurosat", "eurosat"),
    "split_zhou_UCF101.json": ("ucf101", "ucf"),
    "classnames.txt": ("imagenet", "imagenet"),
    "UCF-101-midframes.zip": ("ucf101", "ucf_frames"),
}
DG_COPIES = ["imagenetv2", "imagenet-sketch", "imagenet-adversarial", "imagenet-rendition"]


def parse_links(text):
    links = {}
    for line in text.splitlines():
        m = re.search(r"https://drive\.google\.com/file/d/[\w-]+/view[^)\s]*", line)
        if not m:
            continue
        for fname in TARGETS:
            if fname in line and fname not in links:
                links[fname] = m.group(0)
    return links


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--only", nargs="*", default=None,
                   help="short names: caltech pets cars flowers food sun dtd eurosat ucf imagenet")
    p.add_argument("--ucf_frames", action="store_true", help="also download and extract UCF-101-midframes.zip")
    p.add_argument("--datasets_md", default="", help="local copy of DATASETS.md (default: fetch from GitHub)")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    if a.datasets_md:
        text = open(a.datasets_md).read()
    else:
        text = urllib.request.urlopen(RAW, timeout=60).read().decode()
    links = parse_links(text)
    missing = [f for f in TARGETS if f not in links]
    if missing:
        print("WARNING: no link found in DATASETS.md for:", missing)

    if not a.dry_run:
        try:
            import gdown
        except ImportError:
            sys.exit("pip install gdown")
    root = os.path.expanduser(a.root)
    for fname, url in links.items():
        folder, short = TARGETS[fname]
        if short == "ucf_frames" and not a.ucf_frames:
            continue
        if a.only and short not in a.only and not (short == "ucf_frames" and "ucf" in a.only):
            continue
        out_dir = os.path.join(root, folder)
        out = os.path.join(out_dir, fname)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"  exists: {out}")
            continue
        print(f"  {fname} -> {out}")
        if a.dry_run:
            continue
        os.makedirs(out_dir, exist_ok=True)
        # uc?id=<ID> works with old and new gdown versions (no 'fuzzy' argument needed)
        file_id = re.search(r"/d/([\w-]+)", url).group(1)
        got = gdown.download(f"https://drive.google.com/uc?id={file_id}", out, quiet=False)
        if not got:
            print(f"  FAILED: {fname}  (open {url} in a browser and copy the file to {out_dir})")
            continue
        if fname.endswith(".zip"):
            subprocess.run(["unzip", "-q", "-o", out, "-d", out_dir], check=True)
            os.remove(out)
        if fname == "classnames.txt":
            for d in DG_COPIES:
                os.makedirs(os.path.join(root, d), exist_ok=True)
                shutil.copy(out, os.path.join(root, d, "classnames.txt"))
    print("done. Check with: python scripts/check_data.py", root)


if __name__ == "__main__":
    main()
