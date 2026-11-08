"""Build a compact copy of an already-extracted dataset (any layout trustprompt can load).

  python scripts/make_compact.py --src_root /dev/shm/raw --dst_root ~/data_compact --dataset eurosat
Then delete the raw copy. Keeps all test images, 64 train images/class, 8 val images/class,
resized so the shorter side is 256 px.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trustprompt.datasets import load_dataset
from compact_common import choose_items, save_image, write_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", required=True)
    p.add_argument("--dst_root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--train_pool", type=int, default=64)
    p.add_argument("--val_pool", type=int, default=8)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    data = load_dataset(a.src_root, a.dataset)
    src_dir = os.path.abspath(a.src_root)
    # relative paths under the source root keep file names unique
    rel = lambda path: os.path.relpath(os.path.abspath(path), src_dir)
    split = {k: [(rel(pth), l, c) for pth, l, c in v] for k, v in data.items() if k in ("train", "val", "test")}
    chosen = choose_items(split, a.train_pool, a.val_pool)
    dst = os.path.join(os.path.expanduser(a.dst_root), a.dataset)
    os.makedirs(dst, exist_ok=True)
    todo = {r for items in chosen.values() for r, _, _ in items}

    def work(r):
        with open(os.path.join(src_dir, r), "rb") as f:
            save_image(dst, r, f.read(), a.size, a.quality)

    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(work, sorted(todo))):
            if i % 5000 == 0:
                print(f"  {i}/{len(todo)}")
    s = write_split(dst, chosen)
    print(f"{a.dataset}: train {len(s['train'])}  val {len(s['val'])}  test {len(s['test'])}  -> {dst}")


if __name__ == "__main__":
    main()
