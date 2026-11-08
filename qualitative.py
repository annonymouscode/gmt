"""E5: qualitative examples. Novel test images that the adapted model assigns to a base class, with
GMT's prediction, plus examples where GMT does not help (gate off).

  python qualitative.py --root ~/data_compact            # MaPLe, seed 1 -> runs/qualitative/
Needs the image files (dataset root) and the saved logits/calibration points; CPU only.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from trustprompt.datasets import load_dataset, base_novel_split, subset
from correct_bias import softmax, sinkhorn
from adaptive_correct import gini
from analysis_suite import fit_threshold

DNAME = {"caltech101": "Caltech101", "dtd": "DTD", "eurosat": "EuroSAT", "fgvc_aircraft": "Aircraft",
         "food101": "Food101", "oxford_flowers": "Flowers102", "oxford_pets": "Pets", "ucf101": "UCF101"}


def dirs(L, s):
    if L == "maple":
        return ("runs/logits", "runs/calib") if s == 1 else (f"runs/logits_s{s}", f"runs/calib_s{s}")
    return f"runs/{L}/logits_s{s}", f"runs/{L}/calib_s{s}"


def gmt_pred(lg, zs, pts):
    g0, tau = fit_threshold(pts["g"], pts["gain"])
    P = softmax(lg); C = P.shape[1]
    Q = sinkhorn(P, np.full(C, 1.0 / C))
    lam = 1.0 / (1.0 + np.exp((gini(softmax(zs).mean(0)) - g0) / tau))
    return ((1 - lam) * np.log(np.maximum(P, 1e-12)) + lam * np.log(np.maximum(Q, 1e-12))).argmax(-1), lam


def nice(n):
    return str(n).replace("_", " ")[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--learner", default="maple")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fixed", default="eurosat,dtd,oxford_pets,ucf101", help="datasets for corrected examples")
    ap.add_argument("--failure", default="fgvc_aircraft", help="dataset for an example GMT does not fix")
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    ld, cd = dirs(a.learner, a.seed)
    rng = np.random.default_rng(0)
    picks = []                                            # (dataset, path, true, adapted, gmt, kind)
    for d, kind in [(x, "fixed") for x in a.fixed.split(",")] + [(a.failure, "failure")]:
        t = np.load(f"{ld}/{d}/logits.npz", allow_pickle=True)
        lg, zs, y, nb = t["logits"], t["zs_logits"], t["labels"], int(t["n_base"])
        names = [str(n) for n in t["classnames"]]
        data = load_dataset(a.root, d)
        base, novel = base_novel_split(data)
        items = subset(data["test"], base + novel)
        if len(items) != len(y) or any(it[1] != yy for it, yy in zip(items, y)):
            print(f"! {d}: test order does not match the saved logits; skipped"); continue
        gp, lam = gmt_pred(lg, zs, np.load(f"{cd}/{d}/points.npz"))
        raw = lg.argmax(-1)
        if kind == "fixed":
            idx = np.where((y >= nb) & (raw < nb) & (gp == y))[0]
        else:
            idx = np.where((y >= nb) & (raw < nb) & (gp < nb))[0]
        n = a.per if kind == "fixed" else 1
        for i in rng.choice(idx, size=min(n, len(idx)), replace=False) if len(idx) else []:
            picks.append((d, items[i][0], names[y[i]], names[raw[i]], names[gp[i]], kind, lam))
        print(f"{d}: lambda {lam:.2f}, {len(idx)} candidate images ({kind})")
    if not picks:
        return
    cols = 5; rows = int(np.ceil(len(picks) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(1.75 * cols, 2.1 * rows))
    for ax in np.atleast_1d(axs).ravel():
        ax.axis("off")
    for ax, (d, path, tr, ad, gm, kind, lam) in zip(np.atleast_1d(axs).ravel(), picks):
        ax.imshow(Image.open(path).convert("RGB").resize((224, 224)))
        ok = "#0072B2" if gm == tr else "#D55E00"
        ax.set_title(f"{DNAME.get(d, d)}  ($\\lambda$={lam:.2f})", fontsize=5.5)
        ax.text(0.5, -0.04, f"true: {nice(tr)}", transform=ax.transAxes, ha="center", va="top", fontsize=5)
        ax.text(0.5, -0.16, f"adapted: {nice(ad)}", transform=ax.transAxes, ha="center", va="top",
                fontsize=5, color="#7f7f7f")
        ax.text(0.5, -0.28, f"GMT: {nice(gm)}", transform=ax.transAxes, ha="center", va="top", fontsize=5, color=ok)
    fig.subplots_adjust(hspace=0.8, wspace=0.15)
    fig.text(0.5, 0.005, "grey: prediction of the adapted model (always a base class); blue: GMT corrects it; orange: GMT does not",
             ha="center", fontsize=5.5)
    os.makedirs("runs/qualitative", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"runs/qualitative/supp_qualitative.{ext}", bbox_inches="tight", dpi=200)
    print(f"saved {len(picks)} examples -> runs/qualitative/supp_qualitative.pdf")


if __name__ == "__main__":
    main()
