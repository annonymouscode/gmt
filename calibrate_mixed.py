"""Calibrate g0 on a *simulated mixed* validation setting, still without touching the test set.

Split the base classes into pseudo-base B1 and pseudo-novel B2, prompt-tune on B1 only, score
B1+B2 validation images (plus unused train-pool images) over the joint label set B1 u B2, then
sweep a synthetic class skew to find the gini at which the correction stops paying for itself.
No novel class and no test image is used at any point.

v2 changes
  * several B1/B2 splits per dataset (--splits 0,1,2); split 0 reproduces the original split
  * calibration logits are saved per split (split<s>/calib_logits.npz) and reused on later runs,
    so re-running with new --alphas / --reps is a CPU-only re-sweep with no training
  * default sweep is 11 skews x 20 reps; points.npz also stores alpha and split ids
  * --summarize reports the per-split g0 spread next to the pooled-over-splits fit

  python calibrate_mixed.py --root ~/data_compact --dataset dtd --out runs/calib/dtd
  python calibrate_mixed.py --root ~/data_compact --dataset dtd --out runs/calib/dtd --reps 40  # re-sweep only
  python calibrate_mixed.py --summarize runs/calib/*/points.npz --tol 0.0 --out runs/calib/calib.json
"""
import argparse, json, os, random
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import clip

from trustprompt.datasets import (load_dataset, few_shot, classnames_of, base_novel_split, subset,
                                  build_transforms, ImageList)
from trustprompt.learners import build_learner, train_loss, fill_defaults, end_epoch
from trustprompt.engine import build_optimizer
from correct_bias import softmax, sinkhorn
from adaptive_correct import gini


def skewed_subset(y, alpha, rng, C):
    w = np.arange(1, C + 1) ** -alpha
    w = w / w.sum(); rng.shuffle(w)
    keep = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        n = max(1, int(round(w[c] * len(y))))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return np.concatenate(keep)


def isotonic_decreasing(x, y):
    """Pool-adjacent-violators fit of a non-increasing function y(x). Returns fitted values in x order."""
    order = np.argsort(x)
    yv = list(-np.asarray(y, float)[order])            # fit non-decreasing on -y
    w = [1.0] * len(yv); blocks = [[i] for i in range(len(yv))]
    i = 0
    while i < len(yv) - 1:
        if yv[i] > yv[i + 1]:
            tot = w[i] + w[i + 1]
            yv[i] = (yv[i] * w[i] + yv[i + 1] * w[i + 1]) / tot; w[i] = tot
            blocks[i] += blocks[i + 1]
            del yv[i + 1], w[i + 1], blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = np.empty(len(x))
    for v, b in zip(yv, blocks):
        for j in b:
            fit[order[j]] = -v
    return fit


def fit_threshold(g, gain, tol=0.0):
    """g0 = gini where the monotone (non-increasing) fit of gain first drops below -tol."""
    g = np.asarray(g, float); gain = np.asarray(gain, float)
    fit = isotonic_decreasing(g, gain)
    o = np.argsort(g); gs, fs = g[o], fit[o]
    below = np.where(fs < -tol)[0]
    if len(below) == 0:
        g0 = float(gs.max() + 0.05)
    elif below[0] == 0:
        g0 = float(gs.min())
    else:
        k = below[0]                                     # interpolate the crossing
        x0, x1, y0, y1 = gs[k - 1], gs[k], fs[k - 1] + tol, fs[k] + tol
        g0 = float(x0 + (x1 - x0) * y0 / (y0 - y1)) if y0 != y1 else float(x1)
    span = gs[(fs < -tol + 3) & (fs > -tol - 3)]          # width of the transition region
    tau = float(max(0.01, (span.max() - span.min()) / 4)) if len(span) > 1 else 0.02
    return g0, tau, fs, gs


def sweep(lg, zs, y, C, alphas, reps):
    """Returns arrays (gini, gain, alpha) over alphas x reps skewed subsets."""
    G, GAIN, A = [], [], []
    u = np.full(C, 1.0 / C)
    for alpha in alphas:
        for r in range(reps):
            rng = np.random.default_rng(r * 977 + int(round(alpha * 100)))
            sub = skewed_subset(y, alpha, rng, C) if alpha > 0 else np.arange(len(y))
            P = softmax(lg[sub]); ys = y[sub]
            raw = (lg[sub].argmax(-1) == ys).mean() * 100
            cor = (sinkhorn(P, u).argmax(-1) == ys).mean() * 100
            G.append(gini(softmax(zs[sub]).mean(0))); GAIN.append(cor - raw); A.append(alpha)
            if alpha == 0:
                break                                    # alpha 0 is the full set: one rep is enough
    return np.array(G), np.array(GAIN), np.array(A)


def split_seed(seed, split):
    return seed if split == 0 else seed * 1000 + split   # split 0 == the original v1 split


def train_and_dump(a, data, names_all, base, split, dev, path):
    s = split_seed(a.seed, split)
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    rng = random.Random(s); shuf = list(base); rng.shuffle(shuf)
    half = len(shuf) // 2
    b1, b2 = sorted(shuf[:half]), sorted(shuf[half:])     # pseudo-base, pseudo-novel
    order = b1 + b2
    names = [names_all[l] for l in order]; b1_names = [names_all[l] for l in b1]
    train_items = few_shot(subset(data["train"], b1), a.shots, s)
    # calibration images: base-class validation images plus train-pool images not sampled for
    # training. All are base-class training-split data; no test image is used.
    used = {p for p, _, _ in train_items}
    order_set = set(order)
    pool = [it for it in data["train"] if it[1] in order_set and it[0] not in used]
    val_items = subset(list(data["val"]) + pool, order)  # relabelled 0..|b1|+|b2|-1
    print(f"{a.dataset} split {split}: pseudo-base {len(b1)} / pseudo-novel {len(b2)}, "
          f"{len(train_items)} train, {len(val_items)} val images", flush=True)

    cm, _ = clip.load(a.backbone, device="cpu", jit=False)   # fresh CLIP per split
    model = build_learner(a.learner, cm.float(), a.dataset, a.depth, a.n_ctx,
                          a.kg_lambda if a.kg_lambda is not None else 8.0).to(dev)
    tr, te = build_transforms(True), build_transforms(False)
    mk = lambda it, tf, sh, bs: DataLoader(ImageList(it, tf), batch_size=bs, shuffle=sh,
                                           num_workers=a.workers, pin_memory=True)
    loader = mk(train_items, tr, True, a.bs)
    opt, sched = build_optimizer(model.trainable_parameters(), a.lr, a.epochs, len(loader))
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp)
    for ep in range(a.epochs):
        model.train()
        for imgs, labs in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                loss = train_loss(model, imgs.to(dev), labs.to(dev), b1_names)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        end_epoch(model, ep, a.epochs)
        print(f"  epoch {ep+1}/{a.epochs} loss {loss.item():.4f}", flush=True)

    model.eval(); L, Z, Y = [], [], []
    with torch.no_grad():
        for imgs, labs in mk(val_items, te, False, 128):
            imgs = imgs.to(dev)
            L.append(model(imgs, names).float().cpu()); Z.append(model.zero_shot(imgs, names).cpu())
            Y.append(labs)
    lg, zs, y = torch.cat(L).numpy(), torch.cat(Z).numpy(), torch.cat(Y).numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"                         # atomic: a killed job leaves no half file
    np.savez(tmp, logits=lg, zs_logits=zs, labels=y, n_b1=len(b1),
             b1=np.array(b1), b2=np.array(b2), seed=s)
    os.replace(tmp, path)
    print(f"  simulated-mixed val acc {(lg.argmax(-1) == y).mean()*100:.2f}  -> {path}", flush=True)
    del model, cm, opt; torch.cuda.empty_cache()


def summarize(a):
    per = {}
    for f in a.summarize:
        d = np.load(f)
        sp = d["split"] if "split" in d.files else np.zeros(len(d["g"]), int)
        per[os.path.basename(os.path.dirname(f))] = (d["g"], d["gain"], sp)
    out = {}
    print(f"{'dataset':16s} {'points':>6s} {'g0':>7s} {'tau':>6s}  {'per-split g0':24s} fitted gain lo..hi")
    for name, (g, gain, sp) in sorted(per.items()):
        g0, tau, fit, gs = fit_threshold(g, gain, a.tol)
        out[name] = (g0, tau)
        splits = sorted(set(sp.tolist()))
        g0s = [fit_threshold(g[sp == s], gain[sp == s], a.tol)[0] for s in splits]
        spread = " ".join(f"{x:.3f}" for x in g0s)
        if len(g0s) > 1:
            spread += f" (sd {np.std(g0s):.3f})"
        print(f"{name:16s} {len(g):6d} {g0:7.3f} {tau:6.3f}  {spread:24s} {fit[0]:+6.2f} .. {fit[-1]:+6.2f}")
    allg = np.concatenate([v[0] for v in per.values()])
    allgain = np.concatenate([v[1] for v in per.values()])
    g0p, taup, _, _ = fit_threshold(allg, allgain, a.tol)
    print(f"{'pooled':16s} {len(allg):6d} {g0p:7.3f} {taup:6.3f}")
    if a.out:
        json.dump({"per_dataset": out, "pooled": (g0p, taup)}, open(a.out, "w"), indent=2)
        print(f"saved -> {a.out}  (use adaptive_correct.py --calib {a.out})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--dataset"); ap.add_argument("--out", default="")
    ap.add_argument("--summarize", nargs="*", default=None)
    ap.add_argument("--tol", type=float, default=0.0,
                    help="net gain we require before applying the correction (points)")
    ap.add_argument("--backbone", default="ViT-B/16"); ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--learner", default="maple", choices=["maple", "ivlp", "coop", "kgcoop", "promptsrc", "cocoop"])
    ap.add_argument("--kg_lambda", type=float, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--n_ctx", type=int, default=None); ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", default="0,1,2", help="B1/B2 split ids; 0 = original split")
    ap.add_argument("--retrain", action="store_true", help="ignore saved calibration logits")
    ap.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    if a.summarize is not None:
        return summarize(a)
    fill_defaults(a)

    splits = [int(s) for s in a.splits.split(",")]
    paths = {s: os.path.join(a.out, f"split{s}", "calib_logits.npz") for s in splits}
    todo = [s for s in splits if a.retrain or not os.path.exists(paths[s])]
    if todo:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"): torch.backends.cuda.enable_cudnn_sdp(False)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = load_dataset(a.root, a.dataset); names_all = classnames_of(data["test"])
        base, _ = base_novel_split(data)
        for s in todo:
            train_and_dump(a, data, names_all, base, s, dev, paths[s])
    else:
        print(f"{a.dataset}: reusing saved logits for splits {splits} (CPU re-sweep)")

    alphas = [float(x) for x in a.alphas.split(",")]
    G, GAIN, A, S = [], [], [], []
    for s in splits:
        d = np.load(paths[s])
        lg, zs, y = d["logits"], d["zs_logits"], d["labels"]
        g, gain, al = sweep(lg, zs, y, lg.shape[1], alphas, a.reps)
        G.append(g); GAIN.append(gain); A.append(al); S.append(np.full(len(g), s))
        print(f"  split {s}: {len(g)} points, gini {g.min():.3f}..{g.max():.3f}")
    np.savez(os.path.join(a.out, "points.npz"), g=np.concatenate(G), gain=np.concatenate(GAIN),
             alpha=np.concatenate(A), split=np.concatenate(S))
    print(f"  saved {sum(map(len, G))} calibration points -> {a.out}/points.npz")


if __name__ == "__main__":
    main()
