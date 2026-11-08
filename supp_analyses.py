"""Remaining supplementary experiments, all on saved logits (CPU; E3 also needs CLIP's text encoder).

  python supp_analyses.py --out runs/supp_analysis.json            # A2, A4, K3, M6 for all learners
  python supp_analyses.py --out runs/supp_analysis.json --clip     # also E3 (needs `import clip`)

  A2  gate signal: gini (GMT) vs entropy gap vs KL to uniform vs log max/min ratio. Each signal gets
      its own g0/tau, fitted on the SAME calibration subsets and gains as GMT (no new Sinkhorn runs:
      the subsets are regenerated with calibrate_mixed.py's seeds and checked against the stored gini).
  A4  Sinkhorn iterations (1, 5, 20, 200 = GMT) and blending in log space (GMT) vs probability space.
  K3  calibration size: 1 / 2 / 3 pseudo-splits, and 5 instead of 20 subsets per skew.
  M6  runtime of the gate + Sinkhorn + blend per test batch (numpy, one CPU thread).
  E6  zero-shot reference: novel->base and base->novel rates and the standard-vs-joint HM drop of
      zero-shot CLIP, the adapted model and GMT; and the label balance of each test set vs the
      zero-shot predicted balance (why the gate misreads FGVC-Aircraft).
  E3  mechanism: for every novel class, its CLIP text similarity to the nearest base class vs the
      share of its images sent to base classes (Spearman correlation per dataset).
make_assets.py turns runs/supp_analysis.json into supplementary tables and a figure.
"""
import argparse, glob, json, os, time
import numpy as np
from correct_bias import softmax, sinkhorn
from adaptive_correct import gini
from analysis_suite import fit_threshold, skewed_subset

LEARNERS = ["coop", "cocoop", "kgcoop", "ivlp", "maple", "promptsrc"]
DATASETS = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers",
            "oxford_pets", "ucf101"]
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
ZS_GMT = False


def dirs(L, s):
    if L == "maple":
        return ("runs/logits", "runs/calib") if s == 1 else (f"runs/logits_s{s}", f"runs/calib_s{s}")
    return f"runs/{L}/logits_s{s}", f"runs/{L}/calib_s{s}"


def signals(m):
    C = len(m); m = np.maximum(m, 1e-12)
    return {"gini": gini(m),
            "entropy": float(1 - (-(m * np.log(m)).sum()) / np.log(C)),
            "kl_uniform": float((m * np.log(m * C)).sum()),
            "maxmin": float(np.log(m.max() / m.min()))}


def regenerate(cfiles, reps):
    """Signals, split id, skew and rep index of every calibration subset, in points.npz order."""
    rows = []
    for k, f in enumerate(cfiles):
        c = np.load(f); y, zs = c["labels"], c["zs_logits"]; C = zs.shape[1]
        pz = softmax(zs)
        for alpha in ALPHAS:
            for r in range(reps):
                rng = np.random.default_rng(r * 977 + int(round(alpha * 100)))
                sub = skewed_subset(y, alpha, rng, C) if alpha > 0 else np.arange(len(y))
                rows.append(dict(split=k, alpha=alpha, rep=r, **signals(pz[sub].mean(0))))
                if alpha == 0:
                    break
    return rows


def lam(s, g0, tau):
    return float(1.0 / (1.0 + np.exp((s - g0) / tau)))


def blend_log(P, Q, l):
    return ((1 - l) * np.log(np.maximum(P, 1e-12)) + l * np.log(np.maximum(Q, 1e-12))).argmax(-1)


def acc(p, y):
    return float((p == y).mean() * 100)


def one(L, s, d, reps):
    ld, cd = dirs(L, s)
    pf = f"{cd}/{d}/points.npz"; lf = f"{ld}/{d}/logits.npz"
    cfiles = [f"{cd}/{d}/split{k}/calib_logits.npz" for k in range(3)]
    if not (os.path.exists(pf) and os.path.exists(lf) and all(map(os.path.exists, cfiles))):
        return None
    pts = np.load(pf); gain = pts["gain"]
    rows = regenerate(cfiles, reps)
    if len(rows) != len(gain) or np.max(np.abs(np.array([r["gini"] for r in rows]) - pts["g"])) > 1e-6:
        print(f"  ! {L} s{s} {d}: regenerated subsets do not match points.npz; skipped")
        return None
    t = np.load(lf, allow_pickle=True)
    lg, zs, y = t["logits"], t["zs_logits"], t["labels"]
    C = lg.shape[1]; u = np.full(C, 1.0 / C)
    P = softmax(lg); Q = sinkhorn(P, u)
    m_test = signals(softmax(zs).mean(0))
    raw = acc(lg.argmax(-1), y)
    out = {"raw": raw}

    # E6: zero-shot reference and asymmetry (is the drop bias, or just more classes competing?)
    nb = int(t["n_base"]); isn = y >= nb
    def rates(pred, logits=None):
        d = {"n2b": float((pred[isn] < nb).mean() * 100), "b2n": float((pred[~isn] >= nb).mean() * 100),
             "mixed": float((pred == y).mean() * 100),
             "base_img": float((pred[~isn] == y[~isn]).mean() * 100), "novel_img": float((pred[isn] == y[isn]).mean() * 100)}
        hm = lambda b, n: 2 * b * n / (b + n) if b + n > 0 else 0.0
        d["joint_hm"] = hm(d["base_img"], d["novel_img"])
        if logits is not None:
            b = float((logits[~isn, :nb].argmax(-1) == y[~isn]).mean() * 100)
            n = float((logits[isn, nb:].argmax(-1) == y[isn] - nb).mean() * 100)
            d["std_hm"] = hm(b, n)
        return d
    out["asym"] = {"zs": rates(zs.argmax(-1), zs), "raw": rates(lg.argmax(-1), lg)}
    if ZS_GMT:
        # fair reference: GMT applied to zero-shot CLIP, with its own gate calibrated on the zero-shot
        # calibration logits (Sinkhorn gains on the same skewed subsets)
        G, GA = [], []
        for f in cfiles:
            c = np.load(f); cy, cz = c["labels"], c["zs_logits"]; Cc = cz.shape[1]; uc = np.full(Cc, 1.0 / Cc)
            for alpha in ALPHAS:
                for r in range(reps):
                    rng = np.random.default_rng(r * 977 + int(round(alpha * 100)))
                    sub = skewed_subset(cy, alpha, rng, Cc) if alpha > 0 else np.arange(len(cy))
                    Pz = softmax(cz[sub])
                    G.append(gini(Pz.mean(0)))
                    GA.append(((sinkhorn(Pz, uc).argmax(-1) == cy[sub]).mean() - (cz[sub].argmax(-1) == cy[sub]).mean()) * 100)
                    if alpha == 0:
                        break
        zg0, ztau = fit_threshold(np.array(G), np.array(GA))
        Pz = softmax(zs); Qz = sinkhorn(Pz, u)
        out["asym"]["zs_gmt"] = rates(blend_log(Pz, Qz, lam(m_test["gini"], zg0, ztau)))
    m_true = np.bincount(y, minlength=C) / len(y)
    out["balance"] = {"label_gini": gini(m_true), "zs_gini": m_test["gini"]}

    # A2: gate signals
    out["signals"] = {}
    for name in ("gini", "entropy", "kl_uniform", "maxmin"):
        g0, tau = fit_threshold(np.array([r[name] for r in rows]), gain)
        out["signals"][name] = acc(blend_log(P, Q, lam(m_test[name], g0, tau)), y) - raw
    g0, tau = fit_threshold(pts["g"], gain)
    l = lam(m_test["gini"], g0, tau)
    out["asym"]["gmt"] = rates(blend_log(P, Q, l))
    out["balance"]["lam"] = l

    # A4: Sinkhorn iterations and blending space
    out["sinkhorn"] = {f"iters_{it}": acc(blend_log(P, sinkhorn(P, u, iters=it), l), y) - raw
                       for it in (1, 5, 20)}
    out["sinkhorn"]["iters_200"] = acc(blend_log(P, Q, l), y) - raw
    out["sinkhorn"]["prob_blend"] = acc(((1 - l) * P + l * Q).argmax(-1), y) - raw

    # K3: calibration size
    split = np.array([r["split"] for r in rows]); rep = np.array([r["rep"] for r in rows])
    variants = {"1 split": split == 0, "2 splits": split <= 1, "3 splits (GMT)": split <= 2,
                "3 splits, 5 subsets per skew": rep < 5, "1 split, 5 subsets per skew": (split == 0) & (rep < 5)}
    out["calib_size"] = {}
    for name, mask in variants.items():
        g0v, tauv = fit_threshold(pts["g"][mask], gain[mask])
        out["calib_size"][name] = {"gain": acc(blend_log(P, Q, lam(m_test["gini"], g0v, tauv)), y) - raw,
                                   "points": int(mask.sum())}
    return out


def runtime(path, reps=50):
    t = np.load(path, allow_pickle=True)
    lg, zs = t["logits"], t["zs_logits"]; N, C = lg.shape
    rng = np.random.default_rng(0); res = {}
    for bs in (16, 64, 256, 1024, N):
        ts = []
        for _ in range(reps if bs < N else 3):
            idx = rng.choice(N, bs, replace=False)
            t0 = time.perf_counter()
            P = softmax(lg[idx]); g = gini(softmax(zs[idx]).mean(0))
            Q = sinkhorn(P, np.full(C, 1.0 / C))
            blend_log(P, Q, lam(g, 0.2, 0.02))
            ts.append(time.perf_counter() - t0)
        key = "full" if bs == N else str(bs)
        res[key] = {"ms_per_batch": 1000 * float(np.median(ts)), "us_per_image": 1e6 * float(np.median(ts)) / bs,
                    "batch": int(bs)}
    res["classes"] = int(C); res["images"] = int(N); res["file"] = path
    return res


def mechanism(learners):
    import torch, clip
    model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
    feats, out = {}, {}
    for L in learners:
        pts = []
        for d in DATASETS:
            n2b_raw, n2b_gmt = [], []
            for s in (1, 2, 3):
                lf = f"{dirs(L, s)[0]}/{d}/logits.npz"
                if not os.path.exists(lf):
                    break
                t = np.load(lf, allow_pickle=True)
                lg, y, nb = t["logits"], t["labels"], int(t["n_base"])
                names = [str(n) for n in t["classnames"]]
                isn = y >= nb
                pred = lg.argmax(-1)
                n2b_raw.append([float((pred[y == c] < nb).mean()) * 100 for c in range(nb, lg.shape[1])])
            if len(n2b_raw) < 3:
                continue
            if d not in feats:
                with torch.no_grad():
                    tf = model.encode_text(clip.tokenize([f"a photo of a {n.replace('_', ' ')}." for n in names])).float()
                feats[d] = (torch.nn.functional.normalize(tf, dim=-1).numpy(), nb)
            T, nb = feats[d]
            sim = (T[nb:] @ T[:nb].T).max(1)
            r = np.mean(n2b_raw, 0)
            pts += [{"dataset": d, "sim": float(a), "n2b": float(b)} for a, b in zip(sim, r)]
        if pts:
            out[L] = pts
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/supp_analysis.json")
    ap.add_argument("--reps", type=int, default=20, help="must match the calibration sweep (default 20)")
    ap.add_argument("--learners", default=",".join(LEARNERS))
    ap.add_argument("--clip", action="store_true", help="also run E3 (needs the clip package)")
    ap.add_argument("--zs-gmt-learner", default="maple",
                    help="learner whose calibration files are used for the zero-shot + GMT reference ('' to skip)")
    a = ap.parse_args()
    Ls = a.learners.split(",")
    R = {"per_run": {}, "runtime": None, "mechanism": None}
    global ZS_GMT
    for L in Ls:
        ZS_GMT = (L == a.zs_gmt_learner)
        R["per_run"][L] = {}
        for s in (1, 2, 3):
            for d in DATASETS:
                r = one(L, s, d, a.reps)
                if r is not None:
                    R["per_run"][L].setdefault(str(s), {})[d] = r
            print(f"{L} seed {s}: {len(R['per_run'][L].get(str(s), {}))}/8 datasets", flush=True)
    big = max(glob.glob("runs/logits*/food101/logits.npz") + glob.glob("runs/*/logits_s*/food101/logits.npz"),
              key=os.path.getsize, default=None)
    if big:
        R["runtime"] = runtime(big)
        print("runtime:", {k: v for k, v in R["runtime"].items() if isinstance(v, dict)}, flush=True)
    if a.clip:
        M = mechanism(Ls)
        R["mechanism"] = {L: {"points": pts,
                              "spearman_per_dataset": {d: spearman([p["sim"] for p in pts if p["dataset"] == d],
                                                                    [p["n2b"] for p in pts if p["dataset"] == d])
                                                       for d in DATASETS if sum(p["dataset"] == d for p in pts) > 3}}
                          for L, pts in M.items()}
        for L, v in R["mechanism"].items():
            print(f"E3 {L}: mean Spearman {np.nanmean(list(v['spearman_per_dataset'].values())):.2f}")
    json.dump(R, open(a.out, "w"))
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
