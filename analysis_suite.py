"""CPU analysis suite on saved logits, for every learner and seed that has finished.

  python analysis_suite.py --out runs/analysis.json            # everything available
  python analysis_suite.py --learners maple --quick              # fast check

Per learner x seed x dataset it computes (experiment ids from the paper plan):
  main  raw / full Sinkhorn / GMT: mixed acc, base-image acc, novel-image acc, N->B, and the
        standard base-to-novel HM (E2, M2)
  K4    GMT with a pooled g0/tau and with leave-one-dataset-out g0/tau
  A3    target marginal: uniform (GMT) vs zero-shot-estimated vs true test marginal (oracle)
  M3    synthetic label shift on the test set: Zipf skew alpha in 0..2 over all test classes
  M4    share of novel images in the test set: 10..90%
  GMT-E (when runs/baselines3_<learner>.json exists) is evaluated in main, M3, M4 and M5 too.
  M5    batch size: GMT and Sinkhorn applied independently per random batch, and online GMT
        (gate on the running zero-shot marginal, Sinkhorn scaling on a sliding window)
The gate threshold g0/tau is fitted from each run's own calibration points (calib dir), so no
calib.json is needed. Everything is label-free except the oracle target in A3.
"""
import argparse, glob, json, os, re
import numpy as np
from correct_bias import softmax, sinkhorn, report
from adaptive_correct import gini


# ---------------- threshold fit (copied from calibrate_mixed.py, numpy only)
def isotonic_decreasing(x, y):
    order = np.argsort(x)
    yv = list(-np.asarray(y, float)[order])
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
    g = np.asarray(g, float); gain = np.asarray(gain, float)
    fit = isotonic_decreasing(g, gain)
    o = np.argsort(g); gs, fs = g[o], fit[o]
    below = np.where(fs < -tol)[0]
    if len(below) == 0:
        g0 = float(gs.max() + 0.05)
    elif below[0] == 0:
        g0 = float(gs.min())
    else:
        k = below[0]
        x0, x1, y0, y1 = gs[k - 1], gs[k], fs[k - 1] + tol, fs[k] + tol
        g0 = float(x0 + (x1 - x0) * y0 / (y0 - y1)) if y0 != y1 else float(x1)
    span = gs[(fs < -tol + 3) & (fs > -tol - 3)]
    tau = float(max(0.01, (span.max() - span.min()) / 4)) if len(span) > 1 else 0.02
    return g0, tau


def skewed_subset(y, alpha, rng, C):                      # identical to calibrate_mixed.py
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


# ---------------- corrections
def lam_of(zs, g0, tau):
    g = gini(softmax(zs).mean(0))
    return float(1.0 / (1.0 + np.exp((g - g0) / tau))), g


def blend(P, Q, lam):
    return ((1 - lam) * np.log(np.maximum(P, 1e-12)) + lam * np.log(np.maximum(Q, 1e-12))).argmax(-1)


def methods(lg, zs, g0, tau):
    """Predictions of raw, full Sinkhorn and GMT on one set of images."""
    P = softmax(lg); C = P.shape[1]
    Q = sinkhorn(P, np.full(C, 1.0 / C))
    lam, _ = lam_of(zs, g0, tau)
    return {"raw": lg.argmax(-1), "full": Q.argmax(-1), "gmt": blend(P, Q, lam)}, lam


def colscale(P, target, w, iters):
    """Sinkhorn written as a per-class column scaling w (equivalent to sinkhorn())."""
    for _ in range(iters):
        Q = P * w
        Q /= np.maximum(Q.sum(-1, keepdims=True), 1e-9)
        w = w * target / np.maximum(Q.mean(0), 1e-9)
    return w


def stream_gmt(P, zs, idx, bs, g0, tau, window=1024, iters=20):
    """Online GMT: images arrive in batches of bs in the order idx. The gate uses the zero-shot
    marginal of every image seen so far; the Sinkhorn class scaling is re-fitted (warm start) on
    a sliding window of the last `window` images and applied to the current batch. Label-free."""
    N, C = P.shape
    target = np.full(C, 1.0 / C); w = np.ones(C)
    pzs = softmax(zs); zsum = np.zeros(C); seen = 0
    pred = np.empty(N, int); lams = []
    for s0 in range(0, N, bs):
        sl = idx[s0:s0 + bs]
        zsum += pzs[sl].sum(0); seen += len(sl)
        lam = float(1.0 / (1.0 + np.exp((gini(zsum / seen) - g0) / tau)))
        buf = idx[max(0, s0 + len(sl) - window):s0 + len(sl)]
        w = colscale(P[buf], target, w, iters)
        Q = P[sl] * w; Q /= Q.sum(-1, keepdims=True)
        pred[sl] = blend(P[sl], Q, lam); lams.append(lam)
    return pred, float(np.mean(lams))


def acc(pred, y):
    return float((pred == y).mean() * 100)


def b2n_hm(lg, y, nb):
    isn = y >= nb
    b = (lg[~isn, :nb].argmax(-1) == y[~isn]).mean() * 100
    n = (lg[isn, nb:].argmax(-1) == y[isn] - nb).mean() * 100
    return float(b), float(n), float(2 * b * n / (b + n)) if b + n > 0 else 0.0


# ---------------- per-dataset experiments
def gmte_pred(lg, zs, gm):
    """GMT-E prediction: GMT on the calibrated zero-shot ensemble, gate re-calibrated for it."""
    E = (1 - gm["beta"]) * lg + gm["beta"] * zs
    pr, l = methods(E, zs, gm["g0"], gm["tau"])
    return pr["gmt"], l


def analyse(lg, zs, y, nb, g0, tau, g0p, taup, g0l, taul, reps, rng_seed=0, gm=None):
    out = {}
    P = softmax(lg); C = P.shape[1]
    Q = sinkhorn(P, np.full(C, 1.0 / C))
    lam, g = lam_of(zs, g0, tau)
    preds = {"raw": lg.argmax(-1), "full": Q.argmax(-1), "gmt": blend(P, Q, lam),
             "gmt_pooled": blend(P, Q, lam_of(zs, g0p, taup)[0]),
             "gmt_lodo": blend(P, Q, lam_of(zs, g0l, taul)[0])}
    # A3: other target marginals, same gate
    m_zs = softmax(zs).mean(0)
    m_true = np.bincount(y, minlength=C) / len(y)
    Qz = sinkhorn(P, m_zs); Qt = sinkhorn(P, m_true)
    preds.update({"full_zs_target": Qz.argmax(-1), "gmt_zs_target": blend(P, Qz, lam),
                  "full_true_target": Qt.argmax(-1), "gmt_true_target": blend(P, Qt, lam)})
    if gm:
        preds["gmte"], _ = gmte_pred(lg, zs, gm)
        preds["zs_ens"] = ((1 - gm["beta"]) * lg + gm["beta"] * zs).argmax(-1)
    out["res"] = {k: report(p, y, nb) for k, p in preds.items()}
    out["b2n"] = dict(zip(["base", "novel", "hm"], b2n_hm(lg, y, nb)))
    out["gate"] = dict(gini=g, lam=lam, g0=g0, tau=tau, g0_lodo=g0l, tau_lodo=taul)

    # M3: synthetic label shift on the test set
    out["skew"] = {}
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
        r = {"raw": [], "full": [], "gmt": [], "lam": []} | ({"gmte": []} if gm else {})
        for rep in range(1 if alpha == 0 else reps):
            rng = np.random.default_rng(1000 + rep * 17 + int(alpha * 10))
            sub = np.arange(len(y)) if alpha == 0 else skewed_subset(y, alpha, rng, C)
            pr, l = methods(lg[sub], zs[sub], g0, tau)
            for k in ("raw", "full", "gmt"):
                r[k].append(acc(pr[k], y[sub]))
            r["lam"].append(l)
            if gm:
                r["gmte"].append(acc(gmte_pred(lg[sub], zs[sub], gm)[0], y[sub]))
        out["skew"][str(alpha)] = {k: float(np.mean(v)) for k, v in r.items()}

    # M4: share of novel images
    out["novel_frac"] = {}
    ib, inn = np.where(y < nb)[0], np.where(y >= nb)[0]
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        r = {"raw": [], "full": [], "gmt": [], "lam": []} | ({"gmte": []} if gm else {})
        for rep in range(reps):
            rng = np.random.default_rng(2000 + rep * 17 + int(p * 10))
            n_nov = min(len(inn), int(round(p / (1 - p) * len(ib))))
            n_base = min(len(ib), int(round((1 - p) / p * n_nov)))
            sub = np.concatenate([rng.choice(ib, n_base, replace=False), rng.choice(inn, n_nov, replace=False)])
            pr, l = methods(lg[sub], zs[sub], g0, tau)
            for k in ("raw", "full", "gmt"):
                r[k].append(acc(pr[k], y[sub]))
            r["lam"].append(l)
            if gm:
                r["gmte"].append(acc(gmte_pred(lg[sub], zs[sub], gm)[0], y[sub]))
        out["novel_frac"][str(p)] = {k: float(np.mean(v)) for k, v in r.items()}

    # M5: batch size
    out["batch"] = {}
    for bs in [16, 32, 64, 128, 256, 512, "full"]:
        r = {"full": [], "gmt": [], "stream": [], "lam": [], "lam_str": []} | \
            ({"gmte": [], "gmte_stream": []} if gm else {})
        for rep in range(1 if bs == "full" else max(1, reps // 2)):
            rng = np.random.default_rng(3000 + rep)
            idx = rng.permutation(len(y)); step = len(y) if bs == "full" else bs
            pf = np.empty(len(y), int); pg = np.empty(len(y), int); ls = []
            for s0 in range(0, len(y), step):
                sl = idx[s0:s0 + step]
                pr, l = methods(lg[sl], zs[sl], g0, tau)
                pf[sl] = pr["full"]; pg[sl] = pr["gmt"]; ls.append(l)
            r["full"].append(acc(pf, y)); r["gmt"].append(acc(pg, y)); r["lam"].append(float(np.mean(ls)))
            if bs == "full":
                r["stream"].append(acc(pg, y)); r["lam_str"].append(float(np.mean(ls)))
            else:
                ps, lst = stream_gmt(P, zs, idx, bs, g0, tau)
                r["stream"].append(acc(ps, y)); r["lam_str"].append(lst)
            if gm:
                E = (1 - gm["beta"]) * lg + gm["beta"] * zs
                pe = np.empty(len(y), int)
                for s0 in range(0, len(y), step):
                    sl = idx[s0:s0 + step]
                    pe[sl] = methods(E[sl], zs[sl], gm["g0"], gm["tau"])[0]["gmt"]
                r["gmte"].append(acc(pe, y))
                if bs == "full":
                    r["gmte_stream"].append(acc(pe, y))
                else:
                    r["gmte_stream"].append(acc(stream_gmt(softmax(E), zs, idx, bs, gm["g0"], gm["tau"])[0], y))
        out["batch"][str(bs)] = {k: float(np.mean(v)) for k, v in r.items()}
    out["batch"]["raw"] = acc(lg.argmax(-1), y)
    return out


def run_dirs(learner, seed):
    if learner == "maple":
        return ("runs/logits", "runs/calib") if seed == 1 else (f"runs/logits_s{seed}", f"runs/calib_s{seed}")
    return f"runs/{learner}/logits_s{seed}", f"runs/{learner}/calib_s{seed}"


def gmte_params(learner, seed):
    """beta, g0, tau of GMT-E for each dataset, as fitted by baselines_cs.py (calibration only)."""
    for f in (f"runs/baselines3_{learner}.json",):
        if os.path.exists(f):
            d = json.load(open(f))
            for spec, rows in d.items():
                m = re.search(r"logits_s(\d+)", spec)
                if int(m.group(1) if m else 1) == seed:
                    return {n: r["gmte"] for n, r in rows.items() if "gmte" in r}
    return {}


def run(learner, seed, reps):
    ld, cd = run_dirs(learner, seed)
    gms = gmte_params(learner, seed)
    pts = {}
    for f in sorted(glob.glob(os.path.join(cd, "*", "points.npz"))):
        d = np.load(f); pts[os.path.basename(os.path.dirname(f))] = (d["g"], d["gain"])
    rows = {}
    names = [n for n in pts if os.path.exists(os.path.join(ld, n, "logits.npz"))]
    if not names:
        return rows
    allg = np.concatenate([pts[n][0] for n in names]); allgain = np.concatenate([pts[n][1] for n in names])
    g0p, taup = fit_threshold(allg, allgain)
    for n in names:
        d = np.load(os.path.join(ld, n, "logits.npz"), allow_pickle=True)
        lg, zs, y, nb = d["logits"], d["zs_logits"], d["labels"], int(d["n_base"])
        g0, tau = fit_threshold(*pts[n])
        others = [m for m in names if m != n]
        g0l, taul = fit_threshold(np.concatenate([pts[m][0] for m in others]),
                                  np.concatenate([pts[m][1] for m in others])) if others else (g0p, taup)
        print(f"  {learner} s{seed} {n}", flush=True)
        rows[n] = analyse(lg, zs, y, nb, g0, tau, g0p, taup, g0l, taul, reps, gm=gms.get(n))
    return rows


def ms(vals):
    v = np.array(vals, float)
    return f"{v.mean():6.2f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:4.2f}"


def summarise(R):
    for L, seeds in R.items():
        common = sorted(set.intersection(*[set(r) for r in seeds.values()])) if seeds else []
        if not common:
            continue
        S = list(seeds.values())
        avg = lambda f: [np.mean([f(r[n]) for n in common]) for r in S]
        print(f"\n################ {L}: {len(S)} seed(s) x {len(common)} datasets ({', '.join(common)})")
        print(f"standard base-to-novel HM        {ms(avg(lambda x: x['b2n']['hm']))}")
        for k, lab in [("raw", "raw"), ("full", "full Sinkhorn"), ("gmt", "GMT"),
                       ("gmt_pooled", "GMT pooled g0"), ("gmt_lodo", "GMT leave-one-dataset-out"),
                       ("full_zs_target", "full SK, zero-shot target"), ("gmt_zs_target", "GMT, zero-shot target"),
                       ("full_true_target", "full SK, true target (oracle)"), ("gmt_true_target", "GMT, true target (oracle)"),
                       ("zs_ens", "zero-shot ensemble (sweep beta)"), ("gmte", "GMT-E")]:
            if k not in S[0][common[0]]["res"]:
                continue
            mix = avg(lambda x: x["res"][k]["mixed"])
            gain = avg(lambda x: x["res"][k]["mixed"] - x["res"]["raw"]["mixed"])
            worst = min(r[n]["res"][k]["mixed"] - r[n]["res"]["raw"]["mixed"] for r in S for n in common)
            print(f"{lab:32s} mixed {ms(mix)}  gain {ms(gain)}  worst {worst:+6.2f}  "
                  f"base-img {np.mean(avg(lambda x: x['res'][k]['base_img'])):6.2f}  "
                  f"novel-img {np.mean(avg(lambda x: x['res'][k]['novel_img'])):6.2f}  "
                  f"N->B {np.mean(avg(lambda x: x['res'][k]['n2b'])):6.2f}")
        has_e = "gmte" in S[0][common[0]]["skew"]["0.0"]
        e1 = ["gmte"] if has_e else []
        e2 = ["gmte", "gmte_stream"] if has_e else []
        for key, title, cols in [("skew", "M3 test label shift (alpha)", ["raw", "full", "gmt"] + e1 + ["lam"]),
                                 ("novel_frac", "M4 share of novel images", ["raw", "full", "gmt"] + e1 + ["lam"]),
                                 ("batch", "M5 batch size (stream = online GMT)", ["full", "gmt", "stream"] + e2 + ["lam", "lam_str"])]:
            print(f"\n{title}:  " + "  ".join(f"{c:>7s}" for c in cols))
            for x in [k for k in S[0][common[0]][key] if k != "raw"]:
                vals = [np.mean(avg(lambda r, c=c: r[key][x][c])) for c in cols]
                print(f"  {x:>8s}{'':22s}" + "  ".join(f"{v:7.2f}" if not c.startswith("lam") else f"{v:7.3f}"
                                                        for v, c in zip(vals, cols)))
            if key == "batch":
                print(f"  {'raw':>8s}{'':22s}{np.mean(avg(lambda r: r['batch']['raw'])):7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", default="maple,coop,kgcoop,ivlp")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="reps=2, for a fast check")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    reps = 2 if a.quick else a.reps
    R = {}
    for L in a.learners.split(","):
        R[L] = {}
        for s in [int(x) for x in a.seeds.split(",")]:
            rows = run(L, s, reps)
            if rows:
                R[L][s] = rows
            else:
                print(f"  {L} s{s}: no finished datasets yet")
    summarise(R)
    if a.out:
        json.dump(R, open(a.out, "w"), indent=1, default=float)
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
