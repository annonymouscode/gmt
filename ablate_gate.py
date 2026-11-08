"""Ablation of the imbalance gate, on saved logits (CPU only, no GPU).

Question: is the gini gate needed, or would a simpler choice of correction strength do?
All variants blend the same two predictions,  softmax((1-lam) log P + lam log Sinkhorn(P, u)),
and differ only in how lam is chosen. Everything except orc-lam is label-free.

  raw       lam = 0
  full SK   lam = 1
  fix-glob  one lam for all datasets, tuned on the simulated-mixed calibration sweep
  fix-ds    one lam per dataset, tuned on that dataset's calibration sweep (no test-time signal)
  hard      lam = 1 if gini < g0 else 0   (per-dataset g0, no sigmoid)
  ours      lam = sigmoid(-(gini - g0)/tau), per-dataset g0/tau          (the method)
  pooled    same, one pooled g0/tau for all datasets
  m-gini    same, but the test gini is computed on random K_base-class subsets, so it has the
            same class count as the calibration gini (label-free)
  tau/2, tau*2   ours with the transition twice as sharp / twice as soft (sensitivity)
  orc-lam   per-dataset lam tuned on TEST labels: upper bound for any gate

  python ablate_gate.py --out runs/ablate_gate.json          # seeds 1-3, ~5 min each
"""
import argparse, glob, json, os
import numpy as np
from correct_bias import softmax, sinkhorn, report
from adaptive_correct import gini

METHODS = ["raw", "full SK", "fix-glob", "fix-ds", "hard", "ours", "pooled", "m-gini",
           "tau/2", "tau*2", "orc-lam"]
SHORT = ["raw", "fullSK", "fixGlb", "fixDs", "hard", "ours", "pooled", "mGini", "tau/2", "tau*2", "orcLam"]
LAMS = np.round(np.linspace(0, 1, 21), 2)


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


def logs(P, Q):
    return np.log(np.maximum(P, 1e-12)), np.log(np.maximum(Q, 1e-12))


def blend_pred(lp, lq, lam):
    return ((1 - lam) * lp + lam * lq).argmax(-1)


def acc_curve(lp, lq, y):
    return np.array([(blend_pred(lp, lq, l) == y).mean() for l in LAMS])


def calib_curve(cfiles, alphas, reps):
    """Accuracy vs lam, averaged over the skewed calibration subsets of every split."""
    curves = []
    for f in cfiles:
        c = np.load(f)
        lg, y = c["logits"], c["labels"]; C = lg.shape[1]
        for alpha in alphas:
            for r in range(reps):
                rng = np.random.default_rng(r * 977 + int(round(alpha * 100)))
                sub = skewed_subset(y, alpha, rng, C) if alpha > 0 else np.arange(len(y))
                P = softmax(lg[sub])
                curves.append(acc_curve(*logs(P, sinkhorn(P, np.full(C, 1.0 / C))), y[sub]))
                if alpha == 0:
                    break
    return np.mean(curves, 0)


def matched_gini(zs, k, draws=50, seed=0):
    rng = np.random.default_rng(seed); C = zs.shape[1]
    return float(np.mean([gini(softmax(zs[:, rng.choice(C, k, replace=False)]).mean(0))
                          for _ in range(draws)]))


def sig(g, g0, tau):
    return float(1.0 / (1.0 + np.exp((g - g0) / tau)))


def run_seed(logits_dir, calib_dir, alphas, reps):
    calib = json.load(open(os.path.join(calib_dir, "calib.json")))
    pg0, ptau = calib["pooled"]
    data, curves = {}, {}
    for f in sorted(glob.glob(os.path.join(logits_dir, "*", "logits.npz"))):
        name = os.path.basename(os.path.dirname(f))
        cfiles = sorted(glob.glob(os.path.join(calib_dir, name, "split*", "calib_logits.npz")))
        if not cfiles or name not in calib["per_dataset"]:
            print(f"  skip {name}: no calibration in {calib_dir}"); continue
        print(f"  {name}: calibration sweep ...", flush=True)
        curves[name] = calib_curve(cfiles, alphas, reps)
        data[name] = (f, cfiles)
    lam_glob = float(LAMS[np.argmax(np.mean(list(curves.values()), 0))])

    rows = {}
    for name, (f, cfiles) in data.items():
        d = np.load(f, allow_pickle=True)
        lg, zs, y, nb = d["logits"], d["zs_logits"], d["labels"], int(d["n_base"])
        C = lg.shape[1]; P = softmax(lg)
        lp, lq = logs(P, sinkhorn(P, np.full(C, 1.0 / C)))
        g0, tau = calib["per_dataset"][name]
        g = gini(softmax(zs).mean(0))
        k_cal = int(np.load(cfiles[0])["logits"].shape[1])
        mg = matched_gini(zs, k_cal)
        lam = {"raw": 0.0, "full SK": 1.0, "fix-glob": lam_glob,
               "fix-ds": float(LAMS[np.argmax(curves[name])]),
               "hard": 1.0 if g < g0 else 0.0, "ours": sig(g, g0, tau),
               "pooled": sig(g, pg0, ptau), "m-gini": sig(mg, g0, tau),
               "tau/2": sig(g, g0, tau / 2), "tau*2": sig(g, g0, tau * 2),
               "orc-lam": float(LAMS[np.argmax(acc_curve(lp, lq, y))])}
        rows[name] = dict(res={m: report(blend_pred(lp, lq, lam[m]), y, nb) for m in METHODS},
                          lam=lam, gini=g, m_gini=mg, g0=g0, tau=tau, k_cal=k_cal)
    return rows, lam_glob


def print_seed(tag, rows, lam_glob):
    print(f"\n=== {tag}: mixed accuracy  (fix-glob lam = {lam_glob:.2f}) ===")
    print(f"{'dataset':15s}" + "".join(f"{s:>7s}" for s in SHORT))
    for n, r in rows.items():
        print(f"{n:15s}" + "".join(f"{r['res'][m]['mixed']:7.2f}" for m in METHODS))
    avg = {m: np.mean([r["res"][m]["mixed"] for r in rows.values()]) for m in METHODS}
    worst = {m: min(r["res"][m]["mixed"] - r["res"]["raw"]["mixed"] for r in rows.values()) for m in METHODS}
    print(f"{'average':15s}" + "".join(f"{avg[m]:7.2f}" for m in METHODS))
    print(f"{'worst vs raw':15s}" + "".join(f"{worst[m]:+7.2f}" for m in METHODS))
    print(f"\n{'gate':15s}{'gini':>7s}{'m-gini':>7s}{'g0':>7s}{'lamOur':>7s}{'lamMg':>7s}"
          f"{'lamDs':>7s}{'lamOrc':>7s}")
    for n, r in rows.items():
        L = r["lam"]
        print(f"{n:15s}{r['gini']:7.3f}{r['m_gini']:7.3f}{r['g0']:7.3f}{L['ours']:7.2f}"
              f"{L['m-gini']:7.2f}{L['fix-ds']:7.2f}{L['orc-lam']:7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None,
                    help="logits_dir:calib_dir, repeatable (default: seeds 1-3)")
    ap.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    ap.add_argument("--reps", type=int, default=5, help="skewed subsets per alpha for fix-glob/fix-ds")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    runs = a.run or ["runs/logits:runs/calib", "runs/logits_s2:runs/calib_s2", "runs/logits_s3:runs/calib_s3"]
    alphas = [float(x) for x in a.alphas.split(",")]

    all_rows = {}
    for spec in runs:
        ld, cd = spec.split(":")
        print(f"{spec}", flush=True)
        rows, lg = run_seed(ld, cd, alphas, a.reps)
        all_rows[spec] = rows
        print_seed(spec, rows, lg)

    if len(all_rows) > 1:
        common = sorted(set.intersection(*[set(r) for r in all_rows.values()]))
        print(f"\n=== over {len(all_rows)} seeds x {len(common)} datasets ===")
        print(f"{'method':10s} {'mixed':>15s} {'gain vs raw':>15s} {'worst pair':>11s} {'n2b':>7s}")
        for m in METHODS:
            av = [np.mean([rs[n]["res"][m]["mixed"] for n in common]) for rs in all_rows.values()]
            gn = [np.mean([rs[n]["res"][m]["mixed"] - rs[n]["res"]["raw"]["mixed"] for n in common])
                  for rs in all_rows.values()]
            wp = min(rs[n]["res"][m]["mixed"] - rs[n]["res"]["raw"]["mixed"]
                     for rs in all_rows.values() for n in common)
            n2 = np.mean([rs[n]["res"][m]["n2b"] for rs in all_rows.values() for n in common])
            print(f"{m:10s} {np.mean(av):7.2f} ± {np.std(av, ddof=1):5.2f} "
                  f"{np.mean(gn):+7.2f} ± {np.std(gn, ddof=1):5.2f} {wp:+11.2f} {n2:7.2f}")
        print("(orc-lam uses test labels: upper bound, not a competitor)")
    if a.out:
        json.dump(all_rows, open(a.out, "w"), indent=1, default=float)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
