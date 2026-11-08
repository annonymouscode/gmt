"""Label-free baselines for the mixed base+novel setting, on saved logits (CPU only, no GPU).

Every tuned baseline gets the SAME information budget as our method: the simulated-mixed
calibration logits (pseudo-base B1 / pseudo-novel B2, split<s>/calib_logits.npz) produced by
calibrate_mixed.py. No test label is used, except by CS-oracle, which is an upper bound.

  raw        argmax of the adapted model (MaPLe)
  zero-shot  argmax of frozen CLIP
  EM prior   Saerens et al. 2002: re-estimate the test label prior and divide it out
  full SK    Sinkhorn to a uniform class marginal, always on
  CS-val     calibrated stacking (Chao et al. 2016): subtract gamma from every base logit;
             gamma maximises accuracy on the simulated-mixed calibration set (mean over splits)
  CS-sweep   same, but gamma maximises accuracy averaged over the skewed calibration subsets
             (the same alpha x reps sweep our g0 is fitted on), so it sees the same skews we do
  ZS-val     ensemble with frozen zero-shot CLIP: (1-beta) * adapted + beta * zero-shot logits;
             beta maximises accuracy on the simulated-mixed calibration set (mean over splits)
  ZS-sweep   same, beta maximises accuracy averaged over the skewed calibration subsets
  ours       imbalance-gated Sinkhorn, g0/tau from <calib_dir>/calib.json
  CS-oracle  gamma tuned on the TEST labels (upper bound for any single base-logit offset)
  ZS-val+ours  GMT applied on top of the ZS-val ensemble (same g0/tau; tests complementarity)
  GMT-E      GMT on the ZS-sweep ensemble, with the gate re-calibrated on the ensembled calibration
             logits (beta, g0 and tau all from calibration; label-free)
  ZS-oracle  beta tuned on the TEST labels (upper bound for any zero-shot ensemble)

  python baselines_cs.py                                    # the three default seeds
  python baselines_cs.py --run runs/logits:runs/calib --out runs/baselines_s1.json
"""
import argparse, glob, json, os
import numpy as np
from correct_bias import softmax, sinkhorn, em_prior, report
from adaptive_correct import adaptive, gini
from analysis_suite import fit_threshold

METHODS = ["raw", "zero-shot", "EM prior", "full SK", "CS-val", "CS-sweep", "ZS-val", "ZS-sweep",
           "ours", "ZS-val+ours", "GMT-E", "CS-oracle", "ZS-oracle"]
BETAS = np.round(np.linspace(0, 1, 21), 2)
GRID = np.round(np.arange(-2.0, 15.0 + 1e-9, 0.05), 3)


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


def correct_matrix(lg, y, nb, grid=GRID):
    """N x G bool: is image i classified correctly after subtracting grid[g] from base logits?
    Base wins ties, matching argmax over [base, novel] columns."""
    d = lg[:, :nb].max(1) - lg[:, nb:].max(1)
    cb = lg[:, :nb].argmax(1) == y
    cn = (lg[:, nb:].argmax(1) + nb) == y
    return np.where(d[:, None] >= grid[None, :], cb[:, None], cn[:, None])


def tune_cs(calib_files, alphas, reps):
    acc_val, acc_sw = [], []
    for f in calib_files:
        c = np.load(f)
        lg, y, nb = c["logits"], c["labels"], int(c["n_b1"])
        M = correct_matrix(lg, y, nb)
        acc_val.append(M.mean(0))
        subs = []
        for alpha in alphas:
            for r in range(reps):
                rng = np.random.default_rng(r * 977 + int(round(alpha * 100)))
                sub = skewed_subset(y, alpha, rng, lg.shape[1]) if alpha > 0 else np.arange(len(y))
                subs.append(M[sub].mean(0))
                if alpha == 0:
                    break
        acc_sw.append(np.mean(subs, 0))
    return float(GRID[np.argmax(np.mean(acc_val, 0))]), float(GRID[np.argmax(np.mean(acc_sw, 0))])


def zs_pred(lg, zs, beta):
    return ((1 - beta) * lg + beta * zs).argmax(-1)


def zs_matrix(lg, zs, y):
    """N x len(BETAS) bool: correct after mixing in zero-shot logits with weight beta."""
    return np.stack([zs_pred(lg, zs, b) == y for b in BETAS], 1)


def tune_zs(calib_files, alphas, reps):
    acc_val, acc_sw = [], []
    for f in calib_files:
        c = np.load(f)
        lg, zs, y = c["logits"], c["zs_logits"], c["labels"]
        M = zs_matrix(lg, zs, y)
        acc_val.append(M.mean(0))
        subs = []
        for alpha in alphas:
            for r in range(reps):
                rng = np.random.default_rng(r * 977 + int(round(alpha * 100)))
                sub = skewed_subset(y, alpha, rng, lg.shape[1]) if alpha > 0 else np.arange(len(y))
                subs.append(M[sub].mean(0))
                if alpha == 0:
                    break
        acc_sw.append(np.mean(subs, 0))
    return float(BETAS[np.argmax(np.mean(acc_val, 0))]), float(BETAS[np.argmax(np.mean(acc_sw, 0))])


def sweep_gain(lg, zs, y, alphas, reps):
    """Gain of full Sinkhorn vs no correction over skewed calibration subsets (as calibrate_mixed.sweep)."""
    C = lg.shape[1]; u = np.full(C, 1.0 / C); G, GAIN = [], []
    for alpha in alphas:
        for r in range(reps):
            rng = np.random.default_rng(r * 977 + int(round(alpha * 100)))
            sub = skewed_subset(y, alpha, rng, C) if alpha > 0 else np.arange(len(y))
            P = softmax(lg[sub]); ys = y[sub]
            raw = (lg[sub].argmax(-1) == ys).mean() * 100
            cor = (sinkhorn(P, u).argmax(-1) == ys).mean() * 100
            G.append(gini(softmax(zs[sub]).mean(0))); GAIN.append(cor - raw)
            if alpha == 0:
                break
    return np.array(G), np.array(GAIN)


def calibrate_ensemble_gate(calib_files, beta, alphas, reps):
    G, GAIN = [], []
    for f in calib_files:
        c = np.load(f)
        E = (1 - beta) * c["logits"] + beta * c["zs_logits"]
        g, gain = sweep_gain(E, c["zs_logits"], c["labels"], alphas, reps)
        G.append(g); GAIN.append(gain)
    return fit_threshold(np.concatenate(G), np.concatenate(GAIN))


def cs_pred(lg, nb, gamma):
    z = lg.copy(); z[:, :nb] -= gamma
    return z.argmax(-1)


def run_seed(logits_dir, calib_dir, alphas, reps):
    calib = json.load(open(os.path.join(calib_dir, "calib.json")))
    rows = {}
    for f in sorted(glob.glob(os.path.join(logits_dir, "*", "logits.npz"))):
        name = os.path.basename(os.path.dirname(f))
        cfiles = sorted(glob.glob(os.path.join(calib_dir, name, "split*", "calib_logits.npz")))
        if not cfiles or name not in calib["per_dataset"]:
            print(f"  skip {name}: no calibration in {calib_dir}"); continue
        d = np.load(f, allow_pickle=True)
        lg, zs, y, nb = d["logits"], d["zs_logits"], d["labels"], int(d["n_base"])
        C = lg.shape[1]; P = softmax(lg)
        g_val, g_sw = tune_cs(cfiles, alphas, reps)
        g_or = float(GRID[np.argmax(correct_matrix(lg, y, nb).mean(0))])
        b_val, b_sw = tune_zs(cfiles, alphas, reps)
        b_or = float(BETAS[np.argmax(zs_matrix(lg, zs, y).mean(0))])
        g0, tau = calib["per_dataset"][name]
        Q, _, lam = adaptive(P, zs, g0, tau)
        Qm, _, _ = adaptive(softmax((1 - b_val) * lg + b_val * zs), zs, g0, tau)   # GMT on the ensemble
        g0E, tauE = calibrate_ensemble_gate(cfiles, b_sw, alphas, reps)
        QE, _, lamE = adaptive(softmax((1 - b_sw) * lg + b_sw * zs), zs, g0E, tauE)
        preds = {"raw": lg.argmax(-1), "zero-shot": zs.argmax(-1), "EM prior": em_prior(P).argmax(-1),
                 "full SK": sinkhorn(P, np.full(C, 1.0 / C)).argmax(-1),
                 "CS-val": cs_pred(lg, nb, g_val), "CS-sweep": cs_pred(lg, nb, g_sw),
                 "ZS-val": zs_pred(lg, zs, b_val), "ZS-sweep": zs_pred(lg, zs, b_sw),
                 "ours": Q.argmax(-1), "ZS-val+ours": Qm.argmax(-1), "GMT-E": QE.argmax(-1), "CS-oracle": cs_pred(lg, nb, g_or),
                 "ZS-oracle": zs_pred(lg, zs, b_or)}
        rows[name] = dict(res={k: report(p, y, nb) for k, p in preds.items()},
                          gamma=dict(val=g_val, sweep=g_sw, oracle=g_or),
                          beta=dict(val=b_val, sweep=b_sw, oracle=b_or), lam=float(lam),
                          gmte=dict(beta=b_sw, g0=g0E, tau=tauE, lam=float(lamE)))
    return rows


def print_seed(tag, rows):
    short = ["raw", "zs", "EM", "fullSK", "CS-val", "CS-swp", "ZS-val", "ZS-swp", "ours", "ZS+our", "GMT-E", "CS-orc", "ZS-orc"]
    print(f"\n=== {tag}: mixed accuracy ===")
    print(f"{'dataset':15s}" + "".join(f"{s:>8s}" for s in short) + f"{'g_val':>7s}{'g_orc':>7s}{'b_val':>7s}{'b_swp':>7s}{'b_orc':>7s}")
    for n, r in rows.items():
        print(f"{n:15s}" + "".join(f"{r['res'][m]['mixed']:8.2f}" for m in METHODS)
              + f"{r['gamma']['val']:7.2f}{r['gamma']['oracle']:7.2f}"
              + f"{r['beta']['val']:7.2f}{r['beta']['sweep']:7.2f}{r['beta']['oracle']:7.2f}")
    avg = {m: np.mean([r["res"][m]["mixed"] for r in rows.values()]) for m in METHODS}
    worst = {m: min(r["res"][m]["mixed"] - r["res"]["raw"]["mixed"] for r in rows.values()) for m in METHODS}
    print(f"{'average':15s}" + "".join(f"{avg[m]:8.2f}" for m in METHODS))
    print(f"{'worst vs raw':15s}" + "".join(f"{worst[m]:+8.2f}" for m in METHODS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None,
                    help="logits_dir:calib_dir, repeatable (default: seeds 1-3)")
    ap.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    runs = a.run or ["runs/logits:runs/calib", "runs/logits_s2:runs/calib_s2", "runs/logits_s3:runs/calib_s3"]
    alphas = [float(x) for x in a.alphas.split(",")]

    all_rows = {}
    for spec in runs:
        ld, cd = spec.split(":")
        all_rows[spec] = run_seed(ld, cd, alphas, a.reps)
        print_seed(spec, all_rows[spec])

    if len(all_rows) > 1:
        common = sorted(set.intersection(*[set(r) for r in all_rows.values()]))
        print(f"\n=== over {len(all_rows)} seeds x {len(common)} datasets ===")
        print(f"{'method':11s} {'mixed':>15s} {'gain vs raw':>15s} {'worst pair':>11s} {'n2b':>7s}")
        for m in METHODS:
            av = [np.mean([rs[n]["res"][m]["mixed"] for n in common]) for rs in all_rows.values()]
            gn = [np.mean([rs[n]["res"][m]["mixed"] - rs[n]["res"]["raw"]["mixed"] for n in common])
                  for rs in all_rows.values()]
            wp = min(rs[n]["res"][m]["mixed"] - rs[n]["res"]["raw"]["mixed"]
                     for rs in all_rows.values() for n in common)
            n2 = np.mean([rs[n]["res"][m]["n2b"] for rs in all_rows.values() for n in common])
            print(f"{m:11s} {np.mean(av):7.2f} ± {np.std(av, ddof=1):5.2f} "
                  f"{np.mean(gn):+7.2f} ± {np.std(gn, ddof=1):5.2f} {wp:+11.2f} {n2:7.2f}")
        print("(CS-oracle and ZS-oracle use test labels: upper bounds, not competitors)")
    if a.out:
        json.dump(all_rows, open(a.out, "w"), indent=1, default=float)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
