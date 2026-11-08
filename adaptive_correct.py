"""Imbalance-aware marginal correction, with no hard threshold.

The strength of the correction is a smooth function of how skewed the test-set class
distribution looks, estimated from the frozen zero-shot head (no labels, no test-set statistics
beyond the predictions the model already makes):

    gini(m)  in [0,1]   skew of the zero-shot predicted class marginal m
    lambda   = sigmoid( -(gini - g0) / tau )     in [0,1]
    output   = softmax( (1-lambda) log P + lambda log Sinkhorn(P, uniform) )

lambda -> 1 for a balanced-looking test set (full correction), -> 0 for a skewed one (fall back
to the raw prediction). g0 and tau are calibrated on base-class validation data, not on the test
set: see calibrate().

  python adaptive_correct.py runs/logits/*/logits.npz
  python adaptive_correct.py runs/logits/*/logits.npz --g0 0.29 --tau 0.02
"""
import argparse, json, os
import numpy as np
from correct_bias import softmax, sinkhorn, report


def gini(m):
    m = np.sort(m) / m.sum()
    n = len(m)
    return float(1 - 2 * np.trapezoid(np.cumsum(m), dx=1.0 / n)) if hasattr(np, "trapezoid") \
        else float(1 - 2 * np.trapz(np.cumsum(m), dx=1.0 / n))


def zs_gini(zs_logits):
    return gini(softmax(zs_logits).mean(0))


def adaptive(P, zs_logits, g0=0.29, tau=0.02, iters=200):
    g = zs_gini(zs_logits)
    lam = 1.0 / (1.0 + np.exp((g - g0) / tau))
    Q = sinkhorn(P, np.full(P.shape[1], 1.0 / P.shape[1]), iters)
    L = (1 - lam) * np.log(np.maximum(P, 1e-12)) + lam * np.log(np.maximum(Q, 1e-12))
    return softmax(L), g, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--g0", type=float, default=0.29, help="gini at which the correction is halved")
    ap.add_argument("--tau", type=float, default=0.02, help="softness of the transition")
    ap.add_argument("--calib", default="", help="json from calibrate_mixed.py --summarize --out")
    ap.add_argument("--pooled", action="store_true", help="use the pooled g0 instead of per-dataset")
    a = ap.parse_args()
    calib = json.load(open(a.calib)) if a.calib else None
    rows = []
    print(f"{'dataset':16s} {'gini':>6s} {'lambda':>7s} {'raw':>7s} {'full':>7s} {'ours':>7s} "
          f"{'gain':>7s} {'n2b raw':>8s} {'n2b ours':>9s}")
    for f in sorted(a.npz):
        d = np.load(f, allow_pickle=True)
        lg, zs, y, nb = d["logits"], d["zs_logits"], d["labels"], int(d["n_base"])
        C = lg.shape[1]; P = softmax(lg)
        name = os.path.basename(os.path.dirname(f))
        r_raw = report(lg.argmax(-1), y, nb)
        r_full = report(sinkhorn(P, np.full(C, 1.0 / C)).argmax(-1), y, nb)
        g0, tau = a.g0, a.tau
        if calib:
            g0, tau = calib["pooled"] if a.pooled or name not in calib["per_dataset"] \
                else calib["per_dataset"][name]
        Q, g, lam = adaptive(P, zs, g0, tau)
        r_our = report(Q.argmax(-1), y, nb)
        rows.append((name, r_raw, r_full, r_our))
        print(f"{name:16s} {g:6.3f} {lam:7.3f} {r_raw['mixed']:7.2f} {r_full['mixed']:7.2f} "
              f"{r_our['mixed']:7.2f} {r_our['mixed']-r_raw['mixed']:+7.2f} "
              f"{r_raw['n2b']:8.2f} {r_our['n2b']:9.2f}")
    if len(rows) > 1:
        mr = np.mean([r[1]["mixed"] for r in rows]); mf = np.mean([r[2]["mixed"] for r in rows])
        mo = np.mean([r[3]["mixed"] for r in rows])
        nr = np.mean([r[1]["n2b"] for r in rows]); no = np.mean([r[3]["n2b"] for r in rows])
        print(f"\n{'average':16s} {'':6s} {'':7s} {mr:7.2f} {mf:7.2f} {mo:7.2f} {mo-mr:+7.2f} "
              f"{nr:8.2f} {no:9.2f}")
        worst = min(r[3]["mixed"] - r[1]["mixed"] for r in rows)
        print(f"worst-case change on any dataset: {worst:+.2f}")


if __name__ == "__main__":
    main()
