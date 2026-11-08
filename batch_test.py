"""How small can the test batch be? Sinkhorn applied per batch instead of to the whole test set.

  python batch_test.py runs/logits/*/logits.npz --sizes 16,32,64,128,256,512,full --reps 5

Batches are random (shuffled) by default; --sequential also reports arrival-order batches, which is
the realistic streaming case. Reports mixed accuracy per batch size, averaged over repetitions.
"""
import argparse, os
import numpy as np
from correct_bias import softmax, sinkhorn, report
from adaptive_correct import adaptive


def batched(P, zs, bs, rng, mode="uniform", g0=0.29, tau=0.02, iters=200):
    """Apply the correction independently within each batch.
    mode='uniform': plain Sinkhorn to a uniform marginal.
    mode='adaptive': gini-gated correction, so a batch that looks skewed is left alone."""
    N, C = P.shape
    idx = rng.permutation(N)
    out = np.empty(N, dtype=np.int64)
    lams = []
    step = N if bs == "full" else int(bs)
    for s in range(0, N, step):
        sl = idx[s:s + step]
        if mode == "adaptive":
            Q, g, lam = adaptive(P[sl], zs[sl], g0, tau, iters)
            lams.append(lam)
        else:
            Q = sinkhorn(P[sl], np.full(C, 1.0 / C), iters)
        out[sl] = Q.argmax(-1)
    return out, (float(np.mean(lams)) if lams else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--sizes", default="16,32,64,128,256,512,full")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mode", default="adaptive", choices=["uniform", "adaptive", "both"])
    ap.add_argument("--g0", type=float, default=0.29)
    ap.add_argument("--tau", type=float, default=0.02)
    a = ap.parse_args()
    sizes = a.sizes.split(",")
    table = {}
    for f in a.npz:
        d = np.load(f, allow_pickle=True)
        lg, y, nb = d["logits"], d["labels"], int(d["n_base"])
        name = os.path.basename(os.path.dirname(f))
        P = softmax(lg)
        raw = report(lg.argmax(-1), y, nb)
        zs = d["zs_logits"]
        modes = ["uniform", "adaptive"] if a.mode == "both" else [a.mode]
        row = {"raw": (raw["mixed"], 0.0, 1.0)}
        for mode in modes:
            for bs in sizes:
                accs, lams = [], []
                reps = 1 if bs == "full" else a.reps
                for r in range(reps):
                    rng = np.random.default_rng(r)
                    pred, lam = batched(P, zs, bs, rng, mode, a.g0, a.tau)
                    accs.append(report(pred, y, nb)["mixed"]); lams.append(lam)
                key = bs if len(modes) == 1 else f"{mode[:4]}-{bs}"
                row[key] = (float(np.mean(accs)), float(np.std(accs)), float(np.mean(lams)))
        table[name] = row
        print(f"\n=== {name} ===")
        print(f"{'batch':>12s} {'mixed':>8s} {'+-':>5s} {'mean lambda':>12s}")
        for k, (m, sd, lam) in row.items():
            print(f"{str(k):>12s} {m:8.2f} {sd:5.2f} {lam:12.3f}")
    if len(table) > 1:
        print("\n=== average mixed accuracy over datasets ===")
        keys = list(next(iter(table.values())).keys())
        print(" ".join(f"{k:>8s}" for k in keys))
        print(" ".join(f"{np.mean([t[k][0] for t in table.values()]):8.2f}" for k in keys))


if __name__ == "__main__":
    main()
