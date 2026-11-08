"""Offline test: can label-marginal correction fix base-class bias in the mixed setting?

Runs on logits.npz from dump_logits.py; no GPU, seconds per dataset.

  python correct_bias.py runs/logits/*/logits.npz

Methods compared (all test-time, training-free):
  raw          argmax of the adapted model
  zero-shot    argmax of frozen CLIP
  prior-sub    subtract a scalar from all base-class logits, scalar tuned on the test set (oracle-ish
               upper bound for the simplest possible fix)
  EM prior     Saerens-Latinne-Decock: iteratively re-estimate the label prior and divide it out
  sinkhorn-u   Sinkhorn so that predictions spread uniformly over classes
  sinkhorn-o   Sinkhorn to the true class marginal (oracle: upper bound of marginal correction)
Reported: mixed accuracy, accuracy on base images, on novel images, novel->base confusion.
"""
import argparse, os
import numpy as np


def softmax(x, T=1.0):
    z = x / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def report(pred, y, nb):
    isn = y >= nb
    return dict(mixed=100 * (pred == y).mean(),
                base_img=100 * (pred[~isn] == y[~isn]).mean(),
                novel_img=100 * (pred[isn] == y[isn]).mean(),
                n2b=100 * (pred[isn] < nb).mean())


def em_prior(P, iters=100):
    """Saerens et al.: re-estimate the test-set class prior, then reweight."""
    C = P.shape[1]
    src = np.full(C, 1.0 / C)
    tgt = src.copy()
    for _ in range(iters):
        W = P * (tgt / src)
        W /= W.sum(-1, keepdims=True)
        new = W.mean(0)
        if np.abs(new - tgt).max() < 1e-7:
            break
        tgt = new
    W = P * (tgt / src)
    return W / W.sum(-1, keepdims=True)


def sinkhorn(P, marginal, iters=200, eps=1e-9):
    """Scale rows/columns so columns match `marginal` and rows stay distributions."""
    Q = P.copy()
    N = Q.shape[0]
    for _ in range(iters):
        col = Q.sum(0) / N
        Q = Q * (marginal / np.maximum(col, eps))
        Q = Q / np.maximum(Q.sum(-1, keepdims=True), eps)
    return Q


def best_base_penalty(logits, y, nb, grid=np.linspace(0, 10, 101)):
    best = (-1, 0.0)
    for s in grid:
        z = logits.copy(); z[:, :nb] -= s
        acc = (z.argmax(-1) == y).mean()
        if acc > best[0]:
            best = (acc, s)
    return best[1]


def zs_marginal(zs_logits, T=1.0, smooth=0.0):
    """Estimate the test-set class marginal from the frozen zero-shot head, which carries no
    base-class bias. Optionally smooth toward uniform."""
    Q = softmax(zs_logits, T)
    m = Q.mean(0)
    if smooth > 0:
        m = (1 - smooth) * m + smooth / len(m)
    return m / m.sum()


def partial_sinkhorn(P, marginal, lam=0.5, iters=200):
    """Interpolate in log space between the raw prediction and the marginal-matched one."""
    Q = sinkhorn(P, marginal, iters)
    L = (1 - lam) * np.log(np.maximum(P, 1e-12)) + lam * np.log(np.maximum(Q, 1e-12))
    return softmax(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--T", type=float, default=1.0, help="temperature before softmax")
    a = ap.parse_args()
    rows = []
    for f in a.npz:
        d = np.load(f, allow_pickle=True)
        lg, zs, y, nb = d["logits"], d["zs_logits"], d["labels"], int(d["n_base"])
        name = os.path.basename(os.path.dirname(f))
        C = lg.shape[1]
        P = softmax(lg, a.T)
        true_marg = np.bincount(y, minlength=C) / len(y)
        res = {
            "raw": report(lg.argmax(-1), y, nb),
            "zero-shot": report(zs.argmax(-1), y, nb),
            "prior-sub": report((lg - np.r_[np.full(nb, best_base_penalty(lg, y, nb)),
                                            np.zeros(C - nb)]).argmax(-1), y, nb),
            "EM prior": report(em_prior(P).argmax(-1), y, nb),
            "sinkhorn-u": report(sinkhorn(P, np.full(C, 1.0 / C)).argmax(-1), y, nb),
            "sinkhorn-o": report(sinkhorn(P, true_marg).argmax(-1), y, nb),
            "sinkhorn-zs": report(sinkhorn(P, zs_marginal(zs)).argmax(-1), y, nb),
            "sinkhorn-zs-sm": report(sinkhorn(P, zs_marginal(zs, smooth=0.3)).argmax(-1), y, nb),
            "partial-zs .5": report(partial_sinkhorn(P, zs_marginal(zs), 0.5).argmax(-1), y, nb),
            "partial-u .5": report(partial_sinkhorn(P, np.full(C, 1.0 / C), 0.5).argmax(-1), y, nb),
        }
        print(f"\n=== {name}  ({len(y)} images, {nb} base / {C-nb} novel classes) ===")
        print(f"{'method':12s} {'mixed':>7s} {'base img':>9s} {'novel img':>10s} {'novel->base':>12s}")
        for k, v in res.items():
            print(f"{k:12s} {v['mixed']:7.2f} {v['base_img']:9.2f} {v['novel_img']:10.2f} {v['n2b']:12.2f}")
        rows.append((name, res))
    if len(rows) > 1:
        print("\n=== average over datasets (mixed accuracy) ===")
        for k in rows[0][1]:
            print(f"{k:12s} {np.mean([r[1][k]['mixed'] for r in rows]):7.2f}")


if __name__ == "__main__":
    main()
