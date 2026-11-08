"""Training-free baselines from DeCoOp (DePT router, Eq. 3) and OTTER (exact OT to a uniform marginal),
on saved test logits. Metrics use report() from correct_bias.py.
  python extra_baselines.py --out runs/extra_baselines.json"""
import argparse, json, os, time
import numpy as np
import ot
from correct_bias import softmax, report

LEARNERS = ["coop", "cocoop", "kgcoop", "ivlp", "maple", "promptsrc"]
OLD = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "ucf101"]

def logit_dir(L, s):
    if L == "maple":
        return "runs/logits" if s == 1 else f"runs/logits_s{s}"
    return f"runs/{L}/logits_s{s}"

def dept_router(lg, zs, nb):
    pz = softmax(zs)
    is_base = pz[:, :nb].max(1) >= pz[:, nb:].max(1)
    return np.where(is_base, lg[:, :nb].argmax(1), nb + zs[:, nb:].argmax(1))

def otter(lg):
    P = softmax(lg); N, C = P.shape
    M = -np.log(np.maximum(P, 1e-12))
    T = ot.emd(np.full(N, 1.0 / N), np.full(C, 1.0 / C), M, numItermax=20_000_000)
    return T.argmax(1)

def hm(b, n):
    return 2 * b * n / (b + n) if b + n > 0 else 0.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/extra_baselines.json"); a = ap.parse_args()
    R = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for L in LEARNERS:
        for s in (1, 2, 3):
            for d in OLD:
                if d in R.get(L, {}).get(str(s), {}):
                    continue
                f = f"{logit_dir(L, s)}/{d}/logits.npz"
                if not os.path.exists(f):
                    print(f"missing {f}", flush=True); continue
                t = np.load(f, allow_pickle=True)
                lg, zs, y, nb = t["logits"], t["zs_logits"], t["labels"], int(t["n_base"])
                t0 = time.time()
                preds = {"raw": lg.argmax(1), "dept": dept_router(lg, zs, nb), "otter": otter(lg)}
                res = {k: report(p, y, nb) for k, p in preds.items()}
                for r in res.values():
                    r["joint_hm"] = hm(r["base_img"], r["novel_img"])
                R.setdefault(L, {}).setdefault(str(s), {})[d] = res
                print(f"{L:9s} s{s} {d:15s} raw {res['raw']['mixed']:6.2f}  DePT {res['dept']['mixed'] - res['raw']['mixed']:+6.2f}  "
                      f"OTTER {res['otter']['mixed'] - res['raw']['mixed']:+6.2f}  ({time.time() - t0:.0f}s)", flush=True)
                json.dump(R, open(a.out, "w"), indent=1, default=float)
    print("\nlearner     method   mean delta (std over seeds)   worst pair   delta joint HM")
    for L in LEARNERS:
        for m in ("dept", "otter"):
            per_seed, worst, dhm = [], [], []
            for s, rows in R.get(L, {}).items():
                dl = [r[m]["mixed"] - r["raw"]["mixed"] for r in rows.values()]
                per_seed.append(np.mean(dl)); worst += dl
                dhm.append(np.mean([r[m]["joint_hm"] - r["raw"]["joint_hm"] for r in rows.values()]))
            if per_seed:
                print(f"{L:11s} {m:7s}  {np.mean(per_seed):+6.2f} ({np.std(per_seed):.2f}){'':16s}{min(worst):+7.2f}    {np.mean(dhm):+6.2f}")
    print(f"saved -> {a.out}")

if __name__ == "__main__":
    main()
