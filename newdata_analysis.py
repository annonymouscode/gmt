"""GMT on datasets it was never calibrated on (ImageNet, SUN397, StanfordCars): CPU, saved logits.

GMT's threshold g0/tau is fitted on the calibration points of the 8 original datasets pooled
(same learner and seed); GMT-E uses the median ensemble weight and gate of those 8 datasets
(from runs/baselines3_<learner>.json). No image or label of the new datasets is used for tuning.

  python newdata_analysis.py --out runs/newdata_analysis.json
"""
import argparse, glob, json, os, re
import numpy as np
from correct_bias import softmax, sinkhorn, report
from adaptive_correct import gini
from analysis_suite import fit_threshold

LEARNERS = ["coop", "cocoop", "kgcoop", "ivlp", "maple", "promptsrc"]
OLD = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "ucf101"]
NEW = ["imagenet", "sun397", "stanford_cars"]


def dirs(L, s):
    if L == "maple":
        return ("runs/logits", "runs/calib") if s == 1 else (f"runs/logits_s{s}", f"runs/calib_s{s}")
    return f"runs/{L}/logits_s{s}", f"runs/{L}/calib_s{s}"


def pooled_gate(L, s):
    _, cd = dirs(L, s)
    G, GA = [], []
    for d in OLD:
        f = f"{cd}/{d}/points.npz"
        if os.path.exists(f):
            p = np.load(f); G.append(p["g"]); GA.append(p["gain"])
    return fit_threshold(np.concatenate(G), np.concatenate(GA)) if G else None


def pooled_gmte(L, s):
    f = f"runs/baselines3_{L}.json"
    if not os.path.exists(f):
        return None
    for spec, rows in json.load(open(f)).items():
        m = re.search(r"logits_s(\d+)", spec)
        if int(m.group(1) if m else 1) == s:
            gm = [r["gmte"] for d, r in rows.items() if d in OLD and "gmte" in r]
            if gm:
                return {k: float(np.median([g[k] for g in gm])) for k in ("beta", "g0", "tau")}
    return None


def lam_of(zs, g0, tau):
    g = gini(softmax(zs).mean(0))
    return float(1.0 / (1.0 + np.exp((g - g0) / tau))), g


def blend(P, Q, l):
    return ((1 - l) * np.log(np.maximum(P, 1e-12)) + l * np.log(np.maximum(Q, 1e-12))).argmax(-1)


def hm(b, n):
    return 2 * b * n / (b + n) if b + n > 0 else 0.0


def analyse(lg, zs, y, nb, gate, gm):
    C = lg.shape[1]; u = np.full(C, 1.0 / C); isn = y >= nb
    P = softmax(lg); Q = sinkhorn(P, u)
    lam, g = lam_of(zs, *gate)
    preds = {"raw": lg.argmax(-1), "zero-shot": zs.argmax(-1), "full SK": Q.argmax(-1), "gmt": blend(P, Q, lam)}
    out = {"gate": {"gini": g, "lam": lam, "g0": gate[0], "tau": gate[1]}}
    if gm:
        E = (1 - gm["beta"]) * lg + gm["beta"] * zs
        PE = softmax(E); QE = sinkhorn(PE, u); lamE, _ = lam_of(zs, gm["g0"], gm["tau"])
        preds["gmte"] = blend(PE, QE, lamE); out["gate"]["lam_e"] = lamE; out["gate"]["beta"] = gm["beta"]
    out["res"] = {k: report(p, y, nb) for k, p in preds.items()}
    for k, r in out["res"].items():
        r["joint_hm"] = hm(r["base_img"], r["novel_img"])
    for k, L in (("raw", lg), ("zero-shot", zs)):
        b = (L[~isn, :nb].argmax(-1) == y[~isn]).mean() * 100
        n = (L[isn, nb:].argmax(-1) == y[isn] - nb).mean() * 100
        out["res"][k]["std_hm"] = hm(b, n)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/newdata_analysis.json")
    a = ap.parse_args()
    R = {}
    for L in LEARNERS:
        for s in (1, 2, 3):
            gate = pooled_gate(L, s); gm = pooled_gmte(L, s)
            for d in NEW:
                f = f"{dirs(L, s)[0]}/{d}/logits.npz"
                if gate is None or not os.path.exists(f):
                    continue
                t = np.load(f, allow_pickle=True)
                r = analyse(t["logits"], t["zs_logits"], t["labels"], int(t["n_base"]), gate, gm)
                R.setdefault(L, {}).setdefault(str(s), {})[d] = r
                g = r["res"]["gmt"]["mixed"] - r["res"]["raw"]["mixed"]
                ge = r["res"].get("gmte", {}).get("mixed", np.nan) - r["res"]["raw"]["mixed"]
                print(f"{L:9s} s{s} {d:14s} raw {r['res']['raw']['mixed']:6.2f}  GMT {g:+6.2f}  GMT-E {ge:+6.2f}  "
                      f"lambda {r['gate']['lam']:.2f}", flush=True)
    json.dump(R, open(a.out, "w"), indent=1, default=float)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
