"""Explanatory figures for the main claims, from saved logits and images (CPU).

  python figures_extra.py --root ~/data_compact          # writes runs/extra_figs/*.pdf|png

  fig_method     method diagram with small inline plots (no data needed)
  fig_story      one novel image, three protocols: standard (novel classes only), joint, joint + GMT
  fig_asym       novel->base vs base->novel errors for zero-shot CLIP, every learner, and + GMT
  fig_marginal   predicted class marginals (base vs novel) of zero-shot, adapted and GMT, with the
                 gate value, on a balanced dataset, an imbalanced one and the failure case
make_assets.py copies runs/extra_figs/ into paper_assets/figures/.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from correct_bias import softmax, sinkhorn
from adaptive_correct import gini
from analysis_suite import fit_threshold, isotonic_decreasing

C_BASE, C_NOVEL, C_RAW, C_ZS, C_GMT, C_FULL = "#E69F00", "#0072B2", "#7f7f7f", "#56B4E9", "#0072B2", "#D55E00"
LNAME = {"coop": "CoOp", "cocoop": "CoCoOp", "kgcoop": "KgCoOp", "ivlp": "IVLP", "maple": "MaPLe", "promptsrc": "PromptSRC"}
LEARNERS = ["coop", "cocoop", "kgcoop", "ivlp", "maple", "promptsrc"]
DNAME = {"caltech101": "Caltech101", "dtd": "DTD", "eurosat": "EuroSAT", "fgvc_aircraft": "FGVC-Aircraft",
         "food101": "Food101", "oxford_flowers": "Flowers102", "oxford_pets": "OxfordPets", "ucf101": "UCF101"}
OUT = "runs/extra_figs"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def dirs(L, s):
    if L == "maple":
        return ("runs/logits", "runs/calib") if s == 1 else (f"runs/logits_s{s}", f"runs/calib_s{s}")
    return f"runs/{L}/logits_s{s}", f"runs/{L}/calib_s{s}"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"saved {OUT}/{name}.pdf")


def gmt_parts(lg, zs, pts):
    g0, tau = fit_threshold(pts["g"], pts["gain"])
    P = softmax(lg); C = P.shape[1]
    Q = sinkhorn(P, np.full(C, 1.0 / C))
    g = gini(softmax(zs).mean(0))
    lam = float(1.0 / (1.0 + np.exp((g - g0) / tau)))
    B = np.exp((1 - lam) * np.log(np.maximum(P, 1e-12)) + lam * np.log(np.maximum(Q, 1e-12)))
    B /= B.sum(1, keepdims=True)
    return P, Q, B, lam, g, g0, tau


# --------------------------------------------------------------------------- method diagram
def box(ax, x, y, w, h, text, fc="#ffffff", ec="#333333", fs=6.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, color="#333333", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=0.8, color=color, ls=ls))


def fig_method():
    fig = plt.figure(figsize=(6.9, 2.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # --- test time (top band)
    ax.text(0.01, 0.95, "Test time (training-free)", fontsize=7, fontweight="bold", color="#333333")
    box(ax, 0.01, 0.62, 0.10, 0.20, "test images\n(base + novel)", fc="#f2f2f2")
    box(ax, 0.15, 0.71, 0.13, 0.13, "adapted model $f$\n(prompt-tuned)", fc="#fff3d6")
    box(ax, 0.15, 0.50, 0.13, 0.13, "frozen zero-shot\nCLIP", fc="#e3f1fb")
    arrow(ax, (0.11, 0.74), (0.15, 0.775)); arrow(ax, (0.11, 0.70), (0.15, 0.565))
    box(ax, 0.33, 0.71, 0.15, 0.13, "Sinkhorn projection\nuniform class marginal", fc="#ffffff")
    arrow(ax, (0.28, 0.775), (0.33, 0.775)); ax.text(0.305, 0.80, "$P$", fontsize=7, ha="center")
    box(ax, 0.33, 0.50, 0.15, 0.13, "class balance of\nmean prediction: Gini $g$", fc="#ffffff")
    arrow(ax, (0.28, 0.565), (0.33, 0.565)); ax.text(0.305, 0.59, "$Z$", fontsize=7, ha="center")
    box(ax, 0.53, 0.50, 0.13, 0.13, r"gate $\lambda=\sigma\!\left(\frac{g_0-g}{\tau}\right)$", fc="#ffffff")
    arrow(ax, (0.48, 0.565), (0.53, 0.565))
    box(ax, 0.71, 0.60, 0.15, 0.15, r"$(1-\lambda)\log P+\lambda\log Q$", fc="#dbe9f6", bold=False)
    arrow(ax, (0.48, 0.775), (0.71, 0.70)); ax.text(0.60, 0.76, "$Q$", fontsize=7)
    arrow(ax, (0.66, 0.565), (0.71, 0.64)); ax.text(0.685, 0.575, r"$\lambda$", fontsize=7)
    arrow(ax, (0.86, 0.675), (0.90, 0.675)); ax.text(0.905, 0.675, "prediction", fontsize=6.5, va="center")
    # inline plot: class marginal before / after projection
    a1 = fig.add_axes([0.345, 0.865, 0.12, 0.10]); c = np.arange(10)
    before = np.r_[np.linspace(1.7, 1.2, 5), np.linspace(0.8, 0.35, 5)]; before /= before.sum()
    a1.bar(c - 0.2, before, 0.4, color=[C_BASE] * 5 + [C_NOVEL] * 5, alpha=0.5)
    a1.bar(c + 0.2, np.full(10, 0.1), 0.4, color=[C_BASE] * 5 + [C_NOVEL] * 5)
    a1.set_xticks([]); a1.set_yticks([]); a1.set_title("class mass: $P$ (light) vs $Q$", fontsize=5.5, pad=1)
    # inline plot: gate
    a2 = fig.add_axes([0.545, 0.865, 0.10, 0.10]); gg = np.linspace(0, 0.5, 100)
    a2.plot(gg, 1 / (1 + np.exp((gg - 0.25) / 0.03)), color=C_GMT, lw=1)
    a2.axvline(0.25, ls="--", lw=0.6, color="#555555"); a2.set_xticks([]); a2.set_yticks([0, 1])
    a2.set_title(r"$\lambda$ vs $g$: open when balanced", fontsize=5.5, pad=1)
    # --- calibration (bottom band)
    ax.plot([0.01, 0.99], [0.42, 0.42], color="#bbbbbb", lw=0.6, ls=":")
    ax.text(0.01, 0.37, "Calibration (base classes only, no novel class, no test label)", fontsize=7,
            fontweight="bold", color="#333333")
    box(ax, 0.01, 0.06, 0.14, 0.22, "split base classes:\npseudo-base $B_1$,\npseudo-novel $B_2$", fc="#fff3d6", fs=6)
    box(ax, 0.18, 0.06, 0.15, 0.22, "tune $f$ on $B_1$;\nscore held-out\nimages over $B_1\\cup B_2$", fc="#fff3d6", fs=6)
    box(ax, 0.36, 0.06, 0.14, 0.22, "skewed subsets\n(Zipf $\\alpha\\in[0,2]$):\nGini $g$, gain of $Q$", fc="#ffffff", fs=6)
    arrow(ax, (0.15, 0.17), (0.18, 0.17)); arrow(ax, (0.33, 0.17), (0.36, 0.17))
    a3 = fig.add_axes([0.55, 0.05, 0.16, 0.25]); rng = np.random.default_rng(0)
    gx = rng.uniform(0.1, 0.45, 160); gy = 6 - 40 * np.maximum(0, gx - 0.22) + rng.normal(0, 1.6, 160)
    a3.scatter(gx, gy, s=2, color="#999999", lw=0); o = np.argsort(gx)
    a3.plot(gx[o], isotonic_decreasing(gx, gy)[o], color=C_GMT, lw=1); a3.axhline(0, color="k", lw=0.5)
    a3.axvline(0.25, ls="--", lw=0.7, color=C_GMT); a3.text(0.26, a3.get_ylim()[1] * 0.75, "$g_0$", fontsize=6, color=C_GMT)
    a3.set_xticks([]); a3.set_yticks([]); a3.set_xlabel("Gini $g$", fontsize=5.5, labelpad=1)
    a3.set_title("gain vs Gini: monotone fit", fontsize=5.5, pad=1)
    arrow(ax, (0.50, 0.17), (0.55, 0.17))
    arrow(ax, (0.62, 0.31), (0.595, 0.50), color=C_GMT, ls="--"); ax.text(0.63, 0.40, r"$g_0,\tau$", fontsize=7, color=C_GMT)
    box(ax, 0.75, 0.08, 0.23, 0.20, "GMT-E: same gate, applied to\n$(1-\\beta)f+\\beta f_{zs}$,\n$\\beta$ and $g_0,\\tau$ from calibration", fc="#f7f7f7", fs=6)
    save(fig, "fig_method")


# --------------------------------------------------------------------------- one image, three protocols
def fig_story(root, learner="maple", seed=1, picks=("eurosat", "dtd", "oxford_pets")):
    from trustprompt.datasets import load_dataset, base_novel_split, subset
    from PIL import Image
    ld, cd = dirs(learner, seed)
    rows = []
    rng = np.random.default_rng(1)
    for d in picks:
        t = np.load(f"{ld}/{d}/logits.npz", allow_pickle=True)
        lg, zs, y, nb = t["logits"], t["zs_logits"], t["labels"], int(t["n_base"])
        names = [str(n).replace("_", " ") for n in t["classnames"]]
        data = load_dataset(root, d); base, novel = base_novel_split(data)
        items = subset(data["test"], base + novel)
        if len(items) != len(y):
            print(f"! {d}: test order mismatch, skipped"); continue
        P, Q, Bl, lam, *_ = gmt_parts(lg, zs, np.load(f"{cd}/{d}/points.npz"))
        std = softmax(lg[:, nb:])
        idx = np.where((y >= nb) & (lg.argmax(-1) < nb) & (Bl.argmax(-1) == y) & (std.argmax(-1) == y - nb))[0]
        if len(idx) == 0:
            continue
        # prefer clear cases: the base class wins clearly in the joint protocol
        margin = P[idx].max(1) - P[idx, y[idx]]
        i = idx[np.argsort(-margin)[:max(1, len(idx) // 5)]]
        i = int(rng.choice(i))
        rows.append(dict(d=d, path=items[i][0], y=int(y[i]), nb=nb, names=names, lam=lam,
                         std=std[i], joint=P[i], gmt=Bl[i]))
    if not rows:
        print("! no story examples found"); return
    fig, axs = plt.subplots(len(rows), 4, figsize=(6.9, 1.35 * len(rows)),
                            gridspec_kw={"width_ratios": [0.75, 0.8, 0.8, 0.8]})
    axs = np.atleast_2d(axs)
    titles = ["standard protocol\n(novel classes only)", "joint protocol\n(all classes)",
              "joint + GMT"]
    for r, ex in enumerate(rows):
        axs[r, 0].imshow(Image.open(ex["path"]).convert("RGB").resize((224, 224))); axs[r, 0].axis("off")
        axs[r, 0].set_title(f"{DNAME[ex['d']]}: true class\n{ex['names'][ex['y']]}", fontsize=6)
        for c, (key, title) in enumerate(zip(["std", "joint", "gmt"], titles)):
            ax = axs[r, c + 1]; p = ex[key]
            off = ex["nb"] if key == "std" else 0
            top = np.argsort(-p)[:3][::-1]
            labels = [ex["names"][k + off] for k in top]
            cols = [C_NOVEL if (k + off) >= ex["nb"] else C_BASE for k in top]
            bars = ax.barh(range(3), p[top], color=cols, height=0.65)
            for b, k in zip(bars, top):
                if k + off == ex["y"]:
                    b.set_edgecolor("black"); b.set_linewidth(1.2)
            ax.set_yticks(range(3)); ax.set_yticklabels([l[:20] for l in labels], fontsize=5.5)
            ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.tick_params(axis="x", labelsize=5)
            ok = (top[-1] + off) == ex["y"]
            ax.text(0.98, 0.05, "correct" if ok else "wrong: base class", transform=ax.transAxes, ha="right",
                    fontsize=5.5, color="#009E73" if ok else C_FULL, fontweight="bold")
            if r == 0:
                ax.set_title(title + (f"" if key != "gmt" else ""), fontsize=6.5)
            if key == "gmt":
                ax.text(0.98, 0.25, rf"$\lambda$={ex['lam']:.2f}", transform=ax.transAxes, ha="right", fontsize=5.5)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_BASE), plt.Rectangle((0, 0), 1, 1, color=C_NOVEL),
                        plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=1.2)],
               labels=["base class", "novel class", "true class"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.6, -0.04), fontsize=6)
    fig.subplots_adjust(wspace=1.05, hspace=0.6)
    save(fig, "fig_story")


# --------------------------------------------------------------------------- asymmetry
def fig_asym(sa_path="runs/supp_analysis.json"):
    SA = json.load(open(sa_path))["per_run"]
    def mean(L, key, k):
        v = [SA[L][s][d]["asym"][key][k] for s in SA[L] for d in SA[L][s] if "asym" in SA[L][s][d]]
        return float(np.mean(v)) if v else np.nan
    Ls = [L for L in LEARNERS if L in SA]
    zl = next((L for L in Ls if "zs_gmt" in next(iter(next(iter(SA[L].values())).values()))["asym"]), Ls[0])
    rows = [("Zero-shot CLIP", mean(zl, "zs", "n2b"), mean(zl, "zs", "b2n"),
             mean(zl, "zs_gmt", "n2b"), mean(zl, "zs_gmt", "b2n"))]
    rows += [(LNAME[L], mean(L, "raw", "n2b"), mean(L, "raw", "b2n"), mean(L, "gmt", "n2b"), mean(L, "gmt", "b2n"))
             for L in Ls]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    y = np.arange(len(rows))[::-1]
    for yy, (name, n2b, b2n, gn, gb) in zip(y, rows):
        ax.barh(yy + 0.18, -n2b, 0.34, color=C_BASE, alpha=0.9)
        ax.barh(yy + 0.18, b2n, 0.34, color=C_NOVEL, alpha=0.9)
        ax.barh(yy - 0.18, -gn, 0.34, color=C_BASE, alpha=0.4)
        ax.barh(yy - 0.18, gb, 0.34, color=C_NOVEL, alpha=0.4)
        ax.text(-n2b - 0.8, yy + 0.18, f"{n2b:.1f}", va="center", ha="right", fontsize=5)
        ax.text(b2n + 0.8, yy + 0.18, f"{b2n:.1f}", va="center", ha="left", fontsize=5)
        ax.text(-gn - 0.8, yy - 0.18, f"{gn:.1f}", va="center", ha="right", fontsize=4.5, color="#666666")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    lo = max(r[1] for r in rows) + 7; hi = max(max(r[2], r[4]) for r in rows) + 7
    ax.set_xlim(-lo, hi)
    ticks = [t for t in range(-int(lo // 10) * 10, int(hi) + 1, 10)]
    ax.set_xticks(ticks); ax.set_xticklabels([str(abs(t)) for t in ticks])
    ax.set_xlabel("novel $\\to$ base (%)            base $\\to$ novel (%)", fontsize=6)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#777777"), plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.4)],
              labels=["joint protocol", "+ GMT"], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=2, fontsize=5.5)
    save(fig, "fig_asym")


# --------------------------------------------------------------------------- class marginals and the gate
def fig_marginal(learner="maple", seed=1, picks=("eurosat", "caltech101", "fgvc_aircraft")):
    """Top: what the gate sees (sorted class shares of true labels and of zero-shot predictions).
    Bottom: what GMT does (share of predictions that go to base classes)."""
    ld, cd = dirs(learner, seed)
    fig, axs = plt.subplots(2, len(picks), figsize=(6.9, 3.0), gridspec_kw={"height_ratios": [1.15, 1]})
    for c, d in enumerate(picks):
        t = np.load(f"{ld}/{d}/logits.npz", allow_pickle=True)
        lg, zs, y, nb = t["logits"], t["zs_logits"], t["labels"], int(t["n_base"])
        P, Q, Bl, lam, g, g0, tau = gmt_parts(lg, zs, np.load(f"{cd}/{d}/points.npz"))
        C = lg.shape[1]
        share = lambda pred: np.bincount(pred, minlength=C) / len(pred)
        truth, zsh = np.bincount(y, minlength=C) / len(y), softmax(zs).mean(0)
        ax = axs[0, c]; r = np.arange(1, C + 1) / C
        ax.plot(r, np.sort(truth)[::-1] * C, color="k", lw=1.0, label="true labels")
        ax.plot(r, np.sort(zsh)[::-1] * C, color=C_ZS, lw=1.0, ls="--", label="zero-shot predictions (gate input)")
        ax.axhline(1, color="#bbbbbb", lw=0.5, ls=":")
        ax.set_ylim(0, max(3.0, float(np.sort(truth)[::-1][0] * C) * 1.05)); ax.set_xlim(0, 1)
        status = "open" if lam > 0.5 else "closed"
        ax.set_title(f"{DNAME[d]}: gate {status} ($\\lambda$={lam:.2f})\nzero-shot Gini {g:.2f}, threshold $g_0$={g0:.2f}", fontsize=6)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["most", "", "least frequent"], fontsize=5.5)
        if c == 0:
            ax.set_ylabel("share per class\n(1 = uniform)", fontsize=6)
        ax2 = axs[1, c]
        vals = [truth[:nb].sum(), share(lg.argmax(-1))[:nb].sum(), share(Bl.argmax(-1))[:nb].sum()]
        labs = ["true labels", "adapted model", "+ GMT"]
        bars = ax2.bar(range(3), np.array(vals) * 100, color=["#333333", C_RAW, C_GMT], width=0.6)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v*100:.0f}%", ha="center", fontsize=5.5)
        ax2.set_xticks(range(3)); ax2.set_xticklabels(labs, fontsize=5.5); ax2.set_ylim(0, 85)
        if c == 0:
            ax2.set_ylabel("predictions that go\nto base classes (%)", fontsize=6)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.28, hspace=0.75)
    save(fig, "fig_marginal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="dataset root (needed for fig_story)")
    ap.add_argument("--only", default="method,story,asym,marginal")
    a = ap.parse_args()
    todo = a.only.split(",")
    if "method" in todo:
        fig_method()
    if "asym" in todo and os.path.exists("runs/supp_analysis.json"):
        fig_asym()
    if "marginal" in todo:
        try:
            fig_marginal()
        except FileNotFoundError as e:
            print("! marginal:", e)
    if "story" in todo and a.root:
        fig_story(a.root)


if __name__ == "__main__":
    main()
