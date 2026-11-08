"""Build every table, figure and in-text number of the paper from the saved result files.
Nothing in the paper is typed by hand: re-run this script whenever a result file changes.

  python make_assets.py                    # reads runs/, writes paper_assets/

Inputs (missing files are skipped with a warning):
  runs/analysis_all.json, runs/analysis_<learner>.json   analysis_suite.py
  runs/baselines3_<learner>.json (or baselines2_)        baselines_cs.py
  runs/ablate_gate.json (MaPLe), runs/ablate_<learner>.json   ablate_gate.py
  calibration points.npz under each learner's calib dirs
Outputs in paper_assets/:
  tables/*.tex     booktabs LaTeX tables
  figures/*.pdf    vector figures
  numbers.tex      \\newcommand macros for every number quoted in the text
  summary.md       the same tables in markdown, for checking
"""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis_suite import fit_threshold, isotonic_decreasing

OUT = "paper_assets"
LEARNERS = ["coop", "cocoop", "kgcoop", "ivlp", "maple", "promptsrc"]
LNAME = {"coop": "CoOp", "cocoop": "CoCoOp", "kgcoop": "KgCoOp", "ivlp": "IVLP", "maple": "MaPLe",
         "promptsrc": "PromptSRC"}
LTAG = {"coop": "Coop", "cocoop": "Cocoop", "kgcoop": "Kgcoop", "ivlp": "Ivlp", "maple": "Maple",
        "promptsrc": "Promptsrc"}
DATASETS = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers",
            "oxford_pets", "ucf101"]
DNAME = {"caltech101": "Caltech101", "dtd": "DTD", "eurosat": "EuroSAT", "fgvc_aircraft": "Aircraft",
         "food101": "Food101", "oxford_flowers": "Flowers102", "oxford_pets": "Pets", "ucf101": "UCF101"}
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
C_RAW, C_FULL, C_GMT, C_STREAM, C_GMTE = "#7f7f7f", "#E69F00", "#0072B2", "#009E73", "#CC79A7"
LCOL = {l: OKABE[i] for i, l in enumerate(LEARNERS)}
WARN = []
SEEDS = ("1", "2", "3")
ALLOW_PARTIAL = "--allow-partial" in sys.argv

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})


# ------------------------------------------------------------------ loading
def seed_of(spec):
    m = re.search(r"logits_s(\d+)", spec)
    return m.group(1) if m else "1"


def load_json(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def load_all():
    A = {}
    files = sorted(set(glob.glob("runs/analysis_*.json")), key=os.path.getmtime)   # newest wins
    for f in files:
        d = load_json(f)
        if d:
            for L, v in d.items():
                if v:
                    A[L] = v
    B, AB = {}, {}
    for L in LEARNERS:
        d = load_json(f"runs/baselines3_{L}.json") or load_json(f"runs/baselines2_{L}.json")
        if d:
            B[L] = {seed_of(k): v for k, v in d.items()}
        X = load_json("runs/extra_baselines.json") or {}      # extra_baselines.py: DePT router, OTTER
        n_merged = 0
        for s_, rows_ in X.get(L, {}).items():
            for d_, r_ in rows_.items():
                tgt = B.get(L, {}).get(s_, {}).get(d_)
                if tgt is not None:
                    tgt["res"]["dept"] = r_["dept"]; tgt["res"]["otter"] = r_["otter"]; n_merged += 1
        if X.get(L) and n_merged == 0:
            WARN.append(f"{LNAME[L]}: extra_baselines.json present but no dataset-seed matched")
        d = load_json("runs/ablate_gate.json" if L == "maple" else f"runs/ablate_{L}.json")
        if d:
            AB[L] = {seed_of(k): v for k, v in d.items()}
    for name, src in [("analysis", A), ("baselines", B), ("ablation", AB)]:
        for L in LEARNERS:
            if L not in src:
                WARN.append(f"{LNAME[L]}: no {name} results")
            elif not ALLOW_PARTIAL and not complete(src[L]):
                have = {s: len(v) for s, v in src[L].items()}
                WARN.append(f"{LNAME[L]}: {name} incomplete (datasets per seed: {have}); left out")
                del src[L]
    return A, B, AB


def complete(src_L):
    return all(s in src_L and all(d in src_L[s] for d in DATASETS) for s in SEEDS)


def common_ds(src_L):
    sets = [set(v) for v in src_L.values()]
    ds = sorted(set.intersection(*sets)) if sets else []
    return [d for d in DATASETS if d in ds]


def per_seed(src_L, f):
    """List over seeds of the dataset-average of f(entry)."""
    ds = common_ds(src_L)
    out = []
    for s in sorted(src_L):
        try:
            out.append(float(np.mean([f(src_L[s][d]) for d in ds])))
        except (KeyError, TypeError):
            return []
    return out


def per_dataset(src_L, f):
    """{dataset: seed-mean of f}."""
    return {d: float(np.mean([f(src_L[s][d]) for s in src_L])) for d in common_ds(src_L)}


def worst(src_L, f):
    ds = common_ds(src_L)
    try:
        return float(min(f(src_L[s][d]) for s in src_L for d in ds))
    except (KeyError, ValueError):
        return None


def ms(v):
    v = np.asarray(v, float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (None, None)


def pm(v, signed=False, bold=False):
    m, s = ms(v)
    if m is None:
        return "--"
    num = f"{m:+.2f}" if signed else f"{m:.2f}"
    body = f"\\textbf{{{num}}}" if bold else num
    return f"{body}\\,{{\\scriptsize$\\pm${s:.2f}}}"


def sgn(x):
    return "--" if x is None else f"{x:+.2f}"


# ------------------------------------------------------------------ table writer
TABLES_MD = []


def write_table(name, caption, header, rows, colspec, label, wide=False, groups=None):
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\linewidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    if groups:
        lines.append(" & ".join(groups[0]) + " \\\\")
        lines += groups[1:]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", f"\\end{{{env}}}", ""]
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    open(f"{OUT}/tables/{name}.tex", "w").write("\n".join(lines))
    def clean(c):
        c = c.replace("\\,{\\scriptsize$\\pm$", " ±")
        c = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", c)
        for a, b in [("}", ""), ("$", ""), ("\\to", "→"), ("\\lambda", "λ"), ("\\tau", "τ"),
                     ("\\Delta", "Δ"), ("\\ ", " "), ("\\", "")]:
            c = c.replace(a, b)
        return c
    cap = re.sub(r"[\\$]", "", caption)
    mdh = list(header)
    if groups:
        names = re.findall(r"\\multicolumn\{(\d+)\}\{c\}\{([^}]*)\}", " ".join(groups[0]))
        exp = [n for k, n in names for _ in range(int(k))]
        mdh = [header[0]] + [f"{n} {h}" for n, h in zip(exp, header[1:])]
    md = [f"### {name}: {cap}", "", "| " + " | ".join(clean(h) for h in mdh) + " |",
          "|" + "---|" * len(header)]
    for r in rows:
        if r != "MID":
            md.append("| " + " | ".join(clean(c) for c in r) + " |")
    TABLES_MD.append("\n".join(md) + "\n")


# ------------------------------------------------------------------ numbers
MACROS = []


def macro(name, value, fmt="{:.2f}"):
    if value is None:
        return
    v = fmt.format(value) if not isinstance(value, str) else value
    head = f"\\newcommand{{\\{name}}}{{"
    MACROS[:] = [m for m in MACROS if not m.startswith(head)]     # one definition per name (last wins)
    MACROS.append(f"{head}{v}}}")


def derived_macros():
    """Macros computed from other macros, so the text never types a count by hand."""
    vals = {m.split("}{", 1)[0][len("\\newcommand{\\"):]: m.split("}{", 1)[1][:-1] for m in MACROS}
    wins = [v for k, v in vals.items() if k.startswith("WinGmt") and "/" in v and not k.startswith("WinGmte")]
    if wins:
        num = [int(w.split("/")[0]) for w in wins]
        macro("MinWinGmt", str(min(num))); macro("NumDatasets", wins[0].split("/")[1])
        macro("NumLearnersMinWin", str(sum(n == min(num) for n in num)))
    pg = [v for k, v in vals.items() if k.startswith("PGmteDept")]
    if pg:
        num = [float(v.replace("<", "")) for v in pg]
        macro("MaxPGmteDept", f"{max(num):.3f}")


# ------------------------------------------------------------------ tables
def hm2(b, n):
    return 2 * b * n / (b + n) if b + n > 0 else 0.0


def table_main(A, B, SA=None):
    header = ["Learner", "Std.", "Joint", "+GMT", "+GMT-E", "Joint", "+GMT $\\Delta$", "+GMT-E $\\Delta$",
              "Joint", "+GMT", "+GMT-E"]
    groups = [["", "\\multicolumn{4}{c}{Base/novel HM}", "\\multicolumn{3}{c}{Accuracy, all test images}",
               "\\multicolumn{3}{c}{N$\\to$B (\\%)}"],
              "\\cmidrule(lr){2-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}"]
    rows, avg = [], {"gain": [], "gaine": [], "gap": []}
    for L in LEARNERS:
        if L not in A:
            continue
        a = A[L]
        hm = per_seed(a, lambda x: x["b2n"]["hm"])
        jhm = per_seed(a, lambda x: hm2(x["res"]["raw"]["base_img"], x["res"]["raw"]["novel_img"]))
        raw = per_seed(a, lambda x: x["res"]["raw"]["mixed"])
        gain = per_seed(a, lambda x: x["res"]["gmt"]["mixed"] - x["res"]["raw"]["mixed"])
        n2b_r = per_seed(a, lambda x: x["res"]["raw"]["n2b"])
        gaine, jhme, n2be = [], [], []
        if L in B:
            gaine = per_seed(B[L], lambda x: x["res"]["GMT-E"]["mixed"] - x["res"]["raw"]["mixed"])
            jhme = per_seed(B[L], lambda x: hm2(x["res"]["GMT-E"]["base_img"], x["res"]["GMT-E"]["novel_img"]))
            n2be = per_seed(B[L], lambda x: x["res"]["GMT-E"]["n2b"])
        jhmg = per_seed(a, lambda x: hm2(x["res"]["gmt"]["base_img"], x["res"]["gmt"]["novel_img"]))
        avg.setdefault("dhm", []).append(ms(jhmg)[0] - ms(jhm)[0])
        macro(f"HMGainGmt{LTAG[L]}", ms(jhmg)[0] - ms(jhm)[0], "{:+.2f}")
        if jhme:
            avg.setdefault("dhme", []).append(ms(jhme)[0] - ms(jhm)[0])
            macro(f"HMGainGmte{LTAG[L]}", ms(jhme)[0] - ms(jhm)[0], "{:+.2f}")
        n2b_g = per_seed(a, lambda x: x["res"]["gmt"]["n2b"])
        rows.append([LNAME[L], pm(hm), pm(jhm), pm(jhmg), pm(jhme), pm(raw), pm(gain, True), pm(gaine, True),
                     pm(n2b_r), pm(n2b_g), pm(n2be)])
        macro(f"JointHMGmt{LTAG[L]}", ms(jhmg)[0])
        avg["gain"].append(ms(gain)[0]); avg["gap"].append(ms(hm)[0] - ms(jhm)[0])
        if gaine:
            avg["gaine"].append(ms(gaine)[0])
        t = LTAG[L]
        macro(f"HM{t}", ms(hm)[0]); macro(f"JointHM{t}", ms(jhm)[0]); macro(f"HiddenGap{t}", ms(hm)[0] - ms(jhm)[0])
        macro(f"JointRaw{t}", ms(raw)[0])
        macro(f"JointGmt{t}", ms(per_seed(a, lambda x: x["res"]["gmt"]["mixed"]))[0])
        macro(f"GainGmt{t}", ms(gain)[0], "{:+.2f}"); macro(f"SdGmt{t}", ms(gain)[1])
        macro(f"WorstGmt{t}", worst(a, lambda x: x["res"]["gmt"]["mixed"] - x["res"]["raw"]["mixed"]), "{:+.2f}")
        macro(f"NtoBRaw{t}", ms(n2b_r)[0]); macro(f"NtoBGmt{t}", ms(per_seed(a, lambda x: x["res"]["gmt"]["n2b"]))[0])
        for k, tag in [("base_img", "BaseImg"), ("novel_img", "NovelImg")]:
            macro(f"{tag}Raw{t}", ms(per_seed(a, lambda x, k=k: x["res"]["raw"][k]))[0])
            macro(f"{tag}Gmt{t}", ms(per_seed(a, lambda x, k=k: x["res"]["gmt"][k]))[0])
        if gaine:
            macro(f"GainGmte{t}", ms(gaine)[0], "{:+.2f}"); macro(f"SdGmte{t}", ms(gaine)[1])
            macro(f"WorstGmte{t}", worst(B[L], lambda x: x["res"]["GMT-E"]["mixed"] - x["res"]["raw"]["mixed"]), "{:+.2f}")
            macro(f"JointHMGmte{t}", ms(jhme)[0]); macro(f"NtoBGmte{t}", ms(n2be)[0])
    pr = {L: v for L, v in ((SA or {}).get("per_run") or {}).items() if v and (ALLOW_PARTIAL or complete(v))}
    has = lambda L, k: L in pr and k in pr[L].get("1", {}).get(DATASETS[0], {}).get("asym", {})
    zl = next((L for L in LEARNERS if has(L, "zs_gmt")), None) or next((L for L in LEARNERS if has(L, "zs")), None)
    if zl:
        z = lambda key, k: per_seed(pr[zl], lambda x: x["asym"][key][k])
        zrow = ["Zero-shot CLIP", pm(z("zs", "std_hm")), pm(z("zs", "joint_hm")), "--", "--", pm(z("zs", "mixed")),
                "--", "--", pm(z("zs", "n2b")), "--", "--"]
        if "zs_gmt" in pr[zl]["1"][DATASETS[0]]["asym"]:
            zrow[3] = pm(z("zs_gmt", "joint_hm"))
            zrow[6] = pm([g - r for g, r in zip(z("zs_gmt", "mixed"), z("zs", "mixed"))], True)
            zrow[9] = pm(z("zs_gmt", "n2b"))
        rows = [zrow, "MID"] + rows
    if rows:
        rows.append("MID")
        rows.append(["Average", "", "", "", "", "", f"{np.mean(avg['gain']):+.2f}",
                     f"{np.mean(avg['gaine']):+.2f}" if avg["gaine"] else "--", "", "", ""])
        macro("GainGmtAll", float(np.mean(avg["gain"])), "{:+.2f}")
        macro("HMGainGmtAll", float(np.mean(avg["dhm"])), "{:+.2f}")
        if avg.get("dhme"):
            macro("HMGainGmteAll", float(np.mean(avg["dhme"])), "{:+.2f}")
        macro("HiddenGapAll", float(np.mean(avg["gap"])))
        if avg["gaine"]:
            macro("GainGmteAll", float(np.mean(avg["gaine"])), "{:+.2f}")
        macro("NumLearners", str(len(avg["gain"])))
        ws = [worst(A[L], lambda x: x["res"]["gmt"]["mixed"] - x["res"]["raw"]["mixed"]) for L in LEARNERS if L in A]
        macro("WorstGmtAll", min(ws), "{:+.2f}"); macro("WorstGmtAllAbs", abs(min(ws)))
        macro("NumPairs", str(sum(len(common_ds(A[L])) * len(A[L]) for L in LEARNERS if L in A)))
        macro("MinGainGmt", min(avg["gain"]), "{:+.2f}"); macro("MaxGainGmt", max(avg["gain"]), "{:+.2f}")
        macro("MinHiddenGap", min(avg["gap"])); macro("MaxHiddenGap", max(avg["gap"]))
    n = len([L for L in LEARNERS if L in A])
    write_table("tab1_main",
                f"Results for {n} prompt learners (mean $\\pm$ std over 3 seeds, 8 datasets). "
                "Std.: standard base-to-novel protocol, where base and novel images are scored against separate label sets. "
                "Joint: every image is scored against all base and novel classes, as in deployment; HM is the harmonic mean of base-image and novel-image accuracy in each protocol. "
                "Zero-shot CLIP is not tuned on base classes and is the reference for the cost of the joint label set; its GMT column uses a gate calibrated for it. "
                "N$\\to$B: share of novel images predicted as a base class.",
                header, rows, "l" + "c" * 10, "tab:main", wide=True, groups=groups)


BASE_ROWS = [("raw", "Adapted model (raw)"), ("zero-shot", "Zero-shot CLIP"),
             ("EM prior", "EM prior re-estimation"), ("full SK", "Sinkhorn, always on"),
             ("otter", "OTTER, uniform target~\\cite{shin2024otter}"),
             ("dept", "DePT router~\\cite{zhou2024decoop}"),
             ("CS-val", "Calibrated stacking (val)"), ("CS-sweep", "Calibrated stacking (sweep)"),
             ("ZS-val", "Zero-shot ensemble (val)"), ("ZS-sweep", "Zero-shot ensemble (sweep)"),
             ("ours", "GMT (ours)"), ("GMT-E", "GMT-E (ours)")]
ORACLE_ROWS = [("CS-oracle", "Calibrated stacking, test-tuned"), ("ZS-oracle", "Zero-shot ensemble, test-tuned")]


def table_baselines(B):
    Ls = [L for L in LEARNERS if L in B]
    if not Ls:
        return
    header = ["Method"] + sum([["$\\Delta$", "worst"] for _ in Ls], [])
    groups = [[""] + [f"\\multicolumn{{2}}{{c}}{{{LNAME[L]}}}" for L in Ls],
              " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(Ls)))]
    gains = {L: {k: per_seed(B[L], lambda x, k=k: x["res"][k]["mixed"] - x["res"]["raw"]["mixed"])
                 for k, _ in BASE_ROWS + ORACLE_ROWS if k in next(iter(B[L].values()))[common_ds(B[L])[0]]["res"]}
             for L in Ls}
    best = {L: max((ms(v)[0], k) for k, v in gains[L].items() if k not in dict(ORACLE_ROWS) and v)[1] for L in Ls}
    rows = []
    for k, name in BASE_ROWS + [("MID", "")] + ORACLE_ROWS:
        if k == "MID":
            rows.append("MID"); continue
        r = [name]
        for L in Ls:
            g = gains[L].get(k, [])
            r += [pm(g, True, bold=(k == best[L])),
                  sgn(worst(B[L], lambda x, k=k: x["res"][k]["mixed"] - x["res"]["raw"]["mixed"])) if g else "--"]
        rows.append(r)
    for k, tag in [("full SK", "Full"), ("CS-val", "Cs"), ("ZS-sweep", "Zs"), ("EM prior", "Em"), ("dept", "Dept"), ("otter", "Otter")]:
        f = lambda x, k=k: x["res"][k]["mixed"] - x["res"]["raw"]["mixed"]
        w = [worst(B[L], f) for L in Ls if gains[L].get(k)]
        if w:
            macro(f"Worst{tag}All", min(w), "{:+.2f}"); macro(f"Worst{tag}AllAbs", abs(min(w)))
            macro(f"Gain{tag}All", float(np.mean([ms(gains[L][k])[0] for L in Ls if gains[L].get(k)])), "{:+.2f}")
    for L in Ls:
        t = LTAG[L]
        for k, tag in [("ZS-sweep", "Zs"), ("full SK", "Full"), ("CS-val", "Cs"), ("EM prior", "Em"), ("dept", "Dept"), ("otter", "Otter")]:
            if gains[L].get(k):
                macro(f"Gain{tag}{t}", ms(gains[L][k])[0], "{:+.2f}")
                macro(f"Worst{tag}{t}", worst(B[L], lambda x, k=k: x["res"][k]["mixed"] - x["res"]["raw"]["mixed"]), "{:+.2f}")
    if all(gains[L].get("dept") for L in Ls):
        mg = lambda L, k: ms(gains[L][k])[0]
        macro("NumGmtAboveDept", str(sum(mg(L, "ours") > mg(L, "dept") for L in Ls)))
        macro("NumGmteAboveDept", str(sum(mg(L, "GMT-E") > mg(L, "dept") for L in Ls)))
        macro("NumDeptHurts", str(sum(mg(L, "dept") < 0 for L in Ls)))
    write_table("tab2_baselines",
                "Label-free corrections applied to each prompt learner: gain in joint accuracy over the raw model (mean $\\pm$ std over 3 seeds) and the worst change on any dataset-seed. Tuned baselines use the same calibration data as GMT. OTTER uses a uniform target, since no label distribution is available without labels. DePT router: the training-free routing of DeCoOp, in which the zero-shot MSP decides per image between the adapted model over base classes and zero-shot CLIP over novel classes. Bold: best label-free method per learner. Bottom: upper bounds tuned on test labels.",
                header, rows, "l" + "rr" * len(Ls), "tab:baselines", wide=True, groups=groups)


ABL_ROWS = [("full SK", "$\\lambda=1$ (always on)"), ("fix-glob", "Fixed $\\lambda$, one for all datasets"),
            ("fix-ds", "Fixed $\\lambda$ per dataset"), ("hard", "Hard threshold"),
            ("pooled", "Pooled $g_0,\\tau$"), ("m-gini", "Class-count-matched gini"),
            ("ours", "GMT (ours)"), ("tau/2", "GMT, $\\tau/2$"), ("tau*2", "GMT, $2\\tau$")]
AN_ROWS = [("gmt_lodo", "Leave-one-dataset-out $g_0,\\tau$"), ("gmt_zs_target", "Zero-shot-estimated target"),
           ("gmt_true_target", "True test marginal (oracle)")]


def table_ablation(A, AB):
    Ls = [L for L in LEARNERS if L in AB or L in A]
    if not Ls:
        return
    header = ["How $\\lambda$ / target is chosen"] + sum([["$\\Delta$", "worst"] for _ in Ls], [])
    groups = [[""] + [f"\\multicolumn{{2}}{{c}}{{{LNAME[L]}}}" for L in Ls],
              " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(Ls)))]
    rows = []
    for src, rowlist, key in [(AB, ABL_ROWS + [("orc-lam", "Oracle $\\lambda$ (test labels)")], None), (A, AN_ROWS, None)]:
        for k, name in rowlist:
            r = [name]
            for L in Ls:
                if L not in src:
                    r += ["--", "--"]; continue
                f = lambda x, k=k: x["res"][k]["mixed"] - x["res"]["raw"]["mixed"]
                g = per_seed(src[L], f)
                r += [pm(g, True, bold=(k == "ours")), sgn(worst(src[L], f)) if g else "--"]
            rows.append(r)
        rows.append("MID")
    rows = rows[:-1]
    for L in Ls:
        if L in A:
            g = per_seed(A[L], lambda x: x["res"]["gmt_lodo"]["mixed"] - x["res"]["raw"]["mixed"])
            macro(f"GainLodo{LTAG[L]}", ms(g)[0], "{:+.2f}")
        if L in AB:
            for k, tag in [("fix-ds", "FixDs"), ("fix-glob", "FixGlob"), ("orc-lam", "OrcLam")]:
                g = per_seed(AB[L], lambda x, k=k: x["res"][k]["mixed"] - x["res"]["raw"]["mixed"])
                macro(f"Gain{tag}{LTAG[L]}", ms(g)[0], "{:+.2f}")
    write_table("supp_ablation_full",
                "Ablations: how the correction strength $\\lambda$ and the target marginal are chosen. Gain in joint accuracy over the raw model, mean $\\pm$ std over 3 seeds, and the worst dataset-seed change. Only the oracle rows use test labels.",
                header, rows, "l" + "rr" * len(Ls), "tab:ablationfull", wide=True, groups=groups)


def table_per_dataset(A, B):
    Ls = [L for L in LEARNERS if L in A]
    if not Ls:
        return
    header = ["Dataset"] + sum([["GMT", "GMT-E"] for _ in Ls], [])
    groups = [[""] + [f"\\multicolumn{{2}}{{c}}{{{LNAME[L]}}}" for L in Ls],
              " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(Ls)))]
    pg = {L: per_dataset(A[L], lambda x: x["res"]["gmt"]["mixed"] - x["res"]["raw"]["mixed"]) for L in Ls}
    pe = {L: per_dataset(B[L], lambda x: x["res"]["GMT-E"]["mixed"] - x["res"]["raw"]["mixed"]) if L in B else {} for L in Ls}
    rows = [[DNAME[d]] + sum([[sgn(pg[L].get(d)), sgn(pe[L].get(d))] for L in Ls], []) for d in DATASETS
            if any(d in pg[L] for L in Ls)]
    win = [sum(v > 0 for v in pg[L].values()) for L in Ls]
    rows += ["MID", ["Datasets improved (GMT)"] + sum([[f"{w}/{len(pg[L])}", ""] for w, L in zip(win, Ls)], [])]
    for L in Ls:
        v = np.array(list(pg[L].values()))
        if len(v):
            macro(f"MedianGmt{LTAG[L]}", float(np.median(v)), "{:+.2f}")
            macro(f"WinGmt{LTAG[L]}", f"{int((v > 0).sum())}/{len(v)}")
            noe = [x for d, x in pg[L].items() if d != "eurosat"]
            macro(f"GainNoEurosat{LTAG[L]}", float(np.mean(noe)) if noe else None, "{:+.2f}")
    write_table("supp_per_dataset", "Per-dataset gain in joint accuracy over the raw model (mean over 3 seeds).",
                header, rows, "l" + "rr" * len(Ls), "tab:perdataset", wide=True, groups=groups)


def calib_dir(L, s):
    if L == "maple":
        return "runs/calib" if s == "1" else f"runs/calib_s{s}"
    return f"runs/{L}/calib_s{s}"


def table_split_g0():
    Ls, rows_data = [], {}
    for L in LEARNERS:
        res = {}
        for s in ("1", "2", "3"):
            for d in DATASETS:
                f = f"{calib_dir(L, s)}/{d}/points.npz"
                if not os.path.exists(f):
                    continue
                p = np.load(f)
                g0 = fit_threshold(p["g"], p["gain"])[0]
                sds = []
                if "split" in p.files:
                    g0s = [fit_threshold(p["g"][p["split"] == k], p["gain"][p["split"] == k])[0]
                           for k in sorted(set(p["split"].tolist()))]
                    sds = [np.std(g0s)]
                res.setdefault(d, {"g0": [], "sd": []})
                res[d]["g0"].append(g0); res[d]["sd"] += sds
        n = sum(len(v["g0"]) for v in res.values())
        if res and (ALLOW_PARTIAL or n == len(SEEDS) * len(DATASETS)):
            Ls.append(L); rows_data[L] = res
        elif res:
            WARN.append(f"{LNAME[L]}: calibration points incomplete ({n}/{len(SEEDS) * len(DATASETS)}); left out of split table")
    if not Ls:
        return
    header = ["Dataset"] + sum([["$g_0$", "split sd"] for _ in Ls], [])
    groups = [[""] + [f"\\multicolumn{{2}}{{c}}{{{LNAME[L]}}}" for L in Ls],
              " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(Ls)))]
    rows = []
    for d in DATASETS:
        r = [DNAME[d]]
        for L in Ls:
            x = rows_data[L].get(d)
            r += [f"{np.mean(x['g0']):.3f}" if x else "--", f"{np.mean(x['sd']):.3f}" if x and x["sd"] else "--"]
        rows.append(r)
    allsd = [v for L in Ls for d in rows_data[L] for v in rows_data[L][d]["sd"]]
    if allsd:
        macro("SplitSdBelow", f"{sum(v < 0.025 for v in allsd)}/{len(allsd)}"); macro("SplitSdThresh", "0.025")
    write_table("supp_split_g0", "Gate threshold $g_0$ fitted on the calibration points (mean over seeds) and its standard deviation across the three pseudo-splits (mean over seeds). Where the correction never helped on calibration (Caltech101), $g_0$ is the smallest calibration gini, which is the same for every learner because the zero-shot head is shared.",
                header, rows, "l" + "rr" * len(Ls), "tab:splitg0", wide=True, groups=groups)


SIG = [("ours", "raw", "GMT vs raw"), ("GMT-E", "raw", "GMT-E vs raw"),
       ("GMT-E", "ZS-sweep", "GMT-E vs zero-shot ensemble"), ("GMT-E", "ours", "GMT-E vs GMT"),
       ("ours", "CS-sweep", "GMT vs calibrated stacking"), ("ours", "dept", "GMT vs DePT router"),
       ("GMT-E", "dept", "GMT-E vs DePT router")]


def table_significance(B):
    """Paired tests over all dataset-seed pairs: wins, mean difference, Wilcoxon signed-rank p."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        WARN.append("scipy not available: significance table skipped"); return
    Ls = [L for L in LEARNERS if L in B]
    if not Ls:
        return
    header = ["Comparison"] + sum([["wins", "$\\Delta$", "$p$"] for _ in Ls], [])
    groups = [[""] + [f"\\multicolumn{{3}}{{c}}{{{LNAME[L]}}}" for L in Ls],
              " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(Ls)))]
    rows = []
    for a, b, name in SIG:
        r = [name]
        for L in Ls:
            ds = common_ds(B[L])
            try:
                d = np.array([B[L][s][x]["res"][a]["mixed"] - B[L][s][x]["res"][b]["mixed"]
                              for s in B[L] for x in ds])
            except KeyError:
                r += ["--", "--", "--"]; continue
            nz = d[np.abs(d) > 1e-9]
            p = wilcoxon(nz).pvalue if len(nz) >= 6 else float("nan")
            r += [f"{int((d > 1e-9).sum())}/{len(d)}", f"{d.mean():+.2f}",
                  "$<$0.001" if p < 1e-3 else f"{p:.3f}"]
            tag = {"GMT vs raw": "GmtRaw", "GMT-E vs raw": "GmteRaw", "GMT-E vs zero-shot ensemble": "GmteZs",
                   "GMT-E vs GMT": "GmteGmt", "GMT vs calibrated stacking": "GmtCs",
                   "GMT vs DePT router": "GmtDept", "GMT-E vs DePT router": "GmteDept"}[name]
            macro(f"Wins{tag}{LTAG[L]}", f"{int((d > 1e-9).sum())}/{len(d)}")
            macro(f"P{tag}{LTAG[L]}", "<0.001" if p < 1e-3 else f"{p:.3f}")
        rows.append(r)
    write_table("supp_significance",
                "Paired comparison over all dataset-seed pairs (8 datasets $\\times$ 3 seeds): number of pairs where the first method is better, mean difference in joint accuracy, and two-sided Wilcoxon signed-rank $p$-value.",
                header, rows, "l" + "rrr" * len(Ls), "tab:significance", wide=True, groups=groups)


# ------------------------------------------------------------------ figures
def savefig(fig, name):
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    fig.savefig(f"{OUT}/figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/figures/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_n2b(A, B):
    Ls = [L for L in LEARNERS if L in A]
    if not Ls:
        return
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.0), gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axs[0]
    hm = [ms(per_seed(A[L], lambda x: x["b2n"]["hm"]))[0] for L in Ls]
    raw = [ms(per_seed(A[L], lambda x: hm2(x["res"]["raw"]["base_img"], x["res"]["raw"]["novel_img"])))[0] for L in Ls]
    ge = [ms(per_seed(B[L], lambda x: hm2(x["res"]["GMT-E"]["base_img"], x["res"]["GMT-E"]["novel_img"])))[0]
          if L in B else np.nan for L in Ls]
    gm = [ms(per_seed(A[L], lambda x: hm2(x["res"]["gmt"]["base_img"], x["res"]["gmt"]["novel_img"])))[0] for L in Ls]
    x = np.arange(len(Ls)); w = 0.2
    ax.bar(x - 1.5 * w, hm, w, color="#c7d7ea", label="standard protocol")
    ax.bar(x - 0.5 * w, raw, w, color=C_RAW, label="joint label set")
    ax.bar(x + 0.5 * w, gm, w, color=C_GMT, label="joint, + GMT")
    ax.bar(x + 1.5 * w, ge, w, color=C_GMTE, label="joint, + GMT-E")
    ax.set_xticks(x); ax.set_xticklabels([LNAME[L] for L in Ls], rotation=20)
    lo = np.nanmin(raw + ge + gm) - 8; top = np.nanmax(hm + ge + gm)
    ax.set_ylim(max(0, lo), top + 0.45 * (top - max(0, lo)))       # head-room for the legend
    ax.set_ylabel("base/novel HM (%)"); ax.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2)
    ax.set_title("(a) the standard protocol hides a drop", fontsize=7)
    ax = axs[1]
    rawd = {L: per_dataset(A[L], lambda x: x["res"]["raw"]["n2b"]) for L in Ls}
    ged = {L: per_dataset(B[L], lambda x: x["res"]["GMT-E"]["n2b"]) for L in Ls if L in B}
    ds = [d for d in DATASETS if all(d in rawd[L] for L in Ls)]
    r = np.array([[rawd[L][d] for L in Ls] for d in ds])
    gmd = {L: per_dataset(A[L], lambda x: x["res"]["gmt"]["n2b"]) for L in Ls}
    gg = np.array([[gmd[L][d] for L in Ls] for d in ds])
    x = np.arange(len(ds)); w = 0.27
    kw = dict(capsize=1.2, error_kw={"lw": 0.5})
    ax.bar(x - w, r.mean(1), w, yerr=r.std(1), color=C_RAW, label="adapted model", **kw)
    ax.bar(x, gg.mean(1), w, yerr=gg.std(1), color=C_GMT, label="+ GMT", **kw)
    if ged and all(d in ged[L] for L in ged for d in ds):
        g = np.array([[ged[L][d] for L in ged] for d in ds])
        ax.bar(x + w, g.mean(1), w, yerr=g.std(1), color=C_GMTE, label="+ GMT-E", **kw)
    ax.set_xticks(x); ax.set_xticklabels([DNAME[d] for d in ds], rotation=30, ha="right")
    ax.set_ylabel("novel images sent\nto a base class (%)"); ax.legend(frameon=False, fontsize=6)
    ax.set_title(f"(b) where the novel images go (mean over {len(Ls)} learners, 3 seeds)", fontsize=7)
    macro("NtoBRawMax", float(r.mean(1).max())); macro("NtoBRawMaxDataset", DNAME[ds[int(r.mean(1).argmax())]])
    fig.subplots_adjust(wspace=0.35)
    savefig(fig, "fig1_teaser")


def fig_calibration(A, learner="maple", seed="1", want=("dtd", "eurosat", "oxford_flowers")):
    if learner not in A:
        learner = next(iter(A), None)
    if learner is None:
        return
    ds = [d for d in want if os.path.exists(f"{calib_dir(learner, seed)}/{d}/points.npz")]
    if not ds:
        ds = [d for d in DATASETS if os.path.exists(f"{calib_dir(learner, seed)}/{d}/points.npz")][:3]
    if not ds:
        return
    fig, axs = plt.subplots(1, len(ds), figsize=(2.3 * len(ds), 1.8), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, d in zip(axs, ds):
        p = np.load(f"{calib_dir(learner, seed)}/{d}/points.npz")
        g, gain = p["g"], p["gain"]
        fit = isotonic_decreasing(g, gain); o = np.argsort(g)
        g0, tau = fit_threshold(g, gain)
        ax.scatter(g, gain, s=3, color=C_RAW, alpha=0.35, lw=0, label="calibration subset")
        ax.plot(g[o], fit[o], color=C_GMT, lw=1.2, label="monotone fit")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(g0, color=C_GMT, ls="--", lw=0.8, label=f"$g_0$")
        try:
            tg = A[learner][seed][d]["gate"]["gini"]
            ax.axvline(tg, color=C_FULL, ls=":", lw=1.0, label="test gini")
        except KeyError:
            pass
        ax.set_title(DNAME[d]); ax.set_xlabel("zero-shot gini")
    axs[0].set_ylabel("gain of Sinkhorn (points)")
    axs[-1].legend(frameon=False, fontsize=6, loc="lower left")
    savefig(fig, "fig3_calibration")


def fig_lambda(AB):
    Ls = [L for L in LEARNERS if L in AB]
    if not Ls:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    xs = np.linspace(-0.2, 0.2, 200)
    for L in Ls:
        X, Y, O = [], [], []
        for s in AB[L]:
            for d, r in AB[L][s].items():
                X.append((r["gini"] - r["g0"]) / max(r["tau"], 1e-6)); Y.append(r["lam"]["ours"]); O.append(r["lam"]["orc-lam"])
        ax.scatter(X, O, s=10, facecolors="none", edgecolors=LCOL[L], lw=0.7)
        ax.scatter(X, Y, s=8, color=LCOL[L], label=LNAME[L])
    z = np.linspace(min(-8, ax.get_xlim()[0]), max(8, ax.get_xlim()[1]), 200)
    ax.plot(z, 1 / (1 + np.exp(z)), color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("(test gini $-\\ g_0$) / $\\tau$"); ax.set_ylabel("correction strength $\\lambda$")
    ax.set_title("filled: GMT gate, hollow: best $\\lambda$ on test labels", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    savefig(fig, "fig4_lambda")


def curve(A, key, xs, f):
    vals = []
    for L in A:
        for s in A[L]:
            for d in A[L][s]:
                try:
                    vals.append([f(A[L][s][d][key][x]) for x in xs])
                except KeyError:
                    pass
    return np.array(vals).mean(0) if vals else None


def fig_robustness(A):
    if not A:
        return
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.8))
    for ax, key, xs, xl in [(axs[0], "skew", ["0.0", "0.5", "1.0", "1.5", "2.0"], "test label skew $\\alpha$"),
                            (axs[1], "novel_frac", ["0.1", "0.3", "0.5", "0.7", "0.9"], "share of novel images")]:
        full = curve(A, key, xs, lambda r: r["full"] - r["raw"])
        gmt = curve(A, key, xs, lambda r: r["gmt"] - r["raw"])
        if full is None:
            continue
        xv = [float(x) for x in xs]
        ax.axhline(0, color="k", lw=0.5)
        ax.plot(xv, full, "o-", color=C_FULL, ms=3, lw=1, label="Sinkhorn, always on")
        ax.plot(xv, gmt, "o-", color=C_GMT, ms=3, lw=1, label="GMT")
        gmte = curve(A, key, xs, lambda r: r["gmte"] - r["raw"])
        if gmte is not None:
            ax.plot(xv, gmte, "s-", color=C_GMTE, ms=3, lw=1, label="GMT-E")
            tag = "Skew" if key == "skew" else "Novel"
            macro(f"Worst{tag}CurveGmte", float(min(gmte)), "{:+.2f}")
        ax.set_xlabel(xl)
        if key == "skew":
            macro("SkewTwoFull", float(full[-1]), "{:+.2f}"); macro("SkewTwoGmt", float(gmt[-1]), "{:+.2f}")
            macro("WorstCurveGmt", float(min(gmt)), "{:+.2f}")
        else:
            macro("NovelTenFull", float(full[0]), "{:+.2f}"); macro("NovelTenGmt", float(gmt[0]), "{:+.2f}")
    axs[0].set_ylabel("$\\Delta$ vs adapted model")
    axs[0].legend(frameon=False, fontsize=6)
    fig.subplots_adjust(wspace=0.35)
    savefig(fig, "fig5_robustness")


def fig_gain_and_batch(A):
    if not A:
        return
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.9))
    ax = axs[0]
    allx, ally = [], []
    for L in [l for l in LEARNERS if l in A]:
        n2b = per_dataset(A[L], lambda x: x["res"]["raw"]["n2b"])
        gain = per_dataset(A[L], lambda x: x["res"]["gmt"]["mixed"] - x["res"]["raw"]["mixed"])
        x = [n2b[d] for d in n2b]; y = [gain[d] for d in n2b]
        ax.scatter(x, y, s=10, color=LCOL[L], label=LNAME[L]); allx += x; ally += y
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("novel$\\to$base rate of adapted model (%)"); ax.set_ylabel("GMT gain (points)")
    if len(allx) > 2:
        r = np.corrcoef(allx, ally)[0, 1]; ax.set_title(f"Pearson r = {r:.2f}", fontsize=6.5); macro("CorrGainBias", r)
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    ax = axs[1]
    bss = ["16", "32", "64", "128", "256", "512"]
    full = curve(A, "batch", bss, lambda r: r["full"]); gmt = curve(A, "batch", bss, lambda r: r["gmt"])
    st = curve(A, "batch", bss, lambda r: r["stream"])
    rawv = np.mean([A[L][s][d]["batch"]["raw"] for L in A for s in A[L] for d in A[L][s]])
    fullset = np.mean([A[L][s][d]["batch"]["full"]["gmt"] for L in A for s in A[L] for d in A[L][s]])
    if full is not None:
        xv = [int(b) for b in bss]
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(fullset - rawv, color=C_GMT, ls=":", lw=0.8, label="GMT, whole test set")
        ax.plot(xv, full - rawv, "o-", color=C_FULL, ms=3, lw=1, label="Sinkhorn per batch")
        ax.plot(xv, gmt - rawv, "o-", color=C_GMT, ms=3, lw=1, label="GMT per batch")
        ax.plot(xv, st - rawv, "s-", color=C_STREAM, ms=3, lw=1, label="online GMT")
        ste = curve(A, "batch", bss, lambda r: r["gmte_stream"])
        if ste is not None:
            fse = np.mean([A[L][s][d]["batch"]["full"]["gmte"] for L in A for s in A[L] for d in A[L][s]
                           if "gmte" in A[L][s][d]["batch"]["full"]])
            ax.axhline(fse - rawv, color=C_GMTE, ls=":", lw=0.8, label="GMT-E, whole test set")
            macro("StreamGmteShareSixteen", float(100 * (ste[0] - rawv) / (fse - rawv)) if fse != rawv else None, "{:.0f}")
            ax.plot(xv, ste - rawv, "s-", color=C_GMTE, ms=3, lw=1, label="online GMT-E")
            macro("StreamGmteGainSixteen", float(ste[0] - rawv), "{:+.2f}")
        ax.set_xscale("log", base=2); ax.set_xticks(xv); ax.set_xticklabels(bss)
        ax.set_xlabel("test batch size"); ax.set_ylabel("$\\Delta$ vs adapted model")
        ax.legend(frameon=False, fontsize=5.5)
        macro("StreamGainSixteen", float(st[0] - rawv), "{:+.2f}")
        macro("FullSetGain", float(fullset - rawv), "{:+.2f}")
        macro("StreamShareSixteen", float(100 * (st[0] - rawv) / (fullset - rawv)) if fullset != rawv else None, "{:.0f}")
        macro("BatchSixteenFull", float(full[0] - rawv), "{:+.2f}")
    fig.subplots_adjust(wspace=0.45)
    savefig(fig, "fig6_gain_batch")


def fig_offset(B):
    pts = [(r["gamma"]["val"], r["gamma"]["oracle"], L) for L in B for s in B[L] for d, r in B[L][s].items()
           if "gamma" in r]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    for L in [l for l in LEARNERS if l in B]:
        p = [(a, b) for a, b, l in pts if l == L]
        ax.scatter(*zip(*p), s=8, color=LCOL[L], label=LNAME[L])
    lo = min(min(a, b) for a, b, _ in pts) - 0.2; hi = max(max(a, b) for a, b, _ in pts) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.6)
    ax.set_xlabel("offset tuned on calibration"); ax.set_ylabel("offset best on test")
    ax.legend(frameon=False, fontsize=5.5)
    savefig(fig, "supp_offset")



COMPACT = [("ab", "full SK", "$\\lambda=1$: Sinkhorn always on"), ("ab", "fix-glob", "Fixed $\\lambda$, one for all datasets"),
           ("ab", "fix-ds", "Fixed $\\lambda$ per dataset"), ("ab", "hard", "Hard threshold instead of sigmoid"),
           ("ab", "pooled", "One pooled $g_0,\\tau$ for all datasets"), ("an", "gmt_lodo", "$g_0,\\tau$ from the other datasets (LODO)"),
           ("sa", "entropy", "Entropy instead of gini as gate signal"), ("an", "gmt_zs_target", "Zero-shot-estimated target marginal"),
           ("ab", "ours", "GMT"), ("ab", "tau*2", "GMT, $2\\tau$"), ("MID", "", ""),
           ("ab", "orc-lam", "Oracle $\\lambda$ (test labels)"), ("an", "gmt_true_target", "True test marginal (test labels)")]


def table_ablation_compact(A, AB, SA):
    pr = {L: v for L, v in ((SA or {}).get("per_run") or {}).items() if v and (ALLOW_PARTIAL or complete(v))}
    src = {"ab": AB, "an": A, "sa": pr}
    Ls = [L for L in LEARNERS if L in AB]
    if not Ls:
        return
    rows = []
    for kind, key, label in COMPACT:
        if kind == "MID":
            rows.append("MID"); continue
        means, worsts, n = [], [], 0
        for L in Ls:
            S = src[kind].get(L)
            if not S:
                continue
            f = (lambda x, k=key: x["signals"][k]) if kind == "sa" else (lambda x, k=key: x["res"][k]["mixed"] - x["res"]["raw"]["mixed"])
            g = per_seed(S, f)
            if g:
                means.append(ms(g)[0]); worsts.append(worst(S, f)); n += 1
        if not means:
            rows.append([label, "--", "--", "--"]); continue
        m = float(np.mean(means))
        body = f"\\textbf{{{m:+.2f}}}" if key == "ours" else f"{m:+.2f}"
        rows.append([label, body, f"{sum(v > 0 for v in means)}/{n}", f"{min(worsts):+.2f}"])
        if key == "ours":
            macro("AblGmtWorstAll", min(worsts), "{:+.2f}")
    write_table("tab3_ablation", f"Ablations, summarised over {len(Ls)} prompt learners: mean gain in joint accuracy over the raw model, number of learners that improve, and the worst change on any of the {len(Ls) * 24} dataset-seed pairs. Per-learner results are in the supplementary material. The last two rows use test labels.",
                ["Variant", "Mean $\\Delta$", "Learners improved", "Worst pair"], rows, "lccc", "tab:ablation")



def table_per_seed(A, B):
    """Every seed separately: dataset-averaged joint accuracy of raw, GMT and GMT-E, and N->B."""
    rows = []
    for L in LEARNERS:
        if L not in A:
            continue
        ds = common_ds(A[L])
        for s in sorted(A[L]):
            m = lambda f: float(np.mean([f(A[L][s][d]) for d in ds]))
            raw = m(lambda x: x["res"]["raw"]["mixed"]); g = m(lambda x: x["res"]["gmt"]["mixed"])
            e = float(np.mean([B[L][s][d]["res"]["GMT-E"]["mixed"] for d in ds])) if L in B and s in B[L] else None
            rows.append([LNAME[L] if s == sorted(A[L])[0] else "", s, f"{raw:.2f}", f"{g:.2f}", f"{g - raw:+.2f}",
                         f"{e:.2f}" if e is not None else "--", f"{e - raw:+.2f}" if e is not None else "--",
                         f"{m(lambda x: x['res']['raw']['n2b']):.2f}", f"{m(lambda x: x['res']['gmt']['n2b']):.2f}"])
        rows.append("MID")
    if rows:
        write_table("supp_per_seed", "Results of every seed separately (joint accuracy and N$\\to$B averaged over the 8 datasets). The main tables report the mean and standard deviation of these three values.",
                    ["Learner", "Seed", "Raw", "+GMT", "$\\Delta$", "+GMT-E", "$\\Delta$", "N$\\to$B raw", "N$\\to$B +GMT"],
                    rows[:-1], "llccccccc", "tab:perseed")

# ------------------------------------------------------------------ supplementary (supp_analyses.py)
def supp_tables(SA):
    if not SA:
        return
    pr = {L: v for L, v in SA.get("per_run", {}).items() if v and (ALLOW_PARTIAL or complete(v))}
    Ls = [L for L in LEARNERS if L in pr]
    groups = [[""] + [f"\\multicolumn{{2}}{{c}}{{{LNAME[L]}}}" for L in Ls],
              " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(Ls)))]
    hdr = sum([["$\\Delta$", "worst"] for _ in Ls], [])

    def block(key, items, name, caption, label, first):
        rows = []
        for k, lab in items:
            f = (lambda x, k=k: x[key][k]["gain"]) if key == "calib_size" else (lambda x, k=k: x[key][k])
            r = [lab]
            for L in Ls:
                g = per_seed(pr[L], f)
                r += [pm(g, True, bold=lab.endswith("(GMT)")), sgn(worst(pr[L], f)) if g else "--"]
            rows.append(r)
        write_table(name, caption, [first] + hdr, rows, "l" + "rr" * len(Ls), label, wide=True, groups=groups)

    if Ls:
        block("signals", [("gini", "Gini coefficient (GMT)"), ("entropy", "1 $-$ normalised entropy"),
                          ("kl_uniform", "KL to uniform"), ("maxmin", "log max/min class share")],
              "supp_gate_signals", "Gate signal computed on the zero-shot class marginal. Each signal gets its own threshold, fitted on the same calibration subsets and gains. Gain in joint accuracy over the raw model (mean $\\pm$ std over 3 seeds) and worst dataset-seed change.",
              "tab:signals", "Gate signal")
        block("sinkhorn", [("iters_1", "1 Sinkhorn iteration"), ("iters_5", "5 iterations"), ("iters_20", "20 iterations"),
                           ("iters_200", "200 iterations (GMT)"), ("prob_blend", "200 iterations, blend in probability space")],
              "supp_sinkhorn", "Sinkhorn iterations and blending space, with the gate value fixed. GMT blends log-probabilities.",
              "tab:sinkhorn", "Variant")
        block("calib_size", [("1 split, 5 subsets per skew", "1 split, 5 subsets per skew"), ("1 split", "1 split"),
                             ("2 splits", "2 splits"), ("3 splits, 5 subsets per skew", "3 splits, 5 subsets per skew"),
                             ("3 splits (GMT)", "3 splits (GMT)")],
              "supp_calib_size", "Size of the calibration set used to fit $g_0$ and $\\tau$: number of pseudo-splits and of skewed subsets per skew level.",
              "tab:calibsize", "Calibration")
        for L in Ls:
            for k, tag in [("entropy", "Entropy"), ("kl_uniform", "Kl"), ("maxmin", "Maxmin")]:
                macro(f"GainSignal{tag}{LTAG[L]}", ms(per_seed(pr[L], lambda x, k=k: x["signals"][k]))[0], "{:+.2f}")
            macro(f"GainOneSplit{LTAG[L]}", ms(per_seed(pr[L], lambda x: x["calib_size"]["1 split"]["gain"]))[0], "{:+.2f}")
    rt = SA.get("runtime")
    if rt:
        keys = [k for k in ("16", "64", "256", "1024", "full") if k in rt]
        rows = [[(f"{rt[k]['batch']}" + (" (whole test set)" if k == "full" else "")),
                 f"{rt[k]['ms_per_batch']:.2f}", f"{rt[k]['us_per_image']:.1f}"] for k in keys]
        write_table("supp_runtime", f"Cost of GMT per test batch on one CPU thread (numpy): zero-shot gini, 200 Sinkhorn iterations and blending, for {rt['classes']} classes ({rt['file'].split('/')[-2]}). The image and text encoders are not included.",
                    ["Batch size", "ms per batch", "$\\mu$s per image"], rows, "lrr", "tab:runtime")
        macro("RuntimeUsPerImageSixteen", rt["16"]["us_per_image"], "{:.1f}"); macro("RuntimeMsFull", rt["full"]["ms_per_batch"], "{:.0f}")


HPARAMS = [("CoOp", "4 (\\textit{a photo of a})", "--", "200", "32", "0.002", "--", "fp16, cached features"),
           ("CoCoOp", "4 (\\textit{a photo of a})", "--", "10", "1", "0.002", "meta-net 512$\\to$32$\\to$512", "fp16, cached features"),
           ("KgCoOp", "4 (\\textit{a photo of a})", "--", "100", "128", "0.002", "$\\lambda=8$, 1 $-$ cosine to hand-crafted prompt", "fp32, cached features"),
           ("IVLP", "2 + 2", "12", "5", "4", "0.0035", "independent vision and text prompts", "fp32"),
           ("MaPLe", "2", "9", "5", "4", "0.0035", "vision prompts coupled to text prompts", "fp32"),
           ("PromptSRC", "4 + 4", "9", "20", "4", "0.0025", "L1 text 25, L1 image 10, logit KL, GPA ($\\mu=15$, $\\sigma=1$)", "fp16")]


def table_hparams():
    write_table("supp_hparams", "Training configuration of every prompt learner: the official base-to-novel configuration of each method, run in one shared code base with ViT-B/16, 16 shots, SGD with cosine schedule and one warm-up epoch. Cached features: the image encoder is frozen, so image features of 20 augmented views per training image are computed once.",
                ["Learner", "Context tokens", "Depth", "Epochs", "Batch", "LR", "Method-specific", "Notes"],
                [list(r) for r in HPARAMS], "lccccrll", "tab:hparams", wide=True)


def fig_mechanism(SA):
    M = (SA or {}).get("mechanism")
    if not M:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.2)); rs = []
    for L in [l for l in LEARNERS if l in M]:
        p = M[L]["points"]
        ax.scatter([q["sim"] for q in p], [q["n2b"] for q in p], s=4, alpha=0.5, color=LCOL[L], label=LNAME[L], lw=0)
        rs += list(M[L]["spearman_per_dataset"].values())
    r = float(np.nanmean(rs)); macro("SpearmanSimNtoB", r)
    ax.set_xlabel("text similarity to the nearest base class"); ax.set_ylabel("images sent to base classes (%)")
    ax.set_title(f"mean Spearman per dataset: {r:.2f}", fontsize=6.5); ax.legend(frameon=False, fontsize=5.5, ncol=2, markerscale=2)
    savefig(fig, "supp_mechanism")

VARIANTS = [("maple", "MaPLe", "ViT-B/16", "16"), ("maple_k4", "MaPLe", "ViT-B/16", "4"),
            ("maple_k1", "MaPLe", "ViT-B/16", "1"), ("maple_b32", "MaPLe", "ViT-B/32", "16"),
            ("coop", "CoOp", "ViT-B/16", "16"), ("coop_k4", "CoOp", "ViT-B/16", "4"),
            ("coop_k1", "CoOp", "ViT-B/16", "1"), ("coop_b32", "CoOp", "ViT-B/32", "16")]


VSTAT = {}


def table_variants(A):
    rows = []
    for tag, name, bb, shots in VARIANTS:
        a = A.get(tag)
        b = load_json(f"runs/baselines3_{tag}.json")
        b = {seed_of(k): v for k, v in b.items()} if b else None
        if not a or not (ALLOW_PARTIAL or complete(a)):
            if tag not in ("maple", "coop"):
                WARN.append(f"variant {tag}: no complete analysis yet")
            continue
        f = lambda x: x["res"]["gmt"]["mixed"] - x["res"]["raw"]["mixed"]
        fe = lambda x: x["res"]["GMT-E"]["mixed"] - x["res"]["raw"]["mixed"]
        hm = per_seed(a, lambda x: x["b2n"]["hm"]); jhm = per_seed(a, lambda x: hm2(x["res"]["raw"]["base_img"], x["res"]["raw"]["novel_img"]))
        ge = per_seed(b, fe) if b and (ALLOW_PARTIAL or complete(b)) else []
        rows.append([name, bb, shots, pm(hm), pm(jhm), pm(per_seed(a, lambda x: x["res"]["raw"]["n2b"])),
                     pm(per_seed(a, f), True), sgn(worst(a, f)), pm(ge, True), sgn(worst(b, fe)) if ge else "--"])
        t = {"maple": "MapleMain", "maple_k4": "MapleFourShot", "maple_k1": "MapleOneShot", "maple_b32": "MapleBThirtyTwo",
             "coop": "CoopMain", "coop_k4": "CoopFourShot", "coop_k1": "CoopOneShot", "coop_b32": "CoopBThirtyTwo"}[tag]
        macro(f"VarGain{t}", ms(per_seed(a, f))[0], "{:+.2f}")
        macro(f"VarGap{t}", ms(hm)[0] - ms(jhm)[0])
        macro(f"VarNtoB{t}", ms(per_seed(a, lambda x: x["res"]["raw"]["n2b"]))[0])
        macro(f"VarWorstGmt{t}", worst(a, f), "{:+.2f}")
        VSTAT.setdefault("gap", []).append(ms(hm)[0] - ms(jhm)[0])
        VSTAT.setdefault("gain", []).append(ms(per_seed(a, f))[0])
        VSTAT.setdefault("worst", []).append(worst(a, f))
        if ge:
            macro(f"VarGainGmte{t}", ms(ge)[0], "{:+.2f}"); macro(f"VarWorstGmte{t}", worst(b, fe), "{:+.2f}")
    if VSTAT.get("gap"):
        macro("VarGapMin", min(VSTAT["gap"])); macro("VarGapMax", max(VSTAT["gap"]))
        macro("VarGainMin", min(VSTAT["gain"]), "{:+.2f}"); macro("VarGainMax", max(VSTAT["gain"]), "{:+.2f}")
        macro("VarWorstGmtAll", min(VSTAT["worst"]), "{:+.2f}"); macro("VarWorstGmtAllAbs", abs(min(VSTAT["worst"])))
    if rows:
        write_table("tab4_variants", "Fewer shots and a second backbone (mean $\\pm$ std over 3 seeds, 8 datasets). The hidden drop (standard vs joint HM) and the gains of GMT and GMT-E persist.",
                    ["Learner", "Backbone", "Shots", "Std.\\ HM", "Joint HM", "N$\\to$B", "+GMT $\\Delta$", "worst",
                     "+GMT-E $\\Delta$", "worst"], rows, "lllccccrcr", "tab:variants", wide=True)


def table_asymmetry(SA):
    pr = {L: v for L, v in (SA or {}).get("per_run", {}).items() if v and (ALLOW_PARTIAL or complete(v))}
    Ls = [L for L in LEARNERS if L in pr and "asym" in next(iter(next(iter(pr[L].values())).values()))]
    if not Ls:
        return
    G0 = lambda L, key, k: ms(per_seed(pr[L], lambda x: x["asym"][key][k]))[0]
    def row(name, L, key, std=True):
        g = lambda k: ms(per_seed(pr[L], lambda x, k=k: x["asym"][key][k]))[0]
        sh = g("std_hm") if std else None
        return [name, f"{sh:.2f}" if std else "--", f"{g('joint_hm'):.2f}", f"{sh - g('joint_hm'):.2f}" if std else "--",
                f"{g('n2b'):.2f}", f"{g('b2n'):.2f}"]
    zl = next((L for L in Ls if "zs_gmt" in next(iter(next(iter(pr[L].values())).values()))["asym"]), None)
    rows = [row("Zero-shot CLIP", Ls[0], "zs")] + ([row("\\quad + GMT", zl, "zs_gmt", std=False)] if zl else []) + ["MID"]
    for L in Ls:
        rows += [row(LNAME[L], L, "raw"), row(f"\\quad + GMT", L, "gmt", std=False)]
    zs_drop = ms(per_seed(pr[Ls[0]], lambda x: x["asym"]["zs"]["std_hm"] - x["asym"]["zs"]["joint_hm"]))[0]
    ad_drop = float(np.mean([ms(per_seed(pr[L], lambda x: x["asym"]["raw"]["std_hm"] - x["asym"]["raw"]["joint_hm"]))[0] for L in Ls]))
    macro("ZsHiddenGap", zs_drop); macro("AdaptedHiddenGap", ad_drop); macro("ExcessHiddenGap", ad_drop - zs_drop)
    macro("ZsNtoB", ms(per_seed(pr[Ls[0]], lambda x: x["asym"]["zs"]["n2b"]))[0])
    macro("ZsBtoN", ms(per_seed(pr[Ls[0]], lambda x: x["asym"]["zs"]["b2n"]))[0])
    macro("AdaptedBtoN", float(np.mean([ms(per_seed(pr[L], lambda x: x["asym"]["raw"]["b2n"]))[0] for L in Ls])))
    macro("AdaptedNtoB", float(np.mean([ms(per_seed(pr[L], lambda x: x["asym"]["raw"]["n2b"]))[0] for L in Ls])))
    ratios = [ms(per_seed(pr[L], lambda x: x["asym"]["raw"]["n2b"]))[0] / max(1e-9, ms(per_seed(pr[L], lambda x: x["asym"]["raw"]["b2n"]))[0]) for L in Ls]
    zr = ms(per_seed(pr[Ls[0]], lambda x: x["asym"]["zs"]["n2b"]))[0] / max(1e-9, ms(per_seed(pr[Ls[0]], lambda x: x["asym"]["zs"]["b2n"]))[0])
    macro("ZsAsymRatio", zr); macro("MinAsymRatio", min(ratios)); macro("MaxAsymRatio", max(ratios))
    # how much GMT moves each error direction (quoted in Sec. 5.1)
    n2b_drop = [G0(L, "raw", "n2b") - G0(L, "gmt", "n2b") for L in Ls]
    b2n_rise = [G0(L, "gmt", "b2n") - G0(L, "raw", "b2n") for L in Ls]
    macro("NtoBDropMin", min(n2b_drop), "{:.1f}"); macro("NtoBDropMax", max(n2b_drop), "{:.1f}")
    macro("BtoNRiseMax", max(b2n_rise), "{:.1f}")
    # advantage of each learner over zero-shot CLIP, as the benchmark reports it and as deployment sees it
    G = lambda L, key, k: ms(per_seed(pr[L], lambda x: x["asym"][key][k]))[0]
    zs_std, zs_joint = G(Ls[0], "zs", "std_hm"), G(Ls[0], "zs", "joint_hm")
    zs_gmt = G(zl, "zs_gmt", "joint_hm") if zl else None
    mrows, mstd, mj, mg, mf = [], [], [], [], []
    for L in Ls:
        a1, a2, a3 = G(L, "raw", "std_hm") - zs_std, G(L, "raw", "joint_hm") - zs_joint, G(L, "gmt", "joint_hm") - zs_joint
        a4 = G(L, "gmt", "joint_hm") - zs_gmt if zl else None
        mstd.append(a1); mj.append(a2); mg.append(a3); mf.append(a4)
        mrows.append([LNAME[L], f"{a1:+.2f}", f"{a2:+.2f}", f"{a3:+.2f}", f"{a4:+.2f}" if a4 is not None else "--"])
        t = LTAG[L]; macro(f"MarginStd{t}", a1, "{:+.2f}"); macro(f"MarginJoint{t}", a2, "{:+.2f}"); macro(f"MarginJointGmt{t}", a3, "{:+.2f}")
    mrows += ["MID", ["Average", f"{np.mean(mstd):+.2f}", f"{np.mean(mj):+.2f}", f"{np.mean(mg):+.2f}",
                      f"{np.mean(mf):+.2f}" if zl else "--"]]
    macro("MarginStdAll", float(np.mean(mstd)), "{:+.2f}"); macro("MarginJointAll", float(np.mean(mj)), "{:+.2f}")
    macro("MarginJointGmtAll", float(np.mean(mg)), "{:+.2f}")
    macro("NumNoGainJoint", str(int(sum(v <= 0 for v in mj)))); macro("NumNoGainStd", str(int(sum(v <= 0 for v in mstd))))
    if zl:
        macro("MarginJointGmtFairAll", float(np.mean(mf)), "{:+.2f}"); macro("ZsGmtJointHM", zs_gmt)
        macro("ZsGmtGain", zs_gmt - zs_joint, "{:+.2f}")
    write_table("supp_margin", "Advantage of each prompt learner over zero-shot CLIP in base/novel HM (points). The standard protocol reports the first column; in the joint label set the advantage shrinks. Correcting only the prompt learner with GMT (third column) is not a fair comparison; the last column corrects zero-shot CLIP too, with its own calibrated gate, and the advantage stays at about half of what the standard protocol reports. Mean over 3 seeds and 8 datasets.",
                ["Learner", "Standard", "Joint", "Joint + GMT", "Joint + GMT vs zero-shot + GMT"], mrows, "lcccc", "tab:margin")
    write_table("supp_asymmetry", "Is the joint-protocol drop caused by base-class bias or only by more classes competing? Zero-shot CLIP, which is not tuned on base classes, is the reference. Drop: standard minus joint HM. N$\\to$B / B$\\to$N: share of novel images predicted as base and of base images predicted as novel. Mean over 3 seeds and 8 datasets.",
                ["Model", "Std.\\ HM", "Joint HM", "Drop", "N$\\to$B", "B$\\to$N"], rows, "lccccc", "tab:asymmetry")


def table_balance(SA):
    pr = (SA or {}).get("per_run", {})
    vals = {}
    for L, seeds in pr.items():
        for s, ds in seeds.items():
            for d, r in ds.items():
                if "balance" in r:
                    v = vals.setdefault(d, {"label": [], "zs": [], "lam": []})
                    v["label"].append(r["balance"]["label_gini"]); v["zs"].append(r["balance"]["zs_gini"])
                    v["lam"].append(r["balance"]["lam"])
    if not vals:
        return
    rows = [[DNAME[d], f"{np.mean(vals[d]['label']):.3f}", f"{np.mean(vals[d]['zs']):.3f}", f"{np.mean(vals[d]['lam']):.2f}"]
            for d in DATASETS if d in vals]
    write_table("supp_balance", "Class balance of each test set (gini of the true label distribution) against the balance of zero-shot predictions that the gate reads, and the gate value $\\lambda$ (mean over learners and seeds). Where the two gini values disagree (FGVC-Aircraft), the gate misreads a balanced test set as skewed.",
                ["Dataset", "True label gini", "Zero-shot predicted gini", "Mean $\\lambda$"], rows, "lccc", "tab:balance")
    if "fgvc_aircraft" in vals:
        macro("AircraftLabelGini", float(np.mean(vals["fgvc_aircraft"]["label"])), "{:.3f}")
        macro("AircraftZsGini", float(np.mean(vals["fgvc_aircraft"]["zs"])), "{:.3f}")


def copy_qualitative():
    """Figures made by qualitative.py and figures_extra.py (they need the images or raw logits)."""
    import shutil
    for f in glob.glob("runs/qualitative/*.p*") + glob.glob("runs/extra_figs/*.p*"):
        if f.endswith((".pdf", ".png")):
            shutil.copy(f, f"{OUT}/figures/{os.path.basename(f)}")


NEWD = [("imagenet", "ImageNet"), ("sun397", "SUN397"), ("stanford_cars", "StanfordCars")]


def table_newdata():
    """GMT on ImageNet, SUN397 and StanfordCars with thresholds pooled from the 8 original datasets."""
    R = load_json("runs/newdata_analysis.json")
    if not R:
        return
    Ls = [L for L in LEARNERS if L in R]
    rows = []
    for L in Ls:
        r = [LNAME[L]]
        for d, _ in NEWD:
            vals = [R[L][s][d] for s in R[L] if d in R[L][s]]
            if not vals:
                r += ["--", "--", "--"]; continue
            g = [v["res"]["gmt"]["mixed"] - v["res"]["raw"]["mixed"] for v in vals]
            ge = [v["res"]["gmte"]["mixed"] - v["res"]["raw"]["mixed"] for v in vals if "gmte" in v["res"]]
            gap = [v["res"]["raw"]["std_hm"] - v["res"]["raw"]["joint_hm"] for v in vals]
            r += [f"{np.mean(gap):.2f}", pm(g, True), pm(ge, True) if ge else "--"]
            macro(f"NewGain{LTAG[L]}{ {'imagenet': 'Imagenet', 'sun397': 'Sun', 'stanford_cars': 'Cars'}[d] }", float(np.mean(g)), "{:+.2f}")
        rows.append(r)
    if not rows:
        return
    allg = [R[L][s][d]["res"]["gmt"]["mixed"] - R[L][s][d]["res"]["raw"]["mixed"] for L in Ls for s in R[L] for d in R[L][s]]
    macro("NewGainAll", float(np.mean(allg)), "{:+.2f}"); macro("NewWorstAll", float(min(allg)), "{:+.2f}")
    macro("NewNumRuns", str(len(allg))); macro("NewNumImproved", str(int(sum(g > 1e-9 for g in allg))))
    macro("NewNumHurt", str(int(sum(g < -1e-9 for g in allg))))
    drops = [np.mean([R[L][s][d]["res"]["raw"]["std_hm"] - R[L][s][d]["res"]["raw"]["joint_hm"] for s in R[L] if d in R[L][s]])
             for L in Ls for d, _ in NEWD if any(d in R[L][s] for s in R[L])]
    macro("NewDropMin", float(min(drops))); macro("NewDropMax", float(max(drops)))
    alle = [R[L][s][d]["res"]["gmte"]["mixed"] - R[L][s][d]["res"]["raw"]["mixed"]
            for L in Ls for s in R[L] for d in R[L][s] if "gmte" in R[L][s][d]["res"]]
    if alle:
        macro("NewGainGmteAll", float(np.mean(alle)), "{:+.2f}"); macro("NewWorstGmteAll", float(min(alle)), "{:+.2f}")
    for d, n in NEWD:
        tag = {"imagenet": "Imagenet", "sun397": "Sun", "stanford_cars": "Cars"}[d]
        macro(f"NewNumLearners{tag}", str(sum(any(d in R[L][s] for s in R[L]) for L in Ls)))
        macro(f"NewNumRuns{tag}", str(sum(d in R[L][s] for L in Ls for s in R[L])))
    groups = [[""] + [f"\\multicolumn{{3}}{{c}}{{{n}}}" for _, n in NEWD],
              " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(NEWD)))]
    write_table("tab5_newdata", "Datasets never used for calibration. GMT uses the threshold pooled from the 8 original datasets and \\GMTE{} their median ensemble weight; nothing is tuned on these datasets. Drop: standard minus joint base/novel HM of the adapted model; $\\Delta$: change in joint accuracy (mean $\\pm$ std over seeds).",
                ["Learner"] + sum([["Drop", "+GMT $\\Delta$", "+GMT-E $\\Delta$"] for _ in NEWD], []), rows,
                "l" + "ccc" * len(NEWD), "tab:newdata", wide=True, groups=groups)

# ------------------------------------------------------------------ main
def main():
    if not os.path.isdir("runs"):
        sys.exit("run this from the trustprompt folder (runs/ not found)")
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)          # no stale tables or figures from earlier runs
    A, B, AB = load_all()
    SA = load_json("runs/supp_analysis.json")
    table_main(A, B, SA); table_baselines(B); table_ablation(A, AB); table_ablation_compact(A, AB, SA)
    table_per_dataset(A, B); table_per_seed(A, B); table_split_g0(); table_significance(B); table_newdata()
    supp_tables(SA); table_hparams(); fig_mechanism(SA); table_variants(A); table_asymmetry(SA); table_balance(SA)
    fig_n2b(A, B); fig_calibration(A); fig_lambda(AB); fig_robustness(A); fig_gain_and_batch(A); fig_offset(B)
    for sub in ("", "/tables", "/figures"):
        os.makedirs(OUT + sub, exist_ok=True)
    copy_qualitative()
    derived_macros()
    open(f"{OUT}/numbers.tex", "w").write("% generated by make_assets.py; do not edit\n" + "\n".join(MACROS) + "\n")
    summary = ["# Paper assets summary", "", "Learners found: " + ", ".join(LNAME[L] for L in LEARNERS if L in A), ""]
    if WARN:
        summary += ["## Missing inputs", ""] + [f"- {w}" for w in WARN] + [""]
    summary += TABLES_MD + ["## In-text numbers", "", "```"] + MACROS + ["```"]
    open(f"{OUT}/summary.md", "w").write("\n".join(summary))
    print(f"tables : {sorted(os.listdir(OUT + '/tables'))}")
    print(f"figures: {sorted(f for f in os.listdir(OUT + '/figures') if f.endswith('.pdf'))}")
    print(f"numbers: {len(MACROS)} macros -> {OUT}/numbers.tex")
    for w in WARN:
        print("missing:", w)


if __name__ == "__main__":
    main()
