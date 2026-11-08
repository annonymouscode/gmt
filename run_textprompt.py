"""Fast CoOp / KgCoOp / CoCoOp runs: test-logit dump + 3-split calibration in one process.

CoOp and KgCoOp never change the image encoder, so image features can be computed once and the
prompts trained on cached features. Official hyperparameters are kept (epochs, batch size, lr,
schedule, loss). The only change is data augmentation: each training image gets --views random
augmented views (random resized crop + flip, as in the official code), and every step samples one
of them, instead of drawing a fresh augmentation each epoch.

Outputs are identical in format to dump_logits.py and calibrate_mixed.py:
  runs/<learner>/logits_s<seed>/<dataset>/logits.npz
  runs/<learner>/calib_s<seed>/<dataset>/split<s>/calib_logits.npz  and  points.npz
Any output that already exists is skipped.

  python run_textprompt.py --root ~/data_compact --dataset dtd --learner coop --seed 1
"""
import argparse, math, os, random, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import clip

from trustprompt.datasets import (load_dataset, few_shot, classnames_of, base_novel_split, subset,
                                  build_transforms, ImageList)
from trustprompt.learners import build_learner, fill_defaults
from trustprompt.engine import build_optimizer
from calibrate_mixed import sweep, split_seed


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def encode(cm, items, transform, dev, workers, views=1, half=False):
    """(views, N, D) L2-normalised image features, in the order of `items`."""
    dl = DataLoader(ImageList(items, transform), batch_size=256, shuffle=False,
                    num_workers=workers, pin_memory=True)
    out = []
    for _ in range(views):
        fs = []
        for imgs, _ in dl:
            with torch.autocast("cuda", dtype=torch.float16, enabled=half and dev.type == "cuda"):
                f = cm.encode_image(imgs.to(dev))
            fs.append(F.normalize(f.float(), dim=-1).half() if half else F.normalize(f.float(), dim=-1))
        out.append(torch.cat(fs))
    return torch.stack(out)


def train_prompts(a, cm, names, bank, labels, dev, seed):
    """Train a fresh CoOp/KgCoOp prompt on cached features. bank: (V, N, D), labels: (N,)."""
    torch.manual_seed(seed)
    cm.cpu()                       # learners tokenize their init prompt on CPU, as in dump_logits.py
    model = build_learner(a.learner, cm, a.dataset, a.depth, a.n_ctx, a.kg_lambda).to(dev)
    V, N = bank.shape[:2]
    steps = math.ceil(N / a.bs)
    opt, sched = build_optimizer(model.trainable_parameters(), a.lr, a.epochs, steps)
    g = torch.Generator().manual_seed(seed)
    t0 = model.handcrafted_text(names, dev) if a.learner == "kgcoop" else None
    amp = a.amp and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for ep in range(a.epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, a.bs):
            idx = perm[i:i + a.bs]
            v = torch.randint(V, (len(idx),), generator=g)
            f = bank[v.to(dev), idx.to(dev)].float()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                if a.learner == "cocoop":
                    logits = model.logits_from_feats(f, names)
                else:
                    t = model.text_features(names)      # returns float32, L2-normalised
                    logits = model.logit_scale * f @ t.t()
            loss = F.cross_entropy(logits.float(), labels[idx.to(dev)])
            if t0 is not None:
                loss = loss + a.kg_lambda * (1.0 - (t.float() * t0).sum(-1).mean())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if (ep + 1) % max(1, a.epochs // 10) == 0 or ep + 1 == a.epochs:
            log(f"  epoch {ep+1}/{a.epochs} loss {loss.item():.4f}")
    model.eval()
    return model


@torch.no_grad()
def score(model, feats, names, dev):
    t0 = model.handcrafted_text(names, dev)
    s = model.logit_scale
    zs = (s * feats @ t0.t()).float().cpu().numpy()
    if hasattr(model, "logits_from_feats"):              # CoCoOp: one prompt set per image
        chunk = max(1, 2048 // len(names)); out = []
        for i in range(0, feats.shape[0], chunk):
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                out.append(model.logits_from_feats(feats[i:i + chunk], names).float().cpu())
        return torch.cat(out).numpy(), zs
    t = model.text_features(names)
    return (s * feats @ t.t()).float().cpu().numpy(), zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--learner", required=True, choices=["coop", "kgcoop", "cocoop"])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--backbone", default="ViT-B/16"); ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--splits", default="0,1,2")
    ap.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    ap.add_argument("--reps", type=int, default=20)
    for k in ("depth", "n_ctx", "epochs", "bs"):
        ap.add_argument(f"--{k}", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None); ap.add_argument("--kg_lambda", type=float, default=None)
    ap.add_argument("--no-calib", action="store_true",
                    help="only dump test logits (new datasets: GMT uses thresholds pooled from other datasets)")
    ap.add_argument("--tag", default=None, help="output folder name under runs/ (default: the learner name)")
    ap.add_argument("--amp", action="store_true",
                    help="mixed-precision prompt training (the official CoOp code trains in fp16)")
    a = fill_defaults(ap.parse_args())
    if a.kg_lambda is None:
        a.kg_lambda = 8.0
    log(f"{a.learner} {a.dataset} seed {a.seed}: n_ctx {a.n_ctx} epochs {a.epochs} bs {a.bs} "
        f"lr {a.lr} views {a.views} amp {a.amp}")

    tag = a.tag or a.learner
    lout = f"runs/{tag}/logits_s{a.seed}/{a.dataset}"
    cout = f"runs/{tag}/calib_s{a.seed}/{a.dataset}"
    splits = [int(s) for s in a.splits.split(",")]
    do_dump = not os.path.exists(f"{lout}/logits.npz")
    todo = [] if a.no_calib else [s for s in splits if not os.path.exists(f"{cout}/split{s}/calib_logits.npz")]
    if not do_dump and (a.no_calib or (not todo and os.path.exists(f"{cout}/points.npz"))):
        log("all outputs exist, nothing to do"); return

    torch.backends.cuda.enable_mem_efficient_sdp(False)              # same as dump_logits.py
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"): torch.backends.cuda.enable_cudnn_sdp(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(a.root, a.dataset); names_all = classnames_of(data["test"])
    base, novel = base_novel_split(data)
    cm, _ = clip.load(a.backbone, device="cpu", jit=False)
    cm = cm.float().to(dev).eval()
    tr, te = build_transforms(True), build_transforms(False)

    # ---------------- test-logit dump (as dump_logits.py)
    if do_dump:
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        order = base + novel
        names = [names_all[l] for l in order]; base_names = [names_all[l] for l in base]
        train_items = few_shot(subset(data["train"], base), a.shots, a.seed)
        test_items = subset(data["test"], order)
        log(f"dump: {len(base)} base + {len(novel)} novel, {len(train_items)} train, {len(test_items)} test")
        bank = encode(cm, train_items, tr, dev, a.workers, a.views, half=True)
        y_tr = torch.tensor([l for _, l, _ in train_items], device=dev)
        log(f"  train bank {tuple(bank.shape)}")
        model = train_prompts(a, cm, base_names, bank, y_tr, dev, a.seed)
        ft = encode(cm, test_items, te, dev, a.workers)[0]
        lg, zs = score(model, ft, names, dev)
        y = np.array([l for _, l, _ in test_items])
        os.makedirs(lout, exist_ok=True)
        np.savez_compressed(f"{lout}/logits.npz", logits=lg, zs_logits=zs, labels=y,
                            n_base=len(base), classnames=np.array(names))
        log(f"  mixed accuracy (raw argmax): {(lg.argmax(-1) == y).mean()*100:.2f} -> {lout}/logits.npz")
        del model, bank, ft; torch.cuda.empty_cache()

    # ---------------- calibration splits (as calibrate_mixed.py)
    if todo:
        base_set = set(base)
        cal_pool = [it for it in list(data["val"]) + list(data["train"]) if it[1] in base_set]
        uniq = list({p: (p, l, c) for p, l, c in cal_pool}.values())
        log(f"calibration: encoding {len(uniq)} base-class val/train images once")
        fu = encode(cm, uniq, te, dev, a.workers)[0]
        row = {p: i for i, (p, _, _) in enumerate(uniq)}
        for split in todo:
            s = split_seed(a.seed, split)
            random.seed(s); np.random.seed(s); torch.manual_seed(s)
            rng = random.Random(s); shuf = list(base); rng.shuffle(shuf)
            half_n = len(shuf) // 2
            b1, b2 = sorted(shuf[:half_n]), sorted(shuf[half_n:])
            order = b1 + b2
            names = [names_all[l] for l in order]; b1_names = [names_all[l] for l in b1]
            train_items = few_shot(subset(data["train"], b1), a.shots, s)
            used = {p for p, _, _ in train_items}
            order_set = set(order)
            pool = [it for it in data["train"] if it[1] in order_set and it[0] not in used]
            val_items = subset(list(data["val"]) + pool, order)
            log(f"split {split}: pseudo-base {len(b1)} / pseudo-novel {len(b2)}, "
                f"{len(train_items)} train, {len(val_items)} val")
            bank = encode(cm, train_items, tr, dev, a.workers, a.views, half=True)
            y_tr = torch.tensor([l for _, l, _ in train_items], device=dev)
            model = train_prompts(a, cm, b1_names, bank, y_tr, dev, s)
            fv = fu[torch.tensor([row[p] for p, _, _ in val_items], device=dev)]
            lg, zs = score(model, fv, names, dev)
            y = np.array([l for _, l, _ in val_items])
            path = f"{cout}/split{split}/calib_logits.npz"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path[:-4] + ".tmp.npz"
            np.savez(tmp, logits=lg, zs_logits=zs, labels=y, n_b1=len(b1),
                     b1=np.array(b1), b2=np.array(b2), seed=s)
            os.replace(tmp, path)
            log(f"  simulated-mixed val acc {(lg.argmax(-1) == y).mean()*100:.2f} -> {path}")
            del model, bank; torch.cuda.empty_cache()

    if a.no_calib:
        log("no calibration requested; done"); return
    alphas = [float(x) for x in a.alphas.split(",")]
    G, GAIN, A, S = [], [], [], []
    for s in splits:
        d = np.load(f"{cout}/split{s}/calib_logits.npz")
        g, gain, al = sweep(d["logits"], d["zs_logits"], d["labels"], d["logits"].shape[1], alphas, a.reps)
        G.append(g); GAIN.append(gain); A.append(al); S.append(np.full(len(g), s))
    np.savez(f"{cout}/points.npz", g=np.concatenate(G), gain=np.concatenate(GAIN),
             alpha=np.concatenate(A), split=np.concatenate(S))
    log(f"saved {sum(map(len, G))} calibration points -> {cout}/points.npz")


if __name__ == "__main__":
    main()
