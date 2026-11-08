"""Train MaPLe on base classes and dump mixed-setting test logits for offline analysis.

  python dump_logits.py --root ~/data_compact --dataset eurosat --out runs/logits/eurosat

Saves logits.npz with: logits (N,C) from the adapted model, zs_logits (N,C) from zero-shot CLIP,
labels (N,), n_base. Classes are ordered base-first, then novel, so label < n_base means base.
"""
import argparse, os, random
import numpy as np, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import clip

from trustprompt.datasets import (load_dataset, few_shot, classnames_of, base_novel_split, subset,
                                  build_transforms, ImageList)
from trustprompt.learners import build_learner, train_loss, fill_defaults, end_epoch
from trustprompt.engine import build_optimizer

p = argparse.ArgumentParser()
p.add_argument("--root", required=True); p.add_argument("--dataset", required=True)
p.add_argument("--out", required=True); p.add_argument("--backbone", default="ViT-B/16")
p.add_argument("--shots", type=int, default=16); p.add_argument("--seed", type=int, default=1)
p.add_argument("--learner", default="maple", choices=["maple", "ivlp", "coop", "kgcoop", "promptsrc", "cocoop"])
p.add_argument("--kg_lambda", type=float, default=None)
p.add_argument("--amp", action="store_true", help="mixed-precision training (official PromptSRC/CoCoOp use fp16)")
p.add_argument("--depth", type=int, default=None); p.add_argument("--n_ctx", type=int, default=None)
p.add_argument("--epochs", type=int, default=None); p.add_argument("--bs", type=int, default=None)
p.add_argument("--lr", type=float, default=None); p.add_argument("--workers", type=int, default=4)
p.add_argument("--dump_val", action="store_true", help="also dump base-class validation logits (for calibration)")
a = fill_defaults(p.parse_args())
print(f"learner {a.learner}: depth {a.depth} n_ctx {a.n_ctx} epochs {a.epochs} bs {a.bs} lr {a.lr}", flush=True)

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
torch.backends.cuda.enable_mem_efficient_sdp(False)
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"): torch.backends.cuda.enable_cudnn_sdp(False)
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_dataset(a.root, a.dataset); names_all = classnames_of(data["test"])
base, novel = base_novel_split(data)
order = base + novel                                    # base first, then novel
names = [names_all[l] for l in order]
base_names = [names_all[l] for l in base]
train_items = few_shot(subset(data["train"], base), a.shots, a.seed)
test_items = subset(data["test"], order)                # relabelled 0..C-1 in this order
print(f"{a.dataset}: {len(base)} base + {len(novel)} novel, {len(test_items)} test images")

cm, _ = clip.load(a.backbone, device="cpu", jit=False)
model = build_learner(a.learner, cm.float(), a.dataset, a.depth, a.n_ctx,
                      a.kg_lambda if a.kg_lambda is not None else 8.0).to(dev)
tr, te = build_transforms(True), build_transforms(False)
mk = lambda it, tf, sh, bs: DataLoader(ImageList(it, tf), batch_size=bs, shuffle=sh,
                                       num_workers=a.workers, pin_memory=True)
loader = mk(train_items, tr, True, a.bs)
opt, sched = build_optimizer(model.trainable_parameters(), a.lr, a.epochs, len(loader))
scaler = torch.cuda.amp.GradScaler(enabled=a.amp)
for ep in range(a.epochs):
    model.train()
    for imgs, labs in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
            loss = train_loss(model, imgs.to(dev), labs.to(dev), base_names)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    end_epoch(model, ep, a.epochs)
    print(f"epoch {ep+1}/{a.epochs} loss {loss.item():.4f}", flush=True)

model.eval(); L, Z, Y = [], [], []
with torch.no_grad():
    for imgs, labs in mk(test_items, te, False, 128):
        imgs = imgs.to(dev)
        L.append(model(imgs, names).float().cpu()); Z.append(model.zero_shot(imgs, names).cpu())
        Y.append(labs)
np.savez_compressed(os.path.join(a.out, "logits.npz"),
                    logits=torch.cat(L).numpy(), zs_logits=torch.cat(Z).numpy(),
                    labels=torch.cat(Y).numpy(), n_base=len(base), classnames=np.array(names))
if a.dump_val:
    # base-class validation images, scored over the base label set only: used to calibrate g0
    val_items = subset(data["val"], base)
    VL, VZ, VY = [], [], []
    with torch.no_grad():
        for imgs, labs in mk(val_items, te, False, 128):
            imgs = imgs.to(dev)
            VL.append(model(imgs, base_names).float().cpu()); VZ.append(model.zero_shot(imgs, base_names).cpu())
            VY.append(labs)
    np.savez_compressed(os.path.join(a.out, "val_logits.npz"),
                        logits=torch.cat(VL).numpy(), zs_logits=torch.cat(VZ).numpy(),
                        labels=torch.cat(VY).numpy(), n_base=len(base))
    print(f"val logits: {len(val_items)} images -> {a.out}/val_logits.npz")

acc = (torch.cat(L).argmax(-1) == torch.cat(Y)).float().mean() * 100
print(f"mixed accuracy (raw argmax): {acc:.2f}  -> saved {a.out}/logits.npz")
