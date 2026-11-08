"""Training and evaluation loops for the prompt learners."""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_prototypes(model, loader, num_classes, device):
    """Class means of the unprompted layer-k0 CLS on the few-shot base training set."""
    model.eval()
    feats, labels = [], []
    for imgs, labs in loader:
        _, cls = model.vision.early(imgs.to(device))
        feats.append(F.normalize(cls.float(), dim=-1).cpu())
        labels.append(labs)
    feats, labels = torch.cat(feats), torch.cat(labels)
    protos = torch.stack([feats[labels == c].mean(0) for c in range(num_classes)])
    protos = F.normalize(protos, dim=-1)
    s = (feats @ protos.t()).max(-1).values
    model.gate.set_prototypes(protos.to(device), s.mean().item(), s.std().item())
    return protos


def sample_episode(num_base, held_out_frac, rng):
    ids = list(range(num_base))
    rng.shuffle(ids)
    n_h = max(1, int(round(held_out_frac * num_base)))
    return sorted(ids[n_h:]), sorted(ids[:n_h])  # seen S, held-out H


def train_epoch(model, loader, opt, base_names, device, sched=None, episodic=True, held_out_frac=0.25,
                lam_h=1.0, rng=None, episode_every=0, log_every=50, amp=False, train_override=None):
    """Seen images: CE over seen classes, gradients to prompts and gate.
    Held-out images: CE over all base classes with prompts detached, gradients to the gate only.
    With episodic=False this is plain prompt learning with the gate trained on all images."""
    model.train()
    rng = rng or random.Random(0)
    nb = len(base_names)
    S, H = sample_episode(nb, held_out_frac, rng) if episodic else (list(range(nb)), [])
    stats = {"loss": 0.0, "g_seen": 0.0, "g_held": 0.0, "n": 0, "n_held": 0}
    for it, (imgs, labs) in enumerate(loader):
        if episodic and episode_every and it > 0 and it % episode_every == 0:
            S, H = sample_episode(nb, held_out_frac, rng)
        imgs, labs = imgs.to(device), labs.to(device)
        S_t = torch.tensor(S, device=device)
        s_index = torch.full((nb,), -1, dtype=torch.long, device=device)
        s_index[S_t] = torch.arange(len(S), device=device)
        is_seen = s_index[labs] >= 0
        loss = imgs.new_zeros(())
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            if is_seen.any():
                lg, g, _ = model(imgs[is_seen], [base_names[i] for i in S], proto_ids=S_t, gate_override=train_override)
                loss = loss + F.cross_entropy(lg.float(), s_index[labs[is_seen]])
                stats["g_seen"] += g.float().sum().item()
            if episodic and (~is_seen).any():
                lg, g, _ = model(imgs[~is_seen], base_names, proto_ids=S_t, detach_prompts=True)
                loss = loss + lam_h * F.cross_entropy(lg.float(), labs[~is_seen])
                stats["g_held"] += g.float().sum().item()
                stats["n_held"] += int((~is_seen).sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        stats["loss"] += loss.item() * imgs.shape[0]
        stats["n"] += imgs.shape[0]
        if log_every and it % log_every == 0:
            print(f"  it {it:4d}  loss {loss.item():.4f}")
    n, nh = max(stats["n"], 1), max(stats["n_held"], 1)
    n_seen = max(stats["n"] - stats["n_held"], 1)
    return {"loss": stats["loss"] / n, "g_seen": stats["g_seen"] / n_seen, "g_held": stats["g_held"] / nh}


@torch.no_grad()
def evaluate(model, loader, classnames, device, gate_override=None, amp=False):
    """Returns accuracy and per-sample arrays (correct, g, s1, pred, label)."""
    model.eval()
    out = {"correct": [], "g": [], "s1": [], "pred": [], "label": []}
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            lg, g, s1 = model(imgs, classnames, gate_override=gate_override)
        pred = lg.float().argmax(-1)
        out["correct"].append((pred == labs).cpu())
        out["g"].append(g.float().cpu())
        out["s1"].append(s1.float().cpu() if s1 is not None else torch.zeros(len(labs)))
        out["pred"].append(pred.cpu())
        out["label"].append(labs.cpu())
    arr = {k: torch.cat(v).numpy() for k, v in out.items()}
    return float(arr["correct"].mean() * 100), arr


def harmonic_mean(a, b):
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def build_optimizer(params, lr, epochs, steps_per_epoch, weight_decay=5e-4, warmup_epochs=1, warmup_lr=1e-5):
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    total = max(epochs * steps_per_epoch, 1)
    warm = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return warmup_lr / lr  # CoOp-style constant warmup
        t = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched
