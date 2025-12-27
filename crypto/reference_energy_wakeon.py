# -*- coding: utf-8 -*-
"""
energy_wakeon_snn_oxytocin_rl.py
1-file runnable on Colab/Jupyter/CLI.

What this does:
- Train a simple SNN (surrogate-gradient LIF) for AG News classification using hash features
- Measure approximate GPU energy via NVML power integration (nvidia-ml-py)
- Train/optional finetune DistilBERT classifier as System-2
- Wake-on-SNN Hybrid:
  - Fixed threshold OR auto threshold (quantile) OR auto threshold (accuracy-constrained)
- Optional RL gate policy to decide waking BERT (System-2) based on SNN uncertainty features
  - Uses REINFORCE with baseline
  - "Oxytocin" is just a name for reward shaping coefficient (not biology)

Notes:
- Fixes HuggingFace datasets indexing: converts to python lists early
- Fixes Jupyter injected args: parse_known_args
- Fixes inplace autograd error: SpikeFn stores u.clone()
- Fixes nonzero list-of-list bug: all wake probabilities and actions are forced 1D (N,)
"""

import os
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# NVML energy (nvidia-ml-py)
# ----------------------------
_NVML_OK = False
_pynvml = None

try:
    # recommended (replaces deprecated pynvml package name)
    import pynvml  # provided by nvidia-ml-py
    _pynvml = pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    requested = (requested or "").lower().strip()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def now_ms() -> float:
    return time.time() * 1000.0


class GpuEnergyMeter:
    """
    Approximates energy by sampling instantaneous GPU power via NVML once at end and
    multiplying by elapsed time (rough but consistent for A/B comparisons).
    Returns Joules. If NVML not available or device not CUDA -> returns None.
    """

    def __init__(self, device: torch.device, gpu_index: int = 0):
        self.device = device
        self.gpu_index = gpu_index
        self._ok = False
        self._handle = None
        self._t0 = None

        if self.device.type == "cuda" and _NVML_OK:
            try:
                _pynvml.nvmlInit()
                self._handle = _pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self._ok = True
            except Exception:
                self._ok = False

    def start(self):
        if not self._ok:
            self._t0 = None
            return
        self._t0 = time.time()
        # warm read
        _ = _pynvml.nvmlDeviceGetPowerUsage(self._handle)

    def stop_joules(self) -> Optional[float]:
        if not self._ok or self._t0 is None:
            return None
        t1 = time.time()
        p_mw = _pynvml.nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
        p_w = p_mw / 1000.0
        dt = max(0.0, t1 - self._t0)
        return p_w * dt


# ----------------------------
# Dataset
# ----------------------------
def load_ag_news_as_lists() -> Tuple[List[str], List[int], List[str], List[int], int]:
    """
    Loads AG News from HF datasets, returns python lists.
    """
    from datasets import load_dataset
    ds = load_dataset("ag_news")  # may be cached in colab
    train_texts = list(ds["train"]["text"])
    train_labels = list(ds["train"]["label"])
    test_texts = list(ds["test"]["text"])
    test_labels = list(ds["test"]["label"])
    return train_texts, train_labels, test_texts, test_labels, 4


# ----------------------------
# Hash vectorizer (cheap features)
# ----------------------------
@dataclass
class HashVectorizer:
    dim: int = 8192
    seed: int = 123
    token_cap: int = 256

    def featurize(self, texts: List[str], device: torch.device) -> torch.Tensor:
        x = torch.zeros((len(texts), self.dim), device=device, dtype=torch.float32)
        for i, t in enumerate(texts):
            toks = t.lower().split()
            for w in toks[: self.token_cap]:
                h = (hash((w, self.seed)) % self.dim)
                x[i, h] += 1.0
        x = x / (x.sum(dim=1, keepdim=True) + 1e-6)
        return x


# ----------------------------
# SNN: surrogate spikes
# ----------------------------
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: torch.Tensor, thr: float):
        out = (u >= thr).to(u.dtype)
        # IMPORTANT: clone to avoid inplace-version issues later
        ctx.save_for_backward(u.clone())
        ctx.thr = thr
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (u,) = ctx.saved_tensors
        thr = ctx.thr
        # surrogate gradient (triangular around threshold)
        x = (u - thr).abs()
        grad = (x < 1.0).to(u.dtype) * (1.0 - x)
        return grad_out * grad, None


def spike(u: torch.Tensor, thr: float) -> torch.Tensor:
    return SpikeFn.apply(u, thr)


class SNNClassifier(nn.Module):
    """
    Simple LIF-like recurrent-in-time SNN with surrogate gradients.
    """

    def __init__(self, in_dim: int, hidden: int, n_classes: int, steps: int = 20, beta: float = 0.9, thr: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, n_classes, bias=True)
        self.steps = steps
        self.beta = beta
        self.thr = thr

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        dev = x.device
        u1 = torch.zeros((B, self.fc1.out_features), device=dev)
        u2 = torch.zeros((B, self.fc2.out_features), device=dev)

        out_sum = torch.zeros((B, self.fc2.out_features), device=dev)
        spk_mean_accum = 0.0

        # compute input current once for speed
        i1 = self.fc1(x)

        for _ in range(self.steps):
            u1 = self.beta * u1 + i1
            s1 = spike(u1, self.thr)
            # reset without inplace ops
            u1 = u1 * (1.0 - s1)

            u2 = self.beta * u2 + self.fc2(s1)
            s2 = spike(u2, self.thr)
            u2 = u2 * (1.0 - s2)

            out_sum = out_sum + s2
            spk_mean_accum += float(s1.sum(dim=1).mean().detach().cpu().item())

        logits = out_sum / float(self.steps)
        mean_spikes = spk_mean_accum / float(self.steps)
        return logits, mean_spikes


@torch.no_grad()
def eval_snn(model: nn.Module, vec: HashVectorizer, texts: List[str], labels: List[int], device: torch.device, batch_size: int = 128):
    model.eval()
    correct = 0
    total = 0
    spikes = []
    for i in range(0, len(texts), batch_size):
        bt = texts[i:i + batch_size]
        by = torch.tensor(labels[i:i + batch_size], device=device)
        x = vec.featurize(bt, device)
        logits, mspk = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == by).sum().item()
        total += by.numel()
        spikes.append(mspk)
    return 100.0 * correct / max(1, total), float(np.mean(spikes))


def train_snn_one_epoch(model: nn.Module, vec: HashVectorizer, texts: List[str], labels: List[int],
                        device: torch.device, batch_size: int = 128, lr: float = 5e-4) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    idx = np.random.permutation(len(texts))
    total_loss = 0.0
    n_steps = 0

    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size].tolist()  # numpy.int64 -> python int
        bt = [texts[int(k)] for k in j]
        by = torch.tensor([labels[int(k)] for k in j], device=device)

        x = vec.featurize(bt, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, by)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += float(loss.detach().cpu().item())
        n_steps += 1

    return total_loss / max(1, n_steps)


# ----------------------------
# Confidence helpers for hybrid
# ----------------------------
@torch.no_grad()
def infer_snn_conf(model: nn.Module, vec: HashVectorizer, texts: List[str], labels: List[int],
                   device: torch.device, batch_size: int = 128):
    """
    Returns:
    - acc
    - conf (Tensor, N) : max softmax probability
    - pred (Tensor, N)
    - true (Tensor, N)
    - mean_spikes
    """
    model.eval()
    all_conf = []
    all_pred = []
    all_true = []
    spikes = []

    for i in range(0, len(texts), batch_size):
        bt = texts[i:i + batch_size]
        by = torch.tensor(labels[i:i + batch_size], device=device)
        x = vec.featurize(bt, device)
        logits, mspk = model(x)
        prob = F.softmax(logits, dim=1)
        conf = prob.max(dim=1).values
        pred = prob.argmax(dim=1)
        all_conf.append(conf.detach().cpu())
        all_pred.append(pred.detach().cpu())
        all_true.append(by.detach().cpu())
        spikes.append(mspk)

    conf = torch.cat(all_conf, dim=0).float()
    pred = torch.cat(all_pred, dim=0).long()
    true = torch.cat(all_true, dim=0).long()

    acc = 100.0 * (pred == true).float().mean().item()
    return acc, conf, pred, true, float(np.mean(spikes))


def print_conf_stats(conf: torch.Tensor):
    q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    qq = {float(k): float(torch.quantile(conf, torch.tensor(k)).item()) for k in q}
    print(f"[SNN-CONF] min={conf.min().item():.4f} mean={conf.mean().item():.4f} max={conf.max().item():.4f}")
    print(f"[SNN-CONF-QUANTILE] {qq}")


# ----------------------------
# DistilBERT baseline / finetune
# ----------------------------
@torch.no_grad()
def infer_distilbert_pred(model, tokenizer, texts: List[str], device: torch.device, batch_size: int = 16, max_len: int = 128):
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        bt = texts[i:i + batch_size]
        enc = tokenizer(bt, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).logits
        preds.append(out.argmax(dim=1).detach().cpu())
    return torch.cat(preds, dim=0).long()


@torch.no_grad()
def eval_distilbert(model, tokenizer, texts: List[str], labels: List[int], device: torch.device, batch_size: int = 16, max_len: int = 128):
    pred = infer_distilbert_pred(model, tokenizer, texts, device, batch_size=batch_size, max_len=max_len)
    true = torch.tensor(labels, dtype=torch.long)
    acc = 100.0 * (pred == true).float().mean().item()
    return acc


def finetune_distilbert_steps(model, tokenizer, texts: List[str], labels: List[int], device: torch.device,
                              steps: int = 200, batch_size: int = 16, max_len: int = 128, lr: float = 5e-5, log_every: int = 160):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(texts)
    last_loss = None

    for s in range(1, steps + 1):
        # random minibatch
        j = np.random.randint(0, n, size=(batch_size,))
        bt = [texts[int(k)] for k in j.tolist()]
        by = torch.tensor([labels[int(k)] for k in j.tolist()], device=device)

        enc = tokenizer(bt, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc).logits
        loss = F.cross_entropy(out, by)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last_loss = float(loss.detach().cpu().item())
        if (s % log_every) == 0 or s == steps:
            print(f"[BERT-train] steps={s}/{steps} loss={last_loss:.4f}")

    return last_loss


# ----------------------------
# Auto-threshold selection (hybrid)
# ----------------------------
def thr_by_wake_quantile(conf_calib: torch.Tensor, wake_q: float) -> float:
    """
    Choose threshold so that wake_rate ~= wake_q, i.e. wake if conf < thr.
    So thr is quantile at wake_q.
    """
    wake_q = float(wake_q)
    wake_q = min(max(wake_q, 0.0), 1.0)
    return float(torch.quantile(conf_calib, torch.tensor(wake_q)).item())


@torch.no_grad()
def thr_by_acc_constraint(conf_calib: torch.Tensor,
                          pred_snn_calib: torch.Tensor,
                          true_calib: torch.Tensor,
                          bert_pred_calib: torch.Tensor,
                          target_acc: float,
                          grid: int = 200) -> float:
    """
    Choose threshold that meets target accuracy with minimum wake rate.
    We simulate hybrid on calib set:
      final = snn_pred; overwrite where conf<thr with bert_pred
    """
    target_acc = float(target_acc)
    # thr candidates across observed range
    lo = float(conf_calib.min().item())
    hi = float(conf_calib.max().item())
    if hi <= lo + 1e-9:
        return hi

    best_thr = hi
    best_wake = 1e9

    # evaluate thresholds from low -> high (low wakes few, high wakes many)
    for k in range(grid):
        thr = lo + (hi - lo) * (k / max(1, grid - 1))
        wake_mask = (conf_calib < thr)
        final = pred_snn_calib.clone()
        final[wake_mask] = bert_pred_calib[wake_mask]
        acc = 100.0 * (final == true_calib).float().mean().item()
        wake = 100.0 * wake_mask.float().mean().item()

        if acc >= target_acc and wake < best_wake:
            best_wake = wake
            best_thr = thr

    return float(best_thr)


# ----------------------------
# RL Gate Policy (Wake decision)
# ----------------------------
class GatePolicy(nn.Module):
    """
    Outputs p(wake) in (0,1).
    Input features per sample: [conf, margin, entropy] (or fewer depending on config)
    """

    def __init__(self, in_dim: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # returns (N,1)
        return self.net(feat)


@torch.no_grad()
def snn_features_for_gate(model: nn.Module, vec: HashVectorizer, texts: List[str], labels: List[int],
                          device: torch.device, batch_size: int = 128):
    """
    Produce gate features and also SNN predictions.

    Returns:
    - feat (N,3) on CPU float32: [conf, margin, entropy]
    - snn_pred (N,) CPU long
    - true (N,) CPU long
    - conf (N,) CPU float
    """
    model.eval()
    feats = []
    preds = []
    trues = []
    confs = []

    for i in range(0, len(texts), batch_size):
        bt = texts[i:i + batch_size]
        by = torch.tensor(labels[i:i + batch_size], device=device)
        x = vec.featurize(bt, device)
        logits, _ = model(x)
        prob = F.softmax(logits, dim=1)  # (B,C)

        # conf
        top2 = torch.topk(prob, k=2, dim=1).values  # (B,2)
        conf = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(prob * (prob.clamp_min(1e-9).log())).sum(dim=1)

        pred = prob.argmax(dim=1)

        feat = torch.stack([conf, margin, entropy], dim=1)

        feats.append(feat.detach().cpu())
        preds.append(pred.detach().cpu())
        trues.append(by.detach().cpu())
        confs.append(conf.detach().cpu())

    feat = torch.cat(feats, dim=0).float()
    snn_pred = torch.cat(preds, dim=0).long()
    true = torch.cat(trues, dim=0).long()
    conf = torch.cat(confs, dim=0).float()
    return feat, snn_pred, true, conf


def rl_train_policy(policy: GatePolicy,
                    feat_calib: torch.Tensor,
                    snn_pred_calib: torch.Tensor,
                    true_calib: torch.Tensor,
                    bert_pred_calib: torch.Tensor,
                    device: torch.device,
                    epochs: int = 8,
                    batch_size: int = 256,
                    lr: float = 1e-3,
                    # reward shaping
                    lambda_energy: float = 0.20,
                    oxytocin_bonus: float = 0.05,
                    baseline_momentum: float = 0.90,
                    decision: str = "sample",
                    p_thr: float = 0.5):
    """
    REINFORCE with baseline.
    Reward per sample:
      +1 if final prediction correct
      - lambda_energy if wake (BERT called)
      + oxytocin_bonus if (correct AND did NOT wake)  -> encourages "calm confidence" (System-1 correct)
    This "oxytocin" is a metaphorical coefficient; not biology.

    decision:
      - "sample": wake ~ Bernoulli(p)
      - "threshold": wake = p >= p_thr (still trained using REINFORCE on sampled actions internally)
    """
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    # move to device for training
    feat = feat_calib.to(device)
    snn_pred = snn_pred_calib.to(device)
    true = true_calib.to(device)
    bert_pred = bert_pred_calib.to(device)

    n = feat.size(0)
    baseline = 0.0

    for ep in range(1, epochs + 1):
        # shuffle
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        ep_reward = 0.0
        ep_wake = 0.0
        steps = 0

        for i in range(0, n, batch_size):
            j = perm[i:i + batch_size]
            f = feat[j]
            y_true = true[j]
            y_snn = snn_pred[j]
            y_bert = bert_pred[j]

            p_wake = policy(f).clamp(1e-6, 1.0 - 1e-6)  # (B,1)
            p_wake = p_wake.squeeze(-1)  # IMPORTANT: (B,)

            # sample action for gradient
            a = torch.bernoulli(p_wake)  # (B,) in {0,1}
            wake = a

            # final prediction
            y_final = y_snn.clone()
            wake_mask = (wake > 0.5)
            y_final[wake_mask] = y_bert[wake_mask]

            correct = (y_final == y_true).float()

            # reward
            r = correct
            r = r - (lambda_energy * wake)
            r = r + (oxytocin_bonus * (correct * (1.0 - wake)))  # "oxytocin-like" calm reward

            # baseline update
            r_mean = float(r.mean().detach().cpu().item())
            baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * r_mean

            adv = r - baseline  # (B,)

            # REINFORCE loss: -E[ adv * log pi(a|p) ]
            logp = wake * torch.log(p_wake) + (1.0 - wake) * torch.log(1.0 - p_wake)
            loss = -(adv.detach() * logp).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            ep_loss += float(loss.detach().cpu().item())
            ep_reward += float(r_mean)
            ep_wake += float(wake.float().mean().detach().cpu().item())
            steps += 1

        print(f"[RL] ep {ep}/{epochs} loss={ep_loss/max(1,steps):.4f} mean_reward={ep_reward/max(1,steps):.4f} wake_rate={100.0*ep_wake/max(1,steps):.2f}%")

    policy.eval()
    return policy


@torch.no_grad()
def rl_eval_policy(policy: GatePolicy,
                   feat_eval: torch.Tensor,
                   snn_pred_eval: torch.Tensor,
                   true_eval: torch.Tensor,
                   bert_pred_eval: torch.Tensor,
                   device: torch.device,
                   decision: str = "threshold",
                   p_thr: float = 0.5):
    """
    Returns:
      - acc (%)
      - wake_rate (%)
      - p_wake (N,) CPU float
      - wake_mask (N,) CPU bool
    """
    policy.eval()
    feat = feat_eval.to(device)
    snn_pred = snn_pred_eval.to(device)
    true = true_eval.to(device)
    bert_pred = bert_pred_eval.to(device)

    p_wake = policy(feat).clamp(1e-6, 1.0 - 1e-6)  # (N,1)
    p_wake = p_wake.squeeze(-1)  # IMPORTANT -> (N,)

    if decision == "sample":
        wake = torch.bernoulli(p_wake).bool()
    else:
        wake = (p_wake >= float(p_thr))

    y_final = snn_pred.clone()
    y_final[wake] = bert_pred[wake]

    acc = 100.0 * (y_final == true).float().mean().item()
    wake_rate = 100.0 * wake.float().mean().item()

    return acc, wake_rate, p_wake.detach().cpu(), wake.detach().cpu()


# ----------------------------
# Main
# ----------------------------
def build_argparser():
    p = argparse.ArgumentParser(add_help=True)

    # core
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=str, default="ag_news")
    p.add_argument("--limit_train", type=int, default=20000)
    p.add_argument("--limit_test", type=int, default=2000)

    # SNN
    p.add_argument("--vec_dim", type=int, default=8192)
    p.add_argument("--snn_hidden", type=int, default=512)
    p.add_argument("--snn_steps", type=int, default=20)
    p.add_argument("--snn_epochs", type=int, default=5)
    p.add_argument("--snn_lr", type=float, default=5e-4)
    p.add_argument("--snn_batch", type=int, default=128)

    # BERT
    p.add_argument("--run_bert", action="store_true")
    p.add_argument("--bert_train", action="store_true")
    p.add_argument("--bert_train_steps", type=int, default=200)
    p.add_argument("--bert_batch", type=int, default=16)
    p.add_argument("--bert_max_len", type=int, default=128)
    p.add_argument("--bert_lr", type=float, default=5e-5)

    # Hybrid threshold
    p.add_argument("--run_hybrid", action="store_true")
    p.add_argument("--thr_mode", type=str, default="fixed",
                   choices=["fixed", "auto_wake_q", "auto_acc_constrained"])
    p.add_argument("--wake_thr", type=float, default=0.34, help="used when thr_mode=fixed")
    p.add_argument("--wake_q", type=float, default=0.30, help="used when thr_mode=auto_wake_q (wake fraction)")
    p.add_argument("--target_acc", type=float, default=88.7, help="used when thr_mode=auto_acc_constrained")
    p.add_argument("--calib_n", type=int, default=500, help="calibration split size from test set")

    # RL Gate
    p.add_argument("--run_rl", action="store_true")
    p.add_argument("--rl_epochs", type=int, default=8)
    p.add_argument("--rl_hidden", type=int, default=32)
    p.add_argument("--rl_lr", type=float, default=1e-3)
    p.add_argument("--rl_lambda_energy", type=float, default=0.20)
    p.add_argument("--rl_oxytocin_bonus", type=float, default=0.05)
    p.add_argument("--rl_decision", type=str, default="threshold", choices=["threshold", "sample"])
    p.add_argument("--rl_p_thr", type=float, default=0.5)

    # save
    p.add_argument("--save_json", type=str, default="results.json")

    return p


def main():
    # ignore Jupyter injected args like -f /root/.local/share/jupyter/runtime/kernel-xxx.json
    args, _unknown = build_argparser().parse_known_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"🚀 device={device}")

    # dataset
    train_texts, train_labels, test_texts, test_labels, n_classes = load_ag_news_as_lists()

    train_texts = train_texts[: args.limit_train]
    train_labels = train_labels[: args.limit_train]
    test_texts = test_texts[: args.limit_test]
    test_labels = test_labels[: args.limit_test]

    # SNN
    vec = HashVectorizer(dim=args.vec_dim, seed=args.seed)
    snn = SNNClassifier(in_dim=args.vec_dim, hidden=args.snn_hidden, n_classes=n_classes,
                        steps=args.snn_steps, beta=0.9, thr=1.0).to(device)

    for ep in range(args.snn_epochs):
        t0 = now_ms()
        loss = train_snn_one_epoch(snn, vec, train_texts, train_labels, device,
                                   batch_size=args.snn_batch, lr=args.snn_lr)
        # quick "train subset acc" to track learning
        tr_acc, tr_spk = eval_snn(snn, vec, train_texts[:2000], train_labels[:2000], device, batch_size=args.snn_batch)
        print(f"[SNN] ep {ep+1}/{args.snn_epochs} loss={loss:.4f} train_subset_acc={tr_acc:.2f}% mean_spikes={tr_spk:.2f} time={(now_ms()-t0)/1000:.1f}s")

    # SNN confidence + energy on full test
    acc_snn, conf_all, pred_all, true_all, spk = infer_snn_conf(snn, vec, test_texts, test_labels, device, batch_size=args.snn_batch)
    print(f"[SNN-CONF] test_acc={acc_snn:.2f}% mean_spikes={spk:.2f}")
    print_conf_stats(conf_all)

    meter_snn = GpuEnergyMeter(device=device, gpu_index=0)
    meter_snn.start()
    t0 = now_ms()
    acc_snn2, _, _, _, _ = infer_snn_conf(snn, vec, test_texts, test_labels, device, batch_size=args.snn_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_snn = (now_ms() - t0) / 1000.0
    j_snn = meter_snn.stop_joules()
    print(f"[SNN] infer_acc={acc_snn2:.2f}% time={dt_snn:.3f}s energy_j={j_snn}")
    if j_snn is not None:
        print(f"[SNN] J/sample={(j_snn/len(test_texts)):.6f}")

    # BERT (System-2)
    bert = None
    tok = None
    j_bert = None
    dt_bert = None
    acc_bert = None

    if args.run_bert:
        from transformers import AutoTokenizer, DistilBertForSequenceClassification

        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=n_classes).to(device)

        if args.bert_train:
            tbt = now_ms()
            last_loss = finetune_distilbert_steps(
                bert, tok, train_texts, train_labels, device,
                steps=args.bert_train_steps,
                batch_size=args.bert_batch,
                max_len=args.bert_max_len,
                lr=args.bert_lr
            )
            print(f"[BERT-train] steps={args.bert_train_steps} last_loss={last_loss:.4f} time={(now_ms()-tbt)/1000:.1f}s")

        meter_bert = GpuEnergyMeter(device=device, gpu_index=0)
        meter_bert.start()
        t1 = now_ms()
        acc_bert = eval_distilbert(bert, tok, test_texts, test_labels, device,
                                  batch_size=args.bert_batch, max_len=args.bert_max_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_bert = (now_ms() - t1) / 1000.0
        j_bert = meter_bert.stop_joules()

        print(f"[BERT] infer_acc={acc_bert:.2f}% time={dt_bert:.3f}s energy_j={j_bert}")
        if j_bert is not None:
            print(f"[BERT] J/sample={(j_bert/len(test_texts)):.6f}")
        if (j_snn is not None) and (j_bert is not None) and (j_snn > 0):
            print(f"[RATIO] BERT/SNN energy ≈ {j_bert/j_snn:.2f}x")

    # Hybrid threshold (System-1 gate)
    hybrid_results = {}
    if args.run_hybrid:
        if bert is None or tok is None:
            raise RuntimeError("--run_hybrid requires --run_bert (BERT model/tokenizer needed).")

        print("========== Wake-on-SNN Hybrid ==========")

        calib_n = int(args.calib_n)
        calib_n = max(1, min(calib_n, len(test_texts) // 2))

        # split test into calibration + evaluation
        calib_texts = test_texts[:calib_n]
        calib_labels = test_labels[:calib_n]
        eval_texts = test_texts[calib_n:]
        eval_labels = test_labels[calib_n:]

        # SNN features/preds on calib+eval
        _, conf_calib, pred_calib, true_calib, _ = infer_snn_conf(snn, vec, calib_texts, calib_labels, device, batch_size=args.snn_batch)
        _, conf_eval, pred_eval, true_eval, _ = infer_snn_conf(snn, vec, eval_texts, eval_labels, device, batch_size=args.snn_batch)

        # BERT preds on calib (needed for acc-constrained auto)
        bert_pred_calib = infer_distilbert_pred(bert, tok, calib_texts, device,
                                               batch_size=args.bert_batch, max_len=args.bert_max_len)

        # choose threshold
        if args.thr_mode == "fixed":
            thr = float(args.wake_thr)
            thr_mode_str = f"fixed({thr:.4f})"
        elif args.thr_mode == "auto_wake_q":
            thr = thr_by_wake_quantile(conf_calib, wake_q=float(args.wake_q))
            thr_mode_str = f"auto(wake_q={float(args.wake_q):.3f})"
        else:
            thr = thr_by_acc_constraint(conf_calib, pred_calib, true_calib, bert_pred_calib,
                                        target_acc=float(args.target_acc), grid=300)
            thr_mode_str = f"auto-acc_constrained(target>={float(args.target_acc):.2f}%)"

        print(f"[HYBRID] thr_mode={thr_mode_str}  thr={thr:.4f}  calib_n={calib_n}  eval_n={len(eval_texts)}")

        # measure SNN pass cost on eval (real cost of System-1)
        meter_h_snn = GpuEnergyMeter(device=device, gpu_index=0)
        meter_h_snn.start()
        t0 = now_ms()
        _, conf_eval2, pred_eval2, true_eval2, _ = infer_snn_conf(snn, vec, eval_texts, eval_labels, device, batch_size=args.snn_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        time_snn = (now_ms() - t0) / 1000.0
        energy_snn = meter_h_snn.stop_joules()

        wake_mask = (conf_eval2 < float(thr))  # CPU tensor
        wake_rate = 100.0 * wake_mask.float().mean().item()
        wake_idx = torch.nonzero(wake_mask).squeeze(1).cpu().tolist()

        print(f"[HYBRID] wake_rate={wake_rate:.2f}%  woke={len(wake_idx)}/{len(conf_eval2)}")

        # measure BERT only on woke samples
        meter_h_bert = GpuEnergyMeter(device=device, gpu_index=0)
        meter_h_bert.start()
        t1 = now_ms()
        if len(wake_idx) > 0:
            woke_texts = [eval_texts[int(i)] for i in wake_idx]
            bert_pred_woke = infer_distilbert_pred(bert, tok, woke_texts, device,
                                                  batch_size=args.bert_batch, max_len=args.bert_max_len)
        else:
            bert_pred_woke = torch.empty((0,), dtype=torch.long)

        if device.type == "cuda":
            torch.cuda.synchronize()
        time_bert = (now_ms() - t1) / 1000.0
        energy_bert = meter_h_bert.stop_joules()

        final_pred = pred_eval2.clone()
        if len(wake_idx) > 0:
            final_pred[wake_idx] = bert_pred_woke

        hybrid_acc = 100.0 * (final_pred == true_eval2).float().mean().item()

        print(f"[HYBRID] acc={hybrid_acc:.2f}%")
        print(f"[HYBRID] time_snn={time_snn:.3f}s  time_bert={time_bert:.3f}s  total={(time_snn+time_bert):.3f}s")
        print(f"[HYBRID] energy_snn_j={energy_snn}  energy_bert_j={energy_bert}")

        total_j = None
        if (energy_snn is not None) and (energy_bert is not None):
            total_j = energy_snn + energy_bert
            print(f"[HYBRID] total_j={total_j}  J/sample={(total_j/len(eval_texts)):.6f}")
            if j_bert is not None and total_j > 0:
                print(f"[SAVING] BERT(always)/HYBRID energy ≈ {j_bert/total_j:.2f}x")

        hybrid_results = {
            "thr_mode": args.thr_mode,
            "thr": float(thr),
            "calib_n": int(calib_n),
            "eval_n": int(len(eval_texts)),
            "wake_rate_pct": float(wake_rate),
            "acc_pct": float(hybrid_acc),
            "time_snn_s": float(time_snn),
            "time_bert_s": float(time_bert),
            "energy_snn_j": None if energy_snn is None else float(energy_snn),
            "energy_bert_j": None if energy_bert is None else float(energy_bert),
            "total_j": None if total_j is None else float(total_j),
        }

    # RL gate (learn wake policy)
    rl_results = {}
    if args.run_rl:
        if bert is None or tok is None:
            raise RuntimeError("--run_rl requires --run_bert (BERT model/tokenizer needed).")

        print("========== Wake-on-SNN Hybrid (RL: Oxytocin-modulated) ==========")

        calib_n = int(args.calib_n)
        calib_n = max(1, min(calib_n, len(test_texts) // 2))

        calib_texts = test_texts[:calib_n]
        calib_labels = test_labels[:calib_n]
        eval_texts = test_texts[calib_n:]
        eval_labels = test_labels[calib_n:]

        # Prepare gate features and SNN preds
        feat_calib, snn_pred_calib, true_calib, _conf_calib = snn_features_for_gate(
            snn, vec, calib_texts, calib_labels, device, batch_size=args.snn_batch
        )
        feat_eval, snn_pred_eval, true_eval, _conf_eval = snn_features_for_gate(
            snn, vec, eval_texts, eval_labels, device, batch_size=args.snn_batch
        )

        # BERT preds on calib/eval (needed for reward + evaluation)
        bert_pred_calib = infer_distilbert_pred(bert, tok, calib_texts, device,
                                               batch_size=args.bert_batch, max_len=args.bert_max_len)
        bert_pred_eval = infer_distilbert_pred(bert, tok, eval_texts, device,
                                              batch_size=args.bert_batch, max_len=args.bert_max_len)

        # Train policy
        policy = GatePolicy(in_dim=3, hidden=int(args.rl_hidden)).to(device)
        policy = rl_train_policy(
            policy,
            feat_calib, snn_pred_calib, true_calib, bert_pred_calib,
            device=device,
            epochs=int(args.rl_epochs),
            batch_size=256,
            lr=float(args.rl_lr),
            lambda_energy=float(args.rl_lambda_energy),
            oxytocin_bonus=float(args.rl_oxytocin_bonus),
            decision="sample",     # train with sampling for gradient
            p_thr=float(args.rl_p_thr),
        )

        # Evaluate policy with chosen decision mode
        acc_rl, wake_rate_rl, p_wake_eval, wake_mask_eval = rl_eval_policy(
            policy,
            feat_eval, snn_pred_eval, true_eval, bert_pred_eval,
            device=device,
            decision=str(args.rl_decision),
            p_thr=float(args.rl_p_thr),
        )

        print("[DBG] p_wake stats:", float(p_wake_eval.min().item()), float(p_wake_eval.mean().item()), float(p_wake_eval.max().item()))
        print(f"[HYBRID-RL] acc={acc_rl:.2f}% wake_rate={wake_rate_rl:.2f}%")

        # Estimate energy/sample using measured per-sample costs (rough):
        # We reuse earlier measured test energy if available; otherwise just report wake_rate.
        # If you want exact, you can wrap energy meters around SNN pass and BERT(woke) pass like hybrid.
        est_j_per_sample = None
        if (j_snn is not None) and (j_bert is not None):
            # approximate: always pay SNN, plus wake_rate fraction of BERT
            wake_frac = wake_rate_rl / 100.0
            # scale by relative set sizes (we assume similar)
            est_total_j = (j_snn * (len(eval_texts)/len(test_texts))) + (j_bert * (len(eval_texts)/len(test_texts)) * wake_frac)
            est_j_per_sample = est_total_j / max(1, len(eval_texts))
            print(f"[HYBRID-RL] est J/sample={est_j_per_sample:.6f}")
            if j_bert is not None and est_total_j > 0:
                print(f"[SAVING-RL] BERT(always)/HYBRID ≈ {j_bert/est_total_j:.2f}x")

        rl_results = {
            "acc_pct": float(acc_rl),
            "wake_rate_pct": float(wake_rate_rl),
            "p_wake_min": float(p_wake_eval.min().item()),
            "p_wake_mean": float(p_wake_eval.mean().item()),
            "p_wake_max": float(p_wake_eval.max().item()),
            "est_j_per_sample": None if est_j_per_sample is None else float(est_j_per_sample),
            "rl_lambda_energy": float(args.rl_lambda_energy),
            "rl_oxytocin_bonus": float(args.rl_oxytocin_bonus),
            "rl_decision": str(args.rl_decision),
            "rl_p_thr": float(args.rl_p_thr),
        }

    # Save results JSON
    out = {
        "device": str(device),
        "snn": {
            "acc_pct": float(acc_snn),
            "energy_j": None if j_snn is None else float(j_snn),
            "j_per_sample": None if j_snn is None else float(j_snn / len(test_texts)),
        },
        "bert": {
            "acc_pct": None if acc_bert is None else float(acc_bert),
            "energy_j": None if j_bert is None else float(j_bert),
            "j_per_sample": None if j_bert is None else float(j_bert / len(test_texts)),
        },
        "hybrid": hybrid_results,
        "rl": rl_results,
        "args": vars(args),
        "nvml_ok": bool(_NVML_OK),
    }

    try:
        import json
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {args.save_json}")
    except Exception as e:
        print("[WARN] failed to save json:", e)


if __name__ == "__main__":
    main()
