"""학습/추론 엔진 — clip-level sliding-window max 추론 + 학습 루프.

plan.md §7: eval 은 sliding window 후 clip_score = max(window P(positive)).
plan.md §10: AdamW, CE, cosine, early stopping(val F1), seeds.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AudioPIIDataset
from metrics import classification_metrics


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_clip_probs(model, df, channels, device, num_workers=4, seed=0):
    """각 clip 의 sliding window P(positive) 를 max 집계 → clip 확률 배열(df 순서)."""
    model.eval()
    ds = AudioPIIDataset(df, mode="eval", channels=channels, seed=seed)
    loader = DataLoader(ds, batch_size=1, num_workers=num_workers,
                        collate_fn=lambda b: b[0])
    probs = np.zeros(len(df), dtype=np.float64)
    for xs, _label, idx in loader:
        xs = xs.to(device)                       # [n_win, C, H, W]
        p = torch.softmax(model(xs), dim=1)[:, 1]
        probs[idx] = float(p.max().item())       # clip = max over windows
    return probs


def train_model(model, tr_df, va_df, channels, device,
                epochs=50, lr=1e-4, weight_decay=1e-4, batch_size=32,
                patience=8, seed=42, augment=True, num_workers=4, smoke=False):
    model.to(device)
    ds = AudioPIIDataset(tr_df, mode="train", channels=channels, seed=seed, augment=augment)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss()

    best_f1, best_state, bad = -1.0, None, 0
    history = []
    n_epochs = 1 if smoke else epochs
    for ep in range(n_epochs):
        model.train()
        run_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)
        sched.step()

        vp = predict_clip_probs(model, va_df, channels, device, num_workers=num_workers)
        vf1 = classification_metrics(va_df.label.values, vp)["f1"]
        history.append({"epoch": ep, "train_loss": run_loss / len(ds), "val_f1": vf1})

        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_f1": best_f1, "history": history}


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))
