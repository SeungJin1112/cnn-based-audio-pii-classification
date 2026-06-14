"""1순위 — Precision/Calibration 분석.

v2 의 약점(고정 0.5 에서 hard-neg 과다검출, hard-neg FPR 0.25~0.54)을 공략:
  1) threshold-독립 성능(ROC/PR-AUC) — calibration 으로 안 바뀜(상한)
  2) calibration: temperature scaling / Platt scaling, ECE(before/after)
  3) 운용점 분석: target recall(0.99/0.95/0.90)을 val 에서 맞춘 threshold 를 test 에 적용 →
     test precision / hard-neg FPR / STT 호출률. 고정 0.5 와 비교.

→ "높은 recall 프리필터로 쓸 때 precision/FPR 이 실제로 어떤지, calibration 이 운용점 선택을 돕는지" 정량화.

usage: python calibration_analysis.py [--epochs 40] [--smoke]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C
from engine import get_device, predict_clip_probs, train_model
from metrics import classification_metrics
from models import build_model
from speaker_pii_experiment import build_qwen_manifest
from splits import split_manifest

RUNS = os.path.join(C.CNN_DIR, "runs")
EPS = 1e-6


def to_logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(val_logit, val_y, grid=np.linspace(0.3, 5.0, 95)):
    """val NLL 최소화하는 T (1D grid search)."""
    best_T, best_nll = 1.0, 1e18
    for T in grid:
        p = np.clip(sigmoid(val_logit / T), EPS, 1 - EPS)
        nll = -np.mean(val_y * np.log(p) + (1 - val_y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


def ece(p, y, n_bins=10):
    """Expected Calibration Error."""
    p = np.asarray(p, float); y = np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(e)


def thr_for_recall(y, p, target):
    """val 에서 recall>=target 을 만족하는 가장 높은 threshold(=precision 최대화)."""
    p = np.asarray(p, float); y = np.asarray(y)
    pos = p[y == 1]
    if len(pos) == 0:
        return 0.5
    # recall>=target 이려면 threshold <= (positive 확률의 (1-target) 분위수)
    q = np.quantile(pos, max(0.0, 1 - target))
    return float(min(max(q, EPS), 1 - EPS))


def op_point(y, p, src, thr):
    m = classification_metrics(y, p, thr)
    pred = (np.asarray(p) >= thr).astype(int)
    st = np.asarray(src)
    hn = (st == "negative_hard")
    fpr_hard = float(pred[hn].mean()) if hn.sum() else float("nan")
    return {"thr": round(float(thr), 4), "recall": m["recall"], "precision": m["precision"],
            "f1": m["f1"], "hard_neg_fpr": fpr_hard, "stt_call_rate": float(pred.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = get_device()
    m = build_qwen_manifest()
    tr, va, te, _ = split_manifest(m, seed=42)
    if args.smoke:
        tr = tr.groupby("label", group_keys=False).sample(n=60, random_state=42)

    model, ch = build_model("simple_cnn")
    print("[train] simple_cnn ...")
    train_model(model, tr, va, ch, device, epochs=args.epochs, seed=42, smoke=args.smoke)

    vy, sy = va.label.values, te.label.values
    vp = predict_clip_probs(model, va, ch, device, seed=42)
    tp = predict_clip_probs(model, te, ch, device, seed=42)

    # threshold-독립 상한
    roc = float(roc_auc_score(sy, tp)); pr = float(average_precision_score(sy, tp))

    # calibration
    vlog, tlog = to_logit(vp), to_logit(tp)
    T = fit_temperature(vlog, vy)
    tp_temp = sigmoid(tlog / T)
    platt = LogisticRegression(C=1e6, solver="lbfgs").fit(vlog.reshape(-1, 1), vy)
    tp_platt = platt.predict_proba(tlog.reshape(-1, 1))[:, 1]

    out = {
        "test_roc_auc": roc, "test_pr_auc": pr, "temperature": T,
        "ece": {"raw": ece(tp, sy), "temp": ece(tp_temp, sy), "platt": ece(tp_platt, sy)},
        "fixed_0.5": {
            "raw": op_point(sy, tp, te.source_type.values, 0.5),
            "temp": op_point(sy, tp_temp, te.source_type.values, 0.5),
        },
        "target_recall": {},
    }
    # 운용점: target recall 을 val(원확률)에서 맞춰 test 에 적용
    for tgt in [0.99, 0.95, 0.90]:
        thr = thr_for_recall(vy, vp, tgt)
        out["target_recall"][str(tgt)] = op_point(sy, tp, te.source_type.values, thr)

    os.makedirs(RUNS, exist_ok=True)
    sfx = "_smoke" if args.smoke else ""
    with open(os.path.join(RUNS, f"calibration{sfx}.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n== threshold-독립 상한 ==  ROC={roc:.3f}  PR-AUC={pr:.3f}")
    print(f"== ECE ==  raw={out['ece']['raw']:.4f}  temp(T={T:.2f})={out['ece']['temp']:.4f}  platt={out['ece']['platt']:.4f}")
    print(f"== 고정 0.5 (raw) ==  {out['fixed_0.5']['raw']}")
    print("== 운용점 (val 에서 target recall 맞춤 → test) ==")
    for tgt, op in out["target_recall"].items():
        print(f"  recall>={tgt}: recall={op['recall']:.3f} precision={op['precision']:.3f} "
              f"hard_neg_FPR={op['hard_neg_fpr']:.3f} STT호출률={op['stt_call_rate']:.3f} (thr={op['thr']})")
    print(f"\nsaved -> {os.path.join(RUNS, f'calibration{sfx}.json')}")


if __name__ == "__main__":
    main()
