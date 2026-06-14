"""Phase 4 — Qwen 데이터(14화자 · 4 PII유형) 학습/평가 드라이버.

재사용: dataset.AudioPIIDataset, engine(train/predict), models.build_model, metrics, splits.split_manifest.

모드:
  --indist   : template-disjoint split 으로 in-distribution 성능(+pii유형별 recall, hard-neg FPR)
  --loso     : 14화자 leave-one-speaker-out. 고정 0.5 vs val-튜닝 threshold 비교(=v1 calibration 약점 공략)
  --heldpii  : held-out PII 유형 일반화(예: rrn 제외 학습 → rrn 평가). "숫자암기" 아닌 "PII형식 일반화" 검증
  --smoke    : 소량/1에폭
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch

import config as C
from engine import get_device, predict_clip_probs, train_model
from metrics import classification_metrics, hard_neg_fpr
from models import build_model
from splits import split_manifest

DATA_QWEN = os.path.join(C.CNN_DIR, "data_qwen")
RUNS = os.path.join(C.CNN_DIR, "runs")


# ---------------------------------------------------------------- manifest
def build_qwen_manifest():
    parts = []
    for cls in ["positive", "negative_hard", "negative_easy"]:
        md = os.path.join(DATA_QWEN, cls, "metadata.csv")
        df = pd.read_csv(md)
        df["filepath"] = [os.path.normpath(os.path.join(DATA_QWEN, cls, fp.replace("\\", "/")))
                          for fp in df["filepath"]]
        if "pii_type" not in df:
            df["pii_type"] = ""
        df["pii_type"] = df["pii_type"].fillna("")
        parts.append(df[["filepath", "label", "source_type", "speaker_id",
                         "template_id", "duration", "pii_type"]])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- helpers
def best_threshold(y, p):
    """val 에서 F1 최대 threshold."""
    ts = np.linspace(0.05, 0.95, 19)
    f1s = [classification_metrics(y, p, t)["f1"] for t in ts]
    return float(ts[int(np.argmax(f1s))])


def per_pii_recall(df, probs, thr=0.5):
    out = {}
    pred = (np.asarray(probs) >= thr).astype(int)
    for pt in ["phone", "rrn", "account", "card"]:
        m = (df.pii_type.values == pt) & (df.label.values == 1)
        if m.sum():
            out[pt] = float(pred[m].mean())
    return out


def train_eval(model_name, tr, va, te, device, epochs, smoke, seed):
    model, ch = build_model(model_name)
    train_model(model, tr, va, ch, device, epochs=epochs, seed=seed, smoke=smoke)
    vp = predict_clip_probs(model, va, ch, device, seed=seed)
    tp = predict_clip_probs(model, te, ch, device, seed=seed)
    return model, ch, vp, tp


# ---------------------------------------------------------------- modes
def run_indist(m, device, epochs, smoke, seed=42):
    tr, va, te, _ = split_manifest(m, seed=seed)
    _, _, vp, tp = train_eval("simple_cnn", tr, va, te, device, epochs, smoke, seed)
    y = te.label.values
    res = classification_metrics(y, tp, 0.5)
    res["hard_neg_fpr@0.5"] = hard_neg_fpr(tp, te.source_type.values, 0.5)
    res["per_pii_recall@0.5"] = per_pii_recall(te, tp, 0.5)
    print("[indist] F1=%.3f recall=%.3f roc=%.3f hardFPR=%.3f | per-pii=%s" % (
        res["f1"], res["recall"], res["roc_auc"], res["hard_neg_fpr@0.5"], res["per_pii_recall@0.5"]))
    return res


def run_loso(m, device, epochs, smoke, seed=42):
    speakers = sorted(m.speaker_id.unique())
    folds = []
    for sp in speakers:
        te = m[m.speaker_id == sp].reset_index(drop=True)
        rest = m[m.speaker_id != sp].reset_index(drop=True)
        tr, va, te2, _ = split_manifest(rest, seed=seed)
        tr = pd.concat([tr, te2], ignore_index=True)   # held-out 화자가 test 이므로 rest 전체 학습
        if smoke:
            tr = tr.groupby("label", group_keys=False).sample(n=40, random_state=seed)
            te = te.groupby("label", group_keys=False).sample(n=30, random_state=seed)
        _, _, vp, tp = train_eval("simple_cnn", tr, va if not smoke else tr, te, device, epochs, smoke, seed)
        thr = best_threshold(va.label.values, vp) if not smoke else 0.5
        y = te.label.values
        m05 = classification_metrics(y, tp, 0.5)
        mtuned = classification_metrics(y, tp, thr)
        folds.append({"held": sp, "thr_tuned": thr,
                      "f1@0.5": m05["f1"], "recall@0.5": m05["recall"], "roc": m05["roc_auc"],
                      "f1@tuned": mtuned["f1"], "recall@tuned": mtuned["recall"],
                      "hardFPR@0.5": hard_neg_fpr(tp, te.source_type.values, 0.5)})
        print("  held=%-6s thr=%.2f | F1@0.5=%.3f F1@tuned=%.3f recall@0.5=%.3f roc=%.3f" % (
            sp, thr, m05["f1"], mtuned["f1"], m05["recall"], m05["roc_auc"]))
    f = pd.DataFrame(folds)
    agg = {k: {"mean": float(f[k].mean()), "std": float(f[k].std(ddof=0))}
           for k in ["f1@0.5", "f1@tuned", "recall@0.5", "recall@tuned", "roc", "hardFPR@0.5"]}
    print("\n[LOSO] F1@0.5=%.3f±%.3f  F1@tuned=%.3f±%.3f  ROC=%.3f±%.3f" % (
        agg["f1@0.5"]["mean"], agg["f1@0.5"]["std"],
        agg["f1@tuned"]["mean"], agg["f1@tuned"]["std"], agg["roc"]["mean"], agg["roc"]["std"]))
    return {"folds": folds, "aggregate": agg}


def run_heldpii(m, device, epochs, smoke, held="rrn", seed=42):
    neg = m[m.label == 0]
    pos = m[m.label == 1]
    pos_tr = pos[pos.pii_type != held]
    pos_te = pos[pos.pii_type == held]
    # train: 모든 neg + held 제외 positive  /  test: held positive + neg 일부
    train_pool = pd.concat([neg, pos_tr], ignore_index=True)
    tr, va, te_extra, _assign = split_manifest(train_pool, seed=seed)
    tr = pd.concat([tr, te_extra], ignore_index=True)   # held-pii 가 test 이므로 train_pool 전체 학습
    test = pd.concat([pos_te, neg.sample(min(len(neg), len(pos_te)*2), random_state=seed)], ignore_index=True)
    if smoke:
        tr = tr.groupby("label", group_keys=False).sample(n=40, random_state=seed)
    _, _, vp, tp = train_eval("simple_cnn", tr, va, test, device, epochs, smoke, seed)
    y = test.label.values
    recall_held = float(((np.asarray(tp) >= 0.5).astype(int)[test.pii_type.values == held]).mean())
    res = classification_metrics(y, tp, 0.5)
    print("[heldpii=%s] held-pii recall@0.5=%.3f | overall F1=%.3f roc=%.3f" % (
        held, recall_held, res["f1"], res["roc_auc"]))
    return {"held": held, "held_recall@0.5": recall_held, "overall": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indist", action="store_true")
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--heldpii", action="store_true")
    ap.add_argument("--held", default="rrn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not (args.indist or args.loso or args.heldpii):
        args.indist = args.loso = args.heldpii = True

    device = get_device()
    m = build_qwen_manifest()
    print("manifest: n=%d | label=%s | speakers=%d | pii=%s" % (
        len(m), dict(m.label.value_counts()), m.speaker_id.nunique(),
        dict(m[m.label == 1].pii_type.value_counts())))

    out = {"n": len(m), "speakers": int(m.speaker_id.nunique())}
    if args.indist:  out["indist"] = run_indist(m, device, args.epochs, args.smoke)
    if args.loso:    out["loso"] = run_loso(m, device, args.epochs, args.smoke)
    if args.heldpii: out["heldpii"] = run_heldpii(m, device, args.epochs, args.smoke, args.held)

    os.makedirs(RUNS, exist_ok=True)
    sfx = "_smoke" if args.smoke else ""
    with open(os.path.join(RUNS, f"qwen_experiment{sfx}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.join(RUNS, f'qwen_experiment{sfx}.json')}")


if __name__ == "__main__":
    main()
