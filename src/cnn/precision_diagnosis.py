"""1순위 진단 — precision 누수의 원인 위치 파악.

calibration 이 precision 을 못 고친다는 결과(calibration.json) 후속:
어떤 hard-neg 카테고리가 PII 로 오검출되는지(template_id 별 FPR), easy vs hard 별 FPR 분해.
→ "포맷이 PII 와 유사한 hard-neg(예: 일련번호 4-4-4)" 가 주범인지 확인.

usage: python precision_diagnosis.py [--epochs 40]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import config as C
from engine import get_device, predict_clip_probs, train_model
from models import build_model
from speaker_pii_experiment import build_qwen_manifest
from splits import split_manifest

RUNS = os.path.join(C.CNN_DIR, "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    device = get_device()
    m = build_qwen_manifest()
    tr, va, te, _ = split_manifest(m, seed=42)
    model, ch = build_model("simple_cnn")
    print("[train] simple_cnn ...")
    train_model(model, tr, va, ch, device, epochs=args.epochs, seed=42)

    tp = predict_clip_probs(model, te, ch, device, seed=42)
    te = te.copy(); te["prob"] = tp; te["pred"] = (tp >= args.thr).astype(int)

    # negative(label=0) 만: source_type/template_id 별 오검출률(FPR)
    neg = te[te.label == 0]
    by_src = neg.groupby("source_type").apply(lambda g: round(g.pred.mean(), 3)).to_dict()
    hard = neg[neg.source_type == "negative_hard"]
    by_cat = hard.groupby("template_id").apply(
        lambda g: {"n": int(len(g)), "fpr": round(float(g.pred.mean()), 3),
                   "mean_prob": round(float(g.prob.mean()), 3)}).to_dict()
    # positive recall by pii_type
    pos = te[te.label == 1]
    by_pii = pos.groupby("pii_type").apply(lambda g: round(float(g.pred.mean()), 3)).to_dict()

    out = {"threshold": args.thr, "fpr_by_source": by_src,
           "hard_neg_fpr_by_category": by_cat, "recall_by_pii": by_pii}
    with open(os.path.join(RUNS, "precision_diagnosis.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n== source_type 별 FPR (label=0, thr={args.thr}) ==")
    for k, v in by_src.items():
        print(f"  {k:16s} FPR={v}")
    print("\n== hard-neg 카테고리별 FPR (오검출률 높은 순) ==")
    for cat, d in sorted(by_cat.items(), key=lambda x: -x[1]["fpr"]):
        print(f"  {cat:10s} n={d['n']:3d}  FPR={d['fpr']:.3f}  mean_prob={d['mean_prob']:.3f}")
    print("\n== PII 유형별 recall ==")
    for k, v in by_pii.items():
        print(f"  {k:10s} recall={v}")
    print(f"\nsaved -> {os.path.join(RUNS, 'precision_diagnosis.json')}")


if __name__ == "__main__":
    main()
