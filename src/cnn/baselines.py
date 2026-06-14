"""Baseline (plan.md §5) — majority / duration-only LR / acoustic-feature LR.

CNN 결과 해석의 기준선. CNN 은 최소한 duration-only / acoustic-LR 을 유의미하게
능가해야 "스펙트로그램 패턴을 배웠다"는 주장이 성립한다.

평가셋(plan §4): controlled_main / length_matched / hard_negative_only / stress_5_5 / original.
seed 반복 후 평균±표준편차로 보고.
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from dataset import build_controlled_manifest, make_eval_sets
from features import DURATION_IDX, feature_matrix
from metrics import classification_metrics, hard_neg_fpr
from splits import build_original_eval, split_manifest

RUNS_DIR = os.path.join(C.CNN_DIR, "runs")
REPORT_METRICS = ["f1", "recall", "precision", "roc_auc", "pr_auc"]


def _eval_model(predict_proba, eval_sets):
    """eval_sets 각각에 대해 지표 + hard-neg FPR 계산."""
    res = {}
    for name, ev in eval_sets.items():
        Xev = feature_matrix(ev)
        y = ev.label.values
        prob = predict_proba(Xev)
        m = classification_metrics(y, prob, threshold=0.5)
        m["hard_neg_fpr"] = hard_neg_fpr(prob, ev.source_type.values, threshold=0.5)
        res[name] = m
    return res


def run_once(seed):
    m = build_controlled_manifest(seed=seed)
    tr, va, te, _ = split_manifest(m, seed=seed)
    eval_sets = make_eval_sets(te, seed=seed)
    eval_sets["original"] = build_original_eval(tr)

    Xtr = feature_matrix(tr)
    ytr = tr.label.values
    maj = int(round(ytr.mean()))  # 다수 클래스

    dur_lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000)).fit(Xtr[:, [DURATION_IDX]], ytr)
    ac_lr = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000)).fit(Xtr, ytr)

    models = {
        "majority":      lambda X: np.full(len(X), float(maj)),
        "duration_only": lambda X: dur_lr.predict_proba(X[:, [DURATION_IDX]])[:, 1],
        "acoustic_lr":   lambda X: ac_lr.predict_proba(X)[:, 1],
    }
    return {name: _eval_model(fn, eval_sets) for name, fn in models.items()}


def aggregate(per_seed):
    """seed 별 결과를 평균±표준편차로 집계."""
    agg = {}
    models = per_seed[0].keys()
    eval_names = next(iter(per_seed[0].values())).keys()
    for model in models:
        agg[model] = {}
        for ev in eval_names:
            agg[model][ev] = {}
            for met in REPORT_METRICS + ["hard_neg_fpr"]:
                vals = [s[model][ev][met] for s in per_seed]
                agg[model][ev][met] = {"mean": float(np.nanmean(vals)),
                                       "std": float(np.nanstd(vals))}
    return agg


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    per_seed = [run_once(s) for s in C.SEEDS]
    agg = aggregate(per_seed)
    out_path = os.path.join(RUNS_DIR, "baselines.json")
    with open(out_path, "w") as f:
        json.dump({"seeds": C.SEEDS, "per_seed": per_seed, "aggregate": agg}, f, indent=2)

    # 콘솔 요약 (controlled_main 중심)
    print(f"\n=== Baseline (seeds={C.SEEDS}) — controlled_main / length_matched / hard_negative_only ===")
    hdr = f"{'model':14s} | {'set':18s} | {'F1':>12s} {'recall':>12s} {'roc_auc':>12s} {'hard_fpr':>12s}"
    print(hdr); print("-" * len(hdr))
    for model in agg:
        for ev in ["controlled_main", "length_matched", "hard_negative_only"]:
            a = agg[model][ev]
            def s(k): return f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}"
            print(f"{model:14s} | {ev:18s} | {s('f1'):>12s} {s('recall'):>12s} {s('roc_auc'):>12s} {s('hard_neg_fpr'):>12s}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
