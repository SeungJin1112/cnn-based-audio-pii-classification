"""평가 지표 (plan.md §9 / §12).

분류 성능(F1/precision/recall/ROC-AUC/PR-AUC) + 프리필터 운용 지표
(hard-negative FPR, threshold별 recall/precision/STT 절감률).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

DEFAULT_THRESHOLDS = (0.2, 0.3, 0.5, 0.7, 0.8)


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"]  = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"]  = float("nan")
    return out


def hard_neg_fpr(y_prob, source_type, threshold=0.5):
    """hard negative 중 positive 로 잘못 예측한 비율 (전부 실제 label 0)."""
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    st = np.asarray(source_type)
    mask = (st == "negative_hard")
    if mask.sum() == 0:
        return float("nan")
    return float(y_pred[mask].mean())


def threshold_table(y_true, y_prob, source_type=None, thresholds=DEFAULT_THRESHOLDS):
    """프리필터 운용표: threshold별 recall/precision/F1 + STT 호출률/절감률 (+hard-neg FPR)."""
    y_prob = np.asarray(y_prob, dtype=float)
    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, y_prob, t)
        row = {
            "threshold": t,
            "recall": m["recall"], "precision": m["precision"], "f1": m["f1"],
            "stt_call_rate": float((y_prob >= t).mean()),     # positive 예측 = STT로 보냄
        }
        row["stt_reduction"] = 1.0 - row["stt_call_rate"]
        if source_type is not None:
            row["hard_neg_fpr"] = hard_neg_fpr(y_prob, source_type, t)
        rows.append(row)
    return rows
