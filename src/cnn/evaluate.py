"""평가 (plan.md §9/§12) — 5개 평가셋 × 지표 + threshold 운용표 + 효율성, seed 집계.

평가셋: controlled_main / length_matched / hard_negative_only / stress_5_5 / original
clip 점수 = sliding-window P(positive) 의 max (engine.predict_clip_probs).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import config as C
from dataset import build_controlled_manifest, make_eval_sets
from engine import count_params, get_device, predict_clip_probs
from metrics import classification_metrics, hard_neg_fpr, threshold_table
from models import build_model
from splits import build_original_eval, split_manifest

RUNS_DIR = os.path.join(C.CNN_DIR, "runs")
REPORTS_DIR = os.path.join(C.CNN_DIR, "reports")
EVAL_SET_NAMES = ["controlled_main", "length_matched", "hard_negative_only", "stress_5_5", "original"]


def _measure_latency(model, channels, device, n_warm=10, n_meas=50):
    model.eval()
    x = torch.randn(1, channels, C.N_MELS, C.N_FRAMES, device=device)
    with torch.no_grad():
        for _ in range(n_warm):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_meas):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_meas * 1000.0   # ms/window


def evaluate_run(model_name, seed, device):
    m = build_controlled_manifest(seed=seed)
    tr, va, te, _ = split_manifest(m, seed=seed)
    eval_sets = make_eval_sets(te, seed=seed)
    eval_sets["original"] = build_original_eval(tr)

    model, channels = build_model(model_name)
    run_dir = os.path.join(RUNS_DIR, f"{model_name}_seed{seed}")
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    out = {"sets": {}, "threshold_table": None}
    for name in EVAL_SET_NAMES:
        ev = eval_sets[name]
        prob = predict_clip_probs(model, ev, channels, device)
        met = classification_metrics(ev.label.values, prob, threshold=0.5)
        met["hard_neg_fpr"] = hard_neg_fpr(prob, ev.source_type.values, threshold=0.5)
        out["sets"][name] = met
        if name == "controlled_main":
            out["threshold_table"] = threshold_table(
                ev.label.values, prob, source_type=ev.source_type.values)

    out["efficiency"] = {
        "params": count_params(model),
        "size_mb": round(os.path.getsize(ckpt) / 1e6, 2),
        "gpu_latency_ms_per_window": round(_measure_latency(model, channels, device), 3),
    }
    return out


def aggregate(runs):
    """runs: {seed: run_dict} → mean±std per (set, metric) + efficiency mean."""
    seeds = list(runs)
    metrics = ["f1", "recall", "precision", "roc_auc", "pr_auc", "hard_neg_fpr"]
    agg = {"sets": {}, "efficiency": {}}
    for name in EVAL_SET_NAMES:
        agg["sets"][name] = {}
        for met in metrics:
            vals = [runs[s]["sets"][name][met] for s in seeds]
            agg["sets"][name][met] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    for k in ["params", "size_mb", "gpu_latency_ms_per_window"]:
        vals = [runs[s]["efficiency"][k] for s in seeds]
        agg["efficiency"][k] = float(np.mean(vals))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    args = ap.parse_args()
    device = get_device()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    results = {}
    for model_name in args.models:
        per_seed = {}
        for seed in args.seeds:
            per_seed[seed] = evaluate_run(model_name, seed, device)
            f1 = per_seed[seed]["sets"]["controlled_main"]["f1"]
            print(f"  {model_name} seed{seed}: controlled F1={f1:.3f}")
        results[model_name] = {"per_seed": per_seed, "aggregate": aggregate(per_seed)}

    with open(os.path.join(RUNS_DIR, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_report(results)
    print(f"\nsaved -> {os.path.join(RUNS_DIR, 'evaluation.json')}")
    print(f"report -> {os.path.join(REPORTS_DIR, 'results.md')}")


def _fmt(d):
    return f"{d['mean']:.3f}±{d['std']:.3f}"


def _write_report(results):
    """plan §13 표 + §11 해석 매트릭스 자동 적용."""
    baselines = {}
    bpath = os.path.join(RUNS_DIR, "baselines.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            baselines = json.load(f).get("aggregate", {})

    L = ["# 결과 리포트 (plan.md §13)", "",
         "clip 점수 = sliding-window P(positive) 의 max. seed 평균±표준편차.", ""]

    # --- 메인 성능표 ---
    L += ["## 1. 메인 성능표", "",
          "| Model | Controlled F1 | Controlled Recall | Length-matched F1 | Hard-neg FPR | ROC-AUC | PR-AUC |",
          "|---|---|---|---|---|---|---|"]
    for name, agg in baselines.items():
        cm, lm = agg["controlled_main"], agg["length_matched"]
        L.append(f"| {name} (baseline) | {_fmt(cm['f1'])} | {_fmt(cm['recall'])} | "
                 f"{_fmt(lm['f1'])} | {_fmt(cm['hard_neg_fpr'])} | {_fmt(cm['roc_auc'])} | {_fmt(cm['pr_auc'])} |")
    for name, r in results.items():
        a = r["aggregate"]["sets"]; cm, lm = a["controlled_main"], a["length_matched"]
        L.append(f"| **{name}** | {_fmt(cm['f1'])} | {_fmt(cm['recall'])} | "
                 f"{_fmt(lm['f1'])} | {_fmt(cm['hard_neg_fpr'])} | {_fmt(cm['roc_auc'])} | {_fmt(cm['pr_auc'])} |")

    # --- 효율성표 ---
    L += ["", "## 2. 효율성표", "",
          "| Model | Params | Size MB | GPU ms/window |", "|---|---|---|---|"]
    for name, r in results.items():
        e = r["aggregate"]["efficiency"]
        L.append(f"| {name} | {int(e['params']):,} | {e['size_mb']:.2f} | {e['gpu_latency_ms_per_window']:.3f} |")

    # --- 평가셋별 F1 ---
    L += ["", "## 3. 평가셋별 F1", "",
          "| Model | " + " | ".join(EVAL_SET_NAMES) + " |",
          "|---|" + "---|" * len(EVAL_SET_NAMES)]
    for name, r in results.items():
        a = r["aggregate"]["sets"]
        L.append(f"| {name} | " + " | ".join(_fmt(a[s]["f1"]) for s in EVAL_SET_NAMES) + " |")

    # --- 해석 (plan §11 매트릭스) ---
    L += ["", "## 4. 해석 (plan §11 매트릭스 자동 적용)", ""]
    ac = baselines.get("acoustic_lr", {}).get("controlled_main", {}).get("roc_auc", {}).get("mean", float("nan"))
    dur_lm = baselines.get("duration_only", {}).get("length_matched", {}).get("roc_auc", {}).get("mean", float("nan"))
    for name, r in results.items():
        a = r["aggregate"]["sets"]
        notes = []
        cm_roc = a["controlled_main"]["roc_auc"]["mean"]
        lm_roc = a["length_matched"]["roc_auc"]["mean"]
        orig_f1 = a["original"]["f1"]["mean"]; cm_f1 = a["controlled_main"]["f1"]["mean"]
        hfpr = a["hard_negative_only"]["hard_neg_fpr"]["mean"]
        if orig_f1 - cm_f1 > 0.1:
            notes.append("Original≫Controlled → naive artifact 의존 의심")
        if cm_roc - lm_roc > 0.1:
            notes.append("Length-matched 급락 → 길이 단서 의존")
        if hfpr > 0.3:
            notes.append(f"hard-neg FPR 높음({hfpr:.2f}) → 숫자 일반 발화 오탐")
        if not np.isnan(ac) and cm_roc > ac + 0.02:
            notes.append(f"acoustic-LR(ROC {ac:.2f}) 능가 → 스펙트로그램 패턴 학습 가능성")
        elif not np.isnan(ac) and cm_roc < ac - 0.02:
            notes.append(f"acoustic-LR(ROC {ac:.2f}) 미달 → CNN 추가 가치 제한적")
        if not notes:
            notes.append("baseline 대비 뚜렷한 이상 신호 없음")
        L.append(f"- **{name}**: " + " / ".join(notes))

    L += ["", "> 주의: 본 결과는 전화번호 형식 숫자열의 음향적 존재 탐지 feasibility 에 한정.",
          "> 문맥상 실제 PII 여부·STT 경로 대체 가능성은 주장하지 않음(plan §11).", ""]

    with open(os.path.join(REPORTS_DIR, "results.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
