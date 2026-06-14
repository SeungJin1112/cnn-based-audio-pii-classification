"""Leave-One-Speaker-Out (LOSO) 강건성 실험 — speaker 누수 confound 직접 통제.

배경(reports/conclusion.md 한계):
  기존 main 실험은 template-disjoint + length-stratified 만 적용하고
  speaker-disjoint 는 "화자 수 한계"로 미적용 → 화자 누수를 제거 못 한 confound 로 명시했다.

그러나 controlled manifest 를 확인하면 화자가 **3명**(InJoon/SunHi/Hyunsu)이고
각 화자가 양 라벨을 고르게(~330/330) 보유한다. 따라서 LOSO 가 가능하다:

  fold k: held-out = speaker k 의 모든 clip(평가),  train = 나머지 2 화자.
  → train/test 의 화자가 완전히 분리되므로 "화자 정체성 암기"로는 못 맞힌다.
    LOSO F1 이 in-distribution(template-disjoint) F1(simple_cnn 0.955)에 근접하면
    모델이 화자가 아니라 전화번호 형식 신호를 학습했다는 더 강한 근거가 된다.

train 내부 val(early stopping)은 train-화자 subset 을 split_manifest 로 한번 더
template-disjoint 분리해 만든다(val 도 train 화자 안에서만, 정보 누수 없음).

usage:
  python speaker_robustness.py --model simple_cnn --seeds 42 43 44 [--epochs 50] [--smoke]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import config as C
from dataset import build_controlled_manifest, make_eval_sets
from engine import get_device, predict_clip_probs, train_model
from metrics import classification_metrics, hard_neg_fpr, threshold_table
from models import build_model
from splits import split_manifest


def _train_val_from_speakers(df, seed):
    """train 화자 subset 을 template-disjoint 로 train/val 분리(test 조각은 버림)."""
    tr, va, te, _ = split_manifest(df, seed=seed)
    # te 는 추가 train 데이터로 흡수(LOSO 의 test 는 held-out 화자이므로 여기선 불필요)
    tr_full = pd.concat([tr, te], ignore_index=True)
    return tr_full, va


def run_fold(model_name, held_speaker, manifest, seed, epochs, smoke, device):
    test_df  = manifest[manifest.speaker_id == held_speaker].reset_index(drop=True)
    train_sp = manifest[manifest.speaker_id != held_speaker].reset_index(drop=True)

    if smoke:
        train_sp = train_sp.groupby("label", group_keys=False).sample(n=60, random_state=seed)
        test_df  = test_df.groupby("label", group_keys=False).sample(n=40, random_state=seed)

    tr_df, va_df = _train_val_from_speakers(train_sp, seed)

    model, channels = build_model(model_name)
    info = train_model(model, tr_df, va_df, channels, device,
                       epochs=epochs, seed=seed, augment=True, smoke=smoke)

    # held-out 화자 전체에 대한 clip-level 예측
    probs = predict_clip_probs(model, test_df, channels, device, seed=seed)
    y = test_df.label.values

    main = classification_metrics(y, probs, threshold=0.5)
    eval_sets = make_eval_sets(test_df, seed=seed)
    hn = eval_sets["hard_negative_only"]
    hn_probs = predict_clip_probs(model, hn, channels, device, seed=seed)
    hn_main = classification_metrics(hn.label.values, hn_probs, threshold=0.5)

    return {
        "held_speaker": held_speaker,
        "n_train": int(len(tr_df)), "n_val": int(len(va_df)), "n_test": int(len(test_df)),
        "best_val_f1": info["best_val_f1"],
        "test_f1": main["f1"], "test_recall": main["recall"],
        "test_precision": main["precision"], "test_roc_auc": main["roc_auc"],
        "hard_neg_fpr@0.5": hard_neg_fpr(probs, test_df.source_type.values, 0.5),
        "hard_neg_only_f1": hn_main["f1"], "hard_neg_only_roc": hn_main["roc_auc"],
        "threshold_table": threshold_table(y, probs, test_df.source_type.values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simple_cnn")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = get_device()
    print(f"[LOSO] model={args.model} seeds={args.seeds} device={device} smoke={args.smoke}")

    # seed 와 무관하게 화자 집합은 동일 → seed42 manifest 로 화자 목록 확보
    speakers = sorted(build_controlled_manifest(seed=42).speaker_id.unique())
    print(f"[LOSO] speakers({len(speakers)}): {speakers}")

    results = {"model": args.model, "seeds": args.seeds, "smoke": args.smoke,
               "speakers": speakers, "folds": []}

    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        manifest = build_controlled_manifest(seed=seed)
        for sp in speakers:
            r = run_fold(args.model, sp, manifest, seed, args.epochs, args.smoke, device)
            r["seed"] = seed
            results["folds"].append(r)
            print(f"  seed{seed} held={sp:32s} "
                  f"test_F1={r['test_f1']:.3f} recall={r['test_recall']:.3f} "
                  f"hardFPR={r['hard_neg_fpr@0.5']:.3f} (val_F1={r['best_val_f1']:.3f})")

    # 집계: held_speaker 별 평균, 전체 평균
    folds = pd.DataFrame(results["folds"])
    agg = {}
    for met in ["test_f1", "test_recall", "test_precision", "test_roc_auc", "hard_neg_fpr@0.5"]:
        agg[met] = {"mean": float(folds[met].mean()), "std": float(folds[met].std(ddof=0))}
    results["aggregate"] = agg
    results["by_speaker"] = {
        sp: {"test_f1_mean": float(g.test_f1.mean()),
             "test_recall_mean": float(g.test_recall.mean())}
        for sp, g in folds.groupby("held_speaker")
    }

    out_dir = os.path.join(C.CNN_DIR, "runs")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = os.path.join(out_dir, f"loso_{args.model}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n== LOSO 집계 (speaker-disjoint) ==")
    print(f"  test F1     : {agg['test_f1']['mean']:.3f} ± {agg['test_f1']['std']:.3f}")
    print(f"  test recall : {agg['test_recall']['mean']:.3f} ± {agg['test_recall']['std']:.3f}")
    print(f"  test ROC-AUC: {agg['test_roc_auc']['mean']:.3f} ± {agg['test_roc_auc']['std']:.3f}")
    print(f"  hard-neg FPR: {agg['hard_neg_fpr@0.5']['mean']:.3f} ± {agg['hard_neg_fpr@0.5']['std']:.3f}")
    print(f"  (참고) in-distribution(template-disjoint) simple_cnn F1 = 0.955")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
