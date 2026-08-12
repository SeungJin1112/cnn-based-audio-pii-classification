"""순서를 보는 헤드가 자릿수 그룹 구조를 쓰는가 — 결정 함수 특정의 후속 통제.

\ref{sec:exp:decision} 절은 simple_cnn 의 결정 변수가 그룹 구조가 아니라 등간격 단음절
연쇄의 분량임을 보였다. 그러나 simple_cnn 의 헤드는 AdaptiveAvgPool2d(1) 이므로 시간축
순서가 설계상 지워진다. 따라서 "그룹 구조를 못 쓴다"가 표현의 한계인지 풀링의 산물인지
그 실험만으로는 가를 수 없다.

이 스크립트는 몸통을 그대로 두고 헤드만 바꾼 세 변형(models/seq_heads.py)을 같은 데이터·
같은 절차로 학습시켜 세 가지를 잰다.

  1) in-distribution 성능 (v2-HN)                    — 헤드 교체가 성능을 해치지 않는지
  2) 제품 일련번호 held-out-category 오탐률           — 4-4-4(12자리)를 걸러내기 시작하는지
  3) 통제 자극 자릿수 곡선 (probe_eval.py 로 별도)    — 그룹 구조 곡선이 갈리는지

(2) 가 가장 진단적이다. 제품 일련번호는 총 자릿수 12로 계좌번호와 같지만 그룹 구조가
4-4-4 대 3-3-6 으로 다르다. 그룹 경계를 결정에 쓰는 모델이라면 이 카테고리의 오탐률이
simple_cnn 의 0.947 보다 낮아져야 한다.

usage:
  python seq_head_exp.py --model gru_cnn --seed 42 --mode indist
  python seq_head_exp.py --model gru_cnn --seed 42 --mode heldcat --cat serial
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import common as _c  # noqa: E402
from engine import get_device, predict_clip_probs  # noqa: E402
from metrics import classification_metrics, hard_neg_fpr  # noqa: E402
from models import build_model  # noqa: E402
from run_v2_seeds import SPLIT_SEED, per_category_fpr, per_pii_recall, train_deterministic  # noqa: E402
from splits import split_manifest  # noqa: E402


def run_indist(m, model_name, device, epochs, seed, tag):
    tr, va, te, _ = split_manifest(m, seed=SPLIT_SEED)
    model, ch = build_model(model_name)
    info = train_deterministic(model, tr, va, ch, device, epochs, seed)
    tp = predict_clip_probs(model, te, ch, device)

    res = classification_metrics(te.label.values, tp, 0.5)
    res["hard_neg_fpr@0.5"] = hard_neg_fpr(tp, te.source_type.values, 0.5)
    res["per_pii_recall@0.5"] = per_pii_recall(te, tp, 0.5)
    res["per_category_fpr@0.5"] = per_category_fpr(te, tp, 0.5)
    res["_meta"] = {"model": model_name, "seed": seed,
                    "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "best_val_f1": info["best_val_f1"], "epochs_run": info["epochs_run"]}
    torch.save(model.state_dict(), os.path.join(_c.OUT_DIR, f"ckpt_{tag}.pt"))
    _c.save_scores(f"scores_{tag}_test.csv", te, tp)
    print("[indist %-14s s%d] F1=%.3f recall=%.3f prec=%.3f hardFPR=%.3f roc=%.3f" % (
        model_name, seed, res["f1"], res["recall"], res["precision"],
        res["hard_neg_fpr@0.5"], res["roc_auc"]), flush=True)
    return res


def run_heldcat(m, model_name, device, epochs, seed, cat, tag):
    held = m[(m.source_type == "negative_hard") & (m.template_id == cat)].reset_index(drop=True)
    pool = m[~((m.source_type == "negative_hard") & (m.template_id == cat))].reset_index(drop=True)
    tr, va, te_extra, _ = split_manifest(pool, seed=SPLIT_SEED)
    tr = pd.concat([tr, te_extra], ignore_index=True)

    model, ch = build_model(model_name)
    info = train_deterministic(model, tr, va, ch, device, epochs, seed)
    hp = predict_clip_probs(model, held, ch, device)
    fpr = float((np.asarray(hp) >= 0.5).mean())
    _c.save_scores(f"scores_{tag}.csv", held, hp)
    print("[heldcat %-8s %-14s s%d] n=%d FPR@0.5=%.3f mean=%.3f" % (
        cat, model_name, seed, len(held), fpr, float(np.mean(hp))), flush=True)
    return {"category": cat, "model": model_name, "seed": seed, "n": int(len(held)),
            "held_fpr@0.5": fpr, "mean_prob": float(np.mean(hp)),
            "best_val_f1": info["best_val_f1"], "epochs_run": info["epochs_run"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["indist", "heldcat"])
    ap.add_argument("--cat", default="serial")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    _c.set_determinism(args.seed)
    # cuDNN 의 RNN 커널은 결정적 알고리즘 목록에 없다. GRU 계열만 cuDNN 을 끄고
    # 순수 구현으로 돌려 결정성을 유지한다(속도는 이 규모에서 문제가 되지 않는다).
    if "gru" in args.model:
        torch.backends.cudnn.enabled = False

    device = get_device()
    m = _c.build_manifest(drop_outliers=True)

    if args.mode == "indist":
        tag = f"seq_indist_{args.model}_s{args.seed}"
        res = run_indist(m, args.model, device, args.epochs, args.seed, tag)
    else:
        tag = f"seq_heldcat_{args.cat}_{args.model}_s{args.seed}"
        res = run_heldcat(m, args.model, device, args.epochs, args.seed, args.cat, tag)
    _c.save_json(f"{tag}.json", res)


if __name__ == "__main__":
    main()
