"""RQ5 재검정 — 학습률 스윕.

논문은 네 모델에 AdamW 1e-4 를 동일 적용했다. 사전학습 백본 전 계층 미세조정에
1e-4 는 통상 과대하며, ConvNeXt 의 시드 간 ±0.13 진동이 그 증상일 수 있다.
"동일 예산·동일 설정"이라는 단서를 "학습률을 각 모델에 유리하게 고른 뒤에도"로
강화하려면 모델별 최적 학습률에서 비교해야 한다.

usage:
  python lr_sweep.py --model convnext_tiny --lr 1e-5 --seed 42
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common as _c  # noqa: E402
from dataset import build_controlled_manifest, make_eval_sets  # noqa: E402
from engine import get_device, predict_clip_probs  # noqa: E402
from metrics import classification_metrics, hard_neg_fpr  # noqa: E402
from models import build_model  # noqa: E402
from run_v2_seeds import train_deterministic  # noqa: E402
from splits import split_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    _c.set_determinism(args.seed)
    device = get_device()
    m = build_controlled_manifest(seed=args.seed)
    tr, va, te, _ = split_manifest(m, seed=args.seed)

    model, ch = build_model(args.model)
    info = train_deterministic(model, tr, va, ch, device, args.epochs, args.seed, lr=args.lr)

    out = {"model": args.model, "lr": args.lr, "seed": args.seed,
           "best_val_f1": info["best_val_f1"], "epochs_run": info["epochs_run"], "sets": {}}
    for name, s in make_eval_sets(te, seed=args.seed).items():
        p = predict_clip_probs(model, s, ch, device)
        r = classification_metrics(s.label.values, p, 0.5)
        r["hard_neg_fpr"] = hard_neg_fpr(p, s.source_type.values, 0.5)
        out["sets"][name] = r
    cm = out["sets"]["controlled_main"]
    print("[%s lr=%g seed=%d] val_f1=%.3f | test F1=%.3f prec=%.3f rec=%.3f PR-AUC=%.3f hardFPR=%.3f" % (
        args.model, args.lr, args.seed, info["best_val_f1"], cm["f1"], cm["precision"],
        cm["recall"], cm["pr_auc"], cm["hard_neg_fpr"]))
    tag = f"lrsweep_{args.model}_lr{args.lr:g}_s{args.seed}"
    _c.save_json(tag + ".json", out)


if __name__ == "__main__":
    main()
