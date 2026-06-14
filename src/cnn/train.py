"""학습 실행 (plan.md §10) — 한 (model, seed) 조합을 학습하고 runs/ 에 저장.

split 은 build_controlled_manifest(seed)+split_manifest(seed) 의 결정성으로 재현되므로
evaluate.py 가 동일 seed 로 같은 train/val/test 를 복원한다.

usage: python train.py --model simple_cnn --seed 42 [--epochs 50] [--smoke]
"""
import argparse
import json
import os

import numpy as np
import torch

import config as C
from dataset import build_controlled_manifest
from engine import count_params, get_device, train_model
from models import build_model
from splits import split_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    m = build_controlled_manifest(seed=args.seed)
    if args.smoke:
        # 빠른 파이프라인 검증용 subset
        m = m.groupby("label", group_keys=False).sample(n=60, random_state=args.seed)
    tr, va, te, _ = split_manifest(m, seed=args.seed)

    model, channels = build_model(args.model)
    info = train_model(model, tr, va, channels, device,
                       epochs=args.epochs, seed=args.seed,
                       augment=not args.no_augment, smoke=args.smoke)

    run_dir = os.path.join(C.CNN_DIR, "runs", f"{args.model}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "checkpoint.pt"))
    with open(os.path.join(run_dir, "train.json"), "w") as f:
        json.dump({
            "model": args.model, "seed": args.seed, "channels": channels,
            "params": count_params(model), "epochs": args.epochs,
            "augment": not args.no_augment, "smoke": args.smoke,
            "best_val_f1": info["best_val_f1"], "history": info["history"],
            "split": {"train": len(tr), "val": len(va), "test": len(te)},
        }, f, indent=2)
    print(f"[{args.model} seed{args.seed}] best_val_f1={info['best_val_f1']:.3f} "
          f"params={count_params(model):,} -> {run_dir}")


if __name__ == "__main__":
    main()
