"""단일 wav 추론 데모 (plan.md §12/§14).

오디오 → log-mel sliding window → clip_score = max(P(positive)) → 전화번호 형식 포함 여부.

usage:
  python infer.py --wav path/to.wav --model simple_cnn --seed 42 [--threshold 0.5]
"""
import argparse
import os

import librosa
import numpy as np
import torch

import config as C
from dataset import sliding_windows, wav_to_logmel, _to_tensor
from engine import get_device
from models import build_model


@torch.no_grad()
def predict_wav(model, channels, device, wav_path, threshold=0.5):
    y, _ = librosa.load(wav_path, sr=C.SAMPLE_RATE, mono=True)
    rng = np.random.default_rng(0)
    wins = sliding_windows(y, rng)
    xs = torch.stack([_to_tensor(wav_to_logmel(w), channels) for w in wins]).to(device)
    p = torch.softmax(model(xs), dim=1)[:, 1]      # window별 P(positive)
    clip_prob = float(p.max().item())              # clip = max
    return clip_prob, (clip_prob >= threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model", default="simple_cnn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = get_device()
    model, channels = build_model(args.model)
    ckpt = os.path.join(C.CNN_DIR, "runs", f"{args.model}_seed{args.seed}", "checkpoint.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    prob, is_phone = predict_wav(model, channels, device, args.wav, args.threshold)
    label = "전화번호 형식 포함(1)" if is_phone else "미포함(0)"
    print(f"[{args.model}] {os.path.basename(args.wav)} | P(phone)={prob:.3f} "
          f"(thr={args.threshold}) → {label}")


if __name__ == "__main__":
    main()
