"""잡음 증강 학습 — 전화망 조건을 학습 시점에 섞어 프리필터를 강건화한다.

\ref{sec:exp:degrade} 절의 열화 평가는 평가 시점에만 열화를 가했다. 학습이 깨끗한 16 kHz
합성음만 본 상태이므로, 관측된 성능 저하가 과제의 한계인지 학습 조건 불일치인지 알 수 없다.
이 스크립트는 학습 데이터에 확률적으로 (a) 8 kHz 대역 제한, (b) G.711 근사 양자화,
(c) 통과대역 제한 잡음을 섞어 그 물음에 답한다.

열화는 파형 단계에서 적용하므로 ffmpeg 없이 numpy 로 근사한다. 대역 제한과 mu-law 는 평가용
열화(degrade_eval.py, 실제 ffmpeg G.711)와 구현이 다르지만, \emph{통과대역 잡음 항은 대역
상수·필터·잡음원·SNR 정의가 평가와 동일한 과정}이고 학습 SNR 범위 [5, 25] dB 가 평가점
{20, 10, 5} dB 를 모두 포함한다. 따라서 이 증강으로 얻는 회복치는 정합 조건(matched condition)
에서의 상한이며, 처음 보는 잡음 종류로의 일반화를 뜻하지 않는다.

usage:
  python noise_aug.py --seed 42 [--epochs 40]
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import common as _c  # noqa: E402
import config as C  # noqa: E402
import dataset as D  # noqa: E402
from engine import get_device, predict_clip_probs  # noqa: E402
from metrics import classification_metrics, hard_neg_fpr  # noqa: E402
from models import build_model  # noqa: E402
from run_v2_seeds import SPLIT_SEED, per_category_fpr, per_pii_recall, train_deterministic  # noqa: E402
from splits import split_manifest  # noqa: E402

TEL_BAND = (300.0, 3400.0)
SNR_RANGE = (5.0, 25.0)
P_DEGRADE = 0.5          # 클립의 절반만 열화 — 깨끗한 조건 성능을 지키기 위함


def _bandpass(x, sr, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def _mulaw_roundtrip(x, mu=255.0):
    """G.711 mu-law 양자화 근사(8비트)."""
    s = np.sign(x)
    y = s * np.log1p(mu * np.abs(np.clip(x, -1, 1))) / np.log1p(mu)
    q = np.round((y + 1) * 127.5) / 127.5 - 1
    return (np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu)) / mu)).astype(np.float32)


def degrade_waveform(y, rng):
    """8 kHz 대역 제한 → mu-law 양자화 → 통과대역 잡음. 각 단계는 확률적으로 적용된다."""
    sr = C.SAMPLE_RATE
    if rng.random() < 0.8:                      # 대역 제한
        y = _bandpass(y, sr, 0.0, 3800.0)
    if rng.random() < 0.6:                      # 코덱 양자화
        peak = np.max(np.abs(y)) + 1e-9
        y = _mulaw_roundtrip(y / peak) * peak
    if rng.random() < 0.9:                      # 통과대역 잡음
        snr = rng.uniform(*SNR_RANGE)
        y_in = _bandpass(y, sr, *TEL_BAND)
        p_sig = float(np.mean(y_in.astype(np.float64) ** 2))
        if p_sig > 0:
            n = _bandpass(rng.standard_normal(len(y)).astype(np.float32), sr, *TEL_BAND)
            p_n = float(np.mean(n.astype(np.float64) ** 2))
            y = y + n * np.sqrt((p_sig / (10.0 ** (snr / 10.0))) / max(p_n, 1e-20))
    peak = np.max(np.abs(y))
    return (y / peak if peak > 1.0 else y).astype(np.float32)


class NoisyDataset(D.AudioPIIDataset):
    """학습 모드에서만 파형 열화를 확률적으로 적용하는 AudioPIIDataset."""

    def __getitem__(self, i):
        if self.mode != "train":
            return super().__getitem__(i)
        row = self.df.iloc[i]
        import librosa
        y, _ = librosa.load(row["filepath"], sr=C.SAMPLE_RATE, mono=True)
        if self.rng.random() < P_DEGRADE:
            y = degrade_waveform(y, self.rng)
        seg = D.random_crop(y, self.rng)
        mel = D.wav_to_logmel(seg)
        if self.augment:
            mel = D.spec_augment(mel, self.rng)
        return D._to_tensor(mel, self.channels), int(row["label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    _c.set_determinism(args.seed)
    device = get_device()
    m = _c.build_manifest(drop_outliers=True)
    tr, va, te, _ = split_manifest(m, seed=SPLIT_SEED)

    # 학습 데이터셋만 교체한다. 검증·평가는 원래 클래스를 그대로 쓴다.
    orig = D.AudioPIIDataset
    D.AudioPIIDataset = NoisyDataset
    try:
        model, ch = build_model("simple_cnn")
        info = train_deterministic(model, tr, va, ch, device, args.epochs, args.seed)
    finally:
        D.AudioPIIDataset = orig

    tag = f"noiseaug_s{args.seed}"
    torch.save(model.state_dict(), os.path.join(_c.OUT_DIR, f"ckpt_{tag}.pt"))
    tp = predict_clip_probs(model, te, ch, device)
    res = classification_metrics(te.label.values, tp, 0.5)
    res["hard_neg_fpr@0.5"] = hard_neg_fpr(tp, te.source_type.values, 0.5)
    res["per_pii_recall@0.5"] = per_pii_recall(te, tp, 0.5)
    res["per_category_fpr@0.5"] = per_category_fpr(te, tp, 0.5)
    res["_meta"] = {"seed": args.seed, "epochs": args.epochs,
                    "p_degrade": P_DEGRADE, "snr_range": list(SNR_RANGE),
                    "best_val_f1": info["best_val_f1"], "epochs_run": info["epochs_run"]}
    _c.save_scores(f"scores_{tag}_test.csv", te, tp)
    _c.save_json(f"{tag}.json", res)
    print("[noiseaug s%d] clean test F1=%.3f recall=%.3f prec=%.3f hardFPR=%.3f" % (
        args.seed, res["f1"], res["recall"], res["precision"], res["hard_neg_fpr@0.5"]))


if __name__ == "__main__":
    main()
