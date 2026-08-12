"""협대역·코덱·잡음 열화 평가.

표적 도메인은 8 kHz 협대역 전화망인데 실험은 16 kHz 광대역에서 수행되었다.
새 데이터 수집 없이 기존 v2-HN 클립을 전화망 조건으로 열화시켜 재평가한다.

조건:
  clean        원본 16 kHz
  nb8k         16k -> 8k -> 16k 리샘플 왕복 (대역 제한만)
  g711         16k -> 8k -> G.711 mu-law 왕복 -> 16k (코덱 양자화 포함)
  g711_snr20   위 + 백색잡음 SNR 20 dB
  g711_snr10   위 + 백색잡음 SNR 10 dB
  snr20/snr10  16 kHz 유지 + 잡음만 (대역과 잡음의 기여 분리)

usage:
  python degrade_eval.py --ckpt out/ckpt_indist_s42.pt --seed 42
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common as _c  # noqa: E402
import config as C  # noqa: E402
from engine import get_device, predict_clip_probs  # noqa: E402
from metrics import classification_metrics, hard_neg_fpr  # noqa: E402
from models import build_model  # noqa: E402
from splits import split_manifest  # noqa: E402

WORK = os.path.join(_c.OUT_DIR, "degraded")
# 전화망 통과대역. 이 대역 밖의 잡음은 실제 협대역 채널이 전달하지 못한다.
TEL_BAND = (300.0, 3400.0)
CONDITIONS = ["clean", "nb8k", "g711", "g711_snr20", "g711_snr10", "snr20", "snr10",
              "g711_bp20", "g711_bp10", "g711_bp05"]


def _ffmpeg(args):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args,
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[:400])


def _bandpass(x, sr, lo, hi):
    """FFT 영역에서 [lo, hi] Hz 밖을 제거한다."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def add_noise(y, snr_db, rng, band=None):
    """목표 SNR 이 되도록 잡음을 더한다.

    band 가 주어지면 잡음을 그 대역으로 제한하고 SNR 도 \emph{대역 내} 전력비로 정의한다.
    협대역 전화망을 거친 신호는 4\,kHz 이상 에너지가 사실상 0이므로, 전대역 백색잡음을
    공칭 SNR 로 주입하면 잡음 전력의 절반이 신호가 없는 대역에 떨어져 실제 통화보다
    훨씬 가혹한 조건이 된다. 통과대역 잡음이 전화망 조건의 올바른 근사이다.
    """
    y = y.astype(np.float32)
    sr = C.SAMPLE_RATE
    if band is None:
        p_sig = float(np.mean(y.astype(np.float64) ** 2))
        if p_sig <= 0:
            return y
        n = rng.standard_normal(len(y)) * np.sqrt(p_sig / (10.0 ** (snr_db / 10.0)))
    else:
        lo, hi = band
        y_in = _bandpass(y, sr, lo, hi)
        p_sig = float(np.mean(y_in.astype(np.float64) ** 2))   # 대역 내 신호 전력
        if p_sig <= 0:
            return y
        n = _bandpass(rng.standard_normal(len(y)).astype(np.float32), sr, lo, hi)
        p_n = float(np.mean(n.astype(np.float64) ** 2))
        n = n * np.sqrt((p_sig / (10.0 ** (snr_db / 10.0))) / max(p_n, 1e-20))
    out = y + n.astype(np.float32)
    peak = np.max(np.abs(out))
    if peak > 1.0:                      # 클리핑 방지(조건 간 동일 규칙)
        out = out / peak
    return out.astype(np.float32)


def degrade_file(src, dst, cond, rng):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if cond == "clean":
        y, sr = sf.read(src, dtype="float32")
        sf.write(dst, y, sr, subtype="PCM_16")
        return
    tmp = dst + ".tmp.wav"
    if cond.startswith("g711"):   # g711_bpNN 도 여기 포함된다
        # 8 kHz 로 내린 뒤 G.711 mu-law 로 부호화하고 다시 16 kHz PCM 으로 복원
        _ffmpeg(["-i", src, "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", "-f", "wav", tmp])
        _ffmpeg(["-i", tmp, "-ar", str(C.SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", dst])
        os.remove(tmp)
    elif cond == "nb8k":
        _ffmpeg(["-i", src, "-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le", tmp])
        _ffmpeg(["-i", tmp, "-ar", str(C.SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", dst])
        os.remove(tmp)
    else:
        y, sr = sf.read(src, dtype="float32")
        sf.write(dst, y, sr, subtype="PCM_16")

    if "snr" in cond:
        snr = int(cond.split("snr")[1])
        y, sr = sf.read(dst, dtype="float32")
        sf.write(dst, add_noise(y, snr, rng), sr, subtype="PCM_16")
    elif "_bp" in cond:                       # 통과대역 제한 잡음
        snr = int(cond.split("_bp")[1])
        y, sr = sf.read(dst, dtype="float32")
        sf.write(dst, add_noise(y, snr, rng, band=TEL_BAND), sr, subtype="PCM_16")


def build_condition(te, cond, seed):
    """조건별 열화 wav 를 만들고 해당 경로로 바꾼 manifest 를 돌려준다.

    캐시 경로에 시드를 포함한다. 시드마다 잡음 실현을 새로 뽑아야 ± 가 학습 분산뿐 아니라
    잡음 실현 분산까지 담기 때문이며, 예전에는 한 시드가 만든 파일을 나머지가 그대로 썼다.
    """
    rng = np.random.default_rng(seed)
    out = te.copy()
    paths = []
    for fp in te.filepath:
        dst = os.path.join(WORK, f"s{seed}", cond, os.path.basename(fp))
        if not os.path.exists(dst):
            degrade_file(fp, dst, cond, rng)
        paths.append(dst)
    out["filepath"] = paths
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    _c.set_determinism(args.seed)
    device = get_device()
    m = _c.build_manifest(drop_outliers=True)
    _, _, te, _ = split_manifest(m, seed=42)

    model, ch = build_model("simple_cnn")
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(HERE, args.ckpt)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()

    rows = []
    for cond in CONDITIONS:
        dfc = build_condition(te, cond, args.seed)
        p = predict_clip_probs(model, dfc, ch, device)
        y = dfc.label.values
        met = classification_metrics(y, p, 0.5)
        rows.append({
            "condition": cond, "f1": met["f1"], "recall": met["recall"],
            "precision": met["precision"], "roc_auc": met["roc_auc"], "pr_auc": met["pr_auc"],
            "hard_neg_fpr": hard_neg_fpr(p, dfc.source_type.values, 0.5),
        })
        _c.save_scores(f"scores_degrade{args.suffix}_{cond}_s{args.seed}.csv", dfc, p)
        print("[%-11s] F1=%.3f recall=%.3f prec=%.3f roc=%.3f hardFPR=%.3f" % (
            cond, met["f1"], met["recall"], met["precision"], met["roc_auc"],
            rows[-1]["hard_neg_fpr"]))

    _c.save_json(f"degrade{args.suffix}_s{args.seed}.json",
                 {"n_test": int(len(te)), "ckpt": ckpt, "rows": rows})
    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
