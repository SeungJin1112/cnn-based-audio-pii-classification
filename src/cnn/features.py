"""Hand-crafted acoustic feature 추출 (plan.md §5 baseline 용).

v1 core feature: duration, RMS, ZCR, spectral centroid (mean/std).
(spectral bandwidth/rolloff/tempo/silence ratio + RF 는 §15 defer.)
파일당 1회만 계산하고 캐시한다(여러 평가셋에서 재사용).
"""
import librosa
import numpy as np

import config as C

FEATURE_NAMES = [
    "duration", "rms_mean", "rms_std",
    "zcr_mean", "zcr_std", "centroid_mean", "centroid_std",
]
DURATION_IDX = FEATURE_NAMES.index("duration")

_cache = {}


def extract_features(filepath):
    if filepath in _cache:
        return _cache[filepath]
    y, _ = librosa.load(filepath, sr=C.SAMPLE_RATE, mono=True)
    dur = len(y) / C.SAMPLE_RATE
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    cen = librosa.feature.spectral_centroid(y=y, sr=C.SAMPLE_RATE)[0]
    feats = np.array([
        dur, float(rms.mean()), float(rms.std()),
        float(zcr.mean()), float(zcr.std()),
        float(cen.mean()), float(cen.std()),
    ], dtype=np.float32)
    _cache[filepath] = feats
    return feats


def feature_matrix(df):
    return np.stack([extract_features(fp) for fp in df["filepath"]])
