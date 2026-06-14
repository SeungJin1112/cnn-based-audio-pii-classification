"""데이터셋 — manifest 병합 + wav→log-mel + 하이브리드 윈도 + 평가셋 구성.

plan.md 반영:
  §3.2  positive 1000 + easy 700(data/x 재사용) + hard 300, 1:1 균형
  §4    5개 평가셋: Original / Controlled-main / Length-matched / Hard-negative-only / Stress-5:5
  §7    log-mel(n_mels=64, n_fft=1024, hop=512), 대칭 random-crop(train) /
        sliding-window+max(eval), 무음 대신 저진폭 노이즈 padding, per-sample 정규화
"""
import hashlib
import os
import re

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as C


# ===========================================================================
# 1. manifest 병합
# ===========================================================================
def _derive_template_id(text, prefix):
    """template_id 컬럼이 없는 소스(easy/naive)는 문장 골격으로 파생.

    숫자/공백 제거 후 해시 → 동일/중복 문장은 같은 id 를 받아
    splits.py 의 template-disjoint 가 그 블록을 분리할 수 있게 한다.
    """
    skeleton = re.sub(r"\s+", "", re.sub(r"\d+", "", str(text)))
    return f"{prefix}_" + hashlib.md5(skeleton.encode("utf-8")).hexdigest()[:8]


def _load_source(meta_dir, label, source_type, pool, sample=None, seed=42):
    """한 소스의 metadata.csv 를 통합 스키마로 로드.

    speaker_id 가 없으면 voice 컬럼으로, template_id 가 없으면 text 골격 해시로 보정.
    """
    df = pd.read_csv(os.path.join(meta_dir, "metadata.csv"))
    abs_paths = [
        os.path.normpath(os.path.join(meta_dir, str(fp).replace("\\", "/")))
        for fp in df["filepath"]
    ]
    if "template_id" in df:
        template_id = df["template_id"]
    elif "text" in df:
        template_id = df["text"].map(lambda t: _derive_template_id(t, source_type))
    else:
        template_id = source_type
    out = pd.DataFrame({
        "filepath":    abs_paths,
        "label":       label,
        "source_type": source_type,
        "speaker_id":  df["speaker_id"] if "speaker_id" in df else df.get("voice", "unknown"),
        "template_id": template_id,
        "duration":    df["duration"] if "duration" in df else np.nan,
        "pool":        pool,
    })
    if sample is not None and len(out) > sample:
        out = out.sample(n=sample, random_state=seed).reset_index(drop=True)
    return out


def _ensure_duration(df):
    """duration 결측이면 wav 헤더에서 채운다 (naive/easy 소스 대비)."""
    import soundfile as sf
    missing = df["duration"].isna()
    if missing.any():
        durs = []
        for fp in df.loc[missing, "filepath"]:
            info = sf.info(fp)
            durs.append(round(info.frames / info.samplerate, 3))
        df.loc[missing, "duration"] = durs
    return df


def build_controlled_manifest(seed=42):
    """주 실험용 통제 manifest: positive 1000 + easy 700 + hard 300."""
    pos  = _load_source(C.POSITIVE_DIR, 1, "positive_controlled", "controlled")
    hard = _load_source(C.HARD_NEG_DIR, 0, "negative_hard",       "controlled")
    easy = _load_source(C.EASY_NEG_DIR, 0, "negative_easy",       "controlled",
                        sample=C.EASY_NEG_COUNT, seed=seed)
    df = pd.concat([pos, easy, hard], ignore_index=True)
    return _ensure_duration(df)


def build_original_manifest():
    """통제 전(naive) 평가용: 00001 positive vs data/x negative.

    주의(검증 에이전트 지적): data/x 는 controlled easy negative 와 동일 소스다.
    controlled-train 에 들어간 data/x 파일이 Original 평가에 다시 등장하면 누수다.
    → splits.py 에서 controlled-train split 의 filepath 를 Original manifest 에서
      제외해 Original 을 깨끗한 평가 조건으로 유지한다. (TODO: splits.py)
    """
    pos = _load_source(C.NAIVE_POS_DIR, 1, "positive_naive", "original")
    neg = _load_source(C.NAIVE_NEG_DIR, 0, "negative_naive", "original")
    df = pd.concat([pos, neg], ignore_index=True)
    return _ensure_duration(df)


# ===========================================================================
# 2. 평가셋 구성 (plan.md §4) — test manifest 를 조건별로 파생
# ===========================================================================
def make_eval_sets(test_df, seed=42):
    """controlled test 셋에서 조건별 평가셋을 파생.

    반환 dict:
      controlled_main     : test 그대로 (easy:hard = 7:3)
      length_matched      : positive/negative 길이 분포를 맞춘 subset
      hard_negative_only  : positive vs hard negative 만
      stress_5_5          : easy:hard = 5:5 로 재가중
    (original 은 별도 manifest 이므로 evaluate 단계에서 따로 다룬다)
    """
    rng = np.random.default_rng(seed)
    pos  = test_df[test_df.label == 1]
    easy = test_df[(test_df.label == 0) & (test_df.source_type == "negative_easy")]
    hard = test_df[(test_df.label == 0) & (test_df.source_type == "negative_hard")]

    sets = {}
    sets["controlled_main"] = test_df.reset_index(drop=True)

    # hard-negative-only: positive vs hard
    sets["hard_negative_only"] = pd.concat([pos, hard], ignore_index=True)

    # stress 5:5 — negative 내부 easy:hard 를 1:1 로 재가중 (positive 는 전부 유지).
    # → positive:negative 균형은 깨지므로, 보고 시 threshold-독립 지표(ROC/PR-AUC)와
    #   per-class 지표(hard-neg FPR 등) 중심으로 해석한다.
    n = min(len(easy), len(hard))
    if n > 0:
        easy_s = easy.sample(n=n, random_state=seed)
        hard_s = hard.sample(n=n, random_state=seed)
        sets["stress_5_5"] = pd.concat([pos, easy_s, hard_s], ignore_index=True)
    else:
        sets["stress_5_5"] = test_df.reset_index(drop=True)

    # length-matched — positive 길이 분포에 맞춰 negative 를 길이 bin 단위로 매칭
    sets["length_matched"] = _length_match(pos, pd.concat([easy, hard]), rng)
    return sets


def _length_match(pos, neg, rng, n_bins=10):
    """positive 길이 히스토그램에 맞춰 negative 를 bin 별로 샘플링."""
    if len(pos) == 0 or len(neg) == 0:
        return pd.concat([pos, neg], ignore_index=True)
    lo = min(pos.duration.min(), neg.duration.min())
    hi = max(pos.duration.max(), neg.duration.max())
    bins = np.linspace(lo, hi + 1e-6, n_bins + 1)
    pos_b = np.digitize(pos.duration, bins)
    neg_b = np.digitize(neg.duration, bins)
    kept_neg = []
    for b in np.unique(pos_b):
        want = int((pos_b == b).sum())
        cand = neg[neg_b == b]
        if len(cand) == 0:
            continue
        take = min(want, len(cand))
        kept_neg.append(cand.sample(n=take, random_state=int(rng.integers(1 << 30))))
    neg_matched = pd.concat(kept_neg, ignore_index=True) if kept_neg else neg.iloc[:0]
    return pd.concat([pos, neg_matched], ignore_index=True)


# ===========================================================================
# 3. 오디오 → log-mel
# ===========================================================================
def wav_to_logmel(y):
    S = librosa.feature.melspectrogram(
        y=y, sr=C.SAMPLE_RATE, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, power=C.POWER,
    )
    S = librosa.power_to_db(S, ref=np.max)          # log scale
    S = (S - S.mean()) / (S.std() + 1e-6)           # per-sample 정규화
    return S.astype(np.float32)                     # [n_mels, T]


def _to_tensor(mel, channels):
    t = torch.from_numpy(mel).unsqueeze(0)          # [1, n_mels, T]
    if channels == 3:
        t = t.repeat(3, 1, 1)
    return t


def _noise_pad(y, target_len, rng):
    """target_len 보다 짧으면 저진폭 노이즈로 우측 padding (무음 leakage 방지)."""
    if len(y) >= target_len:
        return y
    pad = rng.standard_normal(target_len - len(y)).astype(np.float32) * C.NOISE_PAD_LEVEL
    return np.concatenate([y, pad])


def random_crop(y, rng):
    """대칭 random-crop: 길면 무작위 위치 crop, 짧으면 노이즈 padding."""
    L = C.WINDOW_SAMPLES
    if len(y) > L:
        start = int(rng.integers(0, len(y) - L + 1))
        return y[start:start + L]
    return _noise_pad(y, L, rng)


def spec_augment(mel, rng, max_freq=8, max_time=16):
    """SpecAugment (train 전용, 보수적): freq/time 마스크 1개씩 0(≈정규화 평균)으로."""
    mel = mel.copy()
    f = int(rng.integers(0, max_freq + 1))
    if f > 0 and mel.shape[0] - f > 0:
        f0 = int(rng.integers(0, mel.shape[0] - f))
        mel[f0:f0 + f, :] = 0.0
    t = int(rng.integers(0, max_time + 1))
    if t > 0 and mel.shape[1] - t > 0:
        t0 = int(rng.integers(0, mel.shape[1] - t))
        mel[:, t0:t0 + t] = 0.0
    return mel


def sliding_windows(y, rng):
    """eval: 전체 clip 을 sliding window 로 분할 (각 WINDOW_SAMPLES)."""
    L, hop = C.WINDOW_SAMPLES, C.EVAL_HOP_SAMPLES
    if len(y) <= L:
        return [_noise_pad(y, L, rng)]
    starts = list(range(0, len(y) - L + 1, hop))
    if starts[-1] != len(y) - L:
        starts.append(len(y) - L)                   # 마지막 구간 포함
    return [y[s:s + L] for s in starts]


# ===========================================================================
# 4. Dataset
# ===========================================================================
class AudioPIIDataset(Dataset):
    """mode='train' → 단일 random-crop 윈도 [C, n_mels, T] 반환.
    mode='eval'  → clip 의 모든 sliding window 를 [n_win, C, n_mels, T] 로 반환
                   (clip-level score = max(window scores) 용).
    """

    def __init__(self, manifest_df, mode="train", channels=1, seed=42, augment=False):
        assert mode in ("train", "eval")
        self.df = manifest_df.reset_index(drop=True)
        self.mode = mode
        self.channels = channels
        self.augment = augment and mode == "train"
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        y, _ = librosa.load(row["filepath"], sr=C.SAMPLE_RATE, mono=True)
        label = int(row["label"])

        if self.mode == "train":
            seg = random_crop(y, self.rng)
            mel = wav_to_logmel(seg)
            if self.augment:
                mel = spec_augment(mel, self.rng)
            x = _to_tensor(mel, self.channels)
            return x, label

        wins = sliding_windows(y, self.rng)
        xs = torch.stack([_to_tensor(wav_to_logmel(w), self.channels) for w in wins])
        return xs, label, i   # [n_win, C, n_mels, T], label, clip index


# ===========================================================================
# smoke test
# ===========================================================================
if __name__ == "__main__":
    print("== build controlled manifest ==")
    m = build_controlled_manifest(seed=42)
    print("총", len(m), "| label 분포:", dict(m.label.value_counts()))
    print("source_type 분포:", dict(m.source_type.value_counts()))
    print("pos dur mean=%.2f | neg dur mean=%.2f" %
          (m[m.label == 1].duration.mean(), m[m.label == 0].duration.mean()))

    print("\n== train sample shape ==")
    ds_tr = AudioPIIDataset(m.sample(20, random_state=0), mode="train", channels=1)
    x, lab = ds_tr[0]
    print("train tensor:", tuple(x.shape), "label:", lab, "(기대 [1,64,157])")
    x3, _ = AudioPIIDataset(m, mode="train", channels=3)[0]
    print("3채널:", tuple(x3.shape), "(기대 [3,64,157])")

    print("\n== eval sample (sliding window) ==")
    ds_ev = AudioPIIDataset(m.sample(5, random_state=1), mode="eval", channels=1)
    xs, lab, idx = ds_ev[0]
    print("eval tensor:", tuple(xs.shape), "label:", lab, "(n_win 가변, 각 [1,64,157])")

    print("\n== eval sets ==")
    sets = make_eval_sets(m, seed=42)   # smoke 용: 전체 manifest 를 test 로 가정
    for name, s in sets.items():
        bal = dict(s.label.value_counts())
        print(f"  {name:20s} n={len(s):5d}  label={bal}")

    print("\n== original manifest ==")
    o = build_original_manifest()
    print("총", len(o), "| label 분포:", dict(o.label.value_counts()))
    print("\nSMOKE OK")
