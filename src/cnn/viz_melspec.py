"""클래스별 log-mel 스펙트로그램 시각화 (모델이 실제로 보는 입력)."""
import os
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from dataset import wav_to_logmel

DQ = os.path.join(C.CNN_DIR, "data_qwen")
OUT = os.path.join(C.CNN_DIR, "reports", "melspec_examples.png")


def pick(cls, cond=None):
    df = pd.read_csv(os.path.join(DQ, cls, "metadata.csv"))
    if cond is not None:
        df = df[cond(df)]
    r = df.iloc[0]
    return os.path.join(DQ, cls, r["filepath"]), r["text"]


samples = [
    ("Phone number  (PII, label=1)",      pick("positive", lambda d: d.pii_type == "phone")),
    ("Resident reg. no.  (PII, label=1)", pick("positive", lambda d: d.pii_type == "rrn")),
    ("Hard-neg digit string  (label=0)",  pick("negative_hard")),
    ("General speech  (label=0)",         pick("negative_easy")),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, (title, (path, text)) in zip(axes.ravel(), samples):
    y, _ = librosa.load(path, sr=C.SAMPLE_RATE, mono=True)
    mel = wav_to_logmel(y[: C.WINDOW_SAMPLES])          # 모델 입력과 동일 변환
    im = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("time frame"); ax.set_ylabel("mel bin (64)")
    fig.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("Log-Mel Spectrograms (model input, 64 mel x time)", fontsize=12)

plt.tight_layout()
plt.savefig(OUT, dpi=110)
print("saved ->", OUT)
