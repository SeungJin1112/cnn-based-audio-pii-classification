"""클래스별 log-mel 스펙트로그램 시각화 (모델이 실제로 보는 입력).

논문은 이 그림을 2단 조판의 전폭 float 에 폭 170 mm 로 싣는다. figsize 가 13 in 이면
라벨이 4 pt 로 축소되어 인쇄본에서 읽히지 않으므로, 최종 배치 폭에 맞춘 7.2 in 으로 그리고
글자 크기를 본문과 비슷하게 잡는다. 라벨은 본문 언어에 맞추어 한국어로 쓴다.
"""
import os
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm, rcParams

import config as C
from dataset import wav_to_logmel

DQ = os.path.join(C.CNN_DIR, "data_qwen")
OUT = os.path.join(C.CNN_DIR, "reports", "melspec_examples.png")
# 논문 figs/ 에도 같은 파일을 떨어뜨려 수동 복사 단계를 없앤다.
OUT2 = os.path.join(C.CNN_DIR, "paper", "figs", "melspec_examples.png")

# 한글 라벨용 폰트. 없으면 조용히 기본값으로 남는다(라벨이 네모로 깨지는 것을 막지는 못하므로
# 사용 가능 여부를 출력한다).
for _cand in ("UnDotum", "Baekmuk Dotum", "NanumGothic"):
    if any(f.name == _cand for f in fm.fontManager.ttflist):
        rcParams["font.family"] = _cand
        break
else:
    print("경고: 한글 폰트를 찾지 못했다. 라벨이 깨질 수 있다.")
rcParams["axes.unicode_minus"] = False


def pick(cls, cond=None):
    df = pd.read_csv(os.path.join(DQ, cls, "metadata.csv"))
    if cond is not None:
        df = df[cond(df)]
    r = df.iloc[0]
    return os.path.join(DQ, cls, r["filepath"]), r["text"]


samples = [
    ("전화번호 (positive)",        pick("positive", lambda d: d.pii_type == "phone")),
    ("주민등록번호 (positive)",    pick("positive", lambda d: d.pii_type == "rrn")),
    ("비PII 숫자열 (hard-neg)",    pick("negative_hard")),
    ("숫자 없는 일반 발화",        pick("negative_easy")),
]

fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2))
for ax, (title, (path, text)) in zip(axes.ravel(), samples):
    y, _ = librosa.load(path, sr=C.SAMPLE_RATE, mono=True)
    mel = wav_to_logmel(y[: C.WINDOW_SAMPLES])          # 모델 입력과 동일 변환
    im = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("프레임", fontsize=8); ax.set_ylabel("멜 채널", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.tick_params(labelsize=6)

plt.tight_layout()
for _o in (OUT, OUT2):
    plt.savefig(_o, dpi=220)
    print("saved ->", _o)
