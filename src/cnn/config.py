"""중앙 설정 — 경로 / 오디오 파라미터 / 윈도 / split seed.

plan.md §6~§10 의 결정값을 한곳에 모은다.
"""
import os

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
CNN_DIR    = os.path.dirname(os.path.abspath(__file__))      # src/cnn
DATA_DIR   = os.path.join(CNN_DIR, "data")                   # src/cnn/data (controlled)
REPO_DATA  = os.path.normpath(os.path.join(CNN_DIR, "..", "..", "data"))  # repo data/

# controlled (새로 생성)
POSITIVE_DIR = os.path.join(DATA_DIR, "positive")        # label 1
HARD_NEG_DIR = os.path.join(DATA_DIR, "negative_hard")   # label 0 (hard)
# easy negative 는 기존 data/x 재사용
EASY_NEG_DIR = os.path.join(REPO_DATA, "x")              # label 0 (easy)

# naive (통제 전 — Original 평가셋용)
NAIVE_POS_DIR = os.path.join(REPO_DATA, "o", "00001")    # 숫자 그대로 읽은 positive
NAIVE_NEG_DIR = os.path.join(REPO_DATA, "x")             # 일반 문장 negative

# ---------------------------------------------------------------------------
# 오디오 / 멜 스펙트로그램 (plan.md §7)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_MELS      = 64
N_FFT       = 1024
HOP_LENGTH  = 512
POWER       = 2.0

# ---------------------------------------------------------------------------
# 윈도 (하이브리드: train=대칭 random-crop, eval=sliding-window + max) (plan.md §7)
# ---------------------------------------------------------------------------
WINDOW_SEC      = 5.0
WINDOW_SAMPLES  = int(WINDOW_SEC * SAMPLE_RATE)
EVAL_HOP_SEC    = 2.5
EVAL_HOP_SAMPLES = int(EVAL_HOP_SEC * SAMPLE_RATE)
NOISE_PAD_LEVEL = 1e-3   # 무음(0) 대신 저진폭 노이즈로 padding (길이 leakage 방지)

# 멜 시간축 프레임 수 (5초 고정 윈도 기준)
N_FRAMES = 1 + WINDOW_SAMPLES // HOP_LENGTH   # = 157

# ---------------------------------------------------------------------------
# 데이터 구성 (plan.md §3.2 — easy:hard = 7:3, positive:negative = 1:1)
# ---------------------------------------------------------------------------
EASY_NEG_COUNT = 700
HARD_NEG_COUNT = 300   # 실제 생성된 hard negative 수와 일치해야 함

# ---------------------------------------------------------------------------
# split / 재현성 (plan.md §8, §10)
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
SPLIT_RATIO = (0.70, 0.15, 0.15)   # train / val / test

# 통합 manifest 스키마
MANIFEST_COLUMNS = [
    "filepath", "label", "source_type", "speaker_id", "template_id", "duration", "pool",
]
