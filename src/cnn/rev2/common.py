"""feedback2 대응 실험 공통 모듈.

- 결정성 고정(시드·cuDNN·DataLoader)
- v2/v3 manifest 빌드 (초장 클립 제외 옵션 포함)
- 텍스트에서 자릿수 낭독 구조(런 길이·총 자릿수) 복원
- 캐리어 문장 유무 판정
- 클립별 점수 저장/적재

기존 실험 코드(dataset/engine/models/metrics/splits)를 그대로 재사용한다.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

CNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CNN_DIR not in sys.path:
    sys.path.insert(0, CNN_DIR)

import config as C  # noqa: E402

REV2_DIR = os.path.join(CNN_DIR, "rev2")
OUT_DIR = os.path.join(REV2_DIR, "out")
DATA_QWEN = os.path.join(CNN_DIR, "data_qwen")

# 655초짜리 TTS 반복 생성 클립을 제외하는 기준(정상 클립 최대 길이는 10초 미만).
OUTLIER_DURATION_SEC = 60.0


# ===========================================================================
# 결정성
# ===========================================================================
def set_determinism(seed):
    """재학습 시 비트 단위 재현을 목표로 난수원과 커널 선택을 고정한다."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # AdaptiveAvgPool2d 역전파처럼 결정성 커널이 없는 연산이 남을 수 있으므로
    # 하드 실패 대신 경고로 두고, 동일 시드 2회 실행으로 실제 일치를 검증한다.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    ws = torch.initial_seed() % (2 ** 32)
    np.random.seed(ws)
    import random as _random
    _random.seed(ws)


def loader_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ===========================================================================
# manifest
# ===========================================================================
def build_manifest(root=None, drop_outliers=True):
    """data_qwen(또는 동일 스키마 디렉터리)의 세 클래스를 하나의 manifest 로 합친다."""
    root = root or DATA_QWEN
    parts = []
    for cls in ["positive", "negative_hard", "negative_easy"]:
        md = os.path.join(root, cls, "metadata.csv")
        df = pd.read_csv(md)
        df["filepath"] = [os.path.normpath(os.path.join(root, cls, str(fp).replace("\\", "/")))
                          for fp in df["filepath"]]
        for col in ["pii_type", "value", "text"]:
            if col not in df:
                df[col] = ""
            df[col] = df[col].fillna("")
        parts.append(df[["filepath", "label", "source_type", "speaker_id", "template_id",
                         "duration", "pii_type", "value", "text"]])
    m = pd.concat(parts, ignore_index=True)
    m["is_outlier"] = m.duration > OUTLIER_DURATION_SEC
    if drop_outliers:
        m = m[~m.is_outlier].reset_index(drop=True)
    return m


# ===========================================================================
# 자릿수 낭독 구조 복원
# ===========================================================================
DIGIT_SYL = "공일이삼사오육칠팔구"
D = f"[{DIGIT_SYL}]"

CARRIER_LEAD = ["안내 말씀드립니다.", "잠시 안내드립니다.", "확인을 위해 알려드립니다.",
                "고객님께 안내드립니다."]
CARRIER_TAIL = ["확인 부탁드립니다.", "참고 부탁드립니다.", "양해 부탁드립니다.", "감사합니다."]

# hard-negative 카테고리별 원문 골격(generate_qwen.py 의 생성기와 1:1 대응).
HARD_PATTERNS = {
    "date":    re.compile(rf"오늘은 ({D}+)년 ({D}+)월 ({D}+)일입니다\."),
    "order":   re.compile(rf"주문번호는 ({D}+)입니다\."),
    "invoice": re.compile(rf"송장번호는 ({D}+) ({D}+)입니다\."),
    "price":   re.compile(rf"가격은 ({D}+)원입니다\."),
    "postal":  re.compile(rf"우편번호는 ({D}+)입니다\."),
    "seat":    re.compile(rf"좌석은 ({D}+)열 ({D}+)번입니다\."),
    "serial":  re.compile(rf"제품 일련번호는 ({D}+) ({D}+) ({D}+)입니다\."),
    "birth":   re.compile(rf"생일은 ({D}+)월 ({D}+)일이에요\."),
    "bus":     re.compile(rf"({D}+)번 버스를 타시면 됩니다\."),
    "count":   None,   # 아라비아 숫자 없음
}


def strip_carrier(text):
    """캐리어 문장을 떼어내고 (핵심 문장, 캐리어 유무)를 돌려준다."""
    t = text.strip()
    for lead in CARRIER_LEAD:
        if t.startswith(lead):
            return t[len(lead):].strip(), True
    for tail in CARRIER_TAIL:
        if t.endswith(tail):
            return t[: -len(tail)].strip(), True
    return t, False


def has_carrier(text):
    return strip_carrier(text)[1]


def digit_runs(row):
    """클립 하나의 '연속 자릿수 런' 길이 목록.

    positive 는 value(구분자 포함 원본)에서, hard-negative 는 카테고리별 정규식에서,
    easy-negative 는 빈 목록으로 복원한다. 복원 실패 시 None 을 돌려주어
    호출부가 커버리지를 검증할 수 있게 한다.
    """
    st = row["source_type"]
    if st == "negative_easy":
        return []
    if st == "positive":
        v = str(row["value"])
        return [len(g) for g in re.split(r"[-\s]", v) if g]
    cat = str(row["template_id"])
    if cat == "count":
        return []
    pat = HARD_PATTERNS.get(cat)
    if pat is None:
        return None
    core, _ = strip_carrier(str(row["text"]))
    m = pat.search(core)
    if m is None:
        return None
    return [len(g) for g in m.groups()]


def digit_features(df):
    """manifest 에 자릿수 구조 컬럼(max_run/total_digits/n_groups)을 추가한다."""
    df = df.copy()
    runs = [digit_runs(r) for _, r in df.iterrows()]
    unresolved = [i for i, r in enumerate(runs) if r is None]
    if unresolved:
        ex = df.iloc[unresolved[:3]][["source_type", "template_id", "text"]].to_dict("records")
        raise ValueError(f"자릿수 구조 복원 실패 {len(unresolved)}건. 예: {ex}")
    df["runs"] = ["-".join(str(x) for x in r) for r in runs]
    df["max_run"] = [max(r) if r else 0 for r in runs]
    df["total_digits"] = [sum(r) for r in runs]
    df["n_groups"] = [len(r) for r in runs]
    df["has_carrier"] = [has_carrier(str(t)) for t in df["text"]]
    return df


# ===========================================================================
# 산출물 저장
# ===========================================================================
def save_json(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"saved -> {p}")
    return p


def save_scores(name, df, probs, extra=None):
    """클립별 점수를 CSV 로 남긴다(조건부 FPR·부트스트랩·곡선 분석에 재사용)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out = df.copy()
    out["prob"] = np.asarray(probs, dtype=float)
    if extra:
        for k, v in extra.items():
            out[k] = v
    p = os.path.join(OUT_DIR, name)
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"saved -> {p}  (n={len(out)})")
    return p


def load_scores(name):
    return pd.read_csv(os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    m = build_manifest(drop_outliers=False)
    print("manifest n=%d (초장 클립 %d개)" % (len(m), int(m.is_outlier.sum())))
    f = digit_features(m)
    print("\n== source_type x 자릿수 구조 ==")
    print(f.groupby(["source_type", "template_id"])
           .agg(n=("runs", "size"), runs=("runs", lambda s: s.mode().iloc[0]),
                max_run=("max_run", "median"), total=("total_digits", "median"))
           .to_string())
    print("\n== 캐리어 결합률 ==")
    print(f.groupby("source_type").has_carrier.mean().to_string())
    print("\nOK")
