"""viz 에 쓴 4개 샘플을 학습에서 제외하고 학습 후 실제 예측 (정직한 확인)."""
import os
import numpy as np
import pandas as pd

import config as C
from engine import get_device, predict_clip_probs, train_model
from models import build_model
from speaker_pii_experiment import build_qwen_manifest
from splits import split_manifest

DQ = os.path.join(C.CNN_DIR, "data_qwen")


def first_path(cls, cond=None):
    df = pd.read_csv(os.path.join(DQ, cls, "metadata.csv"))
    if cond is not None:
        df = df[cond(df)]
    r = df.iloc[0]
    return os.path.normpath(os.path.join(DQ, cls, r["filepath"]))


targets = {
    "phone(좌상,PII=1)":   first_path("positive", lambda d: d.pii_type == "phone"),
    "rrn(우상,PII=1)":     first_path("positive", lambda d: d.pii_type == "rrn"),
    "hardneg(좌하,0)":     first_path("negative_hard"),
    "easy(우하,0)":        first_path("negative_easy"),
}

m = build_qwen_manifest()
tset = set(targets.values())
test_df = m[m.filepath.isin(tset)].reset_index(drop=True)        # 그 4개
train_pool = m[~m.filepath.isin(tset)].reset_index(drop=True)    # 나머지 전부
tr, va, te, _ = split_manifest(train_pool, seed=42)
tr = pd.concat([tr, te], ignore_index=True)

device = get_device()
model, ch = build_model("simple_cnn")
print("[train] 4개 제외하고 simple_cnn 학습...")
train_model(model, tr, va, ch, device, epochs=40, seed=42)

probs = predict_clip_probs(model, test_df, ch, device, seed=42)
p_by_path = dict(zip(test_df.filepath, probs))
print("\n== 4개 샘플 예측 (학습에서 제외됨, threshold 0.5) ==")
for name, path in targets.items():
    p = p_by_path[path]
    pred = "PII(1)" if p >= 0.5 else "non-PII(0)"
    print(f"  {name:20s} P(PII)={p:.3f} -> {pred}")
