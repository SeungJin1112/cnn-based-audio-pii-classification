"""Split — template-disjoint + length-stratified + label-stratified (plan.md §8).

- 같은 template_id 는 한 split 에만 들어간다(template-disjoint).
- label 별로 70/15/15 목표를 맞춘다(label-stratified).
- 그룹을 길이(mean duration) 순으로 정렬해 부족분이 큰 split 에 배정 → 길이 분포 분산.
- speaker-disjoint 는 적용하지 않는다(화자 2~3명, 통제 불가 confound로 명시).
- Original 평가셋은 controlled-train 과 겹치는 파일을 제외해 누수를 막는다.
"""
import numpy as np
import pandas as pd

import config as C
from dataset import build_original_manifest


def split_manifest(df, ratio=C.SPLIT_RATIO, seed=42):
    """template-disjoint + label-stratified + length-aware split.

    반환: (train_df, val_df, test_df, assign)  assign: template_id -> split
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    parts = {"train": [], "val": [], "test": []}
    assign = {}

    for label in sorted(df.label.unique()):
        sub = df[df.label == label]
        # 그룹(template_id) 단위 정보
        ginfo = [
            (tid, g.index.tolist(), float(g.duration.mean()), len(g))
            for tid, g in sub.groupby("template_id")
        ]
        rng.shuffle(ginfo)                       # 동률 tie-break 랜덤화
        ginfo.sort(key=lambda x: x[2])           # 길이순 정렬

        total = len(sub)
        target = {"train": ratio[0] * total, "val": ratio[1] * total, "test": ratio[2] * total}
        cur = {"train": 0, "val": 0, "test": 0}

        # 길이순으로 돌며 "비례 충원(cur/target 최소)" split 에 배정 →
        # 세 split 이 길이 스펙트럼 전 구간을 고루 나눠 가짐(length-stratified).
        for tid, idxs, _mdur, sz in ginfo:
            s = min(("train", "val", "test"), key=lambda k: cur[k] / target[k] if target[k] > 0 else 1e9)
            assign[tid] = s
            cur[s] += sz
            parts[s].extend(idxs)

    train = df.loc[parts["train"]].reset_index(drop=True)
    val   = df.loc[parts["val"]].reset_index(drop=True)
    test  = df.loc[parts["test"]].reset_index(drop=True)
    return train, val, test, assign


def build_original_eval(controlled_train_df):
    """Original 평가셋: controlled-train 에 쓰인 파일은 제외(누수 방지, plan.md §4)."""
    o = build_original_manifest()
    train_paths = set(controlled_train_df["filepath"])
    return o[~o["filepath"].isin(train_paths)].reset_index(drop=True)


# ===========================================================================
# smoke test
# ===========================================================================
if __name__ == "__main__":
    from dataset import build_controlled_manifest

    m = build_controlled_manifest(seed=42)
    tr, va, te, assign = split_manifest(m, seed=42)
    n = len(m)
    print("== split sizes ==")
    for name, d in [("train", tr), ("val", va), ("test", te)]:
        bal = dict(d.label.value_counts())
        print(f"  {name:5s} n={len(d):4d} ({len(d)/n:.1%})  label={bal}  dur_mean={d.duration.mean():.2f}")

    print("\n== template-disjoint 검증 ==")
    sets = {k: set(d.template_id) for k, d in [("train", tr), ("val", va), ("test", te)]}
    overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
    print(f"  split 간 공유 template_id: {len(overlap)}개 (0이어야 함)")

    print("\n== label 별 비율 (stratified) ==")
    for label in [0, 1]:
        tot = (m.label == label).sum()
        row = {nm: (d.label == label).sum() for nm, d in [("train", tr), ("val", va), ("test", te)]}
        print(f"  label={label}: " + " ".join(f"{k}={v}({v/tot:.0%})" for k, v in row.items()))

    print("\n== Original 누수 제외 ==")
    o_full = build_original_manifest()
    o_eval = build_original_eval(tr)
    print(f"  Original 전체 {len(o_full)} → train 제외 후 {len(o_eval)} (제외 {len(o_full)-len(o_eval)}개)")
    print("\nSMOKE OK")
