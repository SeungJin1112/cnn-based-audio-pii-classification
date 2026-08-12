"""무학습 시간구조 기준선 — 온셋 검출 → 등간격 음절 런 최대 길이 → 임계값.

기존 기준선(duration-only, acoustic-LR)은 시간 구조가 없는 전역 통계여서,
CNN 이 이를 능가한다는 사실은 "시간 구조가 필요하다"까지만 말한다.
학습 가중치가 전혀 없는 시간구조 기준선을 두면 60K CNN 의 기여가 정량화된다.

점수 정의:
  온셋 시각열에서 인접 간격(IOI)이 음절 속도 대역 안에 있고 서로 등간격인
  최장 연쇄의 길이. 자릿수 낭독은 등간격 단음절 연쇄이므로 이 값이 커진다.

임계값(및 tol/대역)은 검증 분할에서만 고른다. 학습되는 가중치는 없다.

usage:
  python onset_baseline.py --data v2     # data_qwen (v2-HN)
  python onset_baseline.py --data v1     # controlled v1
"""
import argparse
import os
import sys

import librosa
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common as _c  # noqa: E402
import config as C  # noqa: E402
from metrics import classification_metrics, hard_neg_fpr  # noqa: E402
from splits import split_manifest  # noqa: E402

IOI_LO, IOI_HI = 0.08, 0.40          # 음절 속도 대역(초)
TOL_GRID = [0.20, 0.30, 0.40, 0.50]
BAND_GRID = [(0.08, 0.40), (0.10, 0.35), (0.06, 0.50)]


def onset_times(path):
    y, _ = librosa.load(path, sr=C.SAMPLE_RATE, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=C.SAMPLE_RATE, hop_length=C.HOP_LENGTH)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=C.SAMPLE_RATE,
                                        hop_length=C.HOP_LENGTH, backtrack=False)
    return librosa.frames_to_time(frames, sr=C.SAMPLE_RATE, hop_length=C.HOP_LENGTH)


def regular_run_stats(times, tol, band):
    """등간격·음절대역 조건을 만족하는 연쇄들의 통계.

    반환: (최장 연쇄의 온셋 수, 모든 연쇄의 온셋 총합, 전체 온셋 수)
    '최장 런'만 쓰면 자릿수 그룹 사이 휴지에서 연쇄가 끊겨 카드 4-4-4-4 형이
    과소평가되므로, 총합 변형을 함께 두어 기준선을 불리하게 잡지 않는다.
    """
    n_all = len(times)
    if n_all < 3:
        return 0, 0, n_all
    iois = np.diff(times)
    ok = (iois >= band[0]) & (iois <= band[1])
    best = total = 0
    run = []

    def flush(r):
        return len(r) + 1 if r else 0

    for i, good in enumerate(ok):
        if not good:
            total += flush(run)
            best = max(best, flush(run))
            run = []
            continue
        cand = run + [iois[i]]
        med = float(np.median(cand))
        if med > 0 and np.all(np.abs(np.array(cand) - med) / med <= tol):
            run = cand
        else:
            total += flush(run)
            best = max(best, flush(run))
            run = [iois[i]]
    total += flush(run)
    best = max(best, flush(run))
    return best, total, n_all


STATS = ["longest_run", "total_regular_onsets", "n_onsets"]


def score_frame(df, cache, tol, band, stat="longest_run"):
    idx = STATS.index(stat)
    return np.array([regular_run_stats(cache[fp], tol, band)[idx] for fp in df.filepath],
                    dtype=float)


def evaluate(df, s, thr):
    y = df.label.values
    met = classification_metrics(y, s, thr)
    met["hard_neg_fpr"] = hard_neg_fpr(s, df.source_type.values, thr)
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2", choices=["v1", "v2"])
    args = ap.parse_args()

    if args.data == "v2":
        m = _c.build_manifest(drop_outliers=True)
    else:
        from dataset import build_controlled_manifest
        m = build_controlled_manifest(seed=42)
    tr, va, te, _ = split_manifest(m, seed=42)

    files = sorted(set(m.filepath))
    print(f"온셋 추출 {len(files)}개 ...")
    cache = {}
    for i, fp in enumerate(files):
        cache[fp] = onset_times(fp)
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(files)}")

    # 검증 분할에서 (통계량, tol, band, threshold) 를 고른다 — test 는 보지 않는다.
    best = None
    for stat in STATS:
        # n_onsets 는 tol/band 와 무관하므로 한 조합만 본다.
        combos = [(TOL_GRID[0], BAND_GRID[0])] if stat == "n_onsets" else \
                 [(t, b) for t in TOL_GRID for b in BAND_GRID]
        for tol, band in combos:
            sv = score_frame(va, cache, tol, band, stat)
            for thr in range(2, 40):
                f1 = classification_metrics(va.label.values, sv, thr)["f1"]
                if best is None or f1 > best["val_f1"]:
                    best = {"stat": stat, "tol": tol, "band": list(band),
                            "threshold": thr, "val_f1": f1}
    print("val 최적:", best)

    band = tuple(best["band"])
    st = score_frame(te, cache, best["tol"], band, best["stat"])
    met = evaluate(te, st, best["threshold"])
    met["roc_auc"] = float(classification_metrics(te.label.values, st)["roc_auc"])

    # 통계량 세 종을 모두 보고해 기준선이 불리하게 설정되지 않았음을 보인다.
    per_stat = {}
    for stat in STATS:
        s_va = score_frame(va, cache, best["tol"], band, stat)
        thr = max(range(2, 40),
                  key=lambda t: classification_metrics(va.label.values, s_va, t)["f1"])
        s_te = score_frame(te, cache, best["tol"], band, stat)
        m = evaluate(te, s_te, thr)
        m["roc_auc"] = float(classification_metrics(te.label.values, s_te)["roc_auc"])
        m["threshold"] = thr
        per_stat[stat] = m
        print("  %-22s thr=%2d  F1=%.3f rec=%.3f prec=%.3f roc=%.3f hardFPR=%.3f" % (
            stat, thr, m["f1"], m["recall"], m["precision"], m["roc_auc"], m["hard_neg_fpr"]))

    res = {"data": args.data, "n_test": int(len(te)), "selection": best,
           "test_best": met, "per_stat": per_stat,
           "score_summary": {
               "positive_median": float(np.median(st[te.label.values == 1])),
               "hard_median": float(np.median(st[(te.source_type == "negative_hard").values])),
               "easy_median": (float(np.median(st[(te.source_type == "negative_easy").values]))
                               if (te.source_type == "negative_easy").any() else None)}}
    _c.save_json(f"onset_baseline_{args.data}.json", res)
    print("[onset %s / %s] F1=%.3f recall=%.3f prec=%.3f roc=%.3f hardFPR=%.3f" % (
        args.data, best["stat"], met["f1"], met["recall"], met["precision"], met["roc_auc"],
        met["hard_neg_fpr"]))

    _c.save_scores(f"scores_onset_{args.data}_test.csv", te, st)


if __name__ == "__main__":
    main()
