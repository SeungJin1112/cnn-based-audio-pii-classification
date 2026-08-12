"""음향 경로와 STT+NLP 경로의 결합 방식 비교 — 캐스케이드 대 합집합.

\ref{sec:exp:decision} 절은 음향 모델의 결정 변수가 자릿수 낭독의 분량이지 PII 형식이
아님을 보였다. 그렇다면 이 모델을 "PII 후보 선별기"로 쓰는 캐스케이드는 전제가 어긋난다.
반면 \ref{sec:exp}장의 유형별 재현율은 두 경로의 실패가 반대 방향임을 보인다 — ASR 은
자릿수가 길수록 무너지고(카드 16자리 0.864) 음향 모델은 길수록 잘한다(0.988).

이 스크립트는 클립 단위 판정을 실제로 결합해 네 설계를 같은 인스턴스에서 비교한다.

  STT 단독        전사 후 규칙 탐지
  CNN 단독        음향 점수 >= tau
  캐스케이드 A∧B   음향이 통과시킨 클립만 전사해 확정 (현재 논문 제안, 고정밀)
  합집합   A∨B    둘 중 하나라도 켜지면 positive (재현율 안전망)

전사는 whisper_configs.py 가 out/tx_{조건}_{설정}.csv 에 캐시해 둔 것을 재사용하므로
GPU 가 필요 없다. 음향 점수는 out/scores_*_test.csv 의 클립 확률을 쓴다.

usage:
  python union_eval.py --cnn gru_cnn --conds clean g711 g711_bp20 g711_bp10 g711_bp05
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "stt"))

import common as _c  # noqa: E402
from metrics import classification_metrics, hard_neg_fpr  # noqa: E402
from nlp_pii_detector import detect_format, detect_format_context  # noqa: E402
from splits import split_manifest  # noqa: E402

SEEDS = (42, 43, 44)
# 음향 점수 파일 이름 규칙. simple_cnn 은 논문 주 결과의 산출물을 그대로 쓴다.
SCORE_PATTERN = {"simple_cnn": "scores_indist_s{s}_test.csv"}


def cnn_probs(cnn, cond, seed):
    """조건별 음향 점수. clean 은 in-distribution 산출물, 열화 조건은 degrade 산출물."""
    if cond == "clean":
        name = SCORE_PATTERN.get(cnn, "scores_seq_indist_" + cnn + "_s{s}_test.csv").format(s=seed)
    else:
        # 열화 평가는 simple_cnn 으로만 수행되어 있다.
        name = f"scores_degrade_{cond}_s{seed}.csv"
    path = os.path.join(_c.OUT_DIR, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # 열화 조건의 점수 파일은 열화 wav 경로로 저장되므로 파일명으로 맞춘다.
    return {os.path.basename(k): v for k, v in zip(df.filepath, df.prob)}


def stt_pred(cond, config, te, variant="format_context"):
    path = os.path.join(_c.OUT_DIR, f"tx_{cond}_{config}.csv")
    if not os.path.exists(path):
        return None
    tx = pd.read_csv(path)
    txt = dict(zip(tx.filepath, tx.transcript.fillna("")))
    fn = detect_format_context if variant == "format_context" else detect_format
    # 전사 캐시는 열화 파일 경로로 저장되어 있으므로 파일명으로 맞춘다.
    by_name = {os.path.basename(k): v for k, v in txt.items()}
    out = []
    for fp in te.filepath:
        t = by_name.get(os.path.basename(fp))
        out.append(0.0 if t is None else float(fn(t)[0]))
    return np.asarray(out)


def block(y, pred, src, te, n_pos):
    m = classification_metrics(y, pred, 0.5)
    per = {}
    for pt in ["rrn", "account", "card"]:
        sel = (te.pii_type.values == pt) & (y == 1)
        if sel.sum():
            per[pt] = float(pred[sel].mean())
    return {"f1": m["f1"], "recall": m["recall"], "precision": m["precision"],
            "hard_neg_fpr": hard_neg_fpr(pred, src, 0.5),
            "missed_pii": float(n_pos * (1 - m["recall"])),
            "per_pii_recall": per}


def agg(blocks):
    """시드별 블록을 평균±표준편차로 접는다."""
    out = {}
    for k in ["f1", "recall", "precision", "hard_neg_fpr", "missed_pii"]:
        v = [b[k] for b in blocks]
        out[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    out["per_pii_recall"] = {
        pt: {"mean": float(np.mean([b["per_pii_recall"][pt] for b in blocks])),
             "std": float(np.std([b["per_pii_recall"][pt] for b in blocks]))}
        for pt in blocks[0]["per_pii_recall"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="gru_cnn")
    ap.add_argument("--config", default="default", help="Whisper 디코딩 설정")
    ap.add_argument("--conds", nargs="+", default=["clean"])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5],
                    help="음향 경로의 임계값. 합집합의 재현율-정밀도 거래를 보려면 여러 개를 준다.")
    ap.add_argument("--out", default="union_eval.json")
    args = ap.parse_args()

    m = _c.build_manifest(drop_outliers=True)
    _, _, te, _ = split_manifest(m, seed=42)
    y, src = te.label.values, te.source_type.values
    n_pos = int((y == 1).sum())
    print(f"test n={len(te)} (positive {n_pos}, hard-neg {int((src=='negative_hard').sum())})")

    res = {"n_test": int(len(te)), "n_pos": n_pos, "cnn": args.cnn,
           "whisper_config": args.config, "conditions": {}}

    for cond in args.conds:
        sp = stt_pred(cond, args.config, te)
        if sp is None:
            print(f"[{cond}] 전사 캐시 없음 — 건너뜀")
            continue
        probs = [cnn_probs(args.cnn, cond, s) for s in SEEDS]
        if any(p is None for p in probs):
            print(f"[{cond}] 음향 점수 없음 — 건너뜀")
            continue
        P = np.stack([[p[os.path.basename(fp)] for fp in te.filepath] for p in probs])

        entry = {"stt_only": block(y, sp, src, te, n_pos)}
        for tau in args.taus:
            key = f"tau{tau:g}"
            cnn_b, casc_b, uni_b = [], [], []
            for i in range(len(SEEDS)):
                c = (P[i] >= tau).astype(float)
                cnn_b.append(block(y, c, src, te, n_pos))
                casc_b.append(block(y, np.minimum(c, sp), src, te, n_pos))   # A and B
                uni_b.append(block(y, np.maximum(c, sp), src, te, n_pos))    # A or B
            entry[key] = {"cnn_only": agg(cnn_b), "cascade": agg(casc_b), "union": agg(uni_b)}
        res["conditions"][cond] = entry

        s, u = entry["stt_only"], entry[f"tau{args.taus[0]:g}"]
        print(f"\n[{cond}] Whisper={args.config}, 음향={args.cnn}, tau={args.taus[0]}")
        print(f"  {'설계':10s} {'F1':>16s} {'Recall':>16s} {'Precision':>16s} {'놓친 PII':>10s}")
        print(f"  {'STT 단독':10s} {s['f1']:>16.3f} {s['recall']:>16.3f} {s['precision']:>16.3f} "
              f"{s['missed_pii']:>10.1f}")
        for name, lab in [("cnn_only", "CNN 단독"), ("cascade", "캐스케이드"), ("union", "합집합")]:
            b = u[name]
            f = lambda k: f"{b[k]['mean']:.3f}±{b[k]['std']:.3f}"
            print(f"  {lab:10s} {f('f1'):>16s} {f('recall'):>16s} {f('precision'):>16s} "
                  f"{b['missed_pii']['mean']:>10.1f}")

    _c.save_json(args.out, res)


if __name__ == "__main__":
    main()
