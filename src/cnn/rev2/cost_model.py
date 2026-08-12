"""프리필터 비용 모형 — 손익분기 분석.

논문은 재현율과 호출률만 보고하고 비용 모형을 세우지 않았다. 프리필터 논문에서
비용 모형이 없으면 "그래서 쓸 만한가"에 답이 없다. 여기서는 관측된 재현율·오탐률을
트래픽 구성 파라미터와 결합해 순절감과 손익분기 조건을 계산한다.

기호:
  p        전체 발화 중 PII 포함 비율
  q        비PII 발화 중 '비PII 숫자열'을 포함하는 비율(나머지는 숫자 없는 일반 발화)
  R        프리필터 재현율
  F_hard   비PII 숫자열에 대한 오탐률
  F_easy   숫자 없는 일반 발화에 대한 오탐률
  c_stt    발화당 STT 단가
  c_cnn    발화당 프리필터 단가
  c_miss   PII 1건을 놓쳤을 때의 기대 손실

STT 호출률:  rho = p*R + (1-p)*(q*F_hard + (1-q)*F_easy)
순절감(발화당): (1 - rho)*c_stt - c_cnn - (1-R)*p*c_miss
손익분기:      c_miss/c_stt < ((1-rho) - c_cnn/c_stt) / ((1-R)*p)

usage: python cost_model.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common as _c  # noqa: E402

# 지연 실측에서 유도한 단가비. CNN 0.344 ms/window vs Whisper 약 360 ms/clip 이므로
# 계산 자원 단가가 같다고 보면 c_cnn/c_stt 는 1e-3 수준이다. 보수적으로 5e-3 을 쓴다.
CNN_COST_RATIO = 5e-3


def call_rate(p, q, R, f_hard, f_easy):
    return p * R + (1 - p) * (q * f_hard + (1 - q) * f_easy)


def analyze(R, f_hard, f_easy, ps=(0.01, 0.05, 0.10, 0.20, 0.50),
            qs=(0.10, 0.30, 1.00), cnn_ratio=CNN_COST_RATIO):
    rows = []
    for p in ps:
        for q in qs:
            rho = call_rate(p, q, R, f_hard, f_easy)
            saving = (1 - rho) - cnn_ratio                      # c_stt 단위
            miss = (1 - R) * p                                  # 놓친 PII 건수/발화
            breakeven = saving / miss if miss > 0 else np.inf   # c_miss/c_stt 상한
            rows.append({"p": p, "q": q, "call_rate": rho,
                         "net_saving_per_utt_in_cstt": saving,
                         "missed_pii_per_utt": miss,
                         "breakeven_cmiss_over_cstt": breakeven})
    return pd.DataFrame(rows)


def main():
    # 관측값: 재현율·오탐률은 rev2 재학습 결과에서, easy 오탐률은 v1 test(누수 없는 유일한 측정)에서.
    import json
    import glob
    R = f_hard = None
    agg_path = os.path.join(_c.OUT_DIR, "agg.json")
    if os.path.exists(agg_path):
        a = json.load(open(agg_path, encoding="utf-8"))
        R = a["indist"]["recall"]["mean"]
        f_hard = a["indist"]["hard_neg_fpr@0.5"]["mean"]
    if R is None:
        raise SystemExit("agg.json 이 없다 — analyze.py --what agg 를 먼저 실행하라")

    # 숫자 없는 일반 발화의 오탐률은 v2 평가 분할에 easy-negative 가 없어 측정할 수 없다.
    # 누수 없이 측정된 유일한 값인 v1 controlled test 에서 시드별로 역산해 쓴다.
    #   fp_easy = fp_total - hard_neg_fpr * n_hard,  n_easy = (tn+fp) - n_hard
    ev = os.path.join(os.path.dirname(HERE), "runs", "evaluation.json")
    easies = []
    if os.path.exists(ev):
        per = json.load(open(ev))["simple_cnn"]["per_seed"]
        for s, d in per.items():
            cm, hn = d["sets"]["controlled_main"], d["sets"]["hard_negative_only"]
            n_pos = cm["tp"] + cm["fn"]
            n_hard = hn["n"] - n_pos
            fp_hard = cm["hard_neg_fpr"] * n_hard
            n_easy = (cm["tn"] + cm["fp"]) - n_hard
            if n_easy > 0:
                easies.append((cm["fp"] - fp_hard) / n_easy)
    F_EASY = float(np.mean(easies)) if easies else 0.0
    print(f"v1 시드별 easy-negative 오탐률: {[round(x, 4) for x in easies]}")

    print(f"입력: R={R:.3f}  F_hard={f_hard:.3f}  F_easy={F_EASY:.3f}  c_cnn/c_stt={CNN_COST_RATIO}")
    df = analyze(R, f_hard, F_EASY)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # 재현율-절감 트레이드오프: 프리필터 운용점별로 같은 계산을 반복
    pf = a["indist"]["prefilter"]
    rows = []
    for row in pf:
        Rt, ft = row["recall"]["mean"], row["hard_neg_fpr"]["mean"]
        for p in (0.05, 0.20):
            rho = call_rate(p, 0.30, Rt, ft, F_EASY)
            miss = (1 - Rt) * p
            rows.append({"target_recall": row["target_recall"],
                         "threshold": row["threshold"]["mean"],
                         "recall": Rt, "hard_fpr": ft, "p": p,
                         "call_rate": rho, "saving": (1 - rho) - CNN_COST_RATIO,
                         "breakeven_cmiss_over_cstt": ((1 - rho) - CNN_COST_RATIO) / miss
                         if miss > 0 else np.inf})
    op = pd.DataFrame(rows)
    print("\n== 운용점별 ==")
    print(op.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    _c.save_json("cost_model.json",
                 {"inputs": {"recall": R, "f_hard": f_hard, "f_easy": F_EASY,
                             "cnn_cost_ratio": CNN_COST_RATIO},
                  "grid": df.to_dict("records"),
                  "operating_points": op.to_dict("records")})


if __name__ == "__main__":
    main()
