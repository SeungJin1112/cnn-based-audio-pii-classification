"""Phase 3 — STT+NLP 평가 & CNN 직접 비교.

- CNN in-dist 와 동일 test split(seed 42, template-disjoint) 재현 → 같은 인스턴스에서 STT+NLP 평가.
- 지표: F1/recall/precision/hard-neg FPR/PII유형별 recall (CNN 과 동일).
- WER/CER + 숫자 정확도 측정(합성데이터 STT 유리 편향 정량화).
- 지연: clip당 STT 시간 vs CNN 0.34ms.
- CNN 수치는 runs/qwen_experiment.json(indist) 에서 로드해 비교표 생성.

usage: python eval_stt_nlp.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # src/cnn 모듈 임포트

import config as C
from metrics import classification_metrics, hard_neg_fpr
from speaker_pii_experiment import build_qwen_manifest
from splits import split_manifest
from nlp_pii_detector import detect_format, detect_format_context, _ko_digits_to_arabic

TX = os.path.join(C.CNN_DIR, "data_qwen", "transcripts.csv")
RUNS = os.path.join(C.CNN_DIR, "runs")


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[-1]


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(len(r), 1)


def cer(ref, hyp):
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    return edit_distance(r, h) / max(len(r), 1)


def digits_only(s):
    import re
    return re.sub(r"\D", "", s)


def metrics_block(y, pred, src):
    pred = np.asarray(pred)
    m = classification_metrics(y, pred.astype(float), 0.5)   # 이진 pred 를 prob 로
    return {"f1": m["f1"], "recall": m["recall"], "precision": m["precision"],
            "hard_neg_fpr": hard_neg_fpr(pred.astype(float), src, 0.5)}


def main():
    if not os.path.exists(TX):
        print("transcripts.csv 없음 — 먼저 transcribe_whisper.py 실행"); return
    tx = pd.read_csv(TX)

    # CNN 과 동일 test split 재현
    m = build_qwen_manifest()
    _, _, te, _ = split_manifest(m, seed=42)
    te = te.merge(tx[["filepath", "transcript", "gt_text", "stt_sec"]], on="filepath", how="left")
    miss = te.transcript.isna().sum()
    te = te.dropna(subset=["transcript"]).reset_index(drop=True)
    print(f"test n={len(te)} (전사 누락 {miss}) | label={dict(te.label.value_counts())}")

    y = te.label.values
    src = te.source_type.values
    pred_fmt = [detect_format(t)[0] for t in te.transcript]
    pred_ctx = [detect_format_context(t)[0] for t in te.transcript]

    res = {"n_test": int(len(te)),
           "stt_nlp_format_only": metrics_block(y, pred_fmt, src),
           "stt_nlp_format_context": metrics_block(y, pred_ctx, src)}

    # PII 유형별 recall (둘 다)
    def per_pii(pred):
        pred = np.asarray(pred); out = {}
        for pt in ["phone", "rrn", "account", "card"]:
            mask = (te.pii_type.values == pt) & (y == 1)
            if mask.sum():
                out[pt] = round(float(pred[mask].mean()), 3)
        return out
    res["per_pii_format_only"] = per_pii(pred_fmt)
    res["per_pii_format_context"] = per_pii(pred_ctx)

    # WER/CER: 숫자 표현 불일치 없는 일반발화(negative_easy) '전체'에서 측정(STT 품질은 split 무관).
    ez = tx[tx.source_type == "negative_easy"]
    wers = np.array([wer(r.gt_text, r.transcript) for _, r in ez.iterrows()])
    cers = np.array([cer(r.gt_text, r.transcript) for _, r in ez.iterrows()])
    # 평균은 Whisper 환각(반복 루프) 1~2개에 크게 휘둘림 → 중앙값·완벽일치율 중심 보고.
    res["stt_quality"] = {
        "n_easy": int(len(ez)),
        "wer_median_easy": round(float(np.median(wers)), 4),
        "wer_mean_easy": round(float(np.mean(wers)), 4),
        "exact_match_rate": round(float((wers == 0).mean()), 4),
        "hallucination_count(wer>2)": int((wers > 2).sum()),
        "note": "일반발화 기준. 중앙값 WER≈0·완벽일치 다수 → 합성음 STT 거의 완벽(best-case). "
                "평균은 Whisper 환각 소수가 왜곡. 실통화에선 WER↑. "
                "긴 숫자열(카드 16자리)은 자릿수 누락/분절로 별도 취약(per-pii recall)."}
    res["latency"] = {"stt_sec_per_clip_mean": round(float(te.stt_sec.mean()), 4),
                      "cnn_ms_per_window": 0.344}

    # CNN 비교(동일 split)
    cnn_path = os.path.join(RUNS, "qwen_experiment.json")
    if os.path.exists(cnn_path):
        ci = json.load(open(cnn_path)).get("indist", {})
        res["cnn_indist"] = {"f1": round(ci.get("f1", float("nan")), 3),
                             "recall": round(ci.get("recall", float("nan")), 3),
                             "precision": round(ci.get("precision", float("nan")), 3),
                             "hard_neg_fpr": round(ci.get("hard_neg_fpr@0.5", float("nan")), 3)}

    os.makedirs(RUNS, exist_ok=True)
    json.dump(res, open(os.path.join(RUNS, "stt_nlp_eval.json"), "w"), indent=2, ensure_ascii=False)

    print("\n=== 비교 (동일 test split) ===")
    def row(name, d): print(f"  {name:24s} F1={d['f1']:.3f} recall={d['recall']:.3f} "
                            f"precision={d['precision']:.3f} hardFPR={d['hard_neg_fpr']:.3f}")
    if "cnn_indist" in res: row("CNN(음향)", res["cnn_indist"])
    row("STT+NLP(형식only)", res["stt_nlp_format_only"])
    row("STT+NLP(형식+문맥)", res["stt_nlp_format_context"])
    print(f"\n  per-PII recall (형식+문맥): {res['per_pii_format_context']}")
    sq = res["stt_quality"]
    print(f"  STT 품질(일반발화 n={sq['n_easy']}): WER중앙값={sq['wer_median_easy']} 완벽일치={sq['exact_match_rate']:.0%} "
          f"환각{sq['hallucination_count(wer>2)']}개 (합성=best-case, 실통화선 WER↑)")
    print(f"  지연: STT {res['latency']['stt_sec_per_clip_mean']}s/clip vs CNN {res['latency']['cnn_ms_per_window']}ms/win")
    print(f"\nsaved -> {os.path.join(RUNS, 'stt_nlp_eval.json')}")


if __name__ == "__main__":
    main()
