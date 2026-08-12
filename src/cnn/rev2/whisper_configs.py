"""Whisper 다중 설정 비교 — "전사가 구조적으로 취약하다"는 주장의 검정.

논문은 Whisper 를 기본 설정으로만 쓰고 16자리 카드번호 재현율 0.852 를 근거로
전사의 구조적 취약성을 주장한다. 설정 미조정일 가능성을 배제하려면 최소 두 개의
디코딩 설정에서 같은 실패가 재현되어야 한다.

설정:
  default        논문과 동일(greedy, language=korean)
  prompt         숫자 형식 예시를 initial prompt 로 제공 (Whisper 의 알려진 완화책)
  beam5          빔 서치 5
  beam5_prompt   빔 서치 + 프롬프트

조건:
  clean / nb8k / g711  (열화 조건에서 CNN 과 ASR 이 각각 얼마나 무너지는지 비교)

usage:
  python whisper_configs.py --conditions clean --configs default prompt beam5 beam5_prompt
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "stt"))

os.environ.setdefault("HF_HOME", "/data/hye0n/hf")

import common as _c  # noqa: E402
import config as C  # noqa: E402
from metrics import classification_metrics, hard_neg_fpr  # noqa: E402
from splits import split_manifest  # noqa: E402
from nlp_pii_detector import detect_format, detect_format_context  # noqa: E402

MODEL = "openai/whisper-large-v3"
# 자릿수 낭독이 아라비아 숫자열로 정규화되도록 형식 예시를 준다.
PROMPT_TEXT = ("전화번호는 010-1234-5678입니다. 주민등록번호는 900101-1234567입니다. "
               "카드번호는 1234-5678-9012-3456입니다. 계좌번호는 123-456-789012입니다.")

CONFIGS = {
    "default":      {"num_beams": 1},
    "prompt":       {"num_beams": 1, "use_prompt": True},
    "beam5":        {"num_beams": 5},
    "beam5_prompt": {"num_beams": 5, "use_prompt": True},
}


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


class Transcriber:
    def __init__(self, device="cuda"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(MODEL)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.float16).to(device).eval()
        self.device = device

    def run(self, paths, cfg, batch_size=8):
        if cfg.get("num_beams", 1) > 1:
            batch_size = max(2, batch_size // 4)   # 빔 서치는 캐시가 빔 배수로 늘어난다
        gen = {"language": "korean", "task": "transcribe",
               "num_beams": cfg.get("num_beams", 1), "max_new_tokens": 200}
        if cfg.get("use_prompt"):
            gen["prompt_ids"] = self.proc.get_prompt_ids(
                PROMPT_TEXT, return_tensors="pt").to(self.device)
            gen["prompt_condition_type"] = "first-segment"
        texts, dts = [], []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            audio = []
            for p in batch:
                y, sr = sf.read(p, dtype="float32")
                if sr != C.SAMPLE_RATE:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=C.SAMPLE_RATE)
                audio.append(y[:C.SAMPLE_RATE * 30])
            feats = self.proc(audio, sampling_rate=C.SAMPLE_RATE, return_tensors="pt",
                              return_attention_mask=True)
            t0 = time.time()
            with torch.no_grad():
                ids = self.model.generate(
                    feats.input_features.to(self.device, torch.float16),
                    attention_mask=feats.attention_mask.to(self.device), **gen)
            dt = (time.time() - t0) / len(batch)
            out = self.proc.batch_decode(ids, skip_special_tokens=True)
            texts.extend(t.strip() for t in out)
            dts.extend([dt] * len(batch))
            print(f"    {min(i + batch_size, len(paths))}/{len(paths)}", flush=True)
        return texts, dts


def evaluate(te, transcripts):
    y, src = te.label.values, te.source_type.values
    out = {}
    for name, fn in [("format_only", detect_format), ("format_context", detect_format_context)]:
        pred = np.array([fn(t)[0] for t in transcripts], dtype=float)
        m = classification_metrics(y, pred, 0.5)
        blk = {"f1": m["f1"], "recall": m["recall"], "precision": m["precision"],
               "hard_neg_fpr": hard_neg_fpr(pred, src, 0.5)}
        blk["per_pii_recall"] = {
            pt: {"n": int(((te.pii_type.values == pt) & (y == 1)).sum()),
                 "recall": float(pred[(te.pii_type.values == pt) & (y == 1)].mean())}
            for pt in ["phone", "rrn", "account", "card"]
            if ((te.pii_type.values == pt) & (y == 1)).sum()}
        out[name] = blk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--wer", action="store_true", help="일반 발화에서 WER 도 측정")
    args = ap.parse_args()

    m = _c.build_manifest(drop_outliers=True)
    _, _, te, _ = split_manifest(m, seed=42)
    print(f"test n={len(te)} | {dict(te.source_type.value_counts())}")

    tr = Transcriber()
    # 이미 측정한 조건은 보존하고 새 조건만 덧붙인다(여러 번 나누어 실행하기 위함).
    prev_path = os.path.join(_c.OUT_DIR, "whisper_configs.json")
    res = (json.load(open(prev_path, encoding="utf-8")) if os.path.exists(prev_path)
           else {"n_test": int(len(te)), "model": MODEL, "conditions": {}})
    res.setdefault("conditions", {})
    for cond in args.conditions:
        if cond == "clean":
            paths = te.filepath.tolist()
        else:
            from degrade_eval import build_condition
            paths = build_condition(te, cond, 42).filepath.tolist()
        res["conditions"][cond] = {}
        for cname in args.configs:
            cache = os.path.join(_c.OUT_DIR, f"tx_{cond}_{cname}.csv")
            print(f"\n[{cond}/{cname}] 전사 중 ...", flush=True)
            if os.path.exists(cache):
                prev = pd.read_csv(cache)
                texts, dts = prev.transcript.fillna("").tolist(), [float("nan")]
                print("  (캐시 재사용)")
            else:
                try:
                    texts, dts = tr.run(paths, CONFIGS[cname])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("  OOM → 배치 2로 재시도", flush=True)
                    texts, dts = tr.run(paths, CONFIGS[cname], batch_size=2)
                pd.DataFrame({"filepath": te.filepath, "transcript": texts}).to_csv(
                    cache, index=False, encoding="utf-8-sig")
            blk = evaluate(te, texts)
            blk["latency_sec_per_clip"] = float(np.nanmean(dts))
            res["conditions"][cond][cname] = blk
            _c.save_json("whisper_configs.json", res)          # 중간 저장
            f = blk["format_context"]
            print("  fmt+ctx F1=%.3f recall=%.3f prec=%.3f hardFPR=%.3f | per-pii=%s" % (
                f["f1"], f["recall"], f["precision"], f["hard_neg_fpr"],
                {k: round(v["recall"], 3) for k, v in f["per_pii_recall"].items()}))

    if args.wer:
        easy = m[m.source_type == "negative_easy"]
        res["wer"] = {}
        for cname in args.configs:
            texts, _ = tr.run(easy.filepath.tolist(), CONFIGS[cname])
            w = np.array([wer(g, h) for g, h in zip(easy.text, texts)])
            res["wer"][cname] = {"n": int(len(w)), "median": float(np.median(w)),
                                 "mean": float(w.mean()),
                                 "exact_match": float((w == 0).mean()),
                                 "hallucination_gt2": int((w > 2).sum())}
            print(f"[WER/{cname}] {res['wer'][cname]}")

    _c.save_json("whisper_configs.json", res)


if __name__ == "__main__":
    main()
