"""Phase 0/1 — Whisper-large-v3 로 data_qwen 전사 + 캐시.

- 입력: data_qwen/{positive,negative_hard,negative_easy}/metadata.csv (filepath, text=정답, label, pii_type, source_type)
- 출력: data_qwen/transcripts.csv (filepath, gt_text, transcript, label, pii_type, source_type, stt_sec)
- 정답 text 는 WER 측정용으로만 보관(NLP 입력은 transcript).

usage:
  python transcribe_whisper.py --smoke      # 3개(전화/주민/일반)만 전사·출력
  python transcribe_whisper.py               # 전 클립 전사 → transcripts.csv
"""
import argparse
import os
import time

import pandas as pd
import torch
from transformers import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
DQ = os.path.normpath(os.path.join(HERE, "..", "data_qwen"))
OUT_CSV = os.path.join(DQ, "transcripts.csv")
MODEL = "openai/whisper-large-v3"


def load_rows():
    rows = []
    for cls in ["positive", "negative_hard", "negative_easy"]:
        df = pd.read_csv(os.path.join(DQ, cls, "metadata.csv"))
        df["pii_type"] = df.get("pii_type", "")
        for _, r in df.iterrows():
            rows.append({
                "filepath": os.path.normpath(os.path.join(DQ, cls, str(r["filepath"]).replace("\\", "/"))),
                "gt_text": r["text"], "label": int(r["label"]),
                "pii_type": r["pii_type"] if pd.notna(r.get("pii_type", "")) else "",
                "source_type": r["source_type"],
            })
    return pd.DataFrame(rows)


def build_asr(device):
    return pipeline(
        "automatic-speech-recognition", model=MODEL,
        torch_dtype=torch.float16, device=device,
        chunk_length_s=30, batch_size=16,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", type=int, default=1)   # cuda index
    args = ap.parse_args()

    rows = load_rows()
    if args.smoke:
        pick = pd.concat([
            rows[(rows.label == 1) & (rows.pii_type == "phone")].head(1),
            rows[(rows.label == 1) & (rows.pii_type == "rrn")].head(1),
            rows[rows.source_type == "negative_easy"].head(1),
        ], ignore_index=True)
        rows = pick

    print(f"[load] Whisper {MODEL} -> cuda:{args.device} | {len(rows)} clips")
    asr = build_asr(args.device)

    gen_kwargs = {"language": "korean", "task": "transcribe"}
    out = []
    t_all = time.time()
    paths = rows.filepath.tolist()
    for i in range(0, len(paths), 16):
        batch = paths[i:i + 16]
        t0 = time.time()
        res = asr(batch, generate_kwargs=gen_kwargs)
        dt = (time.time() - t0) / len(batch)
        for p, r in zip(batch, res):
            out.append((p, r["text"].strip(), round(dt, 3)))
        if not args.smoke:
            print(f"  {min(i+16, len(paths))}/{len(paths)}")

    tx = pd.DataFrame(out, columns=["filepath", "transcript", "stt_sec"])
    merged = rows.merge(tx, on="filepath")

    if args.smoke:
        print(f"\n== 전사 스모크 (총 {time.time()-t_all:.1f}s) ==")
        for _, r in merged.iterrows():
            print(f"  [{r.source_type}/{r.pii_type}]")
            print(f"    정답 : {r.gt_text}")
            print(f"    전사 : {r.transcript}")
        print("\n→ 숫자가 아라비아('010…')인지 한글('공일공…')인지 확인 → NLP 정규화 설계에 반영")
    else:
        merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n[done] {len(merged)} 전사 (평균 {merged.stt_sec.mean():.3f}s/clip) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
