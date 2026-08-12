"""feedback2 대응 신규 합성 — 자릿수 길이 프로브셋 / 캐리어 균형 데이터셋(v3).

기존 데이터와 같은 backend(VoiceDesign 참조 → Base 클론, 14화자)를 그대로 쓰므로
합성 분포가 어긋나지 않는다.

--probe : PII 의미가 전혀 없는 순수 난수 자릿수열을 길이·그룹 구조 축으로 합성.
          모델 점수 대 자릿수 길이 곡선을 그려 결정 함수를 특정하기 위한 통제 자극.
            축 A(연속)   : 4/6/8/10/12/14/16자리 연속 × 100
            축 B(그룹)   : 4-4 / 4-4-4 / 4-4-4-4 × 100  (총 자릿수는 8/12/16 로 A와 겹침)
            축 C(맨숫자) : 캐리어 문틀 없이 숫자만 × 40  (문틀 기여 분리)

--v3    : 캐리어 문장을 세 클래스 모두에 동일 확률(85%)로 부여해 재생성.
          기존 v2 는 캐리어가 hard-negative 에 87.1%, positive 에 0% 로 치우쳐
          어휘·운율 상관이 남아 있었다. 이 confound 자체를 제거한 데이터셋이다.

usage:
  python gen_tts.py --probe --device cuda:5
  python gen_tts.py --v3    --device cuda:6
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CNN_DIR, "generate"))

import qwen_voicebank as VB  # noqa: E402
import generate_qwen as G  # noqa: E402

SEED = 42
PROBE_DIR = os.path.join(CNN_DIR, "data_probe")
V3_DIR = os.path.join(CNN_DIR, "data_qwen_v3")
# 정상 발화는 10초 미만이다. 이를 넘으면 TTS 반복 생성으로 본다.
MAX_CLIP_SEC = 20.0

FIELDS = ["filename", "filepath", "text", "value", "pii_type", "gender", "speaker_id",
          "template_id", "source_type", "label", "duration"]
PROBE_FIELDS = FIELDS + ["n_digits", "n_groups", "max_run", "frame", "axis"]


# ---------------------------------------------------------------- 공통 합성
def synth_all(base, prompts, roster, items, out_dir, fields, log_every=50,
              shard=0, nshards=1):
    """items 전체를 인덱스 기준으로 샤딩해 합성한다.

    화자 배정은 전역 인덱스로 정하므로 샤드 수와 무관하게 동일한 화자 순환이 유지된다.
    샤드별로 metadata_shard{k}.csv 를 남기고, merge_shards 로 합친다.
    """
    os.makedirs(os.path.join(out_dir, "output"), exist_ok=True)
    sids = list(roster.keys())
    rows = []
    mine = [(i, it) for i, it in enumerate(items) if i % nshards == shard]
    for n, (i, it) in enumerate(mine):
        sid = sids[i % len(sids)]
        name = f"{it['source_type']}_{i:05d}_{sid}"
        path = os.path.join(out_dir, "output", name + ".wav")
        # 이미 정상 길이로 만들어진 파일은 건너뛴다(중단 후 재개 지원).
        if os.path.exists(path):
            info = sf.info(path)
            if 0 < info.frames / info.samplerate < MAX_CLIP_SEC:
                wav, sr = None, info.samplerate
                dur = round(info.frames / info.samplerate, 3)
                row = dict(it)
                row.update({"filename": name + ".wav", "filepath": "output/" + name + ".wav",
                            "gender": G.gender_of(roster[sid]), "speaker_id": sid,
                            "duration": dur})
                rows.append({k: row.get(k, "") for k in fields})
                continue
        # TTS 가 반복 생성 루프에 빠지면 수백 초짜리 클립이 나온다(v2 의 655초 클립과 같은 현상).
        # 정상 길이가 나올 때까지 최대 3회 재시도하고, 그래도 길면 앞부분만 남긴다.
        for attempt in range(3):
            wav, sr = VB.synthesize(base, prompts[sid], it["text"])
            if len(wav) / sr < MAX_CLIP_SEC:
                break
            print(f"  shard{shard}: idx {i} 재생성({len(wav)/sr:.0f}초)", flush=True)
        if len(wav) / sr >= MAX_CLIP_SEC:
            wav = wav[: int(MAX_CLIP_SEC * sr)]
        sf.write(path, wav, sr, subtype="PCM_16")
        row = dict(it)
        row.update({"filename": name + ".wav", "filepath": "output/" + name + ".wav",
                    "gender": G.gender_of(roster[sid]), "speaker_id": sid,
                    "duration": round(len(wav) / sr, 3)})
        rows.append({k: row.get(k, "") for k in fields})
        if (n + 1) % log_every == 0 or n + 1 == len(mine):
            print(f"  shard{shard}: {n + 1}/{len(mine)}", flush=True)
    name = "metadata.csv" if nshards == 1 else f"metadata_shard{shard}.csv"
    with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[{out_dir}] shard{shard}: {len(rows)} files")


def merge_shards(out_dir, fields):
    """샤드별 metadata 를 filename 순으로 합쳐 metadata.csv 를 만든다."""
    import glob as _g
    parts = sorted(_g.glob(os.path.join(out_dir, "metadata_shard*.csv")))
    if not parts:
        return
    import pandas as pd
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.sort_values("filename").reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False, encoding="utf-8-sig")
    print(f"[{out_dir}] merged {len(parts)} shards -> {len(df)} rows")


# ---------------------------------------------------------------- 프로브셋
def rnd_digits(n):
    return "".join(str(random.randint(0, 9)) for _ in range(n))


def build_probe_items():
    items = []
    # 축 A — 연속 자릿수열, 공통 문틀
    for n in [4, 6, 8, 10, 12, 14, 16]:
        for _ in range(100):
            d = rnd_digits(n)
            items.append({"text": f"번호는 {G.num_to_korean(d)}입니다.", "value": d,
                          "pii_type": "", "template_id": f"probe_cont_{n}",
                          "source_type": "probe", "label": -1,
                          "n_digits": n, "n_groups": 1, "max_run": n,
                          "frame": "sentence", "axis": "continuous"})
    # 축 B — 4자리 그룹 구조(카드형). 총 자릿수는 축 A와 겹치되 런 길이는 4로 고정
    for k in [2, 3, 4]:
        for _ in range(100):
            groups = [rnd_digits(4) for _ in range(k)]
            d = "-".join(groups)
            items.append({"text": f"번호는 {G.num_to_korean(d)}입니다.", "value": d,
                          "pii_type": "", "template_id": f"probe_grp4x{k}",
                          "source_type": "probe", "label": -1,
                          "n_digits": 4 * k, "n_groups": k, "max_run": 4,
                          "frame": "sentence", "axis": "grouped4"})
    # 축 C — 문틀 없이 숫자만
    for n in [4, 6, 8, 10, 12, 14, 16]:
        for _ in range(40):
            d = rnd_digits(n)
            items.append({"text": f"{G.num_to_korean(d)}.", "value": d,
                          "pii_type": "", "template_id": f"probe_bare_{n}",
                          "source_type": "probe", "label": -1,
                          "n_digits": n, "n_groups": 1, "max_run": n,
                          "frame": "bare", "axis": "continuous"})
    return items


# ---------------------------------------------------------------- v3
def carrier(core, p=0.85):
    """generate_qwen.add_carrier 와 동일 규칙을 클래스 구분 없이 적용."""
    if random.random() < p:
        if random.random() < 0.5:
            return f"{random.choice(G.CARRIER_LEAD)} {core}"
        return f"{core} {random.choice(G.CARRIER_TAIL)}"
    return core


def build_v3_pos_items(n_pos):
    """positive 에도 hard-negative 와 같은 확률(85%)로 캐리어를 부여한 재생성 집합.

    기존 hard-negative(캐리어 87.1%)와 조합하면 캐리어 유무가 클래스와 사실상 무상관이
    된다. 대신 캐리어만큼 positive 가 길어져 길이 상관의 부호가 v2 와 반대가 되므로,
    v2 와 v3 에서 결론이 같게 나오면 캐리어·길이 어느 쪽도 결론을 만들지 않았다는 뜻이다.
    """
    pos = []
    types = list(G.PII_TYPES.keys())
    for i in range(n_pos):
        pt = types[i % len(types)]
        gen, templates = G.PII_TYPES[pt]
        v = gen()
        t = random.choice(templates)
        core = t.format(v=G.num_to_korean(v))
        pos.append({"text": carrier(core), "value": v, "pii_type": pt,
                    "template_id": f"{pt}_{templates.index(t)}",
                    "source_type": "positive", "label": 1})
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--v3", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="샤드 metadata 만 병합하고 종료")
    args = ap.parse_args()

    if args.merge:
        if args.probe:
            merge_shards(PROBE_DIR, PROBE_FIELDS)
        if args.v3:
            merge_shards(os.path.join(V3_DIR, "positive"), FIELDS)
        return

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    from qwen_tts import Qwen3TTSModel
    print(f"[load] Base -> {args.device}", flush=True)
    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                         device_map=args.device, dtype=torch.bfloat16,
                                         attn_implementation="sdpa")
    import pandas as pd
    roster_df = pd.read_csv(os.path.join(VB.VOICEBANK_DIR, "roster.csv"))
    roster = dict(zip(roster_df.speaker_id, roster_df.instruct))
    prompts = VB.build_prompts(base)
    print(f"[roster] {len(roster)} speakers", flush=True)

    if args.probe:
        items = build_probe_items()
        if args.smoke:
            items = items[:6]
        print(f"[probe] {len(items)} items (shard {args.shard}/{args.nshards})", flush=True)
        synth_all(base, prompts, roster, items, PROBE_DIR, PROBE_FIELDS,
                  shard=args.shard, nshards=args.nshards)

    if args.v3:
        items = build_v3_pos_items(6 if args.smoke else 900)
        print(f"[v3/positive] {len(items)} items (shard {args.shard}/{args.nshards})", flush=True)
        synth_all(base, prompts, roster, items, os.path.join(V3_DIR, "positive"), FIELDS,
                  shard=args.shard, nshards=args.nshards)

    print("DONE")


if __name__ == "__main__":
    main()
