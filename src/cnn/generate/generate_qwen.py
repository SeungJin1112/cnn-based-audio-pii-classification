#----------------------------------------------------------------------
# Qwen3-TTS 통합 생성기 (Phase 2+3) — 화자 확장 + PII 이진 확장
#
#   - TTS 백엔드: VoiceDesign ref → Base clone (qwen_voicebank, 14화자, 0009 확정)
#   - positive(label=1): 전화번호 / 주민등록번호 / 계좌 / 카드  (pii_type 구분)
#   - hard_neg(label=0): 전화번호 형식 아닌 숫자열(날짜/주문/송장/가격/우편/좌석/일련번호 등)
#   - easy_neg(label=0): 숫자 없는 일반 발화
#   - 모든 숫자는 "한국어 자릿수 발음"(공일공…) → 클래스 차이가 '읽는 방식'이 아니라 '형식'
#   - 통제 불변식: 세 클래스 모두 동일 14화자 풀 · 동일 16kHz mono · (전처리에서 노이즈패딩)
#   - metadata.csv: filename,filepath,text,value,pii_type,gender,speaker_id,template_id,source_type,label,duration
#
# usage: python generate_qwen.py --pos 900 --hard 450 --easy 450 [--device cuda:1]
#        python generate_qwen.py --smoke           # 클래스당 6개 빠른 검증
#----------------------------------------------------------------------
import argparse
import csv
import os
import random
import re

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

import qwen_voicebank as VB

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(HERE, "..", "data_qwen")
SEED      = 42

DIGIT_KO = {'0':'공','1':'일','2':'이','3':'삼','4':'사','5':'오','6':'육','7':'칠','8':'팔','9':'구'}


def num_to_korean(s):
    """구분자(-, 공백)로 나뉜 숫자 그룹을 그룹별 한국어 자릿수 발음 + 공백 결합."""
    groups = re.split(r"[-\s]", s)
    return " ".join("".join(DIGIT_KO[d] for d in g) for g in groups if g)


def digits_to_korean(text):
    return re.sub(r"\d+", lambda m: "".join(DIGIT_KO[d] for d in m.group()), text)


# ---- PII 값 생성기 (형식만, 실제 유효성 불필요) ----
def fake_phone():
    return f"010-{random.randint(0,9999):04d}-{random.randint(0,9999):04d}"

def fake_rrn():
    yy, mm, dd = random.randint(0,99), random.randint(1,12), random.randint(1,28)
    g = random.choice("1234")
    return f"{yy:02d}{mm:02d}{dd:02d}-{g}{random.randint(0,999999):06d}"

def fake_account():
    return f"{random.randint(100,999)}-{random.randint(10,999)}-{random.randint(0,999999):06d}"

def fake_card():
    return "-".join(f"{random.randint(0,9999):04d}" for _ in range(4))

PII_TYPES = {
    "phone":   (fake_phone, [
        "제 전화번호는 {v}입니다.", "연락처는 {v}이에요.", "{v} 으로 연락 주세요.",
        "급하시면 {v} 으로 바로 전화 주세요.", "문자는 {v} 으로 보내주세요.",
    ]),
    "rrn":     (fake_rrn, [
        "제 주민등록번호는 {v}입니다.", "주민번호는 {v}이에요.",
        "본인 확인을 위해 {v} 불러드릴게요.", "{v} 가 제 주민등록번호입니다.",
    ]),
    "account": (fake_account, [
        "제 계좌번호는 {v}입니다.", "{v} 으로 입금해 주세요.",
        "입금하실 계좌는 {v}이에요.", "환불 계좌는 {v}입니다.",
    ]),
    "card":    (fake_card, [
        "카드번호는 {v}입니다.", "{v} 로 결제 부탁드립니다.", "제 카드번호는 {v}이에요.",
    ]),
}

# ---- hard negative (전화번호 형식 아닌 숫자열) ----
def rnd(n): return "".join(str(random.randint(0,9)) for _ in range(n))

def hn_date():    return f"오늘은 {random.randint(2020,2026)}년 {random.randint(1,12)}월 {random.randint(1,28)}일입니다."
def hn_order():   return f"주문번호는 {rnd(random.choice([9,10,11]))}입니다."
def hn_invoice(): return f"송장번호는 {rnd(4)} {rnd(4)}입니다."
def hn_price():   return f"가격은 {random.randint(1,9)}{rnd(random.choice([3,4]))}원입니다."
def hn_postal():  return f"우편번호는 {rnd(5)}입니다."
def hn_seat():    return f"좌석은 {random.randint(1,30)}열 {random.randint(1,40)}번입니다."
def hn_serial():  return f"제품 일련번호는 {rnd(4)} {rnd(4)} {rnd(4)}입니다."   # 12자리(카드 유사 길이)
def hn_birth():   return f"생일은 {random.randint(1,12)}월 {random.randint(1,28)}일이에요."
def hn_bus():     return f"{random.randint(1,9)}{rnd(2)}번 버스를 타시면 됩니다."
def hn_count():   return "하나 둘 셋 넷 다섯 여섯 일곱."

HARD_CATS = [("date",hn_date),("order",hn_order),("invoice",hn_invoice),("price",hn_price),
             ("postal",hn_postal),("seat",hn_seat),("serial",hn_serial),("birth",hn_birth),
             ("bus",hn_bus),("count",hn_count)]

CARRIER_LEAD = ["안내 말씀드립니다.","잠시 안내드립니다.","확인을 위해 알려드립니다.","고객님께 안내드립니다."]
CARRIER_TAIL = ["확인 부탁드립니다.","참고 부탁드립니다.","양해 부탁드립니다.","감사합니다."]
PHONE_FMT = re.compile(r"0\d{2}[-\s]?\d{3,4}[-\s]?\d{4}")

# ---- easy negative (숫자 없는 일반 발화) ----
EASY = [
    "오늘 날씨가 정말 화창하고 좋습니다.", "주말에 가족과 함께 공원에 다녀왔어요.",
    "이번 프로젝트는 다음 주에 마무리될 예정입니다.", "점심으로 김치찌개를 먹었는데 맛있었어요.",
    "회의 자료는 메일로 미리 보내드리겠습니다.", "요즘 아침 저녁으로 날이 많이 쌀쌀해졌네요.",
    "새로 산 책이 생각보다 훨씬 재미있었습니다.", "운동을 꾸준히 하니 몸이 한결 가벼워졌어요.",
    "다음 휴가에는 제주도로 여행을 가려고 합니다.", "커피 한잔하면서 잠깐 쉬었다 가시죠.",
    "발표 준비는 거의 다 끝나가고 있습니다.", "오랜만에 친구를 만나서 즐거운 시간을 보냈어요.",
]


def add_carrier(core):
    if random.random() < 0.85:
        return f"{random.choice(CARRIER_LEAD)} {core}" if random.random() < 0.5 else f"{core} {random.choice(CARRIER_TAIL)}"
    return core


def gender_of(instruct):
    return "female" if "여성" in instruct else "male"


def write_class(base, prompts, roster, class_dir, source_type, label, items):
    """items: list of dict(text, value, pii_type, template_id). 화자는 균등 순환 배정."""
    out_dir = os.path.join(class_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    sids = list(roster.keys())
    rows = []
    for i, it in enumerate(items):
        sid = sids[i % len(sids)]
        wav, sr = VB.synthesize(base, prompts[sid], it["text"])
        base_name = f"{source_type}_{i:05d}_{sid}"
        wav_path = os.path.join(out_dir, base_name + ".wav")
        sf.write(wav_path, wav, sr, subtype="PCM_16")
        rows.append({
            "filename": base_name + ".wav", "filepath": "output/" + base_name + ".wav",
            "text": it["text"], "value": it.get("value",""), "pii_type": it.get("pii_type",""),
            "gender": gender_of(roster[sid]), "speaker_id": sid,
            "template_id": it["template_id"], "source_type": source_type, "label": label,
            "duration": round(len(wav)/sr, 3),
        })
        if (i+1) % 25 == 0 or i+1 == len(items):
            print(f"  [{source_type}] {i+1}/{len(items)}")
    with open(os.path.join(class_dir, "metadata.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["filename","filepath","text","value","pii_type","gender",
                                          "speaker_id","template_id","source_type","label","duration"])
        w.writeheader(); w.writerows(rows)
    print(f"[{source_type}] {len(rows)} files -> {out_dir}")


def build_pos_items(n):
    items, types = [], list(PII_TYPES.keys())
    for i in range(n):
        pt = types[i % len(types)]                       # phone/rrn/account/card 균등
        gen, templates = PII_TYPES[pt]
        v = gen(); t = random.choice(templates)
        items.append({"text": t.format(v=num_to_korean(v)), "value": v, "pii_type": pt,
                      "template_id": f"{pt}_{templates.index(t)}"})
    return items


def build_hard_items(n):
    items = []
    for _ in range(n):
        cat, gen = random.choice(HARD_CATS)
        core = gen(); guard = 0
        while PHONE_FMT.search(core) and guard < 10:
            core = gen(); guard += 1
        items.append({"text": digits_to_korean(add_carrier(core)), "value": "", "pii_type": "",
                      "template_id": cat})
    return items


def build_easy_items(n):
    return [{"text": random.choice(EASY), "value": "", "pii_type": "", "template_id": "easy"}
            for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=900)
    ap.add_argument("--hard", type=int, default=450)
    ap.add_argument("--easy", type=int, default=450)
    ap.add_argument("--device", default=VB.DEVICE)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.pos = args.hard = args.easy = 6

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"[load] Base -> {args.device}")
    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                         device_map=args.device, dtype=torch.bfloat16,
                                         attn_implementation="sdpa")
    roster_df = pd.read_csv(os.path.join(VB.VOICEBANK_DIR, "roster.csv"))
    roster = dict(zip(roster_df.speaker_id, roster_df.instruct))
    prompts = VB.build_prompts(base)
    print(f"[roster] {len(roster)} speakers")

    write_class(base, prompts, roster, os.path.join(DATA_DIR, "positive"),
                "positive", 1, build_pos_items(args.pos))
    write_class(base, prompts, roster, os.path.join(DATA_DIR, "negative_hard"),
                "negative_hard", 0, build_hard_items(args.hard))
    write_class(base, prompts, roster, os.path.join(DATA_DIR, "negative_easy"),
                "negative_easy", 0, build_easy_items(args.easy))
    print("\nGENERATE DONE")


if __name__ == "__main__":
    main()
