#----------------------------------------------------------------------
# Controlled HARD NEGATIVE generator (label=0)
#   - 숫자는 포함하지만 "전화번호 형식(010-XXXX-XXXX류)이 아닌" 발화
#   - positive와 동일하게 숫자를 한국어 자릿수 발음으로 읽음(digits_to_korean)
#     → 클래스 차이가 "읽는 방식"이 아니라 "숫자열의 형식/그룹핑"이 되도록
#   - 16kHz mono wav, metadata.csv 저장 (label, source_type, duration, speaker_id, template_id)
#
# usage: python generate_neg_hard.py [COUNT]   (default COUNT = 30, pilot)
#----------------------------------------------------------------------
import asyncio
import csv
import os
import random
import re
import subprocess
import sys
#----------------------------------------------------------------------
import edge_tts
import soundfile as sf
#----------------------------------------------------------------------
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(SCRIPT_DIR, "..", "data")
CLASS_DIR         = os.path.join(DATA_DIR, "negative_hard")
OUTPUT_DIR        = os.path.join(CLASS_DIR, "output")
METADATA_PATH     = os.path.join(CLASS_DIR, "metadata.csv")
SOURCE_TYPE       = "negative_hard"
LABEL             = 0
#----------------------------------------------------------------------
TOTAL_COUNT       = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED              = 43
#----------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
#----------------------------------------------------------------------
voices = [
    {"gender": "male",   "voice": "ko-KR-InJoonNeural"},
    {"gender": "male",   "voice": "ko-KR-HyunsuMultilingualNeural"},
    {"gender": "female", "voice": "ko-KR-SunHiNeural"},
]

DIGIT_KO = {'0':'공','1':'일','2':'이','3':'삼','4':'사',
            '5':'오','6':'육','7':'칠','8':'팔','9':'구'}

def digits_to_korean(text):
    return re.sub(r"\d+", lambda m: "".join(DIGIT_KO[d] for d in m.group()), text)

def rnd(n):  # n자리 랜덤 숫자열(문자열)
    return "".join(str(random.randint(0, 9)) for _ in range(n))
#----------------------------------------------------------------------
# 카테고리별 "전화번호 형식이 아닌" 숫자 문장 생성기 (raw digits)
def gen_date():
    y, m, d = random.randint(2020, 2026), random.randint(1, 12), random.randint(1, 28)
    return f"오늘은 {y}년 {m}월 {d}일입니다."

def gen_order():
    return f"주문번호는 {rnd(random.choice([9, 10, 11]))}입니다."

def gen_invoice():
    return f"송장번호는 {rnd(4)} {rnd(4)}입니다."

def gen_room():
    return f"{random.randint(1, 9)}{random.randint(0,9)}{random.randint(1,9)}호로 오시면 됩니다."

def gen_counter():
    return f"{random.randint(1, 9)}번 창구에서 접수해 주세요."

def gen_bus():
    return f"{random.randint(1, 9)}{rnd(2)}번 버스를 타시면 됩니다."

def gen_price():
    return f"가격은 {random.randint(1, 9)}{rnd(random.choice([3,4]))}원입니다."

def gen_quantity():
    return f"총 {random.randint(2, 99)}개가 필요합니다."

def gen_postal():
    return f"우편번호는 {rnd(5)}입니다."

def gen_count_native():  # 숫자(arabic) 없음 → 그대로 native 수세기
    return "하나 둘 셋 넷 다섯 여섯 일곱."

CATEGORIES = [
    ("date",     gen_date),
    ("order",    gen_order),
    ("invoice",  gen_invoice),
    ("room",     gen_room),
    ("counter",  gen_counter),
    ("bus",      gen_bus),
    ("price",    gen_price),
    ("quantity", gen_quantity),
    ("postal",   gen_postal),
    ("count",    gen_count_native),
]

# positive와 길이 분포를 맞추기 위한 carrier 문맥(숫자 없음 → 누수 위험 없음)
# core 평균 ~3.6s, positive 평균 ~5.2s → carrier 하나(~1.8s)를 확률적으로 붙여 정렬
CARRIER_ADD_PROB = 0.85
CARRIER_LEAD = [
    "안내 말씀드립니다.",
    "잠시 안내드립니다.",
    "확인을 위해 알려드립니다.",
    "고객님께 안내드립니다.",
]
CARRIER_TAIL = [
    "확인 부탁드립니다.",
    "참고 부탁드립니다.",
    "양해 부탁드립니다.",
    "감사합니다.",
]

PHONE_FMT = re.compile(r"0\d{2}[-\s]?\d{3,4}[-\s]?\d{4}")  # 전화번호 형식 안전장치
#----------------------------------------------------------------------
async def make_tts(text, voice, mp3_path, retries=5):
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%", volume="+0%")
            await communicate.save(mp3_path)
            return
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"[TTS][ERROR]<{attempt + 1}/{retries - 1}><WAIT {wait}s> {e}")
                await asyncio.sleep(wait)
            else:
                raise

def mp3_to_wav(mp3_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )

def wav_duration(wav_path):
    info = sf.info(wav_path)
    return round(info.frames / info.samplerate, 3)
#----------------------------------------------------------------------
async def main():
    rows = []
    for i in range(TOTAL_COUNT):
        speaker = random.choice(voices)
        cat_name, gen = random.choice(CATEGORIES)
        core = gen()
        # 안전장치: 혹시라도 전화번호 형식이 생기면 스킵하고 재생성
        guard = 0
        while PHONE_FMT.search(core) and guard < 10:
            core = gen(); guard += 1
        # positive와 길이 정렬: carrier 문맥 하나를 확률적으로 앞 또는 뒤에 붙임
        if random.random() < CARRIER_ADD_PROB:
            if random.random() < 0.5:
                raw = f"{random.choice(CARRIER_LEAD)} {core}"
            else:
                raw = f"{core} {random.choice(CARRIER_TAIL)}"
        else:
            raw = core
        text = digits_to_korean(raw)

        base_name = f"hardneg_{i:05d}_{speaker['gender']}"
        mp3_path  = os.path.join(OUTPUT_DIR, base_name + ".mp3")
        wav_path  = os.path.join(OUTPUT_DIR, base_name + ".wav")

        print(f"[HARDNEG]<{i + 1}/{TOTAL_COUNT}> {speaker['gender']} | [{cat_name}] {text}")
        await make_tts(text=text, voice=speaker["voice"], mp3_path=mp3_path)
        mp3_to_wav(mp3_path, wav_path)
        os.remove(mp3_path)
        await asyncio.sleep(0.3)

        rows.append({
            "filename"    : base_name + ".wav",
            "filepath"    : "output/" + base_name + ".wav",
            "text"        : text,
            "phone"       : "",
            "gender"      : speaker["gender"],
            "voice"       : speaker["voice"],
            "speaker_id"  : speaker["voice"],
            "template_id" : cat_name,
            "source_type" : SOURCE_TYPE,
            "label"       : LABEL,
            "duration"    : wav_duration(wav_path),
        })

    with open(METADATA_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "filepath", "text", "phone", "gender", "voice",
            "speaker_id", "template_id", "source_type", "label", "duration",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[HARDNEG][SUCCESS] {TOTAL_COUNT} files -> {OUTPUT_DIR}")
    print(f"[HARDNEG][SUCCESS] metadata -> {METADATA_PATH}")
#----------------------------------------------------------------------
asyncio.run(main())
#----------------------------------------------------------------------
