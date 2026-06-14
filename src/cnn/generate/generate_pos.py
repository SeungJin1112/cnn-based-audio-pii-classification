#----------------------------------------------------------------------
# Controlled POSITIVE generator (label=1)
#   - 전화번호 형식 숫자열을 "한국어 발음"으로 읽음 (negative와 발음 스타일 정렬)
#   - 전화번호 위치(앞/중간/끝) + 템플릿 다양화로 길이/위치 쏠림 완화
#   - 문맥변형 positive("예시입니다 / 제 번호 아니에요") 포함 → 형식 자체 학습 probe
#   - 16kHz mono wav, metadata.csv 저장 (label, source_type, duration, speaker_id, template_id)
#
# usage: python generate_pos.py [COUNT]   (default COUNT = 30, pilot)
#----------------------------------------------------------------------
import asyncio
import csv
import os
import random
import subprocess
import sys
#----------------------------------------------------------------------
import edge_tts
import soundfile as sf
#----------------------------------------------------------------------
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(SCRIPT_DIR, "..", "data")
CLASS_DIR         = os.path.join(DATA_DIR, "positive")
OUTPUT_DIR        = os.path.join(CLASS_DIR, "output")
METADATA_PATH     = os.path.join(CLASS_DIR, "metadata.csv")
SOURCE_TYPE       = "positive_controlled"
LABEL             = 1
#----------------------------------------------------------------------
TOTAL_COUNT       = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED              = 42
CONTEXT_VARIANT_RATIO = 0.2   # 문맥변형 positive 비율
#----------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
#----------------------------------------------------------------------
voices = [
    {"gender": "male",   "voice": "ko-KR-InJoonNeural"},
    {"gender": "male",   "voice": "ko-KR-HyunsuMultilingualNeural"},
    {"gender": "female", "voice": "ko-KR-SunHiNeural"},
]

# {phone}을 앞/중간/끝 다양한 위치에 배치
templates_normal = [
    "제 전화번호는 {phone}입니다.",
    "연락처는 {phone}이에요.",
    "{phone} 으로 연락 주세요.",
    "번호 알려드릴게요. {phone}입니다.",
    "급하시면 {phone} 으로 바로 전화 주시면 됩니다.",
    "문자는 {phone} 으로 보내주세요.",
    "{phone} 이 제 휴대폰 번호입니다.",
    "담당자 연락처는 {phone} 이니 참고 부탁드립니다.",
    "혹시 모르니 {phone} 도 같이 적어 두세요.",
    "예약 확인은 {phone} 으로 도와드리겠습니다.",
]

# 문맥상 실제 PII가 아니지만 전화번호 형식은 들림 → 그래도 label=1
templates_context_variant = [
    "{phone} 형식은 예시입니다.",
    "{phone} 같은 번호는 제 번호가 아니에요.",
    "이건 그냥 샘플인데 {phone} 처럼 적으시면 됩니다.",
    "{phone} 는 실제 번호가 아니라 안내용입니다.",
]
#----------------------------------------------------------------------
DIGIT_KO = {'0':'공','1':'일','2':'이','3':'삼','4':'사',
            '5':'오','6':'육','7':'칠','8':'팔','9':'구'}

def make_fake_phone():
    mid  = random.randint(0, 9999)
    last = random.randint(0, 9999)
    return f"010-{mid:04d}-{last:04d}"

def phone_to_korean(phone):
    # "010-1234-5678" → "공일공 일이삼사 오육칠팔"
    return ' '.join(
        ''.join(DIGIT_KO[d] for d in part)
        for part in phone.split('-')
    )
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
        speaker  = random.choice(voices)
        phone    = make_fake_phone()
        phone_ko = phone_to_korean(phone)

        is_variant = (random.random() < CONTEXT_VARIANT_RATIO)
        if is_variant:
            t_idx = random.randrange(len(templates_context_variant))
            template = templates_context_variant[t_idx]
            template_id = f"ctx_{t_idx}"
        else:
            t_idx = random.randrange(len(templates_normal))
            template = templates_normal[t_idx]
            template_id = f"norm_{t_idx}"
        text = template.format(phone=phone_ko)

        base_name = f"pos_{i:05d}_{speaker['gender']}"
        mp3_path  = os.path.join(OUTPUT_DIR, base_name + ".mp3")
        wav_path  = os.path.join(OUTPUT_DIR, base_name + ".wav")

        print(f"[POS]<{i + 1}/{TOTAL_COUNT}> {speaker['gender']} | {text}")
        await make_tts(text=text, voice=speaker["voice"], mp3_path=mp3_path)
        mp3_to_wav(mp3_path, wav_path)
        os.remove(mp3_path)
        await asyncio.sleep(0.3)

        rows.append({
            "filename"    : base_name + ".wav",
            "filepath"    : "output/" + base_name + ".wav",
            "text"        : text,
            "phone"       : phone,
            "gender"      : speaker["gender"],
            "voice"       : speaker["voice"],
            "speaker_id"  : speaker["voice"],
            "template_id" : template_id,
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

    print(f"\n[POS][SUCCESS] {TOTAL_COUNT} files -> {OUTPUT_DIR}")
    print(f"[POS][SUCCESS] metadata -> {METADATA_PATH}")
#----------------------------------------------------------------------
asyncio.run(main())
#----------------------------------------------------------------------
