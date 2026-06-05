#----------------------------------------------------------------------
# 1. install ffmpeg (https://ffmpeg.org/download.html)
# 2. install python 3.8+ (https://www.python.org/downloads/)
# 3. pip install edge-tts
#----------------------------------------------------------------------
import asyncio
import csv
import os
import random
import subprocess
#----------------------------------------------------------------------
import edge_tts
#----------------------------------------------------------------------
# user settings
INPUT_NUMBER      = "00001"
#----------------------------------------------------------------------
TOTAL_COUNT       = 1000
DATA_DIR          = "../../data"
O_DIR             = os.path.join(DATA_DIR,   "o")
NUMBER_DIR        = os.path.join(O_DIR,      INPUT_NUMBER)
OUTPUT_DIR        = os.path.join(NUMBER_DIR, "output")
METADATA_FILENAME = "metadata.csv"
#----------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
#----------------------------------------------------------------------
voices = [
    {
        "gender": "male",
        "voice": "ko-KR-InJoonNeural",
    },
    {
        "gender": "female",
        "voice": "ko-KR-SunHiNeural",
    },
]

templates = [
    "제 전화번호는 {phone}입니다.",
    "연락처는 {phone}입니다.",
    "전화번호는 {phone}이에요.",
    "번호 알려드릴게요. {phone}입니다.",
    "제 번호는 {phone}입니다.",
    "문의는 {phone}으로 주세요.",
    "연락 가능 번호는 {phone}입니다.",
    "휴대폰 번호는 {phone}입니다.",
    "전화는 {phone}으로 주시면 됩니다.",
    "문자 주실 번호는 {phone}입니다.",
]
#----------------------------------------------------------------------
def make_fake_phone():
    mid  = random.randint(0, 9999)
    last = random.randint(0, 9999)
    return f"010-{mid:04d}-{last:04d}"
#----------------------------------------------------------------------
async def make_tts(text, voice, mp3_path, retries=5):
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(
                text    = text,
                voice   = voice,
                rate    = "+0%",
                volume  = "+0%",
            )
            await communicate.save(mp3_path)
            return
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"[TTS][ERROR]<{attempt + 1}/{retries - 1}><WAIT {wait}s> {e}")
                await asyncio.sleep(wait)
            else:
                raise
#----------------------------------------------------------------------
def mp3_to_wav(mp3_path, wav_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",  mp3_path,
            "-ar", "16000",
            "-ac", "1",
            wav_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
#----------------------------------------------------------------------
async def main():
    rows = []

    for i in range(TOTAL_COUNT):
        speaker = random.choice(voices)
        phone   = make_fake_phone()
        text    = random.choice(templates).format(phone=phone)

        base_name = f"phone_{i:05d}_{speaker['gender']}"
        mp3_path  = os.path.join(OUTPUT_DIR, base_name + ".mp3")
        wav_path  = os.path.join(OUTPUT_DIR, base_name + ".wav")

        print(f"[TTS][INFO]<{i + 1}/{TOTAL_COUNT}> {speaker['gender']} | {text}")

        await make_tts(
            text     = text,
            voice    = speaker["voice"],
            mp3_path = mp3_path,
        )

        mp3_to_wav(mp3_path, wav_path)
        os.remove(mp3_path)

        await asyncio.sleep(0.3)

        rows.append({
            "filename" : base_name + ".wav",
            "filepath" : wav_path,
            "text"     : text,
            "phone"    : phone,
            "gender"   : speaker["gender"],
            "voice"    : speaker["voice"],
        })

    with open("metadata.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "filepath",
                "text",
                "phone",
                "gender",
                "voice"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n")
    print(f"[TTS][INFO]<SUCCESS> {TOTAL_COUNT} files generated and metadata.csv saved.")
#----------------------------------------------------------------------
asyncio.run(main())
#----------------------------------------------------------------------
