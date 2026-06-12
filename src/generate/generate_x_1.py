#----------------------------------------------------------------------
# 1. install ffmpeg (https://ffmpeg.org/download.html)
# 2. install python 3.8+ (https://www.python.org/downloads/)
# 3. pip install edge-tts
#----------------------------------------------------------------------
import asyncio
import csv
import os
import random
import re
import subprocess
#----------------------------------------------------------------------
import edge_tts
#----------------------------------------------------------------------
# user settings
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data"))
X_DIR             = os.path.join(DATA_DIR, "x")
INPUT_CSV_PATH    = os.path.join(X_DIR, "sample.csv")
OUTPUT_DIR        = os.path.join(X_DIR, "output")
METADATA_FILENAME = "metadata.csv"
#----------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
#----------------------------------------------------------------------
voices = [
    # male
    {"gender": "male",   "voice": "ko-KR-InJoonNeural"},
    {"gender": "male",   "voice": "ko-KR-HyunsuMultilingualNeural"},
    # female
    {"gender": "female", "voice": "ko-KR-SunHiNeural"},
]

DIGIT_KO = {
    "0": "공",
    "1": "일",
    "2": "이",
    "3": "삼",
    "4": "사",
    "5": "오",
    "6": "육",
    "7": "칠",
    "8": "팔",
    "9": "구",
}

def load_samples(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def digits_to_korean(text):
    return re.sub(r"\d+", lambda match: "".join(DIGIT_KO[digit] for digit in match.group()), text)
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
    samples = load_samples(INPUT_CSV_PATH)
    rows = []

    for i, sample in enumerate(samples):
        speaker = random.choice(voices)
        text    = digits_to_korean(sample["content"].strip())
        seq     = sample["seq"].strip()

        base_name = f"x_{int(seq):05d}_{speaker['gender']}"
        mp3_path  = os.path.join(OUTPUT_DIR, base_name + ".mp3")
        wav_path  = os.path.join(OUTPUT_DIR, base_name + ".wav")
        file_path = "output/" + base_name + ".wav"

        print(f"[TTS][INFO]<{i + 1}/{len(samples)}> {speaker['gender']} | {text}")

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
            "filepath" : file_path,
            "text"     : text,
            "phone"    : "",
            "gender"   : speaker["gender"],
            "voice"    : speaker["voice"],
            "detect"   : 0,
        })

    metadata_path = os.path.join(X_DIR, METADATA_FILENAME)

    with open(metadata_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "filepath",
                "text",
                "phone",
                "gender",
                "voice",
                "detect"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n")
    print(f"[TTS][INFO]<SUCCESS> {len(samples)} files generated and metadata.csv saved.")
#----------------------------------------------------------------------
asyncio.run(main())
#----------------------------------------------------------------------
