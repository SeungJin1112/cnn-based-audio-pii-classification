"""Phase 0 게이트 — Qwen3-TTS VoiceDesign 한국어 스모크.

확인 항목:
  1) Korean 지원 여부 (get_supported_languages)
  2) instruct(자연어 설명)로 서로 다른 2 화자 생성 → 구별되는가
  3) 같은 instruct 2회 생성 → 같은 화자로 일관적인가 (within vs between)
  4) 16kHz mono wav 변환 정상

출력: generate/qwen_smoke_out/*.wav + 콘솔 리포트.
"""
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_smoke_out")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda:1"            # 여유 GPU (실행 전 nvidia-smi 확인)
TARGET_SR = 16000

TEXT = "안녕하세요. 제 전화번호는 010-1234-5678입니다."
SPEAKERS = {
    "spk_f_young": "20대 여성, 밝고 또렷한 목소리, 약간 빠른 말투",
    "spk_m_old":   "50대 남성, 낮고 차분한 목소리, 느린 말투",
}


def save_16k(wav, sr, path):
    wav = np.asarray(wav, dtype=np.float32)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    sf.write(path, wav, TARGET_SR, subtype="PCM_16")
    return len(wav) / TARGET_SR


def main():
    print(f"[load] VoiceDesign -> {DEVICE}")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map=DEVICE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",   # flash-attn 미설치 → sdpa
    )
    try:
        print("[langs]", model.get_supported_languages())
    except Exception as e:
        print("[langs] 조회 실패:", e)

    # 1) 서로 다른 두 화자 + 2) 같은 화자 일관성용 재생성(spk_f_young_2)
    jobs = list(SPEAKERS.items()) + [("spk_f_young_2", SPEAKERS["spk_f_young"])]
    feats = {}
    for name, instruct in jobs:
        wavs, sr = model.generate_voice_design(text=TEXT, instruct=instruct, language="Korean")
        path = os.path.join(OUT, f"{name}.wav")
        dur = save_16k(wavs[0], sr, path)
        y = librosa.resample(np.asarray(wavs[0], np.float32), orig_sr=sr, target_sr=TARGET_SR)
        mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=20).mean(axis=1)
        feats[name] = mfcc
        print(f"  [{name}] sr={sr} dur={dur:.2f}s -> {path}")

    # 화자 구별/일관성 거리 (MFCC 평균 코사인 거리, 거친 지표)
    def cos(a, b):
        return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print("\n== MFCC 거리(거친 지표) ==")
    print(f"  between (다른 화자  f_young vs m_old) : {cos(feats['spk_f_young'], feats['spk_m_old']):.4f}")
    print(f"  within  (같은 화자  f_young vs f_young_2): {cos(feats['spk_f_young'], feats['spk_f_young_2']):.4f}")
    print("  → between > within 이면 화자 구별 OK (정밀 검증은 speaker-verification 모델로 후속)")
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
