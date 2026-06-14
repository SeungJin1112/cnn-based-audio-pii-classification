"""Qwen3-TTS 화자 뱅크 (Phase 1 결과 = 방안 B 파이프라인의 재사용 backend).

파이프라인(0009 확정):
  VoiceDesign 으로 화자별 reference 1개 생성 → Base 로 모든 발화를 clone (화자 일관성 확보).

제공:
  build_voicebank(out_dir)         : 로스터 reference 생성·캐시 + 화자 분리도 리포트 + roster.csv
  load_models(device)              : (vd, base) 로드
  build_prompts(base, roster_dir)  : {sid: clone_prompt} (cache 된 ref wav 로부터)
  synthesize(base, prompt, text)   : 16kHz mono wav (clone)

단독 실행 시 build_voicebank() 수행.
"""
import itertools
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

HERE = os.path.dirname(os.path.abspath(__file__))
VOICEBANK_DIR = os.path.join(HERE, "voicebank")     # ref wav + roster.csv
DEVICE = "cuda:1"
TARGET_SR = 16000
REF_TEXT = "안녕하세요, 오늘 하루도 좋은 하루 되시기 바랍니다."

# 14명 — 성별 7:7, 연령 20~60, 톤/속도/음색/스타일을 강하게 대비
ROSTER = {
    "spk01": "20대 여성, 매우 높고 밝은 목소리, 빠른 말투, 발랄한 느낌",
    "spk02": "60대 남성, 매우 낮고 굵은 목소리, 느린 말투, 묵직한 느낌",
    "spk03": "30대 여성, 중간 톤의 차분한 목소리, 보통 속도, 또박또박한 발음",
    "spk04": "40대 남성, 약간 높은 목소리, 빠른 말투, 활기찬 느낌",
    "spk05": "50대 여성, 낮고 따뜻한 목소리, 느린 말투, 부드러운 느낌",
    "spk06": "20대 남성, 높고 가벼운 목소리, 매우 빠른 말투, 경쾌한 느낌",
    "spk07": "60대 여성, 가늘고 떨리는 목소리, 느린 말투, 인자한 느낌",
    "spk08": "30대 남성, 낮고 단단한 목소리, 보통 속도, 신뢰감 있는 느낌",
    "spk09": "40대 여성, 또렷하고 높은 목소리, 빠른 말투, 사무적인 느낌",
    "spk10": "50대 남성, 허스키하고 낮은 목소리, 느린 말투, 차분한 느낌",
    "spk11": "20대 여성, 부드럽고 중간 톤의 목소리, 느린 말투, 나긋한 느낌",
    "spk12": "30대 남성, 밝고 높은 목소리, 빠른 말투, 친근한 느낌",
    "spk13": "40대 남성, 굵고 낮은 목소리, 보통 속도, 권위적인 느낌",
    "spk14": "50대 여성, 맑고 높은 목소리, 보통 속도, 우아한 느낌",
}


def to16k(wav, sr):
    wav = np.asarray(wav, dtype=np.float32)
    return wav if sr == TARGET_SR else librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)


def load_models(device=DEVICE):
    vd = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                                       device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa")
    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                         device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa")
    return vd, base


def _xvector(base, wav, sr):
    items = base.create_voice_clone_prompt(ref_audio=(np.asarray(wav, np.float32), sr), x_vector_only_mode=True)
    e = items[0].ref_spk_embedding.float().flatten().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def build_prompts(base, roster_dir=VOICEBANK_DIR):
    """cache 된 ref wav 로부터 {sid: clone_prompt} 구성 (데이터 생성용)."""
    roster = pd.read_csv(os.path.join(roster_dir, "roster.csv"))
    prompts = {}
    for _, r in roster.iterrows():
        y, sr = librosa.load(os.path.join(roster_dir, r["ref_wav"]), sr=TARGET_SR, mono=True)
        prompts[r["speaker_id"]] = base.create_voice_clone_prompt(ref_audio=(y, TARGET_SR), ref_text=REF_TEXT)
    return prompts


def synthesize(base, prompt, text, language="Korean"):
    w, sr = base.generate_voice_clone(text=text, language=language, voice_clone_prompt=prompt)
    return to16k(w[0], sr), TARGET_SR


def build_voicebank(out_dir=VOICEBANK_DIR, device=DEVICE):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[load] VoiceDesign + Base -> {device}")
    vd, base = load_models(device)

    rows, embs = [], {}
    for sid, instruct in ROSTER.items():
        w, sr = vd.generate_voice_design(text=REF_TEXT, instruct=instruct, language="Korean")
        ref_wav = f"{sid}_ref.wav"
        sf.write(os.path.join(out_dir, ref_wav), to16k(w[0], sr), TARGET_SR, subtype="PCM_16")
        embs[sid] = _xvector(base, w[0], sr)
        rows.append({"speaker_id": sid, "instruct": instruct, "ref_wav": ref_wav})
        print(f"  {sid} ref 생성 완료")

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "roster.csv"), index=False, encoding="utf-8-sig")

    # 화자 분리도: 모든 쌍의 x-vector 코사인 거리
    pairs = [(a, b, 1 - float(np.dot(embs[a], embs[b]))) for a, b in itertools.combinations(embs, 2)]
    d = np.array([p[2] for p in pairs])
    closest = min(pairs, key=lambda p: p[2])
    print("\n== 로스터 화자 분리도 (x-vector 코사인 거리) ==")
    print(f"  화자 수: {len(embs)}  쌍 수: {len(pairs)}")
    print(f"  거리 min={d.min():.4f}  mean={d.mean():.4f}  max={d.max():.4f}")
    print(f"  가장 가까운 쌍: {closest[0]}–{closest[1]} = {closest[2]:.4f}")
    print(f"  (참고: 0009 within(같은 화자)=0.0106. min 쌍 거리가 이보다 충분히 크면 구별 OK)")
    print(f"\nroster -> {os.path.join(out_dir, 'roster.csv')}")
    print("VOICEBANK DONE")


if __name__ == "__main__":
    build_voicebank()
