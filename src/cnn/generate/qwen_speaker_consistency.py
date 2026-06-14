"""Phase 1 — 화자 일관성 검증: 방안 A(VoiceDesign 직접) vs 방안 B(VoiceDesign ref → Base clone).

핵심 질문: 같은 화자(instruct)가 서로 다른 발화에서도 acoustically 일관적인가?
  - LOSO 가 유효하려면 speaker_id 가 안정적이어야 한다.
  - 방안 A: 매 발화 VoiceDesign.generate_voice_design(text_k, instruct)  ← 호출마다 변동 우려
  - 방안 B: VoiceDesign 으로 reference 1개 생성 → Base.voice_clone 으로 전 발화 생성(화자 고정)

VoiceDesign 체크포인트는 clone/x-vector 미지원 → Base(1.7B)가 clone + create_voice_clone_prompt 담당.
지표(Base x-vector = VoiceClonePromptItem.ref_spk_embedding):
  within=같은 화자 발화 간 평균 코사인거리(작을수록 일관), between=화자 평균임베딩 간 거리(클수록 구별)
  separation ratio = between / within
"""
import itertools
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_consistency_out")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda:1"
TARGET_SR = 16000

SPEAKERS = {
    "s1": "20대 여성, 밝고 또렷한 목소리, 약간 빠른 말투",
    "s2": "50대 남성, 낮고 차분한 목소리, 느린 말투",
    "s3": "30대 남성, 활기차고 명료한 목소리, 보통 속도",
    "s4": "40대 여성, 부드럽고 차분한 목소리, 느린 말투",
    "s5": "20대 남성, 높고 가벼운 목소리, 빠른 말투",
    "s6": "60대 여성, 따뜻하고 또렷한 목소리, 느린 말투",
}
TEXTS = [
    "오늘 날씨가 정말 화창하고 좋습니다.",
    "회의는 내일 오후 세 시에 시작될 예정입니다.",
    "주문하신 상품이 오늘 배송될 예정입니다.",
]
REF_TEXT = "안녕하세요, 반갑습니다."


def to16k(wav, sr):
    wav = np.asarray(wav, dtype=np.float32)
    return wav if sr == TARGET_SR else librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)


def xvector(base, wav, sr):
    items = base.create_voice_clone_prompt(ref_audio=(np.asarray(wav, np.float32), sr),
                                           x_vector_only_mode=True)
    emb = items[0].ref_spk_embedding.float().flatten().cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-9)


def cos_dist(a, b):
    return 1 - float(np.dot(a, b))


def within_between(emb_by_spk):
    within = []
    for embs in emb_by_spk.values():
        within += [cos_dist(a, b) for a, b in itertools.combinations(embs, 2)]
    means = {s: np.mean(e, axis=0) for s, e in emb_by_spk.items()}
    means = {s: m / (np.linalg.norm(m) + 1e-9) for s, m in means.items()}
    between = [cos_dist(means[a], means[b]) for a, b in itertools.combinations(means, 2)]
    w, b = float(np.mean(within)), float(np.mean(between))
    return w, b, b / (w + 1e-9)


def main():
    print(f"[load] VoiceDesign + Base -> {DEVICE}")
    vd = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                                       device_map=DEVICE, dtype=torch.bfloat16, attn_implementation="sdpa")
    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                         device_map=DEVICE, dtype=torch.bfloat16, attn_implementation="sdpa")

    emb_A, emb_B = {}, {}
    for sid, instruct in SPEAKERS.items():
        emb_A[sid], emb_B[sid] = [], []

        # 방안 A: 매 발화 VoiceDesign 직접
        for ti, text in enumerate(TEXTS):
            w, sr = vd.generate_voice_design(text=text, instruct=instruct, language="Korean")
            sf.write(os.path.join(OUT, f"A_{sid}_t{ti}.wav"), to16k(w[0], sr), TARGET_SR, subtype="PCM_16")
            emb_A[sid].append(xvector(base, w[0], sr))

        # 방안 B: VoiceDesign reference 1개 → Base clone 고정
        rw, rsr = vd.generate_voice_design(text=REF_TEXT, instruct=instruct, language="Korean")
        sf.write(os.path.join(OUT, f"B_{sid}_ref.wav"), to16k(rw[0], rsr), TARGET_SR, subtype="PCM_16")
        prompt = base.create_voice_clone_prompt(ref_audio=(np.asarray(rw[0], np.float32), rsr), ref_text=REF_TEXT)
        for ti, text in enumerate(TEXTS):
            w, sr = base.generate_voice_clone(text=text, language="Korean", voice_clone_prompt=prompt)
            sf.write(os.path.join(OUT, f"B_{sid}_t{ti}.wav"), to16k(w[0], sr), TARGET_SR, subtype="PCM_16")
            emb_B[sid].append(xvector(base, w[0], sr))
        print(f"  {sid} done (A={len(emb_A[sid])}, B={len(emb_B[sid])})")

    print("\n== 일관성/구별 (Base x-vector, 코사인 거리) ==")
    wA, bA, rA = within_between(emb_A)
    wB, bB, rB = within_between(emb_B)
    print(f"  방안 A (VoiceDesign 직접): within={wA:.4f}  between={bA:.4f}  ratio={rA:.2f}")
    print(f"  방안 B (ref→Base clone) : within={wB:.4f}  between={bB:.4f}  ratio={rB:.2f}")
    print(f"\n  → separation ratio 우위: 방안 {'B' if rB > rA else 'A'}  (within 작고 ratio 클수록 화자 안정)")
    print("\nCONSISTENCY DONE")


if __name__ == "__main__":
    main()
