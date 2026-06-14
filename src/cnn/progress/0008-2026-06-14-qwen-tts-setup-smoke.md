---
date: 2026-06-14
phase: 화자 확장 (Qwen3-TTS 환경/스모크)
status: done
---

# 0008 — Qwen3-TTS VoiceDesign 환경 구성 + 한국어 스모크 게이트

## 무엇을 했나
- edge-tts 한국어 보이스가 **3개뿐**(InJoon/SunHi/Hyunsu, 이미 전부 사용)으로 확인 → 화자 확장 불가.
- 대체 TTS로 **Qwen3-TTS-12Hz-1.7B-VoiceDesign**(instruct 자연어로 화자 자유 설계) + Tokenizer 다운로드(HF cache, 4.3G + 651M).
- 신규 conda env `qwen-tts`(python 3.12) + `pip install -U qwen-tts`(0.1.1, torch 2.12, transformers 4.57).
- API introspection: VoiceDesign 호출은 `generate_voice_design(text, instruct, language="Korean")`. (일관성용 `create_voice_clone_prompt`/`generate_voice_clone`도 존재.)
- 스모크: `generate/qwen_tts_smoke.py` — 한국어 한 문장을 instruct 2종으로 생성 + 같은 instruct 재생성.

## 왜
- v1 결론의 최대 약점이 화자 3명 한계(LOSO F1 0.616±0.28 불안정). 화자를 늘려야 일반화 추정이 안정.
- VoiceDesign 은 프리셋 화자가 아니라 프롬프트로 화자를 무제한 설계 → 다수 한국어 화자 확보에 적합.

## 결과 / 수치
- 한국어 지원 확인: languages 에 `korean` 포함.
- 생성 정상: sr=24000 → 16kHz mono 변환 OK. 발화 길이 5.9~8.2s.
- 화자 구별(거친 MFCC 평균 코사인 거리): between(다른 화자) 0.0237 > within(같은 instruct 재생성) 0.0054 → **약 4.4배**. 화자 구별 신호 양호.
- GPU: H100 ×4 공유, cuda:1(92G free) 사용.

## 이슈 / 한계
- within 거리가 0(완전 동일)이 아님 → **같은 instruct도 호출마다 미세 변동**. 수백 발화 규모에서 speaker_id 가 흔들리면 LOSO 무의미해질 수 있음 → Phase 1 에서 일관성 방안(B: reference 1개 생성 후 voice-clone 고정) 검증 필요.
- MFCC 평균은 거친 지표 → 정밀 화자 구별은 speaker-verification 임베딩으로 후속 검증 예정.
- flash-attn 미설치 → `attn_implementation="sdpa"` 사용(정상 동작).

## 다음
- Phase 1: 화자 로스터 12~16명 instruct 설계 + 일관성 방안 확정(VoiceDesign 단독 vs reference→voice-clone). 화자 임베딩으로 within≫between 검증.
