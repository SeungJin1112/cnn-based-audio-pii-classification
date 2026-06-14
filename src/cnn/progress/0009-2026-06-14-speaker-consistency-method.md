---
date: 2026-06-14
phase: 화자 확장 (Phase 1 — 일관성 방안 확정)
status: done
---

# 0009 — 화자 일관성: 방안 B(reference→Base clone) 확정

## 무엇을 했나
- `generate/qwen_speaker_consistency.py` — 6 화자 × 3 발화로 두 생성 방식 비교.
  - 방안 A: 매 발화 `VoiceDesign.generate_voice_design(text, instruct)`.
  - 방안 B: VoiceDesign 으로 reference 1개 → `Base.create_voice_clone_prompt` → `Base.generate_voice_clone` 로 전 발화.
- 화자 임베딩: **Base** 의 x-vector(`VoiceClonePromptItem.ref_spk_embedding`). within/between 코사인 거리로 평가.
- 발견: **VoiceDesign 체크포인트는 clone/x-vector 미지원** → Base(1.7B, ~4G) 추가 다운로드(HF cache).

## 왜
- LOSO 가 유효하려면 같은 speaker_id 의 발화들이 acoustically 일관돼야 한다(within ≪ between).
- VoiceDesign 단독 일관성이 의심돼(0008) 정량 검증 필요.

## 결과 / 수치
| 방안 | within(같은 화자) | between(다른 화자) | ratio=between/within |
|---|---|---|---|
| A (VoiceDesign 직접) | 0.0377 | 0.0229 | **0.61** ❌ |
| B (ref→Base clone) | **0.0106** | 0.0242 | **2.29** ✅ |

- **방안 A 실패**: within > between (ratio<1) → 같은 instruct 라도 발화마다 변동이 화자 간 차이보다 커서 speaker_id 가 정의 불가. 이대로 데이터 만들면 LOSO 무의미.
- **방안 B 채택**: clone 으로 within 0.0377→0.0106(약 1/3.5) 축소, between 유지 → 화자 안정.
- → **데이터 생성 파이프라인 확정: VoiceDesign 으로 화자 설계(reference 1개) → Base 로 모든 발화 clone.** speaker_id = reference 정체성.

## 이슈 / 한계
- between 절대거리(0.024)가 큰 편은 아님 → 6 화자 다양성이 아주 크진 않음. **full 로스터는 instruct 를 더 대비되게** 설계해 between 을 키워야(연령/성별/톤/속도 폭 확대).
- x-vector 는 Base 모델 내장 임베딩(절대 스케일 해석은 주의), 상대 비교 지표로 사용.

## 다음
- Phase 1 마무리: full 화자 로스터 12~16명 instruct 설계 → reference 생성·캐시 → 로스터 전체 between 분리도 확인.
- Phase 2: 기존 generate_pos/neg_hard 텍스트·라벨 로직 재사용, TTS 백엔드를 "VoiceDesign ref → Base clone"으로 교체해 통제 데이터 재생성(동일 화자풀·ffmpeg·노이즈패딩 불변식 유지).
