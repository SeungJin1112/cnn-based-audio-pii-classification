# Plan v2 — Qwen3-TTS 화자 확장 + PII 이진 확장

> 작성: 2026-06-14. 기준 커밋 `3b842cd`(v1 파이프라인 + LOSO) 이후 후속 계획.
> 목적: proof-of-concept 를 "더 제대로" 굳히기 — (1) 화자 일반화 안정화, (2) 전화번호 외 PII 일반화.

## Context (왜)

v1 결론의 두 가지 약점:
1. **화자 3명뿐(edge-tts 한계)** → LOSO 가 ROC 0.99(랭킹 일반화)인데 고정 0.5 임계값 F1 0.616±0.28 로 출렁이고, 화자별 calibration 이 어긋남. 추정 자체가 noisy.
2. **전화번호 단일 PII** → "민감정보 탐지" 일반 주장엔 부족.

해결: **Qwen3-TTS-12Hz-1.7B-VoiceDesign**(자연어 instruct 로 화자 자유 설계, 한국어 지원)으로
다수의 구별되는 한국어 화자를 생성하고, PII 유형을 주민등록번호·계좌/카드번호까지 확장한다.
분류 프레이밍은 **이진(민감정보 포함 여부)**.

## 환경 / 모델 (Phase 0 — 진행 중)

- 모델: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`(~4.52GB) + `Qwen/Qwen3-TTS-Tokenizer-12Hz` → HF cache 다운로드 중.
- 추가 후보(Phase 1 일관성 방안 B 채택 시): `Qwen/Qwen3-TTS-12Hz-1.7B-Base`(voice clone).
- 런타임: 신규 conda env `qwen-tts`(python 3.12), `pip install -U qwen-tts` (+ flash-attn). 분류 env `audio-pii` 와 분리 보존.
- GPU: H100 NVL ×4 공유 환경 → 여유 GPU 지정(`device_map="cuda:N"`). 실행 직전 `nvidia-smi` 로 free 메모리 확인.
- 스모크 게이트: 한국어 한 문장을 서로 다른 instruct 2개로 생성 → 16kHz mono wav 변환 → 청취/스펙트로그램으로 화자 구별·품질 확인. **여기서 한국어 품질·일관성이 안 나오면 계획 재검토.**

## Phase 1 — 화자 로스터 설계 (핵심)

- VoiceDesign `instruct` 로 **N=12~16명** 구별 한국어 화자 설계. 속성 그리드로 다양화:
  성별 × 연령대(20s~60s) × 톤(낮음/높음) × 속도 × 스타일(차분/활기). `speakers.csv` 로 정의.
- **일관성 문제**(같은 화자가 발화마다 동일해야 LOSO 가 유효):
  - 방안 A: 화자별 고정 instruct + 고정 seed (모델 seed 지원 시).
  - **방안 B(권장)**: VoiceDesign 로 화자별 reference utterance 1개 생성 → `Base`(voice clone)의 `ref_audio` 로 모든 발화 생성 → 화자 정체성 고정.
  - 게이트: 화자 임베딩(예: speaker-verification 모델)으로 **within-speaker 유사도 ≫ between-speaker** 확인 후 방안 확정.

## Phase 2 — 데이터 재생성 (통제 원칙 유지)

- 기존 `generate/generate_pos.py`·`generate_neg_hard.py` 의 텍스트·템플릿·라벨 로직 **재사용**, TTS 백엔드만 edge-tts→Qwen3-TTS 로 교체.
- **통제 불변식(기존 결론의 핵심 — 절대 깨면 안 됨)**:
  - positive / hard-neg / easy-neg **모두 동일 화자 풀 · 동일 ffmpeg(16kHz mono) · 동일 노이즈 패딩**.
  - → 출처 지문(코덱·무음·화자) shortcut confound 차단. (v1 에서 이 통제가 결론을 지탱했다.)
- metadata 컬럼 확장: `speaker_id`(설계 화자), `gender`, `age_band`, `pii_type`, `label`.

## Phase 3 — PII 이진 확장

- 신규 positive 생성기(낭독 스타일은 기존 controlled 와 통일):
  - **주민등록번호**: `YYMMDD-Gxxxxxx` 형식 가짜(체크섬 불필요, 형식만). 주민번호 템플릿 문장.
  - **계좌/카드번호**: 10~16자리 그룹 숫자. 은행·카드 템플릿 문장.
- 신규 **hard-negative**(자릿수 유사 비민감 숫자열 — "숫자 탐지기" 전락 방지의 핵심):
  - 주민번호 ↔ 생년월일/날짜(6자리, "1990년 1월 1일").
  - 계좌/카드 ↔ 주문번호·송장·가격·우편번호.
- 라벨: 어떤 PII 든 포함 → 1, 일반문장+비민감 숫자열 → 0 (**이진**).
- 평가 확장: PII **유형별 recall 분해**(전화/주민/계좌) + 유형별 hard-neg FPR.
- **held-out PII 유형 일반화 테스트**(강한 증거, LOSO 의 PII 판):
  전화+계좌로 학습 → **주민번호로 평가**. 통과하면 "숫자열 암기"가 아니라 "PII 형식 일반화" 입증.

## Phase 4 — 학습 / 평가 / Calibration

- 기존 `dataset/splits/engine/models/evaluate` 거의 그대로 재사용. `simple_cnn` 우선(가볍고 v1 최강).
- split: template-disjoint + length-stratified 유지 + **화자 충분 → speaker-disjoint(LOSO) 정식 적용**.
- **calibration 추가(LOSO 약점 직접 공략)**: val 에서 temperature scaling / per-speaker score 정규화 →
  고정 임계값 운용점이 **새 화자에서도 유지되는지** 재평가. (v1 의 핵심 미해결 지점)
- 보고: in-distribution vs LOSO(N화자) F1/recall, calibration 전후 운용점 안정성, PII 유형별·held-out-PII 일반화.

## Phase 5 — 문서화

- `progress/0008+` 기록(화자/PII 확장 단계별), `reports/conclusion.md` 갱신(화자 일반화 + PII 일반화 정량화).

## 리스크 / 게이트

- VoiceDesign 한국어 **품질·일관성 미검증** → Phase 1 스모크가 분기 게이트.
- 여전히 **합성 데이터**(실통화 아님) — 정직한 한계 명시 유지.
- GPU **공유 환경** → 실행 전 여유 GPU 확인.
- 일관성 방안 B 채택 시 Base 모델 추가 다운로드(~4GB) 필요.

## 즉시 다음 작업

1. (진행중) 모델 다운로드 완료 확인.
2. `qwen-tts` env 구성 + `pip install qwen-tts`.
3. VoiceDesign 한국어 스모크(서로 다른 화자 2명) → 품질/일관성 게이트 통과 여부 판단.
