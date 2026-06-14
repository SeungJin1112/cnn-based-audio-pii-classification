# Plan v3 — STT + NLP 파이프라인 (정설 경로, CNN 비교 기준선)

> 작성: 2026-06-15. CNN 단계 종료 후, "정설 경로(STT→텍스트→NLP PII 탐지)"를 구성해
> **음향 CNN 접근과 동일 조건에서 직접 비교**하는 것이 목표.
> STT = Whisper-large-v3. NLP = 기존 데이터셋 구조(pii_type/label) 그대로 사용.

## Context (왜)
- CNN 은 "STT 없이 음향만으로 PII 형식 선별" 가능성을 입증했으나, precision 약점(문맥 부재로 주문번호↔전화/계좌 형식 충돌)이 한계였다.
- 정설 경로(STT+NLP)는 텍스트·문맥을 직접 보므로 그 약점을 메울 수 있다. 두 접근을 **같은 데이터·같은 라벨·같은 평가셋·같은 지표**로 비교해 트레이드오프(정확도 vs 지연/비용)를 정량화한다.

## ⭐ 공정 비교 원칙 (가장 중요 — "최대한 동일 조건")
| 항목 | 맞추는 방식 |
|---|---|
| **데이터** | 동일 `data_qwen` (1800클립, 14화자, 16kHz mono) — CNN이 쓴 그 wav 그대로 |
| **라벨/과제** | 동일 이진(PII 포함=1 / 비포함=0), 동일 pii_type(phone/rrn/account/card) |
| **평가셋** | CNN in-distribution 과 **동일 test split**(seed 42, template-disjoint) 재현 → 같은 인스턴스에서 비교 |
| **지표** | 동일: F1 / recall / precision / hard-neg FPR / PII유형별 recall |
| **비교축 추가** | **지연·비용**: clip당 처리시간 (CNN 0.34ms/win vs Whisper) — 프로젝트의 핵심 동기 |

> 주의: STT+NLP 는 **학습이 없다**(Whisper pretrained + 규칙기반 NLP). 따라서 train/val 은 쓰지 않고
> CNN 과 **동일한 test split 인스턴스**에서 평가해 사과-대-사과 비교를 한다. (전체셋 결과도 부가 보고)
> 두 접근의 정보 접근량이 다른 것(음향 only vs 텍스트+문맥)은 **패러다임 차이 자체가 비교 대상**이므로 인위적으로 제한하지 않는다.

> ⚠️ **합성 데이터 편향(STT 유리)을 반드시 감안**: 데이터가 깨끗한 TTS 합성음이라 Whisper 의 WER/CER 이
> 실제 통화 대비 **현저히 낮다**. 즉 STT 단계가 "최상의 조건"이라 STT+NLP 결과는 **best-case(상한)** 이다.
> 실제 음성에서는 WER↑ 로 STT+NLP 가 크게 열화될 수 있다. 따라서 (1) WER/CER 을 측정해 이 편향을 정량화하고,
> (2) 모든 비교표에 "best-case STT" 단서를 명시하며, (3) 잡음·전화코덱 열화 실험(Phase 4)으로 민감도를 본다.

## Phase 0 — Whisper-large-v3 준비
- 모델: `openai/whisper-large-v3` (HF, ~3GB) 다운로드.
- 런타임 후보: **faster-whisper(large-v3, CT2)** 권장(GPU fp16/int8, 1800클립 빠른 전사) — 또는 transformers `WhisperForConditionalGeneration`.
- env: 신규 `stt` env 또는 기존 GPU env 재사용(torch+transformers). H100 여유 GPU 지정.
- 스모크: data_qwen 샘플 3개(전화/주민/일반) 전사 → 한국어 출력·숫자 표기(아라비아 "010…" vs 한글 "공일공…") 형태 확인.

## Phase 1 — 전사(transcription) + 캐시
- `data_qwen` 전 클립(1800)을 Whisper 로 한국어 전사 → `data_qwen/transcripts.csv` (filepath, transcript, stt_sec).
- **한 번만 실행하고 캐시** → NLP 규칙 반복 개발 시 STT 재실행 불필요.
- 부가 기록: clip당 STT 소요시간(지연 비교용), 전사 실패/공백 케이스.

## Phase 2 — NLP PII 탐지기 (규칙 기반, 데이터셋 구조 그대로)
- 입력: Whisper 전사 텍스트(정답 text 아님 — 그건 부정행위). STT 오류 포함이 곧 STT+NLP 의 실제 성능.
- **숫자 정규화**: 한글 자릿수("공일공 일이삼사…") ↔ 아라비아 변환(생성기 `digits_to_korean` 의 역매핑) + 공백/구분 정리. Whisper 가 어느 쪽으로 내든 처리.
- **패턴 매칭**(기존 pii_type 형식 그대로):
  - phone: `01\d-?\d{3,4}-?\d{4}` (11자리)
  - rrn: `\d{6}-?\d{7}` (13자리, YYMMDD-G……)
  - account/card: 그룹 숫자(카드 16자리 4-4-4-4, 계좌 가변 그룹)
- **문맥 키워드(옵션, 별도 변형으로 평가)**: "전화번호/연락처/주민등록번호/계좌/카드" → 형식+문맥 결합본도 만들어 형식-only 와 비교(주문번호 오검출 줄이는지).
- 출력: 패턴 매칭 시 1, 아니면 0 (+ 매칭된 pii_type).

## Phase 3 — 평가 & CNN 직접 비교
- CNN in-dist test split(seed 42) 재현 → 그 인스턴스에서 STT+NLP 예측.
- 산출:
  1. **비교표**: CNN vs STT+NLP(형식-only) vs STT+NLP(형식+문맥) — F1/recall/precision/hard-neg FPR/유형별 recall.
  2. **누수 원인 대비**: CNN 이 약했던 주문번호(9~11자리) 오검출을 STT+NLP 가 문맥으로 잡는지.
  3. **지연/비용표**: clip당 ms (CNN 0.34ms vs Whisper ~수백 ms~초), 처리량.
  4. **STT 품질(WER/CER) 측정** — 생성 메타데이터의 정답 `text` 를 reference 로 Whisper 전사와 비교.
     전체 WER/CER + **숫자 부분만의 정확도**(자릿수 오인식률)를 따로 보고. → 합성데이터로 인한 "STT 유리" 편향을 수치로 박는다.
     모든 비교표 캡션에 측정된 WER 을 명시("WER X% 인 best-case STT 기준").
- 결과 → `runs/stt_nlp_eval.json`, `reports/cnn_vs_sttnlp.md`.

## Phase 4 — 실음성 열화 민감도 (합성 편향 보정) + 2단계 파이프라인
- **WER 민감도 실험(중요)**: data_qwen 오디오에 실통화 모사 열화 적용 후 재전사·재평가 →
  STT+NLP 성능이 WER 증가에 따라 어떻게 떨어지는지 곡선화. 열화 예: 8kHz/μ-law 전화코덱, 배경잡음(SNR 5~20dB), 약한 리버브.
  → "합성에서의 best-case STT+NLP" 가 실음성에서 얼마나 무너지는지 추정. (CNN 도 동일 열화로 비교하면 공정한 robustness 대조.)
- **2단계 파이프라인(선택)**: CNN(경량·고recall 프리필터)로 후보 거른 뒤 STT+NLP(정밀·문맥)로 확정 → 지연·비용 절감 + precision 회복.

## 산출물 / 코드(예정)
- `stt/transcribe_whisper.py` — 전사+캐시
- `stt/nlp_pii_detector.py` — 정규화+패턴(+문맥) 탐지
- `stt/eval_stt_nlp.py` — 동일 split 평가 + CNN 비교표
- progress 기록(0013+), `reports/cnn_vs_sttnlp.md`

## 리스크 / 게이트
- **합성 데이터 → STT 유리 편향(핵심 해석 주의)**: TTS 합성음은 깨끗해 Whisper WER/CER 이 실통화보다 훨씬 낮다.
  → STT+NLP 수치는 **상한(best-case)** 이며, 실음성에선 열화. 비교 결론은 반드시 이 단서와 함께 제시(WER 측정·열화 실험으로 정량화).
- **Whisper 숫자 표기 비일관**(한글 vs 아라비아, 오인식) → Phase 0 스모크로 형태 확인, 정규화 견고화가 핵심.
- 긴 숫자열은 Whisper 도 오인식 가능 → STT+NLP 의 진짜 약점이 드러날 수 있음(흥미로운 비교 포인트).
- 규칙기반 NLP 는 점수가 이진 → ROC/PR 대신 운용점 지표(F1/precision/recall/FPR)로 비교.
- 동일 test split 재현(seed 42) 반드시 일치시켜 공정성 확보.

## 즉시 다음
1. Whisper-large-v3 다운로드 + 전사 스모크(숫자 표기 형태 확인).
2. 전 클립 전사 캐시.
3. NLP 탐지기(형식 / 형식+문맥) 구현 → 동일 split 평가 → CNN 비교표.
