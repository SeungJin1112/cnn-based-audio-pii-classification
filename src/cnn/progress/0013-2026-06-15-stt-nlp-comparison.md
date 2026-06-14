---
date: 2026-06-15
phase: STT+NLP 파이프라인 (Phase 0~3 — 구성 + CNN 비교)
status: done
---

# 0013 — STT(Whisper-v3)+NLP 정설경로 구성 & CNN 직접 비교

## 무엇을 했나
- STT: `openai/whisper-large-v3` 다운로드, `stt/transcribe_whisper.py` 로 data_qwen 1800클립 전사(평균 0.43s/clip) → `data_qwen/transcripts.csv`.
- NLP: `stt/nlp_pii_detector.py` — 숫자 형식 매칭(전화11/주민13/카드16/계좌10~14) + 문맥 키워드 변형(형식only / 형식+문맥).
- 평가: `stt/eval_stt_nlp.py` — **CNN in-dist 와 동일 test split(seed 42, n=331)** 에서 동일 지표로 비교 + WER + 지연.

## 왜
- CNN 의 precision 약점(형식충돌 숫자열)을 텍스트·문맥으로 푸는 정설경로를 동일조건에서 비교 → 트레이드오프 정량화.

## 결과 / 수치 (동일 test split)
| 접근 | F1 | recall | precision | hard-neg FPR | 지연 |
|---|---|---|---|---|---|
| CNN(음향) | 0.826 | **0.989** | 0.708 | 0.542 | **0.344ms** |
| STT+NLP(형식only) | 0.883 | 0.921 | 0.849 | 0.218 | ~360ms |
| STT+NLP(형식+문맥) | **0.935** | 0.921 | **0.951** | **0.063** | ~360ms |

- per-PII recall(형식+문맥): rrn 1.0 / account 0.946 / **card 0.852**.
- STT 품질(일반발화): **WER 중앙값 0, 완벽일치 80%**, 환각 1개 → 합성음 거의 완벽(best-case).

## 해석
- STT+NLP(형식+문맥)이 **precision 0.951 / FPR 0.063** 으로 CNN 약점을 문맥으로 해결(0012 진단과 일치).
- CNN 은 recall 0.989 + **약 1000배 빠름** → 상보적. 2단계(CNN 프리필터→STT+NLP 확정)가 자연스러운 결론.

## 이슈 / 한계
- **합성 STT 유리 편향**: WER 중앙값 0(best-case). 실통화선 WER↑ 로 STT+NLP 열화 예상 → 위 우위는 합성 한정 상한.
- best-case 에서도 **카드(16자리) recall 0.852** — Whisper 가 긴 숫자열 자릿수 누락/분절(쉼표 삽입 등). STT+NLP 구조적 약점.
- 디버깅 기록: 초기 NLP 가 한글 음절('이에요'→2 등) 과잉변환으로 card recall 0.31 까지 떨어짐 → Whisper 가 아라비아 출력이므로 변환 제거(연속4자�+ 숫자없을때만 fallback) → 0.852 회복. WER 평균은 환각 1개로 왜곡 → 중앙값/완벽일치율로 보고.
- 규칙기반 NLP 는 이진 출력 → ROC/PR 대신 운용점 지표로 비교.

## 다음 (Phase 4)
- 실음성 열화 민감도: 8kHz/μ-law 코덱 + 잡음(SNR) + 리버브 → 재전사·재평가로 STT+NLP 열화곡선(합성 편향 보정). CNN 도 동일 열화로 대조.
- 2단계 파이프라인(CNN→STT+NLP) 지연·비용·정밀 결합 검증.
