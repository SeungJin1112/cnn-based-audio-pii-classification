---
date: 2026-06-14
phase: 화자 확장 + PII 확장 (Phase 4 — 학습/평가 결과)
status: done
---

# 0010 — Qwen 14화자 · 4 PII유형 데이터 학습/평가 결과

## 무엇을 했나
- 데이터(Phase 2+3): `generate_qwen.py` 로 **1800클립** 생성 — positive 900(전화/주민/계좌/카드 각 225) + hard-neg 450 + easy-neg 450. 14화자(Qwen VoiceDesign ref→Base clone), 한국어 자릿수 발음, 16kHz mono. (`data_qwen/`)
- 실험(Phase 4): `speaker_pii_experiment.py`, simple_cnn, 기존 engine/metrics/models/splits 재사용.
  - in-distribution(template-disjoint split), 14화자 **LOSO**(고정 0.5 vs val-튜닝 threshold), **held-out-PII**(rrn 제외 학습→rrn 평가).

## 왜
- v1 약점 2개를 정량적으로 메우기 위해: (1) 화자 3명 → LOSO 불안정(F1 0.616±0.28), (2) 전화번호 단일 PII.

## 결과 / 수치
**in-distribution** (simple_cnn): F1 0.826, recall 0.989, precision 0.708, ROC 0.902, PR 0.919, **hard-neg FPR 0.542**.
- per-PII recall: rrn 0.962, account 1.000, card 1.000. (phone 은 template-disjoint 배분상 test 에 미포함 → LOSO 로 커버)

**LOSO (14화자, speaker-disjoint)** — v1 대비 핵심 개선:
| 지표 | v1 (3화자) | **v2 (14화자)** |
|---|---|---|
| F1@0.5 | 0.616 ± 0.28 | **0.861 ± 0.102** |
| recall@0.5 | 0.712 ± 0.41 | 0.891 ± 0.177 |
| ROC-AUC | 0.992 | 0.964 ± 0.029 |
| hard-neg FPR | — | 0.251 ± 0.178 |
- val-튜닝 threshold: F1 0.855±0.094 ≈ 고정 0.5 → 14화자에선 운용점이 이미 안정(v1 calibration 약점 크게 완화).

**held-out-PII (rrn 학습 제외 → rrn 평가)**: held-rrn **recall 0.978**, overall F1 0.720, ROC 0.935.
- → 주민번호를 한 번도 학습하지 않고도 탐지. "숫자 암기"가 아니라 **"PII 형식 숫자열" 일반화** 입증(가장 강한 증거).

## 이슈 / 한계
- **precision/hard-neg FPR 이 약점**: 4 PII유형 + 더 어려운 숫자열 hard-neg(일련번호 12자리 등)로 과제가 어려워져, 고정 0.5 에서 비PII 숫자열을 과다검출(in-dist FPR 0.542, LOSO 0.251). threshold-독립 ROC 는 견고(0.90~0.96)하나 운용 precision 개선 필요.
- 여전히 **합성 음성**(이제 단일 Qwen TTS 계열) → 실통화 gap 잔존. positive/hard/easy 가 동일 14화자·동일 백엔드라 출처 shortcut 은 통제됨.
- 화자 14명 중 최근접 쌍(spk01/spk09) 분리도 marginal(0009).
- in-dist phone recall 미측정(template-disjoint 배분).

## 다음
- `reports/conclusion.md` v2 섹션 갱신(아래 반영).
- (추후) precision 개선: 더 어려운 hard-neg 비중↑, calibration(temperature/Platt), threshold-독립 운용. 실데이터 검증. 화자 수·다양성 추가 확대.
