---
date: 2026-06-15
phase: 화자/PII 확장 (held-out PII 전 유형 + 샘플 검증)
status: done
---

# 0011 — held-out PII 전 유형 일반화 + viz 4샘플 실제 예측

## 무엇을 했나
- held-out PII 테스트를 **3개 비전화 유형 전부**로 확장(`speaker_pii_experiment.py --heldpii --held {rrn,card,account}`).
  각 유형을 학습에서 완전히 제외 → 그 유형으로만 평가.
- `viz_melspec.py` 에 쓴 4개 대표 샘플을 **학습에서 제외 후 실제 예측**(`predict_viz_samples.py`)으로 검증.
- log-mel 시각화 영어 라벨본 생성(`reports/melspec_examples.png`).

## 왜
- 0010 의 held-out 일반화가 주민번호(rrn) 1종에만 입증됨 → "PII 형식 일반화"가 유형 전반에 걸치는지 확인 필요.
- viz 이미지는 시각화용 선택이었을 뿐 → 그 샘플들이 실제로 맞는지 정직하게 확인.

## 결과 / 수치
**held-out PII (각 유형 학습 제외 후 그 유형 평가, simple_cnn):**
| 제외 PII | held-out recall@0.5 | overall F1 | ROC |
|---|---|---|---|
| 주민번호 rrn | 0.978 | 0.720 | 0.935 |
| 카드 card | 0.911 | 0.878 | 0.968 |
| 계좌 account | 0.884 | 0.831 | 0.938 |

→ **세 유형 모두 학습 없이 recall 0.88~0.98 로 검출**. "특정 PII 암기"가 아니라 "PII 형식 숫자열" 공통 특징을 유형 너머로 일반화함이 단일 유형이 아닌 전 유형에서 재현.

**viz 4샘플 실제 예측 (4개를 학습에서 제외, threshold 0.5):**
| 샘플 | 정답 | P(PII) | 판정 |
|---|---|---|---|
| 전화번호 | 1 | 0.973 | PII ✅ |
| 주민번호 | 1 | 0.989 | PII ✅ |
| hard-neg 숫자열 | 0 | 0.107 | non-PII ✅ |
| 일반 발화 | 0 | 0.030 | non-PII ✅ |
→ 4개 전부 정답(학습 제외 상태).

## 이슈 / 한계
- held-out overall F1(0.72~0.88)은 held recall 보다 낮음 — precision 약점(비PII 숫자열 과다검출)이 v2 공통 한계로 여전. recall 은 견고.
- 합성(Qwen) 음성 한정은 동일.

## 다음
- (추후) precision/calibration 개선, 실데이터 검증. 본 결과는 `runs/qwen_experiment*.json`·로그 기반.
