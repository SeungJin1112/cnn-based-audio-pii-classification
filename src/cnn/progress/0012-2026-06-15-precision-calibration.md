---
date: 2026-06-15
phase: 후속연구 1순위 — precision / calibration
status: done
---

# 0012 — Precision/Calibration 분석 + 누수 원인 진단

## 무엇을 했나
- `calibration_analysis.py`: threshold-독립 성능(ROC/PR) + temperature/Platt scaling(ECE) + target-recall 운용점 분석.
- `precision_diagnosis.py`: hard-neg 카테고리(template_id)별 FPR 분해 + PII 유형별 recall.
- 대상: v2 Qwen 데이터(14화자·4 PII), simple_cnn, in-distribution(template-disjoint) split.

## 왜
- v2 약점 = 고정 0.5 에서 hard-neg 과다검출(FPR 0.25~0.54). "프리필터로 쓸 만한가"를 운용점/원인 차원에서 정량화.

## 결과 / 수치
**calibration (`runs/calibration.json`)**
- threshold-독립 상한: ROC 0.901, PR-AUC 0.915 (랭킹은 강함).
- ECE: raw 0.135 → temperature(T=2.85) 0.142(악화) → Platt 0.112(소폭 개선).
- 운용점(val 에서 target recall 맞춰 test 적용):
  | target | test recall | precision | hard-neg FPR | STT 호출률 |
  |---|---|---|---|---|
  | ≥0.99 | 1.000 | 0.663 | 0.676 | 0.861 |
  | ≥0.95 | 0.995 | 0.679 | 0.627 | 0.837 |
  | ≥0.90 | 0.968 | 0.723 | 0.493 | 0.764 |
- → **calibration 으로 precision 안 풀림**(ECE 거의 불변, temp 는 악화). 문제는 확률 보정이 아니라 **표현/데이터 차원**. 고recall 운용 시 STT 호출률 86%(절감 14%뿐) — v1(42% 절감)보다 프리필터 효용 낮음.

**누수 원인 진단 (`runs/precision_diagnosis.json`, thr 0.5)**
- hard-neg 카테고리별 FPR: **order(주문번호 9~11자리) 0.733** ≫ count 0.365 > seat 0.156.
- PII recall: 카드 1.0 / 계좌 0.982 / 주민 0.865.
- → 오검출은 **PII 와 길이·그룹핑이 겹치는 긴 연속 숫자열(주문번호)** 에 집중. 자릿수로 읽으면 전화/계좌와 음향 형식이 사실상 동일 → **형식만으로는 구분 불가능한 본질적 한계**. 짧은 숫자(좌석)는 안전.

## 이슈 / 한계
- template-disjoint split 탓에 test 에 hard-neg 카테고리 3종(order/count/seat)만 등장 → 전 카테고리 진단은 seed/split 확대 필요.
- 결론: **precision 병목은 calibration 이 아니라 "PII 와 형식이 충돌하는 숫자열"** 이다. post-hoc 보정으로는 못 넘는다.

## 다음 (precision 개선 방향 재설정)
- **문맥 단서 활용**: 윈도/특징을 늘려 "주문번호/계좌번호" 같은 **선행 단어(carrier)** 를 더 잡게 → 형식 충돌을 문맥으로 분리.
- **충돌형 hard-neg 강화 학습**: order 류(9~13자리 비PII)를 train 에 더 투입.
- 형식-only 의 한계를 인정하고 **경량 문맥 결합**(예: 키워드 음향 + 형식) 고려.
- 실데이터 검증(2순위)으로 이동.
