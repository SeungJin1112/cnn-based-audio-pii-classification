---
date: 2026-06-14
phase: 강건성 실험 (speaker-disjoint LOSO)
status: done
---

# 0007 — Leave-One-Speaker-Out(LOSO) 강건성 실험

## 무엇을 했나
- `src/cnn/speaker_robustness.py` 신규 구현 — speaker-disjoint 교차검증.
  - 발견: controlled manifest 의 화자는 **3명**(InJoon/SunHi/Hyunsu)이고 각자 양 라벨을 ~330/330 으로 고르게 보유 → LOSO 가능.
  - fold k: held-out = 화자 k 의 모든 clip(평가), train = 나머지 2 화자. val(early stopping)은 train-화자 안에서 template-disjoint 로 분리(누수 없음).
  - 기존 `engine/dataset/metrics/splits/models` 전부 재사용. `simple_cnn` 3 fold × seeds[42,43,44] = **9 런**.
- 결과: `runs/loso_simple_cnn.json`. 스모크(`--smoke`) end-to-end 통과 후 full 실행.

## 왜
- `reports/conclusion.md` 가 명시한 **유일한 미통제 confound = speaker 누수**(main 실험은 template/length-disjoint 만 적용, speaker-disjoint 는 "화자 수 한계"로 생략).
- 화자가 train/test 에 동시에 등장하면 모델이 "화자 정체성"으로 맞힐 수 있어, main 의 F1 0.955 가 화자 누수로 부풀려졌는지 검증 필요.

## 결과 / 수치
전체 9 fold 집계 (speaker-disjoint):

| 지표 | LOSO | (참고) in-distribution main |
|---|---|---|
| test F1 | **0.616 ± 0.280** | 0.955 |
| test recall | **0.712 ± 0.409** | 0.927 |
| test precision | 0.788 ± 0.171 | — |
| **test ROC-AUC** | **0.992 ± 0.008** | 0.997 |
| hard-neg FPR@0.5 | 0.339 ± 0.277 | 0.006 |

held-out 화자별 (seed 평균):

| held 화자 | F1 | recall | precision | ROC-AUC | hard-neg FPR |
|---|---|---|---|---|---|
| Hyunsu | 0.873 | 1.000 | 0.776 | 0.997 | 0.372 |
| InJoon | 0.234 | 0.135 | 1.000 | 0.984 | 0.000 |
| SunHi | 0.741 | 1.000 | 0.588 | 0.995 | 0.645 |

**핵심 해석**:
- **랭킹은 화자에 일반화된다.** held-out 화자 3명 모두 ROC-AUC 0.98~0.997 → 모델은 화자 정체성이 아니라 전화번호 형식의 음향 신호를 학습했다. 주 가설(feasibility/separability)은 speaker-disjoint 에서도 성립.
- **그러나 고정 0.5 임계값은 화자별로 어긋난다.** InJoon held → 과소예측(recall 0.135, 거의 전부 negative 판정), SunHi held → 과다예측(hard-neg FPR 0.645). 절대 확률 calibration 이 화자마다 이동.
- → main 의 절대 F1/recall/FPR(0.955/0.927/0.006)은 **in-distribution 임계 보정 덕에 부풀려진 값**이다. 분리 가능성은 진짜지만, 운용점(threshold) 성능은 새 화자에서 그대로 유지되지 않는다.

## 이슈 / 한계
- 화자 **3명뿐** → fold 간 분산이 크다(F1 ±0.28). 더 많은 화자가 있어야 안정적 추정 가능.
- 배포 관점: 새 화자에는 **화자별 calibration 또는 threshold 튜닝**이 필요. 고정 0.5 운용점 일반화는 미보장.
- `conclusion.md`/`results.md` 의 절대 운용 지표(STT 절감률 등)는 "in-distribution 화자" 단서를 달아 해석해야 함.

## 다음
- `reports/conclusion.md` 한계 항목에 LOSO 결과 반영(speaker 누수를 confound→정량화된 사실로 갱신).
- (범위 밖) 화자 수 확대 재실험, per-speaker/global calibration(temperature scaling) 후 운용점 재평가, ROC 기반(threshold-독립) 보고로 전환 검토.
