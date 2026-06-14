# 결론 — 오디오 전화번호 형식 검출 feasibility (v1)

> 손으로 작성한 해석 문서. 수치는 `results.md`(자동 생성) / `runs/evaluation.json` 근거.
> plan.md §11(정직한 결론 톤)·§19를 따른다.

## 핵심 결과

**주 가설(feasibility) — 성립.** 멜 스펙트로그램 기반 CNN 4종 모두 baseline(majority/duration-only/acoustic-LR)을 유의미하게 능가했다.

| Model | Controlled F1 | Length-matched F1 | Hard-neg FPR | ROC-AUC |
|---|---|---|---|---|
| duration-only (baseline) | 0.63 | 0.64 (ROC 0.53) | 0.81 | 0.70 |
| acoustic-LR (baseline) | 0.76 | 0.77 | 0.26 | 0.88 |
| **simple_cnn** | **0.955** | **0.956** | **0.006** | **0.997** |
| efficientnet_b0 | 0.867 | 0.871 | 0.011 | 0.994 |
| convnext_tiny | 0.804±0.13 | 0.808 | 0.056 | 0.962 |
| mobilenet_v3_small | 0.776 | 0.777 | 0.021 | 0.945 |

## artifact 통제 검증 (가장 중요)

설계가 의도한 통제 실험이 모두 통과했다:

1. **길이 단서 의존 아님.** duration-only baseline은 length-matched에서 ROC 0.70→0.53(≈우연)으로 붕괴했지만, **CNN은 length-matched F1이 controlled와 사실상 동일**(simple_cnn 0.956 vs 0.955). 즉 길이로 맞히는 게 아니다.
2. **"숫자 존재"가 아니라 "전화번호 형식"을 학습.** hard-negative(날짜·주문번호·가격 등 숫자-비전화번호)에 대한 **FPR이 0.006~0.056**으로 매우 낮다. 모델은 "숫자가 들린다"가 아니라 전화번호 형식 패턴을 구분한다.
3. seed 3회 반복에서 simple_cnn/efficientnet은 안정적(±0.01). convnext_tiny만 ±0.13으로 불안정.

## 보조 가설(프리필터 가치) — 성립

simple_cnn controlled_main 운용점(seed 평균):

| threshold | recall | precision | STT 호출률 | STT 절감률 | hard-neg FPR |
|---|---|---|---|---|---|
| 0.2 | 0.998 | 0.887 | 0.584 | **41.6%** | 0.089 |
| 0.3 | 0.994 | 0.933 | 0.551 | **44.9%** | 0.061 |
| 0.5 | 0.927 | 0.986 | 0.485 | 51.5% | 0.006 |

→ **recall ~99.8%를 유지하면서 STT 호출을 ~42% 절감.** 게다가 가장 가벼운 모델이 최고 성능.

## 효율성 — 경량 모델이 최선

| Model | Params | Size MB | GPU ms/window |
|---|---|---|---|
| **simple_cnn** | **60,706** | **0.26** | **0.34** |
| mobilenet_v3_small | 1.5M | 6.2 | 2.03 |
| efficientnet_b0 | 4.0M | 16.3 | 2.92 |
| convnext_tiny | 27.8M | 111.3 | 2.32 |

직접 설계한 **60K 파라미터 simple_cnn이 28M ConvNeXt를 능가**. 프리필터 목적에 경량 모델이 정확히 부합하며, ConvNeXt(성능 상한 비교군)는 소량 데이터에서 오히려 불안정.

## 한계 (정직한 명시)

- **합성 TTS·화자 2~3명** 데이터. val F1 ≈ 1.0은 이 통제 합성 환경에서 과제가 *분리 가능*함을 뜻하며, 실데이터·다화자에서는 더 어려울 것이다.
- **Original(naive) F1이 controlled보다 낮음**(simple_cnn 0.82 vs 0.96). 이는 artifact 문제가 아니라, controlled(한국어 발음 "공일공")로 학습한 모델에 naive(숫자 그대로 "010-…") 낭독 스타일이 분포 밖이기 때문(train/test 도메인 시프트).
- **speaker-disjoint 미적용**(화자 수 한계) → 화자 누수는 제거 못 한 confound로 명시. template-disjoint·length-stratified는 적용.
- 본 결과는 **전화번호 형식 숫자열의 음향적 존재 탐지**에 한정. 문맥상 실제 PII 여부, STT 정설 경로 대체 가능성은 주장하지 않는다.

## 한 줄 결론 (v1)

> STT 없이 멜 스펙트로그램 + 경량 CNN만으로, **길이·숫자존재 artifact에 의존하지 않고** 음성 내 전화번호 형식 숫자열을 높은 신뢰도로 선별할 수 있음을 통제 실험으로 확인했다. high-recall 운용점에서 STT 호출을 ~42% 줄이는 경량 프리필터로서의 가치도 입증됐다. (합성 데이터 한정, 실데이터 검증은 추후 과제.)

---

# v2 — 화자 확장(14명) + PII 이진 확장 (2026-06-14)

> v1 의 두 약점(화자 3명 한계 → LOSO 불안정 / 전화번호 단일 PII)을 메우기 위한 후속 실험.
> 데이터: Qwen3-TTS(VoiceDesign 으로 14화자 설계 → Base 로 일관 clone) 1800클립. positive=전화/주민/계좌/카드, hard-neg=비PII 숫자열, easy-neg=일반 발화. 모두 동일 14화자·동일 백엔드(출처 shortcut 통제). 상세 수치는 `runs/qwen_experiment.json`, `progress/0008~0010`.

## 핵심 결과

**1. 화자 일반화가 안정적으로 굳어짐 (가장 중요).**
speaker-disjoint LOSO 가 화자 3→14 확장으로 크게 개선·안정화됐다.

| LOSO 지표 | v1 (3화자) | **v2 (14화자)** |
|---|---|---|
| F1@0.5 | 0.616 ± **0.28** | **0.861 ± 0.10** |
| recall@0.5 | 0.712 ± 0.41 | 0.891 ± 0.18 |
| ROC-AUC | 0.992 | 0.964 ± 0.03 |

→ 처음 보는 화자에서도 F1 0.86 으로 안정(분산 1/3). v1 의 "운용점이 화자마다 어긋난다"는 문제도 크게 완화(val-튜닝 threshold 0.855 ≈ 고정 0.5).

**2. PII 형식이 *유형을 넘어* 일반화됨 (숫자 암기 아님).**
주민등록번호(rrn)를 **학습에서 전부 제외**(전화+계좌+카드만 학습)하고 평가했을 때 **held-out rrn recall = 0.978**. 모델은 특정 PII 패턴이 아니라 "PII 형식 숫자열"이라는 일반화된 신호를 학습했다.
in-distribution per-PII recall 도 rrn 0.96 / account 1.0 / card 1.0 으로 4종 모두 탐지.

## 한계 (정직한 명시 — v2)

- **precision 이 새 약점**: 4 PII유형 + 더 어려운 숫자열 hard-neg(일련번호 12자리 등)로 과제가 어려워져, 고정 0.5 에서 비PII 숫자열을 과다검출(in-dist hard-neg FPR 0.542, LOSO 0.251, in-dist precision 0.708). threshold-독립 ROC 는 견고(0.90~0.96)하나 **운용 precision/calibration 개선이 다음 과제**.
- 여전히 **합성 음성**(이제 단일 Qwen TTS 계열) → 실통화 gap 잔존.
- 화자 14명은 설계 합성 화자(최근접 쌍 분리도 marginal). 실화자 다양성과는 다름.

## 한 줄 결론 (v2)

> 화자를 14명으로 늘리자 speaker-disjoint 성능이 **F1 0.86 으로 안정화**(v1 0.62±0.28 → v2 0.86±0.10)됐고, **학습하지 않은 PII 유형(주민번호)도 recall 0.98 로 탐지**되어 모델이 "PII 형식"을 유형 너머로 일반화함을 확인했다. 즉 STT 없는 음향 PII 선별의 **feasibility(분리·일반화)는 화자/유형 양축에서 강해졌다.** 남은 과제는 운용 precision(calibration)과 실데이터 검증이다.
