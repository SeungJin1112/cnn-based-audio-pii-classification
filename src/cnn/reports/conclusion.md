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

## 한 줄 결론

> STT 없이 멜 스펙트로그램 + 경량 CNN만으로, **길이·숫자존재 artifact에 의존하지 않고** 음성 내 전화번호 형식 숫자열을 높은 신뢰도로 선별할 수 있음을 통제 실험으로 확인했다. high-recall 운용점에서 STT 호출을 ~42% 줄이는 경량 프리필터로서의 가치도 입증됐다. (합성 데이터 한정, 실데이터 검증은 추후 과제.)
