---
date: 2026-06-14
phase: 본 학습 + 평가 + 결론
status: done
---

# 0006 — 본 학습(4모델×3시드) + 평가 결과 + 결론

## 무엇을 했나
- 4모델(simple_cnn / mobilenet_v3_small / efficientnet_b0 / convnext_tiny) × seeds[42,43,44] = 12런 학습.
- 5개 평가셋 전체 평가 → `runs/evaluation.json`, `reports/results.md`(자동), `reports/conclusion.md`(해석).
- `infer.py` — 단일 wav 추론 데모. positive→P 0.53(검출), hard-neg→P 0.04(미검출) 정상 확인.

## 결과 / 수치
- **CNN 4종 모두 baseline(acoustic-LR ROC 0.88) 능가.** simple_cnn 최고: controlled F1 0.955 / ROC 0.997.
- **artifact 통제 통과**:
  - length-matched F1 ≈ controlled F1 (simple_cnn 0.956 vs 0.955) → 길이 단서 의존 아님 (duration-only는 length-matched ROC 0.53로 붕괴).
  - hard-neg FPR 0.006~0.056 → "숫자 존재"가 아니라 "전화번호 형식" 학습.
- **프리필터 가치(보조 가설) 입증**: simple_cnn thr 0.2에서 recall 0.998 / STT 절감 41.6%, thr 0.3에서 recall 0.994 / 44.9%.
- **경량 모델 우위**: simple_cnn(60K, 0.26MB, 0.34ms)이 convnext_tiny(28M, 111MB)를 능가. ConvNeXt는 ±0.13으로 불안정.

## 이슈 / 한계
- 합성 TTS·화자 2~3명 → val F1 ≈ 1.0은 통제 합성 환경의 분리 가능성. 실데이터는 더 어려움.
- Original(naive) F1 < controlled (0.82 vs 0.96): artifact가 아니라 학습(한국어 발음)↔평가(숫자 그대로) 도메인 시프트.
- speaker-disjoint 미적용(화자 한계) → 화자 누수는 통제 불가 confound로 명시.

## plan 충족 여부
- plan.md §14 로드맵 1~8 전부 완료(데이터 통제·전처리·split·baseline·스모크·학습·평가·결론).
- 주 가설(feasibility)·보조 가설(프리필터)·artifact 통제 모두 plan §11 기준으로 방어 가능하게 결론.

## 다음 (범위 밖 / 추후)
- 실데이터·다화자 검증, STT+NLP 정설 경로 직접 비교, 전화번호 외 PII 확장, 윈도/벤치마크 ablation(§15).
