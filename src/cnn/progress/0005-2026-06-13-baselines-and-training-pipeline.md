---
date: 2026-06-13
phase: Baseline + 학습/평가 파이프라인 구현
status: done
---

# 0005 — baseline 결과 + 모델/학습/평가 엔진 구현

## 무엇을 했나
- `metrics.py` — 분류 지표(F1/precision/recall/ROC-AUC/PR-AUC) + hard-neg FPR + threshold 운용표(STT 호출률/절감률).
- `features.py` — acoustic feature(duration, RMS, ZCR, spectral centroid mean/std), 파일 캐시.
- `baselines.py` — majority / duration-only LR / acoustic-feature LR, seed 3회 집계 → `runs/baselines.json`.
- `models/` — `simple_cnn`(1ch) / `mobilenet_v3_small` / `efficientnet_b0` / `convnext_tiny`(3ch) + `build_model` 팩토리.
- `engine.py` — clip-level sliding-window **max** 추론(`predict_clip_probs`), AdamW+Cosine+CE+early-stopping 학습 루프.
- `dataset.py` — train 전용 SpecAugment(freq/time mask) 추가.
- `train.py` — (model, seed) 학습 → `runs/<model>_seed<seed>/`(checkpoint+train.json). split은 seed 결정성으로 evaluate와 일치.
- `evaluate.py` — 5개 평가셋 × 지표 + threshold표 + 효율성(params/size/GPU latency), seed 집계, `reports/results.md` 자동 생성(§11 해석 매트릭스 포함).

## 왜
- plan §5/§9/§12. CNN 결과를 해석하려면 baseline(특히 duration-only, acoustic-LR)이 기준선으로 필요.

## 결과 / 수치 (baseline, seeds=[42,43,44])
| 모델 | controlled F1 / ROC | length-matched ROC | hard-neg FPR |
|---|---|---|---|
| majority | 0.00 / 0.50 | 0.50 | — |
| duration-only | 0.63 / 0.70 | **0.53 (≈우연)** | 0.81 |
| acoustic-LR | 0.76 / 0.88 | 0.84 | 0.26 |

- **핵심 진단 작동**: duration-only가 length-matched에서 ROC 0.53으로 우연 수준 추락 → 길이 단서가 제거되면 무력화(설계 의도 확인).
- CNN이 넘어야 할 바: **acoustic-LR (controlled ROC 0.88 / length-matched 0.84)**.
- 학습/평가 파이프라인 스모크 통과(end-to-end, 체크포인트 저장·리포트 생성 확인).

## 다음
- 0006: 본 학습(4모델×3시드) + 전체 평가 결과·결론.
