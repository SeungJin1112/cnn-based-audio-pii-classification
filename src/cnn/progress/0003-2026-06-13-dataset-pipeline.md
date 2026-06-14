---
date: 2026-06-13
phase: 데이터 파이프라인 구현
status: done
---

# 0003 — config.py + dataset.py 구현 및 검증 에이전트 통과

## 무엇을 했나
- `src/cnn/config.py` — 경로·오디오 파라미터·윈도·seed 중앙 설정.
- `src/cnn/dataset.py`:
  - `build_controlled_manifest()` — positive 1000 + easy 700(data/x 재사용) + hard 300 통합(1:1).
  - `build_original_manifest()` — naive(00001 vs data/x) 평가용.
  - `make_eval_sets()` — controlled-main / length-matched / hard-negative-only / stress-5:5 파생(+ Original 별도).
  - `wav_to_logmel()` — n_mels=64, n_fft=1024, hop=512, log + per-sample 정규화.
  - 하이브리드 윈도: `random_crop`(train, 대칭) / `sliding_windows`(eval) + 저진폭 노이즈 padding.
  - `AudioPIIDataset` — train=[C,64,157] 단일 / eval=[n_win,C,64,157] 반환.
- 스모크 테스트로 shape·라벨 균형·평가셋 구성 확인.

## 왜
- plan.md §3·§4·§7의 데이터/전처리 결정을 코드로 고정. 윈도는 찬반 종합 결정(대칭 crop + eval max)을 그대로 반영.

## 결과 / 수치
- controlled manifest 2000 (pos 1000 / neg 1000 = easy 700 + hard 300), 1:1.
- train tensor [1,64,157], 3채널 [3,64,157], eval [n_win,1,64,157] 정상.
- 평가셋: controlled_main 2000 / hard_negative_only 1300 / stress_5_5 1600 / length_matched 1690.
- **검증 에이전트**: 핵심 plan 항목(멜 파라미터·하이브리드 윈도·5초·노이즈 padding·7:3·1:1·5개 평가셋·3채널·seed) 전부 PASS.
- 에이전트 지적 반영:
  - easy negative `template_id` 단일값 → text 골격 해시로 파생(고유 247개) → template-disjoint 유효.
  - noise-pad를 전역 np.random → `self.rng`로 라우팅(재현성).
  - stress-5:5 positive 전부 유지로 pos:neg 불균형 → 해석 시 ROC/PR-AUC·per-class 지표 중심(코드 주석 명시).

## 이슈 / 한계
- data/x가 controlled-easy와 Original-neg에 공유됨 → Original 누수 가능. **splits.py에서 controlled-train filepath를 Original에서 제외**하도록 TODO 명시(코드 docstring).
- clip-level `max(window_scores)` 집계와 eval 가변 n_win(batch_size=1/커스텀 collate)은 `evaluate.py`에서 구현 예정.

## 다음
- 0004: `splits.py` — template-disjoint + length-stratified + stratified, seeds=[42,43,44], Original 누수 제외 처리.
