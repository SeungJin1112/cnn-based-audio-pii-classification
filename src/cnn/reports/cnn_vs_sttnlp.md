# CNN(음향) vs STT+NLP(정설 경로) 비교

> 동일 데이터(`data_qwen`)·동일 test split(seed 42, template-disjoint, n=331)·동일 라벨/지표.
> STT = Whisper-large-v3, NLP = 규칙기반(숫자 형식 / 형식+문맥). 수치: `runs/stt_nlp_eval.json`, CNN: `runs/qwen_experiment.json`.

## 메인 비교표 (같은 test 인스턴스)
| 접근 | F1 | recall | precision | hard-neg FPR | 지연 |
|---|---|---|---|---|---|
| **CNN (음향)** | 0.826 | **0.989** | 0.708 | 0.542 | **0.344 ms/win** |
| STT+NLP (형식only) | 0.883 | 0.921 | 0.849 | 0.218 | ~360 ms/clip |
| **STT+NLP (형식+문맥)** | **0.935** | 0.921 | **0.951** | **0.063** | ~360 ms/clip |

PII 유형별 recall (형식+문맥): rrn 1.0 / account 0.946 / **card 0.852**.

## 핵심 해석
- **정확도**: STT+NLP(형식+문맥)이 F1 0.935 로 최고. 특히 **precision 0.951 / hard-neg FPR 0.063** 으로,
  CNN 이 못 풀던 약점(주문번호 등 형식충돌 숫자열 과다검출, FPR 0.542)을 **문맥(선행 단어)으로 해결**.
  → 0012 에서 "precision 은 calibration 이 아니라 문맥이 필요"라 한 진단과 정확히 일치.
- **recall·속도**: CNN 이 recall 0.989(STT+NLP 0.921)로 더 높고, **약 1000배 빠르다**(0.344ms vs 360ms).
  STT+NLP recall 손실은 주로 **카드(16자리) 0.852** — Whisper 가 긴 숫자열에서 자릿수 누락/분절을 내기 때문(아래 편향 참조).
- **상보성**: CNN=고recall·초고속·저정밀, STT+NLP=고정밀·문맥강함·느림 → **2단계(CNN 프리필터→STT+NLP 확정)** 가 자연스러운 결론.

## ⚠️ 합성 데이터 → STT 유리 편향 (반드시 감안)
- STT 품질(일반발화 n=450): **WER 중앙값 0, 완벽일치 80%**, 환각(반복 루프) 1개. → 합성음에서 Whisper 가 **거의 완벽** = STT+NLP 는 **best-case(상한)**.
- 실통화(잡음·억양·코덱)에서는 WER↑ 로 STT+NLP 가 열화될 것. 즉 위 STT+NLP 우위는 **합성 환경 한정 상한**이며, 현실 격차는 더 좁거나 역전될 수 있다.
- 그럼에도 **card 0.852** 처럼 best-case 에서조차 긴 숫자열 recall 이 새는 점은 STT+NLP 의 구조적 약점.

## 결론
- 정확도(특히 precision)는 STT+NLP(형식+문맥)이 우위 — 단 **합성 best-case 기준**.
- 비용/지연·recall 은 CNN 이 압도(≈1000× 속도, recall 0.99).
- **실용 설계**: CNN 으로 고recall·초저비용 1차 선별 → STT+NLP 로 정밀 확정하는 **2단계 파이프라인**이 두 접근의 장점을 결합. (Phase 4 에서 실음성 열화 민감도와 함께 검증 예정.)
