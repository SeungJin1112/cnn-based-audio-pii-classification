# 논문 상세 아웃라인 — STT 없는 음향 기반 PII 탐지의 Feasibility

> 타깃: 기술 리포트 / 졸업논문 (분량 여유). 중심 주장: **Feasibility 입증**.
> 근거 문서: `SUMMARY.md`, `reports/{conclusion,results,cnn_vs_sttnlp,why_simplecnn_and_feasibility}.md`, `progress/0001~0013`, `runs/*.json`.
> 이 파일은 절→근거→주장→들어갈 표/그림을 매핑한 집필 설계도.

---

## 제목(안)
- 국문: "STT 없이 음향 특징만으로 개인정보(PII) 형식 숫자열을 탐지하는 경량 CNN의 타당성 연구"
- 영문: "Feasibility of STT-Free Acoustic Detection of PII-Format Digit Strings with a Lightweight CNN"

## 초록 (Abstract) — 마지막에 작성
- 문제: 음성 PII 탐지의 표준은 ASR→NLP 캐스케이드인데, 비용·지연·민감 텍스트 중간산출·전사오류 전파 문제.
- 제안: STT를 건너뛰고 멜 스펙트로그램 + 경량 CNN이 "PII 형식 숫자열"을 직접 판별.
- 핵심 결과 3개: (1) baseline 초과 + artifact 4겹 통제로 feasibility 입증(F1 0.955, hard-neg FPR 0.006), (2) 화자·PII유형 이중 일반화(LOSO F1 0.86, held-out-PII recall 0.88~0.98), (3) STT+NLP 대비 recall·속도 우위(~1000×)로 2단계 프리필터 가치.
- 경계: 합성 TTS 한정, 운용 precision은 후속과제.

---

## 1. 서론 (Introduction)
**1.1 배경·동기**
- 콜센터·상담 녹취에 전화·주민·계좌·카드번호 등 PII가 자주 노출 → 규제(개인정보보호법) 준수를 위한 자동 탐지·마스킹 수요.
- 표준 파이프라인: 음성 → ASR 전사 → NER/정규식 PII 탐지·레닥션. (근거: Azure Conversation PII, Gladia, Kixie 가이드)

**1.2 문제 제기 (표준 파이프라인의 한계)**
- (a) STT 비용·지연: 전 발화를 무조건 전사 → 대량 녹취에서 비쌈.
- (b) 민감 텍스트 중간산출: 전사 텍스트 자체가 PII 유출 표면.
- (c) 전사오류 전파: "전사 시 구어 숫자를 단어로 오인식" → 긴 숫자열에서 자릿수 누락(본 연구 실험에서 Whisper card recall 0.852로 재현). (근거: Kixie/Limina 가이드, cnn_vs_sttnlp.md)

**1.3 연구 질문**
- RQ1: STT 없이 음향 특징만으로 PII 형식 숫자열을 탐지하는 것이 **원리적으로 가능한가**(feasibility)?
- RQ2: 모델이 진짜 "PII 형식"을 학습하는가, 아니면 길이·숫자존재·화자 같은 artifact로 맞히는가?
- RQ3: 학습하지 않은 화자·PII유형으로 일반화되는가?
- RQ4: 정설 경로(STT+NLP) 대비 어떤 운용 위치(프리필터)가 합리적인가?

**1.4 기여 (Contributions)**
1. STT-free 음향 PII **형식** 탐지 과제를 정식화하고, artifact를 차단한 통제 데이터셋·평가 프로토콜 설계.
2. 4겹 통제(baseline 초과·길이·숫자존재·출처)로 feasibility를 **방어 가능하게** 입증.
3. 화자(LOSO)·PII유형(held-out) 이중 일반화 검증 — "형식을 유형 너머로 일반화".
4. STT+NLP 정면 비교로 상보성 규명 → **CNN 프리필터 → STT+NLP 확정** 2단계 설계 제시.
5. 60K 경량 CNN이 28M ImageNet 백본을 능가함을 보이고 원인 분석.

**1.5 논문 구성**

---

## 2. 관련 연구 (Related Work) — 4갈래 + 위치잡기
**2.1 ASR→NLP 캐스케이드 PII 레닥션** (표준·비교 대상)
- Azure Conversation PII, 상용 레닥션 파이프라인: NER+패턴매칭. 한계: 전사오류·문맥분절 취약(구어 숫자→단어 오인식). → 본 연구가 우회하려는 대상.

**2.2 ASR-free / End-to-End SLU** (가장 가까운 방법론적 이웃)
- 음성에서 의미(intent/slot)를 ASR 없이 직접 추론, ASR 오류전파 회피·시스템 단순화가 동기. (arXiv 1910.10599, 2305.02937, 1909.13332)
- 차이: 기존 E2E SLU는 intent/slot(Fluent Speech Commands)에 집중. 본 연구는 **PII "형식" 이진 탐지 + 프리필터 목적 + feasibility 통제**가 초점.

**2.3 Keyword Spotting(KWS) / Spoken Digit Recognition** (기법적 이웃)
- 저전력 always-on에서 멜/ MFCC + 경량 CNN으로 키워드·숫자 탐지, 풀 ASR 불필요. (arXiv 2104.00769, MATLAB spoken-digit CNN)
- 차이: KWS는 특정 어휘/개별 숫자 인식. 본 연구는 개별 숫자가 아니라 **"PII 형식 숫자열의 존재"라는 상위 패턴**을 탐지(hard-neg로 '숫자존재≠형식' 분리).

**2.4 음성 콘텐츠 프라이버시** (동기적 이웃)
- 콘텐츠 프라이버시 과제, 의도적 열화로 내용 은닉하며 분류는 유지. (arXiv 2301.08925, Preech 1909.04198, MASK 2510.18493)
- 차이: 본 연구는 "민감 텍스트를 아예 만들지 않는" 탐지 단계 자체의 프라이버시 이점을 설계 동기로 삼음.

**2.5 연구 공백(Gap) 정리** — 표 1
> 선행연구 어디에도 "PII 형식을 음향에서 직접, 경량 프리필터로, artifact 통제하 feasibility로 입증하고 화자·유형 일반화까지 확인"한 사례 없음 → 본 연구 위치.

---

## 3. 과제 정식화 및 데이터셋 (Task & Dataset)
**3.1 과제 정의**
- 입력: 5초 음향 윈도(멜 스펙트로그램). 출력: PII 형식 숫자열 포함 이진(1/0).
- 라벨: positive=전화/주민/계좌/카드, negative=hard(비PII 숫자열)+easy(일반 발화).

**3.2 데이터 생성 (통제 설계가 핵심)**
- v1: edge-tts 3화자, 전화번호 단일. (`generate/`, `progress/0002~0003`)
- v2: Qwen3-TTS 14화자, 4 PII유형, 1800클립. (`data_qwen/`, `progress/0008~0010`)
- **읽기 방식 통제**: 모든 숫자를 한국어 자릿수 발음("공일공…") → 클래스 차이가 '읽기 스타일'이 아니라 '숫자열 형식'.
- **출처 지문 통제**: 노이즈 패딩(무음 대신) + 전 클래스 동일 화자·동일 TTS·동일 ffmpeg → 코덱/화자 shortcut 차단.

**3.3 화자 일관성 방법론** (v2 기여) — `progress/0008~0009`
- VoiceDesign 직접 호출은 발화마다 변동(within>between, ratio 0.61) → **VoiceDesign으로 reference 1개 설계 후 Base로 일관 clone**(within 0.0106≪between, ratio 2.29). 14화자 voicebank 확정.
- (그림) within/between 화자 유사도 분포.

**3.4 데이터 통계** — 표: 클래스·화자·유형별 클립 수, 길이 분포.

---

## 4. 방법 (Method)
**4.1 특징 추출** (`config.py`, `features.py`)
- 16kHz mono, log-mel: n_mels=64, n_fft=1024, hop=512, power=2.0. 5초 윈도 → 64×157.
- 하이브리드 윈도: train=대칭 random-crop, eval=sliding-window(hop 2.5s)+max 집계.

**4.2 모델** (`models/`)
- simple_cnn: Conv(3×3)-BN-ReLU-MaxPool 4블록(16→32→64→64) + GAP + Dropout0.3 + Linear. 60,706 params.
- 비교군: mobilenet_v3_small, efficientnet_b0, convnext_tiny (ImageNet pretrained, 1→3채널 복제).

**4.3 학습**(`engine.py`) — AdamW, CE, cosine, early-stopping(val). seed 42/43/44.

**4.4 Baseline**(`baselines.py`) — majority / duration-only / acoustic-LR(수작업 음향통계 7개).

**4.5 평가 프로토콜**(`splits.py`, `metrics.py`) — **누수 없는 측정이 feasibility의 전제**
- 70/15/15, **template-disjoint + length-stratified**, test는 학습·선택 미사용.
- 지표: F1/recall/precision, ROC-AUC, PR-AUC, **hard-neg FPR**(핵심), 지연.
- 평가셋 5종: controlled_main / length_matched / hard_negative_only / stress_5_5 / original.

**4.6 STT+NLP 비교 경로**(`stt/`) — Whisper-large-v3 전사 + 규칙 NLP(형식only / 형식+문맥). 동일 test split.

---

## 5. 실험 및 결과 (Experiments) — Feasibility를 축으로
**5.1 Exp1: Feasibility — baseline 초과** (표 2, `results.md`)
- simple_cnn F1 0.955 / ROC 0.997 vs acoustic-LR 0.76/0.88 → 전역 음향통계로 안 잡히는 시간-주파수 패턴 학습.

**5.2 Exp2: Artifact 통제 검증** (feasibility의 핵심)
- 길이: duration-only는 length-matched에서 ROC 0.70→0.53(우연) 붕괴, CNN은 0.956≈0.955 유지.
- 숫자존재≠형식: hard-neg FPR 0.006.
- (그림) 멜 스펙트로그램 예시 positive vs hard-neg (`reports/melspec_examples.png`).

**5.3 Exp3: 모델 비교 — 경량 우위** (표 3+효율표, `why_simplecnn_and_feasibility.md`)
- 60K simple_cnn > 28M convnext(0.955 vs 0.804±0.13). 원인 4가지: 소량데이터 과적합·도메인불일치·저수준과제·암묵정규화.

**5.4 Exp4: 화자 일반화 — LOSO** (표 4, `progress/0007,0010`)
- v1 F1 0.616±0.28 → v2 0.861±0.10, 분산 1/3. 운용점 안정화(val-튜닝 0.855≈고정 0.5).

**5.5 Exp5: PII 유형 일반화 — held-out-PII** (표 5, `progress/0011`) — 가장 강한 증거
- 주민 0.978 / 카드 0.911 / 계좌 0.884 (학습 완전 제외). viz 4샘플 4/4 정답.

**5.6 Exp6: 프리필터 가치** (표 6, `conclusion.md`)
- threshold 0.3: recall 0.994, STT 호출 44.9% 절감.

**5.7 Exp7: CNN vs STT+NLP** (표 7, `cnn_vs_sttnlp.md`)
- STT+NLP(형식+문맥) F1 0.935 최고 precision, 단 CNN recall 0.989·~1000× 속도.
- **합성 데이터 → STT best-case 편향** 반드시 명시(Whisper WER 중앙값 0).

---

## 6. 논의 (Discussion)
**6.1 무엇이 입증됐나 / 아직 아닌가** (`why_simplecnn_and_feasibility.md §2.6`)
**6.2 precision 약점의 원인 진단** (`precision_diagnosis.py`, `progress/0012`)
- v2 hard-neg FPR 0.25~0.54, 주문번호(9~11자리) FPR 0.733 집중 → 긴 연속 숫자열은 전화/계좌와 음향 형식 사실상 동일 = **형식-only의 본질적 상한**.
- calibration(temperature/Platt)으로 ECE 거의 불변 → 확률보정이 아니라 표현/데이터/문맥 차원 문제.
**6.3 설계 함의 — 2단계 파이프라인**: CNN(고recall·초저비용) → STT+NLP(고정밀·문맥). 상보성.
**6.4 프라이버시 관점**: 민감 텍스트 미생성 프리필터의 이점.

---

## 7. 한계 및 향후 연구 (Limitations & Future Work)
- 합성 TTS 한정 → 실통화(잡음·억양·코덱) 검증 필요(Phase 4).
- 운용 precision/calibration, 문맥 결합(선행 단어 "주문번호/계좌번호").
- 화자 14명은 설계 합성화자(최근접쌍 marginal) → 실화자 다양성.

## 8. 결론 (Conclusion)
> STT 없이 멜+경량CNN만으로, artifact 비의존으로, 화자·PII유형을 일반화하여 PII 형식 숫자열을 높은 recall로 선별 가능함을 통제 실험으로 입증. 운용 precision과 실데이터 검증이 남은 과제.

## 참고문헌 / 부록
- 부록 A: 하이퍼파라미터·화자 목록·템플릿 문장.
- 부록 B: seed별 원수치(`runs/*.json`), 추가 평가셋.

---
## 인용 후보 (Related Work)
- ASR-free SLU: arXiv 1910.10599, 2305.02937, 1909.13332
- KWS/Spoken digit: arXiv 2104.00769; MATLAB spoken-digit CNN
- 콘텐츠 프라이버시: arXiv 2301.08925, 1909.04198(Preech), 2510.18493(MASK)
- 상용 PII 레닥션: Azure Conversation PII, Gladia, Kixie/Limina 가이드
