# 오디오 기반 전화번호 PII 음성 프리필터 — feasibility 실험 설계 (plan.md)

> 본 문서는 외부 평가서(`rebuttal.md`, `rebuttal2_sec.md`)의 confound 지적과, 찬성/반대 두 검토를 종합해 개정한 버전이다.
> 핵심 변경: (1) 실험 범위·결론을 정직하게 낮춤, (2) confound 통제를 "주의사항"이 아닌 **실험 설계의 중심**으로 승격, (3) 통제된 데이터셋을 **새로 생성**, (4) baseline·평가셋·split·지표를 방어 가능한 수준으로 보강.
> 후보 최종본(`rebuttal2_sec.md`) 대비 결정: **하이브리드 윈도/집계 채택**(대칭 random-crop 5s + eval sliding-window max), **speaker-disjoint 미적용**(화자 2~3명 → 통제 불가 confound로 명시), **Stress-5:5 평가셋 추가**, 학습 설정 블록·결과 해석 매트릭스 흡수. 윈도 ablation·정식 벤치마크·RF·MIL 타임스탬프는 §15로 defer.

---

## 0. 한 줄 요약

STT 없이 `오디오 → 멜 스펙트로그램(이미지) → CNN`으로 **전화번호 낭독을 경량으로 선별(pre-filter)** 할 수 있는지를 검증하는 **feasibility test**.
높은 성능이 나오더라도 "PII 의미 이해" 또는 "정설 경로 대체"를 주장하지 않으며, **artifact 의존이 아님을 통제 실험으로 입증한 범위 안에서만** 결론을 낸다.

## 1. 실험 목적과 범위 (재정의)

정설(conventional) PII 탐지 경로:
```
오디오 → STT → 텍스트 → 정규식/NLP로 PII 패턴 탐지
```
본 실험이 검증하는 단순 경로:
```
오디오 → 멜 스펙트로그램(이미지) → CNN 계열 분류기로 "전화번호 포함 여부" 판별
```

### 탐지 대상 정의 (평가서 §6 + 최종판단 반영 — 매우 중요)
- 본 모델의 탐지 대상은 **"음성 안에 전화번호 형식의 숫자열이 음향적으로 존재하는지"** 이다.
- 그것이 문맥상 **실제 개인정보인지 / 예시인지 / 부정 표현("제 번호 아니에요")인지는 본 모델의 판단 대상이 아니다.** 그 문맥 판단은 후단 STT/NLP 단계의 역할로 둔다.
- 따라서 "전화번호 형식이 들리지만 예시/부정 문맥"인 발화는 **negative가 아니라 positive**로 라벨링한다(음향적으로 전화번호 낭독이 존재하므로 프리필터는 통과시켜야 함). 이렇게 해야 **실험 목적·라벨·프리필터 역할이 서로 일치**한다.
- 본 실험의 PII는 일반 개인정보 전체가 아니라 **전화번호 형식 숫자열에 한정**하며, 과제명은 **"음성에서 전화번호 형식 숫자열 검출(phone-number-pattern-in-speech detection)"** 에 가깝다.

### 가설 (claim을 낮춤, 평가서 §9 반영)
- **주 가설(feasibility)**: 멜 스펙트로그램 기반 CNN이 **전화번호 형식 숫자열이 들리는 음성**을 **그렇지 않은 음성**과 구분할 수 있는가.
- **보조 가설(pre-filter 가치)**: 성능이 정설 경로에 못 미쳐도, 경량성 + **high-recall 운용점**에서 STT 호출량을 줄이는 1차 프리필터로 가치가 있는가.
- **명시적 비주장**: STT+NLP baseline을 이번 범위에서 직접 돌리지 않으므로 **"정설 경로 대체 가능성(replacement)"은 본 실험만으로 주장하지 않는다.**

## 2. 핵심 위협: artifact detection (실험 설계의 중심)

> 모델이 잘 맞혀도, 그것이 "전화번호 PII를 감지했다"가 아니라
> **"긴 음성 / 숫자 또박또박 나열 / 발음 스타일 / 템플릿·화자 패턴을 감지했다"** 는 증거일 수 있다.

현재(naive) 데이터에서 확인된 confound:

| confound | 현재 상태 | 위험 |
|---|---|---|
| 음성 길이 | positive ≈6.7s vs negative ≈4.5s | 길이만으로 분류 가능 |
| 발음 스타일 | positive는 숫자 그대로, negative는 한국어 발음 | 발화 방식 차이로 분류 |
| 숫자 존재 여부 | negative에 숫자 거의 없음 | "숫자 나열 vs 아님" 문제로 전락 |
| 템플릿/화자 | 템플릿 기반 TTS, split 무통제 | train/test 누수, 암기 |
| padding | 8s pad → negative에 무음 영역 多 | 무음 길이로 분류 (치명적 leakage) |

→ 이 통제가 본 실험의 **성패를 결정**한다. 아래 §3~§8은 모두 이 위협을 제거/측정하기 위한 설계다.

## 3. 데이터셋 재설계 (새로 생성)

기존 데이터(`data/o/00001`, `data/x`)는 **naive(uncontrolled) 조건**으로 보존하여, "통제 전/후" 비교(ablation)에 사용한다. 그 위에 **통제된 데이터셋을 새로 생성**한다.

### 3.1 Positive (전화번호 형식 숫자열 존재) — 발음 스타일 통제
- `generate_o_2.py`(이미 존재)를 사용해 전화번호를 **한국어 발음("공일공 일이삼사 ...")** 으로 읽은 positive를 `data/o/00002`로 생성 → negative의 숫자 발음 방식과 정렬 (평가서 §1·§11-가능 반영).
- 다양한 삽입 템플릿/문장 길이로 생성하여 길이가 한쪽으로 쏠리지 않게 함.
- **문맥 변형 positive 포함**: "…는 예시입니다 / 제 번호는 아니에요" 같은 예시·부정 문맥에 전화번호 형식 숫자열이 들어간 발화도 **positive로 포함**한다. 음향적으로 전화번호 낭독이 존재하므로 모델은 이를 *문맥과 무관하게* 검출해야 한다(문맥 판단은 후단 몫). 이 샘플들이 모델이 형식 자체를 학습했는지(문맥 단서가 아니라) 확인하는 역할도 한다.

### 3.2 Negative — easy + **hard negative** 추가 (평가서 §5, easy:hard = 7:3)
모두 한국어 숫자 발음으로 통일. **negative의 정의는 "전화번호 형식 숫자열이 없음"** 이다(숫자 자체는 있어도 됨).

| 유형 | 내용 | 라벨 | 비중 |
|---|---|---|---|
| Easy negative | 일반 문장 (기존 `data/x` 재사용) | 0 | **70%** |
| **Hard negative** | 숫자는 있으나 **전화번호 형식이 아님** | 0 | **30%** |

Hard negative 카테고리(신규 스크립트 `generate_x_hard.py`로 생성) — **전화번호 형식(010-XXXX-XXXX류)이 아닌** 숫자열:
- 날짜: "오늘은 이공이육년 육월 십삼일입니다."
- 주문/송장번호: "주문번호는 이공이육공육일삼공공일입니다."
- 방/창구/버스 번호: "삼백일호로 오세요." / "삼번 창구로 가세요."
- 가격/수량: "가격은 만 오천 원입니다."
- 우편번호: "우편번호는 공육이삼사입니다."
- 단순 수 세기: "하나 둘 셋 넷 다섯."

> 목적: 모델이 "숫자 나열 vs 아님"이 아니라 "**전화번호 형식 vs 그 외 숫자열**"을 풀게 만든다. hard negative에서의 오탐(False Positive)이 실사용성의 핵심 지표.
> **주의(평가서 최종판단):** "전화번호 형식인데 예시/부정 문맥"은 hard negative가 **아니다** → §3.1의 positive로 분류한다. 문맥 판단을 CNN에 떠넘기지 않는다.

### 3.3 길이 통제 (평가서 §4)
- positive/negative 모두 **유사한 길이 분포**가 되도록 템플릿 설계 + 생성 후 길이 기록.
- 추가로 **length-matched 평가 subset**을 별도 구성(아래 §4).

## 4. 평가셋 분리 (평가서 §4 반영)

단일 test가 아니라 여러 조건의 평가셋으로 나눠 결론의 출처를 분리한다.

| 평가셋 | 구성 | 측정 목적 |
|---|---|---|
| **Original** | naive 데이터(00001 vs x) | 통제 전 성능(상한, artifact 포함) |
| **Controlled-main** | 00002(한국어발음) vs easy:hard=7:3 neg | **주 평가 조건** |
| **Length-matched** | 길이 분포 맞춘 subset | 길이 단서 제거 후 성능 |
| **Hard-negative-only** | hard negative만 | 숫자-비PII 오탐(FPR) 측정 |
| **Stress-5:5** | easy:hard = 5:5 (test 재가중) | hard negative 비중↑ 시 견고성 |

- 최종 주장은 **Controlled-main / Length-matched / Hard-negative-only** 중심.
- Stress-5:5는 *학습* 비율 7:3은 그대로 두고(결정 사항), **평가 시점에만** 기존 test 음성을 5:5로 재가중하는 것이라 추가 데이터 생성 비용이 거의 없다. "7:3이라 FPR이 좋아 보인 것 아니냐"는 반박을 선제 차단하는 용도.
- 그 외 추가 rigor(윈도 길이 ablation, 정식 벤치마크 프로토콜, RF·확장 feature)는 **v1에서 defer** → §15 참고.

> "Original에서 0.95인데 Controlled-main/Length-matched에서 급락"하면, 그 성능은 artifact 의존이었다는 강력한 증거.

## 5. Baseline (평가서 §2 — 반드시 추가)

CNN이 baseline을 못 이기면 "스펙트로그램 패턴을 배웠다"는 주장은 성립하지 않는다.

| Baseline | 설명 | 의미 |
|---|---|---|
| Majority | 다수 클래스 예측 | 클래스 균형 sanity (50%) |
| **Duration-only** | 길이 1-feature 로지스틱 회귀 | 길이만으로 얼마나 맞나 |
| Acoustic-feature LR | RMS, ZCR, spectral centroid, duration 등 + 로지스틱 회귀 | 단순 음향 특징의 상한 |
| (defer) Acoustic RF + 확장 feature bank | spectral bandwidth/rolloff/tempo/silence ratio 등 + random forest | v1 이후 — §15 |

- **v1 핵심 baseline = majority / duration-only / acoustic-feature LR.** RF와 확장 feature bank는 핵심 artifact 주장에 한계 가치라 v1에서는 defer.
- **CNN/ConvNeXt는 최소한 duration-only와 acoustic-feature LR baseline을 유의미하게 능가해야** 의미가 있다. 결과표에 baseline을 항상 함께 보고.

## 6. 모델 라인업 (평가서 §7 반영)

목표가 "경량 프리필터"이므로 모델을 역할별로 재배치한다.

| 모델 | 역할 | params(대략) |
|---|---|---|
| **Simple CNN** (직접 설계) | 경량 baseline | ~수십만~1M |
| **MobileNetV3-Small** | 실시간/모바일 프리필터 후보(주력) | ~2.5M |
| **EfficientNet-B0** | 소형 이미지 분류 비교군 | ~5M |
| **ConvNeXt-Tiny** | 성능 **상한** 확인용 (프리필터 후보 아님) | 28.6M |

- 입력: 멜 스펙트로그램 1채널 → 3채널 복제 또는 첫 conv 채널 수정.
- 모든 모델 **동일 입력 전처리 / 동일 split / 동일 seed**로 공정 비교.
- ConvNeXt-Tiny는 "CNN 계열이 이 입력에서 도달할 수 있는 성능 상한"으로만 해석.

## 7. 전처리 & leakage 통제 (평가서 §4)

- librosa 16kHz 로드 → **log-mel**: `n_mels=64`, `n_fft=1024`, `hop=512` (초기값).
- **고정 길이 처리에서 padding leakage 제거 (하이브리드 윈도 — 찬/반 종합 결정)**:
  - **학습**: positive·negative **동일 정책**으로 고정 길이 **random crop** (기본 **5초** 윈도). 양 클래스에 같은 crop 정책을 써서 *클래스별 비대칭 crop이 새로운 confound가 되는 것*을 방지(반대 측 지적 반영).
    - 5초로 둔 이유: 전화번호 한국어 낭독("공일공 일이삼사 오육칠팔")이 3초에 잘려 패턴이 훼손되는 위험 회피(찬성 측 지적 반영).
  - **평가**: 전체 clip을 sliding window로 나눠 각 window score 계산 후 **`clip_score = max(window_scores)`** 로 집계. "음성 *어딘가에* 전화번호 형식이 존재하는가"라는 문제 정의에 평균보다 max가 부합.
  - **채택하지 않음(defer)**: `phone_start/end` 타임스탬프 메타데이터 + positive-aware crop. positive label noise는 5초 윈도가 대부분의 낭독 구간을 포함시켜 완화하며, 비대칭 crop의 confound 위험이 더 크다고 판단. v1 결과가 baseline을 넘으면 재검토.
  - 불가피한 padding은 **무음(0) 대신 저진폭 노이즈**로 채우거나 padding mask 적용.
- per-sample 정규화.
- (시각화) 일부 샘플을 PNG로 저장해 "이미지 분류"임을 확인.

## 8. Split 전략 (평가서 §3)

stratified random만으로 부족 → 누수 차단 split 채택.

**적용하는 제약:**
- **Template-disjoint**: 동일 문장 템플릿이 train/test에 섞이지 않게 → 템플릿 암기 차단(핵심).
- **Length-stratified**: positive/negative 길이 분포 균형.
- **Stratified**: label 비율 유지.
- **Seed 고정 + 시드 3회 반복**(`seeds = [42, 43, 44]`) 후 평균±표준편차 보고(2000개 규모는 split 운에 민감).
- 비율 train/val/test = 70/15/15.

**적용하지 않는 제약 (정직한 한계 명시):**
- **Speaker-disjoint는 적용하지 않는다.** TTS 화자가 2~3명뿐이라, 한 화자를 test로 빼면 모델이 *처음 듣는 음색*을 평가받게 되어 전화번호 가설과 무관한 이유로 성능이 무너진다(또는 균형 잡힌 test 구성 자체가 불가). 따라서 화자 누수는 **제거할 수 없는 confound로 인정·명시**하고, template/length 통제로 대신한다. 보고서에 다음을 명시:
  > "TTS 화자 수 제한(2~3명)으로 speaker-disjoint split은 적용하지 않았으며, 화자 누수는 통제 불가 confound로 명시한다. template-disjoint와 length-stratified를 우선 적용했다."

## 9. 평가 지표

### (A) 분류 성능
- accuracy, precision, recall, **F1**, confusion matrix
- **PR curve / ROC-AUC** (threshold 운용 분석용, 평가서 §8)
- 위 §4의 4개 평가셋 각각에 대해 보고

### (B) 프리필터 운용 지표 (평가서 §8 — F1이 아니라 recall 중심)
프리필터는 "PII를 놓치면 안 됨" → **high-recall operating point** 기준.

| 항목 | 기준 |
|---|---|
| Recall | 높게 유지 (예: ≥0.95) |
| False Negative Rate | 핵심 — 놓친 PII |
| Precision | 다소 낮아도 허용 |
| **STT 호출 절감률** | 프리필터의 실제 가치 |
| End-to-end latency | **멜 변환 포함** |

threshold별 운용표를 산출:

| Threshold | Recall | Precision | STT 호출 비율 | STT 절감률 |
|---:|---:|---:|---:|---:|
| 0.2 | … | … | … | … |
| 0.5 | … | … | … | … |
| 0.8 | … | … | … | … |

> "Recall 95%를 유지하며 STT 호출을 몇 % 줄이는가"가 보조 가설의 판정 기준.

### (C) 효율성 지표
- 파라미터 수, 모델 파일 크기(MB)
- 1샘플 latency (CPU/GPU), throughput, (가능 시) FLOPs/MACs
- 멜 변환 포함 end-to-end 지연

## 10. 학습 설정 & 데이터 증강

### 학습 기본 설정 (기본값, 고정 commitment 아님)
| 항목 | 값 |
|---|---|
| optimizer | AdamW |
| loss | CrossEntropyLoss |
| scheduler | CosineAnnealing 또는 ReduceLROnPlateau |
| lr / weight_decay | 1e-4 / 1e-4 |
| batch size | 32 (GPU 메모리 맞게 조정) |
| max epochs | 50 |
| early stopping | val F1 또는 val loss, patience=8 |
| seeds | [42, 43, 44] |

### 데이터 증강 (train 전용)
- noise 추가, speed perturbation, volume jitter, SpecAugment(time/freq masking).
- **train에만 적용, val/test는 깨끗하게 유지.**
- 증강 강도는 **보수적**으로 — 전화번호 낭독 패턴 자체를 훼손하지 않게.

## 11. 결론 해석 기준 (평가서 §9 — 정직하게)

### 결과 패턴 → 해석 매트릭스 (결과 보기 전에 미리 고정 = post-hoc 해석 방지)
| 결과 패턴 | 해석 |
|---|---|
| Original 높음, Controlled-main 낮음 | naive artifact 의존 가능성 큼 |
| Controlled 높음, Length-matched 급락 | 길이 단서 의존 |
| Controlled 높음, Hard-negative FPR 높음 | "숫자 발화 일반"을 전화번호로 오탐 |
| duration-only/acoustic-LR과 비슷 | CNN의 추가 가치 제한적 |
| seed별 편차 큼 | 데이터 규모·split 안정성 부족 |
| ConvNeXt만 높고 경량 모델은 낮음 | 경량 프리필터로는 부적합 |
| Controlled/Length-matched/Hard-neg 모두 양호 + baseline 능가 | 전화번호 형식 패턴을 어느 정도 학습했을 가능성 |

### 결론 문구 기준
보고 시 다음 톤을 넘지 않는다:
```
본 실험은 음성 안에 전화번호 형식의 숫자열이 존재하는지를
STT 없이 멜 스펙트로그램 기반 CNN으로 선별할 수 있는지 검증한다.

문맥상 실제 개인정보인지, 예시인지, 부정 표현인지는 본 모델의 판단 대상이 아니며,
해당 판단은 후단 STT/NLP 단계의 역할로 둔다.

PII 의미 이해나 정설 경로 대체 가능성은 본 실험만으로 주장하지 않는다.
duration-only baseline, length-matched test, hard-negative test로
artifact 의존 여부를 점검하고, high-recall threshold에서
STT 호출 절감 가능성을 보조적으로 평가한다.
```
- 멜 스펙트로그램도 완전 비민감은 아님(화자 음향·발화 패턴 잔존)을 명시.

## 12. 디렉터리 구조 (예정)

```
src/
├─ generate/
│  ├─ generate_o_1.py        # (기존) naive positive: 숫자 그대로
│  ├─ generate_o_2.py        # (기존) 통제 positive: 한국어 발음 → data/o/00002 생성
│  ├─ generate_x_1.py        # (기존) easy negative
│  └─ generate_x_hard.py     # (신규) hard negative: 숫자-비PII
└─ cnn/
   ├─ plan.md                # (현재 문서)
   ├─ rebuttal.md            # 외부 평가서 1차
   ├─ rebuttal2_sec.md       # 외부 평가서 2차(후보 최종본)
   ├─ config.py              # 경로/오디오/하이퍼파라미터/seed
   ├─ dataset.py             # 메타데이터 병합 + wav→log-mel + 5s random-crop(train) / sliding-window(eval) + 평가셋 분리
   ├─ splits.py              # template-disjoint + length-stratified + stratified split (speaker-disjoint 미적용)
   ├─ baselines.py           # majority / duration-only / acoustic-feature LR
   ├─ models/
   │  ├─ simple_cnn.py
   │  ├─ mobilenet.py        # MobileNetV3-Small 래퍼
   │  ├─ efficientnet.py     # EfficientNet-B0 래퍼
   │  └─ convnext.py         # ConvNeXt-Tiny 래퍼 (성능 상한)
   ├─ train.py               # 모델 선택, seed 반복, 학습/검증
   ├─ evaluate.py            # 4개 평가셋 × 지표(A/B/C), PR/ROC, threshold 운용표
   └─ infer.py               # 단일 wav 추론
```

## 13. 실행 환경 (준비 완료)
- conda 환경: `audio-pii` (Python 3.11)
- 하드웨어: NVIDIA H100 NVL × 4, CUDA 13.0
- 패키지: torch 2.12.0+cu130, torchvision 0.27.0+cu130, librosa 0.11.0, timm 1.0.27, pandas, scikit-learn, soundfile, matplotlib, tqdm
- 백본: convnext_tiny(28.6M) 가중치 사전 다운로드 완료. MobileNetV3-Small / EfficientNet-B0는 torchvision·timm로 로드(필요 시 사전 다운로드).
- 실행 예: `conda run -n audio-pii python src/cnn/train.py ...`

## 14. 작업 로드맵 (우선순위 = 평가서 §11 "반드시" 우선)

1. **[데이터 통제]** `generate_o_2.py`로 `data/o/00002`(한국어 발음 positive) 생성 + `generate_x_hard.py`로 hard negative 생성. easy negative는 기존 `data/x` 재사용.
2. **[전처리]** `dataset.py` — naive/controlled 병합, log-mel, **5s random-crop(train) / sliding-window+max(eval)**, 5개 평가셋(Original/Controlled-main/Length-matched/Hard-neg-only/Stress-5:5) 구성.
3. **[split]** `splits.py` — template-disjoint + length-stratified + stratified, seeds=[42,43,44]. (speaker-disjoint 미적용, 한계 명시)
4. **[baseline]** `baselines.py` — majority / duration-only / acoustic-feature LR. CNN 해석의 기준선.
5. **[스모크 테스트]** subset 1 epoch 파이프라인 검증.
6. **[모델 학습]** Simple CNN → MobileNetV3-S → EfficientNet-B0 → ConvNeXt-Tiny, seeds=[42,43,44].
7. **[평가]** `evaluate.py` — 5 평가셋 × (A 성능 / B 프리필터 / C 효율성), PR·ROC, threshold·STT절감표, §11 해석 매트릭스 적용.
8. **[비교·결론]** baseline 대비·평가셋 간 격차 분석 → §11 톤으로 결론. 추론 데모(`infer.py`).

## 15. v1 이후로 defer (feasibility 확인 후 추가)
후보 최종본(`rebuttal2_sec.md`)에서 좋지만 v1에는 과한 항목 — CNN이 baseline을 유의미하게 넘으면 추가:
- **윈도 길이 ablation**(3/5/8초): v1은 5초 단일.
- **정식 벤치마크 프로토콜**(warmup=50/measure=500, p50/p95, CPU+GPU+FLOPs, `benchmark.py`): v1은 params/size/대략 latency만.
- **Acoustic RF + 확장 feature bank**: v1은 majority/duration-only/acoustic-LR.
- **`phone_start/end` 타임스탬프 + positive-aware crop / MIL**: §7 사유로 보류.
- **controlled easy negative 재생성(`generate_x_2.py`)**: v1은 기존 `data/x` 재사용(화자 2~3명이라 음색 매칭 이득 작음).

## 16. 범위 밖 (실험 자체에서 다루지 않음)
- STT+NLP 정설 경로 baseline과의 직접 성능 비교(= replacement 주장 가능 조건).
- 문맥상 실제 개인정보 여부 판단(후단 STT/NLP 역할).
- 전화번호 외 PII(주민번호·계좌번호 등) 확장.
- 실데이터(비합성) 검증.
- 실시간 스트리밍 추론 파이프라인 / 마스킹·후처리.
