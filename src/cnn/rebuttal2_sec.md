아래는 지금까지 정리한 기준을 반영해서 새로 쓴 `plan.md`입니다. 그대로 붙여 넣어도 되는 형태로 작성했습니다.

# 오디오 기반 전화번호 탐지 CNN 실험 설계

## 0. 한 줄 요약

본 실험은 STT 없이 `오디오 → 멜 스펙트로그램 → CNN 계열 분류기` 경로만으로 **음성 안에 전화번호 형식의 숫자열이 포함되어 있는지** 탐지할 수 있는지 검증하는 feasibility test이다.

본 실험은 다음을 주장하지 않는다.

* 일반적인 개인정보 전체 탐지
* 문맥상 실제 개인정보인지 여부 판단
* STT 기반 파이프라인과의 성능 비교 또는 대체 가능성
* 전화번호 외 주민번호, 계좌번호, 주소 등 다른 PII 유형 탐지

본 실험의 목표는 오직 다음이다.

> 멜 스펙트로그램 기반 CNN이 전화번호 형식의 숫자열 발화를 우연 수준 및 단순 baseline 이상으로 구분할 수 있는가?

---

## 1. 실험 목적

### 1.1 검증 대상

기존의 일반적인 음성 PII 탐지 접근은 보통 다음 경로를 따른다.

```text
오디오 → STT → 텍스트 → 정규식/NLP 기반 탐지
```

하지만 본 실험에서는 STT를 사용하지 않는다.

본 실험의 경로는 다음과 같다.

```text
오디오 → log-mel spectrogram → CNN 계열 이미지 분류기 → 전화번호 형식 포함 여부
```

즉, 음성 자체를 텍스트로 변환하지 않고, 음향 패턴만으로 전화번호 형식 숫자열이 포함되어 있는지를 이진 분류한다.

---

## 2. 문제 정의

### 2.1 분류 과제

본 실험은 binary classification 문제로 정의한다.

| 클래스      | 라벨 | 의미                         |
| -------- | -: | -------------------------- |
| Positive |  1 | 음성 안에 전화번호 형식 숫자열이 포함됨     |
| Negative |  0 | 음성 안에 전화번호 형식 숫자열이 포함되지 않음 |

여기서 중요한 점은 **문맥상 실제 개인정보인지 여부는 판단하지 않는다**는 것이다.

예를 들어 다음 문장은 전화번호 형식이 들리므로 positive로 본다.

```text
공일공 일이삼사 오육칠팔 형식은 예시입니다.
```

이 문장은 문맥상 실제 개인정보가 아닐 수 있지만, 본 실험의 목표는 문맥 판단이 아니라 **전화번호 형식 숫자열의 음향적 존재 여부 탐지**이므로 라벨은 1이다.

반대로 다음 문장들은 숫자를 포함하지만 전화번호 형식이 아니므로 negative로 본다.

```text
오늘은 이공이육년 육월 십삼일입니다.
주문번호는 이공이육공육일삼공공일입니다.
삼번 창구로 가세요.
우편번호는 공육이삼사입니다.
가격은 만 오천 원입니다.
```

---

## 3. 핵심 위험 요소: artifact detection

이 실험에서 가장 큰 위험은 모델이 전화번호 패턴을 학습하는 것이 아니라, 데이터셋의 표면적 차이를 학습하는 것이다.

예를 들어 모델은 다음과 같은 단서만으로도 높은 성능을 낼 수 있다.

| 위험 요소           | 설명                                            |
| --------------- | --------------------------------------------- |
| 음성 길이 차이        | positive가 negative보다 길면 길이만으로 분류 가능           |
| 숫자 존재 여부        | negative에 숫자가 거의 없으면 “숫자 발화 여부” 문제로 전락        |
| 발음 스타일 차이       | positive만 숫자를 또박또박 읽으면 리듬 차이로 분류 가능           |
| 템플릿 누수          | 같은 문장 템플릿이 train/test에 섞이면 암기 가능              |
| 화자 누수           | 같은 TTS 화자가 train/test에 섞이면 화자별 artifact 학습 가능 |
| padding leakage | 고정 길이 padding에서 무음 구간 길이만 보고 분류 가능            |

따라서 본 실험은 단순히 CNN 성능만 보는 것이 아니라, 위 artifact를 얼마나 제거하거나 측정했는지를 함께 평가한다.

---

## 4. 데이터셋 설계

### 4.1 전체 구조

데이터셋은 크게 naive 데이터와 controlled 데이터로 나눈다.

| 데이터셋               | 용도         | 설명                                  |
| ------------------ | ---------- | ----------------------------------- |
| Naive dataset      | 통제 전 성능 확인 | 기존 `data/o/00001`, 기존 `data/x` 사용   |
| Controlled dataset | 주 실험       | 발음 방식, 길이, 숫자 포함 여부, 템플릿을 통제해 새로 생성 |

Naive dataset은 최종 주장을 위한 데이터가 아니라, 통제 전후 성능 차이를 확인하기 위한 비교군으로만 사용한다.

---

### 4.2 Positive 데이터

Positive는 전화번호 형식의 숫자열이 포함된 음성이다.

기존 naive positive는 다음과 같다.

```text
data/o/00001/
```

이 데이터는 숫자를 그대로 읽는 방식이므로, controlled 실험의 주 데이터로 사용하지 않는다.

Controlled positive는 다음 스크립트로 새로 생성한다.

```text
src/generate/generate_o_2.py
```

생성 위치는 다음과 같다.

```text
data/o/00002/
```

Controlled positive의 조건은 다음과 같다.

* 전화번호를 한국어 발음으로 읽음

  * 예: `010-1234-5678`
  * 발화: `공일공 일이삼사 오육칠팔`
* 다양한 문장 템플릿 사용
* 다양한 전화번호 위치 사용

  * 문장 앞
  * 문장 중간
  * 문장 끝
* positive/negative 간 길이 분포가 과도하게 벌어지지 않도록 생성 후 duration 기록
* 가능하면 여러 TTS speaker 사용
* 각 샘플에 전화번호 발화 구간 metadata 저장

필수 metadata 예시는 다음과 같다.

```csv
filepath,label,source_type,duration,speaker_id,template_id,phone_start_sec,phone_end_sec
```

| 컬럼                | 설명                    |
| ----------------- | --------------------- |
| `filepath`        | wav 파일 경로             |
| `label`           | 1                     |
| `source_type`     | `positive_controlled` |
| `duration`        | 전체 음성 길이              |
| `speaker_id`      | TTS 화자 ID             |
| `template_id`     | 문장 템플릿 ID             |
| `phone_start_sec` | 전화번호 발화 시작 시각         |
| `phone_end_sec`   | 전화번호 발화 종료 시각         |

`phone_start_sec`, `phone_end_sec`는 random crop이나 sliding window 학습 시 positive label noise를 줄이기 위해 사용한다.

---

### 4.3 Negative 데이터

Negative는 전화번호 형식 숫자열이 포함되지 않은 음성이다.

Negative는 easy negative와 hard negative로 나눈다.

| 유형            | 라벨 | 설명                    |
| ------------- | -: | --------------------- |
| Easy negative |  0 | 일반 문장, 숫자가 거의 없거나 없음  |
| Hard negative |  0 | 숫자는 포함하지만 전화번호 형식은 아님 |

Controlled negative는 기존 `data/x`를 그대로 쓰지 않고, 가능하면 positive와 동일한 TTS 조건으로 새로 생성한다.

```text
src/generate/generate_x_2.py
src/generate/generate_x_hard.py
```

생성 위치는 다음과 같다.

```text
data/x/00002_easy/
data/x/00002_hard/
```

---

### 4.4 Hard negative 설계

Hard negative의 목적은 모델이 단순히 “숫자가 들리는 음성”을 positive로 오탐하지 않는지 확인하는 것이다.

Hard negative에는 다음 유형을 포함한다.

| 유형       | 예시                    |
| -------- | --------------------- |
| 날짜       | 오늘은 이공이육년 육월 십삼일입니다.  |
| 주문번호     | 주문번호는 이공이육공육일삼공공일입니다. |
| 송장번호     | 송장번호는 삼팔구일 이칠오사입니다.   |
| 방 번호     | 삼백일호로 오세요.            |
| 창구 번호    | 삼번 창구에서 접수하세요.        |
| 버스/노선 번호 | 칠백오십일번 버스를 타세요.       |
| 가격       | 가격은 만 오천 원입니다.        |
| 수량       | 총 스물세 개가 필요합니다.       |
| 우편번호     | 우편번호는 공육이삼사입니다.       |
| 단순 수 세기  | 하나 둘 셋 넷 다섯.          |

주의할 점은 다음과 같다.

* 전화번호 형식의 숫자열이 포함된 문장은 negative로 넣지 않는다.
* “예시입니다”, “제 번호는 아닙니다”처럼 문맥상 부정이더라도 전화번호 형식이 들리면 label은 1로 둔다.
* 본 실험은 문맥상 실제 개인정보 여부가 아니라 전화번호 형식의 음향적 존재 여부를 탐지한다.

---

### 4.5 Easy:Hard 비율

Controlled dataset의 negative 내부 비율은 기본적으로 다음을 사용한다.

```text
easy negative : hard negative = 7 : 3
```

예를 들어 전체 샘플 수가 2000개이고 positive:negative를 1:1로 맞춘다면 다음과 같이 구성한다.

| 유형            |   개수 |
| ------------- | ---: |
| Positive      | 1000 |
| Easy negative |  700 |
| Hard negative |  300 |
| Total         | 2000 |

`7:3`을 기본값으로 두는 이유는 다음과 같다.

* hard negative를 충분히 포함해 숫자 artifact를 통제할 수 있음
* hard negative가 과도하게 많아져 recall이 불필요하게 하락하는 것을 방지
* 메인 실험 분포로는 5:5보다 덜 인위적임

단, hard negative에 대한 견고성은 별도 평가셋에서 강하게 측정한다.

---

## 5. 평가셋 구성

단일 test set만 사용하지 않는다. 다음 평가셋을 분리해 보고한다.

| 평가셋                | 구성                                              | 목적                           |
| ------------------ | ----------------------------------------------- | ---------------------------- |
| Original           | `00001` positive vs 기존 `data/x`                 | 통제 전 naive 성능 확인             |
| Controlled-main    | controlled positive vs easy:hard = 7:3 negative | 주 평가셋                        |
| Length-matched     | positive/negative duration 분포를 맞춘 subset        | 길이 artifact 제거 후 성능 확인       |
| Hard-negative-only | hard negative만 negative로 사용                     | 숫자 포함 비전화번호 오탐률 측정           |
| Stress-5:5         | easy:hard = 5:5                                 | hard negative 비중 증가 시 견고성 확인 |

최종 주장은 `Controlled-main`, `Length-matched`, `Hard-negative-only` 결과를 중심으로 한다.

`Original`에서 성능이 높더라도 controlled 조건에서 급락하면, naive 성능은 artifact 의존 가능성이 크다고 해석한다.

---

## 6. 전처리

### 6.1 오디오 로드

* Sampling rate: 16kHz
* Channel: mono
* Format: wav
* Loader: `librosa` 또는 `torchaudio`

기본 로드 방식:

```python
y, sr = librosa.load(filepath, sr=16000, mono=True)
```

---

### 6.2 Log-mel spectrogram

기본 파라미터는 다음과 같다.

| 파라미터          |       값 |
| ------------- | ------: |
| `sample_rate` |   16000 |
| `n_mels`      |      64 |
| `n_fft`       |    1024 |
| `hop_length`  |     512 |
| `power`       |     2.0 |
| scale         | log-mel |

출력 텐서 형태:

```text
[C, n_mels, T]
```

기본적으로 1채널 spectrogram을 사용한다.

ImageNet pretrained 모델을 사용할 경우 다음 중 하나를 선택한다.

1. 1채널 입력을 3채널로 복제
2. 첫 convolution layer를 1채널 입력으로 수정

초기 구현에서는 단순성을 위해 3채널 복제를 기본값으로 둔다.

---

### 6.3 정규화

기본 정규화는 per-sample normalization을 사용한다.

```text
mel = (mel - mean) / (std + eps)
```

필요 시 train set 전체 mean/std 기반 정규화도 비교할 수 있다.

---

### 6.4 Padding leakage 방지

고정 길이 padding은 음성 길이 정보를 모델에 노출할 수 있으므로, 단순 zero padding을 피한다.

기본 전략은 다음과 같다.

#### 학습 시

* fixed-length window를 사용한다.
* positive 샘플은 전화번호 발화 구간이 window 안에 포함되도록 crop한다.
* negative 샘플은 random crop한다.
* window 길이는 3초와 5초를 비교한다.

#### 평가 시

* 전체 clip을 sliding window로 나눈다.
* 각 window에 대해 score를 계산한다.
* clip-level score는 window score의 max를 사용한다.

```text
clip_score = max(window_scores)
```

이 방식은 “음성 어딘가에 전화번호 형식이 포함되어 있는가”라는 문제 정의와 잘 맞는다.

---

### 6.5 Window 길이 ablation

전화번호 전체 패턴이 3초 안에 들어가지 않을 수 있으므로 window 길이를 비교한다.

| window length | 목적                  |
| ------------: | ------------------- |
|            3초 | 짧은 구간 기반 탐지 가능성 확인  |
|            5초 | 전화번호 패턴을 더 안정적으로 포함 |
|            8초 | clip-level에 가까운 비교  |

기본 실험은 5초 window를 주 설정으로 사용하고, 3초는 ablation으로 둔다.

---

## 7. Split 전략

기본 split 비율은 다음과 같다.

```text
train : val : test = 70 : 15 : 15
```

단순 stratified random split만 사용하지 않는다.

가능한 범위에서 다음 제약을 적용한다.

| split 조건                | 목적                                    |
| ----------------------- | ------------------------------------- |
| Stratified split        | label 비율 유지                           |
| Template-disjoint split | 같은 문장 템플릿이 train/test에 섞이지 않도록 함      |
| Speaker-disjoint split  | 같은 TTS speaker가 train/test에 섞이지 않도록 함 |
| Length-stratified split | positive/negative 길이 분포 차이를 줄임        |

단, speaker 수가 부족하면 speaker-disjoint split은 완전히 적용하지 못할 수 있다.
이 경우 문서에 다음처럼 명시한다.

```text
TTS speaker 수 제한으로 speaker-disjoint split은 완전 적용하지 못했으며,
template-disjoint와 length-stratified split을 우선 적용했다.
```

실험은 seed를 바꿔 3회 이상 반복하고 평균±표준편차를 보고한다.

```text
seeds = [42, 43, 44]
```

---

## 8. Baseline

CNN 결과를 해석하기 위해 단순 baseline을 반드시 함께 보고한다.

| Baseline            | 입력                             | 모델                  | 목적             |
| ------------------- | ------------------------------ | ------------------- | -------------- |
| Majority            | 없음                             | 다수 클래스 예측           | 최소 기준          |
| Duration-only       | duration 1개 feature            | Logistic Regression | 길이만으로 분류되는지 확인 |
| Acoustic-feature LR | hand-crafted acoustic features | Logistic Regression | 단순 음향 특징 기준    |
| Acoustic-feature RF | hand-crafted acoustic features | Random Forest       | 비딥러닝 특징 기반 상한  |

Acoustic feature 후보는 다음과 같다.

| Feature                     | 설명                 |
| --------------------------- | ------------------ |
| duration                    | 전체 음성 길이           |
| RMS mean/std                | 에너지 통계             |
| ZCR mean/std                | zero-crossing rate |
| spectral centroid mean/std  | 주파수 중심             |
| spectral bandwidth mean/std | 대역폭                |
| spectral rolloff mean/std   | rolloff            |
| tempo 또는 onset 관련 feature   | 발화 리듬 단서           |
| silence ratio               | 무음 비율              |

CNN 모델은 최소한 duration-only baseline과 acoustic-feature baseline을 유의미하게 넘어야 한다.

---

## 9. 모델 라인업

본 실험에서는 다음 모델을 비교한다.

| 모델                | 역할            | 해석                         |
| ----------------- | ------------- | -------------------------- |
| Simple CNN        | 경량 baseline   | 직접 설계한 작은 CNN              |
| MobileNetV3-Small | 경량 CNN 후보     | 실시간/경량 모델 후보               |
| EfficientNet-B0   | 소형 이미지 분류 비교군 | pretrained CNN 비교          |
| ConvNeXt-Tiny     | 성능 상한 확인      | 프리필터 후보라기보다 upper-bound 모델 |

### 9.1 Simple CNN

구조 예시:

```text
Conv-BN-ReLU-Pool
Conv-BN-ReLU-Pool
Conv-BN-ReLU-Pool
Conv-BN-ReLU-Pool
Global Average Pooling
Linear
```

목표는 과도한 성능이 아니라, 작은 모델이 baseline 이상으로 안정적으로 작동하는지 확인하는 것이다.

### 9.2 MobileNetV3-Small

* `torchvision.models.mobilenet_v3_small`
* pretrained 사용 가능
* 1채널 mel을 3채널로 복제해 입력

### 9.3 EfficientNet-B0

* `torchvision` 또는 `timm` 사용
* pretrained 사용 가능
* MobileNet보다 약간 큰 소형 비교군

### 9.4 ConvNeXt-Tiny

* `torchvision.models.convnext_tiny`
* ImageNet pretrained 사용 가능
* 약 28.6M parameters
* 경량 모델이라기보다는 mel-spectrogram 입력에서 CNN 계열이 도달할 수 있는 성능 상한 확인용

---

## 10. 학습 설정

기본 학습 설정은 다음과 같다.

| 항목             | 값                                    |
| -------------- | ------------------------------------ |
| optimizer      | AdamW                                |
| loss           | CrossEntropyLoss                     |
| scheduler      | CosineAnnealing 또는 ReduceLROnPlateau |
| batch size     | GPU 메모리에 맞게 설정                       |
| epochs         | 최대 50                                |
| early stopping | validation F1 또는 validation loss 기준  |
| seeds          | 3개 이상                                |
| augmentation   | train에만 적용                           |

초기값 예시:

```text
lr = 1e-4
weight_decay = 1e-4
batch_size = 32
max_epochs = 50
early_stop_patience = 8
```

---

## 11. 데이터 증강

증강은 train set에만 적용한다.
validation/test에는 적용하지 않는다.

사용 가능한 증강은 다음과 같다.

| 증강                 | 설명           |
| ------------------ | ------------ |
| noise injection    | 저강도 배경 잡음 추가 |
| volume jitter      | 볼륨 변화        |
| speed perturbation | 말속도 변화       |
| time masking       | SpecAugment  |
| frequency masking  | SpecAugment  |

증강은 artifact를 줄이고 일반화를 높이기 위한 목적이다.
단, 증강이 전화번호 발화 패턴 자체를 훼손하지 않도록 강도는 보수적으로 설정한다.

---

## 12. 평가 지표

### 12.1 기본 분류 성능

각 평가셋에 대해 다음 지표를 보고한다.

| 지표               | 설명                           |
| ---------------- | ---------------------------- |
| Accuracy         | 전체 정확도                       |
| Precision        | positive 예측 중 실제 positive 비율 |
| Recall           | 실제 positive 중 탐지한 비율         |
| F1               | precision과 recall의 조화 평균     |
| Confusion matrix | TP, FP, TN, FN               |
| ROC-AUC          | threshold 독립적 분류 성능          |
| PR-AUC           | positive 탐지 관점 성능            |

특히 본 실험에서는 F1과 Recall을 중요하게 본다.

---

### 12.2 Hard negative 지표

Hard negative 성능은 별도로 보고한다.

핵심 지표는 다음이다.

```text
Hard-negative FPR = hard negative 중 positive로 잘못 예측한 비율
```

즉,

```text
Hard-negative FPR = FP_hard / N_hard_negative
```

이 값이 높으면 모델이 “숫자 포함 음성”을 전화번호로 오탐하는 경향이 강하다는 뜻이다.

---

### 12.3 Length-matched 지표

Length-matched 평가셋에서의 성능은 길이 artifact 제거 후에도 모델이 작동하는지 확인하기 위한 핵심 지표다.

다음과 같은 해석 기준을 둔다.

| 결과 패턴                                         | 해석                         |
| --------------------------------------------- | -------------------------- |
| Original 높음, Controlled 낮음                    | naive artifact 의존 가능성 큼    |
| Controlled 높음, Length-matched 급락              | 길이 단서 의존 가능성               |
| Controlled 높음, Hard-negative FPR 높음           | 숫자 일반 발화를 전화번호로 오탐         |
| Controlled/Length-matched/Hard-negative 모두 양호 | 전화번호 형식 패턴을 어느 정도 학습했을 가능성 |

---

### 12.4 Threshold 분석

최종 성능은 단일 threshold 0.5만 보지 않는다.

threshold별 precision/recall/F1을 보고한다.

예시 표:

| Threshold | Precision | Recall |  F1 | Hard-neg FPR |
| --------: | --------: | -----: | --: | -----------: |
|       0.2 |       ... |    ... | ... |          ... |
|       0.3 |       ... |    ... | ... |          ... |
|       0.5 |       ... |    ... | ... |          ... |
|       0.7 |       ... |    ... | ... |          ... |
|       0.8 |       ... |    ... | ... |          ... |

운영 관점에서는 recall을 높게 유지하는 threshold를 별도로 확인한다.

---

### 12.5 효율성 지표

모델별 효율성도 함께 보고한다.

| 지표                     | 설명                   |
| ---------------------- | -------------------- |
| parameter count        | 모델 파라미터 수            |
| model size             | 저장된 checkpoint 크기    |
| latency CPU batch=1    | CPU 단건 추론 지연         |
| latency GPU batch=1    | GPU 단건 추론 지연         |
| throughput GPU batch=N | 배치 처리량               |
| mel preprocessing time | 오디오→mel 변환 시간        |
| end-to-end latency     | mel 변환 + 모델 추론 전체 시간 |
| FLOPs/MACs             | 가능하면 측정              |

Latency 측정 시 다음을 고정한다.

```text
warmup = 50
measure = 500
report = mean, p50, p95
```

H100 GPU 결과만으로 경량성을 주장하지 않는다.
가능하면 CPU batch=1 latency도 함께 보고한다.

---

## 13. 결과 보고 형식

### 13.1 메인 성능표

예시:

| Model            | Controlled F1 | Controlled Recall | Length-matched F1 | Hard-neg FPR | ROC-AUC | PR-AUC |
| ---------------- | ------------: | ----------------: | ----------------: | -----------: | ------: | -----: |
| Majority         |           ... |               ... |               ... |          ... |     ... |    ... |
| Duration-only LR |           ... |               ... |               ... |          ... |     ... |    ... |
| Acoustic LR      |           ... |               ... |               ... |          ... |     ... |    ... |
| Acoustic RF      |           ... |               ... |               ... |          ... |     ... |    ... |
| Simple CNN       |           ... |               ... |               ... |          ... |     ... |    ... |
| MobileNetV3-S    |           ... |               ... |               ... |          ... |     ... |    ... |
| EfficientNet-B0  |           ... |               ... |               ... |          ... |     ... |    ... |
| ConvNeXt-Tiny    |           ... |               ... |               ... |          ... |     ... |    ... |

각 값은 seed 3회 이상의 평균±표준편차로 보고한다.

---

### 13.2 효율성 표

예시:

| Model           | Params | Size MB | CPU p95 ms | GPU p95 ms | E2E p95 ms | Throughput |
| --------------- | -----: | ------: | ---------: | ---------: | ---------: | ---------: |
| Simple CNN      |    ... |     ... |        ... |        ... |        ... |        ... |
| MobileNetV3-S   |    ... |     ... |        ... |        ... |        ... |        ... |
| EfficientNet-B0 |    ... |     ... |        ... |        ... |        ... |        ... |
| ConvNeXt-Tiny   |    ... |     ... |        ... |        ... |        ... |        ... |

---

### 13.3 평가셋별 비교표

예시:

| Model           | Original F1 | Controlled F1 | Length-matched F1 | Stress-5:5 F1 | Hard-neg FPR |
| --------------- | ----------: | ------------: | ----------------: | ------------: | -----------: |
| Simple CNN      |         ... |           ... |               ... |           ... |          ... |
| MobileNetV3-S   |         ... |           ... |               ... |           ... |          ... |
| EfficientNet-B0 |         ... |           ... |               ... |           ... |          ... |
| ConvNeXt-Tiny   |         ... |           ... |               ... |           ... |          ... |

---

## 14. 결론 해석 기준

본 실험의 결론은 다음 기준으로 제한한다.

### 14.1 긍정적 결론 가능 조건

다음 조건을 만족하면 “mel-spectrogram 기반 CNN이 전화번호 형식 숫자열 탐지에 대해 feasibility를 보였다”고 말할 수 있다.

* Controlled-main에서 majority baseline보다 명확히 높음
* Duration-only baseline보다 높음
* Acoustic-feature baseline보다 높음
* Length-matched 평가셋에서도 성능이 크게 무너지지 않음
* Hard-negative-only에서 FPR이 과도하게 높지 않음
* seed 반복에서 성능이 안정적임

### 14.2 부정적 결론 기준

다음 중 하나가 나타나면 접근의 한계를 명시한다.

| 결과                             | 해석                     |
| ------------------------------ | ---------------------- |
| Duration-only와 성능이 비슷함         | 길이 단서 의존 가능성           |
| Acoustic-feature baseline과 비슷함 | CNN의 추가 가치 제한적         |
| Length-matched에서 급락            | 길이 artifact 의존         |
| Hard-negative FPR이 높음          | 숫자 발화 일반에 취약           |
| seed별 편차가 큼                    | 데이터 규모 또는 split 안정성 부족 |
| ConvNeXt만 높고 경량 모델은 낮음         | 경량 프리필터로는 부적합 가능성      |

---

## 15. 디렉터리 구조

예정 디렉터리 구조는 다음과 같다.

```text
src/
├─ generate/
│  ├─ generate_o_1.py          # 기존 naive positive: 숫자 그대로
│  ├─ generate_o_2.py          # controlled positive: 한국어 전화번호 발음
│  ├─ generate_x_1.py          # 기존 easy negative
│  ├─ generate_x_2.py          # controlled easy negative
│  └─ generate_x_hard.py       # controlled hard negative: 숫자 포함 비전화번호
│
└─ cnn/
   ├─ plan.md                  # 현재 문서
   ├─ rebuttal.md              # 외부 검토 및 반박/수정 기록
   ├─ config.py                # 경로, 오디오 파라미터, 하이퍼파라미터, seed
   ├─ dataset.py               # metadata 로딩, wav→mel, windowing, Dataset/DataLoader
   ├─ splits.py                # template/speaker/length-aware split
   ├─ baselines.py             # majority, duration-only, acoustic feature baseline
   ├─ features.py              # acoustic feature extraction
   ├─ metrics.py               # F1, AUC, hard-neg FPR, threshold table
   ├─ benchmark.py             # latency, throughput, model size 측정
   ├─ models/
   │  ├─ simple_cnn.py
   │  ├─ mobilenet.py
   │  ├─ efficientnet.py
   │  └─ convnext.py
   ├─ train.py                 # 학습 실행
   ├─ evaluate.py              # 평가셋별 성능 산출
   ├─ infer.py                 # 단일 wav 추론
   └─ runs/                    # checkpoint, logs, metrics
```

---

## 16. 실행 환경

현재 준비된 실행 환경은 다음과 같다.

| 항목          | 값                                                 |
| ----------- | ------------------------------------------------- |
| conda 환경    | `audio-pii`                                       |
| Python      | 3.11                                              |
| GPU         | NVIDIA H100 NVL × 4                               |
| CUDA        | 13.0                                              |
| torch       | 2.12.0+cu130                                      |
| torchvision | 0.27.0+cu130                                      |
| librosa     | 0.11.0                                            |
| timm        | 1.0.27                                            |
| 기타          | pandas, scikit-learn, soundfile, matplotlib, tqdm |

ConvNeXt-Tiny pretrained weight는 사전 다운로드되어 있다.

```text
~/.cache/torch/hub/checkpoints/
```

실행 예시는 다음과 같다.

```bash
conda run -n audio-pii python src/cnn/train.py --model simple_cnn --config src/cnn/config.py
conda run -n audio-pii python src/cnn/evaluate.py --run-dir src/cnn/runs/simple_cnn_seed42
conda run -n audio-pii python src/cnn/infer.py --wav path/to/sample.wav --checkpoint path/to/checkpoint.pt
```

---

## 17. 작업 로드맵

### Step 1. Controlled 데이터 생성

* `generate_o_2.py`로 controlled positive 생성
* `generate_x_2.py`로 controlled easy negative 생성
* `generate_x_hard.py`로 controlled hard negative 생성
* 모든 wav에 대해 metadata 저장
* duration, speaker_id, template_id, phone_start/end 기록

산출물:

```text
data/o/00002/
data/x/00002_easy/
data/x/00002_hard/
metadata_controlled.csv
```

---

### Step 2. 데이터 검증

* positive/negative 개수 확인
* easy/hard 비율 확인
* duration 분포 시각화
* speaker/template 분포 확인
* 전화번호 형식이 negative에 잘못 들어갔는지 검사

산출물:

```text
reports/data_distribution.png
reports/duration_distribution.png
reports/metadata_summary.md
```

---

### Step 3. 전처리 및 Dataset 구현

* wav 로드
* log-mel spectrogram 변환
* positive-aware crop
* sliding window evaluation
* DataLoader 구현

산출물:

```text
src/cnn/dataset.py
```

---

### Step 4. Split 구현

* train/val/test = 70/15/15
* stratified split
* template-disjoint 적용
* speaker-disjoint 가능 시 적용
* length-stratified 적용
* seed 3개 이상 지원

산출물:

```text
src/cnn/splits.py
splits/seed42/
splits/seed43/
splits/seed44/
```

---

### Step 5. Baseline 구현

* majority baseline
* duration-only logistic regression
* acoustic-feature logistic regression
* acoustic-feature random forest

산출물:

```text
src/cnn/features.py
src/cnn/baselines.py
```

---

### Step 6. 모델 구현

* Simple CNN
* MobileNetV3-Small
* EfficientNet-B0
* ConvNeXt-Tiny

산출물:

```text
src/cnn/models/simple_cnn.py
src/cnn/models/mobilenet.py
src/cnn/models/efficientnet.py
src/cnn/models/convnext.py
```

---

### Step 7. 스모크 테스트

각 클래스 소량 샘플로 1 epoch만 실행한다.

목적:

* DataLoader 정상 동작 확인
* mel shape 확인
* 모델 forward 확인
* loss 감소 여부 확인
* evaluation script 정상 동작 확인

예시:

```bash
conda run -n audio-pii python src/cnn/train.py --model simple_cnn --smoke-test
```

---

### Step 8. 본 학습

모델별, seed별로 학습한다.

```text
models = [simple_cnn, mobilenet_v3_small, efficientnet_b0, convnext_tiny]
seeds = [42, 43, 44]
```

각 run은 다음을 저장한다.

```text
checkpoint.pt
config.json
metrics_val.json
metrics_test.json
training_log.csv
```

---

### Step 9. 평가

각 모델에 대해 다음 평가셋을 모두 평가한다.

* Original
* Controlled-main
* Length-matched
* Hard-negative-only
* Stress-5:5

각 평가에서 다음을 산출한다.

* accuracy
* precision
* recall
* F1
* confusion matrix
* ROC-AUC
* PR-AUC
* hard-negative FPR
* threshold table

---

### Step 10. 효율성 측정

모델별로 다음을 측정한다.

* parameter count
* checkpoint size
* CPU latency
* GPU latency
* throughput
* mel preprocessing time
* end-to-end latency

산출물:

```text
reports/efficiency_table.csv
reports/efficiency_summary.md
```

---

### Step 11. 결과 정리

결과 보고서는 다음 구조로 작성한다.

```text
1. 실험 목적
2. 데이터셋 구성
3. confound 통제 방식
4. baseline 결과
5. CNN 모델 결과
6. 평가셋별 성능 비교
7. hard negative 분석
8. 효율성 분석
9. 한계
10. 결론
```

---

## 18. 범위 밖

본 실험에서 다루지 않는 항목은 다음과 같다.

* STT 기반 파이프라인 구현
* STT+Regex/NLP와의 직접 비교
* 정설 경로 대체 가능성 주장
* 문맥상 실제 개인정보 여부 판단
* 전화번호 외 PII 탐지
* 실데이터 검증
* 실시간 스트리밍 시스템 구현
* 개인정보 마스킹 또는 후처리

---

## 19. 최종 결론 문구 가이드

결과가 좋을 경우 사용할 수 있는 결론 문구는 다음 수준으로 제한한다.

```text
본 실험은 STT를 사용하지 않고 log-mel spectrogram 기반 CNN만으로
음성 내 전화번호 형식 숫자열을 탐지할 수 있는지 검증했다.

Controlled dataset, length-matched evaluation, hard-negative-only evaluation에서
baseline 대비 안정적인 성능을 보인다면, 이는 CNN이 단순 길이 차이나 숫자 존재 여부만이 아니라
전화번호 형식 발화 패턴을 어느 정도 활용하고 있음을 시사한다.

다만 본 실험은 전화번호 형식 탐지에 한정되며,
문맥상 실제 개인정보 여부 판단이나 STT 기반 방식과의 대체 가능성은 주장하지 않는다.
```

결과가 좋지 않을 경우 사용할 결론 문구는 다음과 같다.

```text
본 실험에서 CNN 모델은 naive 조건에서는 높은 성능을 보였으나,
controlled dataset, length-matched evaluation, hard-negative-only evaluation에서 성능이 하락했다.

이는 모델이 전화번호 형식 자체보다는 길이, 숫자 포함 여부, 템플릿 차이와 같은 artifact에 의존했을 가능성을 시사한다.

따라서 현재 설정에서는 mel-spectrogram 기반 CNN만으로 전화번호 형식 숫자열을 안정적으로 탐지한다고 보기 어렵다.
```
