# 프로젝트 요약 — STT 없는 음성 PII 탐지 (CNN)

> 목표: STT 없이 음성의 음향 특징(멜 스펙트로그램)만으로 개인정보(PII) 포함 여부를 CNN으로 분류.
> 본 문서는 v1(전화번호 단일·edge-tts 3화자) → v2(4 PII유형·Qwen 14화자) 전체 작업을 정리한다.
> 상세 단계 기록은 `progress/0001~0010`, 수치는 `runs/*.json`, 해석은 `reports/conclusion.md`.

---

## 1. 데이터

| 세트 | TTS | 화자 | 클래스 | 규모 | 경로 |
|---|---|---|---|---|---|
| v1 controlled | edge-tts | 3 | 전화 positive / hard-neg / easy-neg | ~2000 | `data/` |
| **v2** | **Qwen3-TTS** | **14** | **전화·주민·계좌·카드** positive / hard-neg / easy-neg | **1800** | `data_qwen/` |

- 모든 숫자는 **한국어 자릿수 발음**("공일공…") → 클래스 차이가 '읽는 방식'이 아니라 '숫자열 형식'.
- 통제 불변식: 세 클래스 모두 **동일 화자 풀·동일 16kHz mono·노이즈 패딩** → 출처(코덱/화자) shortcut 차단.
- v2 화자 생성(0008~0009): VoiceDesign 은 직접 호출 시 같은 instruct 라도 발화마다 변동(within>between, ratio 0.61) → **VoiceDesign 으로 reference 1개 설계 → Base 로 일관 clone**(within 0.0106≪between, ratio 2.29) 방식 확정. 14화자 `voicebank/`.

## 2. 파이프라인 (코드)

| 파일 | 역할 |
|---|---|
| `config.py` / `features.py` / `dataset.py` | 멜 파라미터, 수작업 음향 feature, manifest+log-mel+하이브리드 윈도(train crop / eval sliding-window+max) |
| `splits.py` | template-disjoint + length-stratified split |
| `models/` (`simple_cnn` 외) | simple_cnn(60K) + mobilenet/efficientnet/convnext 비교군 |
| `engine.py` / `metrics.py` | 학습 루프(AdamW/CE/cosine/early-stop), clip=max(window), F1/recall/ROC/PR + hard-neg FPR |
| `baselines.py` | majority / duration-only / acoustic-LR (CNN 이 능가해야 할 기준선) |
| `evaluate.py` | 5개 평가셋 일괄 평가 |
| `speaker_robustness.py` | v1 LOSO(화자 누수 정량화) |
| `generate/qwen_voicebank.py` | 14화자 reference 생성 + VoiceDesign→Base clone backend |
| `generate/generate_qwen.py` | v2 데이터(4 PII + hard/easy) 생성기 |
| `speaker_pii_experiment.py` | v2 학습/평가: in-dist + 14화자 LOSO + held-out-PII |

## 3. 핵심 결과

### baseline 대비 (v1) — feasibility 성립
| Model | Controlled F1 | length-matched F1 | hard-neg FPR | ROC |
|---|---|---|---|---|
| acoustic-LR (기준선) | 0.76 | 0.77 | 0.26 | 0.88 |
| **simple_cnn** | **0.955** | **0.956** | **0.006** | **0.997** |

- artifact 통제 통과: 길이 의존 아님(duration-only 는 length-matched 에서 ROC 0.70→0.53 붕괴, CNN 은 유지), "숫자 존재"가 아니라 "전화번호 형식" 학습(hard-neg FPR 0.006).
- 프리필터 가치: recall ~0.99 유지하며 STT 호출 ~42% 절감.

### 화자 일반화 (v1→v2) — LOSO
| LOSO | v1 (3화자) | **v2 (14화자)** |
|---|---|---|
| F1@0.5 | 0.616 ± **0.28** | **0.861 ± 0.10** |
| recall@0.5 | 0.712 ± 0.41 | 0.891 ± 0.18 |
| ROC | 0.992 | 0.964 ± 0.03 |

- v1: 화자 3명뿐 → speaker-disjoint 불안정, 고정 임계값이 화자별로 어긋남(ROC 높지만 F1 출렁).
- v2: 화자 14명 → **F1 0.86 으로 안정(분산 1/3)**, 운용점도 안정화(val-튜닝 0.855 ≈ 고정 0.5).

### PII 유형 일반화 (v2) — held-out-PII (전 유형 검증, 0011)
각 PII 유형을 **학습에서 완전히 제외**하고 그 유형으로만 평가:
| 제외 PII | held-out recall | ROC |
|---|---|---|
| 주민번호 | 0.978 | 0.935 |
| 카드 | 0.911 | 0.968 |
| 계좌 | 0.884 | 0.938 |
- 세 유형 모두 **학습 없이 recall 0.88~0.98** → "숫자 암기"가 아니라 **"PII 형식"을 유형 너머로 일반화**(가장 강한 증거).
- viz 4샘플(학습 제외) 실제 예측: 전화 0.973·주민 0.989(PII) / hard-neg 0.107·일반 0.030(non-PII) → 4/4 정답.

## 4. 한계 (정직한 명시)
- **precision/hard-neg FPR 이 v2 의 약점**: 4 PII + 더 어려운 숫자열 hard-neg 로 과제가 어려워져 고정 0.5 에서 과다검출(in-dist FPR 0.542, LOSO 0.251). ROC 는 견고(0.90~0.96)하나 운용 precision·calibration 개선 필요.
- 여전히 **합성 음성**(v2 는 단일 Qwen 계열) → 실통화 gap 잔존. 14화자도 설계 합성 화자(최근접 쌍 marginal).
- 본 결과는 **PII 형식 숫자열의 음향 탐지** feasibility 에 한정. 문맥상 실제 PII 여부·STT 정설 경로 대체는 주장하지 않음.

## 5. 결론 한 줄
> STT 없는 음향 PII 선별의 **feasibility(분리·일반화)는 화자·PII유형 양축에서 검증·강화**됐다(LOSO F1 0.62→0.86, 미학습 PII recall 0.98). 남은 과제는 **운용 precision(calibration)** 과 **실데이터 검증**이다.

## 6. 다음 후보
- precision 개선: hard-neg 비중·난이도↑, calibration(temperature/Platt), threshold-독립 운용.
- 실데이터(실통화) 검증, 화자/TTS 다양성 추가, 문맥(실제 PII 여부) 결합.
