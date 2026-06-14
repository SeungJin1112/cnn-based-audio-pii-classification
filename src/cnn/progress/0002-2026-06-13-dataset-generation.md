---
date: 2026-06-13
phase: 데이터셋 생성
status: done
---

# 0002 — 통제 데이터셋 생성 (controlled positive + hard negative)

## 무엇을 했나
- `src/cnn/` 안에 통제 데이터셋을 새로 생성. 생성 스크립트 2종 작성:
  - `src/cnn/generate/generate_pos.py` — positive(label=1). 전화번호를 한국어 자릿수 발음("공일공 일이삼사 ...")으로 읽고, 전화번호 위치(앞/중간/끝)·템플릿 다양화. 문맥변형 positive("예시입니다/제 번호 아니에요")도 ~20% 포함.
  - `src/cnn/generate/generate_neg_hard.py` — hard negative(label=0). 날짜·주문/송장·방/창구/버스·가격·수량·우편번호·수세기 등 "전화번호 형식이 아닌" 숫자열. positive와 동일하게 숫자는 한국어 자릿수 발음. 전화번호 형식 누수 방지 정규식 가드 + carrier 문맥으로 길이 정렬.
- 파일럿(각 30) → 검증 → 길이 confound 발견 후 carrier 튜닝 → full 생성.
- metadata 스키마: `filename,filepath,text,phone,gender,voice,speaker_id,template_id,source_type,label,duration`.
- easy negative는 기존 `data/x`(1000개) 재사용 — 중복 생성 안 함.

## 왜
- plan.md 설계대로 발음 스타일·길이·숫자 형식 confound를 *소스에서* 통제하기 위함. hard negative는 "숫자 나열 vs 아님"이 아니라 "전화번호 형식 vs 그 외 숫자열"을 풀게 만드는 핵심 요소.

## 결과 / 수치
- **positive 1000개**: 16kHz mono, label=1, duration mean **5.16s** (std 0.66, 3.72–6.70). 화자 3종 균형(337/336/327). 문맥변형 205 / 일반 795.
- **hard negative 300개**: 16kHz mono, label=0, duration mean **5.56s** (std 0.99, 2.59–7.82). 화자 3종 균형. 카테고리 10종 고루 분포. **전화번호 형식 누수 0건**.
- easy negative(data/x) 참고 길이 mean 4.41s.
- 길이 정렬: naive(6.7 vs 4.5, 격차 2.2s) → positive 5.16 vs hard 5.56 (격차 0.4s)로 대폭 개선.
- 위치: `src/cnn/data/positive/`, `src/cnn/data/negative_hard/`.
- TTS 일시 오류 2건 발생했으나 재시도로 전부 복구(최종 1300개 정상). 생성은 GPU 미사용(네트워크/CPU).

## 이슈 / 한계
- easy negative(data/x) mean 4.41s로 positive(5.16)보다 ~0.75s 짧음 → 잔여 길이 confound. **Length-matched 평가셋 + 5s random-crop 윈도**로 학습/평가 단계에서 중화 예정.
- 파일럿→full 사이 스크립트 수정으로 gender suffix가 달라져 생긴 orphan wav 20개는 삭제 완료(데이터셋은 metadata 기반이라 영향 없음).

## 다음
- 0003: `dataset.py` — positive + (easy 700 from data/x) + hard 300 병합, log-mel 변환, 5s random-crop(train)/sliding-window(eval), 5개 평가셋 구성.
