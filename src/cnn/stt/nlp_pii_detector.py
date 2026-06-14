"""Phase 2 — 규칙 기반 NLP PII 탐지기 (Whisper 전사 텍스트 입력).

Whisper 는 한글 자릿수를 아라비아 숫자로 정규화하되 구분자(하이픈/공백) 그룹핑은 발화 멈춤 기준
→ 구분자 무시하고 "연속 숫자 토큰의 자릿수/선두 패턴" + (옵션) 문맥 키워드로 PII 판정.

두 변형:
  detect_format(text)         : 형식(숫자 패턴)만
  detect_format_context(text) : 형식 + 문맥 키워드(전화/주민/계좌/카드) 결합

반환: (label 0/1, matched_pii_type or "")
"""
import re

# 한글 자릿수도 혹시 나오면 대비(역정규화)
_KO = {'공':'0','영':'0','일':'1','이':'2','삼':'3','사':'4','오':'5','육':'6','칠':'7','팔':'8','구':'9'}


def _ko_digits_to_arabic(text):
    # 한글 자릿수 '연속 4자 이상'만 숫자로 변환(일반 음절 '이에요/사실' 오변환 방지).
    # Whisper 는 보통 아라비아로 출력하므로 fallback 성격.
    return re.sub(r"[공영일이삼사오육칠팔구]{4,}",
                  lambda m: "".join(_KO[c] for c in m.group()), text)


def _prep(text):
    """Whisper 가 아라비아 숫자를 주면 그대로, 숫자가 전혀 없으면만 한글→아라비아 fallback."""
    return text if re.search(r"\d", text) else _ko_digits_to_arabic(text)


def _digit_runs(text):
    """구분자(-, 공백, 점, 쉼표)를 무시하고 인접 숫자 그룹을 하나의 숫자열로 묶어 길이 리스트 반환.
       예: '010-1824-0409' -> ['01018240409'(11)]; '9292-5155, 3483-8179' -> ['9292515534838179'(16)]
       (Whisper 가 긴 번호에 쉼표/공백을 끼워 넣는 경우를 한 번호로 본다.)"""
    runs = re.findall(r"\d[\d\-\s\.,]*\d|\d", text)
    return [re.sub(r"\D", "", r) for r in runs]


# 문맥 키워드
KW = {
    "phone":   ["전화", "연락처", "휴대폰", "핸드폰", "번호로", "연락"],
    "rrn":     ["주민등록번호", "주민번호"],
    "account": ["계좌", "입금", "환불"],
    "card":    ["카드", "결제"],
}
# 비PII 신호(문맥으로 제외) — 형식 충돌하는 긴 숫자열 카테고리
NEG_KW = ["주문번호", "송장", "운송장", "우편번호", "좌석", "버스", "일련번호", "제품"]


def _format_match(runs):
    """자릿수/선두 기반 PII 형식 매칭 → pii_type or None."""
    for d in runs:
        n = len(d)
        if n == 11 and d.startswith("01"):
            return "phone"
        if n == 13:
            return "rrn"
        if n == 16:
            return "card"
        if 10 <= n <= 14:          # 계좌(가변) — 단, phone/rrn/card 우선 후 잔여
            return "account"
    return None


def detect_format(text):
    text = _prep(text)
    pt = _format_match(_digit_runs(text))
    return (1, pt) if pt else (0, "")


def detect_format_context(text):
    text = _prep(text)
    runs = _digit_runs(text)
    pt = _format_match(runs)
    if pt is None:
        return 0, ""
    # 문맥으로 제외: 형식은 PII 같지만 비PII 키워드가 있으면 0
    if any(k in text for k in NEG_KW) and not any(
            k in text for kws in KW.values() for k in kws):
        return 0, ""
    return 1, pt


if __name__ == "__main__":
    tests = [
        "010-1824-0409으로 연락 주세요.",
        "제 주민등록번호는 310405-1709-570입니다.",
        "커피 한잔하면서 잠깐 쉬었다 가시죠.",
        "주문번호는 1234567890입니다.",          # 형식 충돌(10자리) — 문맥변형이 잡아야
        "카드번호는 1234-5678-9012-3456입니다.",
    ]
    for t in tests:
        print(f"  fmt={detect_format(t)}  fmt+ctx={detect_format_context(t)}  | {t}")
