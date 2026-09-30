import asyncio
import json
import os
import re

from google import genai
from google.genai import errors, types

MODEL = "gemini-2.5-flash"
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0
MAX_RATE_LIMIT_WAIT = 90.0


def _extract_retry_delay(error: errors.APIError) -> float | None:
    """429 응답의 RetryInfo.retryDelay(예: '58s')를 초 단위로 파싱."""
    details = getattr(error, "details", None)
    if not isinstance(details, dict):
        return None
    error_details = details.get("error", details).get("details", [])
    for d in error_details:
        if d.get("@type", "").endswith("RetryInfo"):
            m = re.match(r"([\d.]+)s?", str(d.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    return None

SYSTEM_PROMPT = """# 역할
Naver Smartstore 상품명 및 키워드 최적화 전문가

# 작업
제공된 기존 상품명을 분석하여 Naver Smartstore 판매에 최적화된 새로운 상품명과 해당 상품에 대한 15개 이상의 관련 키워드 후보 목록을 생성합니다.

# 상품명 최적화 규칙 (반드시 준수)
1. **무조건 변경**: 최적화된 상품명은 기존 상품명과 반드시 달라야 합니다. 원문 그대로 반환하는 것은 허용되지 않습니다. 최소 1개 이상의 단어를 실제로 교체/제거하세요.
2. **불필요한 수식어/브랜드명 제거가 기본**: 실제 판매 데이터 분석 결과, 최적화된 상품명은 원본보다 **짧아지는 경우가 대부분(약 77%)**입니다. 아래와 같은 범용적인 수식어나 특정 제조사/브랜드명이 원본에 포함되어 있다면 과감히 제거하고, 핵심 키워드 위주로 간결하게 정리하는 것을 기본으로 합니다:
   - 용도: 다용도, 다목적, 차량용, 휴대용, 여행용
   - 재질: 스텐, 플라스틱, PC, 실리콘, 우드
   - 형태: 사각, 원형, 접이식, 부착식
   - 색상: 컬러, 블랙, 투명
   - 크기: 미니, 소형
   - 홍보/분위기: 프리미엄, 클래식, 심플, 데일리, 클린, 디자인, 다기능, 캐릭터, 빈티지, 인테리어, 포인트, 전설의, 착한, 인기, 고급, 국산, 가정용, 일반, 초간단
   - 막연한 분류어: 장식용소품, 소품, 장식품, 장난감, 도구, 보관, 모양, DIY, 만들기 (단 상품 종류 자체를 나타내는 단어면 유지)
   - 선택/랜덤 표기: 택1, 랜덤, 색상랜덤, 디자인랜덤, 3종, 2종
   - 특정 브랜드명/제조사명
   예: "시스맥스 르와 데스크 오거나이저" -> "데스크 오거나이저" / "띄움 실리콘 주방집게" -> "실리콘 주방집게" / "컬러 파티컵 4P" -> "파티컵 4P"
   (다만 검색량이 높은 메인키워드가 여러 개 존재한다면 예외적으로 길어질 수 있습니다. 이 경우 하나의 핵심 명사에 서로 다른 수식어를 각각 붙여 별도의 메인키워드 형태로 나열합니다. 예: "생일 펠트 왕관" -> "생일왕관 펠트왕관")
3. **대체어 사용**: 흔히 쓰이는 유사어/대체어가 있다면 적극적으로 바꿔 씁니다. (예: 스테이플러->호치키스, 공갈칼->가짜칼, 고리->걸이)
4. **수량 표현은 불필요하면 생략**: 수량이 1(1개, 1P, 1세트 등)인 경우는 반드시 생략합니다. 수량이 2 이상이면 그대로 유지합니다 (실제 데이터에서 90% 유지, 예: 컬러 파티컵 4P -> 파티컵 4P). 영문 약어/단위 표기는 대문자로 정규화합니다. 예: "10p"->"10P", "a4"->"A4", "usb"->"USB".
5. **특수문자 금지**: `|`, `/`, `()`, `-`, `*`, `~`, `&` 등 특수문자를 사용하지 않습니다. 한글, 영문, 숫자, 띄어쓰기만 사용합니다.
6. **실제 검색어 기반**: 실제 구매자가 검색할 법한 자연스럽고 창의적인 표현을 고려합니다. 사전적으로만 맞는 딱딱한 표현보다 실사용 검색어에 가깝게 만듭니다.
7. **SEO/네이버쇼핑 로직 고려**: 핵심 키워드를 앞쪽에 배치하고, 같은 의미의 단어를 중복 나열하지 않으며, 과도한 키워드 나열은 지양하는 자연스러운 상품명 형태를 유지합니다.
8. **원본에 없던 설명 단어를 덧붙이지 않기**: 실제 데이터에서 추가되는 단어는 거의 수량(10P, 2P)뿐입니다. '세트', '파티용품', '인테리어' 같은 단어를 새로 붙이지 마세요. 새 단어는 원본 단어를 더 많이 검색되는 대체어로 '교체'할 때만 씁니다. (예: 생일 축하 어깨띠 -> 생일어깨띠 O, 생일어깨띠 생일파티용품 X / 일반색연필 24색 -> 색연필 24색 O, 일반색연필 24색 세트 X)
9. **이미 최적이면 거의 그대로**: 원본이 이미 짧고 핵심 단어만 있으면(예: 눈사람 키링, T형 가구손잡이) 띄어쓰기/수량 정리 정도만 합니다.

예시: 공갈칼 -> 가짜칼 / 시스맥스 르와 데스크 오거나이저 -> 데스크 오거나이저 / 띄움 실리콘 주방집게 -> 실리콘 주방집게 / 부착식 다용도 스텐 걸이 1P -> 부착식 스텐 걸이 / 생일 펠트 왕관 -> 생일왕관 펠트왕관
실제 수정 사례 추가: 도루코 커터날S 10PCS -> 커터날S 10PCS (브랜드 제거) / 칼라 점멸 LED 캔들 -> LED 캔들 / 플러시 헤어 스크런치 -> 헤어 스크런치 / 클래식 샤워헤드 -> 샤워헤드 / 핸디형선풍기 -> 손풍기 (더 많이 쓰는 대체어) / 슬림도마 -> 얇은도마 / 팬시돼지 -> 돼지저금통 (상품 정체를 드러내는 명사로) / 마이룸 4단 디럭스 캐비넷 -> 4단 책상서랍 / 휴대용 접이식 수박모양 부채 -> 수박 부채 / 부착식 6구 후크 고리 -> 부착식 고리 6구 / 미니학사모 1개입 -> 미니학사모 (수량 1 생략)

# 비슷한 상품 수정 사례 활용
- 입력에 '비슷한 상품을 사람이 실제로 수정한 사례'가 있으면, 그 사람의 수정 스타일을 가장 우선으로 따르세요: 어떤 단어를 지웠는지, 어떤 대체어를 썼는지, 띄어쓰기와 수량 표기, 키워드를 어떤 식으로 조합했는지.
- 사례의 상품이 이 상품과 같은 종류면 그 수정 방식과 키워드 패턴을 적극적으로 따라 하세요. 다른 물건이면 스타일만 참고하고, 그 상품의 단어를 이 상품에 가져오지 마세요.
- 키워드는 사례처럼 '핵심 명사(와 동의어) + 사용장소/대상/용도/특징' 조합을 중심으로 뽑으세요. (예: 휴지통 -> 사무실휴지통, 화장대미니쓰레기통)

# 진행 방식
0. **상품 정체 파악 (가장 중요)**: 이 상품이 실제로 무엇인지(어떤 물건이고, 누가, 어디에 쓰는지)를 먼저 한 문장으로 정리합니다.
   - 이 스토어는 생활잡화/소품/주방/문구/파티/반려동물/취미용품을 파는 위탁판매 스토어이고, 구매자는 일반 소비자입니다. 전문 용어처럼 보이는 단어도 공업용/산업용/의료기관용 뜻보다 일반 소비자가 쓰는 생활용품 뜻을 먼저 고려하세요.
     (예: 링게이지 = 반지 호수(사이즈) 측정기, 공업용 측정공구 X / 세발기 = 누워서 머리 감기는 간병용품)
   - 함께 제공되는 "네이버 연관검색어"는 실제 구매자들이 이 상품명과 함께 검색하는 단어입니다. 상품의 정체를 판단하는 가장 중요한 근거로 사용하세요.
     예: "나비노리개"의 연관검색어가 보자기/명절선물보자기/포장노리개/매듭끈이라면, 아기 장난감이 아니라 선물 포장용 장식 노리개입니다.
   - 상품을 구별해 주는 디자인/형태/소재 단어(예: 나비노리개의 "나비", 옥노리개의 "옥", 펠트왕관의 "펠트")는 범용 수식어가 아니므로 유지합니다. 제거 대상은 규칙 2에 나열된 범용 수식어와 브랜드명뿐입니다.
   - 연관검색어에 나온 다른 물건(예: 노리개 연관어의 "매듭끈", "보자기")을 상품명에 끼워 넣지 마세요. 상품명에는 이 상품 자체를 가리키는 단어만 씁니다.
   - 단어의 다른 뜻으로 착각하지 마세요. 확신이 없으면 원본 상품명의 핵심 명사를 그대로 유지하는 쪽이 안전합니다. 다른 종류의 상품으로 바꾸는 것은 가장 큰 실패입니다.
     (예: 노리개 -> 치발기/딸랑이 X, 손가락못 -> 가짜손톱/손가락보호대 X)
   - 물건의 형태를 나타내는 단어(스틱, 바, 롤러, 패드, 커버, 매트, 링 등)는 다른 형태로 바꾸지 마세요. 형태가 바뀌면 다른 물건입니다. (예: 쿨링스틱 -> 마사지롤러 X, 핸들바 -> 핸들커버 X)
   - 연관검색어 목록에는 느슨하게 관련된 고검색량 단어(다른 상품)도 섞여 있습니다. 검색량이 높다고 해서 그 단어로 상품 종류를 바꾸지 마세요.
   - 흔한 생활용품/소품(머리띠, 스탬프, 넥카라, 모자, 스티커 등)에 생소한 수식어가 붙어 있으면, 그 수식어는 기능이 아니라 모양/디자인/테마일 가능성이 높습니다. 파티/장난/코스튬/반려동물용 소품인 경우가 많습니다.
     (예: 와이파이 머리띠 = 와이파이 아이콘 모양의 파티용 머리띠, 전자기기 X / 벌레자국 스탬프 = 벌레 자국 모양을 찍는 도장, "자국"을 빼면 다른 물건)
1. 기존 상품명 분석: 핵심 기능, 특징, 대상을 파악하고 불필요하거나 검색 효율이 낮은 요소를 식별합니다.
2. 위 규칙에 따라 최적화된 상품명을 생성합니다.
3. 관련 키워드 후보 추출: 잠재 고객이 Naver에서 검색할 만한 키워드를 최소 15개 이상 폭넓게 추출합니다. 동의어, 유사어, 사용 목적, 재질, 대상 고객 등 다양한 관점에서 탐색합니다.
   - 키워드는 모두 '이 상품을 사려는 사람'이 검색할 단어여야 합니다. 다른 종류의 상품(예: 스탬프 상품에 스탬프잉크/네임스탬프/스탬프제작, 파티 머리띠에 여자머리띠/러닝머리띠)은 넣지 마세요.
   - '참고 - 핵심 명사가 들어간 검색어' 목록은 쓰임새 파악용일 뿐이니 그대로 옮겨 적지 말고, 이 상품과 같은 물건인 것만 고르세요.
   이 후보들은 이후 실제 네이버 검색광고 API로 검색량을 검증할 것이므로, 정확한 수치보다는 "실제로 존재할 법한 자연스러운 검색어"인지에 집중합니다.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "product_type": {"type": "string"},
        "optimized_name": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product_type", "optimized_name", "keywords"],
    "propertyOrdering": ["product_type", "optimized_name", "keywords"],
}

REFINE_SYSTEM_PROMPT = """너는 네이버 스마트스토어 상품명을 실제 검색량 데이터로 최종 개선하는 전문가다.

아래 정보를 참고해서 '초안 상품명'을 더 나은 상품명으로 다듬어라.
- 원본 상품명
- 초안 상품명 (1차로 만들어진 것)
- 상품 정체 (이 상품이 실제로 무엇인지 1차에서 정리한 설명)
- 실제 네이버 검색량이 확인된 연관 키워드 목록 (검색량 높은 순)

# 판단 절차
1. 검색량 목록을 훑어보며, 아래 3가지 조건을 **모두** 만족하는 키워드가 있는지 확인한다:
   - **메인키워드**다: 상품 종류 자체를 가리키는 핵심 단어/복합어다 (수식어가 잔뜩 붙은 롱테일 키워드가 아님)
   - **같은 물건이다**: '상품 정체' 설명과 비교했을 때 사실상 같은 물건을 가리킨다. 용도나 대상이 달라지는 단어(예: 한복/포장용 노리개 -> 치발기), 더 넓은 상위 분류(예: 핀스크린 -> 피젯토이, 장난감), 부위만 겹치는 다른 상품(예: 손가락못 -> 손가락보호대)은 모두 탈락이다. 검색량이 아무리 높아도 다른 물건이면 쓰지 않는다.
     특히 아래는 모두 탈락이다:
     · 함께 쓰는 다른 물건: 생일왕관 -> 생일토퍼, 노리개 -> 매듭끈
     · 특정 용도로 범위를 좁힌 것: 다용도 걸이 -> 수건걸이 (원본이 수건용이라고 명시하지 않았으면 좁히지 않는다)
     · 더 넓은 상위 분류: 핀스크린 -> 장난감, 집게 -> 주방도구
     · 형태가 다른 물건: 탬핑스테이션 -> 탬핑매트, 쿨링스틱 -> 쿨링롤러, 핸들바 -> 핸들커버 (스테이션/매트/스틱/롤러/바/커버처럼 형태를 나타내는 단어는 바꾸지 않는다)
   - **명칭이 다르다**: 초안 상품명에 쓰인 단어와는 다른 표현/동의어다 (완전히 같은 단어면 교체할 의미가 없음)
2. 위 조건을 만족하는 키워드가 여러 개라면, 그 중 **검색량이 가장 높은 것**을 고른다.
3. 초안 상품명에서 해당 단어(보통 상품 종류를 나타내는 핵심 명사)만 그 키워드로 자연스럽게 교체한다. 전체를 새로 쓰지 말고 해당 부분만 치환한다.
   예: 초안 "정원딸랑이" + 검색량목록에 메인키워드 "종"이 "딸랑이"보다 훨씬 높음 -> "정원종"
   예: 초안 "바벨펜홀더" + 검색량목록에 메인키워드 "펜꽂이"가 "펜홀더"보다 훨씬 높음 -> "바벨펜꽂이"
4. 조건을 모두 만족하는 명확한 대체 후보가 없다면 억지로 바꾸지 말고 초안을 그대로 유지한다. 애매하면 유지가 정답이다. 초안의 디자인/소재 단어(나비, 옥, 펠트 등)는 없애지 않는다.
5. 특수문자(|, /, (), -, *, ~, & 등)는 사용하지 않는다. 수량이 1(1개, 1P 등)인 표현은 넣지 않는다.
6. 설명 없이 최종 상품명 한 줄만 출력한다. 따옴표나 다른 텍스트를 붙이지 않는다.
"""


class GeminiOptimizer:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. backend/.env에 키를 입력해주세요.")
        self.client = genai.Client(api_key=key)

    async def generate_candidates(self, product_name: str, context: str = "") -> dict:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=MODEL,
                    contents=f"{SYSTEM_PROMPT}\n\n기존 상품명: {product_name}{context}",
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                    ),
                )
                data = json.loads(response.text)
                return {
                    "product_type": data.get("product_type", ""),
                    "optimized_name": data["optimized_name"],
                    "keywords": data["keywords"],
                }
            except errors.ServerError as e:
                # 503(과부하) 등 구글 서버 측 일시적 오류 - 지수 백오프 후 재시도
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            except errors.ClientError as e:
                if getattr(e, "code", None) == 429 and attempt < MAX_RETRIES - 1:
                    last_error = e
                    wait = _extract_retry_delay(e)
                    if wait is None:
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(min(wait, MAX_RATE_LIMIT_WAIT))
                else:
                    raise
        raise RuntimeError(f"Gemini 서버가 계속 응답하지 않습니다 (여러 번 재시도 실패): {last_error}")

    async def refine_name(self, product_name: str, draft_name: str, verified_keywords: list[dict], product_type: str = "") -> str:
        keyword_str = ", ".join(f"{k['keyword']}({k['total']})" for k in verified_keywords)
        prompt = f"{REFINE_SYSTEM_PROMPT}\n\n원본 상품명: {product_name}\n초안 상품명: {draft_name}\n상품 정체: {product_type}\n검색량 확인된 키워드(검색량 높은 순): {keyword_str}"

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0),
                )
                text = (response.text or "").strip()
                return text.strip('"').strip("'").splitlines()[0].strip() if text else draft_name
            except errors.ServerError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            except errors.ClientError as e:
                if getattr(e, "code", None) == 429 and attempt < MAX_RETRIES - 1:
                    last_error = e
                    wait = _extract_retry_delay(e)
                    if wait is None:
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(min(wait, MAX_RATE_LIMIT_WAIT))
                else:
                    return draft_name  # 개선 실패 시 초안 이름으로 폴백
        return draft_name
