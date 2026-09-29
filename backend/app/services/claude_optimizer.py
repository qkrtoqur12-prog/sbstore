import json
import os
from anthropic import AsyncAnthropic

MODEL = "claude-haiku-4-5"

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
   - 특정 브랜드명/제조사명
   예: "시스맥스 르와 데스크 오거나이저" -> "데스크 오거나이저" / "띄움 실리콘 주방집게" -> "실리콘 주방집게" / "컬러 파티컵 4P" -> "파티컵 4P"
   (다만 검색량이 높은 메인키워드가 여러 개 존재한다면 예외적으로 길어질 수 있습니다. 이 경우 하나의 핵심 명사에 서로 다른 수식어를 각각 붙여 별도의 메인키워드 형태로 나열합니다. 예: "생일 펠트 왕관" -> "생일왕관 펠트왕관")
3. **대체어 사용**: 흔히 쓰이는 유사어/대체어가 있다면 적극적으로 바꿔 씁니다. (예: 스테이플러->호치키스, 공갈칼->가짜칼, 고리->걸이)
4. **수량 표현은 불필요하면 생략**: 수량이 1(1개, 1P, 1세트 등)인 경우는 반드시 생략합니다. 수량이 2 이상이면서 상품 구성상 꼭 필요한 정보일 때만 유지합니다. 영문 약어/단위 표기는 대문자로 정규화합니다. 예: "10p"->"10P", "a4"->"A4", "usb"->"USB".
5. **특수문자 금지**: `|`, `/`, `()`, `-`, `*`, `~`, `&` 등 특수문자를 사용하지 않습니다. 한글, 영문, 숫자, 띄어쓰기만 사용합니다.
6. **실제 검색어 기반**: 실제 구매자가 검색할 법한 자연스럽고 창의적인 표현을 고려합니다. 사전적으로만 맞는 딱딱한 표현보다 실사용 검색어에 가깝게 만듭니다.
7. **SEO/네이버쇼핑 로직 고려**: 핵심 키워드를 앞쪽에 배치하고, 같은 의미의 단어를 중복 나열하지 않으며, 과도한 키워드 나열은 지양하는 자연스러운 상품명 형태를 유지합니다.

예시: 공갈칼 -> 가짜칼 / 시스맥스 르와 데스크 오거나이저 -> 데스크 오거나이저 / 띄움 실리콘 주방집게 -> 실리콘 주방집게 / 부착식 다용도 스텐 걸이 1P -> 부착식 스텐 걸이 / 생일 펠트 왕관 -> 생일왕관 펠트왕관

# 진행 방식
1. 기존 상품명 분석: 핵심 기능, 특징, 대상을 파악하고 불필요하거나 검색 효율이 낮은 요소를 식별합니다.
2. 위 규칙에 따라 최적화된 상품명을 생성합니다.
3. 관련 키워드 후보 추출: 잠재 고객이 Naver에서 검색할 만한 키워드를 최소 15개 이상 폭넓게 추출합니다. 동의어, 유사어, 사용 목적, 재질, 대상 고객, 관련 상품 등 다양한 관점에서 탐색합니다.
   이 후보들은 이후 실제 네이버 검색광고 API로 검색량을 검증할 것이므로, 정확한 수치보다는 "실제로 존재할 법한 자연스러운 검색어"인지에 집중합니다.

# 출력 형식
아래 JSON 형식으로만 응답하세요. 다른 설명이나 마크다운 코드펜스 없이 순수 JSON만 출력합니다.
{"optimized_name": "최적화된 상품명", "keywords": ["키워드1", "키워드2", "..."]}
"""

REFINE_SYSTEM_PROMPT = """너는 네이버 스마트스토어 상품명을 실제 검색량 데이터로 최종 개선하는 전문가다.

아래 정보를 참고해서 '초안 상품명'을 더 나은 상품명으로 다듬어라.
- 원본 상품명
- 초안 상품명 (1차로 만들어진 것)
- 실제 네이버 검색량이 확인된 연관 키워드 목록 (검색량 높은 순)

# 판단 절차
1. 검색량 목록을 훑어보며, 아래 3가지 조건을 **모두** 만족하는 키워드가 있는지 확인한다:
   - **메인키워드**다: 상품 종류 자체를 가리키는 핵심 단어/복합어다 (수식어가 잔뜩 붙은 롱테일 키워드가 아님)
   - **밀접하다**: 이 상품과 의미상 사실상 같은 것을 가리킨다 (전혀 다른 상품 카테고리가 아님)
   - **명칭이 다르다**: 초안 상품명에 쓰인 단어와는 다른 표현/동의어다 (완전히 같은 단어면 교체할 의미가 없음)
2. 위 조건을 만족하는 키워드가 여러 개라면, 그 중 **검색량이 가장 높은 것**을 고른다.
3. 초안 상품명에서 해당 단어(보통 상품 종류를 나타내는 핵심 명사)만 그 키워드로 자연스럽게 교체한다. 전체를 새로 쓰지 말고 해당 부분만 치환한다.
   예: 초안 "정원딸랑이" + 검색량목록에 메인키워드 "종"이 "딸랑이"보다 훨씬 높음 -> "정원종"
   예: 초안 "바벨펜홀더" + 검색량목록에 메인키워드 "펜꽂이"가 "펜홀더"보다 훨씬 높음 -> "바벨펜꽂이"
4. 조건을 모두 만족하는 명확한 대체 후보가 없다면 억지로 바꾸지 말고 초안을 그대로 유지한다.
5. 특수문자(|, /, (), -, *, ~, & 등)는 사용하지 않는다. 수량이 1(1개, 1P 등)인 표현은 넣지 않는다.
6. 설명 없이 최종 상품명 한 줄만 출력한다. 따옴표나 다른 텍스트를 붙이지 않는다.
"""


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
    return t.strip()


class ClaudeOptimizer:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다. backend/.env에 키를 입력해주세요.")
        # 100개 이상 대량 처리 시 일시적 429/5xx 오류에 대비해 SDK 기본 재시도(2회)보다 넉넉하게 설정
        self.client = AsyncAnthropic(api_key=key, max_retries=5)

    async def generate_candidates(self, product_name: str) -> dict:
        resp = await self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"기존 상품명: {product_name}"}],
        )
        raw = "".join(block.text for block in resp.content if block.type == "text")
        cleaned = _strip_code_fence(raw)
        data = json.loads(cleaned)
        return {
            "optimized_name": data["optimized_name"],
            "keywords": data["keywords"],
        }

    async def refine_name(self, product_name: str, draft_name: str, verified_keywords: list[dict]) -> str:
        keyword_str = ", ".join(f"{k['keyword']}({k['total']})" for k in verified_keywords)
        resp = await self.client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=REFINE_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"원본 상품명: {product_name}\n초안 상품명: {draft_name}\n검색량 확인된 키워드(검색량 높은 순): {keyword_str}",
            }],
        )
        raw = "".join(block.text for block in resp.content if block.type == "text")
        return raw.strip().strip('"').strip("'").splitlines()[0].strip() if raw.strip() else draft_name
