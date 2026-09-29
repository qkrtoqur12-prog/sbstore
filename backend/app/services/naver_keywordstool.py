import asyncio
import hmac
import hashlib
import base64
import time
import httpx

TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=5.0)
MAX_RETRIES = 5


def _parse_count(value) -> int:
    """monthlyPcQcCnt/monthlyMobileQcCnt는 소량일 때 '< 10' 문자열로 옴."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        if "<" in value:
            return 5
        try:
            return int(value.replace(",", ""))
        except ValueError:
            return 0
    return 0


class NaverKeywordsTool:
    BASE_URL = "https://api.searchad.naver.com"

    def __init__(self, customer_id: str, api_key: str, secret_key: str):
        self.customer_id = customer_id
        self.api_key = api_key
        self.secret_key = secret_key

    def _signature(self, timestamp: str, method: str, uri: str) -> str:
        message = f"{timestamp}.{method}.{uri}"
        raw = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        return base64.b64encode(raw).decode("utf-8")

    def _headers(self, method: str, uri: str) -> dict:
        ts = str(int(time.time() * 1000))
        path_only = uri.split("?")[0]
        return {
            "X-Timestamp": ts,
            "X-API-KEY": self.api_key,
            "X-Customer": str(self.customer_id),
            "X-Signature": self._signature(ts, method, path_only),
            "Content-Type": "application/json",
        }

    async def _query_chunk(self, hint_keywords: list[str]) -> list[dict]:
        cleaned = [k.replace(" ", "") for k in hint_keywords if k.strip()][:5]
        if not cleaned:
            return []
        uri = f"/keywordstool?hintKeywords={','.join(cleaned)}&showDetail=1"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # 대량 처리 시 초당 호출 제한(429)이나 일시적 5xx가 나므로 간격을 늘려가며 재시도 (서명 timestamp는 매번 새로 생성)
            for attempt in range(MAX_RETRIES):
                r = await client.get(f"{self.BASE_URL}{uri}", headers=self._headers("GET", uri))
                if r.is_success:
                    return r.json().get("keywordList", [])
                if (r.status_code != 429 and r.status_code < 500) or attempt == MAX_RETRIES - 1:
                    raise Exception(f"키워드도구 조회 실패 [{r.status_code}]: {r.text}")
                await asyncio.sleep(2 ** attempt)

    async def get_related_keywords(self, hint_keywords: list[str]) -> list[dict]:
        """hint_keywords를 5개씩 나눠 조회 후 합쳐서 반환. 중복 relKeyword는 제거."""
        seen: dict[str, dict] = {}
        for i in range(0, len(hint_keywords), 5):
            chunk = hint_keywords[i : i + 5]
            rows = await self._query_chunk(chunk)
            for row in rows:
                kw = row.get("relKeyword", "").strip()
                if not kw:
                    continue
                pc = _parse_count(row.get("monthlyPcQcCnt"))
                mobile = _parse_count(row.get("monthlyMobileQcCnt"))
                if kw not in seen or (pc + mobile) > (seen[kw]["pc"] + seen[kw]["mobile"]):
                    seen[kw] = {"keyword": kw, "pc": pc, "mobile": mobile, "total": pc + mobile}
        return sorted(seen.values(), key=lambda x: x["total"], reverse=True)
