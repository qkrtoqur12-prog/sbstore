import base64
import json
import os
import time
from pathlib import Path

import bcrypt
import httpx

TOKEN_URL = "https://api.commerce.naver.com/external/v1/oauth2/token"
CATEGORIES_URL = "https://api.commerce.naver.com/external/v1/categories"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "naver_categories_cache.json"
CACHE_MAX_AGE_SECONDS = 30 * 24 * 3600  # 30일 - 카테고리 체계는 자주 안 바뀜


def _sign(client_id: str, client_secret: str) -> tuple[str, str]:
    timestamp = str(int((time.time() - 3) * 1000))
    password = f"{client_id}_{timestamp}"
    hashed = bcrypt.hashpw(password.encode("utf-8"), client_secret.encode("utf-8"))
    sign = base64.standard_b64encode(hashed).decode("utf-8")
    return timestamp, sign


class NaverCommerceClient:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        self.client_id = client_id or os.environ.get("NAVER_COMMERCE_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("NAVER_COMMERCE_CLIENT_SECRET", "")
        if not self.client_id or not self.client_secret:
            raise RuntimeError("NAVER_COMMERCE_CLIENT_ID/SECRET이 설정되지 않았습니다. backend/.env에 키를 입력해주세요.")
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    async def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        timestamp, sign = _sign(self.client_id, self.client_secret)
        data = {
            "client_id": self.client_id,
            "timestamp": timestamp,
            "client_secret_sign": sign,
            "grant_type": "client_credentials",
            "type": "SELF",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(TOKEN_URL, data=data, headers={"content-type": "application/x-www-form-urlencoded"})
            if not r.is_success:
                raise Exception(f"네이버 커머스API 토큰 발급 실패 [{r.status_code}]: {r.text}")
            body = r.json()
            self._token = body["access_token"]
            self._token_expires_at = time.time() + body.get("expires_in", 10800)
            return self._token

    async def fetch_all_categories(self) -> list[dict]:
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(CATEGORIES_URL, headers={"Authorization": f"Bearer {token}"})
            if not r.is_success:
                raise Exception(f"카테고리 조회 실패 [{r.status_code}]: {r.text}")
            return r.json()


_memory_cache: list[dict] | None = None


async def load_leaf_categories(client: NaverCommerceClient | None = None) -> list[dict]:
    """리프 카테고리(last=true) 목록을 프로세스 메모리 -> 로컬 캐시(30일) -> API 순으로 읽는다."""
    global _memory_cache
    if _memory_cache is not None:
        return _memory_cache

    if CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            with open(CACHE_PATH, encoding="utf-8") as f:
                _memory_cache = json.load(f)
                return _memory_cache

    if client is None:
        client = NaverCommerceClient()
    categories = await client.fetch_all_categories()
    leaf = [c for c in categories if c.get("last")]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(leaf, f, ensure_ascii=False)

    _memory_cache = leaf
    return leaf


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else ({s} if s else set())


FUZZY_OVERLAP_THRESHOLD = 0.5
FUZZY_MAX_SCORE = 45


def find_candidate_categories(search_terms: list[str], leaf_categories: list[dict], top_n: int = 25) -> list[dict]:
    """검색어(상품명/키워드)와 카테고리명을 로컬에서 대략 매칭해 상위 후보만 추린다.

    - "갤럭시 케이스" 같은 카테고리명은 "갤럭시S24케이스"처럼 중간에 모델명이 끼면
      연속 문자열로는 안 잡히므로, 카테고리명을 공백 기준 토큰으로 쪼개 검색어 전체에서
      비연속으로라도 모든 토큰이 등장하는지도 함께 확인한다.
    - "가디건"/"카디건"처럼 한두 글자만 다른 철자 변이는 2-gram 겹침 비율로 보조 매칭한다."""
    terms = [t.replace(" ", "") for t in search_terms if t and t.strip()]
    combined = "".join(terms)
    combined_bigrams = _bigrams(combined)
    scored: list[tuple[int, dict]] = []

    for cat in leaf_categories:
        raw_name = cat["name"].strip()
        name = raw_name.replace(" ", "")
        name_tokens = [tok for tok in raw_name.split() if tok]
        best = 0
        for term in terms:
            if not term:
                continue
            if term == name:
                best = max(best, 100)
            elif name in term or term in name:
                best = max(best, 60)
        if len(name_tokens) >= 2 and all(tok in combined for tok in name_tokens):
            best = max(best, 55)

        name_bigrams = _bigrams(name)
        if name_bigrams:
            overlap = len(name_bigrams & combined_bigrams) / len(name_bigrams)
            if overlap >= FUZZY_OVERLAP_THRESHOLD:
                best = max(best, round(overlap * FUZZY_MAX_SCORE))

        if best > 0:
            scored.append((best, cat))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_n]]
