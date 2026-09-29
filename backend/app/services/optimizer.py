from app.services.gemini_optimizer import GeminiOptimizer
from app.services.naver_keywordstool import NaverKeywordsTool

MIN_KEYWORDS = 10
REFINE_CANDIDATE_COUNT = 12


def _is_redundant_with_name(keyword: str, name: str) -> bool:
    """keyword가 name(공백제거)의 부분 문자열이면(=name 안에 포함되는 조각이면) 중복으로 간주.
    반대로 keyword가 name보다 길면서 name을 포함하는 경우(예: 아기+옥노리개)는 새 단어가 추가된 것이므로 허용."""
    kw = keyword.replace(" ", "")
    nm = name.replace(" ", "")
    if not kw or not nm:
        return False
    return len(kw) <= len(nm) and kw in nm


class ProductOptimizer:
    def __init__(self, name_generator: GeminiOptimizer, naver: NaverKeywordsTool):
        self.name_generator = name_generator
        self.naver = naver

    async def optimize(self, product_name: str) -> dict:
        candidates = await self.name_generator.generate_candidates(product_name)
        optimized_name = candidates["optimized_name"]
        candidate_keywords = candidates["keywords"]

        # 네이버 키워드도구는 hint마다 관련성이 느슨한 키워드를 대량으로 반환하므로(수백~수천개),
        # naver_results 전체가 아니라 "LLM이 실제로 제안한 후보 키워드"만 골라서 신뢰도 있는 후보로 사용한다.
        hint_pool = [optimized_name] + candidate_keywords
        naver_results = await self.naver.get_related_keywords(hint_pool)
        volume_by_keyword = {row["keyword"]: row for row in naver_results}

        relevant_candidates = []
        for kw in candidate_keywords:
            key = kw.replace(" ", "")
            if key in volume_by_keyword:
                relevant_candidates.append(volume_by_keyword[key])
        relevant_candidates.sort(key=lambda x: x["total"], reverse=True)

        # 실제 검색량 데이터를 참고해 상품명의 핵심 단어를 더 검색되는 동의어로 교체 시도
        # (관련성 낮은 naver_results 전체가 아니라 LLM이 제안한 후보 키워드 중 상위만 사용)
        top_for_refine = relevant_candidates[:REFINE_CANDIDATE_COUNT]
        try:
            refined = await self.name_generator.refine_name(product_name, optimized_name, top_for_refine)
            if refined and refined.strip():
                optimized_name = refined.strip()
        except Exception:
            pass  # 개선 실패 시 1차 초안 이름 그대로 사용

        # 이름이 교체되어 기존 조회 결과에 없을 수 있으므로 실제 검색량 재조회
        if optimized_name.replace(" ", "") not in volume_by_keyword:
            extra = await self.naver.get_related_keywords([optimized_name])
            for row in extra:
                volume_by_keyword[row["keyword"]] = row
                if row["keyword"] not in {r["keyword"] for r in naver_results}:
                    naver_results.append(row)

        verified: list[dict] = []
        seen = set()
        for kw in candidate_keywords:
            key = kw.replace(" ", "")
            if key in volume_by_keyword and key not in seen and not _is_redundant_with_name(kw, optimized_name):
                verified.append(volume_by_keyword[key])
                seen.add(key)

        verified.sort(key=lambda x: x["total"], reverse=True)

        if len(verified) < MIN_KEYWORDS:
            for row in naver_results:
                if row["keyword"] in seen:
                    continue
                if _is_redundant_with_name(row["keyword"], optimized_name):
                    continue
                verified.append(row)
                seen.add(row["keyword"])
                if len(verified) >= MIN_KEYWORDS:
                    break

        remaining = [
            kw for kw in candidate_keywords
            if kw.replace(" ", "") not in seen and not _is_redundant_with_name(kw, optimized_name)
        ]
        for kw in remaining:
            if len(verified) >= MIN_KEYWORDS:
                break
            verified.append({"keyword": kw, "pc": 0, "mobile": 0, "total": 0})
            seen.add(kw.replace(" ", ""))

        name_volume = volume_by_keyword.get(optimized_name.replace(" ", ""), {"pc": 0, "mobile": 0, "total": 0})

        return {
            "original_name": product_name,
            "optimized_name": optimized_name,
            "optimized_name_pc": name_volume["pc"],
            "optimized_name_mobile": name_volume["mobile"],
            "keywords": verified[:20],
        }
