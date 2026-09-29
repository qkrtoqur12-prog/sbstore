import re

from app.services.gemini_optimizer import GeminiOptimizer
from app.services.naver_keywordstool import NaverKeywordsTool

MIN_KEYWORDS = 10
REFINE_CANDIDATE_COUNT = 12
# 2차 정제(검색량 높은 동의어로 핵심 단어 교체)는 실데이터 평가에서 오히려 정확도를 떨어뜨려 기본 비활성화.
# (2026-09-30, 사람이 수정한 상품명 80건 비교: 글자유사도 켬 0.51/0.60 -> 끔 0.57/0.66. 카네이션브로치 -> 꽃브로치, 골드링 -> 금반지처럼 넓은 고검색량어로 바꿈)
USE_REFINE = False


def _is_redundant_with_name(keyword: str, name: str) -> bool:
    """keyword가 name(공백제거)의 부분 문자열이면(=name 안에 포함되는 조각이면) 중복으로 간주.
    반대로 keyword가 name보다 길면서 name을 포함하는 경우(예: 아기+옥노리개)는 새 단어가 추가된 것이므로 허용."""
    kw = keyword.replace(" ", "")
    nm = name.replace(" ", "")
    if not kw or not nm:
        return False
    return len(kw) <= len(nm) and kw in nm


def _normalize(name: str) -> str:
    return name.replace(" ", "").lower()


def _narrows_scope(product_name: str, draft: str, refined: str) -> bool:
    """2차 정제가 초안 단어 앞뒤에 원본에 없던 단어를 붙여 범위를 좁혔는지 확인 (예: 걸이 -> 수건걸이).
    프롬프트로 막아도 검색량 높은 롱테일로 바꾸는 경향이 있어 코드로 한 번 더 거른다."""
    original = _normalize(product_name)
    for rt in refined.split():
        for dt in draft.split():
            if len(dt) >= 2 and dt in rt and rt != dt:
                extra = rt.replace(dt, "", 1).lower()
                if extra and extra not in original:
                    return True
    return False


def _only_deletes(draft: str, refined: str) -> bool:
    """2차 정제 결과가 초안에서 글자만 빼낸 것(부분 수열)인지 확인 (예: 나비 한복노리개 -> 한복노리개, 생일왕관 펠트왕관 -> 생일왕관).
    2차 정제의 역할은 핵심 단어 '교체'이지 삭제가 아니라서, 디자인/소재 단어가 빠지는 걸 막는다."""
    d, r = _normalize(draft), _normalize(refined)
    if len(r) >= len(d):
        return False
    it = iter(d)
    return all(ch in it for ch in r)


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)}


_JUNG_SWAP = {1: 5, 5: 1}  # ㅐ <-> ㅔ (위탁 상품명에서 가장 흔한 오타, 예: 템핑스테이션 -> 탬핑스테이션)


def _spelling_variants(word: str, limit: int = 4) -> list[str]:
    """ㅐ/ㅔ를 한 글자씩 바꾼 표기 변형. 오타 표기는 네이버 연관어가 비어 있어 상품 정체를 추측하게 되므로 올바른 표기로도 조회한다."""
    variants = []
    for i, ch in enumerate(word):
        code = ord(ch) - 0xAC00
        if not 0 <= code < 11172:
            continue
        jung = (code // 28) % 21
        if jung in _JUNG_SWAP:
            swapped = chr(0xAC00 + code + (_JUNG_SWAP[jung] - jung) * 28)
            variants.append(word[:i] + swapped + word[i + 1:])
    return variants[:limit]


def _head_noun(product_name: str) -> str:
    """상품명의 마지막 한글/영문 단어 = 보통 상품 종류를 나타내는 핵심 명사 (예: 불꽃 핸들바 -> 핸들바)"""
    tokens = ["".join(ch for ch in t if ch.isalpha()) for t in product_name.split()]
    tokens = [t for t in tokens if t]
    return tokens[-1] if tokens else ""


def _keeps_head(product_name: str, name: str) -> bool:
    head = _head_noun(product_name).lower()
    return len(head) < 2 or head in _normalize(name)


# 프롬프트 규칙 2의 "빼도 되는" 범용 수식어. 이 외의 단어는 상품을 구별하는 단어일 수 있다.
_GENERIC_WORDS = {
    "다용도", "다목적", "차량용", "휴대용", "여행용", "스텐", "플라스틱", "pc", "실리콘", "우드",
    "사각", "원형", "접이식", "부착식", "컬러", "칼라", "블랙", "화이트", "투명", "미니", "소형", "심플", "초간단",
    # 실데이터 4,266건에서 사람이 대부분 지운 단어
    "프리미엄", "클래식", "데일리", "클린", "디자인", "다기능", "캐릭터", "빈티지", "인테리어", "포인트",
    "전설의", "착한", "인기", "고급", "국산", "가정용", "일반", "장식용소품", "소품", "장식품", "장난감",
    "도구", "보관", "모양", "diy", "만들기", "택1", "랜덤", "색상랜덤", "디자인랜덤",
}
_QUANTITY = re.compile(r"^\d+(p|ea|개|장|쌍|세트|종|구|색|묶음)?$")


def _dropped_words(product_name: str, name: str) -> list[str]:
    """원본에서 빠진 단어 중 범용 수식어/수량이 아닌 것 (예: 벌레자국 스탬프 -> 곤충자국 스탬프 에서 '벌레자국')"""
    nm = _normalize(name)
    words = ["".join(ch for ch in t if ch.isalnum()).lower() for t in product_name.replace("(", " ").replace(")", " ").split()]
    return [w for w in words if len(w) >= 2 and w not in _GENERIC_WORDS and not _QUANTITY.match(w) and w not in nm]


class ProductOptimizer:
    def __init__(self, name_generator: GeminiOptimizer, naver: NaverKeywordsTool):
        self.name_generator = name_generator
        self.naver = naver

    async def _build_context(self, product_name: str) -> tuple[str, bool]:
        """LLM이 상품 정체를 착각하지 않도록(예: 포장용 노리개 -> 치발기), 원본 상품명 기준 네이버 연관검색어를 근거로 붙인다.
        - 상품명 전체로 조회한 연관어는 그대로 사용
        - 띄어쓰기 단위 단어로 조회한 연관어는 무관한 고볼륨어가 많아 원본과 글자(2-gram)가 겹치는 것만 사용"""
        whole = product_name.replace(" ", "")
        # 단어 하나씩 조회하면 수식어가 빠져 엉뚱한 상품으로 유도되므로(정원 딸랑이 -> "딸랑이"는 아기용품,
        # 바벨 펜홀더 -> "바벨"은 헬스용품) 인접한 두 단어를 붙인 조합으로만 보조 조회한다 (예: 데스크오거나이저).
        # 괄호/특수문자가 섞이면 '전구색(오렌지색)(5개' 같은 깨진 조회어가 되므로 단어에서 기호를 뗀다
        words = [w for w in ("".join(ch for ch in t if ch.isalnum()) for t in product_name.replace("(", " ").replace(")", " ").split()) if w]
        pairs = [words[i] + words[i + 1] for i in range(len(words) - 1)]
        pairs = [p for p in pairs if p != whole][:4]
        variants = _spelling_variants(whole)
        try:
            all_whole = await self.naver.get_related_keywords([whole] + variants)
            token_rows = await self.naver.get_related_keywords(pairs) if pairs else []
        except Exception:
            return "", True  # 조회 자체가 실패한 경우는 "데이터 없음"으로 단정하지 않음

        spellings = {whole, *variants}
        volume = {r["keyword"]: r["total"] for r in all_whole if r["keyword"] in spellings}
        spelling_note = ""
        best = max(variants, key=lambda v: volume.get(v, 0), default=None)
        if best and volume.get(best, 0) > volume.get(whole, 0):
            spelling_note = (
                f"\n표기 참고: 원본 '{whole}'(검색량 {volume.get(whole, 0)})보다 '{best}'(검색량 {volume[best]}) 표기가 더 많이 검색됩니다. "
                f"오타일 가능성이 높으니 상품명과 키워드에는 '{best}' 표기를 쓰세요. 이 두 표기 외에 새로운 표기(글자를 바꾼 변형)를 만들지 마세요."
            )

        name_bigrams = _bigrams(whole) | {b for v in variants for b in _bigrams(v)}
        whole_rows = [r for r in all_whole if r["keyword"] not in spellings]
        # 네이버 결과는 검색량 순이라 느슨하게 관련된 고볼륨어(쿨링스틱 -> 다리마사지기)가 위로 온다. 원본과 글자가 겹치는 것부터 보여준다.
        whole_rows.sort(key=lambda r: not (_bigrams(r["keyword"]) & name_bigrams))
        rows = whole_rows[:20]
        seen = {r["keyword"] for r in rows} | spellings
        for r in token_rows:
            if len(rows) >= 30:
                break
            if r["keyword"] not in seen and _bigrams(r["keyword"]) & name_bigrams:
                rows.append(r)
                seen.add(r["keyword"])

        if not rows:
            # 상품명 전체로는 데이터가 없을 때, 핵심 명사만 조회해서 그 명사가 들어간 검색어로 쓰임새를 짐작하게 한다
            # (예: 삐에로 넥카라 -> 강아지넥카라/고양이넥카라 = 반려동물용). 수식어가 빠진 결과라 '참고용'으로만 준다.
            head = _head_noun(product_name)
            head_hint = ""
            if len(head) >= 2 and head != whole:
                try:
                    head_rows = [r for r in await self.naver.get_related_keywords([head]) if head in r["keyword"] and r["keyword"] != head][:15]
                except Exception:
                    head_rows = []
                if head_rows:
                    head_hint = (
                        f"\n참고 - 핵심 명사 '{head}'가 들어간 검색어(수식어가 빠진 결과라 쓰임새 파악용으로만 참고): "
                        + ", ".join(f"{r['keyword']}({r['total']})" for r in head_rows)
                    )
            return (
                f"{spelling_note}\n네이버 연관검색어: (데이터 없음 - 검색 데이터가 거의 없는 생소한 상품입니다. "
                "상품 정체를 추측해서 다른 상품명으로 바꾸지 말고, 원본의 단어를 그대로 포함한 채 표현만 다듬으세요.)" + head_hint
            ), False
        return spelling_note + "\n네이버 연관검색어(실제 구매자 검색어, 월 검색량): " + ", ".join(f"{r['keyword']}({r['total']})" for r in rows), True

    async def optimize(self, product_name: str) -> dict:
        context, has_data = await self._build_context(product_name)
        candidates = await self.name_generator.generate_candidates(product_name, context)

        def guessed(name: str) -> bool:
            # 근거 데이터가 없는데 원본과 글자가 안 겹치거나(손가락못 -> 네일팁) 핵심 명사를 바꾼 경우(핸들바 -> 핸들커버) = 상품을 추측한 것
            return not has_data and (
                not (_bigrams(_normalize(name)) & _bigrams(_normalize(product_name)))
                or not _keeps_head(product_name, name)
                or bool(_dropped_words(product_name, name))  # 벌레자국 스탬프 -> 벌레 스탬프
            )

        if guessed(candidates["optimized_name"]):
            strict_note = (
                f"{context}\n주의: 이전 응답 '{candidates['optimized_name']}'은 원본과 다른 상품으로 바꾼 것입니다. "
                f"원본 단어 {', '.join(_dropped_words(product_name, candidates['optimized_name'])) or _head_noun(product_name)}를 반드시 그대로 포함하세요."
            )
            retried = await self.name_generator.generate_candidates(product_name, strict_note)
            if not guessed(retried["optimized_name"]):
                candidates = retried
            else:
                candidates = {**retried, "optimized_name": product_name}
        if _normalize(candidates["optimized_name"]) == _normalize(product_name):
            # 규칙 1(무조건 변경) 위반 - 한 번만 다시 요청
            retry_note = f"{context}\n주의: 이전 응답이 원본과 똑같은 '{candidates['optimized_name']}'였습니다. 같은 물건을 가리키는 범위 안에서 반드시 다르게 바꾸세요 (띄어쓰기, 대체어, 영문 대문자 정규화, 불필요한 단어 제거 등)."
            retried = await self.name_generator.generate_candidates(product_name, retry_note)
            if _normalize(retried["optimized_name"]) != _normalize(product_name) and not guessed(retried["optimized_name"]):
                candidates = retried
        optimized_name = candidates["optimized_name"]
        candidate_keywords = candidates["keywords"]
        product_type = candidates.get("product_type", "")

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
        if USE_REFINE:
            try:
                refined = await self.name_generator.refine_name(product_name, optimized_name, top_for_refine, product_type)
                # 2차 정제가 원본으로 되돌려 놓았거나, 범위를 좁혔거나(걸이 -> 수건걸이), 단어만 삭제한 경우엔 1차 초안 유지.
                # 원본 근거 데이터가 없는 상품은 검색량 후보도 추측에서 나온 것이라, 원본과 글자가 하나도 안 겹치는 교체도 막는다.
                if (
                    refined and refined.strip()
                    and _normalize(refined) != _normalize(product_name)
                    and not _narrows_scope(product_name, optimized_name, refined)
                    and not _only_deletes(optimized_name, refined)
                    and (has_data or (_bigrams(_normalize(refined)) & _bigrams(_normalize(product_name)) and _keeps_head(product_name, refined) and not _dropped_words(product_name, refined)))
                ):
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
            "product_type": product_type,
            "original_name": product_name,
            "optimized_name": optimized_name,
            "optimized_name_pc": name_volume["pc"],
            "optimized_name_mobile": name_volume["mobile"],
            "keywords": verified[:20],
        }
