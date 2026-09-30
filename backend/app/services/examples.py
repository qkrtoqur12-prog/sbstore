"""사람이 실제로 수정한 상품명 사례(상품명_예시_양식.xlsx) 중 입력 상품과 비슷한 것을 찾아 LLM에 보여준다.
규칙만으로는 설명되지 않는 사람의 수정 스타일(대체어 선택, 지우는 단어, 키워드 고르는 방식)을 따라하게 하기 위함.
데이터 파일이 없으면 사례 없이 동작한다."""
import json
import math
import os
from collections import Counter

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "name_examples.json")
# 평가할 때는 입력과 원본이 똑같은 사례(=정답)를 빼야 공정하다
EXCLUDE_EXACT = os.environ.get("EXAMPLES_EXCLUDE_EXACT") == "1"
# 평가용: 이 유사도 이상인 거의 같은 상품(형제 상품)도 빼서 "처음 보는 상품" 기준 정확도를 잰다
EXCLUDE_NEAR = float(os.environ.get("EXAMPLES_EXCLUDE_NEAR", "0") or 0)


def _norm(s: str) -> str:
    return "".join(str(s).split()).lower()


def _grams(s: str) -> Counter:
    n = _norm(s)
    return Counter(n[i:i + 2] for i in range(len(n) - 1)) or Counter([n])


class ExampleStore:
    def __init__(self, path: str = DATA_PATH):
        self.rows: list[dict] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.rows = json.load(f)
        # 브랜드명(리템, 시스맥스 등)처럼 여러 상품에 흔히 나오는 글자 조합은 가중치를 낮추고(IDF),
        # 슈즈/홀더처럼 상품을 구별하는 조합에 가중치를 준다
        grams = [_grams(r["original"]) for r in self.rows]
        df = Counter(g for gs in grams for g in gs)
        n = len(grams) or 1
        self._idf = {g: math.log(n / c) for g, c in df.items()}
        self._default_idf = math.log(n)
        self._vecs = [self._vector(g) for g in grams]

    def _vector(self, grams: Counter) -> dict:
        vec = {g: c * self._idf.get(g, self._default_idf) for g, c in grams.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {g: v / norm for g, v in vec.items()}

    def similar(self, product_name: str, k: int = 8, min_score: float = 0.2) -> list[dict]:
        """글자 2-gram TF-IDF 코사인 유사도로 비슷한 원본 상품명을 가진 사례 k개"""
        if not self.rows:
            return []
        q = self._vector(_grams(product_name))
        qn = _norm(product_name)
        scored = []
        for row, v in zip(self.rows, self._vecs):
            if EXCLUDE_EXACT and _norm(row["original"]) == qn:
                continue
            score = sum(w * v[g] for g, w in q.items() if g in v)
            if score and not (EXCLUDE_NEAR and score >= EXCLUDE_NEAR):
                scored.append((score, row))
        scored.sort(key=lambda x: -x[0])
        return [r for s, r in scored[:k] if s >= min_score]

    def context(self, product_name: str, k: int = 8, max_keywords: int = 8) -> str:
        rows = self.similar(product_name, k)
        if not rows:
            return ""
        lines = [
            f"- {r['original']} -> {r['changed']} | 키워드: {','.join(r['keywords'][:max_keywords])}"
            for r in rows
        ]
        return "\n비슷한 상품을 사람이 실제로 수정한 사례(원본 -> 수정 | 뽑은 키워드):\n" + "\n".join(lines)


_store: ExampleStore | None = None


def get_store() -> ExampleStore:
    global _store
    if _store is None:
        _store = ExampleStore()
    return _store
