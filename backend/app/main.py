import asyncio
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.excel_utils import build_result_excel, read_product_names
from app.services.claude_optimizer import ClaudeOptimizer
from app.services.gemini_optimizer import GeminiOptimizer
from app.services.naver_keywordstool import NaverKeywordsTool
from app.services.optimizer import ProductOptimizer

app = FastAPI(title="네이버 상품명/키워드 최적화 도구")

APP_USERNAME = os.environ.get("APP_USERNAME", "admin")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")


@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    if not APP_PASSWORD:
        return await call_next(request)  # 비밀번호 미설정 시 로컬 전용으로 간주하고 통과

    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Basic "):
        import base64

        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            username, _, password = decoded.partition(":")
        except Exception:
            username, password = "", ""
        if secrets.compare_digest(username, APP_USERNAME) and secrets.compare_digest(password, APP_PASSWORD):
            return await call_next(request)

    return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})


def get_optimizer(provider: str) -> ProductOptimizer:
    if provider == "claude":
        name_generator = ClaudeOptimizer()
    elif provider == "gemini":
        name_generator = GeminiOptimizer()
    else:
        raise HTTPException(status_code=400, detail=f"알 수 없는 provider: {provider}")

    naver = NaverKeywordsTool(
        customer_id=os.environ["NAVER_CUSTOMER_ID"],
        api_key=os.environ["NAVER_API_KEY"],
        secret_key=os.environ["NAVER_SECRET_KEY"],
    )
    return ProductOptimizer(name_generator, naver)


class OptimizeRequest(BaseModel):
    product_name: str
    provider: str = "gemini"


class ExportRequest(BaseModel):
    results: list[dict]


@app.post("/api/optimize")
async def optimize_single(req: OptimizeRequest):
    try:
        optimizer = get_optimizer(req.provider)
        return await optimizer.optimize(req.product_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/bulk")
async def optimize_bulk(file: UploadFile, provider: str = Form("gemini")):
    content = await file.read()
    names = read_product_names(content)
    if not names:
        raise HTTPException(status_code=400, detail="엑셀에서 '상품명' 컬럼(또는 첫번째 컬럼)을 찾지 못했습니다.")

    try:
        optimizer = get_optimizer(provider)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Gemini 무료 티어의 분당 요청 한도를 넘지 않도록 순차 처리 + 요청 간 텀을 둔다. (Claude는 한도가 넉넉해 텀 없이 처리)
    results = []
    for i, name in enumerate(names):
        if i > 0 and provider == "gemini":
            await asyncio.sleep(2.0)
        try:
            results.append(await optimizer.optimize(name))
        except Exception as e:
            results.append({"original_name": name, "optimized_name": f"오류: {e}", "optimized_name_pc": 0, "optimized_name_mobile": 0, "keywords": []})

    return results


@app.post("/api/export/excel")
async def export_excel(req: ExportRequest):
    data = build_result_excel(req.results)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=optimized_products.xlsx"},
    )


frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
