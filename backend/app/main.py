import asyncio
import os
import secrets
import time
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


# 엑셀 일괄 처리는 백그라운드 작업으로 돌린다.
# Render 앞단의 Cloudflare가 100초 안에 응답이 없으면 연결을 끊어서(524), 100건(약 15분)을 한 요청으로 기다리면 결과가 버려진다.
# 요청은 작업 번호만 바로 돌려주고, 화면이 몇 초마다 진행 상황과 새로 끝난 결과를 받아간다.
JOBS: dict[str, dict] = {}
_job_tasks: set[asyncio.Task] = set()  # 작업이 도중에 가비지 컬렉션되지 않도록 참조 유지
JOB_TTL_SEC = 6 * 3600
MAX_CONSECUTIVE_ERRORS = 5  # API 한도 초과처럼 계속 실패하면 나머지는 돌리지 않고 멈춘다


async def _run_bulk_job(job: dict, optimizer: ProductOptimizer, names: list[str], provider: str) -> None:
    consecutive_errors = 0
    for i, name in enumerate(names):
        # Gemini 무료 티어의 분당 요청 한도를 넘지 않도록 요청 간 텀을 둔다. (Claude는 한도가 넉넉해 텀 없이 처리)
        if i > 0 and provider == "gemini":
            await asyncio.sleep(2.0)
        try:
            job["results"].append(await optimizer.optimize(name))
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            job["results"].append({"original_name": name, "optimized_name": f"오류: {e}", "optimized_name_pc": 0, "optimized_name_mobile": 0, "keywords": []})
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                job["status"] = "failed"
                job["error"] = f"{MAX_CONSECUTIVE_ERRORS}건 연속 실패로 중단했습니다 ({len(job['results'])}/{job['total']}건까지 처리). 마지막 오류: {e}"
                return
    job["status"] = "done"


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

    now = time.time()
    for old_id in [k for k, j in JOBS.items() if now - j["created"] > JOB_TTL_SEC]:
        del JOBS[old_id]

    job_id = secrets.token_hex(8)
    job = {"status": "running", "total": len(names), "results": [], "error": None, "created": now}
    JOBS[job_id] = job
    async def run_safely():
        try:
            await _run_bulk_job(job, optimizer, names, provider)
        except Exception as e:  # 예상 못한 오류로 작업이 '처리 중'에 영원히 멈춰 보이지 않도록
            job["status"] = "failed"
            job["error"] = f"처리 중 오류로 중단했습니다 ({len(job['results'])}/{job['total']}건까지 처리): {e}"

    task = asyncio.create_task(run_safely())
    _job_tasks.add(task)
    task.add_done_callback(_job_tasks.discard)
    return {"job_id": job_id, "total": len(names)}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, since: int = 0):
    """진행 상황 + since번째 이후 새로 끝난 결과만 돌려준다 (매번 전체를 보내지 않도록)"""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다. 서버가 재시작되어 작업 기록이 사라졌을 수 있습니다.")
    return {
        "status": job["status"],
        "total": job["total"],
        "done": len(job["results"]),
        "error": job["error"],
        "results": job["results"][since:],
    }


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
