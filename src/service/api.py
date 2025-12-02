from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import fitz  # PyMuPDF

from .config import ServiceSettings, get_settings
from .models import BenchmarkRequest, CheckAcceptedResponse, CheckRequest, TaskStatus, TaskStatusResponse
from .tasks import TaskQueue
from .worker import start_workers

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Copyless Plagiarism Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
)

templates_dir = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def get_task_queue(settings: ServiceSettings = Depends(get_settings)) -> TaskQueue:
    if not hasattr(app.state, "task_queue"):
        app.state.task_queue = TaskQueue(settings)
    return app.state.task_queue


@app.on_event("startup")
async def startup_event():
    settings = get_settings()
    queue = TaskQueue(settings)
    app.state.task_queue = queue
    app.state.worker_task = asyncio.create_task(start_workers(queue, settings))


@app.on_event("shutdown")
async def shutdown_event():
    worker_task: Optional[asyncio.Task] = getattr(app.state, "worker_task", None)
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


@app.post("/v1/papers/check", response_model=CheckAcceptedResponse, status_code=202)
async def submit_paper(req: CheckRequest, queue: TaskQueue = Depends(get_task_queue)):
    if not req.content:
        raise HTTPException(status_code=400, detail="content is required")

    try:
        state = queue.submit(metadata=req.metadata)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    queue._items[state.task_id].metadata["callback_url"] = req.callback_url or ""
    queue._items[state.task_id].metadata["content_length"] = str(len(req.content))
    queue._items[state.task_id].metadata["content"] = req.content

    return CheckAcceptedResponse(task_id=state.task_id)


@app.post("/v1/papers/upload_pdf", response_model=CheckAcceptedResponse, status_code=202)
async def upload_pdf(file: UploadFile = File(...), queue: TaskQueue = Depends(get_task_queue)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    try:
        state = queue.submit(metadata={"filename": file.filename})
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    queue._items[state.task_id].metadata["content_length"] = str(len(text))
    queue._items[state.task_id].metadata["content"] = text

    return CheckAcceptedResponse(task_id=state.task_id)


@app.post("/v1/benchmarks/run", response_model=CheckAcceptedResponse, status_code=202)
async def run_benchmark(req: BenchmarkRequest, queue: TaskQueue = Depends(get_task_queue)):
    data_path = Path(req.data_path).expanduser()
    if not data_path.exists():
        raise HTTPException(status_code=400, detail="data_path does not exist")

    metadata = {
        "task_type": "benchmark",
        "benchmark_kind": req.kind.value,
        "data_path": str(data_path.resolve()),
    }

    config = {}
    for field in [
        "model",
        "batch_size",
        "threshold",
        "backend",
        "qdrant_url",
        "qdrant_api_key",
        "collection",
        "doc_min_pairs",
        "doc_min_ratio",
        "show_progress",
    ]:
        value = getattr(req, field)
        if value is not None:
            config[field] = value

    if "show_progress" not in config:
        config["show_progress"] = False

    if config:
        metadata["benchmark_config"] = json.dumps(config)

    if req.callback_url:
        metadata["callback_url"] = req.callback_url

    try:
        state = queue.submit(metadata=metadata)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    return CheckAcceptedResponse(task_id=state.task_id)


@app.get("/v1/reports/{task_id}", response_model=TaskStatusResponse)
async def get_report(task_id: str, queue: TaskQueue = Depends(get_task_queue)):
    state = queue.get(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="task not found")

    if state.status in {TaskStatus.PENDING, TaskStatus.PROCESSING}:
        return TaskStatusResponse(
            task_id=state.task_id,
            status=state.status,
            report=None,
            error=None,
            submitted_at=state.submitted_at,
            started_at=state.started_at,
            completed_at=state.completed_at,
            benchmark=None,
        )

    if state.status == TaskStatus.FAILED:
        return TaskStatusResponse(
            task_id=state.task_id,
            status=state.status,
            report=None,
            error=state.error,
            submitted_at=state.submitted_at,
            started_at=state.started_at,
            completed_at=state.completed_at,
            benchmark=None,
        )

    return TaskStatusResponse(
        task_id=state.task_id,
        status=state.status,
        report=state.report,
        error=None,
        submitted_at=state.submitted_at,
        started_at=state.started_at,
        completed_at=state.completed_at,
        benchmark=state.benchmark,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    settings = get_settings()
    context = {
        "defaults": {
            "top_k": settings.top_k,
            "score_threshold": settings.score_threshold,
            "rerank_top_k": settings.rerank_top_k,
        }
    }
    return templates.TemplateResponse(request, "index.html", context)
