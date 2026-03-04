from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

from .config import ServiceSettings, get_settings
from .models import BenchmarkPayload, ReportPayload, TaskState, TaskStatus

logger = logging.getLogger(__name__)

# Default TTL for completed/failed tasks: 1 hour
_COMPLETED_TASK_TTL = dt.timedelta(hours=1)


class TaskQueue:
    def __init__(self, settings: Optional[ServiceSettings] = None):
        self.settings = settings or get_settings()
        self._queue: Deque[str] = deque()
        self._lock = threading.Lock()
        self._items: Dict[str, TaskState] = {}
        self._cache_dir = Path(self.settings.cache_dir)
        if self.settings.use_disk_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _dump_state(self, state: TaskState) -> None:
        if not self.settings.use_disk_cache:
            return
        path = self._cache_dir / f"{state.task_id}.json"
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def _load_state(self, task_id: str) -> Optional[TaskState]:
        if task_id in self._items:
            return self._items[task_id]
        if not self.settings.use_disk_cache:
            return None
        path = self._cache_dir / f"{task_id}.json"
        if not path.exists():
            return None
        return TaskState.model_validate_json(path.read_text("utf-8"))

    def _purge_expired(self) -> None:
        """Remove completed/failed tasks older than TTL to prevent memory leak."""
        now = dt.datetime.utcnow()
        expired = [
            tid for tid, state in self._items.items()
            if state.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
            and state.completed_at is not None
            and (now - state.completed_at) > _COMPLETED_TASK_TTL
        ]
        for tid in expired:
            del self._items[tid]
        if expired:
            logger.debug("Purged %d expired tasks", len(expired))

    def submit(self, metadata: Optional[Dict[str, str]] = None) -> TaskState:
        with self._lock:
            self._purge_expired()
            if len(self._queue) >= self.settings.max_queue_size:
                raise RuntimeError("Queue is full")
            task_id = str(uuid.uuid4())
            state = TaskState(
                task_id=task_id,
                status=TaskStatus.PENDING,
                submitted_at=dt.datetime.utcnow(),
                metadata=metadata or {},
            )
            self._queue.append(task_id)
            self._items[task_id] = state
            self._dump_state(state)
        return state

    def fetch_next(self) -> Optional[TaskState]:
        with self._lock:
            if not self._queue:
                return None
            task_id = self._queue.popleft()
            state = self._items[task_id]
            state.status = TaskStatus.PROCESSING
            state.started_at = dt.datetime.utcnow()
            self._dump_state(state)
            return state

    def set_completed(self, task_id: str, report: Optional[ReportPayload] = None, benchmark: Optional[BenchmarkPayload] = None) -> TaskState:
        with self._lock:
            state = self._items.get(task_id) or self._load_state(task_id)
            if state is None:
                raise KeyError(f"Task {task_id} not found")
            state.status = TaskStatus.COMPLETED
            state.completed_at = dt.datetime.utcnow()
            state.report = report
            state.benchmark = benchmark
            self._items[task_id] = state
            self._dump_state(state)
            return state

    def set_failed(self, task_id: str, error: str) -> TaskState:
        with self._lock:
            state = self._items.get(task_id) or self._load_state(task_id)
            if state is None:
                raise KeyError(f"Task {task_id} not found")
            state.status = TaskStatus.FAILED
            state.completed_at = dt.datetime.utcnow()
            state.error = error
            self._items[task_id] = state
            self._dump_state(state)
            return state

    def update_metadata(self, task_id: str, updates: Dict[str, str]) -> None:
        """Safely update task metadata via public interface."""
        with self._lock:
            state = self._items.get(task_id)
            if state is None:
                raise KeyError(f"Task {task_id} not found")
            state.metadata.update(updates)
            self._dump_state(state)

    def get(self, task_id: str) -> Optional[TaskState]:
        with self._lock:
            state = self._items.get(task_id)
        if state:
            return state
        return self._load_state(task_id)
