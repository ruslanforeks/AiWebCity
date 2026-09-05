"""Фоновые задачи распознавания.

Один запрос к /api/identify идёт от 40 до 120 секунд: обратный поиск, несколько
сравнений через Vision, проверка адреса, архивные источники. Браузер столько
ждать не обязан — Safari обрывает fetch примерно на минуте и показывает
«Load failed». Именно поэтому со второй попытки всё работало: результат поиска
лежал в кэше, и запрос успевал уложиться в лимит.

Поэтому распознавание запускается фоном, а клиент опрашивает статус. Заодно
можно честно показывать, что происходит прямо сейчас.

Фотография пользователя здесь НЕ хранится: она живёт только в аргументах
работающей корутины и исчезает вместе с ней.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable

JOB_TTL_SECONDS = 900.0
MAX_JOBS = 40

_JOBS: dict[str, dict[str, Any]] = {}


def _cleanup() -> None:
    now = time.time()
    stale = [key for key, job in _JOBS.items() if now - job["updated_at"] > JOB_TTL_SECONDS]
    for key in stale:
        _JOBS.pop(key, None)
    if len(_JOBS) > MAX_JOBS:
        for key in sorted(_JOBS, key=lambda k: _JOBS[k]["updated_at"])[: len(_JOBS) - MAX_JOBS]:
            _JOBS.pop(key, None)


def create() -> str:
    _cleanup()
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {
        "status": "running",
        "stage": "Начинаем поиск",
        "result": None,
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    return job_id


def set_stage(job_id: str, stage: str) -> None:
    job = _JOBS.get(job_id)
    if job and job["status"] == "running":
        job["stage"] = stage
        job["updated_at"] = time.time()


def get(job_id: str) -> dict[str, Any] | None:
    job = _JOBS.get(job_id)
    if not job:
        return None
    return {
        "status": job["status"],
        "stage": job["stage"],
        "result": job["result"],
        "error": job["error"],
        "elapsed": round(time.time() - job["created_at"], 1),
    }


async def run(job_id: str, coroutine: Awaitable[dict[str, Any]]) -> None:
    job = _JOBS.get(job_id)
    try:
        result = await coroutine
        if job is not None:
            job["result"] = result
            job["status"] = "done"
            job["stage"] = "Готово"
    except Exception as exc:  # noqa: BLE001 — задача не должна ронять сервер
        if job is not None:
            job["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            job["status"] = "error"
            job["stage"] = "Ошибка"
    finally:
        if job is not None:
            job["updated_at"] = time.time()


def spawn(coroutine_factory: Callable[[str], Awaitable[dict[str, Any]]]) -> str:
    """Создаёт задачу и запускает её в фоне. Возвращает идентификатор."""
    job_id = create()
    asyncio.create_task(run(job_id, coroutine_factory(job_id)))
    return job_id
