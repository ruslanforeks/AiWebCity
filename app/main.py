from __future__ import annotations

import base64
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
RESULTS_DIR = DATA_DIR / "results"
STATIC_DIR = BASE_DIR / "static"
for directory in (ARCHIVE_DIR, RESULTS_DIR, STATIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TIMEWEB_API_BASE = os.getenv("TIMEWEB_API_BASE", "https://api.timeweb.ai/v1")
TIMEWEB_TOKEN = os.getenv("TIMEWEB_AI_TOKEN", "").strip()
VISION_MODEL = os.getenv("TIMEWEB_VISION_MODEL", "openai/gpt-4.1-mini")
IMAGE_MODEL = os.getenv("TIMEWEB_IMAGE_MODEL", "openai/gpt-image-2")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))

app = FastAPI(title="AiWebCity", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def require_token() -> None:
    if not TIMEWEB_TOKEN:
        raise HTTPException(503, "TIMEWEB_AI_TOKEN не настроен на сервере.")


def data_url(image_bytes: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    return {"raw": text}


async def timeweb_chat(messages: list[dict[str, Any]], model: str) -> str:
    require_token()
    headers = {"Authorization": f"Bearer {TIMEWEB_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.1}
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(f"{TIMEWEB_API_BASE}/chat/completions", headers=headers, json=payload)
    if response.status_code >= 400:
        raise HTTPException(502, f"Timeweb AI error {response.status_code}: {response.text[:600]}")
    body = response.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(502, f"Некорректный ответ Timeweb AI: {body}") from exc
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


async def analyze_photo(image_bytes: bytes, content_type: str, address: str, year: str) -> dict[str, Any]:
    prompt = f"""
Ты — исторический исследователь для бесплатного городского сервиса AiWebCity о Новороссийске.
Проанализируй современную фотографию здания.
Адрес пользователя: {address}
Запрошенный период: {year or 'ближайший подтверждённый исторический период'}

Нужно вернуть строго JSON:
{{
  "building_description": "...",
  "visible_features": ["..."],
  "historical_hypotheses": ["..."],
  "reconstruction_prompt": "...",
  "confidence_without_archive": 0-100,
  "warning": "..."
}}

Ключевое правило: НЕ выдавай исторические детали за факты. По одной современной фотографии нельзя достоверно определить старый фасад.
reconstruction_prompt должен явно требовать сохранить геометрию, число этажей, положение окон и дверей современного здания и менять только исторически обоснованные элементы.
"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(image_bytes, content_type)}},
        ],
    }]
    result = await timeweb_chat(messages, VISION_MODEL)
    return extract_json(result)


def find_archive_candidates(address: str, year: str, limit: int = 8) -> list[dict[str, Any]]:
    """Lightweight archive index for MVP.

    We intentionally avoid paid embedding calls. Files are selected by filename/path tokens.
    Later this can be replaced by local CLIP/DINO retrieval without changing the API.
    """
    tokens = [t.lower() for t in re.findall(r"[а-яёa-z0-9]+", f"{address} {year}") if len(t) > 2]
    candidates: list[tuple[int, Path]] = []
    for path in ARCHIVE_DIR.rglob("*"):
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        haystack = str(path.relative_to(ARCHIVE_DIR)).lower()
        score = sum(1 for token in tokens if token in haystack)
        candidates.append((score, path))
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    output = []
    for score, path in candidates[:limit]:
        output.append({"filename": str(path.relative_to(ARCHIVE_DIR)), "score": score})
    return output


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "token_configured": bool(TIMEWEB_TOKEN),
        "archive_images": sum(1 for p in ARCHIVE_DIR.rglob('*') if p.suffix.lower() in ALLOWED_EXTENSIONS),
        "image_model": IMAGE_MODEL,
    }


@app.post("/api/reconstruct")
async def reconstruct(
    photo: UploadFile = File(...),
    address: str = Form(...),
    year: str = Form(""),
) -> dict[str, Any]:
    require_token()
    raw = await photo.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Фото слишком большое. Максимум {MAX_UPLOAD_MB} МБ.")
    content_type = photo.content_type or "image/jpeg"
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Поддерживаются JPG, PNG и WEBP.")
    try:
        Image.open(__import__('io').BytesIO(raw)).verify()
    except Exception as exc:
        raise HTTPException(400, "Не удалось прочитать изображение.") from exc

    request_id = uuid.uuid4().hex
    modern_path = RESULTS_DIR / f"{request_id}_modern{Path(photo.filename or 'photo.jpg').suffix.lower() or '.jpg'}"
    modern_path.write_bytes(raw)

    analysis = await analyze_photo(raw, content_type, address.strip(), year.strip())
    candidates = find_archive_candidates(address, year)

    if not candidates:
        return {
            "request_id": request_id,
            "status": "insufficient_evidence",
            "message": "В архиве пока нет подходящих фотографий. Загрузка архивов обязательна, чтобы реконструкция не превращалась в выдумку.",
            "analysis": analysis,
            "archive_candidates": [],
        }

    return {
        "request_id": request_id,
        "status": "needs_archive_review",
        "message": "Найдены архивные кандидаты. В этой версии они пока не передаются автоматически в генератор, чтобы исключить ложную уверенность без проверки источников.",
        "analysis": analysis,
        "archive_candidates": candidates,
    }
