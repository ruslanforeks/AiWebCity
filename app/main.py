from __future__ import annotations

import base64
import io
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

TIMEWEB_API_BASE = os.getenv("TIMEWEB_API_BASE", "https://api.timeweb.ai/v1").rstrip("/")
TIMEWEB_TOKEN = os.getenv("TIMEWEB_AI_TOKEN", "").strip()
VISION_MODEL = os.getenv("TIMEWEB_VISION_MODEL", "openai/gpt-4.1-mini")
IMAGE_MODEL = os.getenv("TIMEWEB_IMAGE_MODEL", "openai/gpt-image-2")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))
MAX_ARCHIVE_IMAGES = int(os.getenv("MAX_ARCHIVE_IMAGES", "4"))

app = FastAPI(title="AiWebCity", version="0.3.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/archive", StaticFiles(directory=str(ARCHIVE_DIR)), name="archive")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def require_token() -> None:
    if not TIMEWEB_TOKEN:
        raise HTTPException(503, "TIMEWEB_AI_TOKEN не настроен на сервере.")


def data_url(image_bytes: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def media_type(path: Path) -> str:
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                for key in ("text", "output_text"):
                    if item.get(key):
                        pieces.append(str(item[key]))
        return "".join(pieces)
    if isinstance(content, dict):
        return str(content.get("text", content.get("output_text", "")))
    return ""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"raw": text}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {"raw": text}
            except json.JSONDecodeError:
                pass
    return {"raw": text}


async def timeweb_chat(messages: list[dict[str, Any]], model: str, *, temperature: float = 0.1) -> dict[str, Any]:
    require_token()
    headers = {"Authorization": f"Bearer {TIMEWEB_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    async with httpx.AsyncClient(timeout=240) as client:
        response = await client.post(f"{TIMEWEB_API_BASE}/chat/completions", headers=headers, json=payload)
    if response.status_code >= 400:
        raise HTTPException(502, f"Timeweb AI error {response.status_code}: {response.text[:800]}")
    return response.json()


async def analyze_photo(image_bytes: bytes, content_type: str, address: str, year: str) -> dict[str, Any]:
    prompt = f"""
Ты — исторический исследователь бесплатного городского проекта AiWebCity о Новороссийске.
Пользователь сфотографировал современное здание.
Адрес: {address}
Нужный год/период: {year or 'ближайший подтвержденный период'}

Проанализируй ТОЛЬКО то, что видно на современной фотографии. Нельзя выдавать неизвестные исторические детали за факты.
Верни строго JSON следующей формы:
{{
  "building_description": "краткое описание объекта",
  "visible_features": ["этажи", "окна", "двери", "материалы", "декор"],
  "historical_hypotheses": ["только гипотезы"],
  "reconstruction_prompt": "инструкция для генератора, которая сохраняет современную геометрию здания и меняет только подтвержденные исторические признаки",
  "confidence_without_archive": 0,
  "warning": "почему архивные источники обязательны"
}}
"""
    body = await timeweb_chat([
        {"role": "system", "content": "Отвечай на русском. Не выдумывай исторические факты."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(image_bytes, content_type)}},
        ]},
    ], VISION_MODEL)
    choices = body.get("choices") or []
    if not choices:
        raise HTTPException(502, f"Timeweb AI вернул ответ без choices: {body}")
    content = choices[0].get("message", {}).get("content", "")
    return extract_json(extract_text(content))


def load_archive_manifest() -> list[dict[str, Any]]:
    manifest_path = DATA_DIR / "archive.json"
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def safe_archive_path(relative: str) -> Path | None:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    candidate = (ARCHIVE_DIR / path).resolve()
    try:
        candidate.relative_to(ARCHIVE_DIR.resolve())
    except ValueError:
        return None
    if candidate.suffix.lower() not in ALLOWED_EXTENSIONS or not candidate.is_file():
        return None
    return candidate


def find_archive_candidates(address: str, year: str, limit: int = 8) -> list[dict[str, Any]]:
    manifest = load_archive_manifest()
    tokens = [t.lower() for t in re.findall(r"[а-яёa-z0-9]+", f"{address} {year}") if len(t) > 2]
    rows: list[tuple[int, dict[str, Any]]] = []
    for item in manifest:
        relative = str(item.get("path", ""))
        path = safe_archive_path(relative)
        if path is None:
            continue
        haystack = " ".join(str(item.get(key, "")) for key in ("path", "address", "year", "description", "source")).lower()
        score = sum(3 for token in tokens if token in haystack)
        item_copy = dict(item)
        item_copy["path"] = relative
        item_copy["score"] = score
        rows.append((score, item_copy))

    # Even when metadata has no exact match, return a small deterministic sample.
    if not rows:
        for path in sorted(ARCHIVE_DIR.rglob("*")):
            if path.suffix.lower() in ALLOWED_EXTENSIONS and path.is_file():
                rows.append((0, {"path": str(path.relative_to(ARCHIVE_DIR)), "score": 0}))

    rows.sort(key=lambda x: (-x[0], str(x[1].get("path", ""))))
    return [item for _, item in rows[:limit]]


def evidence_payload(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for candidate in candidates[:MAX_ARCHIVE_IMAGES]:
        path = safe_archive_path(str(candidate["path"]))
        if path is None:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        evidence.append({"meta": candidate, "image": data_url(raw, media_type(path))})
    return evidence


def parse_generated_image(body: dict[str, Any]) -> bytes | None:
    def decode_value(value: Any) -> bytes | None:
        if isinstance(value, dict):
            raw = value.get("b64_json") or value.get("base64") or value.get("data")
            if isinstance(raw, str):
                if raw.startswith("data:image"):
                    raw = raw.split(",", 1)[1]
                try:
                    return base64.b64decode(raw)
                except Exception:
                    return None
        if isinstance(value, str) and value.startswith("data:image"):
            try:
                return base64.b64decode(value.split(",", 1)[1])
            except Exception:
                return None
        return None

    for choice in body.get("choices") or []:
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        for key in ("images", "image"):
            value = message.get(key)
            values = value if isinstance(value, list) else [value]
            for item in values:
                decoded = decode_value(item)
                if decoded:
                    return decoded
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            if isinstance(part, dict):
                if part.get("type") in {"image", "image_url", "output_image"}:
                    for key in ("image", "image_url", "data"):
                        decoded = decode_value(part.get(key))
                        if decoded:
                            return decoded
    return None


async def generate_reconstruction(modern_bytes: bytes, modern_type: str, address: str, year: str, analysis: dict[str, Any], candidates: list[dict[str, Any]]) -> bytes:
    evidence = evidence_payload(candidates)
    if not evidence:
        raise HTTPException(422, "Нет читаемых архивных изображений. Реконструкция заблокирована до появления источников.")

    content: list[dict[str, Any]] = [{"type": "text", "text": f"""
Создай реалистичную историческую реконструкцию здания в Новороссийске.
Адрес: {address}
Период: {year or 'ближайший подтвержденный период'}

Сначала рассмотрите современное фото как геометрический reference. Сохрани ракурс, перспективу, этажность, общую форму здания, положение окон и дверей.
Далее используй архивные фотографии как доказательства исторического состояния. Меняй только признаки, которые действительно подтверждаются источниками.
Не придумывай башни, этажи, окна, двери, декор, вывески или другие элементы. Если признак не подтвержден, не добавляй его.
Не делай изображение в стиле картины или AI-art: результат должен выглядеть как настоящая фотография соответствующего исторического времени.

Анализ современного фото:
{json.dumps(analysis, ensure_ascii=False)}
"""}, {"type": "image_url", "image_url": {"url": data_url(modern_bytes, modern_type)}}]

    for item in evidence:
        meta = item["meta"]
        content.append({"type": "text", "text": f"АРХИВНЫЙ ИСТОЧНИК: год={meta.get('year', 'не указан')}; адрес={meta.get('address', '')}; описание={meta.get('description', '')}; источник={meta.get('source', '')}"})
        content.append({"type": "image_url", "image_url": {"url": item["image"]}})

    body = await timeweb_chat([{ "role": "user", "content": content }], IMAGE_MODEL, temperature=0.3)
    image_bytes = parse_generated_image(body)
    if image_bytes:
        return image_bytes
    raise HTTPException(502, "Модель генерации не вернула изображение. Укажите точный image-model ID из Timeweb AI Gateway в TIMEWEB_IMAGE_MODEL.")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "token_configured": bool(TIMEWEB_TOKEN),
        "archive_images": sum(1 for p in ARCHIVE_DIR.rglob('*') if p.suffix.lower() in ALLOWED_EXTENSIONS),
        "archive_manifest": len(load_archive_manifest()),
        "vision_model": VISION_MODEL,
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
        Image.open(io.BytesIO(raw)).verify()
    except Exception as exc:
        raise HTTPException(400, "Не удалось прочитать изображение.") from exc
    address = address.strip()
    if not address:
        raise HTTPException(400, "Укажите адрес здания.")

    request_id = uuid.uuid4().hex
    extension = Path(photo.filename or "photo.jpg").suffix.lower() or ".jpg"
    if extension not in ALLOWED_EXTENSIONS:
        extension = ".jpg"
    modern_path = RESULTS_DIR / f"{request_id}_modern{extension}"
    modern_path.write_bytes(raw)

    analysis = await analyze_photo(raw, content_type, address, year.strip())
    candidates = find_archive_candidates(address, year.strip())
    if not candidates:
        return {
            "request_id": request_id,
            "status": "insufficient_evidence",
            "message": "В архиве пока нет изображений. Реконструкция не создаётся, потому что сервис не должен выдавать выдумку за историю.",
            "analysis": analysis,
            "archive_candidates": [],
            "modern_photo_url": f"/results/{modern_path.name}",
        }

    image_bytes = await generate_reconstruction(raw, content_type, address, year.strip(), analysis, candidates)
    result_path = RESULTS_DIR / f"{request_id}_reconstruction.png"
    result_path.write_bytes(image_bytes)

    return {
        "request_id": request_id,
        "status": "completed",
        "reconstruction_url": f"/results/{result_path.name}",
        "modern_photo_url": f"/results/{modern_path.name}",
        "analysis": analysis,
        "archive_candidates": candidates,
        "message": "Готово. Архивные источники использованы как evidence для реконструкции.",
    }
