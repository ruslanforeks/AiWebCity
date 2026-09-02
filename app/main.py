from __future__ import annotations

import base64
import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
STATIC_DIR = BASE_DIR / "static"
for directory in (RESULTS_DIR, STATIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TIMEWEB_API_BASE = os.getenv("TIMEWEB_API_BASE", "https://api.timeweb.ai/v1").rstrip("/")
TIMEWEB_TOKEN = os.getenv("TIMEWEB_AI_TOKEN", "").strip()
VISION_MODEL = os.getenv("TIMEWEB_VISION_MODEL", "openai/gpt-4.1-mini")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "").strip()
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "").strip()
YANDEX_IMAGE_URL = "https://searchapi.api.cloud.yandex.net/v2/image/search_by_image"
PASTVU_API_URL = "https://api.pastvu.com/api2"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PASTPHOTO_BASE = "https://pastphoto.ru/place/"

app = FastAPI(title="AiWebCity", version="0.4.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


def require_token() -> None:
    if not TIMEWEB_TOKEN:
        raise HTTPException(503, "TIMEWEB_AI_TOKEN не настроен на сервере.")


def data_url(image_bytes: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "output_text"):
                    if item.get(key):
                        parts.append(str(item[key]))
        return "".join(parts)
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
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
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
Ты — модуль компьютерного зрения городского проекта AiWebCity для Новороссийска.
Твоя задача на этом этапе — НЕ реконструировать историю, а помочь точно понять, что изображено на фотографии.

Адрес от пользователя: {address}
Запрошенный период: {year or 'не указан'}

Правила:
1. Описывай только то, что видно на фотографии, и отдельно то, что можно предположить по контексту адреса.
2. Нельзя придумывать исторические факты, даты строительства, архитектора, старое название или происхождение здания.
3. Сформируй несколько кандидатов на идентичность объекта только при наличии оснований.
4. Поисковые запросы должны быть практичными: название объекта + Новороссийск, адрес, улица, ориентир.
5. Отдельно укажи визуальные признаки, которые помогут сопоставить фотографию с другими снимками здания.
6. Если точная идентификация невозможна, честно укажи это.

Верни строго JSON:
{
  "what_is_visible": "краткое описание",
  "object_type": "жилой дом / административное здание / вокзал / храм / промышленный объект / другое",
  "identity_candidates": [
    {"name": "кандидат", "reason": "почему", "confidence": "high|medium|low"}
  ],
  "visual_fingerprint": ["форма фасада", "этажность", "характерные окна", "углы/выступы", "материалы", "вывески"],
  "search_queries": ["до 6 запросов для поиска фотографий и страниц"],
  "historical_claims": [],
  "limits": "что по этой фотографии установить нельзя"
}
"""
    body = await timeweb_chat([
        {"role": "system", "content": "Отвечай на русском. Это модуль идентификации, а не генерации истории."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(image_bytes, content_type)}},
        ]},
    ], VISION_MODEL)
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    return extract_json(extract_text(content))


async def geocode_address(address: str) -> dict[str, Any] | None:
    headers = {"User-Agent": "AiWebCity/0.4 (+https://aiweb.su/)"}
    params = {"q": address, "format": "jsonv2", "limit": 1, "addressdetails": 1}
    try:
        async with httpx.AsyncClient(timeout=15, headers=headers) as client:
            response = await client.get(NOMINATIM_URL, params=params)
        if response.status_code >= 400:
            return None
        rows = response.json()
        if not rows:
            return None
        row = rows[0]
        return {"lat": float(row["lat"]), "lon": float(row["lon"]), "display_name": row.get("display_name", "")}
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_year(text: str) -> int | None:
    years = [int(value) for value in re.findall(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)", text)]
    return min(years) if years else None


async def yandex_image_search(image_bytes: bytes) -> list[dict[str, Any]]:
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        return []
    headers = {"Api-Key": YANDEX_API_KEY, "Content-Type": "application/json"}
    payload = {"folderId": YANDEX_FOLDER_ID, "data": base64.b64encode(image_bytes).decode("ascii"), "page": "0", "familyMode": "MODERATE"}
    try:
        async with httpx.AsyncClient(timeout=40) as client:
            response = await client.post(YANDEX_IMAGE_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            return []
        body = response.json()
        results: list[dict[str, Any]] = []
        for item in body.get("images", [])[:12]:
            if not isinstance(item, dict):
                continue
            image_url = norm_text(item.get("url"))
            page_url = norm_text(item.get("pageUrl"))
            if not image_url:
                continue
            title = norm_text(item.get("pageTitle")) or "Результат Яндекс Картинок"
            passage = norm_text(item.get("passage"))
            results.append({
                "image_url": image_url,
                "page_url": page_url,
                "title": title,
                "description": passage,
                "source": "Яндекс Картинки",
                "kind": "similar",
                "year": extract_year(" ".join([title, passage, page_url])),
            })
        return results
    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        return []


async def pastvu_search(lat: float, lon: float, year_to: int = 1999) -> list[dict[str, Any]]:
    params_obj: dict[str, Any] = {"geo": [lat, lon], "distance": 1500, "year2": year_to, "limit": 30}
    params = {"method": "photo.giveNearestPhotos", "params": json.dumps(params_obj, ensure_ascii=False, separators=(",", ":"))}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(PASTVU_API_URL, params=params)
        if response.status_code >= 400:
            return []
        body = response.json()
        rows = body.get("result", {}).get("photo", [])
        results: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            file_id = norm_text(item.get("file"))
            if not file_id:
                continue
            cid = item.get("cid")
            page_url = f"https://pastvu.com/p/{cid}" if cid else "https://pastvu.com/"
            try:
                year_value = int(item.get("year")) if item.get("year") is not None else None
            except (TypeError, ValueError):
                year_value = None
            results.append({
                "image_url": f"https://img.pastvu.com/d/{file_id}",
                "page_url": page_url,
                "title": norm_text(item.get("title")) or "Историческая фотография",
                "description": norm_text(item.get("desc") or item.get("description")),
                "source": "PastVu",
                "kind": "historical",
                "year": year_value,
                "distance_m": item.get("distance"),
            })
        return results
    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        return []


async def wikimedia_search(queries: list[str], limit: int = 12) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "AiWebCity/0.4"}) as client:
        for query in queries[:4]:
            params = {
                "action": "query", "generator": "search", "gsrnamespace": 6,
                "gsrsearch": query, "gsrlimit": 8, "prop": "imageinfo",
                "iiprop": "url|extmetadata", "iiurlwidth": 900,
                "format": "json", "formatversion": 2,
            }
            try:
                response = await client.get(WIKIMEDIA_API_URL, params=params)
                if response.status_code >= 400:
                    continue
                body = response.json()
                for page in body.get("query", {}).get("pages", []):
                    infos = page.get("imageinfo") or []
                    if not infos:
                        continue
                    info = infos[0]
                    image_url = norm_text(info.get("thumburl") or info.get("url"))
                    page_url = norm_text(page.get("canonicalurl"))
                    if not image_url:
                        continue
                    ext = info.get("extmetadata") or {}
                    title = norm_text(page.get("title"))
                    description = norm_text((ext.get("ImageDescription") or {}).get("value"))
                    key = page_url or image_url
                    merged[key] = {
                        "image_url": image_url,
                        "page_url": page_url or "https://commons.wikimedia.org/",
                        "title": title or "Wikimedia Commons",
                        "description": re.sub(r"<[^>]+>", " ", description),
                        "source": "Wikimedia Commons",
                        "kind": "similar",
                        "year": extract_year(" ".join([title, description])),
                    }
            except (httpx.HTTPError, ValueError, json.JSONDecodeError):
                continue
    return list(merged.values())[:limit]


def build_pastphoto_link(address: str) -> str:
    path = quote(address.replace(", Россия", "").strip(), safe="")
    return f"{PASTPHOTO_BASE}{path}"


def dedupe_results(items: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = norm_text(item.get("image_url")) or norm_text(item.get("page_url"))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def identity_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    candidates = analysis.get("identity_candidates") if isinstance(analysis.get("identity_candidates"), list) else []
    cleaned = []
    for item in candidates[:5]:
        if not isinstance(item, dict):
            continue
        confidence = str(item.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        cleaned.append({"name": norm_text(item.get("name")), "reason": norm_text(item.get("reason")), "confidence": confidence})
    return {"candidates": cleaned}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "token_configured": bool(TIMEWEB_TOKEN), "yandex_configured": bool(YANDEX_API_KEY and YANDEX_FOLDER_ID), "vision_model": VISION_MODEL}


@app.post("/api/identify")
async def identify(photo: UploadFile = File(...), address: str = Form(...), year: str = Form("")) -> dict[str, Any]:
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
    modern_path = RESULTS_DIR / f"{request_id}_modern{extension}"
    modern_path.write_bytes(raw)

    analysis = await analyze_photo(raw, content_type, address, year.strip())
    geo = await geocode_address(address)

    queries = analysis.get("search_queries") if isinstance(analysis.get("search_queries"), list) else []
    queries = [norm_text(q) for q in queries if norm_text(q)]
    candidates = analysis.get("identity_candidates") if isinstance(analysis.get("identity_candidates"), list) else []
    candidate_names = [norm_text(item.get("name")) for item in candidates if isinstance(item, dict) and norm_text(item.get("name"))]
    queries = list(dict.fromkeys([f"{name} Новороссийск" for name in candidate_names] + queries))[:6]

    similar = await yandex_image_search(raw)
    similar.extend(await wikimedia_search(queries or [f"Новороссийск {address}"], limit=12))

    historical: list[dict[str, Any]] = []
    if geo:
        requested_year = extract_year(year)
        historical.extend(await pastvu_search(geo["lat"], geo["lon"], requested_year or 1999))

    yandex_historical = [item for item in similar if isinstance(item.get("year"), int) and item["year"] <= 2000]
    historical.extend(yandex_historical)
    similar = [item for item in similar if item not in yandex_historical]

    sources = [
        {"name": "PastPhoto", "url": build_pastphoto_link(address), "description": "Открытый интерактивный фотоархив исторических снимков по месту."},
        {"name": "PastVu", "url": "https://pastvu.com/", "description": "Исторические фотографии в радиусе от найденных координат."},
        {"name": "Wikimedia Commons", "url": "https://commons.wikimedia.org/", "description": "Открытая база изображений и категорий."},
        {"name": "Яндекс Картинки", "url": "https://yandex.ru/images/", "description": "Обратный поиск по исходной фотографии." if YANDEX_API_KEY and YANDEX_FOLDER_ID else "Не активирован: не заданы YANDEX_API_KEY и/или YANDEX_FOLDER_ID."},
    ]

    return {
        "request_id": request_id,
        "status": "completed",
        "modern_photo_url": f"/results/{modern_path.name}",
        "address": address,
        "year": year.strip(),
        "geocode": geo,
        "identification": identity_summary(analysis),
        "analysis": analysis,
        "similar_images": dedupe_results(similar, 18),
        "historical_images": dedupe_results(historical, 18),
        "sources": sources,
        "generation": {"enabled": False, "reason": "Генерация изображений отключена на этапе идентификации и поиска источников."},
    }
