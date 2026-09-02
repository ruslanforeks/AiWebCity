from __future__ import annotations

import io
import re
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from .main import (
    STATIC_DIR, MAX_UPLOAD_MB, TIMEWEB_TOKEN, analyze_photo, geocode_address,
    extract_year, norm_text, wikimedia_search, pastvu_search,
    identity_summary, build_pastphoto_link, dedupe_results,
)
from .reverse_search import search as reverse_image_search

app = FastAPI(title="AiWebCity", version="0.8.0-picimage")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index_pic.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "token_configured": bool(TIMEWEB_TOKEN),
        "reverse_search": "PicImageSearch",
        "reverse_search_engines": ["Yandex Images", "Google Lens"],
        "city_scope": "Новороссийск",
        "photo_persistence": False,
    }


def _is_novorossiysk_geo(geo: dict[str, Any] | None) -> bool:
    if not geo:
        return True  # Do not block the service when geocoding is temporarily unavailable.
    text = norm_text(geo.get("display_name")).lower().replace("ё", "е")
    return "новороссийск" in text or "novorossiysk" in text


def _address_is_novorossiysk(address: str) -> bool:
    text = re.sub(r"\s+", " ", address.lower().replace("ё", "е")).strip()
    return "новороссийск" in text or "novorossiysk" in text


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
    if not _address_is_novorossiysk(address):
        raise HTTPException(400, "AiWebCity сейчас работает только по Новороссийску. Укажите адрес в Новороссийске.")

    # Privacy: the uploaded photo stays in memory and is never written to disk.
    analysis = await analyze_photo(raw, content_type, address, year.strip())
    geo = await geocode_address(address)
    if not _is_novorossiysk_geo(geo):
        raise HTTPException(400, "Указанный адрес не относится к Новороссийску. Проект сейчас ограничен Новороссийском.")

    raw_queries = analysis.get("search_queries") if isinstance(analysis.get("search_queries"), list) else []
    queries = [norm_text(q) for q in raw_queries if norm_text(q)]
    candidates = analysis.get("identity_candidates") if isinstance(analysis.get("identity_candidates"), list) else []
    candidate_names = [
        norm_text(item.get("name"))
        for item in candidates
        if isinstance(item, dict) and norm_text(item.get("name"))
    ]
    queries = list(dict.fromkeys([f"{name} Новороссийск" for name in candidate_names] + queries))[:6]

    reverse_result = await reverse_image_search(raw, address=address)
    reverse_items = reverse_result["results"]

    # Wikimedia is also city-scoped. Search terms always include Novorossiysk and the address.
    city_queries = list(dict.fromkeys([
        f"Новороссийск {address}",
        *[f"Новороссийск {q}" for q in queries],
    ]))[:6]
    open_images = await wikimedia_search(city_queries, limit=12)
    similar = dedupe_results(reverse_items + open_images, 12)

    historical: list[dict[str, Any]] = []
    if geo:
        requested_year = extract_year(year)
        historical.extend(await pastvu_search(geo["lat"], geo["lon"], requested_year or 1999))
    for item in reverse_items:
        if isinstance(item.get("year"), int) and item["year"] <= 2000:
            historical.append({**item, "kind": "historical"})

    engine_status = [
        {"name": e["engine"], "ok": e["ok"], "results": len(e["results"]), "search_url": e.get("search_url"), "error": e.get("error")}
        for e in reverse_result["engines"]
    ]

    return {
        "status": "completed",
        "city": "Новороссийск",
        "address": address,
        "year": year.strip(),
        "geocode": geo,
        "identification": identity_summary(analysis),
        "analysis": analysis,
        "reverse_image_search": {"provider": "PicImageSearch", "engines": engine_status, "results": reverse_items},
        "similar_images": similar,
        "historical_images": dedupe_results(historical, 24),
        "sources": [
            {"name": "PastPhoto", "url": build_pastphoto_link(address), "description": "Исторические фотографии по месту."},
            {"name": "PastVu", "url": "https://pastvu.com/", "description": "Исторические фотографии рядом с координатами."},
            {"name": "Wikimedia Commons", "url": "https://commons.wikimedia.org/", "description": "Открытая база изображений."},
            {"name": "OpenStreetMap / Nominatim", "url": "https://www.openstreetmap.org/", "description": "Геокодирование адреса."},
        ],
        "privacy": {
            "server_storage": False,
            "note": "AiWebCity не сохраняет загруженное фото на VPS. Для reverse image search оно временно передаётся Yandex и Google через PicImageSearch; правила обработки этих внешних сервисов применяются отдельно.",
        },
        "generation": {"enabled": False},
    }
