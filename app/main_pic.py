from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import io

from .main import (
    STATIC_DIR,
    MAX_UPLOAD_MB,
    analyze_photo,
    geocode_address,
    extract_year,
    norm_text,
    wikimedia_search,
    pastvu_search,
    identity_summary,
    build_pastphoto_link,
    dedupe_results,
)
from .reverse_search import search as reverse_image_search

app = FastAPI(title="AiWebCity", version="0.7.0-picimage")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "token_configured": True,
        "reverse_search": "PicImageSearch",
        "reverse_search_engines": ["Yandex Images", "Google Lens"],
        "photo_persistence": False,
    }


@app.post("/api/identify")
async def identify(
    photo: UploadFile = File(...),
    address: str = Form(...),
    year: str = Form(""),
) -> dict:
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

    # The uploaded photo remains in memory only; it is never written to disk.
    analysis = await analyze_photo(raw, content_type, address, year.strip())
    geo = await geocode_address(address)

    raw_queries = analysis.get("search_queries") if isinstance(analysis.get("search_queries"), list) else []
    queries = [norm_text(q) for q in raw_queries if norm_text(q)]
    candidates = analysis.get("identity_candidates") if isinstance(analysis.get("identity_candidates"), list) else []
    candidate_names = [norm_text(i.get("name")) for i in candidates if isinstance(i, dict) and norm_text(i.get("name"))]
    queries = list(dict.fromkeys([f"{name} Новороссийск" for name in candidate_names] + queries))[:6]

    reverse_result = await reverse_image_search(raw)
    reverse_items = reverse_result["results"]

    similar = dedupe_results(reverse_items + await wikimedia_search(queries or [f"Новороссийск {address}"], limit=12), 30)

    historical = []
    if geo:
        requested_year = extract_year(year)
        historical.extend(await pastvu_search(geo["lat"], geo["lon"], requested_year or 1999))
    for item in reverse_items:
        if isinstance(item.get("year"), int) and item["year"] <= 2000:
            historical.append({**item, "kind": "historical"})

    engine_status = [
        {
            "name": e["engine"],
            "ok": e["ok"],
            "results": len(e["results"]),
            "search_url": e.get("search_url"),
            "error": e.get("error"),
        }
        for e in reverse_result["engines"]
    ]

    return {
        "status": "completed",
        "address": address,
        "year": year.strip(),
        "geocode": geo,
        "identification": identity_summary(analysis),
        "analysis": analysis,
        "reverse_image_search": {
            "provider": "PicImageSearch",
            "engines": engine_status,
            "results": reverse_items,
        },
        "similar_images": similar,
        "historical_images": dedupe_results(historical, 24),
        "sources": [
            {"name": "Yandex Images", "url": reverse_result["engines"][0].get("search_url") or "https://yandex.com/images/", "description": "Reverse image search через PicImageSearch."},
            {"name": "Google Lens", "url": reverse_result["engines"][1].get("search_url") or "https://lens.google.com/", "description": "Дополнительный визуальный поиск через PicImageSearch."},
            {"name": "PastPhoto", "url": build_pastphoto_link(address), "description": "Исторические фотографии по месту."},
            {"name": "PastVu", "url": "https://pastvu.com/", "description": "Исторические фотографии рядом с координатами."},
            {"name": "Wikimedia Commons", "url": "https://commons.wikimedia.org/", "description": "Открытая база изображений."},
            {"name": "OpenStreetMap / Nominatim", "url": "https://www.openstreetmap.org/", "description": "Геокодирование адреса."},
        ],
        "privacy": {
            "server_storage": False,
            "note": "AiWebCity не сохраняет загруженную фотографию на VPS. Для reverse image search фотография временно передаётся внешним поисковикам Yandex и Google через PicImageSearch. Их собственные правила обработки применяются отдельно.",
        },
        "generation": {"enabled": False},
    }
