from __future__ import annotations

import io
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from .main import (
    STATIC_DIR,
    MAX_UPLOAD_MB,
    TIMEWEB_TOKEN,
    analyze_photo,
    extract_year,
    norm_text,
    wikimedia_search,
    pastvu_search,
    identity_summary,
    build_pastphoto_link,
    dedupe_results,
    resolve_identity,
)
from .reverse_search import search as reverse_image_search

app = FastAPI(title="AiWebCity", version="0.9.0-evidence-first")


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
        "identification_mode": "evidence_first",
    }


def _item_mentions_identified_address(item: dict[str, Any], identity: dict[str, Any]) -> bool:
    best = identity.get("best") if isinstance(identity, dict) else None
    if not best:
        return False
    street = norm_text(best.get("street")).lower().replace("ё", "е")
    house = norm_text(best.get("house")).lower().replace("ё", "е")
    text = " ".join(norm_text(item.get(k)) for k in ("title", "description", "site", "page_url"))
    text = text.lower().replace("ё", "е")
    return bool(street and house and street in text and house in text)


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
        raise HTTPException(400, "Укажите хотя бы адрес-подсказку.")

    # Privacy: raw bytes exist only in memory; nothing is written to VPS storage.
    analysis = await analyze_photo(raw, content_type, address, year.strip())

    reverse_result = await reverse_image_search(raw, address="")
    reverse_items = reverse_result.get("results", [])

    identity = await resolve_identity(analysis, reverse_items, address)
    identified = identity.get("best") if identity.get("status") == "identified" else None

    search_queries = [norm_text(q) for q in analysis.get("search_queries", []) if norm_text(q)]
    visible_terms = [norm_text(v) for v in analysis.get("visible_text", []) if norm_text(v)]
    clue_queries = search_queries + visible_terms[:3]
    if identified:
        clue_queries.insert(0, identified["address"])
    city_queries = list(dict.fromkeys([f"Новороссийск {q}" for q in clue_queries]))[:6]
    open_images = await wikimedia_search(city_queries, limit=12) if city_queries else []
    similar = dedupe_results(reverse_items + open_images, 12)

    historical: list[dict[str, Any]] = []
    if identified and identified.get("geocode"):
        requested_year = extract_year(year) or 1999
        historical.extend(await pastvu_search(identified["geocode"]["lat"], identified["geocode"]["lon"], requested_year))
        historical.extend({**item, "kind": "historical"} for item in reverse_items if _item_mentions_identified_address(item, identity) and isinstance(item.get("year"), int) and item["year"] <= requested_year)

    engine_status = [
        {"name": e["engine"], "ok": e["ok"], "results": len(e["results"]), "error": e.get("error")}
        for e in reverse_result.get("engines", [])
    ]

    return {
        "status": "completed",
        "city": "Новороссийск",
        "user_address_hint": address,
        "year": year.strip(),
        "identification": identity_summary(identity),
        "analysis": analysis,
        "reverse_image_search": {"provider": "PicImageSearch", "engines": engine_status, "results": reverse_items},
        "similar_images": similar,
        "historical_images": dedupe_results(historical, 24),
        "identified_place": identified,
        "sources": {
            "historical": build_pastphoto_link(identified["address"]) if identified else None,
            "pastvu": "https://pastvu.com/",
            "wikimedia": "https://commons.wikimedia.org/",
            "geocoder": "https://www.openstreetmap.org/",
        },
        "privacy": {
            "server_storage": False,
            "note": "Загруженное фото не сохраняется на VPS. Для reverse image search оно временно передаётся внешним поисковым сервисам через PicImageSearch.",
        },
        "generation": {"enabled": False},
    }
