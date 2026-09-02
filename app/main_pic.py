from __future__ import annotations

import io
import re
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from .main import STATIC_DIR, MAX_UPLOAD_MB, TIMEWEB_TOKEN, build_pastphoto_link, extract_year, norm_text, pastvu_search, wikimedia_search
from .reverse_search import search as reverse_image_search
from .visual_compare import verify_candidates, extract_address_hints

app = FastAPI(title="AiWebCity", version="1.0.0-evidence-first")

NOVOROSSIYSK_TERMS = ("новороссийск", "novorossiysk")


def _is_novorossiysk(text: str) -> bool:
    value = norm_text(text).lower().replace("ё", "е")
    return any(term in value for term in NOVOROSSIYSK_TERMS)


async def _geocode_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    hints = extract_address_hints(candidate)
    source_text = norm_text(" ".join([
        norm_text(candidate.get("title")),
        norm_text(candidate.get("description")),
        norm_text(candidate.get("site")),
    ]))
    if _is_novorossiysk(source_text):
        hints.append(source_text[:160])
    if candidate.get("title"):
        hints.append(norm_text(candidate.get("title"))[:160])

    queries: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        hint = norm_text(hint)
        if hint and hint not in seen:
            seen.add(hint)
            queries.append(hint)

    if not queries:
        return None

    headers = {"User-Agent": "AiWebCity/1.0 (+https://aiweb.su/)"}
    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        for query in queries[:5]:
            q = query if _is_novorossiysk(query) else f"{query}, Новороссийск, Россия"
            try:
                response = await client.get("https://nominatim.openstreetmap.org/search", params={"q": q, "format": "jsonv2", "limit": 3, "addressdetails": 1})
                if response.status_code >= 400:
                    continue
                rows = response.json()
            except (httpx.HTTPError, ValueError):
                continue
            for row in rows if isinstance(rows, list) else []:
                display = norm_text(row.get("display_name"))
                if not _is_novorossiysk(display):
                    continue
                addr = row.get("address") if isinstance(row.get("address"), dict) else {}
                street = norm_text(addr.get("road") or addr.get("pedestrian") or addr.get("residential"))
                house = norm_text(addr.get("house_number"))
                if not street:
                    continue
                return {
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "display_name": display,
                    "street": street,
                    "house_number": house,
                    "city": norm_text(addr.get("city") or addr.get("town") or "Новороссийск"),
                    "address_verified": True,
                    "house_number_verified": bool(house),
                }
    return None


def _pretty_address(location: dict[str, Any] | None) -> str:
    if not location:
        return ""
    street = norm_text(location.get("street"))
    house = norm_text(location.get("house_number"))
    if street and house:
        return f"Новороссийск, {street}, {house}"
    if street:
        return f"Новороссийск, {street}"
    return norm_text(location.get("display_name"))


async def _verify_candidates(original: bytes, content_type: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visual = await verify_candidates(original, content_type, candidates, limit=8)
    enriched: list[dict[str, Any]] = []
    for item in visual:
        same_place = bool(item.get("same_place")) and float(item.get("confidence", 0.0)) >= 0.72
        location = await _geocode_candidate(item) if same_place else None
        copy = dict(item)
        copy["visual_verified"] = same_place
        copy["address_verified"] = bool(location)
        copy["location"] = location
        copy["resolved_address"] = _pretty_address(location)
        enriched.append(copy)
    return sorted(enriched, key=lambda x: (bool(x.get("visual_verified")) and bool(x.get("address_verified")), float(x.get("confidence", 0.0))), reverse=True)


def _dedupe_images(items: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        url = norm_text(item.get("image_url"))
        if not url or url in seen:
            continue
        seen.add(url)
        result.append({k: v for k, v in item.items() if k in {"image_url", "page_url", "year", "source", "kind", "distance_m"}})
        if len(result) >= limit:
            break
    return result


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
        "identification_mode": "reverse-search-then-visual-verification",
        "city_scope": "Новороссийск",
        "photo_persistence": False,
        "generation": False,
    }


@app.post("/api/identify")
async def identify(photo: UploadFile = File(...), address: str = Form(""), year: str = Form("")) -> dict[str, Any]:
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

    address = norm_text(address)
    reverse_result = await reverse_image_search(raw, address=address)
    candidates = reverse_result.get("results", [])
    if not candidates:
        return {
            "status": "uncertain",
            "city": "Новороссийск",
            "place": None,
            "verification": {"visual_match": False, "address_verified": False},
            "candidate_images": [],
            "matched_images": [],
            "historical_images": [],
            "message": "Внешний поиск не нашёл фотографий для визуальной проверки.",
            "privacy": {"server_storage": False},
            "generation": {"enabled": False},
        }

    verifications = await _verify_candidates(raw, content_type, candidates)
    accepted = [x for x in verifications if x.get("visual_verified") and x.get("address_verified")]

    place = None
    matched_images: list[dict[str, Any]] = []
    if accepted:
        best = accepted[0]
        location = best.get("location") or {}
        place = {
            "address": best.get("resolved_address") or location.get("display_name"),
            "street": location.get("street"),
            "house_number": location.get("house_number") or None,
            "lat": location.get("lat"),
            "lon": location.get("lon"),
            "confidence": float(best.get("confidence", 0.0)),
        }
        matched_images = [
            {"image_url": x.get("image_url"), "confidence": float(x.get("confidence", 0.0)), "matching_features": x.get("matching_features", [])}
            for x in accepted[:8]
            if x.get("image_url")
        ]

    historical: list[dict[str, Any]] = []
    if place and place.get("lat") is not None and place.get("lon") is not None:
        requested_year = extract_year(year) or 1999
        historical.extend(await pastvu_search(float(place["lat"]), float(place["lon"]), requested_year))
        terms = [place.get("address") or "", place.get("street") or ""]
        queries = [f"Новороссийск {norm_text(x)}" for x in terms if norm_text(x)]
        if queries:
            historical.extend(await wikimedia_search(queries, limit=12))

    return {
        "status": "identified" if place else "uncertain",
        "city": "Новороссийск",
        "place": place,
        "verification": {"visual_match": bool(place), "address_verified": bool(place)},
        "candidate_images": [{"image_url": x.get("image_url")} for x in verifications[:12] if x.get("image_url")],
        "matched_images": matched_images,
        "historical_images": _dedupe_images(historical, 24),
        "message": (
            "Здание визуально подтверждено, затем его адрес подтверждён по открытым геоданным."
            if place else
            "Похожие фотографии найдены, но надёжного визуального совпадения с подтверждённым адресом нет."
        ),
        "privacy": {
            "server_storage": False,
            "note": "Фотография не записывается на VPS; она используется только в памяти и временно передаётся внешним поисковым и AI-сервисам.",
        },
        "generation": {"enabled": False},
        "debug": {"reverse_candidates": len(candidates), "visual_checks": len(verifications)},
    }
