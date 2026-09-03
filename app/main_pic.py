from __future__ import annotations

import io
import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from . import address_resolve
from .main import MAX_UPLOAD_MB, STATIC_DIR, TIMEWEB_TOKEN, extract_year, norm_text, pastvu_search, wikimedia_search
from .reverse_search import search as reverse_image_search
from .visual_compare import verify_candidates

app = FastAPI(title="AiWebCity", version="1.2.0-recall")

VISUAL_CHECK_LIMIT = int(os.getenv("VISUAL_CHECK_LIMIT", "8"))
VISUAL_CONFIDENCE_THRESHOLD = float(os.getenv("VISUAL_CONFIDENCE_THRESHOLD", "0.72"))


def _dedupe_images(items: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        url = norm_text(item.get("image_url"))
        if not url or url in seen:
            continue
        seen.add(url)
        result.append({
            key: item.get(key)
            for key in ("image_url", "page_url", "year", "source", "kind", "distance_m")
            if item.get(key) is not None
        })
        if len(result) >= limit:
            break
    return result


def _empty_response(message: str, **extra: Any) -> dict[str, Any]:
    """Ответ без подтверждённого объекта. status переопределяется через extra."""
    payload = {
        "status": "uncertain",
        "city": "Новороссийск",
        "place": None,
        "verification": {"visual_match": False, "address_verified": False},
        "candidate_images": [],
        "matched_images": [],
        "historical_images": [],
        "message": message,
        "privacy": {"server_storage": False},
        "generation": {"enabled": False},
    }
    payload.update(extra)
    return payload


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index_pic.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "token_configured": bool(TIMEWEB_TOKEN),
        "reverse_search": "Yandex CBIR: sites + similar + tags + object crops + OCR",
        "reverse_search_engines": ["Yandex Images (exact + visually similar)", "Google Lens"],
        "identification_mode": "object-crops -> candidate-pool -> parallel-visual-verification -> address-cross-check",
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

    address_hint = norm_text(address)
    reverse_result = await reverse_image_search(raw, address=address_hint)
    candidates = reverse_result.get("results", [])
    yandex_tags = reverse_result.get("yandex_tags", [])
    ocr_text = reverse_result.get("ocr_text", "")

    if reverse_result.get("captcha"):
        return _empty_response(
            "Поисковый сервис временно ограничил запросы с нашего сервера. Попробуйте ещё раз через несколько минут.",
            debug={"captcha": True, "passes": reverse_result.get("passes")},
        )
    if not candidates:
        return _empty_response(
            "Обратный поиск не нашёл фотографий, которые можно надёжно проверить визуально.",
            debug={"passes": reverse_result.get("passes"), "elapsed_seconds": reverse_result.get("elapsed_seconds")},
        )

    verifications = await verify_candidates(raw, content_type, candidates, limit=VISUAL_CHECK_LIMIT)

    # Если Vision вообще не отвечает (кончился лимит, упал шлюз), это НЕ то же
    # самое, что «совпадений не найдено». Пользователю нужно сказать правду.
    vision_errors = [norm_text(item.get("visual_error")) for item in verifications if item.get("visual_error")]
    reachable = [item for item in verifications if not item.get("visual_error")]
    quota_blocked = any("usage limit" in error.lower() or "timeweb_42" in error.lower() for error in vision_errors)
    if verifications and not reachable:
        return _empty_response(
            "Сервис визуальной проверки сейчас недоступен"
            + (" (исчерпан лимит Timeweb AI)." if quota_blocked else ".")
            + " Кандидаты найдены, но подтвердить их без визуальной проверки нельзя.",
            status="verification_unavailable",
            candidate_images=[
                {"image_url": item.get("image_url"), "preview_url": item.get("preview_url") or item.get("thumb_url")}
                for item in verifications[:12] if item.get("image_url") or item.get("thumb_url")
            ],
            debug={
                "reverse_candidates": len(candidates),
                "raw_result_count": reverse_result.get("raw_result_count"),
                "yandex_tags": yandex_tags[:6],
                "vision_errors": vision_errors[:3],
                "quota_blocked": quota_blocked,
            },
        )

    accepted = [
        item for item in verifications
        if item.get("same_place") and float(item.get("confidence", 0.0)) >= VISUAL_CONFIDENCE_THRESHOLD
    ]

    resolved = await address_resolve.resolve(
        verified_candidates=accepted,
        yandex_tags=yandex_tags,
        ocr_text=ocr_text,
        user_address=address_hint,
    )
    location = resolved.get("location")
    # Адрес засчитывается только при независимом подтверждении улицы. Иначе он
    # остаётся предположением и не выдаётся как факт.
    address_confirmed = bool(resolved.get("confirmed"))

    place = None
    if accepted:
        best = accepted[0]
        place = {
            "address": (resolved.get("address") or None) if address_confirmed else None,
            "address_guess": (resolved.get("address") or None) if (location and not address_confirmed) else None,
            "address_confirmed": address_confirmed,
            "street": (location or {}).get("street") if address_confirmed else None,
            "house_number": ((location or {}).get("house_number") or None) if address_confirmed else None,
            "lat": (location or {}).get("lat") if address_confirmed else None,
            "lon": (location or {}).get("lon") if address_confirmed else None,
            "confidence": float(best.get("confidence", 0.0)),
            "object": best.get("which_object") or None,
            "address_precision": (location or {}).get("precision"),
        }

    matched_images = [
        {
            "image_url": item.get("image_url"),
            "confidence": float(item.get("confidence", 0.0)),
            "matching_features": item.get("matching_features", [])[:3],
        }
        for item in accepted[:8]
        if item.get("image_url")
    ]

    # Исторические фотографии ищем только вокруг ПОДТВЕРЖДЁННОГО адреса: по
    # неподтверждённой догадке они могут оказаться снимками совсем другого места.
    historical: list[dict[str, Any]] = []
    if address_confirmed and location and location.get("lat") is not None:
        requested_year = extract_year(year) or 1999
        historical.extend(await pastvu_search(float(location["lat"]), float(location["lon"]), requested_year))
        queries = [
            f"Новороссийск {norm_text(resolved.get('address'))}",
            f"Новороссийск {norm_text(location.get('street'))}",
        ]
        queries = list(dict.fromkeys(q for q in queries if q.strip() != "Новороссийск"))
        if queries:
            historical.extend(await wikimedia_search(queries, limit=12))

    if accepted and address_confirmed:
        status, message = "identified", "Здание визуально подтверждено, затем независимо проверен его адрес."
    elif accepted and location:
        status, message = (
            "visual_only",
            f"Здание визуально подтверждено, но адрес не подтверждён независимо. "
            f"Вероятный вариант — {resolved.get('address')} — сервис показывает как предположение, а не как факт.",
        )
    elif accepted:
        status, message = (
            "visual_only",
            "Здание визуально подтверждено на нескольких фотографиях, но определить его адрес по открытым данным не удалось.",
        )
    else:
        status, message = (
            "uncertain",
            "Похожие фотографии найдены, но надёжного визуального совпадения нет. Сомнительный результат не выдаём за подтверждённый.",
        )

    return {
        "status": status,
        "city": "Новороссийск",
        "place": place,
        "verification": {
            "visual_match": bool(accepted),
            "address_verified": address_confirmed,
            "visual_matches": len(accepted),
            "checked": len(verifications),
        },
        "candidate_images": [
            {"image_url": item.get("image_url"), "preview_url": item.get("preview_url") or item.get("thumb_url")}
            for item in verifications[:12]
            if item.get("image_url") or item.get("thumb_url")
        ],
        "matched_images": matched_images,
        "historical_images": _dedupe_images(historical, 24),
        "message": message,
        "privacy": {
            "server_storage": False,
            "note": "Фото не записывается на VPS; оно используется в памяти и временно передаётся внешним поисковым и AI-сервисам.",
        },
        "generation": {"enabled": False},
        "debug": {
            "reverse_candidates": len(candidates),
            "raw_result_count": reverse_result.get("raw_result_count"),
            "passes": reverse_result.get("passes"),
            "detected_objects": reverse_result.get("detected_objects"),
            "yandex_tags": yandex_tags[:6],
            "ocr_text": ocr_text[:200],
            "visual_checks": len(verifications),
            "visual_errors": vision_errors[:5],
            "address_hypotheses": resolved.get("hypotheses"),
            "address_confirmed": address_confirmed,
            "address_guess": resolved.get("address") if not address_confirmed else None,
            "search_seconds": reverse_result.get("elapsed_seconds"),
            "cached_search": reverse_result.get("cached"),
        },
    }
