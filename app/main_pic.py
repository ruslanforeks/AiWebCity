from __future__ import annotations

import io
import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from . import address_resolve, streets
from .address_probe import probe_images
from .main import (MAX_UPLOAD_MB, STATIC_DIR, TIMEWEB_TOKEN, extract_year, norm_text,
                   pastvu_search, wikimedia_geosearch, wikimedia_search)
from .reverse_search import search as reverse_image_search
from .visual_compare import verify_candidates

app = FastAPI(title="AiWebCity", version="1.2.0-recall")

VISUAL_CHECK_LIMIT = int(os.getenv("VISUAL_CHECK_LIMIT", "8"))
VISUAL_CONFIDENCE_THRESHOLD = float(os.getenv("VISUAL_CONFIDENCE_THRESHOLD", "0.72"))
# Сколько адресов-гипотез проверять поиском картинок по адресу и сколько фотографий
# на адрес при этом сравнивать. Каждое сравнение — платный вызов Vision.
ADDRESS_PROBE_LIMIT = int(os.getenv("ADDRESS_PROBE_LIMIT", "2"))
ADDRESS_PROBE_IMAGES = int(os.getenv("ADDRESS_PROBE_IMAGES", "3"))
# Радиусы архивного поиска. Ближний — фотографии самого здания, дальний — улицы
# и квартала: по ним видно, как выглядело место, даже если дом не в кадре.
HISTORY_BUILDING_RADIUS = int(os.getenv("HISTORY_BUILDING_RADIUS", "150"))
HISTORY_STREET_RADIUS = int(os.getenv("HISTORY_STREET_RADIUS", "900"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "48"))


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
            for key in ("image_url", "page_url", "year", "source", "kind", "distance_m", "scope", "title")
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


@app.get("/api/streets")
async def street_suggestions(q: str = "", limit: int = 8) -> dict[str, Any]:
    """Подсказки для поля адреса: улицы и реально существующие номера домов."""
    return {"query": q, "suggestions": streets.suggest(q, limit=max(1, min(limit, 15)))}


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
    probe_report: list[dict[str, Any]] = []

    # Подписи страниц могут вообще не называть улицу — так было на ул. Куникова, 43,
    # где все найденные фотографии были правильные, а адреса в подписях не оказалось.
    # Тогда идём с другой стороны: ищем фотографии ПО АДРЕСУ и сравниваем их с фото
    # пользователя. Совпало — адрес доказан изображением, а не чужим текстом.
    if accepted and not resolved.get("confirmed"):
        targets: list[tuple[str, str, str]] = []
        for street, house in address_resolve.parse_user_hint(address_hint)[:1]:
            targets.append((street, house, "user_hint"))
        for hypothesis in (resolved.get("hypotheses") or []):
            pair = (hypothesis["street"], hypothesis["house"])
            if pair not in {(t[0], t[1]) for t in targets}:
                targets.append((pair[0], pair[1], "hypothesis"))

        for street, house, origin in targets[:ADDRESS_PROBE_LIMIT]:
            reference = await probe_images(street, house, limit=ADDRESS_PROBE_IMAGES + 1)
            if not reference:
                probe_report.append({"address": f"{street} {house}".strip(), "origin": origin, "images": 0, "matched": 0})
                continue
            checks = await verify_candidates(raw, content_type, reference, limit=ADDRESS_PROBE_IMAGES)
            matched = [
                item for item in checks
                if item.get("same_place") and float(item.get("confidence", 0.0)) >= VISUAL_CONFIDENCE_THRESHOLD
            ]
            probe_report.append({
                "address": f"{street} {house}".strip(), "origin": origin,
                "images": len(reference), "matched": len(matched),
            })
            if matched:
                probed = await address_resolve.geocode_single(street, house)
                if probed:
                    probed["sources"] = ["address_probe", origin]
                    probed["evidence"] = [item.get("reason") for item in matched[:2] if item.get("reason")]
                    location = probed
                    resolved["location"] = probed
                    resolved["address"] = address_resolve.pretty_address(probed)
                    resolved["confirmed"] = True
                    accepted.extend(matched)
                    break
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
        lat, lon = float(location["lat"]), float(location["lon"])
        requested_year = extract_year(year) or 1999

        # Ближний радиус идёт первым, поэтому после дедупликации фотографии
        # самого здания оказываются выше снимков квартала.
        # scope говорит только о расстоянии до точки, а не о том, что здание есть
        # в кадре: проверять это должна визуальная сверка, её здесь ещё нет.
        near = await pastvu_search(lat, lon, requested_year, distance=HISTORY_BUILDING_RADIUS)
        far = await pastvu_search(lat, lon, requested_year, distance=HISTORY_STREET_RADIUS)
        for item in near:
            item["scope"] = "building"
        for item in far:
            item.setdefault("scope", "street")
        historical.extend(near)
        historical.extend(far)

        geo_photos = await wikimedia_geosearch(lat, lon, radius=HISTORY_STREET_RADIUS, limit=20)
        for item in geo_photos:
            item["scope"] = "street"
        historical.extend(geo_photos)

        queries = [
            f"Новороссийск {norm_text(resolved.get('address'))}",
            f"Новороссийск {norm_text(location.get('street'))}",
            f"Novorossiysk {norm_text(location.get('street'))}",
        ]
        queries = list(dict.fromkeys(q for q in queries if q.strip() not in {"Новороссийск", "Novorossiysk"}))
        if queries:
            text_photos = await wikimedia_search(queries, limit=16)
            for item in text_photos:
                item["scope"] = "street"
            historical.extend(text_photos)

    if accepted and address_confirmed:
        probed_ok = "address_probe" in ((location or {}).get("sources") or [])
        status = "identified"
        message = (
            "Здание визуально подтверждено. Адрес проверен встречным поиском: фотографии по этому адресу "
            "показывают то же самое здание."
            if probed_ok else
            "Здание визуально подтверждено, затем независимо проверен его адрес."
        )
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
        "historical_images": _dedupe_images(historical, HISTORY_LIMIT),
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
            "address_probe": probe_report,
            "address_confirmed": address_confirmed,
            "address_guess": resolved.get("address") if not address_confirmed else None,
            "search_seconds": reverse_result.get("elapsed_seconds"),
            "cached_search": reverse_result.get("cached"),
        },
    }
