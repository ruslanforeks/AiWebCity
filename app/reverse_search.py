"""Сбор пула кандидатов: Yandex CBIR (+ кропы по детектору) и Google Lens.

Порядок такой же, как в целевом пайплайне проекта:
    полное фото -> детектор объектов Яндекса -> кроп здания -> повторный поиск
    -> объединение -> дедупликация -> текстовый ранкинг под Новороссийск.

Сначала recall, потом precision: отсюда выходит большой пул, а отсечение
делает уже визуальная верификация.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from PIL import Image

from .image_fetch import encode_jpeg
from .yandex_cbir import BUILDING_CATEGORIES, cbir_search, yandex_image_id

REVERSE_SEARCH_TIMEOUT = float(os.getenv("REVERSE_SEARCH_TIMEOUT", "45"))
FINAL_RESULTS = int(os.getenv("REVERSE_SEARCH_RESULTS", "60"))
# Сколько запросов к Яндексу разрешено на одну фотографию. Больше запросов —
# лучше recall, но выше риск капчи на IP сервера.
MAX_YANDEX_PASSES = int(os.getenv("YANDEX_MAX_PASSES", "3"))
GOOGLE_LENS_ENABLED = os.getenv("GOOGLE_LENS_ENABLED", "true").lower() != "false"
CACHE_TTL_SECONDS = float(os.getenv("REVERSE_SEARCH_CACHE_TTL", "900"))

NOVOROSSIYSK_TERMS = {"новороссийск", "novorossiysk", "новороссийского", "новороссийске", "новороссийском", "нврск"}

OTHER_CITIES = {
    "геленджик", "геледжик", "анапа", "краснодар", "сочи", "ростов-на-дону", "ростов на дону",
    "майкоп", "туапсе", "армавир", "керчь", "севастополь", "симферополь", "волгоград", "астрахань",
    "москва", "санкт-петербург", "екатеринбург", "самара", "казань", "воронеж", "саратов",
}

NOISE_TERMS = {
    "интерьер", "квартира", "комната", "кухня", "ванная", "планировка", "обои", "мебель", "диван",
    "товар", "каталог", "автомобиль", "мотоцикл", "телефон", "обои на телефон", "декор",
    "дизайн интерьера", "купить квартиру", "снять квартиру", "продажа квартир",
}

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_url(value: Any) -> str:
    value = norm_text(value)
    if not value:
        return ""
    try:
        parts = urlsplit(value)
        if not parts.netloc:
            return value.lower().rstrip("/")
        keep = [
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in {"from", "ref", "source", "tracking", "n"}
        ]
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(keep), "")).lower()
    except Exception:
        return value.lower().rstrip("/")


def image_identity(candidate: dict[str, Any]) -> str:
    """Устойчивый ключ дедупликации: один и тот же файл на разных страницах — один кандидат."""
    yandex_id = yandex_image_id(norm_text(candidate.get("thumb_url"))) or yandex_image_id(norm_text(candidate.get("image_url")))
    if yandex_id:
        return f"yandex-id:{yandex_id}"
    normalized = normalize_url(candidate.get("image_url"))
    if normalized:
        return normalized
    return normalize_url(candidate.get("page_url"))


def _crop_bytes(image: Image.Image, box: tuple[float, float, float, float], *, pad: float = 0.06) -> bytes:
    width, height = image.size
    x0, y0, x1, y1 = box
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = min(1.0, x1 + pad)
    y1 = min(1.0, y1 + pad)
    crop = image.crop((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)))
    return encode_jpeg(crop, max_side=1400, quality=90)


def pick_object_crops(crops: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """Отбирает bbox, ради которых имеет смысл тратить отдельный запрос.

    Приоритет — здания. Кропы «машина»/«растение» для идентификации места
    бесполезны и только жгут лимит запросов.
    """
    buildings = [c for c in crops if c["category"] in BUILDING_CATEGORIES and 0.06 <= c["area"] <= 0.92]
    return buildings[:limit]


def fallback_crops(image: Image.Image, *, limit: int) -> list[bytes]:
    """Если детектор Яндекса ничего не отдал — режем кадр сами.

    Это сценарий «пользователь снял улицу»: на кадре несколько фасадов,
    и целиком он не ищется ни у кого.
    """
    width, height = image.size
    boxes: list[tuple[float, float, float, float]] = []
    if width / max(height, 1) >= 1.2:
        boxes = [(0.0, 0.0, 0.58, 1.0), (0.42, 0.0, 1.0, 1.0)]
    elif height / max(width, 1) >= 1.2:
        boxes = [(0.0, 0.0, 1.0, 0.62), (0.0, 0.30, 1.0, 0.95)]
    else:
        boxes = [(0.12, 0.0, 0.88, 0.85)]
    return [_crop_bytes(image, box, pad=0.0) for box in boxes[:limit]]


def _tokens_for_address(address: str) -> tuple[list[str], str]:
    clean = norm_text(address).lower().replace("ё", "е")
    tokens = [t for t in re.findall(r"[a-zа-я0-9-]+", clean) if len(t) >= 3]
    excluded = {"россия", "край", "город", "ул", "улица", "проспект", "пр", "дом", "д", "новороссийск", "novorossiysk"}
    street_tokens = [t for t in tokens if t not in excluded]
    number_match = re.search(r"\b(\d{1,4}[а-яa-z]?)\b", clean)
    return street_tokens, number_match.group(1) if number_match else ""


def _result_text(item: dict[str, Any]) -> str:
    return " ".join(
        norm_text(item.get(key)) for key in ("title", "description", "site", "page_url", "image_url")
    ).lower().replace("ё", "е")


def rank_candidates(items: list[dict[str, Any]], address: str, limit: int = FINAL_RESULTS) -> list[dict[str, Any]]:
    """Текстовый предварительный ранкинг.

    Это только порядок, в котором кандидаты пойдут на визуальную проверку.
    Текст здесь НЕ является доказательством совпадения — он лишь решает,
    кого показать Vision-модели первым при ограниченном бюджете.
    """
    address_lower = norm_text(address).lower().replace("ё", "е")
    street_tokens, house_number = _tokens_for_address(address)
    scored: list[tuple[float, int, dict[str, Any]]] = []

    for position, item in enumerate(items):
        if not norm_text(item.get("image_url")) and not norm_text(item.get("thumb_url")):
            continue
        text = _result_text(item)

        score = 0.0
        # Точное совпадение файла — сильнее, чем «похожее изображение».
        score += 45.0 if item.get("match_type") == "exact" else 0.0
        # Порядок внутри выдачи Яндекса — это его собственная оценка похожести.
        score += max(0.0, 26.0 - float(item.get("rank", 0)) * 0.6)
        # Кроп здания важнее, чем поиск по всему кадру с улицей. Кроп по рамке
        # от Vision — единственный проход, который смотрел именно на постройку,
        # поэтому весит заметно больше слепой нарезки кадра.
        search_pass = item.get("search_pass", "")
        if search_pass == "crop:vision":
            score += 18.0
        elif search_pass.startswith("crop"):
            score += 6.0

        if any(term in text for term in NOVOROSSIYSK_TERMS):
            score += 34.0
        elif any(city in text for city in OTHER_CITIES):
            # Другой город в подписи — почти наверняка не наш объект,
            # но полностью выкидывать нельзя: подпись бывает мусорной.
            score -= 55.0

        if address_lower and address_lower in text:
            score += 20.0
        matched_street = sum(1 for token in street_tokens if token in text)
        if matched_street:
            score += min(matched_street, 4) * 11.0
        if house_number and re.search(rf"(?<!\d){re.escape(house_number)}(?!\d)", text):
            score += 14.0
        if any(term in text for term in NOISE_TERMS):
            score -= 45.0
        if not norm_text(item.get("image_url")):
            score -= 8.0

        copy = dict(item)
        copy["text_score"] = round(score, 1)
        scored.append((score, -position, copy))

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for _, _, item in sorted(scored, key=lambda row: (row[0], row[1]), reverse=True):
        key = image_identity(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def dedupe(items: list[dict[str, Any]], limit: int = 200) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = image_identity(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _candidates_from_pass(result: dict[str, Any], pass_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(result.get("sites") or []) + list(result.get("similar") or []):
        row = dict(item)
        row["source"] = "Yandex Images"
        row["kind"] = "reverse_image"
        row["search_pass"] = pass_name
        row["image_quality"] = "original" if row.get("image_url") and "avatars.mds.yandex.net" not in row["image_url"] else "yandex_thumb"
        row["preview_url"] = row.get("thumb_url") or row.get("image_url")
        rows.append(row)
    return rows


async def google_lens_search(image_bytes: bytes) -> dict[str, Any]:
    """Google Lens как дополнительный источник. В наших тестах почти всегда пуст,
    поэтому он полностью опционален и никогда не роняет основной поиск."""
    if not GOOGLE_LENS_ENABLED:
        return {"engine": "Google Lens", "ok": False, "results": [], "error": "disabled"}
    try:
        from PicImageSearch import GoogleLens, Network

        async with Network(timeout=REVERSE_SEARCH_TIMEOUT) as client:
            response = await asyncio.wait_for(
                GoogleLens(client=client, search_type="all", hl="ru", country="RU").search(file=image_bytes),
                timeout=REVERSE_SEARCH_TIMEOUT,
            )
        results = []
        for position, item in enumerate(response.raw or []):
            thumb = norm_text(getattr(item, "thumbnail", ""))
            results.append({
                "title": norm_text(getattr(item, "title", "")),
                "description": norm_text(getattr(item, "content", "")),
                "page_url": norm_text(getattr(item, "url", "")),
                "site": norm_text(getattr(item, "source", "")),
                "image_url": thumb,
                "thumb_url": thumb,
                "preview_url": thumb,
                "source": "Google Lens",
                "kind": "reverse_image",
                "match_type": "similar",
                "rank": position,
                "search_pass": "full",
                "image_quality": "preview",
            })
        return {"engine": "Google Lens", "ok": True, "results": results, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"engine": "Google Lens", "ok": False, "results": [], "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


async def search(
    image_bytes: bytes,
    address: str = "",
    building_box_provider: Any = None,
) -> dict[str, Any]:
    cache_key = hashlib.sha256(image_bytes).hexdigest() + "|" + norm_text(address).lower()
    cached = _CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return {**cached[1], "cached": True}

    started = now
    passes: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []

    lens_task = asyncio.create_task(google_lens_search(image_bytes))

    first = await cbir_search(image_bytes, timeout=REVERSE_SEARCH_TIMEOUT)
    passes.append({"name": "full", "ok": first["ok"], "error": first["error"], "sites": len(first["sites"]), "similar": len(first["similar"])})
    all_candidates.extend(_candidates_from_pass(first, "full"))

    tags = list(first.get("tags") or [])
    ocr_text = norm_text(first.get("ocr_text"))
    crops_meta = list(first.get("crops") or [])
    captcha = bool(first.get("captcha"))

    # Второй и третий проход — по объектам, найденным детектором Яндекса.
    # При капче дальше не идём, чтобы не усугублять ситуацию с IP.
    remaining = max(0, MAX_YANDEX_PASSES - 1)
    vision_box = None
    if remaining and not captcha and building_box_provider and not pick_object_crops(crops_meta, limit=1):
        # Детектор Яндекса здания не нашёл. На кадре, который занимает дерево, он
        # честно отвечает «pinus», и обратный поиск начинает искать сосну. Просим
        # рамку у Vision — это один вызов и только в таком случае.
        try:
            vision_box = await building_box_provider(image_bytes)
        except Exception:
            vision_box = None

    crop_payloads: list[tuple[str, bytes]] = []
    if remaining and not captcha:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                selected = pick_object_crops(crops_meta, limit=remaining)
                crop_payloads = [
                    (f"crop:{crop['category']}", _crop_bytes(image, crop["box"])) for crop in selected
                ]
                if not crop_payloads and vision_box:
                    crop_payloads = [("crop:vision", _crop_bytes(image, vision_box["box"]))]
                if not crop_payloads:
                    crop_payloads = [
                        (f"crop:auto{index}", data)
                        for index, data in enumerate(fallback_crops(image, limit=remaining))
                    ]
        except Exception:
            crop_payloads = []

        for pass_name, payload in crop_payloads[:remaining]:
            await asyncio.sleep(1.2)  # вежливая пауза между запросами к Яндексу
            result = await cbir_search(payload, timeout=REVERSE_SEARCH_TIMEOUT)
            passes.append({"name": pass_name, "ok": result["ok"], "error": result["error"], "sites": len(result["sites"]), "similar": len(result["similar"])})
            all_candidates.extend(_candidates_from_pass(result, pass_name))
            for tag in result.get("tags") or []:
                if tag not in tags:
                    tags.append(tag)
            if result.get("captcha"):
                captcha = True
                break

    lens = await lens_task
    all_candidates.extend(lens.get("results") or [])

    raw = dedupe(all_candidates, limit=240)
    ranked = rank_candidates(raw, address, limit=FINAL_RESULTS)

    payload = {
        "enabled": True,
        "results": ranked,
        "raw_result_count": len(raw),
        "passes": passes,
        "engines": [{"engine": "Yandex Images", "ok": first["ok"], "error": first["error"]}, {"engine": lens["engine"], "ok": lens["ok"], "error": lens["error"]}],
        "yandex_tags": tags,
        "ocr_text": ocr_text,
        "detected_objects": [{"category": c["category"], "area": round(c["area"], 3)} for c in crops_meta],
        "vision_building_box": vision_box,
        "captcha": captcha,
        "elapsed_seconds": round(time.time() - started, 1),
        "cached": False,
    }
    _CACHE[cache_key] = (time.time(), payload)
    if len(_CACHE) > 64:
        for key in sorted(_CACHE, key=lambda k: _CACHE[k][0])[:16]:
            _CACHE.pop(key, None)
    return payload
