from __future__ import annotations

import asyncio
import io
import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from PicImageSearch import GoogleLens, Network, Yandex
from PIL import Image

REVERSE_SEARCH_TIMEOUT = 45.0
REVERSE_SEARCH_RESULTS = 18
FINAL_RESULTS = 36
SEARCH_VARIANTS = 4

NOVOROSSIYSK_TERMS = {"новороссийск", "novorossiysk", "новороссийского", "новороссийске", "новороссийском"}

OTHER_CITIES = {"геленджик", "геледжик", "анапа", "краснодар", "сочи", "ростов-на-дону", "ростов на дону", "майкоп", "туапсе", "армавир", "керчь", "севастополь", "симферополь"}

NOISE_TERMS = {"интерьер", "квартира", "комната", "кухня", "ванная", "планировка", "обои", "мебель", "диван", "товар", "каталог", "автомобиль", "машина", "мотоцикл", "телефон", "обои на телефон", "декор", "дизайн интерьера"}


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
        keep = []
        for key, val in parse_qsl(parts.query, keep_blank_values=True):
            if key.lower().startswith("utm_") or key.lower() in {"from", "ref", "source", "tracking"}:
                continue
            keep.append((key, val))
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(keep), "")).lower()
    except Exception:
        return value.lower().rstrip("/")


def image_identity(image_url: str) -> str:
    normalized = normalize_url(image_url)
    if not normalized:
        return ""
    match = re.search(r"[?&]id=([a-z0-9_-]{12,})", normalized, re.I)
    if match:
        return f"yandex-id:{match.group(1).lower()}"
    match = re.search(r"/i\?id=([a-z0-9_-]{12,})", normalized, re.I)
    if match:
        return f"yandex-id:{match.group(1).lower()}"
    return normalized


def _absolute_url(value: Any) -> str:
    value = norm_text(value)
    return "https:" + value if value.startswith("//") else value


def _extract_yandex_originals(response: Any) -> list[dict[str, Any]]:
    try:
        origin = getattr(response, "origin", None)
        root = origin.find('div.Root[id^="ImagesApp-"]') if origin is not None else None
        state = root.attr("data-state") if root is not None else None
        if not state:
            return []
        data = json.loads(str(state))
        sites = (((data or {}).get("initialState") or {}).get("cbirSites") or {}).get("sites") or []
        if not isinstance(sites, list):
            return []
        result = []
        for site in sites:
            if not isinstance(site, dict):
                continue
            original = site.get("originalImage") if isinstance(site.get("originalImage"), dict) else {}
            url = next((_absolute_url(original.get(key)) for key in ("url", "originUrl", "origin_url", "src") if _absolute_url(original.get(key))), "")
            result.append({"url": url, "width": original.get("width"), "height": original.get("height")})
        return result
    except Exception:
        return []


def item_to_dict(item: Any, source: str, *, original_url: str = "", original_size: str = "", variant: str = "full") -> dict[str, Any]:
    title = norm_text(getattr(item, "title", ""))
    page_url = norm_text(getattr(item, "url", ""))
    thumb = norm_text(getattr(item, "thumbnail", ""))
    return {
        "image_url": norm_text(original_url) or thumb,
        "preview_url": thumb,
        "page_url": page_url,
        "title": title or f"Результат {source}",
        "description": norm_text(getattr(item, "content", "")),
        "source": source,
        "kind": "reverse_image",
        "site": norm_text(getattr(item, "source", "")),
        "size": original_size or norm_text(getattr(item, "size", "")),
        "image_quality": "original" if original_url else "preview_fallback",
        "search_variant": variant,
    }


def _make_search_variants(image_bytes: bytes) -> list[tuple[str, bytes]]:
    variants = [("full", image_bytes)]
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            width, height = image.size
            if width / max(height, 1) >= 1.25:
                crop_width = max(1, int(width * 0.62))
                for name, start in (("left", 0.0), ("center", 0.19), ("right", 0.38)):
                    left = min(width - crop_width, max(0, int(width * start)))
                    crop = image.crop((left, 0, left + crop_width, height))
                    out = io.BytesIO(); crop.save(out, format="JPEG", quality=92, optimize=True)
                    variants.append((name, out.getvalue()))
            else:
                crop_height = max(1, int(height * 0.68))
                for name, start in (("top", 0.0), ("center", 0.16), ("bottom", 0.32)):
                    top = min(height - crop_height, max(0, int(height * start)))
                    crop = image.crop((0, top, width, top + crop_height))
                    out = io.BytesIO(); crop.save(out, format="JPEG", quality=92, optimize=True)
                    variants.append((name, out.getvalue()))
    except Exception:
        return variants
    return variants[:SEARCH_VARIANTS]


def _tokens_for_address(address: str) -> tuple[list[str], str]:
    clean = norm_text(address).lower().replace("ё", "е")
    tokens = [t for t in re.findall(r"[a-zа-я0-9-]+", clean) if len(t) >= 3]
    excluded = {"россия", "край", "город", "ул", "улица", "проспект", "пр", "дом", "д", "новороссийск", "novorossiysk"}
    street_tokens = [t for t in tokens if t not in excluded]
    number_match = re.search(r"\b(\d{1,4}[а-яa-z]?)\b", clean)
    return street_tokens, number_match.group(1) if number_match else ""


def _result_text(item: dict[str, Any]) -> str:
    return " ".join(norm_text(item.get(key)) for key in ("title", "description", "site", "page_url", "image_url")).lower().replace("ё", "е")


def rank_novorossiysk(items: list[dict[str, Any]], address: str, limit: int = FINAL_RESULTS) -> list[dict[str, Any]]:
    address_lower = norm_text(address).lower().replace("ё", "е")
    street_tokens, house_number = _tokens_for_address(address)
    scored = []
    for position, item in enumerate(items):
        if not norm_text(item.get("image_url")):
            continue
        text = _result_text(item)
        if any(city in text for city in OTHER_CITIES):
            continue
        score = max(0, 24 - position // 2)
        if any(term in text for term in NOVOROSSIYSK_TERMS):
            score += 40
        if address_lower and address_lower in text:
            score += 25
        matched_street = sum(1 for token in street_tokens if token in text)
        if matched_street:
            score += min(matched_street, 4) * 14
        if house_number and re.search(rf"(?<!\d){re.escape(house_number)}(?!\d)", text):
            score += 20
        if any(term in norm_text(item.get("page_url")).lower() for term in street_tokens[:3]):
            score += 10
        if any(term in text for term in NOISE_TERMS):
            score -= 80
        copy = dict(item); copy["candidate_rank_score"] = score
        scored.append((score, -position, copy))
    seen = set(); result = []
    for _, _, item in sorted(scored, key=lambda row: (row[0], row[1]), reverse=True):
        key = image_identity(norm_text(item.get("image_url"))) or normalize_url(item.get("page_url"))
        if not key or key in seen:
            continue
        seen.add(key); item.pop("candidate_rank_score", None); result.append(item)
        if len(result) >= limit:
            break
    return result


def dedupe(items: list[dict[str, Any]], limit: int = 120) -> list[dict[str, Any]]:
    seen = set(); out = []
    for item in items:
        key = image_identity(norm_text(item.get("image_url"))) or normalize_url(item.get("page_url"))
        if not key or key in seen:
            continue
        seen.add(key); out.append(item)
        if len(out) >= limit:
            break
    return out


async def yandex_search(image_bytes: bytes, variant: str = "full") -> dict[str, Any]:
    try:
        async with Network(timeout=REVERSE_SEARCH_TIMEOUT) as client:
            response = await asyncio.wait_for(Yandex(client=client, base_url="https://yandex.com").search(file=image_bytes), timeout=REVERSE_SEARCH_TIMEOUT)
        originals = _extract_yandex_originals(response)
        raw_items = response.raw or []
        results = []
        for index, item in enumerate(raw_items[:REVERSE_SEARCH_RESULTS]):
            original_url = norm_text(originals[index].get("url")) if index < len(originals) else ""
            width = originals[index].get("width") if index < len(originals) else None
            height = originals[index].get("height") if index < len(originals) else None
            size = f"{width}x{height}" if width and height else ""
            results.append(item_to_dict(item, "Yandex Images", original_url=original_url, original_size=size, variant=variant))
        return {"engine": "Yandex Images", "ok": True, "results": results, "search_url": norm_text(getattr(response, "url", "")) or None, "error": None, "original_urls": sum(1 for x in results if x.get("image_quality") == "original"), "variant": variant}
    except Exception as exc:
        return {"engine": "Yandex Images", "ok": False, "results": [], "search_url": None, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "original_urls": 0, "variant": variant}


async def google_lens_search(image_bytes: bytes, variant: str = "full") -> dict[str, Any]:
    try:
        async with Network(timeout=REVERSE_SEARCH_TIMEOUT) as client:
            response = await asyncio.wait_for(GoogleLens(client=client, search_type="all", hl="ru", country="RU").search(file=image_bytes), timeout=REVERSE_SEARCH_TIMEOUT)
        return {"engine": "Google Lens", "ok": True, "results": [item_to_dict(i, "Google Lens", variant=variant) for i in (response.raw or [])][:REVERSE_SEARCH_RESULTS], "search_url": norm_text(getattr(response, "url", "")) or None, "error": None, "variant": variant}
    except Exception as exc:
        return {"engine": "Google Lens", "ok": False, "results": [], "search_url": None, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "variant": variant}


async def _search_variant(name: str, data: bytes, sem: asyncio.Semaphore):
    async with sem:
        return await asyncio.gather(yandex_search(data, name), google_lens_search(data, name))


async def search(image_bytes: bytes, address: str = "") -> dict[str, Any]:
    variants = _make_search_variants(image_bytes)
    sem = asyncio.Semaphore(2)
    pairs = await asyncio.gather(*(_search_variant(name, data, sem) for name, data in variants))
    yandex_results = [item for pair in pairs for item in (pair[0].get("results") or [])]
    lens_results = [item for pair in pairs for item in (pair[1].get("results") or [])]
    raw = dedupe(yandex_results + lens_results, limit=120)
    filtered = rank_novorossiysk(raw, address, limit=FINAL_RESULTS)
    return {"enabled": True, "engines": [pair[0] for pair in pairs] + [pair[1] for pair in pairs], "results": filtered, "raw_result_count": len(raw), "search_variants": [name for name, _ in variants]}
