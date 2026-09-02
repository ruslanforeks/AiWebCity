from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from PicImageSearch import GoogleLens, Network, Yandex

REVERSE_SEARCH_TIMEOUT = 45.0
REVERSE_SEARCH_RESULTS = 24
FINAL_RESULTS = 24

NOVOROSSIYSK_TERMS = {
    "новороссийск",
    "novorossiysk",
    "новороссийского",
    "новороссийске",
    "новороссийском",
}

OTHER_CITIES = {
    "геленджик",
    "геледжик",
    "анапа",
    "краснодар",
    "сочи",
    "ростов-на-дону",
    "ростов на дону",
    "майкоп",
    "туапсе",
    "армавир",
    "керчь",
    "севастополь",
    "симферополь",
}
NOISE_TERMS = {
    "интерьер",
    "квартира",
    "комната",
    "кухня",
    "ванная",
    "планировка",
    "обои",
    "мебель",
    "диван",
    "товар",
    "каталог",
    "автомобиль",
    "машина",
    "мотоцикл",
    "телефон",
    "обои на телефон",
    "декор",
    "дизайн интерьера",
}


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
            key_lower = key.lower()
            if key_lower.startswith("utm_") or key_lower in {"from", "ref", "source", "tracking"}:
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
    if value.startswith("//"):
        return "https:" + value
    return value


def _extract_yandex_originals(response: Any) -> list[dict[str, Any]]:
    """Extract originalImage URLs from PicImageSearch's raw Yandex response."""
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
        results: list[dict[str, Any]] = []
        for site in sites:
            if not isinstance(site, dict):
                continue
            original = site.get("originalImage") if isinstance(site.get("originalImage"), dict) else {}
            url = ""
            for key in ("url", "originUrl", "origin_url", "src"):
                url = _absolute_url(original.get(key))
                if url:
                    break
            width = original.get("width")
            height = original.get("height")
            try:
                width = int(width) if width is not None else None
            except (TypeError, ValueError):
                width = None
            try:
                height = int(height) if height is not None else None
            except (TypeError, ValueError):
                height = None
            results.append({"url": url, "width": width, "height": height})
        return results
    except Exception:
        return []


def item_to_dict(
    item: Any,
    source: str,
    *,
    original_url: str = "",
    original_size: str = "",
) -> dict[str, Any]:
    title = norm_text(getattr(item, "title", ""))
    page_url = norm_text(getattr(item, "url", ""))
    thumb = norm_text(getattr(item, "thumbnail", ""))
    desc = norm_text(getattr(item, "content", ""))
    site = norm_text(getattr(item, "source", ""))
    size = original_size or norm_text(getattr(item, "size", ""))
    image_url = norm_text(original_url) or thumb
    return {
        "image_url": image_url,
        "preview_url": thumb,
        "page_url": page_url,
        "title": title or f"Результат {source}",
        "description": desc,
        "source": source,
        "kind": "reverse_image",
        "site": site,
        "size": size,
        "image_quality": "original" if original_url else "preview_fallback",
    }


def _tokens_for_address(address: str) -> tuple[list[str], str]:
    clean = norm_text(address).lower().replace("ё", "е")
    tokens = [t for t in re.findall(r"[a-zа-я0-9-]+", clean) if len(t) >= 3]
    street_tokens = [
        t for t in tokens
        if t not in {
            "россия", "край", "город", "г", "ул", "улица", "проспект",
            "пр", "дом", "д", "новороссийск", "novorossiysk",
        }
    ]
    number_match = re.search(r"\b(\d{1,4}[а-яa-z]?)\b", clean)
    number = number_match.group(1) if number_match else ""
    return street_tokens, number


def _result_text(item: dict[str, Any]) -> str:
    return " ".join(
        norm_text(item.get(key)) for key in ("title", "description", "site", "page_url", "image_url")
    ).lower().replace("ё", "е")


def rank_novorossiysk(items: list[dict[str, Any]], address: str, limit: int = FINAL_RESULTS) -> list[dict[str, Any]]:
    address_lower = norm_text(address).lower().replace("ё", "е")
    street_tokens, house_number = _tokens_for_address(address)
    scored: list[tuple[int, int, dict[str, Any]]] = []

    for position, item in enumerate(items):
        image_url = norm_text(item.get("image_url"))
        if not image_url:
            continue
        text = _result_text(item)
        if any(city in text for city in OTHER_CITIES):
            continue

        score = max(0, 12 - position // 2)
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

        item = dict(item)
        item["candidate_rank_score"] = score
        scored.append((score, -position, item))

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for _, _, item in sorted(scored, key=lambda row: (row[0], row[1]), reverse=True):
        keys = [normalize_url(item.get("page_url")), image_identity(norm_text(item.get("image_url")))]
        key = next((k for k in keys if k), "")
        if not key or key in seen:
            continue
        seen.add(key)
        item.pop("candidate_rank_score", None)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def dedupe(items: list[dict[str, Any]], limit: int = 60) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        keys = [normalize_url(item.get("page_url")), image_identity(norm_text(item.get("image_url")))]
        key = next((k for k in keys if k), "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


async def yandex_search(image_bytes: bytes) -> dict[str, Any]:
    try:
        async with Network(timeout=REVERSE_SEARCH_TIMEOUT) as client:
            engine = Yandex(client=client, base_url="https://yandex.com")
            response = await asyncio.wait_for(engine.search(file=image_bytes), timeout=REVERSE_SEARCH_TIMEOUT)
        originals = _extract_yandex_originals(response)
        raw_items = response.raw or []
        results: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items[:REVERSE_SEARCH_RESULTS]):
            original_url = ""
            original_size = ""
            if index < len(originals):
                original_url = norm_text(originals[index].get("url"))
                width = originals[index].get("width")
                height = originals[index].get("height")
                if width and height:
                    original_size = f"{width}x{height}"
            results.append(
                item_to_dict(
                    item,
                    "Yandex Images",
                    original_url=original_url,
                    original_size=original_size,
                )
            )
        return {
            "engine": "Yandex Images",
            "ok": True,
            "results": results,
            "search_url": norm_text(getattr(response, "url", "")) or None,
            "error": None,
            "original_urls": sum(1 for x in results if x.get("image_quality") == "original"),
        }
    except Exception as exc:
        return {
            "engine": "Yandex Images",
            "ok": False,
            "results": [],
            "search_url": None,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "original_urls": 0,
        }


async def google_lens_search(image_bytes: bytes) -> dict[str, Any]:
    try:
        async with Network(timeout=REVERSE_SEARCH_TIMEOUT) as client:
            engine = GoogleLens(client=client, search_type="all", hl="ru", country="RU")
            response = await asyncio.wait_for(engine.search(file=image_bytes), timeout=REVERSE_SEARCH_TIMEOUT)
        return {
            "engine": "Google Lens",
            "ok": True,
            "results": [item_to_dict(i, "Google Lens") for i in (response.raw or [])][:REVERSE_SEARCH_RESULTS],
            "search_url": norm_text(getattr(response, "url", "")) or None,
            "error": None,
        }
    except Exception as exc:
        return {
            "engine": "Google Lens",
            "ok": False,
            "results": [],
            "search_url": None,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }


async def search(image_bytes: bytes, address: str = "") -> dict[str, Any]:
    yandex, lens = await asyncio.gather(yandex_search(image_bytes), google_lens_search(image_bytes))
    raw = dedupe(yandex["results"] + lens["results"], limit=60)
    filtered = rank_novorossiysk(raw, address, limit=FINAL_RESULTS) if address else raw[:FINAL_RESULTS]
    return {
        "enabled": True,
        "engines": [yandex, lens],
        "results": filtered,
        "raw_result_count": len(raw),
    }
