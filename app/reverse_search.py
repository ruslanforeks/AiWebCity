from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from PicImageSearch import GoogleLens, Network, Yandex

REVERSE_SEARCH_TIMEOUT = 45.0
REVERSE_SEARCH_RESULTS = 24
FINAL_RESULTS = 24

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


def extract_year(text: str) -> int | None:
    years = [int(x) for x in re.findall(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)", text)]
    return min(years) if years else None


def item_to_dict(item: Any, source: str) -> dict[str, Any]:
    title = norm_text(getattr(item, "title", ""))
    page_url = norm_text(getattr(item, "url", ""))
    thumb = norm_text(getattr(item, "thumbnail", ""))
    desc = norm_text(getattr(item, "content", ""))
    site = norm_text(getattr(item, "source", ""))
    size = norm_text(getattr(item, "size", ""))
    return {"image_url": thumb, "page_url": page_url, "title": title or f"Результат {source}", "description": desc, "source": source, "kind": "reverse_image", "site": site, "size": size, "year": extract_year(" ".join([title, desc, page_url]))}


def _result_text(item: dict[str, Any]) -> str:
    return " ".join(norm_text(item.get(key)) for key in ("title", "description", "site", "page_url", "image_url")).lower().replace("ё", "е")


def rank_novorossiysk(items: list[dict[str, Any]], address: str = "", limit: int = FINAL_RESULTS) -> list[dict[str, Any]]:
    """Keep reverse-search results useful for the resolver without trusting the user hint as truth."""
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for position, original in enumerate(items):
        item = dict(original)
        image_url = norm_text(item.get("image_url"))
        if not image_url:
            continue
        text = _result_text(item)
        if any(city in text for city in OTHER_CITIES):
            continue
        score = 0
        if any(term in text for term in NOVOROSSIYSK_TERMS):
            score += 45
        if any(term in text for term in NOISE_TERMS):
            score -= 80
        # The user-entered address is deliberately only a weak prior.
        if address:
            hint = norm_text(address).lower().replace("ё", "е")
            hint_words = [w for w in re.findall(r"[a-zа-я0-9-]+", hint) if len(w) >= 4 and w not in {"новороссийск", "россия", "улица", "ул", "дом"}]
            score += min(18, sum(2 for w in hint_words if w in text))
        score += max(0, 12 - position // 2)
        scored.append((score, -position, item))

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for _, _, item in sorted(scored, key=lambda row: (row[0], row[1]), reverse=True):
        key = normalize_url(item.get("page_url")) or image_identity(norm_text(item.get("image_url")))
        if not key or key in seen:
            continue
        seen.add(key)
        item.pop("match_score", None)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def dedupe(items: list[dict[str, Any]], limit: int = 48) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = normalize_url(item.get("page_url")) or image_identity(norm_text(item.get("image_url")))
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
            response = await asyncio.wait_for(Yandex(client=client, base_url="https://yandex.com").search(file=image_bytes), timeout=REVERSE_SEARCH_TIMEOUT)
        return {"engine":"Yandex Images","ok":True,"results":[item_to_dict(i,"Yandex Images") for i in (response.raw or [])][:REVERSE_SEARCH_RESULTS],"search_url":norm_text(getattr(response,"url","")) or None,"error":None}
    except Exception as exc:
        return {"engine":"Yandex Images","ok":False,"results":[],"search_url":None,"error":f"{type(exc).__name__}: {str(exc)[:300]}"}


async def google_lens_search(image_bytes: bytes) -> dict[str, Any]:
    try:
        async with Network(timeout=REVERSE_SEARCH_TIMEOUT) as client:
            response = await asyncio.wait_for(GoogleLens(client=client, search_type="all", hl="ru", country="RU").search(file=image_bytes), timeout=REVERSE_SEARCH_TIMEOUT)
        return {"engine":"Google Lens","ok":True,"results":[item_to_dict(i,"Google Lens") for i in (response.raw or [])][:REVERSE_SEARCH_RESULTS],"search_url":norm_text(getattr(response,"url","")) or None,"error":None}
    except Exception as exc:
        return {"engine":"Google Lens","ok":False,"results":[],"search_url":None,"error":f"{type(exc).__name__}: {str(exc)[:300]}"}


async def search(image_bytes: bytes, address: str = "") -> dict[str, Any]:
    yandex, lens = await asyncio.gather(yandex_search(image_bytes), google_lens_search(image_bytes))
    raw = dedupe(yandex["results"] + lens["results"], limit=48)
    filtered = rank_novorossiysk(raw, address, limit=FINAL_RESULTS)
    return {"enabled":True,"engines":[yandex,lens],"results":filtered,"raw_result_count":len(raw)}
