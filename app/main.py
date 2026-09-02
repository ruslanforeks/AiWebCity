from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TIMEWEB_API_BASE = os.getenv("TIMEWEB_API_BASE", "https://api.timeweb.ai/v1").rstrip("/")
TIMEWEB_TOKEN = os.getenv("TIMEWEB_AI_TOKEN", "").strip()
VISION_MODEL = os.getenv("TIMEWEB_VISION_MODEL", "openai/gpt-4.1-mini")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))
PASTVU_API_URL = "https://api.pastvu.com/api2"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PASTPHOTO_BASE = "https://pastphoto.ru/place/"
CITY_RU = "Новороссийск"
CITY_EN = "Novorossiysk"
OTHER_CITIES = {"геленджик", "gelendzhik", "анапа", "anapa", "краснодар", "krasnodar", "сочи", "sochi", "туапсе", "tuapse", "майкоп", "армавир", "керчь", "севастополь", "симферополь"}


def require_token() -> None:
    if not TIMEWEB_TOKEN:
        raise HTTPException(503, "TIMEWEB_AI_TOKEN не настроен на сервере.")


def data_url(image_bytes: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and (item.get("text") or item.get("output_text")):
                out.append(str(item.get("text") or item.get("output_text")))
        return "".join(out)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("output_text") or "")
    return ""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if match:
        text = match.group(1)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                pass
    return {"raw": text}


async def timeweb_chat(messages: list[dict[str, Any]], model: str, *, temperature: float = 0.1) -> dict[str, Any]:
    require_token()
    headers = {"Authorization": f"Bearer {TIMEWEB_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    async with httpx.AsyncClient(timeout=240) as client:
        response = await client.post(f"{TIMEWEB_API_BASE}/chat/completions", headers=headers, json=payload)
    if response.status_code >= 400:
        raise HTTPException(502, f"Timeweb AI error {response.status_code}: {response.text[:800]}")
    return response.json()


async def analyze_photo(image_bytes: bytes, content_type: str, address: str, year: str) -> dict[str, Any]:
    prompt = f"""
Ты — компьютерное зрение проекта AiWebCity. Проект работает ТОЛЬКО по Новороссийску.
Тебе дана фотография здания, нескольких зданий, улицы или городской сцены.
Адрес пользователя — только слабая подсказка и может быть неправильным.

НЕ называй конкретное здание и НЕ придумывай адрес. Твоя задача — извлечь доказательства для внешнего поиска.
visible_text — только реально читаемые слова и цифры на фото.
address_clues — только реально видимые адресные признаки.
landmark_clues — видимые ориентиры.
visual_fingerprint — устойчивые признаки фасада/сцены.
search_queries — запросы только из наблюдаемых признаков, без придуманных названий.
Если объектов несколько, перечисли их признаки, не выбирая выдуманный главный объект.

Адрес-подсказка: {address or 'не указан'}
Период: {year or 'не указан'}

Верни строго JSON:
{{
  "what_is_visible":"...",
  "object_type":"building|street|intersection|landmark|multiple_buildings|unknown",
  "visible_text":["..."],
  "address_clues":[{{"text":"...","type":"street|house_number|sign|other","confidence":"high|medium"}}],
  "landmark_clues":["..."],
  "visual_fingerprint":["..."],
  "search_queries":["до 6 запросов"],
  "limits":"..."
}}
"""
    body = await timeweb_chat([
        {"role": "system", "content": "Отвечай на русском. Не угадывай конкретные здания и адреса. Извлекай только наблюдаемые признаки."},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url(image_bytes, content_type)}}]},
    ], VISION_MODEL)
    result = extract_json(extract_text(body.get("choices", [{}])[0].get("message", {}).get("content", "")))
    for key in ("visible_text", "address_clues", "landmark_clues", "visual_fingerprint", "search_queries"):
        if not isinstance(result.get(key), list):
            result[key] = []
    return result


async def geocode_address(address: str) -> dict[str, Any] | None:
    if not norm_text(address):
        return None
    try:
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": "AiWebCity/0.9 (+https://aiweb.su/)"}) as client:
            response = await client.get(NOMINATIM_URL, params={"q": address, "format": "jsonv2", "limit": 1, "addressdetails": 1})
        if response.status_code >= 400:
            return None
        rows = response.json()
        if not rows:
            return None
        row = rows[0]
        return {"lat": float(row["lat"]), "lon": float(row["lon"]), "display_name": row.get("display_name", "")}
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def extract_year(text: str) -> int | None:
    years = [int(x) for x in re.findall(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)", text)]
    return min(years) if years else None

_STREET_PREFIX = r"(?:ул(?:ица)?|проспект|просп\.?|пр-т|пер(?:еулок)?|наб(?:ережная)?|площадь|пл\.?|шоссе|бульвар|бул\.?|проезд|тупик|аллея)"
_ADDRESS_RE = re.compile(rf"\b{_STREET_PREFIX}\.?\s+([\wА-Яа-яЁёA-Za-z][\wА-Яа-яЁёA-Za-z'’-]*(?:\s+[\wА-Яа-яЁёA-Za-z][\wА-Яа-яЁёA-Za-z'’-]*){{0,3}})\s*[,№\s]+\s*(\d{{1,4}}[А-Яа-яЁёA-Za-z]?(?:/\d{{1,4}})?)\b", re.I)


def normalize_street(value: str) -> str:
    value = norm_text(value).lower().replace("ё", "е")
    value = re.sub(rf"^{_STREET_PREFIX}\.?\s*", "", value, flags=re.I)
    return value.strip(" ,.")


def normalize_house(value: str) -> str:
    return norm_text(value).lower().replace("ё", "е").strip()


def extract_addresses(text: str) -> list[dict[str, str]]:
    out = []
    for match in _ADDRESS_RE.finditer(norm_text(text).replace("ё", "е")):
        street, house = normalize_street(match.group(1)), normalize_house(match.group(2))
        if street and house:
            out.append({"street": street, "house": house, "display": f"{street}, {house}"})
    return out


def parse_hint_address(address: str) -> tuple[str, str]:
    found = extract_addresses(address)
    if found:
        return found[0]["street"], found[0]["house"]
    clean = norm_text(address).lower().replace("ё", "е")
    nums = re.findall(r"\b\d{1,4}[а-яa-z]?\b", clean)
    return normalize_street(clean), nums[0] if nums else ""


def build_identity_evidence(analysis: dict[str, Any], reverse_items: list[dict[str, Any]], user_address: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    def add(street: str, house: str, source: str, text: str, points: int) -> None:
        key = (normalize_street(street), normalize_house(house))
        if not key[0] or not key[1]:
            return
        row = groups.setdefault(key, {"street": key[0], "house": key[1], "score": 0, "sources": [], "evidence": []})
        row["score"] += points
        if source not in row["sources"]:
            row["sources"].append(source)
        snippet = norm_text(text)
        if snippet and snippet not in row["evidence"]:
            row["evidence"].append(snippet[:280])

    visible = " ".join(norm_text(v) for v in analysis.get("visible_text", []) if norm_text(v))
    for clue in analysis.get("address_clues", []):
        if isinstance(clue, dict):
            clue_text = norm_text(clue.get("text"))
            for match in extract_addresses(clue_text):
                add(match["street"], match["house"], "photo", clue_text, 90)
    for match in extract_addresses(visible):
        add(match["street"], match["house"], "photo", visible, 80)

    hint_street, hint_house = parse_hint_address(user_address)
    if hint_street and hint_house:
        add(hint_street, hint_house, "user_hint", user_address, 8)

    for idx, item in enumerate(reverse_items):
        text = " ".join(norm_text(item.get(k)) for k in ("title", "description", "site", "page_url", "image_url"))
        lowered = text.lower().replace("ё", "е")
        if any(city in lowered for city in OTHER_CITIES):
            continue
        for match in extract_addresses(text):
            add(match["street"], match["house"], "reverse_search", text, max(18, 34 - idx))

    rows = list(groups.values())
    rows.sort(key=lambda r: (-r["score"], r["street"], r["house"]))
    return rows[:8]


async def geocode_novorossiysk_candidate(address: str) -> dict[str, Any] | None:
    for query in (f"{address}, Новороссийск, Россия", f"{address}, Novorossiysk, Russia", address):
        geo = await geocode_address(query)
        if geo:
            text = norm_text(geo.get("display_name")).lower().replace("ё", "е")
            if CITY_RU.lower() in text or CITY_EN.lower() in text:
                return geo
    return None


async def resolve_identity(analysis: dict[str, Any], reverse_items: list[dict[str, Any]], user_address: str) -> dict[str, Any]:
    resolved = []
    for candidate in build_identity_evidence(analysis, reverse_items, user_address):
        display = f"{candidate['street']}, {candidate['house']}"
        geo = await geocode_novorossiysk_candidate(display)
        if not geo:
            continue
        sources = set(candidate["sources"])
        score = candidate["score"] + (35 if "photo" in sources else 0) + min(30, max(0, len(sources) - 1) * 15)
        strong = ("photo" in sources and "reverse_search" in sources) or ("reverse_search" in sources and len(sources) >= 2)
        confidence = "high" if strong and score >= 100 else "medium" if strong or score >= 60 else "low"
        resolved.append({"address": f"Новороссийск, {display}", "street": candidate["street"], "house": candidate["house"], "confidence": confidence, "score": score, "strong": strong, "sources": candidate["sources"], "evidence": candidate["evidence"][:4], "geocode": geo})
    resolved.sort(key=lambda x: (not x["strong"], -x["score"]))
    best = resolved[0] if resolved else None
    return {"status": "identified" if best and best["strong"] else "uncertain", "best": best, "candidates": resolved[:5]}


async def pastvu_search(lat: float, lon: float, year_to: int = 1999) -> list[dict[str, Any]]:
    params = {"method": "photo.giveNearestPhotos", "params": json.dumps({"geo": [lat, lon], "distance": 400, "year2": year_to, "limit": 30}, separators=(",", ":"))}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(PASTVU_API_URL, params=params)
        if response.status_code >= 400:
            return []
        out = []
        for item in response.json().get("result", {}).get("photo", []):
            fid = norm_text(item.get("file"))
            if not fid:
                continue
            try:
                year = int(item.get("year")) if item.get("year") is not None else None
            except (TypeError, ValueError):
                year = None
            out.append({"image_url": f"https://img.pastvu.com/d/{fid}", "page_url": f"https://pastvu.com/p/{item.get('cid')}" if item.get("cid") else "https://pastvu.com/", "title": norm_text(item.get("title")) or "Историческая фотография", "description": norm_text(item.get("desc") or item.get("description")), "source": "PastVu", "kind": "historical", "year": year, "distance_m": item.get("distance")})
        return out
    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        return []


async def wikimedia_search(queries: list[str], limit: int = 12) -> list[dict[str, Any]]:
    merged = {}
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "AiWebCity/0.9"}) as client:
        for query in queries[:6]:
            if not norm_text(query):
                continue
            params = {"action": "query", "generator": "search", "gsrnamespace": 6, "gsrsearch": query, "gsrlimit": 8, "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 900, "format": "json", "formatversion": 2}
            try:
                response = await client.get(WIKIMEDIA_API_URL, params=params)
                if response.status_code >= 400:
                    continue
                for page in response.json().get("query", {}).get("pages", []):
                    info = (page.get("imageinfo") or [{}])[0]
                    image_url = norm_text(info.get("thumburl") or info.get("url"))
                    if not image_url:
                        continue
                    page_url = norm_text(page.get("canonicalurl"))
                    ext = info.get("extmetadata") or {}
                    title = norm_text(page.get("title"))
                    desc = re.sub(r"<[^>]+>", " ", norm_text((ext.get("ImageDescription") or {}).get("value")))
                    merged[page_url or image_url] = {"image_url": image_url, "page_url": page_url or "https://commons.wikimedia.org/", "title": title or "Wikimedia Commons", "description": desc, "source": "Wikimedia Commons", "kind": "similar", "year": extract_year(f"{title} {desc}")}
            except (httpx.HTTPError, ValueError, json.JSONDecodeError):
                continue
    return list(merged.values())[:limit]


def build_pastphoto_link(address: str) -> str:
    return f"{PASTPHOTO_BASE}{quote(norm_text(address).replace(', Россия', ''), safe='')}"


def dedupe_results(items: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    seen = set(); out = []
    for item in items:
        key = norm_text(item.get("image_url")) or norm_text(item.get("page_url"))
        if not key or key in seen:
            continue
        seen.add(key); out.append(item)
        if len(out) >= limit:
            break
    return out


def identity_summary(identity: dict[str, Any]) -> dict[str, Any]:
    best = identity.get("best") if isinstance(identity, dict) else None
    if not best:
        return {"status": "uncertain", "address": None, "confidence": "low", "evidence": [], "sources": []}
    return {"status": identity.get("status", "uncertain"), "address": best.get("address"), "confidence": best.get("confidence"), "evidence": best.get("evidence", []), "sources": best.get("sources", [])}
