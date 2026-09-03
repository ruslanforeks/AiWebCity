"""Определение адреса ПОСЛЕ визуального подтверждения объекта.

Принцип проекта: адрес — не доказательство, а перепроверка. Поэтому сюда
попадают уже подтверждённые визуально кандидаты, и только их подписи имеют
большой вес. Отдельно учитываются подсказки самого Яндекса (cbirTags) и текст,
распознанный на фотографии (OCR) — они получены из изображения, а не из
непроверенной страницы.

Адрес пользователя намеренно имеет минимальный вес: он не должен сам себя
подтверждать.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx

from .main import USER_AGENT, norm_text

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {"User-Agent": USER_AGENT}
# Nominatim разрешает не больше одного запроса в секунду.
NOMINATIM_MIN_INTERVAL = 1.1
_last_nominatim_call = 0.0
_nominatim_lock = asyncio.Lock()

CITY = "Новороссийск"
# Ограничивающий прямоугольник Новороссийска: отсекает совпадения из других городов.
VIEWBOX = "37.6300,44.8300,37.9200,44.6000"

STREET_PREFIX = r"(?:ул(?:ица)?|проспект|просп|пр-т|пер(?:еулок)?|наб(?:ережная)?|площадь|пл|шоссе|бульвар|бул|проезд|тупик|аллея)"

# «ул. Советов, 34» / «улица Свободы 23» / «Советов, д. 34»
RE_PREFIXED = re.compile(
    rf"\b{STREET_PREFIX}\.?\s+([а-яa-z][а-яa-z0-9'’-]*(?:\s+[а-яa-z][а-яa-z0-9'’-]*){{0,2}})\s*[,№\s]+\s*(?:д\.?|дом\s*)?(\d{{1,4}}[а-я]?(?:/\d{{1,4}})?)\b",
    re.I,
)
# «советов 34 новороссийск» — формат подсказок Яндекса, без слова «улица».
RE_BARE = re.compile(
    r"\b([а-яa-z][а-яa-z0-9'’-]{3,}(?:\s+[а-яa-z][а-яa-z0-9'’-]{2,}){0,1})\s+(\d{1,4}[а-я]?)\s+новоросси",
    re.I,
)
RE_BARE_CITY_FIRST = re.compile(
    rf"новоросси\w*\s+{STREET_PREFIX}\.?\s+([а-яa-z][а-яa-z0-9'’-]*(?:\s+[а-яa-z][а-яa-z0-9'’-]*){{0,2}})\s+(\d{{1,4}}[а-я]?)\b",
    re.I,
)

STOP_STREET_WORDS = {
    "новороссийск", "россия", "край", "город", "фото", "карта", "дом", "квартира", "купить",
    "продажа", "аренда", "панорама", "яндекс", "google", "wikimedia", "commons", "объявления",
}


def _clean(text: str) -> str:
    return norm_text(text).lower().replace("ё", "е")


def _valid_street(street: str) -> bool:
    street = street.strip()
    if len(street) < 4:
        return False
    words = street.split()
    return not any(word in STOP_STREET_WORDS for word in words)


def extract_street_house(text: str) -> list[tuple[str, str]]:
    """Достаёт пары (улица, дом) из произвольного текста."""
    text = _clean(text)
    found: list[tuple[str, str]] = []
    for pattern in (RE_PREFIXED, RE_BARE_CITY_FIRST, RE_BARE):
        for street, house in pattern.findall(text):
            street = re.sub(rf"^{STREET_PREFIX}\.?\s*", "", street.strip(" ,.")).strip()
            house = house.strip(" ,.")
            if _valid_street(street) and house and (street, house) not in found:
                found.append((street, house))
    return found


def extract_place_names(tags: list[str]) -> list[str]:
    """Подсказки-названия без номера дома: «новороссийский медицинский колледж»."""
    names: list[str] = []
    for tag in tags:
        clean = _clean(tag)
        if re.search(r"\d", clean):
            continue
        if len(clean.split()) < 2 or len(clean) < 10:
            continue
        if clean in {"дом", "здание"}:
            continue
        names.append(clean)
    return names[:3]


def collect_evidence(
    *,
    verified_candidates: list[dict[str, Any]],
    yandex_tags: list[str],
    ocr_text: str,
    user_address: str,
) -> list[dict[str, Any]]:
    """Складывает адресные гипотезы с весами и источниками."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    def add(street: str, house: str, source: str, weight: float, snippet: str) -> None:
        key = (street, house)
        row = groups.setdefault(key, {"street": street, "house": house, "score": 0.0, "sources": [], "evidence": []})
        row["score"] += weight
        if source not in row["sources"]:
            row["sources"].append(source)
        snippet = norm_text(snippet)[:200]
        if snippet and snippet not in row["evidence"]:
            row["evidence"].append(snippet)

    # Подписи визуально подтверждённых кандидатов — самое сильное свидетельство.
    for position, candidate in enumerate(verified_candidates):
        confidence = float(candidate.get("confidence", 0.0))
        text = " ".join(norm_text(candidate.get(k)) for k in ("title", "description", "page_url"))
        weight = (55.0 + 35.0 * confidence) * (0.85 ** position)
        for street, house in extract_street_house(text):
            add(street, house, "visual_match", weight, text)

    # Подсказки Яндекса получены из самой картинки, но не проверены визуально.
    for position, tag in enumerate(yandex_tags[:6]):
        for street, house in extract_street_house(tag):
            add(street, house, "yandex_tag", max(12.0, 40.0 - position * 6.0), tag)

    # Текст, реально видимый на фотографии.
    for street, house in extract_street_house(ocr_text):
        add(street, house, "photo_ocr", 45.0, ocr_text)

    # Подсказка пользователя — только как слабый тай-брейк.
    for street, house in extract_street_house(user_address):
        add(street, house, "user_hint", 8.0, user_address)

    rows = list(groups.values())
    for row in rows:
        # Согласие нескольких независимых источников важнее одного громкого.
        row["score"] += 25.0 * max(0, len(row["sources"]) - 1)
    rows.sort(key=lambda r: -r["score"])
    return rows[:8]


async def _nominatim(client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
    global _last_nominatim_call
    async with _nominatim_lock:
        wait = NOMINATIM_MIN_INTERVAL - (time.monotonic() - _last_nominatim_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_nominatim_call = time.monotonic()
    try:
        response = await client.get(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 5,
                "addressdetails": 1,
                "viewbox": VIEWBOX,
                "bounded": 1,
                "countrycodes": "ru",
            },
        )
        if response.status_code >= 400:
            return []
        rows = response.json()
        return rows if isinstance(rows, list) else []
    except (httpx.HTTPError, ValueError):
        return []


def _is_novorossiysk(text: str) -> bool:
    value = _clean(text)
    return "новороссийск" in value or "novorossiysk" in value


async def geocode_candidates(rows: list[dict[str, Any]], place_names: list[str]) -> dict[str, Any] | None:
    """Проверяет гипотезы через OpenStreetMap. Возвращает лучшую подтверждённую."""
    async with httpx.AsyncClient(timeout=20, headers=NOMINATIM_HEADERS) as client:
        for row in rows[:5]:
            query = f"{row['street']} {row['house']}, {CITY}, Россия"
            for entry in await _nominatim(client, query):
                display = norm_text(entry.get("display_name"))
                if not _is_novorossiysk(display):
                    continue
                address = entry.get("address") if isinstance(entry.get("address"), dict) else {}
                street = norm_text(address.get("road") or address.get("pedestrian") or address.get("residential"))
                house = norm_text(address.get("house_number"))
                if not street:
                    continue
                return {
                    "lat": float(entry["lat"]),
                    "lon": float(entry["lon"]),
                    "display_name": display,
                    "street": street,
                    "house_number": house or row["house"],
                    "house_number_verified": bool(house),
                    "matched_hypothesis": f"{row['street']} {row['house']}",
                    "score": row["score"],
                    "sources": row["sources"],
                    "evidence": row["evidence"][:3],
                    "precision": "house" if house else "street",
                }

        # Ни одна пара «улица+дом» не подтвердилась — пробуем именованный объект.
        for name in place_names[:2]:
            for entry in await _nominatim(client, f"{name}, {CITY}, Россия"):
                display = norm_text(entry.get("display_name"))
                if not _is_novorossiysk(display):
                    continue
                address = entry.get("address") if isinstance(entry.get("address"), dict) else {}
                street = norm_text(address.get("road") or address.get("pedestrian"))
                return {
                    "lat": float(entry["lat"]),
                    "lon": float(entry["lon"]),
                    "display_name": display,
                    "street": street,
                    "house_number": norm_text(address.get("house_number")),
                    "house_number_verified": bool(address.get("house_number")),
                    "matched_hypothesis": name,
                    "score": 30.0,
                    "sources": ["yandex_tag_place_name"],
                    "evidence": [name],
                    "precision": "place",
                }
    return None


def pretty_address(location: dict[str, Any] | None) -> str:
    if not location:
        return ""
    street = norm_text(location.get("street"))
    house = norm_text(location.get("house_number"))
    if street and house:
        return f"{CITY}, {street}, {house}"
    if street:
        return f"{CITY}, {street}"
    return norm_text(location.get("display_name"))


async def resolve(
    *,
    verified_candidates: list[dict[str, Any]],
    yandex_tags: list[str],
    ocr_text: str,
    user_address: str,
) -> dict[str, Any]:
    rows = collect_evidence(
        verified_candidates=verified_candidates,
        yandex_tags=yandex_tags,
        ocr_text=ocr_text,
        user_address=user_address,
    )
    place_names = extract_place_names(yandex_tags)
    location = await geocode_candidates(rows, place_names)
    return {
        "location": location,
        "address": pretty_address(location),
        "hypotheses": [{"street": r["street"], "house": r["house"], "score": round(r["score"], 1), "sources": r["sources"]} for r in rows[:5]],
        "place_names": place_names,
    }
