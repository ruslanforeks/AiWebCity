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
# Разделитель после «ул» — либо точка (пробел уже необязателен: «Ул.Губернского 2»),
# либо обычный пробел. Раньше пробел был обязателен, и такие подписи терялись.
_SEP = r"(?:\.\s*|\s+)"
RE_PREFIXED = re.compile(
    rf"\b{STREET_PREFIX}{_SEP}([а-яa-z][а-яa-z0-9'’-]*(?:\s+[а-яa-z][а-яa-z0-9'’-]*){{0,2}})\s*[,№\s]+\s*(?:д\.?|дом\s*)?(\d{{1,4}}[а-я]?(?:/\d{{1,4}})?)\b",
    re.I,
)
# «советов 34 новороссийск» — формат подсказок Яндекса, без слова «улица».
RE_BARE = re.compile(
    r"\b([а-яa-z][а-яa-z0-9'’-]{3,}(?:\s+[а-яa-z][а-яa-z0-9'’-]{2,}){0,1})\s+(\d{1,4}[а-я]?)\s+новоросси",
    re.I,
)
RE_BARE_CITY_FIRST = re.compile(
    rf"новоросси\w*[\s,.]+{STREET_PREFIX}{_SEP}([а-яa-z][а-яa-z0-9'’-]*(?:\s+[а-яa-z][а-яa-z0-9'’-]*){{0,2}})[\s,]+(?:д\.?\s*)?(\d{{1,4}}[а-я]?)\b",
    re.I,
)

# Поле «адрес-подсказка» заведомо содержит адрес, поэтому его можно разбирать
# мягче, чем случайный текст со страницы: без слова «улица» и без названия города.
# Именно из-за строгого разбора ввод вида «куникова 43» терялся целиком.
RE_HINT = re.compile(
    rf"^\s*(?:г\.?\s*)?(?:новоросси\w*[\s,]*)?(?:{STREET_PREFIX}{_SEP})?"
    rf"([а-яa-z][а-яa-z0-9'’-]{{2,}}(?:\s+[а-яa-z][а-яa-z0-9'’-]{{2,}}){{0,2}})"
    rf"[\s,]*(?:д\.?\s*|дом\s*)?(\d{{1,4}}[а-я]?(?:/\d{{1,4}})?)\s*"
    rf"(?:[,\s]+новоросси\w*.*)?$",
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
            street = re.sub(rf"^{STREET_PREFIX}{_SEP}", "", street.strip(" ,.")).strip()
            house = house.strip(" ,.")
            if _valid_street(street) and house and (street, house) not in found:
                found.append((street, house))
    return found


def parse_user_hint(address: str) -> list[tuple[str, str]]:
    """Разбор поля «адрес-подсказка». Сначала обычными шаблонами, затем мягким."""
    found = extract_street_house(address)
    if found:
        return found
    match = RE_HINT.match(_clean(address))
    if not match:
        return []
    street = re.sub(rf"^{STREET_PREFIX}{_SEP}", "", match.group(1).strip(" ,.")).strip()
    house = match.group(2).strip(" ,.")
    return [(street, house)] if _valid_street(street) and house else []


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


def house_key(house: str) -> str:
    """«2а», «2/2», «2 » → «2». Нужно, чтобы подсказка «губернского 2а» и подпись
    «Губернского 2» считались одним и тем же домом."""
    digits = re.match(r"\d{1,4}", _clean(house))
    return digits.group(0) if digits else _clean(house)


def image_derived_addresses(yandex_tags: list[str], ocr_text: str) -> tuple[set[str], set[tuple[str, str]]]:
    """Адресные признаки, полученные ИЗ САМОЙ ФОТОГРАФИИ.

    Подсказки Яндекса и OCR построены по изображению, а подписи кандидатов — по
    страницам, которые мы не проверяли. Поэтому они и служат подтверждением.
    """
    streets: set[str] = set()
    houses: set[tuple[str, str]] = set()
    for text in list(yandex_tags) + [ocr_text]:
        for street, house in extract_street_house(text):
            streets.add(street)
            houses.add((street, house_key(house)))
    return streets, houses


def collect_evidence(
    *,
    verified_candidates: list[dict[str, Any]],
    yandex_tags: list[str],
    ocr_text: str,
    user_address: str,
) -> list[dict[str, Any]]:
    """Складывает адресные гипотезы с весами, источниками и подтверждениями."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    def add(street: str, house: str, source: str, weight: float, snippet: str, candidate_id: str = "") -> None:
        key = (street, house)
        row = groups.setdefault(key, {
            "street": street, "house": house, "score": 0.0,
            "sources": [], "evidence": [], "candidate_ids": set(),
        })
        row["score"] += weight
        if source not in row["sources"]:
            row["sources"].append(source)
        if candidate_id:
            row["candidate_ids"].add(candidate_id)
        snippet = norm_text(snippet)[:200]
        if snippet and snippet not in row["evidence"]:
            row["evidence"].append(snippet)

    # Подписи визуально подтверждённых кандидатов — основной вес.
    for position, candidate in enumerate(verified_candidates):
        confidence = float(candidate.get("confidence", 0.0))
        text = " ".join(norm_text(candidate.get(k)) for k in ("title", "description", "page_url"))
        weight = (55.0 + 35.0 * confidence) * (0.85 ** position)
        candidate_id = norm_text(candidate.get("image_url")) or f"cand{position}"
        for street, house in extract_street_house(text):
            add(street, house, "visual_match", weight, text, candidate_id)

    for position, tag in enumerate(yandex_tags[:6]):
        for street, house in extract_street_house(tag):
            add(street, house, "yandex_tag", max(12.0, 40.0 - position * 6.0), tag)

    for street, house in extract_street_house(ocr_text):
        add(street, house, "photo_ocr", 45.0, ocr_text)

    # Подсказка пользователя — слабый тай-брейк, сама себя подтверждать не должна.
    for street, house in parse_user_hint(user_address):
        add(street, house, "user_hint", 8.0, user_address)

    photo_streets, photo_houses = image_derived_addresses(yandex_tags, ocr_text)
    hint_streets = {street for street, _ in parse_user_hint(user_address)}

    rows = list(groups.values())
    for row in rows:
        key = (row["street"], house_key(row["house"]))
        independent_candidates = len(row["candidate_ids"])

        # Совпадение с признаком, извлечённым из самой фотографии.
        if key in photo_houses:
            row["score"] += 60.0
        elif row["street"] in photo_streets:
            row["score"] += 30.0

        row["score"] += 25.0 * max(0, len(row["sources"]) - 1)
        row["score"] += 18.0 * max(0, min(independent_candidates, 3) - 1)

        # Улица должна подтверждаться независимым источником: подсказкой Яндекса,
        # OCR с фотографии или адресом пользователя. Иначе адрес остаётся догадкой,
        # даже если все подписи кандидатов дружно называют одно и то же место —
        # так система выдавала здание другого корпуса того же колледжа в 2 км.
        # Подсказка пользователя подтверждает улицу только тогда, когда её назвал
        # ещё кто-то: иначе любой введённый адрес подтверждал бы сам себя.
        hint_supported = row["street"] in hint_streets and any(src != "user_hint" for src in row["sources"])
        row["street_corroborated"] = row["street"] in photo_streets or hint_supported
        row["house_corroborated"] = key in photo_houses
        row["independent_candidates"] = independent_candidates
        row.pop("candidate_ids", None)

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
                    "street_corroborated": bool(row.get("street_corroborated")),
                    "house_corroborated": bool(row.get("house_corroborated")),
                    "independent_candidates": row.get("independent_candidates", 0),
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
                    "street_corroborated": False,
                    "house_corroborated": False,
                    "independent_candidates": 0,
                }
    return None


async def geocode_single(street: str, house: str) -> dict[str, Any] | None:
    """Геокодирует одну пару «улица+дом». Нужно для проверки адреса поиском."""
    street = norm_text(street)
    house = norm_text(house)
    if not street:
        return None
    query = f"{street} {house}, {CITY}, Россия" if house else f"{street}, {CITY}, Россия"
    async with httpx.AsyncClient(timeout=20, headers=NOMINATIM_HEADERS) as client:
        for entry in await _nominatim(client, query):
            display = norm_text(entry.get("display_name"))
            if not _is_novorossiysk(display):
                continue
            address = entry.get("address") if isinstance(entry.get("address"), dict) else {}
            found_street = norm_text(address.get("road") or address.get("pedestrian") or address.get("residential"))
            found_house = norm_text(address.get("house_number"))
            if not found_street:
                continue
            return {
                "lat": float(entry["lat"]),
                "lon": float(entry["lon"]),
                "display_name": display,
                "street": found_street,
                "house_number": found_house or house,
                "house_number_verified": bool(found_house),
                "matched_hypothesis": f"{street} {house}".strip(),
                "precision": "house" if found_house else "street",
                "street_corroborated": True,
                "house_corroborated": True,
                "sources": ["address_probe"],
                "evidence": [],
                "independent_candidates": 0,
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
    # Адрес считается подтверждённым только если улицу называет независимый
    # источник — подсказка Яндекса, OCR с фотографии или сам пользователь.
    confirmed = bool(location) and bool(location.get("street_corroborated"))
    return {
        "location": location,
        "address": pretty_address(location),
        "confirmed": confirmed,
        "hypotheses": [
            {
                "street": r["street"], "house": r["house"], "score": round(r["score"], 1),
                "sources": r["sources"], "street_corroborated": r.get("street_corroborated"),
                "independent_candidates": r.get("independent_candidates", 0),
            }
            for r in rows[:5]
        ],
        "place_names": place_names,
    }
