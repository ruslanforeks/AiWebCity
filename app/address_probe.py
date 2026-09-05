"""Проверка адреса поиском по нему картинок и визуальным сравнением.

Раньше адрес подтверждался тем, что его называет подпись найденной страницы.
Но подпись — это текст с сайта, который мы не проверяли, и на реальном здании
(ул. Куникова, 43) подписей с адресом среди совпадений просто не оказалось,
хотя сами фотографии были правильные.

Здесь всё наоборот: мы берём адрес-гипотезу, ищем в Яндексе фотографии ПО ЭТОМУ
АДРЕСУ и сравниваем их с фотографией пользователя. Совпало — адрес доказан
изображением. Не совпало — гипотеза отбрасывается, даже если её дружно называли
все подписи.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from lxml import html as lxml_html

from .yandex_cbir import BROWSER_HEADERS

YANDEX_SEARCH_URL = "https://yandex.com/images/search"
PROBE_IMAGES = 4


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _absolute(value: Any) -> str:
    value = _norm(value)
    return "https:" + value if value.startswith("//") else value


# Мусор в начале заголовков Яндекс.Карт: «Панорама: Атэк, теплоснабжение, ...»
TITLE_PREFIX_RE = re.compile(
    r"^(?:панорама|фото|отзывы\s+об?|больше\s+не\s+работает|видео|карта)\s*[:—-]?\s*",
    re.I,
)
# Категории организаций — это не название объекта, а рубрика справочника.
CATEGORY_WORDS = {
    "теплоснабжение", "колледж", "школа", "магазин", "кафе", "ресторан", "аптека", "банк",
    "отделение", "салон", "офис", "supermarket", "супермаркет", "поликлиника", "больница",
    "детский сад", "университет", "институт", "техникум", "гостиница", "парикмахерская",
}


def place_name_from_title(title: str) -> str:
    """Название организации/объекта из заголовка результата поиска.

    «Панорама: Атэк, теплоснабжение, ул. Куникова, 43, Новороссийск» -> «Атэк».
    Это не догадка модели, а текст, который поисковик привязал к фотографии;
    проверяться он всё равно будет встречным поиском по изображению.
    """
    clean = TITLE_PREFIX_RE.sub("", _norm(title))
    head = clean.split(",")[0].strip(" .:-—«»\"'")
    if len(head) < 3 or len(head) > 60:
        return ""
    lowered = head.lower()
    if lowered in CATEGORY_WORDS or re.fullmatch(r"[\d\s.,-]+", head):
        return ""
    if any(word in lowered for word in ("википедия", "wikimedia", "commons", "яндекс карты", "file:")):
        return ""
    return head


def build_query(street: str, house: str) -> str:
    street = _norm(street)
    house = _norm(house)
    parts = [p for p in (street, house) if p]
    return f"{' '.join(parts)} Новороссийск"


def _parse_serp(page_html: str) -> list[dict[str, Any]]:
    """Достаёт картинки из обычной текстовой выдачи Яндекс.Картинок."""
    rows: list[dict[str, Any]] = []
    try:
        tree = lxml_html.fromstring(page_html)
    except Exception:
        return rows
    for node in tree.xpath('//div[starts-with(@id, "ImagesApp-")]'):
        state = node.get("data-state")
        if not state:
            continue
        try:
            data = json.loads(state)
        except json.JSONDecodeError:
            continue
        entities = ((((data.get("initialState") or {}).get("serpList") or {}).get("items") or {}).get("entities") or {})
        if not isinstance(entities, dict):
            continue
        for position, entity in enumerate(entities.values()):
            if not isinstance(entity, dict):
                continue
            origin = entity.get("origin") if isinstance(entity.get("origin"), dict) else {}
            image_url = _absolute(entity.get("origUrl") or origin.get("url"))
            thumb = _absolute((entity.get("image") or "") if isinstance(entity.get("image"), str) else origin.get("previewUrl"))
            snippet = entity.get("snippet") if isinstance(entity.get("snippet"), dict) else {}
            if not image_url and not thumb:
                continue
            rows.append({
                "title": _norm(snippet.get("title")),
                "description": _norm(snippet.get("text")),
                "page_url": _norm(snippet.get("url")),
                "site": _norm(snippet.get("domain")),
                "image_url": image_url or thumb,
                "thumb_url": thumb,
                "preview_url": thumb or image_url,
                "source": "Yandex Images (поиск по адресу)",
                "kind": "address_probe",
                "match_type": "address_probe",
                "rank": position,
            })
    return rows


async def probe_by_query(query: str, *, timeout: float = 30.0, limit: int = PROBE_IMAGES) -> list[dict[str, Any]]:
    """Фотографии по произвольному текстовому запросу."""
    return await _search(query, timeout=timeout, limit=limit)


async def probe_by_name(name: str, *, timeout: float = 30.0, limit: int = PROBE_IMAGES) -> list[dict[str, Any]]:
    """Фотографии по названию объекта: «Атэк», «Новороссийский медицинский колледж»."""
    name = _norm(name)
    if len(name) < 3:
        return []
    return await _search(f"{name} Новороссийск", timeout=timeout, limit=limit)


async def probe_images(street: str, house: str, *, timeout: float = 30.0, limit: int = PROBE_IMAGES) -> list[dict[str, Any]]:
    """Фотографии, которые Яндекс отдаёт по текстовому запросу с адресом."""
    query = build_query(street, house)
    if not query.strip() or query.strip() == "Новороссийск":
        return []
    return await _search(query, timeout=timeout, limit=limit)


async def _search(query: str, *, timeout: float, limit: int) -> list[dict[str, Any]]:
    if not _norm(query):
        return []
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=BROWSER_HEADERS, follow_redirects=True) as client:
            response = await client.get(YANDEX_SEARCH_URL, params={"text": query})
        if response.status_code >= 400:
            return []
        rows = _parse_serp(response.text)
    except (httpx.HTTPError, ValueError):
        return []

    # Панорама, привязанная к точке на карте, — единственный тип результата,
    # который отвечает именно за АДРЕС. Карточка организации отвечает за
    # организацию, а у неё бывает несколько корпусов по разным адресам: так проба
    # «Рубина, 5» подтверждалась фотографиями колледжа, стоящего на Советов, 38.
    def priority(row: dict[str, Any]) -> tuple[int, int, int]:
        text = f"{row.get('title')} {row.get('site')} {row.get('page_url')}".lower()
        image = str(row.get("image_url") or "").lower()
        geo_anchored = "static-pano.maps.yandex" in image or "на карте" in text
        useful = any(word in text for word in ("панорама", "яндекс карты", "фото", "здание", "улица", "2gis", "дом"))
        bad = any(word in text for word in ("интерьер", "квартир", "планировк", "схема", "карта проезда"))
        return (1 if geo_anchored else 0, 0 if bad else (1 if not useful else 2), -int(row.get("rank", 0)))

    rows.sort(key=priority, reverse=True)
    return rows[:limit]
