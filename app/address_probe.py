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


async def probe_images(street: str, house: str, *, timeout: float = 30.0, limit: int = PROBE_IMAGES) -> list[dict[str, Any]]:
    """Фотографии, которые Яндекс отдаёт по текстовому запросу с адресом."""
    query = build_query(street, house)
    if not query.strip() or query.strip() == "Новороссийск":
        return []
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=BROWSER_HEADERS, follow_redirects=True) as client:
            response = await client.get(YANDEX_SEARCH_URL, params={"text": query})
        if response.status_code >= 400:
            return []
        rows = _parse_serp(response.text)
    except (httpx.HTTPError, ValueError):
        return []

    # Панорамы и карточки организаций — самые полезные: на них здание целиком
    # и снято с улицы, а не интерьер и не документ.
    def priority(row: dict[str, Any]) -> tuple[int, int]:
        text = f"{row.get('title')} {row.get('site')}".lower()
        good = any(word in text for word in ("панорама", "яндекс карты", "фото", "здание", "улица", "2gis", "дом"))
        bad = any(word in text for word in ("интерьер", "квартир", "планировк", "схема", "карта проезда"))
        return (0 if bad else (1 if not good else 2), -int(row.get("rank", 0)))

    rows.sort(key=priority, reverse=True)
    return rows[:limit]
