"""Низкоуровневый клиент Yandex CBIR (поиск по изображению).

PicImageSearch читает из ответа Яндекса только `initialState.cbirSites.sites` —
это блок «сайты, где встречается ЭТА ЖЕ картинка». Для фотографии, которой нет
в интернете (обычный кадр с телефона), он пуст, и весь сервис получает ноль
кандидатов.

В том же самом ответе лежит гораздо больше:

    cbirSimilar.thumbs   — до 40 визуально похожих изображений
    cbirTags.tags        — текстовые догадки Яндекса («губернского 1 новороссийск»)
    cbirPreview.crops    — bbox объектов от детектора Яндекса («здание», «машина»)
    cbirOcr.plainText    — текст, распознанный на фотографии

Этот модуль делает ровно один HTTP-запрос и возвращает всё перечисленное.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
from lxml import html as lxml_html

YANDEX_SEARCH_URL = "https://yandex.com/images/search"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ru,en;q=0.9",
}

# Категории детектора Яндекса, которые для нас означают «архитектурный объект».
BUILDING_CATEGORIES = {"здание", "дом", "строение", "архитектура", "building", "house"}

CAPTCHA_MARKERS = ("showcaptcha", "smartcaptcha", "captcha-container", "Подтвердите, что запросы отправляли вы")


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _absolute(value: Any) -> str:
    value = _norm(value)
    if value.startswith("//"):
        return "https:" + value
    return value


def _deep(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _original_from_link(link_url: str) -> str:
    """Из ссылки похожего изображения достаёт полноразмерный оригинал.

    linkUrl выглядит так:
        /images/search?img_url=https%3A%2F%2F...%2Fphoto.jpg&cbir_id=...&cbir_page=similar
    Параметр img_url — это и есть оригинал в полном разрешении.
    """
    if not link_url:
        return ""
    try:
        query = parse_qs(urlsplit(link_url).query)
    except ValueError:
        return ""
    for key in ("img_url", "url"):
        values = query.get(key) or []
        for value in values:
            value = _absolute(value)
            if value.startswith("http") and "avatars.mds.yandex.net/get-images-cbir" not in value:
                return value
    return ""


def yandex_image_id(url: str) -> str:
    """Идентификатор изображения по URL превью Яндекса (avatars.mds.yandex.net/i?id=...)."""
    if not url:
        return ""
    match = re.search(r"[?&]id=([A-Za-z0-9_%\-]{16,})", url)
    return match.group(1).lower() if match else ""


def _parse_state(page_html: str) -> dict[str, Any]:
    tree = lxml_html.fromstring(page_html)
    for node in tree.xpath('//div[starts-with(@id, "ImagesApp-")]'):
        state = node.get("data-state")
        if state:
            try:
                return json.loads(state)
            except json.JSONDecodeError:
                continue
    return {}


def _parse_sites(initial: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, site in enumerate(_deep(initial, "cbirSites.sites") or []):
        if not isinstance(site, dict):
            continue
        original = site.get("originalImage") if isinstance(site.get("originalImage"), dict) else {}
        thumb = site.get("thumb") if isinstance(site.get("thumb"), dict) else {}
        rows.append({
            "title": _norm(site.get("title")),
            "description": _norm(site.get("description")),
            "page_url": _absolute(site.get("url")),
            "site": _norm(site.get("domain")),
            "image_url": _absolute(original.get("url")),
            "thumb_url": _absolute(thumb.get("url")),
            "width": original.get("width"),
            "height": original.get("height"),
            "match_type": "exact",
            "rank": position,
        })
    return rows


def _parse_similar(initial: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, thumb in enumerate(_deep(initial, "cbirSimilar.thumbs") or []):
        if not isinstance(thumb, dict):
            continue
        thumb_url = _absolute(thumb.get("imageUrl"))
        link_url = _norm(thumb.get("linkUrl"))
        original = _original_from_link(link_url)
        if not original and not thumb_url:
            continue
        rows.append({
            "title": _norm(thumb.get("title")),
            "description": "",
            "page_url": ("https://yandex.com" + link_url) if link_url.startswith("/") else link_url,
            "site": urlsplit(original).netloc if original else "",
            "image_url": original or thumb_url,
            "thumb_url": thumb_url,
            "width": thumb.get("width"),
            "height": thumb.get("height"),
            "match_type": "similar",
            "rank": position,
        })
    return rows


def _parse_crops(initial: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for crop in _deep(initial, "cbirPreview.crops") or []:
        if not isinstance(crop, dict):
            continue
        box = crop.get("orig") if isinstance(crop.get("orig"), dict) else {}
        try:
            x0, y0, x1, y1 = (float(box["x0"]), float(box["y0"]), float(box["x1"]), float(box["y1"]))
        except (KeyError, TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append({
            "category": _norm(crop.get("category")).lower(),
            "box": (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)),
            "area": (min(1.0, x1) - max(0.0, x0)) * (min(1.0, y1) - max(0.0, y0)),
        })
    rows.sort(key=lambda row: row["area"], reverse=True)
    return rows


def _parse_tags(initial: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for tag in _deep(initial, "cbirTags.tags") or []:
        text = _norm(tag.get("text")) if isinstance(tag, dict) else _norm(tag)
        if text and text not in tags:
            tags.append(text)
    return tags


def parse_cbir_html(page_html: str, final_url: str = "") -> dict[str, Any]:
    """Разбирает HTML страницы результатов Яндекса в структурированный результат."""
    lowered = f"{final_url} {page_html[:4000]}".lower()
    if any(marker.lower() in lowered for marker in CAPTCHA_MARKERS):
        return {"ok": False, "error": "captcha", "captcha": True, "sites": [], "similar": [], "tags": [], "crops": [], "ocr_text": "", "cbir_id": "", "search_url": final_url}

    state = _parse_state(page_html)
    initial = state.get("initialState") if isinstance(state, dict) else None
    if not isinstance(initial, dict):
        return {"ok": False, "error": "no_data_state", "captcha": False, "sites": [], "similar": [], "tags": [], "crops": [], "ocr_text": "", "cbir_id": "", "search_url": final_url}

    return {
        "ok": True,
        "error": None,
        "captcha": False,
        "cbir_id": _norm(_deep(initial, "cbirPreview.cbirId")),
        "search_url": final_url,
        "sites": _parse_sites(initial),
        "similar": _parse_similar(initial),
        "tags": _parse_tags(initial),
        "crops": _parse_crops(initial),
        "ocr_text": _norm(_deep(initial, "cbirOcr.plainText")),
    }


def _empty(error: str) -> dict[str, Any]:
    return {"ok": False, "error": error, "captcha": error == "captcha", "sites": [], "similar": [], "tags": [], "crops": [], "ocr_text": "", "cbir_id": "", "search_url": ""}


async def cbir_search(image_bytes: bytes, *, timeout: float = 45.0, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    """Один запрос к Яндексу по картинке. Никогда не бросает исключение."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout, headers=BROWSER_HEADERS, follow_redirects=True, http2=False)
    try:
        response = await client.post(
            YANDEX_SEARCH_URL,
            params={"rpt": "imageview", "cbir_page": "sites"},
            data={"prg": 1},
            files={"upfile": ("image.jpg", image_bytes, "image/jpeg")},
        )
        if response.status_code >= 400:
            return _empty(f"http_{response.status_code}")
        return parse_cbir_html(response.text, str(response.url))
    except httpx.HTTPError as exc:
        return _empty(f"{type(exc).__name__}: {str(exc)[:160]}")
    except Exception as exc:  # noqa: BLE001 — движок внешний, падать нельзя
        return _empty(f"{type(exc).__name__}: {str(exc)[:160]}")
    finally:
        if own_client:
            await client.aclose()
