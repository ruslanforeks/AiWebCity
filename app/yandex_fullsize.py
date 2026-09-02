from __future__ import annotations

import json
from typing import Any

import httpx
from bs4 import BeautifulSoup


YANDEX_IMAGES_URL = "https://yandex.com/images/search"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _absolute(value: Any) -> str:
    value = _norm(value)
    if value.startswith("//"):
        return "https:" + value
    return value


def _site_dicts_from_response_html(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one('div.Root[id^="ImagesApp-"]')
    if not root:
        return []
    state = root.get("data-state")
    if not state:
        return []
    try:
        data = json.loads(state)
    except (TypeError, json.JSONDecodeError):
        return []
    sites = data
    for key in ("initialState", "cbirSites", "sites"):
        if isinstance(sites, dict):
            sites = sites.get(key, {})
    if not isinstance(sites, list):
        return []
    return [item for item in sites if isinstance(item, dict)]


def _pick_original_image(item: dict[str, Any]) -> tuple[str, int | None, int | None]:
    original = item.get("originalImage")
    if isinstance(original, dict):
        for key in ("url", "originUrl", "origin_url", "src"):
            url = _absolute(original.get(key))
            if url:
                return url, _to_int(original.get("width")), _to_int(original.get("height"))
    for key in ("img_href", "image", "url"):
        url = _absolute(item.get(key))
        if url and not _looks_like_thumbnail(url):
            return url, None, None
    return "", None, None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _looks_like_thumbnail(url: str) -> bool:
    value = url.lower()
    return any(marker in value for marker in ("avatars.mds.yandex", "im0-tub-ru.yandex", "thumb", "preview"))


def _build_result(raw: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    source = fallback or {}
    snippet = raw.get("snippet") if isinstance(raw.get("snippet"), dict) else {}
    thumb = raw.get("thumb") if isinstance(raw.get("thumb"), dict) else {}
    preview = _absolute(thumb.get("url"))
    original_url, width, height = _pick_original_image(raw)
    if not original_url and isinstance(source, dict):
        original_url = _absolute(source.get("img_href"))
        width = width or _to_int(source.get("width"))
        height = height or _to_int(source.get("height"))
    return {
        "image_url": original_url or preview,
        "preview_url": preview,
        "page_url": _norm(raw.get("url") or source.get("url")),
        "title": _norm(raw.get("title") or snippet.get("title")) or "Результат Yandex Images",
        "description": _norm(raw.get("description") or snippet.get("text")),
        "source": "Yandex Images",
        "kind": "reverse_image",
        "site": _norm(raw.get("domain") or snippet.get("domain")),
        "size": f"{width}x{height}" if width and height else "",
        "image_quality": "original" if original_url else "preview_fallback",
    }


async def search_fullsize(image_bytes: bytes, limit: int = 40) -> dict[str, Any]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }
    timeout = httpx.Timeout(45.0, connect=15.0)
    try:
        async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
            upload = await client.post(
                YANDEX_IMAGES_URL,
                params={
                    "rpt": "imageview",
                    "format": "json",
                    "request": json.dumps({"blocks": [{"block": "b-page_type_search-by-image__link"}]}, ensure_ascii=False),
                },
                files={"upfile": ("photo.jpg", image_bytes, "image/jpeg")},
            )
            upload.raise_for_status()
            payload = upload.json()
            blocks = payload.get("blocks") if isinstance(payload, dict) else None
            params_url = ""
            if isinstance(blocks, list) and blocks:
                params_url = _norm((blocks[0].get("params") or {}).get("url"))
            if not params_url:
                return {"ok": False, "results": [], "search_url": None, "error": "yandex_upload_url_missing"}

            if params_url.startswith("http://") or params_url.startswith("https://"):
                search_url = params_url
            else:
                separator = "&" if "?" in params_url else "?"
                search_url = f"{YANDEX_IMAGES_URL}?{params_url.lstrip('?')}{separator if params_url else ''}"

            search_response = await client.get(search_url)
            search_response.raise_for_status()
            sites = _site_dicts_from_response_html(search_response.text)

            # Prefer the structured cbirSites payload: it contains the original image
            # URL separately from the thumbnail URL used by the search UI.
            results: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in sites:
                result = _build_result(item)
                key = _norm(result.get("image_url")) or _norm(result.get("preview_url"))
                if not key or key in seen:
                    continue
                seen.add(key)
                results.append(result)
                if len(results) >= limit:
                    break

            return {
                "ok": True,
                "results": results,
                "search_url": search_url,
                "error": None,
            }
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "results": [],
            "search_url": None,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
