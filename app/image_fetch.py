"""Загрузка и нормализация изображений-кандидатов.

Раньше URL кандидата отдавался Vision-модели как есть, и шлюз сам пытался его
скачать. Часть источников отдаёт 403 без браузерных заголовков, часть — HTML
вместо картинки, часть просто мертва. Такие кандидаты молча превращались в
«не совпадает».

Здесь картинка скачивается на нашей стороне, проверяется, ужимается и уходит
в модель как data URL. Это и надёжнее, и заметно дешевле по токенам.
"""

from __future__ import annotations

import base64
import io
from typing import Iterable

import httpx
from PIL import Image, ImageOps

FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "ru,en;q=0.9",
}

# На VPS с 1 ГБ памяти декодирование картинки дороже, чем её скачивание:
# 5000x4000 JPEG разворачивается в ~80 МБ пикселей. Поэтому ограничиваем и
# размер загрузки, и число пикселей, и декодируем JPEG сразу уменьшенным.
MAX_DOWNLOAD_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MIN_USEFUL_SIDE = 120

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def encode_jpeg(image: Image.Image, *, max_side: int, quality: int = 85) -> bytes:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def normalize_bytes(raw: bytes, *, max_side: int, quality: int = 85) -> bytes | None:
    """Приводит произвольные байты к валидному JPEG нужного размера.

    draft() просит JPEG-декодер сразу отдать уменьшенное изображение (1/2, 1/4,
    1/8). Для больших исходников это экономит десятки мегабайт на кадр.
    """
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format == "JPEG":
                image.draft("RGB", (max_side, max_side))
            image.load()
            if min(image.size) < MIN_USEFUL_SIDE:
                return None
            return encode_jpeg(image, max_side=max_side, quality=quality)
    except Exception:
        return None


def to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


async def fetch_one(client: httpx.AsyncClient, url: str, *, max_side: int) -> bytes | None:
    if not url or not url.startswith("http"):
        return None
    try:
        response = await client.get(url)
    except (httpx.HTTPError, UnicodeError):
        return None
    if response.status_code >= 400:
        return None
    content_type = response.headers.get("content-type", "").lower()
    if content_type and not content_type.startswith("image/"):
        return None
    if len(response.content) > MAX_DOWNLOAD_BYTES:
        return None
    return normalize_bytes(response.content, max_side=max_side)


async def fetch_first_usable(client: httpx.AsyncClient, urls: Iterable[str], *, max_side: int) -> tuple[bytes | None, str]:
    """Пробует URL по очереди: сначала полноразмерный оригинал, потом превью Яндекса."""
    seen: set[str] = set()
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        data = await fetch_one(client, url, max_side=max_side)
        if data:
            return data, url
    return None, ""


def new_client(timeout: float = 20.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        headers=FETCH_HEADERS,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=12, max_keepalive_connections=6),
    )


async def fetch_candidate(client: httpx.AsyncClient, candidate: dict, *, max_side: int) -> tuple[bytes | None, str]:
    """Полноразмерный оригинал, а если он недоступен — превью Яндекса."""
    urls = [candidate.get("image_url"), candidate.get("thumb_url"), candidate.get("preview_url")]
    return await fetch_first_usable(client, [u for u in urls if u], max_side=max_side)
