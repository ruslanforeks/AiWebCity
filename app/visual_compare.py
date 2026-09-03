"""Визуальная верификация: одно ли это здание на двух фотографиях.

Отличия от первой версии:
  * кандидаты скачиваются и нормализуются у нас, а не «пусть шлюз сам сходит»;
  * исходное фото ужимается один раз, а не гоняется целиком в каждый запрос;
  * сравнения идут параллельно, а не по очереди.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import httpx

from .image_fetch import fetch_candidate, new_client, normalize_bytes, to_data_url
from .main import TIMEWEB_API_BASE, TIMEWEB_TOKEN, VISION_MODEL, extract_json, extract_text, norm_text

VISION_CONCURRENCY = int(os.getenv("VISION_CONCURRENCY", "3"))
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT", "90"))
ORIGINAL_MAX_SIDE = int(os.getenv("VISION_ORIGINAL_MAX_SIDE", "1024"))
CANDIDATE_MAX_SIDE = int(os.getenv("VISION_CANDIDATE_MAX_SIDE", "768"))

SYSTEM_PROMPT = (
    "Ты строгий visual verifier. Не угадывай адрес и не используй текстовые метаданные как доказательство. "
    "Отвечай только валидным JSON."
)

PROMPT = """
Ты — модуль визуальной верификации AiWebCity.
У тебя есть ДВЕ фотографии:
IMAGE A — исходное фото пользователя.
IMAGE B — фото-кандидат, найденное во внешнем поиске.

Ответь только на вопрос: являются ли IMAGE A и IMAGE B фотографиями ОДНОГО И ТОГО ЖЕ КОНКРЕТНОГО ЗДАНИЯ/ОБЪЕКТА/МЕСТА?

Критически важно:
- НЕ определяй здание по названию сайта, URL, подписи, адресу или тексту вокруг изображения.
- НЕ исходи из того, что фотографии совпадают только потому, что они из Новороссийска.
- Не пиши общее описание вроде «семиэтажное здание». Нужны именно совпадающие и противоречащие визуальные детали.
- Сравни геометрию и композицию объекта: характерные выступы и углы, относительное расположение окон, балконов, колонн, входов, этажей, крыш, соседних объектов, вывесок и других уникальных деталей.
- IMAGE B часто снято с ДРУГОГО ракурса, в другой сезон, в другое время суток или содержит здание не целиком. Это нормально: ищи совпадение по устойчивым архитектурным признакам, а не по совпадению кадра.
- Если на IMAGE A видно несколько зданий, считай совпадением ситуацию, когда объект с IMAGE B — это одно из зданий, отчётливо присутствующих на IMAGE A. Тогда укажи его в поле which_object.
- Если на снимках может быть один и тот же объект, но доказательств мало — ставь same_place=false и низкую уверенность. Лучше ошибиться в сторону «не подтверждено», чем выдать ложное совпадение.

Верни строго JSON:
{
  "same_place": true,
  "confidence": 0.0,
  "which_object": "какой именно объект кадра A совпал, если их несколько",
  "matching_features": ["конкретные визуальные совпадения"],
  "contradictions": ["конкретные визуальные противоречия"],
  "reason": "краткое решение без общего описания здания"
}
"""


def _fail(candidate: dict[str, Any], error: str) -> dict[str, Any]:
    return {**candidate, "same_place": False, "confidence": 0.0, "visual_error": error}


async def _compare_one(
    client: httpx.AsyncClient,
    original_data_url: str,
    candidate: dict[str, Any],
    candidate_jpeg: bytes,
    fetched_url: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "text", "text": "IMAGE A: исходная фотография пользователя"},
                {"type": "image_url", "image_url": {"url": original_data_url}},
                {"type": "text", "text": "IMAGE B: фотография-кандидат из внешнего поиска"},
                {"type": "image_url", "image_url": {"url": to_data_url(candidate_jpeg)}},
            ],
        },
    ]
    payload = {"model": VISION_MODEL, "messages": messages, "temperature": 0.0}
    headers = {"Authorization": f"Bearer {TIMEWEB_TOKEN}", "Content-Type": "application/json"}
    try:
        response = await client.post(f"{TIMEWEB_API_BASE}/chat/completions", headers=headers, json=payload)
    except (httpx.HTTPError, ValueError) as exc:
        return _fail(candidate, f"{type(exc).__name__}: {str(exc)[:160]}")
    if response.status_code >= 400:
        return _fail(candidate, f"timeweb_{response.status_code}: {response.text[:160]}")
    try:
        body = response.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    except (ValueError, json.JSONDecodeError, IndexError, AttributeError):
        return _fail(candidate, "bad_response")

    verdict = extract_json(extract_text(content))
    try:
        confidence = max(0.0, min(1.0, float(verdict.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        **candidate,
        "image_url": fetched_url or candidate.get("image_url"),
        "same_place": verdict.get("same_place") is True,
        "confidence": confidence,
        "which_object": norm_text(verdict.get("which_object")),
        "matching_features": [norm_text(x) for x in verdict.get("matching_features", []) if norm_text(x)][:6],
        "contradictions": [norm_text(x) for x in verdict.get("contradictions", []) if norm_text(x)][:6],
        "reason": norm_text(verdict.get("reason")),
        "visual_error": None,
    }


async def verify_candidates(
    image_bytes: bytes,
    content_type: str,
    candidates: list[dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Проверяет top-N кандидатов параллельно и сортирует по уверенности."""
    selected = [c for c in candidates if norm_text(c.get("image_url")) or norm_text(c.get("thumb_url"))][:limit]
    if not selected:
        return []
    if not TIMEWEB_TOKEN:
        return [_fail(c, "TIMEWEB_AI_TOKEN_not_configured") for c in selected]

    original_jpeg = normalize_bytes(image_bytes, max_side=ORIGINAL_MAX_SIDE, quality=88)
    if not original_jpeg:
        return [_fail(c, "bad_original_image") for c in selected]
    original_data_url = to_data_url(original_jpeg)

    # Картинка кандидата скачивается прямо перед сравнением и сразу отпускается:
    # на VPS с 1 ГБ памяти держать все кандидаты одновременно нельзя.
    semaphore = asyncio.Semaphore(VISION_CONCURRENCY)

    async with new_client() as fetch_client, httpx.AsyncClient(timeout=VISION_TIMEOUT) as vision_client:
        async def worker(candidate: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                data, used_url = await fetch_candidate(fetch_client, candidate, max_side=CANDIDATE_MAX_SIDE)
                if not data:
                    return _fail(candidate, "candidate_image_unreachable")
                try:
                    return await _compare_one(vision_client, original_data_url, candidate, data, used_url)
                finally:
                    del data

        results = list(await asyncio.gather(*(worker(c) for c in selected)))

    return sorted(results, key=lambda x: (bool(x.get("same_place")), float(x.get("confidence", 0.0))), reverse=True)


def extract_candidate_location_text(candidate: dict[str, Any]) -> str:
    return norm_text(" ".join([
        norm_text(candidate.get("title")),
        norm_text(candidate.get("description")),
        norm_text(candidate.get("site")),
        norm_text(candidate.get("page_url")),
    ]))


def extract_address_hints(candidate: dict[str, Any]) -> list[str]:
    text = extract_candidate_location_text(candidate).replace("ё", "е")
    patterns = [
        r"(?i)(?:г\.?\s*|город\s+)?новороссийск[^,;|]{0,80}",
        r"(?i)(?:ул\.?|улица|просп\.?|проспект|пер\.?|переулок|шоссе|наб\.?|набережная)\s+[а-яa-z0-9 .-]{2,60}\s*,?\s*(?:д\.?|дом)?\s*\d{1,4}[а-яa-z]?(?:[/-]\d{1,4})?",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            cleaned = norm_text(match).strip(" ,.;:-")
            if len(cleaned) >= 6 and cleaned not in found:
                found.append(cleaned)
    return found[:5]
