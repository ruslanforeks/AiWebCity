"""Привязка архивной фотографии к конкретному зданию.

Раньше архивный снимок попадал в раздел «рядом с домом», если он сделан в 150
метрах от точки. Это утверждение о расстоянии, а не о том, что дом есть в кадре.
На улице Куникова, 43 такими «ближними» оказались фотографии соседнего пивзавода
и цеха «Пепси-колы»: они действительно рядом, но это не та котельная.

Здесь архивный снимок сравнивается с современной фотографией здания. Модель
отвечает не «похоже или нет», а конкретнее: есть ли здание в кадре, насколько
видно фасад и под каким углом он снят. Эти три вещи потом нужны и для отбора
референса под реконструкцию, и для честной оценки достоверности.

Сравнение через десятилетия: вывески, окна, транспорт и деревья меняются, а
геометрия — нет. Именно по ней и надо решать.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

from .image_fetch import fetch_candidate, new_client, normalize_bytes, to_data_url
from .main import TIMEWEB_API_BASE, TIMEWEB_TOKEN, VISION_MODEL, extract_json, extract_text, norm_text

HISTORY_MATCH_LIMIT = int(os.getenv("HISTORY_MATCH_LIMIT", "8"))
HISTORY_MATCH_ENOUGH = int(os.getenv("HISTORY_MATCH_ENOUGH", "3"))
HISTORY_MATCH_CONCURRENCY = int(os.getenv("HISTORY_MATCH_CONCURRENCY", "3"))
HISTORY_MATCH_TIMEOUT = float(os.getenv("HISTORY_MATCH_TIMEOUT", "90"))
HISTORY_MATCH_THRESHOLD = float(os.getenv("HISTORY_MATCH_THRESHOLD", "0.6"))
MODERN_MAX_SIDE = int(os.getenv("HISTORY_MODERN_MAX_SIDE", "1024"))
ARCHIVE_MAX_SIDE = int(os.getenv("HISTORY_ARCHIVE_MAX_SIDE", "1024"))

VISIBILITY_ORDER = {"full": 3, "partial": 2, "edge": 1, "none": 0}
ANGLE_ORDER = {"same": 3, "similar": 2, "different": 1, "unknown": 0}

PROMPT = """
IMAGE A — современная фотография конкретного здания.
IMAGE B — старая архивная фотография того же города, снятая десятки лет назад.

Вопрос один: присутствует ли здание с IMAGE A на фотографии IMAGE B?

Как решать:
- Сравнивай устойчивую геометрию: число и ритм этажей, расположение и форму окон,
  выступы, углы, форму крыши, входы, колонны, характерные детали фасада.
- За десятилетия меняются вывески, рамы окон, транспорт, деревья, покрытие дороги,
  кондиционеры и антенны. Это НЕ повод сказать, что здание другое.
- А вот другая этажность, другой ритм окон, другая форма крыши — повод.
- Соседнее здание того же времени и стиля — это НЕ то же самое здание.
  Похожая эпоха и похожий тип застройки совпадением не считаются.
- Если здание могло измениться до неузнаваемости или доказательств мало,
  честно ставь building_present = false.

Поля ответа:
- building_present — есть ли здание с IMAGE A в кадре IMAGE B.
- visibility — сколько видно: "full" целиком, "partial" заметная часть,
  "edge" только краем или вдали, "none" не видно.
- angle — насколько ракурс IMAGE B близок к ракурсу IMAGE A:
  "same" почти тот же, "similar" близкий, "different" сильно другой, "unknown".
- confidence — уверенность от 0 до 1.

Верни строго JSON:
{
  "building_present": true,
  "visibility": "full",
  "angle": "similar",
  "confidence": 0.0,
  "matching_features": ["конкретные совпавшие детали"],
  "reason": "коротко"
}
"""


def _fail(photo: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        **photo,
        "scope": "street",
        "building_present": False,
        "visibility": "none",
        "angle": "unknown",
        "match_confidence": 0.0,
        "match_error": error,
    }


async def _compare(
    vision_client: httpx.AsyncClient,
    modern_data_url: str,
    photo: dict[str, Any],
    archive_jpeg: bytes,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "Ты строгий visual verifier. Отвечай только валидным JSON. Не используй подписи и метаданные как доказательство."},
        {"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "text", "text": "IMAGE A: современная фотография здания"},
            {"type": "image_url", "image_url": {"url": modern_data_url}},
            {"type": "text", "text": "IMAGE B: архивная фотография"},
            {"type": "image_url", "image_url": {"url": to_data_url(archive_jpeg)}},
        ]},
    ]
    payload = {"model": VISION_MODEL, "messages": messages, "temperature": 0.0}
    headers = {"Authorization": f"Bearer {TIMEWEB_TOKEN}", "Content-Type": "application/json"}
    try:
        response = await vision_client.post(f"{TIMEWEB_API_BASE}/chat/completions", headers=headers, json=payload)
    except (httpx.HTTPError, ValueError) as exc:
        return _fail(photo, f"{type(exc).__name__}: {str(exc)[:120]}")
    if response.status_code >= 400:
        return _fail(photo, f"timeweb_{response.status_code}")
    try:
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except (ValueError, json.JSONDecodeError, IndexError, AttributeError):
        return _fail(photo, "bad_response")

    verdict = extract_json(extract_text(content))
    try:
        confidence = max(0.0, min(1.0, float(verdict.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    present = verdict.get("building_present") is True and confidence >= HISTORY_MATCH_THRESHOLD
    visibility = norm_text(verdict.get("visibility")).lower()
    angle = norm_text(verdict.get("angle")).lower()
    return {
        **photo,
        "scope": "building" if present else "street",
        "building_present": present,
        "visibility": visibility if visibility in VISIBILITY_ORDER else "none",
        "angle": angle if angle in ANGLE_ORDER else "unknown",
        "match_confidence": confidence,
        "matching_features": [norm_text(x) for x in verdict.get("matching_features", []) if norm_text(x)][:4],
        "match_reason": norm_text(verdict.get("reason")),
        "match_error": None,
    }


async def match_archive_photos(
    modern_image: bytes,
    archive_photos: list[dict[str, Any]],
    *,
    limit: int = HISTORY_MATCH_LIMIT,
    enough: int = HISTORY_MATCH_ENOUGH,
) -> list[dict[str, Any]]:
    """Проверяет архивные снимки на присутствие здания. Возвращает их же с вердиктами.

    Проверяются первые `limit` снимков; как только набралось `enough`
    подтверждений, остальные остаются с меткой «эта улица» без лишних вызовов.
    """
    if not archive_photos or not TIMEWEB_TOKEN:
        return archive_photos

    modern_jpeg = normalize_bytes(modern_image, max_side=MODERN_MAX_SIDE, quality=88)
    if not modern_jpeg:
        return archive_photos
    modern_data_url = to_data_url(modern_jpeg)

    checked: list[dict[str, Any]] = []
    confirmed = 0
    semaphore = asyncio.Semaphore(HISTORY_MATCH_CONCURRENCY)
    to_check = archive_photos[:limit]

    async with new_client() as fetch_client, httpx.AsyncClient(timeout=HISTORY_MATCH_TIMEOUT) as vision_client:
        async def worker(photo: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                data, _ = await fetch_candidate(fetch_client, photo, max_side=ARCHIVE_MAX_SIDE)
                if not data:
                    return _fail(photo, "archive_image_unreachable")
                try:
                    return await _compare(vision_client, modern_data_url, photo, data)
                finally:
                    del data

        for start in range(0, len(to_check), HISTORY_MATCH_CONCURRENCY):
            wave = to_check[start:start + HISTORY_MATCH_CONCURRENCY]
            results = list(await asyncio.gather(*(worker(p) for p in wave)))
            checked.extend(results)
            confirmed += sum(1 for r in results if r.get("building_present"))
            if confirmed >= enough:
                break
            if all("timeweb_" in norm_text(r.get("match_error")) for r in results):
                break

    rest = archive_photos[len(checked):]
    for photo in rest:
        photo.setdefault("scope", "street")
        photo.setdefault("building_present", False)
        photo.setdefault("match_confidence", 0.0)

    # Снимки, где здание действительно видно, идут первыми: сначала подтверждённые,
    # среди них — те, где фасад виден полнее и ракурс ближе к современному.
    def rank(photo: dict[str, Any]) -> tuple:
        return (
            1 if photo.get("building_present") else 0,
            VISIBILITY_ORDER.get(photo.get("visibility", "none"), 0),
            ANGLE_ORDER.get(photo.get("angle", "unknown"), 0),
            float(photo.get("match_confidence", 0.0)),
        )

    return sorted(checked + rest, key=rank, reverse=True)


def best_reference(photos: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Лучший архивный снимок для реконструкции: здание видно полнее всего."""
    confirmed = [p for p in photos if p.get("building_present")]
    if not confirmed:
        return None
    return max(confirmed, key=lambda p: (
        VISIBILITY_ORDER.get(p.get("visibility", "none"), 0),
        ANGLE_ORDER.get(p.get("angle", "unknown"), 0),
        float(p.get("match_confidence", 0.0)),
    ))
