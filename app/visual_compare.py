from __future__ import annotations

import base64
import json
import re
from typing import Any

import httpx

from .main import TIMEWEB_API_BASE, TIMEWEB_TOKEN, VISION_MODEL, extract_text, extract_json, norm_text


async def _compare_one(image_bytes: bytes, content_type: str, candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_url = norm_text(candidate.get("image_url"))
    if not candidate_url:
        return {**candidate, "same_place": False, "confidence": 0.0, "error": "no_candidate_image"}
    if not TIMEWEB_TOKEN:
        return {**candidate, "same_place": False, "confidence": 0.0, "error": "TIMEWEB_AI_TOKEN_not_configured"}

    original = base64.b64encode(image_bytes).decode("ascii")
    original_url = f"data:{content_type};base64,{original}"
    prompt = """
Ты — модуль визуальной верификации AiWebCity.
У тебя есть ДВЕ фотографии одного возможного места:
IMAGE A — исходное фото пользователя.
IMAGE B — фото-кандидат, найденное во внешнем поиске.

Ответь только на вопрос: являются ли IMAGE A и IMAGE B фотографиями ОДНОГО И ТОГО ЖЕ КОНКРЕТНОГО ЗДАНИЯ/ОБЪЕКТА/МЕСТА?

Критически важно:
- НЕ определяй здание по названию сайта, URL, подписи, адресу или тексту вокруг изображения.
- НЕ исходи из того, что фотографии совпадают только потому, что они из Новороссийска.
- Не пиши общее описание вроде «семиэтажное здание». Нужны именно совпадающие и противоречащие визуальные детали.
- Сравни геометрию и композицию объекта: характерные выступы и углы, относительное расположение окон, балконов, колонн, входов, этажей, крыш, соседних объектов, вывесок и других уникальных деталей.
- Разный ракурс, сезон, качество, цвет, транспорт или временные изменения НЕ являются сами по себе причиной для отрицания.
- Если на снимках может быть один и тот же объект, но доказательств мало — ставь same_place=false и низкую уверенность. Лучше ошибиться в сторону «не подтверждено», чем выдать ложное совпадение.

Верни строго JSON:
{
  "same_place": true,
  "confidence": 0.0,
  "matching_features": ["конкретные визуальные совпадения"],
  "contradictions": ["конкретные визуальные противоречия"],
  "reason": "краткое решение без общего описания здания"
}
"""
    messages = [
        {"role": "system", "content": "Ты строгий visual verifier. Не угадывай адрес и не используй текстовые метаданные как доказательство."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "IMAGE A: исходная фотография пользователя"},
                {"type": "image_url", "image_url": {"url": original_url}},
                {"type": "text", "text": "IMAGE B: фотография-кандидат из внешнего поиска"},
                {"type": "image_url", "image_url": {"url": candidate_url}},
            ],
        },
    ]
    headers = {"Authorization": f"Bearer {TIMEWEB_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": VISION_MODEL, "messages": messages, "temperature": 0.0}
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(f"{TIMEWEB_API_BASE}/chat/completions", headers=headers, json=payload)
        if response.status_code >= 400:
            return {**candidate, "same_place": False, "confidence": 0.0, "error": f"timeweb_{response.status_code}"}
        body = response.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        verdict = extract_json(extract_text(content))
        same_place = bool(verdict.get("same_place") is True)
        try:
            confidence = float(verdict.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            **candidate,
            "same_place": same_place,
            "confidence": confidence,
            "matching_features": [norm_text(x) for x in verdict.get("matching_features", []) if norm_text(x)],
            "contradictions": [norm_text(x) for x in verdict.get("contradictions", []) if norm_text(x)],
            "reason": norm_text(verdict.get("reason")),
        }
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        return {**candidate, "same_place": False, "confidence": 0.0, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}


async def verify_candidates(image_bytes: bytes, content_type: str, candidates: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    selected = [c for c in candidates if norm_text(c.get("image_url"))][:limit]
    results: list[dict[str, Any]] = []
    for candidate in selected:
        results.append(await _compare_one(image_bytes, content_type, candidate))
    return sorted(results, key=lambda x: (bool(x.get("same_place")), float(x.get("confidence", 0.0))), reverse=True)


def extract_candidate_location_text(candidate: dict[str, Any]) -> str:
    return norm_text(" ".join([
        norm_text(candidate.get("title")),
        norm_text(candidate.get("description")),
        norm_text(candidate.get("site")),
        norm_text(candidate.get("page_url")),
    ]))


def extract_address_hints(candidate: dict[str, Any]) -> list[str]:
    text = extract_candidate_location_text(candidate)
    text = text.replace("ё", "е")
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
