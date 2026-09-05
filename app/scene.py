"""Два точечных обращения к Vision: рамка здания и текст на фотографии.

Оба намеренно узкие. Модель здесь ничего не называет и не угадывает — она
показывает, где на кадре здание, и читает то, что реально написано. Всё, что
похоже на «определи, что это за дом», остаётся за пределами этого модуля:
такие догадки проверить нечем, и именно из них берутся «девятиэтажные древние
здания».
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .image_fetch import normalize_bytes, to_data_url
from .main import TIMEWEB_API_BASE, TIMEWEB_TOKEN, VISION_MODEL, extract_json, extract_text, norm_text

SCENE_TIMEOUT = float(os.getenv("SCENE_TIMEOUT", "60"))
SCENE_MAX_SIDE = int(os.getenv("SCENE_MAX_SIDE", "896"))

BBOX_PROMPT = """
На фотографии городская сцена. Найди ГЛАВНОЕ ЗДАНИЕ — то, ради которого сделан снимок.

Правила:
- Рамка должна охватывать именно постройку: фасад, крышу, стены.
- НЕ включай в рамку деревья, кусты, машины, столбы, провода, людей, небо и дорогу.
- Если зданий несколько, выбери то, что занимает больше кадра и снято прямее.
- Если здания на фотографии нет вообще (только природа, интерьер, предмет, человек),
  верни has_building = false и НЕ придумывай рамку.

Координаты — доли от размера кадра, от 0 до 1: x0 и y0 — левый верхний угол, x1 и y1 — правый нижний.

Верни строго JSON:
{"has_building": true, "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0, "note": "коротко, что в рамке"}
"""

TEXT_PROMPT = """
Прочитай ТОЛЬКО тот текст, который реально виден на фотографии: таблички с номерами домов,
названия улиц, вывески организаций, надписи на фасаде.

Правила:
- Ничего не додумывай. Не видно — оставляй пустым.
- Не выводи текст, который ты предполагаешь по виду здания.
- house_number — только если на табличке действительно видна цифра.
- street_name — только если на табличке действительно написано название улицы.
- signs — вывески и надписи, как они написаны.

Верни строго JSON:
{"house_number": "", "street_name": "", "signs": [], "all_text": ""}
"""


async def _ask(image_bytes: bytes, prompt: str, system: str) -> dict[str, Any]:
    if not TIMEWEB_TOKEN:
        return {}
    prepared = normalize_bytes(image_bytes, max_side=SCENE_MAX_SIDE, quality=88)
    if not prepared:
        return {}
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": to_data_url(prepared)}},
        ]},
    ]
    payload = {"model": VISION_MODEL, "messages": messages, "temperature": 0.0}
    headers = {"Authorization": f"Bearer {TIMEWEB_TOKEN}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=SCENE_TIMEOUT) as client:
            response = await client.post(f"{TIMEWEB_API_BASE}/chat/completions", headers=headers, json=payload)
        if response.status_code >= 400:
            return {}
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except (httpx.HTTPError, ValueError, json.JSONDecodeError, IndexError, AttributeError):
        return {}
    return extract_json(extract_text(content))


async def detect_main_building(image_bytes: bytes) -> dict[str, Any] | None:
    """Рамка главного здания в долях кадра.

    Нужна, когда детектор Яндекса не нашёл здание: на фотографии, где кадр
    занимает дерево, он честно отвечает «pinus», и обратный поиск ищет сосну.
    """
    verdict = await _ask(
        image_bytes, BBOX_PROMPT,
        "Ты детектор объектов. Отвечай только валидным JSON. Не называй здание и не угадывай адрес.",
    )
    if not verdict or verdict.get("has_building") is not True:
        return None
    try:
        x0, y0, x1, y1 = (float(verdict[k]) for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, ValueError):
        return None
    x0, y0 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, y0))
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    if x1 - x0 < 0.08 or y1 - y0 < 0.08:
        return None
    return {"box": (x0, y0, x1, y1), "area": (x1 - x0) * (y1 - y0), "note": norm_text(verdict.get("note"))}


async def read_photo_text(image_bytes: bytes) -> dict[str, Any]:
    """Текст, реально видимый на снимке: номер дома, улица, вывески.

    Собственный OCR Яндекса на таких кадрах выдаёт мусор вроде «1-1-I l omaRossin»,
    а номер дома на табличке — это прямое доказательство, а не догадка.
    """
    verdict = await _ask(
        image_bytes, TEXT_PROMPT,
        "Ты OCR. Отвечай только валидным JSON. Выводи лишь то, что реально написано на фотографии.",
    )
    if not verdict:
        return {"house_number": "", "street_name": "", "signs": [], "all_text": ""}
    signs = [norm_text(x) for x in verdict.get("signs", []) if norm_text(x)][:6]
    return {
        "house_number": norm_text(verdict.get("house_number")),
        "street_name": norm_text(verdict.get("street_name")),
        "signs": signs,
        "all_text": norm_text(verdict.get("all_text")) or " ".join(signs),
    }
