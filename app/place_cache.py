"""Кэш разобранных мест — общий для всех пользователей.

Кэшируется только то, что НЕ зависит от ракурса съёмки: подтверждённый адрес,
найденные архивные фотографии и вердикты их привязки к зданию. Эти вещи
одинаковы для всех, кто снял дом с любой стороны, и стоят дороже всего —
несколько обращений к Vision на здание.

Готовая реконструкция сюда НЕ попадает. Она рисуется под кадр конкретного
человека, и отдать её другому — значит показать ему чужой ракурс и назвать это
его местом.

Побочная выгода: со временем накапливаются разобранные здания города, и по
популярным домам ответ становится почти бесплатным.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(os.getenv("PLACE_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "data" / "place_cache")))
CACHE_TTL_DAYS = float(os.getenv("PLACE_CACHE_TTL_DAYS", "30"))
MAX_ENTRIES = int(os.getenv("PLACE_CACHE_MAX_ENTRIES", "600"))


def _key(address: str) -> str:
    clean = re.sub(r"\s+", " ", str(address or "")).strip().lower().replace("ё", "е")
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:24]


def _path(address: str) -> Path:
    return CACHE_DIR / f"{_key(address)}.json"


def load(address: str) -> dict[str, Any] | None:
    if not address:
        return None
    path = _path(address)
    try:
        with path.open(encoding="utf-8") as handle:
            entry = json.load(handle)
    except (OSError, ValueError):
        return None
    if time.time() - float(entry.get("saved_at", 0)) > CACHE_TTL_DAYS * 86400:
        return None
    return entry


def save(address: str, payload: dict[str, Any]) -> None:
    if not address:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"address": address, "saved_at": time.time(), **payload}
        tmp = _path(address).with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False)
        tmp.replace(_path(address))
        _trim()
    except OSError:
        pass


def _trim() -> None:
    try:
        files = sorted(CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for path in files[: max(0, len(files) - MAX_ENTRIES)]:
        try:
            path.unlink()
        except OSError:
            pass


def stats() -> dict[str, Any]:
    try:
        files = list(CACHE_DIR.glob("*.json"))
        return {"places": len(files), "bytes": sum(f.stat().st_size for f in files)}
    except OSError:
        return {"places": 0, "bytes": 0}
