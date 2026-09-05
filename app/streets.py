"""Подсказки адресов Новороссийска.

База — data/streets_novorossiysk.json, выгрузка адресов из OpenStreetMap
(tools/fetch_streets.py). Она НЕ доказывает, что здание на фотографии стоит по
этому адресу: это делает только сравнение изображений. Её задача проще и
полезнее — помочь пользователю ввести адрес в понятном виде и без опечаток,
а нам получить каноническое название улицы вместо «куникова».
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "streets_novorossiysk.json"

STREET_PREFIX_RE = re.compile(
    r"^(?:г\.?\s*новоросси\w*[\s,]*)?(?:ул(?:ица)?|пер(?:еулок)?|просп(?:ект)?|пр-т|наб(?:ережная)?|"
    r"пл(?:ощадь)?|шоссе|бул(?:ьвар)?|проезд|тупик|аллея|мкр|микрорайон)\.?\s*",
    re.I,
)

_DB: dict[str, Any] | None = None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower().replace("ё", "е")


def _bare(street: str) -> str:
    """«улица Куникова» -> «куникова». Тип улицы для сопоставления не нужен."""
    return _normalize(STREET_PREFIX_RE.sub("", _normalize(street))).strip(" ,.")


def load() -> dict[str, Any]:
    global _DB
    if _DB is None:
        try:
            with DATA_PATH.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError):
            raw = {"streets": {}, "street_coords": {}}
        streets = raw.get("streets") or {}
        houses = raw.get("houses") or {}
        # Плоский список домов с координатами: по нему ищутся адреса рядом с
        # точкой съёмки. 25 тысяч записей просматриваются за доли миллисекунды.
        points: list[tuple[float, float, str, str]] = []
        for street, entries in houses.items():
            for house, point in entries.items():
                if isinstance(point, list) and len(point) == 2:
                    points.append((float(point[0]), float(point[1]), street, house))
        _DB = {
            "streets": streets,
            "houses": houses,
            "points": points,
            "coords": raw.get("street_coords") or {},
            # индекс «куникова» -> «улица Куникова»
            "by_bare": {_bare(name): name for name in streets},
        }
    return _DB


def split_query(query: str) -> tuple[str, str]:
    """Делит ввод на «улица» и «начало номера дома»."""
    clean = _normalize(query)
    match = re.search(r"[\s,]+(?:д\.?\s*|дом\s*)?(\d[\da-zа-я/-]*)\s*$", clean)
    if match:
        return clean[: match.start()].strip(" ,."), match.group(1)
    return clean.strip(" ,."), ""


def canonical_street(street: str) -> str:
    """Каноническое название улицы из базы, если оно там есть."""
    return load()["by_bare"].get(_bare(street), "")


def street_center(street: str) -> tuple[float, float] | None:
    canonical = canonical_street(street) or street
    point = load()["coords"].get(canonical)
    if isinstance(point, list) and len(point) == 2:
        return float(point[0]), float(point[1])
    return None


def _house_sort_key(house: str) -> tuple[int, str]:
    match = re.match(r"(\d+)", house)
    return (int(match.group(1)) if match else 99999, house)


def suggest(query: str, limit: int = 8) -> list[dict[str, str]]:
    """Подсказки для поля адреса.

    Пока не введён номер дома — предлагаем улицы. Как только появился номер —
    предлагаем реально существующие дома на этой улице.
    """
    database = load()
    streets: dict[str, list[str]] = database["streets"]
    if not streets:
        return []

    street_part, house_part = split_query(query)
    if len(street_part) < 2:
        return []

    bare_query = _bare(street_part)
    matches: list[tuple[int, str]] = []
    for name in streets:
        bare = _bare(name)
        if bare.startswith(bare_query):
            matches.append((0, name))
        elif bare_query in bare:
            matches.append((1, name))
    matches.sort(key=lambda row: (row[0], len(row[1]), row[1]))

    results: list[dict[str, str]] = []
    if house_part:
        for _, name in matches[:3]:
            houses = [h for h in streets[name] if _normalize(h).startswith(_normalize(house_part))]
            if not houses:
                continue
            for house in sorted(houses, key=_house_sort_key)[:limit]:
                results.append({"value": f"{name}, {house}", "street": name, "house": house})
                if len(results) >= limit:
                    return results
        # Введённого дома в базе нет — предложим сам ввод и соседние дома улицы.
        if not results and matches:
            name = matches[0][1]
            results.append({"value": f"{name}, {house_part}", "street": name, "house": house_part})
            for house in sorted(streets[name], key=_house_sort_key)[: limit - 1]:
                results.append({"value": f"{name}, {house}", "street": name, "house": house})
        return results[:limit]

    for _, name in matches[:limit]:
        results.append({"value": name, "street": name, "house": "", "houses": str(len(streets[name]))})
    return results


def nearby_addresses(
    lat: float,
    lon: float,
    *,
    radius_m: float = 120.0,
    heading: float | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Дома рядом с точкой съёмки, ближние первыми.

    Если известен азимут камеры, дома «за спиной» уходят вниз списка: человек
    фотографировал то, на что смотрел, а не то, что было позади. Полностью
    отбрасывать их нельзя — азимут в EXIF бывает неточным.
    """
    from .photo_meta import angle_diff, bearing_deg, distance_m

    rows: list[tuple[float, dict[str, Any]]] = []
    for plat, plon, street, house in load()["points"]:
        # Дешёвая отбраковка по прямоугольнику до честного расстояния.
        if abs(plat - lat) > 0.0025 or abs(plon - lon) > 0.0035:
            continue
        distance = distance_m(lat, lon, plat, plon)
        if distance > radius_m:
            continue
        penalty = 0.0
        off_axis = None
        if heading is not None:
            off_axis = angle_diff(heading, bearing_deg(lat, lon, plat, plon))
            penalty = distance * (off_axis / 90.0)
        rows.append((distance + penalty, {
            "street": street,
            "house": house,
            "lat": plat,
            "lon": plon,
            "distance_m": round(distance, 1),
            "off_axis_deg": round(off_axis, 1) if off_axis is not None else None,
        }))

    rows.sort(key=lambda row: row[0])
    return [row[1] for row in rows[:limit]]
