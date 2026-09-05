"""Координаты и направление съёмки из EXIF фотографии.

Телефон записывает в снимок точку съёмки и азимут камеры. Для нас это самый
прямой сигнал из всех: он говорит, где человек стоял и куда смотрел, а у нас
есть база домов Новороссийска с координатами.

Данные используются в памяти на время обработки и никуда не сохраняются —
как и сама фотография.

Учтите: мессенджеры (Telegram, WhatsApp) вырезают EXIF, а iOS умеет отдавать
снимок без геометки. Поэтому GPS — приятный бонус, а не то, на что можно
рассчитывать.
"""

from __future__ import annotations

import io
import math
from typing import Any

from PIL import ExifTags, Image

EARTH_RADIUS_M = 6371000.0


def _to_degrees(values: Any, ref: Any) -> float | None:
    try:
        degrees, minutes, seconds = (float(x) for x in values)
    except (TypeError, ValueError):
        return None
    result = degrees + minutes / 60.0 + seconds / 3600.0
    if str(ref).upper() in {"S", "W"}:
        result = -result
    return result


def extract_gps(image_bytes: bytes) -> dict[str, Any] | None:
    """Точка съёмки и азимут камеры, если они есть в снимке."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            exif = image.getexif()
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        return None
    if not gps:
        return None

    lat = _to_degrees(gps.get(2), gps.get(1, "N"))
    lon = _to_degrees(gps.get(4), gps.get(3, "E"))
    if lat is None or lon is None or (abs(lat) < 0.0001 and abs(lon) < 0.0001):
        return None

    heading = None
    raw_heading = gps.get(17)  # GPSImgDirection — куда смотрела камера
    try:
        if raw_heading is not None:
            heading = float(raw_heading) % 360.0
    except (TypeError, ValueError):
        heading = None

    accuracy = None
    try:
        if gps.get(31) is not None:
            accuracy = float(gps.get(31))
    except (TypeError, ValueError):
        accuracy = None

    return {"lat": round(lat, 6), "lon": round(lon, 6), "heading": heading, "accuracy_m": accuracy}


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Азимут из точки 1 в точку 2, градусы от севера по часовой стрелке."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_diff(a: float, b: float) -> float:
    """Наименьшая разница между двумя азимутами, 0..180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)
