"""Выгружает адреса Новороссийска из OpenStreetMap через Overpass.

Запуск разовый, результат коммитится в data/streets_novorossiysk.json:

    python tools/fetch_streets.py

Хранятся координаты КАЖДОГО дома, а не только улицы: по ним ищутся ближайшие
адреса к точке съёмки из EXIF фотографии.
"""

import collections
import json
import re
import time
import urllib.parse
import urllib.request

MIRROR = "https://overpass-api.de/api/interpreter"
OUT_PATH = "data/streets_novorossiysk.json"

# Город режется на клетки: одним запросом Overpass весь Новороссийск не отдаёт (504).
LAT0, LAT1, LON0, LON1 = 44.58, 44.82, 37.62, 37.93
STEPS = 4


def fetch(query: str, tries: int = 3) -> dict:
    for _ in range(tries):
        try:
            payload = urllib.parse.urlencode({"data": query}).encode()
            request = urllib.request.Request(MIRROR, data=payload, headers={"User-Agent": "AiWebCity/1.2"})
            return json.load(urllib.request.urlopen(request, timeout=200))
        except Exception as exc:  # noqa: BLE001
            print("   повтор:", type(exc).__name__, str(exc)[:60])
            time.sleep(20)
    return {"elements": []}


def house_sort(house: str):
    match = re.match(r"(\d+)", house)
    return (int(match.group(1)) if match else 99999, house)


def main() -> None:
    houses: dict[str, dict[str, list[float]]] = collections.defaultdict(dict)
    dlat = (LAT1 - LAT0) / STEPS
    dlon = (LON1 - LON0) / STEPS

    for i in range(STEPS):
        for j in range(STEPS):
            box = f"{LAT0 + i * dlat:.4f},{LON0 + j * dlon:.4f},{LAT0 + (i + 1) * dlat:.4f},{LON0 + (j + 1) * dlon:.4f}"
            query = (
                f'[out:json][timeout:150];('
                f'node({box})["addr:street"]["addr:housenumber"];'
                f'way({box})["addr:street"]["addr:housenumber"];'
                f');out tags center;'
            )
            result = fetch(query)
            print(f"  клетка {box}: {len(result.get('elements', []))}")
            for element in result.get("elements", []):
                tags = element.get("tags") or {}
                street = (tags.get("addr:street") or "").strip()
                house = (tags.get("addr:housenumber") or "").strip()
                if not street or not house:
                    continue
                center = element.get("center") or {"lat": element.get("lat"), "lon": element.get("lon")}
                if not center.get("lat"):
                    continue
                houses[street][house] = [round(float(center["lat"]), 6), round(float(center["lon"]), 6)]
            time.sleep(6)

    street_coords = {}
    for street, points in houses.items():
        values = list(points.values())
        street_coords[street] = [
            round(sum(p[0] for p in values) / len(values), 5),
            round(sum(p[1] for p in values) / len(values), 5),
        ]

    result = {
        "city": "Новороссийск",
        "source": "OpenStreetMap via Overpass API",
        "streets": {s: sorted(h, key=house_sort) for s, h in sorted(houses.items())},
        "houses": {s: dict(sorted(h.items(), key=lambda kv: house_sort(kv[0]))) for s, h in sorted(houses.items())},
        "street_coords": street_coords,
    }
    with open(OUT_PATH, "w") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))

    total = sum(len(v) for v in houses.values())
    print(f"\nулиц: {len(houses)}  адресов с координатами: {total}")


main()
