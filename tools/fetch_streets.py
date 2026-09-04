"""Выкачивает адреса Новороссийска из OpenStreetMap через Overpass.

Запуск разовый, результат коммитится в data/streets_novorossiysk.json:
    python tools/fetch_streets.py
"""
import collections, json, re, urllib.parse, urllib.request

# Границы города по координатам надёжнее, чем поиск area по имени: у
# Новороссийска несколько административных отношений с разным admin_level.
BBOX = "44.58,37.62,44.82,37.93"

QUERY = f"""
[out:json][timeout:180];
(
  node({BBOX})["addr:street"]["addr:housenumber"];
  way({BBOX})["addr:street"]["addr:housenumber"];
);
out tags center;
"""

OUT_PATH = "data/streets_novorossiysk.json"


def house_sort(house):
    match = re.match(r"(\d+)", house)
    return (int(match.group(1)) if match else 99999, house)


def main():
    payload = urllib.parse.urlencode({"data": QUERY}).encode()
    request = urllib.request.Request(
        "https://overpass-api.de/api/interpreter", data=payload,
        headers={"User-Agent": "AiWebCity/1.2 (city history project)"},
    )
    raw = json.load(urllib.request.urlopen(request, timeout=300))

    streets = collections.defaultdict(set)
    coords = {}
    for element in raw.get("elements", []):
        tags = element.get("tags") or {}
        street = (tags.get("addr:street") or "").strip()
        house = (tags.get("addr:housenumber") or "").strip()
        if not street or not house:
            continue
        streets[street].add(house)
        if street not in coords:
            center = element.get("center") or {"lat": element.get("lat"), "lon": element.get("lon")}
            if center.get("lat"):
                coords[street] = [round(float(center["lat"]), 5), round(float(center["lon"]), 5)]

    result = {
        "city": "Новороссийск",
        "source": "OpenStreetMap via Overpass API",
        "streets": {s: sorted(h, key=house_sort) for s, h in sorted(streets.items())},
        "street_coords": coords,
    }
    with open(OUT_PATH, "w") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))

    total = sum(len(v) for v in streets.values())
    print(f"улиц: {len(streets)}  адресов: {total}")
    for street in list(sorted(streets))[:6]:
        print(f"   {street}: {len(streets[street])} домов, напр. {sorted(streets[street], key=house_sort)[:6]}")


main()

