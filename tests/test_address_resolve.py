"""Проверка адресной логики на РЕАЛЬНЫХ подписях кандидатов.

Сеть и Vision не нужны — проверяются ранжирование гипотез и флаг подтверждения
улицы независимым источником.

Запуск:
    python tests/test_address_resolve.py

Кейсы взяты из живых прогонов и закрывают два реальных промаха:
  * «Ул.Губернского 2» без пробела после точки не извлекался регуляркой,
    и побеждала единственная подпись с пробелом — «Губернского 32»;
  * все подписи вели на другой корпус того же колледжа (ул. Рубина, 5),
    и сервис уверенно выдавал адрес в двух километрах от правильного.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from app.address_resolve import collect_evidence

CASES = {
    "gubernskogo2 (правильно: Губернского 2)": {
        "expect_top": ("губернского", "2"),
        "expect_corroborated": True,
        "candidates": [
            ("Губернского 32 новороссийск Shtampik.com", 1.00),
            ("File:Ул.Губернского 2.jpg - Wikimedia Commons", 1.00),
            ("В Новороссийске отреставрируют здание бывшего женского училища", 0.95),
            ("Category:Buildings in Novorossiysk - Wikimedia Commons", 0.90),
            ("Особенности морского агентства ИпотекЦентр в Новороссийске, улица Губернского, 4", 0.90),
        ],
        "tags": ["губернского 2а новороссийск", "советов 26 новороссийск",
                 "новороссийск ул.новороссийской республики 28а", "ул.пролетарская новороссийск"],
    },
    "sovetov38 (правильно: Советов 38; все кандидаты ведут на Рубина 5)": {
        "expect_top": ("рубина", "5"),
        "expect_corroborated": False,   # улицу не называет ни один независимый источник
        "candidates": [
            ("Новороссийский колледж строительства и экономики, колледж, ул. Рубина, 5", 0.95),
            ("Панорама: Роспечать, точка продажи прессы, ул. Рубина, 5, Новороссийск", 0.90),
            ("Фото: Новороссийский колледж строительства и экономики, ул. Рубина, 5", 0.90),
        ],
        "tags": ["новороссийский колледж строительства и экономики", "нксэ новороссийск",
                 "новороссийск ул.новороссийской республики 28а"],
    },
    "svobody23 (правильно: Свободы 23)": {
        "expect_top": ("свободы", "23"),
        "expect_corroborated": True,    # улица «свободы» есть в подсказках Яндекса
        "candidates": [
            ("Новороссийский медицинский колледж, ул. Свободы, 23", 1.00),
            ("Фото: ГБПОУ Новороссийский медицинский колледж, ул. Свободы, 23, Новороссийск", 0.95),
        ],
        "tags": ["новороссийский медицинский колледж", "свободы 54 новороссийск",
                 "улица свободы 35 новороссийск"],
    },
}

ok = True
for label, case in CASES.items():
    rows = collect_evidence(
        verified_candidates=[{"title": t, "confidence": c, "image_url": f"u{i}"}
                             for i, (t, c) in enumerate(case["candidates"])],
        yandex_tags=case["tags"], ocr_text="", user_address="",
    )
    top = rows[0] if rows else None
    got = (top["street"], top["house"]) if top else None
    corr = bool(top and top["street_corroborated"])
    exp_house = case["expect_top"][1]
    hit = bool(top and top["street"] == case["expect_top"][0]
               and top["house"].rstrip("абвг") == exp_house)
    corr_ok = corr == case["expect_corroborated"]
    ok &= hit and corr_ok
    print(f"{'PASS' if hit and corr_ok else 'FAIL'}  {label}")
    print(f"      топ-гипотеза: {got}  подтверждение улицы: {corr} (ожидалось {case['expect_corroborated']})")
    for r in rows[:3]:
        print(f"        {r['street']:16s} {r['house']:5s} score={r['score']:6.1f} "
              f"улица подтв={r['street_corroborated']} источники={r['sources']}")
print("\nИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ ПАДЕНИЯ")
sys.exit(0 if ok else 1)
