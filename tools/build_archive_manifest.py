from __future__ import annotations

import argparse
import json
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/archive.json from folders under data/archive")
    parser.add_argument("--root", default="data/archive")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTS:
            continue
        rel = path.relative_to(root).as_posix()
        rows.append({
            "path": rel,
            "year": "",
            "address": "",
            "description": "",
            "source": "",
        })
    out = root.parent / "archive.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {len(rows)} archive images -> {out}")


if __name__ == "__main__":
    main()
