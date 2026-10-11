"""Check that the Vietnamese deck shows exactly the same numbers as the English deck.

For every slide, collects the numeric tokens of all text (title, body, source line, speaker
notes, SVG labels) and compares the multisets of the two decks. Month names are written as words
in Vietnamese ("tháng Mười", "tháng Mười Hai") so that no digit is added.

Usage: .venv/Scripts/python.exe scripts/check_deck_numbers.py
"""

from __future__ import annotations

import html
import re
import sys
from collections import Counter

from nbp.paths import ROOT

D = ROOT / "reports" / "summary" / "week5_final"
EN, VI = D / "final_slides_v1.html", D / "final_slides_v1_vi.html"
NUM = re.compile(r"\d+(?:[.,]\d+)*%?")


def per_slide(path) -> dict[int, Counter]:
    page = path.read_text(encoding="utf-8")
    out = {}
    for n, body in re.findall(
        r'<section class="[^"]*" data-n="(\d+)">(.*?)</section>', page, re.DOTALL
    ):
        text = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<defs>.*?</defs>", " ", body)))
        out[int(n)] = Counter(NUM.findall(text))
    return out


def main() -> int:
    en, vi = per_slide(EN), per_slide(VI)
    bad = 0
    if en.keys() != vi.keys():
        print("slide count differs", len(en), len(vi))
        return 1
    total = 0
    for n in sorted(en):
        total += sum(en[n].values())
        extra, missing = vi[n] - en[n], en[n] - vi[n]
        if extra or missing:
            bad += 1
            print(f"slide {n}: only in VI {dict(extra)}; only in EN {dict(missing)}")
    print(f"{len(en)} slides, {total} numeric tokens compared, {bad} slides differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
