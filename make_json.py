from __future__ import annotations

"""
Genererar JSON-filer (day|week|month × språk) till `public/`.

Körs nattligen av GitHub Actions, men kan även köras lokalt:

    python make_json.py
"""

import json
import sys
import traceback
import datetime as dt
from pathlib import Path

import pipeline

# ------------------ OUTPUTMAPP -----------------------------
OUTDIR = Path("public")
OUTDIR.mkdir(exist_ok=True)

# ------------------ VERKTYG -------------------------------
def write_json(path: Path, obj) -> None:
    """Skriv prettifierad UTF-8-JSON till given fil."""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def utc_now_iso() -> str:
    """Timezone-aware ISO-8601 (sekundprecision)."""
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

# ------------------ VILKA FILER? ---------------------------
SPAN_INFO = [          #  (filnamns-prefix, antal dagar, hur många kort)
    ("day",   1,  3),
    ("week",  7, 0),
    ("month", 30, 0),
]

LANGS = pipeline.TARGET_LANGS     # hålls i synk med pipeline.py

# -----------------------------------------------------------
#                       HUVUDSLINGA
# -----------------------------------------------------------
for span, days, topn in SPAN_INFO:
    print(f"\n=== {span.upper()}  ({days} dygn, top {topn}) ===", flush=True)

    # 1) hämta & ranka EN gång per span
    docs_raw  = pipeline.collect_articles(days)
    docs_rank = pipeline.choose_top_docs(docs_raw, topn)

    for lang in LANGS:
        print(f"--- {lang} ------------------------------", flush=True)
        cards = []

        # 2) bygg kort med tydlig progress-logg
        for idx, doc in enumerate(docs_rank, 1):
            title_preview = doc["title"][:60].replace("\n", " ")
            print(f"▶️  {lang} {idx:02}/{len(docs_rank)}  {title_preview}", flush=True)
            try:
                cards.append(pipeline.make_card(doc, lang))
            except Exception as exc:
                # logga och fortsätt – enstaka fel får inte döda hela jobben
                print(f"⚠️  Skippades p.g.a. fel: {exc}", file=sys.stderr, flush=True)
                traceback.print_exc()

        # 3) skriv ut JSON-filen
        payload = {
            "span":      span,
            "language":  lang,
            "generated": utc_now_iso(),
            "cards":     cards,
        }
        fname = OUTDIR / f"{span}_{lang}.json"
        write_json(fname, payload)
        print(f"✅ {fname}", flush=True)
