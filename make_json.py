from __future__ import annotations

"""
Genererar JSON-filer (kategori × (day|week|month) × språk) till `public/`.

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

# ------------------ HJÄLP -------------------------------
def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

# ------------------ KONFIG -------------------------------
# Justera efter dina källor (sources.yaml)
CATEGORIES = list((pipeline.SOURCES or {}).keys()) or ["world"]
LANGS = sorted({lg for cat in (pipeline.SOURCES or {}).values() for lg in (cat or {}).keys()}) or ["sv","en","de","es","fr","el"]

SPAN_INFO = [          #  (filnamns-prefix, antal dagar, hur många kort)
    ("day",   1,   6),
    ("week",  7,  25),
    ("month", 30,  40),
]

# -----------------------------------------------------------
#                       HUVUDSLINGA
# -----------------------------------------------------------
for span, days, topn in SPAN_INFO:
    print(f"\n=== {span.upper()}  ({days} dygn, top {topn}) ===", flush=True)

    for category in CATEGORIES:
        for lang in LANGS:
            print(f"\n--- {category}/{lang} ------------------------------", flush=True)

            # 1) hämta & ranka
            docs_raw  = pipeline.collect_articles(category, lang, days)
            docs_rank = pipeline.choose_top_docs(docs_raw, topn)

            fallback_used = len(docs_rank) < pipeline.MIN_REQUIRED

            # 2) bygg kort med tydlig progress-logg
            cards = []
            for idx, doc in enumerate(docs_rank, 1):
                title_preview = (doc.get("title","")[:60]).replace("\n", " ")
                print(f"▶️  {lang} {idx:02}/{len(docs_rank)}  {title_preview}", flush=True)
                try:
                    cards.append(pipeline.make_card(doc))
                except Exception as exc:
                    # logga och fortsätt – enstaka fel får inte döda hela jobben
                    print(f"⚠️  Skippades p.g.a. fel: {exc}", file=sys.stderr, flush=True)
                    traceback.print_exc()

            # 3) skriv ut JSON-filen
            payload = {
                "span":       span,
                "language":   lang,
                "category":   category,
                "generated":  utc_now_iso(),
                "total_raw":  len(docs_raw),
                "total_rank": len(docs_rank),
                "min_required": pipeline.MIN_REQUIRED,
                "fallback_used": fallback_used,
                "cards":      cards,
            }
            fname = OUTDIR / f"{category}_{span}_{lang}.json"
            write_json(fname, payload)
            print(f"✅ {fname}", flush=True)
