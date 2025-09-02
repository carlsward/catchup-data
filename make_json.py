from __future__ import annotations

"""
Genererar JSON-filer (kategori × span × språk) till `public/`.

Kör lokalt:
    python make_json.py
Eller i GitHub Actions (se build.yml).
"""

import os
import json
import sys
import traceback
import datetime as dt
from pathlib import Path

import pipeline

# ------------------ OUTPUT -------------------------------
OUTDIR = Path("public")
OUTDIR.mkdir(exist_ok=True)

def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

# ------------------ KONFIG -------------------------------
# Hämtas från sources.yaml. Fallback om filen är tom.
CATEGORIES = list((pipeline.SOURCES or {}).keys()) or ["world"]
LANGS = sorted({lg for cat in (pipeline.SOURCES or {}).values() for lg in (cat or {}).keys()}) or ["sv","en","de","fr","es","el"]

# Standard: beta-körning → bara 24h och 3 kort
SPAN_INFO = [
    ("day",   1, 3),
    # ("week",  7, 25),
    # ("month", 30, 40),
]

# Miljövariabler för snabb test utan kodändring
TOPN = os.getenv("TOPN")
DAYS = os.getenv("DAYS")
if TOPN or DAYS:
    topn = int(TOPN or 3)
    days = int(DAYS or 1)
    SPAN_INFO = [("day", days, topn)]

# ------------------ HUVUDSLINGA --------------------------
for span, days, topn in SPAN_INFO:
    if topn <= 0:
        print(f"⏭️  Skippar {span} (top_n={topn})")
        continue
    print(f"\n=== {span.upper()}  ({days} dygn, top {topn}) ===", flush=True)

    for category in CATEGORIES:
        for lang in LANGS:
            print(f"\n--- {category}/{lang} ------------------------------", flush=True)

            # 1) hämta & ranka
            docs_raw  = pipeline.collect_articles(category, lang, days)
            docs_rank = pipeline.choose_top_docs(docs_raw, topn)

            fallback_used = len(docs_rank) < pipeline.MIN_REQUIRED

            # 2) bygg kort
            cards = []
            for idx, doc in enumerate(docs_rank, 1):
                title_preview = (doc.get("title","")[:60]).replace("\n", " ")
                print(f"▶️  {lang} {idx:02}/{len(docs_rank)}  {title_preview}", flush=True)
                try:
                    cards.append(pipeline.make_card(doc))
                except Exception as exc:
                    print(f"⚠️  Skippades p.g.a. fel: {exc}", file=sys.stderr, flush=True)
                    traceback.print_exc()

            # 3) skriv ut
            payload = {
                "span":          span,
                "language":      lang,
                "category":      category,
                "generated":     utc_now_iso(),
                "total_raw":     len(docs_raw),
                "total_rank":    len(docs_rank),
                "min_required":  pipeline.MIN_REQUIRED,
                "fallback_used": fallback_used,
                "cards":         cards,
            }
            fname = OUTDIR / f"{category}_{span}_{lang}.json"
            write_json(fname, payload)
            print(f"✅ {fname}", flush=True)

# (Valfritt) index för klienter
index = {
    "categories": CATEGORIES,
    "languages": LANGS,
    "spans": [s for s,_,_ in SPAN_INFO if _ > 0],
    "generated": utc_now_iso(),
}
write_json(OUTDIR / "index.json", index)
print("✅ public/index.json", flush=True)
