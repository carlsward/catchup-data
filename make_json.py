"""
Genererar alla JSON-filer (day/week/month × språk) och sparar i
mappen `public/`.  Körs automatiskt av GitHub Actions nattligen,
men kan givetvis köras lokalt för test:

    python make_json.py
"""
from pathlib import Path
import json, datetime as dt
import pipeline

OUTDIR = Path("public")
OUTDIR.mkdir(exist_ok=True)

# ------------ verktyg -------------------------------------

def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

# ------------ vilka filer som ska skapas ------------------

SPAN_INFO = [
    ("day",   1,  5),
    ("week",  7, 10),
    ("month", 30, 20),
]
LANGS = ["en", "sv", "de", "es", "fr"]

# ------------ huvudslingan --------------------------------
for span, days, topn in SPAN_INFO:
    # 1) hämta & ranka artiklar EN gång per span
    docs_raw  = pipeline.collect_articles(days)
    docs_rank = pipeline.choose_top_docs(docs_raw, topn)

    for lang in LANGS:
        cards = [pipeline.make_card(d, lang) for d in docs_rank]

        payload = {
            "span":      span,
            "language":  lang,
            "generated": dt.datetime.utcnow()
                            .isoformat(timespec="seconds") + "Z",
            "cards":     cards,
        }
        fname = OUTDIR / f"{span}_{lang}.json"
        write_json(fname, payload)
        print(f"✅ {fname}")
