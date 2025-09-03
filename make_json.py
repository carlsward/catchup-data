from __future__ import annotations
import os, json, sys, traceback, datetime as dt
from pathlib import Path
import gc
import pipeline

OUTDIR = Path("public"); OUTDIR.mkdir(exist_ok=True)

def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

CATEGORIES = list((pipeline.SOURCES or {}).keys()) or ["world"]
LANGS = sorted({lg for cat in (pipeline.SOURCES or {}).values() for lg in (cat or {}).keys()}) or ["sv","en","de","fr","es","el"]

SPAN_INFO = [
    ("day",   1,  3),   # 24 h: 6 kort
    ("week",  7, 3),   # 7 dygn: 20 kort
    ("month", 30, 3),  # 30 dygn: 30 kort
]
TOPN = os.getenv("TOPN"); DAYS = os.getenv("DAYS")
if TOPN or DAYS:
    SPAN_INFO = [("day", int(DAYS or 1), int(TOPN or 3))]

KEEP_MODELS_LOADED = os.getenv("KEEP_MODELS_LOADED", "0") == "1"

for span, days, topn in SPAN_INFO:
    if topn <= 0:
        print(f"⏭️  Skippar {span} (top_n={topn})"); continue
    print(f"\n=== {span.upper()}  ({days} dygn, top {topn}) ===", flush=True)

    for category in CATEGORIES:
        for lang in LANGS:
            print(f"\n--- {category}/{lang} ------------------------------", flush=True)

            docs_raw  = pipeline.collect_articles(category, lang, days)
            docs_rank = pipeline.choose_top_docs(docs_raw, top_n=topn, span=span)

            fallback_used = len(docs_rank) < pipeline.MIN_REQUIRED

            cards = []
            for idx, doc in enumerate(docs_rank, 1):
                title_preview = (doc.get("title","")[:60]).replace("\n", " ")
                print(f"▶️  {lang} {idx:02}/{len(docs_rank)}  {title_preview}", flush=True)
                try:
                    cards.append(pipeline.make_card(doc))
                except Exception as exc:
                    print(f"⚠️  Skippades p.g.a. fel: {exc}", file=sys.stderr, flush=True)
                    traceback.print_exc()

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

            # D: släpp modellminne per språk om vi inte vill cacha under körning
            if not KEEP_MODELS_LOADED:
                pipeline.unload_summarizer(lang)
                del cards, docs_raw, docs_rank
                gc.collect()

index = {
    "categories": CATEGORIES,
    "languages": LANGS,
    "spans": [s for s,_,_ in SPAN_INFO if _ > 0],
    "generated": utc_now_iso(),
}
write_json(OUTDIR / "index.json", index)
print("✅ public/index.json", flush=True)
