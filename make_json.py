from __future__ import annotations
import os, json, sys, traceback, datetime as dt, time
from pathlib import Path
import gc

import pipeline

# ===================== OUTPUT =============================
OUTDIR = Path("public")
OUTDIR.mkdir(exist_ok=True)

def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

# ===================== CACHE ==============================
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# hur länge vi spar artiklar i cachen
CACHE_DAYS = int(os.getenv("CACHE_DAYS", "45"))
MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "5000"))  # hårt tak per (cat,lang)

def _cache_path(category: str, lang: str) -> Path:
    return CACHE_DIR / f"{category}_{lang}.ndjson"

def _now_ts() -> float:
    return time.time()

def _coalesce_ts(doc: dict) -> float:
    """Publiceringstid i epoch; fall back till first_seen_ts om saknas."""
    ts = doc.get("published")
    return float(ts) if ts else float(doc.get("first_seen_ts", _now_ts()))

def load_cache(category: str, lang: str) -> list[dict]:
    path = _cache_path(category, lang)
    items: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    return items

def save_cache(category: str, lang: str, items: list[dict]) -> None:
    path = _cache_path(category, lang)
    # sortera nyast först (på coalesced timestamp)
    items_sorted = sorted(items, key=_coalesce_ts, reverse=True)[:MAX_CACHE_ITEMS]
    with path.open("w", encoding="utf-8") as f:
        for it in items_sorted:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"💾 Cache sparad: {path}  ({len(items_sorted)} st)", flush=True)

def update_cache_with_new(cache_items: list[dict], new_docs: list[dict]) -> list[dict]:
    """Mergea nya artiklar in i cachen (per URL), behåll längre text/rubrik."""
    now_ts = _now_ts()
    by_url = {it["url"]: it for it in cache_items if it.get("url")}
    added = 0
    updated = 0
    for d in new_docs:
        url = d.get("url")
        if not url:
            continue
        cur = by_url.get(url)
        if cur is None:
            nd = dict(d)  # kopia
            nd.setdefault("first_seen_ts", now_ts)
            by_url[url] = nd
            added += 1
        else:
            # uppdatera om ny text är längre (mer komplett) eller titel bättre
            if len(d.get("text","")) > len(cur.get("text","")):
                cur["text"] = d["text"]
            if len(d.get("title","")) >= len(cur.get("title","")):
                cur["title"] = d.get("title","")
            # ta in published om saknades
            if not cur.get("published") and d.get("published"):
                cur["published"] = d["published"]
            updated += 1
    print(f"↺ Cache merge: +{added} nya, ~{updated} uppdaterade", flush=True)
    return list(by_url.values())

def prune_cache(items: list[dict]) -> list[dict]:
    """Släng allt äldre än CACHE_DAYS (på publicerad-eller-first_seen)."""
    cutoff = _now_ts() - CACHE_DAYS * 86400.0
    kept = [it for it in items if _coalesce_ts(it) >= cutoff]
    dropped = len(items) - len(kept)
    if dropped:
        print(f"🧹 Cache prune: −{dropped} äldre än {CACHE_DAYS} dagar", flush=True)
    return kept

# ===================== KONFIG =============================
CATEGORIES = list((pipeline.SOURCES or {}).keys()) or ["world"]
LANGS = sorted({lg for cat in (pipeline.SOURCES or {}).values() for lg in (cat or {}).keys()}) or ["sv","en"]

# Standard: bygg alla tre spann
SPAN_INFO = [
    ("day",   1,  3),   # 24 h
    ("week",  7,  3),   # 7 dygn
    ("month", 30, 3),   # 30 dygn
]

# Env-override för snabbtest
TOPN = os.getenv("TOPN")
DAYS = os.getenv("DAYS")
if TOPN or DAYS:
    SPAN_INFO = [("day", int(DAYS or 1), int(TOPN or 3))]

KEEP_MODELS_LOADED = os.getenv("KEEP_MODELS_LOADED", "0") == "1"

# ===================== HUVUDSLINGA ========================
for category in CATEGORIES:
    for lang in LANGS:
        print(f"\n=== {category}/{lang} ===", flush=True)

        # 1) Läs in och uppdatera cache med dagens insamling
        cache_items = load_cache(category, lang)

        max_days = max(days for _, days, _ in SPAN_INFO)
        raw_cap = 200 if max_days >= 30 else 120
        print(f"⏳ Hämtar nya artiklar (≤ {max_days} dygn, cap {raw_cap})…", flush=True)
        new_docs = pipeline.collect_articles(category, lang, days=max_days, raw_limit=raw_cap)

        cache_items = update_cache_with_new(cache_items, new_docs)
        cache_items = prune_cache(cache_items)
        save_cache(category, lang, cache_items)

        # 2) Bygg spans från CACHEN (inte från dagens RSS)
        seen_urls: set[str] = set()

        for span, days, topn in SPAN_INFO:
            if topn <= 0:
                print(f"⏭️  Skippar {span} (top_n={topn})")
                continue

            print(f"\n--- {span.upper()}  ({days} dygn, top {topn}) ---", flush=True)
            cutoff = _now_ts() - days * 86400.0

            # kandidater = allt i cachen inom tidsspannet
            candidates = [it for it in cache_items if _coalesce_ts(it) >= cutoff]

            # ranka + dedupe mot tidigare span
            docs_rank = pipeline.choose_top_docs(
                candidates, top_n=topn, span=span, exclude_urls=seen_urls
            )
            seen_urls.update(d.get("url") for d in docs_rank)

            # bygg kort
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
                "total_raw":     len(candidates),
                "total_rank":    len(docs_rank),
                "min_required":  pipeline.MIN_REQUIRED,
                "fallback_used": len(docs_rank) < pipeline.MIN_REQUIRED,
                "cards":         cards,
            }
            fname = OUTDIR / f"{category}_{span}_{lang}.json"
            write_json(fname, payload)
            print(f"✅ {fname}", flush=True)

        # 3) Släpp sammanfattningsmodeller mellan språk (minne)
        if not KEEP_MODELS_LOADED:
            pipeline.unload_summarizer(lang)
            gc.collect()

# 4) Indexfil
index = {
    "categories": CATEGORIES,
    "languages": LANGS,
    "spans": [s for (s, _days, top) in SPAN_INFO if top > 0],
    "generated": utc_now_iso(),
}
write_json(OUTDIR / "index.json", index)
print("✅ public/index.json", flush=True)
