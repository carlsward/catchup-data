from __future__ import annotations
from zoneinfo import ZoneInfo
import os, json, sys, traceback, datetime as dt, time
from pathlib import Path
import gc

import pipeline
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Stockholm"))

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

CACHE_DAYS = int(os.getenv("CACHE_DAYS", "45"))
MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "5000"))

def _cache_path(category: str, lang: str) -> Path:
    return CACHE_DIR / f"{category}_{lang}.ndjson"

def _now_ts() -> float:
    return time.time()

def _coalesce_ts(doc: dict) -> float:
    ts = doc.get("published")
    return float(ts) if ts else float(doc.get("first_seen_ts", _now_ts()))

def _day_bounds_local_utc(days_ago: int) -> tuple[float, float]:
    """Start/slut för kalenderdygn i LOCAL_TZ, konverterat till UTC-epoch."""
    now_local = dt.datetime.now(LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    start_local = now_local - dt.timedelta(days=days_ago)
    end_local   = start_local + dt.timedelta(days=1)
    start_utc = start_local.astimezone(dt.timezone.utc).timestamp()
    end_utc   = end_local.astimezone(dt.timezone.utc).timestamp()
    return start_utc, end_utc

def _filter_by_window(items: list[dict], start_ts: float, end_ts: float) -> list[dict]:
    return [it for it in items if start_ts <= _coalesce_ts(it) < end_ts]


def load_cache(category: str, lang: str) -> list[dict]:
    p = _cache_path(category, lang)
    items: list[dict] = []
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
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
    p = _cache_path(category, lang)
    items_sorted = sorted(items, key=_coalesce_ts, reverse=True)[:MAX_CACHE_ITEMS]
    with p.open("w", encoding="utf-8") as f:
        for it in items_sorted:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"💾 Cache sparad: {p}  ({len(items_sorted)} st)", flush=True)

def update_cache_with_new(cache_items: list[dict], new_docs: list[dict]) -> list[dict]:
    now_ts = _now_ts()
    by_url = {it["url"]: it for it in cache_items if it.get("url")}
    added = updated = 0
    for d in new_docs:
        url = d.get("url")
        if not url:
            continue
        cur = by_url.get(url)
        if cur is None:
            nd = dict(d)
            nd.setdefault("first_seen_ts", now_ts)
            by_url[url] = nd
            added += 1
        else:
            if len(d.get("text","")) > len(cur.get("text","")):
                cur["text"] = d["text"]
            if len(d.get("title","")) >= len(cur.get("title","")):
                cur["title"] = d.get("title","")
            if not cur.get("published") and d.get("published"):
                cur["published"] = d["published"]
            updated += 1
    print(f"↺ Cache merge: +{added} nya, ~{updated} uppdaterade", flush=True)
    return list(by_url.values())

def prune_cache(items: list[dict]) -> list[dict]:
    cutoff = _now_ts() - CACHE_DAYS * 86400.0
    kept = [it for it in items if _coalesce_ts(it) >= cutoff]
    dropped = len(items) - len(kept)
    if dropped:
        print(f"🧹 Cache prune: −{dropped} äldre än {CACHE_DAYS} dagar", flush=True)
    return kept

# ===================== KONFIG =============================
CATEGORIES = list((pipeline.SOURCES or {}).keys()) or ["world"]
LANGS = ["sv"]  # ← bara svenska

# Bucket-policy: strikt per-dag-urval (ingen cross-dagsfyllning)
BUCKET_POLICY = {
    "week":  {"mode": "strict_daily", "days": 7,  "per_day": 2},  # 2/dag → upp till 14
    "month": {"mode": "strict_daily", "days": 30, "per_day": 1},  # 1/dag → upp till 30
}


# Antal kort per span (matcha bucket-policy där det finns)
SPAN_INFO = [
    ("day",   1,  5),   # 24 h: 5
    ("week",  7,  14),  # 7 d:  14 (2/dag)
    ("month", 30, 30),  # 30 d: 30 (1/dag)
]

# Env-override för snabbtest
TOPN = os.getenv("TOPN")
DAYS = os.getenv("DAYS")
if TOPN or DAYS:
    SPAN_INFO = [("day", int(DAYS or 1), int(TOPN or 5))]

KEEP_MODELS_LOADED = os.getenv("KEEP_MODELS_LOADED", "0") == "1"

def _debug_day_hist(cands: list[dict], days: int):
    from math import ceil
    rows = []
    for i in range(days):
        start, end = _day_bounds_local_utc(i)
        rows.append(len(_filter_by_window(cands, start, end)))
    print(f"📊 Kandidater per dag (nyast→äldst, {days}d): {rows}", flush=True)


# ===================== BUCKET-URVAL =======================

def pick_strict_daily(
    candidates: list[dict],
    *,               # tvinga namngivna argument
    days: int,
    per_day: int,
    span: str,
) -> list[dict]:
    """
    Försök: EXAKT per_day per lokal kalenderdag senaste `days`.
    1) Välj först max 1/artikelkälla per dag.
    2) Fyll upp inom dagen till per_day.
    3) Om några dagar ändå saknas -> global uppfyllnad över hela spannet
       (utan dagskrav) för att nå 'days*per_day'.
    """
    target = days * per_day
    seen: set[str] = set()
    picked: list[dict] = []

    # Steg A: dag-för-dag
    for i in range(days):
        start, end = _day_bounds_local_utc(i)
        day_docs = _filter_by_window(candidates, start, end)
        if not day_docs:
            continue

        # A1) Max 1 per domän
        first = pipeline.choose_top_docs(
            day_docs,
            top_n=per_day,
            span="month",            # neutral recency
            exclude_urls=seen,
            max_per_domain=1,
        )
        chosen: list[dict] = list(first)

        # A2) Fyll upp till per_day inom dagen
        if len(chosen) < per_day:
            remaining = [
                d for d in day_docs
                if d.get("url") not in {x.get("url") for x in chosen}
                and d.get("url") not in seen
            ]
            if remaining:
                fill = pipeline.choose_top_docs(
                    remaining,
                    top_n=per_day - len(chosen),
                    span="month",
                    exclude_urls=seen,
                    max_per_domain=per_day,  # låt samma domän få ännu en om det behövs
                )
                chosen += fill

        for d in chosen[:per_day]:
            u = d.get("url")
            if u and u not in seen:
                picked.append(d)
                seen.add(u)

    # Steg B: Global uppfyllnad om vi inte nått målet
    if len(picked) < target:
        rest = [d for d in candidates if d.get("url") not in seen]
        if rest:
            add = pipeline.choose_top_docs(
                rest,
                top_n=target - len(picked),
                span="month",
                exclude_urls=seen,
                # balansera källor över helheten (dynamisk cap)
                max_per_domain=pipeline.domain_cap(candidates, target),
            )
            for d in add:
                u = d.get("url")
                if u and u not in seen:
                    picked.append(d)
                    seen.add(u)

    # Nyast först i hela spannet
    picked.sort(key=lambda it: _coalesce_ts(it), reverse=True)
    return picked[:target]





def pick_by_buckets(
    candidates: list[dict],
    span: str,
    buckets: int,
    per_bucket: int,
    exclude_urls: set[str] | None = None,
) -> list[dict]:
    now = _now_ts()
    seen = set(exclude_urls or set())
    picked: list[dict] = []
    per_domain: dict[str, int] = {}

    target = buckets * per_bucket
    global_cap = pipeline.domain_cap(candidates, target)  # dynamiskt per domän över HELA urvalet

    for i in range(buckets):
        end = now - i * 86400.0
        start = end - 86400.0
        bucket_docs = _filter_by_window(candidates, start, end)
        if not bucket_docs:
            continue

        cand = [d for d in bucket_docs if d.get("url") not in seen]

        # välj bästa i dagens fack – utan recency-bias, och låt choose_top_docs föreslå fler,
        # vi vaktar global domän-cap när vi lägger till.
        sel = pipeline.choose_top_docs(
            cand, top_n=per_bucket, span="month", exclude_urls=seen
        )

        for d in sel:
            if d.get("url") in seen:
                continue
            dom = d.get("domain", "")
            if per_domain.get(dom, 0) >= global_cap:
                continue
            picked.append(d)
            seen.add(d.get("url"))
            per_domain[dom] = per_domain.get(dom, 0) + 1

        if len(picked) >= target:
            break

    # fyll upp om vi saknar
    need = target - len(picked)
    if need > 0:
        rest = [d for d in candidates if d.get("url") not in seen]
        extra = pipeline.choose_top_docs(rest, top_n=need, span="month", exclude_urls=seen)
        for d in extra:
            dom = d.get("domain", "")
            if per_domain.get(dom, 0) >= global_cap:
                continue
            picked.append(d)
            seen.add(d.get("url"))
            per_domain[dom] = per_domain.get(dom, 0) + 1
            if len(picked) >= target:
                break

    return picked[:target]


# ===================== HUVUDSLINGA ========================
for category in CATEGORIES:
    for lang in LANGS:
        print(f"\n=== {category}/{lang} ===", flush=True)

        # 1) Läs/uppdatera cache med dagens insamling (upp till största span)
        cache_items = load_cache(category, lang)
        max_days = max(days for _, days, _ in SPAN_INFO)
        raw_cap = 400 if max_days >= 30 else 200
        print(f"⏳ Hämtar nya artiklar (≤ {max_days} dygn, cap {raw_cap})…", flush=True)
        new_docs = pipeline.collect_articles(category, lang, days=max_days, raw_limit=raw_cap)

        cache_items = update_cache_with_new(cache_items, new_docs)
        cache_items = prune_cache(cache_items)
        save_cache(category, lang, cache_items)

        # 2) Bygg spans från cache – OBS: ingen dedup mellan spans
        for span, days, topn in SPAN_INFO:
            if topn <= 0:
                print(f"⏭️  Skippar {span} (top_n={topn})")
                continue

            print(f"\n--- {span.upper()}  ({days} dygn, top {topn}) ---", flush=True)
            cutoff = _now_ts() - days * 86400.0
            candidates = [it for it in cache_items if _coalesce_ts(it) >= cutoff]
            _debug_day_hist(candidates, days)


            policy = BUCKET_POLICY.get(span)
            if policy and policy.get("mode") == "strict_daily":
                docs_rank = pick_strict_daily(
                    candidates,
                    days=policy["days"],
                    per_day=policy["per_day"],
                    span=span,
                )

            elif policy:
                # (behåll din gamla bucket-funktion om du vill kunna slå på den senare)
                docs_rank = pick_by_buckets(
                    candidates,
                    span=span,
                    buckets=policy["buckets"],
                    per_bucket=policy["per_bucket"],
                    exclude_urls=None,      # ingen cross-span-dedup
                )
            else:
                docs_rank = pipeline.choose_top_docs(
                    candidates,
                    top_n=topn,
                    span=span,
                    exclude_urls=None,      # ingen cross-span-dedup
                )


            # Bygg kort
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

        if not KEEP_MODELS_LOADED:
            pipeline.unload_summarizer(lang)
            gc.collect()

# 3) Indexfil
index = {
    "categories": CATEGORIES,
    "languages": LANGS,
    "spans": [s for (s, _days, top) in SPAN_INFO if top > 0],
    "generated": utc_now_iso(),
}
write_json(OUTDIR / "index.json", index)
print("✅ public/index.json", flush=True)

