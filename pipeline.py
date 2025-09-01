from __future__ import annotations

# -----------------------------------------------------------
#  CatchUp · nyhetsinsamling · ranking · summering (utan MT)
# -----------------------------------------------------------
# - Hämtar artiklar från RSS-flöden per (kategori, språk)
# - Rankar med BM25 + recency + källdiversitet
# - Summerar på källspråk med flerspråkig summarizer (mT5/XLSum)
# - Export sker från make_json.py
# -----------------------------------------------------------

# ---- shim så newspaper funkar med lxml>=5 -----------------
try:
    import lxml_html_clean as _hc
    import sys
    sys.modules["lxml.html.clean"] = _hc
except ImportError:
    pass
# -----------------------------------------------------------

import math
import time
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
from urllib.parse import urlparse
import feedparser
import newspaper
import yaml
from rank_bm25 import BM25Okapi
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------------------------------------
#  INSTÄLLNINGAR
# -----------------------------------------------------------

# Om du inte har sources.yaml används en tom fallback (ger tomma resultat).
SOURCES_PATH = Path("sources.yaml")

# Minimikrav per vy; om färre hittas markeras fallback_used i JSON.
MIN_REQUIRED = 6

# Hur mycket vi hämtar/rankar innan topp-N (per språk/kategori)
RAW_LIMIT = 200
MAX_PER_DOMAIN = 4
MIN_TEXT_CHARS = 400

# Hämtning
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) CatchUpBot/1.0"
REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0.2  # artighet mellan sidladdningar (sek)

# Summarizer (flerspråkig)
SUMMARIZER_MODEL = "csebuetnlp/mT5_multilingual_XLSum"

# -----------------------------------------------------------
#  KÄLLLADDNING
# -----------------------------------------------------------

def load_sources() -> Dict[str, Dict[str, List[str]]]:
    """
    Läser sources.yaml med struktur:
      category:
        lang:
          - https://.../rss
    """
    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print("⚠️  sources.yaml är ogiltig – förväntade en mapping.")
            return {}
        return data
    print("ℹ️  Hittade ingen sources.yaml – kör med tomma källor.")
    return {}

SOURCES = load_sources()

# -----------------------------------------------------------
#  NEWSPAPER KONFIG
# -----------------------------------------------------------

NP_CFG = newspaper.Config()
NP_CFG.browser_user_agent = USER_AGENT
NP_CFG.request_timeout = REQUEST_TIMEOUT

# -----------------------------------------------------------
#  MODELL (lazy-load)
# -----------------------------------------------------------

_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        print(f"🧠 Laddar summarizer: {SUMMARIZER_MODEL}", flush=True)
        # Undvik Fast-tokenizer (då slipper vi protobuf-kravet/problemet)
        tok = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL, use_fast=False)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
        _summarizer = hf_pipeline("summarization", model=mdl, tokenizer=tok)
    return _summarizer


# -----------------------------------------------------------
#  HJÄLPFUNKTIONER
# -----------------------------------------------------------

def _epoch_from_entry(e) -> Optional[float]:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(e, attr, None)
        if t:
            try:
                return time.mktime(t)
            except Exception:
                return None
    return None

def _iso_utc_from_epoch(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

def _domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

def _shorten_title(title: str, max_len: int = 80) -> str:
    t = (title or "").strip(" \"'\u2019\u201c\u201d")
    parts = re.split(r"\s+[–\-:]\s+", t, maxsplit=1)
    t = parts[0]
    if len(t) > max_len:
        t = t[:max_len].rstrip() + "…"
    return t

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
def _tok(s: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(s or "")]

# -----------------------------------------------------------
#  INSAMLING
# -----------------------------------------------------------

def collect_articles(category: str, lang: str, days: int = 7) -> List[Dict]:
    urls = (SOURCES.get(category, {}) or {}).get(lang, []) or []
    if not urls:
        print(f"⚠️  Inga källor för {category}/{lang}.")
        return []
    since = time.time() - days * 86_400
    seen_urls = set()
    docs: List[Dict] = []

    for feed_url in urls:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as ex:
            print(f"⚠️  Misslyckades läsa feed: {feed_url} ({ex})")
            continue

        for entry in getattr(feed, "entries", []):
            ts = _epoch_from_entry(entry)
            if ts and ts < since:
                continue
            url = getattr(entry, "link", None)
            title = getattr(entry, "title", "") or ""
            if not url or url in seen_urls:
                continue

            art = newspaper.Article(url, config=NP_CFG)
            try:
                art.download()
                art.parse()
            except Exception:
                # hoppa över artikeln men fortsätt
                continue

            text = (art.text or "").strip()
            if len(text) < MIN_TEXT_CHARS:
                continue

            seen_urls.add(url)
            docs.append({
                "title": title,
                "text": text,
                "url": url,
                "published": ts,
                "domain": _domain(url),
                "language": lang,
                "category": category,
            })

            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)

            if len(docs) >= RAW_LIMIT:
                break
        if len(docs) >= RAW_LIMIT:
            break

    print(f"✅ Insamlat {len(docs)} artiklar för {category}/{lang}")
    return docs

# -----------------------------------------------------------
#  RANKNING
# -----------------------------------------------------------

def choose_top_docs(docs: List[Dict], top_n: int) -> List[Dict]:
    if not docs:
        return []

    corpus_tokens = [_tok(f"{d['title']} {d['text']}") for d in docs]
    bm25 = BM25Okapi(corpus_tokens)
    # Neutral query → täckningsmått; komplettera med recency + diversitet
    scores = bm25.get_scores([])

    now = time.time()
    per_domain = {}
    scored = []

    # Sortera först på BM25, använd sedan bonusar
    prelim = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    for s, d in prelim:
        dom = d.get("domain", "")
        if per_domain.get(dom, 0) >= MAX_PER_DOMAIN:
            continue

        rec = 0.0
        if d.get("published"):
            age_days = max(1.0, (now - d["published"]) / 86400.0)
            rec = 1.0 / math.sqrt(age_days)

        total = float(s) + 0.2 * rec
        scored.append((total, d))
        per_domain[dom] = per_domain.get(dom, 0) + 1

        if top_n > 0 and len(scored) >= top_n:
            break

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]

# -----------------------------------------------------------
#  SUMMERING & KORT
# -----------------------------------------------------------

def summarize(text: str) -> str:
    s = get_summarizer()
    out = s(text[:4000], max_length=90, min_length=25, do_sample=False)
    summary = (out[0]["summary_text"] or "").strip()
    # snygga till mellanslag före skiljetecken
    summary = re.sub(r"\s+([.,!?;:])", r"\1", summary)
    return summary

def make_card(doc: Dict) -> Dict:
    title = _shorten_title(doc["title"])
    summary = summarize(doc["text"])
    return {
        "title": title,
        "summary": summary,
        "url": doc["url"],
        "domain": doc.get("domain"),
        "published": _iso_utc_from_epoch(doc.get("published")),
    }

# -----------------------------------------------------------
#  Snabbtest:  python pipeline.py
# -----------------------------------------------------------
if __name__ == "__main__":
    # Exempel: world/sv senaste 1 dygn
    cat, lang, days = "world", "sv", 1
    raw = collect_articles(cat, lang, days)
    top = choose_top_docs(raw, top_n=6)
    print(f"Top {len(top)}")
    for d in top:
        c = make_card(d)
        print(c["title"], "—", c["domain"])
