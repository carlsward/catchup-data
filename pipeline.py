from __future__ import annotations

# -----------------------------------------------------------
#  CatchUp · insamling · ranking · språkvis summering
# -----------------------------------------------------------
# - Hämtar artiklar från RSS per (kategori, språk) ur sources.yaml
# - Rankar med BM25 + recency + källdiversitet
# - Summerar på KÄLLSPRÅKET (ingen MT) med språk-specifika modeller:
#     sv: Gabriel/bart-base-cnn-swe
#     en: facebook/bart-large-cnn
#     de: Shahm/bart-german
#     fr: moussaKam/barthez-orangesum-abstract
#     es: Narrativa/bsc_roberta2roberta_shared-mlsum-summarization
#     el: IMISLab/GreekT5-umt5-base-greeksum
# - Om ett språk saknar modell: fallback = ta de första meningarna ur texten
# -----------------------------------------------------------

# ---- shim så newspaper funkar med lxml>=5 -----------------
try:
    import lxml_html_clean as _hc
    import sys as _sys
    _sys.modules["lxml.html.clean"] = _hc
except Exception:
    pass
# -----------------------------------------------------------

import math
import time
import re
from typing import List, Dict, Optional
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import newspaper
import yaml
from rank_bm25 import BM25Okapi
from transformers import (
    pipeline as hf_pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Trafilatura (renare extraktion). Faller tillbaka till newspaper3k om saknas.
try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None

# -----------------------------------------------------------
#  INSTÄLLNINGAR
# -----------------------------------------------------------

SOURCES_PATH = Path("sources.yaml")  # läses i repo-roten

# Beta/tuning
MIN_REQUIRED = 3         # min kort per vy innan 'fallback_used' markeras
RAW_LIMIT = 60           # hur många råa artiklar som max hämtas per (cat,lang)
MAX_PER_DOMAIN = 4       # diversitet: max kort per domän
MIN_TEXT_CHARS = 400     # släng tunn/tom text
REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0        # 0 under beta
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) CatchUpBot/1.0"

# Sammanfattning
SUMMARY_SENTENCES = 3    # klipp till ~3 meningar
# -----------------------------------------------------------

def load_sources() -> Dict[str, Dict[str, List[str]]]:
    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print("⚠️  sources.yaml ogiltig (måste vara mapping).")
            return {}
        return data
    print("ℹ️  Ingen sources.yaml hittad – tomma källor.")
    return {}

SOURCES = load_sources()

# Newspaper-konfig
NP_CFG = newspaper.Config()
NP_CFG.browser_user_agent = USER_AGENT
NP_CFG.request_timeout = REQUEST_TIMEOUT

# ------------------ Språk → modell -------------------------
#  Kvalitet före hastighet, kör lokalt utan API-nycklar.
SUM_MODELS = {
    "sv": "Gabriel/bart-base-cnn-swe",                                # svensk nyhets-BART
    "en": "facebook/bart-large-cnn",                                   # eng. nyhets-BART (large)
    "de": "Shahm/bart-german",                                         # tysk BART (MLSUM)
    "fr": "moussaKam/barthez-orangesum-abstract",                      # fransk BARThez (OrangeSum)
    "es": "Narrativa/bsc_roberta2roberta_shared-mlsum-summarization",  # spansk RoBERTa2RoBERTa (MLSUM)
    "el": "IMISLab/GreekT5-umt5-base-greeksum",                        # grekisk mT5 (GreekSum)
}

_summarizers: Dict[str, any] = {}

def _load_sum_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    try:
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        # vissa modeller är sparade som EncoderDecoderModel (t.ex. *2* shared-modeller)
        from transformers import EncoderDecoderModel
        mdl = EncoderDecoderModel.from_pretrained(model_name)
    return hf_pipeline("summarization", model=mdl, tokenizer=tok)

def get_summarizer_for(lang: str):
    if lang in _summarizers:
        return _summarizers[lang]
    model = SUM_MODELS.get(lang)
    if not model:
        return None
    print(f"🧠 Laddar summarizer för {lang}: {model}", flush=True)
    _summarizers[lang] = _load_sum_model(model)
    return _summarizers[lang]

# ------------------ Hjälp & textstäd ----------------------
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
    return (t[:max_len].rstrip() + "…") if len(t) > max_len else t

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
def _tok(s: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(s or "")]

_LATIN = "A-Za-zÅÄÖåäöÉÈÊËÂÀÁÍÌÎÏÓÒÔÖÚÙÛÜÑÇß"
def _collapse_spaced_words(s: str) -> str:
    # slå ihop "a n i g ü r e l o s t" → "anigürelost"
    return re.sub(
        rf"(?:(?<=\b)[{_LATIN}]{{1}}\s+){{3,}}[{_LATIN}]{{1}}(?=\b)",
        lambda m: m.group(0).replace(" ", ""),
        s,
    )

def clean_text(s: str) -> str:
    s = (s or "").replace("\u00AD", "")        # mjukt bindestreck
    s = re.sub(r"\s+", " ", s)
    s = _collapse_spaced_words(s)
    return s.strip()

# inkluderar grekiskt ';' (frågetecken)
_SENT_SPLIT = re.compile(r'(?<=[\.\!\?…]|[;])\s+')

def first_n_sentences(text: str, n: int = SUMMARY_SENTENCES) -> str:
    parts = [p.strip(' "\'–-') for p in _SENT_SPLIT.split(text)]
    parts = [p for p in parts if p]
    return " ".join(parts[:n])

def postprocess_summary(s: str) -> str:
    s = _collapse_spaced_words(s)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return s.strip()

# ------------------ Insamling ------------------------------
def _extract_text_with_trafilatura(url: str, html: Optional[str]) -> str:
    if trafilatura is None:
        return ""
    try:
        if html:
            return trafilatura.extract(html, include_comments=False) or ""
    except Exception:
        return ""
    return ""

def collect_articles(category: str, lang: str, days: int = 7) -> List[Dict]:
    urls = (SOURCES.get(category, {}) or {}).get(lang, []) or []
    if not urls:
        print(f"⚠️  Inga källor för {category}/{lang}.")
        return []
    since = time.time() - days * 86_400
    seen = set()
    out: List[Dict] = []

    for feed_url in urls:
        try:
            feed = feedparser.parse(feed_url)
        except Exception:
            continue

        for e in getattr(feed, "entries", []):
            ts = _epoch_from_entry(e)
            if ts and ts < since:
                continue
            url = getattr(e, "link", None)
            title = getattr(e, "title", "") or ""
            if not url or url in seen:
                continue

            art = newspaper.Article(url, config=NP_CFG)
            try:
                art.download()
                art.parse()
            except Exception:
                continue

            text = _extract_text_with_trafilatura(url, getattr(art, "html", None)) or (art.text or "")
            text = clean_text(text)
            if len(text) < MIN_TEXT_CHARS:
                continue

            seen.add(url)
            out.append({
                "title": title,
                "text": text,
                "url": url,
                "published": ts,
                "domain": _domain(url),
                "language": lang,
                "category": category,
            })

            if REQUEST_DELAY:
                time.sleep(REQUEST_DELAY)
            if len(out) >= RAW_LIMIT:
                break
        if len(out) >= RAW_LIMIT:
            break

    print(f"✅ Insamlat {len(out)} artiklar för {category}/{lang}")
    return out

# ------------------ Rankning -------------------------------
def choose_top_docs(docs: List[Dict], top_n: int) -> List[Dict]:
    if not docs:
        return []

    corpus_tokens = [_tok(f"{d['title']} {d['text']}") for d in docs]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores([])  # neutral query

    now = time.time()
    per_domain: Dict[str, int] = {}
    prelim = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    picked: List[Dict] = []

    for s, d in prelim:
        dom = d.get("domain", "")
        if per_domain.get(dom, 0) >= MAX_PER_DOMAIN:
            continue

        rec = 0.0
        if d.get("published"):
            age_days = max(1.0, (now - d["published"]) / 86400.0)
            rec = 1.0 / math.sqrt(age_days)

        d["_score"] = float(s) + 0.2 * rec
        per_domain[dom] = per_domain.get(dom, 0) + 1
        picked.append(d)
        if top_n > 0 and len(picked) >= top_n:
            break

    picked.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return picked

# ------------------ Summering & kort -----------------------
def summarize_for_lang(text: str, lang: str) -> str:
    txt = clean_text(text)[:4000]
    summ = get_summarizer_for(lang)
    if summ is not None:
        out = summ(
            txt,
            max_length=220,   # lite längre → bättre innehåll
            min_length=90,
            do_sample=False,
            num_beams=4
        )
        return first_n_sentences(postprocess_summary(out[0]["summary_text"]), SUMMARY_SENTENCES)

    # Om språket saknar modell: enkel fallback (ingen översättning)
    return first_n_sentences(txt, SUMMARY_SENTENCES)

def summarize(text: str, lang: str) -> str:
    return summarize_for_lang(text, lang)

def make_card(doc: Dict) -> Dict:
    title = _shorten_title(doc["title"])
    summary = summarize(doc["text"], doc.get("language", "en"))
    return {
        "title": title,
        "summary": summary,
        "url": doc["url"],
        "domain": doc.get("domain"),
        "published": _iso_utc_from_epoch(doc.get("published")),
    }

# ------------------ Snabbtest ------------------------------
if __name__ == "__main__":
    # Exempel: world/sv (24h, top 3) – för snabb röktest lokalt
    cat, lang, days = "world", "sv", 1
    raw = collect_articles(cat, lang, days)
    top = choose_top_docs(raw, top_n=3)
    for d in top:
        c = make_card(d)
        print("—", c["title"], f"({c['domain']})")
        print(c["summary"])
        print()
