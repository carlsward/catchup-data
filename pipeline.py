from __future__ import annotations

# -----------------------------------------------------------
#  CatchUp · insamling · ranking · språkvis summering
#  A: språkdetektion in/ut, robust decoding, städning
#  C: ämnes-diversitet (MMR-light) + span-baserad recency
#  D: snål minnesprofil – unload summarizers per språk
#  E: viktighetsrankning för week/month (story-kluster)
#  **Denna version är reducerad till EN + SV.**
# -----------------------------------------------------------

# ---- shim så newspaper funkar med lxml>=5 -----------------
try:
    import lxml_html_clean as _hc
    import sys as _sys
    _sys.modules["lxml.html.clean"] = _hc
except Exception:
    pass
# -----------------------------------------------------------

import gc
import math
import time
import re
from typing import List, Dict, Optional, Any
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

# Trafilatura för renare extraktion (fallback till newspaper3k om den saknas)
try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None

# Språkdetektion (deterministisk)
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0

# -----------------------------------------------------------
#  INSTÄLLNINGAR
# -----------------------------------------------------------

SOURCES_PATH = Path("sources.yaml")  # repo-rot

MIN_REQUIRED = 3
RAW_LIMIT = 60
MAX_PER_DOMAIN = 4
MIN_TEXT_CHARS = 400
REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) CatchUpBot/1.0"

SUMMARY_SENTENCES = 3
LANG_OK_PROB = 0.65

# C: recency/diversitet (används främst för 'day')
RECENCY_WEIGHT = {"day": 0.35, "week": 0.15, "month": 0.05, "_default": 0.20}
DIVERSITY_LAMBDA = 0.60
TITLE_DUP_JACCARD = 0.80

# E: viktighetsrankning (klustring) för week/month
SIM_TEXT_JACCARD = 0.28   # 0.25–0.35 vettigt intervall
SIM_TITLE_JACCARD = 0.55  # titlar kortare → högre tröskel
DOMAIN_REPUTATION = {
    "reuters.com": 1.0,
    "bbc.co.uk": 1,
    "bbc.com": 1,
    "svt.se": 1,
    "dn.se": 1,
    "svd.se": 1,
    "theguardian.com": 1,
    "euronews.com": 1,
    "expressen.se": 1,
}
REP_DEFAULT = 0.60

def load_sources() -> Dict[str, Dict[str, List[str]]]:
    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    return {}

SOURCES = load_sources()

NP_CFG = newspaper.Config()
NP_CFG.browser_user_agent = USER_AGENT
NP_CFG.request_timeout = REQUEST_TIMEOUT

# ------------------ Språk → modell (endast sv/en) ----------
SUM_MODELS = {
    "sv": "Gabriel/bart-base-cnn-swe",
}

DECODING = {
    "_default": dict(max_length=220, min_length=90, num_beams=6,
                     do_sample=False, no_repeat_ngram_size=3, length_penalty=1.0),
}

_summarizers: Dict[str, Any] = {}

def _load_sum_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    try:
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
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

def unload_summarizer(lang: str) -> None:
    """Frigör minne efter att ett språk batchats klart (D)."""
    try:
        if lang in _summarizers:
            obj = _summarizers.pop(lang)
            del obj
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"🧹 Unloaded summarizer for {lang}", flush=True)
    except Exception:
        pass

# ------------------ Hjälp: språkdetektion -----------------
def detect_lang_code(text: str) -> tuple[str | None, float]:
    s = (text or "").strip()
    if not s:
        return None, 0.0
    try:
        cand = detect_langs(s[:2000])  # klipp för hastighet
        best = max(cand, key=lambda x: x.prob)
        return best.lang, float(best.prob)
    except Exception:
        return None, 0.0

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
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) if ts else None

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

_BLOCK_PATTERNS = [
    r"(?mi)^\s*(bbc monitoring|reuters|ap|afp)\s*[-–:]\s*",
    r"(?mi)^\s*(läs mer|läs också|read more|weiterlesen|lire aussi|leer más)\s*:.*?$",
    r"(?mi)^\s*(share this|related articles?|advertisement|annons).*?$",
    r"(?mi)\(\s*report(ing|ed) by .*?\)",
]

def _scrub_noise(s: str) -> str:
    for pat in _BLOCK_PATTERNS:
        s = re.sub(pat, "", s)
    return s

def clean_text(s: str) -> str:
    s = (s or "").replace("\u00AD", "")
    s = _scrub_noise(s)
    s = re.sub(r"\s+", " ", s)
    s = _collapse_spaced_words(s)
    return s.strip()

# inkluderar grekiskt ';' (frågetecken) – ofarligt att ha kvar
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

def collect_articles(category: str, lang: str, days: int = 7, raw_limit: int | None = None) -> List[Dict]:
    """Hämtar artiklar ≤ `days` bakåt. `raw_limit` kan öka/minska max antal råa."""
    urls = (SOURCES.get(category, {}) or {}).get(lang, []) or []
    if not urls:
        print(f"⚠️  Inga källor för {category}/{lang}.")
        return []

    limit = raw_limit or RAW_LIMIT
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
                art.download(); art.parse()
            except Exception:
                continue

            text = _extract_text_with_trafilatura(url, getattr(art, "html", None)) or (art.text or "")
            text = clean_text(text)
            if len(text) < MIN_TEXT_CHARS:
                continue

            det_lang, prob = detect_lang_code(text)
            if det_lang != lang or prob < LANG_OK_PROB:
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
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break

    print(f"✅ Insamlat {len(out)} artiklar för {category}/{lang}")
    return out


# ------------------ Diversitets-hjälp ----------------------
_STOPWORDS = {
    "en": {"the","a","an","of","and","to","in","on","for","with","by","at","from","as","that","this","is","are","be","was","were"},
    "sv": {"en","ett","och","att","som","för","med","till","på","av","är","var","från","om","i"},
}

def _norm_tokens(text: str, lang: str) -> set:
    toks = [t for t in _tok(text) if len(t) >= 3 and not t.isdigit()]
    sw = _STOPWORDS.get(lang, set())
    return set(t for t in toks if t not in sw)

def _title_tokens(title: str, lang: str) -> set:
    t = re.sub(r"[^\w\s]", " ", title.lower())
    return _norm_tokens(t, lang)

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 0.0 if inter == 0 else inter / float(len(a | b))

# ------------------ Rankning -------------------------------
def domain_cap(docs: List[Dict], top_n: int) -> int:
    """
    Rimligt per-domän-tak givet kandidatlistan.
    - Om bara 1 domän finns: tillåt upp till ~70% av platserna (avrundat uppåt),
      men minst MAX_PER_DOMAIN.
    - Om 2 domäner finns: ~60% (så båda syns om det går).
    - Annars: fördela ungefärligt + 1 i slack, aldrig under MAX_PER_DOMAIN.
    """
    doms = {d.get("domain", "") for d in docs if d.get("domain")}
    k = len(doms)
    if k <= 1:
        return max(MAX_PER_DOMAIN, int(math.ceil(top_n * 0.7)))
    if k == 2:
        return max(MAX_PER_DOMAIN, int(math.ceil(top_n * 0.6)))
    return max(MAX_PER_DOMAIN, int(math.ceil(top_n / k)) + 1)


def choose_top_docs(
    docs: List[Dict],
    top_n: int,
    span: str = "day",
    exclude_urls: Optional[set[str]] = None,
    max_per_domain: Optional[int] = None,
) -> List[Dict]:
    if not docs or top_n == 0:
        return []

    # BM25-bas
    corpus_tokens = [_tok(f"{d['title']} {d['text']}") for d in docs]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores([])

    # recency-vikt (svag för week/month i din konfig)
    now = time.time()
    rw = RECENCY_WEIGHT.get(span, RECENCY_WEIGHT["_default"])

    # dynamiskt domän-tak
    cap = max_per_domain if max_per_domain is not None else domain_cap(docs, top_n)

    items = []
    for s, d in zip(bm25_scores, docs):
        if exclude_urls and d.get("url") in exclude_urls:
            continue
        rec = 0.0
        if d.get("published"):
            age_days = max(1.0, (now - d["published"]) / 86400.0)
            rec = 1.0 / math.sqrt(age_days)
        base = float(s) + rw * rec

        lang = d.get("language", "sv")
        items.append({
            "doc": d,
            "base": base,
            "domain": d.get("domain", ""),
            "title_tokens": _title_tokens(d.get("title",""), lang),
            "text_tokens": _norm_tokens(d.get("text","")[:1000], lang),
        })

    per_domain: Dict[str, int] = {}
    selected: List[Dict] = []
    candidates = sorted(items, key=lambda z: z["base"], reverse=True)

    while candidates and len(selected) < top_n:
        best_idx = -1
        best_score = -1e9
        for i, it in enumerate(candidates):
            dom = it["domain"]
            if per_domain.get(dom, 0) >= cap:
                continue

            # diversitet mot redan valda (MMR-light)
            max_sim = 0.0
            title_dup = False
            for s in selected:
                if _jaccard(it["title_tokens"], s["title_tokens"]) >= TITLE_DUP_JACCARD:
                    title_dup = True
                    break
                max_sim = max(max_sim, _jaccard(it["text_tokens"], s["text_tokens"]))
            if title_dup:
                continue

            mmr = it["base"] - DIVERSITY_LAMBDA * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i

        if best_idx < 0:
            break

        chosen = candidates.pop(best_idx)
        d = chosen["doc"]
        per_domain[chosen["domain"]] = per_domain.get(chosen["domain"], 0) + 1
        selected.append({
            "title_tokens": chosen["title_tokens"],
            "text_tokens": chosen["text_tokens"],
            "doc": d,
            "_score": chosen["base"],
        })

    return [s["doc"] for s in selected]



# ------------------ Summering & kort -----------------------
def _decoding_for(lang: str) -> Dict:
    cfg = DECODING.get(lang, {})
    base = DECODING["_default"].copy()
    base.update(cfg)
    return base

def summarize_for_lang(text: str, lang: str) -> str:
    txt = clean_text(text)[:4000]
    summ = get_summarizer_for(lang)
    if summ is None:
        return first_n_sentences(txt, SUMMARY_SENTENCES)

    kwargs = _decoding_for(lang)
    out = summ(txt, **kwargs)
    summ_text = postprocess_summary(out[0]["summary_text"])
    s_lang, s_prob = detect_lang_code(summ_text)

    if s_lang != lang or s_prob < 0.50:
        strict = kwargs.copy()
        strict["num_beams"] = max(6, kwargs.get("num_beams", 6) + 2)
        strict["no_repeat_ngram_size"] = max(4, kwargs.get("no_repeat_ngram_size", 3) + 1)
        strict["max_length"] = int(kwargs.get("max_length", 220) * 0.9)
        out2 = summ(txt, **strict)
        summ_text2 = postprocess_summary(out2[0]["summary_text"])
        s2, p2 = detect_lang_code(summ_text2)
        if s2 == lang and p2 >= 0.50:
            summ_text = summ_text2
        else:
            summ_text = first_n_sentences(txt, SUMMARY_SENTENCES)

    return first_n_sentences(summ_text, SUMMARY_SENTENCES)

def summarize(text: str, lang: str) -> str:
    try:
        return summarize_for_lang(text, lang)
    except Exception:
        return first_n_sentences(clean_text(text), SUMMARY_SENTENCES)

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
    # Exempel: world/sv (24h, top 3)
    cat, lang, days = "world", "sv", 1
    raw = collect_articles(cat, lang, days)
    top = choose_top_docs(raw, top_n=3, span="day")
    for d in top:
        c = make_card(d)
        print("—", c["title"], f"({c['domain']})")
        print(c["summary"])
        print()
