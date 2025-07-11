# ──────────────────────────────────────────────────────────────
#  pipeline.py  – samlar, rankar, summerar och ÖVERSÄTTER nyheter
# ──────────────────────────────────────────────────────────────

# --- shim så newspaper3k kan importera lxml.html.clean även med lxml>=5 ---
try:
    import lxml_html_clean as _hc
    import sys
    sys.modules["lxml.html.clean"] = _hc
except ImportError:
    pass
# ---------------------------------------------------------------------------

import time, re, sys
import feedparser, newspaper
from rank_bm25 import BM25Okapi
from transformers import pipeline as hf_pipeline

# ──────────────────────────────────────────────────────────────
# 1. Konstanter
# ----------------------------------------------------------------
FEEDS = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.svt.se/nyheter/rss",
]

# hur många dygn tillbaka samt hur många kort vi ska spara
SPAN_RULES = {
    "day":   dict(days=1,  keep=5),
    "week":  dict(days=7,  keep=10),
    "month": dict(days=30, keep=20),
}

# ──────────────────────────────────────────────────────────────
# 2. Initialisera modeller EN gång (sparar enorm tid)
# ----------------------------------------------------------------
summarizer = hf_pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-6-6",
    device_map="auto" if not sys.platform.startswith("win") else None,  # GPU lokalt om finns
)

translator = hf_pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-sv",
    device_map="auto" if not sys.platform.startswith("win") else None,
)

# ──────────────────────────────────────────────────────────────
# 3. Hjälpfunktioner
# ----------------------------------------------------------------
def _gather_articles(days_back: int):
    """Hämtar råa artiklar <days_back> dygn bakåt."""
    cutoff = time.time() - days_back * 86_400
    docs = []
    for feed_url in FEEDS:
        for entry in feedparser.parse(feed_url).entries:
            if time.mktime(entry.published_parsed) < cutoff:
                continue
            art = newspaper.Article(entry.link)
            try:
                art.download(); art.parse()
            except Exception:
                continue  # hoppa över trasiga artiklar
            docs.append(
                {
                    "title_en": entry.title,
                    "text_en": art.text,
                    "url": entry.link,
                }
            )
    return docs


def _rank_by_bm25(docs, keep: int):
    """Väljer de <keep> mest relevanta artiklarna med BM25."""
    corpus = [(d["title_en"] + " " + d["text_en"]).split() for d in docs]
    bm25 = BM25Okapi(corpus)
    # väldigt grov "globalt nyhets"-query
    scores = bm25.get_scores(["news", "world", "sweden"])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:keep]]


def _summarise_and_translate(doc):
    """Sammanfattar ENG → översätter till SV. Returnerar färdig kort‐dict."""
    raw_text = doc["text_en"][:2000]  # BART-modellen klarar ~1 k tokens
    summary_en = summarizer(
        raw_text,
        max_length=100,
        min_length=30,
        do_sample=False
    )[0]["summary_text"].strip()

    # punkt- och mellanslagsfix
    summary_en = re.sub(r"\s+([.,!?;:])", r"\1", summary_en)

    # Översätt både titel & summary
    sv_title   = translator(doc["title_en"],   max_length=128)[0]["translation_text"]
    sv_summary = translator(summary_en,        max_length=512)[0]["translation_text"]

    return {
        "title"   : sv_title,
        "summary" : sv_summary,
        "url"     : doc["url"],
        # valfritt: behåll engelska fälten som fallback / debug
        # "title_en": doc["title_en"],
        # "summary_en": summary_en,
    }

# ──────────────────────────────────────────────────────────────
# 4. Publika funktioner som make_json.py anropar
# ----------------------------------------------------------------
def collect_articles(days: int = 7):
    """Bakåtkompatibel wrapper – behålls för ev. extern användning."""
    return _gather_articles(days)


def choose_top20(docs):
    """Bakåtkompatibel wrapper – används bara för 30-dagars‐spannet."""
    return _rank_by_bm25(docs, 20)


def summarize(docs):
    """Sammanfattar & översätter utan att skära ned – används av gamla kod."""
    return [_summarise_and_translate(d) for d in docs]


# ──────────────────────────────────────────────────────────────
# 5. Ny bekväm funktion → ger exakt rätt antal kort per span
# ----------------------------------------------------------------
def build_cards(span_key: str):
    """
    Exempel:
        cards = build_cards("week")  # 7 dagar, max 10 kort
    """
    if span_key not in SPAN_RULES:
        raise ValueError(f"Okänd span: {span_key}")

    cfg   = SPAN_RULES[span_key]
    docs  = _gather_articles(cfg["days"])
    docs  = _rank_by_bm25(docs, cfg["keep"])
    cards = [_summarise_and_translate(d) for d in docs]
    return cards
