"""
Build-skriptet importerar denna modul från make_json.py.

• Hämtar artiklar ur FEEDS
• Väljer top N (N = 5 / 10 / 20 beroende på spann)
• Summerar på engelska (distilbart-cnn)
• Översätter summeringen till svenska
• Returnerar kort med {title, summary, url} för båda språken
"""

# ------------------------- shim för newspaper + lxml ≥ 5 --------------------
try:
    import lxml_html_clean as _hc
    import sys
    sys.modules['lxml.html.clean'] = _hc
except ImportError:
    pass
# ---------------------------------------------------------------------------

import time, json, re, sys
from pathlib import Path
from typing import List, Dict

import feedparser, newspaper
from rank_bm25 import BM25Okapi
from transformers import pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# 1. Konfiguration
# ---------------------------------------------------------------------------

FEEDS = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.svt.se/nyheter/rss",
]

TOP_K = {"day": 5, "week": 10, "month": 20}
LANGS = ("en", "sv")                          # lägg till fler om du vill

# Hugging Face-pipelines
summarizer_en = hf_pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    device="cpu",
    max_length=120,
    min_length=30,
)

translator_en_sv = hf_pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-sv",
    device="cpu",
    max_length=130,
)
# Exempel för fler språk:
# translator_en_de = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")


# ---------------------------------------------------------------------------
# 2. Hjälpfunktioner
# ---------------------------------------------------------------------------

def _fetch_raw_articles(days: int) -> List[Dict]:
    """Läser RSS-flödena och laddar fulltexten med newspaper."""
    since = time.time() - days * 86_400
    out = []
    for feed_url in FEEDS:
        for entry in feedparser.parse(feed_url).entries:
            # om feeden saknar published_parsed → hoppa
            if not hasattr(entry, "published_parsed"):
                continue
            if time.mktime(entry.published_parsed) < since:
                continue

            art = newspaper.Article(entry.link)
            try:
                art.download(); art.parse()
            except Exception:
                continue            # hoppa om newspaper snubblar

            out.append(
                {
                    "title": entry.title,
                    "text": art.text,
                    "url": entry.link,
                }
            )
    return out


def _bm25_select(docs: List[Dict], k: int) -> List[Dict]:
    """Rankar dokument mot några generella nyhetsord och tar topp k."""
    corpus = [d["title"] + " " + d["text"] for d in docs]
    bm25 = BM25Okapi([c.split() for c in corpus])
    scores = bm25.get_scores(["news", "world", "sweden"])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:k]]


def _summarise_and_translate(doc: Dict) -> Dict[str, str]:
    """Returnerar {lang: summary}."""
    snippet = doc["text"][:2000] or doc["title"]
    sum_en = summarizer_en(
        snippet,
        do_sample=False,
    )[0]["summary_text"]

    sum_sv = translator_en_sv(sum_en)[0]["translation_text"]

    # små efter-poleringar
    def _tidy(txt: str) -> str:
        txt = txt.strip()
        txt = re.sub(r"\s+([.,!?;:])", r"\1", txt)
        return txt

    return {"en": _tidy(sum_en), "sv": _tidy(sum_sv)}


# ---------------------------------------------------------------------------
# 3. Publik funktion som make_json.py anropar
# ---------------------------------------------------------------------------

def build_cards(span: str) -> List[Dict]:
    """
    span = 'day' | 'week' | 'month'
    Returnerar list[card], där varje card har språk-specifika titlar/sammanfattningar
    """
    if span not in TOP_K:
        raise ValueError(f"Ogiltigt span '{span}'")

    days = {"day": 1, "week": 7, "month": 30}[span]
    raw = _fetch_raw_articles(days)
    picked = _bm25_select(raw, TOP_K[span])

    cards = []
    for d in picked:
        translations = _summarise_and_translate(d)
        titles = {
            "en": d["title"],
            "sv": translator_en_sv(d["title"])[0]["translation_text"],
        }

        cards.append(
            {
                "title": titles,
                "summary": translations,
                "url": d["url"],
            }
        )
    return cards


# ---------------------------------------------------------------------------
# 4. CLI – kör manuellt om du vill testa lokalt
#    `python pipeline.py day` skriver JSON till stdout
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    span_cli = sys.argv[1] if len(sys.argv) > 1 else "day"
    cards_cli = build_cards(span_cli)
    print(json.dumps({"cards": cards_cli}, ensure_ascii=False, indent=2))
