from __future__ import annotations

# -----------------------------------------------------------
#  Catch-up  ·  nyhetsinsamling · summering · översättning
# -----------------------------------------------------------
#
# 1.  Hämtar artiklar från RSS-flöden (feedparser + newspaper3k)
# 2.  Rankar med BM25 och väljer topp-N
# 3.  Summerar på engelska    (distilbart-cnn-6-6)
# 4.  Översätter om nödvändigt (NLLB-200-distilled-600M)
#
# -----------------------------------------------------------



# ---- shim så newspaper funkar med lxml>=5 -----------------
try:
    import lxml_html_clean as _hc
    import sys
    sys.modules["lxml.html.clean"] = _hc
except ImportError:
    pass
# -----------------------------------------------------------

import time, re, feedparser, newspaper
from typing import List, Dict
from rank_bm25 import BM25Okapi
from transformers import pipeline as hf_pipeline

# -----------------------------------------------------------
#  INSTÄLLNINGAR
# -----------------------------------------------------------

FEEDS: List[str] = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.svt.se/nyheter/rss",
]

TARGET_LANGS: List[str] = ["en", "sv", "de", "es", "fr"]   # skapa dessa JSON-filer

# ISO-639-3 + skript enligt NLLB (behövs av translation-pipen)
_LANG_CODE: Dict[str, str] = {
    "en": "eng_Latn",
    "sv": "swe_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
}

# -----------------------------------------------------------
#  MODELLER – laddas en enda gång vid import
# -----------------------------------------------------------

print("Device set to use cpu")          # → syns i GitHub Actions-loggen

summarizer = hf_pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    device="cpu",
)

translator = hf_pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    device="cpu",
)

# -----------------------------------------------------------
#  HJÄLPFUNKTIONER
# -----------------------------------------------------------

def translate(text: str, tgt_lang: str) -> str:
    """
    Översätter ENG → `tgt_lang` med NLLB-200.  
    Returnerar originalet om `tgt_lang == "en"`.
    """
    if tgt_lang == "en":
        return text
    return translator(
        text,
        src_lang="eng_Latn",
        tgt_lang=_LANG_CODE[tgt_lang],
        max_length=512,
    )[0]["translation_text"]


def collect_articles(days: int = 7) -> List[Dict]:
    """Hämtar artiklar ≤ `days` bakåt och returnerar en lista med dictar."""
    since = time.time() - days * 86_400
    docs: List[Dict] = []
    for feed_url in FEEDS:
        for entry in feedparser.parse(feed_url).entries:
            if time.mktime(entry.published_parsed) < since:
                continue
            art = newspaper.Article(entry.link)
            try:
                art.download(); art.parse()
            except Exception:
                continue
            docs.append(
                {"title": entry.title, "text": art.text, "url": entry.link}
            )
    return docs


def choose_top_docs(docs: List[Dict], top_n: int = 20) -> List[Dict]:
    """Rankar med BM25 och returnerar de `top_n` mest relevanta artiklarna."""
    corpus = [f"{d['title']} {d['text']}" for d in docs]
    bm25   = BM25Okapi([c.split() for c in corpus])
    scores = bm25.get_scores(["news", "world", "sweden"])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]


def make_card(doc: Dict, tgt_lang: str) -> Dict:
    """Bygger ett 'kort' (rubrik + summering) på valt språk."""
    # 1) Engelsk summering
    short_txt = doc["text"][:2_000]
    en_sum = summarizer(
        short_txt,
        max_length=100,
        min_length=30,
        do_sample=False,
    )[0]["summary_text"].strip()
    en_sum = re.sub(r"\s+([.,!?;:])", r"\1", en_sum)

    # 2) Översätt titel + summary vid behov
    title   = translate(doc["title"], tgt_lang)
    summary = translate(en_sum,        tgt_lang)

    return {"title": title, "summary": summary, "url": doc["url"]}

# -----------------------------------------------------------
#  Snabbtest:  python pipeline.py   (lokalt)
# -----------------------------------------------------------
if __name__ == "__main__":
    art = collect_articles(1)[0]
    for lang in TARGET_LANGS:
        print(f"\n--- {lang} ---")
        print(make_card(art, lang))
