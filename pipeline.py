# pipeline.py  – komplett fil
# --- shim så newspaper fungerar med lxml>=5 ---
try:
    import lxml_html_clean as _hc
    import sys
    sys.modules['lxml.html.clean'] = _hc
except ImportError:
    pass
# ------------------------------------------------

import time, re, feedparser, newspaper
from transformers import pipeline as hf_pipeline
from rank_bm25 import BM25Okapi

FEEDS = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.svt.se/nyheter/rss",
]

LANGS = {
    "en": None,            # originaltext (ingen översättning)
    "sv": "Helsinki-NLP/opus-mt-en-sv",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
}

_SUMMARIZER = hf_pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device="cpu",
)

def collect_articles(days: int = 7):
    """
    Hämtar artiklar publicerade de ­senaste `days` dagarna.
    Returnerar lista med dictar: {title, text, url}
    """
    since = time.time() - days * 86_400
    docs = []
    for feed in FEEDS:
        for e in feedparser.parse(feed).entries:
            if time.mktime(e.published_parsed) < since:
                continue
            art = newspaper.Article(e.link)
            art.download(); art.parse()
            docs.append({"title": e.title, "text": art.text, "url": e.link})
    return docs

def choose_top(docs, k: int, query=("news", "world", "sweden")):
    """
    Väljer de `k` mest relevanta artiklarna mha BM25.
    """
    corpus = [d["title"] + " " + d["text"] for d in docs]
    bm25   = BM25Okapi([c.split() for c in corpus])
    scores = bm25.get_scores(list(query))
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:k]]

def summarize(text: str, max_len=100, min_len=30):
    """
    Returnerar en engelsk sammanfattning av `text`.
    """
    short = text[:2000]                              # API-begränsning
    s = _SUMMARIZER(short,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False)[0]["summary_text"]
    s = re.sub(r"\s+([.,!?;:])", r"\1", s.strip())   # snygga till blanksteg
    return s

def translate(text: str, tgt_lang: str):
    """
    Översätter `text` från engelska → `tgt_lang` via Marian-MT.
    Om `tgt_lang == "en"` returneras texten oförändrad.
    """
    model_name = LANGS[tgt_lang]
    if model_name is None:
        return text
    translator = hf_pipeline("translation", model=model_name, device="cpu")
    return translator(text, max_length=400)[0]["translation_text"]

def build_cards(docs, tgt_lang: str):
    """
    Summariserar och (ev) översätter artiklar till kort.
    """
    cards = []
    for d in docs:
        summ_en = summarize(d["text"])
        title   = d["title"] if tgt_lang == "en" else translate(d["title"], tgt_lang)
        body    = summ_en     if tgt_lang == "en" else translate(summ_en,  tgt_lang)
        cards.append({"title": title, "summary": body, "url": d["url"]})
    return cards
