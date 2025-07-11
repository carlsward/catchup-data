# -----------------------------------------------------------
#  Catch-up  ·  Nyhetsinsamling  ·  summering  ·  översättning
# -----------------------------------------------------------
#
# 1. Hämtar artiklar från RSS-flöden (feedparser + newspaper3k)
# 2. Väljer de mest relevanta med BM25
# 3. Summerar (eng)   ── distilbart-cnn-6-6
# 4. Översätter vid behov ── NLLB-200-distilled-600M
#
# -----------------------------------------------------------

# --- shim så newspaper funkar med lxml>=5 ------------------
try:
    import lxml_html_clean as _hc
    import sys
    sys.modules['lxml.html.clean'] = _hc
except ImportError:
    pass
# -----------------------------------------------------------

import feedparser, newspaper, time, re
from rank_bm25 import BM25Okapi

from transformers import (
    pipeline   as hf_pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# ----------- inställningar --------------------------------
FEEDS = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.svt.se/nyheter/rss",
]

TARGET_LANGS = ["en", "sv", "de", "es", "fr"]

# ---- modeller (laddas en gång vid import) -----------------
print("Device set to use cpu")        # loggraden i Actions-runnen
summarizer = hf_pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    device="cpu",
)

translator_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)

_LANG_CODE = {
    "en": "eng_Latn",
    "sv": "swe_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
}

# -----------------------------------------------------------


# ---------- hjälp­funktioner -------------------------------

def translate(text: str, tgt_lang: str) -> str:
    """Översätter ENG → tgt_lang med NLLB. (No-op om tgt=en)."""
    if tgt_lang == "en":
        return text
    tgt = _LANG_CODE[tgt_lang]
    inp = translator_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    out_tokens = translator_model.generate(
        **inp,
        forced_bos_token_id=translator_tokenizer.lang_code_to_id[tgt],
        max_length=512,
        num_beams=4,
    )
    return translator_tokenizer.decode(out_tokens[0], skip_special_tokens=True)


def collect_articles(days: int = 7):
    """Hämtar artiklar ≤ `days` tillbaka, returnerar lista med dictar."""
    since = time.time() - days * 86_400
    docs = []
    for feed_url in FEEDS:
        for entry in feedparser.parse(feed_url).entries:
            if time.mktime(entry.published_parsed) < since:
                continue
            art = newspaper.Article(entry.link)
            try:
                art.download(); art.parse()
            except Exception:
                continue   # hoppade över trasig artikel
            docs.append(
                {
                    "title": entry.title,
                    "text":  art.text,
                    "url":   entry.link,
                }
            )
    return docs


def choose_top_docs(docs, top_n=20):
    """Rankar med BM25 (enkelt sök-query) och returnerar top_n docs."""
    corpus = [f"{d['title']} {d['text']}" for d in docs]
    bm25  = BM25Okapi([c.split() for c in corpus])
    scores = bm25.get_scores(["news", "world", "sweden"])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]


def make_card(doc: dict, tgt_lang: str) -> dict:
    """Bygger ett 'kort' (rubrik + summering) på valt språk."""
    # 1) Summera på engelska
    short_txt = doc["text"][:2000]
    en_sum = summarizer(
        short_txt, max_length=100, min_length=30, do_sample=False
    )[0]["summary_text"].strip()
    en_sum = re.sub(r"\s+([.,!?;:])", r"\1", en_sum)

    # 2) Översätt vid behov
    title   = translate(doc["title"], tgt_lang)
    summary = translate(en_sum,    tgt_lang)

    return {"title": title, "summary": summary, "url": doc["url"]}


# -----------------------------------------------------------
if __name__ == "__main__":
    # snabbtest: skriv ut en svensk sammanfattning av första Reuters-artikel
    art = collect_articles(1)[0]
    print(make_card(art, "sv"))
