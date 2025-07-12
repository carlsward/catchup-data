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

# --- shim så newspaper funkar med lxml>=5 ------------------
try:
    import lxml_html_clean as _hc
    import sys
    sys.modules["lxml.html.clean"] = _hc
except ImportError:
    pass
# -----------------------------------------------------------

import time, re, feedparser, newspaper
from rank_bm25 import BM25Okapi
from transformers import (
    pipeline as hf_pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# -----------------------------------------------------------
#  INSTÄLLNINGAR
# -----------------------------------------------------------

FEEDS = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.svt.se/nyheter/rss",
]

TARGET_LANGS = ["en", "sv", "de", "es", "fr"]       # generera dessa

_LANG_CODE = {      # ISO-639-3 + skript enligt NLLB
    "en": "eng_Latn",
    "sv": "swe_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
}

# Tokensträngar som olika NLLB-/mBART-modeller kan använda
_TOKEN_PATTERNS = [
    "<2{code}>",     # nytt NLLB-200-format
    "__{code}__",    # äldre fairseq-format
    "<<{code}>>",    # äldre mBART-format
]

# -----------------------------------------------------------
#  MODELLER – laddas en enda gång vid import
# -----------------------------------------------------------

print("Device set to use cpu")                 # → visas i GitHub Actions-loggen

summarizer = hf_pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    device="cpu",
)

translator_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", use_fast=False
)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)

# -----------------------------------------------------------
#  HJÄLPFUNKTIONER
# -----------------------------------------------------------

def _try_get_id(tok, code: str) -> int | None:
    """
    Returnerar BOS-ID för given språkkod eller None om det inte hittas.
    Stöd för både äldre och nya transformer-versioner.
    """
    # a) moderna hjälpfunktioner / tabeller med ID
    for attr in ("get_lang_id", "lang_code_to_id", "lang2id"):
        mapping = getattr(tok, attr, None)
        if mapping:
            try:
                return mapping(code) if callable(mapping) else mapping.get(code)
            except Exception:
                pass

    # b) prova välkända token-strängar (<2xx>, __xx__, <<xx>>)
    for pat in ("<2{code}>", "__{code}__", "<<{code}>>"):
        tid = tok.convert_tokens_to_ids(pat.format(code=code))
        if tid not in (tok.unk_token_id, None):
            return tid

    # c) **NYTT** – Transformers 4.50+: lang_code_to_token → str → id
    token_map = getattr(tok, "lang_code_to_token", None)
    if token_map and code in token_map:
        tid = tok.convert_tokens_to_ids(token_map[code])
        if tid not in (tok.unk_token_id, None):
            return tid

    return None



def translate(text: str, tgt_lang: str) -> str:
    """
    Översätter ENG → tgt_lang med NLLB-200 (no-op om tgt_lang == 'en').
    Kastat ValueError om språk-ID inte hittas.
    """
    if tgt_lang == "en":
        return text

    tgt_code = _LANG_CODE[tgt_lang]
    bos_id   = _try_get_id(translator_tokenizer, tgt_code)
    if bos_id is None:
        raise ValueError(f"Hittar inte språk-ID för {tgt_code}")

    enc = translator_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    out = translator_model.generate(
        **enc,
        forced_bos_token_id=bos_id,
        max_length=512,
        num_beams=4,
    )
    return translator_tokenizer.decode(out[0], skip_special_tokens=True)


def collect_articles(days: int = 7) -> list[dict]:
    """Hämtar artiklar ≤ `days` tillbaka och returnerar en lista med dictar."""
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
                continue           # hoppa trasig artikel
            docs.append(
                {"title": entry.title, "text": art.text, "url": entry.link}
            )
    return docs


def choose_top_docs(docs: list[dict], top_n: int = 20) -> list[dict]:
    """Rankar med BM25 och returnerar de top_n mest relevanta artiklarna."""
    corpus  = [f"{d['title']} {d['text']}" for d in docs]
    bm25    = BM25Okapi([c.split() for c in corpus])
    scores  = bm25.get_scores(["news", "world", "sweden"])
    ranked  = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]


def make_card(doc: dict, tgt_lang: str) -> dict:
    """Bygger ett 'kort' (rubrik + summering) på valt språk."""
    # 1) summering på engelska
    short_txt = doc["text"][:2_000]   # begränsa längd
    en_sum = summarizer(
        short_txt, max_length=100, min_length=30, do_sample=False
    )[0]["summary_text"].strip()
    en_sum = re.sub(r"\s+([.,!?;:])", r"\1", en_sum)   # fix mellanslag

    # 2) översätt titel + sammanfattning vid behov
    title   = translate(doc["title"], tgt_lang)
    summary = translate(en_sum,       tgt_lang)

    return {"title": title, "summary": summary, "url": doc["url"]}

# -----------------------------------------------------------
#  snabbtest: kör “python pipeline.py” lokalt för sanity-check
# -----------------------------------------------------------
if __name__ == "__main__":
    art = collect_articles(1)[0]
    for lang in TARGET_LANGS:
        print(f"\n--- {lang} ---")
        print(make_card(art, lang))
