# --- shim so newspaper works with lxml>=5 ---
try:
    import lxml_html_clean as _hc
    import sys
    sys.modules['lxml.html.clean'] = _hc
except ImportError:
    pass
# --------------------------------------------



import feedparser, newspaper, time
from transformers import pipeline as hf_pipeline
from rank_bm25 import BM25Okapi
import re 

FEEDS = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.svt.se/nyheter/rss",
]

summarizer = hf_pipeline("summarization",
                         model="sshleifer/distilbart-cnn-6-6",
                         device="cpu")

def collect_articles(days: int = 7):
    since = time.time() - days * 86400
    docs = []
    for feed_url in FEEDS:
        for entry in feedparser.parse(feed_url).entries:
            if time.mktime(entry.published_parsed) < since:
                continue
            art = newspaper.Article(entry.link)
            art.download(); art.parse()
            docs.append({"title": entry.title,
                         "text": art.text,
                         "url": entry.link})
    return docs

def choose_top20(docs):
    corpus = [d["title"] + " " + d["text"] for d in docs]
    bm25 = BM25Okapi([c.split() for c in corpus])
    scores = bm25.get_scores(["news", "world", "sweden"])
    paired = list(zip(scores, docs))
    paired.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in paired[:20]]

def summarize(docs):
    cards = []
    for d in docs:
        short = d["text"][:2000]
        summary = summarizer(short,
                             max_length=100,
                             min_length=30,
                             do_sample=False)[0]["summary_text"]
        summary = summary.strip()                           # tar bort start/slut-mellan­slag
        summary = re.sub(r"\s+([.,!?;:])", r"\1", summary)  # tar bort space före . , ! ? ; :
        cards.append({"title": d["title"],
                      "summary": summary,
                      "url": d["url"]})
    return cards
