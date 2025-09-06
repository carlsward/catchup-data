#!/usr/bin/env python3
import feedparser

URLS = [
     "https://api.sr.se/api/rss/program/83",
     "https://svenska.yle.fi/rss/utrikes",
     "https://www.svt.se/nyheter/utrikes/rss.xml",
]

for u in URLS:
    # En snäll user-agent kan hjälpa mot 403 ibland
    f = feedparser.parse(u, agent="Mozilla/5.0 (compatible; rss-check/1.0)")
    status = getattr(f, "status", None)
    entries = getattr(f, "entries", []) or []
    print(f"\n{u}\nstatus={status} entries={len(entries)}")
    for e in entries[:2]:
        title = getattr(e, "title", "") or ""
        print("  -", title[:80])
