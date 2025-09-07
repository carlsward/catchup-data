# backfill.py
from __future__ import annotations
"""
CatchUp – smal backfill för 30 dagar:
- Läser källor från sources_backfill.yaml
- Stöd för:
  * type: "sitemap"  – URL till (del) sitemap; filtrera med include/exclude-regex
  * type: "listing"  – kategori-/arkivsida med enkel /page/N/ eller ?page=N pagination
- Respekterar robots.txt, snälla delays
- Extraherar text via Trafilatura; datum från sitemap <lastmod> eller sida (<time>, JSON-LD)
- Returnerar docs i samma format som pipeline.collect_articles()
"""

import re, time, json, logging, math
import urllib.parse
import urllib.robotparser
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET

import requests
import yaml

# Trafilatura behövs för robust extraktion
try:
    import trafilatura
except Exception:
    trafilatura = None

# Återanvänd befintlig kods konstanter/hjälp
import pipeline

# ----------- konfig (env-styrd vid behov) -----------------
BACKFILL_ENABLED = True  # kan överstyras i make_json via env
TIMEOUT = 15
SLEEP = 1.0
MAX_PAGES = int(Path(".env.MAX_PAGES").read_text().strip()) if Path(".env.MAX_PAGES").exists() else 12
MAX_SITEMAPS = 8  # säkerhet ifall en sitemapindex pekar på många mappar

HEADERS = {"User-Agent": pipeline.USER_AGENT}

# ----------- hjälp ----------------------------------------
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _cutoff(days: int) -> datetime:
    return _utc_now() - timedelta(days=days)

def _domain(url: str) -> str:
    return pipeline._domain(url)

def _lang_ok(text: str, lang: str = "sv", rss_fallback: bool = False) -> bool:
    det, prob = pipeline.detect_lang_code(text or "")
    need = pipeline.LANG_OK_PROB_RSS if rss_fallback else pipeline.LANG_OK_PROB
    return (det == lang) or (prob < need)

def _clean(s: str) -> str:
    return pipeline.clean_text(s or "")

def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # försök några vanliga varianter utan externa beroenden
    s = s.strip()
    try:
        # ISO med Z → +00:00
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        pass
    # Fallback: YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except Exception:
            return None
    return None

def _robots_ok(url: str, rp: urllib.robotparser.RobotFileParser) -> bool:
    try:
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True

def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return r

def _iter_listing_pages(base_url: str) -> List[str]:
    # stöder både /page/N/ och ?page=N
    urls = []
    u = base_url.rstrip("/")
    for p in range(1, MAX_PAGES + 1):
        if re.search(r"/page/\d+/?$", u):
            url = re.sub(r"/page/\d+/?$", f"/page/{p}/", u)
        elif u.endswith("/"):
            url = f"{u}page/{p}/"
        else:
            sep = "&" if "?" in u else "?"
            url = f"{u}{sep}page={p}"
        urls.append(url)
    return urls

def _extract_article(url: str, html: str) -> Dict:
    """Trafilatura → JSON, plocka text, titel, datum."""
    if not trafilatura:
        return {}
    data_json = trafilatura.extract(html, url=url, output="json", with_metadata=True)
    if not data_json:
        return {}
    data = json.loads(data_json)
    # titel + text
    title = _clean(data.get("title") or "")
    text = _clean(data.get("text") or "")
    # datum från trafilatura
    pub = _parse_date(data.get("date"))
    if not pub:
        # försök sniffa <time datetime="..."> eller JSON-LD datePublished
        m = re.search(r'datetime="([^"]+)"', html) or re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
        if m:
            pub = _parse_date(m.group(1))
    if not title and not text:
        return {}
    return {"title": title, "text": text, "published_dt": pub}

# ----------- crawlers -------------------------------------
def crawl_sitemap(source: Dict, stop_days: int) -> List[Dict]:
    cutoff = _cutoff(stop_days)
    out: List[Dict] = []

    base = source["url"]
    include_re = re.compile(source.get("include", "."), re.I)
    exclude_re = re.compile(source.get("exclude", r"$a"), re.I)

    # robots
    robots_url = urllib.parse.urljoin(base, "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url); rp.read()
    except Exception:
        pass

    try:
        r = _get(base)
    except Exception as e:
        logging.warning("Backfill sitemap misslyckades: %s: %s", base, e)
        return out

    try:
        root = ET.fromstring(r.text)
    except Exception as e:
        logging.warning("Kunde inte parsa sitemap XML: %s", e)
        return out

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    sitemap_nodes = root.findall(".//sm:sitemap", ns)
    url_nodes = root.findall(".//sm:url", ns)

    def _maybe_add(u: str, lastmod: Optional[str]):
        if not include_re.search(u) or exclude_re.search(u):
            return
        if not _robots_ok(u, rp):
            return
        pub_dt = _parse_date(lastmod) if lastmod else None
        # Snabbdatumsfilter: om lastmod finns och är äldre än cutoff → hoppa
        if pub_dt and pub_dt < cutoff:
            return
        try:
            ar = _get(u)
        except Exception:
            return
        item = _extract_article(u, ar.text)
        if not item:
            return
        # sätt datum: preferera det från artikel om vettigt, annars lastmod
        pub = item.get("published_dt") or pub_dt
        if not pub or pub < cutoff:
            return
        text = item.get("text") or ""
        title = item.get("title") or ""
        if len(text) < pipeline.MIN_TEXT_CHARS_ARTICLE:
            return
        if not _lang_ok(text, "sv", rss_fallback=False):
            return
        out.append({
            "title": title,
            "text": text,
            "url": u,
            "published": int(pub.timestamp()),
            "domain": _domain(u),
            "language": "sv",
            "category": "world",
            "fallback": "html",
        })

    if sitemap_nodes:
        # sitemapindex → loopa igenom ett begränsat antal för att vara snäll
        for i, sm in enumerate(sitemap_nodes[:MAX_SITEMAPS]):
            loc = sm.findtext("sm:loc", default="", namespaces=ns).strip()
            if not loc:
                continue
            try:
                rs = _get(loc)
                child = ET.fromstring(rs.text)
            except Exception:
                continue
            for urlnode in child.findall(".//sm:url", ns):
                loc2 = urlnode.findtext("sm:loc", default="", namespaces=ns).strip()
                lastmod = urlnode.findtext("sm:lastmod", default="", namespaces=ns).strip()
                _maybe_add(loc2, lastmod)
                time.sleep(SLEEP)
    elif url_nodes:
        for urlnode in url_nodes:
            loc = urlnode.findtext("sm:loc", default="", namespaces=ns).strip()
            lastmod = urlnode.findtext("sm:lastmod", default="", namespaces=ns).strip()
            _maybe_add(loc, lastmod)
            time.sleep(SLEEP)

    return out

def crawl_listing(source: Dict, stop_days: int) -> List[Dict]:
    cutoff = _cutoff(stop_days)
    out: List[Dict] = []

    base = source["url"]
    include_re = re.compile(source.get("include", "."), re.I)
    exclude_re = re.compile(source.get("exclude", r"$a"), re.I)

    # robots
    robots_url = urllib.parse.urljoin(base, "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url); rp.read()
    except Exception:
        pass

    for list_url in _iter_listing_pages(base):
        if not _robots_ok(list_url, rp):
            break
        try:
            r = _get(list_url)
        except Exception:
            break

        # plocka länkar
        urls = list(set(re.findall(r'href="(https?://[^"]+)"', r.text)))
        got_any = False
        for u in urls:
            if not include_re.search(u) or exclude_re.search(u):
                continue
            if not _robots_ok(u, rp):
                continue
            try:
                ar = _get(u)
            except Exception:
                continue

            item = _extract_article(u, ar.text)
            if not item:
                continue
            pub = item.get("published_dt")
            if not pub or pub < cutoff:
                continue
            text = item.get("text") or ""
            title = item.get("title") or ""
            if len(text) < pipeline.MIN_TEXT_CHARS_ARTICLE:
                continue
            if not _lang_ok(text, "sv", rss_fallback=False):
                continue

            out.append({
                "title": title,
                "text": text,
                "url": u,
                "published": int(pub.timestamp()),
                "domain": _domain(u),
                "language": "sv",
                "category": "world",
                "fallback": "html",
            })
            got_any = True
        time.sleep(SLEEP)
        if not got_any:
            break  # antag slutet

    return out

# ----------- API ------------------------------------------
def load_backfill_sources() -> List[Dict]:
    p = Path("sources_backfill.yaml")
    if not p.exists():
        return []
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    # förväntat schema:
    # backfill:
    #   world:
    #     sv:
    #       - {type: "sitemap"|"listing", url: "...", include: "regex", exclude: "regex", stop_days: 30}
    bx = cfg.get("backfill", {})
    out = []
    for category, langs in (bx.items() if isinstance(bx, dict) else []):
        for lang, sources in (langs.items() if isinstance(langs, dict) else []):
            for s in (sources or []):
                s = dict(s)
                s["_category"] = category
                s["_lang"] = lang
                out.append(s)
    return out

def run_backfill(category: str, lang: str, days: int = 30) -> List[Dict]:
    """Kör backfill för en kategori/språk. Returnerar docs-lista."""
    all_sources = load_backfill_sources()
    sources = [s for s in all_sources if s.get("_category")==category and s.get("_lang")==lang]
    if not sources:
        print("ℹ️  Backfill: inga källor i sources_backfill.yaml för", category, lang, flush=True)
        return []
    print(f"🔎 Backfill start ({category}/{lang}, ≤{days} d)…", flush=True)
    agg: List[Dict] = []
    for s in sources:
        typ = (s.get("type") or "").lower()
        stop_days = int(s.get("stop_days") or days)
        try:
            if typ == "sitemap":
                docs = crawl_sitemap(s, stop_days)
            elif typ == "listing":
                docs = crawl_listing(s, stop_days)
            else:
                print(f"⚠️  Okänd backfill-typ: {typ}  ({s.get('url')})", flush=True)
                docs = []
        except Exception as e:
            print(f"⚠️  Backfill-fel {typ}: {s.get('url')}: {e}", flush=True)
            docs = []
        if docs:
            print(f"  • {typ}: +{len(docs)}  ({s.get('url')})", flush=True)
            agg.extend(docs)
        time.sleep(0.2)
    print(f"✅ Backfill klart: {len(agg)} artiklar", flush=True)
    return agg
