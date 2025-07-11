# make_json.py  – komplett fil
import json, datetime as dt, pipeline

SPANS = {
    "day":   (1, 5),    # (antal dagar, antal kort)
    "week":  (7, 10),
    "month": (30, 20),
}
LANGS = ["en", "sv", "de", "es", "fr"]   # lägg till/ta bort språk här

def generate():
    for span, (days, n_cards) in SPANS.items():
        docs = pipeline.collect_articles(days)
        top  = pipeline.choose_top(docs, n_cards)
        for lang in LANGS:
            cards = pipeline.build_cards(top, lang)
            out = {
                "generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "span": span,
                "lang": lang,
                "cards": cards,
            }
            fname = f"public/{span}_{lang}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print("✅", fname)

if __name__ == "__main__":
    generate()
