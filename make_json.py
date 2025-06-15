import json, pathlib, datetime as dt
import pipeline

OUT = pathlib.Path("public")
OUT.mkdir(exist_ok=True)

for span, days in {"day":1, "week":7, "month":30}.items():
    docs  = pipeline.collect_articles(days)
    top20 = pipeline.choose_top20(docs)
    cards = pipeline.summarize(top20)
    payload = {
        "generated": dt.datetime.utcnow()
                     .isoformat(timespec="seconds") + "Z",
        "cards": cards
    }
    (OUT / f"{span}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )

print("✔️  JSON-filer klara i public/")
