"""
Manual test: load the first N articles from resources/wiki_clean_complete.jsonl
and send them to the embed service one at a time (sequential, not batched),
so you can watch `nvidia-smi` per-request and see how VRAM/timing behaves
across a range of real article lengths — not just one hand-picked sample.
"""

import json
import time
from pathlib import Path

import requests

ENDPOINT = "http://linux:8000/embed"

# Script lives in scripts/, data lives in resources/ — resolve relative to
# this file so it works regardless of the current working directory.
DATA_PATH = Path(__file__).parent.parent / "resources" / "wiki_clean_complete.jsonl"

NUM_ARTICLES = 5000  # how many articles to pull from the top of the file


def load_first_n_articles(path: Path, n: int):
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            articles.append(json.loads(line))
    return articles


def embed_one(text: str):
    start = time.time()
    response = requests.post(ENDPOINT, json=[text])
    elapsed = round(time.time() - start, 3)
    return response, elapsed


if __name__ == "__main__":
    articles = load_first_n_articles(DATA_PATH, NUM_ARTICLES)
    print(f"loaded {len(articles)} article(s) from {DATA_PATH}", flush=True)

    start =  time.time()
    for i, article in enumerate(articles):
        text = f"{article['title']}\n\n{article['text']}"
        response, elapsed = embed_one(text)

        if response.status_code != 200:
            print(
                f"[{i}] id={article['id']} title={article['title']!r} "
                f"FAILED status={response.status_code} elapsed={elapsed}s",
                flush=True,
            )
            print("error response:", response.text, flush=True)
            continue

        embedding = response.json()[0]
        print(
            f"[{i}] id={article['id']} title={article['title']!r} "
            f"chars={len(text):,} dim={len(embedding)} elapsed={elapsed}s",
            flush=True,
        )
    print(f"{NUM_ARTICLES} proccesses in {time.time()-start}s")