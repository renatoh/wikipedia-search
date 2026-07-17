"""
Run on the MacBook (M5 Pro, 64GB) — separate machine from the embedding GPU
box, so this never competes with Qwen3-Embedding-4B for VRAM.

Reads wiki_clean_complete.jsonl, and for each article builds an
"embedding_source" field (title + body) that's guaranteed to fit under the
2048-token cap used by the embedding service:
  - Short articles: title + full text, unchanged.
  - Long articles (> MAX_TOKENS): title + an abstractive summary of the body,
    instead of naive truncation, so semantic coverage isn't limited to just
    the lead section.

Output: wiki_with_embedding_source.jsonl — same fields as input, plus
"embedding_source". Point bulk_load.py at this file instead, and simplify
the OpenSearch ingest pipeline to skip the Painless combine step (map
"embedding_source" -> "embedding" directly, since it's precomputed here).
"""

import json
import time

from transformers import AutoTokenizer, pipeline

INPUT_PATH = "../../resources/wiki_clean_complete.jsonl"
OUTPUT_PATH = "wiki_with_embedding_source.jsonl"

# Lightweight tokenizer only — no need to load the full 4B embedding model
# just to count tokens.
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-4B"
MAX_TOKENS = 2048

# distilbart is much cheaper than bart-large-cnn and runs well on MPS;
# swap for "facebook/bart-large-cnn" if you want higher quality and don't
# mind the extra compute.
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
SUMMARY_MAX_TOKENS = 512  # leaves room for title + summary well under 2048

LOG_EVERY = 5_000

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device="mps")


def token_count(text: str) -> int:
    return len(tokenizer(text, truncation=False)["input_ids"])


def summarize(text: str) -> str:
    # distilbart has its own ~1024-token input limit; truncate the source
    # text going INTO the summarizer (separate from the final embedding cap).
    result = summarizer(
        text,
        max_length=SUMMARY_MAX_TOKENS,
        min_length=64,
        truncation=True,
        do_sample=False,
    )
    return result[0]["summary_text"]


def build_embedding_source(title: str, text: str) -> str:
    combined = f"{title}\n\n{text}"
    if token_count(combined) <= MAX_TOKENS:
        return combined
    return f"{title}\n\n{summarize(text)}"


if __name__ == "__main__":
    print("starting summarization pass", flush=True)
    count = 0
    summarized_count = 0
    start = time.time()

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            before_tokens = token_count(f"{doc['title']}\n\n{doc['text']}")
            doc["embedding_source"] = build_embedding_source(doc["title"], doc["text"])
            if before_tokens > MAX_TOKENS:
                summarized_count += 1

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

            if count % LOG_EVERY == 0:
                fout.flush()
                elapsed = round(time.time() - start, 1)
                print(
                    f"processed {count:,} ({summarized_count:,} summarized), "
                    f"elapsed: {elapsed}s",
                    flush=True,
                )

    print(
        f"done. total: {count:,}, summarized: {summarized_count:,}, "
        f"time: {round(time.time() - start, 1)}s",
        flush=True,
    )