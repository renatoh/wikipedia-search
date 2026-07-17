"""
Stream wiki_clean_complete.jsonl into OpenSearch without loading the file
into memory. Reads line by line (generator), lets opensearch-py's
parallel_bulk chunk + send batches concurrently.
"""

import json
import time

from opensearchpy import OpenSearch, helpers

INPUT_PATH = "../../resources/wiki_clean_complete.jsonl"
INDEX_NAME = "wikipedia"
BULK_CHUNK_SIZE = 20  # docs per _bulk request — tune against embed throughput
THREAD_COUNT = 1  # concurrent bulk requests in flight
LOG_EVERY = 10_000

client = OpenSearch(
    hosts=[{"host": "linux", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
)


def generate_actions(path):
    """Generator — reads one JSONL line at a time, never holds the whole file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            yield {
                "_index": INDEX_NAME,
                "_id": doc["id"],  # idempotent: reruns overwrite, don't duplicate
                "_source": {
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                },
                # Relies on the index's "default_pipeline": "wiki-embedding-pipeline"
                # setting to trigger embedding. Uncomment to be explicit instead:
                # "pipeline": "wiki-embedding-pipeline",
            }


if __name__ == "__main__":
    success = 0
    errors = 0
    start = time.time()

    for ok, item in helpers.parallel_bulk(
        client,
        generate_actions(INPUT_PATH),
        chunk_size=BULK_CHUNK_SIZE,
        thread_count=THREAD_COUNT,
        raise_on_error=False,
    ):
        if ok:
            success += 1
        else:
            errors += 1
            print(f"error: {item}", flush=True)

        total = success + errors
        if total % LOG_EVERY == 0:
            elapsed = round(time.time() - start, 1)
            rate = round(total / elapsed, 1) if elapsed > 0 else 0
            print(
                f"indexed: {success:,}, errors: {errors:,}, "
                f"elapsed: {elapsed}s, rate: {rate}/s",
                flush=True,
            )

    print(
        f"done. indexed: {success:,}, errors: {errors:,}, "
        f"total time: {round(time.time() - start, 1)}s",
        flush=True,
    )