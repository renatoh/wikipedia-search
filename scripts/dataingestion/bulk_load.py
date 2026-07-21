"""
Stream wiki_clean_complete.jsonl into OpenSearch without loading the file
into memory. Reads line by line (generator).

NO-CONCURRENCY BASELINE: uses streaming_bulk (single-threaded, no client-side
thread pool at all — unlike parallel_bulk) with chunk_size=1, so exactly one
document is sent per _bulk request. This also matters server-side: even a
single client thread sending a chunk_size=200 request could still have
OpenSearch's own internal write thread pool process those 200 documents'
pipeline executions concurrently. chunk_size=1 removes that possibility too,
since there's only ever one document for OpenSearch to process per request.
Once this is confirmed stable, chunk_size/thread_count can be raised back up
incrementally to find where concurrency actually becomes a problem.
"""

import json
import subprocess
import time
from pathlib import Path

from opensearchpy import OpenSearch, helpers

# Script lives in scripts/, data lives in resources/ — resolve relative to
# this file so it works regardless of the current working directory.
INPUT_PATH = Path(__file__).parent.parent / "resources" / "wiki_clean_with_lead.jsonl"
INDEX_NAME = "wikipedia"
BULK_CHUNK_SIZE = 10  # one doc per _bulk request — no concurrency, client or server side
LOG_EVERY = 500

# Resume support: if a previous run was interrupted, set this to how many
# documents are already confirmed indexed (e.g. via `GET wikipedia/_count`).
# The first RESUME_FROM non-blank lines of the input file are then skipped.
# Docs are indexed with _id = wikipedia page id, so reruns overwrite rather
# than duplicate — if unsure of the exact count, round DOWN (skip fewer).
# A few re-embedded docs from the overlap are harmless; skipping too many
# would silently drop documents. Leave at 0 for a fresh run.
RESUME_FROM = 0

client = OpenSearch(
    hosts=[{"host": "linux", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
)


def generate_actions(path, skip=0):
    """Generator — reads one JSONL line at a time, never holds the whole file.
    Skips the first `skip` non-blank lines, for resuming an interrupted run."""
    with open(path, "r", encoding="utf-8") as f:
        skipped = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skipped < skip:
                skipped += 1
                continue
            doc = json.loads(line)
            yield {
                "_index": INDEX_NAME,
                "_id": doc["id"],  # idempotent: reruns overwrite, don't duplicate
                "_source": {
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    # Transient: the ingest pipeline reads this to build
                    # embedding_source, then a remove processor drops it
                    # before storage — it's redundant with `text` otherwise.
                    "lead": doc["lead"],
                },
                # Relies on the index's "default_pipeline": "wiki-embedding-pipeline"
                # setting to trigger embedding. Uncomment to be explicit instead:
                # "pipeline": "wiki-embedding-pipeline",
            }


def count_lines(path):
    """Fast total-line count for the ETA calculation below. Falls back to a
    plain Python count if `wc` isn't available for some reason."""
    try:
        result = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True, check=True)
        return int(result.stdout.split()[0])
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)


if __name__ == "__main__":
    print("counting total lines in input file (for ETA)...", flush=True)
    total_lines = count_lines(INPUT_PATH)
    remaining_at_start = max(total_lines - RESUME_FROM, 0)
    print(f"total lines: {total_lines:,}, already done (RESUME_FROM): {RESUME_FROM:,}, "
          f"remaining: {remaining_at_start:,}", flush=True)

    success = 0
    errors = 0
    start = time.time()

    for ok, item in helpers.streaming_bulk(
        client,
        generate_actions(INPUT_PATH, skip=RESUME_FROM),
        chunk_size=BULK_CHUNK_SIZE,
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
            remaining = max(remaining_at_start - total, 0)
            eta_hours = round(remaining / rate / 3600, 1) if rate > 0 else float("inf")
            print(
                f"indexed: {success:,}, errors: {errors:,}, "
                f"elapsed: {elapsed}s, rate: {rate}/s, eta: {eta_hours}h",
                flush=True,
            )

    print(
        f"done. indexed: {success:,}, errors: {errors:,}, "
        f"total time: {round(time.time() - start, 1)}s",
        flush=True,
    )
