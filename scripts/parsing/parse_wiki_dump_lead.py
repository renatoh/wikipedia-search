"""
Stream-parse enwiki-latest-pages-articles.xml.bz2 (same streaming approach as
parse_wiki_dump.py), but additionally extract each article's lead section
(the text before the first level-2 "== heading ==") into a separate `lead`
field.

Rationale: Wikipedia's own convention is that the lead summarizes the whole
article, and it's naturally short (a few hundred tokens) even for very long
articles. Right now ~21% of the corpus exceeds the ~2048-token embedding
truncation cutoff, forcing a full-length (and therefore slow) GPU forward
pass while still discarding most of the article's content. Embedding `lead`
instead of `text` keeps the same GPU cap but makes those calls dramatically
shorter for that ~21% slice — and for articles with no headings at all (most
stubs), `lead` just equals the full text, so nothing changes for them.

`text` is still parsed and kept in full for BM25/keyword search — only the
embedding_source (built downstream in the ingest pipeline) switches from
title+text to title+lead.

Output goes to a NEW file, resources/wiki_clean_with_lead.jsonl — this does
not touch or overwrite the existing wiki_clean_complete.jsonl.
"""

import bz2
import json
import multiprocessing
import time
from pathlib import Path
from xml.etree.ElementTree import iterparse

import mwparserfromhell

DUMP_PATH = "/Users/renato/Downloads/enwiki-latest-pages-articles.xml.bz2"

# Script lives in scripts/, data lives in resources/ — resolve relative to
# this file so it works regardless of the current working directory.
OUTPUT_PATH = Path(__file__).parent.parent / "resources" / "wiki_clean_with_lead.jsonl"
LOG_EVERY = 10_000


def iter_articles(path=DUMP_PATH):
    with bz2.open(path, "rb") as f:
        context = iterparse(f, events=("start", "end"))
        _, root = next(context)  # grab the root element to read its namespace
        ns_uri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
        tag = lambda name: f"{{{ns_uri}}}{name}" if ns_uri else name

        for event, elem in context:
            if event != "end" or elem.tag != tag("page"):
                continue

            ns_id_elem = elem.find(tag("ns"))
            redirect_elem = elem.find(tag("redirect"))

            # Only main-namespace articles, skip redirects
            if ns_id_elem is not None and ns_id_elem.text == "0" and redirect_elem is None:
                title_elem = elem.find(tag("title"))
                id_elem = elem.find(tag("id"))
                revision_elem = elem.find(tag("revision"))
                text_elem = revision_elem.find(tag("text")) if revision_elem is not None else None

                yield {
                    "id": id_elem.text if id_elem is not None else None,
                    "title": title_elem.text if title_elem is not None else "",
                    "text": text_elem.text if text_elem is not None else "",
                }

            # CRITICAL: free memory, same as parse_wiki_dump.py.
            elem.clear()
            root.clear()


def _clean(wikicode) -> str:
    """Strip <ref> tags and File:/Image:/Category: wikilinks, then render
    to plain text. Shared by both the full-article and lead-only cleanup."""
    for tag in wikicode.filter_tags(matches=lambda n: n.tag == "ref"):
        try:
            wikicode.remove(tag)
        except ValueError:
            pass

    for link in wikicode.filter_wikilinks():
        title = str(link.title).strip().lower()
        if title.startswith(("file:", "image:", "category:")):
            try:
                wikicode.remove(link)
            except ValueError:
                pass

    return wikicode.strip_code()


def clean_article(raw_text: str):
    """Returns (full_clean_text, lead_clean_text)."""
    wikicode = mwparserfromhell.parse(raw_text)

    # Extract the lead's raw wikitext BEFORE mutating/stripping the full
    # tree below. get_sections(levels=[2], include_lead=True) returns the
    # untitled section before the first level-2 heading as index 0; if
    # there are no level-2 headings at all (most stub articles), that
    # section is the entire article.
    sections = wikicode.get_sections(levels=[2], include_lead=True, flat=True)
    lead_raw = str(sections[0]) if sections else raw_text

    full_clean = _clean(wikicode)
    lead_clean = _clean(mwparserfromhell.parse(lead_raw))

    return full_clean, lead_clean


def process_article(article):
    """Top-level (picklable) wrapper so it can run in a worker process."""
    full_clean, lead_clean = clean_article(article["text"])
    article["text"] = full_clean
    article["lead"] = lead_clean
    return article


if __name__ == "__main__":
    print(f"starting parse of {DUMP_PATH}", flush=True)
    count = 0
    start = time.time()
    batch_start = time.time()

    # Same pattern as parse_wiki_dump.py: streaming XML/bz2 decompression
    # stays single-threaded (I/O-bound), CPU-heavy wikitext cleaning fans
    # out across worker processes via imap.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f, \
         multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for article in pool.imap(process_article, iter_articles(), chunksize=200):
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
            count += 1

            if count % LOG_EVERY == 0:
                f.flush()
                print(
                    f"processed {count:,}, time taken for batch "
                    f"{round(time.time() - batch_start, 3)}s... last: {article['title']}",
                    flush=True,
                )
                print(f"total time taken: {round(time.time() - start, 1)}s", flush=True)
                batch_start = time.time()

    print(f"done. total articles: {count:,}, total time: {round(time.time() - start, 1)}s", flush=True)
