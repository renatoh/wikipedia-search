"""
Stream-parse enwiki-latest-pages-articles.xml.bz2 without loading it into memory.

Two tricks make this work:
1. bz2.open() decompresses as a stream (file-like object), never materializing
   the full ~100GB uncompressed XML on disk or in memory.
2. xml.etree.ElementTree.iterparse() parses incrementally, emitting each
   element as it closes. We clear() each <page> after processing so the
   tree doesn't keep growing — without this, iterparse still builds up the
   whole tree in memory just like a normal parse.
"""

import bz2
import json
import multiprocessing
import time
from xml.etree.ElementTree import iterparse

import mwparserfromhell

DUMP_PATH = "/Users/renato/Downloads/enwiki-latest-pages-articles.xml.bz2"
OUTPUT_PATH = "wiki_clean.jsonl"
LOG_EVERY = 10_000

# MediaWiki export XML uses a default namespace, e.g.
# {http://www.mediawiki.org/xml/export-0.11/}page — so tags must be matched
# with that prefix. We detect it from the root element instead of hardcoding
# the version number.


def iter_articles(path=DUMP_PATH):
    with bz2.open(path, "rb") as f:
        context = iterparse(f, events=("start", "end"))
        _, root = next(context)  # grab the root element to read its namespace
        ns_uri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
        ns = {"mw": ns_uri} if ns_uri else {}
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

            # CRITICAL: free memory. Clear the page element itself, and drop
            # its now-empty subtree from the root so root doesn't keep
            # accumulating every processed page forever.
            elem.clear()
            root.clear()


def clean_wikitext(text: str) -> str:
    wikicode = mwparserfromhell.parse(text)

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


def process_article(article):
    """Top-level (picklable) wrapper so it can run in a worker process."""
    article["text"] = clean_wikitext(article["text"])
    return article


if __name__ == "__main__":
    print(f"starting parse of {DUMP_PATH}", flush=True)
    count = 0
    start = time.time()
    batch_start = time.time()

    # iter_articles() stays single-threaded (streaming XML/bz2 decompression
    # is I/O-bound and cheap). The CPU-heavy wikitext cleaning is what's slow,
    # so we fan that out across worker processes via imap — order is
    # preserved, and results stream back lazily so memory stays bounded.
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
