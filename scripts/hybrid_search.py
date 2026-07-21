"""
One-shot hybrid (BM25 + kNN) search against the `wikipedia` index.

Thin CLI wrapper over SearchService — the actual query construction and
server-side (neural) query embedding live there. For an interactive version
that also shows per-component BM25/kNN scores, use search_repl.py.

Usage:
    python3 hybrid_search.py "your query text here"
"""

import sys

from services import SearchService

TOP_K = 10

service = SearchService()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python3 hybrid_search.py "query text"')
        sys.exit(1)

    query_text = " ".join(sys.argv[1:])
    result = service.hybrid_search(query_text, top_k=TOP_K)

    hits = result["hits"]["hits"]
    print(f"query: {query_text!r} — {len(hits)} result(s)\n")
    for i, hit in enumerate(hits):
        source = hit["_source"]
        snippet = source.get("text", "")[:200].replace("\n", " ")
        print(f"[{i + 1}] score={hit['_score']:.4f} id={source.get('id')} title={source.get('title')!r}")
        print(f"    {snippet}...\n")
