"""
Hybrid (BM25 + kNN) search against the `wikipedia` index.

Combines a BM25 multi_match query (title^2 + text) with a kNN vector query
against the `embedding` field. Scores are normalized and combined by the
`hybrid-search-pipeline` search pipeline — set that up once via Dev Tools
before running this (see the PUT _search/pipeline/... command discussed
alongside this script; relative BM25/kNN weighting lives there, not here).

Usage:
    python3 hybrid_search.py "your query text here"
"""

import sys

import requests
from opensearchpy import OpenSearch

EMBED_ENDPOINT = "http://linux:8000/embed"
INDEX_NAME = "wikipedia"
SEARCH_PIPELINE = "hybrid-search-pipeline"
TOP_K = 10

# How many candidates the kNN side considers before normalization/combination
# — not the same as TOP_K, which is how many final results you get back.
KNN_K = 50

client = OpenSearch(
    hosts=[{"host": "linux", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
)


def embed_query(text: str):
    # NOTE: uses the same plain /embed endpoint as document ingestion.
    # Qwen3-Embedding is asymmetric — query-side encoding can be improved
    # later with a prompt_name="query" variant (see embed_server.py notes),
    # but plain encoding on both sides works fine to start.
    response = requests.post(EMBED_ENDPOINT, json=[text])
    response.raise_for_status()
    return response.json()[0]


def hybrid_search(query_text: str, top_k: int = TOP_K):
    vector = embed_query(query_text)

    body = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["title^2", "text"],
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": vector,
                                "k": KNN_K,
                            }
                        }
                    },
                ]
            }
        },
        "_source": ["id", "title", "text"],
    }

    return client.search(
        index=INDEX_NAME,
        body=body,
        params={"search_pipeline": SEARCH_PIPELINE},
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python3 hybrid_search.py "query text"')
        sys.exit(1)

    query_text = " ".join(sys.argv[1:])
    result = hybrid_search(query_text)

    hits = result["hits"]["hits"]
    print(f"query: {query_text!r} — {len(hits)} result(s)\n")
    for i, hit in enumerate(hits):
        source = hit["_source"]
        snippet = source.get("text", "")[:200].replace("\n", " ")
        print(f"[{i + 1}] score={hit['_score']:.4f} id={source.get('id')} title={source.get('title')!r}")
        print(f"    {snippet}...\n")
