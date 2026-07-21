"""
Interactive hybrid search REPL against the `wikipedia` index.

Type a query, see ranked results, type another — no restarting the script
per query. Useful for quickly A/B testing different search pipelines
(e.g. the arithmetic_mean normalization-processor vs. the RRF
score-ranker-processor) by just flipping PIPELINE_NAME below.

Usage:
    python3 search_repl.py
    (then type queries, or "exit"/"quit" to stop)
"""

from services import SearchService

# NOTE: currently just one pipeline exists under this name — it was
# recreated with the RRF score-ranker-processor, overwriting the earlier
# arithmetic_mean normalization-processor version (same pipeline id). If you
# want to A/B both, PUT the RRF version under a different id (e.g.
# "hybrid-search-pipeline-rrf") and flip this constant to compare them.
PIPELINE_NAME = "hybrid-search-pipeline"

service = SearchService(pipeline=PIPELINE_NAME)


def print_results(query_text: str, hybrid_result: dict, bm25_result: dict, knn_result: dict):
    bm25_scores = {hit["_id"]: hit["_score"] for hit in bm25_result["hits"]["hits"]}
    knn_scores = {hit["_id"]: hit["_score"] for hit in knn_result["hits"]["hits"]}
    # 1-based rank within each sub-query's own result list — this is exactly
    # what RRF's combination is computed from (1 / (rank_constant + rank)).
    bm25_ranks = {hit["_id"]: i + 1 for i, hit in enumerate(bm25_result["hits"]["hits"])}
    knn_ranks = {hit["_id"]: i + 1 for i, hit in enumerate(knn_result["hits"]["hits"])}

    hits = hybrid_result["hits"]["hits"]
    print(f"\npipeline: {PIPELINE_NAME}")
    print(f"query: {query_text!r} — {len(hits)} result(s)\n")
    for i, hit in enumerate(hits):
        source = hit["_source"]
        snippet = source.get("text", "")[:200].replace("\n", " ")
        na = f"n/a (outside top-{service.knn_k})"

        bm25_score = bm25_scores.get(hit["_id"])
        knn_score = knn_scores.get(hit["_id"])
        bm25_str = f"{bm25_score:.4f}" if bm25_score is not None else na
        knn_str = f"{knn_score:.4f}" if knn_score is not None else na

        bm25_rank = bm25_ranks.get(hit["_id"])
        knn_rank = knn_ranks.get(hit["_id"])
        bm25_rank_str = f"#{bm25_rank}" if bm25_rank is not None else na
        knn_rank_str = f"#{knn_rank}" if knn_rank is not None else na

        print(
            f"[{i + 1}] combined={hit['_score']:.4f} "
            f"bm25={bm25_str} (rank {bm25_rank_str}) "
            f"knn={knn_str} (rank {knn_rank_str}) "
            f"id={source.get('id')} title={source.get('title')!r}"
        )
        print(f"    {snippet}...\n")


if __name__ == "__main__":
    print(f"hybrid search REPL — pipeline: {PIPELINE_NAME}")
    print("type a query, or 'exit'/'quit' to stop\n")

    while True:
        try:
            query_text = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query_text:
            continue
        if query_text.lower() in ("exit", "quit"):
            break

        try:
            results = service.search_all(query_text)
            print_results(query_text, results["hybrid"], results["bm25"], results["knn"])
        except Exception as e:
            print(f"error: {e}\n")
