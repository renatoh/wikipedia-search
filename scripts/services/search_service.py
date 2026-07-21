"""
Reusable service wrapping the hybrid-search components used by the REPL and
the agent: running the hybrid / BM25-only / kNN-only queries against the
`wikipedia` OpenSearch index.

Query-time embedding is done SERVER-SIDE by OpenSearch via the `neural`
query clause, using the same ML Commons model that the ingest pipeline uses
at index time (model_id below). The client never embeds queries itself —
index-time and query-time embedding share a single code path, and the client
has no dependency on the embed service.
"""

from opensearchpy import OpenSearch

DEFAULT_OS_HOST = "linux"
DEFAULT_OS_PORT = 9200
DEFAULT_INDEX = "wikipedia"
DEFAULT_PIPELINE = "hybrid-search-pipeline"
# ML Commons model registered in OpenSearch (remote connector to the embed
# service). Same model the ingest pipeline uses — see opensearch/dev_tool_setup.txt.
DEFAULT_MODEL_ID = "pt36Up8Bpv1M9GKHBcqd"
DEFAULT_TOP_K = 10
DEFAULT_KNN_K = 50
DEFAULT_OS_TIMEOUT_S = 30


class SearchService:
    def __init__(
        self,
        os_host: str = DEFAULT_OS_HOST,
        os_port: int = DEFAULT_OS_PORT,
        index: str = DEFAULT_INDEX,
        pipeline: str = DEFAULT_PIPELINE,
        model_id: str = DEFAULT_MODEL_ID,
        top_k: int = DEFAULT_TOP_K,
        knn_k: int = DEFAULT_KNN_K,
        os_timeout_s: int = DEFAULT_OS_TIMEOUT_S,
        client: OpenSearch | None = None,
    ):
        self.index = index
        self.pipeline = pipeline
        self.model_id = model_id
        self.top_k = top_k
        self.knn_k = knn_k
        self.client = client or OpenSearch(
            hosts=[{"host": os_host, "port": os_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            timeout=os_timeout_s,
        )

    def hybrid_search(self, query_text: str, top_k: int | None = None) -> dict:
        body = {
            "size": top_k or self.top_k,
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
                            "neural": {
                                "embedding": {
                                    "query_text": query_text,
                                    "model_id": self.model_id,
                                    "k": self.knn_k,
                                }
                            }
                        },
                    ]
                }
            },
            "_source": ["id", "title", "text"],
        }
        return self.client.search(
            index=self.index,
            body=body,
            params={"search_pipeline": self.pipeline},
        )

    def bm25_search(self, query_text: str, size: int | None = None) -> dict:
        """Same multi_match sub-query as the hybrid, run standalone so its
        raw BM25 score is visible — the hybrid response only exposes the
        combined score."""
        body = {
            "size": size or self.knn_k,
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title^2", "text"],
                }
            },
            "_source": False,
        }
        return self.client.search(index=self.index, body=body)

    def knn_search(self, query_text: str, size: int | None = None) -> dict:
        """Same neural sub-query as the hybrid, run standalone so its raw kNN
        score is visible. OpenSearch embeds the query text server-side."""
        body = {
            "size": size or self.knn_k,
            "query": {
                "neural": {
                    "embedding": {
                        "query_text": query_text,
                        "model_id": self.model_id,
                        "k": self.knn_k,
                    }
                }
            },
            "_source": False,
        }
        return self.client.search(index=self.index, body=body)

    def get(self, doc_id: str) -> dict | None:
        """Fetch a single indexed document by id. Returns the `_source` dict
        (id, title, text) or None if not found."""
        try:
            response = self.client.get(index=self.index, id=doc_id, _source=True)
        except Exception:
            return None
        return response.get("_source")

    def search_all(self, query_text: str) -> dict:
        """Run all three queries — the exact shape the REPL needs to print
        combined + per-component scores side by side. No client-side embed:
        OpenSearch embeds the query text itself for the neural clauses."""
        return {
            "hybrid": self.hybrid_search(query_text),
            "bm25": self.bm25_search(query_text),
            "knn": self.knn_search(query_text),
        }
