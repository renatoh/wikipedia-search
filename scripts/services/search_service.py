"""
Reusable service wrapping the hybrid-search components used by the REPL:
embedding the query, and running the hybrid / BM25-only / kNN-only queries
against the `wikipedia` OpenSearch index.
"""

import requests
from opensearchpy import OpenSearch

DEFAULT_EMBED_ENDPOINT = "http://linux:8000/embed"
DEFAULT_OS_HOST = "linux"
DEFAULT_OS_PORT = 9200
DEFAULT_INDEX = "wikipedia"
DEFAULT_PIPELINE = "hybrid-search-pipeline"
DEFAULT_TOP_K = 10
DEFAULT_KNN_K = 50
DEFAULT_OS_TIMEOUT_S = 30
DEFAULT_EMBED_TIMEOUT_S = 30


class SearchService:
    def __init__(
        self,
        embed_endpoint: str = DEFAULT_EMBED_ENDPOINT,
        os_host: str = DEFAULT_OS_HOST,
        os_port: int = DEFAULT_OS_PORT,
        index: str = DEFAULT_INDEX,
        pipeline: str = DEFAULT_PIPELINE,
        top_k: int = DEFAULT_TOP_K,
        knn_k: int = DEFAULT_KNN_K,
        os_timeout_s: int = DEFAULT_OS_TIMEOUT_S,
        embed_timeout_s: int = DEFAULT_EMBED_TIMEOUT_S,
        client: OpenSearch | None = None,
    ):
        self.embed_endpoint = embed_endpoint
        self.index = index
        self.pipeline = pipeline
        self.top_k = top_k
        self.knn_k = knn_k
        self.embed_timeout_s = embed_timeout_s
        self.client = client or OpenSearch(
            hosts=[{"host": os_host, "port": os_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            timeout=os_timeout_s,
        )

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            self.embed_endpoint, json=[text], timeout=self.embed_timeout_s
        )
        response.raise_for_status()
        return response.json()[0]

    def hybrid_search(
        self,
        query_text: str,
        vector: list[float],
        top_k: int | None = None,
    ) -> dict:
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
                            "knn": {
                                "embedding": {
                                    "vector": vector,
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

    def knn_search(self, vector: list[float], size: int | None = None) -> dict:
        """Same knn sub-query as the hybrid, run standalone so its raw kNN
        score is visible."""
        body = {
            "size": size or self.knn_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector,
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
        """Embed once and run all three queries — the exact shape the REPL
        needs to print combined + per-component scores side by side."""
        vector = self.embed(query_text)
        return {
            "vector": vector,
            "hybrid": self.hybrid_search(query_text, vector),
            "bm25": self.bm25_search(query_text),
            "knn": self.knn_search(vector),
        }
