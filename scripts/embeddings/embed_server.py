import threading
import time
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()

# embed_text is a sync `def`, so FastAPI/Starlette runs it in a thread pool —
# concurrent requests (from bulk_load.py's thread_count + OpenSearch's own
# ingest concurrency) can otherwise call model.encode() on the GPU at the
# same time from different threads, multiplying memory use. Serialize GPU
# access so only one encode() runs at a time.
gpu_lock = threading.Lock()

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-4B",
    device="cuda",
    model_kwargs={"torch_dtype": torch.float16},
    truncate_dim=1024,  # Matryoshka truncation: 2560 -> 1024
)

# Cap sequence length so a single very long Wikipedia article (or a batch of
# them) can't blow up GPU memory. Qwen3-Embedding supports up to 32k tokens,
# but full context isn't needed for retrieval quality here, and without a
# cap, batch_size 64 x a long article is what OOM'd the 16GB card.
model.max_seq_length = 2048


# NOTE: this matches OpenSearch's connector.pre_process.default.embedding /
# connector.post_process.default.embedding contract:
#   request body IS a bare JSON array of strings (no wrapper key)
#   response  IS a bare 2D JSON array of floats (no wrapper key)
@app.post("/embed")
def embed_text(texts: List[str]):
    start_time = time.time()

    # NOTE: Qwen3-Embedding is asymmetric — for best retrieval quality,
    # queries should be encoded with prompt_name="query", documents plain.
    # This endpoint encodes everything plain (matches your OpenSearch
    # connector calling one endpoint for both ingest + search). If you
    # want query-time boosts later, add a second route (e.g. /embed_query)
    # that passes prompt_name="query" and point search pipelines at it.
    # Don't trust model.max_seq_length alone to truncate — some model wrappers
    # don't wire it through. Truncate explicitly at the tokenizer level so we
    # KNOW long articles can't slip through untruncated.
    lengths = [len(model.tokenizer(t, truncation=False)["input_ids"]) for t in texts]
    print(f"token lengths: min={min(lengths)} max={max(lengths)}", flush=True)

    with gpu_lock:
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                # Explicit tokenizer-level truncation — belt-and-suspenders
                # alongside model.max_seq_length in case that alone isn't
                # honored by this model's wrapper.
                tokenizer_kwargs={"truncation": True, "max_length": 2048},
            )
        # Release cached-but-unused CUDA memory back to the allocator each
        # request. Without this, memory fragments over the life of a
        # long-running process handling millions of varying-size requests,
        # showing up as "allocated" climbing even though nothing is actually
        # leaking — this is what the 14GB-for-an-8GB-model symptom was.
        torch.cuda.empty_cache()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    print("--- %s seconds ---" % (time.time() - start_time))

    return embeddings.tolist()
