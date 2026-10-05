# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "colbert-ai==0.2.22",
#     "huggingface-hub",
#     "numpy",
#     "torch",
#     # colbert-ai 0.2.22 fails to load checkpoints with transformers 5.
#     "transformers>=4.45,<5",
# ]
# ///
"""Generate ColBERT reference vectors for laurus' parity test (Issue #1349).

Encodes a fixed set of queries and documents with colbert-ai's
`Checkpoint` (the reference implementation) on the CPU in fp32 and writes
a JSON fixture that `laurus/tests/colbert_parity_test.rs` compares
`CandleColbertEmbedder` against.

Usage:

    CUDA_VISIBLE_DEVICES="" uv run scripts/colbert_reference.py \
        colbert-ir/colbertv2.0 laurus/tests/fixtures/colbert/colbertv2.0.json

The model is downloaded at its current commit, which is recorded in the
fixture as `revision`; the Rust test loads the same commit.
"""

import base64
import json
import os
import sys
from importlib.metadata import version

# colbert-ai calls .half() on document vectors when a GPU is visible.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from colbert.infra import ColBERTConfig  # noqa: E402
from colbert.modeling.checkpoint import Checkpoint  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

QUERIES = [
    "what is late interaction retrieval?",
    "rust ownership and borrowing",
    "How do I reset my password?!",
    "café crème brûlée",
    "東京の天気",
    "lifetimes",
    # Longer than query_maxlen (32): truncated.
    "explain in detail how a search engine builds an inverted index, scores documents "
    "with bm25, combines lexical and dense vector retrieval, and then reranks the top "
    "candidates with a late interaction model",
]

DOCUMENTS = [
    "ColBERT encodes queries and documents into token-level embeddings and scores "
    "them with MaxSim.",
    "Rust's ownership model guarantees memory safety without a garbage collector.",
    "To reset your password, open Settings > Account, then click 'Forgot password?'.",
    "Wow!!! (Really?) -- yes: it's [that] good; {trust} me... #1 @home ~ 100% & more.",
    "Une crème brûlée au café, s'il vous plaît.",
    "東京は今日晴れです。",
    "lifetimes",
    # Longer than doc_maxlen: truncated.
    " ".join(
        f"Sentence {i} talks about vector search, inverted indexes, and ranking."
        for i in range(60)
    ),
]

# Documents with more kept rows than this store only the first and last
# EDGE_ROWS vectors, to keep the fixture small.
LONG_ROWS = 40
EDGE_ROWS = 8


def b64(rows: np.ndarray) -> str:
    return base64.b64encode(rows.astype("<f4").tobytes()).decode("ascii")


def main() -> None:
    model, out_path = sys.argv[1], sys.argv[2]
    revision = HfApi().model_info(model).sha
    path = snapshot_download(model, revision=revision)

    torch.manual_seed(0)
    ckpt = Checkpoint(path, colbert_config=ColBERTConfig())
    ckpt.eval()
    config = ckpt.colbert_config

    queries = []
    query_vectors = []
    for text in QUERIES:
        ids, mask = ckpt.query_tokenizer.tensorize([text])
        with torch.no_grad():
            q = ckpt.queryFromText([text])[0].float().cpu().numpy()
        query_vectors.append(q)
        queries.append(
            {
                "text": text,
                "input_ids": ids[0].tolist(),
                "attention_mask": mask[0].tolist(),
                "rows": int(q.shape[0]),
                "vectors": b64(q),
            }
        )

    documents = []
    doc_vectors = []
    for text in DOCUMENTS:
        ids, mask = ckpt.doc_tokenizer.tensorize([text])
        with torch.no_grad():
            d, keep = ckpt.doc(ids, mask, keep_dims="return_mask")
        keep = keep[0].squeeze(-1).cpu().numpy().astype(bool)
        d = d[0].float().cpu().numpy()[keep]
        doc_vectors.append(d)
        stored = list(range(d.shape[0]))
        if d.shape[0] > LONG_ROWS:
            stored = stored[:EDGE_ROWS] + stored[-EDGE_ROWS:]
        documents.append(
            {
                "text": text,
                "input_ids": ids[0].tolist(),
                "kept": np.flatnonzero(keep).tolist(),
                "rows": int(d.shape[0]),
                "stored_rows": stored,
                "vectors": b64(d[stored]),
            }
        )

    scores = [
        [float((q @ d.T).max(axis=1).sum()) for d in doc_vectors] for q in query_vectors
    ]

    fixture = {
        "model": model,
        "revision": revision,
        "generator": {
            name: version(name) for name in ("colbert-ai", "torch", "transformers")
        },
        "config": {
            "query_maxlen": config.query_maxlen,
            "doc_maxlen": config.doc_maxlen,
            "dim": int(query_vectors[0].shape[1]),
        },
        "queries": queries,
        "documents": documents,
        "scores": scores,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
