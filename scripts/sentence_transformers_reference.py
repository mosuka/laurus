# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "huggingface-hub",
#     "numpy",
#     "sentence-transformers==6.1.0",
#     "torch",
# ]
# ///
"""Generate sentence-transformers reference vectors for laurus (Issue #1340).

Encodes a fixed set of texts with sentence-transformers (the reference
implementation) on the CPU in fp32 and writes a JSON fixture that the
parity tests in `laurus/src/embedding/candle_bert_embedder.rs` compare
`CandleBertEmbedder` against.

Usage:

    CUDA_VISIBLE_DEVICES="" uv run scripts/sentence_transformers_reference.py \
        sentence-transformers/all-MiniLM-L6-v2 \
        laurus/tests/fixtures/sentence_transformers/all-MiniLM-L6-v2.json

The model is loaded at its current commit, which is recorded in the
fixture as `revision`; the Rust test loads the same commit.
"""

import base64
import json
import os
import sys
from importlib.metadata import version

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

# No leading or trailing whitespace: older sentence-transformers releases
# stripped inputs, newer ones do not.
TEXTS = [
    "Rust is a systems programming language.",
    "How do I reset my password?!",
    "Wow!!! (Really?) -- yes: it's [that] good; {trust} me... #1 @home ~ 100% & more.",
    "Une crème brûlée au café, s'il vous plaît.",
    "東京は今日晴れです。",
    "lifetimes",
    # Longer than every model's max_seq_length: truncated.
    " ".join(
        f"Sentence {i} talks about vector search, inverted indexes, and ranking."
        for i in range(40)
    ),
]


def b64(vector: np.ndarray) -> str:
    return base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii")


def main() -> None:
    model_id, out_path = sys.argv[1], sys.argv[2]
    revision = HfApi().model_info(model_id).sha

    torch.manual_seed(0)
    model = SentenceTransformer(model_id, revision=revision, device="cpu")
    model.eval()
    normalize = any(type(module).__name__ == "Normalize" for module in model)

    inputs = []
    for text in TEXTS:
        ids = model.tokenizer(
            text, truncation=True, max_length=model.max_seq_length
        )["input_ids"]
        with torch.no_grad():
            vector = model.encode(
                [text], convert_to_numpy=True, normalize_embeddings=False
            )[0]
        inputs.append({"text": text, "input_ids": list(ids), "vector": b64(vector)})

    fixture = {
        "model": model_id,
        "revision": revision,
        "generator": {
            name: version(name) for name in ("sentence-transformers", "torch", "transformers")
        },
        "max_seq_length": model.max_seq_length,
        "normalize": normalize,
        "dim": int(model.get_embedding_dimension()),
        "inputs": inputs,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
