"""Multimodal Search Example — searching across text and image bytes fields.

This example demonstrates how to store raw bytes (e.g. image data) in a Laurus
index and perform multimodal search using pre-computed CLIP embeddings produced
on the Python side.

The Rust `multimodal_search.rs` example uses the built-in `CandleClipEmbedder`.
In Python you can achieve the same result by:
1. Encoding images/text with a CLIP model (e.g. via `transformers` or `open_clip`).
2. Storing the raw bytes in a `bytes` field and the embedding vector in a flat field.
3. Querying with `VectorQuery` using a pre-computed embedding.

Requirements (optional — see fallback below):
    pip install torch transformers Pillow

Run with:
    maturin develop
    python examples/multimodal_search.py
"""

from __future__ import annotations

import io
import math
import random

import laurus

# ---------------------------------------------------------------------------
# CLIP embedding helper
# ---------------------------------------------------------------------------

try:
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from transformers import CLIPModel, CLIPProcessor  # type: ignore

    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _DIM = 512
    _HAS_CLIP = True

    def embed_text(text: str) -> list[float]:
        inputs = _clip_processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            features = _clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    def embed_image(image_bytes: bytes) -> list[float]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = _clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = _clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

except ImportError:
    _DIM = 32
    _HAS_CLIP = False

    def _rand_unit(seed: int) -> list[float]:  # type: ignore[misc]
        rng = random.Random(seed)
        raw = [rng.gauss(0, 1) for _ in range(_DIM)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_text(text: str) -> list[float]:  # type: ignore[misc]
        return _rand_unit(hash(text) & 0xFFFFFFFF)

    def embed_image(image_bytes: bytes) -> list[float]:  # type: ignore[misc]
        return _rand_unit(hash(image_bytes[:128]) & 0xFFFFFFFF)

    print(
        "[NOTE] torch / transformers / Pillow not found — using random fallback vectors.\n"
        "       Semantic similarity will NOT be meaningful.\n"
        "       Install with: pip install torch transformers Pillow\n"
    )


# ---------------------------------------------------------------------------
# Fake image bytes for demo (1x1 white pixel PNG)
# ---------------------------------------------------------------------------

_WHITE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd4n\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _fake_image(label: str) -> bytes:
    """Return placeholder image bytes (real app would read from disk)."""
    return _WHITE_PNG + label.encode()  # make each "image" unique for hashing


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGES = [
    ("img_cat1", "cat_sleeping.jpg", "image"),
    ("img_cat2", "cat_playing.jpg", "image"),
    ("img_dog1", "dog_running.jpg", "image"),
    ("img_dog2", "two_dogs.jpg", "image"),
]

TEXTS = [
    ("txt1", "A cute kitten looking at the camera", "text"),
    ("txt2", "A loyal dog standing in the grass", "text"),
    ("txt3", "Two dogs playing together", "text"),
    ("txt4", "A landscape with mountains and a lake", "text"),
]


def main() -> None:
    print("=== Laurus Multimodal Search Example ===\n")
    if _HAS_CLIP:
        print("Using CLIP (openai/clip-vit-base-patch32) for embeddings.\n")
    else:
        print("Using random fallback vectors (results not semantically meaningful).\n")

    # ── Schema ─────────────────────────────────────────────────────────────
    schema = laurus.Schema()
    schema.add_bytes_field("content")        # raw image bytes or None
    schema.add_text_field("filename")
    schema.add_text_field("type")
    schema.add_text_field("description")
    schema.add_flat_field("content_vec", dimension=_DIM, distance="cosine")

    index = laurus.Index(schema=schema)

    # ── Index images ───────────────────────────────────────────────────────
    print("--- Indexing images ---")
    for doc_id, filename, media_type in IMAGES:
        raw_bytes = _fake_image(filename)   # replace with open(path, "rb").read()
        vec = embed_image(raw_bytes)
        index.add_document(
            doc_id,
            {
                "content": raw_bytes,
                "filename": filename,
                "type": media_type,
                "description": "",
                "content_vec": vec,
            },
        )
        print(f"  Indexed image: {filename}")

    # ── Index text descriptions ────────────────────────────────────────────
    print("\n--- Indexing text descriptions ---")
    for doc_id, text, media_type in TEXTS:
        vec = embed_text(text)
        index.add_document(
            doc_id,
            {
                "content": None,
                "filename": "",
                "type": media_type,
                "description": text,
                "content_vec": vec,
            },
        )
        print(f"  Indexed text: {text!r:.50s}")

    index.commit()
    print()

    # =====================================================================
    # [A] Text-to-Image: find images matching a text query
    # =====================================================================
    print("=" * 60)
    print("[A] Text-to-Image: query='a photo of a cat'")
    print("=" * 60)
    query_vec = embed_text("a photo of a cat")
    _print_results(index.search(laurus.VectorQuery("content_vec", query_vec), limit=3).results)

    # =====================================================================
    # [B] Text-to-Text: find text descriptions
    # =====================================================================
    print("\n" + "=" * 60)
    print("[B] Text-to-Text: query='loyal dog', filter type='text'")
    print("=" * 60)
    request = laurus.SearchRequest(
        vector_query=laurus.VectorQuery("content_vec", embed_text("loyal dog")),
        filter_query=laurus.TermQuery("type", "text"),
        limit=3,
    )
    _print_results(index.search(request).results)

    # =====================================================================
    # [C] Image-to-Anything: find documents similar to a given image
    # =====================================================================
    print("\n" + "=" * 60)
    print("[C] Image-to-Anything: query from 'cat_sleeping.jpg'")
    print("=" * 60)
    query_img_bytes = _fake_image("cat_sleeping.jpg")
    query_vec = embed_image(query_img_bytes)
    _print_results(index.search(laurus.VectorQuery("content_vec", query_vec), limit=3).results)

    print("\nMultimodal search example completed!")


def _print_results(results: list) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        doc = r.document or {}
        label = doc.get("filename") or doc.get("description", "")
        media_type = doc.get("type", "?")
        print(f"  id={r.id!r:8s}  score={r.score:.4f}  [{media_type}] {label!r:.55s}")


if __name__ == "__main__":
    main()
