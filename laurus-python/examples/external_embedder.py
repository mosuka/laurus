"""External Embedder Example — vector and hybrid search with a user-managed embedder.

Demonstrates vector and hybrid search where embeddings are produced outside
laurus and passed as pre-computed vectors via ``VectorQuery``.

This approach is useful when you want to:
- Use any embedding library (sentence-transformers, HuggingFace, etc.)
- Control the embedding model independently of the index schema
- Reuse embeddings across multiple indexes

For an alternative that lets laurus handle embeddings automatically, see
``vector_search.py`` and ``hybrid_search.py`` which use the built-in
``candle_bert`` embedder via ``schema.add_embedder()``.

Run with:
    pip install sentence-transformers   # optional but recommended
    maturin develop
    python examples/external_embedder.py
"""

from __future__ import annotations

import math
import random

import laurus

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _DIM = 384

    def embed(text: str) -> list[float]:
        return _model.encode(text, normalize_embeddings=True).tolist()

    print("Using sentence-transformers/all-MiniLM-L6-v2 for embeddings.\n")

except ImportError:
    # Fallback: deterministic pseudo-embeddings for demo purposes only.
    # Real similarity is not meaningful with these vectors.
    _DIM = 64

    def embed(text: str) -> list[float]:  # type: ignore[misc]
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        raw = [rng.gauss(0, 1) for _ in range(_DIM)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    print(
        "[NOTE] sentence-transformers not found — using random fallback vectors.\n"
        "       Results will NOT reflect semantic similarity.\n"
        "       Install with: pip install sentence-transformers\n"
    )

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CHUNKS = [
    ("django_guide", "Django Web Development", "Django follows the model-template-view architecture pattern for clean separation of concerns.", 1, "framework"),
    ("django_guide", "Django Web Development", "Django ORM maps Python classes to database tables with migrations for schema management.", 2, "framework"),
    ("django_guide", "Django Web Development", "Django middleware processes requests and responses through a chain of pluggable components.", 3, "framework"),
    ("flask_guide", "Flask Microservices", "Flask provides lightweight routing and Jinja2 templating for building APIs and web apps.", 1, "framework"),
    ("flask_guide", "Flask Microservices", "Flask extensions like Flask-SQLAlchemy and Flask-Migrate add database support to Flask projects.", 2, "framework"),
    ("numpy_docs", "NumPy Fundamentals", "NumPy arrays provide vectorized operations that are much faster than Python loops for numerical computing.", 1, "scientific"),
    ("numpy_docs", "NumPy Fundamentals", "Broadcasting in NumPy allows arithmetic operations on arrays of different shapes without copying data.", 2, "scientific"),
    ("pytest_book", "Testing with pytest", "pytest fixtures provide reusable setup and teardown logic for test functions with dependency injection.", 1, "testing"),
    ("pytest_book", "Testing with pytest", "pytest parametrize decorator runs the same test with different input datasets automatically.", 2, "testing"),
]


def main() -> None:
    print("=== Laurus External Embedder Example ===\n")
    print(f"Embedding model dimension: {_DIM}\n")

    # ── Schema ─────────────────────────────────────────────────────────────
    # No embedder is registered in the schema; vectors are provided by the
    # caller at index and query time via embed().
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_text_field("text")
    # keyword analyzer: treats hyphenated values like "type-system" as a single token
    schema.add_text_field("category", analyzer="keyword")
    schema.add_integer_field("page")
    schema.add_flat_field("text_vec", dimension=_DIM, distance="cosine")
    schema.set_default_fields(["text"])

    index = laurus.Index(schema=schema)

    # ── Index ──────────────────────────────────────────────────────────────
    # Embeddings are computed here in Python and stored as float vectors.
    print("--- Indexing chunked documents ---\n")
    for doc_id, title, text, page, category in CHUNKS:
        index.add_document(
            doc_id,
            {
                "title": title,
                "text": text,
                "category": category,
                "page": page,
                "text_vec": embed(text),
            },
        )
    index.commit()
    print(f"Indexed {len(CHUNKS)} chunks.\n")

    # =====================================================================
    # [A] Basic Vector Search
    # =====================================================================
    print("=" * 60)
    print("[A] Vector-only: 'database ORM queries'")
    print("=" * 60)
    _print_results(
        index.search(laurus.VectorQuery("text_vec", embed("database ORM queries")), limit=3).results
    )

    # =====================================================================
    # [B] Filtered Vector Search — category filter
    # =====================================================================
    print("\n" + "=" * 60)
    print("[B] Filtered vector: 'database ORM queries' + category='testing'")
    print("=" * 60)
    request = laurus.SearchRequest(
        vector_query=laurus.VectorQuery("text_vec", embed("database ORM queries")),
        filter_query=laurus.TermQuery("category", "testing"),
        limit=3,
    )
    _print_results(index.search(request).results)

    # =====================================================================
    # [C] Hybrid search — RRF Fusion
    # =====================================================================
    print("\n" + "=" * 60)
    print("[C] Hybrid (RRF k=60): vector='template rendering' + lexical='jinja2'")
    print("=" * 60)
    request = laurus.SearchRequest(
        lexical_query=laurus.TermQuery("text", "jinja2"),
        vector_query=laurus.VectorQuery("text_vec", embed("template rendering")),
        fusion=laurus.RRF(k=60.0),
        limit=3,
    )
    _print_results(index.search(request).results)

    # =====================================================================
    # [D] Hybrid search — WeightedSum Fusion
    # =====================================================================
    print("\n" + "=" * 60)
    print("[D] Hybrid (WeightedSum 0.3/0.7): vector='data analysis' + lexical='arrays'")
    print("=" * 60)
    request = laurus.SearchRequest(
        lexical_query=laurus.TermQuery("text", "arrays"),
        vector_query=laurus.VectorQuery("text_vec", embed("data analysis")),
        fusion=laurus.WeightedSum(lexical_weight=0.3, vector_weight=0.7),
        limit=3,
    )
    _print_results(index.search(request).results)

    # =====================================================================
    # [E] Hybrid search with filter
    # =====================================================================
    print("\n" + "=" * 60)
    print("[E] Hybrid + filter: vector='testing automation' + lexical='parametrize' + category='testing'")
    print("=" * 60)
    request = laurus.SearchRequest(
        lexical_query=laurus.TermQuery("text", "parametrize"),
        vector_query=laurus.VectorQuery("text_vec", embed("testing automation")),
        filter_query=laurus.TermQuery("category", "testing"),
        fusion=laurus.RRF(k=60.0),
        limit=3,
    )
    _print_results(index.search(request).results)

    print("\nExternal embedder example completed!")


def _print_results(results: list) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        doc = r.document or {}
        print(f"  id={r.id!r:8s}  score={r.score:.4f}  text={doc.get('text', '')!r:.60s}")


if __name__ == "__main__":
    main()
