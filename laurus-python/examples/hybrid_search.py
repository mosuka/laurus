"""Hybrid Search Example — combining lexical and vector search.

Demonstrates:
- Lexical-only search (for comparison)
- Vector-only search (for comparison)
- Hybrid search with RRF fusion
- Hybrid search with WeightedSum fusion
- Hybrid search with a filter query

The embedder is registered in the schema and laurus automatically converts
text to vectors at index and query time — no external embedding library needed.

Run with:
    maturin develop --features embeddings-candle
    python examples/hybrid_search.py
"""

from __future__ import annotations

import laurus

# ---------------------------------------------------------------------------
# Embedder configuration
# ---------------------------------------------------------------------------

_EMBEDDER_NAME = "bert"
_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DIM = 384  # dimension for all-MiniLM-L6-v2

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
    print("=== Laurus Hybrid Search Example ===\n")
    print(f"Embedder: {_EMBEDDER_MODEL} (dim={_DIM})\n")

    # ── Schema ─────────────────────────────────────────────────────────────
    schema = laurus.Schema()
    schema.add_embedder(_EMBEDDER_NAME, {"type": "candle_bert", "model": _EMBEDDER_MODEL})
    schema.add_text_field("title")
    schema.add_text_field("text")
    schema.add_text_field("category")
    schema.add_integer_field("page")
    schema.add_flat_field("text_vec", dimension=_DIM, distance="cosine", embedder=_EMBEDDER_NAME)
    schema.set_default_fields(["text"])

    index = laurus.Index(schema=schema)

    # ── Index ──────────────────────────────────────────────────────────────
    # Passing raw text to text_vec lets laurus automatically vectorize it via the built-in embedder.
    print("--- Indexing chunked documents ---\n")
    for doc_id, title, text, page, category in CHUNKS:
        index.add_document(
            doc_id,
            {
                "title": title,
                "text": text,
                "category": category,
                "page": page,
                "text_vec": text,
            },
        )
    index.commit()
    print(f"Indexed {len(CHUNKS)} chunks.\n")

    # =====================================================================
    # [A] Lexical-only search (baseline)
    # =====================================================================
    print("=" * 60)
    print("[A] Lexical-only: term 'fixtures' in text")
    print("=" * 60)
    _print_results(index.search(laurus.TermQuery("text", "fixtures"), limit=3).results)

    # =====================================================================
    # [B] Vector-only search (baseline)
    # =====================================================================
    print("\n" + "=" * 60)
    print("[B] Vector-only: 'database query builder'")
    print("=" * 60)
    _print_results(
        index.search(laurus.VectorTextQuery("text_vec", "database query builder"), limit=3).results
    )

    # =====================================================================
    # [C] Hybrid search — RRF Fusion (default when both queries present)
    # =====================================================================
    print("\n" + "=" * 60)
    print("[C] Hybrid (RRF k=60): vector='template rendering' + lexical='jinja2'")
    print("=" * 60)
    request = laurus.SearchRequest(
        lexical_query=laurus.TermQuery("text", "jinja2"),
        vector_query=laurus.VectorTextQuery("text_vec", "template rendering"),
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
        vector_query=laurus.VectorTextQuery("text_vec", "data analysis"),
        fusion=laurus.WeightedSum(lexical_weight=0.3, vector_weight=0.7),
        limit=3,
    )
    _print_results(index.search(request).results)

    # =====================================================================
    # [E] Hybrid search with filter query
    # =====================================================================
    print("\n" + "=" * 60)
    print("[E] Hybrid + filter: vector='testing automation' + lexical='parametrize' + category='testing'")
    print("=" * 60)
    request = laurus.SearchRequest(
        lexical_query=laurus.TermQuery("text", "parametrize"),
        vector_query=laurus.VectorTextQuery("text_vec", "testing automation"),
        filter_query=laurus.TermQuery("category", "testing"),
        fusion=laurus.RRF(k=60.0),
        limit=3,
    )
    _print_results(index.search(request).results)

    # =====================================================================
    # [F] Hybrid search via SearchRequest with DSL string query
    # =====================================================================
    print("\n" + "=" * 60)
    print("[F] DSL query string: 'text:middleware'")
    print("=" * 60)
    _print_results(index.search("text:middleware", limit=3).results)

    print("\nHybrid search example completed!")


def _print_results(results: list) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        doc = r.document or {}
        text = doc.get("text", "")
        print(f"  id={r.id!r:8s}  score={r.score:.4f}  text={text!r:.60s}")


if __name__ == "__main__":
    main()
