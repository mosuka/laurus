"""Vector Search Example — semantic similarity search with embeddings.

Demonstrates vector search using laurus's built-in CandleBert embedder:
- Basic vector search (semantic similarity)
- Filtered vector search (with lexical filters)

The embedder is registered in the schema and laurus automatically converts
text to vectors at index and query time — no external embedding library needed.

Run with:
    maturin develop --features embeddings-candle
    python examples/vector_search.py
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
    print("=== Laurus Vector Search Example ===\n")
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
    # [A] Basic Vector Search
    # =====================================================================
    print("=" * 60)
    print("[A] Basic Vector Search: 'database ORM queries'")
    print("=" * 60)
    _print_results(
        index.search(laurus.VectorTextQuery("text_vec", "database ORM queries"), limit=3).results
    )

    # =====================================================================
    # [B] Filtered Vector Search — category filter
    # =====================================================================
    print("\n" + "=" * 60)
    print("[B] Filtered Vector Search: 'database ORM queries' + category='testing'")
    print("=" * 60)
    request = laurus.SearchRequest(
        vector_query=laurus.VectorTextQuery("text_vec", "database ORM queries"),
        filter_query=laurus.TermQuery("category", "testing"),
        limit=3,
    )
    _print_results(index.search(request).results)

    # =====================================================================
    # [C] Filtered Vector Search — numeric range filter (page = 1) + query
    # =====================================================================
    print("\n" + "=" * 60)
    print("[C] Filtered Vector Search: 'web server HTTP' + page=1")
    print("=" * 60)
    request = laurus.SearchRequest(
        vector_query=laurus.VectorTextQuery("text_vec", "web server HTTP"),
        filter_query=laurus.NumericRangeQuery("page", min=1, max=1),
        limit=3,
    )
    _print_results(index.search(request).results)

    print("\nVector search example completed!")


def _print_results(results: list) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        doc = r.document or {}
        print(f"  id={r.id!r:8s}  score={r.score:.4f}  text={doc.get('text', '')!r:.60s}")


if __name__ == "__main__":
    main()
