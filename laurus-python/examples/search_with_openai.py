"""Search with OpenAI Embedder — real vector search using OpenAI's API.

This example mirrors `search_with_openai.rs` but produces embeddings on the
Python side using the `openai` SDK, then passes the resulting vectors to
Laurus as `VectorQuery`.

Prerequisites:
    pip install openai
    export OPENAI_API_KEY=your-api-key-here
    maturin develop
    python examples/search_with_openai.py
"""

from __future__ import annotations

import os
import sys

import laurus

# ---------------------------------------------------------------------------
# OpenAI embedding helper
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    print("ERROR: openai package not installed.")
    print("       Install with: pip install openai")
    sys.exit(1)

_MODEL = "text-embedding-3-small"
_DIM = 1536


def embed(client: "OpenAI", text: str) -> list[float]:
    """Call OpenAI Embeddings API and return the vector."""
    response = client.embeddings.create(input=text, model=_MODEL)
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Dataset (same as the Rust example)
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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("       export OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)

    print("=== Laurus Search with OpenAI Embedder ===\n")

    client = OpenAI(api_key=api_key)
    print(f"OpenAI embedder ready (model={_MODEL}, dim={_DIM})\n")

    # ── Schema ─────────────────────────────────────────────────────────────
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_text_field("text")
    schema.add_text_field("category")
    schema.add_integer_field("page")
    schema.add_flat_field("text_vec", dimension=_DIM, distance="cosine")
    schema.set_default_fields(["text"])

    index = laurus.Index(schema=schema)

    # ── Index ──────────────────────────────────────────────────────────────
    print("--- Indexing chunked documents ---\n")
    for doc_id, title, text, page, category in CHUNKS:
        vec = embed(client, text)
        index.add_document(
            doc_id,
            {
                "title": title,
                "text": text,
                "category": category,
                "page": page,
                "text_vec": vec,
            },
        )
        print(f"  Indexed {doc_id} page {page}: {text[:50]!r}...")
    index.commit()
    print(f"\nIndexed {len(CHUNKS)} chunks.\n")

    # =====================================================================
    # [A] Vector Search
    # =====================================================================
    print("=" * 60)
    print("[A] Vector Search: 'database ORM queries'")
    print("=" * 60)
    _print_results(
        index.search(
            laurus.VectorQuery("text_vec", embed(client, "database ORM queries")), limit=3
        ).results
    )

    # =====================================================================
    # [B] Filtered Vector Search — category filter
    # =====================================================================
    print("\n" + "=" * 60)
    print("[B] Filtered Vector Search: 'database ORM queries' + category='testing'")
    print("=" * 60)
    request = laurus.SearchRequest(
        vector_query=laurus.VectorQuery("text_vec", embed(client, "database ORM queries")),
        filter_query=laurus.TermQuery("category", "testing"),
        limit=3,
    )
    _print_results(index.search(request).results)

    # =====================================================================
    # [C] Filtered Vector Search — numeric range filter
    # =====================================================================
    print("\n" + "=" * 60)
    print("[C] Filtered Vector Search: 'web server HTTP' + page=1")
    print("=" * 60)
    request = laurus.SearchRequest(
        vector_query=laurus.VectorQuery("text_vec", embed(client, "web server HTTP")),
        filter_query=laurus.NumericRangeQuery("page", min=1, max=1),
        limit=3,
    )
    _print_results(index.search(request).results)

    # =====================================================================
    # [D] Lexical Search
    # =====================================================================
    print("\n" + "=" * 60)
    print("[D] Lexical Search: 'fixtures'")
    print("=" * 60)
    _print_results(index.search(laurus.TermQuery("text", "fixtures"), limit=3).results)

    # =====================================================================
    # [E] Hybrid Search (RRF)
    # =====================================================================
    print("\n" + "=" * 60)
    print("[E] Hybrid Search (RRF): vector='template rendering' + lexical='jinja2'")
    print("=" * 60)
    request = laurus.SearchRequest(
        lexical_query=laurus.TermQuery("text", "jinja2"),
        vector_query=laurus.VectorQuery("text_vec", embed(client, "template rendering")),
        fusion=laurus.RRF(k=60.0),
        limit=3,
    )
    _print_results(index.search(request).results)

    print("\nSearch with OpenAI example completed!")


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
