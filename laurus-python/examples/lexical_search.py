"""Lexical Search Example — all query types.

Demonstrates every lexical query type Laurus supports:

1. TermQuery         — exact single-term matching
2. PhraseQuery       — exact word sequence matching
3. FuzzyQuery        — approximate matching (typo tolerance)
4. WildcardQuery     — pattern matching with * and ?
5. NumericRangeQuery — numeric range filtering (int and float)
6. GeoDistanceQuery / GeoBoundingBoxQuery — geographic radius / bounding box
7. BooleanQuery      — AND / OR / NOT combinations
8. SpanQuery         — positional / proximity search

Run with:
    maturin develop
    python examples/lexical_search.py
"""

import laurus


def main() -> None:
    print("=== Laurus Lexical Search Example ===\n")

    # ── Setup ──────────────────────────────────────────────────────────────
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_text_field("body")
    schema.add_text_field("category", analyzer="keyword")
    schema.add_text_field("filename", analyzer="keyword")
    schema.add_boolean_field("in_print")
    schema.add_float_field("price")
    schema.add_integer_field("year")
    schema.add_geo_field("location")
    schema.set_default_fields(["body"])

    index = laurus.Index(schema=schema)

    # ── Index documents ────────────────────────────────────────────────────
    docs = [
        (
            "django",
            {
                "title": "Django Web Framework",
                "body": "Django is a high-level Python web framework with batteries-included architecture and ORM",
                "category": "framework",
                "filename": "django_guide.pdf",
                "in_print": True,
                "price": 4.8,
                "year": 2005,
                "location": (52.3676, 4.9041),  # Amsterdam
            },
        ),
        (
            "flask",
            {
                "title": "Flask Micro Framework",
                "body": "Flask is a lightweight Python micro framework for building web applications and APIs",
                "category": "framework",
                "filename": "flask_docs.epub",
                "in_print": True,
                "price": 4.5,
                "year": 2010,
                "location": (47.3769, 8.5417),  # Zurich
            },
        ),
        (
            "numpy",
            {
                "title": "NumPy Scientific Computing",
                "body": "NumPy provides fast numerical arrays and mathematical operations for scientific computing",
                "category": "scientific",
                "filename": "numpy_manual.pdf",
                "in_print": True,
                "price": 4.9,
                "year": 2006,
                "location": (37.7749, -122.4194),  # San Francisco
            },
        ),
        (
            "pandas",
            {
                "title": "Pandas Data Analysis",
                "body": "Pandas provides data structures and analysis tools for handling tabular data in Python",
                "category": "scientific",
                "filename": "pandas_guide.docx",
                "in_print": True,
                "price": 4.7,
                "year": 2008,
                "location": (37.4419, -122.1430),  # Palo Alto
            },
        ),
        (
            "pytest",
            {
                "title": "pytest Testing Framework",
                "body": "pytest is a powerful testing framework for Python with fixtures and plugins",
                "category": "testing",
                "filename": "pytest_docs.pdf",
                "in_print": False,
                "price": 4.6,
                "year": 2004,
                "location": (52.5200, 13.4050),  # Berlin
            },
        ),
        (
            "gunicorn",
            {
                "title": "Gunicorn WSGI Server",
                "body": "The quick green snake slithered through the virtual environment garden",
                "category": "fiction",
                "filename": "gunicorn_story.txt",
                "in_print": False,
                "price": 3.5,
                "year": 2023,
                "location": (34.0522, -118.2437),  # Los Angeles
            },
        ),
    ]

    print(f"  Indexing {len(docs)} documents...")
    for doc_id, doc in docs:
        index.add_document(doc_id, doc)
    index.commit()
    print("  Done.\n")

    # =====================================================================
    # PART 1: TermQuery — exact single-term matching
    # =====================================================================
    print("=" * 60)
    print("PART 1: TermQuery")
    print("=" * 60)

    print("\n[1a] Search for 'django' in body:")
    _print_results(index.search(laurus.TermQuery("body", "django"), limit=5).results)

    print("\n[1b] Search for 'framework' in category (exact):")
    _print_results(index.search(laurus.TermQuery("category", "framework"), limit=5).results)

    print("\n[1c] Search for in_print=true (boolean field):")
    _print_results(index.search(laurus.TermQuery("in_print", "true"), limit=5).results)

    print("\n[1d] DSL: 'body:django':")
    _print_results(index.search("body:django", limit=5).results)

    # =====================================================================
    # PART 2: PhraseQuery — exact word sequence
    # =====================================================================
    print("\n" + "=" * 60)
    print("PART 2: PhraseQuery")
    print("=" * 60)

    print("\n[2a] Phrase 'scientific computing' in body:")
    _print_results(
        index.search(laurus.PhraseQuery("body", ["scientific", "computing"]), limit=5).results
    )

    print("\n[2b] Phrase 'web framework' in body:")
    _print_results(
        index.search(
            laurus.PhraseQuery("body", ["web", "framework"]),
            limit=5,
        ).results
    )

    print("\n[2c] DSL: 'body:\"scientific computing\"':")
    _print_results(index.search('body:"scientific computing"', limit=5).results)

    # =====================================================================
    # PART 3: FuzzyQuery — approximate matching (typo tolerance)
    # =====================================================================
    print("\n" + "=" * 60)
    print("PART 3: FuzzyQuery")
    print("=" * 60)

    print("\n[3a] Fuzzy 'framwork' (missing 'e', max_edits=2):")
    _print_results(
        index.search(laurus.FuzzyQuery("body", "framwork", max_edits=2), limit=5).results
    )

    print("\n[3b] Fuzzy 'numppy' (extra 'p', max_edits=1):")
    _print_results(
        index.search(laurus.FuzzyQuery("body", "numppy", max_edits=1), limit=5).results
    )

    print("\n[3c] DSL: 'framwork~2':")
    _print_results(index.search("framwork~2", limit=5).results)

    # =====================================================================
    # PART 4: WildcardQuery — pattern matching with * and ?
    # =====================================================================
    print("\n" + "=" * 60)
    print("PART 4: WildcardQuery")
    print("=" * 60)

    print("\n[4a] Wildcard '*.pdf' in filename:")
    _print_results(index.search(laurus.WildcardQuery("filename", "*.pdf"), limit=5).results)

    print("\n[4b] Wildcard 'py*' in body:")
    _print_results(index.search(laurus.WildcardQuery("body", "py*"), limit=5).results)

    print("\n[4c] DSL: 'body:py*':")
    _print_results(index.search("body:py*", limit=5).results)

    # =====================================================================
    # PART 5: NumericRangeQuery — numeric range filtering
    # =====================================================================
    print("\n" + "=" * 60)
    print("PART 5: NumericRangeQuery")
    print("=" * 60)

    print("\n[5a] Entries with rating 4.5–4.9 (float range):")
    _print_results(
        index.search(
            laurus.NumericRangeQuery("price", min=4.5, max=4.9), limit=5
        ).results
    )

    print("\n[5b] Entries released from 2008 onwards (integer range):")
    _print_results(
        index.search(laurus.NumericRangeQuery("year", min=2008), limit=5).results
    )

    print("\n[5c] DSL: 'price:[4.5 TO 4.9]':")
    _print_results(index.search("price:[4.5 TO 4.9]", limit=5).results)

    # =====================================================================
    # PART 6: GeoDistanceQuery / GeoBoundingBoxQuery — geographic search
    # =====================================================================
    print("\n" + "=" * 60)
    print("PART 6: GeoDistanceQuery / GeoBoundingBoxQuery")
    print("=" * 60)

    print("\n[6a] Within 100 km of San Francisco (37.77, -122.42):")
    _print_results(
        index.search(
            laurus.GeoDistanceQuery.within_radius(
                "location", 37.7749, -122.4194, 100_000.0
            ),
            limit=5,
        ).results
    )

    print("\n[6b] Bounding box — US West Coast (33, -123) to (48, -117):")
    _print_results(
        index.search(
            laurus.GeoBoundingBoxQuery.within_bounding_box(
                "location", 33.0, -123.0, 48.0, -117.0
            ),
            limit=5,
        ).results
    )

    # =====================================================================
    # PART 7: BooleanQuery — AND / OR / NOT combinations
    # =====================================================================
    print("\n" + "=" * 60)
    print("PART 7: BooleanQuery")
    print("=" * 60)

    print("\n[7a] AND: 'python' in body AND category='scientific':")
    bq = laurus.BooleanQuery()
    bq.must(laurus.TermQuery("body", "python"))
    bq.must(laurus.TermQuery("category", "scientific"))
    _print_results(index.search(bq, limit=5).results)

    print("\n[7b] OR: category='framework' OR category='testing':")
    bq = laurus.BooleanQuery()
    bq.should(laurus.TermQuery("category", "framework"))
    bq.should(laurus.TermQuery("category", "testing"))
    _print_results(index.search(bq, limit=5).results)

    print("\n[7c] NOT: 'python' in body, NOT 'django':")
    bq = laurus.BooleanQuery()
    bq.must(laurus.TermQuery("body", "python"))
    bq.must_not(laurus.TermQuery("body", "django"))
    _print_results(index.search(bq, limit=5).results)

    print("\n[7d] DSL: '+body:python -body:django':")
    _print_results(index.search("+body:python -body:django", limit=5).results)

    # =====================================================================
    # PART 8: SpanQuery — positional / proximity search
    # =====================================================================
    print("\n" + "=" * 60)
    print("PART 8: SpanQuery (no DSL equivalent)")
    print("=" * 60)

    print("\n[8a] SpanNear: 'quick' near 'snake' (slop=1, ordered):")
    span_q = laurus.SpanQuery.near("body", ["quick", "snake"], slop=1, ordered=True)
    _print_results(index.search(span_q, limit=5).results)

    print("\n[8b] SpanContaining: 'quick..snake' containing 'green':")
    big = laurus.SpanQuery.near("body", ["quick", "snake"], slop=1, ordered=True)
    little = laurus.SpanQuery.term("body", "green")
    containing = laurus.SpanQuery.containing("body", big, little)
    _print_results(index.search(containing, limit=5).results)

    print("\nLexical search example completed!")


def _print_results(results: list) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        title = (r.document or {}).get("title", "")
        print(f"  id={r.id!r:8s}  score={r.score:.4f}  title={title!r}")


if __name__ == "__main__":
    main()
