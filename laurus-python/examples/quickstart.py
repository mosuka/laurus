"""Quickstart — Your first full-text search with Laurus.

This minimal example shows how to:
1. Create an in-memory search engine
2. Define a schema with text fields
3. Index documents
4. Search with a simple term query and a DSL string

Run with:
    maturin develop
    python examples/quickstart.py
"""

import laurus


def main() -> None:
    print("=== Laurus Quickstart ===\n")

    # 1. Define a schema
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_text_field("body")
    schema.set_default_fields(["title", "body"])

    # 2. Create an in-memory index
    index = laurus.Index(schema=schema)

    # 3. Index documents
    index.add_document(
        "doc1",
        {
            "title": "Django Web Framework",
            "body": "Django is a high-level Python web framework with batteries-included architecture.",
        },
    )
    index.add_document(
        "doc2",
        {
            "title": "NumPy Scientific Computing",
            "body": "NumPy provides fast numerical arrays and mathematical operations for Python.",
        },
    )
    index.add_document(
        "doc3",
        {
            "title": "pytest Testing Framework",
            "body": "pytest is a powerful testing framework for Python with fixtures and plugins.",
        },
    )
    index.commit()
    print("Indexed 3 documents.\n")

    # 4. Search with a TermQuery object
    print("[Search] TermQuery — 'django' in body:")
    results = index.search(laurus.TermQuery("body", "django"), limit=5).results
    _print_results(results)

    # 5. Search with a DSL string
    print("\n[Search] DSL — 'python':")
    results = index.search("python", limit=5).results
    _print_results(results)

    # 6. put_document replaces an existing document
    print("\n[Update] Replacing doc2 with put_document:")
    index.put_document(
        "doc2",
        {
            "title": "NumPy Scientific Computing",
            "body": "NumPy enables high-performance numerical computing with ndarray and broadcasting.",
        },
    )
    index.commit()

    print("[Search] DSL — 'broadcasting' (should find updated doc2):")
    results = index.search("broadcasting", limit=5).results
    _print_results(results)

    print("\nQuickstart complete!")


def _print_results(results: list) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        title = (r.document or {}).get("title", "")
        print(f"  id={r.id!r:8s}  score={r.score:.4f}  title={title!r}")


if __name__ == "__main__":
    main()
