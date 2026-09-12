"""Integration tests for the per-field `doc_values` schema option (Issue #1047).

The Python binding does not currently expose field-sorted search or
faceting, so these tests cannot observe the DocValues column itself
disappearing on disk. What they cover is the acceptance criterion that
matters at this layer: `doc_values=False` must not lose stored-field
retrieval, and the schema builder must accept the flag on every field
type that carries it without raising.
"""

import laurus


def test_doc_values_false_keeps_stored_field_retrievable():
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_text_field("internal_note", doc_values=False)
    idx = laurus.Index(schema=schema)
    idx.put_document(
        "doc1",
        {"title": "Hello", "internal_note": "opted out of doc values"},
    )
    idx.commit()

    docs = idx.get_documents("doc1")
    assert len(docs) == 1
    assert docs[0]["title"] == "Hello"
    assert docs[0]["internal_note"] == "opted out of doc values"


def test_doc_values_false_field_is_still_searchable():
    """`doc_values` only controls the column-oriented store; it must not
    affect whether the field is indexed for search."""
    schema = laurus.Schema()
    schema.add_text_field("title", doc_values=False)
    idx = laurus.Index(schema=schema)
    idx.put_document("doc1", {"title": "Rust programming"})
    idx.commit()

    results = idx.search(laurus.TermQuery("title", "rust"), limit=5)
    ids = [hit.id for hit in results]
    assert "doc1" in ids


def test_add_field_methods_accept_doc_values_on_every_carrying_type():
    """Every field option except Bytes carries `doc_values`; this must not
    raise for any of them, in either direction."""
    schema = laurus.Schema()
    schema.add_text_field("t", doc_values=False)
    schema.add_integer_field("i", doc_values=False)
    schema.add_float_field("f", doc_values=False)
    schema.add_boolean_field("b", doc_values=False)
    schema.add_datetime_field("d", doc_values=False)
    schema.add_geo_field("g", doc_values=False)
    schema.add_geo3d_field("g3", doc_values=False)
    # And the default (True) must still be accepted explicitly too.
    schema.add_text_field("t2", doc_values=True)

    idx = laurus.Index(schema=schema)
    assert idx is not None
