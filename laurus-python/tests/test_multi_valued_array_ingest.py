"""Regression tests for sending Python lists into multi-valued numeric fields
(Issue #1178).

`py_to_data_value` used to turn every `list` into `DataValue::Vector`, which
the core's `coerce_to_integer` / `coerce_to_float` multi-valued branches
reject — so a `multi_valued=True` Integer/Float field could be *read* from
Python but never *written* from it. Lists now arrive as `Int64Array` (all
`int`) or `Float64Array` (otherwise numeric) and the schema-aware coercion in
the core routes them.
"""

import laurus


def _index_with(field_adder, name, **kwargs):
    schema = laurus.Schema()
    schema.add_text_field("title")
    field_adder(schema, name, **kwargs)
    return laurus.Index(schema=schema)


def test_int_list_round_trips_through_multi_valued_integer_field():
    idx = _index_with(laurus.Schema.add_integer_field, "tags", multi_valued=True)
    # Before #1178 this `put_document` raised: the list arrived as a Vector,
    # which a multi-valued integer field does not accept.
    idx.put_document("doc1", {"title": "t", "tags": [1, 2, 3]})
    idx.commit()

    docs = idx.get_documents("doc1")
    assert len(docs) == 1
    assert docs[0]["tags"] == [1, 2, 3]


def test_float_list_round_trips_through_multi_valued_float_field():
    idx = _index_with(laurus.Schema.add_float_field, "scores", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "scores": [1.5, 2.0]})
    idx.commit()

    docs = idx.get_documents("doc1")
    assert docs[0]["scores"] == [1.5, 2.0]


def test_mixed_int_and_float_list_becomes_float_array():
    """One non-integer element is enough to make the whole list a float
    array; the integers are widened, not rejected."""
    idx = _index_with(laurus.Schema.add_float_field, "scores", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "scores": [1, 2.5]})
    idx.commit()

    docs = idx.get_documents("doc1")
    assert docs[0]["scores"] == [1.0, 2.5]


def test_bool_elements_are_not_treated_as_integers():
    """`bool` is a subclass of `int` in Python; a list of bools must not be
    silently mistaken for an all-integer list. It falls through to the float
    path (Python allows `float(True)`), so the field sees `[1.0, 0.0]`."""
    idx = _index_with(laurus.Schema.add_float_field, "flags", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "flags": [True, False]})
    idx.commit()

    docs = idx.get_documents("doc1")
    assert docs[0]["flags"] == [1.0, 0.0]


def test_empty_list_is_accepted_by_multi_valued_integer_field():
    idx = _index_with(laurus.Schema.add_integer_field, "tags", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "tags": []})
    idx.commit()

    docs = idx.get_documents("doc1")
    assert docs[0]["tags"] == []


def test_list_into_single_valued_integer_field_is_still_rejected():
    """The fix must not loosen the single-valued contract: an array sent to a
    field declared without `multi_valued` is an error, not a truncation."""
    import pytest

    idx = _index_with(laurus.Schema.add_integer_field, "year")
    with pytest.raises(Exception):
        idx.put_document("doc1", {"title": "t", "year": [2020, 2021]})
