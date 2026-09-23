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
    silently mistaken for an all-integer list. Since #1180 it arrives as a
    `BoolArray`, which the core widens element-wise to `[1.0, 0.0]` on a
    multi-valued Float field (the same 0/1 rule a scalar `bool` follows)."""
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


# ---- Multi-valued geo (Issue #1174) ----


def test_tuple_list_round_trips_through_multi_valued_geo_field():
    """A list of ``(lat, lon)`` tuples is a multi-valued geo field and reads
    back as the same list of tuples."""
    idx = _index_with(laurus.Schema.add_geo_field, "spots", multi_valued=True)
    idx.put_document(
        "doc1", {"title": "t", "spots": [(35.68, 139.76), (34.69, 135.50)]}
    )
    idx.commit()

    docs = idx.get_documents("doc1")
    assert docs[0]["spots"] == [(35.68, 139.76), (34.69, 135.50)]


def test_tuple_list_round_trips_through_multi_valued_geo3d_field():
    idx = _index_with(laurus.Schema.add_geo3d_field, "positions", multi_valued=True)
    idx.put_document(
        "doc1", {"title": "t", "positions": [(1.0, 2.0, 3.0), (-4.0, 5.0, -6.0)]}
    )
    idx.commit()

    docs = idx.get_documents("doc1")
    assert docs[0]["positions"] == [(1.0, 2.0, 3.0), (-4.0, 5.0, -6.0)]


def test_single_tuple_is_wrapped_on_multi_valued_geo_field():
    idx = _index_with(laurus.Schema.add_geo_field, "spots", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "spots": (35.68, 139.76)})
    idx.commit()

    assert idx.get_documents("doc1")[0]["spots"] == [(35.68, 139.76)]


def test_empty_list_is_accepted_by_multi_valued_geo_field():
    idx = _index_with(laurus.Schema.add_geo_field, "spots", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "spots": []})
    idx.commit()

    assert idx.get_documents("doc1")[0]["spots"] == []


def test_tuple_list_into_single_valued_geo_field_is_rejected():
    import pytest

    idx = _index_with(laurus.Schema.add_geo_field, "spot")
    with pytest.raises(Exception, match="multi_valued"):
        idx.put_document("doc1", {"title": "t", "spot": [(35.68, 139.76)]})


def test_mixed_tuple_arities_are_rejected():
    """2-tuples and 3-tuples in one list are neither a 2D nor a 3D array."""
    import pytest

    idx = _index_with(laurus.Schema.add_geo_field, "spots", multi_valued=True)
    with pytest.raises(Exception):
        idx.put_document(
            "doc1", {"title": "t", "spots": [(35.68, 139.76), (1.0, 2.0, 3.0)]}
        )


# ---- Multi-valued datetime (Issue #1184) ----


def test_datetime_list_round_trips_through_multi_valued_datetime_field():
    """A list of RFC 3339 strings or ``datetime`` objects is a multi-valued
    datetime field; it reads back as RFC 3339 strings (UTC) and any instant
    matches a range query."""
    from datetime import datetime, timedelta, timezone

    idx = _index_with(laurus.Schema.add_datetime_field, "seen_at", multi_valued=True)
    tokyo = timezone(timedelta(hours=9))
    idx.put_document(
        "doc1",
        {
            "title": "t",
            "seen_at": ["2024-01-01T00:00:00Z", datetime(2024, 6, 15, 21, 0, 0, tzinfo=tokyo)],
        },
    )
    idx.put_document("doc2", {"title": "t", "seen_at": ["2025-03-01T00:00:00Z"]})
    idx.commit()

    docs = idx.get_documents("doc1")
    assert docs[0]["seen_at"] == ["2024-01-01T00:00:00+00:00", "2024-06-15T12:00:00+00:00"]
    hits = idx.search("seen_at:[2024-06-01 TO 2024-12-31]", limit=5)
    assert [h.id for h in hits] == ["doc1"]


def test_single_datetime_is_wrapped_on_multi_valued_datetime_field():
    idx = _index_with(laurus.Schema.add_datetime_field, "seen_at", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "seen_at": "2024-01-01T00:00:00Z"})
    idx.commit()

    assert idx.get_documents("doc1")[0]["seen_at"] == ["2024-01-01T00:00:00+00:00"]


def test_empty_list_is_accepted_by_multi_valued_datetime_field():
    idx = _index_with(laurus.Schema.add_datetime_field, "seen_at", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "seen_at": []})
    idx.commit()

    assert idx.get_documents("doc1")[0]["seen_at"] == []


def test_datetime_list_into_single_valued_datetime_field_is_rejected():
    import pytest

    idx = _index_with(laurus.Schema.add_datetime_field, "seen_at")
    with pytest.raises(Exception, match="multi_valued"):
        idx.put_document("doc1", {"title": "t", "seen_at": ["2024-01-01T00:00:00Z"]})


def test_non_datetime_string_list_is_rejected():
    import pytest

    idx = _index_with(laurus.Schema.add_datetime_field, "seen_at", multi_valued=True)
    with pytest.raises(ValueError, match="datetimes"):
        idx.put_document("doc1", {"title": "t", "seen_at": ["2024-01-01T00:00:00Z", "tomorrow"]})


# ---- Multi-valued boolean (Issue #1180) ----


def test_bool_list_round_trips_through_multi_valued_boolean_field():
    """A list of bools is a multi-valued boolean field; it reads back as a
    list of bools and a term query matches if any element carries the value."""
    idx = _index_with(laurus.Schema.add_boolean_field, "flags", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "flags": [True, False]})
    idx.put_document("doc2", {"title": "t", "flags": [False]})
    idx.commit()

    assert idx.get_documents("doc1")[0]["flags"] == [True, False]
    assert [h.id for h in idx.search("flags:true", limit=5)] == ["doc1"]
    assert sorted(h.id for h in idx.search("flags:false", limit=5)) == ["doc1", "doc2"]


def test_single_bool_is_wrapped_on_multi_valued_boolean_field():
    idx = _index_with(laurus.Schema.add_boolean_field, "flags", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "flags": True})
    idx.commit()

    assert idx.get_documents("doc1")[0]["flags"] == [True]


def test_empty_list_is_accepted_by_multi_valued_boolean_field():
    idx = _index_with(laurus.Schema.add_boolean_field, "flags", multi_valued=True)
    idx.put_document("doc1", {"title": "t", "flags": []})
    idx.commit()

    assert idx.get_documents("doc1")[0]["flags"] == []


def test_bool_list_into_single_valued_boolean_field_is_rejected():
    import pytest

    idx = _index_with(laurus.Schema.add_boolean_field, "flags")
    with pytest.raises(Exception, match="multi_valued"):
        idx.put_document("doc1", {"title": "t", "flags": [True]})


def test_mixed_bool_and_int_list_into_boolean_field_is_rejected():
    """`[True, 1]` is not all-bool, so it takes the numeric path (a float
    array), which a multi-valued Boolean field rejects."""
    import pytest

    idx = _index_with(laurus.Schema.add_boolean_field, "flags", multi_valued=True)
    with pytest.raises(Exception):
        idx.put_document("doc1", {"title": "t", "flags": [True, 1]})
