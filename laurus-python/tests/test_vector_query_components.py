"""A query vector with a NaN or infinite component is rejected (Issue #1573).

The engine checks the components before the search runs, whatever the
index holds, as it checks a document's vector.
"""

import math

import pytest

import laurus


def make_index(with_documents):
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_flat_field("vec", dimension=2)
    index = laurus.Index(schema=schema)
    if with_documents:
        index.add_document("doc1", {"title": "rust", "vec": [1.0, 0.0]})
        index.add_document("doc2", {"title": "go", "vec": [0.6, 0.8]})
        index.commit()
    return index


@pytest.mark.parametrize("with_documents", [False, True])
# 1e39 is finite in Python but too large for a 32-bit float, so it becomes
# infinite when the vector is converted.
@pytest.mark.parametrize("component", [math.nan, math.inf, -math.inf, 1e39])
def test_non_finite_component_raises_value_error(with_documents, component):
    index = make_index(with_documents)
    with pytest.raises(ValueError, match="query vector components must be finite"):
        index.search(laurus.VectorQuery("vec", [1.0, component]))


def test_finite_components_search():
    index = make_index(with_documents=True)
    results = index.search(laurus.VectorQuery("vec", [1.0, 0.0]))
    assert [r.id for r in results] == ["doc1", "doc2"]
