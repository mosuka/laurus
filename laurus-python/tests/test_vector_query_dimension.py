"""A query vector of another dimension than its field is rejected (Issue #1575).

The engine checks the dimension before the search runs, whatever the index
holds, as it checks a document's vector.
"""

import re

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
@pytest.mark.parametrize("dimension", [1, 3])
def test_another_dimension_raises_value_error(with_documents, dimension):
    index = make_index(with_documents)
    expected = f"query vector 0 has dimension {dimension}, but field 'vec' has 2"
    with pytest.raises(ValueError, match=re.escape(expected)):
        index.search(laurus.VectorQuery("vec", [1.0] * dimension))


def test_the_field_dimension_searches():
    index = make_index(with_documents=True)
    results = index.search(laurus.VectorQuery("vec", [1.0, 0.0]))
    assert [r.id for r in results] == ["doc1", "doc2"]
