"""SearchRequest's fusion (Issue #1555).

A fusion that was neither an RRF nor a WeightedSum was dropped, so the
search ran with the default RRF (k = 60) instead of failing.
"""

import pytest

import laurus


@pytest.fixture
def hybrid_index():
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_flat_field("vec", dimension=2)
    index = laurus.Index(schema=schema)
    for doc_id, title, vec in [
        ("a", "rust a", [1.0, 0.0]),
        ("b", "rust b", [0.0, 1.0]),
        ("c", "python c", [0.8, 0.6]),
    ]:
        index.put_document(doc_id, {"title": title, "vec": vec})
    index.commit()
    return index


def hybrid_request(fusion):
    return laurus.SearchRequest(
        lexical_query=laurus.TermQuery("title", "rust"),
        vector_query=laurus.VectorQuery("vec", [1.0, 0.0]),
        fusion=fusion,
        limit=10,
    )


class Other:
    pass


@pytest.mark.parametrize(
    "fusion",
    ["rrf", {"k": 1.0}, 42, Other()],
    ids=["str", "dict", "int", "object"],
)
def test_search_request_rejects_a_fusion_of_another_type(fusion):
    with pytest.raises(TypeError, match=r"^fusion must be a laurus\.RRF or a laurus\.WeightedSum$"):
        hybrid_request(fusion)


def test_search_request_applies_none_rrf_and_weighted_sum(hybrid_index):
    def scores(fusion):
        return {r.id: round(r.score, 6) for r in hybrid_index.search(hybrid_request(fusion))}

    default = scores(None)
    assert scores(laurus.RRF(k=60.0)) == default
    rrf_1 = scores(laurus.RRF(k=1.0))
    assert rrf_1["a"] == 1.0
    assert rrf_1 != default
    assert scores(laurus.WeightedSum(lexical_weight=0.9, vector_weight=0.1)) not in (default, rrf_1)
