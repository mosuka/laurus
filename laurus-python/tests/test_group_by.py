"""Tests for result grouping (``group_by``, Issue #1297).

Uses the corpus of ``laurus-server/tests/grpc_group_by_test.rs``: three
documents added as chunks sharing their ``_id``, with Flat vectors so the
vector ranking against ``[1, 0]`` — and with it the groups — is exact.
"""

import pytest

import laurus

CHUNKS = [
    ("a", "rust rust rust guide", [1.0, 0.0]),
    ("a", "rust rust intro", [0.9, 0.1]),
    ("a", "rust appendix notes", [0.3, 0.7]),
    ("b", "rust rust book chapter", [0.8, 0.2]),
    ("b", "rust history of programming", [0.2, 0.8]),
    ("c", "rust", [0.6, 0.4]),
]

# Each group as (id, titles of its hits, best first) with group_size 2.
EXPECTED = [
    ("a", ["rust rust rust guide", "rust rust intro"]),
    ("b", ["rust rust book chapter", "rust history of programming"]),
    ("c", ["rust"]),
]


@pytest.fixture
def index():
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_flat_field("vec", dimension=2)
    idx = laurus.Index(schema=schema)
    for doc_id, title, vec in CHUNKS:
        idx.add_document(doc_id, {"title": title, "vec": vec})
    idx.commit()
    return idx


def vector_query():
    return laurus.VectorQuery("vec", [1.0, 0.0])


def groups(results):
    out = []
    for r in results:
        assert r.group_key == r.id
        for hit in r.inner_hits:
            assert hit.group_key == r.group_key
            assert hit.inner_hits == []
        out.append((r.id, [h.document["title"] for h in [r, *r.inner_hits]]))
    return out


def test_search_groups_chunks_by_id(index):
    results = index.search(vector_query(), group_by=laurus.GroupBy("_id", group_size=2))
    assert groups(results) == EXPECTED

    one_each = index.search(vector_query(), group_by=laurus.GroupBy("_id"))
    assert [r.id for r in one_each] == ["a", "b", "c"]
    assert all(r.inner_hits == [] for r in one_each)


def test_ungrouped_results_have_no_group(index):
    results = index.search(vector_query())
    assert len(results) == len(CHUNKS)
    assert all(r.group_key is None and r.inner_hits == [] for r in results)


def test_search_request_carries_its_own_group_by(index):
    request = laurus.SearchRequest(
        vector_query=vector_query(), group_by=laurus.GroupBy("_id", group_size=2)
    )
    assert groups(index.search(request)) == EXPECTED


def test_search_batch_groups_every_query(index):
    batch = index.search_batch(
        [vector_query(), vector_query()], group_by=laurus.GroupBy("_id", group_size=2)
    )
    assert [groups(results) for results in batch] == [EXPECTED, EXPECTED]


def test_group_by_repr_and_attributes():
    group_by = laurus.GroupBy("parent", group_size=3)
    assert (group_by.field, group_by.group_size) == ("parent", 3)
    assert repr(group_by) == 'GroupBy(field="parent", group_size=3)'


@pytest.mark.parametrize(
    "group_by",
    [laurus.GroupBy("missing"), laurus.GroupBy("vec"), laurus.GroupBy("_id", group_size=0)],
)
def test_invalid_group_by_raises_value_error(index, group_by):
    with pytest.raises(ValueError, match="group_by"):
        index.search(vector_query(), group_by=group_by)
