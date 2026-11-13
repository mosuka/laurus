"""Integration tests for ``Index.search_batch`` (Phase 3b of #648, issue #717).

These tests cover the Python-facing batched search API:

- Empty input returns an empty list without invoking the engine.
- Single-query batch matches the single-query ``search()`` result.
- Multi-query batch preserves input order and returns one result list
  per input query.
- DSL strings, ``Query`` objects, and ``SearchRequest`` are all accepted.
"""

import pytest
import laurus


@pytest.fixture
def index():
    """Return a fresh in-memory index with three indexed documents."""
    idx = laurus.Index()
    idx.put_document("doc1", {"title": "Introduction to Rust", "body": "Systems programming language."})
    idx.put_document("doc2", {"title": "Python for Data Science", "body": "Data analysis with Python."})
    idx.put_document("doc3", {"title": "Distributed Systems", "body": "Engineering at scale."})
    idx.commit()
    return idx


def test_search_batch_empty(index):
    """Empty list returns empty list."""
    results = index.search_batch([])
    assert results == []


def test_search_batch_single_query_matches_search(index):
    """search_batch([q]) should match search(q) for an equivalent query."""
    serial = index.search("title:rust", limit=5)
    batch = index.search_batch(["title:rust"], limit=5)
    assert len(batch) == 1
    assert len(batch[0]) == len(serial)
    for b, s in zip(batch[0], serial):
        assert b.id == s.id


def test_search_batch_multi_query_preserves_order(index):
    """Multi-query batch returns one list per input, in input order."""
    queries = ["title:rust", "body:python", "title:distributed"]
    expected_top_ids = ["doc1", "doc2", "doc3"]

    batch = index.search_batch(queries, limit=5)
    assert len(batch) == len(queries)

    for results, expected_id in zip(batch, expected_top_ids):
        assert len(results) >= 1, f"expected at least 1 hit for query targeting {expected_id}"
        assert results[0].id == expected_id


def test_search_batch_with_query_objects(index):
    """search_batch accepts ``Query`` objects, not just DSL strings."""
    queries = [
        laurus.TermQuery("title", "rust"),
        laurus.TermQuery("body", "python"),
    ]
    batch = index.search_batch(queries, limit=5)
    assert len(batch) == 2
    assert batch[0][0].id == "doc1"
    assert batch[1][0].id == "doc2"


def test_search_batch_with_search_requests(index):
    """search_batch accepts pre-built ``SearchRequest`` instances."""
    requests = [
        laurus.SearchRequest(lexical_query=laurus.TermQuery("title", "rust"), limit=5),
        laurus.SearchRequest(lexical_query=laurus.TermQuery("body", "python"), limit=5),
    ]
    batch = index.search_batch(requests)
    assert len(batch) == 2
    assert batch[0][0].id == "doc1"
    assert batch[1][0].id == "doc2"


def test_search_batch_no_match_returns_empty_inner_list(index):
    """A query with no matches yields an empty inner list, not a missing entry."""
    queries = ["title:rust", "title:nonexistent_xyz"]
    batch = index.search_batch(queries, limit=5)
    assert len(batch) == 2
    assert len(batch[0]) >= 1
    assert batch[1] == []


def test_search_batch_limit_per_query(index):
    """``limit`` applies to each query independently."""
    queries = ["body:programming OR body:data", "body:programming OR body:data"]
    batch = index.search_batch(queries, limit=1)
    assert len(batch) == 2
    for results in batch:
        assert len(results) <= 1


def test_search_batch_engine_error_names_the_query(index):
    """A query the engine rejects raises the ``ValueError`` that ``search()``
    raises for it, its message starting with the query's position (Issues
    #1553 and #1627)."""
    with pytest.raises(ValueError) as single:
        index.search("nosuch:rust")
    with pytest.raises(ValueError) as batch:
        index.search_batch(["title:rust", "nosuch:rust"])
    assert "unknown field" in str(batch.value)
    assert str(batch.value) == f"queries[1]: {single.value}"


def test_search_batch_conversion_error_names_the_query(index):
    """A query that cannot be converted raises the exception ``search()``
    raises for it, its message starting with the query's position and the
    original exception chained as its cause (Issue #1627)."""
    with pytest.raises(Exception) as single:
        index.search(42)
    with pytest.raises(Exception) as batch:
        index.search_batch(["title:rust", 42])
    assert type(batch.value) is type(single.value)
    assert str(batch.value) == f"queries[1]: {single.value}"
    assert str(batch.value.__cause__) == str(single.value)
