"""Integration tests for search-result highlighting (Issue #1134).

Covers the Python-facing ``highlight`` parameter on ``Index.search`` /
``Index.search_batch``, and the equivalent ``highlight=`` keyword on
``SearchRequest``:

- The list shorthand (``["body"]``) and the dict form (``{"fields": [...],
  ...}``) are both accepted.
- ``SearchResult.highlights`` is empty when ``highlight`` is not passed.
- ``HighlightConfig`` knobs (``tag``, ``max_fragments``, ...) take effect.
- ``search_batch`` applies the same highlight settings to every query.
- ``SearchRequest(highlight=...)`` produces the same highlights as the
  equivalent ``search(..., highlight=...)`` call.
"""

import pytest
import laurus


@pytest.fixture
def index():
    """Return a fresh in-memory index with one stored text document."""
    idx = laurus.Index()
    idx.put_document(
        "doc1",
        {"title": "Introduction to Rust", "body": "Rust is a systems programming language."},
    )
    idx.commit()
    return idx


def test_highlight_list_shorthand_returns_fragments(index):
    results = index.search("body:rust", highlight=["body"])
    assert len(results) == 1
    fragments = results[0].highlights["body"]
    assert len(fragments) == 1
    assert "<mark>Rust</mark>" in fragments[0]


def test_highlight_omitted_leaves_highlights_empty(index):
    results = index.search("body:rust")
    assert len(results) == 1
    assert results[0].highlights == {}


def test_highlight_unrequested_field_is_absent(index):
    results = index.search("body:rust", highlight=["body"])
    assert "title" not in results[0].highlights


def test_highlight_dict_form_applies_config(index):
    results = index.search(
        "body:rust",
        highlight={"fields": ["body"], "tag": "em", "max_fragments": 2},
    )
    fragment = results[0].highlights["body"][0]
    assert "<em>Rust</em>" in fragment
    assert "<mark>" not in fragment


def test_highlight_dict_requires_fields_key(index):
    with pytest.raises(ValueError):
        index.search("body:rust", highlight={"max_fragments": 2})


def test_highlight_invalid_type_raises(index):
    with pytest.raises(ValueError):
        index.search("body:rust", highlight=42)


def test_search_batch_applies_highlight_to_every_query(index):
    # Both queries target "body" so the requested "body" highlight applies
    # to both under the default require_field_match=True.
    batch = index.search_batch(["body:rust", "body:programming"], highlight=["body"])
    assert len(batch) == 2
    for results in batch:
        assert len(results) == 1
        assert "<mark>" in results[0].highlights["body"][0]


def test_search_request_highlight_matches_search_kwarg(index):
    via_kwarg = index.search("body:rust", highlight=["body"])
    request = laurus.SearchRequest(query="body:rust", highlight=["body"])
    via_request = index.search(request)

    assert via_kwarg[0].highlights == via_request[0].highlights
