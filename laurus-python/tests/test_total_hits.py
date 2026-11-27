"""The total hit count of a search (Issue #1472).

`Index.search`, `Index.search_batch` and `SearchRequest` take
`track_total_hits=`, read as the HTTP gateway reads `"track_total_hits"`:
an int counts exactly up to it, `True` counts every match, and `False` or
`None` counts nothing. Each search returns a `SearchResponse` whose
`total_hits` holds the count and whether it is exact (`"eq"`) or a lower
bound (`"gte"`).
"""

import pytest

import laurus

MESSAGE = r"^track_total_hits must be True, False or an integer from 0 to 18446744073709551615, got "


@pytest.fixture
def index():
    """Five documents match `body:rust`, one matches `body:python`."""
    schema = laurus.Schema()
    schema.add_text_field("body")
    index = laurus.Index(schema=schema)
    for i in range(5):
        index.put_document(f"rust{i}", {"body": f"rust {i}"})
    index.put_document("python", {"body": "python"})
    index.commit()
    return index


def total(response):
    t = response.total_hits
    return None if t is None else (t.value, t.relation)


@pytest.mark.parametrize(
    "track, expected",
    [(1000, (5, "eq")), (3, (3, "gte")), (True, (5, "eq")), (False, None), (None, None)],
    ids=["above", "below", "true", "false", "none"],
)
def test_search_counts_the_matches_as_asked(index, track, expected):
    response = index.search("body:rust", limit=2, track_total_hits=track)

    assert isinstance(response, laurus.SearchResponse)
    assert len(response.results) == 2
    assert total(response) == expected


def test_search_counts_nothing_by_default(index):
    response = index.search("body:rust", limit=2)

    assert len(response.results) == 2
    assert response.total_hits is None


def test_search_reads_true_as_every_match_not_as_one(index):
    # `True` is an `int` in Python; read as one, it would count up to 1.
    assert total(index.search("body:rust", limit=2, track_total_hits=True)) == (5, "eq")
    assert total(index.search("body:rust", limit=2, track_total_hits=1)) == (1, "gte")


def test_search_rejects_zero_through_the_engine(index):
    with pytest.raises(ValueError, match=r"track_total_hits must count at least 1 match"):
        index.search("body:rust", track_total_hits=0)


@pytest.mark.parametrize(
    "value, error",
    [("1000", TypeError), (1.5, TypeError), ({"up_to": 3}, TypeError), (-1, ValueError), (2**64, ValueError)],
    ids=["str", "float", "dict", "negative", "too-large"],
)
def test_search_rejects_another_value(index, value, error):
    with pytest.raises(error, match=MESSAGE):
        index.search("body:rust", track_total_hits=value)


def test_search_request_carries_its_own_track_total_hits(index):
    request = laurus.SearchRequest(query="body:rust", limit=2, track_total_hits=3)
    assert total(index.search(request)) == (3, "gte")

    # A SearchRequest uses its own value, as it does its own limit.
    plain = laurus.SearchRequest(query="body:rust", limit=2)
    assert total(index.search(plain, track_total_hits=True)) is None


def test_search_request_rejects_another_value():
    with pytest.raises(TypeError, match=MESSAGE):
        laurus.SearchRequest(query="body:rust", track_total_hits="1000")


def test_search_batch_returns_each_query_s_total(index):
    responses = index.search_batch(
        ["body:rust", "body:python", "body:missing"], limit=2, track_total_hits=1000
    )

    assert [total(r) for r in responses] == [(5, "eq"), (1, "eq"), (0, "eq")]
    assert [len(r.results) for r in responses] == [2, 1, 0]


def test_search_batch_lets_a_search_request_use_its_own_value(index):
    responses = index.search_batch(
        ["body:rust", laurus.SearchRequest(query="body:rust", limit=2, track_total_hits=3)],
        limit=2,
        track_total_hits=True,
    )

    assert [total(r) for r in responses] == [(5, "eq"), (3, "gte")]


def test_search_batch_counts_nothing_by_default(index):
    responses = index.search_batch(["body:rust"], limit=2)

    assert [total(r) for r in responses] == [None]


def test_search_batch_rejects_another_value(index):
    with pytest.raises(TypeError, match=MESSAGE):
        index.search_batch(["body:rust"], track_total_hits="1000")


def test_response_and_total_hits_repr(index):
    response = index.search("body:rust", limit=2, track_total_hits=3)

    assert repr(response) == "SearchResponse(results=2, total_hits=TotalHits(value=3, relation='gte'))"
    assert repr(response.total_hits) == "TotalHits(value=3, relation='gte')"
    assert repr(index.search("body:rust", limit=2)) == "SearchResponse(results=2, total_hits=None)"
