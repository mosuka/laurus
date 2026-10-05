"""Tests for multi-vector fields and the late-interaction rescore (Issue #1351).

Uses the same corpus as the Rust test
(`laurus/tests/late_interaction_rescore_test.rs`): against the query token
vectors ``[[1, 0], [0, 1]]`` the MaxSim scores are c 1.1, b 1.0, d 0.9,
a 0.1, e 0.05 — an order that matches neither the BM25 nor the ``vec``
ranking — so the binding must rank exactly like the Rust API.
"""

import pytest

import laurus

QUERY = [[1.0, 0.0], [0.0, 1.0]]
EXPECTED = ["c", "b", "d", "a", "e"]
CORPUS = [
    ("a", "rust", [1.0, 0.0], [[0.1, 0.0]]),
    ("b", "rust rust", [0.9, 0.1], [[0.5, 0.5], [0.0, 0.2]]),
    ("c", "rust language", [0.5, 0.5], [[0.9, 0.2]]),
    ("d", "rust rust rust", [0.2, 0.8], [[0.3, 0.3], [0.6, 0.0]]),
    ("e", "learning rust today", [0.0, 1.0], [[0.02, 0.03]]),
]


def max_sim(query, tokens):
    return sum(max(sum(q * t for q, t in zip(qv, tv)) for tv in tokens) for qv in query)


@pytest.fixture
def index():
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_flat_field("vec", dimension=2)
    schema.add_multi_vector_field("tokens", dimension=2, distance="dot_product")
    idx = laurus.Index(schema=schema)
    for doc_id, title, vec, tokens in CORPUS:
        idx.put_document(doc_id, {"title": title, "vec": vec, "tokens": tokens})
    idx.commit()
    return idx


def ids(results):
    return [r.id for r in results]


def test_dsl_search_is_reordered_by_late_interaction(index):
    baseline = index.search("title:rust")
    assert ids(baseline) != EXPECTED

    rescore = laurus.LateInteractionRescore("tokens", QUERY)
    results = index.search("title:rust", rescore=rescore)

    assert ids(results) == EXPECTED
    tokens = {doc_id: t for doc_id, _, _, t in CORPUS}
    for r in results:
        assert r.score == pytest.approx(max_sim(QUERY, tokens[r.id]), abs=1e-5)


def test_search_request_is_reordered_by_late_interaction(index):
    rescore = laurus.LateInteractionRescore("tokens", QUERY)
    request = laurus.SearchRequest(query="title:rust", rescore=rescore)
    assert ids(index.search(request)) == EXPECTED

    hybrid = laurus.SearchRequest(
        lexical_query=laurus.TermQuery("title", "rust"),
        vector_query=laurus.VectorQuery("vec", [1.0, 0.0]),
        fusion=laurus.RRF(),
        rescore=rescore,
    )
    assert ids(index.search(hybrid)) == EXPECTED


def test_only_the_window_is_reordered(index):
    baseline = index.search("title:rust")
    rescore = laurus.LateInteractionRescore("tokens", QUERY, window_size=1)
    assert rescore.window_size == 1

    results = index.search("title:rust", rescore=rescore)

    assert ids(results) == ids(baseline)
    tokens = {doc_id: t for doc_id, _, _, t in CORPUS}
    assert results[0].score == pytest.approx(max_sim(QUERY, tokens[results[0].id]), abs=1e-5)
    for r, b in zip(results[1:], baseline[1:]):
        assert r.score == b.score


def test_token_vectors_are_not_stored(index):
    # Token vectors live only in the vector store, unlike a single vector.
    doc = index.get_documents("b")[0]
    assert "tokens" not in doc
    assert doc["vec"] == pytest.approx([0.9, 0.1])


def test_default_window_size():
    assert laurus.LateInteractionRescore("tokens", QUERY).window_size == 100


@pytest.mark.parametrize(
    "query", [[1.0, 0.0], [[1.0], "x"], 42, None], ids=["flat", "mixed", "int", "none"]
)
def test_query_must_be_text_or_token_vectors(query):
    with pytest.raises(TypeError, match="query must be a str or a list of float lists"):
        laurus.LateInteractionRescore("tokens", query)


@pytest.mark.parametrize(
    ("rescore", "message"),
    [
        (laurus.LateInteractionRescore("tokens", "rust"), "has no token-level embedder"),
        (laurus.LateInteractionRescore("vec", QUERY), "needs a MultiVector field"),
        (laurus.LateInteractionRescore("tokens", []), "between 1 and 1024 query vectors"),
        (laurus.LateInteractionRescore("tokens", [[1.0, 0.0, 0.0]]), "has dimension 3"),
        (laurus.LateInteractionRescore("tokens", QUERY, window_size=0), "window_size"),
    ],
    ids=["text-without-embedder", "not-multi-vector", "empty", "dimension", "window"],
)
def test_invalid_rescore_is_rejected(index, rescore, message):
    with pytest.raises(ValueError, match=message):
        index.search("title:rust", rescore=rescore)


@pytest.mark.parametrize(
    ("tokens", "error"),
    [
        ([[1.0, 0.0], [0.0]], ValueError),
        ([[1.0, True]], TypeError),
        ([[1.0, "x"]], TypeError),
    ],
    ids=["ragged", "bool", "str"],
)
def test_invalid_token_vectors_are_rejected(index, tokens, error):
    with pytest.raises(error):
        index.put_document("x", {"title": "rust", "tokens": tokens})


def test_integer_token_vectors_are_accepted(index):
    index.put_document("x", {"title": "rust", "tokens": [[2, 0]]})
    index.commit()
    results = index.search("title:rust", rescore=laurus.LateInteractionRescore("tokens", QUERY))
    assert results[0].id == "x"
    assert results[0].score == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dimension": 0}, "dimension"),
        ({"dimension": 2, "distance": "euclidean"}, "distance"),
    ],
    ids=["zero-dimension", "euclidean"],
)
def test_add_multi_vector_field_rejects_invalid_options(kwargs, message):
    schema = laurus.Schema()
    with pytest.raises(ValueError, match=message):
        schema.add_multi_vector_field("tokens", **kwargs)
    assert schema.field_names() == []


def test_add_embedder_accepts_every_core_type():
    schema = laurus.Schema()
    schema.add_embedder(
        "colbert",
        {
            "type": "candle_colbert",
            "model": "colbert-ir/colbertv2.0",
            "revision": "main",
            "query_maxlen": 16,
            "doc_maxlen": 64,
        },
    )
    schema.add_multi_vector_field("tokens", dimension=128, embedder="colbert")

    restored = laurus.Schema.from_toml(schema.to_toml())
    toml = restored.to_toml()
    assert 'type = "candle_colbert"' in toml
    assert "query_maxlen = 16" in toml
    assert 'embedder = "colbert"' in toml


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"type": "candle_bert"}, "missing field `model`"),
        ({"type": "nope"}, "unknown variant"),
        ({}, "missing field `type`"),
        ("candle_bert", "must be a dict"),
    ],
    ids=["missing-model", "unknown-type", "missing-type", "not-a-dict"],
)
def test_add_embedder_rejects_invalid_config(config, message):
    schema = laurus.Schema()
    with pytest.raises(ValueError, match=message):
        schema.add_embedder("e", config)
