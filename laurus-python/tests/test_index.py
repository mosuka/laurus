"""Basic integration tests for the laurus Python binding."""

import pytest
import laurus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def index():
    """Return a fresh in-memory index with two indexed documents."""
    idx = laurus.Index()
    idx.put_document("doc1", {"title": "Introduction to Rust", "body": "Systems programming language."})
    idx.put_document("doc2", {"title": "Python for Data Science", "body": "Data analysis with Python."})
    idx.commit()
    return idx


@pytest.fixture
def vector_index():
    """Return an in-memory HNSW index with two indexed documents."""
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.add_hnsw_field("embedding", dimension=4)
    idx = laurus.Index(schema=schema)
    idx.put_document("doc1", {"title": "Rust", "embedding": [0.1, 0.2, 0.3, 0.4]})
    idx.put_document("doc2", {"title": "Python", "embedding": [0.9, 0.8, 0.7, 0.6]})
    idx.commit()
    return idx


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------


def test_index_memory():
    idx = laurus.Index()
    assert repr(idx) == "Index()"


def test_index_with_schema():
    schema = laurus.Schema()
    schema.add_text_field("title")
    idx = laurus.Index(schema=schema)
    assert idx is not None


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------


def test_put_and_get_document():
    idx = laurus.Index()
    idx.put_document("doc1", {"title": "Hello"})
    idx.commit()
    docs = idx.get_documents("doc1")
    assert len(docs) == 1


def test_put_replaces_existing(index):
    index.put_document("doc1", {"title": "Updated"})
    index.commit()
    docs = index.get_documents("doc1")
    assert len(docs) == 1


def test_add_document_appends():
    idx = laurus.Index()
    idx.add_document("doc1", {"title": "Chunk 1"})
    idx.add_document("doc1", {"title": "Chunk 2"})
    docs = idx.get_documents("doc1")
    assert len(docs) == 2


def test_delete_documents(index):
    index.delete_documents("doc1")
    index.commit()
    docs = index.get_documents("doc1")
    assert len(docs) == 0


def test_get_documents_unknown_id(index):
    docs = index.get_documents("does_not_exist")
    assert docs == []


# ---------------------------------------------------------------------------
# WAL sync policy
# ---------------------------------------------------------------------------


def test_wal_sync_policy_constructors():
    """All WalSyncPolicy constructors build without error."""
    assert laurus.WalSyncPolicy.per_record() is not None
    assert laurus.WalSyncPolicy.group() is not None
    assert (
        laurus.WalSyncPolicy.group(
            max_records=4096, max_bytes=2 * 1024 * 1024, max_interval_ms=1000
        )
        is not None
    )


def test_index_accepts_wal_sync_policies():
    """Index construction accepts every WalSyncPolicy variant."""
    assert laurus.Index(wal_sync_policy=laurus.WalSyncPolicy.per_record()) is not None
    assert laurus.Index(wal_sync_policy=laurus.WalSyncPolicy.group()) is not None
    assert (
        laurus.Index(
            wal_sync_policy=laurus.WalSyncPolicy.group(
                max_records=8, max_bytes=4096, max_interval_ms=500
            )
        )
        is not None
    )


def test_index_group_commit_flush_wal_then_commit():
    """A document written under group commit is retrievable after flush+commit."""
    idx = laurus.Index(wal_sync_policy=laurus.WalSyncPolicy.group())
    idx.put_document("doc1", {"title": "Group Commit"})
    idx.flush_wal()
    idx.commit()
    docs = idx.get_documents("doc1")
    assert len(docs) == 1


def test_flush_wal_per_record_is_noop():
    """flush_wal() is a harmless no-op under the default per-record policy."""
    idx = laurus.Index()
    idx.put_document("doc1", {"title": "Per Record"})
    idx.flush_wal()
    idx.commit()
    docs = idx.get_documents("doc1")
    assert len(docs) == 1


def test_commit_policy_constructors():
    """All CommitPolicy constructors build without error."""
    assert laurus.CommitPolicy.manual() is not None
    assert laurus.CommitPolicy.every_docs(100) is not None
    # every_docs(0) is valid — it disables auto-commit (equivalent to manual).
    assert laurus.CommitPolicy.every_docs(0) is not None
    assert "every_docs(100)" in repr(laurus.CommitPolicy.every_docs(100))
    assert laurus.CommitPolicy.interval_ms(1000) is not None
    assert "interval_ms(1000)" in repr(laurus.CommitPolicy.interval_ms(1000))


def test_index_accepts_commit_policies():
    """Index construction accepts every CommitPolicy variant, and the default."""
    assert laurus.Index(commit_policy=laurus.CommitPolicy.manual()) is not None
    assert laurus.Index(commit_policy=laurus.CommitPolicy.every_docs(100)) is not None
    assert laurus.Index(commit_policy=laurus.CommitPolicy.every_docs(0)) is not None
    assert laurus.Index(commit_policy=laurus.CommitPolicy.interval_ms(1000)) is not None
    # Omitting commit_policy keeps the default (manual).
    assert laurus.Index() is not None


def test_index_auto_commit_end_to_end():
    """An index built with EveryDocs is usable; a doc is retrievable with no
    explicit commit (the binding path is wired end-to-end)."""
    idx = laurus.Index(commit_policy=laurus.CommitPolicy.every_docs(1))
    idx.put_document("d1", {"title": "Auto"})
    # No explicit idx.commit() here.
    assert len(idx.get_documents("d1")) == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats(index):
    stats = index.stats()
    assert stats["document_count"] == 2


def test_stats_vector_fields(vector_index):
    stats = vector_index.stats()
    assert "embedding" in stats["vector_fields"]
    assert stats["vector_fields"]["embedding"]["count"] == 2
    assert stats["vector_fields"]["embedding"]["dimension"] == 4


# ---------------------------------------------------------------------------
# Lexical search
# ---------------------------------------------------------------------------


def test_search_dsl(index):
    results = index.search("title:rust", limit=5)
    assert len(results) >= 1
    assert results[0].id == "doc1"


def test_search_term_query(index):
    results = index.search(laurus.TermQuery("body", "python"), limit=5)
    assert len(results) >= 1
    assert results[0].id == "doc2"


def test_search_result_has_id_and_score(index):
    results = index.search("title:rust", limit=1)
    r = results[0]
    assert r.id == "doc1"
    assert r.score > 0


def test_search_limit(index):
    results = index.search("body:programming OR body:python", limit=1)
    assert len(results) <= 1


def test_search_offset(index):
    all_results = index.search("body:programming OR body:data", limit=10)
    offset_results = index.search("body:programming OR body:data", limit=10, offset=1)
    if len(all_results) > 1:
        assert offset_results[0].id == all_results[1].id


def test_search_no_results(index):
    results = index.search("title:nonexistent_xyz", limit=5)
    assert results == []


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


def test_vector_query(vector_index):
    results = vector_index.search(laurus.VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]), limit=2)
    assert len(results) >= 1
    assert results[0].id == "doc1"


def test_vector_query_repr():
    q = laurus.VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4])
    assert "embedding" in repr(q)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


def test_search_request_lexical_only(index):
    req = laurus.SearchRequest(lexical_query=laurus.TermQuery("title", "rust"), limit=5)
    results = index.search(req)
    assert len(results) >= 1


def test_search_request_hybrid(vector_index):
    req = laurus.SearchRequest(
        lexical_query=laurus.TermQuery("title", "rust"),
        vector_query=laurus.VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]),
        fusion=laurus.RRF(k=60.0),
        limit=5,
    )
    results = vector_index.search(req)
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Query types
# ---------------------------------------------------------------------------


def test_phrase_query(index):
    results = index.search(laurus.PhraseQuery("title", ["introduction", "rust"]), limit=5)
    assert any(r.id == "doc1" for r in results)


def test_fuzzy_query(index):
    results = index.search(laurus.FuzzyQuery("title", "pythn", max_edits=1), limit=5)
    assert any(r.id == "doc2" for r in results)


def test_numeric_range_query():
    schema = laurus.Schema()
    schema.add_integer_field("year")
    idx = laurus.Index(schema=schema)
    idx.put_document("doc1", {"year": 2020})
    idx.put_document("doc2", {"year": 2023})
    idx.commit()
    results = idx.search(laurus.NumericRangeQuery("year", min=2022, max=2024), limit=5)
    assert len(results) == 1
    assert results[0].id == "doc2"


def _datetime_index():
    schema = laurus.Schema()
    schema.add_datetime_field("created_at")
    idx = laurus.Index(schema=schema)
    idx.put_document("jan", {"created_at": "2024-01-01T00:00:00Z"})
    idx.put_document("jun", {"created_at": "2024-06-15T12:00:00Z"})
    idx.put_document("next", {"created_at": "2025-01-01T00:00:00Z"})
    idx.commit()
    return idx


def _ids(results):
    return sorted(r.id for r in results)


def test_datetime_range_query_with_date_only_bounds():
    """Issue #1179: date-only bounds are midnight UTC, inclusive on both ends."""
    idx = _datetime_index()
    q = laurus.DateTimeRangeQuery("created_at", min="2024-01-01", max="2024-12-31")
    assert _ids(idx.search(q, limit=5)) == ["jan", "jun"]
    assert "DateTimeRangeQuery" in repr(q)


def test_datetime_range_query_accepts_datetime_objects_and_open_bounds():
    from datetime import datetime, timedelta, timezone

    idx = _datetime_index()
    # A timezone-aware datetime is normalized to UTC: 21:00+09:00 == 12:00Z.
    tokyo = timezone(timedelta(hours=9))
    lower = datetime(2024, 6, 15, 21, 0, 0, tzinfo=tokyo)
    q = laurus.DateTimeRangeQuery("created_at", min=lower)
    assert _ids(idx.search(q, limit=5)) == ["jun", "next"]
    # A naive datetime is interpreted as UTC.
    q = laurus.DateTimeRangeQuery("created_at", max=datetime(2024, 6, 15, 12, 0, 0))
    assert _ids(idx.search(q, limit=5)) == ["jan", "jun"]


def test_datetime_range_query_dsl_form_reaches_the_index():
    idx = _datetime_index()
    # The documented DSL form (exclusive braces).
    assert _ids(idx.search("created_at:{2024-01-01 TO 2024-12-31}", limit=5)) == ["jun"]


def test_datetime_range_query_rejects_malformed_bound():
    import pytest

    with pytest.raises(ValueError):
        laurus.DateTimeRangeQuery("created_at", min="yesterday")
    with pytest.raises(ValueError):
        laurus.DateTimeRangeQuery("created_at", min=42)


def test_boolean_query(index):
    q = laurus.BooleanQuery()
    q.must(laurus.TermQuery("body", "programming"))
    q.must_not(laurus.TermQuery("title", "python"))
    results = index.search(q, limit=5)
    assert all(r.id != "doc2" for r in results)


def test_wildcard_query(index):
    results = index.search(laurus.WildcardQuery("title", "py*"), limit=5)
    assert any(r.id == "doc2" for r in results)


# ---------------------------------------------------------------------------
# Fusion algorithms
# ---------------------------------------------------------------------------


def test_rrf_repr():
    rrf = laurus.RRF(k=60.0)
    assert "60" in repr(rrf)


def test_weighted_sum_repr():
    ws = laurus.WeightedSum(lexical_weight=0.3, vector_weight=0.7)
    assert "0.3" in repr(ws)


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------


def test_synonym_dictionary():
    syn = laurus.SynonymDictionary()
    syn.add_synonym_group(["ml", "machine learning"])


def test_whitespace_tokenizer():
    tokenizer = laurus.WhitespaceTokenizer()
    tokens = tokenizer.tokenize("hello world")
    assert len(tokens) == 2
    assert tokens[0].text == "hello"
    assert tokens[1].text == "world"


def test_synonym_graph_filter():
    syn = laurus.SynonymDictionary()
    syn.add_synonym_group(["ml", "machine learning"])
    tokenizer = laurus.WhitespaceTokenizer()
    filt = laurus.SynonymGraphFilter(syn, keep_original=True, boost=0.8)

    tokens = tokenizer.tokenize("ml tutorial")
    result = filt.apply(tokens)
    texts = [t.text for t in result]
    assert "ml" in texts
    assert "machine" in texts or "machine learning" in texts


def test_token_fields():
    tokenizer = laurus.WhitespaceTokenizer()
    tokens = tokenizer.tokenize("hello")
    tok = tokens[0]
    assert tok.text == "hello"
    assert isinstance(tok.position, int)
    assert isinstance(tok.position_increment, int)
    assert isinstance(tok.position_length, int)
    assert isinstance(tok.boost, float)
