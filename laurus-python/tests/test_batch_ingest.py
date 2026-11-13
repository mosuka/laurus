"""Integration tests for ``Index.put_documents`` / ``Index.add_documents`` (#866).

These tests cover the Python-facing batched ingestion API:

- Empty batch is a no-op.
- ``put_documents`` applies the batch and deduplicates duplicate ids
  within one batch (last occurrence wins), matching a sequence of
  ``put_document`` calls.
- ``add_documents`` accumulates repeated ids as separate versions.
- A malformed entry (not an ``(id, dict)`` pair) is rejected naming its
  position.
- A document the engine rejects fails the batch with the exception it
  raises alone (#1627).
- A document that cannot be converted, or a malformed pair, raises a
  caller's error naming its position (#1630).
"""

import pytest
import laurus


def test_put_documents_empty_batch_is_noop():
    idx = laurus.Index()
    idx.put_documents([])
    idx.add_documents([])
    idx.commit()
    assert idx.stats()["document_count"] == 0


def test_put_documents_applies_and_dedupes():
    idx = laurus.Index()
    idx.put_documents(
        [
            ("doc1", {"title": "One"}),
            ("doc2", {"title": "Two"}),
            ("doc1", {"title": "One v2"}),  # duplicate id: last wins
        ]
    )
    idx.commit()

    assert idx.stats()["document_count"] == 2
    docs = idx.get_documents("doc1")
    assert len(docs) == 1
    assert docs[0]["title"] == "One v2"


def test_add_documents_accumulates_chunks():
    idx = laurus.Index()
    idx.add_documents(
        [
            ("doc", {"title": "chunk 0"}),
            ("doc", {"title": "chunk 1"}),
        ]
    )
    idx.commit()

    docs = idx.get_documents("doc")
    assert len(docs) == 2


def test_put_documents_rejects_malformed_entry():
    idx = laurus.Index()
    with pytest.raises(Exception) as excinfo:
        idx.put_documents([("ok", {"title": "fine"}), "not-a-pair"])
    assert "documents[1]" in str(excinfo.value)


@pytest.mark.parametrize("method", ["put_documents", "add_documents"])
def test_batch_raises_the_exception_of_the_failing_document(method):
    """A document the engine rejects fails the batch with the ``ValueError``
    the same document raises alone, the message naming its position and id
    (Issue #1627: it was a ``RuntimeError``)."""
    idx = laurus.Index()
    with pytest.raises(ValueError) as single:
        idx.put_document("b", {"_x": "v"})
    with pytest.raises(ValueError) as batch:
        getattr(idx, method)([("a", {"title": "a"}), ("b", {"_x": "v"})])
    message = str(batch.value)
    assert "doc 1 (id 'b')" in message
    assert message.endswith(str(single.value))


@pytest.mark.parametrize("method", ["put_documents", "add_documents"])
def test_batch_names_the_document_that_cannot_be_converted(method):
    """A document whose content cannot be converted fails the batch with the
    exception ``put_document`` raises for it, its message starting with the
    document's position and the original chained as its cause (Issue
    #1630)."""
    idx = laurus.Index()
    with pytest.raises(Exception) as single:
        idx.put_document("b", {"x": {"a": 1}})
    with pytest.raises(Exception) as batch:
        getattr(idx, method)([("a", {"title": "a"}), ("b", {"x": {"a": 1}})])
    assert type(batch.value) is type(single.value)
    assert str(batch.value) == f"documents[1]: {single.value}"
    assert str(batch.value.__cause__) == str(single.value)


@pytest.mark.parametrize(
    ("entry", "error", "message"),
    [
        ("not-a-pair", TypeError, "documents[1]: expected a (id: str, doc: dict) pair"),
        (("b",), ValueError, "documents[1]: expected a 2-element (id, doc) pair, got 1 elements"),
        (("b", {}, 1), ValueError, "documents[1]: expected a 2-element (id, doc) pair, got 3 elements"),
        ((1, {}), TypeError, "documents[1]: id must be a str"),
        (("b", "x"), TypeError, "documents[1]: document must be a dict"),
    ],
    ids=["not-a-pair", "one-element", "three-elements", "id-not-str", "doc-not-dict"],
)
def test_batch_rejects_a_malformed_pair(entry, error, message):
    """A malformed pair raises ``TypeError`` for a wrong type and
    ``ValueError`` for the wrong number of elements, naming its position
    (Issue #1630: it was a ``RuntimeError``)."""
    idx = laurus.Index()
    with pytest.raises(Exception) as excinfo:
        idx.put_documents([("a", {"title": "a"}), entry])
    assert type(excinfo.value) is error
    assert str(excinfo.value) == message
