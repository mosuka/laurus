"""Tests for `Index.reload()` (Issue #1089).

Before this change, the only way to pick up changes committed by another
process was to construct a brand-new `Index`, which always pays the full
`Engine::build()` cost (embedding-model reload in particular). `reload()`
rebuilds the `Engine` for the same directory in place, reusing the
already-loaded embedder(s) when the schema's embedding configuration hasn't
changed, and works whether the `Index` is currently open or already
`close()`d -- reopening the same directory either way. That close()-interop
is what makes "pick up an external commit" testable deterministically: the
directory lock (Issue #1086) is a single, process-wide exclusive lock, so
simulating "another process wrote" only requires a second `Index` handle
that opens after the first one closes.

Since Issue #1103 releases the GIL during `Index` method calls, `reload()`
(and `close()`) can now genuinely race an in-flight call from another
thread. Both take `&mut self`, which pyo3 only grants as an *exclusive*
borrow -- it cannot be acquired while another call's ordinary `&self`
(shared) borrow is outstanding. See `test_reload_raises_when_it_races_a_
concurrent_call` below.
"""

import threading
import time

import laurus


def _index_with_title_field(path=None):
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.set_default_fields(["title"])
    return laurus.Index(path=path, schema=schema)


def test_reload_noop_when_nothing_changed(tmp_path):
    index = _index_with_title_field(str(tmp_path))
    index.put_document("doc1", {"title": "hello world"})
    index.commit()

    assert index.reload() is False
    results = index.search("title:hello", limit=5).results
    assert len(results) == 1


def test_reload_picks_up_external_commit(tmp_path):
    path = str(tmp_path)
    a = _index_with_title_field(path)
    a.put_document("doc1", {"title": "hello world"})
    a.commit()
    a.close()

    # Simulate another process writing to the same directory while `a` is
    # not holding the lock.
    b = laurus.Index(path=path)
    b.put_document("doc2", {"title": "hello again"})
    b.commit()
    b.close()

    assert a.reload() is True
    results = a.search("title:hello", limit=5).results
    assert len(results) == 2


def test_reload_requires_file_backed_index():
    index = laurus.Index()
    try:
        index.reload()
        assert False, "expected reload() to raise on an in-memory index"
    except ValueError:
        pass


def test_reload_preserves_commit_policy(tmp_path):
    path = str(tmp_path)
    schema = laurus.Schema()
    schema.add_text_field("title")
    policy = laurus.CommitPolicy.every_docs(1)
    index = laurus.Index(path=path, schema=schema, commit_policy=policy)

    index.put_document("doc1", {"title": "auto committed"})
    assert len(index.get_documents("doc1")) == 1  # visible without an explicit commit()

    index.reload()

    # Auto-commit must still apply after reload(), not silently reset to manual.
    index.put_document("doc2", {"title": "still auto committed"})
    assert len(index.get_documents("doc2")) == 1


def test_commit_generation_matches_stats(tmp_path):
    index = _index_with_title_field(str(tmp_path))
    index.put_document("doc1", {"title": "hello world"})
    index.commit()

    assert index.commit_generation() == index.stats()["commit_generation"]


def test_reload_raises_when_it_races_a_concurrent_call(tmp_path):
    index = _index_with_title_field(str(tmp_path))
    index.put_documents([(f"doc{i}", {"title": f"doc {i}"}) for i in range(50_000)])
    index.commit()

    start_barrier = threading.Event()
    result = {}

    def searcher():
        start_barrier.wait()
        # Long enough to keep a shared borrow on `index` outstanding while
        # `reloader` tries to acquire an exclusive one.
        index.search_batch(["title:doc"] * 200, limit=10)

    def reloader():
        start_barrier.wait()
        time.sleep(0.02)  # let the searcher acquire its shared borrow first
        try:
            index.reload()
            result["error"] = None
        except RuntimeError as e:
            result["error"] = str(e)

    t_search = threading.Thread(target=searcher)
    t_reload = threading.Thread(target=reloader)
    t_search.start()
    t_reload.start()
    start_barrier.set()
    t_search.join()
    t_reload.join()

    assert result["error"] == "Already borrowed"
