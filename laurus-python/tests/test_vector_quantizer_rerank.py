"""Tests for the HNSW quantizer / rerank_storage schema options (Issue #797).

These assert that the values configured on ``add_hnsw_field`` actually reach
the Rust core, using deterministic observables rather than only that search
succeeds:

* ``rerank_storage="f32"`` makes the core write a ``*.hnsw.f32`` Stage-2
  sidecar on disk (mirrors the server-side guard in #793/#800); the default
  writes no sidecar.
* ``quantizer="product_quantization"`` forwards ``subvector_count`` to the
  core's PQ training, which rejects a count that does not divide the field
  dimension.
* ``quantizer="float16"`` (Issue #623) separates two vectors that the
  default int8 quantizer stores as the same code.
"""

import pytest
import laurus


# ---------------------------------------------------------------------------
# rerank_storage reaches the core (on-disk sidecar)
# ---------------------------------------------------------------------------


def test_rerank_storage_f32_writes_sidecar(tmp_path):
    schema = laurus.Schema()
    schema.add_hnsw_field("embedding", dimension=4, rerank_storage="f32")
    idx = laurus.Index(path=str(tmp_path), schema=schema)
    idx.put_document("doc1", {"embedding": [0.1, 0.2, 0.3, 0.4]})
    idx.put_document("doc2", {"embedding": [0.9, 0.8, 0.7, 0.6]})
    idx.commit()

    sidecars = list(tmp_path.rglob("*.hnsw.f32"))
    assert sidecars, "rerank_storage='f32' must write a .hnsw.f32 rerank sidecar"


def test_no_rerank_storage_writes_no_sidecar(tmp_path):
    schema = laurus.Schema()
    schema.add_hnsw_field("embedding", dimension=4)
    idx = laurus.Index(path=str(tmp_path), schema=schema)
    idx.put_document("doc1", {"embedding": [0.1, 0.2, 0.3, 0.4]})
    idx.put_document("doc2", {"embedding": [0.9, 0.8, 0.7, 0.6]})
    idx.commit()

    sidecars = list(tmp_path.rglob("*.hnsw.f32"))
    assert not sidecars, "no rerank sidecar should exist without rerank_storage"


# ---------------------------------------------------------------------------
# quantizer reaches the core (PQ training honours subvector_count)
# ---------------------------------------------------------------------------


def test_product_quantization_builds_and_searches():
    schema = laurus.Schema()
    # PQ is an L2 quantizer, so use Euclidean (matching the core's
    # `test_hnsw_pq_search_returns_corpus_neighbour`).
    schema.add_hnsw_field(
        "embedding",
        dimension=4,
        distance="euclidean",
        quantizer="product_quantization",
        subvector_count=2,
    )
    idx = laurus.Index(schema=schema)
    # Mirror the stable two-cluster corpus from the core's
    # `test_hnsw_pq_search_returns_corpus_neighbour` (issue #730), sized to
    # 128 points per cluster (256 total) so the segment meets the PQ
    # min-train threshold (#880: smaller PQ-configured segments are written
    # as Scalar8Bit and would not exercise PQ training at all).
    def offset(i):
        return [
            (i % 8) * 0.04 - 0.14,
            (i // 8 % 8) * 0.04 - 0.14,
            (i // 64 % 8) * 0.04 - 0.14,
            (i % 16) * 0.04 - 0.32,
        ]

    near_base = [10.0, 10.0, 20.0, 20.0]
    far_base = [-100.0, -100.0, -200.0, -200.0]
    for i in range(128):
        off = offset(i)
        idx.put_document(f"near{i}", {"embedding": [b + o for b, o in zip(near_base, off)]})
        idx.put_document(f"far{i}", {"embedding": [b + o for b, o in zip(far_base, off)]})
    idx.commit()

    results = idx.search(laurus.VectorQuery("embedding", near_base), limit=3)
    assert len(results) == 3
    assert all(r.id.startswith("near") for r in results)


def test_pq_subvector_count_must_divide_dimension():
    """subvector_count=3 does not divide dimension=4, so the core's PQ
    training must reject it at commit — proving the value reached the core."""
    schema = laurus.Schema()
    schema.add_hnsw_field(
        "embedding", dimension=4, quantizer="product_quantization", subvector_count=3
    )
    idx = laurus.Index(schema=schema)
    idx.put_document("doc1", {"embedding": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(Exception):
        idx.commit()


# ---------------------------------------------------------------------------
# quantizer="float16" reaches the core (Issue #623)
# ---------------------------------------------------------------------------


def _assert_float16_ranks_exact_match_first(add_field):
    """``low`` / ``high`` stretch the segment's int8 range to [-100, 100]
    (step ~0.78), so ``a`` and ``b`` share one int8 code and tie under the
    default quantizer; Float16 keeps them apart, so the exact match ``b``
    must rank first."""
    schema = laurus.Schema()
    add_field(schema)
    idx = laurus.Index(schema=schema)
    idx.put_document("low", {"embedding": [-100.0, -100.0]})
    idx.put_document("high", {"embedding": [100.0, 100.0]})
    idx.put_document("a", {"embedding": [0.2, 0.0]})
    idx.put_document("b", {"embedding": [0.25, 0.0]})
    idx.commit()

    results = idx.search(laurus.VectorQuery("embedding", [0.25, 0.0]), limit=2)
    assert [r.id for r in results] == ["b", "a"]


def test_float16_hnsw_field_ranks_nearest_first():
    _assert_float16_ranks_exact_match_first(
        lambda s: s.add_hnsw_field(
            "embedding", dimension=2, distance="euclidean", quantizer="float16"
        )
    )


def test_float16_flat_field_ranks_nearest_first():
    _assert_float16_ranks_exact_match_first(
        lambda s: s.add_flat_field(
            "embedding", dimension=2, distance="euclidean", quantizer="float16"
        )
    )


# ---------------------------------------------------------------------------
# Builder-level validation (incoherent configs are rejected up front)
# ---------------------------------------------------------------------------


def test_pq_rejected_for_flat_field():
    schema = laurus.Schema()
    with pytest.raises(ValueError, match="only supported by HNSW"):
        schema.add_flat_field(
            "embedding", dimension=4, quantizer="product_quantization"
        )


def test_unknown_quantizer_rejected():
    schema = laurus.Schema()
    with pytest.raises(ValueError):
        schema.add_hnsw_field("embedding", dimension=4, quantizer="bogus")


def test_pq_requires_subvector_count():
    schema = laurus.Schema()
    with pytest.raises(ValueError):
        schema.add_hnsw_field(
            "embedding", dimension=4, quantizer="product_quantization"
        )


def test_subvector_count_rejected_for_scalar():
    schema = laurus.Schema()
    with pytest.raises(ValueError):
        schema.add_hnsw_field("embedding", dimension=4, subvector_count=2)


def test_unknown_rerank_storage_rejected():
    schema = laurus.Schema()
    with pytest.raises(ValueError):
        schema.add_hnsw_field("embedding", dimension=4, rerank_storage="bogus")
