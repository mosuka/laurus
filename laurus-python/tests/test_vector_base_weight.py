"""Tests for the `base_weight` vector field option (Issue #1084).

Confirms `base_weight` reaches the Rust core and actually affects search
scores, using a deterministic observable rather than only that search
succeeds: two HNSW fields holding the SAME query-matching vector, with
different `base_weight`, must produce scores whose ratio tracks the
`base_weight` ratio.
"""

import laurus


def test_base_weight_defaults_to_one():
    schema = laurus.Schema()
    schema.add_hnsw_field("embedding", dimension=4)
    idx = laurus.Index(schema=schema)
    idx.put_document("doc1", {"embedding": [1.0, 0.0, 0.0, 0.0]})
    idx.commit()

    results = idx.search(laurus.VectorQuery("embedding", [1.0, 0.0, 0.0, 0.0]), limit=1)
    assert len(results) == 1

    schema2 = laurus.Schema()
    schema2.add_hnsw_field("embedding", dimension=4, base_weight=1.0)
    idx2 = laurus.Index(schema=schema2)
    idx2.put_document("doc1", {"embedding": [1.0, 0.0, 0.0, 0.0]})
    idx2.commit()

    results2 = idx2.search(laurus.VectorQuery("embedding", [1.0, 0.0, 0.0, 0.0]), limit=1)
    assert abs(results[0].score - results2[0].score) < 1e-6, (
        "an unset base_weight must score identically to an explicit 1.0"
    )


def test_base_weight_affects_vector_score():
    same_vector = [1.0, 0.0, 0.0, 0.0]

    schema_a = laurus.Schema()
    schema_a.add_hnsw_field("vec_a", dimension=4, base_weight=1.0)
    idx_a = laurus.Index(schema=schema_a)
    idx_a.put_document("doc1", {"vec_a": same_vector})
    idx_a.commit()
    score_a = idx_a.search(laurus.VectorQuery("vec_a", same_vector), limit=1)[0].score

    schema_b = laurus.Schema()
    schema_b.add_hnsw_field("vec_b", dimension=4, base_weight=3.0)
    idx_b = laurus.Index(schema=schema_b)
    idx_b.put_document("doc1", {"vec_b": same_vector})
    idx_b.commit()
    score_b = idx_b.search(laurus.VectorQuery("vec_b", same_vector), limit=1)[0].score

    assert score_b > score_a, (
        f"base_weight=3.0 must score higher than base_weight=1.0: a={score_a}, b={score_b}"
    )
    ratio = score_b / score_a
    assert abs(ratio - 3.0) < 0.05, f"score ratio must track base_weight ratio (~3.0), got {ratio}"


def test_non_positive_base_weight_falls_back_to_one():
    same_vector = [1.0, 0.0, 0.0, 0.0]

    schema_zero = laurus.Schema()
    schema_zero.add_hnsw_field("embedding", dimension=4, base_weight=0.0)
    idx_zero = laurus.Index(schema=schema_zero)
    idx_zero.put_document("doc1", {"embedding": same_vector})
    idx_zero.commit()
    score_zero = idx_zero.search(laurus.VectorQuery("embedding", same_vector), limit=1)[0].score

    schema_one = laurus.Schema()
    schema_one.add_hnsw_field("embedding", dimension=4, base_weight=1.0)
    idx_one = laurus.Index(schema=schema_one)
    idx_one.put_document("doc1", {"embedding": same_vector})
    idx_one.commit()
    score_one = idx_one.search(laurus.VectorQuery("embedding", same_vector), limit=1)[0].score

    assert abs(score_zero - score_one) < 1e-6, (
        "base_weight=0.0 must clamp to 1.0 rather than zeroing the score"
    )
