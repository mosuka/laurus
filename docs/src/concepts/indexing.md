# Indexing

This section explains how Laurus stores and organizes data internally. Understanding the indexing layer will help you choose the right field types and tune performance.

## Topics

### [Lexical Indexing](indexing/lexical_indexing.md)

How text, numeric, and geographic fields are indexed using an inverted index. Covers:

- The inverted index structure (term dictionary, posting lists)
- BKD trees for numeric range queries
- Segment files and their formats
- BM25 scoring

### [Vector Indexing](indexing/vector_indexing.md)

How vector fields are indexed for approximate nearest neighbor search. Covers:

- Index types: Flat, HNSW, IVF
- Parameter tuning (m, ef_construction, n_clusters, n_probe)
- Distance metrics (Cosine, Euclidean, DotProduct)
- Quantization (SQ8, PQ, Float16)
