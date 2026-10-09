#!/usr/bin/env python3
"""Issue #1350: measure whether LADR-style adaptive candidate expansion is
worth adding to laurus's rescore stage.

Uses only the Python standard library plus the `laurus` binding itself (no
numpy / torch / beir package): the engine already generates both the dense
and ColBERT embeddings internally, and `filter_query` + `rescore` let us pin
MaxSim scoring to an arbitrary candidate set, so the whole measurement runs
through the public search API.

Requires a `laurus` build with the `embeddings-candle` feature, e.g.:

    cd laurus-python && VIRTUAL_ENV=$(pwd)/.venv .venv/bin/maturin develop \\
        --features embeddings-candle

Usage:

    laurus-python/.venv/bin/python scripts/beir_ladr_eval.py scifact
    laurus-python/.venv/bin/python scripts/beir_ladr_eval.py nfcorpus \\
        --corpus-size 1000 --max-queries 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import laurus

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache" / "beir"
BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_DIM = 384
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLBERT_DIM = 128

WINDOW_SIZES = [10, 20, 50, 100, 200, 500, 1000]
SEED_COUNTS = [10, 20, 50]  # first-stage seeds handed to the LADR expansion
GRAPH_NEIGHBORS = 16  # neighbors pulled per seed (level-0 default is m*2=32)
RANDOM_SEED = 1234


# ---------------------------------------------------------------------------
# Dataset fetch + parse (BEIR's own zip layout: corpus.jsonl, queries.jsonl,
# qrels/test.tsv)
# ---------------------------------------------------------------------------


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, str]  # doc_id -> "title. text"
    queries: dict[str, str]  # query_id -> text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}


def fetch_dataset(name: str) -> Path:
    dataset_dir = CACHE_DIR / name
    if (dataset_dir / "corpus.jsonl").exists():
        print(f"[beir] {name}: already present, skipping download")
        return dataset_dir

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / f"{name}.zip"
    if not zip_path.exists():
        url = f"{BEIR_BASE_URL}/{name}.zip"
        print(f"[beir] downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as resp, open(zip_path, "wb") as f:
            f.write(resp.read())

    print(f"[beir] extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CACHE_DIR)
    return dataset_dir


def load_dataset(
    name: str,
    corpus_size: int | None = None,
    max_queries: int | None = None,
    seed: int = RANDOM_SEED,
) -> BeirDataset:
    """Load a BEIR dataset, optionally down-sampled.

    Down-sampling is qrels-aware: queries are chosen first (from those with
    at least one judged document), then the corpus is the union of their
    judged documents plus enough randomly-chosen filler documents to reach
    `corpus_size`. A plain prefix-truncated corpus would strand almost every
    query's relevant documents outside the kept prefix, since qrels
    reference documents scattered throughout the file -- this keeps every
    sampled query fully evaluable while still shrinking the corpus.
    """
    dataset_dir = fetch_dataset(name)
    rng = random.Random(seed)

    corpus_all: dict[str, str] = {}
    with open(dataset_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            title = row.get("title", "") or ""
            text = row.get("text", "") or ""
            corpus_all[row["_id"]] = f"{title}. {text}".strip(". ").strip()

    queries_all: dict[str, str] = {}
    with open(dataset_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            queries_all[row["_id"]] = row["text"]

    qrels_all: dict[str, dict[str, int]] = {}
    qrels_path = dataset_dir / "qrels" / "test.tsv"
    with open(qrels_path, encoding="utf-8") as f:
        next(f)  # header: query-id\tcorpus-id\tscore
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            qid, did, rel = parts
            if did not in corpus_all:
                continue
            qrels_all.setdefault(qid, {})[did] = int(rel)

    candidate_qids = [qid for qid in queries_all if qrels_all.get(qid)]
    rng.shuffle(candidate_qids)
    if max_queries is not None:
        candidate_qids = candidate_qids[:max_queries]
    queries = {qid: queries_all[qid] for qid in candidate_qids}
    qrels = {qid: qrels_all[qid] for qid in candidate_qids}

    required_doc_ids: set[str] = set()
    for q_judgments in qrels.values():
        required_doc_ids.update(q_judgments.keys())

    if corpus_size is not None and corpus_size < len(corpus_all):
        fill_candidates = [d for d in corpus_all if d not in required_doc_ids]
        rng.shuffle(fill_candidates)
        fill_needed = max(0, corpus_size - len(required_doc_ids))
        selected_ids = required_doc_ids | set(fill_candidates[:fill_needed])
        corpus = {d: corpus_all[d] for d in selected_ids}
    else:
        corpus = corpus_all

    print(
        f"[beir] {name}: {len(corpus)} docs ({len(required_doc_ids)} judged-required + fill), "
        f"{len(queries)} queries ({sum(len(v) for v in qrels.values())} qrel rows)"
    )
    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def build_index(dataset: BeirDataset, index_path: Path | None = None) -> laurus.Index:
    """Build (or, if `index_path` already holds a committed index, reopen)
    the evaluation index. ColBERT ingestion is the dominant cost of this
    whole script on a loaded machine (tens of seconds per document), so a
    persistent path lets later runs -- e.g. after fixing an eval-loop bug --
    reuse the embeddings already computed instead of redoing them."""
    if index_path is not None and (index_path / "schema.toml").exists():
        print(f"[index] reopening persisted index at {index_path} (skipping re-ingestion)")
        return laurus.Index(path=str(index_path))

    schema = laurus.Schema()
    schema.add_text_field("doc_id", analyzer="keyword")
    schema.add_text_field("all", analyzer="keyword")
    schema.add_text_field("text")
    schema.add_embedder("dense", {"type": "candle_bert", "model": DENSE_MODEL})
    schema.add_hnsw_field("dense_hnsw", DENSE_DIM, embedder="dense")
    schema.add_flat_field("dense_flat", DENSE_DIM, embedder="dense")
    schema.add_embedder("colbert", {"type": "candle_colbert", "model": COLBERT_MODEL})
    schema.add_multi_vector_field("colbert", COLBERT_DIM, embedder="colbert")

    if index_path is not None:
        index_path.mkdir(parents=True, exist_ok=True)
        index = laurus.Index(path=str(index_path), schema=schema)
    else:
        index = laurus.Index(schema=schema)

    t0 = time.time()
    total = len(dataset.corpus)
    progress_every = max(1, min(64, total // 10))
    batch = []
    for i, (doc_id, text) in enumerate(dataset.corpus.items()):
        batch.append(
            (
                doc_id,
                {
                    "doc_id": doc_id,
                    "all": "1",
                    "text": text,
                    "dense_hnsw": text,
                    "dense_flat": text,
                    "colbert": text,
                },
            )
        )
        if len(batch) >= 16:
            index.put_documents(batch)
            batch = []
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"[index] {i + 1}/{total} docs ({elapsed:.0f}s elapsed)")
    if batch:
        index.put_documents(batch)
    index.commit()
    print(f"[index] indexed {len(dataset.corpus)} docs in {time.time() - t0:.0f}s")
    return index


# ---------------------------------------------------------------------------
# Harness self-checks (A5): confirm the measurement's own preconditions
# ---------------------------------------------------------------------------


def self_check(index: laurus.Index, dataset: BeirDataset) -> None:
    n = len(dataset.corpus)
    ceiling = index.search(
        laurus.SearchRequest(
            query=laurus.TermQuery("all", "1"),
            limit=n,
            rescore=laurus.LateInteractionRescore("colbert", "self-check", window_size=n),
        )
    )
    if len(ceiling) != n:
        raise AssertionError(
            f"self-check failed: ceiling query returned {len(ceiling)} hits, expected {n} "
            "(colbert field coverage is not 100%, or match-all field is broken)"
        )
    print(f"[self-check] ceiling condition returns exactly {n} hits: OK")


# ---------------------------------------------------------------------------
# Conditions C0-C5
# ---------------------------------------------------------------------------


@dataclass
class Ranking:
    doc_ids: list[str]
    # cost accounting
    scored_tokens: int = 0  # Sigma |D| x |Q| approximation (scored_docs x query_len proxy)
    first_stage_depth: int = 0
    graph_lookups: int = 0
    candidate_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.candidate_size = len(self.doc_ids)


def _query_len_tokens(text: str) -> int:
    # Cheap proxy: whitespace tokens, capped like the ColBERT query_maxlen default (32).
    return min(32, max(1, len(text.split())))


def condition_c0_ceiling(index: laurus.Index, query_text: str, corpus_size: int) -> Ranking:
    req = laurus.SearchRequest(
        query=laurus.TermQuery("all", "1"),
        limit=corpus_size,
        rescore=laurus.LateInteractionRescore("colbert", query_text, window_size=corpus_size),
    )
    results = index.search(req)
    qlen = _query_len_tokens(query_text)
    return Ranking(
        doc_ids=[r.id for r in results],
        scored_tokens=corpus_size * qlen,
        first_stage_depth=corpus_size,
        graph_lookups=0,
    )


def build_lexical_query(query_text: str) -> laurus.BooleanQuery:
    bq = laurus.BooleanQuery()
    for term in query_text.lower().split():
        bq.should(laurus.TermQuery("text", term))
    return bq


def run_condition_c1(index: laurus.Index, query_text: str, window: int) -> Ranking:
    req = laurus.SearchRequest(
        lexical_query=build_lexical_query(query_text),
        vector_query=laurus.VectorTextQuery("dense_hnsw", query_text),
        fusion=laurus.RRF(k=60.0),
        limit=window,
        rescore=laurus.LateInteractionRescore("colbert", query_text, window_size=window),
    )
    results = index.search(req)
    qlen = _query_len_tokens(query_text)
    return Ranking(
        doc_ids=[r.id for r in results],
        scored_tokens=min(window, len(results)) * qlen,
        first_stage_depth=window,
        graph_lookups=0,
    )


# Corpus-graph lookups are query-independent (the neighbors of document X
# never change), so they are memoized across the *entire* run, not just
# within one query. Without this, the original version re-ran a fresh
# nearest-neighbor search for every (query, seed) pair -- the dominant cost
# of the whole evaluation, since every call re-embeds the seed's text.
_graph_cache: dict[tuple[str, str], list[str]] = {}


def _graph_field_query(field: str, doc_id: str, doc_text: str, m: int) -> list[str]:
    """Top-M neighbor doc ids of `doc_id`, found by re-querying the dense
    field with the document's own text (stand-in for the real per-segment
    HNSW neighbor list, which has no public resolution API yet)."""
    key = (field, doc_id)
    cached = _graph_cache.get(key)
    if cached is not None:
        return cached
    results = index_singleton.search(
        laurus.VectorTextQuery(field, doc_text),
        limit=m + 1,  # a document is its own nearest neighbor
    )
    neighbors = [r.id for r in results if r.id != doc_id][:m]
    _graph_cache[key] = neighbors
    return neighbors


def _hybrid_seed_ranking(index: laurus.Index, query_text: str, max_seeds: int) -> list[str]:
    """The hybrid first-stage ranking (no rescore), once per query at the
    largest seed count needed; every smaller `seeds` value is a prefix of
    this same ranking, so conditions must slice it instead of re-querying
    per seed count (previously 3x redundant hybrid searches per query)."""
    seed_req = laurus.SearchRequest(
        lexical_query=build_lexical_query(query_text),
        vector_query=laurus.VectorTextQuery("dense_hnsw", query_text),
        fusion=laurus.RRF(k=60.0),
        limit=max_seeds,
    )
    return [r.id for r in index.search(seed_req)]


def run_condition_c2_ladr(
    index: laurus.Index,
    dataset: BeirDataset,
    seed_ranking: list[str],
    query_text: str,
    seeds: int,
    graph_field: str,
    neighbors_per_seed: int,
) -> Ranking:
    seed_hits = seed_ranking[:seeds]

    # Expand: union of seeds and each seed's graph neighbors.
    candidate_set = set(seed_hits)
    graph_lookups = 0
    for doc_id in seed_hits:
        neighbors = _graph_field_query(graph_field, doc_id, dataset.corpus[doc_id], neighbors_per_seed)
        candidate_set.update(neighbors)
        graph_lookups += 1

    return _score_candidate_set(
        index, query_text, candidate_set, first_stage_depth=seeds, graph_lookups=graph_lookups
    )


def run_condition_c3_placebo(
    index: laurus.Index,
    dataset: BeirDataset,
    seed_ranking: list[str],
    query_text: str,
    seeds: int,
    candidate_count: int,
    rng: random.Random,
) -> Ranking:
    seed_hits = seed_ranking[:seeds]
    candidate_set = set(seed_hits)
    all_doc_ids = list(dataset.corpus.keys())
    while len(candidate_set) < candidate_count:
        candidate_set.add(rng.choice(all_doc_ids))

    return _score_candidate_set(index, query_text, candidate_set, first_stage_depth=seeds, graph_lookups=0)


def _score_candidate_set(
    index: laurus.Index,
    query_text: str,
    candidate_set: set[str],
    first_stage_depth: int,
    graph_lookups: int,
) -> Ranking:
    filter_bq = laurus.BooleanQuery()
    for doc_id in candidate_set:
        filter_bq.should(laurus.TermQuery("doc_id", doc_id))
    n = len(candidate_set)
    req = laurus.SearchRequest(
        query=laurus.TermQuery("all", "1"),
        filter_query=filter_bq,
        limit=n,
        rescore=laurus.LateInteractionRescore("colbert", query_text, window_size=n),
    )
    results = index.search(req)
    if len(results) != n:
        raise AssertionError(
            f"self-check failed: candidate-set query returned {len(results)} hits, expected {n}"
        )
    qlen = _query_len_tokens(query_text)
    return Ranking(
        doc_ids=[r.id for r in results],
        scored_tokens=n * qlen,
        first_stage_depth=first_stage_depth,
        graph_lookups=graph_lookups,
    )


# `index_singleton` lets `_graph_field_query` reuse the module-level index
# without threading it through every call; set once per dataset run.
index_singleton: laurus.Index | None = None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def ceiling_recovery_at_k(ranking_ids: list[str], ceiling_ids: list[str], k: int = 10) -> float:
    ceiling_top_k = set(ceiling_ids[:k])
    if not ceiling_top_k:
        return float("nan")
    hit = len(set(ranking_ids[:k]) & ceiling_top_k)
    return hit / len(ceiling_top_k)


def rbo(list_a: list[str], list_b: list[str], p: float = 0.9, k: int = 10) -> float:
    """Rank-biased overlap (Webber et al. 2010), truncated at k."""
    a_set: set[str] = set()
    b_set: set[str] = set()
    total = 0.0
    for d in range(1, k + 1):
        if d - 1 < len(list_a):
            a_set.add(list_a[d - 1])
        if d - 1 < len(list_b):
            b_set.add(list_b[d - 1])
        overlap = len(a_set & b_set)
        total += (overlap / d) * (p ** (d - 1))
    return (1 - p) * total


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranking_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    gains = [qrels.get(doc_id, 0) for doc_id in ranking_ids[:k]]
    dcg = dcg_at_k(gains, k)
    ideal = sorted(qrels.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else float("nan")


def recall_at_k(ranking_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return float("nan")
    hit = len(set(ranking_ids[:k]) & relevant)
    return hit / len(relevant)


def unjudged_at_k(ranking_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    top_k = ranking_ids[:k]
    if not top_k:
        return float("nan")
    unjudged = sum(1 for doc_id in top_k if doc_id not in qrels)
    return unjudged / len(top_k)


def paired_bootstrap_ci(
    a: list[float], b: list[float], n_resamples: int = 10000, seed: int = RANDOM_SEED
) -> tuple[float, float, float]:
    """Mean paired difference (a - b) and its 95% bootstrap CI, resampling
    query indices with replacement. NaNs (an undefined metric for a query
    with no qrels at that cutoff) are dropped pairwise."""
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    diffs = [x - y for x, y in pairs]
    mean_diff = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    resampled_means = []
    for _ in range(n_resamples):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(resample) / n)
    resampled_means.sort()
    lo = resampled_means[int(0.025 * n_resamples)]
    hi = resampled_means[min(n_resamples - 1, int(0.975 * n_resamples))]
    return mean_diff, lo, hi


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class ConditionResult:
    label: str
    ceiling_recovery_10: list[float] = field(default_factory=list)
    rbo_10: list[float] = field(default_factory=list)
    ndcg_10: list[float] = field(default_factory=list)
    recall_10: list[float] = field(default_factory=list)
    recall_100: list[float] = field(default_factory=list)
    unjudged_10: list[float] = field(default_factory=list)
    scored_tokens: list[int] = field(default_factory=list)
    first_stage_depth: list[int] = field(default_factory=list)
    graph_lookups: list[int] = field(default_factory=list)
    candidate_size: list[int] = field(default_factory=list)

    def record(self, ranking: Ranking, ceiling_ids: list[str], qrels: dict[str, int]) -> None:
        self.ceiling_recovery_10.append(ceiling_recovery_at_k(ranking.doc_ids, ceiling_ids, 10))
        self.rbo_10.append(rbo(ranking.doc_ids, ceiling_ids, k=10))
        self.ndcg_10.append(ndcg_at_k(ranking.doc_ids, qrels, 10))
        self.recall_10.append(recall_at_k(ranking.doc_ids, qrels, 10))
        self.recall_100.append(recall_at_k(ranking.doc_ids, qrels, 100))
        self.unjudged_10.append(unjudged_at_k(ranking.doc_ids, qrels, 10))
        self.scored_tokens.append(ranking.scored_tokens)
        self.first_stage_depth.append(ranking.first_stage_depth)
        self.candidate_size.append(ranking.candidate_size)
        self.graph_lookups.append(ranking.graph_lookups)

    def mean(self, attr: str) -> float:
        values = [v for v in getattr(self, attr) if not math.isnan(v)]
        return sum(values) / len(values) if values else float("nan")

    def summary_row(self) -> dict[str, float | str]:
        return {
            "condition": self.label,
            "ceiling_recovery@10": round(self.mean("ceiling_recovery_10"), 4),
            "rbo@10": round(self.mean("rbo_10"), 4),
            "ndcg@10": round(self.mean("ndcg_10"), 4),
            "recall@10": round(self.mean("recall_10"), 4),
            "recall@100": round(self.mean("recall_100"), 4),
            "unjudged@10": round(self.mean("unjudged_10"), 4),
            "avg_scored_tokens": round(self.mean("scored_tokens"), 1),
            "avg_first_stage_depth": round(self.mean("first_stage_depth"), 1),
            "avg_graph_lookups": round(self.mean("graph_lookups"), 1),
            "avg_candidate_size": round(self.mean("candidate_size"), 1),
        }


def main() -> None:
    global index_singleton

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["scifact", "nfcorpus"])
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=None,
        help="down-sample the corpus to about this many docs (qrels-aware; see load_dataset)",
    )
    parser.add_argument("--max-queries", type=int, default=None, help="cap number of queries evaluated")
    parser.add_argument("--out", type=str, default=None, help="write results table as JSON to this path")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="use an ephemeral in-memory index instead of caching it under .cache/beir/ "
        "(every run re-embeds the whole corpus)",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, corpus_size=args.corpus_size, max_queries=args.max_queries)
    if not dataset.queries:
        print("[error] no judged queries found", file=sys.stderr)
        sys.exit(1)

    query_ids = list(dataset.queries.keys())
    index_path = None
    if not args.no_persist:
        cs_tag = args.corpus_size if args.corpus_size is not None else "full"
        mq_tag = args.max_queries if args.max_queries is not None else "all"
        index_path = CACHE_DIR / args.dataset / f"index_cs{cs_tag}_mq{mq_tag}_seed{RANDOM_SEED}"
    index = build_index(dataset, index_path=index_path)
    index_singleton = index
    self_check(index, dataset)

    corpus_size = len(dataset.corpus)
    rng = random.Random(RANDOM_SEED)

    results: dict[str, ConditionResult] = {}

    def result_for(label: str) -> ConditionResult:
        return results.setdefault(label, ConditionResult(label=label))

    for qi, qid in enumerate(query_ids):
        query_text = dataset.queries[qid]
        qrels = dataset.qrels.get(qid, {})

        ceiling = condition_c0_ceiling(index, query_text, corpus_size)
        result_for("C0 ceiling (exhaustive MaxSim)").record(ceiling, ceiling.doc_ids, qrels)

        for window in WINDOW_SIZES:
            if window > corpus_size:
                continue
            c1 = run_condition_c1(index, query_text, window)
            result_for(f"C1 current (W={window})").record(c1, ceiling.doc_ids, qrels)

        usable_seed_counts = [s for s in SEED_COUNTS if s <= corpus_size]
        seed_ranking = (
            _hybrid_seed_ranking(index, query_text, max(usable_seed_counts))
            if usable_seed_counts
            else []
        )
        for seeds in usable_seed_counts:
            c2 = run_condition_c2_ladr(
                index, dataset, seed_ranking, query_text, seeds, "dense_hnsw", GRAPH_NEIGHBORS
            )
            result_for(f"C2 LADR ANN-graph (S={seeds},M={GRAPH_NEIGHBORS})").record(
                c2, ceiling.doc_ids, qrels
            )

            # Matched-cost control: C1 (plain window widening) at exactly
            # |C2's candidate set| for *this query* -- the direct go/no-go
            # competitor, not an interpolation across the WINDOW_SIZES sweep.
            c1_matched = run_condition_c1(index, query_text, c2.candidate_size)
            result_for(f"C1 matched-cost (S={seeds})").record(c1_matched, ceiling.doc_ids, qrels)

            c3 = run_condition_c3_placebo(
                index, dataset, seed_ranking, query_text, seeds, len(c2.doc_ids), rng
            )
            result_for(f"C3 placebo (S={seeds})").record(c3, ceiling.doc_ids, qrels)

            c5 = run_condition_c2_ladr(
                index, dataset, seed_ranking, query_text, seeds, "dense_flat", GRAPH_NEIGHBORS
            )
            result_for(f"C5 LADR exact-kNN-graph (S={seeds},M={GRAPH_NEIGHBORS})").record(
                c5, ceiling.doc_ids, qrels
            )

        # C4 positive control: BM25-only first stage, small window.
        c4 = run_condition_c1_lexical_only(index, query_text, window=10)
        result_for("C4 positive control (BM25-only, W=10)").record(c4, ceiling.doc_ids, qrels)

        print(f"[eval] {qi + 1}/{len(query_ids)} queries done")

    print(f"\n=== {args.dataset}: {len(query_ids)} queries, {corpus_size} docs ===\n")
    rows = [r.summary_row() for r in results.values()]
    header = list(rows[0].keys())
    widths = {h: max(len(h), *(len(str(row[h])) for row in rows)) for h in header}
    print(" | ".join(h.ljust(widths[h]) for h in header))
    print("-+-".join("-" * widths[h] for h in header))
    for row in rows:
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in header))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[beir] wrote {args.out}")

    print(
        "\n=== go/no-go: C2 (graph expansion) vs. C1 matched-cost "
        "(plain window widening) and C3 (placebo), paired by query ===\n"
    )
    print(f"{'seeds':>6} | {'C2-vs-C1matched diff':>22} | {'95% CI':>22} | {'C2-vs-C3 diff':>14} | {'95% CI':>22}")
    for seeds in usable_seed_counts:
        c2_key = f"C2 LADR ANN-graph (S={seeds},M={GRAPH_NEIGHBORS})"
        c1m_key = f"C1 matched-cost (S={seeds})"
        c3_key = f"C3 placebo (S={seeds})"
        if c2_key not in results:
            continue
        d1, lo1, hi1 = paired_bootstrap_ci(
            results[c2_key].ceiling_recovery_10, results[c1m_key].ceiling_recovery_10
        )
        d2, lo2, hi2 = paired_bootstrap_ci(
            results[c2_key].ceiling_recovery_10, results[c3_key].ceiling_recovery_10
        )
        print(
            f"{seeds:>6} | {d1:>22.4f} | [{lo1:.4f}, {hi1:.4f}]{'':<6} | "
            f"{d2:>14.4f} | [{lo2:.4f}, {hi2:.4f}]"
        )
    print(
        "\nGo/no-go rule (pre-registered): go requires the C2-vs-C1matched CI lower\n"
        "bound above +0.02 AND the C2-vs-C3 CI lower bound above 0, at matched cost."
    )


def run_condition_c1_lexical_only(index: laurus.Index, query_text: str, window: int) -> Ranking:
    """C4 positive control: an artificially weak (BM25-only) first stage.
    If expansion can't beat widening W here, the harness itself is
    suspect -- a real, strong effect should show up most easily where the
    first stage is worst."""
    req = laurus.SearchRequest(
        lexical_query=build_lexical_query(query_text),
        limit=window,
        rescore=laurus.LateInteractionRescore("colbert", query_text, window_size=window),
    )
    results = index.search(req)
    qlen = _query_len_tokens(query_text)
    return Ranking(
        doc_ids=[r.id for r in results],
        scored_tokens=min(window, len(results)) * qlen,
        first_stage_depth=window,
        graph_lookups=0,
    )


if __name__ == "__main__":
    main()
