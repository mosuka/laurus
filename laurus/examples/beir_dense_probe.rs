//! Issue #623: BEIR nDCG@10 and Recall@10 of dense vector storage kinds
//! through the engine.
//!
//! Encodes a BEIR corpus once with `sentence-transformers/all-MiniLM-L6-v2`
//! (384 dimensions, L2-normalized), caches the f32 vectors on disk, and
//! ingests the same vectors into one engine with one Cosine field per
//! configuration under test (Flat and HNSW, `Scalar8Bit` and `Float16`, and
//! a `Scalar8Bit` Flat field with an f32 rerank sidecar). Every field sees
//! identical vectors and the same commit cadence, so the only difference is
//! the stored element kind on the production write and search paths.
//!
//! Reported per field:
//!
//! - **nDCG@10** against the qrels, with the paired per-query difference
//!   from exact f32 search (computed in this probe, independent of the
//!   engine) and its 95% bootstrap confidence interval.
//! - **Recall@10 vs f32**: overlap of the field's top 10 with the exact f32
//!   top 10.
//! - **bytes per vector** of the field's segment files (and sidecars).
//! - **median query latency**, wall clock; only comparable within one run.
//!
//! nDCG@10 follows trec_eval's `ndcg_cut_10` as BEIR runs it: linear gain,
//! `1 / log2(rank + 1)` discount, ideal DCG over every judged document with
//! relevance > 0, ties broken by doc id descending, and results whose doc
//! id equals the query id dropped (BEIR's `ignore_identical_ids`).
//!
//! # Usage
//!
//! ```sh
//! ./scripts/fetch-beir.sh scifact
//! cargo run --release --features embeddings-candle \
//!     --example beir_dense_probe -- --dataset scifact
//! ```
//!
//! The first run downloads the model and encodes the corpus; the vectors
//! are cached under `.cache/beir/<dataset>/` and reused by later runs.
//!
//! Other flags:
//!
//! - `--data-dir <dir>`: where the datasets live (default `<repo>/.cache/beir`).
//! - `--commit-every <n>`: commit after every `n` documents (default 500).
//! - `--storage memory|file:<dir>`: `memory` (the default) loads segments
//!   eagerly; `file:<dir>` uses an mmap-backed `FileStorage` in `<dir>`
//!   (which must not exist yet), the default deployment, where Float16 rows
//!   are read in place.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use laurus::embedding::candle_bert_embedder::CandleBertOptions;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::{FlatOption, HnswOption, Vector};
use laurus::{
    CandleBertEmbedder, DataValue, DistanceMetric, Document, EmbedInput, Embedder, Engine,
    FieldOption, QuantizationMethod, QueryVector, Schema, SearchRequestBuilder, Storage,
    StorageConfig, StorageFactory, VectorSearchQuery,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

type AnyResult<T> = std::result::Result<T, Box<dyn Error>>;

/// sentence-transformers/all-MiniLM-L6-v2 at a fixed commit.
const MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
const REVISION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
const DIMENSION: usize = 384;

const NDCG_K: usize = 10;
/// Rerank window for the sidecar field: `top_k × factor` candidates.
const RERANK_FACTOR: usize = 4;
const BOOTSTRAP_RESAMPLES: usize = 10_000;
const BOOTSTRAP_SEED: u64 = 623;
const ENCODE_CHUNK: usize = 64;
const CACHE_MAGIC: &[u8; 4] = b"LDV1";

/// One field under test.
#[derive(Clone, Copy)]
struct Kind {
    name: &'static str,
    hnsw: bool,
    quantizer: QuantizationMethod,
    rerank: bool,
}

const KINDS: [Kind; 5] = [
    Kind {
        name: "flat_sq8",
        hnsw: false,
        quantizer: QuantizationMethod::Scalar8Bit,
        rerank: false,
    },
    Kind {
        name: "flat_sq8_rerank",
        hnsw: false,
        quantizer: QuantizationMethod::Scalar8Bit,
        rerank: true,
    },
    Kind {
        name: "flat_f16",
        hnsw: false,
        quantizer: QuantizationMethod::Float16,
        rerank: false,
    },
    Kind {
        name: "hnsw_sq8",
        hnsw: true,
        quantizer: QuantizationMethod::Scalar8Bit,
        rerank: false,
    },
    Kind {
        name: "hnsw_f16",
        hnsw: true,
        quantizer: QuantizationMethod::Float16,
        rerank: false,
    },
];

// ── Arguments ──────────────────────────────────────────────────────────────

struct Args {
    dataset: String,
    data_dir: PathBuf,
    commit_every: usize,
    storage: Option<PathBuf>,
}

fn parse_args() -> AnyResult<Args> {
    let mut args = Args {
        dataset: String::new(),
        data_dir: Path::new(env!("CARGO_MANIFEST_DIR")).join("../.cache/beir"),
        commit_every: 500,
        storage: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--dataset" => args.dataset = value()?,
            "--data-dir" => args.data_dir = PathBuf::from(value()?),
            "--commit-every" => args.commit_every = value()?.parse()?,
            "--storage" => {
                let v = value()?;
                args.storage = match v.as_str() {
                    "memory" => None,
                    _ => Some(PathBuf::from(
                        v.strip_prefix("file:")
                            .ok_or("--storage takes memory or file:<dir>")?,
                    )),
                };
            }
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if args.dataset.is_empty() {
        return Err("--dataset is required".into());
    }
    Ok(args)
}

// ── Dataset ────────────────────────────────────────────────────────────────

/// Judged relevance per query: `query id -> doc id -> relevance`.
type Qrels = HashMap<String, HashMap<String, i32>>;

/// A BEIR dataset restricted to its test split.
struct Dataset {
    doc_ids: Vec<String>,
    doc_texts: Vec<String>,
    query_ids: Vec<String>,
    query_texts: Vec<String>,
    qrels: Qrels,
}

/// Parse a BEIR qrels file (`query-id \t corpus-id \t score`, with a header
/// line).
fn parse_qrels(reader: impl BufRead) -> AnyResult<Qrels> {
    let mut qrels = Qrels::new();
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with("query-id") {
            continue;
        }
        let mut cols = line.split('\t');
        let (Some(query_id), Some(doc_id), Some(score)) = (cols.next(), cols.next(), cols.next())
        else {
            return Err(format!("malformed qrels line: {line:?}").into());
        };
        qrels
            .entry(query_id.to_string())
            .or_default()
            .insert(doc_id.to_string(), score.trim().parse()?);
    }
    Ok(qrels)
}

fn load_dataset(dir: &Path) -> AnyResult<Dataset> {
    let qrels = parse_qrels(BufReader::new(File::open(dir.join("qrels/test.tsv"))?))?;
    let mut doc_ids = Vec::new();
    let mut doc_texts = Vec::new();
    for line in BufReader::new(File::open(dir.join("corpus.jsonl"))?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(&line)?;
        let field = |name: &str| row[name].as_str().unwrap_or_default().to_string();
        doc_ids.push(field("_id"));
        // `title + " " + text`, as BEIR's sentence-transformers models encode.
        doc_texts.push(
            format!("{} {}", field("title"), field("text"))
                .trim()
                .to_string(),
        );
    }
    let mut query_ids = Vec::new();
    let mut query_texts = Vec::new();
    for line in BufReader::new(File::open(dir.join("queries.jsonl"))?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(&line)?;
        let id = row["_id"].as_str().unwrap_or_default();
        if qrels.contains_key(id) {
            query_ids.push(id.to_string());
            query_texts.push(row["text"].as_str().unwrap_or_default().to_string());
        }
    }
    if query_ids.len() != qrels.len() {
        return Err(format!(
            "{} test queries have qrels but only {} are in queries.jsonl",
            qrels.len(),
            query_ids.len()
        )
        .into());
    }
    Ok(Dataset {
        doc_ids,
        doc_texts,
        query_ids,
        query_texts,
        qrels,
    })
}

// ── Vector cache ───────────────────────────────────────────────────────────

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_str(r: &mut impl Read) -> io::Result<String> {
    let mut buf = vec![0u8; read_u32(r)? as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Layout (little-endian): magic, dimension, revision, count, then per item
/// its id and `DIMENSION` floats. Written to a temporary file and renamed.
fn write_cache(path: &Path, ids: &[String], vectors: &[Vec<f32>]) -> io::Result<()> {
    let tmp = path.with_extension("bin.part");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(CACHE_MAGIC)?;
        write_u32(&mut w, DIMENSION as u32)?;
        write_str(&mut w, REVISION)?;
        write_u32(&mut w, ids.len() as u32)?;
        for (id, v) in ids.iter().zip(vectors) {
            write_str(&mut w, id)?;
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        w.flush()?;
    }
    fs::rename(tmp, path)
}

/// `Ok(None)` when the cache is missing or was written for another model,
/// revision or id list.
fn read_cache(path: &Path, ids: &[String]) -> io::Result<Option<Vec<Vec<f32>>>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut r = BufReader::new(file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != CACHE_MAGIC
        || read_u32(&mut r)? as usize != DIMENSION
        || read_str(&mut r)? != REVISION
        || read_u32(&mut r)? as usize != ids.len()
    {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(ids.len());
    let mut bytes = vec![0u8; DIMENSION * 4];
    for id in ids {
        if read_str(&mut r)? != *id {
            return Ok(None);
        }
        r.read_exact(&mut bytes)?;
        out.push(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        );
    }
    Ok(Some(out))
}

/// Cached vectors for `texts`, encoding (and caching) them on a miss.
async fn cached_vectors(
    embedder: &mut Option<CandleBertEmbedder>,
    path: &Path,
    ids: &[String],
    texts: &[String],
    label: &str,
) -> AnyResult<Vec<Vec<f32>>> {
    if let Some(v) = read_cache(path, ids)? {
        println!("[cache] {label}: reused {}", path.display());
        return Ok(v);
    }
    if embedder.is_none() {
        let e = tokio::task::spawn_blocking(|| {
            CandleBertEmbedder::with_options(MODEL, CandleBertOptions::default().revision(REVISION))
        })
        .await??;
        *embedder = Some(e);
    }
    let embedder = embedder.as_ref().unwrap();
    let started = Instant::now();
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(ENCODE_CHUNK) {
        let inputs: Vec<EmbedInput<'_>> = chunk.iter().map(|t| EmbedInput::Text(t)).collect();
        for v in embedder.embed_batch(&inputs).await? {
            if v.data.len() != DIMENSION {
                return Err(
                    format!("expected {DIMENSION} dimensions, got {}", v.data.len()).into(),
                );
            }
            out.push(v.data.to_vec());
        }
        println!(
            "[encode] {label}: {}/{} ({:.0}s)",
            out.len(),
            texts.len(),
            started.elapsed().as_secs_f64()
        );
    }
    write_cache(path, ids, &out)?;
    println!("[cache] {label}: wrote {}", path.display());
    Ok(out)
}

// ── Engine ─────────────────────────────────────────────────────────────────

fn field_option(kind: Kind) -> FieldOption {
    let rerank_storage = kind.rerank.then_some(RerankStorageKind::F32);
    if kind.hnsw {
        FieldOption::Hnsw(HnswOption {
            dimension: DIMENSION,
            distance: DistanceMetric::Cosine,
            quantizer: kind.quantizer,
            rerank_storage,
            ..HnswOption::default()
        })
    } else {
        FieldOption::Flat(FlatOption {
            dimension: DIMENSION,
            distance: DistanceMetric::Cosine,
            quantizer: kind.quantizer,
            rerank_storage,
            ..FlatOption::default()
        })
    }
}

async fn build_engine(
    storage: Arc<dyn Storage>,
    dataset: &Dataset,
    doc_vectors: &[Vec<f32>],
    commit_every: usize,
) -> AnyResult<Engine> {
    let mut builder = Schema::builder();
    for kind in KINDS {
        builder = builder.add_field(kind.name, field_option(kind));
    }
    let engine = Engine::new(storage, builder.build()).await?;
    let started = Instant::now();
    for (i, (id, v)) in dataset.doc_ids.iter().zip(doc_vectors).enumerate() {
        let mut doc = Document::builder();
        for kind in KINDS {
            doc = doc.add_field(kind.name, DataValue::Vector(v.clone()));
        }
        engine.put_document(id, doc.build()).await?;
        if commit_every > 0 && (i + 1) % commit_every == 0 {
            engine.commit().await?;
        }
    }
    engine.commit().await?;
    println!(
        "[ingest] {} docs (commit every {commit_every}) in {:.1}s",
        dataset.doc_ids.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(engine)
}

/// Bytes of `field`'s segment files and of its rerank sidecars.
fn field_bytes(storage: &dyn Storage, field: &str) -> AnyResult<(u64, u64)> {
    let (mut segments, mut sidecars) = (0, 0);
    for name in storage.list_files()? {
        if !name.split('/').any(|part| part == field) {
            continue;
        }
        if name.ends_with(".f32") {
            sidecars += storage.file_size(&name)?;
        } else if name.ends_with(".flat") || name.ends_with(".hnsw") {
            segments += storage.file_size(&name)?;
        }
    }
    Ok((segments, sidecars))
}

async fn search(
    engine: &Engine,
    kind: Kind,
    query: &[f32],
    limit: usize,
) -> AnyResult<(Vec<(String, f32)>, f64)> {
    let mut builder = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query.to_vec()),
            weight: 1.0,
            fields: Some(vec![kind.name.to_string()]),
        }]))
        .limit(limit);
    if kind.rerank {
        builder = builder.vector_rerank_factor(RERANK_FACTOR);
    }
    let started = Instant::now();
    let results = engine.search(builder.build()).await?;
    let millis = started.elapsed().as_secs_f64() * 1e3;
    Ok((
        results.into_iter().map(|r| (r.id, r.score)).collect(),
        millis,
    ))
}

// ── Metrics ────────────────────────────────────────────────────────────────

/// trec_eval's view of a ranking: the query's own id dropped, then ordered
/// by score descending and, on ties, by doc id descending.
fn trec_order(query_id: &str, mut results: Vec<(String, f32)>) -> Vec<(String, f32)> {
    results.retain(|(doc_id, _)| doc_id != query_id);
    results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
    results
}

fn dcg(gains: impl Iterator<Item = f64>) -> f64 {
    gains
        .enumerate()
        .map(|(i, gain)| gain / ((i + 2) as f64).log2())
        .sum()
}

/// nDCG@k with linear gain over every judged document with relevance > 0.
fn ndcg_at(ranked: &[(String, f32)], judged: &HashMap<String, i32>, k: usize) -> f64 {
    let gain = |doc_id: &str| judged.get(doc_id).copied().unwrap_or(0).max(0) as f64;
    let actual = dcg(ranked.iter().take(k).map(|(doc_id, _)| gain(doc_id)));
    let mut ideal: Vec<f64> = judged
        .values()
        .filter(|&&r| r > 0)
        .map(|&r| r as f64)
        .collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let ideal = dcg(ideal.into_iter().take(k));
    if ideal > 0.0 { actual / ideal } else { 0.0 }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn median(xs: &[f64]) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
}

/// Percentile bootstrap 95% confidence interval of the mean of paired
/// per-query differences.
fn bootstrap_ci(deltas: &[f64]) -> (f64, f64) {
    if deltas.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = StdRng::seed_from_u64(BOOTSTRAP_SEED);
    let mut means: Vec<f64> = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| {
            (0..deltas.len())
                .map(|_| deltas[rng.random_range(0..deltas.len())])
                .sum::<f64>()
                / deltas.len() as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let at =
        |q: f64| means[((BOOTSTRAP_RESAMPLES as f64 * q) as usize).min(BOOTSTRAP_RESAMPLES - 1)];
    (at(0.025), at(0.975))
}

/// Exact f32 Cosine ranking of the whole corpus (the vectors are already
/// normalized, as the Cosine fields store them).
fn exact_ranking(dataset: &Dataset, docs: &[Vec<f32>], query: &[f32]) -> Vec<(String, f32)> {
    dataset
        .doc_ids
        .iter()
        .zip(docs)
        .map(|(id, d)| (id.clone(), d.iter().zip(query).map(|(a, b)| a * b).sum()))
        .collect()
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    let args = parse_args()?;
    let dir = args.data_dir.join(&args.dataset);
    let dataset = load_dataset(&dir)?;
    println!(
        "[dataset] {}: {} documents, {} queries",
        args.dataset,
        dataset.doc_ids.len(),
        dataset.query_ids.len()
    );
    let rev = &REVISION[..8];
    let mut embedder = None;
    let mut docs = cached_vectors(
        &mut embedder,
        &dir.join(format!("dense-minilm-{rev}.docs.bin")),
        &dataset.doc_ids,
        &dataset.doc_texts,
        "docs",
    )
    .await?;
    let mut queries = cached_vectors(
        &mut embedder,
        &dir.join(format!("dense-minilm-{rev}.queries.bin")),
        &dataset.query_ids,
        &dataset.query_texts,
        "queries",
    )
    .await?;
    for v in docs.iter_mut().chain(queries.iter_mut()) {
        normalize(v);
    }

    let storage = match &args.storage {
        None => StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?,
        Some(path) => {
            if path.exists() {
                return Err(format!("{} already exists", path.display()).into());
            }
            fs::create_dir_all(path)?;
            let mut config = FileStorageConfig::new(path);
            config.use_mmap = true;
            StorageFactory::create(StorageConfig::File(config))?
        }
    };
    println!(
        "[storage] {} (loading mode {:?})",
        args.storage
            .as_ref()
            .map_or("memory".to_string(), |p| format!("file:{}", p.display())),
        storage.loading_mode()
    );
    let engine = build_engine(storage.clone(), &dataset, &docs, args.commit_every).await?;

    // Request one extra hit so dropping the query's own id still leaves 10.
    let limit = NDCG_K + 1;
    let mut exact_ndcg = Vec::new();
    let mut exact_top: Vec<Vec<String>> = Vec::new();
    for (qi, query) in queries.iter().enumerate() {
        let qid = &dataset.query_ids[qi];
        let ranked = trec_order(qid, exact_ranking(&dataset, &docs, query));
        exact_ndcg.push(ndcg_at(&ranked, &dataset.qrels[qid], NDCG_K));
        exact_top.push(ranked.into_iter().take(NDCG_K).map(|(d, _)| d).collect());
    }
    println!(
        "| field | nDCG@10 | Δ vs exact f32 [95% CI] | Recall@10 vs f32 | bytes/vector (segments + sidecars) | median ms/query |"
    );
    println!("| --- | --- | --- | --- | --- | --- |");
    println!(
        "| exact f32 (probe) | {:.4} | — | 1.0000 | {} | — |",
        mean(&exact_ndcg),
        DIMENSION * 4
    );
    for kind in KINDS {
        let mut ndcg = Vec::new();
        let mut overlap = Vec::new();
        let mut millis = Vec::new();
        for (qi, query) in queries.iter().enumerate() {
            let qid = &dataset.query_ids[qi];
            let (results, ms) = search(&engine, kind, query, limit).await?;
            millis.push(ms);
            let ranked = trec_order(qid, results);
            ndcg.push(ndcg_at(&ranked, &dataset.qrels[qid], NDCG_K));
            let exact: HashSet<&String> = exact_top[qi].iter().collect();
            let hits = ranked
                .iter()
                .take(NDCG_K)
                .filter(|(d, _)| exact.contains(d))
                .count();
            overlap.push(hits as f64 / NDCG_K as f64);
        }
        let deltas: Vec<f64> = ndcg.iter().zip(&exact_ndcg).map(|(a, b)| a - b).collect();
        let (lo, hi) = bootstrap_ci(&deltas);
        let (segments, sidecars) = field_bytes(storage.as_ref(), kind.name)?;
        let n = dataset.doc_ids.len() as f64;
        println!(
            "| {} | {:.4} | {:+.4} [{:+.4}, {:+.4}] | {:.4} | {:.0} + {:.0} | {:.3} |",
            kind.name,
            mean(&ndcg),
            mean(&deltas),
            lo,
            hi,
            mean(&overlap),
            segments as f64 / n,
            sidecars as f64 / n,
            median(&millis)
        );
    }
    Ok(())
}
