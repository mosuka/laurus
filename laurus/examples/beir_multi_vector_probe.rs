//! Issue #1346: BEIR nDCG@10 of ColBERT MaxSim rescoring for every
//! `MultiVectorStorage` kind.
//!
//! Encodes a BEIR corpus once with a real ColBERT model, caches the f32
//! token vectors on disk, and ingests the same vectors into one engine that
//! has one MultiVector field per storage kind (`tokens_f32`, `tokens_f16`,
//! `tokens_int8`, `tokens_binary`). Every kind therefore sees identical
//! vectors and an identical first stage; the only difference is the on-disk
//! element kind, decoded on the production rescore path. The default
//! commit interval puts 500 documents in the first commit, enough for the
//! `Binary` field to learn its parameters there, so every segment of it is
//! `Binary`.
//!
//! Two conditions are measured:
//!
//! - **exhaustive**: every document is a first-stage candidate and MaxSim
//!   ranks the whole corpus (`window_size` = corpus size, so the corpus is
//!   capped at [`RescoreOptions::MAX_WINDOW_SIZE`]). This isolates the
//!   storage kind's effect on ranking quality.
//! - **bm25**: a BM25 first stage over the document text and a MaxSim
//!   rescore of its top [`BM25_WINDOW`], the way an application uses the
//!   field. The BM25-only ranking is reported as the baseline.
//!
//! nDCG@10 follows trec_eval's `ndcg_cut_10` as BEIR runs it (through
//! pytrec_eval): linear gain, `1 / log2(rank + 1)` discount, ideal DCG over
//! every judged document with relevance > 0 (retrieved or not), ties broken
//! by doc id descending, and results whose doc id equals the query id
//! dropped (BEIR's `ignore_identical_ids`). `--trec-run` writes the rankings
//! as TREC run files so the numbers can be re-checked with trec_eval.
//!
//! # Usage
//!
//! ```sh
//! ./scripts/fetch-beir.sh scifact
//! cargo run --release --features embeddings-candle \
//!     --example beir_multi_vector_probe -- --dataset scifact
//! ```
//!
//! The first run downloads the model and encodes the corpus; the vectors
//! are cached under `.cache/beir/<dataset>/` and reused by later runs.
//!
//! Other flags:
//!
//! - `--data-dir <dir>`: where the datasets live (default `<repo>/.cache/beir`).
//! - `--commit-every <n>`: commit after every `n` documents (default 500;
//!   `0` commits once after the last document).
//! - `--ingest-only`: build the engine, report the ingest time, and exit.
//! - `--time-encode <n>`: encode the first `n` documents, report the
//!   throughput, and exit without touching the cache.
//! - `--trec-run <dir>`: write one TREC run file per condition and kind.

use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use laurus::analysis::StandardAnalyzer;
use laurus::lexical::{BooleanQueryBuilder, TermQuery};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::{
    Analyzer, CandleColbertEmbedder, CandleColbertOptions, DistanceMetric, Document, EmbedInput,
    EmbedRole, Engine, FieldOption, LexicalSearchQuery, MultiVectorOption, MultiVectorStorage,
    RescoreOptions, Schema, SearchRequest, SearchRequestBuilder, Storage, TextOption,
    TokenEmbedder,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

type AnyResult<T> = std::result::Result<T, Box<dyn Error>>;

/// answerdotai/answerai-colbert-small-v1 at the commit the parity fixtures
/// and `tests/colbert_end_to_end_test.rs` use.
const MODEL: &str = "answerdotai/answerai-colbert-small-v1";
const REVISION: &str = "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e";
/// The model's own `artifact.metadata` values, passed explicitly so the
/// cache key does not depend on what the checkpoint happens to declare.
const QUERY_MAXLEN: usize = 32;
const DOC_MAXLEN: usize = 300;
const DIMENSION: usize = 96;

/// Storage kinds under test, F32 first: it is the reference the others
/// are compared against.
const KINDS: [MultiVectorStorage; 4] = [
    MultiVectorStorage::F32,
    MultiVectorStorage::F16,
    MultiVectorStorage::Int8,
    MultiVectorStorage::Binary,
];

const BODY: &str = "body";
/// Holds the same term in every document so one `TermQuery` matches the
/// whole corpus (the exhaustive condition's first stage).
const ALL: &str = "all";
const ALL_TERM: &str = "1";

const NDCG_K: usize = 10;
/// Candidates the BM25 condition rescores, and the depth of every ranking.
const BM25_WINDOW: usize = 100;
const BOOTSTRAP_RESAMPLES: usize = 10_000;
const BOOTSTRAP_SEED: u64 = 1346;
/// Documents per `embed_tokens` call; the embedder batches 32 at a time
/// internally, this only sets how often progress is printed.
const ENCODE_CHUNK: usize = 256;
const CACHE_MAGIC: &[u8; 4] = b"LBVC";
const CACHE_VERSION: u32 = 1;

/// Published nDCG@10 for the model (Hugging Face model card). The card
/// does not say whether it was measured with PLAID retrieval or by
/// reranking, so this is a sanity reference, not an expected value.
fn published_ndcg(dataset: &str) -> Option<f64> {
    match dataset {
        "scifact" => Some(0.7477),
        "nfcorpus" => Some(0.373),
        _ => None,
    }
}

fn field_name(kind: MultiVectorStorage) -> &'static str {
    match kind {
        MultiVectorStorage::F32 => "tokens_f32",
        MultiVectorStorage::F16 => "tokens_f16",
        MultiVectorStorage::Int8 => "tokens_int8",
        MultiVectorStorage::Binary => "tokens_binary",
    }
}

fn kind_label(kind: MultiVectorStorage) -> &'static str {
    match kind {
        MultiVectorStorage::F32 => "f32",
        MultiVectorStorage::F16 => "f16",
        MultiVectorStorage::Int8 => "int8",
        MultiVectorStorage::Binary => "binary",
    }
}

// ── Command line ───────────────────────────────────────────────────────────

struct Args {
    dataset: String,
    data_dir: PathBuf,
    commit_every: usize,
    ingest_only: bool,
    time_encode: Option<usize>,
    trec_run: Option<PathBuf>,
}

fn parse_args() -> AnyResult<Args> {
    let mut args = Args {
        dataset: String::new(),
        data_dir: Path::new(env!("CARGO_MANIFEST_DIR")).join("../.cache/beir"),
        commit_every: 500,
        ingest_only: false,
        time_encode: None,
        trec_run: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--dataset" => args.dataset = value()?,
            "--data-dir" => args.data_dir = PathBuf::from(value()?),
            "--commit-every" => args.commit_every = value()?.parse()?,
            "--ingest-only" => args.ingest_only = true,
            "--time-encode" => args.time_encode = Some(value()?.parse()?),
            "--trec-run" => args.trec_run = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if args.dataset.is_empty() {
        return Err("usage: beir_multi_vector_probe --dataset <scifact|nfcorpus> [flags]".into());
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
/// line and CRLF line endings). Every row is kept; rows with relevance ≤ 0
/// add no gain, as in trec_eval.
fn parse_qrels(reader: impl BufRead) -> AnyResult<Qrels> {
    let mut qrels = Qrels::new();
    for line in reader.lines() {
        // `lines()` strips both LF and CRLF line endings.
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

/// `title + " " + text`, trimmed: the document text BEIR's own
/// sentence-transformers models encode.
fn document_text(title: &str, text: &str) -> String {
    format!("{title} {text}").trim().to_string()
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
        doc_texts.push(document_text(&field("title"), &field("text")));
    }

    // queries.jsonl holds every split; keep the test queries, in file order.
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
    let known: std::collections::HashSet<&str> = doc_ids.iter().map(String::as_str).collect();
    let missing = qrels
        .values()
        .flat_map(|judged| judged.keys())
        .filter(|doc_id| !known.contains(doc_id.as_str()))
        .count();
    if missing > 0 {
        return Err(format!("{missing} judged documents are not in corpus.jsonl").into());
    }

    Ok(Dataset {
        doc_ids,
        doc_texts,
        query_ids,
        query_texts,
        qrels,
    })
}

// ── Token-vector cache ─────────────────────────────────────────────────────

/// What a cache file must have been encoded with to be reused.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheKey {
    model: String,
    revision: String,
    query_maxlen: u32,
    doc_maxlen: u32,
    dimension: u32,
}

impl CacheKey {
    fn current() -> Self {
        Self {
            model: MODEL.to_string(),
            revision: REVISION.to_string(),
            query_maxlen: QUERY_MAXLEN as u32,
            doc_maxlen: DOC_MAXLEN as u32,
            dimension: DIMENSION as u32,
        }
    }

    fn file_name(&self, role: &str) -> String {
        format!(
            "colbert-{}-q{}-d{}.{role}.bin",
            &self.revision[..8.min(self.revision.len())],
            self.query_maxlen,
            self.doc_maxlen
        )
    }
}

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

/// Write `vectors[i]` (row-major, `key.dimension` floats per token) for
/// `ids[i]`. Written to a temporary file and renamed, so an interrupted
/// run never leaves a truncated cache behind.
///
/// Layout (little-endian): magic, version, dimension, query_maxlen,
/// doc_maxlen, model, revision, count, then per item its id, row count and
/// rows.
fn write_cache(
    path: &Path,
    key: &CacheKey,
    ids: &[String],
    vectors: &[Vec<f32>],
) -> io::Result<()> {
    let tmp = path.with_extension("bin.part");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(CACHE_MAGIC)?;
        for v in [
            CACHE_VERSION,
            key.dimension,
            key.query_maxlen,
            key.doc_maxlen,
        ] {
            write_u32(&mut w, v)?;
        }
        write_str(&mut w, &key.model)?;
        write_str(&mut w, &key.revision)?;
        write_u32(&mut w, ids.len() as u32)?;
        for (id, rows) in ids.iter().zip(vectors) {
            write_str(&mut w, id)?;
            write_u32(&mut w, (rows.len() / key.dimension as usize) as u32)?;
            for x in rows {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        w.flush()?;
    }
    fs::rename(tmp, path)
}

/// Read a cache written by [`write_cache`]. `Ok(None)` when the file is
/// missing or was encoded with a different key or for different ids (the
/// caller then re-encodes); `Err` when it cannot be read.
fn read_cache(path: &Path, key: &CacheKey, ids: &[String]) -> io::Result<Option<Vec<Vec<f32>>>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut r = BufReader::new(file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != CACHE_MAGIC || read_u32(&mut r)? != CACHE_VERSION {
        return Ok(None);
    }
    let found = CacheKey {
        dimension: read_u32(&mut r)?,
        query_maxlen: read_u32(&mut r)?,
        doc_maxlen: read_u32(&mut r)?,
        model: read_str(&mut r)?,
        revision: read_str(&mut r)?,
    };
    if &found != key || read_u32(&mut r)? as usize != ids.len() {
        return Ok(None);
    }
    let dim = key.dimension as usize;
    let mut vectors = Vec::with_capacity(ids.len());
    let mut bytes = Vec::new();
    for id in ids {
        if read_str(&mut r)? != *id {
            return Ok(None);
        }
        let rows = read_u32(&mut r)? as usize;
        bytes.resize(rows * dim * 4, 0);
        r.read_exact(&mut bytes)?;
        vectors.push(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        );
    }
    Ok(Some(vectors))
}

// ── Encoding ───────────────────────────────────────────────────────────────

async fn load_embedder() -> AnyResult<Arc<CandleColbertEmbedder>> {
    let started = Instant::now();
    let embedder = tokio::task::spawn_blocking(|| {
        CandleColbertEmbedder::with_options(
            MODEL,
            CandleColbertOptions::default()
                .revision(REVISION)
                .query_maxlen(QUERY_MAXLEN)
                .doc_maxlen(DOC_MAXLEN),
        )
    })
    .await??;
    assert_eq!(
        embedder.token_dimension(),
        DIMENSION,
        "unexpected token dimension"
    );
    println!(
        "[encode] loaded {MODEL}@{} in {:.1}s",
        &REVISION[..8],
        started.elapsed().as_secs_f64()
    );
    Ok(Arc::new(embedder))
}

/// Encode `texts`, returning each item's token vectors flattened row-major.
async fn encode(
    embedder: &CandleColbertEmbedder,
    texts: &[String],
    role: EmbedRole,
    label: &str,
) -> AnyResult<Vec<Vec<f32>>> {
    let started = Instant::now();
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(ENCODE_CHUNK) {
        let inputs: Vec<EmbedInput<'_>> = chunk.iter().map(|t| EmbedInput::Text(t)).collect();
        for tokens in embedder.embed_tokens(&inputs, role).await? {
            out.push(tokens.iter().flat_map(|v| v.data.iter().copied()).collect());
        }
        let secs = started.elapsed().as_secs_f64();
        println!(
            "[encode] {label}: {}/{} ({secs:.0}s, {:.1}/s)",
            out.len(),
            texts.len(),
            out.len() as f64 / secs.max(1e-9)
        );
    }
    Ok(out)
}

/// Cached token vectors for `texts`, encoding (and caching) them on a miss.
async fn cached_vectors(
    embedder: &mut Option<Arc<CandleColbertEmbedder>>,
    path: &Path,
    ids: &[String],
    texts: &[String],
    role: EmbedRole,
    label: &str,
) -> AnyResult<Vec<Vec<f32>>> {
    let key = CacheKey::current();
    if let Some(vectors) = read_cache(path, &key, ids)? {
        println!("[cache] {label}: reused {}", path.display());
        return Ok(vectors);
    }
    if embedder.is_none() {
        *embedder = Some(load_embedder().await?);
    }
    let vectors = encode(embedder.as_ref().unwrap(), texts, role, label).await?;
    write_cache(path, &key, ids, &vectors)?;
    println!("[cache] {label}: wrote {}", path.display());
    Ok(vectors)
}

// ── Engine ─────────────────────────────────────────────────────────────────

fn schema() -> Schema {
    let mut builder = Schema::builder()
        .add_text_field(
            BODY,
            TextOption::default().stored(false).term_vectors(false),
        )
        .add_text_field(ALL, TextOption::default().stored(false).term_vectors(false));
    for kind in KINDS {
        builder = builder.add_field(
            field_name(kind),
            FieldOption::MultiVector(
                MultiVectorOption::new(DIMENSION)
                    .distance(DistanceMetric::Cosine)
                    .storage(kind),
            ),
        );
    }
    builder.build()
}

/// Ingest every document into a fresh engine, committing after every
/// `commit_every` documents (`0`: once at the end).
async fn build_engine(
    storage: Arc<MemoryStorage>,
    analyzer: Arc<dyn Analyzer>,
    dataset: &Dataset,
    doc_vectors: &[Vec<f32>],
    commit_every: usize,
) -> AnyResult<Engine> {
    let engine = Engine::builder(storage, schema())
        .analyzer(analyzer)
        .build()
        .await?;
    let started = Instant::now();
    for (i, ((id, text), flat)) in dataset
        .doc_ids
        .iter()
        .zip(&dataset.doc_texts)
        .zip(doc_vectors)
        .enumerate()
    {
        let rows: Vec<Vec<f32>> = flat
            .as_chunks::<DIMENSION>()
            .0
            .iter()
            .map(|row| row.to_vec())
            .collect();
        let mut doc = Document::builder()
            .add_text(BODY, text.as_str())
            .add_text(ALL, ALL_TERM);
        for kind in KINDS {
            doc = doc.add_vector_array(field_name(kind), rows.clone());
        }
        engine.put_document(id, doc.build()).await?;
        if commit_every > 0 && (i + 1) % commit_every == 0 {
            engine.commit().await?;
        }
    }
    engine.commit().await?;
    println!(
        "[ingest] {} docs (commit every {}) in {:.1}s",
        dataset.doc_ids.len(),
        if commit_every == 0 {
            "-".to_string()
        } else {
            commit_every.to_string()
        },
        started.elapsed().as_secs_f64()
    );
    Ok(engine)
}

/// Bytes of every `.mv` segment file of `field` in `storage`.
fn segment_bytes(storage: &MemoryStorage, field: &str) -> AnyResult<u64> {
    let mut total = 0;
    for name in storage.list_files()? {
        if name.ends_with(".mv") && name.contains(field) {
            total += storage.file_size(&name)?;
        }
    }
    Ok(total)
}

fn query_vectors(flat: &[f32]) -> Vec<Vector> {
    flat.as_chunks::<DIMENSION>()
        .0
        .iter()
        .map(|row| Vector::new(row.to_vec()))
        .collect()
}

/// SHOULD clauses of the query's analyzed terms; `None` when every term is
/// dropped by the analyzer (stop words only).
fn bm25_query(analyzer: &dyn Analyzer, text: &str) -> AnyResult<Option<LexicalSearchQuery>> {
    let mut builder = BooleanQueryBuilder::new();
    let mut terms = 0;
    for token in analyzer.analyze(text)? {
        builder = builder.should(Box::new(TermQuery::new(BODY, token.text)));
        terms += 1;
    }
    Ok((terms > 0).then(|| LexicalSearchQuery::Obj(Box::new(builder.build()))))
}

fn match_all() -> LexicalSearchQuery {
    LexicalSearchQuery::Obj(Box::new(TermQuery::new(ALL, ALL_TERM)))
}

/// MaxSim over the top `window` first-stage candidates on `kind`'s field.
fn late_interaction(kind: MultiVectorStorage, query: &[Vector], window: usize) -> RescoreOptions {
    RescoreOptions::late_interaction(field_name(kind), query.to_vec()).window_size(window)
}

/// The top `limit` hits of the `lexical` first stage, optionally rescored.
fn request(
    lexical: LexicalSearchQuery,
    rescore: Option<RescoreOptions>,
    limit: usize,
) -> SearchRequest {
    let builder = SearchRequestBuilder::new()
        .lexical_query(lexical)
        .limit(limit);
    match rescore {
        Some(rescore) => builder.rescore(rescore).build(),
        None => builder.build(),
    }
}

/// Run `request` and return its `(doc id, score)` ranking and wall time.
async fn search(engine: &Engine, request: SearchRequest) -> AnyResult<(Vec<(String, f32)>, f64)> {
    let started = Instant::now();
    let results = engine.search(request).await?;
    let millis = started.elapsed().as_secs_f64() * 1e3;
    Ok((
        results.into_iter().map(|r| (r.id, r.score)).collect(),
        millis,
    ))
}

// ── Metrics ────────────────────────────────────────────────────────────────

/// trec_eval's view of a ranking: results whose doc id equals the query id
/// dropped (BEIR's `ignore_identical_ids`), then ordered by score
/// descending and, on ties, by doc id descending.
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
/// A query with no relevant document scores 0, as in trec_eval.
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

/// Fraction of the relevant (relevance > 0) documents found in the top `k`.
fn recall_at(ranked: &[(String, f32)], judged: &HashMap<String, i32>, k: usize) -> f64 {
    let relevant = judged.values().filter(|&&r| r > 0).count();
    if relevant == 0 {
        return 0.0;
    }
    let found = ranked
        .iter()
        .take(k)
        .filter(|(doc_id, _)| judged.get(doc_id).is_some_and(|&r| r > 0))
        .count();
    found as f64 / relevant as f64
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
fn bootstrap_ci(deltas: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    if deltas.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| {
            (0..deltas.len())
                .map(|_| deltas[rng.random_range(0..deltas.len())])
                .sum::<f64>()
                / deltas.len() as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let at = |q: f64| means[((resamples as f64 * q) as usize).min(resamples - 1)];
    (at(0.025), at(0.975))
}

// ── Evaluation ─────────────────────────────────────────────────────────────

/// One ranking per query, for one condition and storage kind.
#[derive(Default)]
struct Run {
    rankings: Vec<Vec<(String, f32)>>,
    millis: Vec<f64>,
}

impl Run {
    /// Run `request` for query `qid` (`None`: the query has no BM25 terms,
    /// recorded as an empty ranking) and append its trec_eval-ordered
    /// ranking.
    async fn push(
        &mut self,
        engine: &Engine,
        qid: &str,
        request: Option<SearchRequest>,
    ) -> AnyResult<()> {
        let (ranked, millis) = match request {
            Some(request) => search(engine, request).await?,
            None => (Vec::new(), 0.0),
        };
        self.rankings.push(trec_order(qid, ranked));
        self.millis.push(millis);
        Ok(())
    }

    fn ndcg(&self, dataset: &Dataset) -> Vec<f64> {
        self.rankings
            .iter()
            .zip(&dataset.query_ids)
            .map(|(ranked, qid)| ndcg_at(ranked, &dataset.qrels[qid], NDCG_K))
            .collect()
    }

    fn recall(&self, dataset: &Dataset, k: usize) -> Vec<f64> {
        self.rankings
            .iter()
            .zip(&dataset.query_ids)
            .map(|(ranked, qid)| recall_at(ranked, &dataset.qrels[qid], k))
            .collect()
    }

    fn empty(&self) -> usize {
        self.rankings.iter().filter(|r| r.is_empty()).count()
    }

    fn write_trec(&self, path: &Path, dataset: &Dataset, tag: &str) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        for (ranked, qid) in self.rankings.iter().zip(&dataset.query_ids) {
            for (rank, (doc_id, score)) in ranked.iter().enumerate() {
                writeln!(w, "{qid} Q0 {doc_id} {} {score} {tag}", rank + 1)?;
            }
        }
        w.flush()
    }
}

/// The doc ids of a ranking's top [`NDCG_K`].
fn top_ids(ranked: &[(String, f32)]) -> Vec<&str> {
    ranked
        .iter()
        .take(NDCG_K)
        .map(|(doc_id, _)| doc_id.as_str())
        .collect()
}

/// Print one condition's table: per kind, nDCG@10, its paired difference
/// from F32 with a 95% CI, the queries whose nDCG@10 went up / down, the
/// queries whose top 10 (ids or order) differs from F32's, and the median
/// search time.
fn report(condition: &str, dataset: &Dataset, runs: &[Run], baseline: Option<f64>) {
    println!();
    println!("== {condition} ==");
    if let Some(bm25) = baseline {
        println!("bm25 only: nDCG@{NDCG_K} {bm25:.4}");
    }
    println!(
        "{:<6} {:>9} {:>10} {:>23} {:>9} {:>11} {:>6} {:>9}",
        "kind", "nDCG@10", "Δ vs f32", "95% CI", "+/- q", "top10≠f32", "empty", "ms/query"
    );
    let reference = runs[0].ndcg(dataset);
    for (kind, run) in KINDS.iter().zip(runs) {
        let ndcg = run.ndcg(dataset);
        let deltas: Vec<f64> = ndcg.iter().zip(&reference).map(|(a, b)| a - b).collect();
        let (lo, hi) = bootstrap_ci(&deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED);
        let better = deltas.iter().filter(|&&d| d > 0.0).count();
        let worse = deltas.iter().filter(|&&d| d < 0.0).count();
        let changed = run
            .rankings
            .iter()
            .zip(&runs[0].rankings)
            .filter(|(a, b)| top_ids(a) != top_ids(b))
            .count();
        println!(
            "{:<6} {:>9.4} {:>+10.5} {:>23} {:>9} {:>11} {:>6} {:>9.2}",
            kind_label(*kind),
            mean(&ndcg),
            mean(&deltas),
            format!("[{lo:+.5}, {hi:+.5}]"),
            format!("{better}/{worse}"),
            changed,
            run.empty(),
            median(&run.millis)
        );
    }
}

/// Every query's ranking under every condition.
struct Evaluation {
    /// MaxSim over the whole corpus, one run per kind in [`KINDS`] order.
    exhaustive: Vec<Run>,
    /// The BM25 top [`BM25_WINDOW`] rescored by MaxSim, one run per kind.
    rescored: Vec<Run>,
    /// The BM25 first stage alone.
    bm25: Run,
}

/// Run both conditions for every query and storage kind.
async fn evaluate(
    engine: &Engine,
    analyzer: &dyn Analyzer,
    dataset: &Dataset,
    query_vectors_flat: &[Vec<f32>],
) -> AnyResult<Evaluation> {
    let n = dataset.doc_ids.len();
    let mut eval = Evaluation {
        exhaustive: KINDS.iter().map(|_| Run::default()).collect(),
        rescored: KINDS.iter().map(|_| Run::default()).collect(),
        bm25: Run::default(),
    };
    for (q, (qid, text)) in dataset
        .query_ids
        .iter()
        .zip(&dataset.query_texts)
        .enumerate()
    {
        let vectors = query_vectors(&query_vectors_flat[q]);
        // `LexicalSearchQuery` is not `Clone`; build one per request.
        let bm25 = || bm25_query(analyzer, text);

        let first_stage = bm25()?.map(|lexical| request(lexical, None, BM25_WINDOW));
        eval.bm25.push(engine, qid, first_stage).await?;

        // Rotate the kind order per query so no kind always runs first.
        for step in 0..KINDS.len() {
            let k = (q + step) % KINDS.len();
            let exhaustive = request(
                match_all(),
                Some(late_interaction(KINDS[k], &vectors, n)),
                BM25_WINDOW,
            );
            eval.exhaustive[k]
                .push(engine, qid, Some(exhaustive))
                .await?;

            let rescored = bm25()?.map(|lexical| {
                request(
                    lexical,
                    Some(late_interaction(KINDS[k], &vectors, BM25_WINDOW)),
                    BM25_WINDOW,
                )
            });
            eval.rescored[k].push(engine, qid, rescored).await?;
        }
        if (q + 1) % 50 == 0 {
            println!("[eval] {}/{} queries", q + 1, dataset.query_ids.len());
        }
    }
    Ok(eval)
}

/// Print each kind's row size, payload bytes (`row_bytes × token count`)
/// and the actual bytes of its segment files.
fn print_sizes(storage: &MemoryStorage, total_rows: usize) -> AnyResult<()> {
    println!();
    println!(
        "{:<6} {:>10} {:>14} {:>14} {:>10}",
        "kind", "bytes/row", "payload bytes", "segment bytes", "ratio"
    );
    for kind in KINDS {
        let row_bytes = kind.row_bytes(DIMENSION);
        println!(
            "{:<6} {:>10} {:>14} {:>14} {:>9.2}x",
            kind_label(kind),
            row_bytes,
            total_rows * row_bytes,
            segment_bytes(storage, field_name(kind))?,
            MultiVectorStorage::F32.row_bytes(DIMENSION) as f64 / row_bytes as f64
        );
    }
    Ok(())
}

/// Write one TREC run file per condition and kind into `out`.
fn write_runs(out: &Path, name: &str, dataset: &Dataset, eval: &Evaluation) -> io::Result<()> {
    fs::create_dir_all(out)?;
    let path = |condition: &str, kind: &str| out.join(format!("{name}-{condition}-{kind}.run"));
    eval.bm25
        .write_trec(&path("bm25", "none"), dataset, "bm25")?;
    for (kind, (a, b)) in KINDS.iter().zip(eval.exhaustive.iter().zip(&eval.rescored)) {
        let label = kind_label(*kind);
        a.write_trec(&path("exhaustive", label), dataset, label)?;
        b.write_trec(&path("bm25-rescore", label), dataset, label)?;
    }
    Ok(())
}

/// `--time-encode`: encode the first `limit` documents and report the
/// throughput, without touching the cache.
async fn time_encode(dataset: &Dataset, limit: usize) -> AnyResult<()> {
    let n = dataset.doc_texts.len();
    let embedder = load_embedder().await?;
    let texts = &dataset.doc_texts[..limit.min(n)];
    let started = Instant::now();
    let vectors = encode(&embedder, texts, EmbedRole::Document, "documents").await?;
    let secs = started.elapsed().as_secs_f64();
    let tokens: usize = vectors.iter().map(|v| v.len() / DIMENSION).sum();
    println!(
        "[time-encode] {} docs, {tokens} tokens in {secs:.1}s: {:.1} docs/s, {:.0} tokens/s; \
         full corpus ≈ {:.1} min",
        texts.len(),
        texts.len() as f64 / secs,
        tokens as f64 / secs,
        n as f64 / (texts.len() as f64 / secs) / 60.0
    );
    Ok(())
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    let args = parse_args()?;
    let dir = args.data_dir.join(&args.dataset);
    if !dir.join("corpus.jsonl").exists() {
        return Err(format!(
            "{} not found; run ./scripts/fetch-beir.sh {}",
            dir.display(),
            args.dataset
        )
        .into());
    }
    let dataset = load_dataset(&dir)?;
    let n = dataset.doc_ids.len();
    println!(
        "[beir] {}: {n} documents, {} test queries, {} judgments",
        args.dataset,
        dataset.query_ids.len(),
        dataset.qrels.values().map(HashMap::len).sum::<usize>()
    );

    if let Some(limit) = args.time_encode {
        return time_encode(&dataset, limit).await;
    }
    if n > RescoreOptions::MAX_WINDOW_SIZE {
        return Err(format!(
            "the exhaustive condition needs window_size = {n}, above MAX_WINDOW_SIZE ({})",
            RescoreOptions::MAX_WINDOW_SIZE
        )
        .into());
    }

    let key = CacheKey::current();
    let mut embedder = None;
    let doc_vectors = cached_vectors(
        &mut embedder,
        &dir.join(key.file_name("docs")),
        &dataset.doc_ids,
        &dataset.doc_texts,
        EmbedRole::Document,
        "documents",
    )
    .await?;
    let query_vectors_flat = cached_vectors(
        &mut embedder,
        &dir.join(key.file_name("queries")),
        &dataset.query_ids,
        &dataset.query_texts,
        EmbedRole::Query,
        "queries",
    )
    .await?;
    drop(embedder);

    let rows_per_doc: Vec<usize> = doc_vectors.iter().map(|v| v.len() / DIMENSION).collect();
    let total_rows: usize = rows_per_doc.iter().sum();
    assert!(
        rows_per_doc.iter().all(|&r| (1..=DOC_MAXLEN).contains(&r)),
        "document token count out of range"
    );
    assert!(
        query_vectors_flat
            .iter()
            .all(|v| v.len() == QUERY_MAXLEN * DIMENSION),
        "every query should have exactly {QUERY_MAXLEN} vectors"
    );
    println!(
        "[beir] {total_rows} document tokens ({:.1} per document, max {})",
        total_rows as f64 / n as f64,
        rows_per_doc.iter().max().copied().unwrap_or(0)
    );

    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let engine = build_engine(
        storage.clone(),
        analyzer.clone(),
        &dataset,
        &doc_vectors,
        args.commit_every,
    )
    .await?;
    drop(doc_vectors);

    print_sizes(&storage, total_rows)?;
    if args.ingest_only {
        return Ok(());
    }

    // Self-check: the exhaustive first stage must hand MaxSim every document.
    let everything = request(
        match_all(),
        Some(late_interaction(
            KINDS[0],
            &query_vectors(&query_vectors_flat[0]),
            n,
        )),
        n,
    );
    let (all, _) = search(&engine, everything).await?;
    assert_eq!(
        all.len(),
        n,
        "the exhaustive first stage should match every document"
    );

    let eval = evaluate(&engine, analyzer.as_ref(), &dataset, &query_vectors_flat).await?;
    report(
        &format!("exhaustive MaxSim over all {n} documents"),
        &dataset,
        &eval.exhaustive,
        None,
    );
    if let Some(published) = published_ndcg(&args.dataset) {
        println!("published nDCG@10 (model card, setup unstated): {published:.4}");
    }
    report(
        &format!("BM25 top {BM25_WINDOW} → MaxSim rescore"),
        &dataset,
        &eval.rescored,
        Some(mean(&eval.bm25.ndcg(&dataset))),
    );
    println!(
        "bm25 only: Recall@{BM25_WINDOW} {:.4} (ceiling of the rescore condition), {} queries with no BM25 hit",
        mean(&eval.bm25.recall(&dataset, BM25_WINDOW)),
        eval.bm25.empty()
    );

    if let Some(out) = &args.trec_run {
        write_runs(out, &args.dataset, &dataset, &eval)?;
        println!("[trec] run files written to {}", out.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judged(pairs: &[(&str, i32)]) -> HashMap<String, i32> {
        pairs.iter().map(|(d, r)| (d.to_string(), *r)).collect()
    }

    fn ranking(ids: &[&str]) -> Vec<(String, f32)> {
        ids.iter()
            .enumerate()
            .map(|(i, d)| (d.to_string(), 100.0 - i as f32))
            .collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_ndcg_counts_unretrieved_relevant_documents_in_the_ideal() {
        // DCG = 1/log2(2) + 1/log2(4) = 1.5; the unretrieved "c" still
        // counts in the ideal: IDCG = 1 + 1/log2(3) + 1/log2(4).
        let qrels = judged(&[("a", 1), ("b", 1), ("c", 1)]);
        let expected = 1.5 / (1.0 + 1.0 / 3f64.log2() + 0.5);
        assert!(close(
            ndcg_at(&ranking(&["a", "x", "b"]), &qrels, 10),
            expected
        ));
    }

    #[test]
    fn test_ndcg_uses_linear_graded_gain() {
        // Linear gain (not 2^rel - 1): DCG = 1 + 2/log2(3), IDCG = 2 + 1/log2(3).
        let qrels = judged(&[("a", 2), ("b", 1)]);
        let expected = (1.0 + 2.0 / 3f64.log2()) / (2.0 + 1.0 / 3f64.log2());
        assert!(close(ndcg_at(&ranking(&["b", "a"]), &qrels, 10), expected));
    }

    #[test]
    fn test_ndcg_ignores_ranks_past_k_and_non_positive_judgments() {
        let qrels = judged(&[("a", 1), ("zero", 0), ("neg", -1)]);
        let mut ids: Vec<String> = (0..10).map(|i| format!("x{i}")).collect();
        ids.push("a".to_string());
        let ranked: Vec<(String, f32)> = ids.iter().map(|d| (d.clone(), 1.0)).collect();
        assert_eq!(ndcg_at(&ranked, &qrels, 10), 0.0);
        assert!(close(
            ndcg_at(&ranking(&["zero", "neg", "a"]), &qrels, 10),
            0.5
        ));
    }

    #[test]
    fn test_ndcg_is_zero_without_relevant_documents_or_results() {
        assert_eq!(ndcg_at(&ranking(&["a"]), &judged(&[("a", 0)]), 10), 0.0);
        assert_eq!(ndcg_at(&[], &judged(&[("a", 1)]), 10), 0.0);
    }

    #[test]
    fn test_trec_order_breaks_ties_by_doc_id_descending_and_drops_the_query_id() {
        let results = vec![
            ("a".to_string(), 1.0),
            ("q1".to_string(), 9.0),
            ("c".to_string(), 2.0),
            ("b".to_string(), 1.0),
        ];
        let ordered: Vec<String> = trec_order("q1", results)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(ordered, ["c", "b", "a"]);
    }

    #[test]
    fn test_recall_counts_relevant_documents_in_the_top_k() {
        let qrels = judged(&[("a", 1), ("b", 2), ("c", 0)]);
        assert!(close(recall_at(&ranking(&["c", "a", "x"]), &qrels, 2), 0.5));
        assert!(close(recall_at(&ranking(&["c", "a", "b"]), &qrels, 3), 1.0));
    }

    #[test]
    fn test_parse_qrels_handles_the_header_and_crlf() {
        // A blank CRLF line must be skipped like a blank LF line.
        let input =
            "query-id\tcorpus-id\tscore\r\n1\t31715818\t1\r\n1\t42\t2\r\n\r\n3\t14717500\t0\r\n";
        let qrels = parse_qrels(input.as_bytes()).unwrap();
        assert_eq!(qrels.len(), 2);
        assert_eq!(qrels["1"]["31715818"], 1);
        assert_eq!(qrels["1"]["42"], 2);
        assert_eq!(qrels["3"]["14717500"], 0);
    }

    #[test]
    fn test_document_text_joins_title_and_text() {
        assert_eq!(document_text("Title.", "Body text."), "Title. Body text.");
        assert_eq!(document_text("", "Body text."), "Body text.");
    }

    #[test]
    fn test_cache_round_trips_and_rejects_another_key_or_id_list() {
        let path =
            std::env::temp_dir().join(format!("beir-probe-cache-{}.bin", std::process::id()));
        let key = CacheKey {
            dimension: 2,
            ..CacheKey::current()
        };
        let ids = vec!["d1".to_string(), "d2".to_string()];
        let vectors = vec![vec![0.5, -0.25, 1.0, 2.0], vec![3.0, 4.0]];
        write_cache(&path, &key, &ids, &vectors).unwrap();

        assert_eq!(read_cache(&path, &key, &ids).unwrap(), Some(vectors));
        let other_revision = CacheKey {
            revision: "deadbeef".to_string(),
            ..key.clone()
        };
        assert_eq!(read_cache(&path, &other_revision, &ids).unwrap(), None);
        let other_ids = vec!["d1".to_string(), "d3".to_string()];
        assert_eq!(read_cache(&path, &key, &other_ids).unwrap(), None);
        assert_eq!(read_cache(&path, &key, &ids[..1]).unwrap(), None);

        fs::remove_file(&path).unwrap();
        assert_eq!(read_cache(&path, &key, &ids).unwrap(), None);
    }

    #[test]
    fn test_bootstrap_is_deterministic_and_degenerate_for_constant_differences() {
        let deltas = [0.1, -0.2, 0.05, 0.0, 0.3];
        assert_eq!(
            bootstrap_ci(&deltas, 1000, 7),
            bootstrap_ci(&deltas, 1000, 7)
        );
        let (lo, hi) = bootstrap_ci(&deltas, 1000, 7);
        assert!(lo < mean(&deltas) && mean(&deltas) < hi);
        let (lo, hi) = bootstrap_ci(&[0.02; 8], 1000, 7);
        assert!(close(lo, 0.02) && close(hi, 0.02));
    }
}
