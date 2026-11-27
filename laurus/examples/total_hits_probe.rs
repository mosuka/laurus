//! Issue #1472 Phase 0: the cost of counting a lexical total up to a
//! threshold.
//!
//! Builds an inverted index over the first `--docs` passages of the MS MARCO
//! passage collection (`./scripts/fetch-msmarco.sh`), with the passages in
//! one text field and the standard analyzer, caches it under
//! `.cache/total_hits_probe/<shape>_<docs>/` on a `FileStorage`, and times
//! top-10 searches with no total and with a total counted up to 1,000, up
//! to 10,000 and exactly, against `count()`, which visits every match.
//!
//! Shapes:
//!
//! - `one`: one segment (one flush).
//! - `eight`: eight segments of equal size, no merging.
//! - `default`: `InvertedIndexConfig::default()` (a flush every 10,000
//!   documents and the default merge policy).
//!
//! The query terms are picked from the index by document frequency: the
//! most frequent of a list of common English words, and the words closest
//! to 5% and to 2% of the documents. The standard analyzer drops stop
//! words, so the most frequent of them is in about 11% of the passages. A
//! three-term OR of the next three most frequent words exercises
//! Block-Max-WAND on one segment.
//!
//! For every query and mode it prints the median and the 90th percentile of
//! the wall time over `--reps` runs, the CPU time of the process per run
//! (user and system, all threads; Linux only), and the total the collector
//! counted.
//!
//! ```bash
//! ./scripts/fetch-msmarco.sh
//! cargo run --release --example total_hits_probe -- --shape default
//! ```
//!
//! Options: `--shape one|eight|default`, `--docs <n>` (default 1,000,000),
//! `--reps <n>` (default 30), `--rebuild`, `--modes <list>` (comma-separated
//! from `none,1000,10000,exact,count`; default all).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use laurus::Document;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::index::inverted::searcher::InvertedIndexSearcher;
use laurus::lexical::query::Query;
use laurus::lexical::query::boolean::BooleanQuery;
use laurus::lexical::query::collector::{Collector, TopDocsCollector};
use laurus::lexical::query::term::TermQuery;
use laurus::lexical::{InvertedIndexConfig, LexicalSearchRequest};
use laurus::storage::Storage;
use laurus::storage::file::{FileStorage, FileStorageConfig};

const FIELD: &str = "body";
const TOP_K: usize = 10;

/// Common English words to pick the query terms from.
const CANDIDATES: &[&str] = &[
    "the",
    "of",
    "and",
    "to",
    "a",
    "in",
    "is",
    "for",
    "that",
    "are",
    "with",
    "as",
    "on",
    "it",
    "be",
    "or",
    "by",
    "an",
    "from",
    "this",
    "can",
    "your",
    "you",
    "which",
    "at",
    "not",
    "have",
    "has",
    "was",
    "also",
    "more",
    "used",
    "one",
    "will",
    "may",
    "other",
    "they",
    "their",
    "all",
    "most",
    "when",
    "about",
    "many",
    "than",
    "use",
    "time",
    "how",
    "water",
    "years",
    "number",
    "people",
    "body",
    "first",
    "two",
    "between",
    "new",
    "after",
    "known",
    "into",
    "states",
    "united",
    "called",
    "cost",
    "average",
    "blood",
    "food",
    "state",
    "day",
    "work",
    "health",
    "system",
    "made",
    "usually",
    "include",
    "different",
    "american",
    "world",
    "high",
];

struct Args {
    shape: String,
    docs: usize,
    reps: usize,
    rebuild: bool,
    modes: Vec<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        shape: "default".to_string(),
        docs: 1_000_000,
        reps: 30,
        rebuild: false,
        modes: ["none", "1000", "10000", "exact", "count"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--shape" => args.shape = it.next().expect("--shape needs a value"),
            "--docs" => args.docs = it.next().expect("--docs needs a value").parse().unwrap(),
            "--reps" => args.reps = it.next().expect("--reps needs a value").parse().unwrap(),
            "--rebuild" => args.rebuild = true,
            "--modes" => {
                args.modes = it
                    .next()
                    .expect("--modes needs a value")
                    .split(',')
                    .map(|s| s.to_string())
                    .collect()
            }
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// The index configuration of a shape.
fn config_for(shape: &str, docs: usize) -> InvertedIndexConfig {
    let base = InvertedIndexConfig {
        analyzer: Arc::new(StandardAnalyzer::new().expect("standard analyzer")),
        ..Default::default()
    };
    match shape {
        "one" => InvertedIndexConfig {
            max_buffered_docs: docs + 1,
            max_buffer_memory: usize::MAX,
            max_segments: 1_000,
            ..base
        },
        "eight" => InvertedIndexConfig {
            max_buffered_docs: docs.div_ceil(8),
            max_buffer_memory: usize::MAX,
            max_segments: 1_000,
            ..base
        },
        "default" => base,
        other => panic!("unknown shape {other}"),
    }
}

fn open_storage(dir: &Path) -> Arc<dyn Storage> {
    std::fs::create_dir_all(dir).expect("create the index directory");
    Arc::new(FileStorage::new(dir, FileStorageConfig::new(dir)).expect("open the FileStorage"))
}

/// Build the index of a shape, or open the cached one.
fn index_for(args: &Args) -> InvertedIndex {
    let dir = repo_root()
        .join(".cache/total_hits_probe")
        .join(format!("{}_{}", args.shape, args.docs));
    let marker = dir.join("BUILT");
    if args.rebuild && dir.exists() {
        std::fs::remove_dir_all(&dir).expect("remove the cached index");
    }
    let config = config_for(&args.shape, args.docs);
    if marker.exists() {
        eprintln!("[probe] opening the cached index at {}", dir.display());
        return InvertedIndex::open(open_storage(&dir), config).expect("open the cached index");
    }
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("remove a partial index");
    }
    let tsv = repo_root().join(".cache/msmarco/collection.tsv");
    let file = std::fs::File::open(&tsv)
        .unwrap_or_else(|e| panic!("{}: {e}; run ./scripts/fetch-msmarco.sh", tsv.display()));
    let started = Instant::now();
    let index = InvertedIndex::create(open_storage(&dir), config).expect("create the index");
    {
        let mut writer = index.writer().expect("open the writer");
        for (i, line) in BufReader::new(file).lines().take(args.docs).enumerate() {
            let line = line.expect("read a passage");
            let passage = line.split_once('\t').map(|(_, p)| p).unwrap_or(&line);
            writer
                .add_document(Document::builder().add_text(FIELD, passage).build())
                .expect("add a passage");
            if (i + 1) % 100_000 == 0 {
                eprintln!("[probe] {} passages in {:.0?}", i + 1, started.elapsed());
            }
        }
        writer.commit().expect("commit");
    }
    std::fs::write(&marker, b"").expect("write the marker");
    eprintln!("[probe] built in {:.0?}", started.elapsed());
    index
}

/// The collector of a mode: no threshold for `none`.
fn collector_for(threshold: Option<u64>) -> TopDocsCollector {
    TopDocsCollector::new(TOP_K).with_total_hits_threshold(threshold)
}

/// User plus system CPU time of the process, all threads.
#[cfg(target_os = "linux")]
fn cpu_time() -> Duration {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1_000);
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

#[cfg(not(target_os = "linux"))]
fn cpu_time() -> Duration {
    Duration::ZERO
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i]
}

fn main() {
    let args = parse_args();
    let index = index_for(&args);
    let reader = index.reader().expect("open the reader");
    let doc_count = reader.doc_count();
    let segments = reader
        .as_any()
        .downcast_ref::<laurus::lexical::index::inverted::reader::InvertedIndexReader>()
        .map(|r| r.segment_count())
        .unwrap_or(0);
    println!(
        "shape={} docs={} segments={} reps={}",
        args.shape, doc_count, segments, args.reps
    );

    // Pick the query terms by document frequency.
    let mut df: Vec<(&str, u64)> = CANDIDATES
        .iter()
        .filter_map(|w| {
            reader
                .term_info(FIELD, w)
                .ok()
                .flatten()
                .map(|info| (*w, info.doc_freq))
        })
        .collect();
    df.sort_by_key(|(_, f)| std::cmp::Reverse(*f));
    eprintln!(
        "[probe] document frequencies: {}",
        df.iter()
            .map(|(w, f)| format!("{w}={f}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let closest = |fraction: f64| -> &str {
        let target = doc_count as f64 * fraction;
        df.iter()
            .min_by(|a, b| {
                (a.1 as f64 - target)
                    .abs()
                    .total_cmp(&(b.1 as f64 - target).abs())
            })
            .map(|(w, _)| *w)
            .expect("no candidate term in the index")
    };
    let common = df[0].0;
    let mid = closest(0.05);
    let rare = closest(0.02);
    let or_terms: Vec<&str> = df.iter().skip(1).take(3).map(|(w, _)| *w).collect();
    let freq = |w: &str| {
        df.iter()
            .find(|(x, _)| *x == w)
            .map(|(_, f)| *f)
            .unwrap_or(0)
    };

    type Make = Box<dyn Fn() -> Box<dyn Query>>;
    let queries: Vec<(String, Make)> = vec![
        (
            format!("term {common} (df {})", freq(common)),
            Box::new(move || Box::new(TermQuery::new(FIELD, common)) as Box<dyn Query>),
        ),
        (
            format!("term {mid} (df {})", freq(mid)),
            Box::new(move || Box::new(TermQuery::new(FIELD, mid)) as Box<dyn Query>),
        ),
        (
            format!("term {rare} (df {})", freq(rare)),
            Box::new(move || Box::new(TermQuery::new(FIELD, rare)) as Box<dyn Query>),
        ),
        (format!("or {}", or_terms.join("|")), {
            let terms: Vec<String> = or_terms.iter().map(|s| s.to_string()).collect();
            Box::new(move || {
                let mut q = BooleanQuery::new();
                for t in &terms {
                    q.add_should(Box::new(TermQuery::new(FIELD, t.as_str())));
                }
                Box::new(q) as Box<dyn Query>
            })
        }),
    ];

    let searcher = InvertedIndexSearcher::from_arc(reader.clone());
    println!("query\tmode\tmedian_us\tp90_us\tcpu_us_per_run\ttotal");
    for (label, make) in &queries {
        for mode in &args.modes {
            let threshold = match mode.as_str() {
                "none" | "count" => None,
                "exact" => Some(u64::MAX),
                n => Some(n.parse::<u64>().expect("a threshold")),
            };
            // One warm-up run, then the measured runs.
            let mut total = 0;
            let mut times = Vec::with_capacity(args.reps);
            let cpu_before = cpu_time();
            for rep in 0..=args.reps {
                let started = Instant::now();
                if mode == "count" {
                    total = searcher
                        .count(LexicalSearchRequest::new(make()))
                        .expect("count");
                } else {
                    let collected = searcher
                        .search_with_collector(make(), collector_for(threshold))
                        .expect("search");
                    total = collected.total_hits();
                }
                if rep > 0 {
                    times.push(started.elapsed());
                }
            }
            let cpu = (cpu_time() - cpu_before) / (args.reps as u32 + 1);
            times.sort();
            println!(
                "{label}\t{mode}\t{}\t{}\t{}\t{total}",
                percentile(&times, 0.5).as_micros(),
                percentile(&times, 0.9).as_micros(),
                cpu.as_micros(),
            );
        }
    }
}
