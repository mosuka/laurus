//! Issue #1704: a fixed number of searches of one query shape, to compare
//! the instruction counts of the scoring hot paths between two builds.
//!
//! Builds an inverted index of `--docs` documents with the uniform corpus
//! of `benches/lexical_search_bench.rs` (`build_body`) and the standard
//! analyzer, caches it under `.cache/scoring_hot_path_probe/<docs>/` on a
//! `FileStorage`, and runs `--iters` top-10 searches of the `--query` shape
//! on one thread. It prints the query's description and a hash of the last
//! search's hits (document ids and score bits), so two builds can be checked
//! to return the same hits.
//!
//! The synonym and graph phrase shapes are parsed with a query-time
//! `SynonymGraphFilter`, so the same source builds on a commit without
//! weighted synonyms; there, the `_wboost` shapes (a filter with
//! `with_boost(0.8)`) give today's unweighted queries.
//!
//! Shapes: `term`, `must_and`, `should_or`, `synonym`, `should_or_synonym`,
//! `synonym_wboost`, `should_or_synonym_wboost`, `phrase`, `graph_phrase`,
//! `graph_phrase_wboost`.
//!
//! ```bash
//! cargo build --release --example scoring_hot_path_probe
//! RAYON_NUM_THREADS=1 perf stat -e instructions:u \
//!     target/release/examples/scoring_hot_path_probe --query term --iters 20000
//! ```
//!
//! The instructions of one search are the difference between `--iters N`
//! and `--iters 0` (the index opening and the query parsing), divided by N.
//! Options: `--query <shape>`, `--iters <n>` (default 1,000), `--docs <n>`
//! (default 10,000), `--rebuild`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use laurus::Document;
use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::pipeline::PipelineAnalyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::analysis::synonym::dictionary::SynonymDictionary;
use laurus::analysis::token_filter::lowercase::LowercaseFilter;
use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;
use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use laurus::lexical::InvertedIndexConfig;
use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::index::inverted::searcher::InvertedIndexSearcher;
use laurus::lexical::query::Query;
use laurus::lexical::query::boolean::BooleanQuery;
use laurus::lexical::query::collector::{Collector, TopDocsCollector};
use laurus::lexical::query::parser::LexicalQueryParser;
use laurus::lexical::query::term::TermQuery;
use laurus::storage::Storage;
use laurus::storage::file::{FileStorage, FileStorageConfig};

const FIELD: &str = "body";
const TOP_K: usize = 10;

// The uniform corpus of `benches/lexical_search_bench.rs` (`build_body`),
// copied so that the probe does not depend on the bench.
const COMMON_TERMS: &str = "search engine system data query";
const TOPIC_PHRASES: &[&str] = &[
    "rust programming language systems safety concurrency memory ownership",
    "python data science machine learning artificial intelligence numpy",
    "javascript web development frontend backend node react framework",
    "database query optimization indexing performance search engine",
    "network protocol distributed systems cloud computing infrastructure",
    "security cryptography authentication authorization encryption",
    "algorithms data structures sorting searching graph traversal",
    "operating systems kernel processes threads scheduling memory",
];
const LONG_TAIL: &[&str] = &[
    "compaction",
    "histogram",
    "tarpaulin",
    "miri",
    "loom",
    "fuzzer",
    "afl",
];
const LONG_TAIL_PER_DOC: usize = 5;

fn build_body(i: usize) -> String {
    let topic = TOPIC_PHRASES[i % TOPIC_PHRASES.len()];
    let tail: Vec<&str> = (0..LONG_TAIL_PER_DOC)
        .map(|k| LONG_TAIL[(i.wrapping_mul(7) + k * 11) % LONG_TAIL.len()])
        .collect();
    format!(
        "Document {i} {COMMON_TERMS} {topic} {} should match relevant terms",
        tail.join(" ")
    )
}

struct Args {
    query: String,
    iters: usize,
    docs: usize,
    rebuild: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        query: "term".to_string(),
        iters: 1_000,
        docs: 10_000,
        rebuild: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--query" => args.query = it.next().expect("--query <shape>"),
            "--iters" => args.iters = it.next().expect("--iters <n>").parse().expect("a number"),
            "--docs" => args.docs = it.next().expect("--docs <n>").parse().expect("a number"),
            "--rebuild" => args.rebuild = true,
            other => panic!("unknown option {other}"),
        }
    }
    args
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn open_storage(dir: &Path) -> Arc<dyn Storage> {
    std::fs::create_dir_all(dir).expect("create the index directory");
    Arc::new(FileStorage::new(dir, FileStorageConfig::new(dir)).expect("open the FileStorage"))
}

/// Build the index, or open the cached one.
fn index_for(args: &Args) -> InvertedIndex {
    let dir = repo_root()
        .join(".cache/scoring_hot_path_probe")
        .join(args.docs.to_string());
    let marker = dir.join("BUILT");
    if args.rebuild && dir.exists() {
        std::fs::remove_dir_all(&dir).expect("remove the cached index");
    }
    let config = InvertedIndexConfig {
        analyzer: Arc::new(StandardAnalyzer::new().expect("standard analyzer")),
        max_buffered_docs: args.docs + 1,
        max_buffer_memory: usize::MAX,
        ..Default::default()
    };
    if marker.exists() {
        return InvertedIndex::open(open_storage(&dir), config).expect("open the cached index");
    }
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("remove a partial index");
    }
    let index = InvertedIndex::create(open_storage(&dir), config).expect("create the index");
    {
        let mut writer = index.writer().expect("open the writer");
        for i in 0..args.docs {
            writer
                .add_document(Document::builder().add_text(FIELD, build_body(i)).build())
                .expect("add a document");
        }
        writer.commit().expect("commit");
    }
    std::fs::write(&marker, b"").expect("write the marker");
    index
}

/// A query-time analyzer with one synonym group, its synonyms weighted by
/// `with_boost(boost)` when `boost` is given.
fn synonym_analyzer(group: &[&str], boost: Option<f32>) -> Arc<dyn Analyzer> {
    let mut dict = SynonymDictionary::new(None).expect("synonym dictionary");
    dict.add_synonym_group(group.iter().map(|s| s.to_string()).collect());
    let mut filter = SynonymGraphFilter::new(dict, true);
    if let Some(boost) = boost {
        filter = filter.with_boost(boost);
    }
    Arc::new(
        PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
            .add_filter(Arc::new(LowercaseFilter::new()))
            .add_filter(Arc::new(filter)),
    )
}

fn parse(analyzer: Arc<dyn Analyzer>, dsl: &str) -> Box<dyn Query> {
    LexicalQueryParser::new(analyzer)
        .parse(dsl)
        .expect("parse the query")
}

fn term(text: &str) -> Box<dyn Query> {
    Box::new(TermQuery::new(FIELD, text))
}

fn query_for(shape: &str) -> Box<dyn Query> {
    let plain = || synonym_analyzer(&["unused", "unmatched"], None);
    let synonym = |boost| parse(synonym_analyzer(&["memory", "cloud"], boost), "body:memory");
    let should_or_synonym = |boost| {
        let mut q = BooleanQuery::new();
        q.add_should(term("python"));
        q.add_should(synonym(boost));
        Box::new(q) as Box<dyn Query>
    };
    let graph_phrase = |boost| {
        parse(
            synonym_analyzer(&["data", "data query"], boost),
            "body:\"system data\"",
        )
    };
    match shape {
        "term" => term("memory"),
        "must_and" => {
            let mut q = BooleanQuery::new();
            q.add_must(term("memory"));
            q.add_must(term("rust"));
            Box::new(q)
        }
        "should_or" => {
            let mut q = BooleanQuery::new();
            for t in ["memory", "cloud", "python"] {
                q.add_should(term(t));
            }
            Box::new(q)
        }
        "synonym" => synonym(None),
        "should_or_synonym" => should_or_synonym(None),
        "synonym_wboost" => synonym(Some(0.8)),
        "should_or_synonym_wboost" => should_or_synonym(Some(0.8)),
        "phrase" => parse(plain(), "body:\"search engine\""),
        "graph_phrase" => graph_phrase(None),
        "graph_phrase_wboost" => graph_phrase(Some(0.8)),
        other => panic!("unknown query shape {other}"),
    }
}

fn main() {
    let args = parse_args();
    let index = index_for(&args);
    let reader = index.reader().expect("open the reader");
    let searcher = InvertedIndexSearcher::from_arc(reader);
    let query = query_for(&args.query);
    eprintln!("[probe] {}: {}", args.query, query.description());

    let mut hash = DefaultHasher::new();
    for iter in 0..args.iters {
        let collector = searcher
            .search_with_collector(query.clone_box(), TopDocsCollector::new(TOP_K))
            .expect("search");
        if iter + 1 == args.iters {
            for hit in collector.results() {
                hit.doc_id.hash(&mut hash);
                hit.score.to_bits().hash(&mut hash);
            }
        }
    }
    println!("{}\t{}\t{:016x}", args.query, args.iters, hash.finish());
}
