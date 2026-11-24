//! Issue #1579: `update_field` rebuilds a vector field as another of the
//! kinds HNSW, Flat and IVF, keeping its vectors.
//!
//! Each segment records the kind it was written as, and reading, searching,
//! merging and reclaiming go by it. The schema only decides the kind of the
//! segments written from now on, so a field whose segments are of another
//! kind than the schema's opens and is searched: after a crash between the
//! rebuild and saving the schema, after a schema hook that failed, and after
//! `schema.toml` was edited by hand.

#[path = "common/log_capture.rs"]
mod log_capture;

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use laurus::lexical::TermQuery;
use laurus::storage::manifest::{load_checksummed_json, save_checksummed_json};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::prefixed::PrefixedStorage;
use laurus::storage::{FileMetadata, LoadingMode, Storage, StorageInput, StorageOutput};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::index::flat::reader::FlatVectorIndexReader;
use laurus::vector::index::hnsw::reader::HnswIndexReader;
use laurus::vector::index::ivf::reader::IvfIndexReader;
use laurus::{
    Document, Engine, FieldChangeKind, FieldOption, FlatOption, HnswOption, IvfOption, LaurusError,
    QuantizationMethod, QueryVector, RerankStorageKind, Result, Schema, SearchRequestBuilder,
    TextOption, UpdateFieldOptions, VectorSearchQuery,
};

const DIM: usize = 8;

const MANIFEST: &str = "vector/vec/segments.json";

const REINDEX: UpdateFieldOptions = UpdateFieldOptions {
    reindex: true,
    dry_run: false,
};

const DRY_RUN: UpdateFieldOptions = UpdateFieldOptions {
    reindex: false,
    dry_run: true,
};

// --- kinds ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Flat,
    Hnsw,
    Ivf,
}

impl Kind {
    const ALL: [Kind; 3] = [Kind::Flat, Kind::Hnsw, Kind::Ivf];

    /// The primary file suffix of the kind, without its dot.
    fn suffix(self) -> &'static str {
        match self {
            Kind::Flat => "flat",
            Kind::Hnsw => "hnsw",
            Kind::Ivf => "ivf",
        }
    }

    fn option(self) -> FieldOption {
        self.option_with(QuantizationMethod::Scalar8Bit, None)
    }

    /// The option of the kind with Euclidean distance, whose score is
    /// `exp(-distance)`.
    fn option_with(
        self,
        quantizer: QuantizationMethod,
        rerank_storage: Option<RerankStorageKind>,
    ) -> FieldOption {
        self.option_of(DistanceMetric::Euclidean, quantizer, rerank_storage)
    }

    /// The option of the kind. IVF probes every list, which makes it exact.
    fn option_of(
        self,
        distance: DistanceMetric,
        quantizer: QuantizationMethod,
        rerank_storage: Option<RerankStorageKind>,
    ) -> FieldOption {
        match self {
            Kind::Flat => FieldOption::Flat(FlatOption {
                dimension: DIM,
                distance,
                quantizer,
                rerank_storage,
                ..FlatOption::default()
            }),
            Kind::Hnsw => FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance,
                quantizer,
                rerank_storage,
                ..HnswOption::default()
            }),
            Kind::Ivf => FieldOption::Ivf(IvfOption {
                dimension: DIM,
                distance,
                quantizer,
                rerank_storage,
                n_clusters: 2,
                n_probe: 2,
                ..IvfOption::default()
            }),
        }
    }
}

/// The six changes of kind, with a label.
fn pairs() -> Vec<(String, Kind, Kind)> {
    Kind::ALL
        .into_iter()
        .flat_map(|from| Kind::ALL.into_iter().map(move |to| (from, to)))
        .filter(|(from, to)| from != to)
        .map(|(from, to)| (format!("{} -> {}", from.suffix(), to.suffix()), from, to))
        .collect()
}

fn schema_with(option: FieldOption) -> Schema {
    Schema::builder()
        .add_field("vec", option)
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build()
}

// --- storage ---

type Files = BTreeMap<String, Vec<u8>>;

/// A failure [`TestStorage`] injects while armed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    /// Creating a segment file of this kind fails.
    CreateSegment(Kind),
    /// Renaming the field's manifest into place fails.
    RenameManifest,
    /// The sync after the rename of the field's manifest fails.
    SyncManifest,
    /// Deleting a segment file fails, as on Windows while it is mapped.
    DeleteSegment,
}

/// Memory storage without locks, so an index can be opened again while a
/// dropped engine's lock is not released yet. It can fail an operation, and
/// record a copy of every file after each operation that changes one.
#[derive(Debug)]
struct TestStorage {
    inner: MemoryStorage,
    recording: AtomicBool,
    copies: Mutex<Vec<(String, Files)>>,
    fault: Mutex<Option<Fault>>,
    manifest_renamed: AtomicBool,
    injected: AtomicUsize,
}

impl TestStorage {
    fn new() -> Arc<Self> {
        Self::with_files(&Files::new())
    }

    fn with_files(files: &Files) -> Arc<Self> {
        let storage = Self {
            inner: MemoryStorage::new(MemoryStorageConfig::default()),
            recording: AtomicBool::new(false),
            copies: Mutex::new(Vec::new()),
            fault: Mutex::new(None),
            manifest_renamed: AtomicBool::new(false),
            injected: AtomicUsize::new(0),
        };
        for (name, bytes) in files {
            storage.write(name, bytes);
        }
        Arc::new(storage)
    }

    fn write(&self, name: &str, bytes: &[u8]) {
        let mut output = self.inner.create_output(name).unwrap();
        output.write_all(bytes).unwrap();
        output.close().unwrap();
    }

    fn files(&self) -> Files {
        self.inner
            .list_files()
            .unwrap()
            .into_iter()
            .map(|name| {
                let mut bytes = Vec::new();
                let mut input = self.inner.open_input(&name).unwrap();
                input.read_to_end(&mut bytes).unwrap();
                (name, bytes)
            })
            .collect()
    }

    /// The names of the field's segment files.
    fn segment_files(&self) -> Vec<String> {
        self.files()
            .into_keys()
            .filter_map(|name| {
                name.strip_prefix("vector/vec/")
                    .filter(|file| file.starts_with("segment_"))
                    .map(str::to_string)
            })
            .collect()
    }

    fn arm(&self, fault: Fault) {
        self.manifest_renamed.store(false, Ordering::SeqCst);
        *self.fault.lock().unwrap() = Some(fault);
    }

    /// Stop injecting the fault, and return how often it was injected.
    fn disarm(&self) -> usize {
        *self.fault.lock().unwrap() = None;
        self.injected.swap(0, Ordering::SeqCst)
    }

    fn armed(&self, fault: Fault) -> bool {
        *self.fault.lock().unwrap() == Some(fault)
    }

    fn inject(&self, what: &str) -> LaurusError {
        self.injected.fetch_add(1, Ordering::SeqCst);
        LaurusError::storage(format!("injected failure: {what}"))
    }

    /// Record a copy of every file, unless it equals the last one.
    fn record(&self, step: String) {
        if !self.recording.load(Ordering::SeqCst) {
            return;
        }
        let files = self.files();
        let mut copies = self.copies.lock().unwrap();
        if copies.last().is_none_or(|(_, last)| *last != files) {
            copies.push((step, files));
        }
    }

    fn start_recording(&self) {
        self.recording.store(true, Ordering::SeqCst);
        self.record("before the first step".to_string());
    }

    fn stop_recording(&self) -> Vec<(String, Files)> {
        self.recording.store(false, Ordering::SeqCst);
        std::mem::take(&mut *self.copies.lock().unwrap())
    }
}

fn is_segment_file(name: &str) -> bool {
    name.starts_with("vector/vec/segment_")
}

impl Storage for TestStorage {
    fn loading_mode(&self) -> LoadingMode {
        self.inner.loading_mode()
    }
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        for kind in Kind::ALL {
            if self.armed(Fault::CreateSegment(kind))
                && is_segment_file(name)
                && name.contains(&format!(".{}", kind.suffix()))
            {
                return Err(self.inject(&format!("create {name}")));
            }
        }
        let output = self.inner.create_output(name)?;
        self.record(format!("create {name}"));
        Ok(output)
    }
    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }
    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }
    fn delete_file(&self, name: &str) -> Result<()> {
        if self.armed(Fault::DeleteSegment) && is_segment_file(name) {
            return Err(self.inject(&format!("delete {name}")));
        }
        self.inner.delete_file(name)?;
        self.record(format!("delete {name}"));
        Ok(())
    }
    fn list_files(&self) -> Result<Vec<String>> {
        self.inner.list_files()
    }
    fn file_size(&self, name: &str) -> Result<u64> {
        self.inner.file_size(name)
    }
    fn metadata(&self, name: &str) -> Result<FileMetadata> {
        self.inner.metadata(name)
    }
    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        if new_name == MANIFEST {
            if self.armed(Fault::RenameManifest) {
                return Err(self.inject(&format!("rename {old_name}")));
            }
            self.manifest_renamed.store(true, Ordering::SeqCst);
        }
        self.inner.rename_file(old_name, new_name)?;
        self.record(format!("rename {old_name} to {new_name}"));
        Ok(())
    }
    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }
    fn sync(&self) -> Result<()> {
        if self.armed(Fault::SyncManifest) && self.manifest_renamed.swap(false, Ordering::SeqCst) {
            return Err(self.inject("sync"));
        }
        self.inner.sync()?;
        self.record("sync".to_string());
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

async fn open(storage: &Arc<TestStorage>, option: FieldOption) -> Result<Engine> {
    Engine::new(storage.clone() as Arc<dyn Storage>, schema_with(option)).await
}

// --- documents ---

/// Deterministic values in `[-1, 1)`, far apart for different seeds.
fn values(seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..DIM)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The live documents: the number of the id, and the seed of its values.
type Live = BTreeMap<u64, u64>;

fn title(id: u64) -> &'static str {
    if id.is_multiple_of(2) { "even" } else { "odd" }
}

async fn put(engine: &Engine, live: &mut Live, id: u64, seed: u64) {
    let doc = Document::builder()
        .add_vector("vec", values(seed))
        .add_text("title", title(id))
        .build();
    engine.put_document(&format!("d{id}"), doc).await.unwrap();
    live.insert(id, seed);
}

async fn delete(engine: &Engine, live: &mut Live, id: u64) {
    engine.delete_documents(&format!("d{id}")).await.unwrap();
    live.remove(&id);
}

/// Four segments with an upsert, a delete and a re-added id among them, and
/// after the last commit a new document, an upsert, a delete and a delete
/// followed by a put of the same id.
async fn fill(engine: &Engine) -> Live {
    let mut live = Live::new();
    for id in 0..6 {
        put(engine, &mut live, id, id).await;
    }
    engine.commit().await.unwrap();
    for id in 6..12 {
        put(engine, &mut live, id, id).await;
    }
    put(engine, &mut live, 1, 101).await;
    delete(engine, &mut live, 3).await;
    engine.commit().await.unwrap();
    for id in 12..18 {
        put(engine, &mut live, id, id).await;
    }
    delete(engine, &mut live, 2).await;
    engine.commit().await.unwrap();
    put(engine, &mut live, 3, 103).await;
    engine.commit().await.unwrap();

    put(engine, &mut live, 18, 18).await;
    put(engine, &mut live, 5, 105).await;
    delete(engine, &mut live, 4).await;
    delete(engine, &mut live, 6).await;
    put(engine, &mut live, 6, 106).await;
    live
}

// --- checks ---

fn request(seed: u64, limit: usize) -> SearchRequestBuilder {
    request_on(seed, limit, Some(vec!["vec".to_string()]))
}

fn request_on(seed: u64, limit: usize, fields: Option<Vec<String>>) -> SearchRequestBuilder {
    SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(values(seed)),
            weight: 1.0,
            fields,
        }]))
        .limit(limit)
}

async fn search(
    engine: &Engine,
    request: SearchRequestBuilder,
) -> std::result::Result<Vec<(String, f32)>, String> {
    engine
        .search(request.build())
        .await
        .map(|hits| hits.into_iter().map(|hit| (hit.id, hit.score)).collect())
        .map_err(|e| format!("a search failed: {e}"))
}

fn ids<'a>(numbers: impl Iterator<Item = &'a u64>) -> BTreeSet<String> {
    numbers.map(|id| format!("d{id}")).collect()
}

/// Why the field does not hold exactly `live`, if it does not: a search
/// for everything returns each document once, and a search for the values
/// of each returns it first, with a score that tells its values are those.
async fn wrong_documents(engine: &Engine, live: &Live) -> Option<String> {
    let all = match search(engine, request(0, 100)).await {
        Ok(hits) => hits,
        Err(e) => return Some(e),
    };
    let found: BTreeSet<String> = all.iter().map(|(id, _)| id.clone()).collect();
    let expected = ids(live.keys());
    if found != expected || all.len() != expected.len() {
        let missing: Vec<_> = expected.difference(&found).collect();
        let extra: Vec<_> = found.difference(&expected).collect();
        return Some(format!(
            "{} hits for {} documents: {missing:?} are missing, {extra:?} should not be found",
            all.len(),
            expected.len()
        ));
    }
    for (id, seed) in live {
        match search(engine, request(*seed, 1)).await {
            Ok(hits) => match hits.first() {
                Some((found, score)) if *found == format!("d{id}") && *score > 0.9 => {}
                other => return Some(format!("the values of d{id} find {other:?}")),
            },
            Err(e) => return Some(e),
        }
    }
    None
}

/// The field's manifest, as JSON, or null when there is none.
fn manifest(storage: &TestStorage) -> serde_json::Value {
    load_checksummed_json(storage, MANIFEST, None)
        .unwrap()
        .map_or(serde_json::Value::Null, |(manifest, _)| manifest)
}

/// The `(segment id, kind)` pairs the field's manifest lists.
fn listed(storage: &TestStorage) -> BTreeSet<(String, String)> {
    manifest(storage)["segments"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|segment| {
            let id = segment["segment_id"].as_str().unwrap().to_string();
            let kind = segment["kind"].as_str().unwrap_or("not recorded");
            (id, kind.to_string())
        })
        .collect()
}

/// Why the field's segment files are not exactly those of the segments its
/// manifest lists, if they are not.
fn wrong_files(storage: &TestStorage) -> Option<String> {
    let listed = listed(storage);
    let mut primaries = BTreeSet::new();
    for file in storage.segment_files() {
        let mut parts = file.split('.');
        let id = parts.next().unwrap_or_default().to_string();
        let kind = parts.next().unwrap_or_default().to_string();
        let rest: Vec<&str> = parts.collect();
        if !listed.contains(&(id.clone(), kind.clone())) || rest.contains(&"tmp") {
            return Some(format!(
                "file {file} is left, the manifest lists {listed:?}"
            ));
        }
        if rest.is_empty() {
            primaries.insert((id, kind));
        }
    }
    if primaries != listed {
        return Some(format!(
            "the manifest lists {listed:?}, with files {primaries:?}"
        ));
    }
    None
}

/// Why the field is not of kind `kind` in one segment, or none if it has
/// no vector, if it is not.
fn wrong_kind(storage: &TestStorage, kind: Kind) -> Option<String> {
    let listed = listed(storage);
    if listed.len() > 1 || listed.iter().any(|(_, k)| k != kind.suffix()) {
        return Some(format!("the manifest lists {listed:?}"));
    }
    wrong_files(storage)
}

fn report(failures: &mut Vec<String>, what: &str, failure: Option<String>) {
    if let Some(failure) = failure {
        failures.push(format!("{what}: {failure}"));
    }
}

async fn check_open(
    storage: &Arc<TestStorage>,
    option: FieldOption,
    live: &Live,
    what: &str,
    failures: &mut Vec<String>,
) -> Option<Engine> {
    match open(storage, option).await {
        Ok(engine) => {
            report(failures, what, wrong_documents(&engine, live).await);
            Some(engine)
        }
        Err(e) => {
            failures.push(format!("{what}: the open failed: {e}"));
            None
        }
    }
}

// --- tests ---

/// Every change between the three kinds is a Reindex that keeps every
/// vector: what was committed, in several segments, and what was not, with
/// the deletes, the upserts and the re-added ids. The field is then one
/// segment of the new kind, and no file of the old kind is left, also after
/// the next commit and a reopen.
#[tokio::test(flavor = "multi_thread")]
async fn a_change_of_kind_is_a_reindex_that_keeps_every_vector() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        let storage = TestStorage::new();
        let engine = open(&storage, from.option()).await.unwrap();
        let live = fill(&engine).await;

        let dry = engine
            .update_field("vec", to.option(), DRY_RUN)
            .await
            .unwrap();
        if dry.classification != FieldChangeKind::Reindex {
            failures.push(format!("{pair}: the dry run says {:?}", dry.classification));
        }
        let outcome = engine
            .update_field("vec", to.option(), REINDEX)
            .await
            .unwrap();
        if outcome.classification != FieldChangeKind::Reindex {
            failures.push(format!(
                "{pair}: it was applied as {:?}",
                outcome.classification
            ));
        }
        let schema = engine.schema();
        if format!("{:?}", schema.fields["vec"]) != format!("{:?}", to.option()) {
            failures.push(format!("{pair}: the schema has {:?}", schema.fields["vec"]));
        }
        if schema.pending_reindex.contains("vec") {
            failures.push(format!("{pair}: pending_reindex names the field"));
        }
        for (what, commit) in [("after", false), ("after the next commit", true)] {
            if commit {
                engine.commit().await.unwrap();
            }
            let what = format!("{pair}, {what}");
            report(&mut failures, &what, wrong_documents(&engine, &live).await);
            report(&mut failures, &what, wrong_kind(&storage, to));
        }
        drop(engine);
        let what = format!("{pair}, after a reopen");
        let engine = check_open(&storage, to.option(), &live, &what, &mut failures).await;
        report(&mut failures, &what, wrong_kind(&storage, to));
        drop(engine);
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A Cosine field, whose segments hold normalized vectors, keeps them, and
/// so does one with Float16 rows.
#[tokio::test(flavor = "multi_thread")]
async fn a_change_of_kind_keeps_the_vectors_of_a_cosine_field() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        for quantizer in [QuantizationMethod::Scalar8Bit, QuantizationMethod::Float16] {
            let what = format!("{pair}, {quantizer:?}");
            let option = |kind: Kind| kind.option_of(DistanceMetric::Cosine, quantizer, None);
            let storage = TestStorage::new();
            let engine = open(&storage, option(from)).await.unwrap();
            let live = fill(&engine).await;
            engine
                .update_field("vec", option(to), REINDEX)
                .await
                .unwrap();
            report(&mut failures, &what, wrong_documents(&engine, &live).await);
            engine.commit().await.unwrap();
            drop(engine);
            for first in [from, to] {
                let what = format!("{what}, reopened as {first:?}");
                let copy = TestStorage::with_files(&storage.files());
                drop(check_open(&copy, option(first), &live, &what, &mut failures).await);
                report(&mut failures, &what, wrong_kind(&copy, to));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Opening a field that has segments of another kind warns once, with its
/// name and how many they are. Opening one that has none, and the rebuild,
/// which opens the old kind's segments under the new one, warn of nothing.
#[tokio::test(flavor = "multi_thread")]
async fn an_open_warns_of_segments_of_another_kind() {
    log_capture::install();
    // The tests of this file run in parallel; the name tells the field.
    const FIELD: &str = "warned";
    let warnings = || -> Vec<String> {
        log_capture::warnings()
            .into_iter()
            .map(|(_, message)| message)
            .filter(|message| message.contains(&format!("'{FIELD}'")))
            .collect()
    };
    let open = |storage: &Arc<TestStorage>, kind: Kind| {
        let schema = Schema::builder().add_field(FIELD, kind.option()).build();
        Engine::new(storage.clone() as Arc<dyn Storage>, schema)
    };
    let storage = TestStorage::new();
    let engine = open(&storage, Kind::Hnsw).await.unwrap();
    for id in 0..3 {
        let doc = Document::builder().add_vector(FIELD, values(id)).build();
        engine.put_document(&format!("d{id}"), doc).await.unwrap();
        engine.commit().await.unwrap();
    }
    drop(engine);
    drop(open(&storage, Kind::Hnsw).await.unwrap());
    assert_eq!(warnings(), Vec::<String>::new());

    let engine = open(&storage, Kind::Flat).await.unwrap();
    assert_eq!(
        warnings(),
        [
            "vector field 'warned' has 3 segments that were written as another index kind; a \
             search scans them exactly, until a merge or an update of the field that rebuilds \
             it rewrites them"
        ]
    );
    engine
        .update_field(FIELD, Kind::Ivf.option(), REINDEX)
        .await
        .unwrap();
    drop(engine);
    drop(open(&storage, Kind::Ivf).await.unwrap());
    assert_eq!(warnings().len(), 1);
}

/// A field whose documents are all deleted, and one that never had any,
/// change their kind, take documents afterwards, and reopen.
#[tokio::test(flavor = "multi_thread")]
async fn a_change_of_kind_of_a_field_with_no_live_vector_leaves_it_empty() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        for committed in [None, Some(false), Some(true)] {
            let what = match committed {
                None => format!("{pair}, no document"),
                Some(false) => format!("{pair}, deleted since the commit"),
                Some(true) => format!("{pair}, deleted and committed"),
            };
            let storage = TestStorage::new();
            let engine = open(&storage, from.option()).await.unwrap();
            let mut live = Live::new();
            if let Some(committed) = committed {
                for id in 0..4 {
                    put(&engine, &mut live, id, id).await;
                }
                engine.commit().await.unwrap();
                for id in 0..4 {
                    delete(&engine, &mut live, id).await;
                }
                if committed {
                    engine.commit().await.unwrap();
                }
            }
            match engine.update_field("vec", to.option(), REINDEX).await {
                Ok(outcome) if outcome.classification == FieldChangeKind::Reindex => {}
                other => failures.push(format!("{what}: {:?}", other.map(|o| o.classification))),
            }
            report(&mut failures, &what, wrong_documents(&engine, &live).await);
            put(&engine, &mut live, 7, 7).await;
            engine.commit().await.unwrap();
            report(&mut failures, &what, wrong_documents(&engine, &live).await);
            drop(engine);
            let what = format!("{what}, reopened");
            drop(check_open(&storage, to.option(), &live, &what, &mut failures).await);
            report(&mut failures, &what, wrong_files(&storage));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The f32 rerank sidecar of segment `id` of kind `kind`.
fn sidecar_values(storage: &Arc<TestStorage>, id: &str, kind: Kind, doc: u64) -> Option<Vec<f32>> {
    let field: Arc<dyn Storage> = Arc::new(PrefixedStorage::new(
        "vector/vec".to_string(),
        storage.clone() as Arc<dyn Storage>,
    ));
    let metric = DistanceMetric::Euclidean;
    let pool = match kind {
        Kind::Flat => FlatVectorIndexReader::load(field, id, metric)
            .unwrap()
            .rerank_storage()
            .cloned(),
        Kind::Hnsw => HnswIndexReader::load(field, id, metric)
            .unwrap()
            .rerank_storage()
            .cloned(),
        Kind::Ivf => IvfIndexReader::load(field, id, metric)
            .unwrap()
            .rerank_storage()
            .cloned(),
    };
    pool?.get_f32_slice(doc, "vec").map(<[f32]>::to_vec)
}

/// With the f32 sidecar, the rebuilt segment holds the exact values the
/// documents were put with, not those decoded from the old segments.
#[tokio::test(flavor = "multi_thread")]
async fn a_change_of_kind_takes_the_exact_values_of_the_sidecar() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        let with_sidecar = |kind: Kind| {
            kind.option_with(QuantizationMethod::Scalar8Bit, Some(RerankStorageKind::F32))
        };
        let storage = TestStorage::new();
        let engine = open(&storage, with_sidecar(from)).await.unwrap();
        let mut live = Live::new();
        for id in 0..4 {
            put(&engine, &mut live, id, id).await;
        }
        engine.commit().await.unwrap();
        engine
            .update_field("vec", with_sidecar(to), REINDEX)
            .await
            .unwrap();
        report(&mut failures, &pair, wrong_documents(&engine, &live).await);
        report(&mut failures, &pair, wrong_kind(&storage, to));
        let Some((segment, _)) = listed(&storage).pop_first() else {
            failures.push(format!("{pair}: the manifest lists no segment"));
            continue;
        };
        // The internal ids of the four documents are the first ones.
        let held: Vec<Vec<f32>> = (0..20)
            .filter_map(|doc| sidecar_values(&storage, &segment, to, doc))
            .collect();
        let expected: Vec<Vec<f32>> = (0..4).map(values).collect();
        if held != expected {
            failures.push(format!("{pair}: the sidecar holds {held:?}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A product-quantized HNSW field keeps every document as Flat and IVF,
/// and the exact values when it has the sidecar.
#[tokio::test(flavor = "multi_thread")]
async fn a_product_quantized_hnsw_field_becomes_flat_or_ivf() {
    const DOCS: u64 = 300;
    let mut failures = Vec::new();
    let quantizer = QuantizationMethod::ProductQuantization { subvector_count: 2 };
    for to in [Kind::Flat, Kind::Ivf] {
        for sidecar in [None, Some(RerankStorageKind::F32)] {
            let what = format!("{to:?}, sidecar {sidecar:?}");
            let storage = TestStorage::new();
            let engine = open(&storage, Kind::Hnsw.option_with(quantizer, sidecar))
                .await
                .unwrap();
            let mut live = Live::new();
            for id in 0..DOCS {
                put(&engine, &mut live, id, id).await;
            }
            engine.commit().await.unwrap();
            engine
                .update_field(
                    "vec",
                    to.option_with(QuantizationMethod::Float16, None),
                    REINDEX,
                )
                .await
                .unwrap();
            report(&mut failures, &what, wrong_kind(&storage, to));
            if sidecar.is_some() {
                // 300 searches are slow; every tenth document tells.
                live.retain(|id, _| id.is_multiple_of(10));
            }
            let all = search(&engine, request(0, 1000)).await.unwrap();
            if all.len() != DOCS as usize {
                failures.push(format!("{what}: {} hits", all.len()));
            }
            if sidecar.is_some() {
                for (id, seed) in &live {
                    let hits = search(&engine, request(*seed, 1)).await.unwrap();
                    if hits
                        .first()
                        .is_none_or(|(found, score)| *found != format!("d{id}") || *score <= 0.9)
                    {
                        failures.push(format!("{what}: the values of d{id} find {hits:?}"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A crash after any step of the change that touches the storage leaves
/// every document readable under the schema the index is opened with, the
/// old one or the new one, with what was written since the last commit. The
/// index then takes writes, opens under the other schema too, and the files
/// its manifest does not list are reclaimed, of either kind.
#[tokio::test(flavor = "multi_thread")]
async fn a_crash_at_any_step_leaves_every_vector_readable_under_either_schema() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        let storage = TestStorage::new();
        let engine = open(&storage, from.option()).await.unwrap();
        let live = fill(&engine).await;
        storage.start_recording();
        engine
            .update_field("vec", to.option(), REINDEX)
            .await
            .unwrap();
        let mut copies = storage.stop_recording();
        copies.push(("after it returned".to_string(), storage.files()));
        drop(engine);

        // The publish is among the steps, with both kinds' files on
        // storage, and so is the deletion of the replaced segments.
        for step in [format!("to {MANIFEST}"), "delete".to_string()] {
            if !copies.iter().any(|(s, _)| s.contains(&step)) {
                failures.push(format!("{pair}: no step '{step}' was recorded"));
            }
        }

        for (step, files) in &copies {
            for (first, second) in [(from, to), (to, from)] {
                let what = format!("{pair}, crash after '{step}', opened as {first:?}");
                let copy = TestStorage::with_files(files);
                let mut live = live.clone();
                let Some(engine) =
                    check_open(&copy, first.option(), &live, &what, &mut failures).await
                else {
                    continue;
                };
                report(&mut failures, &what, wrong_files(&copy));
                put(&engine, &mut live, 30, 30).await;
                put(&engine, &mut live, 7, 107).await;
                delete(&engine, &mut live, 8).await;
                engine.commit().await.unwrap();
                report(&mut failures, &what, wrong_documents(&engine, &live).await);
                drop(engine);
                let what = format!("{what} and then as {second:?}");
                drop(check_open(&copy, second.option(), &live, &what, &mut failures).await);
                report(&mut failures, &what, wrong_files(&copy));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A change of kind that fails before it publishes leaves the field as it
/// was, also when the failed rebuild left a file under the number the next
/// segment takes: a reopen reclaims it, and the change succeeds when tried
/// again.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_change_of_kind_leaves_the_field_as_it_was() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        for fault in [Fault::CreateSegment(to), Fault::RenameManifest] {
            let what = format!("{pair}, {fault:?}");
            let storage = TestStorage::new();
            let engine = open(&storage, from.option()).await.unwrap();
            let mut live = fill(&engine).await;
            // What is uncommitted would reach the manifest while armed.
            engine.commit().await.unwrap();
            storage.arm(fault);
            let result = engine.update_field("vec", to.option(), REINDEX).await;
            let injected = storage.disarm();
            if result.is_ok() || injected == 0 {
                failures.push(format!(
                    "{what}: {injected} failures were injected, and it returned {:?}",
                    result.map(|outcome| outcome.classification)
                ));
            }
            if format!("{:?}", engine.schema().fields["vec"]) != format!("{:?}", from.option()) {
                failures.push(format!("{what}: the schema changed"));
            }
            report(&mut failures, &what, wrong_documents(&engine, &live).await);

            // The next segment of the field has the number the failed one
            // had; under both kinds, if the failure left its file.
            put(&engine, &mut live, 30, 30).await;
            engine.commit().await.unwrap();
            report(&mut failures, &what, wrong_documents(&engine, &live).await);
            let left: Vec<String> = storage
                .segment_files()
                .into_iter()
                .filter(|file| file.ends_with(&format!(".{}", to.suffix())))
                .collect();
            if fault == Fault::RenameManifest {
                let number = left.first().map(|file| file.split('.').next().unwrap());
                let same_number = listed(&storage)
                    .iter()
                    .any(|(id, _)| Some(id.as_str()) == number);
                if !same_number {
                    failures.push(format!(
                        "{what}: the failure left {left:?}, none under a number listed since"
                    ));
                }
            }
            drop(engine);

            for first in [from, to] {
                let what = format!("{what}, reopened as {first:?}");
                let copy = TestStorage::with_files(&storage.files());
                drop(check_open(&copy, first.option(), &live, &what, &mut failures).await);
                report(&mut failures, &what, wrong_files(&copy));
            }

            let engine = open(&storage, from.option()).await.unwrap();
            engine
                .update_field("vec", to.option(), REINDEX)
                .await
                .unwrap();
            let what = format!("{what}, tried again");
            report(&mut failures, &what, wrong_documents(&engine, &live).await);
            report(&mut failures, &what, wrong_kind(&storage, to));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A change of kind that has published succeeds whatever fails after: the
/// sync of the manifest, or the deletion of the replaced segments. The
/// next open reclaims what they leave.
#[tokio::test(flavor = "multi_thread")]
async fn a_published_change_of_kind_leaves_its_cleanup_to_the_next_open() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        for fault in [Fault::SyncManifest, Fault::DeleteSegment] {
            let what = format!("{pair}, {fault:?}");
            let storage = TestStorage::new();
            let engine = open(&storage, from.option()).await.unwrap();
            let live = fill(&engine).await;
            engine.commit().await.unwrap();
            storage.arm(fault);
            let result = engine.update_field("vec", to.option(), REINDEX).await;
            let injected = storage.disarm();
            if result.is_err() || injected == 0 {
                failures.push(format!(
                    "{what}: {injected} failures were injected, and it returned {:?}",
                    result.map(|outcome| outcome.classification)
                ));
            }
            report(&mut failures, &what, wrong_documents(&engine, &live).await);
            let old_kind = |file: &String| file.contains(&format!(".{}", from.suffix()));
            if !storage.segment_files().iter().any(old_kind) {
                failures.push(format!("{what}: no file of the old kind is left"));
            }
            drop(engine);
            for first in [to, from] {
                let what = format!("{what}, reopened as {first:?}");
                let copy = TestStorage::with_files(&storage.files());
                drop(check_open(&copy, first.option(), &live, &what, &mut failures).await);
                report(&mut failures, &what, wrong_kind(&copy, to));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A schema hook that fails leaves the engine on the new kind, with the
/// old schema saved or the new one. The engine goes on taking writes and
/// commits, the index opens under either schema, and the same change tried
/// again saves the schema.
#[tokio::test(flavor = "multi_thread")]
async fn a_schema_hook_that_fails_leaves_the_field_open_under_either_schema() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        let storage = TestStorage::new();
        let failing = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let hook = {
            let (failing, calls) = (Arc::clone(&failing), Arc::clone(&calls));
            move |_: &Schema| {
                calls.fetch_add(1, Ordering::SeqCst);
                if failing.load(Ordering::SeqCst) {
                    return Err(LaurusError::storage("injected failure: save the schema"));
                }
                Ok(())
            }
        };
        let engine = Engine::builder(
            storage.clone() as Arc<dyn Storage>,
            schema_with(from.option()),
        )
        .persist_schema_with(Arc::new(hook))
        .build()
        .await
        .unwrap();
        let mut live = fill(&engine).await;
        failing.store(true, Ordering::SeqCst);
        if engine
            .update_field("vec", to.option(), REINDEX)
            .await
            .is_ok()
        {
            failures.push(format!("{pair}: the hook's failure was not returned"));
        }
        put(&engine, &mut live, 30, 30).await;
        put(&engine, &mut live, 7, 107).await;
        delete(&engine, &mut live, 8).await;
        engine.commit().await.unwrap();
        report(&mut failures, &pair, wrong_documents(&engine, &live).await);

        for (first, second) in [(from, to), (to, from)] {
            let what = format!("{pair}, reopened as {first:?}");
            let mut live = live.clone();
            let copy = TestStorage::with_files(&storage.files());
            let Some(reopened) =
                check_open(&copy, first.option(), &live, &what, &mut failures).await
            else {
                continue;
            };
            put(&reopened, &mut live, 31, 31).await;
            reopened.commit().await.unwrap();
            drop(reopened);
            let what = format!("{what} and then as {second:?}");
            drop(check_open(&copy, second.option(), &live, &what, &mut failures).await);
            report(&mut failures, &what, wrong_files(&copy));
        }

        failing.store(false, Ordering::SeqCst);
        let before = calls.load(Ordering::SeqCst);
        let again = engine
            .update_field("vec", to.option(), REINDEX)
            .await
            .unwrap();
        if again.classification != FieldChangeKind::MetadataOnly
            || calls.load(Ordering::SeqCst) != before + 1
        {
            failures.push(format!(
                "{pair}: tried again, it was {:?} and the hook ran {} times",
                again.classification,
                calls.load(Ordering::SeqCst) - before
            ));
        }
        report(&mut failures, &pair, wrong_documents(&engine, &live).await);
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A commit does not wait for `update_field`, so one from another thread,
/// or of the commit timer, which is the same, can run between the rebuild
/// and saving the schema. A crash there leaves the index open under either
/// schema.
#[tokio::test(flavor = "multi_thread")]
async fn a_commit_before_the_schema_is_saved_opens_under_either_schema() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        let storage = TestStorage::new();
        let shared: Arc<OnceLock<Weak<Engine>>> = Arc::new(OnceLock::new());
        let in_hook = Arc::new(Mutex::new(None));
        let hook = {
            let storage = Arc::clone(&storage);
            let shared = Arc::clone(&shared);
            let in_hook = Arc::clone(&in_hook);
            move |_: &Schema| {
                let engine = shared.get().and_then(Weak::upgrade).unwrap();
                let before = engine.commit_generation();
                std::thread::spawn(move || {
                    tokio::runtime::Builder::new_current_thread()
                        .build()
                        .unwrap()
                        .block_on(engine.commit())
                        .unwrap();
                    assert_eq!(engine.commit_generation(), before + 1);
                })
                .join()
                .unwrap();
                *in_hook.lock().unwrap() = Some(storage.files());
                Ok(())
            }
        };
        let engine = Engine::builder(
            storage.clone() as Arc<dyn Storage>,
            schema_with(from.option()),
        )
        .persist_schema_with(Arc::new(hook))
        .build()
        .await
        .unwrap();
        let engine = Arc::new(engine);
        shared.set(Arc::downgrade(&engine)).unwrap();
        let live = fill(&engine).await;
        engine
            .update_field("vec", to.option(), REINDEX)
            .await
            .unwrap();
        drop(engine);
        let files = in_hook.lock().unwrap().take();
        let Some(files) = files else {
            failures.push(format!("{pair}: the hook did not run"));
            continue;
        };
        for first in [from, to] {
            let what = format!("{pair}, crash in the hook, opened as {first:?}");
            let copy = TestStorage::with_files(&files);
            drop(check_open(&copy, first.option(), &live, &what, &mut failures).await);
            report(&mut failures, &what, wrong_files(&copy));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A field opened under another kind than its segments' is searched, takes
/// writes of the new kind next to the old segments, and goes back. Filters,
/// a minimum score and a search with no field cover the segments of both
/// kinds, and the newest copy of a document wins across them.
#[tokio::test(flavor = "multi_thread")]
async fn a_field_with_segments_of_two_kinds_is_searched_across_them() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        let storage = TestStorage::new();
        let engine = open(&storage, from.option()).await.unwrap();
        let mut live = fill(&engine).await;
        engine.commit().await.unwrap();
        drop(engine);

        let what = format!("{pair}, opened");
        let Some(engine) = check_open(&storage, to.option(), &live, &what, &mut failures).await
        else {
            continue;
        };
        // An upsert of a document and a delete of another in the old
        // segments, and enough new documents that the commit does not
        // compact the field.
        put(&engine, &mut live, 0, 100).await;
        for id in 30..42 {
            put(&engine, &mut live, id, id).await;
        }
        delete(&engine, &mut live, 9).await;
        engine.commit().await.unwrap();
        let kinds: BTreeSet<String> = listed(&storage).into_iter().map(|(_, kind)| kind).collect();
        let both = [from, to].map(|kind| kind.suffix().to_string());
        if kinds != BTreeSet::from(both) {
            failures.push(format!("{pair}: the manifest lists kinds {kinds:?}"));
        }
        let what = format!("{pair}, both kinds");
        report(&mut failures, &what, wrong_documents(&engine, &live).await);

        // The old values of the upserted document do not find it.
        match search(&engine, request(0, 100)).await {
            Ok(hits) => {
                if hits.iter().any(|(id, score)| id == "d0" && *score > 0.5) {
                    failures.push(format!("{what}: the old values of d0 find it"));
                }
            }
            Err(e) => failures.push(format!("{what}: {e}")),
        }
        for word in ["even", "odd"] {
            let expected = ids(live.keys().filter(|id| title(**id) == word));
            let filtered = request(0, 100).filter_query(Box::new(TermQuery::new("title", word)));
            match search(&engine, filtered).await {
                Ok(hits) => {
                    let found: BTreeSet<String> = hits.into_iter().map(|(id, _)| id).collect();
                    if found != expected {
                        failures.push(format!("{what}: filter {word} finds {found:?}"));
                    }
                }
                Err(e) => failures.push(format!("{what}: {e}")),
            }
        }
        // d10 is in an old segment, d30 in a new one.
        for (id, seed) in [(10, 10), (30, 30)] {
            let min_score = request(seed, 100).vector_min_score(0.9);
            match search(&engine, min_score).await {
                Ok(hits) if hits.len() == 1 && hits[0].0 == format!("d{id}") => {}
                other => failures.push(format!("{what}: a minimum score finds {other:?}")),
            }
            match search(&engine, request_on(seed, 1, None)).await {
                Ok(hits)
                    if hits
                        .first()
                        .is_some_and(|(found, _)| *found == format!("d{id}")) => {}
                other => failures.push(format!("{what}: a search with no field finds {other:?}")),
            }
        }
        drop(engine);

        for again in [to, from] {
            let what = format!("{pair}, both kinds, reopened as {again:?}");
            drop(check_open(&storage, again.option(), &live, &what, &mut failures).await);
            report(&mut failures, &what, wrong_files(&storage));
        }

        // A change of another option rewrites the segments of both kinds.
        let engine = open(&storage, to.option()).await.unwrap();
        let outcome = engine
            .update_field(
                "vec",
                to.option_with(QuantizationMethod::Float16, None),
                REINDEX,
            )
            .await
            .unwrap();
        if outcome.classification != FieldChangeKind::Reindex {
            failures.push(format!("{pair}: {:?}", outcome.classification));
        }
        let what = format!("{pair}, rewritten");
        report(&mut failures, &what, wrong_documents(&engine, &live).await);
        report(&mut failures, &what, wrong_kind(&storage, to));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The merge of a commit takes segments of both kinds and writes one of
/// the field's kind.
#[tokio::test(flavor = "multi_thread")]
async fn a_commits_merge_takes_segments_of_both_kinds() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        let storage = TestStorage::new();
        let mut live = Live::new();
        let engine = open(&storage, from.option()).await.unwrap();
        for id in 0..5 {
            put(&engine, &mut live, id, id).await;
            engine.commit().await.unwrap();
        }
        drop(engine);
        let engine = match open(&storage, to.option()).await {
            Ok(engine) => engine,
            Err(e) => {
                failures.push(format!("{pair}: the open failed: {e}"));
                continue;
            }
        };
        for id in 5..10 {
            put(&engine, &mut live, id, id).await;
            engine.commit().await.unwrap();
        }
        // Ten segments of one document are one merge.
        report(&mut failures, &pair, wrong_documents(&engine, &live).await);
        report(&mut failures, &pair, wrong_kind(&storage, to));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A manifest an older build wrote has no kind per segment, and one older
/// still has no record of the field either. The kind of its segments is the
/// recorded one, or the one whose file exists, also when an older build that
/// had no check wrote segments of two kinds. The index opens under either
/// schema and records the kinds with its next commit.
#[tokio::test(flavor = "multi_thread")]
async fn a_manifest_without_kinds_opens_under_either_schema() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        for mixed in [false, true] {
            let storage = TestStorage::new();
            let engine = open(&storage, from.option()).await.unwrap();
            let mut live = fill(&engine).await;
            engine.commit().await.unwrap();
            drop(engine);
            if mixed {
                let engine = match open(&storage, to.option()).await {
                    Ok(engine) => engine,
                    Err(e) => {
                        failures.push(format!("{pair}: the open failed: {e}"));
                        continue;
                    }
                };
                for id in 50..60 {
                    put(&engine, &mut live, id, id).await;
                }
                engine.commit().await.unwrap();
            }

            for with_record in [true, false] {
                for first in [from, to] {
                    let what = format!(
                        "{pair}, mixed: {mixed}, record: {with_record}, opened as {first:?}"
                    );
                    let copy = TestStorage::with_files(&storage.files());
                    let mut old = manifest(&copy);
                    for segment in old["segments"].as_array_mut().unwrap() {
                        segment.as_object_mut().unwrap().remove("kind");
                    }
                    if !with_record {
                        old.as_object_mut().unwrap().remove("field");
                    }
                    save_checksummed_json(copy.as_ref(), MANIFEST, None, &old).unwrap();

                    let mut live = live.clone();
                    let Some(engine) =
                        check_open(&copy, first.option(), &live, &what, &mut failures).await
                    else {
                        continue;
                    };
                    put(&engine, &mut live, 30, 30).await;
                    engine.commit().await.unwrap();
                    drop(engine);
                    report(&mut failures, &what, wrong_files(&copy));
                    let kinds: BTreeSet<String> =
                        listed(&copy).into_iter().map(|(_, kind)| kind).collect();
                    let mut expected = BTreeSet::from([from, first].map(Kind::suffix));
                    if mixed {
                        expected.insert(to.suffix());
                    }
                    if kinds != expected.into_iter().map(str::to_string).collect() {
                        failures.push(format!("{what}: the manifest lists kinds {kinds:?}"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A file of another kind under the number of a listed segment is
/// reclaimed when the manifest records the segment's kind, and kept when
/// the kind is only inferred.
#[tokio::test(flavor = "multi_thread")]
async fn a_file_of_another_kind_is_reclaimed_only_under_a_recorded_kind() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        let storage = TestStorage::new();
        let engine = open(&storage, from.option()).await.unwrap();
        let live = fill(&engine).await;
        engine.commit().await.unwrap();
        drop(engine);
        let Some((segment, _)) = listed(&storage).pop_first() else {
            failures.push(format!("{pair}: the manifest lists no segment"));
            continue;
        };
        for recorded in [true, false] {
            let what = format!("{pair}, recorded: {recorded}");
            let copy = TestStorage::with_files(&storage.files());
            if !recorded {
                let mut old = manifest(&copy);
                for listed in old["segments"].as_array_mut().unwrap() {
                    listed.as_object_mut().unwrap().remove("kind");
                }
                save_checksummed_json(copy.as_ref(), MANIFEST, None, &old).unwrap();
            }
            let other = [
                format!("vector/vec/{segment}.{}", to.suffix()),
                format!("vector/vec/{segment}.{}.f32", to.suffix()),
            ];
            for file in &other {
                copy.write(file, b"not a segment");
            }
            drop(check_open(&copy, from.option(), &live, &what, &mut failures).await);
            for file in &other {
                if copy.file_exists(file) == recorded {
                    failures.push(format!("{what}: {file} exists: {}", !recorded));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A change of the kind of field `a` leaves field `a/b`, whose directory is
/// inside the one of `a`, and its codebook alone.
#[tokio::test(flavor = "multi_thread")]
async fn a_change_of_kind_leaves_the_files_of_a_nested_field_alone() {
    let mut failures = Vec::new();
    for (pair, from, to) in pairs() {
        let schema = |a: Kind| {
            Schema::builder()
                .add_field("a", a.option())
                .add_field("a/b", from.option())
                .build()
        };
        let storage = TestStorage::new();
        let engine = Engine::new(storage.clone() as Arc<dyn Storage>, schema(from))
            .await
            .unwrap();
        for id in 0..6 {
            let doc = Document::builder()
                .add_vector("a", values(id))
                .add_vector("a/b", values(50 + id))
                .build();
            engine.put_document(&format!("d{id}"), doc).await.unwrap();
        }
        engine.commit().await.unwrap();
        storage.write("vector/a/b.pqcb", b"a codebook");
        let nested = |storage: &TestStorage| -> Files {
            storage
                .files()
                .into_iter()
                .filter(|(name, _)| name.starts_with("vector/a/b"))
                .collect()
        };
        let before = nested(&storage);
        engine
            .update_field("a", to.option(), REINDEX)
            .await
            .unwrap();
        for (what, reopen) in [("after", false), ("reopened", true)] {
            let reopened = if reopen {
                Some(
                    Engine::new(storage.clone() as Arc<dyn Storage>, schema(to))
                        .await
                        .unwrap(),
                )
            } else {
                None
            };
            let engine = reopened.as_ref().unwrap_or(&engine);
            for (field, seed) in [("a", 3), ("a/b", 53)] {
                let request = request_on(seed, 1, Some(vec![field.to_string()]));
                match search(engine, request).await {
                    Ok(hits)
                        if hits
                            .first()
                            .is_some_and(|(id, score)| id == "d3" && *score > 0.9) => {}
                    other => failures.push(format!("{pair}, {what}: {field} finds {other:?}")),
                }
            }
            if nested(&storage) != before {
                failures.push(format!("{pair}, {what}: the files of a/b changed"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
