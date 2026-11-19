//! Correctness of ingestion driven from several threads at once (#546).
//!
//! Today the lexical writer sits behind a single mutex in `LexicalStore`,
//! so concurrent `put_document` / `add_document` calls are serialized and
//! these tests pass by construction. They exist because the DWPT work
//! removes that serialization: every invariant asserted here is one that
//! per-thread ingestion can break silently — a document indexed but not
//! findable, an update that leaves both versions live, counters that
//! drift from the documents actually present, or WAL state that replays
//! to something different from what was acknowledged.
//!
//! Each test therefore asserts an *observable* property, never an
//! implementation detail, so it keeps its meaning across the refactor.

use std::sync::Arc;

use laurus::lexical::TextOption;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document, Engine, FieldOption, Schema, SearchRequestBuilder};

/// Threads used by the concurrent tests. Fixed rather than derived from
/// the host's core count so a CI runner and a workstation exercise the
/// same interleaving pressure.
const THREADS: usize = 8;
/// Documents per thread.
const PER_THREAD: usize = 50;

fn schema() -> Schema {
    Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("owner", FieldOption::Text(TextOption::default()))
        .build()
}

fn doc(title: &str, owner: &str) -> Document {
    Document::builder()
        .add_field("title", DataValue::Text(title.into()))
        .add_field("owner", DataValue::Text(owner.into()))
        .build()
}

/// Run `body` on `THREADS` OS threads, each with its own current-thread
/// runtime, and wait for all of them.
///
/// Deliberately not `tokio::spawn` on a shared multi-thread runtime: the
/// point is to have genuinely concurrent callers of the engine, not tasks
/// that a single executor may end up running one after another.
fn in_parallel<F>(body: F)
where
    F: Fn(usize) + Send + Sync,
{
    let body = &body;
    std::thread::scope(|scope| {
        for t in 0..THREADS {
            scope.spawn(move || body(t));
        }
    });
}

/// Every document ingested concurrently must be present exactly once.
///
/// The failure this guards against is the one that stays invisible to
/// `document_count`: a document that was accepted, counted, and yet is not
/// retrievable — or is retrievable twice.
#[test]
fn concurrent_ingest_keeps_every_document_exactly_once() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let setup = tokio::runtime::Runtime::new().unwrap();
    let engine = Arc::new(setup.block_on(Engine::new(storage, schema())).unwrap());

    {
        let engine = Arc::clone(&engine);
        in_parallel(move |t| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                for i in 0..PER_THREAD {
                    let id = format!("t{t}-d{i}");
                    engine
                        .put_document(&id, doc(&format!("title {t} {i}"), &format!("owner{t}")))
                        .await
                        .unwrap();
                }
            });
        });
    }

    setup.block_on(engine.commit()).unwrap();

    let total = THREADS * PER_THREAD;
    let stats = engine.stats().unwrap();
    assert_eq!(
        stats.document_count, total as u64,
        "the engine must count every concurrently ingested document"
    );

    // Counters agreeing is not enough: each document must actually be
    // retrievable, and exactly once.
    setup.block_on(async {
        for t in 0..THREADS {
            for i in 0..PER_THREAD {
                let id = format!("t{t}-d{i}");
                let docs = engine.get_documents(&id).await.unwrap();
                assert_eq!(docs.len(), 1, "{id} must resolve to exactly one document");
            }
        }
    });
}

/// Concurrent updates of the same ids must leave exactly one version each.
///
/// `put_document` is delete-then-add. Per-thread ingestion is where that
/// can go wrong quietly: if a thread cannot see a document another thread
/// has buffered, its update adds a second live copy instead of replacing
/// the first, and the duplicate only shows up as an extra search hit.
#[test]
fn concurrent_updates_leave_exactly_one_version_per_id() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let setup = tokio::runtime::Runtime::new().unwrap();
    let engine = Arc::new(setup.block_on(Engine::new(storage, schema())).unwrap());

    // Seed one version of every id, committed.
    setup.block_on(async {
        for i in 0..PER_THREAD {
            engine
                .put_document(&format!("shared-{i}"), doc("original", "seed"))
                .await
                .unwrap();
        }
        engine.commit().await.unwrap();
    });

    // Every thread rewrites every id: each id ends up with one winner,
    // whichever thread got there last.
    {
        let engine = Arc::clone(&engine);
        in_parallel(move |t| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                for i in 0..PER_THREAD {
                    engine
                        .put_document(
                            &format!("shared-{i}"),
                            doc(&format!("rewritten by {t}"), &format!("owner{t}")),
                        )
                        .await
                        .unwrap();
                }
            });
        });
    }

    setup.block_on(engine.commit()).unwrap();

    setup.block_on(async {
        for i in 0..PER_THREAD {
            let id = format!("shared-{i}");
            let docs = engine.get_documents(&id).await.unwrap();
            assert_eq!(
                docs.len(),
                1,
                "{id} must have exactly one live version after concurrent updates"
            );
        }
    });

    let stats = engine.stats().unwrap();
    assert_eq!(
        stats.document_count, PER_THREAD as u64,
        "concurrent updates must not inflate the document count"
    );
}

/// Documents ingested concurrently and never committed must all come back
/// after a reopen.
///
/// This is the crash-shaped invariant. Per-thread ingestion aggregates a
/// WAL checkpoint across writers, and taking the maximum rather than the
/// minimum would silently drop the records of whichever writer lagged —
/// visible only after a restart, never on the happy path.
#[test]
fn uncommitted_concurrent_ingest_survives_a_reopen() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let total = THREADS * PER_THREAD;

    {
        let setup = tokio::runtime::Runtime::new().unwrap();
        let engine = Arc::new(
            setup
                .block_on(Engine::new(storage.clone(), schema()))
                .unwrap(),
        );
        let ingest = Arc::clone(&engine);
        in_parallel(move |t| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                for i in 0..PER_THREAD {
                    ingest
                        .put_document(&format!("t{t}-d{i}"), doc("durable", &format!("owner{t}")))
                        .await
                        .unwrap();
                }
            });
        });
        // Dropped without commit: durability must come from the WAL alone.
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let reopened = rt.block_on(Engine::new(storage, schema())).unwrap();
    rt.block_on(reopened.commit()).unwrap();

    let stats = reopened.stats().unwrap();
    assert_eq!(
        stats.document_count, total as u64,
        "every acknowledged document must replay from the WAL after a reopen"
    );
    rt.block_on(async {
        for t in 0..THREADS {
            for i in 0..PER_THREAD {
                let id = format!("t{t}-d{i}");
                let docs = reopened.get_documents(&id).await.unwrap();
                assert_eq!(docs.len(), 1, "{id} must survive the reopen");
            }
        }
    });
}

/// Concurrent deletes must remove exactly the targeted documents.
///
/// Deletions are group-committed and the writer resolves a document to its
/// segment by doc-id range. Per-thread ingestion makes those ranges
/// overlap, at which point a delete can be attributed to a segment that
/// never held the document — inflating the deleted count while leaving the
/// real one alive.
#[test]
fn concurrent_deletes_remove_exactly_their_targets() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let setup = tokio::runtime::Runtime::new().unwrap();
    let engine = Arc::new(setup.block_on(Engine::new(storage, schema())).unwrap());

    setup.block_on(async {
        for t in 0..THREADS {
            for i in 0..PER_THREAD {
                engine
                    .put_document(&format!("t{t}-d{i}"), doc("target", &format!("owner{t}")))
                    .await
                    .unwrap();
            }
        }
        engine.commit().await.unwrap();
    });

    // Each thread deletes only its own even-numbered documents.
    {
        let engine = Arc::clone(&engine);
        in_parallel(move |t| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                for i in (0..PER_THREAD).step_by(2) {
                    engine
                        .delete_documents(&format!("t{t}-d{i}"))
                        .await
                        .unwrap();
                }
            });
        });
    }

    setup.block_on(engine.commit()).unwrap();

    setup.block_on(async {
        for t in 0..THREADS {
            for i in 0..PER_THREAD {
                let id = format!("t{t}-d{i}");
                let docs = engine.get_documents(&id).await.unwrap();
                let expected = usize::from(i % 2 != 0);
                assert_eq!(
                    docs.len(),
                    expected,
                    "{id} must be {} after the concurrent deletes",
                    if expected == 0 { "gone" } else { "present" }
                );
            }
        }
    });

    let survivors = (THREADS * PER_THREAD - THREADS * PER_THREAD.div_ceil(2)) as u64;
    let stats = engine.stats().unwrap();
    assert_eq!(
        stats.document_count, survivors,
        "the count must match the documents that actually survived"
    );
}

/// Dynamic puts from several threads that add the same new fields all
/// succeed, and leave each field in the schema once, with the type of its
/// values (Issue #1498).
///
/// Every document has a field all threads add, one only its own thread
/// adds and one that two threads add, so puts that need a field race each
/// other's `add_field`, and each put then coerces its values under the
/// schema lock to the type the field was added with.
#[test]
fn concurrent_dynamic_puts_add_each_new_field_once() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let setup = tokio::runtime::Runtime::new().unwrap();
    // `schema()` keeps the default policy, `DynamicFieldPolicy::Dynamic`.
    let engine = Arc::new(setup.block_on(Engine::new(storage, schema())).unwrap());
    let doc = |t: usize, i: usize| {
        Document::builder()
            .add_field("title", DataValue::Text(format!("title {t} {i}")))
            .add_field("shared", DataValue::Int64(i as i64))
            .add_field(format!("own_{t}"), DataValue::Text(format!("word{i}")))
            .add_field(
                format!("pair_{}", t / 2),
                DataValue::Bool(i.is_multiple_of(2)),
            )
            .build()
    };

    {
        let engine = Arc::clone(&engine);
        in_parallel(move |t| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                for i in 0..PER_THREAD {
                    engine
                        .put_document(&format!("t{t}-d{i}"), doc(t, i))
                        .await
                        .unwrap();
                }
            });
        });
    }

    setup.block_on(engine.commit()).unwrap();

    let schema = engine.schema();
    let mut wrong = Vec::new();
    if !matches!(schema.fields.get("shared"), Some(FieldOption::Integer(_))) {
        wrong.push(format!("shared: {:?}", schema.fields.get("shared")));
    }
    for t in 0..THREADS {
        let own = format!("own_{t}");
        if !matches!(schema.fields.get(&own), Some(FieldOption::Text(_))) {
            wrong.push(format!("{own}: {:?}", schema.fields.get(&own)));
        }
        let pair = format!("pair_{}", t / 2);
        if !matches!(schema.fields.get(&pair), Some(FieldOption::Boolean(_))) {
            wrong.push(format!("{pair}: {:?}", schema.fields.get(&pair)));
        }
    }
    // title and owner from `schema()`, shared, one own_ per thread and one
    // pair_ per two threads.
    let expected_fields = 3 + THREADS + THREADS / 2;
    if schema.fields.len() != expected_fields {
        wrong.push(format!(
            "{} fields, not {expected_fields}: {:?}",
            schema.fields.len(),
            schema.fields.keys().collect::<Vec<_>>()
        ));
    }

    setup.block_on(async {
        for t in 0..THREADS {
            for i in 0..PER_THREAD {
                let id = format!("t{t}-d{i}");
                let docs = engine.get_documents(&id).await.unwrap();
                let want = doc(t, i);
                let fits = docs.len() == 1
                    && want
                        .fields
                        .iter()
                        .all(|(name, value)| docs[0].fields.get(name) == Some(value));
                if !fits {
                    wrong.push(format!("{id}: {docs:?}"));
                }
            }
        }
        let request = SearchRequestBuilder::new()
            .query_dsl(format!("shared:[0 TO {}]", PER_THREAD - 1))
            .limit(THREADS * PER_THREAD + 1)
            .build();
        let hits = engine.search(request).await.unwrap().len();
        if hits != THREADS * PER_THREAD {
            wrong.push(format!("shared:[..] finds {hits} documents"));
        }
    });
    assert!(wrong.is_empty(), "{wrong:#?}");
}
