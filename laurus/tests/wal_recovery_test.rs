use tempfile::TempDir;

use laurus::Engine;
use laurus::lexical::TermQuery;
use laurus::lexical::TextOption;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::FlatOption;
use laurus::{CommitPolicy, FieldOption, LexicalSearchQuery, Schema, WalSyncPolicy};
use laurus::{DataValue, Document};

#[tokio::test(flavor = "multi_thread")]
async fn test_wal_recovery_uncommitted() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Config
    let vector_opt = FlatOption::default();
    let lexical_opt = TextOption::default();

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(lexical_opt))
        .add_field("embedding", FieldOption::Flat(vector_opt))
        .build();

    // 3. Round 1: Index but DO NOT commit
    {
        let engine = Engine::new(storage.clone(), config.clone()).await?;

        // Initial state
        let query = Box::new(TermQuery::new("title", "rust"));
        let search_request = laurus::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(query))
            .build();
        let search_results = engine.search(search_request).await?.results;
        assert_eq!(search_results.len(), 0);

        let doc1 = Document::builder()
            .add_field("title", DataValue::Text("Rust Programming".into()))
            .add_field("embedding", DataValue::Vector(vec![0.1; 128]))
            .build();

        engine.put_document("doc1", doc1).await?;

        // Drop engine WITHOUT commit
    }

    // 4. Round 2: Recover from WAL
    {
        // Re-open engine on SAME storage
        let engine = Engine::new(storage.clone(), config.clone()).await?;

        // Commit to ensure flushed to searchable index
        engine.commit().await?;

        // Should have recovered doc1 from WAL and now committed
        let query = Box::new(TermQuery::new("title", "rust"));
        let search_request = laurus::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(query))
            .build();
        let search_results = engine.search(search_request).await?.results;
        assert_eq!(
            search_results.len(),
            1,
            "Document should be recovered from WAL"
        );
    }

    Ok(())
}

/// Under the group-commit policy (Issue #542, Phase 4) appends defer their
/// fsync, so durability is reached via [`laurus::Engine::flush_wal`] rather than
/// per-record. A record made durable by `flush_wal` (without a full commit) must
/// still recover after an uncommitted reopen, just like the per-record case.
#[tokio::test(flavor = "multi_thread")]
async fn test_wal_recovery_group_commit_flush_wal() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("embedding", FieldOption::Flat(FlatOption::default()))
        .build();

    // Round 1: index under the group-commit policy and force the WAL durable
    // with flush_wal(), but DO NOT commit.
    {
        let engine = Engine::builder(storage.clone(), config.clone())
            .wal_sync_policy(WalSyncPolicy::group_with_defaults())
            .build()
            .await?;

        let doc1 = Document::builder()
            .add_field("title", DataValue::Text("Rust Programming".into()))
            .add_field("embedding", DataValue::Vector(vec![0.1; 128]))
            .build();
        engine.put_document("doc1", doc1).await?;

        // The group policy defers the fsync; flush_wal makes the partial batch
        // durable. Drop the engine WITHOUT commit.
        engine.flush_wal()?;
    }

    // Round 2: recover from the WAL.
    {
        let engine = Engine::new(storage.clone(), config.clone()).await?;
        engine.commit().await?;

        let query = Box::new(TermQuery::new("title", "rust"));
        let search_request = laurus::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(query))
            .build();
        let search_results = engine.search(search_request).await?.results;
        assert_eq!(
            search_results.len(),
            1,
            "flush_wal'd group-commit record should recover from WAL"
        );
    }

    Ok(())
}

/// #890: an engine with `CommitPolicy::EveryDocs` must keep the exact same
/// crash/recovery semantics as manual commits. Ingesting a number of documents
/// that is not a multiple of the threshold leaves an uncommitted tail in the
/// WAL; a crash (drop without an explicit final commit) followed by a reopen
/// must recover every document — the auto-committed prefix from the stores and
/// the uncommitted tail from WAL replay — with no loss or duplication.
#[tokio::test(flavor = "multi_thread")]
async fn test_commit_policy_every_docs_crash_recovery() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("embedding", FieldOption::Flat(FlatOption::default()))
        .build();

    // Round 1: ingest 5 docs under EveryDocs(3). Doc 3 auto-commits (docs 1-3
    // durable in the stores); docs 4-5 remain uncommitted in the WAL. Drop the
    // engine WITHOUT an explicit final commit — the crash.
    {
        let engine = Engine::builder(storage.clone(), config.clone())
            .commit_policy(CommitPolicy::EveryDocs(3))
            .build()
            .await?;

        for i in 0..5 {
            let doc = Document::builder()
                .add_field("title", DataValue::Text(format!("rust doc {i}")))
                .add_field("embedding", DataValue::Vector(vec![0.1; 128]))
                .build();
            engine.put_document(&format!("doc{i}"), doc).await?;
        }
        // Drop without commit — docs 4-5 are only in the WAL.
    }

    // Round 2: reopen on the same storage. Recovery replays the uncommitted
    // tail; all 5 documents must be present.
    {
        let engine = Engine::new(storage.clone(), config.clone()).await?;
        let query = Box::new(TermQuery::new("title", "rust"));
        let search_request = laurus::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(query))
            .limit(10)
            .build();
        let results = engine.search(search_request).await?.results;
        assert_eq!(
            results.len(),
            5,
            "all 5 docs must survive a crash under EveryDocs: the auto-committed \
             prefix from the stores plus the uncommitted tail from WAL replay"
        );
    }

    Ok(())
}
