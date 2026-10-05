//! One-shot search query execution.
//!
//! Opens the index and hands the raw query string to the engine as a
//! [`SearchQuery::Dsl`](laurus::SearchQuery::Dsl). The engine parses it
//! with its own `UnifiedQueryParser`
//! ([`Engine::unified_query_parser`](laurus::Engine::unified_query_parser)),
//! which is built from the index's `PerFieldAnalyzer` — so a field
//! configured with a Japanese (Lindera) analyzer in `schema.toml` is
//! analyzed with that analyzer at query time too. Building a
//! `LexicalQueryParser::with_standard_analyzer()` here instead silently
//! ignored per-field analyzers and made Japanese queries return nothing.

use std::path::Path;

use anyhow::{Context, Result};
use laurus::vector::Vector;
use laurus::{RescoreOptions, SearchRequestBuilder, SearchResult};

use crate::cli::SearchCommand;
use crate::context;
use crate::output::{self, OutputFormat};

/// Open the index and execute the query, returning the engine's raw hits.
///
/// Split out of [`run`] so tests can assert on hits without capturing
/// stdout.
///
/// # Arguments
///
/// * `cmd` - Parsed [`SearchCommand`] carrying the query string, limit and
///   offset.
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Errors
///
/// Returns an error if the index cannot be opened, the query string is not
/// valid DSL, it references a field outside the schema, or execution fails.
pub(crate) async fn execute(cmd: &SearchCommand, index_dir: &Path) -> Result<Vec<SearchResult>> {
    let engine = context::open_index(index_dir).await?;

    let mut builder = SearchRequestBuilder::new()
        .query_dsl(cmd.query.clone())
        .limit(cmd.limit)
        .offset(cmd.offset);
    if !cmd.highlight.is_empty() {
        builder = builder.highlight(cmd.highlight.clone());
    }
    if let Some(rescore) = rescore_options(cmd)? {
        builder = builder.rescore(rescore);
    }

    engine
        .search(builder.build())
        .await
        .context("Failed to execute search")
}

/// The rescore the `--rescore-*` flags ask for, if any (Issue #1351).
///
/// clap already enforces that `--rescore-field` comes with exactly one of
/// `--rescore-text` and `--rescore-vectors`; the engine validates the
/// values.
///
/// # Errors
///
/// Returns an error if `--rescore-vectors` is not a JSON array of numeric
/// arrays.
fn rescore_options(cmd: &SearchCommand) -> Result<Option<RescoreOptions>> {
    let Some(field) = &cmd.rescore_field else {
        return Ok(None);
    };
    let options = match (&cmd.rescore_text, &cmd.rescore_vectors) {
        (Some(text), _) => RescoreOptions::late_interaction_text(field.clone(), text.clone()),
        (None, Some(json)) => {
            let vectors: Vec<Vec<f32>> = serde_json::from_str(json)
                .context("--rescore-vectors must be a JSON array of numeric arrays")?;
            RescoreOptions::late_interaction(
                field.clone(),
                vectors.into_iter().map(Vector::new).collect(),
            )
        }
        (None, None) => anyhow::bail!("--rescore-field needs --rescore-text or --rescore-vectors"),
    };
    Ok(Some(match cmd.rescore_window {
        Some(window_size) => options.window_size(window_size),
        None => options,
    }))
}

/// Execute a search command against the index and print the results.
///
/// Opens the index at `index_dir` and runs the query string as DSL,
/// letting the engine parse it with the schema's own per-field analyzers.
///
/// # Arguments
///
/// * `cmd` - Parsed [`SearchCommand`] containing the query string, limit,
///   and offset.
/// * `index_dir` - Path to the index directory holding the index.
/// * `format` - The desired output format (table or JSON).
///
/// # Returns
///
/// Returns `Ok(())` on success after printing results.
///
/// # Errors
///
/// See [`execute`].
pub async fn run(cmd: SearchCommand, index_dir: &Path, format: OutputFormat) -> Result<()> {
    let results = execute(&cmd, index_dir).await?;
    output::print_search_results(&results, format);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A schema whose `body` field uses a bigram analyzer. Bigrams give a
    /// dictionary-free stand-in for a morphological tokenizer: one bare
    /// query term analyzes into several tokens, exactly the situation the
    /// CLI used to get wrong (it always analyzed queries with
    /// `StandardAnalyzer`, ignoring this per-field analyzer).
    const BIGRAM_SCHEMA: &str = r#"
default_fields = ["body"]

[fields.body.Text]
indexed = true
stored = true
analyzer = "ja_bigram"

[analyzers.ja_bigram]
tokenizer = { type = "ngram", min_gram = 2, max_gram = 2 }
"#;

    fn search_command(query: &str) -> SearchCommand {
        SearchCommand {
            query: query.to_string(),
            limit: 10,
            offset: 0,
            highlight: Vec::new(),
            rescore_field: None,
            rescore_text: None,
            rescore_vectors: None,
            rescore_window: None,
        }
    }

    /// The direct regression: a query against a field with a non-default
    /// analyzer must be analyzed with *that* analyzer, not
    /// `StandardAnalyzer`. Without the fix, the query term never matches
    /// any bigram token and this returns zero hits.
    #[tokio::test]
    async fn search_analyzes_the_query_with_the_schema_analyzer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), BIGRAM_SCHEMA).unwrap();

        let engine = context::open_index(dir.path()).await.unwrap();
        engine
            .put_document(
                "doc1",
                laurus::Document::builder()
                    .add_field("body", "吾輩は猫である")
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();
        // Issue #1086: `Engine::build()` now takes an exclusive lock on
        // the storage directory, so this handle must close before
        // `execute` opens a second `Engine` over the same directory --
        // exactly what a real CLI invocation would do anyway (each
        // `laurus` command is a separate process that exits when done).
        drop(engine);

        let results = execute(&search_command("吾輩は猫"), dir.path())
            .await
            .unwrap();
        assert!(
            !results.is_empty(),
            "query must match via the field's own analyzer"
        );
    }

    /// A bare (unquoted) multi-token query ORs its analyzed tokens, so a
    /// document sharing only *some* of them still matches — the CLI-level
    /// combination of the analyzer fix (this file) and the OR-not-phrase
    /// change (`lexical/query/parser.rs`).
    #[tokio::test]
    async fn search_ors_the_analyzed_tokens_of_a_bare_query() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), BIGRAM_SCHEMA).unwrap();

        let engine = context::open_index(dir.path()).await.unwrap();
        engine
            .put_document(
                "doc1",
                laurus::Document::builder()
                    .add_field("body", "吾輩は猫である")
                    .build(),
            )
            .await
            .unwrap();
        engine
            .put_document(
                "doc2",
                laurus::Document::builder()
                    .add_field("body", "猫が吾輩を見た")
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();
        // Issue #1086: `Engine::build()` now takes an exclusive lock on
        // the storage directory, so this handle must close before
        // `execute` opens a second `Engine` over the same directory --
        // exactly what a real CLI invocation would do anyway (each
        // `laurus` command is a separate process that exits when done).
        drop(engine);

        let results = execute(&search_command("吾輩は猫"), dir.path())
            .await
            .unwrap();
        assert_eq!(
            results.len(),
            2,
            "both docs share bigrams with the query, expected 2 hits, got {results:?}"
        );
    }

    /// `_id` must be queryable — the reserved field is not in
    /// `schema.fields`, so `known_fields` (engine.rs) must special-case it.
    #[tokio::test]
    async fn search_by_reserved_id_field_is_not_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), BIGRAM_SCHEMA).unwrap();

        let engine = context::open_index(dir.path()).await.unwrap();
        engine
            .put_document(
                "doc1",
                laurus::Document::builder()
                    .add_field("body", "吾輩は猫である")
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();
        // Issue #1086: `Engine::build()` now takes an exclusive lock on
        // the storage directory, so this handle must close before
        // `execute` opens a second `Engine` over the same directory --
        // exactly what a real CLI invocation would do anyway (each
        // `laurus` command is a separate process that exits when done).
        drop(engine);

        let results = execute(&search_command("_id:doc1"), dir.path())
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    /// A typo'd / undeclared field name must fail with a message naming
    /// it — the DSL path validates field references, unlike the previous
    /// hand-built `LexicalQueryParser` which silently matched nothing.
    #[tokio::test]
    async fn search_rejects_a_typoed_field_with_a_named_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), BIGRAM_SCHEMA).unwrap();
        context::open_index(dir.path()).await.unwrap();

        let err = execute(&search_command("bod:猫"), dir.path())
            .await
            .expect_err("undeclared field must be rejected");
        // `execute` wraps the engine's error with `.context(...)`; anyhow's
        // `Display` shows only the outermost message, so check the full
        // chain (`Debug`, which anyhow renders with "Caused by: ...").
        let chain = format!("{err:?}");
        assert!(chain.contains("bod"), "{chain}");
    }

    /// `limit` and `offset` from the command must still be honoured
    /// through the DSL path.
    #[tokio::test]
    async fn search_honours_limit_and_offset() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), BIGRAM_SCHEMA).unwrap();

        let engine = context::open_index(dir.path()).await.unwrap();
        for i in 0..5 {
            engine
                .put_document(
                    &format!("doc{i}"),
                    laurus::Document::builder()
                        .add_field("body", "吾輩は猫である")
                        .build(),
                )
                .await
                .unwrap();
        }
        engine.commit().await.unwrap();
        // Issue #1086: `Engine::build()` now takes an exclusive lock on
        // the storage directory, so this handle must close before
        // `execute` opens a second `Engine` over the same directory --
        // exactly what a real CLI invocation would do anyway (each
        // `laurus` command is a separate process that exits when done).
        drop(engine);

        // The bigram analyzer needs a 2+ character query to produce any
        // token at all (min_gram = 2).
        let limited = execute(
            &SearchCommand {
                limit: 2,
                ..search_command("猫で")
            },
            dir.path(),
        )
        .await
        .unwrap();
        assert_eq!(limited.len(), 2);
    }

    /// Table-formatting a Japanese result must not panic — this is the
    /// `output.rs` char-boundary fix exercised through the real search
    /// path.
    #[tokio::test]
    async fn search_on_a_japanese_result_prints_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), BIGRAM_SCHEMA).unwrap();

        let engine = context::open_index(dir.path()).await.unwrap();
        let long_text = "吾輩は猫である。名前はまだ無い。".repeat(10);
        engine
            .put_document(
                "doc1",
                laurus::Document::builder()
                    .add_field("body", long_text)
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();
        // Issue #1086: `Engine::build()` now takes an exclusive lock on
        // the storage directory, so this handle must close before
        // `execute` opens a second `Engine` over the same directory --
        // exactly what a real CLI invocation would do anyway (each
        // `laurus` command is a separate process that exits when done).
        drop(engine);

        // 2+ characters: the bigram analyzer needs at least min_gram (2).
        let results = execute(&search_command("猫で"), dir.path()).await.unwrap();
        output::print_search_results(&results, OutputFormat::Table);
        output::print_search_results(&results, OutputFormat::Json);
    }

    /// `--highlight body` (Issue #1134) must reach the engine and come back
    /// as fragments on the matching hit.
    #[tokio::test]
    async fn search_with_highlight_returns_fragments_for_the_requested_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), BIGRAM_SCHEMA).unwrap();

        let engine = context::open_index(dir.path()).await.unwrap();
        engine
            .put_document(
                "doc1",
                laurus::Document::builder()
                    .add_field("body", "吾輩は猫である")
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();
        // Issue #1086: `Engine::build()` now takes an exclusive lock on
        // the storage directory, so this handle must close before
        // `execute` opens a second `Engine` over the same directory --
        // exactly what a real CLI invocation would do anyway (each
        // `laurus` command is a separate process that exits when done).
        drop(engine);

        let results = execute(
            &SearchCommand {
                highlight: vec!["body".to_string()],
                ..search_command("吾輩は猫")
            },
            dir.path(),
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 1);
        let fragments = results[0]
            .highlights
            .get("body")
            .expect("body must be highlighted");
        assert!(
            fragments.iter().any(|f| f.contains("<mark>")),
            "{fragments:?}"
        );
    }

    /// Without `--highlight`, every hit's `highlights` stays empty — the
    /// default `SearchCommand` (no flag) must not turn highlighting on.
    #[tokio::test]
    async fn search_without_highlight_flag_leaves_highlights_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), BIGRAM_SCHEMA).unwrap();

        let engine = context::open_index(dir.path()).await.unwrap();
        engine
            .put_document(
                "doc1",
                laurus::Document::builder()
                    .add_field("body", "吾輩は猫である")
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();
        drop(engine);

        let results = execute(&search_command("吾輩は猫"), dir.path())
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.highlights.is_empty()));
    }

    /// The corpus of `laurus/tests/late_interaction_rescore_test.rs`.
    const RESCORE_SCHEMA: &str = r#"
[fields.title.Text]
indexed = true
stored = true

[fields.vec.Flat]
dimension = 2

[fields.tokens.MultiVector]
dimension = 2
distance = "DotProduct"
"#;

    /// Issue #1351: `--rescore-*` rescores the top results, giving the
    /// Rust API's ranking for the shared corpus.
    #[tokio::test]
    async fn search_rescores_with_token_vectors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), RESCORE_SCHEMA).unwrap();
        let engine = context::open_index(dir.path()).await.unwrap();
        let corpus = [
            ("a", "rust", [1.0, 0.0], vec![vec![0.1, 0.0]]),
            (
                "b",
                "rust rust",
                [0.9, 0.1],
                vec![vec![0.5, 0.5], vec![0.0, 0.2]],
            ),
            ("c", "rust language", [0.5, 0.5], vec![vec![0.9, 0.2]]),
            (
                "d",
                "rust rust rust",
                [0.2, 0.8],
                vec![vec![0.3, 0.3], vec![0.6, 0.0]],
            ),
            (
                "e",
                "learning rust today",
                [0.0, 1.0],
                vec![vec![0.02, 0.03]],
            ),
        ];
        for (id, title, vec, tokens) in corpus {
            let doc = laurus::Document::builder()
                .add_text("title", title)
                .add_vector("vec", vec.to_vec())
                .add_vector_array("tokens", tokens)
                .build();
            engine.put_document(id, doc).await.unwrap();
        }
        engine.commit().await.unwrap();
        drop(engine);

        let results = execute(
            &SearchCommand {
                rescore_field: Some("tokens".to_string()),
                rescore_vectors: Some("[[1, 0], [0, 1]]".to_string()),
                ..search_command("title:rust")
            },
            dir.path(),
        )
        .await
        .unwrap();
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["c", "b", "d", "a", "e"]);

        // Text needs a token-level embedder, which this field lacks.
        let err = execute(
            &SearchCommand {
                rescore_field: Some("tokens".to_string()),
                rescore_text: Some("rust".to_string()),
                ..search_command("title:rust")
            },
            dir.path(),
        )
        .await
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("no token-level embedder"),
            "{err:#}"
        );
    }

    #[test]
    fn rescore_flags_must_come_together() {
        use clap::Parser;

        let parse = |args: &[&str]| {
            SearchCommand::try_parse_from(["search", "title:rust"].iter().chain(args))
        };
        assert!(parse(&["--rescore-field", "tokens"]).is_err(), "no query");
        assert!(parse(&["--rescore-text", "q"]).is_err(), "no field");
        assert!(parse(&["--rescore-window", "5"]).is_err(), "no field");
        assert!(
            parse(&[
                "--rescore-field",
                "tokens",
                "--rescore-text",
                "q",
                "--rescore-vectors",
                "[[1]]"
            ])
            .is_err(),
            "two queries"
        );

        let cmd = parse(&[
            "--rescore-field",
            "tokens",
            "--rescore-vectors",
            "[[1, 0]]",
            "--rescore-window",
            "5",
        ])
        .unwrap();
        let rescore = rescore_options(&cmd).unwrap().unwrap();
        assert_eq!(rescore.window_size, 5);

        let cmd = parse(&["--rescore-field", "tokens", "--rescore-vectors", "[1, 0]"]).unwrap();
        assert!(rescore_options(&cmd).is_err(), "not an array of arrays");
    }
}
