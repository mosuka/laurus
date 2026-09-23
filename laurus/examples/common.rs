//! Shared helpers for Laurus examples.
//!
//! Provides common setup utilities so that each example can focus on
//! demonstrating its specific feature rather than repeating boilerplate.

#![allow(dead_code)]

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::keyword::KeywordAnalyzer;
use laurus::analysis::analyzer::per_field::PerFieldAnalyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::Vector;
use laurus::{DataValue, EmbedInput, EmbedInputType, Embedder, Result, SearchResult};

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

/// Create an in-memory storage backend.
pub fn memory_storage() -> Result<Arc<dyn Storage>> {
    StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))
}

// ---------------------------------------------------------------------------
// Analyzer helpers
// ---------------------------------------------------------------------------

/// Create a `PerFieldAnalyzer` where `keyword_fields` use `KeywordAnalyzer`
/// (exact match) and all other fields use `StandardAnalyzer` (tokenization +
/// lowercasing).
pub fn per_field_analyzer(keyword_fields: &[&str]) -> Arc<PerFieldAnalyzer> {
    let std_analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::default());
    let kw_analyzer: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
    let pfa = PerFieldAnalyzer::new(Arc::clone(&std_analyzer));
    for &field in keyword_fields {
        pfa.add_analyzer(field, Arc::clone(&kw_analyzer));
    }
    Arc::new(pfa)
}

// ---------------------------------------------------------------------------
// MockEmbedder (for vector / hybrid search examples)
// ---------------------------------------------------------------------------

/// A mock embedder that converts text into a 4-dimensional vector based on
/// keyword matching. In production, use a real embedding model.
#[derive(Debug, Clone)]
pub struct MockEmbedder;

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(t) => {
                let t = t.to_lowercase();
                // Semantic dimensions: [systems, language, memory, concurrency]
                let vec = if t.contains("ownership")
                    || t.contains("borrow")
                    || t.contains("lifetime")
                {
                    [0.2, 0.3, 0.9, 0.1] // Memory safety
                } else if t.contains("async") || t.contains("thread") || t.contains("concurrent") {
                    [0.3, 0.1, 0.2, 0.9] // Concurrency
                } else if t.contains("type") || t.contains("generic") || t.contains("trait") {
                    [0.4, 0.8, 0.2, 0.1] // Type system
                } else if t.contains("cargo") || t.contains("crate") || t.contains("build") {
                    [0.8, 0.3, 0.1, 0.2] // Build system
                } else if t.contains("memory") || t.contains("safe") {
                    [0.3, 0.2, 0.8, 0.2] // Memory
                } else {
                    [0.0, 0.0, 0.0, 0.0]
                };
                Ok(Vector::new(vec.to_vec()))
            }
            _ => Ok(Vector::new(vec![0.0; 4])),
        }
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn supports_text(&self) -> bool {
        true
    }

    fn supports_image(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "MockEmbedder"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------

/// Print unified (Engine) search results in a human-readable format.
///
/// Displays all document fields (except `_id` and vector fields).
pub fn print_search_results(results: &[SearchResult]) {
    if results.is_empty() {
        println!("  (No results found)");
        return;
    }
    for (i, hit) in results.iter().enumerate() {
        println!("  {}. (score: {:.4})", i + 1, hit.score);
        if let Some(doc) = &hit.document {
            let mut fields: Vec<_> = doc.fields.iter().collect();
            fields.sort_by_key(|(k, _)| (*k).clone());
            for (name, value) in fields {
                if name == "_id" || matches!(value, DataValue::Vector(_)) {
                    continue;
                }
                let formatted = format_data_value(value);
                // Truncate by character count, not byte count: slicing
                // raw bytes panicked on any multi-byte (e.g. Japanese)
                // field value whose 80th byte fell inside a character.
                let display = if formatted.chars().count() > 80 {
                    let head: String = formatted.chars().take(80).collect();
                    format!("{head}...")
                } else {
                    formatted
                };
                println!("     {}: {}", name, display);
            }
        }
    }
}

/// Format a `DataValue` for display.
fn format_data_value(value: &DataValue) -> String {
    match value {
        DataValue::Text(s) => s.clone(),
        DataValue::Int64(n) => n.to_string(),
        DataValue::Float64(f) => format!("{f:.2}"),
        DataValue::Bool(b) => b.to_string(),
        DataValue::DateTime(dt) => dt.to_rfc3339(),
        DataValue::Geo(p) => format!("({:.4}, {:.4})", p.lat, p.lon),
        DataValue::GeoEcef(p) => format!("({:.2}, {:.2}, {:.2})", p.x, p.y, p.z),
        DataValue::Vector(v) => format!("{v:?}"),
        DataValue::Bytes(b, _) => format!("[{} bytes]", b.len()),
        DataValue::Null => "null".to_string(),
        DataValue::Int64Array(arr) => format!("{arr:?}"),
        DataValue::Float64Array(arr) => format!("{arr:?}"),
        DataValue::GeoArray(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .map(|p| format!("({:.4}, {:.4})", p.lat, p.lon))
                .collect();
            format!("[{}]", parts.join(", "))
        }
        DataValue::GeoEcefArray(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .map(|p| format!("({:.2}, {:.2}, {:.2})", p.x, p.y, p.z))
                .collect();
            format!("[{}]", parts.join(", "))
        }
        DataValue::DateTimeArray(arr) => {
            let parts: Vec<String> = arr.iter().map(|dt| dt.to_rfc3339()).collect();
            format!("[{}]", parts.join(", "))
        }
    }
}
