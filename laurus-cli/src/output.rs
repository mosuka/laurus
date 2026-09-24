//! Output formatting utilities for CLI results.
//!
//! This module provides functions to render search results, documents, and
//! index statistics in either a human-readable table or JSON format. The
//! desired format is selected via the [`OutputFormat`] enum.

use std::collections::HashMap;

use base64::Engine as _;
use clap::ValueEnum;
use laurus::{DataValue, Document, EngineStats, SearchResult};
use serde_json::json;
use tabled::settings::location::ByColumnName;
use tabled::settings::{Remove, Style};
use tabled::{Table, Tabled};

/// Output format for CLI results.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable table.
    Table,
    /// JSON output.
    Json,
}

/// Print search results to stdout.
///
/// # Arguments
///
/// * `results` - Slice of [`SearchResult`] entries returned by the engine.
/// * `format` - The desired output format (table or JSON).
pub fn print_search_results(results: &[SearchResult], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            let json_results = search_results_to_json(results);
            println!("{}", serde_json::to_string_pretty(&json_results).unwrap());
        }
        OutputFormat::Table => {
            if results.is_empty() {
                println!("No results found.");
                return;
            }

            let has_highlights = results.iter().any(|r| !r.highlights.is_empty());
            let rows: Vec<SearchResultRow> = results
                .iter()
                .map(|r| {
                    let fields = r
                        .document
                        .as_ref()
                        .map(|doc| format_fields_compact(&doc.fields))
                        .unwrap_or_default();
                    SearchResultRow {
                        id: r.id.clone(),
                        score: format!("{:.4}", r.score),
                        fields,
                        highlights: format_highlights_compact(&r.highlights),
                    }
                })
                .collect();

            let mut table = Table::new(&rows);
            table.with(Style::rounded());
            // Highlighting was not requested for this search (Issue #1134):
            // drop the column instead of rendering it empty on every row.
            if !has_highlights {
                table.with(Remove::column(ByColumnName::new("Highlights")));
            }
            println!("{table}");
        }
    }
}

/// Convert search results to the JSON shape [`print_search_results`] prints:
/// `{"id", "score", "fields"?, "highlights"?}` per result. `"highlights"` is
/// present only when at least one field actually highlighted (Issue #1134),
/// matching the HTTP gateway's and laurus-mcp's `SearchResult` -> JSON
/// conversions.
fn search_results_to_json(results: &[SearchResult]) -> Vec<serde_json::Value> {
    results
        .iter()
        .map(|r| {
            let mut obj = json!({
                "id": r.id,
                "score": r.score,
            });
            if let Some(ref doc) = r.document {
                obj["fields"] = fields_to_json(&doc.fields);
            }
            if !r.highlights.is_empty() {
                obj["highlights"] = json!(r.highlights);
            }
            obj
        })
        .collect()
}

/// Print documents to stdout.
///
/// # Arguments
///
/// * `id` - The external document ID used for display.
/// * `documents` - Slice of [`Document`] entries (may contain multiple chunks).
/// * `format` - The desired output format (table or JSON).
pub fn print_documents(id: &str, documents: &[Document], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            let json_docs: Vec<serde_json::Value> = documents
                .iter()
                .map(|doc| {
                    json!({
                        "id": id,
                        "fields": fields_to_json(&doc.fields),
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json_docs).unwrap());
        }
        OutputFormat::Table => {
            if documents.is_empty() {
                println!("No documents found for id '{id}'.");
                return;
            }

            let rows: Vec<DocumentRow> = documents
                .iter()
                .enumerate()
                .map(|(i, doc)| DocumentRow {
                    id: if i == 0 {
                        id.to_string()
                    } else {
                        format!("{id} (chunk {i})")
                    },
                    fields: format_fields_compact(&doc.fields),
                })
                .collect();

            let table = Table::new(&rows).with(Style::rounded()).to_string();
            println!("{table}");
        }
    }
}

/// Print index statistics to stdout.
///
/// # Arguments
///
/// * `stats` - [`EngineStats`] containing document count and per-field vector
///   statistics.
/// * `format` - The desired output format (table or JSON).
pub fn print_stats(stats: &EngineStats, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            let vector_fields_json: serde_json::Value = stats
                .vector_fields
                .iter()
                .map(|(name, fs)| {
                    (
                        name.clone(),
                        json!({
                            "vector_count": fs.vector_count,
                            "dimension": fs.dimension,
                        }),
                    )
                })
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into();
            let output = json!({
                "document_count": stats.document_count,
                "vector_fields": vector_fields_json,
            });
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        OutputFormat::Table => {
            println!("Document count: {}", stats.document_count);

            if !stats.vector_fields.is_empty() {
                let rows: Vec<FieldStatsRow> = stats
                    .vector_fields
                    .iter()
                    .map(|(name, fs)| FieldStatsRow {
                        field: name.clone(),
                        vector_count: fs.vector_count,
                        dimension: fs.dimension,
                    })
                    .collect();

                let table = Table::new(&rows).with(Style::rounded()).to_string();
                println!("\nVector fields:\n{table}");
            }
        }
    }
}

// --- Helper types and functions ---

#[derive(Tabled)]
struct SearchResultRow {
    #[tabled(rename = "ID")]
    id: String,
    #[tabled(rename = "Score")]
    score: String,
    #[tabled(rename = "Fields")]
    fields: String,
    #[tabled(rename = "Highlights")]
    highlights: String,
}

#[derive(Tabled)]
struct DocumentRow {
    #[tabled(rename = "ID")]
    id: String,
    #[tabled(rename = "Fields")]
    fields: String,
}

#[derive(Tabled)]
struct FieldStatsRow {
    #[tabled(rename = "Field")]
    field: String,
    #[tabled(rename = "Vectors")]
    vector_count: usize,
    #[tabled(rename = "Dimension")]
    dimension: usize,
}

/// Convert field data to a compact display string.
fn format_fields_compact(fields: &HashMap<String, DataValue>) -> String {
    let mut parts: Vec<String> = fields
        .iter()
        .filter(|(k, _)| k.as_str() != "_id")
        .map(|(k, v)| format!("{k}: {}", format_data_value(v)))
        .collect();
    parts.sort();
    parts.join(", ")
}

/// Convert fields to JSON value.
fn fields_to_json(fields: &HashMap<String, DataValue>) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = fields
        .iter()
        .filter(|(k, _)| k.as_str() != "_id")
        .map(|(k, v)| (k.clone(), data_value_to_json(v)))
        .collect();
    serde_json::Value::Object(map)
}

/// Convert highlights (Issue #1134) to a compact display string, fields in
/// sorted order for deterministic output: `field: frag1 | frag2, other: ...`.
/// Each fragment is truncated the same way a text field value is.
fn format_highlights_compact(highlights: &HashMap<String, Vec<String>>) -> String {
    let mut fields: Vec<&String> = highlights.keys().collect();
    fields.sort();
    fields
        .into_iter()
        .map(|field| {
            let fragments = highlights[field]
                .iter()
                .map(|fragment| truncate_preview(fragment, TEXT_PREVIEW_CHARS))
                .collect::<Vec<_>>()
                .join(" | ");
            format!("{field}: {fragments}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Maximum number of characters rendered for a text value in table output.
///
/// Values longer than this are cut to `TEXT_PREVIEW_CHARS - 3` characters
/// and suffixed with `...` so the rendered cell never exceeds
/// `TEXT_PREVIEW_CHARS` characters.
const TEXT_PREVIEW_CHARS: usize = 80;

/// Truncate `s` to at most `max_chars` characters, on character boundaries
/// rather than byte boundaries — the previous inline version of this logic
/// (`&s[..77]`) sliced raw bytes and panicked whenever the cut point landed
/// inside a multi-byte character, i.e. on essentially any Japanese value.
/// A value cut short is suffixed with `...`.
///
/// `s.len()` (bytes) is always >= the character count, so the byte-length
/// guard short-circuits the O(n) `chars().count()` for the common
/// already-short case without changing the result.
fn truncate_preview(s: &str, max_chars: usize) -> String {
    if s.len() > max_chars && s.chars().count() > max_chars {
        let head: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{head}...")
    } else {
        s.to_string()
    }
}

/// Format a DataValue for compact display.
fn format_data_value(value: &DataValue) -> String {
    match value {
        DataValue::Null => "null".to_string(),
        DataValue::Bool(b) => b.to_string(),
        DataValue::Int64(i) => i.to_string(),
        DataValue::Float64(f) => f.to_string(),
        DataValue::Text(s) => truncate_preview(s, TEXT_PREVIEW_CHARS),
        DataValue::Bytes(b, _) => format!("<{} bytes>", b.len()),
        DataValue::Vector(v) => format!("<vector dim={}>", v.len()),
        DataValue::DateTime(dt) => dt.to_rfc3339(),
        DataValue::Geo(p) => format!("({}, {})", p.lat, p.lon),
        DataValue::GeoEcef(p) => format!("({}, {}, {})", p.x, p.y, p.z),
        DataValue::Int64Array(arr) => format!(
            "[{}]",
            arr.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        DataValue::Float64Array(arr) => format!(
            "[{}]",
            arr.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        DataValue::GeoArray(arr) => format!(
            "[{}]",
            arr.iter()
                .map(|p| format!("({}, {})", p.lat, p.lon))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        DataValue::GeoEcefArray(arr) => format!(
            "[{}]",
            arr.iter()
                .map(|p| format!("({}, {}, {})", p.x, p.y, p.z))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        DataValue::DateTimeArray(arr) => format!(
            "[{}]",
            arr.iter()
                .map(|dt| dt.to_rfc3339())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        DataValue::BoolArray(arr) => format!(
            "[{}]",
            arr.iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        DataValue::TextArray(arr) => format!("[{}]", arr.join(", ")),
    }
}

/// Convert DataValue to serde_json::Value.
///
/// The output uses the same field-value shapes that
/// [`laurus::json_to_document`] accepts as input, so `get docs --format
/// json` output round-trips through `put docs` / `add doc --data`. In
/// particular, [`DataValue::Bytes`] is rendered as `{"data": "<base64>",
/// "mime": ...}` — the same object shape `json_to_document` parses back into
/// bytes — rather than a byte count, which would lose the payload entirely.
fn data_value_to_json(value: &DataValue) -> serde_json::Value {
    match value {
        DataValue::Null => serde_json::Value::Null,
        DataValue::Bool(b) => json!(b),
        DataValue::Int64(i) => json!(i),
        DataValue::Float64(f) => json!(f),
        DataValue::Text(s) => json!(s),
        DataValue::Bytes(b, mime) => {
            json!({"data": base64::engine::general_purpose::STANDARD.encode(b), "mime": mime})
        }
        DataValue::Vector(v) => json!(v),
        DataValue::DateTime(dt) => json!(dt.to_rfc3339()),
        DataValue::Geo(p) => json!({"lat": p.lat, "lon": p.lon}),
        DataValue::GeoEcef(p) => json!({"x": p.x, "y": p.y, "z": p.z}),
        DataValue::Int64Array(arr) => json!(arr),
        DataValue::Float64Array(arr) => json!(arr),
        // Same per-point object shape as the single-valued arms, so the
        // output feeds back into `json_to_document`'s geo-array inference.
        DataValue::GeoArray(arr) => json!(
            arr.iter()
                .map(|p| json!({"lat": p.lat, "lon": p.lon}))
                .collect::<Vec<_>>()
        ),
        DataValue::GeoEcefArray(arr) => json!(
            arr.iter()
                .map(|p| json!({"x": p.x, "y": p.y, "z": p.z}))
                .collect::<Vec<_>>()
        ),
        // RFC 3339 strings, the shape `json_to_document` infers back into a
        // multi-valued datetime (#1184).
        DataValue::DateTimeArray(arr) => {
            json!(arr.iter().map(|dt| dt.to_rfc3339()).collect::<Vec<_>>())
        }
        // JSON booleans, the shape `json_to_document` infers back into a
        // multi-valued boolean (#1180).
        DataValue::BoolArray(arr) => json!(arr),
        // JSON strings, the shape `json_to_document` infers back into a
        // multi-valued text field (#1175) — or a datetime array when every
        // element is RFC 3339.
        DataValue::TextArray(arr) => json!(arr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `format_data_value` used to slice `DataValue::Text` at byte offset
    /// 77, which panics with "byte index 77 is not a char boundary" on any
    /// multi-byte text whose 77th byte falls inside a character. Table
    /// output of a Japanese field triggered this on every search.
    #[test]
    fn format_data_value_truncates_japanese_text_without_panicking() {
        // 100 chars x 3 bytes = 300 bytes; byte 77 splits the 26th char.
        let text = "あ".repeat(100);
        let out = format_data_value(&DataValue::Text(text));
        assert!(out.ends_with("..."), "long text must be elided: {out}");
    }

    /// The 80 limit counts characters, not bytes: 79 Japanese characters
    /// (237 bytes) must survive untouched even though the byte length
    /// exceeds the old byte-based threshold.
    #[test]
    fn format_data_value_truncation_counts_characters_not_bytes() {
        let text = "あ".repeat(79);
        let out = format_data_value(&DataValue::Text(text.clone()));
        assert_eq!(out, text, "79 chars is under the limit despite 237 bytes");
    }

    /// ASCII behavior must stay exactly as before: byte length and
    /// character count coincide for ASCII input.
    #[test]
    fn format_data_value_keeps_short_ascii_text_verbatim() {
        let text = "hello world".to_string();
        let out = format_data_value(&DataValue::Text(text.clone()));
        assert_eq!(out, text);
    }

    /// A 4-byte character (e.g. an emoji) landing exactly at the cut point
    /// must not panic, and the result must remain valid UTF-8 (guaranteed
    /// by construction since we build it from `chars()`).
    #[test]
    fn format_data_value_handles_emoji_at_the_cut_point() {
        let mut text = "a".repeat(TEXT_PREVIEW_CHARS - 3);
        text.push('🍎');
        text.push_str(&"b".repeat(20));
        let out = format_data_value(&DataValue::Text(text));
        assert!(out.ends_with("..."), "long text must be elided: {out}");
    }

    fn result_with_highlights(id: &str, highlights: HashMap<String, Vec<String>>) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score: 1.0,
            document: None,
            highlights,
        }
    }

    #[test]
    fn format_highlights_compact_sorts_fields_and_joins_fragments() {
        let mut highlights = HashMap::new();
        highlights.insert("title".to_string(), vec!["<mark>Rust</mark>".to_string()]);
        highlights.insert(
            "body".to_string(),
            vec!["frag one".to_string(), "frag two".to_string()],
        );

        let out = format_highlights_compact(&highlights);
        assert_eq!(
            out, "body: frag one | frag two, title: <mark>Rust</mark>",
            "fields sorted alphabetically (body before title): {out}"
        );
    }

    #[test]
    fn format_highlights_compact_is_empty_for_no_highlights() {
        assert_eq!(format_highlights_compact(&HashMap::new()), "");
    }

    /// `search_results_to_json` (Issue #1134): the JSON output must gain a
    /// `"highlights"` key only for a result that actually has one.
    #[test]
    fn search_results_to_json_includes_highlights_only_when_present() {
        let mut highlights = HashMap::new();
        highlights.insert("body".to_string(), vec!["<mark>Rust</mark>".to_string()]);
        let results = vec![
            result_with_highlights("doc1", highlights),
            result_with_highlights("doc2", HashMap::new()),
        ];

        let json = search_results_to_json(&results);
        assert_eq!(
            json[0]["highlights"]["body"],
            serde_json::json!(["<mark>Rust</mark>"])
        );
        assert!(
            json[1].get("highlights").is_none(),
            "a result with no highlights must not have the key: {}",
            json[1]
        );
    }

    /// Table rendering with and without highlights must not panic, and the
    /// `Highlights` column must actually disappear when nothing in the
    /// result set has any (rather than rendering an all-empty column).
    #[test]
    fn print_search_results_table_drops_the_highlights_column_when_empty() {
        let no_highlights = vec![result_with_highlights("doc1", HashMap::new())];
        print_search_results(&no_highlights, OutputFormat::Table);
        print_search_results(&no_highlights, OutputFormat::Json);

        let mut highlights = HashMap::new();
        highlights.insert("body".to_string(), vec!["<mark>Rust</mark>".to_string()]);
        let with_highlights = vec![result_with_highlights("doc1", highlights)];
        print_search_results(&with_highlights, OutputFormat::Table);
        print_search_results(&with_highlights, OutputFormat::Json);
    }
}
