//! Implementations for the `create` subcommand.
//!
//! Handles creating new resources:
//!
//! - [`run_index`] - Create a new index from a schema TOML file.
//! - [`run_schema`] - Interactive schema TOML generation wizard.
//!
//! The schema wizard guides the user through an interactive terminal session
//! to define index fields and their options, then writes the resulting schema
//! as a TOML file. Supports all field types provided by the laurus engine,
//! including lexical fields (Text, Integer, Float, etc.) and vector index
//! fields (HNSW, Flat, IVF).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use dialoguer::{Confirm, Input, MultiSelect, Select};

use crate::context;

/// Execute the `create index` command.
///
/// If `schema_path` is `Some`, creates a new index from the given schema TOML
/// file. If `None`, launches the interactive schema wizard to build a schema
/// in-memory and then creates the index directly without writing a separate
/// schema file (the schema is persisted inside the index directory as
/// `schema.toml`).
///
/// With `train_pq_codebook`, every HNSW field configuring
/// `ProductQuantization` + `pq_codebook_path` gets its shared codebook
/// trained from the given JSONL file immediately after creation (Issue
/// #920) — the very first commit can already encode against it, removing
/// the train-before-first-commit ordering hazard the #918 failure policy
/// otherwise leaves to the user. Validation (JSONL exists, at least one
/// eligible field) runs **before** anything is created so a failure never
/// leaves a half-initialized index behind.
///
/// # Arguments
///
/// * `schema_path` - Optional path to a schema TOML file. When `None`, the
///   interactive wizard is used instead.
/// * `train_pq_codebook` - Optional JSONL training file (bulk-ingest
///   shape); trains the shared codebook(s) as part of creation.
/// * `index_dir` - Path to the index directory for the new index.
///
/// # Errors
///
/// Returns an error if:
/// - The schema file cannot be read or parsed (when `schema_path` is given).
/// - The interactive wizard fails (when `schema_path` is `None`).
/// - `train_pq_codebook` is given but the file does not exist, or the
///   effective schema has no `ProductQuantization` + `pq_codebook_path`
///   field.
/// - The index cannot be created, or codebook training fails.
pub async fn run_index(
    schema_path: Option<&Path>,
    train_pq_codebook: Option<&Path>,
    index_dir: &Path,
) -> Result<()> {
    // Determine the schema that will actually be persisted, replicating
    // init_index's recovery rule: an existing schema.toml without store/
    // wins over the argument/wizard schema.
    let schema = if index_dir.join("schema.toml").exists() && !index_dir.join("store").exists() {
        context::read_schema(index_dir)?
    } else {
        match schema_path {
            Some(path) => {
                let content =
                    std::fs::read_to_string(path).context("Failed to read schema file")?;
                toml::from_str(&content).context("Failed to parse schema TOML")?
            }
            None => build_schema_interactive()?,
        }
    };

    // Pre-creation validation for --train-pq-codebook: fail before
    // creating anything so a bad invocation never leaves a created-but-
    // untrained index behind.
    let pq_fields = match train_pq_codebook {
        Some(jsonl) => {
            if !jsonl.exists() {
                anyhow::bail!(
                    "--train-pq-codebook file '{}' does not exist",
                    jsonl.display()
                );
            }
            let fields = eligible_pq_fields(&schema);
            if fields.is_empty() {
                anyhow::bail!(
                    "--train-pq-codebook was given but no HNSW field configures \
                     ProductQuantization + pq_codebook_path; nothing to train \
                     (set pq_codebook_path on the field, or drop the flag)"
                );
            }
            fields
        }
        None => Vec::new(),
    };

    context::create_index_from_schema(index_dir, schema).await?;
    println!("Index created at {}.", index_dir.display());

    if let Some(jsonl) = train_pq_codebook {
        let engine = context::open_index(index_dir).await?;
        for field in &pq_fields {
            let vectors = crate::commands::train::collect_vectors_from_jsonl(field, jsonl, None)?;
            if vectors.is_empty() {
                anyhow::bail!(
                    "no training vectors found in '{}' for field '{field}'",
                    jsonl.display()
                );
            }
            println!(
                "Training PQ codebook for field '{field}' on {} vectors...",
                vectors.len()
            );
            // output = None writes to the field's configured
            // pq_codebook_path, so the schema persisted above and the
            // trained file agree by construction.
            let info = engine.train_pq_codebook(field, &vectors, None)?;
            println!(
                "Trained codebook '{}' (m = {}, k = {}, sub_dim = {}, dimension = {}) \
                 from {} vectors.",
                info.path,
                info.subvector_count,
                info.centroids,
                info.sub_dimension,
                info.dimension,
                info.training_vectors
            );
        }
    }
    Ok(())
}

/// Collect the fields eligible for create-time codebook training: HNSW
/// fields configuring `ProductQuantization` (or, with the `pq-fastscan`
/// feature, `ProductQuantizationFastScan` — Issue #920) with a
/// `pq_codebook_path`, sorted by field name. `Schema::fields` is a
/// `BTreeMap` (Issue #1060) so iteration is already in that order; the
/// explicit sort is kept to make the ordering guarantee obvious here
/// rather than relying on that implicit fact.
fn eligible_pq_fields(schema: &Schema) -> Vec<String> {
    use laurus::vector::core::quantization::QuantizationMethod;
    fn is_pq_variant(quantizer: &QuantizationMethod) -> bool {
        match quantizer {
            QuantizationMethod::ProductQuantization { .. } => true,
            #[cfg(feature = "pq-fastscan")]
            QuantizationMethod::ProductQuantizationFastScan { .. } => true,
            _ => false,
        }
    }
    let mut fields: Vec<String> = schema
        .fields
        .iter()
        .filter_map(|(name, option)| match option {
            FieldOption::Hnsw(o) if is_pq_variant(&o.quantizer) && o.pq_codebook_path.is_some() => {
                Some(name.clone())
            }
            _ => None,
        })
        .collect();
    fields.sort();
    fields
}
use laurus::lexical::core::field::{
    BooleanOption, BytesOption, DateTimeOption, FloatOption, Geo3dOption, GeoOption, IntegerOption,
    TextOption,
};
use laurus::vector::DistanceMetric;
use laurus::vector::core::field::{FlatOption, HnswOption, IvfOption};
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::{AnalyzerSpec, BuiltinAnalyzerSpec, FieldOption, Schema};

/// Field type names shown in the interactive prompt.
const FIELD_TYPES: &[&str] = &[
    "Text", "Integer", "Float", "Boolean", "DateTime", "Geo", "Geo3d", "Bytes", "Hnsw", "Flat",
    "Ivf",
];

/// Distance metric names shown in the interactive prompt.
const DISTANCE_METRICS: &[&str] = &["Cosine", "Euclidean", "Manhattan", "DotProduct", "Angular"];

/// Run the interactive schema generation wizard (`create schema`).
///
/// Prompts the user to define fields one by one, asks for default search
/// fields among the lexical fields, previews the resulting TOML, and writes
/// it to `output` upon confirmation.
///
/// # Arguments
///
/// * `output` - Destination file path for the generated schema TOML.
///
/// # Returns
///
/// Returns `Ok(())` on success, or if the user cancels before writing.
///
/// # Errors
///
/// Returns an error if:
/// - An interactive prompt fails (e.g. terminal I/O error).
/// - The schema cannot be serialised to TOML.
/// - The output file cannot be written.
pub fn run_schema(output: &Path) -> Result<()> {
    let schema = build_schema_interactive()?;

    // Show preview.
    let toml_str = toml::to_string_pretty(&schema).context("Failed to serialize schema to TOML")?;
    println!("\n--- Preview ---");
    println!("{toml_str}");
    println!("---------------\n");

    if !Confirm::new()
        .with_prompt(format!("Write to {}?", output.display()))
        .default(true)
        .interact()?
    {
        println!("Cancelled.");
        return Ok(());
    }

    std::fs::write(output, &toml_str).context("Failed to write schema file")?;
    println!("Schema written to {}.", output.display());

    Ok(())
}

/// Run the interactive schema wizard and return the resulting [`Schema`].
///
/// Prompts the user to define fields one by one, then asks for default
/// search fields among the lexical fields. Returns the built schema without
/// writing it to disk.
///
/// # Errors
///
/// Returns an error if an interactive prompt fails (e.g. terminal I/O error).
pub fn build_schema_interactive() -> Result<Schema> {
    println!("\n=== Laurus Schema Generator ===\n");

    let mut fields: BTreeMap<String, FieldOption> = BTreeMap::new();
    let mut field_order: Vec<String> = Vec::new();

    loop {
        let name = prompt_field_name(&fields)?;
        let field_option = prompt_field_type_and_options()?;

        println!(
            "\nField \"{}\" ({}) added.\n",
            name,
            field_type_label(&field_option)
        );

        field_order.push(name.clone());
        fields.insert(name, field_option);

        if !Confirm::new()
            .with_prompt("Add another field?")
            .default(true)
            .interact()?
        {
            break;
        }
        println!();
    }

    // Collect lexical field names for default field selection.
    let lexical_fields: Vec<&str> = field_order
        .iter()
        .filter(|name| fields.get(*name).map(is_lexical_field).unwrap_or(false))
        .map(|s| s.as_str())
        .collect();

    let default_fields = if lexical_fields.is_empty() {
        Vec::new()
    } else {
        prompt_default_fields(&lexical_fields)?
    };

    Ok(Schema {
        analyzers: BTreeMap::new(),
        embedders: BTreeMap::new(),
        fields,
        default_fields,
        dynamic_field_policy: Default::default(),
        pending_reindex: Default::default(),
    })
}

/// Prompt for a unique field name.
fn prompt_field_name(existing: &BTreeMap<String, FieldOption>) -> Result<String> {
    loop {
        let name: String = Input::new().with_prompt("Field name").interact_text()?;

        if name.is_empty() {
            println!("Field name cannot be empty.");
            continue;
        }

        if existing.contains_key(&name) {
            println!(
                "Field \"{}\" already exists. Please choose a different name.",
                name
            );
            continue;
        }

        return Ok(name);
    }
}

/// Prompt for field type selection and then type-specific options.
fn prompt_field_type_and_options() -> Result<FieldOption> {
    let type_index = Select::new()
        .with_prompt("Field type")
        .items(FIELD_TYPES)
        .default(0)
        .interact()?;

    match FIELD_TYPES[type_index] {
        "Text" => prompt_text_option(),
        "Integer" => prompt_indexed_stored_option("Integer"),
        "Float" => prompt_indexed_stored_option("Float"),
        "Boolean" => prompt_indexed_stored_option("Boolean"),
        "DateTime" => prompt_indexed_stored_option("DateTime"),
        "Geo" => prompt_indexed_stored_option("Geo"),
        "Geo3d" => prompt_indexed_stored_option("Geo3d"),
        "Bytes" => prompt_bytes_option(),
        "Hnsw" => prompt_hnsw_option(),
        "Flat" => prompt_flat_option(),
        "Ivf" => prompt_ivf_option(),
        _ => unreachable!(),
    }
}

/// Prompt for TextOption (indexed, stored, term_vectors, analyzer).
fn prompt_text_option() -> Result<FieldOption> {
    let indexed = Confirm::new()
        .with_prompt("Indexed?")
        .default(true)
        .interact()?;
    let stored = Confirm::new()
        .with_prompt("Stored?")
        .default(true)
        .interact()?;
    let term_vectors = Confirm::new()
        .with_prompt("Term vectors?")
        .default(true)
        .interact()?;

    let analyzer_choices = [
        "standard", "keyword", "english", "japanese", "simple", "noop",
    ];
    let analyzer_idx = dialoguer::Select::new()
        .with_prompt("Analyzer")
        .items(analyzer_choices)
        .default(0)
        .interact()?;
    let analyzer = if analyzer_choices[analyzer_idx] == "japanese" {
        // The Japanese preset requires a Lindera dictionary path.
        let dict: String = dialoguer::Input::new()
            .with_prompt("Lindera dictionary path (e.g. /var/lib/lindera/ipadic)")
            .interact_text()?;
        // Lindera 3.x's `Mode::from_str` only accepts "normal"/"decompose"
        // (there is no "search" mode); offering it here would let the
        // wizard write a schema.toml that fails at index-open time.
        let mode_choices = ["normal", "decompose"];
        let mode_idx = dialoguer::Select::new()
            .with_prompt("Lindera segmentation mode")
            .items(mode_choices)
            .default(0)
            .interact()?;
        let user_dict: String = dialoguer::Input::new()
            .with_prompt("User dictionary path (leave empty for none)")
            .allow_empty(true)
            .interact_text()?;
        Some(AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
            mode: mode_choices[mode_idx].to_string(),
            dict,
            user_dict: if user_dict.is_empty() {
                None
            } else {
                Some(user_dict)
            },
        }))
    } else {
        Some(AnalyzerSpec::Named(
            analyzer_choices[analyzer_idx].to_string(),
        ))
    };

    Ok(FieldOption::Text(TextOption {
        indexed,
        stored,
        term_vectors,
        doc_values: true,
        analyzer,
    }))
}

/// Prompt for field types that have indexed + stored options
/// (Integer, Float, Boolean, DateTime, Geo, Geo3d).
fn prompt_indexed_stored_option(type_name: &str) -> Result<FieldOption> {
    let indexed = Confirm::new()
        .with_prompt("Indexed?")
        .default(true)
        .interact()?;
    let stored = Confirm::new()
        .with_prompt("Stored?")
        .default(true)
        .interact()?;

    let multi_valued = if matches!(type_name, "Integer" | "Float") {
        Confirm::new()
            .with_prompt("Multi-valued? (accepts arrays of values)")
            .default(false)
            .interact()?
    } else {
        false
    };

    Ok(match type_name {
        "Integer" => FieldOption::Integer(IntegerOption {
            indexed,
            stored,
            multi_valued,
            doc_values: true,
        }),
        "Float" => FieldOption::Float(FloatOption {
            indexed,
            stored,
            multi_valued,
            doc_values: true,
        }),
        "Boolean" => FieldOption::Boolean(BooleanOption {
            indexed,
            stored,
            doc_values: true,
        }),
        "DateTime" => FieldOption::DateTime(DateTimeOption {
            indexed,
            stored,
            doc_values: true,
        }),
        "Geo" => FieldOption::Geo(GeoOption {
            indexed,
            stored,
            doc_values: true,
        }),
        "Geo3d" => FieldOption::Geo3d(Geo3dOption {
            indexed,
            stored,
            doc_values: true,
        }),
        _ => unreachable!(),
    })
}

/// Prompt for BytesOption (stored only).
fn prompt_bytes_option() -> Result<FieldOption> {
    let stored = Confirm::new()
        .with_prompt("Stored?")
        .default(true)
        .interact()?;
    Ok(FieldOption::Bytes(BytesOption { stored }))
}

/// Prompt for a distance metric selection.
fn prompt_distance_metric() -> Result<DistanceMetric> {
    let idx = Select::new()
        .with_prompt("Distance metric")
        .items(DISTANCE_METRICS)
        .default(0)
        .interact()?;

    Ok(match DISTANCE_METRICS[idx] {
        "Cosine" => DistanceMetric::Cosine,
        "Euclidean" => DistanceMetric::Euclidean,
        "Manhattan" => DistanceMetric::Manhattan,
        "DotProduct" => DistanceMetric::DotProduct,
        "Angular" => DistanceMetric::Angular,
        _ => unreachable!(),
    })
}

/// Prompt for a positive usize value with a default.
fn prompt_usize(prompt: &str, default: usize) -> Result<usize> {
    let val: usize = Input::new()
        .with_prompt(prompt)
        .default(default)
        .interact_text()?;
    Ok(val)
}

/// Prompt for a vector field's `base_weight` (Issue #1084): this field's
/// relative scoring priority when searched alongside other vector fields.
/// Defaults to `1.0`. Only matters when a query targets two or more
/// specific vector fields at once; has no effect on the lexical-vs-vector
/// balance of a hybrid search.
fn prompt_base_weight() -> Result<f32> {
    let val: f64 = Input::new()
        .with_prompt("Base weight (relative priority vs. other vector fields in this schema)")
        .default(1.0)
        .interact_text()?;
    Ok(val as f32)
}

/// Prompt for HnswOption.
fn prompt_hnsw_option() -> Result<FieldOption> {
    let dimension = prompt_usize("Dimension", 128)?;
    let distance = prompt_distance_metric()?;
    let m = prompt_usize("M (max connections per node)", 16)?;
    let ef_construction = prompt_usize("ef_construction", 200)?;
    let quantizer = prompt_quantization_method(dimension)?;
    let pq_codebook_path = prompt_pq_codebook_path(&quantizer)?;
    let rerank_storage = prompt_rerank_storage()?;
    let base_weight = prompt_base_weight()?;

    Ok(FieldOption::Hnsw(HnswOption {
        dimension,
        distance,
        m,
        ef_construction,
        default_ef_search: None,
        base_weight,
        quantizer,
        rerank_storage,
        embedder: None,
        pq_codebook_path,
    }))
}

/// Prompt for the quantization method.
///
/// Default = Scalar8Bit (Stage 1, 4x compression, recall ~0.95).
/// Optional Product Quantization (Stage 3, Issue #481) requires the
/// user to pick an `M` that divides the vector dimension; the
/// codebook uses K = 256 centroids per sub-vector and is either
/// trained per segment (the default) or trained once and shared
/// across segments via `pq_codebook_path` / `laurus train
/// pq-codebook` (Issue #631). PQ delivers 8-19x compression at the
/// cost of recall — usually paired with rerank storage to compensate.
fn prompt_quantization_method(
    dimension: usize,
) -> Result<laurus::vector::core::quantization::QuantizationMethod> {
    use laurus::vector::core::quantization::QuantizationMethod;
    let kinds = [
        "Scalar8Bit (Stage 1, 4x)",
        "ProductQuantization (Stage 3, 8-32x)",
    ];
    let idx = Select::new()
        .with_prompt("Quantization method")
        .items(kinds.as_slice())
        .default(0)
        .interact()?;
    match idx {
        0 => Ok(QuantizationMethod::Scalar8Bit),
        _ => {
            // Default M = max divisor of dim in {32, 16, 8} so the
            // codebook is well-formed and the codes-to-dim ratio is
            // sensible (sub_dim ≥ 2 to keep ADC meaningful).
            let default_m = [32usize, 16, 8]
                .iter()
                .copied()
                .find(|m| dimension.is_multiple_of(*m) && dimension / m >= 2)
                .unwrap_or(8);
            let subvector_count = prompt_usize("M (subvector count, must divide dim)", default_m)?;
            if !dimension.is_multiple_of(subvector_count) {
                return Err(anyhow::anyhow!(
                    "subvector_count {subvector_count} does not divide dimension {dimension}"
                ));
            }
            Ok(QuantizationMethod::ProductQuantization { subvector_count })
        }
    }
}

/// Prompt for an optional shared PQ codebook file name (Issue #631).
///
/// Only asked when the quantizer is Product Quantization; empty input
/// (the default) keeps per-segment training. When set, commits refuse
/// to encode until `laurus train pq-codebook` has trained the named
/// codebook — there is no silent fallback to per-segment training.
fn prompt_pq_codebook_path(
    quantizer: &laurus::vector::core::quantization::QuantizationMethod,
) -> Result<Option<String>> {
    use laurus::vector::core::quantization::QuantizationMethod;
    if !matches!(quantizer, QuantizationMethod::ProductQuantization { .. }) {
        return Ok(None);
    }
    let path: String = Input::new()
        .with_prompt(
            "Shared PQ codebook file (train once via `laurus train pq-codebook`; \
             empty = train per segment)",
        )
        .allow_empty(true)
        .default(String::new())
        .interact_text()?;
    Ok(if path.is_empty() { None } else { Some(path) })
}

/// Prompt the user to optionally enable Stage 2 rerank storage
/// (Issue #481). Defaults to disabled (Stage 1 int8-only behavior).
///
/// When enabled, the writer emits a per-field f32 sidecar so the
/// HNSW searcher can run two-stage rerank: a wide int8 candidate
/// fetch followed by an exact f32 rescoring of the top
/// `top_k * rerank_factor` candidates.
fn prompt_rerank_storage() -> Result<Option<RerankStorageKind>> {
    let enable = Confirm::new()
        .with_prompt("Enable Stage 2 rerank storage (extra f32 sidecar, +4 bytes/dim)?")
        .default(false)
        .interact()?;
    Ok(if enable {
        Some(RerankStorageKind::F32)
    } else {
        None
    })
}

/// Prompt for FlatOption.
fn prompt_flat_option() -> Result<FieldOption> {
    let dimension = prompt_usize("Dimension", 128)?;
    let distance = prompt_distance_metric()?;
    let base_weight = prompt_base_weight()?;

    Ok(FieldOption::Flat(FlatOption {
        dimension,
        distance,
        base_weight,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    }))
}

/// Prompt for IvfOption.
fn prompt_ivf_option() -> Result<FieldOption> {
    let dimension = prompt_usize("Dimension", 128)?;
    let distance = prompt_distance_metric()?;
    let n_clusters = prompt_usize("Number of clusters", 100)?;
    let n_probe = prompt_usize("Number of probes", 1)?;
    let base_weight = prompt_base_weight()?;

    Ok(FieldOption::Ivf(IvfOption {
        dimension,
        distance,
        n_clusters,
        n_probe,
        base_weight,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
    }))
}

/// Prompt for default search fields from lexical fields.
fn prompt_default_fields(lexical_fields: &[&str]) -> Result<Vec<String>> {
    if lexical_fields.is_empty() {
        return Ok(Vec::new());
    }

    let selections = MultiSelect::new()
        .with_prompt("Select default search fields")
        .items(lexical_fields)
        .interact()?;

    Ok(selections
        .into_iter()
        .map(|i| lexical_fields[i].to_string())
        .collect())
}

/// Check if a field option is a lexical (non-vector) field type.
fn is_lexical_field(option: &FieldOption) -> bool {
    matches!(
        option,
        FieldOption::Text(_)
            | FieldOption::Integer(_)
            | FieldOption::Float(_)
            | FieldOption::Boolean(_)
            | FieldOption::DateTime(_)
            | FieldOption::Geo(_)
            | FieldOption::Geo3d(_)
            | FieldOption::Bytes(_)
    )
}

/// Return a human-readable label for a field option variant.
fn field_type_label(option: &FieldOption) -> &'static str {
    match option {
        FieldOption::Text(_) => "Text",
        FieldOption::Integer(_) => "Integer",
        FieldOption::Float(_) => "Float",
        FieldOption::Boolean(_) => "Boolean",
        FieldOption::DateTime(_) => "DateTime",
        FieldOption::Geo(_) => "Geo",
        FieldOption::Geo3d(_) => "Geo3d",
        FieldOption::Bytes(_) => "Bytes",
        FieldOption::Hnsw(_) => "Hnsw",
        FieldOption::Flat(_) => "Flat",
        FieldOption::Ivf(_) => "Ivf",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interactive wizard must list `Geo3d` so users can declare a 3D
    /// ECEF field without dropping down to TOML hand-editing. Tracked
    /// originally in #337.
    #[test]
    fn field_types_includes_geo3d() {
        assert!(
            FIELD_TYPES.contains(&"Geo3d"),
            "FIELD_TYPES must include \"Geo3d\""
        );
    }

    /// Sanity-check the existing display path: a `FieldOption::Geo3d` value
    /// must render as `"Geo3d"` and be classified as a lexical field.
    #[test]
    fn field_type_label_handles_geo3d() {
        let opt = FieldOption::Geo3d(Geo3dOption {
            indexed: true,
            stored: true,
            doc_values: true,
        });
        assert_eq!(field_type_label(&opt), "Geo3d");
        assert!(is_lexical_field(&opt));
    }

    /// Stage 2 (Issue #481): the schema TOML accepts `rerank_storage =
    /// "F32"` on an HNSW field and the field round-trips through the
    /// same toml::from_str path the cli uses to load schema.toml.
    /// Omitting the key keeps the Stage 1 default (None).
    #[test]
    fn schema_toml_round_trips_rerank_storage_on_hnsw_field() {
        let toml_with = r#"
[fields.embedding.Hnsw]
dimension = 4
distance = "Cosine"
m = 4
ef_construction = 16
rerank_storage = "F32"
"#;
        let schema: Schema =
            toml::from_str(toml_with).expect("schema TOML with rerank_storage must parse");
        let opt = schema
            .fields
            .get("embedding")
            .expect("embedding field must be present");
        match opt {
            FieldOption::Hnsw(h) => {
                assert_eq!(h.rerank_storage, Some(RerankStorageKind::F32));
            }
            other => panic!("expected Hnsw field, got {:?}", field_type_label(other)),
        }

        let toml_without = r#"
[fields.embedding.Hnsw]
dimension = 4
distance = "Cosine"
m = 4
ef_construction = 16
"#;
        let schema: Schema = toml::from_str(toml_without)
            .expect("schema TOML without rerank_storage must still parse");
        let opt = schema.fields.get("embedding").unwrap();
        match opt {
            FieldOption::Hnsw(h) => assert!(h.rerank_storage.is_none()),
            other => panic!("expected Hnsw field, got {:?}", field_type_label(other)),
        }
    }

    // --- create index --train-pq-codebook (Issue #920) ---

    use laurus::{DataValue, Document};

    const DIM: usize = 32;

    /// Write a schema TOML declaring `fields` as HNSW + PQ +
    /// pq_codebook_path, and a JSONL training file carrying `count`
    /// deterministic vectors for every one of those fields per line.
    fn setup_pq_schema_and_jsonl(
        dir: &Path,
        fields: &[&str],
        count: usize,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut schema = String::new();
        for field in fields {
            schema.push_str(&format!(
                "[fields.{field}.Hnsw]\ndimension = 32\ndistance = \"Euclidean\"\n\
                 m = 8\nef_construction = 32\npq_codebook_path = \"{field}.pqcb\"\n\n\
                 [fields.{field}.Hnsw.quantizer.ProductQuantization]\nsubvector_count = 4\n\n"
            ));
        }
        std::fs::create_dir_all(dir).unwrap();
        let schema_path = dir.join("input-schema.toml");
        std::fs::write(&schema_path, schema).unwrap();

        let mut state = 0x2468_ACE0_u64;
        let mut jsonl = String::new();
        for i in 0..count {
            let mut cells = Vec::new();
            for field in fields {
                let data: Vec<String> = (0..DIM)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        format!(
                            "{:.4}",
                            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                        )
                    })
                    .collect();
                cells.push(format!("\"{field}\": [{}]", data.join(", ")));
            }
            jsonl.push_str(&format!(
                "{{\"id\": \"doc{i}\", \"fields\": {{{}}}}}\n",
                cells.join(", ")
            ));
        }
        let jsonl_path = dir.join("train.jsonl");
        std::fs::write(&jsonl_path, jsonl).unwrap();
        (schema_path, jsonl_path)
    }

    /// Put one small document per field and commit on a freshly opened
    /// engine — succeeds only when every PQ field's codebook is trained.
    async fn ingest_and_commit(index_dir: &Path, fields: &[&str]) -> anyhow::Result<()> {
        let engine = context::open_index(index_dir).await?;
        let mut builder = Document::builder();
        for field in fields {
            builder = builder.add_field(
                *field,
                DataValue::Vector((0..DIM).map(|j| j as f32 * 0.01).collect()),
            );
        }
        engine.put_document("probe", builder.build()).await?;
        engine.commit().await?;
        Ok(())
    }

    /// Issue #920: the flag trains the codebook as part of creation, so
    /// the very first commit encodes against it — the direct regression
    /// for the train-before-first-commit hazard.
    #[tokio::test]
    async fn create_with_train_flag_makes_first_commit_succeed() {
        let dir = tempfile::tempdir().unwrap();
        let (schema, jsonl) = setup_pq_schema_and_jsonl(dir.path(), &["embedding"], 300);

        run_index(Some(&schema), Some(&jsonl), dir.path())
            .await
            .unwrap();

        assert!(dir.path().join("store/vector/embedding.pqcb").exists());
        ingest_and_commit(dir.path(), &["embedding"])
            .await
            .expect("first commit must encode against the create-time codebook");
    }

    /// Control pinning the hazard itself: without the flag, the same
    /// schema's first commit hard-errors per the #918 failure policy.
    #[tokio::test]
    async fn create_without_train_flag_leaves_first_commit_failing() {
        let dir = tempfile::tempdir().unwrap();
        let (schema, _jsonl) = setup_pq_schema_and_jsonl(dir.path(), &["embedding"], 1);

        run_index(Some(&schema), None, dir.path()).await.unwrap();

        assert!(!dir.path().join("store/vector/embedding.pqcb").exists());
        let err = ingest_and_commit(dir.path(), &["embedding"])
            .await
            .expect_err("commit without a trained codebook must fail");
        assert!(
            err.to_string().contains("pq_codebook_path"),
            "the failure must be the untrained-codebook hard-error: {err}"
        );
    }

    /// The flag with no eligible PQ field bails before creating anything.
    #[tokio::test]
    async fn create_with_train_flag_rejects_schema_without_pq_field() {
        let dir = tempfile::tempdir().unwrap();
        let schema_path = dir.path().join("input-schema.toml");
        // HNSW but Scalar8Bit (default quantizer), no pq_codebook_path.
        std::fs::write(
            &schema_path,
            "[fields.embedding.Hnsw]\ndimension = 32\ndistance = \"Euclidean\"\n\
             m = 8\nef_construction = 32\n",
        )
        .unwrap();
        let jsonl = dir.path().join("train.jsonl");
        std::fs::write(&jsonl, "").unwrap();

        let index_dir = dir.path().join("idx");
        let err = run_index(Some(&schema_path), Some(&jsonl), &index_dir)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no HNSW field configures"),
            "error must explain why nothing can be trained: {err}"
        );
        assert!(
            !index_dir.join("schema.toml").exists(),
            "a rejected invocation must not leave a half-created index"
        );
    }

    /// Multiple eligible fields all get their codebooks trained.
    #[tokio::test]
    async fn create_with_train_flag_trains_every_pq_field() {
        let dir = tempfile::tempdir().unwrap();
        let (schema, jsonl) = setup_pq_schema_and_jsonl(dir.path(), &["emb_a", "emb_b"], 300);

        run_index(Some(&schema), Some(&jsonl), dir.path())
            .await
            .unwrap();

        assert!(dir.path().join("store/vector/emb_a.pqcb").exists());
        assert!(dir.path().join("store/vector/emb_b.pqcb").exists());
    }
}
