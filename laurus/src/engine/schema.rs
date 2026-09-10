pub mod analyzer;
pub mod embedder;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use self::analyzer::AnalyzerDefinition;
use self::embedder::EmbedderDefinition;

use crate::lexical::core::field::{
    BooleanOption, BytesOption, DateTimeOption, FloatOption, Geo3dOption, GeoOption, IntegerOption,
    TextOption,
};
use crate::vector::core::field::{FlatOption, HnswOption, IvfOption};

/// Policy for fields that are not declared in the schema.
///
/// Applied when a document is ingested with field names that do not appear in
/// [`Schema::fields`]. The default is [`DynamicFieldPolicy::Dynamic`], which
/// mirrors the "schema-less onboarding" design goal: users can start indexing
/// immediately without defining a schema upfront.
///
/// # Variants
///
/// - [`Strict`](Self::Strict): Unknown fields cause the ingest to fail. Use
///   when you want to enforce an exact schema contract.
/// - [`Dynamic`](Self::Dynamic) (default): Unknown fields are accepted; their
///   type is inferred from the value and a new field definition is added to
///   the schema automatically.
/// - [`Ignore`](Self::Ignore): Unknown fields are silently dropped. Use when
///   you want to ingest partially-structured data without rejecting it but
///   also without expanding the schema.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynamicFieldPolicy {
    /// Fail the ingest when any field is not declared in the schema.
    Strict,
    /// Infer the type of unknown fields and add them to the schema.
    #[default]
    Dynamic,
    /// Silently drop unknown fields.
    Ignore,
}

impl std::str::FromStr for DynamicFieldPolicy {
    type Err = crate::error::LaurusError;

    /// Parse a policy name (case-insensitive).
    ///
    /// Accepted values: `"strict"`, `"dynamic"`, `"ignore"`. This is the
    /// canonical policy parser used by all language bindings so the accepted
    /// spelling is identical across Python, Node.js, WASM, Ruby, and PHP.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LaurusError::invalid_argument`] for any
    /// unrecognised value.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "strict" => Ok(DynamicFieldPolicy::Strict),
            "dynamic" => Ok(DynamicFieldPolicy::Dynamic),
            "ignore" => Ok(DynamicFieldPolicy::Ignore),
            other => Err(crate::error::LaurusError::invalid_argument(format!(
                "unknown dynamic field policy '{other}' \
                 (expected 'strict', 'dynamic', or 'ignore')"
            ))),
        }
    }
}

/// Name of the automatically-injected external document ID field.
///
/// This is the sole field name with a `_` prefix that the engine accepts from
/// user code; all other `_`-prefixed names are rejected by
/// [`validate_field_name`].
pub const RESERVED_ID_FIELD: &str = "_id";

/// Returns `true` if `name` is a reserved field name that user code is
/// allowed to reference explicitly (currently only [`RESERVED_ID_FIELD`]).
///
/// # Arguments
///
/// * `name` - The field name to check.
pub fn is_allowed_reserved_field(name: &str) -> bool {
    name == RESERVED_ID_FIELD
}

/// Validates that a user-supplied field name does not collide with the
/// engine's reserved namespace.
///
/// Field names whose first character is `_` are reserved for the engine
/// (e.g. [`RESERVED_ID_FIELD`]) and cannot be declared by users. The only
/// exception is the allow-listed names returned by
/// [`is_allowed_reserved_field`].
///
/// # Arguments
///
/// * `name` - The field name to validate.
///
/// # Errors
///
/// Returns [`crate::error::LaurusError::invalid_argument`] if the name starts
/// with `_` and is not in the allow-list.
pub fn validate_field_name(name: &str) -> crate::error::Result<()> {
    if name.starts_with('_') && !is_allowed_reserved_field(name) {
        return Err(crate::error::LaurusError::invalid_argument(format!(
            "Field name '{name}' is reserved: names starting with '_' are \
             reserved for system fields (allowed: '{RESERVED_ID_FIELD}')"
        )));
    }
    Ok(())
}

/// Schema for the unified engine.
///
/// Declares what fields exist, their index types (lexical or vector),
/// and optional custom analyzer definitions. Custom analyzers are
/// referenced by name from [`TextOption::analyzer`].
///
/// The schema also carries a [`DynamicFieldPolicy`] that controls how
/// undeclared fields are handled during document ingestion. The default is
/// [`DynamicFieldPolicy::Dynamic`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Custom analyzer definitions, keyed by name.
    /// These can be referenced from text field `analyzer` settings.
    ///
    /// A `BTreeMap` (not `HashMap`), so serialization (`to_toml`/`to_json`)
    /// always emits keys in the same sorted order (Issue #1060) instead of
    /// varying with `HashMap`'s per-process hash seed.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub analyzers: BTreeMap<String, AnalyzerDefinition>,
    /// Embedder definitions, keyed by name.
    /// These can be referenced from vector field `embedder` settings.
    ///
    /// A `BTreeMap` for the same determinism reason as [`Self::analyzers`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub embedders: BTreeMap<String, EmbedderDefinition>,
    /// Options for each field.
    ///
    /// A `BTreeMap` for the same determinism reason as [`Self::analyzers`].
    pub fields: BTreeMap<String, FieldOption>,
    /// Default fields for search.
    #[serde(default)]
    pub default_fields: Vec<String>,
    /// Policy for fields not declared in [`fields`](Self::fields).
    /// Defaults to [`DynamicFieldPolicy::Dynamic`].
    #[serde(default)]
    pub dynamic_field_policy: DynamicFieldPolicy,
    /// Fields whose `FieldOption` was changed via [`Engine::update_field`]
    /// with a [`FieldChangeKind::Destructive`] classification, and which
    /// therefore no longer have valid on-disk data matching the current
    /// schema (Issue #1079).
    ///
    /// This is a visibility mechanism, not an enforcement one: unlike Solr
    /// (which silently allows a schema and its index to drift apart, see
    /// #1077's investigation), laurus records every destructive change
    /// here so `GetSchema`/`laurus get schema` can surface it instead of
    /// leaving the inconsistency undiscoverable. A field is added when a
    /// destructive [`FieldChangeKind`] is applied and is expected to be
    /// removed once the field has been rebuilt (rebuilding itself is out
    /// of scope for this phase — see #1080/#1081).
    ///
    /// [`Engine::update_field`]: crate::engine::Engine::update_field
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub pending_reindex: BTreeSet<String>,
}

impl Schema {
    pub fn new() -> Self {
        Self {
            analyzers: BTreeMap::new(),
            embedders: BTreeMap::new(),
            fields: BTreeMap::new(),
            default_fields: Vec::new(),
            dynamic_field_policy: DynamicFieldPolicy::default(),
            pending_reindex: BTreeSet::new(),
        }
    }

    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Schema {
    /// Parse a schema from a TOML string, in the same format
    /// `laurus-cli create index --schema` accepts (and what
    /// [`Schema::to_toml`] produces).
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LaurusError::Schema`] if `s` is not valid
    /// TOML or does not match the schema shape.
    pub fn from_toml(s: &str) -> crate::error::Result<Self> {
        toml::from_str(s).map_err(|e| {
            crate::error::LaurusError::schema(format!(
                "invalid schema TOML: {}",
                e.to_string().trim_end()
            ))
        })
    }

    /// Serialize this schema to a TOML string, in the same format
    /// `laurus-cli create index --schema` accepts.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LaurusError::Schema`] if serialization
    /// fails (this should not happen for a schema built through the
    /// public API).
    pub fn to_toml(&self) -> crate::error::Result<String> {
        toml::to_string_pretty(self).map_err(|e| {
            crate::error::LaurusError::schema(format!("failed to serialize schema to TOML: {e}"))
        })
    }
}

/// Options for a single field in the unified schema.
///
/// Each variant directly represents a concrete field type.
/// For hybrid search, define separate fields for vector and lexical indexing.
///
/// Serializes using serde's externally tagged representation:
/// ```json
/// { "Text": { "indexed": true, "stored": true, "term_vectors": false } }
/// { "Hnsw": { "dimension": 384, "distance": "Cosine" } }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldOption {
    /// Text field options (lexical search).
    Text(TextOption),
    /// Integer field options.
    Integer(IntegerOption),
    /// Float field options.
    Float(FloatOption),
    /// Boolean field options.
    Boolean(BooleanOption),
    /// DateTime field options.
    DateTime(DateTimeOption),
    /// 2D geo field options.
    Geo(GeoOption),
    /// 3D ECEF geo field options.
    Geo3d(Geo3dOption),
    /// Bytes field options.
    Bytes(BytesOption),
    /// HNSW vector index options.
    Hnsw(HnswOption),
    /// Flat vector index options.
    Flat(FlatOption),
    /// IVF vector index options.
    Ivf(IvfOption),
}

impl FieldOption {
    /// Returns true if this is a vector field.
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Hnsw(_) | Self::Flat(_) | Self::Ivf(_))
    }

    /// Returns true if this is a lexical field.
    pub fn is_lexical(&self) -> bool {
        matches!(
            self,
            Self::Text(_)
                | Self::Integer(_)
                | Self::Float(_)
                | Self::Boolean(_)
                | Self::DateTime(_)
                | Self::Geo(_)
                | Self::Geo3d(_)
                | Self::Bytes(_)
        )
    }

    /// Converts to the vector-subsystem's `FieldOption` if this is a vector field.
    pub fn to_vector(&self) -> Option<crate::vector::core::field::FieldOption> {
        match self {
            Self::Hnsw(o) => Some(crate::vector::core::field::FieldOption::Hnsw(o.clone())),
            Self::Flat(o) => Some(crate::vector::core::field::FieldOption::Flat(o.clone())),
            Self::Ivf(o) => Some(crate::vector::core::field::FieldOption::Ivf(o.clone())),
            _ => None,
        }
    }

    /// Returns the embedder name if this is a vector field with an embedder configured.
    pub fn embedder_name(&self) -> Option<&str> {
        match self {
            Self::Hnsw(o) => o.embedder.as_deref(),
            Self::Flat(o) => o.embedder.as_deref(),
            Self::Ivf(o) => o.embedder.as_deref(),
            _ => None,
        }
    }

    /// Converts to the lexical-subsystem's `FieldOption` if this is a lexical field.
    pub fn to_lexical(&self) -> Option<crate::lexical::core::field::FieldOption> {
        match self {
            Self::Text(o) => Some(crate::lexical::core::field::FieldOption::Text(o.clone())),
            Self::Integer(o) => Some(crate::lexical::core::field::FieldOption::Integer(o.clone())),
            Self::Float(o) => Some(crate::lexical::core::field::FieldOption::Float(o.clone())),
            Self::Boolean(o) => Some(crate::lexical::core::field::FieldOption::Boolean(o.clone())),
            Self::DateTime(o) => Some(crate::lexical::core::field::FieldOption::DateTime(
                o.clone(),
            )),
            Self::Geo(o) => Some(crate::lexical::core::field::FieldOption::Geo(o.clone())),
            Self::Geo3d(o) => Some(crate::lexical::core::field::FieldOption::Geo3d(o.clone())),
            Self::Bytes(o) => Some(crate::lexical::core::field::FieldOption::Bytes(o.clone())),
            _ => None,
        }
    }
}

/// The impact of changing a field's [`FieldOption`] via
/// [`Engine::update_field`](crate::engine::Engine::update_field) (Issue #1079).
///
/// Ordered from least to most disruptive (the derived [`Ord`] follows
/// declaration order): when several parameters change at once, the overall
/// classification is the most severe of the individual per-parameter
/// classifications — see [`classify_change`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FieldChangeKind {
    /// Only a schema-level setting changes; no existing on-disk data needs
    /// to be touched. For example, an HNSW field's `default_ef_search` is
    /// read only when a searcher is constructed, never persisted to a
    /// segment.
    MetadataOnly,
    /// Existing on-disk data must be rebuilt, but the values needed to do
    /// so are still available (the document store's stored fields, or —
    /// for vector fields — the raw vectors already on disk).
    Reindex,
    /// The change cannot be applied from existing on-disk data at all.
    /// Applying it discards the field's existing data (e.g. a dimension
    /// change, or any change to a field that is not `stored`).
    Destructive,
}

/// Classify a field option change from `old` to `new` (Issue #1079).
///
/// This is the core policy decision behind
/// [`Engine::update_field`](crate::engine::Engine::update_field): it never
/// touches storage and never fails, so it can be used both to decide how to
/// apply a change and, with `dry_run`, to report the impact of a
/// not-yet-applied one.
///
/// # Cross-type changes
///
/// Changing between two lexical variants (e.g. `Text` -> `Integer`) is a
/// [`FieldChangeKind::Reindex`] — analysis just needs to run again over the
/// stored values. Changing between two vector variants (e.g. `Hnsw` ->
/// `Flat`) is likewise a [`FieldChangeKind::Reindex`] in principle (the raw
/// vectors on disk are the same regardless of index algorithm), escalated
/// to [`FieldChangeKind::Destructive`] if `dimension`, `distance`, or
/// `embedder` also changed. Changing between a lexical and a vector variant
/// is always [`FieldChangeKind::Destructive`]: the two subsystems have no
/// shared "rebuild from existing data" path (vector fields are always
/// stored as raw vectors; lexical fields honor `stored` independently).
pub fn classify_change(old: &FieldOption, new: &FieldOption) -> FieldChangeKind {
    match (old, new) {
        (FieldOption::Text(o), FieldOption::Text(n)) => classify_text(o, n),
        (FieldOption::Integer(o), FieldOption::Integer(n)) => classify_numeric_lexical(
            o.indexed,
            n.indexed,
            o.stored,
            o.multi_valued,
            n.multi_valued,
        ),
        (FieldOption::Float(o), FieldOption::Float(n)) => classify_numeric_lexical(
            o.indexed,
            n.indexed,
            o.stored,
            o.multi_valued,
            n.multi_valued,
        ),
        (FieldOption::Boolean(o), FieldOption::Boolean(n)) => {
            classify_indexed_only(o.indexed, n.indexed, o.stored)
        }
        (FieldOption::DateTime(o), FieldOption::DateTime(n)) => {
            classify_indexed_only(o.indexed, n.indexed, o.stored)
        }
        (FieldOption::Geo(o), FieldOption::Geo(n)) => {
            classify_indexed_only(o.indexed, n.indexed, o.stored)
        }
        (FieldOption::Geo3d(o), FieldOption::Geo3d(n)) => {
            classify_indexed_only(o.indexed, n.indexed, o.stored)
        }
        // `stored` changes (for every lexical variant, including Bytes) are
        // always metadata-only: they only affect documents ingested after
        // the change, never data already on disk.
        (FieldOption::Bytes(_), FieldOption::Bytes(_)) => FieldChangeKind::MetadataOnly,

        (FieldOption::Hnsw(o), FieldOption::Hnsw(n)) => {
            classify_vector_common(old, new).max(classify_hnsw_specific(o, n))
        }
        (FieldOption::Flat(o), FieldOption::Flat(n)) => {
            classify_vector_common(old, new).max(classify_flat_specific(o, n))
        }
        (FieldOption::Ivf(o), FieldOption::Ivf(n)) => {
            classify_vector_common(old, new).max(classify_ivf_specific(o, n))
        }

        (o, n) if o.is_lexical() && n.is_lexical() => FieldChangeKind::Reindex,
        (o, n) if o.is_vector() && n.is_vector() => {
            classify_vector_common(o, n).max(FieldChangeKind::Reindex)
        }
        _ => FieldChangeKind::Destructive,
    }
}

/// Shared classification for `indexed`-only lexical options (Boolean,
/// DateTime, Geo, Geo3d).
///
/// Only the `false -> true` transition requires rebuilding: documents
/// ingested while the field was `indexed: false` have no postings to
/// search, so turning indexing on only takes effect for documents ingested
/// afterward unless the field is rebuilt. The reverse direction
/// (`true -> false`) leaves existing postings in place (the query parser
/// does not consult `indexed` — see `laurus/src/lexical/index/inverted/writer.rs`,
/// which is the only place `indexed` is read, at document-ingestion time),
/// so it is metadata-only, matching the same limitation `delete_field`
/// already has (Issue #1077's investigation).
///
/// A rebuild needs the field's original values, which only exist when
/// `old_stored` is `true` (Issue #1081: laurus's lexical rebuild sources
/// original values from the segment's own stored fields, not a document
/// store, but the constraint is the same either way — nothing is stored
/// for `stored: false` fields once a segment is sealed). Without them the
/// change can only be applied by discarding the field's existing data, so
/// it is classified `Destructive` instead of `Reindex`.
fn classify_indexed_only(
    old_indexed: bool,
    new_indexed: bool,
    old_stored: bool,
) -> FieldChangeKind {
    if !old_indexed && new_indexed {
        if old_stored {
            FieldChangeKind::Reindex
        } else {
            FieldChangeKind::Destructive
        }
    } else {
        FieldChangeKind::MetadataOnly
    }
}

/// Classification for `TextOption`: `indexed` follows
/// [`classify_indexed_only`]; an `analyzer` change always requires
/// rebuilding from the field's original values, since existing postings
/// were built with the old analyzer — classified `Reindex` when
/// `old.stored` (the rebuild can source original text from the segment's
/// stored fields) or `Destructive` otherwise (see
/// [`classify_indexed_only`]'s doc comment).
///
/// `term_vectors: false -> true` needs the same rebuild as an analyzer
/// change: existing postings were written without positions, and there is
/// no way to recover them other than re-deriving the field from its
/// original value — `Reindex` when `old.stored`, `Destructive` otherwise.
/// `true -> false` is `MetadataOnly`: unlike `multi_valued` (where a stale
/// on-disk shape can be misread under the new setting), a reader tells
/// positions apart per-posting via the segment's own `any_positions`
/// header, so leftover positions on disk are simply never read once the
/// field's setting is `false` — no correctness risk either way (#1083).
fn classify_text(old: &TextOption, new: &TextOption) -> FieldChangeKind {
    let mut kind = classify_indexed_only(old.indexed, new.indexed, old.stored);
    if old.analyzer != new.analyzer {
        kind = kind.max(if old.stored {
            FieldChangeKind::Reindex
        } else {
            FieldChangeKind::Destructive
        });
    }
    if !old.term_vectors && new.term_vectors {
        kind = kind.max(if old.stored {
            FieldChangeKind::Reindex
        } else {
            FieldChangeKind::Destructive
        });
    }
    kind
}

/// Classification shared by `IntegerOption`/`FloatOption`: `indexed`
/// follows [`classify_indexed_only`] (`stored`-dependent, since turning
/// indexing on has no BKD points to source from for a `stored: false`
/// field that was never indexed). `multi_valued: false -> true` is
/// metadata-only (existing single values are still valid single-element
/// matches under "any match" semantics); the reverse could leave stale
/// multi-value postings misread as single-valued, so it conservatively
/// requires a reindex — always `Reindex`, never `Destructive`, regardless
/// of `stored`: numeric points are read back from the segment's BKD tree
/// (the authoritative source, Issue #758), not from stored fields, so a
/// field that was already `indexed: true` always has a rebuild source.
fn classify_numeric_lexical(
    old_indexed: bool,
    new_indexed: bool,
    old_stored: bool,
    old_multi_valued: bool,
    new_multi_valued: bool,
) -> FieldChangeKind {
    let mut kind = classify_indexed_only(old_indexed, new_indexed, old_stored);
    if old_multi_valued && !new_multi_valued {
        kind = kind.max(FieldChangeKind::Reindex);
    }
    kind
}

/// Classification for the parameters shared by every vector variant
/// (`dimension`, `distance`, `embedder`), regardless of whether `old`/`new`
/// are the same variant.
///
/// All three are [`FieldChangeKind::Destructive`]: `dimension` makes
/// existing vectors the wrong shape; `embedder` changes what a document's
/// raw input should have produced (the stored vectors were produced by the
/// old embedder); `distance` is deceptively simple but
/// `laurus/src/vector/index/config.rs` normalizes stored vectors when
/// `distance == Cosine`, so a naive rebuild under a new distance would
/// misinterpret already-normalized vectors.
///
/// # Panics
///
/// Panics if `old` or `new` is not a vector `FieldOption` — callers must
/// only invoke this after matching on a vector variant.
fn classify_vector_common(old: &FieldOption, new: &FieldOption) -> FieldChangeKind {
    let old_v = old
        .to_vector()
        .expect("classify_vector_common: `old` must be a vector FieldOption");
    let new_v = new
        .to_vector()
        .expect("classify_vector_common: `new` must be a vector FieldOption");
    if old_v.dimension() != new_v.dimension()
        || old_v.distance() != new_v.distance()
        || old.embedder_name() != new.embedder_name()
    {
        FieldChangeKind::Destructive
    } else {
        FieldChangeKind::MetadataOnly
    }
}

/// Classification for HNSW-specific parameters. `default_ef_search` and
/// `base_weight` are metadata-only: both are read only at query time
/// (`base_weight` by `VectorStore::search_impl`'s per-field query-weight
/// computation, Issue #1084; `default_ef_search` by the searcher
/// construction path), so neither needs an on-disk rebuild. Known gap
/// (shared by both, not yet fixed): `Engine::update_field`'s
/// `MetadataOnly` arm updates the persisted schema but does not push the
/// new value into an already-open `VectorStore` — the change only takes
/// effect the next time the index is opened, not immediately, despite the
/// `MetadataOnly` classification's usual "no rebuild needed" implication.
/// `m`/`ef_construction`/`quantizer`/`rerank_storage`/`pq_codebook_path`
/// all require rebuilding the graph from the raw vectors already on disk.
fn classify_hnsw_specific(old: &HnswOption, new: &HnswOption) -> FieldChangeKind {
    let mut kind = FieldChangeKind::MetadataOnly;
    if old.m != new.m
        || old.ef_construction != new.ef_construction
        || old.quantizer != new.quantizer
        || old.rerank_storage != new.rerank_storage
        || old.pq_codebook_path != new.pq_codebook_path
    {
        kind = kind.max(FieldChangeKind::Reindex);
    }
    kind
}

/// Classification for Flat-specific parameters. `base_weight` is
/// metadata-only (read only at query time by `VectorStore::search_impl`'s
/// per-field query-weight computation, Issue #1084 — see
/// `classify_hnsw_specific`'s doc comment for the known "not live-reflected
/// without reopening" gap); `quantizer`/`rerank_storage` require rebuilding
/// from the raw vectors already on disk.
fn classify_flat_specific(old: &FlatOption, new: &FlatOption) -> FieldChangeKind {
    if old.quantizer != new.quantizer || old.rerank_storage != new.rerank_storage {
        FieldChangeKind::Reindex
    } else {
        FieldChangeKind::MetadataOnly
    }
}

/// Classification for IVF-specific parameters. `n_probe` and `base_weight`
/// are metadata-only (search-time only; the persisted `n_probe` value's
/// read-back is discarded — see `laurus/src/vector/index/ivf/writer.rs`).
/// `base_weight` is read at query time by `VectorStore::search_impl`'s
/// per-field query-weight computation (Issue #1084) — see
/// `classify_hnsw_specific`'s doc comment for the known "not live-reflected
/// without reopening" gap. `n_clusters` requires re-running k-means;
/// `quantizer`/`rerank_storage` require rebuilding from the raw vectors
/// already on disk.
fn classify_ivf_specific(old: &IvfOption, new: &IvfOption) -> FieldChangeKind {
    if old.n_clusters != new.n_clusters
        || old.quantizer != new.quantizer
        || old.rerank_storage != new.rerank_storage
    {
        FieldChangeKind::Reindex
    } else {
        FieldChangeKind::MetadataOnly
    }
}

#[derive(Default)]
pub struct SchemaBuilder {
    analyzers: BTreeMap<String, AnalyzerDefinition>,
    embedders: BTreeMap<String, EmbedderDefinition>,
    fields: BTreeMap<String, FieldOption>,
    default_fields: Vec<String>,
    dynamic_field_policy: DynamicFieldPolicy,
}

impl SchemaBuilder {
    pub fn add_field(mut self, name: impl Into<String>, option: FieldOption) -> Self {
        let name = name.into();
        self.fields.insert(name, option);
        self
    }

    pub fn add_text_field(self, name: impl Into<String>, option: impl Into<TextOption>) -> Self {
        self.add_field(name, FieldOption::Text(option.into()))
    }

    pub fn add_integer_field(
        self,
        name: impl Into<String>,
        option: impl Into<IntegerOption>,
    ) -> Self {
        self.add_field(name, FieldOption::Integer(option.into()))
    }

    pub fn add_float_field(self, name: impl Into<String>, option: impl Into<FloatOption>) -> Self {
        self.add_field(name, FieldOption::Float(option.into()))
    }

    pub fn add_boolean_field(
        self,
        name: impl Into<String>,
        option: impl Into<BooleanOption>,
    ) -> Self {
        self.add_field(name, FieldOption::Boolean(option.into()))
    }

    pub fn add_datetime_field(
        self,
        name: impl Into<String>,
        option: impl Into<DateTimeOption>,
    ) -> Self {
        self.add_field(name, FieldOption::DateTime(option.into()))
    }

    pub fn add_geo_field(self, name: impl Into<String>, option: impl Into<GeoOption>) -> Self {
        self.add_field(name, FieldOption::Geo(option.into()))
    }

    /// Add a 3D ECEF geo field, indexed in a 3D BKD tree for sphere /
    /// k-NN queries (queries themselves arrive with #300–#302).
    pub fn add_geo3d_field(self, name: impl Into<String>, option: impl Into<Geo3dOption>) -> Self {
        self.add_field(name, FieldOption::Geo3d(option.into()))
    }

    pub fn add_bytes_field(self, name: impl Into<String>, option: impl Into<BytesOption>) -> Self {
        self.add_field(name, FieldOption::Bytes(option.into()))
    }

    pub fn add_hnsw_field(self, name: impl Into<String>, option: impl Into<HnswOption>) -> Self {
        self.add_field(name, FieldOption::Hnsw(option.into()))
    }

    pub fn add_flat_field(self, name: impl Into<String>, option: impl Into<FlatOption>) -> Self {
        self.add_field(name, FieldOption::Flat(option.into()))
    }

    pub fn add_ivf_field(self, name: impl Into<String>, option: impl Into<IvfOption>) -> Self {
        self.add_field(name, FieldOption::Ivf(option.into()))
    }

    pub fn add_default_field(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.default_fields.push(name);
        self
    }

    /// Add a custom analyzer definition to the schema.
    ///
    /// # Arguments
    ///
    /// * `name` - The analyzer name (referenced from `TextOption::analyzer`).
    /// * `definition` - The analyzer definition.
    pub fn add_analyzer(mut self, name: impl Into<String>, definition: AnalyzerDefinition) -> Self {
        self.analyzers.insert(name.into(), definition);
        self
    }

    /// Add an embedder definition to the schema.
    ///
    /// # Arguments
    ///
    /// * `name` - The embedder name (referenced from vector field `embedder`).
    /// * `definition` - The embedder definition.
    pub fn add_embedder(mut self, name: impl Into<String>, definition: EmbedderDefinition) -> Self {
        self.embedders.insert(name.into(), definition);
        self
    }

    /// Sets the policy for fields not declared in the schema.
    ///
    /// See [`DynamicFieldPolicy`] for the available options. When not set,
    /// the default is [`DynamicFieldPolicy::Dynamic`].
    ///
    /// # Arguments
    ///
    /// * `policy` - The dynamic field policy to apply during ingestion.
    pub fn dynamic_field_policy(mut self, policy: DynamicFieldPolicy) -> Self {
        self.dynamic_field_policy = policy;
        self
    }

    /// Build the schema, validating reserved field names.
    ///
    /// # Errors
    ///
    /// Returns an error if any field name starts with `_` and is not in the
    /// reserved allow-list (see [`validate_field_name`]).
    pub fn try_build(self) -> crate::error::Result<Schema> {
        for name in self.fields.keys() {
            validate_field_name(name)?;
        }
        Ok(Schema {
            analyzers: self.analyzers,
            embedders: self.embedders,
            fields: self.fields,
            default_fields: self.default_fields,
            dynamic_field_policy: self.dynamic_field_policy,
            pending_reindex: BTreeSet::new(),
        })
    }

    /// Build the schema.
    ///
    /// # Panics
    ///
    /// Panics if any field name collides with a reserved name. Use
    /// [`try_build`](Self::try_build) for a fallible variant.
    pub fn build(self) -> Schema {
        self.try_build()
            .expect("SchemaBuilder::build: field name validation failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::core::field::TextOption;

    #[test]
    fn default_dynamic_field_policy_is_dynamic() {
        assert_eq!(DynamicFieldPolicy::default(), DynamicFieldPolicy::Dynamic);
    }

    #[test]
    fn schema_new_uses_default_policy() {
        let schema = Schema::new();
        assert_eq!(schema.dynamic_field_policy, DynamicFieldPolicy::Dynamic);
    }

    #[test]
    fn schema_builder_sets_policy() {
        let schema = Schema::builder()
            .dynamic_field_policy(DynamicFieldPolicy::Strict)
            .build();
        assert_eq!(schema.dynamic_field_policy, DynamicFieldPolicy::Strict);
    }

    #[test]
    fn validate_field_name_accepts_regular_name() {
        assert!(validate_field_name("title").is_ok());
        assert!(validate_field_name("year_2024").is_ok());
        assert!(validate_field_name("a").is_ok());
    }

    #[test]
    fn validate_field_name_accepts_id() {
        assert!(validate_field_name(RESERVED_ID_FIELD).is_ok());
    }

    #[test]
    fn validate_field_name_rejects_underscore_prefix() {
        let err = validate_field_name("_score").unwrap_err();
        assert!(
            err.to_string().contains("reserved"),
            "unexpected error: {err}"
        );
        assert!(validate_field_name("_custom").is_err());
        assert!(validate_field_name("__foo").is_err());
    }

    #[test]
    fn schema_builder_try_build_rejects_reserved_name() {
        let result = Schema::builder()
            .add_field("_bad", FieldOption::Text(TextOption::default()))
            .try_build();
        assert!(result.is_err());
    }

    #[test]
    fn schema_builder_try_build_accepts_regular_names() {
        let result = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .try_build();
        assert!(result.is_ok());
    }

    #[test]
    fn schema_builder_add_geo3d_field_round_trips() {
        let schema = Schema::builder()
            .add_geo3d_field("position", Geo3dOption::default())
            .build();
        let opt = schema.fields.get("position").expect("field declared");
        match opt {
            FieldOption::Geo3d(g3d) => {
                assert!(g3d.indexed);
                assert!(g3d.stored);
            }
            other => panic!("expected FieldOption::Geo3d, got {other:?}"),
        }

        // The engine schema -> lexical schema bridge must preserve Geo3d.
        let lexical = opt.to_lexical().expect("Geo3d is a lexical field");
        assert!(matches!(
            lexical,
            crate::lexical::core::field::FieldOption::Geo3d(_)
        ));
        assert!(opt.is_lexical());
        assert!(!opt.is_vector());
    }

    #[test]
    fn dynamic_field_policy_serde_round_trip() {
        for policy in [
            DynamicFieldPolicy::Strict,
            DynamicFieldPolicy::Dynamic,
            DynamicFieldPolicy::Ignore,
        ] {
            let json = serde_json::to_string(&policy).unwrap();
            let back: DynamicFieldPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(policy, back);
        }
    }

    /// Issue #1079: an empty `pending_reindex` (the common case — no
    /// destructive `update_field` has ever been applied) is omitted from
    /// the TOML entirely, keeping `schema.toml` unchanged for schemas that
    /// never touch the new field.
    #[test]
    fn pending_reindex_empty_is_omitted_from_toml() {
        let schema = Schema::new();
        let toml = schema.to_toml().unwrap();
        assert!(
            !toml.contains("pending_reindex"),
            "empty pending_reindex should not appear in TOML: {toml}"
        );
    }

    /// Issue #1079: a non-empty `pending_reindex` round-trips through TOML,
    /// so `laurus get schema` (or reopening the index) still surfaces which
    /// fields need a rebuild.
    #[test]
    fn pending_reindex_round_trips_through_toml() {
        let mut schema = Schema::new();
        schema.pending_reindex.insert("embedding".to_string());
        schema.pending_reindex.insert("body".to_string());

        let toml = schema.to_toml().unwrap();
        let back = Schema::from_toml(&toml).unwrap();
        assert_eq!(back.pending_reindex, schema.pending_reindex);
    }

    /// Issue #1079: table-driven coverage of `classify_change` for all 11
    /// `FieldOption` variants, plus the cross-variant (lexical<->lexical,
    /// vector<->vector, lexical<->vector) rules.
    #[test]
    fn classify_change_table() {
        use crate::vector::core::distance::DistanceMetric;
        use crate::vector::core::quantization::QuantizationMethod;
        use crate::vector::core::rerank::RerankStorageKind;
        use FieldChangeKind::{Destructive, MetadataOnly, Reindex};

        let text = |f: fn(TextOption) -> TextOption| FieldOption::Text(f(TextOption::default()));
        let integer = |f: fn(IntegerOption) -> IntegerOption| {
            FieldOption::Integer(f(IntegerOption::default()))
        };
        let float =
            |f: fn(FloatOption) -> FloatOption| FieldOption::Float(f(FloatOption::default()));
        let boolean = |f: fn(BooleanOption) -> BooleanOption| {
            FieldOption::Boolean(f(BooleanOption::default()))
        };
        let datetime = |f: fn(DateTimeOption) -> DateTimeOption| {
            FieldOption::DateTime(f(DateTimeOption::default()))
        };
        let geo = |f: fn(GeoOption) -> GeoOption| FieldOption::Geo(f(GeoOption::default()));
        let geo3d =
            |f: fn(Geo3dOption) -> Geo3dOption| FieldOption::Geo3d(f(Geo3dOption::default()));
        let bytes =
            |f: fn(BytesOption) -> BytesOption| FieldOption::Bytes(f(BytesOption::default()));
        let hnsw = |f: fn(HnswOption) -> HnswOption| FieldOption::Hnsw(f(HnswOption::default()));
        let flat = |f: fn(FlatOption) -> FlatOption| FieldOption::Flat(f(FlatOption::default()));
        let ivf = |f: fn(IvfOption) -> IvfOption| FieldOption::Ivf(f(IvfOption::default()));

        let cases: Vec<(&str, FieldOption, FieldOption, FieldChangeKind)> = vec![
            // ---- Text ----
            (
                "text: stored toggle is metadata-only",
                text(|o| o),
                text(|o| o.stored(false)),
                MetadataOnly,
            ),
            (
                "text: indexed false->true requires reindex",
                text(|o| o.indexed(false)),
                text(|o| o.indexed(true)),
                Reindex,
            ),
            (
                "text: indexed true->false is metadata-only",
                text(|o| o.indexed(true)),
                text(|o| o.indexed(false)),
                MetadataOnly,
            ),
            (
                "text: analyzer change requires reindex",
                text(|o| o),
                text(|o| o.analyzer("english")),
                Reindex,
            ),
            (
                "text: analyzer change on a stored:false field is destructive (no original text to re-analyze)",
                text(|o| o.stored(false)),
                text(|o| o.stored(false).analyzer("english")),
                Destructive,
            ),
            (
                "text: indexed false->true on a stored:false field is destructive (no original text to index)",
                text(|o| o.stored(false).indexed(false)),
                text(|o| o.stored(false).indexed(true)),
                Destructive,
            ),
            (
                "text: term_vectors false->true requires reindex",
                text(|o| o.term_vectors(false)),
                text(|o| o.term_vectors(true)),
                Reindex,
            ),
            (
                "text: term_vectors true->false is metadata-only",
                text(|o| o.term_vectors(true)),
                text(|o| o.term_vectors(false)),
                MetadataOnly,
            ),
            (
                "text: term_vectors false->true on a stored:false field is destructive (no original text to re-derive positions from)",
                text(|o| o.stored(false).term_vectors(false)),
                text(|o| o.stored(false).term_vectors(true)),
                Destructive,
            ),
            // ---- Integer ----
            (
                "integer: stored toggle is metadata-only",
                integer(|o| o),
                integer(|o| o.stored(false)),
                MetadataOnly,
            ),
            (
                "integer: indexed false->true requires reindex",
                integer(|o| o.indexed(false)),
                integer(|o| o.indexed(true)),
                Reindex,
            ),
            (
                "integer: multi_valued false->true is metadata-only",
                integer(|o| o),
                integer(|mut o| {
                    o.multi_valued = true;
                    o
                }),
                MetadataOnly,
            ),
            (
                "integer: multi_valued true->false requires reindex",
                integer(|mut o| {
                    o.multi_valued = true;
                    o
                }),
                integer(|o| o),
                Reindex,
            ),
            (
                "integer: indexed false->true on a stored:false field is destructive (no BKD points, never indexed)",
                integer(|o| o.stored(false).indexed(false)),
                integer(|o| o.stored(false).indexed(true)),
                Destructive,
            ),
            (
                "integer: multi_valued true->false on a stored:false field stays a reindex (BKD is the source, not stored fields)",
                integer(|mut o| {
                    o.stored = false;
                    o.multi_valued = true;
                    o
                }),
                integer(|o| o.stored(false)),
                Reindex,
            ),
            // ---- Float ----
            (
                "float: stored toggle is metadata-only",
                float(|o| o),
                float(|o| o.stored(false)),
                MetadataOnly,
            ),
            (
                "float: indexed false->true requires reindex",
                float(|o| o.indexed(false)),
                float(|o| o.indexed(true)),
                Reindex,
            ),
            (
                "float: multi_valued true->false requires reindex",
                float(|mut o| {
                    o.multi_valued = true;
                    o
                }),
                float(|o| o),
                Reindex,
            ),
            // ---- Boolean ----
            (
                "boolean: stored toggle is metadata-only",
                boolean(|o| o),
                boolean(|o| o.stored(false)),
                MetadataOnly,
            ),
            (
                "boolean: indexed false->true requires reindex",
                boolean(|o| o.indexed(false)),
                boolean(|o| o.indexed(true)),
                Reindex,
            ),
            // ---- DateTime ----
            (
                "datetime: stored toggle is metadata-only",
                datetime(|o| o),
                datetime(|o| o.stored(false)),
                MetadataOnly,
            ),
            (
                "datetime: indexed false->true requires reindex",
                datetime(|o| o.indexed(false)),
                datetime(|o| o.indexed(true)),
                Reindex,
            ),
            // ---- Geo ----
            (
                "geo: stored toggle is metadata-only",
                geo(|o| o),
                geo(|o| o.stored(false)),
                MetadataOnly,
            ),
            (
                "geo: indexed false->true requires reindex",
                geo(|o| o.indexed(false)),
                geo(|o| o.indexed(true)),
                Reindex,
            ),
            // ---- Geo3d ----
            (
                "geo3d: stored toggle is metadata-only",
                geo3d(|o| o),
                geo3d(|o| o.stored(false)),
                MetadataOnly,
            ),
            (
                "geo3d: indexed false->true requires reindex",
                geo3d(|o| o.indexed(false)),
                geo3d(|o| o.indexed(true)),
                Reindex,
            ),
            // ---- Bytes (no `indexed`; only `stored`) ----
            (
                "bytes: stored toggle is metadata-only",
                bytes(|o| o),
                bytes(|o| o.stored(false)),
                MetadataOnly,
            ),
            // ---- Hnsw ----
            (
                "hnsw: default_ef_search change is metadata-only",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.default_ef_search = Some(64);
                    o
                }),
                MetadataOnly,
            ),
            (
                "hnsw: base_weight change is metadata-only",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.base_weight = 2.0;
                    o
                }),
                MetadataOnly,
            ),
            (
                "hnsw: m change requires reindex",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.m = 32;
                    o
                }),
                Reindex,
            ),
            (
                "hnsw: ef_construction change requires reindex",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.ef_construction = 400;
                    o
                }),
                Reindex,
            ),
            (
                "hnsw: quantizer change requires reindex",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.quantizer = QuantizationMethod::ProductQuantization { subvector_count: 8 };
                    o
                }),
                Reindex,
            ),
            (
                "hnsw: rerank_storage change requires reindex",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.rerank_storage = Some(RerankStorageKind::F32);
                    o
                }),
                Reindex,
            ),
            (
                "hnsw: pq_codebook_path change requires reindex",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.pq_codebook_path = Some("codebook.pqcb".to_string());
                    o
                }),
                Reindex,
            ),
            (
                "hnsw: dimension change is destructive",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.dimension = 256;
                    o
                }),
                Destructive,
            ),
            (
                "hnsw: distance change is destructive",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.distance = DistanceMetric::Euclidean;
                    o
                }),
                Destructive,
            ),
            (
                "hnsw: embedder change is destructive",
                hnsw(|o| o),
                hnsw(|mut o| {
                    o.embedder = Some("new-embedder".to_string());
                    o
                }),
                Destructive,
            ),
            // ---- Flat ----
            (
                "flat: base_weight change is metadata-only",
                flat(|o| o),
                flat(|mut o| {
                    o.base_weight = 2.0;
                    o
                }),
                MetadataOnly,
            ),
            (
                "flat: quantizer change requires reindex",
                flat(|o| o),
                flat(|mut o| {
                    o.quantizer = QuantizationMethod::ProductQuantization { subvector_count: 8 };
                    o
                }),
                Reindex,
            ),
            (
                "flat: rerank_storage change requires reindex",
                flat(|o| o),
                flat(|mut o| {
                    o.rerank_storage = Some(RerankStorageKind::F32);
                    o
                }),
                Reindex,
            ),
            (
                "flat: dimension change is destructive",
                flat(|o| o),
                flat(|mut o| {
                    o.dimension = 256;
                    o
                }),
                Destructive,
            ),
            // ---- Ivf ----
            (
                "ivf: n_probe change is metadata-only",
                ivf(|o| o),
                ivf(|mut o| {
                    o.n_probe = 8;
                    o
                }),
                MetadataOnly,
            ),
            (
                "ivf: base_weight change is metadata-only",
                ivf(|o| o),
                ivf(|mut o| {
                    o.base_weight = 2.0;
                    o
                }),
                MetadataOnly,
            ),
            (
                "ivf: n_clusters change requires reindex",
                ivf(|o| o),
                ivf(|mut o| {
                    o.n_clusters = 200;
                    o
                }),
                Reindex,
            ),
            (
                "ivf: quantizer change requires reindex",
                ivf(|o| o),
                ivf(|mut o| {
                    o.quantizer = QuantizationMethod::ProductQuantization { subvector_count: 8 };
                    o
                }),
                Reindex,
            ),
            (
                "ivf: dimension change is destructive",
                ivf(|o| o),
                ivf(|mut o| {
                    o.dimension = 256;
                    o
                }),
                Destructive,
            ),
            // ---- Cross-variant: lexical <-> lexical ----
            (
                "text -> integer is a reindex (lexical type change)",
                text(|o| o),
                integer(|o| o),
                Reindex,
            ),
            (
                "bytes -> text is a reindex (lexical type change)",
                bytes(|o| o),
                text(|o| o),
                Reindex,
            ),
            // ---- Cross-variant: vector <-> vector ----
            (
                "hnsw -> flat with identical dimension/distance/embedder is a reindex",
                hnsw(|o| o),
                flat(|o| o),
                Reindex,
            ),
            (
                "hnsw -> flat with a dimension change is destructive",
                hnsw(|o| o),
                flat(|mut o| {
                    o.dimension = 256;
                    o
                }),
                Destructive,
            ),
            (
                "flat -> ivf with identical dimension/distance/embedder is a reindex",
                flat(|o| o),
                ivf(|o| o),
                Reindex,
            ),
            // ---- Cross-variant: lexical <-> vector ----
            (
                "text -> hnsw is destructive (different storage subsystem)",
                text(|o| o),
                hnsw(|o| o),
                Destructive,
            ),
            (
                "hnsw -> text is destructive (different storage subsystem)",
                hnsw(|o| o),
                text(|o| o),
                Destructive,
            ),
        ];

        for (desc, old, new, expected) in cases {
            assert_eq!(classify_change(&old, &new), expected, "case failed: {desc}");
        }
    }

    /// Issue #1060: `Schema`'s `analyzers`/`embedders`/`fields` are
    /// `BTreeMap`s, not `HashMap`s, so `to_toml()` must produce
    /// byte-identical output across repeated calls on the same schema.
    /// With `HashMap`, table order would instead vary with the
    /// process's per-instance hash seed.
    #[test]
    fn schema_toml_serialization_is_deterministic_across_calls() {
        use crate::engine::schema::analyzer::{AnalyzerDefinition, TokenizerConfig};
        use crate::engine::schema::embedder::EmbedderDefinition;
        use crate::lexical::core::field::{BooleanOption, DateTimeOption, IntegerOption};

        let schema = Schema::builder()
            .add_text_field("title", TextOption::default())
            .add_text_field("body", TextOption::default())
            .add_integer_field("year", IntegerOption::default())
            .add_boolean_field("published", BooleanOption::default())
            .add_datetime_field("created_at", DateTimeOption::default())
            .add_analyzer(
                "analyzer_z",
                AnalyzerDefinition {
                    char_filters: vec![],
                    tokenizer: TokenizerConfig::Whitespace,
                    token_filters: vec![],
                },
            )
            .add_analyzer(
                "analyzer_a",
                AnalyzerDefinition {
                    char_filters: vec![],
                    tokenizer: TokenizerConfig::Whitespace,
                    token_filters: vec![],
                },
            )
            .add_embedder("embedder_z", EmbedderDefinition::Precomputed)
            .add_embedder("embedder_a", EmbedderDefinition::Precomputed)
            .build();

        let first = schema.to_toml().unwrap();
        for _ in 0..10 {
            assert_eq!(
                schema.to_toml().unwrap(),
                first,
                "TOML output must be byte-identical across repeated calls"
            );
        }
    }

    /// Issue #1060: a `CharFilterConfig::Mapping`'s dictionary is also a
    /// `BTreeMap` for the same reason -- it's serialized as part of the
    /// schema's `analyzers` table.
    #[test]
    fn schema_toml_serialization_is_deterministic_with_mapping_char_filter() {
        use crate::CharFilterConfig;
        use crate::engine::schema::analyzer::{AnalyzerDefinition, TokenizerConfig};

        let mut mapping = std::collections::BTreeMap::new();
        mapping.insert("z".to_string(), "zed".to_string());
        mapping.insert("a".to_string(), "ay".to_string());
        mapping.insert("m".to_string(), "em".to_string());

        let schema = Schema::builder()
            .add_analyzer(
                "with_mapping",
                AnalyzerDefinition {
                    char_filters: vec![CharFilterConfig::Mapping { mapping }],
                    tokenizer: TokenizerConfig::Whitespace,
                    token_filters: vec![],
                },
            )
            .build();

        let first = schema.to_toml().unwrap();
        for _ in 0..10 {
            assert_eq!(
                schema.to_toml().unwrap(),
                first,
                "TOML output must be byte-identical across repeated calls"
            );
        }
    }
}
