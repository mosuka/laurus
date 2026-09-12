//! Faceted search functionality for categorizing and filtering search results.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::core::field::FieldValue;
use crate::lexical::query::Hit;
use crate::lexical::query::Query;
use crate::lexical::reader::LexicalIndexReader;

/// Represents a facet field and its hierarchical structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FacetPath {
    /// The field name this facet belongs to.
    pub field: String,
    /// Hierarchical path components (e.g., ["Electronics", "Computers", "Laptops"]).
    pub path: Vec<String>,
}

impl FacetPath {
    /// Create a new facet path.
    pub fn new(field: String, path: Vec<String>) -> Self {
        FacetPath { field, path }
    }

    /// Create a facet path from a single value.
    pub fn from_value(field: String, value: String) -> Self {
        FacetPath {
            field,
            path: vec![value],
        }
    }

    /// Create a facet path from a delimited string.
    pub fn from_delimited(field: String, path_str: &str, delimiter: &str) -> Self {
        let path = path_str.split(delimiter).map(|s| s.to_string()).collect();
        FacetPath { field, path }
    }

    /// Get the depth of this facet path.
    pub fn depth(&self) -> usize {
        self.path.len()
    }

    /// Check if this path is a parent of another path.
    pub fn is_parent_of(&self, other: &FacetPath) -> bool {
        if self.field != other.field || self.depth() >= other.depth() {
            return false;
        }

        self.path.iter().zip(other.path.iter()).all(|(a, b)| a == b)
    }

    /// Get the parent path (one level up).
    pub fn parent(&self) -> Option<FacetPath> {
        if self.path.len() > 1 {
            let mut parent_path = self.path.clone();
            parent_path.pop();
            Some(FacetPath {
                field: self.field.clone(),
                path: parent_path,
            })
        } else {
            None
        }
    }

    /// Create a child path by appending a component.
    pub fn child(&self, component: String) -> FacetPath {
        let mut child_path = self.path.clone();
        child_path.push(component);
        FacetPath {
            field: self.field.clone(),
            path: child_path,
        }
    }

    /// Convert to a string representation.
    pub fn to_string_with_delimiter(&self, delimiter: &str) -> String {
        self.path.join(delimiter)
    }
}

/// Represents a facet count for a specific path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetCount {
    /// The facet path.
    pub path: FacetPath,
    /// Number of documents matching this facet.
    pub count: u64,
    /// Child facets (for hierarchical display).
    pub children: Vec<FacetCount>,
}

impl FacetCount {
    /// Create a new facet count.
    pub fn new(path: FacetPath, count: u64) -> Self {
        FacetCount {
            path,
            count,
            children: Vec::new(),
        }
    }

    /// Add a child facet count.
    pub fn add_child(&mut self, child: FacetCount) {
        self.children.push(child);
    }

    /// Sort children by count (descending) or name (ascending).
    pub fn sort_children(&mut self, by_count: bool) {
        if by_count {
            self.children.sort_by_key(|c| std::cmp::Reverse(c.count));
        } else {
            self.children
                .sort_by(|a, b| a.path.path.last().cmp(&b.path.path.last()));
        }

        // Recursively sort children
        for child in &mut self.children {
            child.sort_children(by_count);
        }
    }
}

/// Configuration for facet collection.
#[derive(Debug, Clone)]
pub struct FacetConfig {
    /// Maximum number of facet values to return per field.
    pub max_facets_per_field: usize,
    /// Maximum depth for hierarchical facets.
    pub max_depth: usize,
    /// Minimum count threshold for including a facet.
    pub min_count: u64,
    /// Whether to include zero counts for missing facets.
    pub include_zero_counts: bool,
    /// Sort facets by count (true) or alphabetically (false).
    pub sort_by_count: bool,
}

impl Default for FacetConfig {
    fn default() -> Self {
        FacetConfig {
            max_facets_per_field: 100,
            max_depth: 10,
            min_count: 1,
            include_zero_counts: false,
            sort_by_count: true,
        }
    }
}

/// Facet collector that accumulates facet counts during search.
///
/// Internally counts are keyed by interned `(field_id, value_id)` pairs
/// (#409): the `entry().or_insert(0) += 1` hot path no longer pays the
/// `FacetPath` clone + per-`String` hash that the original
/// `HashMap<FacetPath, _>` representation incurred on every increment.
/// Two specialised counter maps keep keys cheap to hash and parent walks
/// allocation-free:
///
/// - `flat_counts: HashMap<u64, u64>` — depth-1 paths. The 64-bit key
///   packs `(field_id << 32) | value_id`, so each increment hashes a
///   single `u64` instead of a `String + Vec<String>` pair.
/// - `hier_counts: HashMap<Box<[u32]>, u64>` — depth-N paths. The boxed
///   slice stores `[field_id, level0_id, level1_id, …]`; the parent walk
///   shrinks a local `Vec<u32>` by `pop()` at each level and only pays a
///   single `Box<[u32]>` allocation when the entry doesn't yet exist.
///
/// Field names and value strings are interned in two `String → u32`
/// maps owned by the collector and decoded back to strings only at
/// `finalize` time.
#[derive(Debug)]
pub struct FacetCollector {
    /// Configuration for facet collection.
    config: FacetConfig,
    /// Fields to collect facets for.
    facet_fields: Vec<String>,
    /// Interned ids for each entry in `facet_fields`, populated once at
    /// construction so the per-doc loop never re-interns the field name.
    field_ids: Vec<u32>,
    /// Reverse map for the field-name interner: `field_names[id]` reads
    /// the original facet field name back. Field interning is performed
    /// only at construction (no per-doc inserts) so we don't keep the
    /// forward map around.
    field_names: Vec<String>,
    /// Interner: facet value (single path component) → `u32` id. Shared
    /// across all fields — distinct fields are kept separate by the
    /// `field_id` portion of the counter key.
    value_interner: HashMap<String, u32>,
    /// Reverse map for `value_interner`. Indexed by id.
    value_names: Vec<String>,
    /// Counter map for depth-1 paths. Key encodes
    /// `(field_id << 32) | value_id`.
    flat_counts: HashMap<u64, u64>,
    /// Counter map for depth ≥ 2 paths. Key is `[field_id, level0_id,
    /// level1_id, …]`.
    hier_counts: HashMap<Box<[u32]>, u64>,
    /// Per-field DocValues availability, parallel to `facet_fields`
    /// (Issue #597). Lazily populated on the first `collect_doc` call from
    /// `reader.has_doc_values(field)` — availability is doc-independent, so
    /// it is resolved once instead of re-probing the (lock-guarded) reader
    /// for every collected hit. Empty until the first call.
    field_has_dv: Vec<bool>,
}

/// Append the facet path components of a single field `value` to `out`.
///
/// Shared by the DocValues fast path and the stored-document fallback in
/// [`FacetCollector::collect_doc`] so both derive identical facet paths
/// (Issue #597). A `Text` value containing `/` is split into a hierarchical
/// path; scalar values stringify to a single component; any other variant
/// falls back to its `Debug` form, matching the original inline behaviour.
fn push_path_components(value: &crate::data::DataValue, out: &mut Vec<String>) {
    match value {
        crate::data::DataValue::Text(value) => {
            if value.contains('/') {
                for s in value.split('/') {
                    out.push(s.to_string());
                }
            } else {
                out.push(value.clone());
            }
        }
        crate::data::DataValue::Int64(v) => out.push(v.to_string()),
        crate::data::DataValue::Float64(v) => out.push(v.to_string()),
        crate::data::DataValue::Bool(v) => out.push(v.to_string()),
        _ => out.push(format!("{value:?}")),
    }
}

impl FacetCollector {
    /// Create a new facet collector.
    pub fn new(config: FacetConfig, facet_fields: Vec<String>) -> Self {
        let mut field_interner: HashMap<String, u32> = HashMap::new();
        let mut field_names: Vec<String> = Vec::new();
        let mut field_ids: Vec<u32> = Vec::with_capacity(facet_fields.len());
        for name in &facet_fields {
            // Pre-intern declared facet fields so `collect_doc` only ever
            // does an O(1) `Vec<u32>` index, not a `HashMap` probe per
            // document per field.
            if let Some(&id) = field_interner.get(name) {
                field_ids.push(id);
            } else {
                let id = field_names.len() as u32;
                field_names.push(name.clone());
                field_interner.insert(name.clone(), id);
                field_ids.push(id);
            }
        }

        FacetCollector {
            config,
            facet_fields,
            field_ids,
            field_names,
            value_interner: HashMap::new(),
            value_names: Vec::new(),
            flat_counts: HashMap::new(),
            hier_counts: HashMap::new(),
            field_has_dv: Vec::new(),
        }
    }

    /// Intern a value string and return its `u32` id, allocating a
    /// reverse-map entry on first sight. Subsequent calls for the same
    /// string are an O(1) `HashMap` probe.
    #[inline]
    fn intern_value(&mut self, value: &str) -> u32 {
        if let Some(&id) = self.value_interner.get(value) {
            return id;
        }
        let id = self.value_names.len() as u32;
        self.value_names.push(value.to_string());
        self.value_interner.insert(value.to_string(), id);
        id
    }

    /// Add a document to the facet counts.
    pub fn collect_doc(&mut self, doc_id: u64, reader: &dyn LexicalIndexReader) -> Result<()> {
        // Resolve per-field DocValues availability once (Issue #597). It is
        // doc-independent, so caching it here keeps `collect_doc` free of a
        // lock-guarded `has_doc_values` probe per hit. NOTE: `true` here
        // means "some segment has this column" (`InvertedIndexReader::
        // has_doc_values` is `any(...)` across segments) -- it does not
        // guarantee `get_doc_value` finds a value for THIS doc, since
        // segments can disagree on whether a field has a column (Issue
        // #1047: mixed old/new segments, or a `doc_values: false` field).
        if self.field_has_dv.len() != self.facet_fields.len() {
            self.field_has_dv = self
                .facet_fields
                .iter()
                .map(|f| reader.has_doc_values(f))
                .collect();
        }

        // Fetched lazily and cached for the rest of this call (#409: at
        // most once per `collect_doc`) the first time a field actually
        // needs it -- either because it has no DocValues column at all,
        // or because this document's segment misses despite the field
        // having DocValues elsewhere (#1047). Fields that always hit
        // DocValues never pay for this. Uses `document_fields` (only the
        // facet fields), not `document`, to avoid cloning every field of
        // a wide-schema document.
        let mut doc_fields: Option<
            Result<Option<std::collections::HashMap<String, crate::data::DataValue>>>,
        > = None;

        // Reusable scratch buffers — allocated once per call, cleared at
        // each field iteration. Avoids per-field `Vec` reallocations that
        // dominated the per-doc cost on flat fields where the HashMap
        // hot path is otherwise tight.
        let mut path_components: Vec<String> = Vec::new();
        let mut path_ids: Vec<u32> = Vec::new();

        for field_idx in 0..self.facet_fields.len() {
            let field_id = self.field_ids[field_idx];
            let has_dv = self.field_has_dv[field_idx];

            // Phase 1: extract path components. Borrows
            // `self.facet_fields[field_idx]` only until the end of this
            // block, so `intern_value` (which needs `&mut self`) is free to
            // run in phase 2 without a conflict.
            path_components.clear();
            {
                let field_name: &str = &self.facet_fields[field_idx];
                // DocValues fast path (#597). `FieldValue` is `DataValue`,
                // so the value maps to facet path components exactly as
                // the stored document would.
                let dv_hit = has_dv
                    .then(|| reader.get_doc_value(field_name, doc_id).ok().flatten())
                    .flatten();

                if let Some(value) = dv_hit {
                    push_path_components(&value, &mut path_components);
                } else {
                    // No DocValues column, or a miss despite `has_dv`
                    // (#1047) -- fall back to the stored document.
                    let result = doc_fields.get_or_insert_with(|| {
                        let field_refs: Vec<&str> =
                            self.facet_fields.iter().map(String::as_str).collect();
                        reader.document_fields(doc_id, &field_refs)
                    });
                    match result {
                        Ok(Some(fields)) => {
                            if let Some(val) = fields.get(field_name) {
                                push_path_components(val, &mut path_components);
                            }
                        }
                        Ok(None) => {
                            // Document not found — no facet contribution.
                        }
                        Err(_) => {
                            // Synthetic fallback preserved from the pre-#409
                            // implementation: 5 distinct values stratified
                            // by `doc_id`.
                            path_components.push(format!("value_{}", doc_id % 5));
                        }
                    }
                }
            }

            if path_components.is_empty() {
                continue;
            }

            // Phase 2: intern path components and bump counters.
            let depth = path_components.len();
            if depth == 1 {
                // Depth-1 fast path. Single hash on a `u64` key, no
                // boxed-slice allocation.
                let value_id = self.intern_value(&path_components[0]);
                let key = ((field_id as u64) << 32) | (value_id as u64);
                *self.flat_counts.entry(key).or_insert(0) += 1;
            } else {
                // Depth-N path. Build `[field_id, level0_id, …]` once
                // into the scratch `Vec<u32>`, then `pop()` the last id
                // at each step of the parent walk.
                path_ids.clear();
                path_ids.push(field_id);
                for component in &path_components {
                    let id = self.intern_value(component);
                    path_ids.push(id);
                }
                while path_ids.len() > 1 {
                    // Allocates a fresh `Box<[u32]>` per `entry()` —
                    // unavoidable with the std `HashMap::entry` API, but
                    // the box is `4 + 4*depth` bytes and hashes an
                    // integer slice rather than a string, so the
                    // per-step cost is ~30-40 ns vs the ~160-200 ns of
                    // cloning + hashing a `FacetPath`.
                    let key: Box<[u32]> = path_ids.as_slice().into();
                    *self.hier_counts.entry(key).or_insert(0) += 1;
                    path_ids.pop();
                }
            }
        }

        Ok(())
    }

    /// Finalize and return the collected facet counts.
    pub fn finalize(self) -> Result<FacetResults> {
        let mut field_facets: HashMap<String, Vec<FacetCount>> = HashMap::new();

        // Decode the depth-1 (`flat_counts`) tier. The 64-bit key splits
        // into `(field_id << 32) | value_id`; both ids index into the
        // collector's reverse-name maps so we can reconstruct the
        // original `FacetPath`.
        for (key, count) in &self.flat_counts {
            if *count < self.config.min_count {
                continue;
            }
            let field_id = (key >> 32) as u32;
            let value_id = (*key & 0xFFFF_FFFF) as u32;
            let field_name = &self.field_names[field_id as usize];
            let value = &self.value_names[value_id as usize];
            let facet_path = FacetPath::from_value(field_name.clone(), value.clone());
            field_facets
                .entry(field_name.clone())
                .or_default()
                .push(FacetCount::new(facet_path, *count));
        }

        // Decode the depth ≥ 2 (`hier_counts`) tier. Slot 0 is the
        // `field_id`; slots 1.. are interned path components in order.
        for (key, count) in &self.hier_counts {
            if *count < self.config.min_count {
                continue;
            }
            let field_id = key[0];
            let field_name = &self.field_names[field_id as usize];
            let path_components: Vec<String> = key[1..]
                .iter()
                .map(|&id| self.value_names[id as usize].clone())
                .collect();
            let facet_path = FacetPath::new(field_name.clone(), path_components);
            field_facets
                .entry(field_name.clone())
                .or_default()
                .push(FacetCount::new(facet_path, *count));
        }

        // Build hierarchical structure and sort
        for facet_counts in field_facets.values_mut() {
            FacetCollector::build_hierarchy_static(facet_counts);

            // Sort top-level facets
            if self.config.sort_by_count {
                facet_counts.sort_by_key(|c| std::cmp::Reverse(c.count));
            } else {
                facet_counts.sort_by(|a, b| a.path.path.first().cmp(&b.path.path.first()));
            }

            // Limit number of facets
            facet_counts.truncate(self.config.max_facets_per_field);

            // Sort children recursively
            for facet_count in facet_counts {
                facet_count.sort_children(self.config.sort_by_count);
            }
        }

        Ok(FacetResults { field_facets })
    }

    /// Build hierarchical structure from flat facet counts.
    fn build_hierarchy_static(facet_counts: &mut [FacetCount]) {
        // This is a simplified implementation
        // In a real implementation, we would:
        // 1. Identify parent-child relationships
        // 2. Move child facets under their parents
        // 3. Build the hierarchical tree structure

        // For now, just sort by depth
        facet_counts.sort_by_key(|c| c.path.depth());
    }
}

/// Results of facet collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetResults {
    /// Facet counts grouped by field.
    pub field_facets: HashMap<String, Vec<FacetCount>>,
}

impl FacetResults {
    /// Create empty facet results.
    pub fn empty() -> Self {
        FacetResults {
            field_facets: HashMap::new(),
        }
    }

    /// Get facet counts for a specific field.
    pub fn get_field_facets(&self, field_name: &str) -> Option<&Vec<FacetCount>> {
        self.field_facets.get(field_name)
    }

    /// Get the total number of unique facet values across all fields.
    pub fn total_facet_count(&self) -> usize {
        self.field_facets.values().map(|facets| facets.len()).sum()
    }

    /// Merge with another facet results.
    pub fn merge(&mut self, other: FacetResults) {
        for (field, other_facets) in other.field_facets {
            let field_facets = self.field_facets.entry(field).or_default();
            field_facets.extend(other_facets);
        }
    }
}

/// Facet filter for constraining search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetFilter {
    /// Facet paths that must match (AND condition).
    pub required_paths: Vec<FacetPath>,
    /// Facet paths that must not match (NOT condition).
    pub excluded_paths: Vec<FacetPath>,
}

impl FacetFilter {
    /// Create a new empty facet filter.
    pub fn new() -> Self {
        FacetFilter {
            required_paths: Vec::new(),
            excluded_paths: Vec::new(),
        }
    }

    /// Add a required facet path.
    pub fn require(&mut self, path: FacetPath) {
        self.required_paths.push(path);
    }

    /// Add an excluded facet path.
    pub fn exclude(&mut self, path: FacetPath) {
        self.excluded_paths.push(path);
    }

    /// Check if a document matches this filter.
    pub fn matches_doc(&self, doc_facets: &[FacetPath]) -> bool {
        // Check required paths
        for required_path in &self.required_paths {
            let matches = doc_facets.iter().any(|doc_facet| {
                // Check exact match or if doc_facet is a child of required_path
                doc_facet == required_path || required_path.is_parent_of(doc_facet)
            });

            if !matches {
                return false;
            }
        }

        // Check excluded paths
        for excluded_path in &self.excluded_paths {
            let matches = doc_facets.iter().any(|doc_facet| {
                // Check exact match or if doc_facet is a child of excluded_path
                doc_facet == excluded_path || excluded_path.is_parent_of(doc_facet)
            });

            if matches {
                return false;
            }
        }

        true
    }
}

impl Default for FacetFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Faceted search engine that combines full-text search with facet collection.
#[derive(Debug)]
pub struct FacetedSearchEngine {
    /// Configuration for facet collection.
    facet_config: FacetConfig,
}

impl FacetedSearchEngine {
    /// Create a new faceted search engine.
    pub fn new(facet_config: FacetConfig) -> Self {
        FacetedSearchEngine { facet_config }
    }

    /// Perform a faceted search.
    pub fn search<Q: Query>(
        &self,
        query: Q,
        facet_fields: Vec<String>,
        facet_filter: Option<FacetFilter>,
        reader: &dyn LexicalIndexReader,
    ) -> Result<FacetedSearchResults> {
        // Execute the base query
        let _matcher = query.matcher(reader)?;
        let _scorer = query.scorer(reader)?;

        let mut hits = Vec::new();
        let mut facet_collector = FacetCollector::new(self.facet_config.clone(), facet_fields);

        // Collect matching documents
        // Note: Simplified implementation as matcher.next() returns bool not Option<u32>
        for doc_id in 0..10u64 {
            // Placeholder logic
            let score = 1.0f32; // Placeholder score as scorer.score needs different arguments

            // Apply facet filter if provided
            if let Some(ref filter) = facet_filter {
                let doc_facets = self.get_document_facets(doc_id, reader)?;
                if !filter.matches_doc(&doc_facets) {
                    continue;
                }
            }

            hits.push(Hit {
                doc_id,
                score,
                fields: HashMap::new(), // TODO: Load actual field values
            });

            // Collect facets for this document
            facet_collector.collect_doc(doc_id, reader)?;
        }

        // Sort hits by score
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));

        // Finalize facet collection
        let facet_results = facet_collector.finalize()?;

        let total_hits = hits.len() as u64;
        Ok(FacetedSearchResults {
            hits,
            facets: facet_results,
            total_hits,
        })
    }

    /// Get facet paths for a document.
    fn get_document_facets(
        &self,
        _doc_id: u64,
        _reader: &dyn LexicalIndexReader,
    ) -> Result<Vec<FacetPath>> {
        // This is a simplified implementation
        // In a real implementation, we would:
        // 1. Load the document from the index
        // 2. Extract facet field values
        // 3. Parse them into FacetPath objects

        // For now, return empty list
        Ok(vec![])
    }
}

/// Results of a faceted search.
#[derive(Debug, Serialize, Deserialize)]
pub struct FacetedSearchResults {
    /// Search hits.
    pub hits: Vec<Hit>,
    /// Facet results.
    pub facets: FacetResults,
    /// Total number of hits.
    pub total_hits: u64,
}

impl FacetedSearchResults {
    /// Create empty faceted search results.
    pub fn empty() -> Self {
        FacetedSearchResults {
            hits: Vec::new(),
            facets: FacetResults::empty(),
            total_hits: 0,
        }
    }
}

/// Facet field definition for schema.
#[derive(Debug, Clone)]
pub struct FacetField {
    /// Field name.
    pub name: String,
    /// Whether this is a hierarchical facet.
    pub hierarchical: bool,
    /// Delimiter for hierarchical paths.
    pub delimiter: String,
    /// Whether to store facet values.
    pub stored: bool,
}

impl FacetField {
    /// Create a new facet field.
    pub fn new(name: String) -> Self {
        FacetField {
            name,
            hierarchical: false,
            delimiter: "/".to_string(),
            stored: true,
        }
    }

    /// Make this a hierarchical facet field.
    pub fn hierarchical(mut self, delimiter: String) -> Self {
        self.hierarchical = true;
        self.delimiter = delimiter;
        self
    }

    /// Set whether to store facet values.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }
}

/// Grouping functionality for search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupConfig {
    /// Field to group by
    pub group_field: String,
    /// Maximum number of groups to return
    pub max_groups: usize,
    /// Maximum number of documents per group
    pub max_docs_per_group: usize,
    /// Sort groups by count (true) or field value (false)
    pub sort_by_count: bool,
}

impl Default for GroupConfig {
    fn default() -> Self {
        GroupConfig {
            group_field: String::new(),
            max_groups: 100,
            max_docs_per_group: 10,
            sort_by_count: true,
        }
    }
}

/// A group of search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchGroup {
    /// The group key (field value)
    pub group_key: String,
    /// Documents in this group
    pub documents: Vec<Hit>,
    /// Total number of documents in this group (may be larger than documents.len())
    pub total_docs: u64,
    /// Representative document for this group (usually the highest scoring)
    pub representative_doc: Option<Hit>,
}

impl SearchGroup {
    /// Create a new search group.
    pub fn new(group_key: String) -> Self {
        SearchGroup {
            group_key,
            documents: Vec::new(),
            total_docs: 0,
            representative_doc: None,
        }
    }

    /// Add a document to this group.
    pub fn add_document(&mut self, hit: Hit) {
        // Set representative document (highest scoring)
        if self.representative_doc.is_none()
            || hit.score > self.representative_doc.as_ref().unwrap().score
        {
            self.representative_doc = Some(hit.clone());
        }

        self.documents.push(hit);
        self.total_docs += 1;
    }

    /// Sort documents in this group by score.
    pub fn sort_by_score(&mut self) {
        self.documents.sort_by(|a, b| b.score.total_cmp(&a.score));
    }

    /// Limit the number of documents in this group.
    pub fn limit_documents(&mut self, max_docs: usize) {
        if self.documents.len() > max_docs {
            self.documents.truncate(max_docs);
        }
    }
}

/// Results of grouped search.
#[derive(Debug, Serialize, Deserialize)]
pub struct GroupedSearchResults {
    /// Groups of search results
    pub groups: Vec<SearchGroup>,
    /// Total number of documents across all groups
    pub total_docs: u64,
    /// Total number of groups found
    pub total_groups: u64,
    /// Configuration used for grouping
    pub group_config: GroupConfig,
}

impl GroupedSearchResults {
    /// Create empty grouped search results.
    pub fn empty(group_config: GroupConfig) -> Self {
        GroupedSearchResults {
            groups: Vec::new(),
            total_docs: 0,
            total_groups: 0,
            group_config,
        }
    }

    /// Get the total number of unique groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Get a group by its key.
    pub fn get_group(&self, group_key: &str) -> Option<&SearchGroup> {
        self.groups.iter().find(|g| g.group_key == group_key)
    }
}

/// Grouped search engine that organizes results by field values.
#[derive(Debug)]
pub struct GroupedSearchEngine {
    /// Configuration for grouping
    group_config: GroupConfig,
}

impl GroupedSearchEngine {
    /// Create a new grouped search engine.
    pub fn new(group_config: GroupConfig) -> Self {
        GroupedSearchEngine { group_config }
    }

    /// Perform a grouped search.
    pub fn search<Q: Query>(
        &self,
        query: Q,
        reader: &dyn LexicalIndexReader,
    ) -> Result<GroupedSearchResults> {
        let _matcher = query.matcher(reader)?;
        let scorer = query.scorer(reader)?;

        let mut groups: HashMap<String, SearchGroup> = HashMap::new();
        let mut total_docs = 0u64;

        // Collect matching documents and group them
        // Note: This is a simplified implementation
        for doc_id in 0..100u64 {
            // Placeholder iteration
            let score = scorer.score(doc_id, 1.0, None);
            if score > 0.0 {
                // Get group key for this document
                let group_key = self.get_document_group_key(doc_id, reader)?;

                let hit = Hit {
                    doc_id,
                    score,
                    fields: self.load_document_fields(doc_id, reader)?,
                };

                // Add to appropriate group
                groups
                    .entry(group_key.clone())
                    .or_insert_with(|| SearchGroup::new(group_key))
                    .add_document(hit);

                total_docs += 1;
            }
        }

        // Convert groups to vector and sort
        let mut group_vec: Vec<SearchGroup> = groups.into_values().collect();

        // Sort groups
        if self.group_config.sort_by_count {
            group_vec.sort_by_key(|g| std::cmp::Reverse(g.total_docs));
        } else {
            group_vec.sort_by(|a, b| a.group_key.cmp(&b.group_key));
        }

        // Process each group
        for group in &mut group_vec {
            group.sort_by_score();
            group.limit_documents(self.group_config.max_docs_per_group);
        }

        // Limit number of groups
        let total_groups = group_vec.len() as u64;
        group_vec.truncate(self.group_config.max_groups);

        Ok(GroupedSearchResults {
            groups: group_vec,
            total_docs,
            total_groups,
            group_config: self.group_config.clone(),
        })
    }

    /// Get the group key for a document.
    fn get_document_group_key(
        &self,
        doc_id: u64,
        reader: &dyn LexicalIndexReader,
    ) -> Result<String> {
        // Try to get the document and extract the group field value
        match reader.document(doc_id) {
            Ok(Some(document)) => {
                if let Some(field_value) = document.get_field(&self.group_config.group_field) {
                    match field_value {
                        FieldValue::Text(value) => Ok(value.clone()),
                        FieldValue::Int64(value) => Ok(value.to_string()),
                        FieldValue::Float64(value) => Ok(value.to_string()),
                        FieldValue::Bool(value) => Ok(value.to_string()),
                        _ => Ok(format!("{field_value:?}")),
                    }
                } else {
                    Ok("unknown".to_string())
                }
            }
            _ => {
                // Fallback: create synthetic group keys
                Ok(format!("group_{}", doc_id % 5))
            }
        }
    }

    /// Load document fields for display.
    fn load_document_fields(
        &self,
        doc_id: u64,
        reader: &dyn LexicalIndexReader,
    ) -> Result<HashMap<String, String>> {
        let mut fields = HashMap::new();

        match reader.document(doc_id) {
            Ok(Some(document)) => {
                for (field_name, field_value) in &document.fields {
                    let value_str = match field_value {
                        crate::data::DataValue::Text(value) => value.clone(),
                        crate::data::DataValue::Int64(value) => value.to_string(),
                        crate::data::DataValue::Float64(value) => value.to_string(),
                        crate::data::DataValue::Bool(value) => value.to_string(),
                        _ => format!("{field_value:?}"),
                    };
                    fields.insert(field_name.clone(), value_str);
                }
            }
            _ => {
                // Fallback: add synthetic fields
                fields.insert("id".to_string(), doc_id.to_string());
                fields.insert("title".to_string(), format!("Document {doc_id}"));
            }
        }

        Ok(fields)
    }
}

/// Range faceting for numeric and date fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeFacet {
    /// Field name
    pub field: String,
    /// Range definitions
    pub ranges: Vec<FacetRange>,
}

/// A range definition for faceting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetRange {
    /// Range label
    pub label: String,
    /// Minimum value (inclusive)
    pub min: Option<f64>,
    /// Maximum value (exclusive)
    pub max: Option<f64>,
    /// Number of documents in this range
    pub count: u64,
}

impl FacetRange {
    /// Create a new facet range.
    pub fn new(label: String, min: Option<f64>, max: Option<f64>) -> Self {
        FacetRange {
            label,
            min,
            max,
            count: 0,
        }
    }

    /// Check if a value falls within this range.
    pub fn contains(&self, value: f64) -> bool {
        let min_ok = self.min.is_none_or(|min| value >= min);
        let max_ok = self.max.is_none_or(|max| value < max);
        min_ok && max_ok
    }
}

impl RangeFacet {
    /// Create a new range facet.
    pub fn new(field: String, ranges: Vec<FacetRange>) -> Self {
        RangeFacet { field, ranges }
    }

    /// Create numeric ranges automatically.
    pub fn numeric_ranges(field: String, min: f64, max: f64, count: usize) -> Self {
        let mut ranges = Vec::new();
        let step = (max - min) / count as f64;

        for i in 0..count {
            let range_min = min + (i as f64 * step);
            let range_max = if i == count - 1 {
                None
            } else {
                Some(min + ((i + 1) as f64 * step))
            };

            let label = if let Some(max_val) = range_max {
                format!("[{range_min:.1} TO {max_val:.1})")
            } else {
                format!("[{range_min:.1} TO *]")
            };

            ranges.push(FacetRange::new(label, Some(range_min), range_max));
        }

        RangeFacet::new(field, ranges)
    }

    /// Count documents in each range.
    pub fn count_ranges(&mut self, values: &[f64]) {
        // Reset counts
        for range in &mut self.ranges {
            range.count = 0;
        }

        // Count values in each range
        for &value in values {
            for range in &mut self.ranges {
                if range.contains(value) {
                    range.count += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Document;
    use crate::data::DataValue;
    use crate::lexical::index::structures::bkd_tree::BKDTree;
    use crate::lexical::reader::{FieldStats, PostingIterator, ReaderTermInfo};
    use std::any::Any;
    use std::collections::HashSet;
    use std::sync::Arc;

    /// Configurable reader for `FacetCollector::collect_doc` DocValues tests
    /// (Issue #597). `dv_fields` lists the fields exposed via DocValues;
    /// `panic_on_document` asserts the stored-document path is never taken.
    #[derive(Debug)]
    struct DvMockReader {
        docs: Vec<Document>,
        dv_fields: HashSet<String>,
        panic_on_document: bool,
        /// Doc ids for which `get_doc_value` reports `Ok(None)` even though
        /// the field is listed in `dv_fields` -- simulating a segment that
        /// has the DocValues column but lacks this particular document's
        /// value (Issue #1047 mixed-segment case).
        dv_miss_doc_ids: HashSet<u64>,
    }

    impl DvMockReader {
        fn new(docs: Vec<Document>, dv_fields: &[&str], panic_on_document: bool) -> Self {
            Self {
                docs,
                dv_fields: dv_fields.iter().map(|s| (*s).to_string()).collect(),
                panic_on_document,
                dv_miss_doc_ids: HashSet::new(),
            }
        }

        /// Like `new`, but `get_doc_value` misses for every doc id in
        /// `miss_doc_ids` while `has_doc_values` still reports `true`.
        fn with_dv_miss(docs: Vec<Document>, dv_fields: &[&str], miss_doc_ids: &[u64]) -> Self {
            Self {
                docs,
                dv_fields: dv_fields.iter().map(|s| (*s).to_string()).collect(),
                panic_on_document: false,
                dv_miss_doc_ids: miss_doc_ids.iter().copied().collect(),
            }
        }
    }

    impl LexicalIndexReader for DvMockReader {
        fn doc_count(&self) -> u64 {
            self.docs.len() as u64
        }
        fn max_doc(&self) -> u64 {
            self.docs.len() as u64
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(&self, doc_id: u64) -> Result<Option<Document>> {
            assert!(
                !self.panic_on_document,
                "document() must not be called when all facet fields have DocValues"
            );
            Ok(self.docs.get(doc_id as usize).cloned())
        }
        fn term_info(&self, _field: &str, _term: &str) -> Result<Option<ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(&self, _field: &str, _term: &str) -> Result<Option<Box<dyn PostingIterator>>> {
            Ok(None)
        }
        fn field_stats(&self, _field: &str) -> Result<Option<FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn get_bkd_tree(&self, _field: &str) -> Result<Option<Arc<dyn BKDTree>>> {
            Ok(None)
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn has_doc_values(&self, field: &str) -> bool {
            self.dv_fields.contains(field)
        }
        fn get_doc_value(&self, field: &str, doc_id: u64) -> Result<Option<FieldValue>> {
            if !self.dv_fields.contains(field) || self.dv_miss_doc_ids.contains(&doc_id) {
                return Ok(None);
            }
            Ok(self
                .docs
                .get(doc_id as usize)
                .and_then(|d| d.get(field).cloned()))
        }
    }

    /// Build a document from `(field, text-value)` pairs.
    fn text_doc(pairs: &[(&str, &str)]) -> Document {
        let mut b = Document::builder();
        for (f, v) in pairs {
            b = b.add_field(*f, DataValue::Text((*v).to_string()));
        }
        b.build()
    }

    /// Recursively flatten a field's facet counts into sorted `(path, count)`
    /// pairs, so two collection runs can be compared for exact equivalence.
    fn flatten(results: &FacetResults, field: &str) -> Vec<(Vec<String>, u64)> {
        fn walk(c: &FacetCount, out: &mut Vec<(Vec<String>, u64)>) {
            out.push((c.path.path.clone(), c.count));
            for ch in &c.children {
                walk(ch, out);
            }
        }
        let mut out = Vec::new();
        if let Some(counts) = results.get_field_facets(field) {
            for c in counts {
                walk(c, &mut out);
            }
        }
        out.sort();
        out
    }

    /// Run a full facet collection over `docs` and return the results.
    fn collect(docs: Vec<Document>, fields: &[&str], dv: &[&str], panic_doc: bool) -> FacetResults {
        let n = docs.len() as u64;
        let reader = DvMockReader::new(docs, dv, panic_doc);
        let mut collector = FacetCollector::new(
            FacetConfig::default(),
            fields.iter().map(|s| s.to_string()).collect(),
        );
        for doc_id in 0..n {
            collector
                .collect_doc(doc_id, &reader)
                .expect("collect_doc must not error");
        }
        collector.finalize().expect("finalize must not error")
    }

    #[test]
    fn facet_docvalues_counts_match_stored_doc() {
        // Identical corpus: flat field `brand` + hierarchical field `cat`.
        let docs = vec![
            text_doc(&[("brand", "apple"), ("cat", "a/x")]),
            text_doc(&[("brand", "apple"), ("cat", "a/y")]),
            text_doc(&[("brand", "dell"), ("cat", "b/x")]),
        ];
        // DocValues path: all fields have DocValues, so `document()` would
        // panic if the collector took the stored-doc path.
        let via_dv = collect(docs.clone(), &["brand", "cat"], &["brand", "cat"], true);
        // Stored-document fallback: no DocValues.
        let via_doc = collect(docs, &["brand", "cat"], &[], false);

        assert_eq!(flatten(&via_dv, "brand"), flatten(&via_doc, "brand"));
        assert_eq!(flatten(&via_dv, "cat"), flatten(&via_doc, "cat"));
        // Guard against "both empty": assert the concrete flat counts.
        assert_eq!(
            flatten(&via_dv, "brand"),
            vec![
                (vec!["apple".to_string()], 2),
                (vec!["dell".to_string()], 1),
            ]
        );
    }

    #[test]
    fn facet_docvalues_skips_document_fetch() {
        // Every facet field has DocValues → `document()` must never be called
        // (the mock panics if it is).
        let docs = vec![text_doc(&[("cat", "a")]), text_doc(&[("cat", "b")])];
        let results = collect(docs, &["cat"], &["cat"], true);
        assert_eq!(
            flatten(&results, "cat"),
            vec![(vec!["a".to_string()], 1), (vec!["b".to_string()], 1)]
        );
    }

    #[test]
    fn facet_falls_back_to_document_without_docvalues() {
        let docs = vec![text_doc(&[("cat", "a")]), text_doc(&[("cat", "a")])];
        let results = collect(docs, &["cat"], &[], false);
        assert_eq!(flatten(&results, "cat"), vec![(vec!["a".to_string()], 2)]);
    }

    #[test]
    fn facet_mixed_docvalues_and_stored() {
        // `brand` has DocValues; `cat` does not → `document()` is fetched for
        // `cat` while `brand` is read from DocValues. Both must be counted.
        let docs = vec![
            text_doc(&[("brand", "apple"), ("cat", "a")]),
            text_doc(&[("brand", "dell"), ("cat", "a")]),
        ];
        let results = collect(docs, &["brand", "cat"], &["brand"], false);
        assert_eq!(
            flatten(&results, "brand"),
            vec![
                (vec!["apple".to_string()], 1),
                (vec!["dell".to_string()], 1),
            ]
        );
        assert_eq!(flatten(&results, "cat"), vec![(vec!["a".to_string()], 2)]);
    }

    #[test]
    fn facet_falls_back_when_has_dv_is_true_but_this_docs_value_is_missing() {
        // Issue #1047 regression: `has_doc_values("cat")` reports `true`
        // (another segment has the column), but THIS doc's DocValues lookup
        // misses (`Ok(None)`). Before the fix, `collect_doc` treated any
        // non-`Ok(Some(_))` DV read under `has_dv == true` as "no
        // contribution" and never fell back to the stored document, so
        // doc 1's `cat` facet would be silently dropped.
        let docs = vec![
            text_doc(&[("cat", "a")]), // doc_id 0: DV hit
            text_doc(&[("cat", "b")]), // doc_id 1: DV miss -> must fall back
        ];
        let reader = DvMockReader::with_dv_miss(docs, &["cat"], &[1]);
        let mut collector = FacetCollector::new(FacetConfig::default(), vec!["cat".to_string()]);
        collector
            .collect_doc(0, &reader)
            .expect("collect_doc must not error");
        collector
            .collect_doc(1, &reader)
            .expect("collect_doc must not error");
        let results = collector.finalize().expect("finalize must not error");
        assert_eq!(
            flatten(&results, "cat"),
            vec![(vec!["a".to_string()], 1), (vec!["b".to_string()], 1)],
            "a DocValues miss with has_dv=true must still fall back to the stored document"
        );
    }

    #[test]
    fn test_facet_path_creation() {
        let path = FacetPath::new(
            "category".to_string(),
            vec!["Electronics".to_string(), "Computers".to_string()],
        );
        assert_eq!(path.field, "category");
        assert_eq!(path.depth(), 2);

        let single_path = FacetPath::from_value("brand".to_string(), "Apple".to_string());
        assert_eq!(single_path.depth(), 1);
        assert_eq!(single_path.path[0], "Apple");

        let delimited_path =
            FacetPath::from_delimited("tags".to_string(), "tech/computers/laptops", "/");
        assert_eq!(delimited_path.depth(), 3);
        assert_eq!(delimited_path.path, vec!["tech", "computers", "laptops"]);
    }

    #[test]
    fn test_facet_path_hierarchy() {
        let parent = FacetPath::new("category".to_string(), vec!["Electronics".to_string()]);
        let child = FacetPath::new(
            "category".to_string(),
            vec!["Electronics".to_string(), "Computers".to_string()],
        );

        assert!(parent.is_parent_of(&child));
        assert!(!child.is_parent_of(&parent));

        let grandchild = child.child("Laptops".to_string());
        assert_eq!(grandchild.depth(), 3);
        assert!(child.is_parent_of(&grandchild));
        assert!(parent.is_parent_of(&grandchild));

        let child_parent = child.parent().unwrap();
        assert_eq!(child_parent, parent);
    }

    #[test]
    fn test_facet_count() {
        let path = FacetPath::from_value("category".to_string(), "Electronics".to_string());
        let mut facet_count = FacetCount::new(path, 42);

        assert_eq!(facet_count.count, 42);
        assert_eq!(facet_count.children.len(), 0);

        let child_path = FacetPath::from_value("category".to_string(), "Computers".to_string());
        let child_count = FacetCount::new(child_path, 15);
        facet_count.add_child(child_count);

        assert_eq!(facet_count.children.len(), 1);
        assert_eq!(facet_count.children[0].count, 15);
    }

    #[test]
    fn test_facet_filter() {
        let mut filter = FacetFilter::new();
        filter.require(FacetPath::from_value(
            "category".to_string(),
            "Electronics".to_string(),
        ));
        filter.exclude(FacetPath::from_value(
            "brand".to_string(),
            "Acme".to_string(),
        ));

        // Test matching document
        let doc_facets = vec![
            FacetPath::from_value("category".to_string(), "Electronics".to_string()),
            FacetPath::from_value("brand".to_string(), "Apple".to_string()),
        ];
        assert!(filter.matches_doc(&doc_facets));

        // Test non-matching document (missing required facet)
        let doc_facets2 = vec![FacetPath::from_value(
            "category".to_string(),
            "Books".to_string(),
        )];
        assert!(!filter.matches_doc(&doc_facets2));

        // Test non-matching document (has excluded facet)
        let doc_facets3 = vec![
            FacetPath::from_value("category".to_string(), "Electronics".to_string()),
            FacetPath::from_value("brand".to_string(), "Acme".to_string()),
        ];
        assert!(!filter.matches_doc(&doc_facets3));
    }

    #[test]
    fn test_facet_config() {
        let config = FacetConfig::default();
        assert_eq!(config.max_facets_per_field, 100);
        assert_eq!(config.max_depth, 10);
        assert_eq!(config.min_count, 1);
        assert!(!config.include_zero_counts);
        assert!(config.sort_by_count);
    }

    #[test]
    fn test_facet_results() {
        let mut results = FacetResults::empty();
        assert_eq!(results.total_facet_count(), 0);

        let path = FacetPath::from_value("category".to_string(), "Electronics".to_string());
        let facet_count = FacetCount::new(path, 42);
        results
            .field_facets
            .insert("category".to_string(), vec![facet_count]);

        assert_eq!(results.total_facet_count(), 1);
        assert!(results.get_field_facets("category").is_some());
        assert!(results.get_field_facets("nonexistent").is_none());
    }

    #[test]
    fn test_group_config() {
        let config = GroupConfig::default();
        assert!(config.group_field.is_empty());
        assert_eq!(config.max_groups, 100);
        assert_eq!(config.max_docs_per_group, 10);
        assert!(config.sort_by_count);
    }

    #[test]
    fn test_search_group() {
        let mut group = SearchGroup::new("Electronics".to_string());
        assert_eq!(group.group_key, "Electronics");
        assert_eq!(group.total_docs, 0);
        assert!(group.representative_doc.is_none());

        let hit1 = Hit {
            doc_id: 1,
            score: 0.8,
            fields: HashMap::new(),
        };
        let hit2 = Hit {
            doc_id: 2,
            score: 0.9,
            fields: HashMap::new(),
        };

        group.add_document(hit1);
        group.add_document(hit2);

        assert_eq!(group.total_docs, 2);
        assert_eq!(group.documents.len(), 2);
        assert_eq!(group.representative_doc.as_ref().unwrap().score, 0.9);

        group.sort_by_score();
        assert_eq!(group.documents[0].score, 0.9);
        assert_eq!(group.documents[1].score, 0.8);

        group.limit_documents(1);
        assert_eq!(group.documents.len(), 1);
    }

    #[test]
    fn test_grouped_search_results() {
        let config = GroupConfig {
            group_field: "category".to_string(),
            max_groups: 10,
            max_docs_per_group: 5,
            sort_by_count: true,
        };

        let results = GroupedSearchResults::empty(config.clone());
        assert_eq!(results.group_count(), 0);
        assert_eq!(results.total_docs, 0);
        assert_eq!(results.total_groups, 0);
        assert!(results.get_group("Electronics").is_none());
    }

    #[test]
    fn test_facet_range() {
        let range = FacetRange::new("[0.0 TO 10.0)".to_string(), Some(0.0), Some(10.0));

        assert!(range.contains(5.0));
        assert!(range.contains(0.0)); // Inclusive minimum
        assert!(!range.contains(10.0)); // Exclusive maximum
        assert!(!range.contains(-1.0));
        assert!(!range.contains(15.0));
    }

    #[test]
    fn test_range_facet_creation() {
        let range_facet = RangeFacet::numeric_ranges("price".to_string(), 0.0, 100.0, 5);

        assert_eq!(range_facet.field, "price");
        assert_eq!(range_facet.ranges.len(), 5);

        // Check first range
        assert_eq!(range_facet.ranges[0].min, Some(0.0));
        assert_eq!(range_facet.ranges[0].max, Some(20.0));

        // Check last range
        assert_eq!(range_facet.ranges[4].min, Some(80.0));
        assert_eq!(range_facet.ranges[4].max, None); // Open-ended
    }

    #[test]
    fn test_range_facet_counting() {
        let mut range_facet = RangeFacet::numeric_ranges("score".to_string(), 0.0, 10.0, 2);
        let values = vec![1.0, 3.0, 7.0, 9.0, 15.0]; // 15.0 should not count (out of range)

        range_facet.count_ranges(&values);

        // First range [0.0 TO 5.0): should count 1.0, 3.0
        assert_eq!(range_facet.ranges[0].count, 2);

        // Second range [5.0 TO *]: should count 7.0, 9.0, 15.0
        assert_eq!(range_facet.ranges[1].count, 3);
    }
}
