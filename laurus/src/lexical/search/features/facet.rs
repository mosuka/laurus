//! Faceted search functionality for categorizing and filtering search results.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Result;
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
    ///
    /// Empty components are dropped, exactly as [`FacetCollector`] does for
    /// `/`-delimited values, so `"/a//b"` gives `["a", "b"]` and the path
    /// matches the one the collector counts. A string with no non-empty
    /// component gives the empty path: the root of `field`, which
    /// [`is_parent_of`](Self::is_parent_of) every other path of that field.
    pub fn from_delimited(field: String, path_str: &str, delimiter: &str) -> Self {
        let path = path_str
            .split(delimiter)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
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
    /// Number of documents whose value is this path or lies under it. A
    /// document is counted once per path even when a multi-valued field
    /// repeats the element or several elements share a hierarchical
    /// ancestor (Issue #1187), so no child outnumbers its parent.
    pub count: u64,
    /// Facets one level deeper for hierarchical drill-down (`["a", "b"]`
    /// under `["a"]`), filtered, sorted and truncated like the top level
    /// (Issue #1192).
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

    /// Sort children by count (descending, ties by name) or name
    /// (ascending), the order [`FacetCollector::finalize`] produces.
    pub fn sort_children(&mut self, by_count: bool) {
        if by_count {
            self.children.sort_by(|a, b| {
                b.count
                    .cmp(&a.count)
                    .then_with(|| a.path.path.last().cmp(&b.path.path.last()))
            });
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
    /// Maximum number of facet values kept per level: at the top level of
    /// each field and, separately, among the children of each node. It is
    /// applied after sorting, so a level keeps its best entries.
    pub max_facets_per_field: usize,
    /// Maximum depth of a hierarchical facet. Each path is cut to its first
    /// `max_depth` components while collecting, so deeper levels are never
    /// counted: with `2`, `a/b/c` counts `a` and `a/b`. `0` counts nothing
    /// and `usize::MAX` keeps every level.
    pub max_depth: usize,
    /// Minimum document count for a facet value to be returned. A value
    /// below it is dropped together with its children, which never count
    /// more. Values no collected document has are never returned, so `0`
    /// behaves like `1`.
    pub min_count: u64,
    /// Sort each level by count, descending, with ties broken by label
    /// (`true`), or by label alone (`false`). Labels compare as strings.
    pub sort_by_count: bool,
}

impl Default for FacetConfig {
    fn default() -> Self {
        FacetConfig {
            max_facets_per_field: 100,
            max_depth: 10,
            min_count: 1,
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
/// - `flat_counts: HashMap<u64, Slot>` — the root (first component) of
///   every path, whether the path is a flat value `a` or a hierarchical
///   `a/b`, so both share one counter for `a` (Issue #1192). The 64-bit
///   key packs `(field_id << 32) | value_id`, so each increment hashes a
///   single `u64` instead of a `String + Vec<String>` pair.
/// - `hier_counts: HashMap<Box<[u32]>, Slot>` — the depth ≥ 2 prefixes of
///   hierarchical paths. The boxed slice stores `[field_id, level0_id,
///   level1_id, …]`; the parent walk shrinks a local `Vec<u32>` by `pop()`
///   at each level and only pays a `Box<[u32]>` allocation when the entry
///   doesn't yet exist.
///
/// [`finalize`](Self::finalize) nests both tiers into one tree per field.
///
/// A multi-valued field value expands to one path per element (Issue
/// #1187), so a single document can reach the same key more than once —
/// through a repeated element (`["rust", "rust"]`) or through the shared
/// ancestor of two hierarchical elements (`["a/b", "a/c"]` → `a`). Every
/// [`Slot`] remembers the generation (`doc_gen`, advanced once per
/// `collect_doc`) of its last increment and refuses a second increment in
/// the same generation, so counts stay per document — Lucene's
/// `SortedSetDocValuesFacetCounts` semantics — at the cost of one compare
/// on the scalar hot path.
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
    /// Counter map for the root of every path. Key is
    /// [`flat_key`]`(field_id, value_id)`.
    flat_counts: HashMap<u64, Slot>,
    /// Counter map for depth ≥ 2 prefixes. Key is `[field_id, level0_id,
    /// level1_id, …]`.
    hier_counts: HashMap<Box<[u32]>, Slot>,
    /// Per-field DocValues availability, parallel to `facet_fields`
    /// (Issue #597). Lazily populated on the first `collect_doc` call from
    /// `reader.has_doc_values(field)` — availability is doc-independent, so
    /// it is resolved once instead of re-probing the (lock-guarded) reader
    /// for every collected hit. Empty until the first call.
    field_has_dv: Vec<bool>,
    /// Generation counter, advanced at the start of every `collect_doc`
    /// (Issue #1187). `Slot::generation == doc_gen` means "already counted
    /// for the document being collected"; `0` is never a live generation.
    doc_gen: u64,
}

/// Counter slot for one facet key: the document count and the generation
/// of the last increment (Issue #1187).
#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    count: u64,
    /// `FacetCollector::doc_gen` at the last increment; `0` = never.
    generation: u64,
}

impl Slot {
    /// Increment at most once per generation. Returns `false` when the key
    /// was already counted for the current document.
    #[inline]
    fn bump(&mut self, generation: u64) -> bool {
        if self.generation == generation {
            return false;
        }
        self.generation = generation;
        self.count += 1;
        true
    }
}

/// Reusable scratch holding the facet paths derived from one field value
/// (Issue #1187): `components` concatenates every path's components and
/// `ends[i]` is the end index of path `i`. A scalar yields one path, an
/// array one path per element. `clear` keeps the allocated capacity so the
/// per-field loop in [`FacetCollector::collect_doc`] stays allocation-free
/// once warmed up.
#[derive(Debug, Default)]
struct FacetPaths {
    components: Vec<String>,
    ends: Vec<usize>,
}

impl FacetPaths {
    fn clear(&mut self) {
        self.components.clear();
        self.ends.clear();
    }

    fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Start a new path and let `fill` push its components. A path that
    /// ends up with no components is discarded.
    fn push_path(&mut self, fill: impl FnOnce(&mut Vec<String>)) {
        let start = self.components.len();
        fill(&mut self.components);
        if self.components.len() > start {
            self.ends.push(self.components.len());
        }
    }

    /// The paths in insertion order, each as its component slice.
    fn paths(&self) -> impl Iterator<Item = &[String]> + '_ {
        let mut start = 0;
        self.ends.iter().map(move |&end| {
            let path = &self.components[start..end];
            start = end;
            path
        })
    }
}

/// Push the facet path of one text value: a `/`-delimited string becomes a
/// hierarchical path, anything else a single component. Empty components
/// are dropped (Issue #1192) — Lucene rejects them outright — so `"/a/b"`
/// and `"a//b"` both give `a/b`, `"a/"` gives `a`, and `""` or `"/"` give
/// no path at all.
fn push_text_path(text: &str, out: &mut FacetPaths) {
    out.push_path(|components| {
        if text.contains('/') {
            components.extend(
                text.split('/')
                    .filter(|c| !c.is_empty())
                    .map(str::to_string),
            );
        } else if !text.is_empty() {
            // The common flat value: one push, no split iterator.
            components.push(text.to_string());
        }
    });
}

/// Facet label of a float: always carries a fraction (`2.0`, `2.5`) so a
/// float facet never shares a label with an integer one. Non-finite values
/// keep their `Display` form (`NaN`, `inf`).
fn format_facet_float(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

/// Facet label of a datetime: RFC 3339 in UTC (`+00:00`), floored to
/// microseconds. DocValues archive datetimes at microsecond precision
/// (`MicroSeconds` in `data.rs`) while the stored document keeps the full
/// precision, so without the floor one instant could get two labels
/// depending on which copy the collector read (Issue #1187).
fn format_facet_datetime(value: &DateTime<Utc>) -> String {
    DateTime::<Utc>::from_timestamp_micros(value.timestamp_micros())
        .unwrap_or(*value)
        .to_rfc3339()
}

/// Append the facet paths of one field `value` to `out` (Issue #1187).
///
/// Shared by the DocValues fast path and the stored-document fallback in
/// [`FacetCollector::collect_doc`] so both derive identical facet paths
/// (Issue #597). A scalar yields one path — a `Text` containing `/` is
/// split into a hierarchical path — and an array yields one path per
/// element, rendered exactly like the scalar of the same type. `Null`, geo
/// points, vectors and bytes are not facetable and yield nothing. The match
/// is exhaustive on purpose: a new `DataValue` variant has to decide here
/// whether, and how, it facets.
fn push_facet_paths(value: &crate::data::DataValue, out: &mut FacetPaths) {
    use crate::data::DataValue as V;

    fn scalar(out: &mut FacetPaths, label: String) {
        out.push_path(|components| components.push(label));
    }

    match value {
        V::Text(text) => push_text_path(text, out),
        V::Int64(v) => scalar(out, v.to_string()),
        V::Float64(v) => scalar(out, format_facet_float(*v)),
        V::Bool(v) => scalar(out, v.to_string()),
        V::DateTime(dt) => scalar(out, format_facet_datetime(dt)),
        V::TextArray(items) => items.iter().for_each(|text| push_text_path(text, out)),
        V::Int64Array(items) => items.iter().for_each(|v| scalar(out, v.to_string())),
        V::Float64Array(items) => items
            .iter()
            .for_each(|v| scalar(out, format_facet_float(*v))),
        V::BoolArray(items) => items.iter().for_each(|v| scalar(out, v.to_string())),
        V::DateTimeArray(items) => items
            .iter()
            .for_each(|dt| scalar(out, format_facet_datetime(dt))),
        V::Null
        | V::Geo(_)
        | V::GeoEcef(_)
        | V::GeoArray(_)
        | V::GeoEcefArray(_)
        | V::Vector(_)
        | V::Bytes(_, _) => {}
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
            doc_gen: 0,
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

    /// Add a document to the facet counts. Every facet key is incremented at
    /// most once per call, however many paths the document's values expand
    /// to (Issue #1187).
    ///
    /// # Errors
    ///
    /// Returns the reader's error when a facet field has to be read from
    /// the stored document and that read fails. Fields collected earlier in
    /// the same call are already counted by then, so the collector holds
    /// partial counts afterwards and should be discarded.
    pub fn collect_doc(&mut self, doc_id: u64, reader: &dyn LexicalIndexReader) -> Result<()> {
        // One generation per document: `Slot::bump` refuses a second
        // increment of the same key in the same generation.
        self.doc_gen += 1;
        let generation = self.doc_gen;

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
        let mut doc_fields: Option<Option<HashMap<String, crate::data::DataValue>>> = None;

        // Reusable scratch buffers — allocated once per call, cleared at
        // each field iteration. Avoids per-field `Vec` reallocations that
        // dominated the per-doc cost on flat fields where the HashMap
        // hot path is otherwise tight.
        let mut paths = FacetPaths::default();
        let mut path_ids: Vec<u32> = Vec::new();

        for field_idx in 0..self.facet_fields.len() {
            let field_id = self.field_ids[field_idx];
            let has_dv = self.field_has_dv[field_idx];

            // Phase 1: derive the facet paths of this field's value.
            // Borrows `self.facet_fields[field_idx]` only until the end of
            // this block, so `intern_value` (which needs `&mut self`) is
            // free to run in phase 2 without a conflict.
            paths.clear();
            {
                let field_name: &str = &self.facet_fields[field_idx];
                // DocValues fast path (#597). `FieldValue` is `DataValue`,
                // so the value maps to facet paths exactly as the stored
                // document would. A hit that yields no path (a geo value,
                // an empty array, `Null`) is final: the stored document
                // holds the same value, so there is nothing to fall back to.
                let dv_hit = has_dv
                    .then(|| reader.get_doc_value(field_name, doc_id).ok().flatten())
                    .flatten();

                if let Some(value) = dv_hit {
                    push_facet_paths(&value, &mut paths);
                } else {
                    // No DocValues column, or a miss despite `has_dv`
                    // (#1047) -- fall back to the stored document. A read
                    // error is returned rather than counted: this document's
                    // facet values are unknown.
                    if doc_fields.is_none() {
                        let field_refs: Vec<&str> =
                            self.facet_fields.iter().map(String::as_str).collect();
                        doc_fields = Some(reader.document_fields(doc_id, &field_refs)?);
                    }
                    // `None` = document not found: no facet contribution.
                    if let Some(Some(fields)) = &doc_fields
                        && let Some(val) = fields.get(field_name)
                    {
                        push_facet_paths(val, &mut paths);
                    }
                }
            }

            if paths.is_empty() {
                continue;
            }

            // Phase 2: intern path components and bump counters, once per
            // key per document (Issue #1187). A path is cut to its first
            // `max_depth` components, and its root always lands in
            // `flat_counts` — the counter a flat value with the same label
            // uses — so a label reached at depth 1 has a single counter per
            // field however many paths reach it (Issue #1192).
            for path in paths.paths() {
                let depth = path.len().min(self.config.max_depth);
                if depth == 0 {
                    continue;
                }
                let root_id = self.intern_value(&path[0]);
                if depth > 1 {
                    // Build `[field_id, root_id, level1_id, …]` once into
                    // the scratch `Vec<u32>` for the parent walk.
                    path_ids.clear();
                    path_ids.push(field_id);
                    path_ids.push(root_id);
                    for component in &path[1..depth] {
                        let id = self.intern_value(component);
                        path_ids.push(id);
                    }
                    if !self.bump_hier_prefixes(&mut path_ids, generation) {
                        continue;
                    }
                }
                self.flat_counts
                    .entry(flat_key(field_id, root_id))
                    .or_default()
                    .bump(generation);
            }
        }

        Ok(())
    }

    /// Bump the depth ≥ 2 prefixes of `path_ids` (`[field_id, root_id, …]`)
    /// in `hier_counts`, leaf to root, popping one id per step. Returns
    /// `false` when a prefix was already counted for this document: the
    /// earlier walk that counted it also counted every shorter prefix, root
    /// included, so the caller must not bump the root either.
    fn bump_hier_prefixes(&mut self, path_ids: &mut Vec<u32>, generation: u64) -> bool {
        while path_ids.len() > 2 {
            // Probe with the borrowed slice first so a hit allocates
            // nothing; only a key seen for the first time pays for its
            // `Box<[u32]>`.
            let fresh = match self.hier_counts.get_mut(path_ids.as_slice()) {
                Some(slot) => slot.bump(generation),
                None => self
                    .hier_counts
                    .entry(path_ids.as_slice().into())
                    .or_default()
                    .bump(generation),
            };
            if !fresh {
                return false;
            }
            path_ids.pop();
        }
        true
    }

    /// Finalize and return the collected facet counts, one tree per field.
    ///
    /// Every level — the top level and the children of each node — is
    /// filtered by `min_count`, sorted and cut to `max_facets_per_field` on
    /// its own, so ancestors never use up their descendants' budget (Issue
    /// #1192). A field with nothing left is absent from the results.
    pub fn finalize(self) -> Result<FacetResults> {
        // Assemble one trie per field from both tiers. The counters are
        // prefix-closed — whenever a path is counted for a document, so is
        // each of its ancestors — so every node receives a count of its own
        // below and no child outnumbers its parent.
        let mut roots: Vec<HashMap<u32, TrieNode>> = (0..self.field_names.len())
            .map(|_| HashMap::new())
            .collect();
        for (&key, slot) in &self.flat_counts {
            // Unpack `flat_key`: `(field_id << 32) | root_id`.
            let (field_id, root_id) = ((key >> 32) as usize, key as u32);
            roots[field_id].entry(root_id).or_default().count = slot.count;
        }
        for (key, slot) in &self.hier_counts {
            // `[field_id, root_id, level1_id, …]`.
            let mut node = roots[key[0] as usize].entry(key[1]).or_default();
            for &id in &key[2..] {
                node = node.children.entry(id).or_default();
            }
            node.count = slot.count;
        }

        let mut field_facets = HashMap::new();
        for (field_id, level) in roots.into_iter().enumerate() {
            let field = &self.field_names[field_id];
            let facets = self.build_level(level, field, &[]);
            if !facets.is_empty() {
                field_facets.insert(field.clone(), facets);
            }
        }
        Ok(FacetResults { field_facets })
    }

    /// Turn one trie level into `FacetCount`s: drop the nodes under
    /// `min_count` (with their subtrees, which count no more), sort, keep
    /// the first `max_facets_per_field`, and recurse into the kept nodes'
    /// children. `parent` holds the labels leading to this level.
    fn build_level(
        &self,
        level: HashMap<u32, TrieNode>,
        field: &str,
        parent: &[String],
    ) -> Vec<FacetCount> {
        let mut nodes: Vec<(&str, TrieNode)> = level
            .into_iter()
            .filter(|(_, node)| node.count >= self.config.min_count)
            .map(|(id, node)| (self.value_names[id as usize].as_str(), node))
            .collect();
        // Sibling labels are distinct, so either order is total and the
        // truncation below is deterministic. Count ties go to the smaller
        // label, as in Lucene (whose ordinals follow label order) and
        // Tantivy; interned ids would follow first-seen order instead.
        if self.config.sort_by_count {
            nodes.sort_unstable_by(|(a_label, a), (b_label, b)| {
                b.count.cmp(&a.count).then_with(|| a_label.cmp(b_label))
            });
        } else {
            nodes.sort_unstable_by_key(|&(label, _)| label);
        }
        nodes.truncate(self.config.max_facets_per_field);

        nodes
            .into_iter()
            .map(|(label, node)| {
                debug_assert!(node.count > 0, "facet trie node `{label}` has no count");
                let mut path = parent.to_vec();
                path.push(label.to_string());
                let children = self.build_level(node.children, field, &path);
                FacetCount {
                    path: FacetPath::new(field.to_string(), path),
                    count: node.count,
                    children,
                }
            })
            .collect()
    }
}

/// `flat_counts` key of a root label: `(field_id << 32) | value_id`, a
/// single `u64` to hash instead of a `String + Vec<String>` pair.
#[inline]
fn flat_key(field_id: u32, value_id: u32) -> u64 {
    (u64::from(field_id) << 32) | u64::from(value_id)
}

/// Node of the per-field trie [`FacetCollector::finalize`] assembles from
/// the counters, keyed by interned label id.
#[derive(Debug, Default)]
struct TrieNode {
    count: u64,
    children: HashMap<u32, TrieNode>,
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

    /// Get the total number of unique facet values across all fields,
    /// nested hierarchical values included.
    pub fn total_facet_count(&self) -> usize {
        fn count(facets: &[FacetCount]) -> usize {
            facets.iter().map(|f| 1 + count(&f.children)).sum()
        }
        self.field_facets.values().map(|facets| count(facets)).sum()
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
    use crate::data::{DataValue, GeoEcefPoint, GeoPoint};
    use crate::lexical::core::field::FieldValue;
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
        /// `document()` returns an error, simulating a failed stored-field
        /// read.
        fail_document: bool,
    }

    impl DvMockReader {
        fn new(docs: Vec<Document>, dv_fields: &[&str], panic_on_document: bool) -> Self {
            Self {
                docs,
                dv_fields: dv_fields.iter().map(|s| (*s).to_string()).collect(),
                panic_on_document,
                dv_miss_doc_ids: HashSet::new(),
                fail_document: false,
            }
        }

        /// Like `new`, but `get_doc_value` misses for every doc id in
        /// `miss_doc_ids` while `has_doc_values` still reports `true`.
        fn with_dv_miss(docs: Vec<Document>, dv_fields: &[&str], miss_doc_ids: &[u64]) -> Self {
            Self {
                dv_miss_doc_ids: miss_doc_ids.iter().copied().collect(),
                ..Self::new(docs, dv_fields, false)
            }
        }

        /// Like `new` with no DocValues, but every stored-document read fails.
        fn with_failing_document(docs: Vec<Document>) -> Self {
            Self {
                fail_document: true,
                ..Self::new(docs, &[], false)
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
            if self.fail_document {
                return Err(crate::error::LaurusError::storage(
                    "simulated stored-field read failure",
                ));
            }
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

    /// Build a document from `(field, value)` pairs of any `DataValue`
    /// (Issue #1187 array tests).
    fn doc(pairs: &[(&str, DataValue)]) -> Document {
        let mut b = Document::builder();
        for (f, v) in pairs {
            b = b.add_field(*f, v.clone());
        }
        b.build()
    }

    /// `DataValue::TextArray` from string slices.
    fn texts(items: &[&str]) -> DataValue {
        DataValue::TextArray(items.iter().map(|s| (*s).to_string()).collect())
    }

    /// Facet path from string slices, for `flatten` comparisons.
    fn p(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_string()).collect()
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

    /// Render a field's facet tree in result order as `label:count`,
    /// siblings space-separated and children in brackets, e.g.
    /// `"a:2[b:1 c:1] d:1"`. Unlike `flatten`, it pins the nesting and the
    /// order of every level.
    fn tree(results: &FacetResults, field: &str) -> String {
        fn render(facets: &[FacetCount]) -> String {
            facets
                .iter()
                .map(|c| {
                    let label = c.path.path.last().map(String::as_str).unwrap_or("");
                    if c.children.is_empty() {
                        format!("{label}:{}", c.count)
                    } else {
                        format!("{label}:{}[{}]", c.count, render(&c.children))
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        results
            .get_field_facets(field)
            .map(|facets| render(facets))
            .unwrap_or_default()
    }

    /// Run a full facet collection over `docs` with `config` and return
    /// the results.
    fn collect_with(
        config: FacetConfig,
        docs: Vec<Document>,
        fields: &[&str],
        dv: &[&str],
        panic_doc: bool,
    ) -> FacetResults {
        let n = docs.len() as u64;
        let reader = DvMockReader::new(docs, dv, panic_doc);
        let mut collector =
            FacetCollector::new(config, fields.iter().map(|s| s.to_string()).collect());
        for doc_id in 0..n {
            collector
                .collect_doc(doc_id, &reader)
                .expect("collect_doc must not error");
        }
        collector.finalize().expect("finalize must not error")
    }

    /// Run a full facet collection over `docs` and return the results.
    fn collect(docs: Vec<Document>, fields: &[&str], dv: &[&str], panic_doc: bool) -> FacetResults {
        collect_with(FacetConfig::default(), docs, fields, dv, panic_doc)
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

    // ---- Multi-valued (array) values, Issue #1187 -----------------------

    #[test]
    fn facet_expands_text_array_elements() {
        let docs = vec![doc(&[("tags", texts(&["rust", "search"]))])];
        let results = collect(docs, &["tags"], &["tags"], true);
        // Exact match: no `TextArray([...])` label and no `rust/search`
        // hierarchical path.
        assert_eq!(
            flatten(&results, "tags"),
            vec![(p(&["rust"]), 1), (p(&["search"]), 1)]
        );
    }

    #[test]
    fn facet_text_array_counts_across_documents() {
        // The generation must advance per `collect_doc`, or the second
        // document's `a` would be refused as "already counted".
        let docs = vec![
            doc(&[("tags", texts(&["a"]))]),
            doc(&[("tags", texts(&["a"]))]),
            doc(&[("tags", texts(&["a", "b"]))]),
        ];
        let results = collect(docs, &["tags"], &["tags"], true);
        assert_eq!(
            flatten(&results, "tags"),
            vec![(p(&["a"]), 3), (p(&["b"]), 1)]
        );
    }

    #[test]
    fn facet_arrays_do_not_bleed_across_fields() {
        let docs = vec![doc(&[("tags", texts(&["a"])), ("cats", texts(&["a"]))])];
        let results = collect(docs, &["tags", "cats"], &["tags", "cats"], true);
        assert_eq!(flatten(&results, "tags"), vec![(p(&["a"]), 1)]);
        assert_eq!(flatten(&results, "cats"), vec![(p(&["a"]), 1)]);
    }

    /// The array corpus shared by the parity test and the per-rule tests:
    /// a plain array, a duplicate element, two hierarchical elements with a
    /// shared ancestor, and an empty array.
    fn array_corpus() -> Vec<Document> {
        vec![
            doc(&[("tags", texts(&["rust", "search"]))]),
            doc(&[("tags", texts(&["rust", "rust"]))]),
            doc(&[("tags", texts(&["a/b", "a/c"]))]),
            doc(&[("tags", texts(&[]))]),
        ]
    }

    fn array_corpus_expected() -> Vec<(Vec<String>, u64)> {
        vec![
            (p(&["a"]), 1),
            (p(&["a", "b"]), 1),
            (p(&["a", "c"]), 1),
            (p(&["rust"]), 2),
            (p(&["search"]), 1),
        ]
    }

    #[test]
    fn facet_text_array_docvalues_matches_stored_doc() {
        // #597 parity for arrays: the DocValues path (`document()` would
        // panic) and the stored-document fallback must agree exactly.
        let via_dv = collect(array_corpus(), &["tags"], &["tags"], true);
        let via_doc = collect(array_corpus(), &["tags"], &[], false);
        assert_eq!(flatten(&via_dv, "tags"), flatten(&via_doc, "tags"));
        assert_eq!(flatten(&via_dv, "tags"), array_corpus_expected());
    }

    #[test]
    fn facet_repeated_array_element_counts_once() {
        let docs = vec![doc(&[("tags", texts(&["rust", "rust", "rust"]))])];
        let results = collect(docs, &["tags"], &["tags"], true);
        assert_eq!(flatten(&results, "tags"), vec![(p(&["rust"]), 1)]);
    }

    #[test]
    fn facet_shared_ancestor_counts_once_per_document() {
        let docs = vec![doc(&[("cat", texts(&["a/b", "a/c"]))])];
        let results = collect(docs, &["cat"], &["cat"], true);
        assert_eq!(
            flatten(&results, "cat"),
            vec![(p(&["a"]), 1), (p(&["a", "b"]), 1), (p(&["a", "c"]), 1)]
        );
    }

    #[test]
    fn facet_hierarchical_text_array_elements() {
        // Each element is its own path: a hierarchical one and a flat one.
        let docs = vec![doc(&[("cat", texts(&["a/b", "c"]))])];
        let results = collect(docs, &["cat"], &["cat"], true);
        assert_eq!(
            flatten(&results, "cat"),
            vec![(p(&["a"]), 1), (p(&["a", "b"]), 1), (p(&["c"]), 1)]
        );
    }

    // ---- Hierarchical facet tree, Issue #1192 ---------------------------

    #[test]
    fn facet_hierarchical_path_builds_nested_children() {
        let results = collect(
            vec![text_doc(&[("cat", "a/b/c")])],
            &["cat"],
            &["cat"],
            true,
        );
        assert_eq!(tree(&results, "cat"), "a:1[b:1[c:1]]");
        // Every node carries its full path, not just its label.
        let a = &results.get_field_facets("cat").unwrap()[0];
        assert_eq!(a.children[0].children[0].path.path, p(&["a", "b", "c"]));
    }

    #[test]
    fn facet_flat_value_and_hierarchical_ancestor_share_one_node() {
        // Within one document: the flat `a` and the root of `a/b` are the
        // same facet, counted once, in either element order.
        for elements in [["a", "a/b"], ["a/b", "a"]] {
            let results = collect(
                vec![doc(&[("cat", texts(&elements))])],
                &["cat"],
                &["cat"],
                true,
            );
            assert_eq!(tree(&results, "cat"), "a:1[b:1]", "{elements:?}");
        }
        // Across documents the two contributions add up on one node.
        let results = collect(
            vec![text_doc(&[("cat", "a")]), text_doc(&[("cat", "a/b")])],
            &["cat"],
            &["cat"],
            true,
        );
        assert_eq!(tree(&results, "cat"), "a:2[b:1]");
    }

    #[test]
    fn facet_max_facets_per_field_applies_per_level() {
        // a: 3 docs (children x: 2, y: 1), b: 1 doc (child z). With one
        // facet per level, ancestors no longer eat the budget: `a` keeps
        // its best child instead of being the only entry returned.
        let docs = vec![
            text_doc(&[("cat", "a/x")]),
            text_doc(&[("cat", "a/x")]),
            text_doc(&[("cat", "a/y")]),
            text_doc(&[("cat", "b/z")]),
        ];
        let config = || FacetConfig {
            max_facets_per_field: 1,
            ..Default::default()
        };
        let results = collect_with(config(), docs.clone(), &["cat"], &["cat"], true);
        assert_eq!(tree(&results, "cat"), "a:3[x:2]");

        let results = collect_with(
            FacetConfig {
                max_facets_per_field: 2,
                ..config()
            },
            docs,
            &["cat"],
            &["cat"],
            true,
        );
        assert_eq!(tree(&results, "cat"), "a:3[x:2 y:1] b:1[z:1]");
    }

    #[test]
    fn facet_min_count_prunes_whole_subtree() {
        let docs = vec![
            text_doc(&[("cat", "a/x")]),
            text_doc(&[("cat", "a/x")]),
            text_doc(&[("cat", "a/y")]),
            text_doc(&[("cat", "b/z/w")]),
        ];
        let results = collect_with(
            FacetConfig {
                min_count: 2,
                ..Default::default()
            },
            docs,
            &["cat"],
            &["cat"],
            true,
        );
        // `y` (1) goes; `b` (1) goes with its whole subtree.
        assert_eq!(tree(&results, "cat"), "a:3[x:2]");
    }

    #[test]
    fn facet_max_depth_truncates_paths() {
        let docs = || {
            vec![
                // Two elements that collapse to the same `a/b` prefix under
                // `max_depth: 2` — still one document for `a/b`.
                doc(&[("cat", texts(&["a/b/c", "a/b/d"]))]),
                text_doc(&[("cat", "a/e/f/g")]),
            ]
        };
        let with_depth = |max_depth| {
            let config = FacetConfig {
                max_depth,
                ..Default::default()
            };
            tree(
                &collect_with(config, docs(), &["cat"], &["cat"], true),
                "cat",
            )
        };
        assert_eq!(with_depth(2), "a:2[b:1 e:1]");
        assert_eq!(with_depth(1), "a:2");
        assert_eq!(with_depth(usize::MAX), "a:2[b:1[c:1 d:1] e:1[f:1[g:1]]]");
        let nothing = collect_with(
            FacetConfig {
                max_depth: 0,
                ..Default::default()
            },
            docs(),
            &["cat"],
            &["cat"],
            true,
        );
        assert!(nothing.get_field_facets("cat").is_none());
    }

    #[test]
    fn facet_count_ties_sort_by_label() {
        // 20 values with the same count, inserted in reverse label order:
        // the kept three must be the smallest labels, in order, not
        // whichever the hash map happened to yield first.
        let labels: Vec<String> = (0..20).rev().map(|i| format!("v{i:02}")).collect();
        let docs: Vec<Document> = labels
            .iter()
            .map(|label| text_doc(&[("cat", &format!("root/{label}")), ("flat", label)]))
            .collect();
        let results = collect_with(
            FacetConfig {
                max_facets_per_field: 3,
                ..Default::default()
            },
            docs,
            &["cat", "flat"],
            &["cat", "flat"],
            true,
        );
        assert_eq!(tree(&results, "flat"), "v00:1 v01:1 v02:1");
        assert_eq!(tree(&results, "cat"), "root:20[v00:1 v01:1 v02:1]");
    }

    #[test]
    fn facet_sort_by_name_orders_every_level() {
        let docs = vec![
            text_doc(&[("cat", "b/z")]),
            text_doc(&[("cat", "b/y")]),
            text_doc(&[("cat", "b/y")]),
            text_doc(&[("cat", "a/x")]),
        ];
        let results = collect_with(
            FacetConfig {
                sort_by_count: false,
                ..Default::default()
            },
            docs,
            &["cat"],
            &["cat"],
            true,
        );
        assert_eq!(tree(&results, "cat"), "a:1[x:1] b:3[y:2 z:1]");
    }

    #[test]
    fn facet_empty_components_are_dropped() {
        // Scalars and array elements follow the same rule.
        let values = ["/a/b", "a//b", "a/", "", "/"];
        let array = collect(
            vec![doc(&[("cat", texts(&values))])],
            &["cat"],
            &["cat"],
            true,
        );
        let scalars = collect(
            values.iter().map(|v| text_doc(&[("cat", v)])).collect(),
            &["cat"],
            &["cat"],
            true,
        );
        assert_eq!(tree(&array, "cat"), "a:1[b:1]");
        // Three documents reach `a`, two of them `a/b`; `""` and `"/"`
        // count nothing.
        assert_eq!(tree(&scalars, "cat"), "a:3[b:2]");
        let nothing = collect(
            vec![text_doc(&[("cat", "")]), text_doc(&[("cat", "//")])],
            &["cat"],
            &["cat"],
            true,
        );
        assert!(nothing.get_field_facets("cat").is_none());
    }

    #[test]
    fn facet_path_from_delimited_drops_empty_components() {
        let path = FacetPath::from_delimited("cat".to_string(), "/a//b/", "/");
        assert_eq!(path.path, p(&["a", "b"]));
        // No non-empty component: the field root, parent of every path.
        let root = FacetPath::from_delimited("cat".to_string(), "//", "/");
        assert_eq!(root.depth(), 0);
        assert!(root.is_parent_of(&path));
    }

    #[test]
    fn facet_total_facet_count_includes_nested_values() {
        let results = collect(
            vec![text_doc(&[("cat", "a/b/c")]), text_doc(&[("cat", "d")])],
            &["cat"],
            &["cat"],
            true,
        );
        assert_eq!(results.total_facet_count(), 4);
    }

    #[test]
    fn facet_int64_and_bool_arrays_expand() {
        let docs = vec![doc(&[
            ("n", DataValue::Int64Array(vec![1, 2, 2])),
            ("flags", DataValue::BoolArray(vec![true, false, true])),
        ])];
        let results = collect(docs, &["n", "flags"], &["n", "flags"], true);
        assert_eq!(flatten(&results, "n"), vec![(p(&["1"]), 1), (p(&["2"]), 1)]);
        assert_eq!(
            flatten(&results, "flags"),
            vec![(p(&["false"]), 1), (p(&["true"]), 1)]
        );
    }

    #[test]
    fn facet_float_values_always_carry_a_fraction() {
        // One document carrying every field, so the all-DocValues run never
        // misses a value (a miss falls back to `document()`, which the mock
        // turns into a panic by design).
        let docs = vec![doc(&[
            ("price", DataValue::Float64(1.0)),
            ("count", DataValue::Int64(1)),
            ("prices", DataValue::Float64Array(vec![1.5, 2.0, f64::NAN])),
        ])];
        let fields = ["price", "prices", "count"];
        let results = collect(docs, &fields, &fields, true);
        // `1.0` and `1` are different labels: the float carries a fraction.
        assert_eq!(flatten(&results, "price"), vec![(p(&["1.0"]), 1)]);
        assert_eq!(flatten(&results, "count"), vec![(p(&["1"]), 1)]);
        assert_eq!(
            flatten(&results, "prices"),
            vec![(p(&["1.5"]), 1), (p(&["2.0"]), 1), (p(&["NaN"]), 1)]
        );
    }

    #[test]
    fn facet_datetime_values_render_as_rfc3339() {
        let jan: DateTime<Utc> = "2024-01-01T00:00:00Z".parse().unwrap();
        // Nanoseconds are floored to microseconds, matching the DocValues
        // encoding, so both copies of an instant get one label.
        let nanos = DateTime::<Utc>::from_timestamp(1_700_000_000, 123_456_789).unwrap();
        let docs = vec![
            doc(&[("ts", DataValue::DateTime(jan))]),
            doc(&[("ts", DataValue::DateTimeArray(vec![jan, nanos]))]),
        ];
        let results = collect(docs, &["ts"], &["ts"], true);
        assert_eq!(
            flatten(&results, "ts"),
            vec![
                (p(&["2023-11-14T22:13:20.123456+00:00"]), 1),
                (p(&["2024-01-01T00:00:00+00:00"]), 2),
            ]
        );
    }

    #[test]
    fn facet_non_facetable_values_are_skipped() {
        // Reached through the stored-document fallback (no DocValues),
        // which is the only way `Vector` / `Bytes` can get here at all.
        let docs = vec![doc(&[
            ("loc", DataValue::Geo(GeoPoint::new(35.68, 139.77))),
            (
                "locs",
                DataValue::GeoArray(vec![GeoPoint::new(35.68, 139.77)]),
            ),
            (
                "ecef",
                DataValue::GeoEcefArray(vec![GeoEcefPoint {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                }]),
            ),
            ("vec", DataValue::Vector(vec![0.1, 0.2])),
            ("blob", DataValue::Bytes(vec![1, 2, 3], None)),
            ("nothing", DataValue::Null),
            ("tags", texts(&["kept"])),
        ])];
        let fields = ["loc", "locs", "ecef", "vec", "blob", "nothing", "tags"];
        let results = collect(docs, &fields, &[], false);
        for field in &fields[..6] {
            assert!(
                results.get_field_facets(field).is_none(),
                "{field} must contribute no facet value, got {:?}",
                flatten(&results, field)
            );
        }
        assert_eq!(flatten(&results, "tags"), vec![(p(&["kept"]), 1)]);
    }

    #[test]
    fn facet_docvalues_hit_with_no_paths_does_not_fall_back() {
        // A DocValues hit that yields no path is final: `document()` must
        // not be called (the mock panics if it is).
        let docs = vec![doc(&[
            ("tags", texts(&[])),
            ("loc", DataValue::Geo(GeoPoint::new(35.68, 139.77))),
            ("nothing", DataValue::Null),
        ])];
        let fields = ["tags", "loc", "nothing"];
        let results = collect(docs, &fields, &fields, true);
        for field in &fields {
            assert!(results.get_field_facets(field).is_none(), "{field}");
        }
    }

    #[test]
    fn facet_mixed_scalar_and_array_fields() {
        // One scalar field with DocValues, one array field without.
        let docs = vec![
            doc(&[
                ("brand", DataValue::Text("apple".into())),
                ("tags", texts(&["x", "y"])),
            ]),
            doc(&[
                ("brand", DataValue::Text("apple".into())),
                ("tags", texts(&["y"])),
            ]),
        ];
        let results = collect(docs, &["brand", "tags"], &["brand"], false);
        assert_eq!(flatten(&results, "brand"), vec![(p(&["apple"]), 2)]);
        assert_eq!(
            flatten(&results, "tags"),
            vec![(p(&["x"]), 1), (p(&["y"]), 2)]
        );
    }

    #[test]
    fn facet_arrays_respect_min_count_and_max_facets_per_field() {
        // Distinct counts (a: 3, b: 2, c: 1) so the count-sorted truncation
        // is deterministic.
        let corpus = || {
            vec![
                doc(&[("tags", texts(&["a", "b", "c"]))]),
                doc(&[("tags", texts(&["a", "b"]))]),
                doc(&[("tags", texts(&["a"]))]),
            ]
        };
        let min_two = collect_with(
            FacetConfig {
                min_count: 2,
                ..Default::default()
            },
            corpus(),
            &["tags"],
            &["tags"],
            true,
        );
        assert_eq!(
            flatten(&min_two, "tags"),
            vec![(p(&["a"]), 3), (p(&["b"]), 2)]
        );
        let top_one = collect_with(
            FacetConfig {
                max_facets_per_field: 1,
                ..Default::default()
            },
            corpus(),
            &["tags"],
            &["tags"],
            true,
        );
        assert_eq!(flatten(&top_one, "tags"), vec![(p(&["a"]), 3)]);
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
    fn facet_collect_doc_propagates_stored_document_error() {
        // A failed stored-document read is an error, not a facet value: the
        // collector used to count a made-up `value_{doc_id % 5}` instead.
        let reader = DvMockReader::with_failing_document(vec![text_doc(&[("cat", "a")])]);
        let mut collector = FacetCollector::new(FacetConfig::default(), vec!["cat".to_string()]);
        let err = collector
            .collect_doc(0, &reader)
            .expect_err("a stored-document read error must be returned");
        assert!(
            err.to_string()
                .contains("simulated stored-field read failure"),
            "unexpected error: {err}"
        );
        let results = collector.finalize().expect("finalize must not error");
        assert!(
            results.get_field_facets("cat").is_none(),
            "nothing may be counted for a document that could not be read, got {:?}",
            flatten(&results, "cat")
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
