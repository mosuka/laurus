//! Phrase query over a token graph, such as a quoted value through
//! multi-word synonyms (#1271).
//!
//! A quoted value whose analyzer emits alternatives of different lengths is
//! a graph of arcs, and every path through it is a phrase. Their number is
//! the product of the alternatives at each match, so [`GraphPhraseQuery`]
//! matches the graph itself instead of one phrase per path: each step of the
//! phrase matcher depends only on the previous matched position
//! ([`next_in_window`]), so the positions a path can reach are carried from
//! node to node along the arcs.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::error::Result;
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::phrase::{PhraseMatch, PhraseMatcher, PhraseScorer, next_in_window};
use crate::lexical::query::scorer::{BM25Scorer, Scorer};
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// One arc of a phrase graph: any of `terms` spans the positions from node
/// `from` to node `to`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhraseArc {
    /// The node the arc leaves.
    pub from: u32,
    /// The node the arc reaches, after `from`.
    pub to: u32,
    /// The alternative terms, any of which matches the arc.
    pub terms: Vec<String>,
}

impl PhraseArc {
    /// Create an arc from `from` to `to` matching any of `terms`.
    pub fn new(from: u32, to: u32, terms: Vec<String>) -> Self {
        PhraseArc { from, to, terms }
    }
}

/// The arcs of `arcs` that lie on a path from node `first` to node `last`,
/// with the nodes renumbered from 0 in order and the arcs sorted by
/// `(from, to)`. Each `(from, to)` must appear once.
///
/// An arc no path uses, such as a dead end of a hand-built token stream, is
/// dropped. The result is empty when no path goes from `first` to `last`.
pub(crate) fn complete_paths(mut arcs: Vec<PhraseArc>, first: u32, last: u32) -> Vec<PhraseArc> {
    arcs.sort_by_key(|arc| (arc.from, arc.to));

    // Arcs only go forward, so one pass in each direction settles which
    // nodes a path from `first` reaches and which reach `last`.
    let mut reached = BTreeSet::from([first]);
    for arc in &arcs {
        if reached.contains(&arc.from) {
            reached.insert(arc.to);
        }
    }
    let mut reaching = BTreeSet::from([last]);
    for arc in arcs.iter().rev() {
        if reaching.contains(&arc.to) {
            reaching.insert(arc.from);
        }
    }
    arcs.retain(|arc| reached.contains(&arc.from) && reaching.contains(&arc.to));

    let nodes: BTreeMap<u32, u32> = arcs
        .iter()
        .flat_map(|arc| [arc.from, arc.to])
        .collect::<BTreeSet<u32>>()
        .into_iter()
        .zip(0..)
        .collect();
    for arc in &mut arcs {
        arc.from = nodes[&arc.from];
        arc.to = nodes[&arc.to];
    }
    arcs
}

/// A phrase query whose positions form a graph: it matches where any path
/// through the graph matches as a [`PhraseQuery`](super::PhraseQuery) with
/// the same slop would.
///
/// The query parser builds one for a quoted value with two or more paths of
/// two or more positions, such as `"ml is"` with `ml` and `machine learning`
/// as synonyms. Each path is matched with the phrase matcher's rule, so the
/// documents matched are those that some path's phrase matches, but the
/// work grows with the arcs, not with the number of paths. A document gets
/// one phrase score, from the anchors where some path completes.
#[derive(Debug, Clone)]
pub struct GraphPhraseQuery {
    /// The field to search in.
    field: String,
    /// The arcs, sorted by `(from, to)`, over nodes `0..=last`.
    arcs: Vec<PhraseArc>,
    /// The last node, which every path ends at.
    last: u32,
    /// The most arcs on one path, as the index lays the graph out.
    longest: usize,
    /// The boost factor for this query.
    boost: f32,
    /// The largest gap allowed after each position, as in `PhraseQuery`.
    slop: u32,
}

impl GraphPhraseQuery {
    /// Create a graph phrase over `arcs` as [`complete_paths`] returns them:
    /// nodes numbered from 0, sorted by `(from, to)`, every arc on a path
    /// from node 0 to the last node, and no arc straight from the first node
    /// to the last (a one-position path is a term, not a phrase).
    pub(crate) fn from_arcs<S: Into<String>>(field: S, arcs: Vec<PhraseArc>) -> Self {
        let last = arcs.iter().map(|arc| arc.to).max().unwrap_or(0);
        debug_assert!(!arcs.is_empty(), "a graph phrase needs arcs");
        debug_assert!(
            arcs.windows(2)
                .all(|pair| (pair[0].from, pair[0].to) < (pair[1].from, pair[1].to)),
            "arcs must be sorted by (from, to), each once: {arcs:?}"
        );
        debug_assert!(
            arcs.iter().all(|arc| arc.from < arc.to),
            "arcs must go forward: {arcs:?}"
        );
        debug_assert!(
            !arcs.iter().any(|arc| arc.from == 0 && arc.to == last),
            "a one-position path is a term: {arcs:?}"
        );
        debug_assert_eq!(
            complete_paths(arcs.clone(), 0, last),
            arcs,
            "arcs must be complete_paths' output"
        );

        // Arcs are sorted by `from`, so every arc into a node comes before
        // the arcs leaving it.
        let mut depth = vec![0usize; last as usize + 1];
        for arc in &arcs {
            let through = depth[arc.from as usize] + 1;
            let reached = &mut depth[arc.to as usize];
            *reached = (*reached).max(through);
        }

        GraphPhraseQuery {
            field: field.into(),
            arcs,
            last,
            longest: depth[last as usize],
            boost: 1.0,
            slop: 0,
        }
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set the largest gap allowed after each position (0 = exact).
    pub fn with_slop(mut self, slop: u32) -> Self {
        self.slop = slop;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the arcs, sorted by `(from, to)`, over nodes from 0 to the last.
    pub fn arcs(&self) -> &[PhraseArc] {
        &self.arcs
    }

    /// Get the slop value.
    pub fn slop(&self) -> u32 {
        self.slop
    }

    /// Find the documents where some path matches, sorted by document ID.
    ///
    /// Each distinct term's postings are read once. A term missing from the
    /// index only drops its alternative, as in `PhraseQuery`.
    pub(crate) fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<PhraseMatch>> {
        let mut terms: Vec<&str> = Vec::new();
        let mut term_ids: HashMap<&str, usize> = HashMap::new();
        let arc_terms: Vec<Vec<usize>> = self
            .arcs
            .iter()
            .map(|arc| {
                arc.terms
                    .iter()
                    .map(|term| {
                        *term_ids.entry(term.as_str()).or_insert_with(|| {
                            terms.push(term.as_str());
                            terms.len() - 1
                        })
                    })
                    .collect()
            })
            .collect();

        // Per candidate document, each term's sorted positions.
        let mut docs: HashMap<u64, Vec<Vec<u64>>> = HashMap::new();
        for (id, term) in terms.iter().enumerate() {
            let Some(mut iter) = reader.postings(&self.field, term)? else {
                continue;
            };
            while iter.next()? {
                let doc_id = iter.doc_id();
                if doc_id == u64::MAX {
                    break;
                }
                docs.entry(doc_id)
                    .or_insert_with(|| vec![Vec::new(); terms.len()])[id]
                    .extend(iter.positions()?);
            }
        }

        let mut matches = Vec::new();
        for (doc_id, mut positions) in docs {
            for term_positions in &mut positions {
                term_positions.sort_unstable();
                term_positions.dedup();
            }
            // Alternatives share positions when stacked (synonyms).
            let occurrences: Vec<Cow<'_, [u64]>> = arc_terms
                .iter()
                .map(|ids| match ids.as_slice() {
                    [id] => Cow::Borrowed(positions[*id].as_slice()),
                    ids => {
                        let mut merged: Vec<u64> = ids
                            .iter()
                            .flat_map(|&id| positions[id].iter().copied())
                            .collect();
                        merged.sort_unstable();
                        merged.dedup();
                        Cow::Owned(merged)
                    }
                })
                .collect();

            let anchors = complete_anchors(&self.arcs, self.last, &occurrences, self.slop);
            if !anchors.is_empty() {
                matches.push(PhraseMatch {
                    doc_id,
                    phrase_freq: u32::try_from(anchors.len()).unwrap_or(u32::MAX),
                    positions: anchors,
                });
            }
        }
        matches.sort_by_key(|m| m.doc_id);
        Ok(matches)
    }

    /// A phrase scorer over `matches`, like `PhraseQuery`'s, with the
    /// length boost of the longest path.
    fn scorer_for(
        &self,
        reader: &dyn LexicalIndexReader,
        matches: &[PhraseMatch],
    ) -> Box<dyn Scorer> {
        let total_docs = reader.doc_count();
        if total_docs == 0 {
            return Box::new(BM25Scorer::new(0, 0, 0, 1.0, 1, self.boost));
        }
        let avg_field_length = reader
            .field_statistics(&self.field)
            .map(|stats| stats.avg_field_length)
            .unwrap_or(10.0);
        let boost = self.boost * (1.0 + 0.2 * (self.longest as f32 - 1.0));
        Box::new(PhraseScorer::new(
            matches,
            total_docs,
            avg_field_length,
            boost,
        ))
    }
}

/// The anchors (positions of a first term) from which some path through
/// `arcs` completes in one document, sorted.
///
/// `occurrences` holds, per arc, the sorted positions of its terms. A path
/// anchors on any occurrence of its first arc, and each next arc must occur
/// where [`next_in_window`] finds it after the previous one, exactly as
/// `PhraseMatcher` checks one phrase. That step depends only on the previous
/// position, so the (anchor, previous position) pairs some path brings to a
/// node are enough to go on from it, whichever path brought them.
fn complete_anchors<O: AsRef<[u64]>>(
    arcs: &[PhraseArc],
    last: u32,
    occurrences: &[O],
    slop: u32,
) -> Vec<u64> {
    let mut states: Vec<Vec<(u64, u64)>> = vec![Vec::new(); last as usize + 1];
    let mut settled = 0;
    for (arc, occurrences) in arcs.iter().zip(occurrences) {
        let occurrences = occurrences.as_ref();
        let (before, after) = states.split_at_mut(arc.to as usize);
        let reached = &mut after[0];
        if arc.from == 0 {
            reached.extend(occurrences.iter().map(|&position| (position, position)));
            continue;
        }

        // Every arc into `from` came earlier, so its states are final.
        let states_at = &mut before[arc.from as usize];
        if arc.from != settled {
            settled = arc.from;
            states_at.sort_unstable();
            states_at.dedup();
        }
        for &(anchor, previous) in states_at.iter() {
            if let Some(position) = next_in_window(occurrences, previous, slop) {
                reached.push((anchor, position));
            }
        }
    }

    let mut anchors: Vec<u64> = states[last as usize]
        .iter()
        .map(|&(anchor, _)| anchor)
        .collect();
    anchors.sort_unstable();
    anchors.dedup();
    anchors
}

impl Query for GraphPhraseQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(PhraseMatcher::from_matches(
            self.find_matches(reader)?,
        )))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let matches = self.find_matches(reader)?;
        Ok(self.scorer_for(reader, &matches))
    }

    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        let matches = self.find_matches(reader)?;
        let scorer = self.scorer_for(reader, &matches);
        Ok((Box::new(PhraseMatcher::from_matches(matches)), scorer))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        format!(
            "GraphPhraseQuery(field:{}, arcs:{}, slop:{})",
            self.field,
            describe_arcs(&self.arcs),
            self.slop
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.arcs.is_empty())
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        let terms: usize = self.arcs.iter().map(|arc| arc.terms.len()).sum();
        Ok(terms as u64 * 100) // Rough estimate, as for PhraseQuery
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn collect_positional_field_refs(&self, out: &mut std::collections::HashSet<String>) {
        // Every path has two or more positions.
        out.insert(self.field.clone());
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        if field.is_none_or(|f| f == self.field) {
            out.push(HighlightTerm::GraphPhrase {
                arcs: self.arcs.clone(),
                slop: self.slop,
            });
        }
    }

    fn cache_key(&self) -> Option<String> {
        // Field + arcs + slop determine the matched set; boost is
        // score-only and excluded.
        Some(format!(
            "graph_phrase|{:?}|{}|{}",
            self.field,
            describe_arcs(&self.arcs),
            self.slop
        ))
    }
}

/// `arcs` as `[0-1 ["machine"], 0-2 ["ml"], ...]`.
fn describe_arcs(arcs: &[PhraseArc]) -> String {
    let arcs: Vec<String> = arcs
        .iter()
        .map(|arc| format!("{}-{} {:?}", arc.from, arc.to, arc.terms))
        .collect();
    format!("[{}]", arcs.join(", "))
}

/// Graphs and paths for tests comparing a graph phrase with its paths.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{PhraseArc, complete_paths};

    pub(crate) fn arc(from: u32, to: u32, terms: &[&str]) -> PhraseArc {
        PhraseArc::new(from, to, terms.iter().map(|t| t.to_string()).collect())
    }

    /// Every path from node 0 through `arcs`, as phrase positions.
    pub(crate) fn paths(arcs: &[PhraseArc]) -> Vec<Vec<Vec<String>>> {
        fn walk(
            node: u32,
            last: u32,
            arcs: &[PhraseArc],
            path: &mut Vec<Vec<String>>,
            paths: &mut Vec<Vec<Vec<String>>>,
        ) {
            if node == last {
                paths.push(path.clone());
                return;
            }
            for arc in arcs.iter().filter(|arc| arc.from == node) {
                path.push(arc.terms.clone());
                walk(arc.to, last, arcs, path, paths);
                path.pop();
            }
        }
        let last = arcs.iter().map(|arc| arc.to).max().unwrap();
        let mut paths = Vec::new();
        walk(0, last, arcs, &mut Vec::new(), &mut paths);
        paths
    }

    /// Reproducible pseudo-random numbers, without a dependency.
    pub(crate) struct Lcg(pub(crate) u64);

    impl Lcg {
        pub(crate) fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % n
        }
    }

    /// The words of random texts and graphs. Texts never use "z", so an arc
    /// with it has a term missing from the index.
    pub(crate) const WORDS: &[&str] = &["a", "b", "c", "d", "z"];

    /// `count` texts of 2 to 10 words, without "z".
    pub(crate) fn random_texts(rng: &mut Lcg, count: usize) -> Vec<String> {
        (0..count)
            .map(|_| {
                let words: Vec<&str> = (0..2 + rng.below(9))
                    .map(|_| WORDS[rng.below(4) as usize])
                    .collect();
                words.join(" ")
            })
            .collect()
    }

    /// Random arcs over up to six nodes, cut to their complete paths, with
    /// no arc from the first node straight to the last.
    pub(crate) fn random_graph(rng: &mut Lcg) -> Option<Vec<PhraseArc>> {
        let last = 2 + rng.below(4) as u32;
        let mut arcs = Vec::new();
        for from in 0..last {
            for to in from + 1..=(from + 3).min(last) {
                if (from, to) == (0, last) || rng.below(2) == 0 {
                    continue;
                }
                let mut terms: Vec<String> = (0..1 + rng.below(2))
                    .map(|_| WORDS[rng.below(WORDS.len() as u64) as usize].to_string())
                    .collect();
                terms.sort();
                terms.dedup();
                arcs.push(PhraseArc::new(from, to, terms));
            }
        }
        let arcs = complete_paths(arcs, 0, last);
        (!arcs.is_empty()).then_some(arcs)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashSet};
    use std::sync::Arc;

    use super::test_support::{Lcg, arc, paths, random_graph, random_texts};
    use super::*;
    use crate::analysis::analyzer::analyzer::Analyzer;
    use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
    use crate::analysis::synonym::dictionary::SynonymDictionary;
    use crate::analysis::token_filter::synonym_graph::SynonymGraphFilter;
    use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;
    use crate::data::Document;
    use crate::lexical::index::LexicalIndex;
    use crate::lexical::index::config::InvertedIndexConfig;
    use crate::lexical::index::inverted::InvertedIndex;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn whitespace_analyzer() -> PipelineAnalyzer {
        PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
    }

    /// Stacks the synonym `groups` at index time, as the Engine does.
    fn stacking_analyzer(groups: &[&[&str]]) -> PipelineAnalyzer {
        let mut dict = SynonymDictionary::new(None).unwrap();
        for group in groups {
            dict.add_synonym_group(group.iter().map(|s| s.to_string()).collect());
        }
        whitespace_analyzer().add_filter(Arc::new(SynonymGraphFilter::new(dict, true)))
    }

    /// Index each text as one document of field `body`; doc ids follow the
    /// order of `texts`.
    fn index(analyzer: Arc<dyn Analyzer>, texts: &[&str]) -> Arc<dyn LexicalIndexReader> {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexConfig {
            analyzer,
            ..Default::default()
        };
        let index = InvertedIndex::create(storage, config).unwrap();
        let mut writer = index.writer().unwrap();
        for text in texts {
            writer
                .add_document(Document::builder().add_text("body", *text).build())
                .unwrap();
        }
        writer.commit().unwrap();
        index.reader().unwrap()
    }

    /// What a quoted value matched before #1271: each path as a phrase of
    /// its own. Per matching document, the union of the paths' anchors.
    fn path_expansion(
        reader: &dyn LexicalIndexReader,
        arcs: &[PhraseArc],
        slop: u32,
    ) -> Vec<(u64, Vec<u64>)> {
        let mut by_doc: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
        for path in paths(arcs) {
            for found in PhraseMatcher::find_phrase_matches(reader, "body", &path, slop).unwrap() {
                by_doc
                    .entry(found.doc_id)
                    .or_default()
                    .extend(found.positions);
            }
        }
        by_doc
            .into_iter()
            .map(|(doc_id, anchors)| (doc_id, anchors.into_iter().collect()))
            .collect()
    }

    fn graph_matches(
        reader: &dyn LexicalIndexReader,
        arcs: &[PhraseArc],
        slop: u32,
    ) -> Vec<(u64, Vec<u64>)> {
        GraphPhraseQuery::from_arcs("body", arcs.to_vec())
            .with_slop(slop)
            .find_matches(reader)
            .unwrap()
            .into_iter()
            .map(|found| {
                assert_eq!(found.phrase_freq as usize, found.positions.len());
                (found.doc_id, found.positions)
            })
            .collect()
    }

    /// The graph walk matches exactly the documents, at exactly the anchors,
    /// that one phrase per path matches, at every slop, over positions laid
    /// out plainly and with stacked multi-word synonyms.
    #[test]
    fn graph_phrase_matches_the_union_of_its_paths() {
        let mut rng = Lcg(1271);
        let analyzers: [Arc<dyn Analyzer>; 2] = [
            Arc::new(whitespace_analyzer()),
            Arc::new(stacking_analyzer(&[&["a", "b c"], &["d", "c"]])),
        ];
        let mut compared = 0;
        let mut matched = 0;
        for round in 0..40 {
            let texts = random_texts(&mut rng, 12);
            let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
            for analyzer in &analyzers {
                let reader = index(analyzer.clone(), &texts);
                for _ in 0..20 {
                    let Some(arcs) = random_graph(&mut rng) else {
                        continue;
                    };
                    for slop in 0..3 {
                        let expected = path_expansion(reader.as_ref(), &arcs, slop);
                        assert_eq!(
                            graph_matches(reader.as_ref(), &arcs, slop),
                            expected,
                            "round {round}, slop {slop}, arcs {arcs:?}, texts {texts:?}"
                        );
                        compared += 1;
                        matched += usize::from(!expected.is_empty());
                    }
                }
            }
        }
        assert!(compared > 1500, "only {compared} comparisons");
        assert!(
            matched > compared / 4,
            "only {matched} of {compared} matched"
        );
    }

    /// The greedy rule's misses are kept: `[a][b][c]~1` takes b@1 in
    /// "a b b x c" and then finds no c, though b@2 would have led to one.
    #[test]
    fn greedy_misses_match_phrase_query() {
        let reader = index(Arc::new(whitespace_analyzer()), &["a b b x c"]);
        let arcs = vec![
            arc(0, 1, &["a"]),
            arc(0, 2, &["d"]),
            arc(1, 2, &["b"]),
            arc(2, 3, &["c"]),
        ];
        assert!(graph_matches(reader.as_ref(), &arcs, 1).is_empty());
        assert_eq!(graph_matches(reader.as_ref(), &arcs, 2), vec![(0, vec![0])]);
        for slop in 0..3 {
            assert_eq!(
                graph_matches(reader.as_ref(), &arcs, slop),
                path_expansion(reader.as_ref(), &arcs, slop)
            );
        }
    }

    /// A term missing from the index drops only the paths through it.
    #[test]
    fn a_missing_term_drops_only_its_paths() {
        let reader = index(
            Arc::new(whitespace_analyzer()),
            &["new york is big", "big apple is big"],
        );
        let arcs = vec![
            arc(0, 1, &["new"]),
            arc(0, 2, &["huge"]),
            arc(1, 3, &["york"]),
            arc(2, 3, &["apple"]),
            arc(3, 4, &["is"]),
        ];
        assert_eq!(graph_matches(reader.as_ref(), &arcs, 0), vec![(0, vec![0])]);
    }

    /// With synonyms stacked at index time, "ml is fun" is stored as ml@0
    /// machine@0 learning@1 is@2: `[ml][is]` misses it and
    /// `[machine][learning][is]` finds it, both through one graph.
    #[test]
    fn paths_of_different_lengths_meet_on_flattened_positions() {
        let analyzer = stacking_analyzer(&[&["ml", "machine learning"]]);
        let reader = index(
            Arc::new(analyzer),
            &["ml is fun", "machine learning is fun", "learning is fun"],
        );
        let arcs = vec![
            arc(0, 1, &["machine"]),
            arc(0, 2, &["ml"]),
            arc(1, 2, &["learning"]),
            arc(2, 3, &["is"]),
        ];
        let found = graph_matches(reader.as_ref(), &arcs, 0);
        assert_eq!(found, path_expansion(reader.as_ref(), &arcs, 0));
        let docs: Vec<u64> = found.iter().map(|(doc_id, _)| *doc_id).collect();
        assert_eq!(docs, vec![0, 1]);
    }

    /// A chain is one path, so it matches as the `PhraseQuery` of its arcs.
    #[test]
    fn a_chain_matches_like_phrase_query() {
        let reader = index(
            Arc::new(whitespace_analyzer()),
            &["a big dog", "a large dog", "a very large dog", "a big cat"],
        );
        let arcs = vec![
            arc(0, 1, &["a"]),
            arc(1, 2, &["big", "large"]),
            arc(2, 3, &["dog"]),
        ];
        for slop in 0..3 {
            let phrase: Vec<(u64, Vec<u64>)> =
                PhraseMatcher::find_phrase_matches(reader.as_ref(), "body", &paths(&arcs)[0], slop)
                    .unwrap()
                    .into_iter()
                    .map(|found| (found.doc_id, found.positions))
                    .collect();
            assert_eq!(
                graph_matches(reader.as_ref(), &arcs, slop),
                phrase,
                "{slop}"
            );
        }
    }

    /// Arcs off every path from the first node to the last are dropped:
    /// one before the first node, a dead end, and one past the last node.
    #[test]
    fn complete_paths_keeps_only_arcs_on_a_path() {
        let arcs = vec![
            arc(4, 5, &["before"]),
            arc(5, 6, &["a"]),
            arc(5, 7, &["dead"]),
            arc(6, 8, &["b"]),
            arc(8, 9, &["after"]),
        ];
        assert_eq!(
            complete_paths(arcs.clone(), 5, 8),
            vec![arc(0, 1, &["a"]), arc(1, 2, &["b"])]
        );
        assert_eq!(complete_paths(arcs, 5, 7), vec![arc(0, 1, &["dead"])]);
        assert!(complete_paths(vec![arc(0, 1, &["a"]), arc(2, 3, &["b"])], 0, 3).is_empty());
    }

    fn ml_is() -> GraphPhraseQuery {
        GraphPhraseQuery::from_arcs(
            "body",
            vec![
                arc(0, 1, &["machine"]),
                arc(0, 2, &["ml"]),
                arc(1, 2, &["learning"]),
                arc(2, 3, &["is"]),
            ],
        )
    }

    #[test]
    fn query_surface() {
        let query = ml_is().with_slop(1).with_boost(2.0);
        assert_eq!(query.field(), "body");
        assert_eq!(query.slop(), 1);
        assert_eq!(query.boost(), 2.0);
        assert_eq!(
            query.description(),
            "GraphPhraseQuery(field:body, arcs:[0-1 [\"machine\"], 0-2 [\"ml\"], \
             1-2 [\"learning\"], 2-3 [\"is\"]], slop:1)"
        );

        // The boost changes the score only, so not the cache key.
        assert_eq!(query.cache_key(), ml_is().with_slop(1).cache_key());
        assert_ne!(query.cache_key(), ml_is().cache_key());

        let mut fields = HashSet::new();
        query.collect_positional_field_refs(&mut fields);
        assert_eq!(fields, HashSet::from(["body".to_string()]));

        let mut terms = Vec::new();
        query.collect_highlight_terms(Some("other"), &mut terms);
        assert!(terms.is_empty());
        query.collect_highlight_terms(Some("body"), &mut terms);
        match terms.as_slice() {
            [HighlightTerm::GraphPhrase { arcs, slop: 1 }] => assert_eq!(arcs, query.arcs()),
            other => panic!("expected one GraphPhrase, got {other:?}"),
        }
    }

    /// Matched documents score above zero, with the length boost of the
    /// longest path (`[machine][learning][is]`, three positions).
    #[test]
    fn matched_docs_score_with_the_longest_paths_length() {
        let analyzer = stacking_analyzer(&[&["ml", "machine learning"]]);
        let reader = index(
            Arc::new(analyzer),
            &["ml is fun", "machine learning is fun", "fun is ml"],
        );
        let query = ml_is().with_boost(2.0);
        let (mut matcher, scorer) = query.matcher_scorer(reader.as_ref()).unwrap();
        assert!(
            (scorer.boost() - 2.0 * 1.4).abs() < 1e-6,
            "{}",
            scorer.boost()
        );

        let mut docs = Vec::new();
        while !matcher.is_exhausted() {
            let doc_id = matcher.doc_id();
            assert!(scorer.score(doc_id, 1.0, None) > 0.0, "{doc_id}");
            docs.push(doc_id);
            matcher.next().unwrap();
        }
        assert_eq!(docs, vec![0, 1]);
    }
}
