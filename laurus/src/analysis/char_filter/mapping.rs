use aho_corasick::{AhoCorasick, MatchKind};
use std::collections::BTreeMap;

use super::{CharFilter, Transformation};
use crate::error::LaurusError;

pub struct MappingCharFilter {
    ac: AhoCorasick,
    replacements: Vec<String>,
}

impl MappingCharFilter {
    /// Create a char filter that replaces each key of `mapping` with its
    /// value, the longest key first.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] if the keys are too large to
    /// search for.
    pub fn new(mapping: BTreeMap<String, String>) -> crate::error::Result<Self> {
        let mut keys = Vec::new();
        let mut replacements = Vec::new();

        // Sort keys to ensure deterministic order if needed, though AhoCorasick handles ordering options.
        // We iterate mapping. AhoCorasick uses parallel arrays for construction.
        for (k, v) in mapping {
            keys.push(k);
            replacements.push(v);
        }

        let ac = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&keys)
            .map_err(|e| LaurusError::invalid_argument(format!("Invalid mapping: {e}")))?;

        Ok(Self { ac, replacements })
    }
}

impl CharFilter for MappingCharFilter {
    fn filter(&self, input: &str) -> (String, Vec<Transformation>) {
        let mut output = String::with_capacity(input.len());
        let mut transformations = Vec::new();

        let mut last_match_end = 0;

        // Find all non-overlapping matches
        for m in self.ac.find_iter(input) {
            let match_start = m.start();
            let match_end = m.end();
            let pattern_index = m.pattern();
            let replacement = &self.replacements[pattern_index.as_usize()];

            // transform: last_match_end..match_start is unchanged
            output.push_str(&input[last_match_end..match_start]);

            let new_start = output.len();
            output.push_str(replacement);
            let new_end = output.len();

            // Record transformation
            transformations.push(Transformation::new(
                match_start,
                match_end,
                new_start,
                new_end,
            ));

            last_match_end = match_end;
        }

        output.push_str(&input[last_match_end..]);

        (output, transformations)
    }

    fn name(&self) -> &'static str {
        "mapping"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_char_filter() {
        let mut mapping = BTreeMap::new();
        mapping.insert("ph".to_string(), "f".to_string());
        mapping.insert("qu".to_string(), "k".to_string());

        let filter = MappingCharFilter::new(mapping).unwrap();
        let input = "phone queue";
        let (output, trans) = filter.filter(input);

        assert_eq!(output, "fone keue");
        // "ph" -> "f", "qu" -> "k"
        assert_eq!(trans.len(), 2);

        assert_eq!(trans[0].original_start, 0); // ph
        assert_eq!(trans[0].original_end, 2);
        assert_eq!(trans[0].new_start, 0); // f
        assert_eq!(trans[0].new_end, 1);

        // "one " unchanged. original[2..6] ("one ") -> new[1..5] ("one ")
        // Next match "qu" at original 6..8. new starts at 5.

        assert_eq!(trans[1].original_start, 6); // qu
        assert_eq!(trans[1].original_end, 8);
        assert_eq!(trans[1].new_start, 5); // k
        assert_eq!(trans[1].new_end, 6);
    }

    #[test]
    fn test_mapping_expansion() {
        let mut mapping = BTreeMap::new();
        mapping.insert("a".to_string(), "aaa".to_string());
        let filter = MappingCharFilter::new(mapping).unwrap();
        let (output, trans) = filter.filter("bab");
        assert_eq!(output, "baaab");
        assert_eq!(trans.len(), 1);
        assert_eq!(trans[0].original_start, 1);
        assert_eq!(trans[0].original_end, 2);
        assert_eq!(trans[0].new_start, 1);
        assert_eq!(trans[0].new_end, 4);
    }

    #[test]
    fn test_mapping_deletion() {
        let mut mapping = BTreeMap::new();
        mapping.insert("foo".to_string(), "".to_string());
        let filter = MappingCharFilter::new(mapping).unwrap();
        let (output, trans) = filter.filter("afoob");
        assert_eq!(output, "ab");
        assert_eq!(trans.len(), 1);
        assert_eq!(trans[0].original_start, 1);
        assert_eq!(trans[0].original_end, 4);
        assert_eq!(trans[0].new_start, 1);
        assert_eq!(trans[0].new_end, 1);
    }

    #[test]
    fn test_mapping_overlap() {
        let mut mapping = BTreeMap::new();
        mapping.insert("ab".to_string(), "1".to_string());
        mapping.insert("abc".to_string(), "2".to_string());
        let filter = MappingCharFilter::new(mapping).unwrap();

        // "abc" should match "abc" -> "2" (longest match)
        let (output, trans) = filter.filter("abc");
        assert_eq!(output, "2");
        assert_eq!(trans.len(), 1);
        assert_eq!(trans[0].original_start, 0);
        assert_eq!(trans[0].original_end, 3);
        assert_eq!(trans[0].new_start, 0);
        assert_eq!(trans[0].new_end, 1);
    }

    #[test]
    fn test_mapping_multibyte() {
        let mut mapping = BTreeMap::new();
        mapping.insert("壱".to_string(), "1".to_string());
        let filter = MappingCharFilter::new(mapping).unwrap();

        let (output, trans) = filter.filter("第壱位");
        assert_eq!(output, "第1位");
        assert_eq!(trans.len(), 1);
        // "壱" is 3 bytes (starts at 3)
        assert_eq!(trans[0].original_start, 3);
        assert_eq!(trans[0].original_end, 6);
        assert_eq!(trans[0].new_start, 3);
        assert_eq!(trans[0].new_end, 4);
    }
}
