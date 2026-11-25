//! Finite automaton support for query matching.
//!
//! This module provides the automata that decide which terms a prefix,
//! wildcard, regexp or fuzzy query matches.

use regex_syntax::hir::Look;
use regex_syntax::hir::literal::{ExtractKind, Extractor};

use crate::error::Result;
use crate::util::levenshtein::{
    damerau_levenshtein_distance_threshold, levenshtein_distance_threshold,
};

/// A simple Levenshtein automaton for fuzzy string matching.
///
/// This is a simplified implementation that checks if a candidate string
/// is within the specified edit distance from the pattern.
///
use regex::Regex;

/// A trait for finite automata used in term filtering.
pub trait Automaton: Send + Sync {
    /// Check if a candidate string matches the automaton.
    fn matches(&self, candidate: &str) -> bool;

    /// A prefix of every string the automaton matches, if it has one.
    ///
    /// The enumeration of a term dictionary, which is sorted, starts at the
    /// first term with this prefix and ends after the last one. A prefix that
    /// a matching term does not have would drop that term.
    fn initial_seek_term(&self) -> Option<String>;
}

/// An automaton that matches the strings with a prefix.
#[derive(Debug, Clone)]
pub struct PrefixAutomaton {
    /// The prefix.
    prefix: String,
}

impl PrefixAutomaton {
    /// Create an automaton that matches the strings that start with `prefix`.
    pub fn new(prefix: impl Into<String>) -> Self {
        PrefixAutomaton {
            prefix: prefix.into(),
        }
    }
}

impl Automaton for PrefixAutomaton {
    fn matches(&self, candidate: &str) -> bool {
        candidate.starts_with(&self.prefix)
    }

    fn initial_seek_term(&self) -> Option<String> {
        (!self.prefix.is_empty()).then(|| self.prefix.clone())
    }
}

/// A simple Levenshtein automaton for fuzzy string matching.
///
/// This is a simplified implementation that checks if a candidate string
/// is within the specified edit distance from the pattern.
#[derive(Debug, Clone)]
pub struct LevenshteinAutomaton {
    /// The pattern string to match against
    pattern: String,
    /// Precomputed prefix of `pattern` of length `prefix_length`, if any. Stored
    /// to avoid recomputing `pattern.chars().take(prefix_length).collect()` on
    /// every `matches` call — that allocation showed up as a per-term hot
    /// spot (see #530).
    pattern_prefix: Option<String>,
    /// Maximum edit distance
    max_edits: u32,
    /// Minimum prefix length that must match exactly
    prefix_length: usize,
    /// Whether to use Damerau-Levenshtein (transpositions count as 1 edit)
    transpositions: bool,
}

impl LevenshteinAutomaton {
    /// Create a new Levenshtein automaton.
    pub fn new(
        pattern: impl Into<String>,
        max_edits: u32,
        prefix_length: usize,
        transpositions: bool,
    ) -> Self {
        let pattern: String = pattern.into();
        let pattern_prefix = if prefix_length > 0 {
            Some(pattern.chars().take(prefix_length).collect())
        } else {
            None
        };
        LevenshteinAutomaton {
            pattern,
            pattern_prefix,
            max_edits,
            prefix_length,
            transpositions,
        }
    }

    /// Get the pattern string.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Get the maximum edit distance.
    pub fn max_edits(&self) -> u32 {
        self.max_edits
    }

    /// Get the prefix length.
    pub fn prefix_length(&self) -> usize {
        self.prefix_length
    }

    /// Check if transpositions are enabled.
    pub fn uses_transpositions(&self) -> bool {
        self.transpositions
    }

    /// The edit distance between the pattern and `candidate`.
    ///
    /// # Returns
    ///
    /// `None` if the automaton does not match `candidate`: the distance is
    /// above the maximum, or the required prefix differs.
    pub fn distance(&self, candidate: &str) -> Option<usize> {
        // Check prefix requirement against the precomputed prefix.
        if let Some(prefix) = &self.pattern_prefix
            && !candidate.starts_with(prefix.as_str())
        {
            return None;
        }

        // Calculate edit distance using the threshold-aware variants. They use
        // a 2-row (or 3-row for Damerau) sliding buffer instead of the full
        // N × M matrix, terminate early when the per-row minimum exceeds the
        // threshold, and avoid heap allocations for the matrix on the
        // non-matching majority of candidates.
        let threshold = self.max_edits as usize;
        if self.transpositions {
            damerau_levenshtein_distance_threshold(&self.pattern, candidate, threshold)
        } else {
            levenshtein_distance_threshold(&self.pattern, candidate, threshold)
        }
    }
}

impl Automaton for LevenshteinAutomaton {
    fn matches(&self, candidate: &str) -> bool {
        self.distance(candidate).is_some()
    }

    fn initial_seek_term(&self) -> Option<String> {
        self.pattern_prefix.clone()
    }
}

/// An automaton backed by a regular expression.
#[derive(Debug, Clone)]
pub struct RegexAutomaton {
    regex: Regex,
    #[allow(dead_code)]
    pattern: String,
    initial_seek_term: Option<String>,
}

impl RegexAutomaton {
    /// Create a new regex automaton.
    pub fn new(pattern: &str) -> Result<Self> {
        let regex = Regex::new(pattern).map_err(|e| {
            crate::error::LaurusError::analysis(format!("Invalid regexp pattern: {e}"))
        })?;

        let initial_seek_term = Self::required_prefix(pattern);

        Ok(RegexAutomaton {
            regex,
            pattern: pattern.to_string(),
            initial_seek_term,
        })
    }

    /// Create from existing compiled regex.
    pub fn from_regex(regex: Regex, pattern: String) -> Self {
        let initial_seek_term = Self::required_prefix(&pattern);
        RegexAutomaton {
            regex,
            pattern,
            initial_seek_term,
        }
    }

    /// The literal prefix of every string `pattern` matches, if it has one.
    ///
    /// It is read off the parsed pattern, since a character after `^` is not
    /// part of every match when it is optional, when the pattern has an
    /// alternative, or when it is an escape (Issue #1740).
    fn required_prefix(pattern: &str) -> Option<String> {
        let hir = regex_syntax::parse(pattern).ok()?;
        // Unless it is anchored, a match may start anywhere in the string.
        if !hir.properties().look_set_prefix().contains(Look::Start) {
            return None;
        }
        let literals = Extractor::new().kind(ExtractKind::Prefix).extract(&hir);
        let prefix = literals.longest_common_prefix()?;
        // The common bytes of two characters may end inside them.
        let prefix = match std::str::from_utf8(prefix) {
            Ok(prefix) => prefix,
            Err(e) => std::str::from_utf8(&prefix[..e.valid_up_to()]).ok()?,
        };
        (!prefix.is_empty()).then(|| prefix.to_string())
    }
}

impl Automaton for RegexAutomaton {
    fn matches(&self, candidate: &str) -> bool {
        self.regex.is_match(candidate)
    }

    fn initial_seek_term(&self) -> Option<String> {
        self.initial_seek_term.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_automaton() {
        let automaton = LevenshteinAutomaton::new("hello", 1, 0, true);

        assert!(automaton.matches("hello")); // exact match
        assert!(automaton.matches("helo")); // 1 deletion
        assert!(automaton.matches("hallo")); // 1 substitution
        assert!(automaton.matches("helllo")); // 1 insertion
        assert!(automaton.matches("ehllo")); // 1 transposition

        assert!(!automaton.matches("world")); // too different
        assert!(!automaton.matches("hi")); // too different (2 edits)
    }

    /// The distance is that of the edits, within the maximum, and of the
    /// strings with the required prefix only (Issue #1740).
    #[test]
    fn a_levenshtein_automaton_tells_the_distance() {
        for (prefix_length, transpositions, candidate, expected) in [
            (0, true, "hello", Some(0)),
            (0, true, "hallo", Some(1)),
            (0, true, "ehllo", Some(1)),
            (0, false, "ehllo", Some(2)),
            (0, true, "hxllx", Some(2)),
            (0, true, "xxllx", None),
            (2, true, "hello", Some(0)),
            (2, true, "hello!", Some(1)),
            (2, true, "hallo", None),
            (2, true, "", None),
        ] {
            let automaton = LevenshteinAutomaton::new("hello", 2, prefix_length, transpositions);
            assert_eq!(
                automaton.distance(candidate),
                expected,
                "{candidate}, prefix {prefix_length}, transpositions {transpositions}"
            );
            assert_eq!(automaton.matches(candidate), expected.is_some());
        }
    }

    /// The prefix is one that every match starts with (Issue #1740).
    #[test]
    fn a_regexps_prefix_is_that_of_every_match() {
        for (pattern, expected) in [
            ("^abc.*", Some("abc")),
            (r"^abc\.def", Some("abc.def")),
            (r"\Aabc", Some("abc")),
            ("^a(b|c)", Some("a")),
            ("^(?:ab|ac).*", Some("a")),
            ("^(?:ab|cd)", None),
            ("^abc?$", Some("ab")),
            ("^ab*c", Some("a")),
            ("^ab{0,1}c", Some("a")),
            ("^ab+c", Some("ab")),
            ("^a{3}b", Some("aaab")),
            ("^x|z$", None),
            ("^ab|^ac", Some("a")),
            ("^ab|^cd", None),
            (r"^\d+$", None),
            (r"^a\d", Some("a")),
            (r"^a\x62c", Some("abc")),
            ("^[a-c]", None),
            ("^[a]b", Some("ab")),
            (".*abc", None),
            ("abc", None),
            ("^", None),
            ("^$", None),
            ("", None),
            ("(^a)?b", None),
            ("(?i)^ab", None),
            ("(?m)^ab", None),
            ("^über", Some("über")),
            // The two share their first byte only.
            ("^(?:ü|é)a", None),
            ("^x(?:ü|é)a", Some("x")),
        ] {
            let automaton = RegexAutomaton::new(pattern).unwrap();
            assert_eq!(
                automaton.initial_seek_term().as_deref(),
                expected,
                "{pattern}"
            );
        }
    }

    #[test]
    fn a_prefix_automaton_matches_the_strings_with_its_prefix() {
        let automaton = PrefixAutomaton::new("he");
        assert!(automaton.matches("he"));
        assert!(automaton.matches("hello"));
        assert!(!automaton.matches("h"));
        assert!(!automaton.matches("the"));
        assert_eq!(automaton.initial_seek_term().as_deref(), Some("he"));
        let automaton = PrefixAutomaton::new("");
        assert!(automaton.matches(""));
        assert!(automaton.matches("hello"));
        assert_eq!(automaton.initial_seek_term(), None);
    }

    #[test]
    fn test_prefix_constraint() {
        let automaton = LevenshteinAutomaton::new("hello", 2, 2, true);

        assert!(automaton.matches("hello")); // exact match
        assert!(automaton.matches("heLLo")); // 2 edits, prefix "he" matches

        assert!(!automaton.matches("xello")); // prefix doesn't match
        assert!(!automaton.matches("world")); // prefix doesn't match
    }
}
