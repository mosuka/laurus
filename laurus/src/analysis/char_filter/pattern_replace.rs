use regex::Regex;

use super::{CharFilter, Transformation};
use crate::error::LaurusError;

/// A char filter that replaces characters matching a regex pattern.
pub struct PatternReplaceCharFilter {
    pattern: Regex,
    replacement: String,
}

impl PatternReplaceCharFilter {
    /// Create a new pattern replace char filter.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] if `pattern` is not a valid
    /// regular expression.
    pub fn new(pattern: &str, replacement: &str) -> crate::error::Result<Self> {
        Ok(Self {
            pattern: Regex::new(pattern).map_err(|e| {
                LaurusError::invalid_argument(format!("Invalid regex pattern: {e}"))
            })?,
            replacement: replacement.to_string(),
        })
    }
}

impl CharFilter for PatternReplaceCharFilter {
    fn filter(&self, input: &str) -> (String, Vec<Transformation>) {
        let mut output = String::with_capacity(input.len());
        let mut transformations = Vec::new();
        let mut last_match_end = 0;

        for m in self.pattern.find_iter(input) {
            let match_start = m.start();
            let match_end = m.end();

            // Append unchanged part
            output.push_str(&input[last_match_end..match_start]);

            let replacement_start = output.len();
            output.push_str(&self.replacement);
            let replacement_end = output.len();

            // Record transformation if length changed or content changed
            // Even if lengths are equal, content changed, so we might want to map.
            // But strictly, offset correction focuses on position mapping.
            // If lengths are same, linear mapping works without explicit transformation record?
            // Ideally record all replacements.
            if match_end - match_start != replacement_end - replacement_start {
                transformations.push(Transformation::new(
                    match_start,
                    match_end,
                    replacement_start,
                    replacement_end,
                ));
            }
            // For now, only recording if length diff exists or explicitly for content?
            // Standard approach: Record all substitutions that are not identity.
            // Since we replaced something, let's record it.

            last_match_end = match_end;
        }

        output.push_str(&input[last_match_end..]);

        (output, transformations)
    }

    fn name(&self) -> &'static str {
        "pattern_replace"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_replace() {
        let filter = PatternReplaceCharFilter::new(r"(\d+)", "NUM").unwrap();
        let input = "Year 2024";
        let (output, transformations) = filter.filter(input);
        assert_eq!(output, "Year NUM");
        assert_eq!(transformations.len(), 1);
        assert_eq!(transformations[0].original_start, 5); // "2"
        assert_eq!(transformations[0].original_end, 9); // after "4"
        assert_eq!(transformations[0].new_start, 5); // "N"
        assert_eq!(transformations[0].new_end, 8); // after "M"
    }

    #[test]
    fn test_remove_pattern() {
        let filter = PatternReplaceCharFilter::new(r"-", "").unwrap();
        let input = "123-456-789";
        let (output, transformations) = filter.filter(input);
        assert_eq!(output, "123456789");
        assert_eq!(transformations.len(), 2);
    }
}
