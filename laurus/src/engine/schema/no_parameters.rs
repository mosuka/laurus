//! Strict reading of a schema variant that takes no parameters (Issue #1700).
//!
//! An internally tagged enum such as [`TokenFilterConfig`] reads a unit
//! variant by discarding every key besides its tag, even when the enum sets
//! `#[serde(deny_unknown_fields)]`. A unit variant that names [`deserialize`]
//! with `#[serde(deserialize_with = ...)]` reads those keys as an empty struct
//! that denies unknown fields instead, so `{"type": "lowercase", "words":
//! ["the"]}` is an error naming `words`. Serialization is not affected: the
//! variant is still written as `{"type": "lowercase"}`.
//!
//! [`TokenFilterConfig`]: super::analyzer::TokenFilterConfig

use serde::{Deserialize, Deserializer};

/// The parameters of a variant that takes none.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoParameters {}

/// Read the keys of a parameterless variant, other than its tag, failing on
/// any.
///
/// # Errors
///
/// Returns the deserializer's error, naming the key, if any key besides the
/// tag is present.
pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<(), D::Error> {
    NoParameters::deserialize(deserializer).map(|NoParameters {}| ())
}
