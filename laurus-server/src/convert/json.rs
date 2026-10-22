//! Readers for the optional keys of a JSON object, shared by the HTTP
//! gateway's conversions and by the JSON parsers of [`super::search`], which
//! the MCP search tools call too (Issue #1523).
//!
//! Each reader takes the object holding the key and names the key in its
//! error, so a value of the wrong type is rejected instead of being read as
//! omitted. A missing key and `null` are `None`.

use serde_json::{Map, Value};

/// Parse a JSON integer that must fit `u32`, naming `key` in the error, so
/// an out-of-range or mistyped value is rejected rather than wrapped or
/// replaced by a default.
fn json_to_u32(value: &Value, key: &str) -> Result<u32, String> {
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| format!("{key} must be an integer between 0 and {}", u32::MAX))
}

/// Read the optional integer `key` of `obj`, which must fit `u32` when
/// present.
///
/// A missing key and `null` are `None`, so the caller can leave the value
/// unset for the engine's default (Issue #1479) instead of reading it as 0.
///
/// This and the other `json_optional_*` helpers take an object rather than
/// a [`Value`], so a caller must check that the value holding `key` is an
/// object first: [`Value::get`] answers `None` for anything else, which
/// read a field option such as `{"text": 1}` as all defaults (Issue #1524).
///
/// # Errors
///
/// Returns an error when the value is present and not `null` but fails
/// [`json_to_u32`] (`16.0`, `"16"`, `-1`, `4294967296`).
pub(crate) fn json_optional_u32(
    obj: &Map<String, Value>,
    key: &str,
) -> Result<Option<u32>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => json_to_u32(value, key).map(Some),
    }
}

/// Parse a JSON integer that must fit `u64`, naming `key` in the error, the
/// `uint64` counterpart of [`json_to_u32`] (Issue #1490).
fn json_to_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{key} must be an integer between 0 and {}", u64::MAX))
}

/// Read the optional integer `key` of `obj`, which must fit `u64` when
/// present; the `uint64` counterpart of [`json_optional_u32`] (Issue
/// #1490).
///
/// # Errors
///
/// Returns an error when the value is present and not `null` but fails
/// [`json_to_u64`] (`500.0`, `"500"`, `-1`, `18446744073709551616`).
pub(crate) fn json_optional_u64(
    obj: &Map<String, Value>,
    key: &str,
) -> Result<Option<u64>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => json_to_u64(value, key).map(Some),
    }
}

/// Read the optional boolean `key` of `obj`, the boolean counterpart of
/// [`json_optional_u32`] (Issue #1511).
///
/// A missing key and `null` are `None`, so the caller decides what an
/// omitted key means: `false` for a plain proto `bool`, or the engine's
/// default for an `optional bool`.
///
/// # Errors
///
/// Returns an error naming `key` when the value is present and not `null`
/// but not `true` or `false` (`"true"`, `1`), instead of reading it as
/// omitted, which made `"dry_run": "true"` apply a field change.
pub(crate) fn json_optional_bool(
    obj: &Map<String, Value>,
    key: &str,
) -> Result<Option<bool>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(format!("{key} must be true or false")),
    }
}

/// Read the optional number `key` of `obj`, the number counterpart of
/// [`json_optional_u32`] (Issue #1524).
///
/// # Errors
///
/// Returns an error naming `key` when the value is present and not `null`
/// but not a number (`"2"`, `[2]`), instead of reading it as omitted.
pub(crate) fn json_optional_f64(
    obj: &Map<String, Value>,
    key: &str,
) -> Result<Option<f64>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be a number")),
    }
}

/// Read the optional string `key` of `obj`, the string counterpart of
/// [`json_optional_u32`] (Issue #1524).
///
/// # Errors
///
/// Returns an error naming `key` when the value is present and not `null`
/// but not a string (`1`, `["a"]`), instead of reading it as omitted.
pub(crate) fn json_optional_str<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
) -> Result<Option<&'a str>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(format!("{key} must be a string")),
    }
}

/// Read the optional object `key` of `obj`, the object counterpart of
/// [`json_optional_u32`] (Issue #1524).
///
/// # Errors
///
/// Returns an error naming `key` when the value is present and not `null`
/// but not an object (`5`, `"x"`, `[]`), instead of reading it as omitted.
pub(crate) fn json_optional_object<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
) -> Result<Option<&'a Map<String, Value>>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(map)) => Ok(Some(map)),
        Some(_) => Err(format!("{key} must be an object")),
    }
}

/// Read the optional array `key` of `obj`, the array counterpart of
/// [`json_optional_u32`] (Issue #1524). The caller reads the items.
///
/// # Errors
///
/// Returns an error naming `key` when the value is present and not `null`
/// but not an array (`"lowercase"`, `{}`), instead of reading it as omitted.
pub(crate) fn json_optional_array<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
) -> Result<Option<&'a Vec<Value>>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => Ok(Some(items)),
        Some(_) => Err(format!("{key} must be an array")),
    }
}

/// Read the optional array of strings `key` of `obj` (Issue #1524).
///
/// # Errors
///
/// Returns an error naming `key` when the value is present and not `null`
/// but not an array, or has an item that is not a string (`[1, "title"]`),
/// instead of reading it as omitted or dropping the item.
pub(crate) fn json_optional_string_array(
    obj: &Map<String, Value>,
    key: &str,
) -> Result<Option<Vec<String>>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => json_to_string_array(value)
            .map(Some)
            .ok_or_else(|| format!("{key} must be an array of strings")),
    }
}

/// The strings of `value`, or `None` when it is not an array or has an item
/// that is not a string.
pub(crate) fn json_to_string_array(value: &Value) -> Option<Vec<String>> {
    value
        .as_array()?
        .iter()
        .map(|item| item.as_str().map(str::to_string))
        .collect()
}
