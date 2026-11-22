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

/// Reject a key of `obj` that is not one of `known`, the keys its reader
/// reads (Issue #1706).
///
/// Each reader of an object with fixed keys calls this first, with the keys
/// it reads, so a misspelled key such as `"indexd"` or `"dry_rn"` is an
/// error instead of being dropped, which left the setting at its default.
/// A key whose value is `null` is not reported, since `null` is the same as
/// omitting a key for every reader here; it sets nothing.
///
/// `what` names the object in the error (`"the text option"`). The keys of
/// a [`Map`] iterate in name order, so the key reported is the first
/// unknown one by name.
///
/// # Errors
///
/// Returns `unknown key "<key>" in <what> (expected <known>)` for the first
/// unknown key.
pub(crate) fn reject_unknown_keys(
    obj: &Map<String, Value>,
    what: &str,
    known: &[&str],
) -> Result<(), String> {
    match obj
        .iter()
        .find(|(key, value)| !value.is_null() && !known.contains(&key.as_str()))
    {
        Some((key, _)) => Err(format!(
            "unknown key \"{key}\" in {what} (expected {})",
            quoted_choices(known)
        )),
        None => Ok(()),
    }
}

/// `"a"`, `"a" or "b"`, or `"a", "b", or "c"`, as the gateway's other
/// errors list the values they expect.
fn quoted_choices(choices: &[&str]) -> String {
    let quoted: Vec<String> = choices.iter().map(|c| format!("\"{c}\"")).collect();
    match quoted.as_slice() {
        [] => "no keys".to_string(),
        [one] => one.clone(),
        [first, second] => format!("{first} or {second}"),
        [rest @ .., last] => format!("{}, or {last}", rest.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    /// Issue #1706: an unknown key is named, with the keys the reader takes.
    #[test]
    fn reject_unknown_keys_names_the_key_and_the_known_ones() {
        let obj = object(json!({"indexed": true, "indexd": false}));
        let err = reject_unknown_keys(&obj, "the text option", &["indexed", "stored"]).unwrap_err();
        assert_eq!(
            err,
            "unknown key \"indexd\" in the text option (expected \"indexed\" or \"stored\")"
        );

        let err =
            reject_unknown_keys(&object(json!({"x": 1})), "it", &["a", "b", "c"]).unwrap_err();
        assert_eq!(
            err,
            "unknown key \"x\" in it (expected \"a\", \"b\", or \"c\")"
        );
    }

    /// Issue #1706: the known keys pass, and so does an unknown key set to
    /// `null`, which is the same as omitting it.
    #[test]
    fn reject_unknown_keys_passes_known_keys_and_null() {
        let known = ["indexed", "stored"];
        assert!(reject_unknown_keys(&object(json!({})), "it", &known).is_ok());
        assert!(reject_unknown_keys(&object(json!({"indexed": true})), "it", &known).is_ok());
        assert!(reject_unknown_keys(&object(json!({"indexd": null})), "it", &known).is_ok());
    }
}
