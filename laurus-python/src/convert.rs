//! Conversions between Python objects and Laurus types.

use chrono::{DateTime, Utc};
use laurus::{DataValue, Document};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

/// Convert a Python `dict` to a [`Document`].
pub fn dict_to_document(py: Python, dict: &Bound<PyDict>) -> PyResult<Document> {
    let mut builder = Document::builder();
    for (key, value) in dict.iter() {
        let field: String = key.extract()?;
        let dv = py_to_data_value(py, &value)?;
        builder = builder.add_field(&field, dv);
    }
    Ok(builder.build())
}

/// Convert a Python value to a [`DataValue`].
///
/// Type mapping:
/// - `None`                → `DataValue::Null`
/// - `bool`                → `DataValue::Bool`  (must be checked before int)
/// - `int`                 → `DataValue::Int64`
/// - `float`               → `DataValue::Float64`
/// - `str`                 → `DataValue::Text`
/// - `bytes`               → `DataValue::Bytes`
/// - `list[int]`           → `DataValue::Int64Array`
/// - `list[float|int]`     → `DataValue::Float64Array` (vector fields cast
///   either array to `Vector` downstream; an empty list is an empty `Int64Array`)
/// - `list[(lat, lon)]`    → `DataValue::GeoArray` (multi-valued geo, #1174)
/// - `list[(x, y, z)]`     → `DataValue::GeoEcefArray`
/// - `list[datetime | str]` → `DataValue::DateTimeArray` (multi-valued datetime, #1184)
/// - `list[bool]`          → `DataValue::BoolArray` (multi-valued boolean, #1180)
/// - `(lat, lon)` tuple    → `DataValue::Geo`
/// - `(x, y, z)` tuple     → `DataValue::GeoEcef` (3D ECEF Cartesian, meters)
pub fn py_to_data_value(py: Python, obj: &Bound<PyAny>) -> PyResult<DataValue> {
    if obj.is_none() {
        return Ok(DataValue::Null);
    }
    // bool must come before int because Python bool is a subclass of int
    if obj.is_instance_of::<PyBool>() {
        let b: bool = obj.extract()?;
        return Ok(DataValue::Bool(b));
    }
    if obj.is_instance_of::<PyInt>() {
        let i: i64 = obj.extract()?;
        return Ok(DataValue::Int64(i));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f: f64 = obj.extract()?;
        return Ok(DataValue::Float64(f));
    }
    if obj.is_instance_of::<PyString>() {
        let s: String = obj.extract()?;
        return Ok(DataValue::Text(s));
    }
    if obj.is_instance_of::<PyBytes>() {
        let b: Vec<u8> = obj.extract()?;
        return Ok(DataValue::Bytes(b, None));
    }
    if obj.is_instance_of::<PyList>() {
        let list = obj.cast::<PyList>()?;
        // Numeric lists become the most informative array shape and let the
        // core's schema-aware `coerce_value` route them: vector fields cast
        // to `Vector`, multi-valued numeric fields keep the array,
        // single-valued fields reject it (#1178). Emitting `Vector` here
        // unconditionally made multi-valued numeric fields unreachable from
        // Python. `bool` is a subclass of `int`, so it is excluded from the
        // all-integer check. An empty list becomes an empty `Int64Array`:
        // `coerce_to_vector` casts it to the same empty `Vector` as before,
        // while multi-valued numeric fields — which reject `Vector` outright
        // — now accept it.
        if list.is_empty() {
            return Ok(DataValue::Int64Array(Vec::new()));
        }
        // A list of tuples is a multi-valued geo field (#1174). Each tuple
        // goes through the same 2-/3-tuple checks as a single point below,
        // so key semantics and range validation are shared; lists of lists
        // are not treated as geo.
        if list.iter().all(|item| item.is_instance_of::<PyTuple>()) {
            return py_tuple_list_to_geo_array(py, list);
        }
        // A list of `str` / `datetime` objects is a multi-valued datetime
        // field (#1184); each element is parsed like a single datetime.
        if list.iter().all(|item| {
            item.is_instance_of::<PyString>() || item.hasattr("isoformat").unwrap_or(false)
        }) {
            return py_datetime_list_to_datetime_array(list);
        }
        // A list of bools is a multi-valued boolean field (#1180). Checked
        // before the integer gate — `bool` is a subclass of `int` — so it no
        // longer falls through to the float path as `[1.0, 0.0]`; the core
        // still widens a `BoolArray` element-wise for numeric fields.
        if list.iter().all(|item| item.is_instance_of::<PyBool>()) {
            let flags: Vec<bool> = list
                .iter()
                .map(|item| item.extract::<bool>())
                .collect::<PyResult<_>>()?;
            return Ok(DataValue::BoolArray(flags));
        }
        let all_ints = list
            .iter()
            .all(|item| item.is_instance_of::<PyInt>() && !item.is_instance_of::<PyBool>());
        if all_ints {
            let ints: Vec<i64> = list
                .iter()
                .map(|item| item.extract::<i64>())
                .collect::<PyResult<_>>()?;
            return Ok(DataValue::Int64Array(ints));
        }
        let floats: Vec<f64> = list
            .iter()
            .map(|item| item.extract::<f64>())
            .collect::<PyResult<_>>()?;
        return Ok(DataValue::Float64Array(floats));
    }
    // Try tuple (lat, lon) for Geo
    if let Ok(tup) = obj.cast::<pyo3::types::PyTuple>()
        && tup.len() == 2
        && let (Ok(lat), Ok(lon)) = (
            tup.get_item(0)?.extract::<f64>(),
            tup.get_item(1)?.extract::<f64>(),
        )
    {
        let point = laurus::lexical::GeoPoint::try_new(lat, lon)
            .map_err(|e| PyValueError::new_err(format!("invalid geo point: {e}")))?;
        return Ok(DataValue::Geo(point));
    }
    // Try tuple (x, y, z) for Geo3d (ECEF Cartesian, meters). Must come
    // after the 2-tuple Geo check so existing semantics are preserved.
    if let Ok(tup) = obj.cast::<pyo3::types::PyTuple>()
        && tup.len() == 3
        && let (Ok(x), Ok(y), Ok(z)) = (
            tup.get_item(0)?.extract::<f64>(),
            tup.get_item(1)?.extract::<f64>(),
            tup.get_item(2)?.extract::<f64>(),
        )
    {
        return Ok(DataValue::GeoEcef(laurus::GeoEcefPoint::new(x, y, z)));
    }
    // Try Python datetime.datetime (anything exposing `isoformat()`).
    if let Ok(dt_str) = obj.call_method0("isoformat")
        && let Ok(s) = dt_str.extract::<String>()
        && let Some(dt) = parse_py_datetime_text(&s)
    {
        return Ok(DataValue::DateTime(dt));
    }
    Err(PyValueError::new_err(format!(
        "Cannot convert Python value of type {} to DataValue",
        obj.get_type().name()?
    )))
}

/// Parse the text form Python hands us for a datetime: an RFC 3339 / ISO
/// 8601 string with an offset, then a naive `YYYY-MM-DDTHH:MM:SS` (as UTC).
fn parse_py_datetime_text(s: &str) -> Option<DateTime<Utc>> {
    if let Ok(dt) = s.parse::<DateTime<Utc>>() {
        return Some(dt);
    }
    // Try without timezone suffix
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
        .ok()
        .map(|ndt| DateTime::<Utc>::from_naive_utc_and_offset(ndt, Utc))
}

/// Convert a non-empty list whose elements are all `str` or expose
/// `isoformat()` into a [`DataValue::DateTimeArray`] (#1184). Each element
/// is parsed exactly like a single datetime; anything else is an error.
fn py_datetime_list_to_datetime_array(list: &Bound<PyList>) -> PyResult<DataValue> {
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let text: String = if let Ok(s) = item.extract::<String>() {
            s
        } else {
            item.call_method0("isoformat")?.extract()?
        };
        let dt = parse_py_datetime_text(&text).ok_or_else(|| {
            PyValueError::new_err(format!(
                "a list of datetimes must be all RFC 3339 / ISO 8601 datetimes \
                 (multi-valued text fields are not supported), got {text:?}"
            ))
        })?;
        out.push(dt);
    }
    Ok(DataValue::DateTimeArray(out))
}

/// Convert a non-empty list whose elements are all tuples into a
/// [`DataValue::GeoArray`] (all `(lat, lon)`) or [`DataValue::GeoEcefArray`]
/// (all `(x, y, z)`), rejecting a mix or a tuple of any other arity.
fn py_tuple_list_to_geo_array(py: Python, list: &Bound<PyList>) -> PyResult<DataValue> {
    let mixed = || {
        PyValueError::new_err("a list of tuples must be all (lat, lon) or all (x, y, z) geo points")
    };
    let mut geo = Vec::with_capacity(list.len());
    let mut ecef = Vec::with_capacity(list.len());
    for item in list.iter() {
        match py_to_data_value(py, &item)? {
            DataValue::Geo(p) => geo.push(p),
            DataValue::GeoEcef(p) => ecef.push(p),
            _ => return Err(mixed()),
        }
    }
    match (geo.is_empty(), ecef.is_empty()) {
        (false, true) => Ok(DataValue::GeoArray(geo)),
        (true, false) => Ok(DataValue::GeoEcefArray(ecef)),
        _ => Err(mixed()),
    }
}

/// Convert a [`Document`] to a Python `dict`.
pub fn document_to_dict(py: Python, doc: &Document) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (field, value) in &doc.fields {
        let py_value = data_value_to_py(py, value)?;
        dict.set_item(field, py_value)?;
    }
    Ok(dict.into_any().unbind())
}

/// Convert a [`DataValue`] to a Python object.
pub fn data_value_to_py(py: Python, value: &DataValue) -> PyResult<Py<PyAny>> {
    match value {
        DataValue::Null => Ok(py.None()),
        DataValue::Bool(b) => Ok((*(*b).into_pyobject(py)?).clone().unbind().into_any()),
        DataValue::Int64(i) => Ok((*i).into_pyobject(py)?.unbind().into_any()),
        DataValue::Float64(f) => Ok((*f).into_pyobject(py)?.unbind().into_any()),
        DataValue::Text(s) => Ok(s.clone().into_pyobject(py)?.unbind().into_any()),
        DataValue::Bytes(b, _mime) => Ok(PyBytes::new(py, b).unbind().into_any()),
        DataValue::Vector(v) => Ok(v.clone().into_pyobject(py)?.unbind().into_any()),
        DataValue::DateTime(dt) => Ok(dt.to_rfc3339().into_pyobject(py)?.unbind().into_any()),
        DataValue::Geo(p) => {
            let tup = pyo3::types::PyTuple::new(py, [p.lat, p.lon])?;
            Ok(tup.unbind().into_any())
        }
        DataValue::GeoEcef(p) => {
            let tup = pyo3::types::PyTuple::new(py, [p.x, p.y, p.z])?;
            Ok(tup.unbind().into_any())
        }
        DataValue::Int64Array(arr) => Ok(arr.clone().into_pyobject(py)?.unbind().into_any()),
        DataValue::Float64Array(arr) => Ok(arr.clone().into_pyobject(py)?.unbind().into_any()),
        // Lists of the same tuples the single-valued arms produce, so the
        // output feeds back into `py_to_data_value` unchanged.
        DataValue::GeoArray(arr) => {
            let items = arr
                .iter()
                .map(|p| PyTuple::new(py, [p.lat, p.lon]))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.unbind().into_any())
        }
        DataValue::GeoEcefArray(arr) => {
            let items = arr
                .iter()
                .map(|p| PyTuple::new(py, [p.x, p.y, p.z]))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.unbind().into_any())
        }
        // A list of the same RFC 3339 strings the single-valued arm produces.
        DataValue::DateTimeArray(arr) => {
            let items: Vec<String> = arr.iter().map(|dt| dt.to_rfc3339()).collect();
            Ok(PyList::new(py, items)?.unbind().into_any())
        }
        // A list of Python bools (#1180).
        DataValue::BoolArray(arr) => Ok(arr.clone().into_pyobject(py)?.unbind().into_any()),
    }
}

/// Maximum nesting depth accepted by [`py_to_json_value`].
///
/// Guards against a self-referential container (e.g. `d = {}; d["x"] = d`)
/// blowing the Rust call stack, which would abort the process rather than
/// raise a Python exception.
const MAX_JSON_VALUE_DEPTH: usize = 32;

/// Convert an arbitrary Python object into a [`serde_json::Value`].
///
/// Used to bridge Python `dict`/`list` literals (e.g. the `tokenizer` /
/// `char_filters` / `token_filters` arguments of `Schema.add_analyzer`)
/// into serde-deserializable JSON so they can be decoded with exactly the
/// same semantics (field defaults, `snake_case` variant names, etc.) as the
/// TOML schema format.
///
/// Type mapping:
/// - `None`         → `Value::Null`
/// - `bool`         → `Value::Bool`  (must be checked before int)
/// - `int`          → `Value::Number` (rejected if it fits neither `i64` nor `u64`)
/// - `float`        → `Value::Number` (rejected if NaN or infinite)
/// - `str`          → `Value::String`
/// - `list`/`tuple` → `Value::Array`
/// - `dict`         → `Value::Object` (keys must be `str`)
///
/// Any other type, or nesting deeper than [`MAX_JSON_VALUE_DEPTH`], is
/// rejected with a `ValueError`/`TypeError`.
pub fn py_to_json_value(obj: &Bound<PyAny>) -> PyResult<serde_json::Value> {
    py_to_json_value_inner(obj, 0)
}

fn py_to_json_value_inner(obj: &Bound<PyAny>, depth: usize) -> PyResult<serde_json::Value> {
    if depth > MAX_JSON_VALUE_DEPTH {
        return Err(PyValueError::new_err(format!(
            "value is nested too deeply (max depth: {MAX_JSON_VALUE_DEPTH})"
        )));
    }
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    // bool must come before int because Python bool is a subclass of int.
    if obj.is_instance_of::<PyBool>() {
        let b: bool = obj.extract()?;
        return Ok(serde_json::Value::Bool(b));
    }
    if obj.is_instance_of::<PyInt>() {
        // Try i64 first, then fall back to u64 for large unsigned values;
        // anything wider than that has no lossless serde_json representation.
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(serde_json::Value::Number(i.into()));
        }
        let u: u64 = obj.extract().map_err(|_| {
            PyValueError::new_err("integer is too large to represent in a schema value")
        })?;
        return Ok(serde_json::Value::Number(u.into()));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f: f64 = obj.extract()?;
        let n = serde_json::Number::from_f64(f)
            .ok_or_else(|| PyValueError::new_err("float value must be finite (not NaN or inf)"))?;
        return Ok(serde_json::Value::Number(n));
    }
    if obj.is_instance_of::<PyString>() {
        let s: String = obj.extract()?;
        return Ok(serde_json::Value::String(s));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let items: Vec<serde_json::Value> = obj
            .try_iter()?
            .map(|item| py_to_json_value_inner(&item?, depth + 1))
            .collect::<PyResult<_>>()?;
        return Ok(serde_json::Value::Array(items));
    }
    if obj.is_instance_of::<PyDict>() {
        let dict = obj.cast::<PyDict>()?;
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let key: String = key.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "dict keys must be str, got {}",
                    key.get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_default()
                ))
            })?;
            map.insert(key, py_to_json_value_inner(&value, depth + 1)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    Err(PyValueError::new_err(format!(
        "Cannot convert Python value of type {} to a schema value",
        obj.get_type().name()?
    )))
}
