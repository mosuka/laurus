//! Conversions between Ruby values and Laurus types.

use chrono::{DateTime, Utc};
use laurus::{DataValue, Document};
use magnus::prelude::*;
use magnus::r_hash::ForEach;
use magnus::{Error, RArray, RHash, RString, Ruby, Symbol, Value};

/// Convert a Ruby `Hash` to a [`Document`].
///
/// Each key must be a `String` or `Symbol`; values are converted via
/// [`rb_to_data_value`].
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `hash` - Ruby Hash mapping field names to values.
///
/// # Returns
///
/// A `Document` with fields populated from the Hash.
pub fn hash_to_document(ruby: &Ruby, hash: RHash) -> Result<Document, Error> {
    let mut builder = Document::builder();
    hash.foreach(|key: Value, value: Value| {
        let field: String = if key.is_kind_of(ruby.class_symbol()) {
            let sym = Symbol::from_value(key)
                .ok_or_else(|| Error::new(ruby.exception_type_error(), "expected Symbol key"))?;
            sym.name()?.to_string()
        } else {
            let s = RString::from_value(key).ok_or_else(|| {
                Error::new(
                    ruby.exception_type_error(),
                    "hash key must be String or Symbol",
                )
            })?;
            s.to_string()?
        };
        let dv = rb_to_data_value(ruby, value)?;
        builder = std::mem::take(&mut builder).add_field(&field, dv);
        Ok(ForEach::Continue)
    })?;
    Ok(builder.build())
}

/// Convert a Ruby value to a [`DataValue`].
///
/// # Type mapping
///
/// | Ruby type                     | DataValue variant    |
/// |-------------------------------|----------------------|
/// | `nil`                         | `Null`               |
/// | `true` / `false`              | `Bool`               |
/// | `Integer`                     | `Int64`              |
/// | `Float`                       | `Float64`            |
/// | `String`                      | `Text`               |
/// | `Array` of `Integer`          | `Int64Array`         |
/// | `Array` of numerics           | `Float64Array` (vector fields cast either array to `Vector`; empty is an empty `Int64Array`) |
/// | `Array` of `lat`/`lon` Hashes | `GeoArray` (multi-valued geo, #1174) |
/// | `Array` of `x`/`y`/`z` Hashes | `GeoEcefArray`       |
/// | `Hash` with `"lat"`, `"lon"`  | `Geo`                |
/// | `Hash` with `"x"`, `"y"`, `"z"` | `GeoEcef` (3D ECEF Cartesian, meters) |
/// | `Time` / ISO 8601 string      | `DateTime`           |
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `value` - Arbitrary Ruby value to convert.
///
/// # Returns
///
/// The corresponding `DataValue`, or an error if the type is unsupported.
pub fn rb_to_data_value(ruby: &Ruby, value: Value) -> Result<DataValue, Error> {
    // nil → Null
    if value.is_nil() {
        return Ok(DataValue::Null);
    }
    // bool must come before Integer (Ruby true/false are not Integer)
    if value.is_kind_of(ruby.class_true_class()) || value.is_kind_of(ruby.class_false_class()) {
        let b: bool = magnus::TryConvert::try_convert(value)?;
        return Ok(DataValue::Bool(b));
    }
    // Integer → Int64
    if value.is_kind_of(ruby.class_integer()) {
        let i: i64 = magnus::TryConvert::try_convert(value)?;
        return Ok(DataValue::Int64(i));
    }
    // Float → Float64
    if value.is_kind_of(ruby.class_float()) {
        let f: f64 = magnus::TryConvert::try_convert(value)?;
        return Ok(DataValue::Float64(f));
    }
    // String → Text
    if value.is_kind_of(ruby.class_string()) {
        let s: String = magnus::TryConvert::try_convert(value)?;
        return Ok(DataValue::Text(s));
    }
    // Array of numerics → Int64Array / Float64Array (Geo hashes are handled below)
    if value.is_kind_of(ruby.class_array()) {
        let arr = RArray::from_value(value)
            .ok_or_else(|| Error::new(ruby.exception_type_error(), "expected Array"))?;
        // Numeric arrays become the most informative array shape and let the
        // core's schema-aware `coerce_value` route them: vector fields cast
        // to `Vector`, multi-valued numeric fields keep the array,
        // single-valued fields reject it (#1178). Emitting `Vector` here
        // unconditionally made multi-valued numeric fields unreachable from
        // Ruby. An empty array becomes an empty `Int64Array`:
        // `coerce_to_vector` casts it to the same empty `Vector` as before,
        // while multi-valued numeric fields — which reject `Vector` outright
        // — now accept it.
        if arr.is_empty() {
            return Ok(DataValue::Int64Array(Vec::new()));
        }
        // `Value` is a GC-tracked handle, not `TryConvertOwned`, so iterate
        // rather than `to_vec::<Value>()`.
        let elements: Vec<Value> = arr.into_iter().collect();
        // An Array of Hashes is a multi-valued geo field (#1174). Each Hash
        // goes through the same lat/lon and x/y/z checks as a single point
        // below, so key semantics and range validation are shared.
        if elements.iter().all(|v| v.is_kind_of(ruby.class_hash())) {
            return rb_hash_array_to_geo_array(ruby, &elements);
        }
        if elements.iter().all(|v| v.is_kind_of(ruby.class_integer())) {
            let ints = elements
                .iter()
                .map(|v| magnus::TryConvert::try_convert(*v))
                .collect::<Result<Vec<i64>, Error>>()?;
            return Ok(DataValue::Int64Array(ints));
        }
        let floats = elements
            .iter()
            .map(|v| magnus::TryConvert::try_convert(*v))
            .collect::<Result<Vec<f64>, Error>>()?;
        return Ok(DataValue::Float64Array(floats));
    }
    // Hash with "lat"/"lon" → Geo, or with "x"/"y"/"z" → Geo3d
    if value.is_kind_of(ruby.class_hash()) {
        let hash = RHash::from_value(value)
            .ok_or_else(|| Error::new(ruby.exception_type_error(), "expected Hash"))?;
        let lat_val: Option<Value> = hash.get(ruby.str_new("lat"));
        let lon_val: Option<Value> = hash.get(ruby.str_new("lon"));
        if let (Some(lat_v), Some(lon_v)) = (lat_val, lon_val) {
            let lat: f64 = magnus::TryConvert::try_convert(lat_v)?;
            let lon: f64 = magnus::TryConvert::try_convert(lon_v)?;
            let point = laurus::lexical::GeoPoint::try_new(lat, lon).map_err(|e| {
                Error::new(
                    ruby.exception_arg_error(),
                    format!("invalid geo point: {e}"),
                )
            })?;
            return Ok(DataValue::Geo(point));
        }
        // Geo3d (3D ECEF Cartesian point) — must come after the {lat, lon}
        // check so existing 2D Geo semantics are preserved.
        let x_val: Option<Value> = hash.get(ruby.str_new("x"));
        let y_val: Option<Value> = hash.get(ruby.str_new("y"));
        let z_val: Option<Value> = hash.get(ruby.str_new("z"));
        if let (Some(xv), Some(yv), Some(zv)) = (x_val, y_val, z_val) {
            let x: f64 = magnus::TryConvert::try_convert(xv)?;
            let y: f64 = magnus::TryConvert::try_convert(yv)?;
            let z: f64 = magnus::TryConvert::try_convert(zv)?;
            return Ok(DataValue::GeoEcef(laurus::GeoEcefPoint::new(x, y, z)));
        }
        return Err(Error::new(
            ruby.exception_arg_error(),
            "Hash must have 'lat' and 'lon' keys for Geo conversion, or 'x', 'y', 'z' keys for Geo3d",
        ));
    }
    // Try Time → DateTime (call .iso8601 or .to_s)
    if let Ok(s) = value.funcall::<_, _, String>("iso8601", ())
        && let Ok(dt) = s.parse::<DateTime<Utc>>()
    {
        return Ok(DataValue::DateTime(dt));
    }

    Err(Error::new(
        ruby.exception_type_error(),
        format!(
            "cannot convert Ruby value of type {} to DataValue",
            value.class()
        ),
    ))
}

/// Convert a non-empty Array whose elements are all Hashes into a
/// [`DataValue::GeoArray`] (all `lat`/`lon`) or [`DataValue::GeoEcefArray`]
/// (all `x`/`y`/`z`), rejecting a mix or a Hash of any other shape.
fn rb_hash_array_to_geo_array(ruby: &Ruby, elements: &[Value]) -> Result<DataValue, Error> {
    let mixed = || {
        Error::new(
            ruby.exception_arg_error(),
            "an Array of Hashes must be all { lat, lon } or all { x, y, z } geo points",
        )
    };
    let mut geo = Vec::with_capacity(elements.len());
    let mut ecef = Vec::with_capacity(elements.len());
    for &element in elements {
        match rb_to_data_value(ruby, element)? {
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

/// Convert a [`Document`] to a Ruby `Hash`.
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `doc` - Document to convert.
///
/// # Returns
///
/// A Ruby Hash mapping field names to Ruby values.
pub fn document_to_hash(ruby: &Ruby, doc: &Document) -> Result<RHash, Error> {
    let hash = ruby.hash_new();
    for (field, value) in &doc.fields {
        let rb_value = data_value_to_rb(ruby, value)?;
        hash.aset(ruby.str_new(field), rb_value)?;
    }
    Ok(hash)
}

/// Convert a [`DataValue`] to a Ruby value.
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `value` - DataValue to convert.
///
/// # Returns
///
/// The corresponding Ruby value.
pub fn data_value_to_rb(ruby: &Ruby, value: &DataValue) -> Result<Value, Error> {
    match value {
        DataValue::Null => Ok(ruby.qnil().as_value()),
        DataValue::Bool(b) => Ok(if *b {
            ruby.qtrue().as_value()
        } else {
            ruby.qfalse().as_value()
        }),
        DataValue::Int64(i) => Ok(ruby.integer_from_i64(*i).as_value()),
        DataValue::Float64(f) => Ok(ruby.float_from_f64(*f).as_value()),
        DataValue::Text(s) => Ok(ruby.str_new(s).as_value()),
        DataValue::Bytes(b, _mime) => Ok(ruby.str_from_slice(b).as_value()),
        DataValue::Vector(v) => {
            let arr = ruby.ary_new_capa(v.len());
            for &f in v {
                arr.push(ruby.float_from_f64(f as f64))?;
            }
            Ok(arr.as_value())
        }
        DataValue::DateTime(dt) => Ok(ruby.str_new(&dt.to_rfc3339()).as_value()),
        DataValue::Geo(p) => {
            let hash = ruby.hash_new();
            hash.aset(ruby.str_new("lat"), ruby.float_from_f64(p.lat))?;
            hash.aset(ruby.str_new("lon"), ruby.float_from_f64(p.lon))?;
            Ok(hash.as_value())
        }
        DataValue::GeoEcef(p) => {
            let hash = ruby.hash_new();
            hash.aset(ruby.str_new("x"), ruby.float_from_f64(p.x))?;
            hash.aset(ruby.str_new("y"), ruby.float_from_f64(p.y))?;
            hash.aset(ruby.str_new("z"), ruby.float_from_f64(p.z))?;
            Ok(hash.as_value())
        }
        DataValue::Int64Array(arr) => {
            let out = ruby.ary_new_capa(arr.len());
            for &v in arr {
                out.push(ruby.integer_from_i64(v))?;
            }
            Ok(out.as_value())
        }
        DataValue::Float64Array(arr) => {
            let out = ruby.ary_new_capa(arr.len());
            for &v in arr {
                out.push(ruby.float_from_f64(v))?;
            }
            Ok(out.as_value())
        }
        // Arrays of the same Hashes the single-valued arms produce, so the
        // output feeds back into `rb_to_data_value` unchanged.
        DataValue::GeoArray(arr) => {
            let out = ruby.ary_new_capa(arr.len());
            for &p in arr {
                out.push(data_value_to_rb(ruby, &DataValue::Geo(p))?)?;
            }
            Ok(out.as_value())
        }
        DataValue::GeoEcefArray(arr) => {
            let out = ruby.ary_new_capa(arr.len());
            for &p in arr {
                out.push(data_value_to_rb(ruby, &DataValue::GeoEcef(p))?)?;
            }
            Ok(out.as_value())
        }
    }
}
