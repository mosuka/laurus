//! Conversions between PHP values and Laurus types.

use chrono::{DateTime, Utc};
use ext_php_rs::boxed::ZBox;
use ext_php_rs::convert::FromZval;
use ext_php_rs::prelude::PhpResult;
use ext_php_rs::types::array::ArrayKey;
use ext_php_rs::types::{ZendHashTable, Zval};
use laurus::{DataValue, Document};

/// Convert a PHP associative array (HashTable) to a [`Document`].
///
/// Each key must be a string; values are converted via [`zval_to_data_value`].
///
/// # Arguments
///
/// * `ht` - PHP HashTable (associative array) mapping field names to values.
///
/// # Returns
///
/// A `Document` with fields populated from the array.
pub fn hashtable_to_document(ht: &ZendHashTable) -> PhpResult<Document> {
    let mut builder = Document::builder();
    for (key, val) in ht.iter() {
        let field = match key {
            ArrayKey::String(s) => s,
            ArrayKey::Str(s) => s.to_string(),
            ArrayKey::ZendString(s) => s
                .as_str()
                .map_err(|_| "array key must be valid UTF-8")?
                .to_string(),
            ArrayKey::Long(_) => {
                return Err("array key must be a string, not an integer".into());
            }
        };
        let dv = zval_to_data_value(val)?;
        builder = builder.add_field(&field, dv);
    }
    Ok(builder.build())
}

/// Convert a PHP [`Zval`] to a [`DataValue`].
///
/// # Type mapping
///
/// | PHP type                                  | DataValue variant    |
/// |-------------------------------------------|----------------------|
/// | `null`                                    | `Null`               |
/// | `bool`                                    | `Bool`               |
/// | `int`                                     | `Int64`              |
/// | `float`                                   | `Float64`            |
/// | `string`                                  | `Text`               |
/// | `array` of ints (sequential)              | `Int64Array`         |
/// | `array` of numerics (sequential)          | `Float64Array` (vector fields cast either array to `Vector`; empty is an empty `Int64Array`) |
/// | `array` of `lat`/`lon` arrays (sequential) | `GeoArray` (multi-valued geo, #1174) |
/// | `array` of `x`/`y`/`z` arrays (sequential) | `GeoEcefArray`      |
/// | `array` with `"lat"`, `"lon"` keys        | `Geo`                |
/// | `array` with `"x"`, `"y"`, `"z"` keys     | `GeoEcef`            |
/// | ISO 8601 string (fallback)                | `DateTime`           |
///
/// # Arguments
///
/// * `zv` - PHP Zval to convert.
///
/// # Returns
///
/// The corresponding `DataValue`, or an error if the type is unsupported.
pub fn zval_to_data_value(zv: &Zval) -> PhpResult<DataValue> {
    // null
    if zv.is_null() {
        return Ok(DataValue::Null);
    }
    // bool
    if zv.is_bool() {
        let b = bool::from_zval(zv).ok_or("failed to convert bool")?;
        return Ok(DataValue::Bool(b));
    }
    // int (long)
    if zv.is_long() {
        let i = i64::from_zval(zv).ok_or("failed to convert int")?;
        return Ok(DataValue::Int64(i));
    }
    // float (double)
    if zv.is_double() {
        let f = f64::from_zval(zv).ok_or("failed to convert float")?;
        return Ok(DataValue::Float64(f));
    }
    // string
    if zv.is_string() {
        let s = String::from_zval(zv).ok_or("failed to convert string")?;
        // Try ISO 8601 datetime parse
        if let Ok(dt) = s.parse::<DateTime<Utc>>() {
            return Ok(DataValue::DateTime(dt));
        }
        return Ok(DataValue::Text(s));
    }
    // array
    if zv.is_array() {
        let ht = zv.array().ok_or("failed to get array")?;

        // Check for geo: associative array with "lat" and "lon" keys
        if let (Some(lat_zv), Some(lon_zv)) = (ht.get("lat"), ht.get("lon")) {
            let lat = f64::from_zval(lat_zv).ok_or("'lat' must be a float")?;
            let lon = f64::from_zval(lon_zv).ok_or("'lon' must be a float")?;
            let point = laurus::lexical::GeoPoint::try_new(lat, lon)
                .map_err(|e| format!("invalid geo point: {e}"))?;
            return Ok(DataValue::Geo(point));
        }

        // Check for geo3d: associative array with "x", "y", "z" keys (must
        // come after the {lat, lon} check so existing 2D Geo semantics are
        // preserved).
        if let (Some(x_zv), Some(y_zv), Some(z_zv)) = (ht.get("x"), ht.get("y"), ht.get("z")) {
            let x = f64::from_zval(x_zv).ok_or("'x' must be a float")?;
            let y = f64::from_zval(y_zv).ok_or("'y' must be a float")?;
            let z = f64::from_zval(z_zv).ok_or("'z' must be a float")?;
            return Ok(DataValue::GeoEcef(laurus::GeoEcefPoint::new(x, y, z)));
        }

        // Otherwise a sequential numeric array. It becomes the most
        // informative array shape and the core's schema-aware `coerce_value`
        // routes it: vector fields cast to `Vector`, multi-valued numeric
        // fields keep the array, single-valued fields reject it (#1178).
        // Emitting `Vector` here unconditionally made multi-valued numeric
        // fields unreachable from PHP. An empty array becomes an empty
        // `Int64Array`: `coerce_to_vector` casts it to the same empty `Vector`
        // as before, while multi-valued numeric fields — which reject `Vector`
        // outright — now accept it.
        if ht.is_empty() {
            return Ok(DataValue::Int64Array(Vec::new()));
        }
        // A sequential array of arrays is a multi-valued geo field (#1174):
        // the outer array has no `lat`/`x` keys so the marker checks above
        // fall through to here, and each element takes the same
        // associative-array path as a single point.
        if ht.iter().all(|(_, val)| val.is_array()) {
            return zval_array_of_arrays_to_geo_array(ht);
        }
        if ht.iter().all(|(_, val)| val.is_long()) {
            let mut ints = Vec::with_capacity(ht.len());
            for (_, val) in ht.iter() {
                ints.push(i64::from_zval(val).ok_or("integer array elements must be int")?);
            }
            return Ok(DataValue::Int64Array(ints));
        }
        // `f64::from_zval` only accepts PHP doubles, so a mixed `[1, 2.5]`
        // array needs its int elements widened explicitly.
        let mut floats = Vec::with_capacity(ht.len());
        for (_, val) in ht.iter() {
            let f = if val.is_long() {
                i64::from_zval(val).map(|i| i as f64)
            } else {
                f64::from_zval(val)
            };
            floats.push(f.ok_or("numeric array elements must be numeric")?);
        }
        return Ok(DataValue::Float64Array(floats));
    }

    Err(format!(
        "cannot convert PHP value of type {:?} to DataValue",
        zv.get_type()
    )
    .into())
}

/// Convert a non-empty array whose elements are all arrays into a
/// [`DataValue::GeoArray`] (all `lat`/`lon`) or [`DataValue::GeoEcefArray`]
/// (all `x`/`y`/`z`), rejecting a mix or an element of any other shape.
fn zval_array_of_arrays_to_geo_array(ht: &ZendHashTable) -> PhpResult<DataValue> {
    const MIXED: &str =
        "an array of arrays must be all ['lat', 'lon'] or all ['x', 'y', 'z'] geo points";
    let mut geo = Vec::with_capacity(ht.len());
    let mut ecef = Vec::with_capacity(ht.len());
    for (_, val) in ht.iter() {
        match zval_to_data_value(val)? {
            DataValue::Geo(p) => geo.push(p),
            DataValue::GeoEcef(p) => ecef.push(p),
            _ => return Err(MIXED.into()),
        }
    }
    match (geo.is_empty(), ecef.is_empty()) {
        (false, true) => Ok(DataValue::GeoArray(geo)),
        (true, false) => Ok(DataValue::GeoEcefArray(ecef)),
        _ => Err(MIXED.into()),
    }
}

/// Convert a [`Document`] to a PHP associative array (HashTable).
///
/// # Arguments
///
/// * `doc` - Document to convert.
///
/// # Returns
///
/// A `ZendHashTable` mapping field names to PHP values.
pub fn document_to_hashtable(doc: &Document) -> PhpResult<ZBox<ZendHashTable>> {
    let mut ht = ZendHashTable::new();
    for (field, value) in &doc.fields {
        let zv = data_value_to_zval(value)?;
        ht.insert(field.as_str(), zv)
            .map_err(|_| format!("failed to insert field '{field}' into array"))?;
    }
    Ok(ht)
}

/// Convert a [`DataValue`] to a PHP [`Zval`].
///
/// # Arguments
///
/// * `value` - DataValue to convert.
///
/// # Returns
///
/// The corresponding PHP Zval.
pub fn data_value_to_zval(value: &DataValue) -> PhpResult<Zval> {
    let mut zv = Zval::new();
    match value {
        DataValue::Null => {
            zv.set_null();
        }
        DataValue::Bool(b) => {
            zv.set_bool(*b);
        }
        DataValue::Int64(i) => {
            zv.set_long(*i);
        }
        DataValue::Float64(f) => {
            zv.set_double(*f);
        }
        DataValue::Text(s) => {
            zv.set_string(s, false)
                .map_err(|_| "failed to set string")?;
        }
        DataValue::Bytes(b, _mime) => {
            zv.set_binary(b.clone());
        }
        DataValue::Vector(v) => {
            let mut arr = ZendHashTable::new();
            for (i, &f) in v.iter().enumerate() {
                let mut fzv = Zval::new();
                fzv.set_double(f as f64);
                arr.insert_at_index(i as i64, fzv)
                    .map_err(|_| "failed to insert vector element")?;
            }
            zv.set_hashtable(arr);
        }
        DataValue::DateTime(dt) => {
            zv.set_string(&dt.to_rfc3339(), false)
                .map_err(|_| "failed to set datetime string")?;
        }
        DataValue::Geo(p) => {
            let mut arr = ZendHashTable::new();
            let mut lat_zv = Zval::new();
            lat_zv.set_double(p.lat);
            let mut lon_zv = Zval::new();
            lon_zv.set_double(p.lon);
            arr.insert("lat", lat_zv)
                .map_err(|_| "failed to insert lat")?;
            arr.insert("lon", lon_zv)
                .map_err(|_| "failed to insert lon")?;
            zv.set_hashtable(arr);
        }
        DataValue::GeoEcef(p) => {
            let mut arr = ZendHashTable::new();
            let mut x_zv = Zval::new();
            x_zv.set_double(p.x);
            let mut y_zv = Zval::new();
            y_zv.set_double(p.y);
            let mut z_zv = Zval::new();
            z_zv.set_double(p.z);
            arr.insert("x", x_zv).map_err(|_| "failed to insert x")?;
            arr.insert("y", y_zv).map_err(|_| "failed to insert y")?;
            arr.insert("z", z_zv).map_err(|_| "failed to insert z")?;
            zv.set_hashtable(arr);
        }
        DataValue::Int64Array(values) => {
            let mut arr = ZendHashTable::new();
            for v in values {
                let mut item = Zval::new();
                item.set_long(*v);
                arr.push(item).map_err(|_| "failed to push integer")?;
            }
            zv.set_hashtable(arr);
        }
        DataValue::Float64Array(values) => {
            let mut arr = ZendHashTable::new();
            for v in values {
                let mut item = Zval::new();
                item.set_double(*v);
                arr.push(item).map_err(|_| "failed to push float")?;
            }
            zv.set_hashtable(arr);
        }
        // Arrays of the same associative arrays the single-valued arms
        // produce, so the output feeds back into `zval_to_data_value`.
        DataValue::GeoArray(points) => {
            let mut arr = ZendHashTable::new();
            for p in points {
                arr.push(data_value_to_zval(&DataValue::Geo(*p))?)
                    .map_err(|_| "failed to push geo point")?;
            }
            zv.set_hashtable(arr);
        }
        DataValue::GeoEcefArray(points) => {
            let mut arr = ZendHashTable::new();
            for p in points {
                arr.push(data_value_to_zval(&DataValue::GeoEcef(*p))?)
                    .map_err(|_| "failed to push geo3d point")?;
            }
            zv.set_hashtable(arr);
        }
    }
    Ok(zv)
}
