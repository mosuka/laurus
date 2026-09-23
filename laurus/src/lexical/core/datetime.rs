//! The single DateTime ⇄ BKD encoding and the query-literal grammar
//! (Issue #1179).
//!
//! A `DateTime` field is indexed as a one-dimensional BKD point. Before
//! #1179 the writer and the standalone document parser encoded that point
//! differently (whole seconds vs. fractional seconds), and the query side
//! had no encoder at all. Everything now goes through
//! [`datetime_to_point`], so a bound and a stored value that denote the
//! same instant always compare equal.

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};

/// Encode a datetime as its BKD point: seconds since the Unix epoch with
/// microsecond fraction, as `f64`.
///
/// Microseconds are the persisted precision of a `DataValue::DateTime`
/// (rkyv `MicroSeconds` in `crate::data`), so the stored-document fallback
/// and the BKD path see the same value. Whole seconds are exact
/// (`1.6e15 µs < 2^53`; the correctly-rounded division yields the integer
/// second, bit-identical to the whole-second points written before #1179),
/// and distinct microsecond instants stay distinct (the ULP at current
/// epochs is 2⁻²² s ≈ 0.24 µs).
pub(crate) fn datetime_to_point(dt: &DateTime<Utc>) -> f64 {
    dt.timestamp_micros() as f64 / 1_000_000.0
}

/// Parse a query-side datetime literal.
///
/// Accepted forms, tried in this order:
///
/// 1. RFC 3339 (`2024-01-01T00:00:00Z`, `2024-01-01T09:00:00.5+09:00`) —
///    normalized to UTC.
/// 2. Naive `YYYY-MM-DDTHH:MM:SS[.fff]` (no offset) — interpreted as UTC.
/// 3. Date only `YYYY-MM-DD` — midnight UTC of that day (Lucene /
///    Elasticsearch default), so `[2024-01-01 TO 2024-12-31]` includes
///    `2024-12-31T00:00:00Z` and nothing later that day.
///
/// Anything else (including bare numbers and `*`) yields `None`; callers
/// decide whether that means "numeric bound", "open bound", or an error.
/// `str::parse::<DateTime<Utc>>` is deliberately not used: its relaxed
/// RFC 3339 grammar still requires an offset and would not cover form 2.
pub(crate) fn parse_datetime_literal(s: &str) -> Option<DateTime<Utc>> {
    let s = s.trim();
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(naive) = s.parse::<NaiveDateTime>() {
        return Some(naive.and_utc());
    }
    if let Ok(date) = s.parse::<NaiveDate>() {
        return Some(date.and_hms_opt(0, 0, 0)?.and_utc());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn datetime_to_point_is_exact_for_whole_seconds() {
        for secs in [
            0i64,
            1_500_000_000,
            1_600_000_000,
            1_700_000_000,
            4_102_444_800,
        ] {
            let dt = Utc.timestamp_opt(secs, 0).unwrap();
            // Bit-identical to the pre-#1179 whole-second encoding.
            assert_eq!(datetime_to_point(&dt).to_bits(), (secs as f64).to_bits());
        }
    }

    #[test]
    fn datetime_to_point_matches_micros_precision() {
        let base = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
        let plus_one_micro = Utc.timestamp_opt(1_700_000_000, 1_000).unwrap();
        let plus_half = Utc.timestamp_opt(1_700_000_000, 500_000_000).unwrap();
        assert!(datetime_to_point(&base) < datetime_to_point(&plus_one_micro));
        assert!(datetime_to_point(&plus_one_micro) < datetime_to_point(&plus_half));
        assert_eq!(datetime_to_point(&plus_half), 1_700_000_000.5);
        // Sub-microsecond detail is not persisted, so it must not create a
        // point the stored value cannot reproduce.
        let plus_nanos = Utc.timestamp_opt(1_700_000_000, 999).unwrap();
        assert_eq!(datetime_to_point(&plus_nanos), datetime_to_point(&base));
    }

    #[test]
    fn parses_rfc3339_with_offset_to_utc() {
        let dt = parse_datetime_literal("2024-01-01T09:00:00+09:00").unwrap();
        assert_eq!(dt, Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap());
        let dt = parse_datetime_literal("2024-01-01T00:00:00.25Z").unwrap();
        assert_eq!(dt.timestamp_subsec_millis(), 250);
        // Surrounding whitespace is tolerated.
        assert!(parse_datetime_literal(" 2024-01-01T00:00:00Z ").is_some());
    }

    #[test]
    fn parses_naive_datetime_as_utc() {
        let dt = parse_datetime_literal("2024-06-15T12:34:56").unwrap();
        assert_eq!(dt, Utc.with_ymd_and_hms(2024, 6, 15, 12, 34, 56).unwrap());
        let dt = parse_datetime_literal("2024-06-15T12:34:56.5").unwrap();
        assert_eq!(dt.timestamp_subsec_millis(), 500);
    }

    #[test]
    fn parses_date_only_as_midnight_utc() {
        let dt = parse_datetime_literal("2024-12-31").unwrap();
        assert_eq!(dt, Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap());
    }

    #[test]
    fn rejects_numbers_star_and_garbage() {
        for s in [
            "*",
            "",
            "2024",
            "1700000000",
            "1.5",
            "yesterday",
            "2024-13-01",
            "2024-01-01 00:00:00",
            "abc",
        ] {
            assert!(parse_datetime_literal(s).is_none(), "{s:?} must not parse");
        }
    }
}
