//! Issue #1600: a range on a field without a BKD tree matches the same
//! documents as on a field with one, through `Engine::search`.
//!
//! A segment builds a BKD tree for a numeric or datetime field only when the
//! field is indexed. Without one, `NumericRangeQuery` (and
//! `DateTimeRangeQuery`, which delegates to it) scans the stored documents,
//! and that scan compared the big-endian bytes of the values, so negative
//! values and bounds, `-0.0` and instants before 1970 missed. Each field kind
//! here is in the schema twice, `*_idx` with a BKD tree and `*_scan` with
//! `indexed: false, stored: true`, plus a `Text` field, which never has a
//! tree. Every range must find the documents written in the test on both.

use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use laurus::lexical::{DateTimeRangeQuery, NumericRangeQuery, Query};
use laurus::storage::memory::MemoryStorage;
use laurus::{
    DateTimeOption, Document, Engine, FloatOption, IntegerOption, LexicalSearchQuery, Schema,
    SearchRequestBuilder, TextOption,
};

/// A UTC instant.
fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, second)
        .unwrap()
}

/// The documents: `(id, float, integer, instant)`. The `Text` field holds
/// the float as Rust writes it (`-0` for -0.0).
fn docs() -> Vec<(&'static str, f64, i64, DateTime<Utc>)> {
    vec![
        ("n5", -5.0, -5, at(1960, 6, 1, 0, 0, 0)),
        ("h", -1.5, -2, at(1966, 1, 1, 0, 0, 0)),
        ("n1", -1.0, -1, at(1969, 12, 31, 23, 59, 59)),
        ("z", -0.0, 0, at(1970, 1, 1, 0, 0, 0)),
        ("p3", 3.0, 3, at(1975, 1, 1, 0, 0, 0)),
    ]
}

/// An engine over the fields of the reproduction in the Issue, each kind
/// with and without a BKD tree, holding [`docs`] in one segment.
async fn engine() -> Engine {
    let multi = |indexed| FloatOption {
        indexed,
        stored: true,
        multi_valued: true,
        doc_values: true,
    };
    let schema = Schema::builder()
        .add_float_field("f_idx", FloatOption::default())
        .add_float_field("f_scan", FloatOption::default().indexed(false))
        .add_float_field("fa_idx", multi(true))
        .add_float_field("fa_scan", multi(false))
        .add_integer_field("i_idx", IntegerOption::default())
        .add_integer_field("i_scan", IntegerOption::default().indexed(false))
        .add_datetime_field("d_idx", DateTimeOption::default())
        .add_datetime_field("d_scan", DateTimeOption::default().indexed(false))
        .add_text_field("t_scan", TextOption::default())
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    for (id, float, integer, instant) in docs() {
        let doc = Document::builder()
            .add_float("f_idx", float)
            .add_float("f_scan", float)
            .add_float64_array("fa_idx", vec![float])
            .add_float64_array("fa_scan", vec![float])
            .add_integer("i_idx", integer)
            .add_integer("i_scan", integer)
            .add_datetime("d_idx", instant)
            .add_datetime("d_scan", instant)
            .add_text("t_scan", float.to_string())
            .build();
        engine.put_document(id, doc).await.unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

/// The ids `request` finds, sorted, or the error it returns.
async fn ids(engine: &Engine, request: SearchRequestBuilder) -> Result<Vec<String>, String> {
    let mut ids: Vec<String> = engine
        .search(request.build())
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|result| result.id)
        .collect();
    ids.sort();
    Ok(ids)
}

/// A lexical search for `query`.
fn object(query: Box<dyn Query>) -> SearchRequestBuilder {
    SearchRequestBuilder::new().lexical_query(LexicalSearchQuery::Obj(query))
}

/// Record in `wrong` a search whose ids are not `want`.
fn check(wrong: &mut Vec<String>, name: String, got: Result<Vec<String>, String>, want: &[&str]) {
    let want: Vec<String> = want.iter().map(|id| id.to_string()).collect();
    if got.as_ref() != Ok(&want) {
        wrong.push(format!("{name}: got {got:?}, want {want:?}"));
    }
}

/// A numeric range: as the DSL writes it, as bounds for the Rust API
/// (both inclusive or both exclusive), and the ids it should find on the
/// float fields and on the integer fields.
struct NumericCase {
    dsl: &'static str,
    lower: Option<f64>,
    upper: Option<f64>,
    inclusive: bool,
    floats: &'static [&'static str],
    integers: &'static [&'static str],
}

const NUMERIC_CASES: &[NumericCase] = &[
    NumericCase {
        dsl: "[-5 TO 5]",
        lower: Some(-5.0),
        upper: Some(5.0),
        inclusive: true,
        floats: &["h", "n1", "n5", "p3", "z"],
        integers: &["h", "n1", "n5", "p3", "z"],
    },
    NumericCase {
        dsl: "[-3 TO 0]",
        lower: Some(-3.0),
        upper: Some(0.0),
        inclusive: true,
        floats: &["h", "n1", "z"],
        integers: &["h", "n1", "z"],
    },
    NumericCase {
        dsl: "[0 TO 5]",
        lower: Some(0.0),
        upper: Some(5.0),
        inclusive: true,
        floats: &["p3", "z"],
        integers: &["p3", "z"],
    },
    NumericCase {
        dsl: "{-5 TO 3}",
        lower: Some(-5.0),
        upper: Some(3.0),
        inclusive: false,
        floats: &["h", "n1", "z"],
        integers: &["h", "n1", "z"],
    },
    NumericCase {
        dsl: "[* TO -2]",
        lower: None,
        upper: Some(-2.0),
        inclusive: true,
        floats: &["n5"],
        integers: &["h", "n5"],
    },
    NumericCase {
        dsl: "[-1 TO *]",
        lower: Some(-1.0),
        upper: None,
        inclusive: true,
        floats: &["n1", "p3", "z"],
        integers: &["n1", "p3", "z"],
    },
    NumericCase {
        dsl: "[-1.5 TO -1.5]",
        lower: Some(-1.5),
        upper: Some(-1.5),
        inclusive: true,
        floats: &["h"],
        integers: &[],
    },
];

/// Issue #1600: numeric ranges find the same documents with and without a
/// BKD tree, through the DSL, a `Float` `NumericRangeQuery` and, for an
/// inclusive range with whole-number bounds, `i64_range` (the bindings'
/// integer range). `t_scan` holds the floats as text.
#[tokio::test(flavor = "multi_thread")]
async fn numeric_ranges_find_the_same_documents_without_a_bkd_tree() {
    let engine = engine().await;
    let fields: [(&str, bool); 7] = [
        ("f_idx", false),
        ("f_scan", false),
        ("fa_idx", false),
        ("fa_scan", false),
        ("t_scan", false),
        ("i_idx", true),
        ("i_scan", true),
    ];
    let mut wrong = Vec::new();
    for case in NUMERIC_CASES {
        for (field, integer) in fields {
            let want = if integer { case.integers } else { case.floats };
            let dsl = format!("{field}:{}", case.dsl);
            let got = ids(&engine, SearchRequestBuilder::new().query_dsl(&dsl)).await;
            check(&mut wrong, format!("DSL {dsl}"), got, want);

            let float = NumericRangeQuery::new(
                field,
                laurus::lexical::core::field::NumericType::Float,
                case.lower,
                case.upper,
                case.inclusive,
                case.inclusive,
            );
            let got = ids(&engine, object(Box::new(float))).await;
            check(&mut wrong, format!("f64 {field} {}", case.dsl), got, want);

            let whole = |bound: Option<f64>| bound.is_none_or(|b| b.fract() == 0.0);
            if case.inclusive && whole(case.lower) && whole(case.upper) {
                let integer_range = NumericRangeQuery::i64_range(
                    field,
                    case.lower.map(|b| b as i64),
                    case.upper.map(|b| b as i64),
                );
                let got = ids(&engine, object(Box::new(integer_range))).await;
                check(&mut wrong, format!("i64 {field} {}", case.dsl), got, want);
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1600: datetime ranges over instants before 1970, whose BKD points
/// are negative, find the same documents with and without a BKD tree,
/// through the DSL and `DateTimeRangeQuery`.
#[tokio::test(flavor = "multi_thread")]
async fn datetime_ranges_before_1970_find_the_same_documents_without_a_bkd_tree() {
    let engine = engine().await;
    type Build = fn(&str) -> DateTimeRangeQuery;
    let cases: [(&str, Build, &[&str]); 5] = [
        (
            "[1960-01-01 TO 1980-01-01]",
            |f| DateTimeRangeQuery::between(f, at(1960, 1, 1, 0, 0, 0), at(1980, 1, 1, 0, 0, 0)),
            &["h", "n1", "n5", "p3", "z"],
        ),
        (
            "[1960-01-01 TO 1969-12-31T23:59:59Z]",
            |f| {
                DateTimeRangeQuery::between(
                    f,
                    at(1960, 1, 1, 0, 0, 0),
                    at(1969, 12, 31, 23, 59, 59),
                )
            },
            &["h", "n1", "n5"],
        ),
        (
            "[* TO 1965-01-01]",
            |f| DateTimeRangeQuery::on_or_before(f, at(1965, 1, 1, 0, 0, 0)),
            &["n5"],
        ),
        (
            "[1969-01-01 TO 1970-01-01]",
            |f| DateTimeRangeQuery::between(f, at(1969, 1, 1, 0, 0, 0), at(1970, 1, 1, 0, 0, 0)),
            &["n1", "z"],
        ),
        (
            "{1966-01-01 TO 1970-01-01}",
            |f| {
                DateTimeRangeQuery::new(
                    f,
                    Some(at(1966, 1, 1, 0, 0, 0)),
                    Some(at(1970, 1, 1, 0, 0, 0)),
                    false,
                    false,
                )
            },
            &["n1"],
        ),
    ];
    let mut wrong = Vec::new();
    for (range, build, want) in cases {
        for field in ["d_idx", "d_scan"] {
            let dsl = format!("{field}:{range}");
            let got = ids(&engine, SearchRequestBuilder::new().query_dsl(&dsl)).await;
            check(&mut wrong, format!("DSL {dsl}"), got, want);

            let got = ids(&engine, object(Box::new(build(field)))).await;
            check(
                &mut wrong,
                format!("DateTimeRangeQuery {field} {range}"),
                got,
                want,
            );
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
