//! Issue #1700: every type a schema is read into rejects a key it does not
//! declare, at every level of the schema and also for a variant that takes
//! no parameters, so that a misspelled key is an error instead of a setting
//! that silently keeps its default.
//!
//! The tests build one schema holding every type and every variant, and
//! insert an unknown key into each object of its serialized form in turn.

use std::collections::{BTreeMap, BTreeSet};

use laurus::{
    AnalyzerDefinition, AnalyzerSpec, BooleanOption, BuiltinAnalyzerSpec, BytesOption,
    CharFilterConfig, DateTimeOption, DistanceMetric, DynamicFieldPolicy, EmbedderDefinition,
    FieldOption, FlatOption, FloatOption, Geo3dOption, GeoOption, HnswOption, IntegerOption,
    IvfOption, MultiVectorOption, MultiVectorStorage, QuantizationMethod, RerankStorageKind,
    Schema, TextOption, TokenFilterConfig, TokenizerConfig,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// The key inserted into each object; no schema type declares it.
const UNKNOWN: &str = "__unknown_key__";

/// A variant name no schema enum has, to make serde list the ones it has.
const NO_SUCH_VARIANT: &str = "__no_such_variant__";

/// A schema holding every type and every variant a schema can contain.
///
/// When a variant is added to one of the schema enums,
/// `fixture_holds_every_variant` fails until it is added here, so that
/// `every_object_rejects_an_unknown_key` covers it too.
fn fixture() -> Schema {
    let mut analyzers = BTreeMap::new();
    analyzers.insert(
        "every_filter".to_string(),
        AnalyzerDefinition {
            char_filters: vec![
                CharFilterConfig::UnicodeNormalization {
                    form: "nfkc".to_string(),
                },
                CharFilterConfig::PatternReplace {
                    pattern: "a+".to_string(),
                    replacement: "a".to_string(),
                },
                // A mapping is keyed by the text it replaces, so its keys,
                // `type` among them, are not schema keys.
                CharFilterConfig::Mapping {
                    mapping: BTreeMap::from([
                        ("type".to_string(), "kind".to_string()),
                        ("ph".to_string(), "f".to_string()),
                    ]),
                },
                CharFilterConfig::JapaneseIterationMark {
                    kanji: true,
                    kana: false,
                },
            ],
            tokenizer: TokenizerConfig::Regex {
                pattern: r"\w+".to_string(),
                gaps: true,
            },
            token_filters: vec![
                TokenFilterConfig::Lowercase,
                TokenFilterConfig::Stop {
                    words: Some(vec!["the".to_string()]),
                },
                // Serialized to JSON as `"words": null`.
                TokenFilterConfig::Stop { words: None },
                TokenFilterConfig::Stem {
                    stem_type: Some("porter".to_string()),
                },
                TokenFilterConfig::Boost { boost: 2.0 },
                TokenFilterConfig::Limit { limit: 8 },
                TokenFilterConfig::Strip,
                TokenFilterConfig::RemoveEmpty,
                TokenFilterConfig::FlattenGraph,
            ],
        },
    );
    for (name, tokenizer) in [
        ("whitespace", TokenizerConfig::Whitespace),
        ("unicode_word", TokenizerConfig::UnicodeWord),
        (
            "ngram",
            TokenizerConfig::Ngram {
                min_gram: 2,
                max_gram: 3,
            },
        ),
        (
            "lindera",
            TokenizerConfig::Lindera {
                mode: "normal".to_string(),
                dict: "/dict".to_string(),
                user_dict: Some("/user_dict".to_string()),
            },
        ),
        ("whole", TokenizerConfig::Whole),
    ] {
        analyzers.insert(
            format!("tokenizer_{name}"),
            AnalyzerDefinition {
                char_filters: Vec::new(),
                tokenizer,
                token_filters: Vec::new(),
            },
        );
    }

    let embedders = BTreeMap::from([
        ("precomputed".to_string(), EmbedderDefinition::Precomputed),
        (
            "bert".to_string(),
            EmbedderDefinition::CandleBert {
                model: "bert".to_string(),
            },
        ),
        (
            "clip".to_string(),
            EmbedderDefinition::CandleClip {
                model: "clip".to_string(),
            },
        ),
        (
            "openai".to_string(),
            EmbedderDefinition::Openai {
                model: "text-embedding-3-small".to_string(),
            },
        ),
        (
            "colbert".to_string(),
            EmbedderDefinition::CandleColbert {
                model: "colbert".to_string(),
                revision: Some("main".to_string()),
                query_maxlen: Some(32),
                doc_maxlen: Some(180),
            },
        ),
    ]);

    let mut fields = BTreeMap::new();
    fields.insert(
        "text_named".to_string(),
        FieldOption::Text(TextOption {
            analyzer: Some(AnalyzerSpec::Named("every_filter".to_string())),
            ..TextOption::default()
        }),
    );
    fields.insert(
        "text_builtin".to_string(),
        FieldOption::Text(TextOption {
            analyzer: Some(AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
                mode: "normal".to_string(),
                dict: "/dict".to_string(),
                user_dict: Some("/user_dict".to_string()),
            })),
            ..TextOption::default()
        }),
    );
    fields.insert(
        "integer".to_string(),
        FieldOption::Integer(IntegerOption::default()),
    );
    fields.insert(
        "float".to_string(),
        FieldOption::Float(FloatOption::default()),
    );
    fields.insert(
        "boolean".to_string(),
        FieldOption::Boolean(BooleanOption::default()),
    );
    fields.insert(
        "date_time".to_string(),
        FieldOption::DateTime(DateTimeOption::default()),
    );
    fields.insert("geo".to_string(), FieldOption::Geo(GeoOption::default()));
    fields.insert(
        "geo3d".to_string(),
        FieldOption::Geo3d(Geo3dOption::default()),
    );
    fields.insert(
        "bytes".to_string(),
        FieldOption::Bytes(BytesOption::default()),
    );
    fields.insert(
        "hnsw".to_string(),
        FieldOption::Hnsw(HnswOption {
            distance: DistanceMetric::Euclidean,
            default_ef_search: Some(64),
            quantizer: QuantizationMethod::ProductQuantization { subvector_count: 4 },
            rerank_storage: Some(RerankStorageKind::F32),
            embedder: Some("precomputed".to_string()),
            pq_codebook_path: Some("/codebook".to_string()),
            ..HnswOption::default()
        }),
    );
    fields.insert(
        "flat".to_string(),
        FieldOption::Flat(FlatOption {
            quantizer: QuantizationMethod::Float16,
            ..FlatOption::default()
        }),
    );
    fields.insert(
        "ivf".to_string(),
        FieldOption::Ivf(IvfOption {
            quantizer: QuantizationMethod::Scalar8Bit,
            ..IvfOption::default()
        }),
    );
    #[cfg(feature = "pq-fastscan")]
    fields.insert(
        "ivf_fastscan".to_string(),
        FieldOption::Ivf(IvfOption {
            quantizer: QuantizationMethod::ProductQuantizationFastScan { subvector_count: 4 },
            ..IvfOption::default()
        }),
    );
    fields.insert(
        "multi_vector".to_string(),
        FieldOption::MultiVector(MultiVectorOption {
            storage: MultiVectorStorage::F16,
            pool_factor: 2,
            embedder: Some("colbert".to_string()),
            ..MultiVectorOption::default()
        }),
    );

    Schema {
        analyzers,
        embedders,
        fields,
        default_fields: vec!["text_named".to_string()],
        dynamic_field_policy: DynamicFieldPolicy::Strict,
        pending_reindex: BTreeSet::from(["flat".to_string()]),
    }
}

/// One step of a path into a JSON value.
#[derive(Clone, Debug)]
enum Step {
    Key(String),
    Index(usize),
}

/// Render `path` as `$.fields.hnsw.Hnsw`, for messages.
fn render(path: &[Step]) -> String {
    let mut out = String::from("$");
    for step in path {
        match step {
            Step::Key(key) => {
                out.push('.');
                out.push_str(key);
            }
            Step::Index(index) => out.push_str(&format!("[{index}]")),
        }
    }
    out
}

fn key_at(path: &[Step], index: usize) -> Option<&str> {
    match path.get(index) {
        Some(Step::Key(key)) => Some(key),
        _ => None,
    }
}

/// Whether the object at `path` is keyed by names the user chose, not by
/// schema keys: the analyzer, embedder and field maps, and a mapping char
/// filter's `mapping`.
fn is_name_map(path: &[Step], parent: Option<&Value>) -> bool {
    if path.len() == 1 && matches!(key_at(path, 0), Some("analyzers" | "embedders" | "fields")) {
        return true;
    }
    matches!(path.last(), Some(Step::Key(key)) if key == "mapping")
        && parent.and_then(|p| p.get("type")) == Some(&json!("mapping"))
}

/// Whether the object at `path` is the wrapper of an externally tagged
/// enum, whose only key is the variant name: a field option
/// (`{"Text": {...}}`) or a quantizer struct variant
/// (`{"ProductQuantization": {...}}`). serde rejects a second key there in
/// its own words; `externally_tagged_wrappers_reject_a_second_key` covers
/// them.
fn is_wrapper(path: &[Step]) -> bool {
    (path.len() == 2 && key_at(path, 0) == Some("fields"))
        || matches!(path.last(), Some(Step::Key(key)) if key == "quantizer")
}

/// Collect the path of every object in `value` into `out`, with whether an
/// unknown key inserted there must be reported by name.
fn collect_objects(
    value: &Value,
    parent: Option<&Value>,
    path: &mut Vec<Step>,
    out: &mut Vec<(Vec<Step>, bool)>,
) {
    match value {
        Value::Object(map) => {
            if !is_name_map(path, parent) {
                out.push((path.clone(), !is_wrapper(path)));
            }
            for (key, child) in map {
                path.push(Step::Key(key.clone()));
                collect_objects(child, Some(value), path, out);
                path.pop();
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                path.push(Step::Index(index));
                collect_objects(child, parent, path, out);
                path.pop();
            }
        }
        _ => {}
    }
}

/// `value` with `UNKNOWN` inserted into the object at `path`.
fn with_unknown_key(value: &Value, path: &[Step]) -> Value {
    let mut copy = value.clone();
    let mut node = &mut copy;
    for step in path {
        node = match step {
            Step::Key(key) => node.get_mut(key.as_str()).expect("path from the value"),
            Step::Index(index) => node.get_mut(*index).expect("path from the value"),
        };
    }
    node.as_object_mut()
        .expect("path of an object")
        .insert(UNKNOWN.to_string(), json!(1));
    copy
}

/// An unknown key inserted into any object of a schema is an error that
/// names it: the schema itself, every field option, the builtin analyzer,
/// every analyzer definition, component (including the parameterless
/// ones) and embedder, and a quantizer's struct variant.
#[test]
fn every_object_rejects_an_unknown_key() {
    let json = serde_json::to_value(fixture()).unwrap();
    let mut positions = Vec::new();
    collect_objects(&json, None, &mut Vec::new(), &mut positions);
    let expected = format!("unknown field `{UNKNOWN}`");

    let mut failures = Vec::new();
    let mut checked = 0;
    for (path, named) in &positions {
        if !named {
            continue;
        }
        checked += 1;
        match serde_json::from_value::<Schema>(with_unknown_key(&json, path)) {
            Ok(_) => failures.push(format!("{}: accepted", render(path))),
            Err(e) if !e.to_string().contains(&expected) => {
                failures.push(format!(
                    "{}: rejected without naming the key: {e}",
                    render(path)
                ));
            }
            Err(_) => {}
        }
    }
    assert!(checked > 30, "only {checked} positions were checked");
    assert!(
        failures.is_empty(),
        "{} of {checked} positions did not reject `{UNKNOWN}`:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A key next to the variant name of an externally tagged enum (a field
/// option, a quantizer struct variant) is rejected too. serde does that by
/// itself, without naming the key; this pins it.
#[test]
fn externally_tagged_wrappers_reject_a_second_key() {
    let json = serde_json::to_value(fixture()).unwrap();
    let mut positions = Vec::new();
    collect_objects(&json, None, &mut Vec::new(), &mut positions);

    let wrappers: Vec<&Vec<Step>> = positions
        .iter()
        .filter(|(_, named)| !named)
        .map(|(path, _)| path)
        .collect();
    assert!(
        wrappers
            .iter()
            .any(|path| key_at(path, 0) == Some("fields") && path.len() == 2),
        "no field option wrapper in {wrappers:?}"
    );
    assert!(
        wrappers
            .iter()
            .any(|path| matches!(path.last(), Some(Step::Key(key)) if key == "quantizer")),
        "no quantizer wrapper in {wrappers:?}"
    );

    let accepted: Vec<String> = wrappers
        .iter()
        .filter(|path| serde_json::from_value::<Schema>(with_unknown_key(&json, path)).is_ok())
        .map(|path| render(path))
        .collect();
    assert!(accepted.is_empty(), "accepted a second key at {accepted:?}");
}

/// The variant names serde lists for `T` when it reads `probe`, which names
/// a variant `T` does not have.
fn declared_variants<T: DeserializeOwned + std::fmt::Debug>(probe: Value) -> BTreeSet<String> {
    let message = serde_json::from_value::<T>(probe)
        .expect_err("the probe names no variant")
        .to_string();
    let listed = message
        .split_once(", expected ")
        .unwrap_or_else(|| panic!("no list of variants in {message:?}"))
        .1;
    listed
        .split('`')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

/// The single key of `value`'s object, or `value` itself if it is a string.
fn variant_name(value: &Value) -> String {
    match value {
        Value::String(name) => name.clone(),
        Value::Object(map) if map.len() == 1 => map.keys().next().unwrap().clone(),
        other => panic!("not a variant: {other}"),
    }
}

/// The fixture holds every variant of every schema enum, including those
/// behind a feature, so that `every_object_rejects_an_unknown_key` checks
/// each of them. The lists come from serde, so a new variant fails this
/// test until it is added to the fixture.
#[test]
fn fixture_holds_every_variant() {
    let json = serde_json::to_value(fixture()).unwrap();
    let mut held: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    let mut hold = |kind: &'static str, name: String| {
        held.entry(kind).or_default().insert(name);
    };

    for definition in json["analyzers"].as_object().unwrap().values() {
        hold("tokenizer", variant_name(&definition["tokenizer"]["type"]));
        for filter in definition["char_filters"].as_array().into_iter().flatten() {
            hold("char filter", variant_name(&filter["type"]));
        }
        for filter in definition["token_filters"].as_array().into_iter().flatten() {
            hold("token filter", variant_name(&filter["type"]));
        }
    }
    for embedder in json["embedders"].as_object().unwrap().values() {
        hold("embedder", variant_name(&embedder["type"]));
    }
    for option in json["fields"].as_object().unwrap().values() {
        let kind = variant_name(option);
        let inner = &option[kind.as_str()];
        if let Some(language) = inner.get("analyzer").and_then(|a| a.get("language")) {
            hold("builtin analyzer", variant_name(language));
        }
        if let Some(quantizer) = inner.get("quantizer") {
            hold("quantizer", variant_name(quantizer));
        }
        hold("field option", kind);
    }

    let declared: BTreeMap<&str, BTreeSet<String>> = BTreeMap::from([
        (
            "tokenizer",
            declared_variants::<TokenizerConfig>(json!({ "type": NO_SUCH_VARIANT })),
        ),
        (
            "char filter",
            declared_variants::<CharFilterConfig>(json!({ "type": NO_SUCH_VARIANT })),
        ),
        (
            "token filter",
            declared_variants::<TokenFilterConfig>(json!({ "type": NO_SUCH_VARIANT })),
        ),
        (
            "embedder",
            declared_variants::<EmbedderDefinition>(json!({ "type": NO_SUCH_VARIANT })),
        ),
        (
            "builtin analyzer",
            declared_variants::<BuiltinAnalyzerSpec>(json!({ "language": NO_SUCH_VARIANT })),
        ),
        (
            "quantizer",
            declared_variants::<QuantizationMethod>(json!(NO_SUCH_VARIANT)),
        ),
        (
            "field option",
            declared_variants::<FieldOption>(json!({ NO_SUCH_VARIANT: {} })),
        ),
    ]);

    let missing: Vec<String> = declared
        .iter()
        .flat_map(|(kind, names)| {
            let held = held.get(kind).cloned().unwrap_or_default();
            names
                .difference(&held)
                .map(move |name| format!("{kind} `{name}`"))
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        missing.is_empty(),
        "the fixture does not hold {missing:?}; add them to `fixture`"
    );
}

/// `Schema::from_toml` rejects this issue's examples with an error naming
/// the key. The line is not checked: toml reports the key's line for a
/// struct field but the table's line inside an internally tagged enum.
#[test]
fn from_toml_names_the_unknown_key() {
    let unknown_field = |key: &str| format!("unknown field `{key}`");
    let cases = [
        (
            "a regex tokenizer's misspelled gaps",
            "[fields.body.Text]\n\n[analyzers.a.tokenizer]\ntype = \"regex\"\ngap = true\n",
            unknown_field("gap"),
        ),
        (
            "a key on a filter that takes none",
            "[fields.body.Text]\n\n[analyzers.a.tokenizer]\ntype = \"whitespace\"\n\n\
             [[analyzers.a.token_filters]]\ntype = \"lowercase\"\nwords = [\"the\"]\n",
            unknown_field("words"),
        ),
        (
            "a misspelled token_filters",
            "[fields.body.Text]\n\n[analyzers.a.tokenizer]\ntype = \"whitespace\"\n\n\
             [[analyzers.a.token_filter]]\ntype = \"lowercase\"\n",
            unknown_field("token_filter"),
        ),
        (
            "a misspelled field option",
            "[fields.body.Text]\nindexd = false\n",
            unknown_field("indexd"),
        ),
        (
            "a misspelled default_fields",
            "default_field = [\"body\"]\n\n[fields.body.Text]\n",
            unknown_field("default_field"),
        ),
        (
            "a misspelled builtin analyzer key",
            "[fields.body.Text]\nanalyzer = { language = \"japanese\", dict = \"/d\", mod = \"normal\" }\n",
            unknown_field("mod"),
        ),
        (
            "a key on an embedder that takes none",
            "[fields.body.Text]\n\n[embedders.e]\ntype = \"precomputed\"\nmodle = \"x\"\n",
            unknown_field("modle"),
        ),
        // toml checks the keys of an externally tagged struct variant by
        // itself, in its own words, before serde's `deny_unknown_fields`
        // does; JSON relies on the latter (`every_object_rejects_an_unknown_key`).
        (
            "a misspelled product quantization key",
            "[fields.v.Hnsw]\ndimension = 4\nquantizer = { ProductQuantization = { subvectors = 2 } }\n",
            "unexpected keys in table: subvectors".to_string(),
        ),
    ];

    let mut failures = Vec::new();
    for (label, toml, expected) in cases {
        match Schema::from_toml(toml) {
            Ok(_) => failures.push(format!("{label}: accepted")),
            Err(e) if !e.to_string().contains(&expected) => {
                failures.push(format!("{label}: {e}"));
            }
            Err(_) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "each example must fail naming its key:\n{}",
        failures.join("\n")
    );
}

/// The types accept what they write: the fixture round-trips through
/// `to_toml` and `from_toml`, and through JSON (where `Stop { words: None }`
/// is written as `null`).
#[test]
fn every_type_reads_its_own_output() {
    let schema = fixture();
    let json = serde_json::to_value(&schema).unwrap();

    let from_json: Schema = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&from_json).unwrap(), json);

    let toml = schema.to_toml().unwrap();
    let from_toml = Schema::from_toml(&toml).unwrap_or_else(|e| panic!("{e}\n{toml}"));
    assert_eq!(serde_json::to_value(&from_toml).unwrap(), json);
}

/// `AnalyzerSpec` is read by the shape of the input: a string is a name, a
/// table a builtin analyzer whose own error (naming an unknown key) is
/// kept, and anything else asks for one of the two.
#[test]
fn analyzer_spec_reads_a_name_or_a_builtin_table() {
    assert_eq!(
        serde_json::from_value::<AnalyzerSpec>(json!("standard")).unwrap(),
        AnalyzerSpec::Named("standard".to_string())
    );
    assert_eq!(
        serde_json::from_value::<AnalyzerSpec>(json!({ "language": "japanese", "dict": "/d" }))
            .unwrap(),
        AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
            mode: "normal".to_string(),
            dict: "/d".to_string(),
            user_dict: None,
        })
    );

    let unknown = serde_json::from_value::<AnalyzerSpec>(
        json!({ "language": "japanese", "dict": "/d", "mod": "normal" }),
    )
    .unwrap_err()
    .to_string();
    assert!(unknown.contains("unknown field `mod`"), "{unknown}");

    let number = serde_json::from_value::<AnalyzerSpec>(json!(1))
        .unwrap_err()
        .to_string();
    assert!(
        number.contains("expected an analyzer name or a builtin analyzer table"),
        "{number}"
    );

    let schema = Schema::from_toml(
        "[fields.a.Text]\nanalyzer = \"standard\"\n\n\
         [fields.b.Text]\nanalyzer = { language = \"japanese\", dict = \"/d\" }\n",
    )
    .unwrap();
    let analyzer = |name: &str| match &schema.fields[name] {
        FieldOption::Text(option) => option.analyzer.clone(),
        other => panic!("{other:?}"),
    };
    assert_eq!(
        analyzer("a"),
        Some(AnalyzerSpec::Named("standard".to_string()))
    );
    assert!(matches!(
        analyzer("b"),
        Some(AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese { .. }))
    ));
}
