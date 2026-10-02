"""Tests for `Schema.add_analyzer` and TOML schema loading/saving (Issue #1057).

These deliberately avoid the Lindera tokenizer: this repository ships no
Lindera dictionary, and `embedded://*` dictionary URIs are only resolvable
from `laurus`'s own dev-dependency test builds, not from `laurus-python`.
`whitespace`/`ngram`/`regex` tokenizers exercise the same code paths
(`add_analyzer`'s dict-to-core-enum conversion, and the TOML
serialize/deserialize round trip) without that dependency.

Each behavioural test proves the analyzer actually reached the query
engine (a deterministic search-result difference), not just that
`add_analyzer`/`from_toml` didn't raise.
"""

import pathlib

import pytest
import laurus


# ---------------------------------------------------------------------------
# add_analyzer: behavioural — the analyzer actually takes effect
# ---------------------------------------------------------------------------


def test_ngram_tokenizer_enables_substring_match():
    schema = laurus.Schema()
    schema.add_analyzer("ngram3", {"type": "ngram", "min_gram": 3, "max_gram": 3})
    schema.add_text_field("title", analyzer="ngram3")
    schema.add_text_field("plain")  # default "standard" analyzer: whole-word tokens

    idx = laurus.Index(schema=schema)
    idx.put_document("doc1", {"title": "hello", "plain": "hello"})
    idx.commit()

    # "ell" is a substring of "hello", only reachable via 3-grams (hel/ell/llo).
    assert len(idx.search("title:ell", limit=5)) == 1
    assert len(idx.search("plain:ell", limit=5)) == 0


def test_token_filters_apply_lowercase():
    schema = laurus.Schema()
    schema.add_analyzer("ws", {"type": "whitespace"})
    schema.add_analyzer("ws_lower", {"type": "whitespace"}, token_filters=[{"type": "lowercase"}])
    schema.add_text_field("raw", analyzer="ws")
    schema.add_text_field("lower", analyzer="ws_lower")

    idx = laurus.Index(schema=schema)
    idx.put_document("doc1", {"raw": "HELLO World", "lower": "HELLO World"})
    idx.commit()

    # Without a lowercase filter, neither the indexed token nor the query
    # term is folded, so an all-lowercase query term can't match "HELLO".
    assert len(idx.search("raw:hello", limit=5)) == 0
    # With the filter applied on both sides (index + query time), it matches.
    assert len(idx.search("lower:hello", limit=5)) == 1


def test_char_filters_apply_pattern_replace():
    schema = laurus.Schema()
    schema.add_analyzer("dash", {"type": "whitespace"})
    schema.add_analyzer(
        "dash_split",
        {"type": "whitespace"},
        char_filters=[{"type": "pattern_replace", "pattern": "-", "replacement": " "}],
    )
    schema.add_text_field("raw", analyzer="dash")
    schema.add_text_field("split", analyzer="dash_split")

    idx = laurus.Index(schema=schema)
    idx.put_document("doc1", {"raw": "state-of-the-art", "split": "state-of-the-art"})
    idx.commit()

    # Whitespace alone doesn't split on hyphens, so "art" is never its own token.
    assert len(idx.search("raw:art", limit=5)) == 0
    # The char filter turns hyphens into spaces before tokenization.
    assert len(idx.search("split:art", limit=5)) == 1


def test_add_analyzer_defaults_to_no_filters():
    schema = laurus.Schema()
    schema.add_analyzer("ws", {"type": "whitespace"})  # char_filters/token_filters omitted
    assert schema.analyzer_names() == ["ws"]


# ---------------------------------------------------------------------------
# add_analyzer: error surface
# ---------------------------------------------------------------------------


def test_unknown_tokenizer_type_rejected():
    schema = laurus.Schema()
    with pytest.raises(ValueError, match="tokenizer"):
        schema.add_analyzer("bad", {"type": "kuromoji"})


def test_missing_required_tokenizer_field_rejected():
    schema = laurus.Schema()
    with pytest.raises(ValueError, match="tokenizer"):
        schema.add_analyzer("bad", {"type": "ngram", "min_gram": 2})  # missing max_gram


def test_unknown_char_filter_type_rejected():
    schema = laurus.Schema()
    with pytest.raises(ValueError, match=r"char_filters\[0\]"):
        schema.add_analyzer(
            "bad", {"type": "whitespace"}, char_filters=[{"type": "unknown_filter"}]
        )


def test_negative_limit_rejected():
    schema = laurus.Schema()
    with pytest.raises(ValueError, match=r"token_filters\[0\]"):
        schema.add_analyzer(
            "bad", {"type": "whitespace"}, token_filters=[{"type": "limit", "limit": -1}]
        )


def test_boost_accepts_python_int():
    schema = laurus.Schema()
    # boost is f32 in the core; a Python int must not be rejected.
    schema.add_analyzer(
        "b", {"type": "whitespace"}, token_filters=[{"type": "boost", "boost": 2}]
    )
    assert schema.analyzer_names() == ["b"]


def test_bool_not_coerced_to_int():
    schema = laurus.Schema()
    # gaps is bool in the core; must accept a Python bool (not misread as int).
    schema.add_analyzer("r", {"type": "regex", "pattern": r"\w+", "gaps": True})
    assert schema.analyzer_names() == ["r"]


def test_non_str_mapping_key_rejected():
    schema = laurus.Schema()
    with pytest.raises(TypeError):
        schema.add_analyzer(
            "m",
            {"type": "whitespace"},
            char_filters=[{"type": "mapping", "mapping": {1: "a"}}],
        )


def test_tokenizer_must_be_dict_not_string():
    schema = laurus.Schema()
    with pytest.raises(ValueError, match="tokenizer"):
        schema.add_analyzer("bad", "whitespace")


# ---------------------------------------------------------------------------
# Names reserved for built-in analyzers (Issue #1310)
# ---------------------------------------------------------------------------

RESERVED_NAMES = ["standard", "keyword", "english", "simple", "noop"]

# The entry tokenizes on whitespace without lowercasing, so a lowercase query
# matches "Hello" only when the built-in `standard` analyzer is used.
RESERVED_SCHEMA_TOML = """
[analyzers.standard]
tokenizer = { type = "whitespace" }

[fields.body.Text]
analyzer = "standard"
"""


@pytest.mark.parametrize("name", RESERVED_NAMES)
def test_add_analyzer_rejects_reserved_name(name):
    schema = laurus.Schema()
    with pytest.raises(
        ValueError, match=f"Analyzer name '{name}' is reserved for a built-in analyzer"
    ):
        schema.add_analyzer(name, {"type": "whitespace"})
    assert schema.analyzer_names() == []


def test_add_analyzer_accepts_japanese_name():
    schema = laurus.Schema()
    schema.add_analyzer("japanese", {"type": "whitespace"})
    assert schema.analyzer_names() == ["japanese"]


def test_in_memory_index_rejects_reserved_analyzer_name():
    schema = laurus.Schema.from_toml(RESERVED_SCHEMA_TOML)
    with pytest.raises(ValueError, match="reserved for a built-in analyzer"):
        laurus.Index(schema=schema)


def test_path_index_rejects_reserved_analyzer_name_and_writes_nothing(tmp_path):
    schema = laurus.Schema.from_toml(RESERVED_SCHEMA_TOML)
    index_dir = tmp_path / "idx"
    with pytest.raises(ValueError, match="reserved for a built-in analyzer"):
        laurus.Index(str(index_dir), schema=schema)
    assert not index_dir.exists()


def test_existing_index_with_reserved_analyzer_name_still_opens(tmp_path):
    (tmp_path / "schema.toml").write_text(RESERVED_SCHEMA_TOML)
    idx = laurus.Index(str(tmp_path))
    idx.put_document("doc1", {"body": "Hello World"})
    idx.commit()
    assert len(idx.search("body:hello", limit=5)) == 1


# ---------------------------------------------------------------------------
# from_toml / from_toml_file
# ---------------------------------------------------------------------------

SCHEMA_TOML = """
default_fields = ["title"]

[analyzers.ngram3]
tokenizer = { type = "ngram", min_gram = 3, max_gram = 3 }

[fields.title.Text]
indexed = true
stored = true
term_vectors = false
analyzer = "ngram3"
"""


def test_from_toml_then_search():
    schema = laurus.Schema.from_toml(SCHEMA_TOML)
    assert schema.field_names() == ["title"]
    assert schema.analyzer_names() == ["ngram3"]

    idx = laurus.Index(schema=schema)
    idx.put_document("doc1", {"title": "hello"})
    idx.commit()
    assert len(idx.search("title:ell", limit=5)) == 1


def test_from_toml_file_accepts_path_and_str(tmp_path):
    p = tmp_path / "schema.toml"
    p.write_text(SCHEMA_TOML)

    from_path = laurus.Schema.from_toml_file(p)
    from_str = laurus.Schema.from_toml_file(str(p))
    assert from_path.field_names() == from_str.field_names() == ["title"]


def test_from_toml_file_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        laurus.Schema.from_toml_file(tmp_path / "does_not_exist.toml")


def test_from_toml_parse_error_raises_value_error():
    with pytest.raises(ValueError, match="TOML"):
        laurus.Schema.from_toml("not = [valid")


def test_from_toml_file_repo_fixture():
    """Sanity check against a real fixture used elsewhere in the repo."""
    fixture = pathlib.Path(__file__).resolve().parents[2] / "resources" / "schema.toml"
    if not fixture.exists():
        pytest.skip("resources/schema.toml not present in this checkout")
    schema = laurus.Schema.from_toml_file(fixture)
    assert schema.field_names()


# ---------------------------------------------------------------------------
# to_toml / to_toml_file: round trip
# ---------------------------------------------------------------------------


def test_to_toml_then_from_toml_round_trip():
    schema = laurus.Schema()
    schema.add_analyzer(
        "ngram3",
        {"type": "ngram", "min_gram": 3, "max_gram": 3},
        char_filters=[{"type": "unicode_normalization", "form": "nfkc"}],
        token_filters=[{"type": "lowercase"}],
    )
    schema.add_text_field("title", analyzer="ngram3")
    schema.set_default_fields(["title"])

    toml_str = schema.to_toml()
    restored = laurus.Schema.from_toml(toml_str)

    # Compare parsed structure rather than raw text: it's the more direct
    # check for round-tripping, even though table order in the TOML text
    # is now stable and sorted by key (Issue #1060).
    assert restored.field_names() == schema.field_names()
    assert restored.analyzer_names() == schema.analyzer_names()

    idx = laurus.Index(schema=restored)
    idx.put_document("doc1", {"title": "hello"})
    idx.commit()
    assert len(idx.search("title:ell", limit=5)) == 1


def test_to_toml_file_then_from_toml_file_round_trip(tmp_path):
    schema = laurus.Schema()
    schema.add_analyzer("ws", {"type": "whitespace"})
    schema.add_text_field("title", analyzer="ws")

    p = tmp_path / "schema.toml"
    schema.to_toml_file(p)
    restored = laurus.Schema.from_toml_file(p)

    assert restored.field_names() == schema.field_names()
    assert restored.analyzer_names() == schema.analyzer_names()
