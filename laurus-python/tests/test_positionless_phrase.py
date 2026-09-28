"""Quoted queries on fields indexed with `term_vectors = false` (Issue #1247).

Such a field stores no term positions. A quoted value that analyzes to one
token is a term match and must still find its document; a phrase of two or
more terms needs positions and must raise instead of returning no hits.
"""

import pytest
import laurus


SCHEMA_TOML = """
[analyzers.url_list]
tokenizer = { type = "whitespace" }

[fields.path.Text]
indexed = true
stored = true
term_vectors = false
analyzer = "keyword"

[fields.links.Text]
indexed = true
stored = true
term_vectors = false
analyzer = "url_list"

[fields.body.Text]
indexed = true
stored = true
term_vectors = false

[fields.title.Text]
indexed = true
stored = true
term_vectors = true
"""


@pytest.fixture
def index():
    idx = laurus.Index(schema=laurus.Schema.from_toml(SCHEMA_TOML))
    idx.put_document(
        "a",
        {
            "path": "file:///m/a.md",
            "links": "https://example.com/a https://example.com/b?x=1",
            "body": "rust search engine",
            "title": "rust search engine",
        },
    )
    idx.put_document(
        "b",
        {
            "path": "file:///m/b.md",
            "links": "https://other.example/z",
            "body": "search for rust",
            "title": "search for rust",
        },
    )
    idx.commit()
    return idx


def ids(results):
    return sorted(hit.id for hit in results)


def test_quoted_single_token_matches_without_positions(index):
    assert ids(index.search('path:"file:///m/a.md"')) == ["a"]
    assert ids(index.search('links:"https://example.com/a"')) == ["a"]


def test_one_term_phrase_query_object_matches_without_positions(index):
    query = laurus.PhraseQuery("path", ["file:///m/a.md"])
    assert ids(index.search(query)) == ["a"]


def test_term_query_is_unaffected(index):
    assert ids(index.search(laurus.TermQuery("path", "file:///m/a.md"))) == ["a"]


def test_phrase_on_a_field_with_positions_still_matches(index):
    assert ids(index.search('title:"rust search"')) == ["a"]


def test_multi_term_phrase_without_positions_raises(index):
    with pytest.raises(ValueError, match="body"):
        index.search('body:"rust search"')
    with pytest.raises(ValueError, match="term_vectors"):
        index.search(laurus.PhraseQuery("body", ["rust", "search"]))
