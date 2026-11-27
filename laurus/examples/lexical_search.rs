//! Lexical Search Example — all query types via Engine API.
//!
//! Demonstrates every lexical query type Laurus supports:
//!
//! 1. **TermQuery**         — exact single-term matching
//! 2. **PhraseQuery**       — exact word sequence matching
//! 3. **FuzzyQuery**        — approximate matching (typo tolerance)
//! 4. **WildcardQuery**     — pattern matching with `*` and `?`
//! 5. **NumericRangeQuery**  — numeric range filtering
//! 6. **GeoDistanceQuery / GeoBoundingBoxQuery** — geographic radius / bounding box (Builder API only)
//! 7. **BooleanQuery**      — AND / OR / NOT combinations
//! 8. **SpanQuery**         — positional / proximity search (Builder API only)
//!
//! Each query type (where supported) includes both a Builder API example
//! and a QueryParser DSL example showing the equivalent text syntax.
//!
//! Run with: `cargo run --example lexical_search`

mod common;

use laurus::lexical::core::field::{BooleanOption, FloatOption, GeoOption, IntegerOption};
use laurus::lexical::span::{SpanQueryBuilder, SpanQueryWrapper};
use laurus::lexical::{
    BooleanQuery, FuzzyQuery, GeoBoundingBoxQuery, GeoDistanceQuery, LexicalQueryParser,
    NumericRangeQuery, PhraseQuery, TermQuery, TextOption, WildcardQuery,
};
use laurus::{
    DataValue, Document, Engine, LexicalSearchQuery, Result, Schema, SearchRequestBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Laurus Lexical Search Example ===\n");

    // ─── Setup ───────────────────────────────────────────────────────────
    let storage = common::memory_storage()?;
    let analyzer = common::per_field_analyzer(&["category", "filename"]);

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_text_field("category", TextOption::default())
        .add_text_field("filename", TextOption::default())
        .add_boolean_field("in_print", BooleanOption::default())
        .add_float_field("price", FloatOption::default())
        .add_integer_field("year", IntegerOption::default())
        .add_geo_field("location", GeoOption::default())
        .add_default_field("body")
        .build();

    let engine = Engine::builder(storage, schema)
        .analyzer(analyzer.clone())
        .build()
        .await?;

    // ─── Index documents ─────────────────────────────────────────────────
    let docs = vec![
        ("book1", Document::builder()
            .add_text("title", "The Rust Programming Language")
            .add_text("body", "Rust is a systems programming language focused on safety, speed, and concurrency")
            .add_text("category", "programming")
            .add_text("filename", "rust_book.pdf")
            .add_boolean("in_print", true)
            .add_field("price", DataValue::Float64(49.99))
            .add_integer("year", 2019)
            .add_field("location", DataValue::Geo(laurus::lexical::GeoPoint::new(37.7749, -122.4194))) // San Francisco
            .build()),
        ("book2", Document::builder()
            .add_text("title", "Python for Data Science")
            .add_text("body", "Python is a versatile programming language widely used in data science and machine learning")
            .add_text("category", "data-science")
            .add_text("filename", "python_data.epub")
            .add_boolean("in_print", true)
            .add_field("price", DataValue::Float64(39.99))
            .add_integer("year", 2021)
            .add_field("location", DataValue::Geo(laurus::lexical::GeoPoint::new(40.7128, -74.0060))) // New York
            .build()),
        ("book3", Document::builder()
            .add_text("title", "JavaScript Web Development")
            .add_text("body", "JavaScript powers the modern web from frontend frameworks to backend services")
            .add_text("category", "web-development")
            .add_text("filename", "javascript_web.pdf")
            .add_boolean("in_print", true)
            .add_field("price", DataValue::Float64(54.99))
            .add_integer("year", 2022)
            .add_field("location", DataValue::Geo(laurus::lexical::GeoPoint::new(51.5074, -0.1278))) // London
            .build()),
        ("book4", Document::builder()
            .add_text("title", "Machine Learning Algorithms")
            .add_text("body", "Understanding algorithms used in machine learning and artificial intelligence applications")
            .add_text("category", "data-science")
            .add_text("filename", "ml_algorithms.docx")
            .add_boolean("in_print", true)
            .add_field("price", DataValue::Float64(72.99))
            .add_integer("year", 2020)
            .add_field("location", DataValue::Geo(laurus::lexical::GeoPoint::new(37.4419, -122.1430))) // Palo Alto
            .build()),
        ("book5", Document::builder()
            .add_text("title", "Database Design Principles")
            .add_text("body", "Learn database design, SQL queries, and data management for modern applications")
            .add_text("category", "database")
            .add_text("filename", "db_design.pdf")
            .add_boolean("in_print", false)
            .add_field("price", DataValue::Float64(45.50))
            .add_integer("year", 2018)
            .add_field("location", DataValue::Geo(laurus::lexical::GeoPoint::new(47.6062, -122.3321))) // Seattle
            .build()),
        ("book6", Document::builder()
            .add_text("title", "The quick brown fox")
            .add_text("body", "The quick brown fox jumped over the lazy dog in a sunny meadow")
            .add_text("category", "fiction")
            .add_text("filename", "fox_story.txt")
            .add_boolean("in_print", false)
            .add_field("price", DataValue::Float64(12.99))
            .add_integer("year", 2023)
            .add_field("location", DataValue::Geo(laurus::lexical::GeoPoint::new(34.0522, -118.2437))) // Los Angeles
            .build()),
    ];

    println!("  Indexing {} documents...", docs.len());
    for (id, doc) in &docs {
        engine.add_document(id, doc.clone()).await?;
    }
    engine.commit().await?;
    println!("  Done.\n");

    // ─── Query parser for DSL examples ──────────────────────────────────
    let parser = LexicalQueryParser::new(analyzer).with_default_field("body");

    // =====================================================================
    // PART 1: TermQuery — exact single-term matching
    // =====================================================================
    println!("{}", "=".repeat(60));
    println!("PART 1: TermQuery");
    println!("{}", "=".repeat(60));

    println!("\n[1a] Search for 'rust' in body:");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "body", "rust",
                ))))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[1b] Search for 'programming' in category (KeywordAnalyzer — exact):");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "category",
                    "programming",
                ))))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[1c] Search for in_print=true (boolean field):");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "in_print", "true",
                ))))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[1d] DSL: 'body:rust'");
    let query = parser.parse("body:rust")?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // =====================================================================
    // PART 2: PhraseQuery — exact word sequence
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("PART 2: PhraseQuery");
    println!("{}", "=".repeat(60));

    println!("\n[2a] Phrase 'machine learning' in body:");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(PhraseQuery::new(
                    "body",
                    vec!["machine".into(), "learning".into()],
                ))))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[2b] Phrase 'systems programming language' in body:");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(PhraseQuery::new(
                    "body",
                    vec!["systems".into(), "programming".into(), "language".into()],
                ))))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[2c] DSL: 'body:\"machine learning\"'");
    let query = parser.parse("body:\"machine learning\"")?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // =====================================================================
    // PART 3: FuzzyQuery — approximate matching (typo tolerance)
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("PART 3: FuzzyQuery");
    println!("{}", "=".repeat(60));

    println!("\n[3a] Fuzzy 'programing' (missing 'm', edit distance 2):");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(
                    FuzzyQuery::new("body", "programing").max_edits(2),
                )))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[3b] Fuzzy 'javascritp' (transposed, edit distance 1):");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(
                    FuzzyQuery::new("body", "javascritp").max_edits(1),
                )))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[3c] DSL: 'programing~2'");
    let query = parser.parse("programing~2")?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // =====================================================================
    // PART 4: WildcardQuery — pattern matching with * and ?
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("PART 4: WildcardQuery");
    println!("{}", "=".repeat(60));

    println!("\n[4a] Wildcard '*.pdf' in filename:");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(WildcardQuery::new(
                    "filename", "*.pdf",
                )?)))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[4b] Wildcard 'pro*' in body:");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(WildcardQuery::new(
                    "body", "pro*",
                )?)))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[4c] DSL: 'body:pro*'");
    let query = parser.parse("body:pro*")?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // =====================================================================
    // PART 5: NumericRangeQuery — numeric range filtering
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("PART 5: NumericRangeQuery");
    println!("{}", "=".repeat(60));

    println!("\n[5a] Books with price $40–$60:");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(
                    NumericRangeQuery::f64_range("price", Some(40.0), Some(60.0)),
                )))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[5b] Books published after 2020:");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(
                    NumericRangeQuery::i64_range("year", Some(2021), None),
                )))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[5c] DSL: 'price:[40 TO 60]'");
    let query = parser.parse("price:[40 TO 60]")?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // =====================================================================
    // PART 6: GeoDistanceQuery / GeoBoundingBoxQuery — geographic search
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("PART 6: GeoDistanceQuery / GeoBoundingBoxQuery");
    println!("{}", "=".repeat(60));

    println!("\n[6a] Within 100km of San Francisco (37.77, -122.42):");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(
                    GeoDistanceQuery::within_radius("location", 37.7749, -122.4194, 100.0)?,
                )))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[6b] Bounding box — US West Coast (33, -123) to (48, -117):");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(
                    GeoBoundingBoxQuery::within_bounding_box(
                        "location", 33.0, -123.0, 48.0, -117.0,
                    )?,
                )))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // =====================================================================
    // PART 7: BooleanQuery — AND / OR / NOT combinations
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("PART 7: BooleanQuery");
    println!("{}", "=".repeat(60));

    println!("\n[7a] AND: 'programming' in body AND category='data-science':");
    let mut bq = BooleanQuery::new();
    bq.add_must(Box::new(TermQuery::new("body", "programming")));
    bq.add_must(Box::new(TermQuery::new("category", "data-science")));
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[7b] OR: category='programming' OR category='web-development':");
    let mut bq = BooleanQuery::new();
    bq.add_should(Box::new(TermQuery::new("category", "programming")));
    bq.add_should(Box::new(TermQuery::new("category", "web-development")));
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[7c] NOT: 'programming' in body, NOT 'python':");
    let mut bq = BooleanQuery::new();
    bq.add_must(Box::new(TermQuery::new("body", "programming")));
    bq.add_must_not(Box::new(TermQuery::new("body", "python")));
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[7d] DSL: '+body:programming -body:python'");
    let query = parser.parse("+body:programming -body:python")?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // =====================================================================
    // PART 8: SpanQuery — positional / proximity search (Builder API only)
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("PART 8: SpanQuery (Builder API only — no DSL equivalent)");
    println!("{}", "=".repeat(60));

    let sb = SpanQueryBuilder::new("body");

    println!("\n[8a] SpanNear: 'quick' near 'fox' (slop=1, ordered):");
    let span_near = sb.near(
        vec![Box::new(sb.term("quick")), Box::new(sb.term("fox"))],
        1,
        true,
    );
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(SpanQueryWrapper::new(
                    Box::new(span_near),
                ))))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[8b] SpanContaining: 'quick..fox' containing 'brown':");
    let big = sb.near(
        vec![Box::new(sb.term("quick")), Box::new(sb.term("fox"))],
        1,
        true,
    );
    let little = sb.term("brown");
    let containing = sb.containing(Box::new(big), Box::new(little));
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(SpanQueryWrapper::new(
                    Box::new(containing),
                ))))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\nLexical search example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexical_search_example() {
        let result = main();
        assert!(result.is_ok(), "lexical_search failed: {:?}", result.err());
    }
}
