# frozen_string_literal: true

# Lexical Search Example — all query types.
#
# Demonstrates every lexical query type Laurus supports:
#
# 1. TermQuery         — exact single-term matching
# 2. PhraseQuery       — exact word sequence matching
# 3. FuzzyQuery        — approximate matching (typo tolerance)
# 4. WildcardQuery     — pattern matching with * and ?
# 5. NumericRangeQuery — numeric range filtering (int and float)
# 6. GeoDistanceQuery / GeoBoundingBoxQuery — geographic radius / bounding box
# 7. BooleanQuery      — AND / OR / NOT combinations
# 8. SpanQuery         — positional / proximity search
#
# Usage:
#   cd laurus-ruby
#   bundle exec rake compile
#   ruby -Ilib examples/lexical_search.rb

require "laurus"

def print_results(results)
  if results.empty?
    puts "  (no results)"
    return
  end
  results.each do |r|
    title = r.document&.fetch("title", "") || ""
    printf "  id=%-8s  score=%.4f  title=%s\n", r.id.inspect, r.score, title.inspect
  end
end

def main
  puts "=== Laurus Lexical Search Example ===\n\n"

  # ── Setup ──────────────────────────────────────────────────────────────
  schema = Laurus::Schema.new
  schema.add_text_field("title")
  schema.add_text_field("body")
  schema.add_text_field("category", analyzer: "keyword")
  schema.add_text_field("filename", analyzer: "keyword")
  schema.add_boolean_field("in_print")
  schema.add_float_field("price")
  schema.add_integer_field("year")
  schema.add_geo_field("location")
  schema.set_default_fields(%w[body])

  index = Laurus::Index.new(schema: schema)

  # ── Index documents ────────────────────────────────────────────────────
  docs = [
    ["rails", {
      "title" => "Ruby on Rails",
      "body" => "Rails is a full-stack web application framework following convention over configuration and the MVC pattern",
      "category" => "framework",
      "filename" => "rails_guide.pdf",
      "in_print" => true,
      "price" => 49.99,
      "year" => 2004,
      "location" => { "lat" => 37.7749, "lon" => -122.4194 } # San Francisco
    }],
    ["sinatra", {
      "title" => "Sinatra DSL",
      "body" => "Sinatra is a lightweight Ruby DSL for creating web applications with minimal effort and elegant routing",
      "category" => "framework",
      "filename" => "sinatra_docs.epub",
      "in_print" => true,
      "price" => 29.99,
      "year" => 2007,
      "location" => { "lat" => 37.4419, "lon" => -122.1430 } # Palo Alto
    }],
    ["rspec", {
      "title" => "RSpec Testing Framework",
      "body" => "RSpec is a behavior-driven development framework for writing readable and expressive Ruby test specifications",
      "category" => "testing",
      "filename" => "rspec_manual.pdf",
      "in_print" => true,
      "price" => 44.99,
      "year" => 2005,
      "location" => { "lat" => 52.5200, "lon" => 13.4050 } # Berlin
    }],
    ["rubocop", {
      "title" => "RuboCop Style Guide",
      "body" => "RuboCop is a static code analyzer and formatter enforcing community Ruby style guidelines and best practices",
      "category" => "tooling",
      "filename" => "rubocop_guide.docx",
      "in_print" => true,
      "price" => 35.50,
      "year" => 2013,
      "location" => { "lat" => 51.5074, "lon" => -0.1278 } # London
    }],
    ["bundler", {
      "title" => "Bundler Dependency Manager",
      "body" => "Bundler manages gem dependencies for Ruby projects ensuring consistent environments across machines",
      "category" => "tooling",
      "filename" => "bundler_docs.pdf",
      "in_print" => false,
      "price" => 25.00,
      "year" => 2010,
      "location" => { "lat" => 47.6062, "lon" => -122.3321 } # Seattle
    }],
    ["matz", {
      "title" => "The elegant red gem",
      "body" => "The elegant red gem sparkled beneath the cherry blossom tree in a quiet garden",
      "category" => "fiction",
      "filename" => "gem_story.txt",
      "in_print" => false,
      "price" => 12.99,
      "year" => 2023,
      "location" => { "lat" => 34.0522, "lon" => -118.2437 } # Los Angeles
    }]
  ]

  puts "  Indexing #{docs.size} documents..."
  docs.each { |doc_id, doc| index.add_document(doc_id, doc) }
  index.commit
  puts "  Done.\n\n"

  # =====================================================================
  # PART 1: TermQuery — exact single-term matching
  # =====================================================================
  puts "=" * 60
  puts "PART 1: TermQuery"
  puts "=" * 60

  puts "\n[1a] Search for 'ruby' in body:"
  print_results(index.search(Laurus::TermQuery.new("body", "ruby"), limit: 5).results)

  puts "\n[1b] Search for 'framework' in category (exact):"
  print_results(index.search(Laurus::TermQuery.new("category", "framework"), limit: 5).results)

  puts "\n[1c] Search for in_print=true (boolean field):"
  print_results(index.search(Laurus::TermQuery.new("in_print", "true"), limit: 5).results)

  puts "\n[1d] DSL: 'body:ruby':"
  print_results(index.search("body:ruby", limit: 5).results)

  # =====================================================================
  # PART 2: PhraseQuery — exact word sequence
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "PART 2: PhraseQuery"
  puts "=" * 60

  puts "\n[2a] Phrase 'convention over configuration' in body:"
  print_results(index.search(Laurus::PhraseQuery.new("body", %w[convention over configuration]), limit: 5).results)

  puts "\n[2b] Phrase 'web application framework' in body:"
  print_results(index.search(Laurus::PhraseQuery.new("body", %w[web application framework]), limit: 5).results)

  puts "\n[2c] DSL: 'body:\"convention over configuration\"':"
  print_results(index.search('body:"convention over configuration"', limit: 5).results)

  # =====================================================================
  # PART 3: FuzzyQuery — approximate matching (typo tolerance)
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "PART 3: FuzzyQuery"
  puts "=" * 60

  puts "\n[3a] Fuzzy 'rubyy' (extra 'y', max_edits=2):"
  print_results(index.search(Laurus::FuzzyQuery.new("body", "rubyy", max_edits: 2), limit: 5).results)

  puts "\n[3b] Fuzzy 'sinatraa' (extra 'a', max_edits=1):"
  print_results(index.search(Laurus::FuzzyQuery.new("body", "sinatraa", max_edits: 1), limit: 5).results)

  puts "\n[3c] DSL: 'rubyy~2':"
  print_results(index.search("rubyy~2", limit: 5).results)

  # =====================================================================
  # PART 4: WildcardQuery — pattern matching with * and ?
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "PART 4: WildcardQuery"
  puts "=" * 60

  puts "\n[4a] Wildcard '*.pdf' in filename:"
  print_results(index.search(Laurus::WildcardQuery.new("filename", "*.pdf"), limit: 5).results)

  puts "\n[4b] Wildcard 'rub*' in body:"
  print_results(index.search(Laurus::WildcardQuery.new("body", "rub*"), limit: 5).results)

  puts "\n[4c] DSL: 'body:rub*':"
  print_results(index.search("body:rub*", limit: 5).results)

  # =====================================================================
  # PART 5: NumericRangeQuery — numeric range filtering
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "PART 5: NumericRangeQuery"
  puts "=" * 60

  puts "\n[5a] Books with price $30-$50 (float range):"
  print_results(index.search(Laurus::NumericRangeQuery.new("price", min: 30.0, max: 50.0), limit: 5).results)

  puts "\n[5b] Books published from 2010 onwards (integer range):"
  print_results(index.search(Laurus::NumericRangeQuery.new("year", min: 2010), limit: 5).results)

  puts "\n[5c] DSL: 'price:[30 TO 50]':"
  print_results(index.search("price:[30 TO 50]", limit: 5).results)

  # =====================================================================
  # PART 6: GeoDistanceQuery / GeoBoundingBoxQuery — geographic search
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "PART 6: GeoDistanceQuery / GeoBoundingBoxQuery"
  puts "=" * 60

  puts "\n[6a] Within 100 km of San Francisco (37.77, -122.42):"
  print_results(index.search(Laurus::GeoDistanceQuery.within_radius("location", 37.7749, -122.4194, 100_000.0), limit: 5).results)

  puts "\n[6b] Bounding box — US West Coast (33, -123) to (48, -117):"
  print_results(index.search(Laurus::GeoBoundingBoxQuery.within_bounding_box("location", 33.0, -123.0, 48.0, -117.0), limit: 5).results)

  # =====================================================================
  # PART 7: BooleanQuery — AND / OR / NOT combinations
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "PART 7: BooleanQuery"
  puts "=" * 60

  puts "\n[7a] AND: 'ruby' in body AND category='framework':"
  bq = Laurus::BooleanQuery.new
  bq.must(Laurus::TermQuery.new("body", "ruby"))
  bq.must(Laurus::TermQuery.new("category", "framework"))
  print_results(index.search(bq, limit: 5).results)

  puts "\n[7b] OR: category='framework' OR category='testing':"
  bq = Laurus::BooleanQuery.new
  bq.should(Laurus::TermQuery.new("category", "framework"))
  bq.should(Laurus::TermQuery.new("category", "testing"))
  print_results(index.search(bq, limit: 5).results)

  puts "\n[7c] NOT: 'ruby' in body, NOT 'sinatra':"
  bq = Laurus::BooleanQuery.new
  bq.must(Laurus::TermQuery.new("body", "ruby"))
  bq.must_not(Laurus::TermQuery.new("body", "sinatra"))
  print_results(index.search(bq, limit: 5).results)

  puts "\n[7d] DSL: '+body:ruby -body:sinatra':"
  print_results(index.search("+body:ruby -body:sinatra", limit: 5).results)

  # =====================================================================
  # PART 8: SpanQuery — positional / proximity search
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "PART 8: SpanQuery (no DSL equivalent)"
  puts "=" * 60

  puts "\n[8a] SpanNear: 'elegant' near 'gem' (slop=1, ordered):"
  span_q = Laurus::SpanQuery.near("body", %w[elegant gem], slop: 1, ordered: true)
  print_results(index.search(span_q, limit: 5).results)

  puts "\n[8b] SpanContaining: 'elegant..gem' containing 'red':"
  big = Laurus::SpanQuery.near("body", %w[elegant gem], slop: 1, ordered: true)
  little = Laurus::SpanQuery.term("body", "red")
  containing = Laurus::SpanQuery.containing("body", big, little)
  print_results(index.search(containing, limit: 5).results)

  puts "\nLexical search example completed!"
end

main
