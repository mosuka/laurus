# frozen_string_literal: true

# Hybrid Search Example — combining lexical and vector search.
#
# Demonstrates:
# - Lexical-only search (for comparison)
# - Vector-only search (for comparison)
# - Hybrid search with RRF fusion
# - Hybrid search with WeightedSum fusion
# - Hybrid search with a filter query
#
# The embedder is registered in the schema and laurus automatically converts
# text to vectors at index and query time — no external embedding library needed.
#
# Usage:
#   cd laurus-ruby
#   bundle exec rake compile  # build with: --features embeddings-candle
#   ruby -Ilib examples/hybrid_search.rb

require "laurus"

EMBEDDER_NAME = "bert"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384 # dimension for all-MiniLM-L6-v2

CHUNKS = [
  ["rails_guide", "Ruby on Rails Tutorial", "Rails follows the model-view-controller pattern with Active Record for database access.", 1, "framework"],
  ["rails_guide", "Ruby on Rails Tutorial", "Rails migrations manage database schema changes with version-controlled Ruby files.", 2, "framework"],
  ["rails_guide", "Ruby on Rails Tutorial", "Action Controller handles HTTP requests and renders responses through configurable routes.", 3, "framework"],
  ["sinatra_guide", "Sinatra Web Apps", "Sinatra provides lightweight routing with blocks for building REST APIs and small web applications.", 1, "framework"],
  ["sinatra_guide", "Sinatra Web Apps", "Sinatra middleware and helpers extend request processing with authentication and logging.", 2, "framework"],
  ["rspec_book", "Testing with RSpec", "RSpec describe and context blocks organize test examples with readable and expressive syntax.", 1, "testing"],
  ["rspec_book", "Testing with RSpec", "RSpec mocks and stubs simulate external dependencies to isolate the unit under test.", 2, "testing"],
  ["rubygems_docs", "RubyGems Fundamentals", "RubyGems is the package manager for Ruby distributing libraries as self-contained gems.", 1, "tooling"],
  ["rubygems_docs", "RubyGems Fundamentals", "Gemspec files define gem metadata, dependencies, and file lists for publishing to rubygems.org.", 2, "tooling"]
].freeze

def print_results(results)
  if results.empty?
    puts "  (no results)"
    return
  end
  results.each do |r|
    doc = r.document || {}
    text = (doc["text"] || "")[0, 60]
    printf "  id=%-8s  score=%.4f  text=%s\n", r.id.inspect, r.score, text.inspect
  end
end

def main
  puts "=== Laurus Hybrid Search Example ===\n\n"
  puts "Embedder: #{EMBEDDER_MODEL} (dim=#{DIM})\n\n"

  # ── Schema ─────────────────────────────────────────────────────────────
  schema = Laurus::Schema.new
  schema.add_embedder(EMBEDDER_NAME, { "type" => "candle_bert", "model" => EMBEDDER_MODEL })
  schema.add_text_field("title")
  schema.add_text_field("text")
  schema.add_text_field("category")
  schema.add_integer_field("page")
  schema.add_flat_field("text_vec", DIM, distance: "cosine", embedder: EMBEDDER_NAME)
  schema.set_default_fields(%w[text])

  index = Laurus::Index.new(schema: schema)

  # ── Index ──────────────────────────────────────────────────────────────
  puts "--- Indexing chunked documents ---\n\n"
  CHUNKS.each do |doc_id, title, text, page, category|
    index.add_document(doc_id, {
      "title" => title,
      "text" => text,
      "category" => category,
      "page" => page,
      "text_vec" => text
    })
  end
  index.commit
  puts "Indexed #{CHUNKS.size} chunks.\n\n"

  # =====================================================================
  # [A] Lexical-only search (baseline)
  # =====================================================================
  puts "=" * 60
  puts "[A] Lexical-only: term 'migrations' in text"
  puts "=" * 60
  print_results(index.search(Laurus::TermQuery.new("text", "migrations"), limit: 3).results)

  # =====================================================================
  # [B] Vector-only search (baseline)
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[B] Vector-only: 'database query builder'"
  puts "=" * 60
  print_results(index.search(Laurus::VectorTextQuery.new("text_vec", "database query builder"), limit: 3).results)

  # =====================================================================
  # [C] Hybrid search — RRF Fusion
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[C] Hybrid (RRF k=60): vector='template rendering' + lexical='action'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    lexical_query: Laurus::TermQuery.new("text", "action"),
    vector_query: Laurus::VectorTextQuery.new("text_vec", "template rendering"),
    fusion: Laurus::RRF.new(k: 60.0),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [D] Hybrid search — WeightedSum Fusion
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[D] Hybrid (WeightedSum 0.3/0.7): vector='testing isolation' + lexical='mocks'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    lexical_query: Laurus::TermQuery.new("text", "mocks"),
    vector_query: Laurus::VectorTextQuery.new("text_vec", "testing isolation"),
    fusion: Laurus::WeightedSum.new(lexical_weight: 0.3, vector_weight: 0.7),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [E] Hybrid search with filter query
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[E] Hybrid + filter: vector='package distribution' + lexical='gem' + category='tooling'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    lexical_query: Laurus::TermQuery.new("text", "gem"),
    vector_query: Laurus::VectorTextQuery.new("text_vec", "package distribution"),
    filter_query: Laurus::TermQuery.new("category", "tooling"),
    fusion: Laurus::RRF.new(k: 60.0),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [F] DSL string query
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[F] DSL query string: 'text:middleware'"
  puts "=" * 60
  print_results(index.search("text:middleware", limit: 3).results)

  puts "\nHybrid search example completed!"
end

main
