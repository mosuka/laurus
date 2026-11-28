# frozen_string_literal: true

# Vector Search Example — semantic similarity search with embeddings.
#
# Demonstrates vector search using laurus's built-in CandleBert embedder:
# - Basic vector search (semantic similarity)
# - Filtered vector search (with lexical filters)
#
# The embedder is registered in the schema and laurus automatically converts
# text to vectors at index and query time — no external embedding library needed.
#
# Usage:
#   cd laurus-ruby
#   bundle exec rake compile  # build with: --features embeddings-candle
#   ruby -Ilib examples/vector_search.rb

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
  puts "=== Laurus Vector Search Example ===\n\n"
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
  # [A] Basic Vector Search
  # =====================================================================
  puts "=" * 60
  puts "[A] Basic Vector Search: 'database ORM queries'"
  puts "=" * 60
  print_results(index.search(Laurus::VectorTextQuery.new("text_vec", "database ORM queries"), limit: 3).results)

  # =====================================================================
  # [B] Filtered Vector Search — category filter
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[B] Filtered Vector Search: 'database ORM queries' + category='testing'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    vector_query: Laurus::VectorTextQuery.new("text_vec", "database ORM queries"),
    filter_query: Laurus::TermQuery.new("category", "testing"),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [C] Filtered Vector Search — numeric range filter (page = 1)
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[C] Filtered Vector Search: 'HTTP web routing' + page = 1"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    vector_query: Laurus::VectorTextQuery.new("text_vec", "HTTP web routing"),
    filter_query: Laurus::NumericRangeQuery.new("page", min: 1, max: 1),
    limit: 3
  )
  print_results(index.search(request).results)

  puts "\nVector search example completed!"
end

main
