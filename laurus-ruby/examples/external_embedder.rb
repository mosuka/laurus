# frozen_string_literal: true

# External Embedder Example — vector and hybrid search with a user-managed embedder.
#
# Demonstrates vector and hybrid search where embeddings are produced outside
# laurus and passed as pre-computed vectors via VectorQuery.
#
# This approach is useful when you want to:
# - Use any embedding library (ruby-openai, informers, etc.)
# - Control the embedding model independently of the index schema
# - Reuse embeddings across multiple indexes
#
# For an alternative that lets laurus handle embeddings automatically, see
# vector_search.rb and hybrid_search.rb which use the built-in
# CandleBert embedder via schema.add_embedder().
#
# Usage:
#   gem install informers        # optional but recommended
#   cd laurus-ruby
#   bundle exec rake compile
#   ruby -Ilib examples/external_embedder.rb

require "laurus"

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

begin
  require "informers"

  _pipeline = Informers.pipeline("feature-extraction", "sentence-transformers/all-MiniLM-L6-v2")
  DIM = 384

  define_method(:embed) do |text|
    result = _pipeline.call(text)
    # Normalize to unit vector
    norm = Math.sqrt(result.sum { |x| x * x })
    norm = 1.0 if norm.zero?
    result.map { |x| x / norm }
  end

  puts "Using informers (sentence-transformers/all-MiniLM-L6-v2) for embeddings.\n\n"
rescue LoadError
  # Fallback: deterministic pseudo-embeddings for demo purposes only.
  # Real similarity is not meaningful with these vectors.
  DIM = 64

  define_method(:embed) do |text|
    seed = text.bytes.reduce(0) { |acc, b| (acc * 31 + b) & 0xFFFFFFFF }
    rng = Random.new(seed)
    raw = Array.new(DIM) { rng.rand(-1.0..1.0) }
    norm = Math.sqrt(raw.sum { |x| x * x })
    norm = 1.0 if norm.zero?
    raw.map { |x| x / norm }
  end

  puts "[NOTE] informers gem not found -- using random fallback vectors."
  puts "       Results will NOT reflect semantic similarity."
  puts "       Install with: gem install informers"
  puts
end

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
  puts "=== Laurus External Embedder Example ===\n\n"
  puts "Embedding dimension: #{DIM}\n\n"

  # -- Schema ---------------------------------------------------------------
  # No embedder is registered in the schema; vectors are provided by the
  # caller at index and query time via embed().
  schema = Laurus::Schema.new
  schema.add_text_field("title")
  schema.add_text_field("text")
  schema.add_text_field("category")
  schema.add_integer_field("page")
  schema.add_flat_field("text_vec", DIM, distance: "cosine")
  schema.set_default_fields(%w[text])

  index = Laurus::Index.new(schema: schema)

  # -- Index ----------------------------------------------------------------
  # Embeddings are computed here in Ruby and stored as float vectors.
  puts "--- Indexing chunked documents ---\n\n"
  CHUNKS.each do |doc_id, title, text, page, category|
    index.add_document(doc_id, {
      "title" => title,
      "text" => text,
      "category" => category,
      "page" => page,
      "text_vec" => embed(text)
    })
  end
  index.commit
  puts "Indexed #{CHUNKS.size} chunks.\n\n"

  # =====================================================================
  # [A] Basic Vector Search
  # =====================================================================
  puts "=" * 60
  puts "[A] Vector-only: 'database ORM queries'"
  puts "=" * 60
  print_results(index.search(Laurus::VectorQuery.new("text_vec", embed("database ORM queries")), limit: 3).results)

  # =====================================================================
  # [B] Filtered Vector Search -- category filter
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[B] Filtered vector: 'database ORM queries' + category='testing'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    vector_query: Laurus::VectorQuery.new("text_vec", embed("database ORM queries")),
    filter_query: Laurus::TermQuery.new("category", "testing"),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [C] Hybrid search -- RRF Fusion
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[C] Hybrid (RRF k=60): vector='template rendering' + lexical='action'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    lexical_query: Laurus::TermQuery.new("text", "action"),
    vector_query: Laurus::VectorQuery.new("text_vec", embed("template rendering")),
    fusion: Laurus::RRF.new(k: 60.0),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [D] Hybrid search -- WeightedSum Fusion
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[D] Hybrid (WeightedSum 0.3/0.7): vector='testing isolation' + lexical='mocks'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    lexical_query: Laurus::TermQuery.new("text", "mocks"),
    vector_query: Laurus::VectorQuery.new("text_vec", embed("testing isolation")),
    fusion: Laurus::WeightedSum.new(lexical_weight: 0.3, vector_weight: 0.7),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [E] Hybrid search with filter
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[E] Hybrid + filter: vector='package distribution' + lexical='gem' + category='tooling'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    lexical_query: Laurus::TermQuery.new("text", "gem"),
    vector_query: Laurus::VectorQuery.new("text_vec", embed("package distribution")),
    filter_query: Laurus::TermQuery.new("category", "tooling"),
    fusion: Laurus::RRF.new(k: 60.0),
    limit: 3
  )
  print_results(index.search(request).results)

  puts "\nExternal embedder example completed!"
end

main
