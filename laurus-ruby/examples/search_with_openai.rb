# frozen_string_literal: true

# Search with OpenAI Embedder — real vector search using OpenAI's API.
#
# This example mirrors search_with_openai.rs but produces embeddings on the
# Ruby side using the ruby-openai gem, then passes the resulting vectors to
# Laurus as VectorQuery.
#
# Prerequisites:
#   gem install ruby-openai
#   export OPENAI_API_KEY=your-api-key-here
#   cd laurus-ruby
#   bundle exec rake compile
#   ruby -Ilib examples/search_with_openai.rb

require "laurus"

# ---------------------------------------------------------------------------
# OpenAI embedding helper
# ---------------------------------------------------------------------------

begin
  require "openai"
rescue LoadError
  puts "ERROR: ruby-openai gem not installed."
  puts "       Install with: gem install ruby-openai"
  exit 1
end

MODEL = "text-embedding-3-small"
DIM = 1536

def embed(client, text)
  response = client.embeddings(parameters: { model: MODEL, input: text })
  response.dig("data", 0, "embedding")
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
  api_key = ENV["OPENAI_API_KEY"]
  unless api_key
    puts "ERROR: OPENAI_API_KEY environment variable not set."
    puts "       export OPENAI_API_KEY=your-api-key-here"
    exit 1
  end

  puts "=== Laurus Search with OpenAI Embedder ===\n\n"

  client = OpenAI::Client.new(access_token: api_key)
  puts "OpenAI embedder ready (model=#{MODEL}, dim=#{DIM})\n\n"

  # -- Schema ---------------------------------------------------------------
  schema = Laurus::Schema.new
  schema.add_text_field("title")
  schema.add_text_field("text")
  schema.add_text_field("category")
  schema.add_integer_field("page")
  schema.add_flat_field("text_vec", DIM, distance: "cosine")
  schema.set_default_fields(%w[text])

  index = Laurus::Index.new(schema: schema)

  # -- Index ----------------------------------------------------------------
  puts "--- Indexing chunked documents ---\n\n"
  CHUNKS.each do |doc_id, title, text, page, category|
    vec = embed(client, text)
    index.add_document(doc_id, {
      "title" => title,
      "text" => text,
      "category" => category,
      "page" => page,
      "text_vec" => vec
    })
    puts "  Indexed #{doc_id} page #{page}: #{text[0, 50].inspect}..."
  end
  index.commit
  puts "\nIndexed #{CHUNKS.size} chunks.\n\n"

  # =====================================================================
  # [A] Vector Search
  # =====================================================================
  puts "=" * 60
  puts "[A] Vector Search: 'database ORM queries'"
  puts "=" * 60
  print_results(index.search(Laurus::VectorQuery.new("text_vec", embed(client, "database ORM queries")), limit: 3).results)

  # =====================================================================
  # [B] Filtered Vector Search -- category filter
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[B] Filtered Vector Search: 'database ORM queries' + category='testing'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    vector_query: Laurus::VectorQuery.new("text_vec", embed(client, "database ORM queries")),
    filter_query: Laurus::TermQuery.new("category", "testing"),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [C] Filtered Vector Search -- numeric range filter
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[C] Filtered Vector Search: 'web server HTTP' + page=1"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    vector_query: Laurus::VectorQuery.new("text_vec", embed(client, "web server HTTP")),
    filter_query: Laurus::NumericRangeQuery.new("page", min: 1, max: 1),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [D] Lexical Search
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[D] Lexical Search: 'middleware'"
  puts "=" * 60
  print_results(index.search(Laurus::TermQuery.new("text", "middleware"), limit: 3).results)

  # =====================================================================
  # [E] Hybrid Search (RRF)
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[E] Hybrid Search (RRF): vector='template rendering' + lexical='action'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    lexical_query: Laurus::TermQuery.new("text", "action"),
    vector_query: Laurus::VectorQuery.new("text_vec", embed(client, "template rendering")),
    fusion: Laurus::RRF.new(k: 60.0),
    limit: 3
  )
  print_results(index.search(request).results)

  puts "\nSearch with OpenAI example completed!"
end

main
