# frozen_string_literal: true

# Quick start example for the laurus Ruby binding.
#
# Usage:
#   cd laurus-ruby
#   bundle exec rake compile
#   ruby -Ilib examples/quickstart.rb

require "laurus"

# Create a schema with two text fields.
schema = Laurus::Schema.new
schema.add_text_field("title")
schema.add_text_field("body")
schema.set_default_fields(%w[title body])

# Create an in-memory index.
index = Laurus::Index.new(schema: schema)

# Add documents.
index.add_document("doc1", { "title" => "Rails Web Framework", "body" => "Convention over configuration for rapid web development." })
index.add_document("doc2", { "title" => "RSpec Testing", "body" => "Behavior-driven development framework for Ruby." })
index.commit

# Search with a DSL string.
puts "=== DSL search ==="
results = index.search("ruby OR development", limit: 5).results
results.each do |r|
  puts "  #{r.id} (score: #{r.score}): #{r.document['title']}"
end

# Search with a TermQuery object.
puts "\n=== TermQuery search ==="
results = index.search(Laurus::TermQuery.new("body", "convention"), limit: 5).results
results.each do |r|
  puts "  #{r.id} (score: #{r.score}): #{r.document['title']}"
end

# Index statistics.
puts "\n=== Stats ==="
stats = index.stats
puts "  Document count: #{stats['document_count']}"
