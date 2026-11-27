# frozen_string_literal: true

# Multimodal Search Example — searching across text and image bytes fields.
#
# This example demonstrates how to store raw bytes (e.g. image data) in a
# Laurus index and perform multimodal search using pre-computed vectors.
#
# The Rust multimodal_search.rs example uses the built-in CandleClipEmbedder.
# In Ruby you can achieve the same result by:
# 1. Encoding images/text with a CLIP model externally.
# 2. Storing the raw bytes in a bytes field and the embedding vector in a flat field.
# 3. Querying with VectorQuery using a pre-computed embedding.
#
# NOTE: Ruby does not have a widely-available native CLIP library, so this
# example uses random fallback vectors. Semantic similarity will NOT be
# meaningful. For production use, call a CLIP API (e.g. via HTTP) and pass
# the resulting vectors to VectorQuery.
#
# Usage:
#   cd laurus-ruby
#   bundle exec rake compile
#   ruby -Ilib examples/multimodal_search.rb

require "laurus"

# ---------------------------------------------------------------------------
# Embedding helper (random fallback -- no Ruby CLIP library available)
# ---------------------------------------------------------------------------

DIM = 32

def embed_text(text)
  seed = text.bytes.reduce(0) { |acc, b| (acc * 31 + b) & 0xFFFFFFFF }
  rng = Random.new(seed)
  raw = Array.new(DIM) { rng.rand(-1.0..1.0) }
  norm = Math.sqrt(raw.sum { |x| x * x })
  norm = 1.0 if norm.zero?
  raw.map { |x| x / norm }
end

def embed_image(image_bytes)
  seed = image_bytes[0, 128].bytes.reduce(0) { |acc, b| (acc * 31 + b) & 0xFFFFFFFF }
  rng = Random.new(seed)
  raw = Array.new(DIM) { rng.rand(-1.0..1.0) }
  norm = Math.sqrt(raw.sum { |x| x * x })
  norm = 1.0 if norm.zero?
  raw.map { |x| x / norm }
end

puts "[NOTE] Using random fallback vectors (no native CLIP library for Ruby)."
puts "       Semantic similarity will NOT be meaningful."
puts "       For production, call a CLIP API and pass vectors to VectorQuery."
puts

# ---------------------------------------------------------------------------
# Fake image bytes for demo (1x1 white pixel PNG)
# ---------------------------------------------------------------------------

WHITE_PNG = (
  "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01" \
  "\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00" \
  "\x00\x01\x01\x00\x05\x18\xd4n\x00\x00\x00\x00IEND\xaeB`\x82"
).dup.force_encoding("BINARY").freeze

def fake_image(label)
  WHITE_PNG + label.encode("BINARY")
end

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGES = [
  ["img_rails1", "rails_console.png", "image"],
  ["img_rails2", "rails_routes.png", "image"],
  ["img_sinatra1", "sinatra_app.png", "image"],
  ["img_rspec1", "rspec_output.png", "image"]
].freeze

TEXTS = [
  ["txt1", "A Rails console session showing Active Record queries", "text"],
  ["txt2", "Sinatra route handler responding with JSON", "text"],
  ["txt3", "RSpec test output with green passing examples", "text"],
  ["txt4", "Bundler dependency graph for a Ruby project", "text"]
].freeze

def print_results(results)
  if results.empty?
    puts "  (no results)"
    return
  end
  results.each do |r|
    doc = r.document || {}
    label = doc["filename"]
    label = doc["description"] if label.nil? || label.empty?
    label ||= ""
    media_type = doc["type"] || "?"
    printf "  id=%-12s  score=%.4f  [%s] %s\n", r.id.inspect, r.score, media_type, label.inspect[0, 55]
  end
end

def main
  puts "=== Laurus Multimodal Search Example ===\n\n"
  puts "Using random fallback vectors (results not semantically meaningful).\n\n"

  # -- Schema ---------------------------------------------------------------
  schema = Laurus::Schema.new
  schema.add_bytes_field("content")       # raw image bytes or nil
  schema.add_text_field("filename")
  schema.add_text_field("type")
  schema.add_text_field("description")
  schema.add_flat_field("content_vec", DIM, distance: "cosine")

  index = Laurus::Index.new(schema: schema)

  # -- Index images ---------------------------------------------------------
  puts "--- Indexing images ---"
  IMAGES.each do |doc_id, filename, media_type|
    raw_bytes = fake_image(filename) # replace with File.binread(path)
    vec = embed_image(raw_bytes)
    index.add_document(doc_id, {
      "content" => raw_bytes,
      "filename" => filename,
      "type" => media_type,
      "description" => "",
      "content_vec" => vec
    })
    puts "  Indexed image: #{filename}"
  end

  # -- Index text descriptions ----------------------------------------------
  puts "\n--- Indexing text descriptions ---"
  TEXTS.each do |doc_id, text, media_type|
    vec = embed_text(text)
    index.add_document(doc_id, {
      "content" => nil,
      "filename" => "",
      "type" => media_type,
      "description" => text,
      "content_vec" => vec
    })
    puts "  Indexed text: #{text[0, 50].inspect}"
  end

  index.commit
  puts

  # =====================================================================
  # [A] Text-to-Image: find images matching a text query
  # =====================================================================
  puts "=" * 60
  puts "[A] Text-to-Image: query='a screenshot of Rails console'"
  puts "=" * 60
  query_vec = embed_text("a screenshot of Rails console")
  print_results(index.search(Laurus::VectorQuery.new("content_vec", query_vec), limit: 3).results)

  # =====================================================================
  # [B] Text-to-Text: find text descriptions
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[B] Text-to-Text: query='testing output', filter type='text'"
  puts "=" * 60
  request = Laurus::SearchRequest.new(
    vector_query: Laurus::VectorQuery.new("content_vec", embed_text("testing output")),
    filter_query: Laurus::TermQuery.new("type", "text"),
    limit: 3
  )
  print_results(index.search(request).results)

  # =====================================================================
  # [C] Image-to-Anything: find documents similar to a given image
  # =====================================================================
  puts "\n#{"=" * 60}"
  puts "[C] Image-to-Anything: query from 'rails_console.png'"
  puts "=" * 60
  query_img_bytes = fake_image("rails_console.png")
  query_vec = embed_image(query_img_bytes)
  print_results(index.search(Laurus::VectorQuery.new("content_vec", query_vec), limit: 3).results)

  puts "\nMultimodal search example completed!"
end

main
