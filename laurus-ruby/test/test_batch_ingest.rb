# frozen_string_literal: true

require_relative "test_helper"

# Integration tests for Laurus::Index#put_documents / #add_documents (#866).
class TestBatchIngest < Minitest::Test
  def test_put_documents_empty_batch_is_noop
    idx = Laurus::Index.new
    idx.put_documents([])
    idx.add_documents([])
    idx.commit
    assert_equal 0, idx.stats["document_count"]
  end

  def test_put_documents_applies_and_dedupes
    idx = Laurus::Index.new
    idx.put_documents([
                        ["doc1", { "title" => "One" }],
                        ["doc2", { "title" => "Two" }],
                        ["doc1", { "title" => "One v2" }] # duplicate id: last wins
                      ])
    idx.commit

    assert_equal 2, idx.stats["document_count"]
    docs = idx.get_documents("doc1")
    assert_equal 1, docs.length
    assert_equal "One v2", docs[0]["title"]
  end

  def test_add_documents_accumulates_chunks
    idx = Laurus::Index.new
    idx.add_documents([
                        ["doc", { "title" => "chunk 0" }],
                        ["doc", { "title" => "chunk 1" }]
                      ])
    idx.commit

    assert_equal 2, idx.get_documents("doc").length
  end

  # Issue #1630: a malformed pair raises TypeError for a wrong type (it was
  # an ArgumentError) and ArgumentError for the wrong number of elements,
  # naming its position.
  def test_put_documents_rejects_malformed_entry
    cases = {
      "not a pair" => ["not-a-pair", TypeError, "documents[1]: expected an [id, hash] pair"],
      "one element" => [["b"], ArgumentError, "documents[1]: expected a 2-element [id, hash] pair, got 1 elements"],
      "three elements" => [["b", {}, 1], ArgumentError,
                           "documents[1]: expected a 2-element [id, hash] pair, got 3 elements"],
      "id not a String" => [[1, {}], TypeError, "documents[1]: id must be a String"],
      "document not a Hash" => [%w[b x], TypeError, "documents[1]: document must be a Hash"]
    }
    cases.each do |label, (entry, error_class, message)|
      idx = Laurus::Index.new
      error = assert_raises(StandardError, label) do
        idx.put_documents([["ok", { "title" => "fine" }], entry])
      end
      assert_equal error_class, error.class, label
      assert_equal message, error.message, label
    end
  end

  # Issue #1630: a document whose content cannot be converted fails the
  # batch with the exception put_document raises for it, its message
  # starting with the document's position.
  def test_batch_names_the_document_that_cannot_be_converted
    %i[put_documents add_documents].each do |method|
      idx = Laurus::Index.new
      single = assert_raises(StandardError) { idx.put_document("b", { "x" => Object.new }) }
      batch = assert_raises(StandardError, method.to_s) do
        idx.public_send(method, [["a", { "title" => "a" }], ["b", { "x" => Object.new }]])
      end
      assert_equal single.class, batch.class, method.to_s
      assert_equal "documents[1]: #{single.message}", batch.message, method.to_s
    end
  end

  # Issue #1627: a document the engine rejects fails the batch with the
  # ArgumentError the same document raises alone (it was a RuntimeError),
  # the message naming its position and id.
  def test_batch_raises_the_exception_of_the_failing_document
    %i[put_documents add_documents].each do |method|
      idx = Laurus::Index.new
      single = assert_raises(ArgumentError) { idx.put_document("b", { "_x" => "v" }) }
      batch = assert_raises(ArgumentError, method.to_s) do
        idx.public_send(method, [["a", { "title" => "a" }], ["b", { "_x" => "v" }]])
      end
      assert_includes batch.message, "doc 1 (id 'b')", method.to_s
      assert batch.message.end_with?(single.message), "#{method}: #{batch.message}"
    end
  end
end
