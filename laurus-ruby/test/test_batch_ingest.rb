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

  def test_put_documents_rejects_malformed_entry
    idx = Laurus::Index.new
    error = assert_raises(ArgumentError) do
      idx.put_documents([["ok", { "title" => "fine" }], "not-a-pair"])
    end
    assert_includes error.message, "documents[1]"
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
