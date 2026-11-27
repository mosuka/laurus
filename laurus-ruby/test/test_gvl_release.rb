# frozen_string_literal: true

require_relative "test_helper"

# Prove that Index methods release the GVL (Global VM Lock) while doing
# Rust/tokio work (Issue #1103).
#
# Before this change, every `Index` method held the GVL for the whole
# duration of its `rt.block_on(...)` call. A background Ruby thread calling
# another `Index` method could not even *start* running until the
# foreground call returned, because acquiring the GVL is a prerequisite for
# making any Ruby call at all -- including entering another `Index` method.
#
# These tests use that fact directly: kick off a slow call on one thread,
# and from a second thread, call a quick `Index` method shortly after the
# slow one starts. If the GVL is genuinely released while the slow call's
# Rust/tokio work runs, the quick call can begin (and finish) *before* the
# slow one completes. If the GVL were held for the whole slow call, the
# quick call could not even begin until the slow call returns the GVL, so
# it would necessarily start after the slow call finishes.
#
# This mirrors the approach used for the Python binding's equivalent test
# (`laurus-python/tests/test_gil_release.py`): it measures actual
# interleaving of two genuine `Index` calls, which is a more reliable
# signal than inferring lock state from a background counter's progress.
class TestGvlRelease < Minitest::Test
  def big_docs(n = 50_000)
    (0...n).map { |i| ["doc#{i}", { "title" => "document number #{i} rust ruby search engine benchmark" }] }
  end

  def run_concurrently(quick_delay: 0.02)
    start_barrier = Queue.new
    times = {}

    slow_thread = Thread.new do
      start_barrier.pop
      times[:slow_start] = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      yield(:slow)
      times[:slow_end] = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    end

    quick_thread = Thread.new do
      start_barrier.pop
      sleep(quick_delay)
      times[:quick_start] = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      yield(:quick)
      times[:quick_end] = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    end

    2.times { start_barrier << true }
    slow_thread.join
    quick_thread.join
    times
  end

  def assert_quick_call_was_not_blocked_by(times, slow_label, quick_label, min_slow_duration: 0.05)
    slow_duration = times[:slow_end] - times[:slow_start]
    assert slow_duration > min_slow_duration,
           "#{slow_label} finished too fast for this test to be meaningful " \
           "(#{slow_duration}s) -- make the workload heavier"

    assert times[:quick_start] < times[:slow_end],
           "#{quick_label} did not start until after the concurrent #{slow_label} " \
           "finished -- the GVL may not be released during #{slow_label}"
  end

  def test_search_starts_before_a_concurrent_put_documents_finishes
    idx = Laurus::Index.new
    idx.put_document("seed", { "title" => "hello world" })
    idx.commit
    docs = big_docs

    times = run_concurrently do |which|
      if which == :slow
        idx.put_documents(docs)
      else
        idx.search("title:hello", limit: 5)
      end
    end
    assert_quick_call_was_not_blocked_by(times, "put_documents", "search")
  end

  def test_search_starts_before_a_concurrent_commit_finishes
    idx = Laurus::Index.new
    idx.put_document("seed", { "title" => "hello world" })
    idx.commit
    idx.put_documents(big_docs) # left uncommitted; commit below applies it

    times = run_concurrently do |which|
      if which == :slow
        idx.commit
      else
        idx.search("title:hello", limit: 5)
      end
    end
    assert_quick_call_was_not_blocked_by(times, "commit", "search", min_slow_duration: 0.01)
  end

  def test_search_starts_before_a_concurrent_search_batch_finishes
    idx = Laurus::Index.new
    idx.put_document("seed", { "title" => "hello world" })
    idx.put_documents(big_docs)
    idx.commit

    times = run_concurrently do |which|
      if which == :slow
        idx.search_batch(["title:document"] * 200, limit: 10)
      else
        idx.search("title:hello", limit: 5)
      end
    end
    assert_quick_call_was_not_blocked_by(times, "search_batch", "search")
  end

  def test_concurrent_searches_from_multiple_threads_all_return_correct_results
    idx = Laurus::Index.new
    idx.put_documents(big_docs(2_000))
    idx.put_document("needle", { "title" => "unique findable term" })
    idx.commit

    # Collect results per thread and assert on the main thread afterward --
    # Minitest's assertions are not documented as thread-safe.
    ids_per_thread = Array.new(8) { [] }
    threads = (0...8).map do |i|
      Thread.new do
        10.times do
          results = idx.search("title:unique", limit: 5).results
          ids_per_thread[i] << results.map(&:id)
        end
      end
    end
    threads.each(&:join)

    ids_per_thread.each do |runs|
      runs.each { |ids| assert_equal ["needle"], ids }
    end
  end
end
