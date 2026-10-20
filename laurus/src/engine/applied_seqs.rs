//! The WAL sequence numbers whose mutation has finished applying (Issue
//! #1499).
//!
//! A mutation appends its WAL record, which assigns the next sequence number,
//! and then applies it to the lexical and the vector store. Mutations of
//! different ids finish in any order, so the highest finished sequence number
//! can be above a record that is still being applied. A commit may truncate
//! the WAL, and record a store checkpoint, only up to the
//! [watermark](AppliedSeqs::watermark): the highest sequence number at or
//! below which every mutation finished.

use std::collections::BTreeSet;

use parking_lot::Mutex;

/// Tracks which WAL records finished applying to both stores.
///
/// Relies on the sequence numbers above the watermark being dense: every
/// number the engine assigns after [`Self::raise_to`] comes from a WAL append
/// whose mutation either calls [`Self::finish`] once both stores hold it, or
/// stops writes until the index is reopened. A number that never finishes
/// holds the watermark below it, so no commit truncates its record or
/// checkpoints past it.
#[derive(Debug, Default)]
pub(super) struct AppliedSeqs {
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    /// Every sequence number at or below this one finished.
    watermark: u64,
    /// Finished sequence numbers above `watermark + 1`, waiting for the gap
    /// below them to close. Empty unless mutations finish out of order.
    finished_above: BTreeSet<u64>,
    /// Grows with every finish and every raise; see [`AppliedSeqs::progress`].
    progress: u64,
}

impl State {
    /// Take the finished numbers that now continue the watermark.
    fn absorb_contiguous(&mut self) {
        while let Some(&next) = self.finished_above.first() {
            if next > self.watermark + 1 {
                break;
            }
            self.finished_above.pop_first();
            self.watermark = self.watermark.max(next);
        }
    }
}

impl AppliedSeqs {
    /// Record that the mutation of WAL record `seq` finished applying to
    /// both stores.
    ///
    /// # Arguments
    ///
    /// * `seq` - The sequence number the record's append assigned. Each one
    ///   finishes at most once.
    pub(super) fn finish(&self, seq: u64) {
        let mut state = self.state.lock();
        debug_assert!(
            seq > state.watermark && !state.finished_above.contains(&seq),
            "WAL record {seq} finished twice"
        );
        if seq == state.watermark + 1 {
            state.watermark = seq;
            state.absorb_contiguous();
        } else if seq > state.watermark {
            state.finished_above.insert(seq);
        } else {
            return;
        }
        state.progress += 1;
    }

    /// The highest sequence number at or below which every mutation
    /// finished: what a commit may truncate the WAL through and record as
    /// the stores' checkpoint.
    pub(super) fn watermark(&self) -> u64 {
        self.state.lock().watermark
    }

    /// A number that grows each time a mutation finishes or
    /// [`Self::raise_to`] raises the watermark, so that a commit can tell
    /// whether anything was applied since the previous one by comparing
    /// them. Neither the watermark nor the highest finished sequence number
    /// does: a mutation that finishes inside a gap below a later one moves
    /// neither.
    pub(super) fn progress(&self) -> u64 {
        self.state.lock().progress
    }

    /// Treat every sequence number at or below `seq` as finished.
    ///
    /// For [`Engine::recover`](super::Engine::recover), once the WAL is
    /// replayed and no mutation can be in flight: everything at or below the
    /// log's last sequence number is then in both stores.
    pub(super) fn raise_to(&self, seq: u64) {
        let mut state = self.state.lock();
        if seq > state.watermark {
            state.watermark = seq;
            state.absorb_contiguous();
            state.progress += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_order_finishes_advance_the_watermark() {
        let applied = AppliedSeqs::default();
        assert_eq!(applied.watermark(), 0);
        applied.finish(1);
        applied.finish(2);
        assert_eq!(applied.watermark(), 2);
        assert_eq!(applied.progress(), 2);
    }

    #[test]
    fn a_later_finish_waits_for_the_gap_below_it() {
        let applied = AppliedSeqs::default();
        applied.finish(1);
        applied.finish(3);
        applied.finish(4);
        assert_eq!(applied.watermark(), 1, "2 has not finished");

        applied.finish(2);
        assert_eq!(applied.watermark(), 4, "2 closes the gap below 3 and 4");
    }

    #[test]
    fn progress_grows_with_every_finish_even_inside_a_gap() {
        let applied = AppliedSeqs::default();
        applied.finish(3);
        let after_three = applied.progress();
        applied.finish(2);
        assert_eq!(applied.watermark(), 0, "1 has not finished");
        assert!(applied.progress() > after_three);
    }

    #[test]
    fn an_unfinished_number_holds_the_watermark() {
        let applied = AppliedSeqs::default();
        for seq in 2..=100 {
            applied.finish(seq);
        }
        assert_eq!(applied.watermark(), 0);
    }

    #[test]
    fn raise_to_covers_everything_below_and_absorbs_what_follows() {
        let applied = AppliedSeqs::default();
        applied.raise_to(10);
        assert_eq!(applied.watermark(), 10);
        assert_eq!(applied.progress(), 1);
        applied.finish(12);
        applied.raise_to(11);
        assert_eq!(applied.watermark(), 12);

        let progress = applied.progress();
        applied.raise_to(5);
        assert_eq!(applied.watermark(), 12, "raise_to never lowers it");
        assert_eq!(applied.progress(), progress, "nor counts as progress");
    }
}
