//! Captures the warnings laurus logs, for integration tests that check what
//! a failure logs (Issue #1500).
//!
//! Include it with `#[path = "common/log_capture.rs"] mod log_capture;`
//! rather than through `common/mod.rs`, so test binaries that do not use it
//! do not compile it.

use std::sync::{Mutex, Once};
use std::thread::{self, ThreadId};

/// Every warning (and error) logged so far, with the thread that logged it.
static WARNINGS: Mutex<Vec<(ThreadId, String)>> = Mutex::new(Vec::new());

/// A logger that keeps every warning and error in [`WARNINGS`].
struct Capture;

impl log::Log for Capture {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            WARNINGS
                .lock()
                .unwrap()
                .push((thread::current().id(), record.args().to_string()));
        }
    }

    fn flush(&self) {}
}

/// Install the capturing logger as this test binary's logger, once.
pub fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        log::set_logger(&Capture).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
    });
}

/// The warnings logged since [`install`], with the thread that logged each.
///
/// The tests of a binary run in parallel, so a test looks only for the
/// warnings its own failure produces: by their text, or by its own thread
/// when the code under test logs on the caller's thread.
pub fn warnings() -> Vec<(ThreadId, String)> {
    WARNINGS.lock().unwrap().clone()
}
