//! `laurus serve` takes the index directory from the global `--index-dir`
//! option or `LAURUS_INDEX_DIR`, with the default `./laurus_index`, as every
//! `laurus` command does, and a config file that names one with
//! `[index] data_dir` stops it from starting (Issue #1718). Before, the
//! server started and ignored the key.
//!
//! These tests run the real binary from an empty directory and read the
//! `Data directory:` line the server logs at startup.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// How long a server may take to log its data directory, or to exit.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

/// The log line, before the path, that names the directory the server opens.
const DATA_DIRECTORY: &str = "Data directory: ";

/// How a `laurus serve` run ended.
enum Outcome {
    /// The server started and logged this data directory; it was then killed.
    Started(String),
    /// The process exited before logging a data directory.
    Exited {
        status: ExitStatus,
        stdout: String,
        stderr: String,
    },
}

/// A `laurus` command run from `dir`, with none of the inherited settings
/// that would change where the index goes or which config file is read.
fn laurus(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_laurus"));
    cmd.current_dir(dir)
        .env("RUST_LOG", "info")
        .env("NO_COLOR", "1");
    for var in [
        "LAURUS_INDEX_DIR",
        "LAURUS_CONFIG",
        "LAURUS_HOST",
        "LAURUS_PORT",
        "LAURUS_HTTP_PORT",
        "LAURUS_MAX_RESULT_WINDOW",
    ] {
        cmd.env_remove(var);
    }
    cmd
}

/// Removes the ANSI escape sequences (`ESC [ ... letter`) a log line may
/// carry.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Runs `cmd` followed by `serve` on an ephemeral local port with `args`,
/// and returns once the server logs its data directory (killing it) or the
/// process exits.
fn serve(mut cmd: Command, args: &[&str]) -> Outcome {
    let mut child = cmd
        .args(["serve", "-H", "127.0.0.1", "-p", "0"])
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn laurus serve");
    let stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let (lines, received) = mpsc::channel();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if lines.send(strip_ansi(&line)).is_err() {
                break;
            }
        }
    });
    let stderr_reader = thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });

    let deadline = Instant::now() + STARTUP_TIMEOUT;
    let mut seen = Vec::new();
    let started = loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match received.recv_timeout(left) {
            Ok(line) => {
                if let Some(at) = line.find(DATA_DIRECTORY) {
                    break Some(line[at + DATA_DIRECTORY.len()..].trim().to_string());
                }
                seen.push(line);
            }
            Err(RecvTimeoutError::Disconnected) => break None,
            Err(RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "laurus serve neither logged a data directory nor exited within {:?}; stdout: {seen:?}",
                    STARTUP_TIMEOUT
                );
            }
        }
    };
    if started.is_some() {
        let _ = child.kill();
    }
    let status = child.wait().expect("wait for laurus serve");
    reader.join().unwrap();
    let stderr = stderr_reader.join().unwrap();
    match started {
        Some(dir) => Outcome::Started(dir),
        None => Outcome::Exited {
            status,
            stdout: seen.join("\n"),
            stderr,
        },
    }
}

/// The data directory `laurus serve` opens, given `cmd`.
fn served_data_dir(cmd: Command) -> String {
    match serve(cmd, &[]) {
        Outcome::Started(dir) => dir,
        Outcome::Exited { status, stderr, .. } => {
            panic!("laurus serve exited ({status}) instead of starting: {stderr}")
        }
    }
}

#[test]
fn the_index_dir_flag_names_the_directory() {
    let dir = tempfile::tempdir().unwrap();
    let mut cmd = laurus(dir.path());
    cmd.args(["--index-dir", "from_flag"]);
    assert_eq!(served_data_dir(cmd), "from_flag");
}

#[test]
fn the_environment_variable_names_the_directory() {
    let dir = tempfile::tempdir().unwrap();
    let mut cmd = laurus(dir.path());
    cmd.env("LAURUS_INDEX_DIR", "from_env");
    assert_eq!(served_data_dir(cmd), "from_env");
}

#[test]
fn the_directory_defaults_to_laurus_index() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(served_data_dir(laurus(dir.path())), "./laurus_index");
}

/// A config file with `[index] data_dir` stops the server before it starts,
/// with an error that names the file and the key and points to
/// `--index-dir`, whether or not the flag is also given.
#[test]
fn data_dir_in_the_config_file_stops_serve_from_starting() {
    for flag in [&[][..], &["--index-dir", "from_flag"][..]] {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.toml"),
            "[index]\ndata_dir = \"/var/lib/laurus/data\"\n",
        )
        .unwrap();
        let mut cmd = laurus(dir.path());
        cmd.args(flag);
        match serve(cmd, &["--config", "config.toml"]) {
            Outcome::Started(served) => {
                panic!("with {flag:?}, laurus serve started on {served} despite data_dir")
            }
            Outcome::Exited {
                status,
                stdout,
                stderr,
            } => {
                assert!(!status.success(), "with {flag:?}: {status}");
                for part in [
                    "Failed to parse config file: config.toml",
                    "data_dir",
                    "--index-dir",
                ] {
                    assert!(
                        stderr.contains(part),
                        "with {flag:?}, missing {part:?}: {stderr}"
                    );
                }
                assert!(
                    !stdout.contains("Laurus server starting"),
                    "with {flag:?}: {stdout}"
                );
            }
        }
    }
}
