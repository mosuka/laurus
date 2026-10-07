//! Server configuration types deserialized from a TOML file.
//!
//! The top-level [`Config`] struct contains sections for the gRPC/HTTP server,
//! index storage, and logging. All sections have sensible defaults so that
//! a minimal (or even empty) TOML file produces a working configuration.

use laurus::{CommitPolicy, DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, WalSyncPolicy};
use serde::Deserialize;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Default for [`ServerConfig::max_result_window`], matching Elasticsearch's
/// `index.max_result_window`.
pub const DEFAULT_MAX_RESULT_WINDOW: NonZeroUsize = NonZeroUsize::new(10_000).unwrap();

/// Top-level configuration loaded from a TOML file.
#[derive(Debug, Deserialize, Default)]
pub struct Config {
    /// Network settings for the gRPC server and the optional HTTP gateway.
    #[serde(default)]
    pub server: ServerConfig,
    /// Index storage settings (e.g. data directory path).
    #[serde(default)]
    pub index: IndexConfig,
}

/// Server network configuration.
#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    /// Listen address for the gRPC server.
    #[serde(default = "default_host")]
    pub host: String,
    /// Listen port for the gRPC server.
    #[serde(default = "default_port")]
    pub port: u16,
    /// Listen port for the HTTP Gateway. The Gateway is started only when this is set.
    #[serde(default)]
    pub http_port: Option<u16>,
    /// Largest `offset + limit` a search may request (Issue #1367). Larger
    /// requests are rejected with `INVALID_ARGUMENT` (HTTP 400). Must be at
    /// least 1.
    #[serde(default = "default_max_result_window")]
    pub max_result_window: NonZeroUsize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            http_port: None,
            max_result_window: default_max_result_window(),
        }
    }
}

/// Index storage settings.
#[derive(Debug, Deserialize)]
pub struct IndexConfig {
    /// Filesystem path where the index data (schema and store) is persisted.
    /// Defaults to `"./laurus_data"`.
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    /// Write-ahead log durability settings. Defaults to per-record fsync.
    #[serde(default)]
    pub wal: WalConfig,
    /// Auto-commit settings. Defaults to manual (caller/server-driven commits).
    #[serde(default)]
    pub commit: CommitConfig,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            wal: WalConfig::default(),
            commit: CommitConfig::default(),
        }
    }
}

/// The WAL sync policy selector deserialized from the config file.
///
/// Mirrors the variants of [`laurus::WalSyncPolicy`] without their parameters;
/// the [`WalConfig`] group thresholds are applied when the policy is
/// [`SyncPolicyKind::Group`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SyncPolicyKind {
    /// Fsync after every record (default); maps to [`WalSyncPolicy::PerRecord`].
    #[default]
    PerRecord,
    /// Batch fsyncs; maps to [`WalSyncPolicy::Group`] with the [`WalConfig`]
    /// thresholds.
    Group,
}

/// Write-ahead log durability configuration.
///
/// Selects the [`laurus::WalSyncPolicy`] used for the index. The default is
/// [`SyncPolicyKind::PerRecord`], preserving per-record durability. When
/// `sync_policy = "group"`, the optional `group_*` thresholds tune the batch;
/// unset thresholds fall back to [`laurus::DEFAULT_GROUP_MAX_RECORDS`] /
/// [`laurus::DEFAULT_GROUP_MAX_BYTES`].
#[derive(Debug, Clone, Copy, Deserialize, Default)]
pub struct WalConfig {
    /// Which durability policy to use. Defaults to `per_record`.
    #[serde(default)]
    pub sync_policy: SyncPolicyKind,
    /// Flush once this many records accumulate (group policy only). Falls back
    /// to [`laurus::DEFAULT_GROUP_MAX_RECORDS`] when unset.
    #[serde(default)]
    pub group_max_records: Option<usize>,
    /// Flush once this many bytes accumulate (group policy only). Falls back to
    /// [`laurus::DEFAULT_GROUP_MAX_BYTES`] when unset.
    #[serde(default)]
    pub group_max_bytes: Option<usize>,
    /// Periodic flush interval in milliseconds (group policy only). When unset
    /// no background timer runs. Honored on native targets only.
    #[serde(default)]
    pub group_max_interval_ms: Option<u64>,
}

impl WalConfig {
    /// Convert this configuration into a [`laurus::WalSyncPolicy`].
    ///
    /// # Returns
    ///
    /// [`WalSyncPolicy::PerRecord`] when `sync_policy` is `per_record`, or
    /// [`WalSyncPolicy::Group`] with the configured thresholds (defaults applied
    /// for unset values) when `sync_policy` is `group`.
    pub fn to_policy(&self) -> WalSyncPolicy {
        match self.sync_policy {
            SyncPolicyKind::PerRecord => WalSyncPolicy::PerRecord,
            SyncPolicyKind::Group => WalSyncPolicy::Group {
                max_records: self.group_max_records.unwrap_or(DEFAULT_GROUP_MAX_RECORDS),
                max_bytes: self.group_max_bytes.unwrap_or(DEFAULT_GROUP_MAX_BYTES),
                max_interval: self.group_max_interval_ms.map(Duration::from_millis),
            },
        }
    }
}

/// The auto-commit policy selector deserialized from the config file.
///
/// Mirrors the variants of [`laurus::CommitPolicy`] without their parameters;
/// the [`CommitConfig::every_docs`] threshold is applied when the policy is
/// [`CommitPolicyKind::EveryDocs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CommitPolicyKind {
    /// Never auto-commit (default); maps to [`CommitPolicy::Manual`].
    #[default]
    Manual,
    /// Commit every `every_docs` applied documents; maps to
    /// [`CommitPolicy::EveryDocs`].
    EveryDocs,
    /// Auto-commit at least every `interval_ms` milliseconds via a background
    /// timer; maps to [`CommitPolicy::Interval`] (Issue #892). Native-only: on
    /// `wasm32` the engine never starts the timer, so this is a no-op there.
    Interval,
}

/// Auto-commit configuration.
///
/// Selects the [`laurus::CommitPolicy`] used for the index. The default is
/// [`CommitPolicyKind::Manual`], preserving caller-driven commits. When
/// `policy = "every_docs"`, the optional `every_docs` threshold sets the
/// document count between commits; an unset (or `0`) threshold disables
/// auto-commit, matching [`CommitPolicy::EveryDocs`]`(0)`. When
/// `policy = "interval"`, the optional `interval_ms` threshold sets the
/// background auto-commit interval in milliseconds; an unset (or `0`)
/// interval disables the timer, matching [`CommitPolicy::Interval`]`(0)`.
#[derive(Debug, Clone, Copy, Deserialize, Default)]
pub struct CommitConfig {
    /// Which auto-commit policy to use. Defaults to `manual`.
    #[serde(default)]
    pub policy: CommitPolicyKind,
    /// Commit after this many applied documents (`every_docs` policy only).
    /// Unset (or `0`) disables auto-commit.
    #[serde(default)]
    pub every_docs: Option<usize>,
    /// Auto-commit interval in milliseconds (`interval` policy only). Unset
    /// (or `0`) disables the timer, matching [`CommitPolicy::Interval`]`(0)`.
    #[serde(default)]
    pub interval_ms: Option<u64>,
}

impl CommitConfig {
    /// Convert this configuration into a [`laurus::CommitPolicy`].
    ///
    /// # Returns
    ///
    /// [`CommitPolicy::Manual`] when `policy` is `manual`,
    /// [`CommitPolicy::EveryDocs`] with the configured threshold (`0` when
    /// unset, which the engine treats as disabled) when `policy` is
    /// `every_docs`, or [`CommitPolicy::Interval`] with the configured
    /// millisecond interval (`0` when unset) when `policy` is `interval`.
    pub fn to_policy(&self) -> CommitPolicy {
        match self.policy {
            CommitPolicyKind::Manual => CommitPolicy::Manual,
            CommitPolicyKind::EveryDocs => CommitPolicy::EveryDocs(self.every_docs.unwrap_or(0)),
            CommitPolicyKind::Interval => {
                CommitPolicy::Interval(Duration::from_millis(self.interval_ms.unwrap_or(0)))
            }
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    50051
}

fn default_max_result_window() -> NonZeroUsize {
    DEFAULT_MAX_RESULT_WINDOW
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("./laurus_data")
}

impl Config {
    /// Load configuration from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Filesystem path to the TOML configuration file.
    ///
    /// # Returns
    ///
    /// A fully populated [`Config`] instance with defaults applied for any
    /// missing sections or fields.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if the TOML content
    /// cannot be deserialized into a [`Config`].
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_wal_policy_is_per_record() {
        // An empty config (no [index.wal] section) must keep per-record fsync.
        let config: Config = toml::from_str("").unwrap();
        assert_eq!(config.index.wal.sync_policy, SyncPolicyKind::PerRecord);
        assert_eq!(config.index.wal.to_policy(), WalSyncPolicy::PerRecord);
    }

    #[test]
    fn group_policy_uses_defaults_when_thresholds_omitted() {
        let toml = r#"
            [index.wal]
            sync_policy = "group"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.index.wal.sync_policy, SyncPolicyKind::Group);
        assert_eq!(
            config.index.wal.to_policy(),
            WalSyncPolicy::Group {
                max_records: DEFAULT_GROUP_MAX_RECORDS,
                max_bytes: DEFAULT_GROUP_MAX_BYTES,
                max_interval: None,
            }
        );
    }

    #[test]
    fn group_policy_honors_explicit_thresholds_and_interval() {
        let toml = r#"
            [index.wal]
            sync_policy = "group"
            group_max_records = 256
            group_max_bytes = 4096
            group_max_interval_ms = 500
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(
            config.index.wal.to_policy(),
            WalSyncPolicy::Group {
                max_records: 256,
                max_bytes: 4096,
                max_interval: Some(Duration::from_millis(500)),
            }
        );
    }

    #[test]
    fn default_commit_policy_is_manual() {
        // An empty config (no [index.commit] section) must keep manual commits.
        let config: Config = toml::from_str("").unwrap();
        assert_eq!(config.index.commit.policy, CommitPolicyKind::Manual);
        assert_eq!(config.index.commit.to_policy(), CommitPolicy::Manual);
    }

    #[test]
    fn every_docs_policy_maps_threshold() {
        let toml = r#"
            [index.commit]
            policy = "every_docs"
            every_docs = 1000
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.index.commit.policy, CommitPolicyKind::EveryDocs);
        assert_eq!(
            config.index.commit.to_policy(),
            CommitPolicy::EveryDocs(1000)
        );
    }

    #[test]
    fn every_docs_policy_defaults_threshold_to_zero_when_omitted() {
        // policy = "every_docs" without a threshold maps to EveryDocs(0), which
        // the engine treats as disabled (equivalent to Manual).
        let toml = r#"
            [index.commit]
            policy = "every_docs"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.index.commit.to_policy(), CommitPolicy::EveryDocs(0));
    }

    #[test]
    fn interval_policy_maps_duration() {
        let toml = r#"
            [index.commit]
            policy = "interval"
            interval_ms = 500
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.index.commit.policy, CommitPolicyKind::Interval);
        assert_eq!(
            config.index.commit.to_policy(),
            CommitPolicy::Interval(Duration::from_millis(500))
        );
    }

    #[test]
    fn max_result_window_defaults_to_10000_and_reads_toml() {
        let empty: Config = toml::from_str("").unwrap();
        assert_eq!(empty.server.max_result_window.get(), 10_000);
        assert_eq!(
            ServerConfig::default().max_result_window,
            DEFAULT_MAX_RESULT_WINDOW
        );

        let set: Config = toml::from_str("[server]\nmax_result_window = 50000").unwrap();
        assert_eq!(set.server.max_result_window.get(), 50_000);
    }

    #[test]
    fn max_result_window_rejects_zero() {
        let err = toml::from_str::<Config>("[server]\nmax_result_window = 0").unwrap_err();
        assert!(err.to_string().contains("max_result_window"), "{err}");
    }
}
