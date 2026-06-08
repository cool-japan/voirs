//! In-house HuggingFace Hub downloader (pure-Rust TLS).
//!
//! This module replaces the external `hf-hub` crate with a small, dependency-light
//! downloader built directly on [`reqwest`] (configured for `rustls-no-provider`).
//! It installs a pure-Rust [`rustls`] [`rustls::crypto::CryptoProvider`] backed by RustCrypto as the
//! process-wide default, so that no `ring` or `aws-lc-rs`/`aws-lc-sys` C/assembly
//! crypto is compiled into the default build.
//!
//! # Crypto provider
//!
//! Because the workspace builds `reqwest` with the `rustls-no-provider` feature, the
//! process **must** install a default [`rustls::crypto::CryptoProvider`] before any TLS handshake.
//! [`ensure_crypto_provider`] does this exactly once via [`std::sync::Once`]; every
//! public entry point in this module calls it first. Binaries and FFI entry points
//! should also call it as early as possible (see `voirs_sdk::ensure_crypto_provider`).
//!
//! # Caching
//!
//! Downloaded files are cached under
//! `dirs::cache_dir()/voirs/hub/<repo with '/'→'--'>/<revision>/<filename>`.
//! A cached file that exists and is non-empty is returned without a network request.
//!
//! # Example
//!
//! ```no_run
//! # async fn run() -> Result<(), voirs_acoustic::hub::HubError> {
//! // Download a single file from a model repository.
//! let path = voirs_acoustic::hub::download_file(
//!     "facebook/fastspeech2-en-ljspeech",
//!     "model.safetensors",
//!     None,
//! )
//! .await?;
//! println!("cached at {}", path.display());
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;
use std::sync::Once;

/// Errors produced by the in-house HuggingFace Hub downloader.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// The HTTP transport failed (DNS, TLS, connection, body read, ...).
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// A local filesystem operation (create dir, write, rename, ...) failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The server responded with a non-success status code.
    #[error("HTTP request returned status {code}")]
    Status {
        /// The HTTP status code returned by the server.
        code: u16,
    },

    /// The requested resource was not found (HTTP 404).
    #[error("resource not found: {repo}/{path}")]
    NotFound {
        /// The repository identifier.
        repo: String,
        /// The path/filename that could not be located.
        path: String,
    },
}

/// Base URL of the HuggingFace Hub.
const HF_ENDPOINT: &str = "https://huggingface.co";

/// Default revision when none is supplied.
const DEFAULT_REVISION: &str = "main";

static INSTALL_PROVIDER: Once = Once::new();

/// Install the pure-Rust [`rustls`] [`rustls::crypto::CryptoProvider`] as the process default.
///
/// This is guarded by [`std::sync::Once`], so it is safe (and cheap) to call from
/// many places — only the first call performs the installation. The provider is the
/// RustCrypto-backed provider exposed by `oxitls-adapter-rustls-rustcrypto`.
///
/// [`CryptoProvider::install_default`](rustls::crypto::CryptoProvider::install_default)
/// takes the provider by value and returns `Err` if a default was already set; the
/// `let _ =` is intentional because a previously-installed provider (e.g. installed
/// by another component) is perfectly acceptable.
pub fn ensure_crypto_provider() {
    INSTALL_PROVIDER.call_once(|| {
        let provider = (*oxitls_adapter_rustls_rustcrypto::pure_provider()).clone();
        let _ = rustls::crypto::CryptoProvider::install_default(provider);
    });
}

/// Normalize a repository id for use as a single filesystem path component.
fn repo_to_dir_component(repo: &str) -> String {
    repo.replace('/', "--")
}

/// Compute the on-disk cache path for `repo`/`revision`/`filename`.
///
/// Layout: `dirs::cache_dir()/voirs/hub/<repo with '/'→'--'>/<revision>/<filename>`.
/// Falls back to the system temp dir when no cache dir can be determined.
fn cache_path(repo: &str, filename: &str, revision: &str) -> PathBuf {
    let base = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
    base.join("voirs")
        .join("hub")
        .join(repo_to_dir_component(repo))
        .join(revision)
        .join(filename)
}

/// Build the `resolve` URL used to download a file's bytes.
fn resolve_url(repo: &str, filename: &str, revision: &str) -> String {
    format!("{HF_ENDPOINT}/{repo}/resolve/{revision}/{filename}")
}

/// Build the model-info API URL used to list a repository's files.
fn api_model_url(repo: &str) -> String {
    format!("{HF_ENDPOINT}/api/models/{repo}")
}

/// Download a single file from a HuggingFace repository, returning its cached path.
///
/// The crypto provider is installed first. If the file is already cached and
/// non-empty, the cached path is returned without any network access. Otherwise the
/// file is fetched from `{endpoint}/{repo}/resolve/{revision}/{filename}` (reqwest
/// follows redirects by default), streamed to a temporary file, and atomically
/// renamed into the cache.
///
/// `revision` defaults to `"main"` when `None`.
///
/// # Errors
///
/// Returns [`HubError::NotFound`] on HTTP 404, [`HubError::Status`] on any other
/// non-success status, [`HubError::Http`] on transport failures, and
/// [`HubError::Io`] on local filesystem failures.
pub async fn download_file(
    repo: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<PathBuf, HubError> {
    ensure_crypto_provider();

    let revision = revision.unwrap_or(DEFAULT_REVISION);
    let target = cache_path(repo, filename, revision);

    // Fast path: return a non-empty cached file without touching the network.
    if let Ok(meta) = tokio::fs::metadata(&target).await {
        if meta.is_file() && meta.len() > 0 {
            tracing::debug!("hub: using cached file {}", target.display());
            return Ok(target);
        }
    }

    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let url = resolve_url(repo, filename, revision);
    tracing::info!("hub: downloading {url}");
    let response = reqwest::get(&url).await?;

    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(HubError::NotFound {
            repo: repo.to_string(),
            path: filename.to_string(),
        });
    }
    if !status.is_success() {
        return Err(HubError::Status {
            code: status.as_u16(),
        });
    }

    let bytes = response.bytes().await?;

    // Write to a unique temp sibling, then atomically rename into place so a partial
    // download can never be observed as a valid cache entry.
    let tmp = temp_sibling(&target);
    tokio::fs::write(&tmp, &bytes).await?;
    match tokio::fs::rename(&tmp, &target).await {
        Ok(()) => {}
        Err(rename_err) => {
            // Best-effort cleanup of the temp file before surfacing the error.
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(HubError::Io(rename_err));
        }
    }

    tracing::info!("hub: cached {} ({} bytes)", target.display(), bytes.len());
    Ok(target)
}

/// List the files contained in a HuggingFace model repository.
///
/// The crypto provider is installed first. The repository metadata is fetched from
/// `{endpoint}/api/models/{repo}` and the `rfilename` of every entry in `siblings`
/// is collected.
///
/// `revision` is currently informational and does not change the queried endpoint.
///
/// # Errors
///
/// Returns [`HubError::NotFound`] on HTTP 404, [`HubError::Status`] on any other
/// non-success status, and [`HubError::Http`] on transport / JSON-decode failures.
pub async fn list_files(repo: &str, revision: Option<&str>) -> Result<Vec<String>, HubError> {
    ensure_crypto_provider();
    let _ = revision; // currently informational

    let url = api_model_url(repo);
    tracing::debug!("hub: listing files via {url}");
    let response = reqwest::get(&url).await?;

    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(HubError::NotFound {
            repo: repo.to_string(),
            path: String::new(),
        });
    }
    if !status.is_success() {
        return Err(HubError::Status {
            code: status.as_u16(),
        });
    }

    let value: serde_json::Value = response.json().await?;
    let files = value
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|entry| {
                    entry
                        .get("rfilename")
                        .and_then(|f| f.as_str())
                        .map(|s| s.to_string())
                })
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();

    Ok(files)
}

/// Build a unique temporary sibling path for atomic-rename downloads.
fn temp_sibling(target: &std::path::Path) -> PathBuf {
    // Combine PID and a process-lifetime counter so concurrent downloads of the same
    // file (or retries) never collide on the temp name.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let file_name = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("download");
    let tmp_name = format!(".{file_name}.{pid}.{n}.tmp");
    match target.parent() {
        Some(parent) => parent.join(tmp_name),
        None => PathBuf::from(tmp_name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repo_to_dir_component() {
        assert_eq!(
            repo_to_dir_component("facebook/fastspeech2-en-ljspeech"),
            "facebook--fastspeech2-en-ljspeech"
        );
        assert_eq!(repo_to_dir_component("no-slash"), "no-slash");
        assert_eq!(repo_to_dir_component("a/b/c"), "a--b--c");
    }

    #[test]
    fn test_cache_path_layout() {
        let p = cache_path("org/model", "model.safetensors", "main");
        let s = p.to_string_lossy().replace('\\', "/");
        assert!(
            s.contains("voirs/hub/org--model/main/model.safetensors"),
            "got {s}"
        );
    }

    #[test]
    fn test_resolve_url() {
        assert_eq!(
            resolve_url("org/model", "model.safetensors", "main"),
            "https://huggingface.co/org/model/resolve/main/model.safetensors"
        );
        assert_eq!(
            resolve_url("microsoft/speecht5_tts", "config.json", "abc123"),
            "https://huggingface.co/microsoft/speecht5_tts/resolve/abc123/config.json"
        );
    }

    #[test]
    fn test_api_model_url() {
        assert_eq!(
            api_model_url("org/model"),
            "https://huggingface.co/api/models/org/model"
        );
    }

    #[test]
    fn test_temp_sibling_is_unique_and_adjacent() {
        let target = cache_path("org/model", "model.bin", "main");
        let a = temp_sibling(&target);
        let b = temp_sibling(&target);
        assert_ne!(a, b, "temp siblings must be unique");
        assert_eq!(
            a.parent(),
            target.parent(),
            "temp file must sit beside target"
        );
    }

    #[test]
    fn test_ensure_crypto_provider_is_idempotent() {
        // Must not panic when called multiple times (Once-guarded).
        ensure_crypto_provider();
        ensure_crypto_provider();
    }

    #[tokio::test]
    #[ignore = "network: downloads a small file from huggingface.co"]
    async fn network_smoke_download_and_list() {
        ensure_crypto_provider();

        // 1) list endpoint over TLS (validates the alpha crypto handshake + /api/models)
        let files = list_files("bert-base-uncased", None)
            .await
            .expect("list_files should succeed");
        assert!(
            files.iter().any(|f| f == "config.json"),
            "expected config.json in repo file list, got: {files:?}"
        );

        // 2) download a tiny real file (validates resolve URL + download + cache)
        let path = download_file("bert-base-uncased", "config.json", None)
            .await
            .expect("download_file should succeed");
        let meta = std::fs::metadata(&path).expect("downloaded file should exist");
        assert!(meta.len() > 0, "downloaded config.json must be non-empty");
        println!("OK: downloaded {} bytes to {}", meta.len(), path.display());
    }
}
