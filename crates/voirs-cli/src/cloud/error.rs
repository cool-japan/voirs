//! Typed errors for VoiRS cloud storage and cloud API integration.
//!
//! These errors exist so that "no credentials configured" and "provider not
//! implemented" are *distinguishable, matchable* outcomes rather than an
//! opaque `anyhow::anyhow!(...)` string -- and, most importantly, so that
//! nothing in [`crate::cloud`] can silently substitute a fabricated result
//! for a real one. Every fallible cloud operation returns one of these
//! variants (or propagates through `anyhow::Error`, which converts from
//! `CloudStorageError` automatically via `std::error::Error`).

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur while talking to a cloud storage or cloud API
/// backend (AWS S3, Azure Blob Storage, Google Cloud Storage, any
/// S3-compatible service such as MinIO / Cloudflare R2, or the VoiRS cloud
/// API).
#[derive(Debug, Error)]
pub enum CloudStorageError {
    /// The requested provider has no usable credentials configured. This is
    /// a deliberate fail-closed error: VoiRS never fabricates network
    /// activity or pretends success when credentials are missing.
    #[error("cloud storage not configured for provider {provider}: {detail}")]
    NotConfigured {
        /// Human-readable provider name (e.g. `"AWS S3"`, `"Azure Blob Storage"`).
        provider: String,
        /// What is missing and how to fix it.
        detail: String,
    },

    /// The provider/operation combination is recognized but has no real
    /// implementation (as opposed to `NotConfigured`, which means the
    /// implementation exists but lacks credentials).
    #[error("cloud storage operation not supported for provider {provider}: {detail}")]
    Unsupported {
        /// Human-readable provider name.
        provider: String,
        /// Explanation of what is unsupported.
        detail: String,
    },

    /// A cryptographic request-signing step failed.
    #[error("request signing failed: {0}")]
    Signing(String),

    /// The remote HTTP endpoint returned a non-success status code.
    #[error("{provider} request failed: HTTP {status} for {url}: {body}")]
    Http {
        /// Human-readable provider name.
        provider: String,
        /// HTTP status code returned by the server.
        status: u16,
        /// The request URL (without credentials).
        url: String,
        /// Response body (truncated by the caller if very large).
        body: String,
    },

    /// A transport-level error (DNS, TLS, connection refused, timeout, ...).
    #[error("{provider} network request to {url} failed: {source}")]
    Network {
        /// Human-readable provider name.
        provider: String,
        /// The request URL (without credentials).
        url: String,
        /// Underlying `reqwest` error.
        #[source]
        source: reqwest::Error,
    },

    /// Local file-system I/O error.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Path the failed I/O operation targeted.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A response body could not be parsed in the expected format
    /// (e.g. malformed XML from a `ListObjectsV2` response).
    #[error("failed to parse {provider} response from {url}: {detail}")]
    InvalidResponse {
        /// Human-readable provider name.
        provider: String,
        /// The request URL that produced the unparsable response.
        url: String,
        /// Explanation of what failed to parse.
        detail: String,
    },
}
