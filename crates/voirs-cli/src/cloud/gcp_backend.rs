//! Google Cloud Storage REST client using OAuth2 bearer-token
//! authentication.
//!
//! Unlike AWS/Azure, the GCS JSON API does not need request signing: a
//! valid OAuth2 access token (from `GOOGLE_OAUTH_TOKEN`, a service-account
//! flow run out-of-band, or `gcloud auth print-access-token`) is sent
//! verbatim as `Authorization: Bearer <token>`. This client only ever runs
//! with a real, caller-supplied token -- if none is configured, the caller
//! (see `crate::cloud::storage`) must fail closed with
//! [`crate::cloud::error::CloudStorageError::NotConfigured`] instead of
//! constructing this backend at all, per VoiRS's "never fabricate cloud
//! I/O" policy.

use crate::cloud::error::CloudStorageError;
use crate::cloud::sigv4::{canonical_query_string, uri_encode};
use reqwest::{Client, Response};
use serde::Deserialize;

const API_ROOT: &str = "https://storage.googleapis.com";

/// Connection parameters for a Google Cloud Storage bucket.
#[derive(Debug, Clone)]
pub struct GcpConfig {
    /// Target bucket name.
    pub bucket: String,
    /// OAuth2 bearer access token. Required and non-empty: callers must
    /// not construct a [`GcpBackend`] without a real token.
    pub token: String,
}

/// One entry returned by [`GcpBackend::list_objects`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcsObjectSummary {
    /// Full object name.
    pub name: String,
    /// Object size in bytes, as reported by the server.
    pub size_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct GcsListResponse {
    #[serde(default)]
    items: Vec<GcsObjectJson>,
}

#[derive(Debug, Deserialize)]
struct GcsObjectJson {
    name: String,
    #[serde(default)]
    size: Option<String>,
}

/// A real Google Cloud Storage client. Every method issues an actual
/// bearer-authenticated HTTP request against the GCS JSON API and
/// propagates real transport/HTTP errors.
pub struct GcpBackend {
    client: Client,
    config: GcpConfig,
}

async fn ensure_success(url: &str, response: Response) -> Result<Response, CloudStorageError> {
    if response.status().is_success() {
        Ok(response)
    } else {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        Err(CloudStorageError::Http {
            provider: "Google Cloud Storage".to_string(),
            status,
            url: url.to_string(),
            body,
        })
    }
}

impl GcpBackend {
    /// Create a new backend. `config.token` must be a real, non-empty
    /// bearer token; the caller is responsible for that invariant (see
    /// [`crate::cloud::storage`]).
    #[must_use]
    pub fn new(client: Client, config: GcpConfig) -> Self {
        Self { client, config }
    }

    fn auth_value(&self) -> String {
        format!("Bearer {}", self.config.token)
    }

    /// Upload `body` as object `name`.
    pub async fn put_object(
        &self,
        name: &str,
        body: Vec<u8>,
        content_type: &str,
    ) -> Result<(), CloudStorageError> {
        let query = canonical_query_string(&[("uploadType", "media"), ("name", name)]);
        let url = format!(
            "{API_ROOT}/upload/storage/v1/b/{}/o?{query}",
            uri_encode(&self.config.bucket, false)
        );

        let response = self
            .client
            .post(&url)
            .header("authorization", self.auth_value())
            .header("content-type", content_type)
            .body(body)
            .send()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: "Google Cloud Storage".to_string(),
                url: url.clone(),
                source,
            })?;

        ensure_success(&url, response).await.map(|_| ())
    }

    /// Download the full contents of object `name`.
    pub async fn get_object(&self, name: &str) -> Result<Vec<u8>, CloudStorageError> {
        let query = canonical_query_string(&[("alt", "media")]);
        let url = format!(
            "{API_ROOT}/storage/v1/b/{}/o/{}?{query}",
            uri_encode(&self.config.bucket, false),
            uri_encode(name, true)
        );

        let response = self
            .client
            .get(&url)
            .header("authorization", self.auth_value())
            .send()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: "Google Cloud Storage".to_string(),
                url: url.clone(),
                source,
            })?;

        let response = ensure_success(&url, response).await?;
        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|source| CloudStorageError::Network {
                provider: "Google Cloud Storage".to_string(),
                url,
                source,
            })
    }

    /// Delete object `name`.
    pub async fn delete_object(&self, name: &str) -> Result<(), CloudStorageError> {
        let url = format!(
            "{API_ROOT}/storage/v1/b/{}/o/{}",
            uri_encode(&self.config.bucket, false),
            uri_encode(name, true)
        );

        let response = self
            .client
            .delete(&url)
            .header("authorization", self.auth_value())
            .send()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: "Google Cloud Storage".to_string(),
                url: url.clone(),
                source,
            })?;

        ensure_success(&url, response).await.map(|_| ())
    }

    /// List objects under `prefix` (single page).
    pub async fn list_objects(
        &self,
        prefix: &str,
    ) -> Result<Vec<GcsObjectSummary>, CloudStorageError> {
        let query = if prefix.is_empty() {
            String::new()
        } else {
            format!("?{}", canonical_query_string(&[("prefix", prefix)]))
        };
        let url = format!(
            "{API_ROOT}/storage/v1/b/{}/o{query}",
            uri_encode(&self.config.bucket, false)
        );

        let response = self
            .client
            .get(&url)
            .header("authorization", self.auth_value())
            .send()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: "Google Cloud Storage".to_string(),
                url: url.clone(),
                source,
            })?;

        let response = ensure_success(&url, response).await?;
        let body = response
            .text()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: "Google Cloud Storage".to_string(),
                url: url.clone(),
                source,
            })?;
        let parsed: GcsListResponse =
            serde_json::from_str(&body).map_err(|e| CloudStorageError::InvalidResponse {
                provider: "Google Cloud Storage".to_string(),
                url: url.clone(),
                detail: format!("malformed JSON list response: {e}"),
            })?;

        Ok(parsed
            .items
            .into_iter()
            .map(|item| GcsObjectSummary {
                name: item.name,
                size_bytes: item.size.and_then(|s| s.parse().ok()).unwrap_or(0),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `reqwest::Client::new()` panics unless a rustls `CryptoProvider` has
    /// been installed process-wide first (this workspace builds reqwest
    /// with `rustls-no-provider`). Route every test `Client` through this
    /// helper instead of calling `Client::new()` directly.
    fn test_client() -> Client {
        voirs_acoustic::hub::ensure_crypto_provider();
        Client::new()
    }

    fn backend() -> GcpBackend {
        GcpBackend::new(
            test_client(),
            GcpConfig {
                bucket: "voirs-cloud".to_string(),
                token: "ya29.test-token".to_string(),
            },
        )
    }

    #[test]
    fn auth_value_uses_real_token() {
        let b = backend();
        assert_eq!(b.auth_value(), "Bearer ya29.test-token");
    }

    #[test]
    fn auth_value_changes_with_token() {
        let b1 = backend();
        let b2 = GcpBackend::new(
            test_client(),
            GcpConfig {
                bucket: "voirs-cloud".to_string(),
                token: "different-token".to_string(),
            },
        );
        assert_ne!(b1.auth_value(), b2.auth_value());
    }

    #[test]
    fn list_response_parses_size_as_string_field() {
        let json = r#"{"items":[{"name":"models/a.bin","size":"1024"},{"name":"models/b.bin","size":"2048"}]}"#;
        let parsed: GcsListResponse = serde_json::from_str(json).unwrap();
        let summaries: Vec<GcsObjectSummary> = parsed
            .items
            .into_iter()
            .map(|item| GcsObjectSummary {
                name: item.name,
                size_bytes: item.size.and_then(|s| s.parse().ok()).unwrap_or(0),
            })
            .collect();
        assert_eq!(
            summaries,
            vec![
                GcsObjectSummary {
                    name: "models/a.bin".to_string(),
                    size_bytes: 1024,
                },
                GcsObjectSummary {
                    name: "models/b.bin".to_string(),
                    size_bytes: 2048,
                },
            ]
        );
    }

    #[test]
    fn list_response_handles_empty_bucket() {
        let json = r#"{}"#;
        let parsed: GcsListResponse = serde_json::from_str(json).unwrap();
        assert!(parsed.items.is_empty());
    }
}
