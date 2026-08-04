//! Azure Blob Storage REST client using the "Shared Key" authorization
//! scheme.
//!
//! Implements the exact `StringToSign` / `CanonicalizedHeaders` /
//! `CanonicalizedResource` algorithm published by Microsoft for the Blob
//! service (version 2009-09-19 and later):
//! <https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key>,
//! directly over `reqwest`, using only `hmac` + `sha2` + `base64`. No Azure
//! SDK, no FFI required.

use crate::cloud::error::CloudStorageError;
use crate::cloud::sigv4::{collapse_whitespace, uri_encode};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use hmac::{Hmac, KeyInit, Mac};
use reqwest::{Client, Response, Url};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Azure Storage REST API version this client speaks.
const API_VERSION: &str = "2021-08-06";

/// Credentials and connection parameters for an Azure Blob Storage
/// container.
#[derive(Debug, Clone)]
pub struct AzureConfig {
    /// Storage account name.
    pub account: String,
    /// Base64-encoded storage account (shared) key.
    pub account_key: String,
    /// Target container name.
    pub container: String,
}

/// A real Azure Blob Storage client, authenticated with Shared Key. Every
/// method issues an actual signed HTTP request against
/// `https://{account}.blob.core.windows.net` and propagates real
/// transport/HTTP errors.
pub struct AzureBackend {
    client: Client,
    config: AzureConfig,
}

async fn ensure_success(
    provider: &str,
    url: &str,
    response: Response,
) -> Result<Response, CloudStorageError> {
    if response.status().is_success() {
        Ok(response)
    } else {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        Err(CloudStorageError::Http {
            provider: provider.to_string(),
            status,
            url: url.to_string(),
            body,
        })
    }
}

/// Current UTC time formatted as an RFC 1123 `Date`/`x-ms-date` header
/// value, e.g. `Thu, 01 Jan 2026 00:00:00 GMT`.
fn http_date_now() -> String {
    chrono::Utc::now()
        .format("%a, %d %b %Y %H:%M:%S GMT")
        .to_string()
}

/// Build the Shared-Key `StringToSign` for the Blob service, per the
/// field order published by Microsoft:
/// `VERB\nContent-Encoding\nContent-Language\nContent-Length\nContent-MD5\n
/// Content-Type\nDate\nIf-Modified-Since\nIf-Match\nIf-None-Match\n
/// If-Unmodified-Since\nRange\nCanonicalizedHeaders + CanonicalizedResource`.
/// `Content-Encoding`, `Content-Language`, `Content-MD5`, `Date`,
/// `If-Modified-Since`, `If-Match`, `If-None-Match`, `If-Unmodified-Since`,
/// and `Range` are always empty in this client (we authenticate with
/// `x-ms-date` and never send conditional headers), so only
/// `content_length_field` (empty string if the body is zero bytes, else
/// the decimal length) and `content_type` vary.
fn build_string_to_sign(
    verb: &str,
    content_length_field: &str,
    content_type: &str,
    canonicalized_headers: &str,
    canonicalized_resource: &str,
) -> String {
    format!(
        "{verb}\n\n\n{content_length_field}\n\n{content_type}\n\n\n\n\n\n\n{canonicalized_headers}{canonicalized_resource}"
    )
}

/// Build `CanonicalizedHeaders`: every `x-ms-*` header, lowercased,
/// whitespace-collapsed, sorted by name, each line `\n`-terminated.
fn canonicalized_headers(headers: &[(&str, &str)]) -> String {
    let mut xms: Vec<(String, String)> = headers
        .iter()
        .filter(|(k, _)| k.to_lowercase().starts_with("x-ms-"))
        .map(|(k, v)| (k.to_lowercase(), collapse_whitespace(v.trim())))
        .collect();
    xms.sort();
    xms.iter().map(|(k, v)| format!("{k}:{v}\n")).collect()
}

impl AzureBackend {
    const PROVIDER: &'static str = "Azure Blob Storage";

    /// Create a new backend using the given HTTP client and config. The
    /// caller is responsible for having installed a rustls
    /// `CryptoProvider` (see `voirs_acoustic::hub::ensure_crypto_provider`)
    /// before building `client`.
    #[must_use]
    pub fn new(client: Client, config: AzureConfig) -> Self {
        Self { client, config }
    }

    fn blob_url(&self, blob: &str) -> Result<Url, CloudStorageError> {
        let url_string = format!(
            "https://{}.blob.core.windows.net/{}/{}",
            self.config.account,
            uri_encode(&self.config.container, false),
            uri_encode(blob, false)
        );
        Url::parse(&url_string).map_err(|e| {
            CloudStorageError::Signing(format!("invalid Azure blob URL '{url_string}': {e}"))
        })
    }

    fn canonicalized_resource(&self, url: &Url) -> String {
        format!("/{}{}", self.config.account, url.path())
    }

    fn authorization_header(&self, string_to_sign: &str) -> Result<String, CloudStorageError> {
        let key_bytes = BASE64.decode(&self.config.account_key).map_err(|e| {
            CloudStorageError::Signing(format!("Azure account key is not valid base64: {e}"))
        })?;
        let mut mac = HmacSha256::new_from_slice(&key_bytes)
            .map_err(|e| CloudStorageError::Signing(format!("HMAC-SHA256 key error: {e}")))?;
        mac.update(string_to_sign.as_bytes());
        let signature = BASE64.encode(mac.finalize().into_bytes());
        Ok(format!("SharedKey {}:{}", self.config.account, signature))
    }

    /// Upload `body` as a block blob named `blob`.
    pub async fn put_blob(
        &self,
        blob: &str,
        body: Vec<u8>,
        content_type: &str,
    ) -> Result<(), CloudStorageError> {
        let url = self.blob_url(blob)?;
        let date = http_date_now();
        let content_length_field = if body.is_empty() {
            String::new()
        } else {
            body.len().to_string()
        };

        let headers = [
            ("x-ms-blob-type", "BlockBlob"),
            ("x-ms-date", date.as_str()),
            ("x-ms-version", API_VERSION),
        ];
        let string_to_sign = build_string_to_sign(
            "PUT",
            &content_length_field,
            content_type,
            &canonicalized_headers(&headers),
            &self.canonicalized_resource(&url),
        );
        let authorization = self.authorization_header(&string_to_sign)?;

        let response = self
            .client
            .put(url.clone())
            .header("x-ms-blob-type", "BlockBlob")
            .header("x-ms-date", &date)
            .header("x-ms-version", API_VERSION)
            .header("content-type", content_type)
            .header("authorization", authorization)
            .body(body)
            .send()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: Self::PROVIDER.to_string(),
                url: url.to_string(),
                source,
            })?;

        ensure_success(Self::PROVIDER, url.as_str(), response)
            .await
            .map(|_| ())
    }

    /// Download the full contents of `blob`.
    pub async fn get_blob(&self, blob: &str) -> Result<Vec<u8>, CloudStorageError> {
        let url = self.blob_url(blob)?;
        let date = http_date_now();

        let headers = [("x-ms-date", date.as_str()), ("x-ms-version", API_VERSION)];
        let string_to_sign = build_string_to_sign(
            "GET",
            "",
            "",
            &canonicalized_headers(&headers),
            &self.canonicalized_resource(&url),
        );
        let authorization = self.authorization_header(&string_to_sign)?;

        let response = self
            .client
            .get(url.clone())
            .header("x-ms-date", &date)
            .header("x-ms-version", API_VERSION)
            .header("authorization", authorization)
            .send()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: Self::PROVIDER.to_string(),
                url: url.to_string(),
                source,
            })?;

        let response = ensure_success(Self::PROVIDER, url.as_str(), response).await?;
        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|source| CloudStorageError::Network {
                provider: Self::PROVIDER.to_string(),
                url: url.to_string(),
                source,
            })
    }

    /// Delete `blob`.
    pub async fn delete_blob(&self, blob: &str) -> Result<(), CloudStorageError> {
        let url = self.blob_url(blob)?;
        let date = http_date_now();

        let headers = [("x-ms-date", date.as_str()), ("x-ms-version", API_VERSION)];
        let string_to_sign = build_string_to_sign(
            "DELETE",
            "",
            "",
            &canonicalized_headers(&headers),
            &self.canonicalized_resource(&url),
        );
        let authorization = self.authorization_header(&string_to_sign)?;

        let response = self
            .client
            .delete(url.clone())
            .header("x-ms-date", &date)
            .header("x-ms-version", API_VERSION)
            .header("authorization", authorization)
            .send()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: Self::PROVIDER.to_string(),
                url: url.to_string(),
                source,
            })?;

        ensure_success(Self::PROVIDER, url.as_str(), response)
            .await
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values cross-checked offline against an independent Python
    // implementation (`hashlib`/`hmac`/`base64`, stdlib) of the exact
    // Shared Key algorithm published by Microsoft at
    // https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key,
    // including its own worked "Get Blob" `CanonicalizedHeaders`/
    // `CanonicalizedResource` example, which this module's structure
    // matches field-for-field.

    /// `reqwest::Client::new()` panics unless a rustls `CryptoProvider` has
    /// been installed process-wide first (this workspace builds reqwest
    /// with `rustls-no-provider`). Route every test `Client` through this
    /// helper instead of calling `Client::new()` directly.
    fn test_client() -> Client {
        voirs_acoustic::hub::ensure_crypto_provider();
        Client::new()
    }

    fn test_backend() -> AzureBackend {
        let config = AzureConfig {
            account: "voirstestaccount".to_string(),
            account_key:
                "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWYwMTIzNDU2Nzg5YWJjZGVmMDEyMzQ1Njc="
                    .to_string(),
            container: "testcontainer".to_string(),
        };
        AzureBackend::new(test_client(), config)
    }

    #[test]
    fn canonicalized_headers_only_includes_x_ms_and_sorts() {
        let headers = [
            ("Content-Type", "application/octet-stream"),
            ("x-ms-version", "2021-08-06"),
            ("x-ms-blob-type", "BlockBlob"),
        ];
        assert_eq!(
            canonicalized_headers(&headers),
            "x-ms-blob-type:BlockBlob\nx-ms-version:2021-08-06\n"
        );
    }

    #[test]
    fn put_blob_string_to_sign_matches_reference() {
        let backend = test_backend();
        let url = backend.blob_url("voirs/model.safetensors").unwrap();
        assert_eq!(url.path(), "/testcontainer/voirs/model.safetensors");

        let date = "Thu, 01 Jan 2026 00:00:00 GMT";
        let headers = [
            ("x-ms-blob-type", "BlockBlob"),
            ("x-ms-date", date),
            ("x-ms-version", API_VERSION),
        ];
        let content_length = "24"; // len(b"hello azure blob storage")
        let string_to_sign = build_string_to_sign(
            "PUT",
            content_length,
            "application/octet-stream",
            &canonicalized_headers(&headers),
            &backend.canonicalized_resource(&url),
        );

        assert_eq!(
            string_to_sign,
            "PUT\n\n\n24\n\napplication/octet-stream\n\n\n\n\n\n\nx-ms-blob-type:BlockBlob\nx-ms-date:Thu, 01 Jan 2026 00:00:00 GMT\nx-ms-version:2021-08-06\n/voirstestaccount/testcontainer/voirs/model.safetensors"
        );

        let authorization = backend.authorization_header(&string_to_sign).unwrap();
        assert_eq!(
            authorization,
            "SharedKey voirstestaccount:iNUJRY6JK9erOQA+/9/xd9v2oBCp0nx4/03ctMcdob8="
        );
    }

    #[test]
    fn get_blob_string_to_sign_matches_reference() {
        let backend = test_backend();
        let url = backend.blob_url("voirs/model.safetensors").unwrap();

        let date = "Thu, 01 Jan 2026 00:00:00 GMT";
        let headers = [("x-ms-date", date), ("x-ms-version", API_VERSION)];
        let string_to_sign = build_string_to_sign(
            "GET",
            "",
            "",
            &canonicalized_headers(&headers),
            &backend.canonicalized_resource(&url),
        );

        assert_eq!(
            string_to_sign,
            "GET\n\n\n\n\n\n\n\n\n\n\n\nx-ms-date:Thu, 01 Jan 2026 00:00:00 GMT\nx-ms-version:2021-08-06\n/voirstestaccount/testcontainer/voirs/model.safetensors"
        );

        let authorization = backend.authorization_header(&string_to_sign).unwrap();
        assert_eq!(
            authorization,
            "SharedKey voirstestaccount:1apGhIOSl9dZD8cbaWZkaIu5m2ej5sPGq1NhI1Smwsw="
        );
    }

    #[test]
    fn authorization_changes_when_account_key_changes() {
        let mut config_a = AzureConfig {
            account: "acct".to_string(),
            account_key: BASE64.encode(b"key-one-key-one-key-one-key-one"),
            container: "c".to_string(),
        };
        let backend_a = AzureBackend::new(test_client(), config_a.clone());
        config_a.account_key = BASE64.encode(b"key-two-key-two-key-two-key-two");
        let backend_b = AzureBackend::new(test_client(), config_a);

        let sts = "GET\n\n\n\n\n\n\n\n\n\n\n\n/acct/c/blob";
        let sig_a = backend_a.authorization_header(sts).unwrap();
        let sig_b = backend_b.authorization_header(sts).unwrap();
        assert_ne!(sig_a, sig_b);
    }

    #[test]
    fn rejects_non_base64_account_key() {
        let config = AzureConfig {
            account: "acct".to_string(),
            account_key: "not valid base64 !!!".to_string(),
            container: "c".to_string(),
        };
        let backend = AzureBackend::new(test_client(), config);
        let result = backend.authorization_header("GET\n\n\n\n\n\n\n\n\n\n\n\n/acct/c/blob");
        assert!(matches!(result, Err(CloudStorageError::Signing(_))));
    }
}
