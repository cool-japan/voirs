//! Minimal, Pure-Rust AWS S3 client.
//!
//! Signs every request with hand-rolled SigV4 (see [`super::sigv4`], built on
//! the RustCrypto `hmac`/`sha2` crates) and sends it with `reqwest`. No AWS
//! SDK, no OpenSSL, no vendored signing library.

use super::sigv4;
use super::{CloudProvider, ObjectMetadata, RetryConfig};
use crate::{DatasetError, Result};
use chrono::{DateTime, Utc};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue},
    Client, Method, StatusCode,
};
use std::collections::{BTreeMap, HashMap};

/// Resolved AWS credentials used to sign and route requests.
#[derive(Clone)]
pub(super) struct AwsCredentials {
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl std::fmt::Debug for AwsCredentials {
    /// Redacts the secret key and session token so this can never leak a
    /// real credential through a stray `{:?}`/`dbg!`/log statement, while
    /// still satisfying `Result::unwrap_err`'s `Debug` bound in tests.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AwsCredentials")
            .field("region", &self.region)
            .field("access_key_id", &self.access_key_id)
            .field("secret_access_key", &"[REDACTED]")
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

/// Read a config field, falling back to the given environment variable when
/// the config value is empty. Returns `None` if neither is set (or the env
/// var is set but empty).
fn resolve_field(config_value: &str, env_var: &str) -> Option<String> {
    if !config_value.is_empty() {
        return Some(config_value.to_string());
    }
    std::env::var(env_var).ok().filter(|v| !v.is_empty())
}

/// Resolve real AWS credentials from a [`CloudProvider`], falling back to
/// the standard `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
/// `AWS_SESSION_TOKEN` / `AWS_REGION` (or `AWS_DEFAULT_REGION`) environment
/// variables wherever the config leaves a field empty.
///
/// GCP and Azure are honestly reported as not implemented: GCS requires an
/// OAuth2 service-account JWT (RS256) exchange and Azure requires its own
/// Shared Key signing scheme, neither of which this client performs. Rather
/// than silently downgrading those providers to fabricated success, callers
/// get a clear, typed error telling them to configure AWS instead.
pub(super) fn resolve_aws_credentials(provider: &CloudProvider) -> Result<AwsCredentials> {
    match provider {
        CloudProvider::AWS {
            region,
            access_key_id,
            secret_access_key,
            session_token,
        } => {
            let access_key_id =
                resolve_field(access_key_id, "AWS_ACCESS_KEY_ID").ok_or_else(|| {
                    DatasetError::Configuration(
                    "AWS access key ID not configured: set CloudProvider::AWS.access_key_id or \
                     the AWS_ACCESS_KEY_ID environment variable"
                        .to_string(),
                )
                })?;
            let secret_access_key = resolve_field(secret_access_key, "AWS_SECRET_ACCESS_KEY")
                .ok_or_else(|| {
                    DatasetError::Configuration(
                        "AWS secret access key not configured: set \
                         CloudProvider::AWS.secret_access_key or the AWS_SECRET_ACCESS_KEY \
                         environment variable"
                            .to_string(),
                    )
                })?;
            let region = resolve_field(region, "AWS_REGION")
                .or_else(|| resolve_field("", "AWS_DEFAULT_REGION"))
                .ok_or_else(|| {
                    DatasetError::Configuration(
                        "AWS region not configured: set CloudProvider::AWS.region or the \
                         AWS_REGION/AWS_DEFAULT_REGION environment variable"
                            .to_string(),
                    )
                })?;
            let session_token = session_token
                .clone()
                .or_else(|| resolve_field("", "AWS_SESSION_TOKEN"));

            Ok(AwsCredentials {
                region,
                access_key_id,
                secret_access_key,
                session_token,
            })
        }
        CloudProvider::GCP { .. } | CloudProvider::Azure { .. } => Err(DatasetError::CloudStorage(
            unsupported_provider_reason(provider),
        )),
    }
}

/// The exact reason a non-AWS provider is not implemented, shared between
/// [`resolve_aws_credentials`]'s `Err` path and
/// [`super::CloudStorageImpl::new`] (which needs the same text without
/// going through a `Result` it would have to unwrap). Only meaningful for
/// [`CloudProvider::GCP`]/[`CloudProvider::Azure`]; returns a generic
/// message for `AWS` (never actually reachable in practice, since callers
/// only invoke this for the non-AWS arms).
pub(super) fn unsupported_provider_reason(provider: &CloudProvider) -> String {
    match provider {
        CloudProvider::GCP { .. } => {
            "Google Cloud Storage is not implemented in this build: GCS requires an OAuth2 \
             service-account JWT (RS256) token exchange, which is not performed here. \
             Configure CloudProvider::AWS to use real cloud storage."
                .to_string()
        }
        CloudProvider::Azure { .. } => {
            "Azure Blob Storage is not implemented in this build: Azure Shared Key request \
             signing is not performed here. Configure CloudProvider::AWS to use real cloud \
             storage."
                .to_string()
        }
        CloudProvider::AWS { .. } => {
            "internal error: unsupported_provider_reason called for CloudProvider::AWS".to_string()
        }
    }
}

/// Exponential backoff delay (milliseconds) for retry `attempt` (0-based),
/// clamped to `retry.max_delay_ms`. Shared by [`S3Client`]'s own retry loop
/// and [`super::CloudStorageImpl::calculate_backoff_delay`] so both use one
/// formula.
pub(super) fn backoff_delay_ms(retry: &RetryConfig, attempt: usize) -> u64 {
    let delay = retry.base_delay_ms as f64 * retry.backoff_multiplier.powi(attempt as i32);
    delay.min(retry.max_delay_ms as f64) as u64
}

/// Minimal S3 REST client bound to one bucket and one set of credentials.
///
/// Builds the standard virtual-hosted-style endpoint
/// (`{bucket}.s3.{region}.amazonaws.com`) and signs every request with
/// SigV4. `scheme`/`host` are private so only this module can override them
/// (see the `#[cfg(test)]` constructor below, used to point the client at a
/// local loopback server), without exposing any endpoint-override knob on
/// the public [`super::CloudProvider`] configuration surface.
#[derive(Clone)]
pub(super) struct S3Client {
    http: Client,
    scheme: &'static str,
    host: String,
    creds: AwsCredentials,
    max_attempts: usize,
    retry: RetryConfig,
}

impl S3Client {
    pub(super) fn new(
        http: Client,
        bucket: String,
        creds: AwsCredentials,
        retry: &RetryConfig,
    ) -> Self {
        let host = format!("{bucket}.s3.{region}.amazonaws.com", region = creds.region);
        Self {
            http,
            scheme: "https",
            host,
            creds,
            max_attempts: retry.max_attempts.max(1),
            retry: retry.clone(),
        }
    }

    /// Test-only constructor: points the client at a local loopback address
    /// over plain HTTP instead of the real AWS endpoint, so tests can verify
    /// real request signing and byte movement against a server the test
    /// itself controls, without ever touching the network or real AWS.
    #[cfg(test)]
    pub(super) fn for_test(
        http: Client,
        addr: std::net::SocketAddr,
        creds: AwsCredentials,
        retry: &RetryConfig,
    ) -> Self {
        Self {
            http,
            scheme: "http",
            host: addr.to_string(),
            creds,
            max_attempts: retry.max_attempts.max(1),
            retry: retry.clone(),
        }
    }

    /// Public HTTPS URL for `key` under this client's endpoint (used as the
    /// return value for upload operations, and as the base for presigned
    /// URLs).
    pub(super) fn object_url(&self, key: &str) -> String {
        format!(
            "{}://{}/{}",
            self.scheme,
            self.host,
            sigv4::uri_encode(key, false)
        )
    }

    fn backoff_delay(&self, attempt: usize) -> u64 {
        backoff_delay_ms(&self.retry, attempt)
    }

    /// Sign and send a single request attempt (no retry). `key` is the
    /// object key with no leading slash (empty string means the bucket
    /// root, used by `list_objects`).
    async fn send_once(
        &self,
        method: Method,
        key: &str,
        query: &[(String, String)],
        extra_headers: &[(&str, String)],
        body: Vec<u8>,
    ) -> Result<reqwest::Response> {
        let now: DateTime<Utc> = Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = now.format("%Y%m%d").to_string();
        let payload_hash = sigv4::sha256_hex(&body);

        let canonical_uri = format!("/{}", sigv4::uri_encode(key, false));
        let canonical_qs = sigv4::canonical_query_string(query);

        let mut headers: BTreeMap<String, String> = BTreeMap::new();
        headers.insert("host".to_string(), self.host.clone());
        headers.insert("x-amz-date".to_string(), amz_date.clone());
        headers.insert("x-amz-content-sha256".to_string(), payload_hash.clone());
        if let Some(token) = &self.creds.session_token {
            headers.insert("x-amz-security-token".to_string(), token.clone());
        }
        for (name, value) in extra_headers {
            headers.insert((*name).to_string(), value.clone());
        }

        let (canonical_headers_str, signed_headers) = sigv4::canonical_headers(&headers);
        let canonical_request = sigv4::canonical_request(
            method.as_str(),
            &canonical_uri,
            &canonical_qs,
            &canonical_headers_str,
            &signed_headers,
            &payload_hash,
        );
        let credential_scope = format!("{date_stamp}/{}/s3/aws4_request", self.creds.region);
        let sts = sigv4::string_to_sign(&amz_date, &credential_scope, &canonical_request);
        let sig = sigv4::signature(
            &self.creds.secret_access_key,
            &date_stamp,
            &self.creds.region,
            "s3",
            &sts,
        )?;

        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, SignedHeaders={signed_headers}, Signature={sig}",
            self.creds.access_key_id
        );

        let mut url_str = format!("{}://{}{}", self.scheme, self.host, canonical_uri);
        if !canonical_qs.is_empty() {
            // `canonical_qs` is already fully percent-encoded and sorted; it
            // is sent on the wire exactly as signed so the server derives
            // the identical canonical query string when verifying.
            url_str.push('?');
            url_str.push_str(&canonical_qs);
        }
        let url = reqwest::Url::parse(&url_str)
            .map_err(|e| DatasetError::CloudStorage(format!("invalid S3 request URL: {e}")))?;

        let mut header_map = HeaderMap::new();
        for (name, value) in &headers {
            if name == "host" {
                // reqwest/hyper set the real `Host` header from the URL
                // authority automatically; it is still included above in
                // the *signed* header set because that is what the server
                // receives and verifies against.
                continue;
            }
            let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                DatasetError::CloudStorage(format!("invalid header name '{name}': {e}"))
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|e| {
                DatasetError::CloudStorage(format!("invalid header value for '{name}': {e}"))
            })?;
            header_map.insert(header_name, header_value);
        }
        header_map.insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_str(&authorization).map_err(|e| {
                DatasetError::CloudStorage(format!("invalid Authorization header: {e}"))
            })?,
        );

        self.http
            .request(method, url)
            .headers(header_map)
            .body(body)
            .send()
            .await
            .map_err(|e| DatasetError::CloudStorage(format!("S3 request failed: {e}")))
    }

    /// Send a request, retrying transient failures (network errors, 5xx,
    /// 429) with exponential backoff. 4xx responses are never retried --
    /// retrying a permanent client error just delays an inevitable failure.
    async fn send_with_retry(
        &self,
        method: Method,
        key: &str,
        query: &[(String, String)],
        extra_headers: &[(&str, String)],
        body: Vec<u8>,
    ) -> Result<reqwest::Response> {
        let mut last_err: Option<DatasetError> = None;
        for attempt in 0..self.max_attempts {
            let is_last = attempt + 1 == self.max_attempts;
            match self
                .send_once(method.clone(), key, query, extra_headers, body.clone())
                .await
            {
                Ok(response) => {
                    let status = response.status();
                    let retryable =
                        status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS;
                    if !retryable || is_last {
                        return Ok(response);
                    }
                    last_err = Some(DatasetError::CloudStorage(format!(
                        "S3 {method} {key} returned retryable status {status}"
                    )));
                }
                Err(e) => {
                    last_err = Some(e);
                    if is_last {
                        break;
                    }
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(
                self.backoff_delay(attempt),
            ))
            .await;
        }
        Err(last_err.unwrap_or_else(|| {
            DatasetError::CloudStorage("S3 request failed after retries".to_string())
        }))
    }

    pub(super) async fn put_object(
        &self,
        key: &str,
        body: Vec<u8>,
        content_type: &str,
    ) -> Result<()> {
        let response = self
            .send_with_retry(
                Method::PUT,
                key,
                &[],
                &[("content-type", content_type.to_string())],
                body,
            )
            .await?;
        let status = response.status();
        if status.is_success() {
            Ok(())
        } else {
            let text = response.text().await.unwrap_or_default();
            Err(DatasetError::CloudStorage(format!(
                "S3 PUT {key} failed: HTTP {status}: {text}"
            )))
        }
    }

    /// Send the GET and return the raw response (for streaming callers, see
    /// [`super::CloudStorageImpl::stream_dataset`]); callers that want the
    /// fully-buffered body should use [`Self::get_object`]. `pub(super)`
    /// rather than private so `mod.rs` can turn the response into an
    /// `AsyncRead` via `reqwest::Response::bytes_stream` without this module
    /// having to name the `bytes` crate's types directly (it is only a
    /// transitive dependency here, not one this crate declares itself).
    pub(super) async fn get_object_response(&self, key: &str) -> Result<reqwest::Response> {
        let response = self
            .send_with_retry(Method::GET, key, &[], &[], Vec::new())
            .await?;
        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            return Err(DatasetError::CloudStorage(format!(
                "S3 object not found: {key}"
            )));
        }
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(DatasetError::CloudStorage(format!(
                "S3 GET {key} failed: HTTP {status}: {text}"
            )));
        }
        Ok(response)
    }

    pub(super) async fn get_object(&self, key: &str) -> Result<Vec<u8>> {
        let response = self.get_object_response(key).await?;
        response.bytes().await.map(|b| b.to_vec()).map_err(|e| {
            DatasetError::CloudStorage(format!("failed to read S3 response body for {key}: {e}"))
        })
    }

    pub(super) async fn delete_object(&self, key: &str) -> Result<()> {
        let response = self
            .send_with_retry(Method::DELETE, key, &[], &[], Vec::new())
            .await?;
        let status = response.status();
        if status.is_success() || status == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            let text = response.text().await.unwrap_or_default();
            Err(DatasetError::CloudStorage(format!(
                "S3 DELETE {key} failed: HTTP {status}: {text}"
            )))
        }
    }

    /// HEAD `key`. Returns `Ok(None)` for a real "does not exist" (404) and
    /// `Err` for anything else that prevented a definitive answer (network
    /// failure, 403 Forbidden, 5xx) -- an authorization failure must never
    /// be silently reported as "object does not exist".
    pub(super) async fn head_object(&self, key: &str) -> Result<Option<ObjectMetadata>> {
        let response = self
            .send_with_retry(Method::HEAD, key, &[], &[], Vec::new())
            .await?;
        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(DatasetError::CloudStorage(format!(
                "S3 HEAD {key} failed: HTTP {status}"
            )));
        }

        let headers = response.headers();
        let size = headers
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let content_type = headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        let etag = headers
            .get(reqwest::header::ETAG)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .trim_matches('"')
            .to_string();
        let last_modified = headers
            .get(reqwest::header::LAST_MODIFIED)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| DateTime::parse_from_rfc2822(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(DateTime::<Utc>::UNIX_EPOCH);

        Ok(Some(ObjectMetadata {
            size,
            content_type,
            last_modified,
            etag,
            metadata: HashMap::new(),
        }))
    }

    /// List every key under `prefix` via `ListObjectsV2`, parsing the real
    /// XML response. An empty bucket/prefix legitimately yields `Ok(vec![])`;
    /// a response that isn't recognizable `ListObjectsV2` XML at all is an
    /// error rather than a silent empty list (an empty `Vec` must always
    /// mean "truly no objects", never "we couldn't tell").
    pub(super) async fn list_objects(&self, prefix: &str) -> Result<Vec<String>> {
        let query = vec![
            ("list-type".to_string(), "2".to_string()),
            ("prefix".to_string(), prefix.to_string()),
        ];
        let response = self
            .send_with_retry(Method::GET, "", &query, &[], Vec::new())
            .await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(DatasetError::CloudStorage(format!(
                "S3 ListObjectsV2 failed: HTTP {status}: {text}"
            )));
        }
        let text = response.text().await.map_err(|e| {
            DatasetError::CloudStorage(format!("failed to read ListObjectsV2 response: {e}"))
        })?;
        parse_list_objects_keys(&text)
    }

    /// Build a presigned `GET` URL using SigV4 query-string signing (as
    /// opposed to the header-based signing every other method uses): the
    /// signing parameters travel in the query string instead of an
    /// `Authorization` header, and the payload hash is the literal
    /// `UNSIGNED-PAYLOAD` sentinel because the request is never actually
    /// sent by this process -- only the URL is handed back to the caller.
    pub(super) fn presigned_get_url(&self, key: &str, expiry_seconds: u64) -> Result<String> {
        const MAX_EXPIRY_SECONDS: u64 = 7 * 24 * 60 * 60; // SigV4 hard limit: 7 days.
        if expiry_seconds == 0 || expiry_seconds > MAX_EXPIRY_SECONDS {
            return Err(DatasetError::Configuration(format!(
                "presigned URL expiry must be between 1 and {MAX_EXPIRY_SECONDS} seconds \
                 (7 days), got {expiry_seconds}"
            )));
        }

        let now = Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = now.format("%Y%m%d").to_string();
        let credential_scope = format!("{date_stamp}/{}/s3/aws4_request", self.creds.region);
        let credential = format!("{}/{credential_scope}", self.creds.access_key_id);

        let mut query: Vec<(String, String)> = vec![
            (
                "X-Amz-Algorithm".to_string(),
                "AWS4-HMAC-SHA256".to_string(),
            ),
            ("X-Amz-Credential".to_string(), credential),
            ("X-Amz-Date".to_string(), amz_date.clone()),
            ("X-Amz-Expires".to_string(), expiry_seconds.to_string()),
            ("X-Amz-SignedHeaders".to_string(), "host".to_string()),
        ];
        if let Some(token) = &self.creds.session_token {
            query.push(("X-Amz-Security-Token".to_string(), token.clone()));
        }

        let canonical_uri = format!("/{}", sigv4::uri_encode(key, false));
        let canonical_qs = sigv4::canonical_query_string(&query);

        let mut headers = BTreeMap::new();
        headers.insert("host".to_string(), self.host.clone());
        let (canonical_headers_str, signed_headers) = sigv4::canonical_headers(&headers);

        let canonical_request = sigv4::canonical_request(
            "GET",
            &canonical_uri,
            &canonical_qs,
            &canonical_headers_str,
            &signed_headers,
            "UNSIGNED-PAYLOAD",
        );
        let sts = sigv4::string_to_sign(&amz_date, &credential_scope, &canonical_request);
        let sig = sigv4::signature(
            &self.creds.secret_access_key,
            &date_stamp,
            &self.creds.region,
            "s3",
            &sts,
        )?;

        Ok(format!(
            "{}://{}{}?{}&X-Amz-Signature={}",
            self.scheme, self.host, canonical_uri, canonical_qs, sig
        ))
    }
}

/// Extract every `<Key>...</Key>` element's decoded text content from a
/// `ListObjectsV2` XML response, in document order.
///
/// This is a narrow tag-scanning parser rather than a full XML parser (no
/// XML crate is in the workspace), but it still fails closed: a response
/// that doesn't even contain a `<ListBucketResult>` root, or a `<Key>` that
/// is never closed, is reported as an error rather than silently degrading
/// to an empty list, since an empty `Vec<String>` must only ever mean
/// "genuinely no matching objects".
fn parse_list_objects_keys(xml: &str) -> Result<Vec<String>> {
    if !xml.contains("<ListBucketResult") {
        let preview: String = xml.chars().take(200).collect();
        return Err(DatasetError::CloudStorage(format!(
            "unexpected ListObjectsV2 response (missing <ListBucketResult>): {preview}"
        )));
    }

    let mut keys = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<Key>") {
        let after_open = &rest[start + "<Key>".len()..];
        let Some(end) = after_open.find("</Key>") else {
            return Err(DatasetError::CloudStorage(
                "malformed ListObjectsV2 response: unterminated <Key> element".to_string(),
            ));
        };
        keys.push(decode_xml_text(&after_open[..end]));
        rest = &after_open[end + "</Key>".len()..];
    }
    Ok(keys)
}

/// Decode the small set of XML entities S3 actually emits inside `<Key>`
/// text nodes. `&amp;` is decoded last so it cannot re-introduce a
/// two-character sequence that looks like another entity.
fn decode_xml_text(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_aws_credentials_uses_config_values() {
        let provider = CloudProvider::AWS {
            region: "eu-west-1".to_string(),
            access_key_id: "AKIAEXAMPLE".to_string(),
            secret_access_key: "secret".to_string(),
            session_token: Some("token".to_string()),
        };
        let creds = resolve_aws_credentials(&provider).unwrap();
        assert_eq!(creds.region, "eu-west-1");
        assert_eq!(creds.access_key_id, "AKIAEXAMPLE");
        assert_eq!(creds.secret_access_key, "secret");
        assert_eq!(creds.session_token.as_deref(), Some("token"));
    }

    #[test]
    fn resolve_aws_credentials_rejects_empty_config_without_env() {
        // Deliberately does not touch process environment variables (tests
        // run as separate processes under nextest, but sharing this
        // property with `cargo test` too keeps the test honest either way):
        // an empty config with no env fallback must fail closed.
        let provider = CloudProvider::AWS {
            region: String::new(),
            access_key_id: String::new(),
            secret_access_key: String::new(),
            session_token: None,
        };
        // Only assert failure if the ambient environment truly has nothing
        // configured, so this test cannot flake on a machine that happens
        // to export real AWS credentials.
        if std::env::var("AWS_ACCESS_KEY_ID").is_err() {
            assert!(resolve_aws_credentials(&provider).is_err());
        }
    }

    #[test]
    fn resolve_aws_credentials_falls_back_to_environment_variables() {
        let provider = CloudProvider::AWS {
            region: String::new(),
            access_key_id: String::new(),
            secret_access_key: String::new(),
            session_token: None,
        };
        // nextest gives each test its own process, so mutating process-wide
        // environment variables here cannot leak into other tests.
        std::env::set_var("AWS_ACCESS_KEY_ID", "env-access-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "env-secret-key");
        std::env::set_var("AWS_REGION", "ap-northeast-1");

        let creds = resolve_aws_credentials(&provider).unwrap();
        assert_eq!(creds.access_key_id, "env-access-key");
        assert_eq!(creds.secret_access_key, "env-secret-key");
        assert_eq!(creds.region, "ap-northeast-1");

        std::env::remove_var("AWS_ACCESS_KEY_ID");
        std::env::remove_var("AWS_SECRET_ACCESS_KEY");
        std::env::remove_var("AWS_REGION");
    }

    #[test]
    fn resolve_aws_credentials_fails_closed_for_gcp_and_azure() {
        let gcp = CloudProvider::GCP {
            project_id: "proj".to_string(),
            service_account_key: "{}".to_string(),
            location: None,
        };
        let err = resolve_aws_credentials(&gcp).unwrap_err().to_string();
        assert!(err.contains("not implemented"), "got: {err}");

        let azure = CloudProvider::Azure {
            account_name: "acct".to_string(),
            account_key: "key".to_string(),
            container: None,
        };
        let err = resolve_aws_credentials(&azure).unwrap_err().to_string();
        assert!(err.contains("not implemented"), "got: {err}");
    }

    #[test]
    fn parse_list_objects_keys_extracts_all_keys_in_order() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>bucket</Name>
  <Contents><Key>datasets/a/manifest.json</Key><Size>10</Size></Contents>
  <Contents><Key>datasets/a &amp; b/manifest.json</Key><Size>20</Size></Contents>
</ListBucketResult>"#;
        let keys = parse_list_objects_keys(xml).unwrap();
        assert_eq!(
            keys,
            vec![
                "datasets/a/manifest.json".to_string(),
                "datasets/a & b/manifest.json".to_string(),
            ]
        );
    }

    #[test]
    fn parse_list_objects_keys_empty_result_is_ok_empty_vec() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>bucket</Name>
  <KeyCount>0</KeyCount>
</ListBucketResult>"#;
        assert_eq!(parse_list_objects_keys(xml).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn parse_list_objects_keys_rejects_unrecognizable_response() {
        let err = parse_list_objects_keys("<html>Access Denied</html>").unwrap_err();
        assert!(err.to_string().contains("unexpected"));
    }

    #[test]
    fn parse_list_objects_keys_rejects_unterminated_key() {
        let xml = "<ListBucketResult><Contents><Key>oops";
        assert!(parse_list_objects_keys(xml).is_err());
    }

    #[test]
    fn presigned_get_url_rejects_out_of_range_expiry() {
        crate::tls::ensure_crypto_provider();
        let creds = AwsCredentials {
            region: "us-east-1".to_string(),
            access_key_id: "AKIAEXAMPLE".to_string(),
            secret_access_key: "secret".to_string(),
            session_token: None,
        };
        let client = S3Client::new(
            Client::new(),
            "bucket".to_string(),
            creds,
            &RetryConfig::default(),
        );
        assert!(client.presigned_get_url("key", 0).is_err());
        assert!(client.presigned_get_url("key", 8 * 24 * 60 * 60).is_err());
        assert!(client.presigned_get_url("key", 3600).is_ok());
    }

    #[test]
    fn presigned_get_url_contains_expected_query_parameters() {
        crate::tls::ensure_crypto_provider();
        let creds = AwsCredentials {
            region: "us-east-1".to_string(),
            access_key_id: "AKIAEXAMPLE".to_string(),
            secret_access_key: "secret".to_string(),
            session_token: None,
        };
        let client = S3Client::new(
            Client::new(),
            "bucket".to_string(),
            creds,
            &RetryConfig::default(),
        );
        let url = client
            .presigned_get_url("datasets/demo/manifest.json", 3600)
            .unwrap();
        assert!(url.starts_with("https://bucket.s3.us-east-1.amazonaws.com/"));
        assert!(url.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"));
        assert!(url.contains("X-Amz-Expires=3600"));
        assert!(url.contains("X-Amz-Signature="));
        assert!(url.contains("X-Amz-SignedHeaders=host"));
    }
}
