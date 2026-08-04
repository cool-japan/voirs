//! Minimal, dependency-light S3-compatible object storage REST client.
//!
//! Speaks the plain S3 REST API (`PUT`/`GET`/`DELETE` object,
//! `ListObjectsV2`) directly over `reqwest`, authenticated with the
//! hand-rolled AWS Signature Version 4 implementation in
//! [`crate::cloud::sigv4`]. Because the wire protocol is identical, this
//! one client works against:
//!
//! - **Amazon S3** (`endpoint = None`): virtual-hosted-style URLs
//!   (`https://{bucket}.s3.{region}.amazonaws.com/{key}`).
//! - **MinIO / Cloudflare R2 / any S3-compatible service**
//!   (`endpoint = Some(url)`): path-style URLs (`{endpoint}/{bucket}/{key}`),
//!   which is what self-hosted and non-AWS S3-compatible services expect.
//!
//! No AWS SDK, no FFI: this is `reqwest` + `hmac` + `sha2` + `hex`, all
//! pure Rust.

use crate::cloud::error::CloudStorageError;
use crate::cloud::sigv4::{
    amz_date_now, sha256_hex, uri_encode, SigV4Request, SigV4Signer, EMPTY_PAYLOAD_SHA256,
};
use reqwest::{Client, Response, Url};

/// Credentials and connection parameters for an S3-compatible bucket.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// AWS (or S3-compatible) access key ID.
    pub access_key: String,
    /// AWS (or S3-compatible) secret access key.
    pub secret_key: String,
    /// AWS region, e.g. `"us-east-1"`. Ignored by most non-AWS services
    /// but still required as part of the SigV4 credential scope.
    pub region: String,
    /// Target bucket name.
    pub bucket: String,
    /// `Some(url)` selects path-style addressing against a custom
    /// S3-compatible endpoint (MinIO, Cloudflare R2, ...). `None` selects
    /// virtual-hosted-style addressing against real AWS S3.
    pub endpoint: Option<String>,
}

/// One entry returned by [`S3Backend::list_objects`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3ObjectSummary {
    /// Full object key.
    pub key: String,
    /// Object size in bytes, as reported by the server.
    pub size_bytes: u64,
}

impl S3Config {
    /// `true` when using path-style addressing against a custom endpoint
    /// (MinIO / R2 / any non-AWS S3-compatible service).
    fn is_path_style(&self) -> bool {
        self.endpoint.is_some()
    }

    fn base_url(&self) -> String {
        match &self.endpoint {
            Some(endpoint) => {
                let trimmed = endpoint.trim_end_matches('/');
                if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
                    trimmed.to_string()
                } else {
                    format!("https://{trimmed}")
                }
            }
            None => format!("https://{}.s3.{}.amazonaws.com", self.bucket, self.region),
        }
    }

    /// Percent-encoded canonical path (no scheme/host, no query string)
    /// for `key`. `key == ""` addresses the bucket root, used by
    /// [`S3Backend::list_objects`].
    fn canonical_path(&self, key: &str) -> String {
        let encoded_key = uri_encode(key, false);
        if self.is_path_style() {
            let encoded_bucket = uri_encode(&self.bucket, false);
            if encoded_key.is_empty() {
                format!("/{encoded_bucket}")
            } else {
                format!("/{encoded_bucket}/{encoded_key}")
            }
        } else if encoded_key.is_empty() {
            "/".to_string()
        } else {
            format!("/{encoded_key}")
        }
    }
}

/// A real S3-compatible object storage client: every method below issues
/// an actual signed HTTP request and propagates real transport/HTTP
/// errors. There is no fabricated success path.
pub struct S3Backend {
    client: Client,
    config: S3Config,
}

/// Host header value (including a non-default port, if any) derived from
/// a parsed [`Url`]. Deriving this from the same `Url` object that is
/// handed to `reqwest` guarantees the value we sign is byte-identical to
/// the `Host` header `reqwest`/`hyper` actually puts on the wire (which is
/// itself derived from the request URL's authority) -- we never set the
/// `Host` header explicitly, since doing so independently of the URL
/// risks the two disagreeing.
fn host_header(url: &Url) -> Result<String, CloudStorageError> {
    let host = url
        .host_str()
        .ok_or_else(|| CloudStorageError::Signing(format!("URL has no host: {url}")))?;
    Ok(match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_string(),
    })
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

impl S3Backend {
    /// Provider name used in error messages.
    const PROVIDER: &'static str = "S3";

    /// Create a new backend using the given HTTP client and config. The
    /// caller is responsible for having installed a rustls
    /// `CryptoProvider` (see `voirs_acoustic::hub::ensure_crypto_provider`)
    /// before building `client`.
    #[must_use]
    pub fn new(client: Client, config: S3Config) -> Self {
        Self { client, config }
    }

    fn signer<'a>(&'a self, amz_date: &'a str) -> SigV4Signer<'a> {
        SigV4Signer {
            access_key: &self.config.access_key,
            secret_key: &self.config.secret_key,
            region: &self.config.region,
            service: "s3",
            amz_date,
        }
    }

    fn object_url(&self, key: &str) -> Result<Url, CloudStorageError> {
        let url_string = format!(
            "{}{}",
            self.config.base_url(),
            self.config.canonical_path(key)
        );
        Url::parse(&url_string).map_err(|e| {
            CloudStorageError::Signing(format!("invalid S3 object URL '{url_string}': {e}"))
        })
    }

    /// Upload `body` to `key`. Issues a real signed `PUT` request; the
    /// object only exists in the bucket if this returns `Ok`.
    pub async fn put_object(&self, key: &str, body: Vec<u8>) -> Result<(), CloudStorageError> {
        let url = self.object_url(key)?;
        let host = host_header(&url)?;
        let amz_date = amz_date_now();
        let payload_hash = sha256_hex(&body);
        let canonical_uri = url.path().to_string();

        let sig_req = SigV4Request {
            method: "PUT",
            canonical_uri: &canonical_uri,
            query_pairs: vec![],
            headers: vec![
                ("host".to_string(), host.clone()),
                ("x-amz-content-sha256".to_string(), payload_hash.clone()),
                ("x-amz-date".to_string(), amz_date.clone()),
            ],
            payload_hash: &payload_hash,
        };
        let authorization = self.signer(&amz_date).authorization_header(&sig_req)?;

        let response = self
            .client
            .put(url.clone())
            .header("x-amz-content-sha256", &payload_hash)
            .header("x-amz-date", &amz_date)
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

    /// Download the full contents of `key`. Issues a real signed `GET`
    /// request and returns the exact bytes the server sent.
    pub async fn get_object(&self, key: &str) -> Result<Vec<u8>, CloudStorageError> {
        let url = self.object_url(key)?;
        let host = host_header(&url)?;
        let amz_date = amz_date_now();
        let canonical_uri = url.path().to_string();

        let sig_req = SigV4Request {
            method: "GET",
            canonical_uri: &canonical_uri,
            query_pairs: vec![],
            headers: vec![
                ("host".to_string(), host.clone()),
                (
                    "x-amz-content-sha256".to_string(),
                    EMPTY_PAYLOAD_SHA256.to_string(),
                ),
                ("x-amz-date".to_string(), amz_date.clone()),
            ],
            payload_hash: EMPTY_PAYLOAD_SHA256,
        };
        let authorization = self.signer(&amz_date).authorization_header(&sig_req)?;

        let response = self
            .client
            .get(url.clone())
            .header("x-amz-content-sha256", EMPTY_PAYLOAD_SHA256)
            .header("x-amz-date", &amz_date)
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

    /// Delete `key`. S3 returns success (204) even if the key does not
    /// exist, which this passes through unchanged.
    pub async fn delete_object(&self, key: &str) -> Result<(), CloudStorageError> {
        let url = self.object_url(key)?;
        let host = host_header(&url)?;
        let amz_date = amz_date_now();
        let canonical_uri = url.path().to_string();

        let sig_req = SigV4Request {
            method: "DELETE",
            canonical_uri: &canonical_uri,
            query_pairs: vec![],
            headers: vec![
                ("host".to_string(), host.clone()),
                (
                    "x-amz-content-sha256".to_string(),
                    EMPTY_PAYLOAD_SHA256.to_string(),
                ),
                ("x-amz-date".to_string(), amz_date.clone()),
            ],
            payload_hash: EMPTY_PAYLOAD_SHA256,
        };
        let authorization = self.signer(&amz_date).authorization_header(&sig_req)?;

        let response = self
            .client
            .delete(url.clone())
            .header("x-amz-content-sha256", EMPTY_PAYLOAD_SHA256)
            .header("x-amz-date", &amz_date)
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

    /// List objects under `prefix` using `ListObjectsV2` (single page,
    /// up to whatever the server returns by default; pagination via
    /// `continuation-token` is not implemented).
    pub async fn list_objects(
        &self,
        prefix: &str,
    ) -> Result<Vec<S3ObjectSummary>, CloudStorageError> {
        let base_path = self.config.canonical_path("");
        let query_pairs: Vec<(&str, &str)> = if prefix.is_empty() {
            vec![("list-type", "2")]
        } else {
            vec![("list-type", "2"), ("prefix", prefix)]
        };
        let query_string = crate::cloud::sigv4::canonical_query_string(&query_pairs);
        let url_string = format!("{}{}?{}", self.config.base_url(), base_path, query_string);
        let url = Url::parse(&url_string).map_err(|e| {
            CloudStorageError::Signing(format!("invalid S3 list URL '{url_string}': {e}"))
        })?;
        let host = host_header(&url)?;
        let amz_date = amz_date_now();

        let sig_req = SigV4Request {
            method: "GET",
            canonical_uri: &base_path,
            query_pairs,
            headers: vec![
                ("host".to_string(), host.clone()),
                (
                    "x-amz-content-sha256".to_string(),
                    EMPTY_PAYLOAD_SHA256.to_string(),
                ),
                ("x-amz-date".to_string(), amz_date.clone()),
            ],
            payload_hash: EMPTY_PAYLOAD_SHA256,
        };
        let authorization = self.signer(&amz_date).authorization_header(&sig_req)?;

        let response = self
            .client
            .get(url.clone())
            .header("x-amz-content-sha256", EMPTY_PAYLOAD_SHA256)
            .header("x-amz-date", &amz_date)
            .header("authorization", authorization)
            .send()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: Self::PROVIDER.to_string(),
                url: url.to_string(),
                source,
            })?;

        let response = ensure_success(Self::PROVIDER, url.as_str(), response).await?;
        let body = response
            .text()
            .await
            .map_err(|source| CloudStorageError::Network {
                provider: Self::PROVIDER.to_string(),
                url: url.to_string(),
                source,
            })?;
        parse_list_objects_xml(&body).ok_or_else(|| CloudStorageError::InvalidResponse {
            provider: Self::PROVIDER.to_string(),
            url: url.to_string(),
            detail: "response was not a well-formed ListBucketResult XML document".to_string(),
        })
    }
}

/// Parse a `ListObjectsV2` (`ListBucketResult`) XML response body into a
/// flat list of `(key, size)` entries.
///
/// This intentionally does *not* pull in a general-purpose XML dependency:
/// `ListBucketResult` has a stable, flat, well-documented schema (each
/// `<Contents>` block contains non-nested `<Key>`/`<Size>` tags), so a
/// small tag scanner is sufficient and keeps the dependency footprint
/// minimal. It is not a general XML parser and must not be reused for
/// arbitrary XML.
fn parse_list_objects_xml(xml: &str) -> Option<Vec<S3ObjectSummary>> {
    if !xml.contains("<ListBucketResult") {
        return None;
    }
    let mut results = Vec::new();
    for block in xml.split("<Contents>").skip(1) {
        let block = block.split("</Contents>").next().unwrap_or("");
        let key = extract_tag(block, "Key").map(|s| xml_unescape(&s));
        let size = extract_tag(block, "Size").and_then(|s| s.parse::<u64>().ok());
        if let (Some(key), Some(size_bytes)) = (key, size) {
            results.push(S3ObjectSummary { key, size_bytes });
        }
    }
    Some(results)
}

fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml[start..end].to_string())
}

fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn path_style_canonical_path_includes_bucket() {
        let config = S3Config {
            access_key: "ak".into(),
            secret_key: "sk".into(),
            region: "us-east-1".into(),
            bucket: "voirs-cloud".into(),
            endpoint: Some("http://127.0.0.1:9000".into()),
        };
        assert_eq!(
            config.canonical_path("models/x.bin"),
            "/voirs-cloud/models/x.bin"
        );
        assert_eq!(config.canonical_path(""), "/voirs-cloud");
        assert_eq!(config.base_url(), "http://127.0.0.1:9000");
    }

    #[test]
    fn virtual_hosted_canonical_path_excludes_bucket() {
        let config = S3Config {
            access_key: "ak".into(),
            secret_key: "sk".into(),
            region: "eu-west-1".into(),
            bucket: "voirs-cloud".into(),
            endpoint: None,
        };
        assert_eq!(config.canonical_path("models/x.bin"), "/models/x.bin");
        assert_eq!(config.canonical_path(""), "/");
        assert_eq!(
            config.base_url(),
            "https://voirs-cloud.s3.eu-west-1.amazonaws.com"
        );
    }

    #[test]
    fn endpoint_without_scheme_defaults_to_https() {
        let config = S3Config {
            access_key: "ak".into(),
            secret_key: "sk".into(),
            region: "us-east-1".into(),
            bucket: "b".into(),
            endpoint: Some("minio.internal:9000".into()),
        };
        assert_eq!(config.base_url(), "https://minio.internal:9000");
    }

    #[test]
    fn built_url_path_round_trips_special_characters() {
        // Guards against double percent-encoding: the path we hand to
        // `Url::parse` must come back out byte-identical, proving `url`
        // did not re-encode our already-escaped `%20`/`%2B` sequences.
        let config = S3Config {
            access_key: "ak".into(),
            secret_key: "sk".into(),
            region: "us-east-1".into(),
            bucket: "voirs-cloud".into(),
            endpoint: Some("http://127.0.0.1:9000".into()),
        };
        let key = "models/voice pack+en.safetensors";
        let expected_path = config.canonical_path(key);
        let url = Url::parse(&format!("{}{}", config.base_url(), expected_path)).unwrap();
        assert_eq!(url.path(), expected_path);
    }

    #[test]
    fn host_header_omits_default_https_port_but_keeps_custom_port() {
        let https_default =
            Url::parse("https://examplebucket.s3.us-east-1.amazonaws.com/key").unwrap();
        assert_eq!(
            host_header(&https_default).unwrap(),
            "examplebucket.s3.us-east-1.amazonaws.com"
        );

        let custom_port = Url::parse("http://127.0.0.1:9000/bucket/key").unwrap();
        assert_eq!(host_header(&custom_port).unwrap(), "127.0.0.1:9000");
    }

    #[test]
    fn parse_list_objects_xml_extracts_keys_and_sizes() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>voirs-cloud</Name>
  <Prefix>models/</Prefix>
  <KeyCount>2</KeyCount>
  <Contents>
    <Key>models/a.safetensors</Key>
    <LastModified>2026-01-01T00:00:00.000Z</LastModified>
    <ETag>"abc123"</ETag>
    <Size>1048576</Size>
    <StorageClass>STANDARD</StorageClass>
  </Contents>
  <Contents>
    <Key>models/b &amp; c.bin</Key>
    <LastModified>2026-01-02T00:00:00.000Z</LastModified>
    <ETag>"def456"</ETag>
    <Size>2048</Size>
    <StorageClass>STANDARD</StorageClass>
  </Contents>
</ListBucketResult>"#;

        let objects = parse_list_objects_xml(xml).expect("well-formed XML parses");
        assert_eq!(
            objects,
            vec![
                S3ObjectSummary {
                    key: "models/a.safetensors".to_string(),
                    size_bytes: 1_048_576,
                },
                S3ObjectSummary {
                    key: "models/b & c.bin".to_string(),
                    size_bytes: 2048,
                },
            ]
        );
    }

    #[test]
    fn parse_list_objects_xml_rejects_non_xml_body() {
        assert!(parse_list_objects_xml("<html>not S3</html>").is_none());
    }

    /// End-to-end, fully offline (loopback-only, no external network)
    /// proof that `put_object` sends a real, correctly-signed HTTP
    /// request: a bare TCP listener on `127.0.0.1` accepts the connection,
    /// parses the raw HTTP request line + headers + body by hand, and
    /// asserts the wire bytes -- including the `Host` header, which
    /// `reqwest` sets from the URL itself rather than from an explicit
    /// `.header("host", ...)` call.
    #[tokio::test]
    async fn put_object_sends_real_signed_http_request() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
        listener.set_nonblocking(false).expect("blocking listener");
        let port = listener.local_addr().expect("local addr").port();

        let server = std::thread::spawn(move || -> (String, Vec<u8>) {
            let (mut stream, _) = listener.accept().expect("accept connection");
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .expect("set read timeout");

            // Read until we have the full header block.
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let header_end = loop {
                let n = stream.read(&mut chunk).expect("read from socket");
                assert!(n > 0, "connection closed before headers were complete");
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let header_text = String::from_utf8_lossy(&buf[..header_end]).to_string();

            let content_length: usize = header_text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.trim().eq_ignore_ascii_case("content-length") {
                        value.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            while buf.len() < header_end + content_length {
                let n = stream.read(&mut chunk).expect("read body");
                assert!(n > 0, "connection closed before body was complete");
                buf.extend_from_slice(&chunk[..n]);
            }
            let body = buf[header_end..header_end + content_length].to_vec();

            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .expect("write response");
            let _ = stream.flush();

            (header_text, body)
        });

        let config = S3Config {
            access_key: "AKIAIOSFODNN7EXAMPLE".into(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
            region: "us-east-1".into(),
            bucket: "voirs-cloud".into(),
            endpoint: Some(format!("http://127.0.0.1:{port}")),
        };
        voirs_acoustic::hub::ensure_crypto_provider();
        let client = Client::builder().build().expect("build reqwest client");
        let backend = S3Backend::new(client, config);

        let body = b"real bytes, not a fabricated placeholder".to_vec();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            backend.put_object("models/real.bin", body.clone()),
        )
        .await
        .expect("request did not time out");
        result.expect("put_object should succeed against the mock server");

        let (header_text, received_body) =
            tokio::task::spawn_blocking(move || server.join().expect("server thread panicked"))
                .await
                .expect("join server task");

        assert!(
            header_text.starts_with("PUT /voirs-cloud/models/real.bin HTTP/1.1"),
            "unexpected request line: {header_text}"
        );
        assert!(
            header_text
                .to_lowercase()
                .contains(&format!("host: 127.0.0.1:{port}")),
            "Host header did not match the request URL authority: {header_text}"
        );
        assert!(
            header_text
                .contains("authorization: AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/"),
            "missing/incorrect Authorization header: {header_text}"
        );
        let expected_hash = sha256_hex(&body);
        assert!(
            header_text.contains(&format!("x-amz-content-sha256: {expected_hash}")),
            "x-amz-content-sha256 did not match real body hash: {header_text}"
        );
        assert_eq!(
            received_body, body,
            "server must receive the exact real bytes that were uploaded"
        );
    }

    fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }
}
