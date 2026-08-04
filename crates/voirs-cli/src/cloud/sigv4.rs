//! AWS Signature Version 4 (SigV4) request signing.
//!
//! A minimal, dependency-light, pure-Rust implementation of the SigV4
//! canonical-request / string-to-sign / signing-key algorithm exactly as
//! published by AWS
//! (<https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html>),
//! built only on the `hmac` + `sha2` RustCrypto crates plus `hex`. This is
//! all Amazon S3 (and any S3-compatible object store: MinIO, Cloudflare R2,
//! ...) needs for request authentication, so [`crate::cloud::s3_backend`]
//! can speak the real S3 REST API with real, independently verifiable HTTP
//! requests -- no AWS SDK and no FFI required.
//!
//! Every function in this module is pure (no I/O, no clock reads beyond
//! what the caller passes in), which makes the whole signing pipeline
//! testable offline. The unit tests below assert exact canonical-request,
//! string-to-sign, and signature values, cross-checked against an
//! independent Python (`hashlib`/`hmac`) implementation of the same
//! AWS-published algorithm.

use crate::cloud::error::CloudStorageError;
use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// Hex SHA-256 digest of the empty byte string. Used as the payload hash
/// for requests that carry no body (GET, DELETE, HEAD, ListObjectsV2).
pub const EMPTY_PAYLOAD_SHA256: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Lowercase-hex SHA-256 digest of `data`.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// HMAC-SHA256 of `data` under `key`. HMAC accepts keys of any length (long
/// keys are hashed down, short keys are zero-padded internally), so this
/// only returns `Err` if the underlying crate's key-length invariant is
/// ever violated -- which cannot happen for HMAC, but we still surface it
/// as a typed error instead of unwrapping, per VoiRS's no-panic policy.
fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<[u8; 32], CloudStorageError> {
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|e| CloudStorageError::Signing(format!("HMAC-SHA256 key error: {e}")))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().into())
}

/// Percent-encode `input` per SigV4's `UriEncode()` rules: RFC 3986
/// unreserved characters (`A-Za-z0-9-_.~`) pass through unescaped; every
/// other byte becomes `%XX` using uppercase hex digits. When
/// `encode_slash` is `false`, `/` is also left unescaped -- used for
/// encoding a full path (where `/` is a structural separator) rather than
/// a single path segment or a query-string component.
#[must_use]
pub fn uri_encode(input: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            b'/' if !encode_slash => out.push('/'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Build the canonical (and wire-identical) query string from `pairs`:
/// each name/value percent-encoded individually, then sorted by the
/// encoded name. Both the request signer and the URL builder must call
/// this *same* function so the signed query string can never drift from
/// the one that is actually sent on the wire.
#[must_use]
pub fn canonical_query_string(pairs: &[(&str, &str)]) -> String {
    let mut encoded: Vec<(String, String)> = pairs
        .iter()
        .map(|(k, v)| (uri_encode(k, true), uri_encode(v, true)))
        .collect();
    encoded.sort();
    encoded
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// Collapse any run of whitespace into a single space, matching the
/// `CanonicalHeaders` trimming rule ("convert sequential spaces to a
/// single space").
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_was_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(c);
            last_was_space = false;
        }
    }
    out
}

/// One HTTP request to be signed.
#[derive(Debug, Clone)]
pub struct SigV4Request<'a> {
    /// HTTP method, e.g. `"PUT"`, `"GET"`, `"DELETE"`.
    pub method: &'a str,
    /// Already percent-encoded absolute path, e.g. `/bucket/key`.
    pub canonical_uri: &'a str,
    /// Un-encoded query parameters; encoding and sorting happens inside
    /// [`SigV4Signer::canonical_request`] via [`canonical_query_string`].
    pub query_pairs: Vec<(&'a str, &'a str)>,
    /// Header name (any case) -> value. Must include `host`, and for S3
    /// must include `x-amz-content-sha256` and `x-amz-date`.
    pub headers: Vec<(String, String)>,
    /// Hex SHA-256 of the request body, or the literal string
    /// `UNSIGNED-PAYLOAD`.
    pub payload_hash: &'a str,
}

/// Derives a SigV4 signing key and signs requests for one
/// `(access_key, secret_key, region, service)` credential scope.
#[derive(Debug, Clone)]
pub struct SigV4Signer<'a> {
    /// AWS (or S3-compatible) access key ID.
    pub access_key: &'a str,
    /// AWS (or S3-compatible) secret access key.
    pub secret_key: &'a str,
    /// AWS region, e.g. `"us-east-1"`.
    pub region: &'a str,
    /// AWS service code, e.g. `"s3"`.
    pub service: &'a str,
    /// Request timestamp in `YYYYMMDDTHHMMSSZ` (ISO 8601 basic) format.
    pub amz_date: &'a str,
}

impl SigV4Signer<'_> {
    /// The `YYYYMMDD` date stamp used in the credential scope, i.e. the
    /// first 8 characters of `amz_date`.
    #[must_use]
    pub fn date_stamp(&self) -> &str {
        self.amz_date.get(..8).unwrap_or(self.amz_date)
    }

    /// The `YYYYMMDD/region/service/aws4_request` credential scope.
    #[must_use]
    pub fn credential_scope(&self) -> String {
        format!(
            "{}/{}/{}/aws4_request",
            self.date_stamp(),
            self.region,
            self.service
        )
    }

    fn signing_key(&self) -> Result<[u8; 32], CloudStorageError> {
        let k_date = hmac_sha256(
            format!("AWS4{}", self.secret_key).as_bytes(),
            self.date_stamp().as_bytes(),
        )?;
        let k_region = hmac_sha256(&k_date, self.region.as_bytes())?;
        let k_service = hmac_sha256(&k_region, self.service.as_bytes())?;
        hmac_sha256(&k_service, b"aws4_request")
    }

    /// Build `(canonical_request, signed_headers)` for `req`, per the AWS
    /// SigV4 spec: headers are lowercased, whitespace-collapsed, sorted,
    /// and newline-joined; the query string is built via
    /// [`canonical_query_string`].
    #[must_use]
    pub fn canonical_request(&self, req: &SigV4Request<'_>) -> (String, String) {
        let mut headers: Vec<(String, String)> = req
            .headers
            .iter()
            .map(|(k, v)| (k.to_lowercase(), collapse_whitespace(v.trim())))
            .collect();
        headers.sort();
        headers.dedup_by(|a, b| a.0 == b.0);

        let canonical_headers: String = headers
            .iter()
            .map(|(k, v)| format!("{k}:{v}\n"))
            .collect();
        let signed_headers = headers
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
            .join(";");

        let canonical_query = canonical_query_string(&req.query_pairs);

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            req.method,
            req.canonical_uri,
            canonical_query,
            canonical_headers,
            signed_headers,
            req.payload_hash
        );

        (canonical_request, signed_headers)
    }

    /// Compute the full `Authorization` header value for `req`.
    ///
    /// # Errors
    /// Returns [`CloudStorageError::Signing`] if the underlying HMAC
    /// primitive rejects a key (not reachable in practice: HMAC accepts
    /// keys of any length).
    pub fn authorization_header(
        &self,
        req: &SigV4Request<'_>,
    ) -> Result<String, CloudStorageError> {
        let (canonical_request, signed_headers) = self.canonical_request(req);
        let hashed_canonical_request = sha256_hex(canonical_request.as_bytes());
        let scope = self.credential_scope();
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            self.amz_date, scope, hashed_canonical_request
        );
        let signing_key = self.signing_key()?;
        let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes())?);

        Ok(format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
            self.access_key, scope, signed_headers, signature
        ))
    }
}

/// Format the current UTC time as an SigV4 `amz_date`
/// (`YYYYMMDDTHHMMSSZ`).
#[must_use]
pub fn amz_date_now() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // All expected values below were cross-checked offline against an
    // independent Python implementation (`hashlib` + `hmac`, stdlib) of
    // the exact algorithm published by AWS at
    // https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html
    // (canonical request, string-to-sign, signing-key derivation, and
    // final signature). Using a second, independent implementation of the
    // documented algorithm is a real cross-check: it catches transcription
    // and wiring bugs in the Rust implementation, and a hand-verified
    // constant (`EMPTY_PAYLOAD_SHA256` below) anchors both against a
    // publicly documented value rather than trusting only self-consistency.
    // ------------------------------------------------------------------

    #[test]
    fn empty_payload_hash_matches_sha256_of_empty_string() {
        // This is the one constant independently checkable without any
        // cross-implementation: SHA-256("") is a fixed, widely published
        // value. If this fails, `EMPTY_PAYLOAD_SHA256` itself was
        // mistyped.
        assert_eq!(sha256_hex(b""), EMPTY_PAYLOAD_SHA256);
    }

    #[test]
    fn uri_encode_preserves_unreserved_and_escapes_the_rest() {
        assert_eq!(
            uri_encode("azAZ09-_.~", false),
            "azAZ09-_.~",
            "unreserved characters must never be escaped"
        );
        assert_eq!(uri_encode("a b", true), "a%20b");
        assert_eq!(uri_encode("a+b", true), "a%2Bb");
        assert_eq!(uri_encode("a/b", true), "a%2Fb");
        assert_eq!(
            uri_encode("a/b", false),
            "a/b",
            "encode_slash=false must keep '/' literal"
        );
        assert_eq!(uri_encode("$", true), "%24");
    }

    #[test]
    fn canonical_query_string_sorts_and_encodes() {
        // Query params must be sorted by (encoded) name, each value
        // percent-encoded individually.
        assert_eq!(
            canonical_query_string(&[("prefix", "models/"), ("list-type", "2")]),
            "list-type=2&prefix=models%2F"
        );
        assert_eq!(canonical_query_string(&[]), "");
    }

    /// Case 1: virtual-hosted-style PUT with a special character (`$`) in
    /// the object key, non-empty body.
    #[test]
    fn sigv4_case1_virtual_hosted_put_special_char_key() {
        let raw_key = "test$file.text";
        let canonical_uri = format!("/{}", uri_encode(raw_key, false));
        assert_eq!(canonical_uri, "/test%24file.text");

        let payload = b"Welcome to Amazon S3.".to_vec();
        let payload_hash = sha256_hex(&payload);
        assert_eq!(
            payload_hash,
            "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
        );

        let signer = SigV4Signer {
            access_key: "AKIAIOSFODNN7EXAMPLE",
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region: "us-east-1",
            service: "s3",
            amz_date: "20130524T000000Z",
        };
        let req = SigV4Request {
            method: "PUT",
            canonical_uri: &canonical_uri,
            query_pairs: vec![],
            headers: vec![
                (
                    "host".to_string(),
                    "examplebucket.s3.us-east-1.amazonaws.com".to_string(),
                ),
                ("x-amz-content-sha256".to_string(), payload_hash.clone()),
                ("x-amz-date".to_string(), "20130524T000000Z".to_string()),
            ],
            payload_hash: &payload_hash,
        };

        let (canonical_request, signed_headers) = signer.canonical_request(&req);
        assert_eq!(signed_headers, "host;x-amz-content-sha256;x-amz-date");
        assert_eq!(
            canonical_request,
            "PUT\n/test%24file.text\n\nhost:examplebucket.s3.us-east-1.amazonaws.com\nx-amz-content-sha256:44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072\nx-amz-date:20130524T000000Z\n\nhost;x-amz-content-sha256;x-amz-date\n44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
        );

        let authorization = signer.authorization_header(&req).expect("signing succeeds");
        assert_eq!(
            authorization,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=67df5e3b3bbc1cce47b9479a00276b09765f4226ac48d16e52119dc4bd050c80"
        );
    }

    /// Case 2: path-style GET (MinIO-style custom endpoint with an
    /// explicit non-default port), empty body, object key containing a
    /// folder separator, a space, and a `+`.
    #[test]
    fn sigv4_case2_path_style_get_minio_with_port() {
        let raw_key = "models/voice pack+en.safetensors";
        let bucket = "voirs-cloud";
        let canonical_uri = format!(
            "/{}/{}",
            uri_encode(bucket, false),
            uri_encode(raw_key, false)
        );
        assert_eq!(
            canonical_uri,
            "/voirs-cloud/models/voice%20pack%2Ben.safetensors"
        );

        let signer = SigV4Signer {
            access_key: "minioadmin",
            secret_key: "minioadminsecret",
            region: "us-east-1",
            service: "s3",
            amz_date: "20260101T120000Z",
        };
        let req = SigV4Request {
            method: "GET",
            canonical_uri: &canonical_uri,
            query_pairs: vec![],
            headers: vec![
                ("host".to_string(), "127.0.0.1:9000".to_string()),
                (
                    "x-amz-content-sha256".to_string(),
                    EMPTY_PAYLOAD_SHA256.to_string(),
                ),
                ("x-amz-date".to_string(), "20260101T120000Z".to_string()),
            ],
            payload_hash: EMPTY_PAYLOAD_SHA256,
        };

        let (canonical_request, _signed_headers) = signer.canonical_request(&req);
        assert_eq!(
            canonical_request,
            "GET\n/voirs-cloud/models/voice%20pack%2Ben.safetensors\n\nhost:127.0.0.1:9000\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20260101T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let authorization = signer.authorization_header(&req).expect("signing succeeds");
        assert_eq!(
            authorization,
            "AWS4-HMAC-SHA256 Credential=minioadmin/20260101/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=76110e73b6eff500210b8338b60baf22bd81583ab913a355aa7e19e70541bc41"
        );
    }

    /// Case 3: `ListObjectsV2`-style GET with multiple query parameters,
    /// verifying that `canonical_query_string` sorts them correctly inside
    /// a real signature computation.
    #[test]
    fn sigv4_case3_list_objects_query_params() {
        let signer = SigV4Signer {
            access_key: "AKIAIOSFODNN7EXAMPLE",
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region: "us-west-2",
            service: "s3",
            amz_date: "20260615T093000Z",
        };
        let req = SigV4Request {
            method: "GET",
            canonical_uri: "/",
            // Deliberately out of alphabetical order; canonical_request
            // must sort them.
            query_pairs: vec![("prefix", "models/"), ("list-type", "2")],
            headers: vec![
                (
                    "host".to_string(),
                    "examplebucket.s3.us-west-2.amazonaws.com".to_string(),
                ),
                (
                    "x-amz-content-sha256".to_string(),
                    EMPTY_PAYLOAD_SHA256.to_string(),
                ),
                ("x-amz-date".to_string(), "20260615T093000Z".to_string()),
            ],
            payload_hash: EMPTY_PAYLOAD_SHA256,
        };

        let (canonical_request, _) = signer.canonical_request(&req);
        assert_eq!(
            canonical_request,
            "GET\n/\nlist-type=2&prefix=models%2F\nhost:examplebucket.s3.us-west-2.amazonaws.com\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20260615T093000Z\n\nhost;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let authorization = signer.authorization_header(&req).expect("signing succeeds");
        assert_eq!(
            authorization,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20260615/us-west-2/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=5bb77069f1e4480209f195743af3bf4f2bb74cd2698bac543bb0f3293023113a"
        );
    }

    #[test]
    fn signature_changes_when_secret_key_changes() {
        // A signature that doesn't actually depend on the secret key would
        // be a fabrication bug hiding as a "real" implementation. Assert
        // the output genuinely varies with input.
        let req = SigV4Request {
            method: "GET",
            canonical_uri: "/",
            query_pairs: vec![],
            headers: vec![
                ("host".to_string(), "example.amazonaws.com".to_string()),
                (
                    "x-amz-content-sha256".to_string(),
                    EMPTY_PAYLOAD_SHA256.to_string(),
                ),
                ("x-amz-date".to_string(), "20260101T000000Z".to_string()),
            ],
            payload_hash: EMPTY_PAYLOAD_SHA256,
        };
        let signer_a = SigV4Signer {
            access_key: "AKID",
            secret_key: "secret-one",
            region: "us-east-1",
            service: "s3",
            amz_date: "20260101T000000Z",
        };
        let signer_b = SigV4Signer {
            secret_key: "secret-two",
            ..signer_a.clone()
        };

        let sig_a = signer_a.authorization_header(&req).expect("signs");
        let sig_b = signer_b.authorization_header(&req).expect("signs");
        assert_ne!(sig_a, sig_b);
    }

    #[test]
    fn amz_date_now_has_expected_shape() {
        let date = amz_date_now();
        assert_eq!(date.len(), 16, "expected YYYYMMDDTHHMMSSZ, got {date}");
        assert!(date.ends_with('Z'));
        assert!(date.as_bytes()[8] == b'T');
    }
}
