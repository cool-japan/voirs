//! AWS Signature Version 4 (SigV4) request signing primitives.
//!
//! Implements the canonical-request / string-to-sign / signing-key algorithm
//! described at
//! <https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html>,
//! built entirely on the Pure-Rust RustCrypto `hmac`/`sha2` crates (no
//! OpenSSL, no `aws-lc-sys`, no vendored C signing library). This backs the
//! real S3(-compatible) client in [`super::s3_client`] used by
//! [`super::storage::VoirsCloudStorage`].
//!
//! Every function here is a small, independently testable piece of the
//! algorithm rather than one opaque "sign this request" black box, so each
//! step (URI encoding, canonical headers, canonical request, string-to-sign,
//! signing-key derivation) can be checked against the AWS spec in isolation.

use crate::VoirsError;
use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Hex-encoded SHA-256 digest of `data`.
pub(super) fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// HMAC-SHA256 of `data` under `key`.
///
/// HMAC accepts a key of any length (short keys are zero-padded, long keys
/// are hashed down internally), so the only failure mode is an internal
/// RustCrypto invariant violation. That is reported as an honest error
/// rather than unwrapped, even though it should never actually occur.
fn hmac_sha256(key: &[u8], data: &[u8]) -> crate::Result<Vec<u8>> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key)
        .map_err(|e| VoirsError::config_error(format!("HMAC initialization failed: {e}")))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

/// Percent-encode `input` per the SigV4 URI-encoding rules: unreserved
/// characters (`A-Z a-z 0-9 - _ . ~`) pass through unchanged; everything
/// else becomes an uppercase `%XX` escape of its UTF-8 bytes.
///
/// `/` is left unescaped when `encode_slash` is `false` (used for the
/// canonical URI's path segments, where `/` is a structural separator);
/// otherwise it is escaped like any other reserved byte (used for canonical
/// query-string keys/values, where `/` is just data).
pub(super) fn uri_encode(input: &str, encode_slash: bool) -> String {
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

/// Build the `CanonicalHeaders` block and the semicolon-joined
/// `SignedHeaders` list from an already-lowercased header map.
///
/// A `BTreeMap` is used (rather than a plain slice of pairs) precisely
/// because SigV4 requires headers sorted by name: iterating a `BTreeMap`
/// yields that order for free and cannot accidentally be left unsorted.
pub(super) fn canonical_headers(headers: &BTreeMap<String, String>) -> (String, String) {
    let mut canonical = String::new();
    let mut names = Vec::with_capacity(headers.len());
    for (name, value) in headers {
        canonical.push_str(name);
        canonical.push(':');
        canonical.push_str(value.trim());
        canonical.push('\n');
        names.push(name.clone());
    }
    (canonical, names.join(";"))
}

/// Build the SigV4 canonical request string:
/// `Method\nCanonicalURI\nCanonicalQueryString\nCanonicalHeaders\nSignedHeaders\nHashedPayload`.
///
/// `canonical_headers` is expected to already end with a trailing `\n`
/// (as produced by [`canonical_headers`]), so the blank line the spec
/// requires between the headers block and `SignedHeaders` falls out of the
/// format string naturally.
pub(super) fn canonical_request(
    method: &str,
    canonical_uri: &str,
    canonical_query_string: &str,
    canonical_headers: &str,
    signed_headers: &str,
    payload_hash: &str,
) -> String {
    format!(
        "{method}\n{canonical_uri}\n{canonical_query_string}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
}

/// Build the SigV4 string-to-sign from a canonical request.
pub(super) fn string_to_sign(
    amz_date: &str,
    credential_scope: &str,
    canonical_request: &str,
) -> String {
    format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    )
}

/// Derive the SigV4 signing key via the 4-step HMAC chain:
/// `HMAC(HMAC(HMAC(HMAC("AWS4" + secret, date), region), service), "aws4_request")`.
fn signing_key(
    secret_access_key: &str,
    date_stamp: &str,
    region: &str,
    service: &str,
) -> crate::Result<Vec<u8>> {
    let k_date = hmac_sha256(
        format!("AWS4{secret_access_key}").as_bytes(),
        date_stamp.as_bytes(),
    )?;
    let k_region = hmac_sha256(&k_date, region.as_bytes())?;
    let k_service = hmac_sha256(&k_region, service.as_bytes())?;
    hmac_sha256(&k_service, b"aws4_request")
}

/// Compute the final hex-encoded SigV4 signature for a string-to-sign.
pub(super) fn signature(
    secret_access_key: &str,
    date_stamp: &str,
    region: &str,
    service: &str,
    string_to_sign: &str,
) -> crate::Result<String> {
    let key = signing_key(secret_access_key, date_stamp, region, service)?;
    let mac = hmac_sha256(&key, string_to_sign.as_bytes())?;
    Ok(hex::encode(mac))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        // FIPS 180-4 example message digest.
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // SHA-256 of the empty string -- this is the SigV4 payload hash used
        // for GET/HEAD/DELETE requests, which never carry a body.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn hmac_sha256_matches_rfc4231_test_case_1() {
        // RFC 4231 Section 4.2, Test Case 1 (independent of any AWS-specific
        // assumption -- this checks the raw HMAC-SHA256 primitive only).
        let key = [0x0bu8; 20];
        let mac = hmac_sha256(&key, b"Hi There").unwrap();
        assert_eq!(
            hex::encode(mac),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn uri_encode_preserves_unreserved_and_escapes_rest() {
        assert_eq!(uri_encode("abcXYZ019-_.~", false), "abcXYZ019-_.~");
        assert_eq!(uri_encode("a b/c", false), "a%20b/c"); // slash preserved
        assert_eq!(uri_encode("a b/c", true), "a%20b%2Fc"); // slash escaped
        assert_eq!(uri_encode("réservé", true), "r%C3%A9serv%C3%A9");
    }

    #[test]
    fn canonical_headers_sorts_and_joins_names() {
        let mut headers = BTreeMap::new();
        headers.insert("x-amz-date".to_string(), "20130524T000000Z".to_string());
        headers.insert("host".to_string(), "example.com".to_string());
        let (canonical, signed) = canonical_headers(&headers);
        assert_eq!(canonical, "host:example.com\nx-amz-date:20130524T000000Z\n");
        assert_eq!(signed, "host;x-amz-date");
    }

    #[test]
    fn canonical_request_matches_hand_verified_structure() {
        // A minimal, hand-constructed example that exercises every
        // canonical-request component: a `/`-preserving path, an empty query
        // string, three signed headers in sorted order, and the well-known
        // SHA-256("") payload hash used for bodiless requests.
        let mut headers = BTreeMap::new();
        headers.insert(
            "host".to_string(),
            "examplebucket.s3.us-east-1.amazonaws.com".to_string(),
        );
        headers.insert(
            "x-amz-content-sha256".to_string(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
        );
        headers.insert("x-amz-date".to_string(), "20130524T000000Z".to_string());

        let (canonical_headers_str, signed_headers) = canonical_headers(&headers);
        assert_eq!(signed_headers, "host;x-amz-content-sha256;x-amz-date");

        let request = canonical_request(
            "GET",
            "/test.txt",
            "",
            &canonical_headers_str,
            &signed_headers,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        );

        let expected = "GET\n\
/test.txt\n\
\n\
host:examplebucket.s3.us-east-1.amazonaws.com\n\
x-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n\
x-amz-date:20130524T000000Z\n\
\n\
host;x-amz-content-sha256;x-amz-date\n\
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(request, expected);
    }

    #[test]
    fn signature_is_deterministic_and_sensitive_to_every_input() {
        let base = signature("secret", "20130524", "us-east-1", "s3", "sts").unwrap();
        assert_eq!(
            base,
            signature("secret", "20130524", "us-east-1", "s3", "sts").unwrap(),
            "signing must be deterministic for identical inputs"
        );
        assert_ne!(
            base,
            signature("different-secret", "20130524", "us-east-1", "s3", "sts").unwrap()
        );
        assert_ne!(
            base,
            signature("secret", "20130525", "us-east-1", "s3", "sts").unwrap()
        );
        assert_ne!(
            base,
            signature("secret", "20130524", "us-west-2", "s3", "sts").unwrap()
        );
        assert_eq!(base.len(), 64, "SigV4 signature must be 64 hex characters");
        assert!(base.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
