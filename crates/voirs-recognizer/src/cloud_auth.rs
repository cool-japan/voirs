//! Cloud authentication signing helpers
//!
//! Implements AWS SigV4, Azure SharedKey, and GCS HMAC signing.

use crate::cloud_storage::CloudStorageError;
use std::collections::HashMap;

#[cfg(feature = "cloud")]
use {
    base64::{engine::general_purpose::STANDARD, Engine as _},
    hmac::{Hmac, KeyInit, Mac},
    sha2::{Digest, Sha256},
    url::Url,
};

#[cfg(feature = "cloud")]
type HmacSha256 = Hmac<Sha256>;

#[cfg(feature = "cloud")]
fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<Vec<u8>, CloudStorageError> {
    let mut mac: HmacSha256 = KeyInit::new_from_slice(key)
        .map_err(|e| CloudStorageError::AuthenticationFailed(format!("HMAC key error: {e}")))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

#[cfg(feature = "cloud")]
fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Returns current UTC timestamp in AWS format: "20130524T000000Z"
#[cfg(feature = "cloud")]
pub fn current_timestamp() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

/// Returns current UTC date in format: "20130524"
#[cfg(feature = "cloud")]
pub fn current_date() -> String {
    chrono::Utc::now().format("%Y%m%d").to_string()
}

/// Sign an AWS S3 request using SigV4.
///
/// Returns a HashMap with headers: "Authorization", "x-amz-date",
/// "x-amz-content-sha256", "host".
#[cfg(feature = "cloud")]
pub fn sign_s3_request(
    method: &str,
    url_str: &str,
    body_sha256: &str,
    access_key: &str,
    secret_key: &str,
    region: &str,
    service: &str,
    timestamp: &str,
) -> Result<HashMap<String, String>, CloudStorageError> {
    let parsed =
        Url::parse(url_str).map_err(|e| CloudStorageError::InvalidConfiguration(e.to_string()))?;
    let host = parsed.host_str().unwrap_or("").to_string();
    let path = parsed.path().to_string();
    let query_string = parsed.query().unwrap_or("").to_string();

    let canonical_headers = format!(
        "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        host, body_sha256, timestamp
    );
    let signed_headers = "host;x-amz-content-sha256;x-amz-date";

    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method, path, query_string, canonical_headers, signed_headers, body_sha256
    );

    let datestamp = &timestamp[..8];
    let credential_scope = format!("{}/{}/{}/aws4_request", datestamp, region, service);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        timestamp,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );

    let signing_key = {
        let k1 = hmac_sha256(
            format!("AWS4{}", secret_key).as_bytes(),
            datestamp.as_bytes(),
        )?;
        let k2 = hmac_sha256(&k1, region.as_bytes())?;
        let k3 = hmac_sha256(&k2, service.as_bytes())?;
        hmac_sha256(&k3, b"aws4_request")?
    };

    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes())?);

    let auth_header = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        access_key, credential_scope, signed_headers, signature
    );

    let mut headers = HashMap::new();
    headers.insert("Authorization".to_string(), auth_header);
    headers.insert("x-amz-date".to_string(), timestamp.to_string());
    headers.insert("x-amz-content-sha256".to_string(), body_sha256.to_string());
    headers.insert("host".to_string(), host);
    Ok(headers)
}

/// Sign an Azure Blob Storage request using SharedKey scheme.
///
/// Returns the full `Authorization` header value.
/// When `blob_name` is empty, generates a list-blobs signature.
#[cfg(feature = "cloud")]
pub fn sign_azure_request(
    method: &str,
    account_name: &str,
    container: &str,
    blob_name: &str,
    content_length: u64,
    content_type: &str,
    x_ms_date: &str,
    x_ms_version: &str,
    account_key_b64: &str,
) -> Result<String, CloudStorageError> {
    let canon_headers = format!("x-ms-date:{}\nx-ms-version:{}\n", x_ms_date, x_ms_version);

    let canon_resource = if blob_name.is_empty() {
        format!(
            "/{}/{}\ncomp:list\nrestype:container",
            account_name, container
        )
    } else {
        format!("/{}/{}/{}", account_name, container, blob_name)
    };

    let content_length_str = if content_length == 0 {
        String::new()
    } else {
        content_length.to_string()
    };

    let string_to_sign = format!(
        "{}\n\n\n{}\n\n{}\n\n\n\n\n\n\n{}{}",
        method, content_length_str, content_type, canon_headers, canon_resource
    );

    let key_bytes = STANDARD.decode(account_key_b64).map_err(|_| {
        CloudStorageError::AuthenticationFailed(
            "Invalid Azure account key (base64 decode failed)".into(),
        )
    })?;

    let sig_bytes = hmac_sha256(&key_bytes, string_to_sign.as_bytes())?;
    let signature = STANDARD.encode(sig_bytes);

    Ok(format!("SharedKey {}:{}", account_name, signature))
}

/// Sign a GCS request using HMAC keys (S3-compatible API).
///
/// Requires HMAC access key and secret. Delegates to [`sign_s3_request`]
/// targeting the GCS S3-compatible endpoint.
#[cfg(feature = "cloud")]
pub fn sign_gcs_request(
    method: &str,
    url_str: &str,
    body_sha256: &str,
    access_key: &str,
    secret_key: &str,
    region: &str,
    timestamp: &str,
) -> Result<HashMap<String, String>, CloudStorageError> {
    if access_key.is_empty() || access_key.starts_with('/') {
        return Err(CloudStorageError::InvalidConfiguration(
            "GCS requires HMAC keys (access_key_id + secret_access_key) for S3-compatible API"
                .to_string(),
        ));
    }
    sign_s3_request(
        method,
        url_str,
        body_sha256,
        access_key,
        secret_key,
        region,
        "s3",
        timestamp,
    )
}
