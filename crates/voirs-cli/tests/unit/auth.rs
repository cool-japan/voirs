//! Unit tests for authentication and rate limiting

use axum::http::{HeaderMap, HeaderValue};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use voirs_cli::commands::server::{
    extract_api_key, extract_client_ip, ApiKeyConfig, AuthState, RateLimitBucket, UsageStats,
};

#[test]
fn test_api_key_config_creation() {
    let config = ApiKeyConfig {
        key: "test-key".to_string(),
        name: "Test Key".to_string(),
        rate_limit: 100,
        enabled: true,
        created_at: SystemTime::now(),
    };

    assert_eq!(config.key, "test-key");
    assert_eq!(config.name, "Test Key");
    assert_eq!(config.rate_limit, 100);
    assert!(config.enabled);
}

#[test]
fn test_rate_limit_bucket() {
    let mut bucket = RateLimitBucket {
        requests: 0,
        window_start: Instant::now(),
        limit: 10,
    };

    // Test incrementing requests
    bucket.requests += 1;
    assert_eq!(bucket.requests, 1);
    assert!(bucket.requests < bucket.limit);

    // Test hitting limit
    bucket.requests = bucket.limit;
    assert_eq!(bucket.requests, bucket.limit);
}

#[test]
fn test_usage_stats_default() {
    let stats = UsageStats::default();

    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.successful_requests, 0);
    assert_eq!(stats.failed_requests, 0);
    assert_eq!(stats.total_audio_seconds, 0.0);
    assert_eq!(stats.bytes_transferred, 0);
    assert!(stats.last_used.is_none());
}

#[test]
fn test_usage_stats_tracking() {
    let mut stats = UsageStats::default();

    stats.total_requests += 1;
    stats.successful_requests += 1;
    stats.total_audio_seconds += 2.5;
    stats.bytes_transferred += 1024;
    stats.last_used = Some(SystemTime::now());

    assert_eq!(stats.total_requests, 1);
    assert_eq!(stats.successful_requests, 1);
    assert_eq!(stats.failed_requests, 0);
    assert_eq!(stats.total_audio_seconds, 2.5);
    assert_eq!(stats.bytes_transferred, 1024);
    assert!(stats.last_used.is_some());
}

#[test]
fn test_extract_api_key_from_authorization_bearer() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer test-api-key-123"),
    );

    let api_key = extract_api_key(&headers);
    assert_eq!(api_key, Some("test-api-key-123".to_string()));
}

#[test]
fn test_extract_api_key_from_authorization_apikey() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("ApiKey my-secret-key"),
    );

    let api_key = extract_api_key(&headers);
    assert_eq!(api_key, Some("my-secret-key".to_string()));
}

#[test]
fn test_extract_api_key_from_x_api_key_header() {
    let mut headers = HeaderMap::new();
    headers.insert("x-api-key", HeaderValue::from_static("another-api-key"));

    let api_key = extract_api_key(&headers);
    assert_eq!(api_key, Some("another-api-key".to_string()));
}

#[test]
fn test_extract_api_key_no_header() {
    let headers = HeaderMap::new();
    let api_key = extract_api_key(&headers);
    assert_eq!(api_key, None);
}

#[test]
fn test_extract_api_key_invalid_format() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("InvalidFormat test-key"),
    );

    let api_key = extract_api_key(&headers);
    assert_eq!(api_key, None);
}

#[test]
fn test_extract_client_ip_from_x_forwarded_for() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-forwarded-for",
        HeaderValue::from_static("192.168.1.100, 10.0.0.1"),
    );

    // Mock request (not used in this function currently)
    let request = axum::extract::Request::builder()
        .body(axum::body::Body::empty())
        .unwrap();

    let ip = extract_client_ip(&headers, &request);
    assert_eq!(ip, "192.168.1.100");
}

#[test]
fn test_extract_client_ip_from_x_real_ip() {
    let mut headers = HeaderMap::new();
    headers.insert("x-real-ip", HeaderValue::from_static("203.0.113.195"));

    let request = axum::extract::Request::builder()
        .body(axum::body::Body::empty())
        .unwrap();

    let ip = extract_client_ip(&headers, &request);
    assert_eq!(ip, "203.0.113.195");
}

#[test]
fn test_extract_client_ip_fallback() {
    let headers = HeaderMap::new();
    let request = axum::extract::Request::builder()
        .body(axum::body::Body::empty())
        .unwrap();

    let ip = extract_client_ip(&headers, &request);
    assert_eq!(ip, "unknown");
}

#[test]
fn test_auth_state_initialization() {
    let auth_state = AuthState {
        api_keys: HashMap::new(),
        rate_limits: HashMap::new(),
        usage_stats: HashMap::new(),
        access_logs: Vec::new(),
    };

    assert!(auth_state.api_keys.is_empty());
    assert!(auth_state.rate_limits.is_empty());
    assert!(auth_state.usage_stats.is_empty());
    assert!(auth_state.access_logs.is_empty());
}

#[test]
fn test_auth_state_with_api_key() {
    let mut auth_state = AuthState {
        api_keys: HashMap::new(),
        rate_limits: HashMap::new(),
        usage_stats: HashMap::new(),
        access_logs: Vec::new(),
    };

    let api_key = ApiKeyConfig {
        key: "test-key".to_string(),
        name: "Test Key".to_string(),
        rate_limit: 60,
        enabled: true,
        created_at: SystemTime::now(),
    };

    auth_state.api_keys.insert("test-key".to_string(), api_key);

    assert_eq!(auth_state.api_keys.len(), 1);
    assert!(auth_state.api_keys.contains_key("test-key"));

    let stored_key = auth_state.api_keys.get("test-key").unwrap();
    assert_eq!(stored_key.name, "Test Key");
    assert_eq!(stored_key.rate_limit, 60);
    assert!(stored_key.enabled);
}

#[test]
fn test_rate_limit_window_expiry() {
    let now = Instant::now();
    let expired_start = now - Duration::from_secs(120); // 2 minutes ago

    let bucket = RateLimitBucket {
        requests: 5,
        window_start: expired_start,
        limit: 10,
    };

    // Check if window has expired (should be reset)
    let window_duration = Duration::from_secs(60);
    let should_reset = now.duration_since(bucket.window_start) >= window_duration;

    assert!(should_reset);
}

#[test]
fn test_multiple_api_key_extraction_priority() {
    // Test that authorization header takes priority over x-api-key
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer priority-key"),
    );
    headers.insert("x-api-key", HeaderValue::from_static("fallback-key"));

    let api_key = extract_api_key(&headers);
    assert_eq!(api_key, Some("priority-key".to_string()));
}
