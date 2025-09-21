use axum::{
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::Response,
    Json,
};
use serde_json::json;
use std::{collections::HashMap, sync::Arc, time::Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Clone)]
pub struct RequestMetrics {
    pub total_requests: Arc<RwLock<u64>>,
    pub active_requests: Arc<RwLock<u64>>,
    pub error_count: Arc<RwLock<u64>>,
    pub request_durations: Arc<RwLock<Vec<f64>>>,
}

impl Default for RequestMetrics {
    fn default() -> Self {
        Self {
            total_requests: Arc::new(RwLock::new(0)),
            active_requests: Arc::new(RwLock::new(0)),
            error_count: Arc::new(RwLock::new(0)),
            request_durations: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl RequestMetrics {
    pub async fn record_request_start(&self) {
        let mut total = self.total_requests.write().await;
        *total += 1;

        let mut active = self.active_requests.write().await;
        *active += 1;
    }

    pub async fn record_request_end(&self, duration: f64, is_error: bool) {
        let mut active = self.active_requests.write().await;
        if *active > 0 {
            *active -= 1;
        }

        let mut durations = self.request_durations.write().await;
        durations.push(duration);

        // Keep only the last 1000 request durations
        if durations.len() > 1000 {
            durations.drain(0..500);
        }

        if is_error {
            let mut errors = self.error_count.write().await;
            *errors += 1;
        }
    }

    pub async fn get_stats(&self) -> serde_json::Value {
        let total = *self.total_requests.read().await;
        let active = *self.active_requests.read().await;
        let errors = *self.error_count.read().await;
        let durations = self.request_durations.read().await;

        let avg_duration = if durations.is_empty() {
            0.0
        } else {
            durations.iter().sum::<f64>() / durations.len() as f64
        };

        let mut sorted_durations = durations.clone();
        sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p95_duration = if sorted_durations.is_empty() {
            0.0
        } else {
            let index = (sorted_durations.len() as f64 * 0.95) as usize;
            sorted_durations.get(index).copied().unwrap_or(0.0)
        };

        json!({
            "total_requests": total,
            "active_requests": active,
            "error_count": errors,
            "error_rate": if total > 0 { errors as f64 / total as f64 } else { 0.0 },
            "average_duration_ms": avg_duration * 1000.0,
            "p95_duration_ms": p95_duration * 1000.0,
            "requests_per_second": if durations.len() > 0 { durations.len() as f64 / durations.iter().sum::<f64>() } else { 0.0 }
        })
    }
}

pub async fn metrics_middleware(
    State(metrics): State<RequestMetrics>,
    request: Request,
    next: Next,
) -> Response {
    let start_time = Instant::now();

    metrics.record_request_start().await;

    let response = next.run(request).await;

    let duration = start_time.elapsed().as_secs_f64();
    let is_error = response.status().is_server_error() || response.status().is_client_error();

    metrics.record_request_end(duration, is_error).await;

    response
}

pub async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    let request_id = Uuid::new_v4().to_string();

    request
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());

    let mut response = next.run(request).await;

    response
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());

    response
}

pub async fn security_headers_middleware(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;

    let headers = response.headers_mut();

    // Add security headers
    headers.insert(
        "X-Content-Type-Options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("X-Frame-Options", HeaderValue::from_static("DENY"));
    headers.insert(
        "X-XSS-Protection",
        HeaderValue::from_static("1; mode=block"),
    );
    headers.insert(
        "Referrer-Policy",
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );
    headers.insert(
        "Content-Security-Policy",
        HeaderValue::from_static("default-src 'self'"),
    );

    response
}

pub async fn rate_limiting_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Simple rate limiting based on IP (in a real implementation, use Redis or similar)
    let client_ip = headers
        .get("x-forwarded-for")
        .or_else(|| headers.get("x-real-ip"))
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");

    // For now, just log the IP and pass through
    // In a real implementation, you'd check against a rate limit store
    tracing::debug!("Request from IP: {}", client_ip);

    Ok(next.run(request).await)
}

pub async fn error_handler_middleware(
    request: Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let response = next.run(request).await;

    if response.status().is_server_error() {
        let error_response = json!({
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "status": response.status().as_u16()
        });

        return Err((response.status(), Json(error_response)));
    }

    Ok(response)
}

pub async fn logging_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start_time = Instant::now();

    tracing::info!("Request started: {} {}", method, uri);

    let response = next.run(request).await;

    let duration = start_time.elapsed();
    let status = response.status();

    tracing::info!(
        "Request completed: {} {} {} in {:?}",
        method,
        uri,
        status,
        duration
    );

    response
}

pub async fn health_check_middleware(request: Request, next: Next) -> Response {
    // For health checks, we might want to bypass some middleware
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }

    next.run(request).await
}

#[derive(Clone)]
pub struct ApiKeyValidator {
    pub valid_keys: Arc<RwLock<HashMap<String, ApiKeyInfo>>>,
}

#[derive(Clone)]
pub struct ApiKeyInfo {
    pub name: String,
    pub permissions: Vec<String>,
    pub rate_limit: Option<u32>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl ApiKeyValidator {
    pub fn new() -> Self {
        Self {
            valid_keys: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add_key(&self, key: String, info: ApiKeyInfo) {
        let mut keys = self.valid_keys.write().await;
        keys.insert(key, info);
    }

    pub async fn validate_key(&self, key: &str) -> Option<ApiKeyInfo> {
        let keys = self.valid_keys.read().await;
        keys.get(key).cloned()
    }
}

pub async fn api_key_middleware(
    State(validator): State<ApiKeyValidator>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let api_key = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|auth| {
            if auth.starts_with("Bearer ") {
                Some(auth.trim_start_matches("Bearer "))
            } else {
                None
            }
        })
        .or_else(|| headers.get("x-api-key").and_then(|h| h.to_str().ok()));

    if let Some(key) = api_key {
        if let Some(_key_info) = validator.validate_key(key).await {
            return Ok(next.run(request).await);
        }
    }

    let error_response = json!({
        "error": "Unauthorized",
        "message": "Valid API key required"
    });

    Err((StatusCode::UNAUTHORIZED, Json(error_response)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request, response::Response};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_request_metrics() {
        let metrics = RequestMetrics::default();

        metrics.record_request_start().await;
        assert_eq!(*metrics.total_requests.read().await, 1);
        assert_eq!(*metrics.active_requests.read().await, 1);

        metrics.record_request_end(0.1, false).await;
        assert_eq!(*metrics.active_requests.read().await, 0);
        assert_eq!(*metrics.error_count.read().await, 0);
    }

    #[tokio::test]
    async fn test_api_key_validator() {
        let validator = ApiKeyValidator::new();

        let key_info = ApiKeyInfo {
            name: "test-key".to_string(),
            permissions: vec!["read".to_string()],
            rate_limit: Some(100),
            created_at: chrono::Utc::now(),
        };

        validator
            .add_key("test-key-123".to_string(), key_info.clone())
            .await;

        let result = validator.validate_key("test-key-123").await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "test-key");

        let invalid_result = validator.validate_key("invalid-key").await;
        assert!(invalid_result.is_none());
    }

    #[tokio::test]
    async fn test_metrics_stats() {
        let metrics = RequestMetrics::default();

        metrics.record_request_start().await;
        metrics.record_request_end(0.1, false).await;

        metrics.record_request_start().await;
        metrics.record_request_end(0.2, true).await;

        let stats = metrics.get_stats().await;
        assert_eq!(stats["total_requests"], 2);
        assert_eq!(stats["error_count"], 1);
        assert_eq!(stats["error_rate"], 0.5);
    }
}
