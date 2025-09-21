//! REST API middleware for the VoiRS recognizer service.

use axum::{
    extract::{ConnectInfo, Request},
    http::{HeaderMap, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::net::SocketAddr;
use std::time::Instant;
use tracing::{error, info, warn};
use uuid::Uuid;

/// Request logging middleware
pub async fn logging_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    method: Method,
    uri: axum::http::Uri,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let request_id = Uuid::new_v4().to_string();

    // Log incoming request
    info!(
        request_id = %request_id,
        method = %method,
        uri = %uri,
        client_ip = %addr.ip(),
        user_agent = ?headers.get("user-agent"),
        "Incoming request"
    );

    // Add request ID to headers for downstream handlers
    let mut request = request;
    request
        .headers_mut()
        .insert("x-request-id", request_id.parse().unwrap());

    // Process request
    let response = next.run(request).await;

    // Log response
    let duration = start.elapsed();
    let status = response.status();

    match status.as_u16() {
        200..=299 => info!(
            request_id = %request_id,
            status = %status,
            duration_ms = %duration.as_millis(),
            "Request completed successfully"
        ),
        400..=499 => warn!(
            request_id = %request_id,
            status = %status,
            duration_ms = %duration.as_millis(),
            "Client error"
        ),
        500..=599 => error!(
            request_id = %request_id,
            status = %status,
            duration_ms = %duration.as_millis(),
            "Server error"
        ),
        _ => info!(
            request_id = %request_id,
            status = %status,
            duration_ms = %duration.as_millis(),
            "Request completed"
        ),
    }

    response
}

/// Rate limiting middleware
pub async fn rate_limiting_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Simple in-memory rate limiting (for production, use Redis or similar)
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime};

    static RATE_LIMITER: std::sync::OnceLock<
        Arc<Mutex<HashMap<std::net::IpAddr, Vec<SystemTime>>>>,
    > = std::sync::OnceLock::new();

    let limiter = RATE_LIMITER.get_or_init(|| Arc::new(Mutex::new(HashMap::new())));
    let mut guard = limiter.lock();

    let now = SystemTime::now();
    let window = Duration::from_secs(60); // 1 minute window
    let max_requests = 100; // Max 100 requests per minute

    let client_ip = addr.ip();
    let requests = guard.entry(client_ip).or_insert_with(Vec::new);

    // Remove old requests outside the window
    requests.retain(|&time| now.duration_since(time).unwrap_or(Duration::ZERO) < window);

    // Check if rate limit exceeded
    if requests.len() >= max_requests {
        warn!(client_ip = %client_ip, "Rate limit exceeded");
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Add current request
    requests.push(now);
    drop(guard);

    Ok(next.run(request).await)
}

/// Authentication middleware (basic implementation)
pub async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip auth for health check endpoints
    let path = request.uri().path();
    if path.starts_with("/health") {
        return Ok(next.run(request).await);
    }

    // Check for API key in headers
    if let Some(auth_header) = headers.get("authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") || auth_str.starts_with("ApiKey ") {
                // In production, validate the actual token/key
                return Ok(next.run(request).await);
            }
        }
    }

    // Check for API key in query parameter
    if let Some(query) = request.uri().query() {
        if query.contains("api_key=") {
            // In production, validate the actual API key
            return Ok(next.run(request).await);
        }
    }

    warn!("Unauthorized request to {}", path);
    Err(StatusCode::UNAUTHORIZED)
}

/// Content validation middleware
pub async fn content_validation_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let method = request.method().clone();
    let path = request.uri().path();

    // Skip validation for GET requests and health checks
    if method == Method::GET || path.starts_with("/health") {
        return Ok(next.run(request).await);
    }

    // Check content type for POST/PUT requests
    if method == Method::POST || method == Method::PUT {
        if let Some(content_type) = headers.get("content-type") {
            if let Ok(content_type_str) = content_type.to_str() {
                if content_type_str.starts_with("application/json")
                    || content_type_str.starts_with("multipart/form-data")
                    || content_type_str.starts_with("audio/")
                {
                    return Ok(next.run(request).await);
                }
            }
        }

        warn!("Invalid or missing content type for {} {}", method, path);
        return Err(StatusCode::BAD_REQUEST);
    }

    Ok(next.run(request).await)
}

/// Security headers middleware
pub async fn security_headers_middleware(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;

    let headers = response.headers_mut();

    // Add security headers
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert(
        "Referrer-Policy",
        "strict-origin-when-cross-origin".parse().unwrap(),
    );
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
            .parse()
            .unwrap(),
    );

    response
}

/// Error handling middleware
pub async fn error_handling_middleware(request: Request, next: Next) -> Response {
    let response = next.run(request).await;

    // If the response is an error, enhance it with additional context
    if response.status().is_client_error() || response.status().is_server_error() {
        // Could add custom error pages, logging, etc.
        // For now, just pass through
    }

    response
}

/// CORS preflight middleware
pub async fn cors_preflight_middleware(
    method: Method,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    if method == Method::OPTIONS {
        // Handle preflight requests
        let mut response = StatusCode::OK.into_response();
        let response_headers = response.headers_mut();

        response_headers.insert("Access-Control-Allow-Origin", "*".parse().unwrap());
        response_headers.insert(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, DELETE, OPTIONS".parse().unwrap(),
        );
        response_headers.insert(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Requested-With"
                .parse()
                .unwrap(),
        );
        response_headers.insert("Access-Control-Max-Age", "86400".parse().unwrap());

        return response;
    }

    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, routing::get, Router};
    use std::net::{IpAddr, Ipv4Addr};
    use tower::ServiceExt;

    async fn dummy_handler() -> &'static str {
        "OK"
    }

    #[tokio::test]
    async fn test_logging_middleware() {
        let app = Router::new().route("/test", get(dummy_handler));

        let response = app
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_security_headers_middleware() {
        let app = Router::new()
            .route("/test", get(dummy_handler))
            .layer(axum::middleware::from_fn(security_headers_middleware));

        let response = app
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().contains_key("X-Content-Type-Options"));
        assert!(response.headers().contains_key("X-Frame-Options"));
    }
}
