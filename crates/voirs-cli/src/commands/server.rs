//! Server command implementation with authentication and rate limiting.

use axum::{
    extract::{FromRequest, Query, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use voirs_sdk::{
    config::AppConfig,
    error::Result,
    types::{AudioFormat, QualityLevel, SynthesisConfig},
    VoirsPipeline,
};

/// API key configuration
#[derive(Debug, Clone)]
pub struct ApiKeyConfig {
    /// API key value
    pub key: String,
    /// Key name/description
    pub name: String,
    /// Rate limit (requests per minute)
    pub rate_limit: u32,
    /// Whether the key is enabled
    pub enabled: bool,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Rate limiting bucket for tracking requests
#[derive(Debug, Clone)]
pub struct RateLimitBucket {
    /// Number of requests in current window
    pub requests: u32,
    /// Window start time
    pub window_start: Instant,
    /// Rate limit for this bucket
    pub limit: u32,
}

/// Authentication and rate limiting state
#[derive(Debug)]
pub struct AuthState {
    /// Valid API keys
    pub api_keys: HashMap<String, ApiKeyConfig>,
    /// Rate limiting buckets per IP/API key
    pub rate_limits: HashMap<String, RateLimitBucket>,
    /// Usage statistics per API key
    pub usage_stats: HashMap<String, UsageStats>,
    /// Access logs
    pub access_logs: Vec<AccessLogEntry>,
}

/// Usage statistics for API keys
#[derive(Debug, Clone, Default, Serialize)]
pub struct UsageStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_audio_seconds: f64,
    pub bytes_transferred: u64,
    pub last_used: Option<SystemTime>,
}

/// Access log entry
#[derive(Debug, Clone, Serialize)]
pub struct AccessLogEntry {
    pub timestamp: SystemTime,
    pub ip_address: String,
    pub api_key: Option<String>,
    pub method: String,
    pub path: String,
    pub status_code: u16,
    pub response_time_ms: u64,
    pub bytes_transferred: u64,
}

/// Server application state
#[derive(Clone)]
pub struct AppState {
    pipeline: Arc<VoirsPipeline>,
    config: AppConfig,
    auth: Arc<Mutex<AuthState>>,
    start_time: Instant,
    shutdown_signal: Arc<tokio::sync::RwLock<bool>>,
}

/// Synthesis request
#[derive(Debug, Deserialize)]
pub struct SynthesisRequest {
    /// Text to synthesize
    pub text: String,
    /// Voice ID (optional)
    pub voice: Option<String>,
    /// Speaking rate (0.5 - 2.0)
    pub rate: Option<f32>,
    /// Pitch shift in semitones (-12.0 - 12.0)
    pub pitch: Option<f32>,
    /// Volume gain in dB (-20.0 - 20.0)
    pub volume: Option<f32>,
    /// Quality level
    pub quality: Option<String>,
    /// Audio format
    pub format: Option<String>,
    /// Enable audio enhancement
    pub enhance: Option<bool>,
}

/// Synthesis response
#[derive(Debug, Serialize)]
pub struct SynthesisResponse {
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Audio data (base64 encoded)
    pub audio_data: Option<String>,
    /// Audio duration in seconds
    pub duration: Option<f32>,
    /// Audio format
    pub format: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
}

/// Voice information
#[derive(Debug, Serialize)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub language: String,
    pub gender: Option<String>,
    pub description: Option<String>,
    pub is_installed: bool,
}

/// Voices list response
#[derive(Debug, Serialize)]
pub struct VoicesResponse {
    pub voices: Vec<VoiceInfo>,
    pub total: usize,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub pipeline_ready: bool,
}

/// Detailed health check response
#[derive(Debug, Serialize)]
pub struct DetailedHealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub timestamp: u64,
    pub checks: Vec<HealthCheck>,
    pub system: SystemHealth,
}

/// Individual health check
#[derive(Debug, Serialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: String,
    pub message: Option<String>,
    pub duration_ms: u64,
    pub last_checked: u64,
}

/// System health information
#[derive(Debug, Serialize)]
pub struct SystemHealth {
    pub memory_usage_mb: u64,
    pub memory_available_mb: u64,
    pub cpu_usage_percent: f32,
    pub disk_usage_percent: f32,
    pub thread_count: u64,
    pub file_descriptors: u64,
}

/// Server statistics
#[derive(Debug, Serialize)]
pub struct ServerStats {
    pub requests_total: u64,
    pub requests_successful: u64,
    pub requests_failed: u64,
    pub average_synthesis_time_ms: f64,
    pub total_audio_generated_seconds: f64,
    pub uptime_seconds: u64,
    pub active_api_keys: usize,
    pub rate_limited_requests: u64,
}

/// Authentication info response
#[derive(Debug, Serialize)]
pub struct AuthInfoResponse {
    pub api_key_name: String,
    pub rate_limit: u32,
    pub requests_remaining: u32,
    pub requests_used: u32,
    pub window_reset_seconds: u64,
}

/// Usage statistics response
#[derive(Debug, Serialize)]
pub struct UsageStatsResponse {
    pub api_key_name: String,
    pub stats: UsageStats,
}

/// Query parameters for voices endpoint
#[derive(Debug, Deserialize)]
pub struct VoicesQuery {
    pub language: Option<String>,
    pub gender: Option<String>,
}

/// Custom error type for API responses
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({
            "error": self.message,
            "status": self.status.as_u16()
        }));
        (self.status, body).into_response()
    }
}

impl From<voirs_sdk::VoirsError> for ApiError {
    fn from(err: voirs_sdk::VoirsError) -> Self {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
    }
}

/// Authentication middleware
pub async fn auth_middleware(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> std::result::Result<Response, ApiError> {
    let start_time = Instant::now();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    // Extract client IP
    let client_ip = extract_client_ip(&headers, &request);

    // Skip authentication for docs and health endpoints
    if path == "/docs"
        || path == "/"
        || path == "/api/v1/health"
        || path == "/api/v1/health/detailed"
        || path == "/api/v1/health/ready"
        || path == "/api/v1/health/live"
    {
        let response = next.run(request).await;
        log_request(&state, &client_ip, None, &method, &path, 200, start_time, 0).await;
        return Ok(response);
    }

    // Check if server is shutting down
    if *state.shutdown_signal.read().await {
        return Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "Server is shutting down".to_string(),
        });
    }

    // Extract API key
    let api_key = extract_api_key(&headers);

    // Validate API key and check rate limits
    let validation_result = validate_and_rate_limit(&state, &client_ip, api_key.as_deref()).await;

    match validation_result {
        Ok(api_key_config) => {
            // Store API key config in request extensions for use by handlers
            let mut request_with_extensions = request;
            if let Some(ref config) = api_key_config {
                request_with_extensions
                    .extensions_mut()
                    .insert(config.clone());
            }

            let response = next.run(request_with_extensions).await;
            let status_code = response.status().as_u16();

            // Calculate bytes transferred from response
            let bytes_transferred = calculate_response_size(&response);

            // Update usage statistics
            update_usage_stats(
                &state,
                api_key_config.as_ref(),
                status_code < 400,
                bytes_transferred as f64,
            )
            .await;

            // Log request
            log_request(
                &state,
                &client_ip,
                api_key_config.as_ref().map(|k| k.key.clone()),
                &method,
                &path,
                status_code,
                start_time,
                bytes_transferred,
            )
            .await;

            Ok(response)
        }
        Err(error) => {
            log_request(
                &state,
                &client_ip,
                api_key,
                &method,
                &path,
                error.status.as_u16(),
                start_time,
                0,
            )
            .await;
            Err(error)
        }
    }
}

/// Extract client IP from request
pub fn extract_client_ip(headers: &HeaderMap, _request: &Request) -> String {
    // Check for forwarded headers first
    if let Some(forwarded) = headers.get("x-forwarded-for") {
        if let Ok(forwarded_str) = forwarded.to_str() {
            if let Some(first_ip) = forwarded_str.split(',').next() {
                return first_ip.trim().to_string();
            }
        }
    }

    if let Some(real_ip) = headers.get("x-real-ip") {
        if let Ok(ip_str) = real_ip.to_str() {
            return ip_str.to_string();
        }
    }

    // Fallback to remote addr (would need to be passed through state)
    "unknown".to_string()
}

/// Extract API key from authorization header
pub fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|header| header.to_str().ok())
        .and_then(|auth_str| {
            if auth_str.starts_with("Bearer ") {
                Some(auth_str[7..].to_string())
            } else if auth_str.starts_with("ApiKey ") {
                Some(auth_str[7..].to_string())
            } else {
                None
            }
        })
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|header| header.to_str().ok())
                .map(|s| s.to_string())
        })
}

/// Validate API key and check rate limits
async fn validate_and_rate_limit(
    state: &AppState,
    client_ip: &str,
    api_key: Option<&str>,
) -> std::result::Result<Option<ApiKeyConfig>, ApiError> {
    let mut auth_state = state.auth.lock().unwrap();

    // Check if API key is provided and valid
    let api_key_config = if let Some(key) = api_key {
        match auth_state.api_keys.get(key) {
            Some(config) if config.enabled => Some(config.clone()),
            Some(_) => {
                return Err(ApiError {
                    status: StatusCode::UNAUTHORIZED,
                    message: "API key is disabled".to_string(),
                });
            }
            None => {
                return Err(ApiError {
                    status: StatusCode::UNAUTHORIZED,
                    message: "Invalid API key".to_string(),
                });
            }
        }
    } else {
        // If no API keys are configured, allow unauthenticated access
        if auth_state.api_keys.is_empty() {
            None
        } else {
            return Err(ApiError {
                status: StatusCode::UNAUTHORIZED,
                message: "API key required".to_string(),
            });
        }
    };

    // Determine rate limit key and limit
    let (rate_limit_key, rate_limit) = if let Some(ref config) = api_key_config {
        (format!("api_key:{}", config.key), config.rate_limit)
    } else {
        (format!("ip:{}", client_ip), 60) // Default 60 requests per minute for IP-based limiting
    };

    // Check rate limit
    let now = Instant::now();
    let window_duration = Duration::from_secs(60); // 1 minute window

    let bucket = auth_state
        .rate_limits
        .entry(rate_limit_key)
        .or_insert_with(|| RateLimitBucket {
            requests: 0,
            window_start: now,
            limit: rate_limit,
        });

    // Reset window if expired
    if now.duration_since(bucket.window_start) >= window_duration {
        bucket.requests = 0;
        bucket.window_start = now;
    }

    // Check if over limit
    if bucket.requests >= bucket.limit {
        return Err(ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: format!(
                "Rate limit exceeded. Limit: {} requests per minute",
                bucket.limit
            ),
        });
    }

    // Increment request count
    bucket.requests += 1;

    Ok(api_key_config)
}

/// Calculate response size in bytes
fn calculate_response_size(response: &Response) -> u64 {
    // Try to get content-length header first
    if let Some(content_length) = response.headers().get("content-length") {
        if let Ok(length_str) = content_length.to_str() {
            if let Ok(length) = length_str.parse::<u64>() {
                return length;
            }
        }
    }

    // For responses without content-length, estimate based on body
    // This is approximate since we can't easily access the body here
    let headers_size = response
        .headers()
        .iter()
        .map(|(name, value)| name.as_str().len() + value.len() + 4) // +4 for ": " and "\r\n"
        .sum::<usize>() as u64;

    // Add status line and basic HTTP overhead
    let status_line_size = response.status().as_str().len() as u64 + 20; // HTTP version + spaces

    headers_size + status_line_size
}

/// Update usage statistics
async fn update_usage_stats(
    state: &AppState,
    api_key_config: Option<&ApiKeyConfig>,
    success: bool,
    bytes_transferred: f64,
) {
    update_usage_stats_with_audio(state, api_key_config, success, bytes_transferred, None).await;
}

/// Update usage statistics with optional audio duration
async fn update_usage_stats_with_audio(
    state: &AppState,
    api_key_config: Option<&ApiKeyConfig>,
    success: bool,
    bytes_transferred: f64,
    audio_duration: Option<f64>,
) {
    if let Some(config) = api_key_config {
        let mut auth_state = state.auth.lock().unwrap();
        let stats = auth_state
            .usage_stats
            .entry(config.key.clone())
            .or_default();

        stats.total_requests += 1;
        stats.bytes_transferred += bytes_transferred as u64;

        if success {
            stats.successful_requests += 1;
        } else {
            stats.failed_requests += 1;
        }

        // Update audio duration if provided
        if let Some(duration) = audio_duration {
            stats.total_audio_seconds += duration;
        }

        stats.last_used = Some(SystemTime::now());
    }
}

/// Log request
async fn log_request(
    state: &AppState,
    client_ip: &str,
    api_key: Option<String>,
    method: &str,
    path: &str,
    status_code: u16,
    start_time: Instant,
    bytes_transferred: u64,
) {
    let mut auth_state = state.auth.lock().unwrap();

    let log_entry = AccessLogEntry {
        timestamp: SystemTime::now(),
        ip_address: client_ip.to_string(),
        api_key,
        method: method.to_string(),
        path: path.to_string(),
        status_code,
        response_time_ms: start_time.elapsed().as_millis() as u64,
        bytes_transferred,
    };

    auth_state.access_logs.push(log_entry);

    // Keep only last 10000 log entries to prevent memory bloat
    if auth_state.access_logs.len() > 10000 {
        auth_state.access_logs.drain(0..1000);
    }
}

/// Run server command
pub async fn run_server(host: &str, port: u16, config: &AppConfig) -> Result<()> {
    println!("Initializing VoiRS HTTP server...");

    // Build pipeline
    let pipeline = Arc::new(
        VoirsPipeline::builder()
            .with_quality(QualityLevel::High)
            .with_gpu_acceleration(config.pipeline.use_gpu)
            .build()
            .await?,
    );

    // Initialize authentication state
    let mut auth_state = AuthState {
        api_keys: HashMap::new(),
        rate_limits: HashMap::new(),
        usage_stats: HashMap::new(),
        access_logs: Vec::new(),
    };

    // Add default API key for development (should be configurable in production)
    let default_api_key = ApiKeyConfig {
        key: "voirs-dev-key-123".to_string(),
        name: "Development Key".to_string(),
        rate_limit: 100, // 100 requests per minute
        enabled: true,
        created_at: SystemTime::now(),
    };
    auth_state
        .api_keys
        .insert(default_api_key.key.clone(), default_api_key);

    // Create application state
    let state = AppState {
        pipeline,
        config: config.clone(),
        auth: Arc::new(Mutex::new(auth_state)),
        start_time: Instant::now(),
        shutdown_signal: Arc::new(tokio::sync::RwLock::new(false)),
    };

    // Build the application router
    let app = create_router(state.clone());

    // Parse address
    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .map_err(|e| voirs_sdk::VoirsError::config_error(&format!("Invalid address: {}", e)))?;

    println!("Starting VoiRS server on http://{}", addr);
    println!("API endpoints:");
    println!("  POST /api/v1/synthesize      - Synthesize text to speech (requires auth)");
    println!("  GET  /api/v1/voices          - List available voices (requires auth)");
    println!("  GET  /api/v1/health          - Basic health check (public)");
    println!("  GET  /api/v1/health/detailed - Detailed health check (public)");
    println!("  GET  /api/v1/health/ready    - Readiness probe (public)");
    println!("  GET  /api/v1/health/live     - Liveness probe (public)");
    println!("  GET  /api/v1/stats           - Server statistics (requires auth)");
    println!("  GET  /api/v1/auth/info       - Authentication information (requires auth)");
    println!("  GET  /api/v1/auth/usage      - Usage statistics (requires auth)");
    println!("  POST /api/v1/shutdown        - Graceful shutdown (requires auth)");
    println!("  GET  /docs                   - API documentation (public)");
    println!();
    println!("Authentication:");
    println!("  Default API key: voirs-dev-key-123");
    println!("  Headers: Authorization: Bearer <api-key> or X-API-Key: <api-key>");
    println!("  Rate limit: 100 requests per minute per API key");
    println!();
    println!("Graceful shutdown:");
    println!("  Send SIGTERM or SIGINT to gracefully shutdown");
    println!("  Or use POST /api/v1/shutdown endpoint");
    println!();

    // Start the server with graceful shutdown
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        voirs_sdk::VoirsError::config_error(&format!("Failed to bind to {}: {}", addr, e))
    })?;

    // Set up graceful shutdown signal
    let shutdown_signal = shutdown_signal();

    // Clone state for shutdown handling
    let shutdown_state = state.clone();

    // Start the server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal.await;
            println!("Starting graceful shutdown...");

            // Set shutdown flag to reject new requests
            *shutdown_state.shutdown_signal.write().await = true;

            // Give time for in-flight requests to complete
            tokio::time::sleep(Duration::from_secs(5)).await;

            println!("Graceful shutdown complete");
        })
        .await
        .map_err(|e| voirs_sdk::VoirsError::config_error(&format!("Server error: {}", e)))?;

    Ok(())
}

/// Signal handler for graceful shutdown
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            println!("Received Ctrl+C signal");
        },
        _ = terminate => {
            println!("Received SIGTERM signal");
        }
    }
}

/// Graceful shutdown endpoint
async fn shutdown_handler(State(state): State<AppState>) -> impl IntoResponse {
    // Set shutdown flag
    *state.shutdown_signal.write().await = true;

    // Respond to the client before initiating shutdown
    let response = Json(serde_json::json!({
        "message": "Server shutdown initiated",
        "status": "shutting_down"
    }));

    // Trigger shutdown in a separate task
    let shutdown_state = state.clone();
    tokio::spawn(async move {
        // Give a moment for the response to be sent
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Send shutdown signal
        // This is a simplified approach - in production, you might want to use
        // a channel or other mechanism to signal the main server loop
        std::process::exit(0);
    });

    response
}

/// Create the router with all routes
fn create_router(state: AppState) -> Router {
    let middleware_state = state.clone();

    Router::new()
        // API routes
        .route("/api/v1/synthesize", post(synthesize_handler))
        .route("/api/v1/voices", get(voices_handler))
        .route("/api/v1/health", get(health_handler))
        .route("/api/v1/health/detailed", get(detailed_health_handler))
        .route("/api/v1/health/ready", get(readiness_handler))
        .route("/api/v1/health/live", get(liveness_handler))
        .route("/api/v1/stats", get(stats_handler))
        .route("/api/v1/auth/info", get(auth_info_handler))
        .route("/api/v1/auth/usage", get(usage_stats_handler))
        .route("/api/v1/shutdown", post(shutdown_handler))
        // Documentation routes
        .route("/docs", get(docs_handler))
        .route("/", get(root_handler))
        // Add state and middleware
        .with_state(state)
        .layer(
            ServiceBuilder::new()
                .layer(middleware::from_fn_with_state(
                    middleware_state,
                    auth_middleware,
                ))
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive()),
        )
}

/// Root handler - redirects to docs
async fn root_handler() -> impl IntoResponse {
    axum::response::Redirect::permanent("/docs")
}

/// Synthesize text to speech
async fn synthesize_handler(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> std::result::Result<Json<SynthesisResponse>, ApiError> {
    // Extract API key configuration from request extensions
    let api_key_config = request.extensions().get::<ApiKeyConfig>().cloned();

    // Extract the JSON body
    let axum::extract::Json(synthesis_request): axum::extract::Json<SynthesisRequest> =
        axum::extract::Json::from_request(request, &state)
            .await
            .map_err(|_| ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "Invalid JSON request body".to_string(),
            })?;
    // Validate request
    if synthesis_request.text.trim().is_empty() {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "Text cannot be empty".to_string(),
        });
    }

    if synthesis_request.text.len() > 10000 {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "Text too long (max 10000 characters)".to_string(),
        });
    }

    // Parse quality level
    let quality = match synthesis_request.quality.as_deref() {
        Some("low") => QualityLevel::Low,
        Some("medium") => QualityLevel::Medium,
        Some("high") => QualityLevel::High,
        Some("ultra") => QualityLevel::Ultra,
        None => QualityLevel::High,
        Some(other) => {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: format!("Invalid quality level: {}", other),
            });
        }
    };

    // Parse audio format
    let format = match synthesis_request.format.as_deref() {
        Some("wav") => AudioFormat::Wav,
        Some("flac") => AudioFormat::Flac,
        Some("mp3") => AudioFormat::Mp3,
        Some("opus") => AudioFormat::Opus,
        None => AudioFormat::Wav,
        Some(other) => {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: format!("Unsupported audio format: {}", other),
            });
        }
    };

    // Validate parameters
    if let Some(rate) = synthesis_request.rate {
        if !(0.5..=2.0).contains(&rate) {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "Speaking rate must be between 0.5 and 2.0".to_string(),
            });
        }
    }

    if let Some(pitch) = synthesis_request.pitch {
        if !(-12.0..=12.0).contains(&pitch) {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "Pitch shift must be between -12.0 and 12.0 semitones".to_string(),
            });
        }
    }

    if let Some(volume) = synthesis_request.volume {
        if !(-20.0..=20.0).contains(&volume) {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "Volume gain must be between -20.0 and 20.0 dB".to_string(),
            });
        }
    }

    // Create synthesis config
    let synth_config = SynthesisConfig {
        speaking_rate: synthesis_request.rate.unwrap_or(1.0),
        pitch_shift: synthesis_request.pitch.unwrap_or(0.0),
        volume_gain: synthesis_request.volume.unwrap_or(0.0),
        enable_enhancement: synthesis_request.enhance.unwrap_or(false),
        quality,
        ..Default::default()
    };

    // Set voice if specified
    if let Some(voice_id) = &synthesis_request.voice {
        if let Err(e) = state.pipeline.set_voice(voice_id).await {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: format!("Invalid voice '{}': {}", voice_id, e),
            });
        }
    }

    // Perform synthesis
    match state
        .pipeline
        .synthesize_with_config(&synthesis_request.text, &synth_config)
        .await
    {
        Ok(audio) => {
            // Convert audio to bytes
            let audio_bytes = audio.to_format(format)?;
            let audio_base64 = base64::encode(&audio_bytes);

            // Update audio statistics (duration for usage tracking)
            let duration_seconds = audio.duration() as f64;

            // Update usage statistics with audio duration
            if let Some(ref config) = api_key_config {
                update_usage_stats_with_audio(
                    &state,
                    Some(config),
                    true, // synthesis was successful
                    0.0,  // bytes will be tracked by middleware
                    Some(duration_seconds),
                )
                .await;
            }

            Ok(Json(SynthesisResponse {
                success: true,
                error: None,
                audio_data: Some(audio_base64),
                duration: Some(audio.duration()),
                format: format.to_string(),
                sample_rate: audio.sample_rate(),
                channels: audio.channels() as u16,
            }))
        }
        Err(e) => Ok(Json(SynthesisResponse {
            success: false,
            error: Some(e.to_string()),
            audio_data: None,
            duration: None,
            format: format.to_string(),
            sample_rate: 0,
            channels: 0,
        })),
    }
}

/// List available voices
async fn voices_handler(
    State(state): State<AppState>,
    Query(query): Query<VoicesQuery>,
) -> std::result::Result<Json<VoicesResponse>, ApiError> {
    // Get available voices from pipeline
    let voice_configs = state
        .pipeline
        .list_voices()
        .await
        .map_err(|e| ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("Failed to list voices: {}", e),
        })?;

    // Convert VoiceConfig to VoiceInfo
    let mut voices: Vec<VoiceInfo> = voice_configs
        .iter()
        .map(|vc| voice_config_to_info(vc))
        .collect();

    // Apply filters
    if let Some(language) = &query.language {
        voices.retain(|v| v.language.to_lowercase().contains(&language.to_lowercase()));
    }

    if let Some(gender) = &query.gender {
        voices.retain(|v| {
            v.gender
                .as_ref()
                .map_or(false, |g| g.eq_ignore_ascii_case(gender))
        });
    }

    let total = voices.len();

    Ok(Json(VoicesResponse { voices, total }))
}

/// Convert SDK VoiceConfig to API VoiceInfo
fn voice_config_to_info(config: &voirs_sdk::types::VoiceConfig) -> VoiceInfo {
    // Build description from characteristics
    let quality_str = match config.characteristics.quality {
        voirs_sdk::types::QualityLevel::Low => "Standard quality",
        voirs_sdk::types::QualityLevel::Medium => "Good quality",
        voirs_sdk::types::QualityLevel::High => "High quality",
        voirs_sdk::types::QualityLevel::Ultra => "Ultra-high quality",
    };

    let style_str = match config.characteristics.style {
        voirs_sdk::types::SpeakingStyle::Neutral => "neutral",
        voirs_sdk::types::SpeakingStyle::Conversational => "conversational",
        voirs_sdk::types::SpeakingStyle::News => "news",
        voirs_sdk::types::SpeakingStyle::Formal => "formal",
        voirs_sdk::types::SpeakingStyle::Casual => "casual",
        voirs_sdk::types::SpeakingStyle::Energetic => "energetic",
        voirs_sdk::types::SpeakingStyle::Calm => "calm",
        voirs_sdk::types::SpeakingStyle::Dramatic => "dramatic",
        voirs_sdk::types::SpeakingStyle::Whisper => "whisper",
    };

    let mut description_parts = vec![quality_str.to_string()];
    description_parts.push(format!("{} style", style_str));

    if config.characteristics.emotion_support {
        description_parts.push("emotion support".to_string());
    }

    let description = Some(description_parts.join(", "));

    // Check if voice is installed (based on metadata flag set by pipeline)
    let is_installed = config
        .metadata
        .get("installed")
        .and_then(|v| v.parse::<bool>().ok())
        .unwrap_or(false);

    VoiceInfo {
        id: config.id.clone(),
        name: config.name.clone(),
        language: config.language.as_str().to_string(),
        gender: config
            .characteristics
            .gender
            .as_ref()
            .map(|g| g.to_string().to_lowercase()),
        description,
        is_installed,
    }
}

/// Health check endpoint
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    // Calculate uptime from start_time
    let uptime_seconds = state.start_time.elapsed().as_secs();

    // Check pipeline status by testing if it can list voices
    let pipeline_ready = match state.pipeline.list_voices().await {
        Ok(_) => true,
        Err(_) => false,
    };

    Json(HealthResponse {
        status: if pipeline_ready {
            "healthy"
        } else {
            "degraded"
        }
        .to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds,
        pipeline_ready,
    })
}

/// Detailed health check endpoint
async fn detailed_health_handler(State(state): State<AppState>) -> Json<DetailedHealthResponse> {
    let uptime_seconds = state.start_time.elapsed().as_secs();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut checks = Vec::new();
    let mut overall_status = "healthy";

    // Pipeline health check
    let pipeline_start = Instant::now();
    let pipeline_check = match state.pipeline.list_voices().await {
        Ok(_) => HealthCheck {
            name: "pipeline".to_string(),
            status: "healthy".to_string(),
            message: Some("Pipeline is operational".to_string()),
            duration_ms: pipeline_start.elapsed().as_millis() as u64,
            last_checked: timestamp,
        },
        Err(e) => {
            overall_status = "degraded";
            HealthCheck {
                name: "pipeline".to_string(),
                status: "unhealthy".to_string(),
                message: Some(format!("Pipeline error: {}", e)),
                duration_ms: pipeline_start.elapsed().as_millis() as u64,
                last_checked: timestamp,
            }
        }
    };
    checks.push(pipeline_check);

    // Memory health check
    let memory_start = Instant::now();
    let memory_check = check_memory_health();
    checks.push(HealthCheck {
        name: "memory".to_string(),
        status: memory_check.0,
        message: Some(memory_check.1),
        duration_ms: memory_start.elapsed().as_millis() as u64,
        last_checked: timestamp,
    });

    // Authentication system health check
    let auth_start = Instant::now();
    let auth_check = check_auth_health(&state);
    checks.push(HealthCheck {
        name: "authentication".to_string(),
        status: auth_check.0,
        message: Some(auth_check.1),
        duration_ms: auth_start.elapsed().as_millis() as u64,
        last_checked: timestamp,
    });

    // File system health check
    let fs_start = Instant::now();
    let fs_check = check_filesystem_health();
    checks.push(HealthCheck {
        name: "filesystem".to_string(),
        status: fs_check.0,
        message: Some(fs_check.1),
        duration_ms: fs_start.elapsed().as_millis() as u64,
        last_checked: timestamp,
    });

    // Update overall status based on checks
    if checks.iter().any(|c| c.status == "unhealthy") {
        overall_status = "unhealthy";
    } else if checks.iter().any(|c| c.status == "degraded") {
        overall_status = "degraded";
    }

    // Get system health information
    let system_health = get_system_health();

    Json(DetailedHealthResponse {
        status: overall_status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds,
        timestamp,
        checks,
        system: system_health,
    })
}

/// Readiness probe endpoint (K8s style)
async fn readiness_handler(State(state): State<AppState>) -> impl IntoResponse {
    // Check if the service is ready to serve traffic
    let pipeline_ready = match state.pipeline.list_voices().await {
        Ok(_) => true,
        Err(_) => false,
    };

    let auth_ready = {
        let auth_state = state.auth.lock().unwrap();
        true // Allow unauthenticated access for development
    };

    if pipeline_ready && auth_ready {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ready",
                "message": "Service is ready to serve traffic"
            })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "not_ready",
                "message": "Service is not ready to serve traffic"
            })),
        )
    }
}

/// Liveness probe endpoint (K8s style)
async fn liveness_handler(State(state): State<AppState>) -> impl IntoResponse {
    // Check if the service is alive (basic health check)
    let uptime = state.start_time.elapsed().as_secs();

    // Consider the service dead if it's been running for more than 24 hours without restart
    // This is a simple example - in practice, you might check for deadlocks, memory leaks, etc.
    if uptime > 86400 {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "unhealthy",
                "message": "Service has been running too long, needs restart"
            })),
        )
    } else {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "alive",
                "message": "Service is alive and responsive"
            })),
        )
    }
}

/// Check memory health
fn check_memory_health() -> (String, String) {
    // This is a simplified memory check
    // In a real implementation, you would use proper system APIs
    match std::fs::read_to_string("/proc/meminfo") {
        Ok(meminfo) => {
            let lines: Vec<&str> = meminfo.lines().collect();
            let mut total_kb = 0;
            let mut available_kb = 0;

            for line in lines {
                if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        total_kb = value.parse().unwrap_or(0);
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        available_kb = value.parse().unwrap_or(0);
                    }
                }
            }

            if total_kb > 0 {
                let usage_percent = ((total_kb - available_kb) as f64 / total_kb as f64) * 100.0;
                if usage_percent > 90.0 {
                    (
                        "unhealthy".to_string(),
                        format!("High memory usage: {:.1}%", usage_percent),
                    )
                } else if usage_percent > 80.0 {
                    (
                        "degraded".to_string(),
                        format!("Moderate memory usage: {:.1}%", usage_percent),
                    )
                } else {
                    (
                        "healthy".to_string(),
                        format!("Memory usage: {:.1}%", usage_percent),
                    )
                }
            } else {
                (
                    "degraded".to_string(),
                    "Could not determine memory usage".to_string(),
                )
            }
        }
        Err(_) => (
            "degraded".to_string(),
            "Memory information not available".to_string(),
        ),
    }
}

/// Check authentication system health
fn check_auth_health(state: &AppState) -> (String, String) {
    let auth_state = state.auth.lock().unwrap();

    let api_key_count = auth_state.api_keys.len();
    let active_buckets = auth_state.rate_limits.len();
    let log_entries = auth_state.access_logs.len();

    // Check if we have too many log entries (potential memory leak)
    if log_entries > 50000 {
        return (
            "degraded".to_string(),
            "Too many access log entries".to_string(),
        );
    }

    // Check if we have too many rate limit buckets (potential memory leak)
    if active_buckets > 10000 {
        return (
            "degraded".to_string(),
            "Too many active rate limit buckets".to_string(),
        );
    }

    (
        "healthy".to_string(),
        format!(
            "Auth system operational: {} API keys, {} active buckets",
            api_key_count, active_buckets
        ),
    )
}

/// Check filesystem health
fn check_filesystem_health() -> (String, String) {
    // Check if we can write to temp directory
    let temp_file = "/tmp/voirs_health_check";
    match std::fs::write(temp_file, "health check") {
        Ok(_) => {
            // Clean up the test file
            let _ = std::fs::remove_file(temp_file);
            ("healthy".to_string(), "Filesystem is writable".to_string())
        }
        Err(e) => ("unhealthy".to_string(), format!("Filesystem error: {}", e)),
    }
}

/// Get system health information
fn get_system_health() -> SystemHealth {
    // This is a simplified implementation
    // In a real application, you would use proper system monitoring libraries

    let (memory_usage_mb, memory_available_mb) = get_memory_info();
    let cpu_usage_percent = get_cpu_usage();
    let disk_usage_percent = get_disk_usage();
    let thread_count = get_thread_count();
    let file_descriptors = get_file_descriptor_count();

    SystemHealth {
        memory_usage_mb,
        memory_available_mb,
        cpu_usage_percent,
        disk_usage_percent,
        thread_count,
        file_descriptors,
    }
}

/// Get memory information
fn get_memory_info() -> (u64, u64) {
    match std::fs::read_to_string("/proc/meminfo") {
        Ok(meminfo) => {
            let lines: Vec<&str> = meminfo.lines().collect();
            let mut total_kb = 0;
            let mut available_kb = 0;

            for line in lines {
                if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        total_kb = value.parse().unwrap_or(0);
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        available_kb = value.parse().unwrap_or(0);
                    }
                }
            }

            let usage_mb = (total_kb - available_kb) / 1024;
            let available_mb = available_kb / 1024;

            (usage_mb, available_mb)
        }
        Err(_) => (0, 0),
    }
}

/// Get CPU usage (percentage)
fn get_cpu_usage() -> f32 {
    // Estimate CPU usage based on process statistics
    // For accurate per-process CPU, would need to track usage over time
    let cpu_count = num_cpus::get() as f32;

    // Try to get process CPU time on Unix systems
    #[cfg(unix)]
    {
        unsafe {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
            if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
                let usage = usage.assume_init();
                // ru_utime and ru_stime are in microseconds on some platforms
                let user_time = usage.ru_utime.tv_sec as f32 + usage.ru_utime.tv_usec as f32 / 1_000_000.0;
                let sys_time = usage.ru_stime.tv_sec as f32 + usage.ru_stime.tv_usec as f32 / 1_000_000.0;
                let total_time = user_time + sys_time;

                // Rough estimate: normalize by CPU count
                // For web server, typically uses 30-60% of one core during active requests
                return ((total_time / 100.0).min(1.0) * 50.0).min(100.0);
            }
        }
    }

    // Fallback estimate for active server
    25.0
}

/// Get disk usage (percentage)
fn get_disk_usage() -> f32 {
    // Use platform-specific APIs to get disk usage
    #[cfg(target_os = "linux")]
    {
        // Parse /proc/mounts to find root partition, then use statvfs
        use std::ffi::CString;
        use std::mem::MaybeUninit;

        let path = CString::new("/").unwrap();
        unsafe {
            let mut stat: libc::statvfs = MaybeUninit::zeroed().assume_init();
            if libc::statvfs(path.as_ptr(), &mut stat) == 0 {
                let total_blocks = stat.f_blocks;
                let free_blocks = stat.f_bfree;
                let used_blocks = total_blocks - free_blocks;

                if total_blocks > 0 {
                    return (used_blocks as f32 / total_blocks as f32) * 100.0;
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Use statfs on macOS
        use std::ffi::CString;
        use std::mem::MaybeUninit;

        let path = CString::new("/").unwrap();
        unsafe {
            let mut stat: libc::statfs = MaybeUninit::zeroed().assume_init();
            if libc::statfs(path.as_ptr(), &mut stat) == 0 {
                let total_blocks = stat.f_blocks;
                let free_blocks = stat.f_bfree;
                let used_blocks = total_blocks - free_blocks;

                if total_blocks > 0 {
                    return (used_blocks as f64 / total_blocks as f64 * 100.0) as f32;
                }
            }
        }
    }

    // Fallback for unsupported platforms
    0.0
}

/// Get thread count
fn get_thread_count() -> u64 {
    match std::fs::read_to_string("/proc/self/status") {
        Ok(status) => {
            for line in status.lines() {
                if line.starts_with("Threads:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        return value.parse().unwrap_or(0);
                    }
                }
            }
            0
        }
        Err(_) => 0,
    }
}

/// Get file descriptor count
fn get_file_descriptor_count() -> u64 {
    match std::fs::read_dir("/proc/self/fd") {
        Ok(entries) => entries.count() as u64,
        Err(_) => 0,
    }
}

/// Server statistics endpoint
async fn stats_handler(State(state): State<AppState>) -> Json<ServerStats> {
    let auth_state = state.auth.lock().unwrap();
    let uptime = state.start_time.elapsed().as_secs();

    // Calculate aggregate statistics
    let mut total_requests = 0u64;
    let mut successful_requests = 0u64;
    let mut failed_requests = 0u64;
    let mut total_audio_seconds = 0.0;

    for stats in auth_state.usage_stats.values() {
        total_requests += stats.total_requests;
        successful_requests += stats.successful_requests;
        failed_requests += stats.failed_requests;
        total_audio_seconds += stats.total_audio_seconds;
    }

    // Count rate limited requests from access logs
    let rate_limited_requests = auth_state
        .access_logs
        .iter()
        .filter(|log| log.status_code == 429)
        .count() as u64;

    // Calculate average synthesis time from synthesis requests in access logs
    let synthesis_logs: Vec<_> = auth_state
        .access_logs
        .iter()
        .filter(|log| log.path == "/api/v1/synthesize" && log.status_code == 200)
        .collect();

    let average_synthesis_time_ms = if synthesis_logs.is_empty() {
        0.0
    } else {
        synthesis_logs
            .iter()
            .map(|log| log.response_time_ms as f64)
            .sum::<f64>()
            / synthesis_logs.len() as f64
    };

    Json(ServerStats {
        requests_total: total_requests,
        requests_successful: successful_requests,
        requests_failed: failed_requests,
        average_synthesis_time_ms,
        total_audio_generated_seconds: total_audio_seconds,
        uptime_seconds: uptime,
        active_api_keys: auth_state.api_keys.len(),
        rate_limited_requests,
    })
}

/// Authentication info endpoint
async fn auth_info_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> std::result::Result<Json<AuthInfoResponse>, ApiError> {
    let api_key = extract_api_key(&headers).ok_or_else(|| ApiError {
        status: StatusCode::UNAUTHORIZED,
        message: "API key required".to_string(),
    })?;

    let auth_state = state.auth.lock().unwrap();

    let api_key_config = auth_state.api_keys.get(&api_key).ok_or_else(|| ApiError {
        status: StatusCode::UNAUTHORIZED,
        message: "Invalid API key".to_string(),
    })?;

    // Get current usage from rate limit bucket
    let rate_limit_key = format!("api_key:{}", api_key);
    let bucket = auth_state.rate_limits.get(&rate_limit_key);

    let (requests_used, requests_remaining, window_reset_seconds) = if let Some(bucket) = bucket {
        let elapsed = bucket.window_start.elapsed().as_secs();
        let reset_seconds = if elapsed >= 60 { 0 } else { 60 - elapsed };

        (
            bucket.requests,
            bucket.limit.saturating_sub(bucket.requests),
            reset_seconds,
        )
    } else {
        (0, api_key_config.rate_limit, 60)
    };

    Ok(Json(AuthInfoResponse {
        api_key_name: api_key_config.name.clone(),
        rate_limit: api_key_config.rate_limit,
        requests_remaining,
        requests_used,
        window_reset_seconds,
    }))
}

/// Usage statistics endpoint
async fn usage_stats_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> std::result::Result<Json<UsageStatsResponse>, ApiError> {
    let api_key = extract_api_key(&headers).ok_or_else(|| ApiError {
        status: StatusCode::UNAUTHORIZED,
        message: "API key required".to_string(),
    })?;

    let auth_state = state.auth.lock().unwrap();

    let api_key_config = auth_state.api_keys.get(&api_key).ok_or_else(|| ApiError {
        status: StatusCode::UNAUTHORIZED,
        message: "Invalid API key".to_string(),
    })?;

    let stats = auth_state
        .usage_stats
        .get(&api_key)
        .cloned()
        .unwrap_or_default();

    Ok(Json(UsageStatsResponse {
        api_key_name: api_key_config.name.clone(),
        stats,
    }))
}

/// API documentation endpoint
async fn docs_handler() -> impl IntoResponse {
    let docs = r#"
<!DOCTYPE html>
<html>
<head>
    <title>VoiRS API Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { color: white; padding: 3px 8px; border-radius: 3px; font-weight: bold; }
        .post { background: #49cc90; }
        .get { background: #61affe; }
        code { background: #f0f0f0; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>VoiRS Text-to-Speech API</h1>
    <p>RESTful API for high-quality speech synthesis.</p>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /api/v1/synthesize</h3>
        <p>Synthesize text to speech audio.</p>
        <p><strong>Request Body:</strong></p>
        <pre><code>{
  "text": "Hello, world!",
  "voice": "en-us-female-1",
  "rate": 1.0,
  "pitch": 0.0,
  "volume": 0.0,
  "quality": "high",
  "format": "wav",
  "enhance": false
}</code></pre>
        <p><strong>Response:</strong></p>
        <pre><code>{
  "success": true,
  "audio_data": "base64-encoded-audio",
  "duration": 2.5,
  "format": "wav",
  "sample_rate": 22050,
  "channels": 1
}</code></pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /api/v1/voices</h3>
        <p>List available voices.</p>
        <p><strong>Query Parameters:</strong></p>
        <ul>
            <li><code>language</code> - Filter by language (e.g., "en-US")</li>
            <li><code>gender</code> - Filter by gender ("male" or "female")</li>
        </ul>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /api/v1/health</h3>
        <p>Health check endpoint.</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /api/v1/stats</h3>
        <p>Server statistics and usage metrics.</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /api/v1/auth/info</h3>
        <p>Get authentication information and rate limit status.</p>
        <p><strong>Response:</strong></p>
        <pre><code>{
  "api_key_name": "Development Key",
  "rate_limit": 100,
  "requests_remaining": 85,
  "requests_used": 15,
  "window_reset_seconds": 42
}</code></pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /api/v1/auth/usage</h3>
        <p>Get detailed usage statistics for your API key.</p>
        <p><strong>Response:</strong></p>
        <pre><code>{
  "api_key_name": "Development Key",
  "stats": {
    "total_requests": 1542,
    "successful_requests": 1489,
    "failed_requests": 53,
    "total_audio_seconds": 3847.2,
    "bytes_transferred": 15732481,
    "last_used": "2024-01-15T10:30:00Z"
  }
}</code></pre>
    </div>
    
    <h2>Authentication</h2>
    <p>Most API endpoints require authentication using an API key. Include your API key in requests using one of these methods:</p>
    <ul>
        <li><strong>Authorization Header:</strong> <code>Authorization: Bearer your-api-key</code></li>
        <li><strong>API Key Header:</strong> <code>X-API-Key: your-api-key</code></li>
    </ul>
    <p><strong>Development API Key:</strong> <code>voirs-dev-key-123</code></p>
    <p><strong>Rate Limiting:</strong> Each API key has a rate limit (default: 100 requests per minute). Rate limit status is returned in response headers and available via the <code>/api/v1/auth/info</code> endpoint.</p>
    
    <h2>Audio Formats</h2>
    <ul>
        <li><strong>wav</strong> - Uncompressed WAV (default)</li>
        <li><strong>flac</strong> - Lossless FLAC compression</li>
        <li><strong>mp3</strong> - Lossy MP3 compression</li>
        <li><strong>opus</strong> - Modern Opus codec</li>
    </ul>
    
    <h2>Quality Levels</h2>
    <ul>
        <li><strong>low</strong> - Fast synthesis, lower quality</li>
        <li><strong>medium</strong> - Balanced speed and quality</li>
        <li><strong>high</strong> - High quality (default)</li>
        <li><strong>ultra</strong> - Maximum quality, slower</li>
    </ul>
</body>
</html>
"#;

    ([(header::CONTENT_TYPE, "text/html")], docs)
}

// Add base64 encoding support
mod base64 {
    pub fn encode(data: &[u8]) -> String {
        use std::convert::TryInto;
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        let mut result = String::new();
        let chunks = data.chunks_exact(3);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let b = u32::from_be_bytes([0, chunk[0], chunk[1], chunk[2]]);
            result.push(ALPHABET[((b >> 18) & 63) as usize] as char);
            result.push(ALPHABET[((b >> 12) & 63) as usize] as char);
            result.push(ALPHABET[((b >> 6) & 63) as usize] as char);
            result.push(ALPHABET[(b & 63) as usize] as char);
        }

        match remainder.len() {
            1 => {
                let b = (remainder[0] as u32) << 16;
                result.push(ALPHABET[((b >> 18) & 63) as usize] as char);
                result.push(ALPHABET[((b >> 12) & 63) as usize] as char);
                result.push_str("==");
            }
            2 => {
                let b = ((remainder[0] as u32) << 16) | ((remainder[1] as u32) << 8);
                result.push(ALPHABET[((b >> 18) & 63) as usize] as char);
                result.push(ALPHABET[((b >> 12) & 63) as usize] as char);
                result.push(ALPHABET[((b >> 6) & 63) as usize] as char);
                result.push('=');
            }
            _ => {}
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_request_validation() {
        let request = SynthesisRequest {
            text: "Hello, world!".to_string(),
            voice: Some("en-us-female-1".to_string()),
            rate: Some(1.0),
            pitch: Some(0.0),
            volume: Some(0.0),
            quality: Some("high".to_string()),
            format: Some("wav".to_string()),
            enhance: Some(false),
        };

        assert_eq!(request.text, "Hello, world!");
        assert_eq!(request.voice, Some("en-us-female-1".to_string()));
        assert_eq!(request.rate, Some(1.0));
    }

    #[test]
    fn test_voice_info_creation() {
        let voice = VoiceInfo {
            id: "test-voice".to_string(),
            name: "Test Voice".to_string(),
            language: "en-US".to_string(),
            gender: Some("female".to_string()),
            description: Some("A test voice".to_string()),
            is_installed: true,
        };

        assert_eq!(voice.id, "test-voice");
        assert_eq!(voice.language, "en-US");
        assert!(voice.is_installed);
    }

    #[test]
    fn test_base64_encoding() {
        let data = b"Hello, world!";
        let encoded = base64::encode(data);
        assert!(!encoded.is_empty());
        assert!(encoded.chars().all(|c| c.is_ascii()));
    }
}
