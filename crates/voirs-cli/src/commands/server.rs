//! Server command implementation.

use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use voirs::{config::AppConfig, error::Result, types::{AudioFormat, QualityLevel, SynthesisConfig}, VoirsPipeline};

/// Server application state
#[derive(Clone)]
pub struct AppState {
    pipeline: Arc<VoirsPipeline>,
    config: AppConfig,
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

/// Server statistics
#[derive(Debug, Serialize)]
pub struct ServerStats {
    pub requests_total: u64,
    pub requests_successful: u64,
    pub requests_failed: u64,
    pub average_synthesis_time_ms: f64,
    pub total_audio_generated_seconds: f64,
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

impl From<voirs::VoirsError> for ApiError {
    fn from(err: voirs::VoirsError) -> Self {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
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
            .await?
    );
    
    // Create application state
    let state = AppState {
        pipeline,
        config: config.clone(),
    };
    
    // Build the application router
    let app = create_router(state);
    
    // Parse address
    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .map_err(|e| voirs::VoirsError::config_error(&format!("Invalid address: {}", e)))?;
    
    println!("Starting VoiRS server on http://{}", addr);
    println!("API endpoints:");
    println!("  POST /api/v1/synthesize - Synthesize text to speech");
    println!("  GET  /api/v1/voices     - List available voices");
    println!("  GET  /api/v1/health     - Health check");
    println!("  GET  /api/v1/stats      - Server statistics");
    println!("  GET  /docs              - API documentation");
    println!();
    
    // Start the server
    let listener = tokio::net::TcpListener::bind(&addr).await
        .map_err(|e| voirs::VoirsError::config_error(&format!("Failed to bind to {}: {}", addr, e)))?;
    
    axum::serve(listener, app).await
        .map_err(|e| voirs::VoirsError::config_error(&format!("Server error: {}", e)))?;
    
    Ok(())
}

/// Create the router with all routes
fn create_router(state: AppState) -> Router {
    Router::new()
        // API routes
        .route("/api/v1/synthesize", post(synthesize_handler))
        .route("/api/v1/voices", get(voices_handler))
        .route("/api/v1/health", get(health_handler))
        .route("/api/v1/stats", get(stats_handler))
        
        // Documentation routes
        .route("/docs", get(docs_handler))
        .route("/", get(root_handler))
        
        // Add state and middleware
        .with_state(state)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        )
}

/// Root handler - redirects to docs
async fn root_handler() -> impl IntoResponse {
    axum::response::Redirect::permanent("/docs")
}

/// Synthesize text to speech
async fn synthesize_handler(
    State(state): State<AppState>,
    Json(request): Json<SynthesisRequest>,
) -> std::result::Result<Json<SynthesisResponse>, ApiError> {
    // Validate request
    if request.text.trim().is_empty() {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "Text cannot be empty".to_string(),
        });
    }
    
    if request.text.len() > 10000 {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "Text too long (max 10000 characters)".to_string(),
        });
    }
    
    // Parse quality level
    let quality = match request.quality.as_deref() {
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
    let format = match request.format.as_deref() {
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
    if let Some(rate) = request.rate {
        if !(0.5..=2.0).contains(&rate) {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "Speaking rate must be between 0.5 and 2.0".to_string(),
            });
        }
    }
    
    if let Some(pitch) = request.pitch {
        if !(-12.0..=12.0).contains(&pitch) {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "Pitch shift must be between -12.0 and 12.0 semitones".to_string(),
            });
        }
    }
    
    if let Some(volume) = request.volume {
        if !(-20.0..=20.0).contains(&volume) {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "Volume gain must be between -20.0 and 20.0 dB".to_string(),
            });
        }
    }
    
    // Create synthesis config
    let synth_config = SynthesisConfig {
        speaking_rate: request.rate.unwrap_or(1.0),
        pitch_shift: request.pitch.unwrap_or(0.0),
        volume_gain: request.volume.unwrap_or(0.0),
        enable_enhancement: request.enhance.unwrap_or(false),
        quality,
        ..Default::default()
    };
    
    // Set voice if specified
    if let Some(voice_id) = &request.voice {
        if let Err(e) = state.pipeline.set_voice(voice_id).await {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: format!("Invalid voice '{}': {}", voice_id, e),
            });
        }
    }
    
    // Perform synthesis
    match state.pipeline.synthesize_with_config(&request.text, &synth_config).await {
        Ok(audio) => {
            // Convert audio to bytes
            let audio_bytes = audio.to_format(format)?;
            let audio_base64 = base64::encode(&audio_bytes);
            
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
        Err(e) => {
            Ok(Json(SynthesisResponse {
                success: false,
                error: Some(e.to_string()),
                audio_data: None,
                duration: None,
                format: format.to_string(),
                sample_rate: 0,
                channels: 0,
            }))
        }
    }
}

/// List available voices
async fn voices_handler(
    State(state): State<AppState>,
    Query(query): Query<VoicesQuery>,
) -> std::result::Result<Json<VoicesResponse>, ApiError> {
    // Get available voices (placeholder implementation)
    let mut voices = vec![
        VoiceInfo {
            id: "en-us-female-1".to_string(),
            name: "Emma (US English)".to_string(),
            language: "en-US".to_string(),
            gender: Some("female".to_string()),
            description: Some("Clear American English female voice".to_string()),
            is_installed: true,
        },
        VoiceInfo {
            id: "en-us-male-1".to_string(),
            name: "Michael (US English)".to_string(),
            language: "en-US".to_string(),
            gender: Some("male".to_string()),
            description: Some("Natural American English male voice".to_string()),
            is_installed: true,
        },
        VoiceInfo {
            id: "en-gb-female-1".to_string(),
            name: "Charlotte (UK English)".to_string(),
            language: "en-GB".to_string(),
            gender: Some("female".to_string()),
            description: Some("Elegant British English female voice".to_string()),
            is_installed: false,
        },
    ];
    
    // Apply filters
    if let Some(language) = &query.language {
        voices.retain(|v| v.language.to_lowercase().contains(&language.to_lowercase()));
    }
    
    if let Some(gender) = &query.gender {
        voices.retain(|v| v.gender.as_ref().map_or(false, |g| g.eq_ignore_ascii_case(gender)));
    }
    
    let total = voices.len();
    
    Ok(Json(VoicesResponse { voices, total }))
}

/// Health check endpoint
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    // TODO: Add actual uptime tracking
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // Placeholder
        pipeline_ready: true, // TODO: Check actual pipeline status
    })
}

/// Server statistics endpoint
async fn stats_handler(State(state): State<AppState>) -> Json<ServerStats> {
    // TODO: Implement actual statistics tracking
    Json(ServerStats {
        requests_total: 0,
        requests_successful: 0,
        requests_failed: 0,
        average_synthesis_time_ms: 0.0,
        total_audio_generated_seconds: 0.0,
    })
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
    
    (
        [(header::CONTENT_TYPE, "text/html")],
        docs,
    )
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