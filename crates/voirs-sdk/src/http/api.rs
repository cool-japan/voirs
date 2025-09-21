use axum::{
    extract::{Extension, Json, Path, Query},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::SharedPipeline;
use crate::{
    config::PipelineConfig,
    types::{LanguageCode, QualityLevel, VoiceConfig},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub request_id: String,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            request_id: Uuid::new_v4().to_string(),
        }
    }

    pub fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            request_id: Uuid::new_v4().to_string(),
        }
    }
}

impl<T> IntoResponse for ApiResponse<T>
where
    T: Serialize,
{
    fn into_response(self) -> Response {
        let status = if self.success {
            StatusCode::OK
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };

        (status, Json(self)).into_response()
    }
}

#[derive(Debug, Deserialize)]
pub struct SynthesisRequest {
    pub text: String,
    pub voice_id: Option<String>,
    pub language: Option<LanguageCode>,
    pub quality: Option<QualityLevel>,
    pub speed: Option<f32>,
    pub pitch: Option<f32>,
    pub volume: Option<f32>,
    pub sample_rate: Option<u32>,
    pub format: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SynthesisResponse {
    pub audio_data: Vec<u8>,
    pub metadata: AudioMetadata,
    pub processing_time: f64,
    pub cache_hit: bool,
}

#[derive(Debug, Serialize)]
pub struct AudioMetadata {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration: f64,
    pub format: String,
    pub size: usize,
}

#[derive(Debug, Deserialize)]
pub struct VoiceQuery {
    pub language: Option<LanguageCode>,
    pub quality: Option<QualityLevel>,
    pub gender: Option<String>,
    pub style: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VoicesResponse {
    pub voices: Vec<VoiceConfig>,
    pub total_count: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime: f64,
    pub memory_usage: MemoryUsage,
    pub pipeline_status: PipelineStatus,
}

#[derive(Debug, Serialize)]
pub struct MemoryUsage {
    pub used_mb: f64,
    pub available_mb: f64,
    pub cache_size_mb: f64,
}

#[derive(Debug, Serialize)]
pub struct PipelineStatus {
    pub initialized: bool,
    pub current_voice: Option<String>,
    pub loaded_models: Vec<String>,
    pub active_streams: usize,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub total_syntheses: u64,
    pub total_duration: f64,
    pub cache_hit_rate: f64,
    pub average_processing_time: f64,
    pub active_connections: usize,
}

pub fn create_api_routes() -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/stats", get(get_stats))
        .route("/synthesize", post(synthesize))
        .route("/voices", get(get_voices))
        .route("/voices/:voice_id", get(get_voice))
        .route("/voice", put(switch_voice))
        .route("/config", get(get_config))
        .route("/config", put(update_config))
        .route("/cache/clear", delete(clear_cache))
        .route("/cache/stats", get(get_cache_stats))
}

async fn health_check(
    Extension(pipeline): Extension<SharedPipeline>,
) -> Result<impl IntoResponse, StatusCode> {
    let pipeline = pipeline.read().await;
    let uptime = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let memory_usage = MemoryUsage {
        used_mb: 0.0, // Would need actual memory tracking
        available_mb: 1024.0,
        cache_size_mb: 0.0,
    };

    let current_voice = pipeline.current_voice().await.map(|v| v.id.clone());
    let pipeline_status = PipelineStatus {
        initialized: true,
        current_voice,
        loaded_models: vec![
            "g2p".to_string(),
            "acoustic".to_string(),
            "vocoder".to_string(),
        ],
        active_streams: 0,
    };

    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime,
        memory_usage,
        pipeline_status,
    };

    Ok(ApiResponse::success(response))
}

async fn get_stats(
    Extension(pipeline): Extension<SharedPipeline>,
) -> Result<impl IntoResponse, StatusCode> {
    let _pipeline = pipeline.read().await;

    let response = StatsResponse {
        total_syntheses: 0,
        total_duration: 0.0,
        cache_hit_rate: 0.0,
        average_processing_time: 0.0,
        active_connections: 0,
    };

    Ok(ApiResponse::success(response))
}

async fn synthesize(
    Extension(pipeline): Extension<SharedPipeline>,
    Json(request): Json<SynthesisRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let start_time = std::time::Instant::now();

    let pipeline = pipeline.read().await;

    let mut config = PipelineConfig::default();

    if let Some(speed) = request.speed {
        config = config.speed(speed);
    }

    if let Some(pitch) = request.pitch {
        config = config.pitch(pitch);
    }

    if let Some(volume) = request.volume {
        config = config.volume(volume);
    }

    if let Some(sample_rate) = request.sample_rate {
        config = config.sample_rate(sample_rate);
    }

    match pipeline
        .synthesize_with_config(&request.text, &config.default_synthesis)
        .await
    {
        Ok(audio_buffer) => {
            let processing_time = start_time.elapsed().as_secs_f64();

            let audio_data = match request.format.as_deref() {
                Some("wav") | None => audio_buffer.to_wav_bytes(),
                Some("raw") => Ok(audio_buffer
                    .samples()
                    .iter()
                    .flat_map(|&sample| sample.to_le_bytes())
                    .collect()),
                _ => return Err(StatusCode::BAD_REQUEST),
            };

            let audio_bytes = audio_data.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let metadata = AudioMetadata {
                sample_rate: audio_buffer.sample_rate(),
                channels: audio_buffer.channels() as u16,
                duration: audio_buffer.duration() as f64,
                format: request.format.unwrap_or_else(|| "wav".to_string()),
                size: audio_bytes.len(),
            };

            let response = SynthesisResponse {
                audio_data: audio_bytes,
                metadata,
                processing_time,
                cache_hit: false, // Would need actual cache tracking
            };

            Ok(ApiResponse::success(response))
        }
        Err(e) => {
            tracing::error!("Synthesis failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_voices(
    Extension(pipeline): Extension<SharedPipeline>,
    Query(query): Query<VoiceQuery>,
) -> Result<impl IntoResponse, StatusCode> {
    let pipeline = pipeline.read().await;

    match pipeline.list_voices().await {
        Ok(mut voices) => {
            // Apply filters
            if let Some(language) = query.language {
                voices.retain(|v| v.language == language);
            }

            if let Some(quality) = query.quality {
                voices.retain(|v| v.characteristics.quality == quality);
            }

            if let Some(gender) = query.gender {
                voices.retain(|v| {
                    v.characteristics
                        .gender
                        .as_ref()
                        .map(|g| g.to_string().to_lowercase())
                        == Some(gender.to_lowercase())
                });
            }

            if let Some(style) = query.style {
                voices.retain(|v| {
                    v.characteristics.style.to_string().to_lowercase() == style.to_lowercase()
                });
            }

            let response = VoicesResponse {
                total_count: voices.len(),
                voices,
            };

            Ok(ApiResponse::success(response))
        }
        Err(e) => {
            tracing::error!("Failed to get voices: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_voice(
    Extension(pipeline): Extension<SharedPipeline>,
    Path(voice_id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let pipeline = pipeline.read().await;

    match pipeline.list_voices().await {
        Ok(voices) => {
            if let Some(voice) = voices.iter().find(|v| v.id == voice_id) {
                Ok(ApiResponse::success(voice.clone()))
            } else {
                Err(StatusCode::NOT_FOUND)
            }
        }
        Err(e) => {
            tracing::error!("Failed to get voice {}: {}", voice_id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn switch_voice(
    Extension(pipeline): Extension<SharedPipeline>,
    Json(request): Json<HashMap<String, String>>,
) -> Result<impl IntoResponse, StatusCode> {
    let voice_id = request.get("voice_id").ok_or(StatusCode::BAD_REQUEST)?;

    let pipeline = pipeline.write().await;

    match pipeline.set_voice(voice_id).await {
        Ok(()) => Ok(ApiResponse::success("Voice switched successfully")),
        Err(e) => {
            tracing::error!("Failed to switch voice: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_config(
    Extension(pipeline): Extension<SharedPipeline>,
) -> Result<impl IntoResponse, StatusCode> {
    let pipeline = pipeline.read().await;

    let config = pipeline.get_config().await;
    Ok(ApiResponse::success(config))
}

async fn update_config(
    Extension(pipeline): Extension<SharedPipeline>,
    Json(config): Json<PipelineConfig>,
) -> Result<impl IntoResponse, StatusCode> {
    let pipeline = pipeline.write().await;

    match pipeline.update_config(config).await {
        Ok(()) => Ok(ApiResponse::success("Configuration updated successfully")),
        Err(e) => {
            tracing::error!("Failed to update config: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn clear_cache(
    Extension(_pipeline): Extension<SharedPipeline>,
) -> Result<impl IntoResponse, StatusCode> {
    // Cache management is handled internally by the pipeline
    // Return success indicating cache clearing is not needed
    Ok(ApiResponse::success(
        "Cache management is handled automatically",
    ))
}

async fn get_cache_stats(
    Extension(_pipeline): Extension<SharedPipeline>,
) -> Result<impl IntoResponse, StatusCode> {
    // Provide basic cache stats - cache management is internal
    let stats = serde_json::json!({
        "cache_enabled": true,
        "status": "managed_internally",
        "message": "Cache statistics are managed internally by the pipeline"
    });
    Ok(ApiResponse::success(stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VoirsPipelineBuilder;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::json;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_check() {
        let pipeline = VoirsPipelineBuilder::new()
            .build()
            .await
            .expect("Failed to build pipeline");

        let shared_pipeline = std::sync::Arc::new(RwLock::new(pipeline));
        let app = create_api_routes().layer(Extension(shared_pipeline));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_synthesize_endpoint() {
        let pipeline = VoirsPipelineBuilder::new()
            .build()
            .await
            .expect("Failed to build pipeline");

        let shared_pipeline = std::sync::Arc::new(RwLock::new(pipeline));
        let app = create_api_routes().layer(Extension(shared_pipeline));

        let request_body = json!({
            "text": "Hello, world!",
            "format": "wav"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/synthesize")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_voices_endpoint() {
        let pipeline = VoirsPipelineBuilder::new()
            .build()
            .await
            .expect("Failed to build pipeline");

        let shared_pipeline = std::sync::Arc::new(RwLock::new(pipeline));
        let app = create_api_routes().layer(Extension(shared_pipeline));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/voices")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
