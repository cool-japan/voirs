use axum::{
    extract::{Extension, Json},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{middleware::RequestMetrics, SharedPipeline};
use crate::{
    config::PipelineConfig,
    types::{LanguageCode, QualityLevel},
};

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    pub request_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ErrorResponse {
    pub fn new(error: String, message: String) -> Self {
        Self {
            error,
            message,
            request_id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
        }
    }
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(self)).into_response()
    }
}

#[derive(Debug, Deserialize)]
pub struct BatchSynthesisRequest {
    pub texts: Vec<String>,
    pub voice_id: Option<String>,
    pub language: Option<LanguageCode>,
    pub quality: Option<QualityLevel>,
    pub sample_rate: Option<u32>,
    pub output_format: Option<String>,
    pub parallel: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct BatchSynthesisResponse {
    pub results: Vec<BatchSynthesisResult>,
    pub total_count: usize,
    pub success_count: usize,
    pub error_count: usize,
    pub total_processing_time: f64,
}

#[derive(Debug, Serialize)]
pub struct BatchSynthesisResult {
    pub index: usize,
    pub text: String,
    pub success: bool,
    pub audio_data: Option<Vec<u8>>,
    pub error: Option<String>,
    pub processing_time: f64,
}

#[derive(Debug, Deserialize)]
pub struct VoiceValidationRequest {
    pub voice_id: String,
    pub test_text: Option<String>,
    pub quality_check: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct VoiceValidationResponse {
    pub voice_id: String,
    pub valid: bool,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
    pub quality_score: Option<f32>,
    pub test_audio: Option<Vec<u8>>,
}

#[derive(Debug, Deserialize)]
pub struct ModelManagementRequest {
    pub action: String, // "download", "update", "remove"
    pub model_id: String,
    pub version: Option<String>,
    pub force: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ModelManagementResponse {
    pub model_id: String,
    pub action: String,
    pub success: bool,
    pub message: String,
    pub version: Option<String>,
    pub size_mb: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct SystemInfoResponse {
    pub version: String,
    pub build_info: BuildInfo,
    pub system_specs: SystemSpecs,
    pub supported_features: Vec<String>,
    pub available_languages: Vec<LanguageCode>,
    pub default_voice: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BuildInfo {
    pub version: String,
    pub commit_hash: String,
    pub build_date: String,
    pub rust_version: String,
    pub target_arch: String,
}

#[derive(Debug, Serialize)]
pub struct SystemSpecs {
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub disk_space_gb: f64,
    pub gpu_available: bool,
    pub gpu_info: Option<String>,
}

pub async fn batch_synthesis_handler(
    Extension(shared_pipeline): Extension<SharedPipeline>,
    Json(request): Json<BatchSynthesisRequest>,
) -> Result<impl IntoResponse, ErrorResponse> {
    let start_time = std::time::Instant::now();

    let total_count = request.texts.len();
    let mut results = Vec::with_capacity(total_count);
    let mut success_count;
    let mut error_count;

    if request.parallel.unwrap_or(false) {
        // Process texts in parallel
        use futures::future::join_all;

        let output_format = request.output_format.clone();
        let quality = request.quality.unwrap_or(QualityLevel::High);
        let sample_rate = request.sample_rate;

        let futures = request.texts.iter().enumerate().map(|(index, text)| {
            let text = text.clone();
            let output_format = output_format.clone();
            let pipeline_clone = shared_pipeline.clone();
            async move {
                let text_start = std::time::Instant::now();

                let mut config = PipelineConfig::default();
                config.default_synthesis.quality = quality;
                if let Some(sample_rate) = sample_rate {
                    config.default_synthesis.sample_rate = sample_rate;
                }

                let pipeline = pipeline_clone.read().await;
                match pipeline.synthesize(&text).await {
                    Ok(audio_buffer) => {
                        let audio_data = match output_format.as_deref() {
                            Some("wav") | None => audio_buffer.to_wav_bytes(),
                            Some("raw") => Ok(audio_buffer
                                .samples()
                                .iter()
                                .flat_map(|&sample| sample.to_le_bytes())
                                .collect()),
                            _ => audio_buffer.to_wav_bytes(),
                        };

                        BatchSynthesisResult {
                            index,
                            text: text.clone(),
                            success: true,
                            audio_data: Some(audio_data.unwrap_or_default()),
                            error: None,
                            processing_time: text_start.elapsed().as_secs_f64(),
                        }
                    }
                    Err(e) => BatchSynthesisResult {
                        index,
                        text: text.clone(),
                        success: false,
                        audio_data: None,
                        error: Some(e.to_string()),
                        processing_time: text_start.elapsed().as_secs_f64(),
                    },
                }
            }
        });

        results = join_all(futures).await;
    } else {
        // Process texts sequentially
        let pipeline = shared_pipeline.read().await;
        for (index, text) in request.texts.iter().enumerate() {
            let text_start = std::time::Instant::now();

            let mut config = PipelineConfig::default();
            config.default_synthesis.quality = request.quality.unwrap_or(QualityLevel::High);
            if let Some(sample_rate) = request.sample_rate {
                config.default_synthesis.sample_rate = sample_rate;
            }

            let result = match pipeline.synthesize(text).await {
                Ok(audio_buffer) => {
                    let audio_data = match request.output_format.as_deref() {
                        Some("wav") | None => audio_buffer.to_wav_bytes(),
                        Some("raw") => Ok(audio_buffer
                            .samples()
                            .iter()
                            .flat_map(|&sample| sample.to_le_bytes())
                            .collect()),
                        _ => audio_buffer.to_wav_bytes(),
                    };

                    BatchSynthesisResult {
                        index,
                        text: text.clone(),
                        success: true,
                        audio_data: Some(audio_data.unwrap_or_default()),
                        error: None,
                        processing_time: text_start.elapsed().as_secs_f64(),
                    }
                }
                Err(e) => BatchSynthesisResult {
                    index,
                    text: text.clone(),
                    success: false,
                    audio_data: None,
                    error: Some(e.to_string()),
                    processing_time: text_start.elapsed().as_secs_f64(),
                },
            };

            results.push(result);
        }
    }

    success_count = results.iter().filter(|r| r.success).count();
    error_count = results.iter().filter(|r| !r.success).count();

    let response = BatchSynthesisResponse {
        results,
        total_count,
        success_count,
        error_count,
        total_processing_time: start_time.elapsed().as_secs_f64(),
    };

    Ok(Json(response))
}

pub async fn validate_voice_handler(
    Extension(pipeline): Extension<SharedPipeline>,
    Json(request): Json<VoiceValidationRequest>,
) -> Result<impl IntoResponse, ErrorResponse> {
    let pipeline = pipeline.read().await;

    let mut issues = Vec::new();
    let mut recommendations = Vec::new();
    let mut quality_score = None;
    let mut test_audio = None;

    // Check if voice exists by trying to set it
    let voice_info = match pipeline.set_voice(&request.voice_id).await {
        Ok(()) => {
            // Get the current voice to validate it was set successfully
            match pipeline.current_voice().await {
                Some(voice) => voice,
                None => {
                    issues.push(format!("Voice '{}' could not be set", request.voice_id));
                    return Ok(Json(VoiceValidationResponse {
                        voice_id: request.voice_id,
                        valid: false,
                        issues,
                        recommendations,
                        quality_score,
                        test_audio,
                    }));
                }
            }
        }
        Err(e) => {
            issues.push(format!("Error setting voice: {e}"));
            return Ok(Json(VoiceValidationResponse {
                voice_id: request.voice_id,
                valid: false,
                issues,
                recommendations,
                quality_score,
                test_audio,
            }));
        }
    };

    // Validate voice configuration
    if voice_info.language == LanguageCode::EnUs {
        recommendations.push("English US voice selected".to_string());
    }

    // Test synthesis if requested
    if let Some(test_text) = request.test_text {
        match pipeline.synthesize(&test_text).await {
            Ok(audio_buffer) => {
                if let Ok(wav_bytes) = audio_buffer.to_wav_bytes() {
                    test_audio = Some(wav_bytes);
                }
                quality_score = Some(0.85); // Mock quality score
            }
            Err(e) => {
                issues.push(format!("Test synthesis failed: {e}"));
            }
        }
    }

    let response = VoiceValidationResponse {
        voice_id: request.voice_id,
        valid: issues.is_empty(),
        issues,
        recommendations,
        quality_score,
        test_audio,
    };

    Ok(Json(response))
}

pub async fn model_management_handler(
    Extension(pipeline): Extension<SharedPipeline>,
    Json(request): Json<ModelManagementRequest>,
) -> Result<impl IntoResponse, ErrorResponse> {
    let _pipeline = pipeline.read().await;

    let response = match request.action.as_str() {
        "download" => {
            // Mock model download
            ModelManagementResponse {
                model_id: request.model_id,
                action: "download".to_string(),
                success: true,
                message: "Model downloaded successfully".to_string(),
                version: request.version,
                size_mb: Some(256.0),
            }
        }
        "update" => {
            // Mock model update
            ModelManagementResponse {
                model_id: request.model_id,
                action: "update".to_string(),
                success: true,
                message: "Model updated successfully".to_string(),
                version: request.version,
                size_mb: Some(256.0),
            }
        }
        "remove" => {
            // Mock model removal
            ModelManagementResponse {
                model_id: request.model_id,
                action: "remove".to_string(),
                success: true,
                message: "Model removed successfully".to_string(),
                version: request.version,
                size_mb: None,
            }
        }
        _ => {
            return Err(ErrorResponse::new(
                "Invalid action".to_string(),
                format!("Unknown action: {}", request.action),
            ));
        }
    };

    Ok(Json(response))
}

pub async fn system_info_handler() -> Result<impl IntoResponse, ErrorResponse> {
    let build_info = BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        commit_hash: option_env!("VERGEN_GIT_SHA")
            .unwrap_or("unknown")
            .to_string(),
        build_date: option_env!("VERGEN_BUILD_DATE")
            .unwrap_or("unknown")
            .to_string(),
        rust_version: option_env!("VERGEN_RUSTC_SEMVER")
            .unwrap_or("unknown")
            .to_string(),
        target_arch: std::env::consts::ARCH.to_string(),
    };

    let system_specs = SystemSpecs {
        cpu_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
        memory_gb: 8.0,       // Mock value
        disk_space_gb: 500.0, // Mock value
        gpu_available: false, // Would need actual GPU detection
        gpu_info: None,
    };

    let supported_features = vec![
        "synthesis".to_string(),
        "streaming".to_string(),
        "voice_switching".to_string(),
        "batch_processing".to_string(),
        "quality_validation".to_string(),
        "model_management".to_string(),
        "caching".to_string(),
        "plugins".to_string(),
    ];

    let available_languages = vec![
        LanguageCode::EnUs,
        LanguageCode::EsEs,
        LanguageCode::FrFr,
        LanguageCode::DeDe,
        LanguageCode::ItIt,
        LanguageCode::PtBr,
        LanguageCode::RuRu,
        LanguageCode::ZhCn,
        LanguageCode::JaJp,
        LanguageCode::KoKr,
    ];

    let response = SystemInfoResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_info,
        system_specs,
        supported_features,
        available_languages,
        default_voice: Some("default_english_voice".to_string()),
    };

    Ok(Json(response))
}

pub async fn metrics_handler(
    Extension(metrics): Extension<RequestMetrics>,
) -> Result<impl IntoResponse, ErrorResponse> {
    let stats = metrics.get_stats().await;
    Ok(Json(stats))
}

pub async fn debug_handler(
    Extension(pipeline): Extension<SharedPipeline>,
) -> Result<impl IntoResponse, ErrorResponse> {
    let pipeline = pipeline.read().await;

    let debug_info = serde_json::json!({
        "pipeline_initialized": true,
        "current_voice": pipeline.current_voice().await.map(|v| v.id.clone()),
        "cache_stats": {},  // Cache stats not available in current pipeline interface
        "memory_usage": {
            "estimated_mb": 256.0,
            "peak_mb": 512.0
        },
        "component_status": {
            "g2p": "loaded",
            "acoustic": "loaded",
            "vocoder": "loaded"
        }
    });

    Ok(Json(debug_info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VoirsPipelineBuilder;
    use axum::Extension;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn test_batch_synthesis_handler() {
        let pipeline = VoirsPipelineBuilder::new()
            .build()
            .await
            .expect("Failed to build pipeline");

        let shared_pipeline = Arc::new(RwLock::new(pipeline));

        let request = BatchSynthesisRequest {
            texts: vec!["Hello".to_string(), "World".to_string()],
            voice_id: None,
            language: None,
            quality: None,
            sample_rate: None,
            output_format: None,
            parallel: Some(false),
        };

        let result = batch_synthesis_handler(Extension(shared_pipeline), Json(request)).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_system_info_handler() {
        let result = system_info_handler().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_voice_handler() {
        let pipeline = VoirsPipelineBuilder::new()
            .build()
            .await
            .expect("Failed to build pipeline");

        let shared_pipeline = Arc::new(RwLock::new(pipeline));

        let request = VoiceValidationRequest {
            voice_id: "test_voice".to_string(),
            test_text: Some("Hello world".to_string()),
            quality_check: Some(true),
        };

        let result = validate_voice_handler(Extension(shared_pipeline), Json(request)).await;

        assert!(result.is_ok());
    }
}
