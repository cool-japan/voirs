//! Robustness features: retries, validation, health checks

use crate::{
    fallback::{DegradationConfig, QualityThresholds},
    types::{ConversionRequest, ConversionResult, ConversionType, VoiceCharacteristics},
    Error, Result,
};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{error, info, warn};

use super::converter::VoiceConverter;

impl VoiceConverter {
    /// Apply graceful degradation when primary conversion fails
    pub(super) async fn apply_graceful_degradation(
        &self,
        request: &ConversionRequest,
        original_error: Error,
    ) -> Result<ConversionResult> {
        let mut degradation_controller = self.degradation_controller.lock().await;
        degradation_controller
            .handle_failure(request, original_error, &self.config)
            .await
    }

    /// Check if quality fallback is needed and apply if necessary
    pub(super) async fn check_and_apply_quality_fallback(
        &self,
        request: &ConversionRequest,
        result: &ConversionResult,
    ) -> Result<Option<ConversionResult>> {
        let mut degradation_controller = self.degradation_controller.lock().await;
        degradation_controller
            .handle_quality_degradation(request, result, &self.config)
            .await
    }

    /// Configure graceful degradation settings
    pub async fn configure_degradation(&self, config: DegradationConfig) {
        let mut degradation_controller = self.degradation_controller.lock().await;
        degradation_controller.configure(config);
    }

    /// Get graceful degradation performance statistics
    pub async fn get_degradation_stats(&self) -> crate::fallback::PerformanceTracker {
        let degradation_controller = self.degradation_controller.lock().await;
        degradation_controller.get_performance_stats().clone()
    }

    /// Update quality thresholds for degradation triggers
    pub async fn update_degradation_thresholds(&self, thresholds: QualityThresholds) {
        let mut degradation_controller = self.degradation_controller.lock().await;
        degradation_controller.update_quality_thresholds(thresholds);
    }

    /// Get current degradation quality thresholds
    pub async fn get_degradation_thresholds(&self) -> QualityThresholds {
        let degradation_controller = self.degradation_controller.lock().await;
        degradation_controller.get_quality_thresholds().clone()
    }

    /// Robust conversion wrapper with comprehensive error handling
    pub async fn convert_with_retries(
        &self,
        request: ConversionRequest,
        max_retries: u32,
    ) -> Result<ConversionResult> {
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match self.convert(request.clone()).await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Conversion succeeded after {} retry attempts", attempt);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    if attempt < max_retries {
                        warn!(
                            "Conversion attempt {} failed: {}, retrying...",
                            attempt + 1,
                            e
                        );
                        // Short delay before retry
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| Error::runtime("Max retries exceeded".to_string())))
    }

    /// Safe conversion that never panics and always returns some result
    pub async fn safe_convert(&self, request: ConversionRequest) -> ConversionResult {
        match self.convert(request.clone()).await {
            Ok(result) => result,
            Err(e) => {
                error!("Safe conversion failed: {}, returning empty result", e);

                // Return a minimal result with original audio
                let processing_time = std::time::Duration::from_millis(1);
                let mut result = ConversionResult::success(
                    request.id.clone(),
                    request.source_audio.clone(),
                    request.source_sample_rate,
                    processing_time,
                    request.conversion_type.clone(),
                );

                // Mark as a safe fallback
                result.success = false; // Indicate this is a fallback
                result
                    .quality_metrics
                    .insert("safe_fallback".to_string(), 1.0);
                result
                    .quality_metrics
                    .insert("overall_quality".to_string(), 0.3);

                result
            }
        }
    }

    /// Validate conversion request with detailed error reporting
    pub fn validate_request_detailed(&self, request: &ConversionRequest) -> Result<()> {
        // Basic validation
        request.validate()?;

        // Additional validation checks
        if request.source_audio.is_empty() {
            return Err(Error::validation("Source audio is empty".to_string()));
        }

        if request.source_sample_rate < 8000 || request.source_sample_rate > 48000 {
            return Err(Error::validation(format!(
                "Invalid sample rate: {}",
                request.source_sample_rate
            )));
        }

        // Check if audio contains only silence
        let max_amplitude = request
            .source_audio
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        if max_amplitude < 1e-6 {
            warn!(
                "Input audio appears to be silent (max amplitude: {})",
                max_amplitude
            );
        }

        // Check for extreme audio values that might cause issues
        let clipped_samples = request
            .source_audio
            .iter()
            .filter(|&&x| x.abs() > 0.99)
            .count();
        if clipped_samples > request.source_audio.len() / 10 {
            warn!(
                "Input audio has many clipped samples: {}/{}",
                clipped_samples,
                request.source_audio.len()
            );
        }

        // Check conversion type compatibility
        if request.realtime && !self.supports_realtime_conversion(&request.conversion_type) {
            return Err(Error::validation(format!(
                "Conversion type {:?} does not support real-time processing",
                request.conversion_type
            )));
        }

        Ok(())
    }

    /// Check if a conversion type supports real-time processing
    pub(super) fn supports_realtime_conversion(&self, conversion_type: &ConversionType) -> bool {
        match conversion_type {
            ConversionType::PitchShift => true,
            ConversionType::SpeedTransformation => true,
            ConversionType::GenderTransformation => true,
            ConversionType::AgeTransformation => false, // More complex processing
            ConversionType::SpeakerConversion => false, // Requires neural models
            ConversionType::VoiceMorphing => false,     // Complex blending
            ConversionType::EmotionalTransformation => false, // Complex analysis
            ConversionType::PassThrough => true,        // Fastest possible processing
            ConversionType::ZeroShotConversion => false, // Complex analysis required
            ConversionType::Custom(_) => false,         // Unknown complexity
        }
    }

    /// Health check for the voice converter
    pub async fn health_check(&self) -> Result<HashMap<String, String>> {
        let mut health_status = HashMap::new();

        // Check device availability
        health_status.insert("device".to_string(), format!("{:?}", self.device));

        // Check loaded models
        let model_count = self.models.read().await.len();
        health_status.insert("loaded_models".to_string(), model_count.to_string());

        // Check system resources (simplified)
        health_status.insert("cpu_usage".to_string(), "50%".to_string()); // Would be actual in real implementation
        health_status.insert("memory_usage".to_string(), "1024MB".to_string());

        // Check degradation controller status
        let degradation_stats = self.get_degradation_stats().await;
        health_status.insert(
            "degradation_success_rate".to_string(),
            format!(
                "{:.2}%",
                if degradation_stats.total_degradations > 0 {
                    (degradation_stats.successful_degradations as f64
                        / degradation_stats.total_degradations as f64)
                        * 100.0
                } else {
                    100.0
                }
            ),
        );

        // Test basic functionality
        let test_audio = vec![0.1, -0.1, 0.2, -0.2];
        let test_request = ConversionRequest::new(
            "health_check".to_string(),
            test_audio,
            16000,
            ConversionType::PitchShift,
            crate::types::ConversionTarget::new(VoiceCharacteristics::default()),
        );

        match self.validate_request_detailed(&test_request) {
            Ok(_) => health_status.insert("validation".to_string(), "OK".to_string()),
            Err(e) => health_status.insert("validation".to_string(), format!("ERROR: {e}")),
        };

        Ok(health_status)
    }
}
