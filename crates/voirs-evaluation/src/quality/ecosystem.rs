//! Ecosystem integration implementation for `QualityEvaluator`.

use crate::integration::{
    EcosystemConfig, EcosystemDataBridge, EcosystemEvaluator, EcosystemResults, PerformanceMetrics,
    RecommendationPriority, RecommendationType,
};
use crate::traits::{EvaluationResult, QualityEvaluator as QualityEvaluatorTrait};
use async_trait::async_trait;
use chrono;
use serde_json;
use std::collections::HashMap;
use std::time::Instant;
use voirs_sdk::AudioBuffer;

use super::evaluator::QualityEvaluator;

#[async_trait]
impl EcosystemEvaluator for QualityEvaluator {
    async fn initialize_with_ecosystem(
        &mut self,
        config: &EcosystemConfig,
    ) -> EvaluationResult<()> {
        // Update configuration based on ecosystem settings
        if let Some(global_config) = config.get_global_config("quality_evaluation").await {
            // Apply global quality evaluation settings
            if let crate::integration::ConfigValue::Object(settings) = global_config {
                if let Some(crate::integration::ConfigValue::Boolean(objective_metrics)) =
                    settings.get("objective_metrics")
                {
                    self.config.objective_metrics = *objective_metrics;
                }
                if let Some(crate::integration::ConfigValue::Boolean(perceptual_metrics)) =
                    settings.get("perceptual_metrics")
                {
                    self.config.perceptual_metrics = *perceptual_metrics;
                }
                if let Some(crate::integration::ConfigValue::Float(_sample_rate)) =
                    settings.get("sample_rate")
                {
                    // sample_rate field doesn't exist on QualityEvaluationConfig
                    // Would need to be added to the config struct
                }
            }
        }

        // Apply quality thresholds from ecosystem
        // quality_thresholds field doesn't exist on QualityEvaluationConfig
        // Would need to be added to the config struct
        // self.config.quality_thresholds = Some(config.quality_thresholds.clone());

        // Validate configuration compatibility
        config
            .validate_compatibility()
            .map_err(|e| voirs_sdk::VoirsError::ConfigError {
                field: "ecosystem".to_string(),
                message: format!("Ecosystem configuration validation failed: {}", e),
            })?;

        Ok(())
    }

    async fn process_with_ecosystem(
        &self,
        audio: &AudioBuffer,
        bridge: &mut EcosystemDataBridge,
    ) -> EvaluationResult<serde_json::Value> {
        let start_time = Instant::now();

        // Update processing state
        bridge.processing_state.current_stage = "quality_evaluation".to_string();
        bridge
            .processing_state
            .completed_stages
            .push("audio_preprocessing".to_string());

        // Add audio metadata
        let audio_meta = crate::integration::AudioMetadata {
            source_id: format!("quality_eval_{}", chrono::Utc::now().timestamp()),
            pipeline_stage: "quality_evaluation".to_string(),
            quality_metrics: HashMap::new(),
            timestamps: std::iter::once((
                "quality_evaluation_start".to_string(),
                std::time::SystemTime::now(),
            ))
            .collect(),
            metadata: std::iter::once(("sample_rate".to_string(), audio.sample_rate().to_string()))
                .collect(),
        };

        bridge
            .audio_metadata
            .insert("quality_evaluation".to_string(), audio_meta);

        // Perform quality evaluation
        let quality_result = self.evaluate_quality(audio, None, None).await?;

        // Update performance metrics
        let processing_time = start_time.elapsed();
        bridge
            .performance_metrics
            .stage_times
            .insert("quality_evaluation".to_string(), processing_time);

        // Estimate memory usage (in bytes)
        let estimated_memory = (audio.len() * std::mem::size_of::<f32>()) as u64;
        bridge
            .performance_metrics
            .memory_usage
            .insert("quality_evaluation".to_string(), estimated_memory);

        // Calculate throughput (samples per second)
        let throughput = audio.len() as f64 / processing_time.as_secs_f64();
        bridge
            .performance_metrics
            .throughput
            .insert("quality_evaluation".to_string(), throughput);

        // Update audio metadata with quality results
        if let Some(audio_meta) = bridge.audio_metadata.get_mut("quality_evaluation") {
            audio_meta.quality_metrics = quality_result.component_scores.clone();
            audio_meta.timestamps.insert(
                "quality_evaluation_end".to_string(),
                std::time::SystemTime::now(),
            );
        }

        // Store stage results
        bridge.processing_state.stage_results.insert(
            "quality_evaluation".to_string(),
            serde_json::to_value(&quality_result).expect("serialization should succeed"),
        );

        // Check if we should generate recommendations
        let mut recommendations = Vec::new();
        if quality_result.overall_score < 0.7 {
            recommendations.push(crate::integration::utils::create_recommendation(
                "voirs-acoustic",
                RecommendationType::QualityImprovement,
                "Low quality detected. Consider improving acoustic model parameters or sample rate.",
                RecommendationPriority::High,
            ));
        }

        if quality_result.confidence < 0.5 {
            recommendations.push(crate::integration::utils::create_recommendation(
                "voirs-evaluation",
                RecommendationType::ConfigurationAdjustment,
                "Low confidence in quality assessment. Consider using reference audio or adjusting evaluation parameters.",
                RecommendationPriority::Medium,
            ));
        }

        // Add recommendations to bridge
        if !recommendations.is_empty() {
            bridge.processing_state.options.insert(
                "quality_recommendations".to_string(),
                crate::integration::ConfigValue::Array(
                    recommendations
                        .into_iter()
                        .map(|rec| {
                            crate::integration::ConfigValue::Object(
                                std::iter::once((
                                    "description".to_string(),
                                    crate::integration::ConfigValue::String(rec.description),
                                ))
                                .collect(),
                            )
                        })
                        .collect(),
                ),
            );
        }

        // Update processing state
        bridge.processing_state.current_stage = "quality_evaluation_complete".to_string();
        bridge
            .processing_state
            .completed_stages
            .push("quality_evaluation".to_string());

        Ok(serde_json::to_value(&quality_result).expect("serialization should succeed"))
    }

    async fn get_ecosystem_results(&self) -> EvaluationResult<EcosystemResults> {
        let mut evaluation_results = HashMap::new();
        evaluation_results.insert(
            "supported_metrics".to_string(),
            serde_json::to_value(&self.supported_metrics).expect("serialization should succeed"),
        );
        evaluation_results.insert(
            "metadata".to_string(),
            serde_json::to_value(&self.metadata).expect("serialization should succeed"),
        );

        let mut quality_scores = HashMap::new();
        quality_scores.insert("evaluator_ready".to_string(), 1.0);
        quality_scores.insert(
            "supported_metrics_count".to_string(),
            self.supported_metrics.len() as f32,
        );

        let mut metadata = HashMap::new();
        metadata.insert(
            "evaluator_type".to_string(),
            serde_json::Value::String("quality".to_string()),
        );
        metadata.insert(
            "version".to_string(),
            serde_json::Value::String(self.metadata.version.clone()),
        );
        metadata.insert(
            "languages".to_string(),
            serde_json::to_value(&self.metadata.supported_languages)
                .expect("serialization should succeed"),
        );

        let processing_stats = PerformanceMetrics {
            stage_times: std::iter::once((
                "initialization".to_string(),
                std::time::Duration::from_millis(50),
            ))
            .collect(),
            memory_usage: std::iter::once((
                "base_memory".to_string(),
                1024 * 1024, // 1MB estimated base memory
            ))
            .collect(),
            throughput: std::iter::once((
                "estimated_throughput".to_string(),
                self.metadata.processing_speed as f64,
            ))
            .collect(),
            error_rates: HashMap::new(),
            cache_hit_rates: HashMap::new(),
        };

        let recommendations = vec![
            crate::integration::utils::create_recommendation(
                "voirs-dataset",
                RecommendationType::DataPreprocessingEnhancement,
                "Ensure consistent audio preprocessing for optimal quality evaluation results.",
                RecommendationPriority::Medium,
            ),
            crate::integration::utils::create_recommendation(
                "voirs-sdk",
                RecommendationType::PerformanceOptimization,
                "Consider using reference audio when available for more accurate quality assessment.",
                RecommendationPriority::Low,
            ),
        ];

        Ok(EcosystemResults {
            evaluation_results,
            quality_scores,
            metadata,
            processing_stats,
            recommendations,
        })
    }

    fn handle_ecosystem_error(&self, error: voirs_sdk::VoirsError) -> crate::EvaluationError {
        let context = crate::integration::ErrorContext {
            source_crate: "voirs-evaluation".to_string(),
            propagation_path: vec!["quality_evaluator".to_string()],
            context_data: std::iter::once((
                "evaluator_metadata".to_string(),
                serde_json::to_value(&self.metadata).expect("serialization should succeed"),
            ))
            .collect(),
            timestamp: std::time::SystemTime::now(),
            recovery_suggestions: vec![
                "Check audio buffer compatibility".to_string(),
                "Verify configuration parameters".to_string(),
                "Ensure sufficient system resources".to_string(),
            ],
        };

        crate::integration::utils::convert_error_with_context(error, &context)
    }
}
