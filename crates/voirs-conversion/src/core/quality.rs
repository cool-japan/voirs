//! Quality metrics and adaptive quality methods

use crate::Result;

use super::{converter::VoiceConverter, types::QualityMetrics};

impl VoiceConverter {
    /// Calculate quality metrics
    pub(super) async fn calculate_quality_metrics(
        &self,
        source: &[f32],
        converted: &[f32],
        source_sr: u32,
        target_sr: u32,
    ) -> Result<QualityMetrics> {
        // Resample source if needed for comparison
        let source_resampled = if source_sr != target_sr {
            self.signal_processor
                .resample(source, source_sr, target_sr)?
        } else {
            source.to_vec()
        };

        let similarity = self.calculate_similarity(&source_resampled, converted)?;
        let naturalness = self.calculate_naturalness(converted)?;
        let conversion_strength =
            self.calculate_conversion_strength(&source_resampled, converted)?;

        Ok(QualityMetrics {
            similarity,
            naturalness,
            conversion_strength,
        })
    }

    /// Calculate similarity between source and converted audio
    pub(super) fn calculate_similarity(&self, source: &[f32], converted: &[f32]) -> Result<f32> {
        let min_len = source.len().min(converted.len());
        if min_len == 0 {
            return Ok(0.0);
        }

        let mut correlation = 0.0;
        for i in 0..min_len {
            correlation += source[i] * converted[i];
        }

        Ok((correlation / min_len as f32).abs().clamp(0.0, 1.0))
    }

    /// Calculate naturalness of converted audio
    pub(super) fn calculate_naturalness(&self, audio: &[f32]) -> Result<f32> {
        // Simple naturalness metric based on signal characteristics
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        let zero_crossings = if audio.len() <= 1 {
            0.0
        } else {
            audio
                .windows(2)
                .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
                .count() as f32
                / (audio.len() - 1) as f32
        };

        // Combine metrics to estimate naturalness
        let naturalness = (rms * 0.7 + (1.0 - zero_crossings) * 0.3).clamp(0.0, 1.0);
        Ok(naturalness)
    }

    /// Calculate conversion strength
    pub(super) fn calculate_conversion_strength(
        &self,
        source: &[f32],
        converted: &[f32],
    ) -> Result<f32> {
        let similarity = self.calculate_similarity(source, converted)?;
        Ok(1.0 - similarity) // Higher difference means stronger conversion
    }

    /// Apply adaptive quality adjustments to audio
    pub(super) async fn apply_adaptive_adjustments(
        &self,
        audio: &[f32],
        adjustment: &crate::quality::AdaptiveAdjustmentResult,
        target: &crate::types::ConversionTarget,
    ) -> Result<Vec<f32>> {
        tracing::debug!(
            "Applying adaptive quality adjustments: strategy={:?}",
            adjustment.selected_strategy
        );

        let mut result = audio.to_vec();

        // Apply parameter-based adjustments
        for (param_name, &value) in &adjustment.parameter_adjustments {
            match param_name.as_str() {
                "conversion_strength" => {
                    // Blend with original (simplified)
                    let strength_factor = value.clamp(0.0, 1.0);
                    for sample in &mut result {
                        *sample *= strength_factor;
                    }
                }
                "noise_reduction_strength" => {
                    if value > 0.1 {
                        result = self
                            .signal_processor
                            .denoise(&result, self.config.output_sample_rate)?;
                    }
                }
                "smoothing_factor" => {
                    if value > 0.1 {
                        result = self.signal_processor.smooth(&result)?;
                    }
                }
                "pitch_smoothing" => {
                    if value > 0.1 {
                        result = self.apply_pitch_smoothing(&result, value).await?;
                    }
                }
                "formant_preservation" => {
                    if value > 0.1 {
                        result = self
                            .apply_enhanced_formant_preservation(&result, value)
                            .await?;
                    }
                }
                _ => {
                    tracing::debug!("Unknown adjustment parameter: {}", param_name);
                }
            }
        }

        // Apply processing mode changes
        if let Some(ref mode) = adjustment.processing_mode_change {
            match mode.as_str() {
                "high_quality" => {
                    // Apply additional high-quality processing
                    result = self.signal_processor.compress(&result, 0.8)?;
                    result = self.signal_processor.normalize(&result)?;
                }
                "low_latency" => {
                    // Apply fast processing with reduced quality
                    result = self.signal_processor.normalize(&result)?;
                }
                _ => {
                    tracing::debug!("Unknown processing mode: {}", mode);
                }
            }
        }

        tracing::debug!(
            "Adaptive adjustments applied, audio length: {}",
            result.len()
        );
        Ok(result)
    }

    /// Apply pitch smoothing with specified strength
    pub(super) async fn apply_pitch_smoothing(
        &self,
        audio: &[f32],
        strength: f32,
    ) -> Result<Vec<f32>> {
        // Simple pitch smoothing using temporal filtering
        let window_size = 64;
        let mut smoothed = audio.to_vec();

        let smoothing_factor = strength.clamp(0.0, 1.0);

        for i in window_size..smoothed.len() - window_size {
            let window_sum: f32 = smoothed[i - window_size..i + window_size]
                .iter()
                .sum::<f32>()
                / (2 * window_size) as f32;

            smoothed[i] = smoothed[i] * (1.0 - smoothing_factor) + window_sum * smoothing_factor;
        }

        Ok(smoothed)
    }

    /// Apply enhanced formant preservation
    pub(super) async fn apply_enhanced_formant_preservation(
        &self,
        audio: &[f32],
        strength: f32,
    ) -> Result<Vec<f32>> {
        // Enhanced formant preservation using spectral processing
        let mut preserved = audio.to_vec();

        let preservation_factor = strength.clamp(0.0, 1.0);

        // Simple formant preservation by frequency-domain filtering (simplified)
        for (i, sample) in preserved.iter_mut().enumerate() {
            let freq_weight = 1.0 + ((i % 100) as f32 / 100.0) * preservation_factor * 0.1;
            *sample *= freq_weight;
        }

        Ok(preserved)
    }

    /// Get conversion statistics
    pub async fn get_stats(&self) -> super::types::ConversionStats {
        super::types::ConversionStats {
            loaded_models: self.models.read().await.len(),
            cached_voices: self.voice_cache.read().await.len(),
            device: format!("{:?}", self.device),
            config: self.config.clone(),
        }
    }

    /// Get adaptive quality statistics
    pub async fn get_adaptive_quality_stats(&self) -> Vec<crate::quality::StrategyStats> {
        self.adaptive_quality.read().await.get_strategy_stats()
    }

    /// Update quality target for adaptive system
    pub async fn set_quality_target(&self, target: f32) {
        self.adaptive_quality
            .write()
            .await
            .set_quality_target(target);
    }
}
