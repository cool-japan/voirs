//! AcousticModel trait implementation for VitsModel
//!
//! This module contains the AcousticModel trait implementation for VITS,
//! providing the main synthesis interface.

use async_trait::async_trait;

use crate::{
    AcousticError, AcousticModel, AcousticModelFeature, AcousticModelMetadata, LanguageCode,
    MelSpectrogram, MemoryOptimizer, Phoneme, Result, SynthesisConfig,
};

use super::{utils::tensor_to_mel_spectrogram, VitsModel};

#[async_trait]
impl AcousticModel for VitsModel {
    async fn synthesize(
        &self,
        phonemes: &[Phoneme],
        config: Option<&SynthesisConfig>,
    ) -> Result<MelSpectrogram> {
        if phonemes.is_empty() {
            return Err(AcousticError::InputError {
                message: "Empty phoneme sequence".to_string(),
            });
        }

        tracing::info!("VITS: Starting synthesis for {} phonemes", phonemes.len());

        // Disable optimizations for reproducible generation when seed is provided
        let use_optimizations =
            self.optimization_enabled && config.is_none_or(|c| c.seed.is_none());

        // Start overall timing
        let _synthesis_timer = if use_optimizations {
            Some(self.performance_monitor.start_timer("synthesize"))
        } else {
            None
        };

        // Track memory usage if optimization is enabled
        if use_optimizations {
            let estimated_memory = MemoryOptimizer::estimate_mel_memory(
                self.config.mel_channels,
                phonemes.len() * 20, // Rough estimate: 20 frames per phoneme
            );
            self.performance_monitor
                .record_memory_usage("synthesis", estimated_memory);
            self.performance_monitor
                .increment_counter("synthesis_requests");
        }

        // Step 0: Apply prosody adjustments
        let adjusted_phonemes = self.apply_prosody_adjustments(phonemes, config)?;
        tracing::debug!(
            "VITS: Applied prosody adjustments to {} phonemes",
            adjusted_phonemes.len()
        );

        // Step 1: Text encoding (using prosody-adjusted phonemes)
        let text_encoding_raw = {
            let _timer = if use_optimizations {
                Some(self.performance_monitor.start_timer("text_encoding"))
            } else {
                None
            };
            self.text_encoder.forward(&adjusted_phonemes, None)?
        };
        tracing::debug!(
            "VITS: Text encoding raw shape: {:?}",
            text_encoding_raw.shape()
        );

        // Transpose to match expected format [batch, features, sequence]
        let text_encoding =
            text_encoding_raw
                .transpose(1, 2)
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Failed to transpose text encoding: {e}"),
                })?;
        tracing::debug!(
            "VITS: Text encoding transposed shape: {:?}",
            text_encoding.shape()
        );

        // Step 1.5: Apply emotion conditioning to text encoding
        let emotion_conditioned_encoding = if self.config.emotion_enabled {
            let _timer = if use_optimizations {
                Some(self.performance_monitor.start_timer("emotion_conditioning"))
            } else {
                None
            };

            // Get emotion from synthesis config, or use default neutral emotion
            let emotion_config = config
                .and_then(|c| c.emotion.as_ref())
                .cloned()
                .unwrap_or_else(crate::speaker::EmotionConfig::default);

            let conditioned = self.apply_emotion_conditioning(&text_encoding, &emotion_config)?;
            tracing::debug!(
                "VITS: Applied emotion conditioning, shape: {:?}",
                conditioned.shape()
            );
            conditioned
        } else {
            text_encoding
        };

        // Step 2: Duration prediction using neural predictor (with prosody-adjusted phonemes)
        let seed = config.and_then(|c| c.seed);
        let durations = {
            let _timer = if use_optimizations {
                Some(self.performance_monitor.start_timer("duration_prediction"))
            } else {
                None
            };
            self.duration_predictor
                .predict_phoneme_durations_with_seed(&adjusted_phonemes, seed)?
        };
        tracing::debug!("VITS: Predicted durations: {:?}", durations);

        // Step 3: Generate latent representation (prior) from emotion-conditioned text encoding
        let z_prior = self.generate_prior(&emotion_conditioned_encoding, &durations, seed)?;
        tracing::debug!("VITS: Generated prior shape: {:?}", z_prior.dims());

        // Step 4: Apply normalizing flows
        let (z_flow, log_det) = {
            let _timer = if use_optimizations {
                Some(self.performance_monitor.start_timer("flows"))
            } else {
                None
            };
            let mut flows = self
                .flows
                .lock()
                .expect("VitsAcousticImpl flows mutex poisoned");
            flows.forward(&z_prior)?
        };
        tracing::debug!(
            "VITS: Flow output shape: {:?}, log_det: {:?}",
            z_flow.dims(),
            log_det.dims()
        );

        // Step 5: Decode to mel spectrogram using neural decoder
        tracing::info!("VITS: Using neural decoder to generate mel spectrogram");

        let mel_tensor = {
            let _timer = if use_optimizations {
                Some(self.performance_monitor.start_timer("decoding"))
            } else {
                None
            };
            self.decoder
                .forward(&z_flow)
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Decoder forward failed: {e}"),
                })?
        };

        tracing::debug!("VITS: Decoder output shape: {:?}", mel_tensor.dims());

        // Convert tensor to MelSpectrogram format
        let hop_length = 256;
        let mel = tensor_to_mel_spectrogram(&mel_tensor, self.config.sample_rate, hop_length)?;

        tracing::info!(
            "VITS: Synthesis complete, generated mel spectrogram: {}x{}",
            mel.n_mels,
            mel.n_frames
        );

        Ok(mel)
    }

    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<MelSpectrogram>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        tracing::info!(
            "VITS: Starting optimized batch synthesis for {} inputs",
            inputs.len()
        );

        // Check if any config has a seed (affects optimization behavior)
        let has_seed = configs.is_some_and(|c| c.iter().any(|conf| conf.seed.is_some()));
        let use_optimizations = self.optimization_enabled && !has_seed;

        // Start batch timing
        let _batch_timer = if use_optimizations {
            Some(self.performance_monitor.start_timer("batch_synthesis"))
        } else {
            None
        };

        // Memory optimization: check if batch fits in memory budget
        if use_optimizations {
            let total_phonemes: usize = inputs.iter().map(|p| p.len()).sum();
            let estimated_memory = MemoryOptimizer::estimate_mel_memory(
                self.config.mel_channels,
                total_phonemes * 20, // Rough estimate: 20 frames per phoneme
            );

            // Warn if memory usage is high (>500MB)
            if estimated_memory > 500 * 1024 * 1024 {
                tracing::warn!(
                    "VITS: Large batch synthesis estimated memory usage: {:.1}MB",
                    estimated_memory as f32 / (1024.0 * 1024.0)
                );
            }

            self.performance_monitor
                .record_memory_usage("batch_synthesis", estimated_memory);
            self.performance_monitor
                .increment_counter("batch_synthesis_requests");
        }

        // Pre-allocate results vector for efficiency
        let mut results = Vec::with_capacity(inputs.len());

        // Batch processing with improved error handling and memory management
        for (i, phonemes) in inputs.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));

            // Add input validation to avoid unnecessary processing
            if phonemes.is_empty() {
                tracing::warn!("VITS: Skipping empty phoneme sequence at index {}", i);
                return Err(AcousticError::InputError {
                    message: format!("Empty phoneme sequence at batch index {i}"),
                });
            }

            match self.synthesize(phonemes, config).await {
                Ok(mel) => {
                    results.push(mel);

                    // Progress logging for large batches
                    if i % 10 == 0 && i > 0 {
                        tracing::debug!("VITS: Completed {}/{} batch items", i + 1, inputs.len());
                    }
                }
                Err(e) => {
                    tracing::error!("VITS: Batch synthesis failed at index {}: {}", i, e);
                    return Err(AcousticError::ProcessingError {
                        message: format!("Batch synthesis failed at index {i}: {e}"),
                    });
                }
            }
        }

        tracing::info!(
            "VITS: Batch synthesis completed successfully - {} mel spectrograms generated",
            results.len()
        );
        Ok(results)
    }

    fn metadata(&self) -> AcousticModelMetadata {
        AcousticModelMetadata {
            name: "VITS".to_string(),
            version: "1.0.0".to_string(),
            architecture: "VITS".to_string(),
            supported_languages: vec![LanguageCode::EnUs, LanguageCode::EnGb, LanguageCode::JaJp],
            sample_rate: self.config.sample_rate,
            mel_channels: self.config.mel_channels as u32,
            is_multi_speaker: self.config.multi_speaker,
            speaker_count: self.config.speaker_count.map(|c| c as u32),
        }
    }

    fn supports(&self, feature: AcousticModelFeature) -> bool {
        match feature {
            AcousticModelFeature::MultiSpeaker => self.config.multi_speaker,
            AcousticModelFeature::EmotionControl => true, // ✅ Implemented with emotion modeling
            AcousticModelFeature::BatchProcessing => true,
            AcousticModelFeature::GpuAcceleration => true,
            AcousticModelFeature::StreamingInference => true, // ✅ Implemented
            AcousticModelFeature::StreamingSynthesis => true, // ✅ Implemented
            AcousticModelFeature::ProsodyControl => true,     // ✅ Implemented
            AcousticModelFeature::StyleTransfer => true,      // ✅ Implemented
            AcousticModelFeature::VoiceCloning => true,       // ✅ Implemented
            AcousticModelFeature::RealTimeInference => true,  // ✅ Optimized with GPU support
        }
    }
}
