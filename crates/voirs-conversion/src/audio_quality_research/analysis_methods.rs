//! High-level analysis orchestration methods

use crate::Error;
use std::collections::HashMap;

use super::researcher::AudioQualityResearcher;
use super::types::*;

impl AudioQualityResearcher {
    /// Perform comprehensive audio quality analysis
    pub fn comprehensive_analysis(
        &mut self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<ComprehensiveQualityAnalysis, Error> {
        let start_time = std::time::Instant::now();

        // Validate inputs
        if original.len() != processed.len() {
            return Err(Error::validation(
                "Original and processed audio must have the same length".to_string(),
            ));
        }

        if original.is_empty() {
            return Err(Error::validation("Audio cannot be empty".to_string()));
        }

        // Perform individual analyses
        let spectral_analysis = self.analyze_spectral_quality(original, processed, sample_rate)?;
        let temporal_analysis = self.analyze_temporal_quality(original, processed, sample_rate)?;
        let psychoacoustic_analysis =
            self.analyze_psychoacoustic_quality(original, processed, sample_rate)?;
        let multidimensional_quality =
            self.analyze_multidimensional_quality(original, processed, sample_rate)?;

        // Calculate individual quality scores
        let pesq_score = if self.config.pesq_analysis {
            self.calculate_pesq_style_score(original, processed, sample_rate)?
        } else {
            3.0 // Default neutral score
        };

        let stoi_score = if self.config.stoi_analysis {
            self.calculate_stoi_style_score(original, processed, sample_rate)?
        } else {
            0.8 // Default good intelligibility
        };

        let pemo_q_score = if self.config.pemo_q_analysis {
            self.calculate_pemo_q_style_score(original, processed, sample_rate)?
        } else {
            0.8 // Default good perceptual quality
        };

        // Calculate overall perceptual quality
        let perceptual_quality = self.calculate_overall_perceptual_quality(
            &spectral_analysis,
            &temporal_analysis,
            &psychoacoustic_analysis,
            &multidimensional_quality,
        );

        // Neural quality prediction
        let neural_prediction = if self.config.neural_models {
            self.predict_neural_quality(
                &spectral_analysis,
                &temporal_analysis,
                &psychoacoustic_analysis,
            )?
        } else {
            perceptual_quality // Fallback to perceptual quality
        };

        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;
        let duration_seconds = original.len() as f32 / sample_rate as f32;

        let mut algorithms_used = vec![
            "perceptual_quality".to_string(),
            "spectral_analysis".to_string(),
            "temporal_analysis".to_string(),
            "psychoacoustic_analysis".to_string(),
        ];
        if self.config.pesq_analysis {
            algorithms_used.push("pesq".to_string());
        }
        if self.config.stoi_analysis {
            algorithms_used.push("stoi".to_string());
        }
        if self.config.pemo_q_analysis {
            algorithms_used.push("pemo_q".to_string());
        }
        if self.config.neural_models {
            algorithms_used.push("neural_prediction".to_string());
        }

        let analysis_stats = AnalysisStatistics {
            processing_time_ms: processing_time,
            frames_analyzed: original.len() / self.config.frame_size,
            sample_rate,
            duration_seconds,
            algorithms_used,
        };

        self.analysis_count += 1;

        Ok(ComprehensiveQualityAnalysis {
            perceptual_quality,
            neural_prediction,
            pesq_score,
            stoi_score,
            pemo_q_score,
            spectral_analysis,
            temporal_analysis,
            psychoacoustic_analysis,
            multidimensional_quality,
            analysis_stats,
        })
    }

    /// Analyze spectral quality
    pub(crate) fn analyze_spectral_quality(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<SpectralQualityAnalysis, Error> {
        // Calculate spectral distortion
        let spectral_distortion = self.calculate_spectral_distortion(original, processed)?;

        // Calculate cepstral distance
        let cepstral_distance = self.calculate_cepstral_distance(original, processed)?;

        // Calculate log spectral distance
        let log_spectral_distance = self.calculate_log_spectral_distance(original, processed)?;

        // Calculate Itakura-Saito distortion
        let itakura_saito_distortion =
            self.calculate_itakura_saito_distortion(original, processed)?;

        // Calculate spectral correlation
        let spectral_correlation = self.calculate_spectral_correlation(original, processed)?;

        // Calculate spectral flatness deviation
        let spectral_flatness_deviation =
            self.calculate_spectral_flatness_deviation(original, processed)?;

        // Analyze harmonic distortion
        let harmonic_distortion =
            self.analyze_harmonic_distortion(original, processed, sample_rate)?;

        Ok(SpectralQualityAnalysis {
            spectral_distortion,
            cepstral_distance,
            log_spectral_distance,
            itakura_saito_distortion,
            spectral_correlation,
            spectral_flatness_deviation,
            harmonic_distortion,
        })
    }

    /// Analyze temporal quality
    pub(crate) fn analyze_temporal_quality(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<TemporalQualityAnalysis, Error> {
        let temporal_coherence = self.calculate_temporal_coherence(original, processed)?;
        let envelope_correlation = self.calculate_envelope_correlation(original, processed)?;
        let zcr_deviation = self.calculate_zcr_deviation(original, processed)?;
        let temporal_smoothness = self.calculate_temporal_smoothness(processed)?;
        let phase_coherence = self.calculate_phase_coherence(original, processed)?;
        let rhythm_preservation = self.calculate_rhythm_preservation(original, processed)?;

        Ok(TemporalQualityAnalysis {
            temporal_coherence,
            envelope_correlation,
            zcr_deviation,
            temporal_smoothness,
            phase_coherence,
            rhythm_preservation,
        })
    }

    /// Analyze psychoacoustic quality
    pub(crate) fn analyze_psychoacoustic_quality(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<PsychoacousticAnalysis, Error> {
        let loudness_deviation = self.calculate_loudness_deviation(original, processed)?;
        let critical_band_analysis =
            self.analyze_critical_bands(original, processed, sample_rate)?;
        let masking_threshold_deviation =
            self.calculate_masking_threshold_deviation(original, processed)?;
        let sharpness_deviation = self.calculate_sharpness_deviation(original, processed)?;
        let roughness = self.calculate_roughness(processed)?;
        let fluctuation_strength = self.calculate_fluctuation_strength(processed)?;
        let tonality = self.analyze_tonality(original, processed)?;

        Ok(PsychoacousticAnalysis {
            loudness_deviation,
            critical_band_analysis,
            masking_threshold_deviation,
            sharpness_deviation,
            roughness,
            fluctuation_strength,
            tonality,
        })
    }

    /// Analyze multidimensional quality
    pub(crate) fn analyze_multidimensional_quality(
        &self,
        original: &[f32],
        processed: &[f32],
        sample_rate: u32,
    ) -> Result<MultidimensionalQuality, Error> {
        let naturalness = self.calculate_naturalness(original, processed, sample_rate)?;
        let clarity = self.calculate_clarity(processed)?;
        let pleasantness = self.calculate_pleasantness(processed)?;
        let intelligibility = self.calculate_intelligibility(original, processed)?;
        let spaciousness = self.calculate_spaciousness(processed)?;
        let warmth = self.calculate_warmth(processed)?;
        let brightness = self.calculate_brightness(processed)?;
        let presence = self.calculate_presence(processed)?;

        Ok(MultidimensionalQuality {
            naturalness,
            clarity,
            pleasantness,
            intelligibility,
            spaciousness,
            warmth,
            brightness,
            presence,
        })
    }

    /// Calculate PESQ-style quality score
    pub(crate) fn calculate_pesq_style_score(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<f32, Error> {
        // Simplified PESQ-style calculation
        let mse = original
            .iter()
            .zip(processed.iter())
            .map(|(&o, &p)| (o - p).powi(2))
            .sum::<f32>()
            / original.len() as f32;

        let snr = if mse > 0.0 {
            10.0 * (1.0 / mse).log10()
        } else {
            60.0 // Very high quality
        };

        // Convert to PESQ scale (1.0-5.0)
        let pesq_score = (snr / 20.0 + 1.0).clamp(1.0, 5.0);
        Ok(pesq_score)
    }

    /// Calculate STOI-style intelligibility score
    pub(crate) fn calculate_stoi_style_score(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<f32, Error> {
        // Simplified STOI-style calculation based on correlation in frequency bands
        let frame_size = 256;
        let mut correlations = Vec::new();

        for chunk in original
            .chunks(frame_size)
            .zip(processed.chunks(frame_size))
        {
            let (orig_chunk, proc_chunk) = chunk;
            if orig_chunk.len() == proc_chunk.len() && orig_chunk.len() == frame_size {
                let correlation = self.calculate_correlation(orig_chunk, proc_chunk);
                correlations.push(correlation);
            }
        }

        let stoi_score = if correlations.is_empty() {
            0.8
        } else {
            correlations.iter().sum::<f32>() / correlations.len() as f32
        };

        Ok(stoi_score.clamp(0.0, 1.0))
    }

    /// Calculate PEMO-Q style perceptual score
    pub(crate) fn calculate_pemo_q_style_score(
        &self,
        original: &[f32],
        processed: &[f32],
        _sample_rate: u32,
    ) -> Result<f32, Error> {
        // Simplified PEMO-Q style calculation
        let loudness_diff = self.calculate_loudness_difference(original, processed)?;
        let sharpness_diff = self.calculate_sharpness_difference(original, processed)?;
        let roughness_level = self.calculate_roughness(processed)?;

        // Combine metrics (simplified PEMO-Q approach)
        let pemo_q_score =
            1.0 - (loudness_diff * 0.4 + sharpness_diff * 0.3 + roughness_level * 0.3);
        Ok(pemo_q_score.clamp(0.0, 1.0))
    }

    /// Predict quality using neural model
    pub(crate) fn predict_neural_quality(
        &self,
        spectral: &SpectralQualityAnalysis,
        temporal: &TemporalQualityAnalysis,
        psychoacoustic: &PsychoacousticAnalysis,
    ) -> Result<f32, Error> {
        // Extract features
        let mut features: HashMap<&str, f32> = HashMap::new();
        features.insert("spectral_distortion", spectral.spectral_distortion);
        features.insert("temporal_coherence", temporal.temporal_coherence);
        features.insert("loudness_deviation", psychoacoustic.loudness_deviation);
        features.insert(
            "harmonic_preservation",
            1.0 - spectral.harmonic_distortion.thd,
        );
        features.insert("noise_level", psychoacoustic.roughness);

        // Simple neural network simulation
        let mut score = 0.5; // Base score
        for (feature_name, &feature_value) in &features {
            if let Some(&weight) = self.neural_model.weights.get(feature_name) {
                let normalized_value = if let Some(&(mean, std)) =
                    self.neural_model.feature_scales.get(feature_name)
                {
                    (feature_value - mean) / std
                } else {
                    feature_value
                };
                score += weight * normalized_value * 0.1; // Scale contribution
            }
        }

        Ok(score.clamp(0.0, 1.0))
    }

    /// Calculate overall perceptual quality
    pub(crate) fn calculate_overall_perceptual_quality(
        &self,
        spectral: &SpectralQualityAnalysis,
        temporal: &TemporalQualityAnalysis,
        psychoacoustic: &PsychoacousticAnalysis,
        multidimensional: &MultidimensionalQuality,
    ) -> f32 {
        // Weighted combination of different quality aspects
        let spectral_quality = 1.0 - spectral.spectral_distortion;
        let temporal_quality = temporal.temporal_coherence;
        let psychoacoustic_quality = 1.0 - psychoacoustic.loudness_deviation;
        let multidimensional_avg = (multidimensional.naturalness
            + multidimensional.clarity
            + multidimensional.pleasantness
            + multidimensional.intelligibility)
            / 4.0;

        // Weighted average
        (spectral_quality * 0.3
            + temporal_quality * 0.25
            + psychoacoustic_quality * 0.25
            + multidimensional_avg * 0.2)
            .clamp(0.0, 1.0)
    }
}
