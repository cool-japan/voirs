//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::perceptual::cross_cultural::{CrossCulturalConfig, CrossCulturalPerceptualModel};
use crate::quality::cross_language_intelligibility::{
    CrossLanguageIntelligibilityConfig, CrossLanguageIntelligibilityEvaluator,
};
use crate::quality::universal_phoneme_mapping::{
    UniversalPhonemeMapper, UniversalPhonemeMappingConfig,
};
use crate::traits::{EvaluationResult, QualityScore};
use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use voirs_recognizer::traits::PhonemeAlignment;
use voirs_sdk::{AudioBuffer, LanguageCode};

use super::data_types::*;

/// Multilingual speaker model evaluator
pub struct MultilingualSpeakerModelEvaluator {
    /// Configuration
    config: MultilingualSpeakerModelConfig,
    /// Universal phoneme mapper
    phoneme_mapper: UniversalPhonemeMapper,
    /// Cross-language intelligibility evaluator
    intelligibility_evaluator: CrossLanguageIntelligibilityEvaluator,
    /// Cross-cultural perceptual model
    cultural_model: CrossCulturalPerceptualModel,
    /// Cached speaker models
    speaker_models: HashMap<String, SpeakerModel>,
    /// Cached voice characteristics
    voice_characteristics_cache: HashMap<(String, LanguageCode), VoiceCharacteristics>,
}
impl MultilingualSpeakerModelEvaluator {
    /// Create new multilingual speaker model evaluator
    pub fn new(config: MultilingualSpeakerModelConfig) -> Self {
        let phoneme_mapper = UniversalPhonemeMapper::new(UniversalPhonemeMappingConfig::default());
        let intelligibility_evaluator = CrossLanguageIntelligibilityEvaluator::new(
            CrossLanguageIntelligibilityConfig::default(),
        );
        let cultural_model = CrossCulturalPerceptualModel::new(CrossCulturalConfig::default());
        Self {
            config,
            phoneme_mapper,
            intelligibility_evaluator,
            cultural_model,
            speaker_models: HashMap::new(),
            voice_characteristics_cache: HashMap::new(),
        }
    }
    /// Evaluate multilingual speaker model
    pub async fn evaluate_multilingual_speaker_model(
        &mut self,
        speaker_id: &str,
        reference_audio: &AudioBuffer,
        reference_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        phoneme_alignments: Option<&HashMap<LanguageCode, PhonemeAlignment>>,
    ) -> EvaluationResult<MultilingualSpeakerModelResult> {
        let start_time = std::time::Instant::now();
        let reference_characteristics = self
            .extract_voice_characteristics(reference_audio, reference_language)
            .await?;
        let speaker_model = self.create_or_update_speaker_model(
            speaker_id,
            reference_language,
            reference_characteristics.clone(),
            target_audios.keys().cloned().collect(),
        );
        let voice_transfer_quality = if self.config.enable_voice_transfer_quality {
            self.evaluate_voice_transfer_quality(
                &reference_characteristics,
                target_audios,
                reference_language,
            )
            .await?
        } else {
            HashMap::new()
        };
        let speaker_identity_preservation = if self.config.enable_speaker_identity_preservation {
            self.evaluate_speaker_identity_preservation(
                &reference_characteristics,
                target_audios,
                reference_language,
            )
            .await?
        } else {
            HashMap::new()
        };
        let language_adaptation = if self.config.enable_language_adaptation {
            self.evaluate_language_adaptation(target_audios, reference_language, phoneme_alignments)
                .await?
        } else {
            HashMap::new()
        };
        let acoustic_consistency = if self.config.enable_acoustic_consistency {
            self.evaluate_acoustic_consistency(
                &reference_characteristics,
                target_audios,
                reference_language,
            )
            .await?
        } else {
            HashMap::new()
        };
        let perceptual_similarity = if self.config.enable_perceptual_similarity {
            self.evaluate_perceptual_similarity(reference_audio, target_audios, reference_language)
                .await?
        } else {
            HashMap::new()
        };
        let cross_language_similarity = self
            .calculate_cross_language_similarity(target_audios)
            .await?;
        let voice_characteristics = self
            .analyze_voice_characteristics(
                &reference_characteristics,
                target_audios,
                reference_language,
            )
            .await?;
        let language_adaptations = self
            .generate_language_adaptations(target_audios, reference_language, phoneme_alignments)
            .await?;
        let problematic_pairs = self.identify_problematic_language_pairs(
            &voice_transfer_quality,
            &speaker_identity_preservation,
            &language_adaptation,
            &acoustic_consistency,
            &perceptual_similarity,
            reference_language,
        );
        let overall_quality = self.calculate_overall_quality(
            &voice_transfer_quality,
            &speaker_identity_preservation,
            &language_adaptation,
            &acoustic_consistency,
            &perceptual_similarity,
        );
        let evaluation_confidence = self.calculate_evaluation_confidence(
            &voice_transfer_quality,
            &speaker_identity_preservation,
            &language_adaptation,
            &acoustic_consistency,
            &perceptual_similarity,
        );
        let processing_time = start_time.elapsed();
        Ok(MultilingualSpeakerModelResult {
            reference_language,
            target_languages: target_audios.keys().cloned().collect(),
            overall_quality,
            voice_transfer_quality,
            speaker_identity_preservation,
            language_adaptation,
            acoustic_consistency,
            perceptual_similarity,
            cross_language_similarity,
            voice_characteristics,
            language_adaptations,
            problematic_pairs,
            evaluation_confidence,
            processing_time,
        })
    }
    /// Extract voice characteristics from audio
    pub async fn extract_voice_characteristics(
        &mut self,
        audio: &AudioBuffer,
        language: LanguageCode,
    ) -> EvaluationResult<VoiceCharacteristics> {
        let cache_key = (audio.samples().len().to_string(), language);
        if let Some(cached_characteristics) = self.voice_characteristics_cache.get(&cache_key) {
            return Ok(cached_characteristics.clone());
        }
        let samples = audio.samples();
        let f0_stats = self.extract_f0_statistics(samples)?;
        let formant_stats = self.extract_formant_statistics(samples)?;
        let spectral_stats = self.extract_spectral_statistics(samples)?;
        let temporal_stats = self.extract_temporal_statistics(samples)?;
        let voice_quality_stats = self.extract_voice_quality_statistics(samples)?;
        let prosodic_stats = self.extract_prosodic_statistics(samples)?;
        let characteristics = VoiceCharacteristics {
            f0_stats,
            formant_stats,
            spectral_stats,
            temporal_stats,
            voice_quality_stats,
            prosodic_stats,
        };
        self.voice_characteristics_cache
            .insert(cache_key, characteristics.clone());
        Ok(characteristics)
    }
    /// Extract F0 statistics
    fn extract_f0_statistics(&self, samples: &[f32]) -> EvaluationResult<F0Statistics> {
        let chunk_size = 1024;
        let mut f0_values = Vec::new();
        for chunk in samples.chunks(chunk_size) {
            let f0 = self.estimate_f0(chunk);
            if f0 > 0.0 {
                f0_values.push(f0);
            }
        }
        if f0_values.is_empty() {
            return Ok(F0Statistics {
                mean_f0: 0.0,
                f0_std: 0.0,
                f0_range: (0.0, 0.0),
                f0_variability: 0.0,
            });
        }
        let mean_f0 = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
        let f0_variance = f0_values
            .iter()
            .map(|&f0| (f0 - mean_f0).powi(2))
            .sum::<f32>()
            / f0_values.len() as f32;
        let f0_std = f0_variance.sqrt();
        let f0_min = f0_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let f0_max = f0_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let f0_variability = if mean_f0 > 0.0 { f0_std / mean_f0 } else { 0.0 };
        Ok(F0Statistics {
            mean_f0,
            f0_std,
            f0_range: (f0_min, f0_max),
            f0_variability,
        })
    }
    /// Estimate F0 using autocorrelation
    fn estimate_f0(&self, samples: &[f32]) -> f32 {
        let min_period = 40;
        let max_period = 400;
        let mut best_corr = 0.0;
        let mut best_period = min_period;
        for period in min_period..=max_period.min(samples.len() / 2) {
            let mut correlation = 0.0;
            let mut count = 0;
            for i in 0..(samples.len() - period) {
                correlation += samples[i] * samples[i + period];
                count += 1;
            }
            if count > 0 {
                correlation /= count as f32;
                if correlation > best_corr {
                    best_corr = correlation;
                    best_period = period;
                }
            }
        }
        if best_corr > 0.3 {
            16000.0 / best_period as f32
        } else {
            0.0
        }
    }
    /// Extract formant statistics using Linear Prediction Coding (LPC)
    fn extract_formant_statistics(&self, samples: &[f32]) -> EvaluationResult<FormantStatistics> {
        if samples.is_empty() {
            return Err(EvaluationError::InvalidInput {
                message: "Empty audio samples for formant extraction".to_string(),
            }
            .into());
        }
        let sample_rate = 16000.0;
        let frame_size = 512;
        let overlap = 256;
        let lpc_order = 12;
        let mut all_formants = Vec::new();
        let mut frame_start = 0;
        while frame_start + frame_size <= samples.len() {
            let frame = &samples[frame_start..frame_start + frame_size];
            let windowed_frame: Vec<f32> = frame
                .iter()
                .enumerate()
                .map(|(i, &sample)| {
                    let window_val = 0.54
                        - 0.46
                            * (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32)
                                .cos();
                    sample * window_val
                })
                .collect();
            if let Ok(formants) = self.lpc_formant_analysis(&windowed_frame, lpc_order, sample_rate)
            {
                all_formants.push(formants);
            }
            frame_start += frame_size - overlap;
        }
        if all_formants.is_empty() {
            return Ok(FormantStatistics {
                mean_formants: vec![700.0, 1300.0, 2500.0],
                formant_stds: vec![100.0, 150.0, 200.0],
                formant_bandwidths: vec![80.0, 120.0, 160.0],
            });
        }
        let num_formants = all_formants[0].len();
        let mut mean_formants = vec![0.0; num_formants];
        let mut formant_values: Vec<Vec<f32>> = vec![Vec::new(); num_formants];
        for frame_formants in &all_formants {
            for (i, &formant) in frame_formants.iter().enumerate() {
                if i < num_formants && formant > 0.0 && formant < sample_rate / 2.0 {
                    formant_values[i].push(formant);
                }
            }
        }
        let mut formant_stds = vec![0.0; num_formants];
        for (i, values) in formant_values.iter().enumerate() {
            if !values.is_empty() {
                mean_formants[i] = values.iter().sum::<f32>() / values.len() as f32;
                let variance = values
                    .iter()
                    .map(|&f| (f - mean_formants[i]).powi(2))
                    .sum::<f32>()
                    / values.len() as f32;
                formant_stds[i] = variance.sqrt();
            }
        }
        let formant_bandwidths: Vec<f32> = mean_formants
            .iter()
            .map(|&f| (f * 0.1).max(50.0).min(300.0))
            .collect();
        Ok(FormantStatistics {
            mean_formants,
            formant_stds,
            formant_bandwidths,
        })
    }
    /// Perform LPC analysis to find formants
    fn lpc_formant_analysis(
        &self,
        samples: &[f32],
        order: usize,
        sample_rate: f32,
    ) -> EvaluationResult<Vec<f32>> {
        let autocorr = self.autocorrelation(samples, order + 1);
        let lpc_coeffs = self.levinson_durbin(&autocorr)?;
        let formants = self.find_formants_from_lpc(&lpc_coeffs, sample_rate);
        Ok(formants)
    }
    /// Calculate autocorrelation function
    fn autocorrelation(&self, samples: &[f32], max_lag: usize) -> Vec<f32> {
        let mut autocorr = vec![0.0; max_lag];
        let n = samples.len();
        for lag in 0..max_lag {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += samples[i] * samples[i + lag];
            }
            autocorr[lag] = sum / (n - lag) as f32;
        }
        autocorr
    }
    /// Levinson-Durbin algorithm for solving Yule-Walker equations
    fn levinson_durbin(&self, autocorr: &[f32]) -> EvaluationResult<Vec<f32>> {
        let n = autocorr.len() - 1;
        let mut a = vec![0.0; n + 1];
        let mut k = vec![0.0; n];
        a[0] = 1.0;
        let mut e = autocorr[0];
        for i in 1..=n {
            let mut sum = 0.0;
            for j in 1..i {
                sum += a[j] * autocorr[i - j];
            }
            if e.abs() < 1e-10 {
                return Err(EvaluationError::InvalidInput {
                    message: "Singular autocorrelation matrix in LPC analysis".to_string(),
                }
                .into());
            }
            k[i - 1] = -(autocorr[i] + sum) / e;
            a[i] = k[i - 1];
            for j in 1..i {
                a[j] += k[i - 1] * a[i - j];
            }
            e *= 1.0 - k[i - 1] * k[i - 1];
        }
        Ok(a)
    }
    /// Find formants from LPC coefficients
    fn find_formants_from_lpc(&self, lpc_coeffs: &[f32], sample_rate: f32) -> Vec<f32> {
        let mut formants = Vec::new();
        let fft_size = 512;
        let freq_resolution = sample_rate / fft_size as f32;
        let mut spectrum = vec![0.0; fft_size / 2];
        for (i, spectrum_val) in spectrum.iter_mut().enumerate() {
            let omega = 2.0 * std::f32::consts::PI * i as f32 / fft_size as f32;
            let mut real_part = lpc_coeffs[0];
            let mut imag_part = 0.0;
            for (k, &coeff) in lpc_coeffs.iter().enumerate().skip(1) {
                let phase = k as f32 * omega;
                real_part += coeff * phase.cos();
                imag_part += coeff * phase.sin();
            }
            let magnitude_sq = real_part * real_part + imag_part * imag_part;
            *spectrum_val = if magnitude_sq > 1e-10 {
                1.0 / magnitude_sq.sqrt()
            } else {
                0.0
            };
        }
        for i in 1..(spectrum.len() - 1) {
            if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.1 {
                let frequency = i as f32 * freq_resolution;
                if frequency > 200.0 && frequency < 4000.0 {
                    formants.push(frequency);
                }
            }
        }
        formants.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        formants.truncate(4);
        while formants.len() < 3 {
            match formants.len() {
                0 => formants.push(700.0),
                1 => formants.push(1300.0),
                2 => formants.push(2500.0),
                _ => break,
            }
        }
        formants
    }
    /// Extract spectral statistics
    fn extract_spectral_statistics(&self, samples: &[f32]) -> EvaluationResult<SpectralStatistics> {
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let spectral_centroid = self.calculate_spectral_centroid(samples);
        Ok(SpectralStatistics {
            spectral_centroid,
            spectral_spread: 1000.0,
            spectral_tilt: -6.0,
            spectral_rolloff: 4000.0,
        })
    }
    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, samples: &[f32]) -> f32 {
        let chunk_size = 512;
        let mut centroid_sum = 0.0;
        let mut count = 0;
        for chunk in samples.chunks(chunk_size) {
            if chunk.len() == chunk_size {
                let mut weighted_sum = 0.0;
                let mut total_energy = 0.0;
                for (i, &sample) in chunk.iter().enumerate() {
                    let energy = sample * sample;
                    weighted_sum += energy * i as f32;
                    total_energy += energy;
                }
                if total_energy > 0.0 {
                    let centroid = weighted_sum / total_energy;
                    centroid_sum += centroid * 16000.0 / chunk_size as f32;
                    count += 1;
                }
            }
        }
        if count > 0 {
            centroid_sum / count as f32
        } else {
            1000.0
        }
    }
    /// Extract temporal statistics
    fn extract_temporal_statistics(&self, samples: &[f32]) -> EvaluationResult<TemporalStatistics> {
        let duration = samples.len() as f32 / 16000.0;
        let speaking_rate = self.estimate_speaking_rate(samples);
        Ok(TemporalStatistics {
            speaking_rate,
            pause_frequency: 0.5,
            pause_duration: 0.3,
            rhythm_regularity: 0.7,
        })
    }
    /// Estimate speaking rate
    fn estimate_speaking_rate(&self, samples: &[f32]) -> f32 {
        let chunk_size = 800;
        let mut energy_peaks = 0;
        let mut prev_energy = 0.0;
        for chunk in samples.chunks(chunk_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
            if energy > prev_energy * 1.5 && energy > 0.01 {
                energy_peaks += 1;
            }
            prev_energy = energy;
        }
        let duration_seconds = samples.len() as f32 / 16000.0;
        if duration_seconds > 0.0 {
            (energy_peaks as f32 / duration_seconds).clamp(2.0, 8.0)
        } else {
            4.0
        }
    }
    /// Extract voice quality statistics
    fn extract_voice_quality_statistics(
        &self,
        samples: &[f32],
    ) -> EvaluationResult<VoiceQualityStatistics> {
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let hnr = if rms > 0.0 {
            20.0 * (peak / rms).log10()
        } else {
            0.0
        };
        Ok(VoiceQualityStatistics {
            jitter: 0.02,
            shimmer: 0.03,
            hnr,
            spectral_noise: 0.1,
        })
    }
    /// Extract prosodic statistics
    fn extract_prosodic_statistics(
        &self,
        _samples: &[f32],
    ) -> EvaluationResult<ProsodicStatistics> {
        Ok(ProsodicStatistics {
            intonation_range: 0.5,
            stress_prominence: 0.7,
            rhythm_consistency: 0.8,
            prosodic_variability: 0.3,
        })
    }
    /// Create or update speaker model
    fn create_or_update_speaker_model(
        &mut self,
        speaker_id: &str,
        reference_language: LanguageCode,
        reference_characteristics: VoiceCharacteristics,
        supported_languages: Vec<LanguageCode>,
    ) -> &SpeakerModel {
        let speaker_model = SpeakerModel {
            speaker_id: speaker_id.to_string(),
            reference_language,
            voice_characteristics: reference_characteristics,
            supported_languages: supported_languages.clone(),
            language_quality: supported_languages
                .iter()
                .map(|&lang| (lang, 0.8))
                .collect(),
        };
        self.speaker_models
            .insert(speaker_id.to_string(), speaker_model);
        self.speaker_models
            .get(speaker_id)
            .expect("value should be present")
    }
    /// Evaluate voice transfer quality
    async fn evaluate_voice_transfer_quality(
        &self,
        reference_characteristics: &VoiceCharacteristics,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_language: LanguageCode,
    ) -> EvaluationResult<HashMap<LanguageCode, f32>> {
        let mut quality_scores = HashMap::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != reference_language {
                let target_characteristics =
                    self.extract_voice_characteristics_sync(target_audio, *target_language)?;
                let transfer_quality = self.calculate_voice_transfer_quality(
                    reference_characteristics,
                    &target_characteristics,
                    reference_language,
                    *target_language,
                );
                quality_scores.insert(*target_language, transfer_quality);
            }
        }
        Ok(quality_scores)
    }
    /// Extract voice characteristics synchronously
    fn extract_voice_characteristics_sync(
        &self,
        audio: &AudioBuffer,
        _language: LanguageCode,
    ) -> EvaluationResult<VoiceCharacteristics> {
        let samples = audio.samples();
        let f0_stats = self.extract_f0_statistics(samples)?;
        let formant_stats = self.extract_formant_statistics(samples)?;
        let spectral_stats = self.extract_spectral_statistics(samples)?;
        let temporal_stats = self.extract_temporal_statistics(samples)?;
        let voice_quality_stats = self.extract_voice_quality_statistics(samples)?;
        let prosodic_stats = self.extract_prosodic_statistics(samples)?;
        Ok(VoiceCharacteristics {
            f0_stats,
            formant_stats,
            spectral_stats,
            temporal_stats,
            voice_quality_stats,
            prosodic_stats,
        })
    }
    /// Calculate voice transfer quality
    pub fn calculate_voice_transfer_quality(
        &self,
        reference_characteristics: &VoiceCharacteristics,
        target_characteristics: &VoiceCharacteristics,
        _reference_language: LanguageCode,
        _target_language: LanguageCode,
    ) -> f32 {
        let f0_similarity = self.calculate_f0_similarity(
            &reference_characteristics.f0_stats,
            &target_characteristics.f0_stats,
        );
        let formant_similarity = self.calculate_formant_similarity(
            &reference_characteristics.formant_stats,
            &target_characteristics.formant_stats,
        );
        let spectral_similarity = self.calculate_spectral_similarity(
            &reference_characteristics.spectral_stats,
            &target_characteristics.spectral_stats,
        );
        let temporal_similarity = self.calculate_temporal_similarity(
            &reference_characteristics.temporal_stats,
            &target_characteristics.temporal_stats,
        );
        let voice_quality_similarity = self.calculate_voice_quality_similarity(
            &reference_characteristics.voice_quality_stats,
            &target_characteristics.voice_quality_stats,
        );
        let prosodic_similarity = self.calculate_prosodic_similarity(
            &reference_characteristics.prosodic_stats,
            &target_characteristics.prosodic_stats,
        );
        let overall_similarity = f0_similarity * 0.2
            + formant_similarity * 0.2
            + spectral_similarity * 0.2
            + temporal_similarity * 0.15
            + voice_quality_similarity * 0.15
            + prosodic_similarity * 0.1;
        overall_similarity.max(0.0).min(1.0)
    }
    /// Calculate F0 similarity
    fn calculate_f0_similarity(
        &self,
        ref_stats: &F0Statistics,
        target_stats: &F0Statistics,
    ) -> f32 {
        if ref_stats.mean_f0 == 0.0 && target_stats.mean_f0 == 0.0 {
            return 1.0;
        }
        let mean_similarity = 1.0
            - (ref_stats.mean_f0 - target_stats.mean_f0).abs()
                / ref_stats.mean_f0.max(target_stats.mean_f0).max(1.0);
        let variability_similarity =
            1.0 - (ref_stats.f0_variability - target_stats.f0_variability).abs();
        (mean_similarity + variability_similarity) / 2.0
    }
    /// Calculate formant similarity
    fn calculate_formant_similarity(
        &self,
        ref_stats: &FormantStatistics,
        target_stats: &FormantStatistics,
    ) -> f32 {
        let mut similarity_sum = 0.0;
        let min_formants = ref_stats
            .mean_formants
            .len()
            .min(target_stats.mean_formants.len());
        for i in 0..min_formants {
            let formant_similarity = 1.0
                - (ref_stats.mean_formants[i] - target_stats.mean_formants[i]).abs()
                    / ref_stats.mean_formants[i]
                        .max(target_stats.mean_formants[i])
                        .max(1.0);
            similarity_sum += formant_similarity;
        }
        if min_formants > 0 {
            similarity_sum / min_formants as f32
        } else {
            0.5
        }
    }
    /// Calculate spectral similarity
    fn calculate_spectral_similarity(
        &self,
        ref_stats: &SpectralStatistics,
        target_stats: &SpectralStatistics,
    ) -> f32 {
        let centroid_similarity = 1.0
            - (ref_stats.spectral_centroid - target_stats.spectral_centroid).abs()
                / ref_stats
                    .spectral_centroid
                    .max(target_stats.spectral_centroid)
                    .max(1.0);
        let tilt_similarity =
            1.0 - (ref_stats.spectral_tilt - target_stats.spectral_tilt).abs() / 20.0;
        (centroid_similarity + tilt_similarity) / 2.0
    }
    /// Calculate temporal similarity
    fn calculate_temporal_similarity(
        &self,
        ref_stats: &TemporalStatistics,
        target_stats: &TemporalStatistics,
    ) -> f32 {
        let rate_similarity =
            1.0 - (ref_stats.speaking_rate - target_stats.speaking_rate).abs() / 10.0;
        let rhythm_similarity =
            1.0 - (ref_stats.rhythm_regularity - target_stats.rhythm_regularity).abs();
        (rate_similarity + rhythm_similarity) / 2.0
    }
    /// Calculate voice quality similarity
    fn calculate_voice_quality_similarity(
        &self,
        ref_stats: &VoiceQualityStatistics,
        target_stats: &VoiceQualityStatistics,
    ) -> f32 {
        let jitter_similarity = 1.0 - (ref_stats.jitter - target_stats.jitter).abs() / 0.1;
        let shimmer_similarity = 1.0 - (ref_stats.shimmer - target_stats.shimmer).abs() / 0.1;
        let hnr_similarity = 1.0 - (ref_stats.hnr - target_stats.hnr).abs() / 40.0;
        (jitter_similarity + shimmer_similarity + hnr_similarity) / 3.0
    }
    /// Calculate prosodic similarity
    fn calculate_prosodic_similarity(
        &self,
        ref_stats: &ProsodicStatistics,
        target_stats: &ProsodicStatistics,
    ) -> f32 {
        let intonation_similarity =
            1.0 - (ref_stats.intonation_range - target_stats.intonation_range).abs();
        let stress_similarity =
            1.0 - (ref_stats.stress_prominence - target_stats.stress_prominence).abs();
        let rhythm_similarity =
            1.0 - (ref_stats.rhythm_consistency - target_stats.rhythm_consistency).abs();
        (intonation_similarity + stress_similarity + rhythm_similarity) / 3.0
    }
    /// Evaluate speaker identity preservation
    async fn evaluate_speaker_identity_preservation(
        &self,
        reference_characteristics: &VoiceCharacteristics,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_language: LanguageCode,
    ) -> EvaluationResult<HashMap<LanguageCode, f32>> {
        let mut preservation_scores = HashMap::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != reference_language {
                let target_characteristics =
                    self.extract_voice_characteristics_sync(target_audio, *target_language)?;
                let preservation_score = self.calculate_speaker_identity_preservation(
                    reference_characteristics,
                    &target_characteristics,
                    reference_language,
                    *target_language,
                );
                preservation_scores.insert(*target_language, preservation_score);
            }
        }
        Ok(preservation_scores)
    }
    /// Calculate speaker identity preservation
    pub fn calculate_speaker_identity_preservation(
        &self,
        reference_characteristics: &VoiceCharacteristics,
        target_characteristics: &VoiceCharacteristics,
        _reference_language: LanguageCode,
        _target_language: LanguageCode,
    ) -> f32 {
        let voice_quality_preservation = self.calculate_voice_quality_similarity(
            &reference_characteristics.voice_quality_stats,
            &target_characteristics.voice_quality_stats,
        );
        let f0_character_preservation = self.calculate_f0_character_preservation(
            &reference_characteristics.f0_stats,
            &target_characteristics.f0_stats,
        );
        let spectral_character_preservation = self.calculate_spectral_character_preservation(
            &reference_characteristics.spectral_stats,
            &target_characteristics.spectral_stats,
        );
        let overall_preservation = voice_quality_preservation * 0.4
            + f0_character_preservation * 0.35
            + spectral_character_preservation * 0.25;
        overall_preservation.max(0.0).min(1.0)
    }
    /// Calculate F0 character preservation
    fn calculate_f0_character_preservation(
        &self,
        ref_stats: &F0Statistics,
        target_stats: &F0Statistics,
    ) -> f32 {
        let variability_preservation =
            1.0 - (ref_stats.f0_variability - target_stats.f0_variability).abs();
        let range_preservation = if ref_stats.f0_range.0 > 0.0 && target_stats.f0_range.0 > 0.0 {
            let ref_range = ref_stats.f0_range.1 - ref_stats.f0_range.0;
            let target_range = target_stats.f0_range.1 - target_stats.f0_range.0;
            1.0 - (ref_range - target_range).abs() / ref_range.max(target_range).max(1.0)
        } else {
            0.5
        };
        (variability_preservation + range_preservation) / 2.0
    }
    /// Calculate spectral character preservation
    fn calculate_spectral_character_preservation(
        &self,
        ref_stats: &SpectralStatistics,
        target_stats: &SpectralStatistics,
    ) -> f32 {
        let tilt_preservation =
            1.0 - (ref_stats.spectral_tilt - target_stats.spectral_tilt).abs() / 20.0;
        let rolloff_preservation = 1.0
            - (ref_stats.spectral_rolloff - target_stats.spectral_rolloff).abs()
                / ref_stats
                    .spectral_rolloff
                    .max(target_stats.spectral_rolloff)
                    .max(1.0);
        (tilt_preservation + rolloff_preservation) / 2.0
    }
    /// Evaluate language adaptation
    async fn evaluate_language_adaptation(
        &self,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_language: LanguageCode,
        _phoneme_alignments: Option<&HashMap<LanguageCode, PhonemeAlignment>>,
    ) -> EvaluationResult<HashMap<LanguageCode, f32>> {
        let mut adaptation_scores = HashMap::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != reference_language {
                let adaptation_score = self.calculate_language_adaptation_score(
                    target_audio,
                    reference_language,
                    *target_language,
                )?;
                adaptation_scores.insert(*target_language, adaptation_score);
            }
        }
        Ok(adaptation_scores)
    }
    /// Calculate language adaptation score
    fn calculate_language_adaptation_score(
        &self,
        target_audio: &AudioBuffer,
        reference_language: LanguageCode,
        target_language: LanguageCode,
    ) -> EvaluationResult<f32> {
        let intelligibility_score = self.intelligibility_evaluator.predict_intelligibility(
            reference_language,
            target_language,
            None,
        );
        let phoneme_coverage = self
            .phoneme_mapper
            .analyze_phoneme_coverage(reference_language, target_language)?;
        let get_language_code = |lang: LanguageCode| -> String {
            match lang {
                LanguageCode::EnUs | LanguageCode::EnGb => "en".to_string(),
                LanguageCode::EsEs | LanguageCode::EsMx | LanguageCode::Es => "es".to_string(),
                LanguageCode::FrFr | LanguageCode::Fr => "fr".to_string(),
                LanguageCode::DeDe | LanguageCode::De => "de".to_string(),
                LanguageCode::JaJp | LanguageCode::Ja => "ja".to_string(),
                LanguageCode::ZhCn => "zh".to_string(),
                LanguageCode::PtBr | LanguageCode::Pt => "pt".to_string(),
                LanguageCode::RuRu | LanguageCode::Ru => "ru".to_string(),
                LanguageCode::ItIt | LanguageCode::It => "it".to_string(),
                LanguageCode::KoKr | LanguageCode::Ko => "ko".to_string(),
                LanguageCode::Ar => "ar".to_string(),
                LanguageCode::Hi => "hi".to_string(),
                _ => "en".to_string(),
            }
        };
        let cultural_adaptation = self.cultural_model.calculate_linguistic_distance_factor(
            &get_language_code(reference_language),
            &get_language_code(target_language),
        );
        let adaptation_score = intelligibility_score * 0.4
            + phoneme_coverage.average_mapping_quality * 0.35
            + cultural_adaptation * 0.25;
        Ok(adaptation_score.max(0.0).min(1.0))
    }
    /// Evaluate acoustic consistency
    async fn evaluate_acoustic_consistency(
        &self,
        reference_characteristics: &VoiceCharacteristics,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_language: LanguageCode,
    ) -> EvaluationResult<HashMap<LanguageCode, f32>> {
        let mut consistency_scores = HashMap::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != reference_language {
                let target_characteristics =
                    self.extract_voice_characteristics_sync(target_audio, *target_language)?;
                let consistency_score = self.calculate_acoustic_consistency_score(
                    reference_characteristics,
                    &target_characteristics,
                    reference_language,
                    *target_language,
                );
                consistency_scores.insert(*target_language, consistency_score);
            }
        }
        Ok(consistency_scores)
    }
    /// Calculate acoustic consistency score
    fn calculate_acoustic_consistency_score(
        &self,
        reference_characteristics: &VoiceCharacteristics,
        target_characteristics: &VoiceCharacteristics,
        _reference_language: LanguageCode,
        _target_language: LanguageCode,
    ) -> f32 {
        let voice_quality_consistency = self.calculate_voice_quality_similarity(
            &reference_characteristics.voice_quality_stats,
            &target_characteristics.voice_quality_stats,
        );
        let spectral_consistency = self.calculate_spectral_character_preservation(
            &reference_characteristics.spectral_stats,
            &target_characteristics.spectral_stats,
        );
        let overall_consistency = voice_quality_consistency * 0.6 + spectral_consistency * 0.4;
        overall_consistency.max(0.0).min(1.0)
    }
    /// Evaluate perceptual similarity
    async fn evaluate_perceptual_similarity(
        &self,
        reference_audio: &AudioBuffer,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_language: LanguageCode,
    ) -> EvaluationResult<HashMap<LanguageCode, f32>> {
        let mut similarity_scores = HashMap::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != reference_language {
                let similarity_score = self
                    .calculate_perceptual_similarity_score(
                        reference_audio,
                        target_audio,
                        reference_language,
                        *target_language,
                    )
                    .await?;
                similarity_scores.insert(*target_language, similarity_score);
            }
        }
        Ok(similarity_scores)
    }
    /// Calculate perceptual similarity score
    async fn calculate_perceptual_similarity_score(
        &self,
        reference_audio: &AudioBuffer,
        target_audio: &AudioBuffer,
        reference_language: LanguageCode,
        target_language: LanguageCode,
    ) -> EvaluationResult<f32> {
        let cultural_profile = crate::perceptual::CulturalProfile {
            region: crate::perceptual::CulturalRegion::NorthAmerica,
            language_familiarity: vec![format!("{:?}", target_language).to_lowercase()],
            musical_training: false,
            accent_tolerance: 0.7,
        };
        let get_language_code = |lang: LanguageCode| -> String {
            match lang {
                LanguageCode::EnUs | LanguageCode::EnGb => "en".to_string(),
                LanguageCode::EsEs | LanguageCode::EsMx | LanguageCode::Es => "es".to_string(),
                LanguageCode::FrFr | LanguageCode::Fr => "fr".to_string(),
                LanguageCode::DeDe | LanguageCode::De => "de".to_string(),
                LanguageCode::JaJp | LanguageCode::Ja => "ja".to_string(),
                LanguageCode::ZhCn => "zh".to_string(),
                LanguageCode::PtBr | LanguageCode::Pt => "pt".to_string(),
                LanguageCode::RuRu | LanguageCode::Ru => "ru".to_string(),
                LanguageCode::ItIt | LanguageCode::It => "it".to_string(),
                LanguageCode::KoKr | LanguageCode::Ko => "ko".to_string(),
                LanguageCode::Ar => "ar".to_string(),
                LanguageCode::Hi => "hi".to_string(),
                _ => "en".to_string(),
            }
        };
        let demographic_profile = crate::perceptual::DemographicProfile {
            age_group: crate::perceptual::AgeGroup::MiddleAged,
            gender: crate::perceptual::Gender::Other,
            education_level: crate::perceptual::EducationLevel::Bachelor,
            native_language: get_language_code(target_language),
            audio_experience: crate::perceptual::ExperienceLevel::Intermediate,
        };
        let reference_adaptation = self.cultural_model.calculate_adaptation_factors(
            &cultural_profile,
            &demographic_profile,
            reference_audio,
            &get_language_code(reference_language),
        )?;
        let target_adaptation = self.cultural_model.calculate_adaptation_factors(
            &cultural_profile,
            &demographic_profile,
            target_audio,
            &get_language_code(target_language),
        )?;
        let adaptation_similarity =
            self.calculate_adaptation_factor_similarity(&reference_adaptation, &target_adaptation);
        Ok(adaptation_similarity)
    }
    /// Calculate adaptation factor similarity
    fn calculate_adaptation_factor_similarity(
        &self,
        reference_adaptation: &crate::perceptual::cross_cultural::CrossCulturalAdaptation,
        target_adaptation: &crate::perceptual::cross_cultural::CrossCulturalAdaptation,
    ) -> f32 {
        let phonetic_similarity = 1.0
            - (reference_adaptation.phonetic_distance_factor
                - target_adaptation.phonetic_distance_factor)
                .abs();
        let prosodic_similarity = 1.0
            - (reference_adaptation.prosodic_mismatch_factor
                - target_adaptation.prosodic_mismatch_factor)
                .abs();
        let communication_similarity = 1.0
            - (reference_adaptation.communication_style_factor
                - target_adaptation.communication_style_factor)
                .abs();
        (phonetic_similarity + prosodic_similarity + communication_similarity) / 3.0
    }
    /// Calculate cross-language similarity matrix
    async fn calculate_cross_language_similarity(
        &self,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
    ) -> EvaluationResult<HashMap<(LanguageCode, LanguageCode), f32>> {
        let mut similarity_matrix = HashMap::new();
        let languages: Vec<LanguageCode> = target_audios.keys().cloned().collect();
        for &lang1 in &languages {
            for &lang2 in &languages {
                if lang1 != lang2 {
                    if let (Some(audio1), Some(audio2)) =
                        (target_audios.get(&lang1), target_audios.get(&lang2))
                    {
                        let similarity = self
                            .calculate_perceptual_similarity_score(audio1, audio2, lang1, lang2)
                            .await?;
                        similarity_matrix.insert((lang1, lang2), similarity);
                    }
                }
            }
        }
        Ok(similarity_matrix)
    }
    /// Analyze voice characteristics
    async fn analyze_voice_characteristics(
        &self,
        reference_characteristics: &VoiceCharacteristics,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_language: LanguageCode,
    ) -> EvaluationResult<VoiceCharacteristicsAnalysis> {
        let f0_characteristics = self.analyze_f0_characteristics(
            reference_characteristics,
            target_audios,
            reference_language,
        )?;
        let formant_characteristics = FormantCharacteristicsAnalysis {
            formant_frequencies: HashMap::new(),
            formant_bandwidths: HashMap::new(),
            formant_consistency: 0.8,
            language_adaptations: HashMap::new(),
        };
        let spectral_characteristics = SpectralCharacteristicsAnalysis {
            spectral_centroid: HashMap::new(),
            spectral_spread: HashMap::new(),
            spectral_tilt: HashMap::new(),
            spectral_consistency: 0.7,
            language_adaptations: HashMap::new(),
        };
        let temporal_characteristics = TemporalCharacteristicsAnalysis {
            speaking_rate: HashMap::new(),
            pause_patterns: HashMap::new(),
            rhythm_characteristics: HashMap::new(),
            temporal_consistency: 0.75,
            language_adaptations: HashMap::new(),
        };
        let voice_quality_characteristics = VoiceQualityCharacteristicsAnalysis {
            jitter: HashMap::new(),
            shimmer: HashMap::new(),
            harmonic_to_noise_ratio: HashMap::new(),
            voice_quality_consistency: 0.85,
            language_adaptations: HashMap::new(),
        };
        let prosodic_characteristics = ProsodicCharacteristicsAnalysis {
            intonation_patterns: HashMap::new(),
            stress_patterns: HashMap::new(),
            prosodic_consistency: 0.7,
            language_adaptations: HashMap::new(),
        };
        Ok(VoiceCharacteristicsAnalysis {
            f0_characteristics,
            formant_characteristics,
            spectral_characteristics,
            temporal_characteristics,
            voice_quality_characteristics,
            prosodic_characteristics,
        })
    }
    /// Analyze F0 characteristics
    fn analyze_f0_characteristics(
        &self,
        reference_characteristics: &VoiceCharacteristics,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_language: LanguageCode,
    ) -> EvaluationResult<F0CharacteristicsAnalysis> {
        let mut mean_f0 = HashMap::new();
        let mut f0_range = HashMap::new();
        let mut f0_variability = HashMap::new();
        let mut language_adaptations = HashMap::new();
        mean_f0.insert(
            reference_language,
            reference_characteristics.f0_stats.mean_f0,
        );
        f0_range.insert(
            reference_language,
            reference_characteristics.f0_stats.f0_range,
        );
        f0_variability.insert(
            reference_language,
            reference_characteristics.f0_stats.f0_variability,
        );
        for (target_language, target_audio) in target_audios {
            if *target_language != reference_language {
                let target_characteristics =
                    self.extract_voice_characteristics_sync(target_audio, *target_language)?;
                mean_f0.insert(*target_language, target_characteristics.f0_stats.mean_f0);
                f0_range.insert(*target_language, target_characteristics.f0_stats.f0_range);
                f0_variability.insert(
                    *target_language,
                    target_characteristics.f0_stats.f0_variability,
                );
                let adaptation = self.analyze_f0_adaptation(
                    &reference_characteristics.f0_stats,
                    &target_characteristics.f0_stats,
                );
                language_adaptations.insert(*target_language, adaptation);
            }
        }
        let f0_consistency = self.calculate_f0_consistency(&mean_f0, &f0_variability);
        Ok(F0CharacteristicsAnalysis {
            mean_f0,
            f0_range,
            f0_variability,
            f0_consistency,
            language_adaptations,
        })
    }
    /// Analyze F0 adaptation
    fn analyze_f0_adaptation(
        &self,
        reference_stats: &F0Statistics,
        target_stats: &F0Statistics,
    ) -> F0Adaptation {
        let mean_difference = (target_stats.mean_f0 - reference_stats.mean_f0).abs();
        let variability_difference =
            (target_stats.f0_variability - reference_stats.f0_variability).abs();
        let adaptation_type = if mean_difference < 10.0 && variability_difference < 0.1 {
            F0AdaptationType::None
        } else if mean_difference > 50.0 {
            F0AdaptationType::MeanShifting
        } else if variability_difference > 0.3 {
            F0AdaptationType::RangeScaling
        } else {
            F0AdaptationType::ContourModification
        };
        let adaptation_magnitude =
            (mean_difference / reference_stats.mean_f0.max(1.0) + variability_difference).min(1.0);
        let adaptation_appropriateness = 1.0 - adaptation_magnitude;
        let adaptation_consistency = 0.8;
        F0Adaptation {
            adaptation_type,
            adaptation_magnitude,
            adaptation_appropriateness,
            adaptation_consistency,
        }
    }
    /// Calculate F0 consistency
    fn calculate_f0_consistency(
        &self,
        mean_f0: &HashMap<LanguageCode, f32>,
        f0_variability: &HashMap<LanguageCode, f32>,
    ) -> f32 {
        if mean_f0.len() < 2 {
            return 1.0;
        }
        let mean_values: Vec<f32> = mean_f0.values().cloned().collect();
        let variability_values: Vec<f32> = f0_variability.values().cloned().collect();
        let mean_variance = self.calculate_variance(&mean_values);
        let variability_variance = self.calculate_variance(&variability_values);
        let consistency = 1.0 - (mean_variance / 10000.0 + variability_variance).min(1.0);
        consistency.max(0.0).min(1.0)
    }
    /// Calculate variance
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance
    }
    /// Generate language adaptations
    async fn generate_language_adaptations(
        &self,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_language: LanguageCode,
        _phoneme_alignments: Option<&HashMap<LanguageCode, PhonemeAlignment>>,
    ) -> EvaluationResult<HashMap<LanguageCode, LanguageAdaptationResult>> {
        let mut adaptations = HashMap::new();
        for (target_language, target_audio) in target_audios {
            if *target_language != reference_language {
                let adaptation_result = self.generate_language_adaptation_result(
                    target_audio,
                    reference_language,
                    *target_language,
                )?;
                adaptations.insert(*target_language, adaptation_result);
            }
        }
        Ok(adaptations)
    }
    /// Generate language adaptation result
    fn generate_language_adaptation_result(
        &self,
        _target_audio: &AudioBuffer,
        reference_language: LanguageCode,
        target_language: LanguageCode,
    ) -> EvaluationResult<LanguageAdaptationResult> {
        let adaptation_quality = self.intelligibility_evaluator.predict_intelligibility(
            reference_language,
            target_language,
            None,
        );
        let specific_adaptations = vec![
            SpecificAdaptation {
                category: AdaptationCategory::Phonetic,
                description: "Phonetic adaptation for target language".to_string(),
                effectiveness: 0.8,
                consistency: 0.75,
            },
            SpecificAdaptation {
                category: AdaptationCategory::Prosodic,
                description: "Prosodic adaptation for target language".to_string(),
                effectiveness: 0.7,
                consistency: 0.8,
            },
        ];
        let adaptation_challenges = vec![AdaptationChallenge {
            challenge_type: AdaptationChallengeType::PhoneticIncompatibility,
            description: "Some phonemes cannot be perfectly mapped".to_string(),
            severity: 0.3,
            suggested_solutions: vec![
                "Use phonetic approximations".to_string(),
                "Implement language-specific acoustic models".to_string(),
            ],
        }];
        Ok(LanguageAdaptationResult {
            language: target_language,
            adaptation_quality,
            adaptation_consistency: 0.75,
            adaptation_appropriateness: 0.8,
            specific_adaptations,
            adaptation_challenges,
        })
    }
    /// Identify problematic language pairs
    fn identify_problematic_language_pairs(
        &self,
        voice_transfer_quality: &HashMap<LanguageCode, f32>,
        speaker_identity_preservation: &HashMap<LanguageCode, f32>,
        language_adaptation: &HashMap<LanguageCode, f32>,
        acoustic_consistency: &HashMap<LanguageCode, f32>,
        perceptual_similarity: &HashMap<LanguageCode, f32>,
        reference_language: LanguageCode,
    ) -> Vec<ProblematicLanguagePair> {
        let mut problematic_pairs = Vec::new();
        for language in voice_transfer_quality.keys() {
            let transfer_quality = voice_transfer_quality.get(language).unwrap_or(&0.5);
            let identity_preservation = speaker_identity_preservation.get(language).unwrap_or(&0.5);
            let adaptation_quality = language_adaptation.get(language).unwrap_or(&0.5);
            let consistency = acoustic_consistency.get(language).unwrap_or(&0.5);
            let similarity = perceptual_similarity.get(language).unwrap_or(&0.5);
            let overall_quality = (transfer_quality
                + identity_preservation
                + adaptation_quality
                + consistency
                + similarity)
                / 5.0;
            if overall_quality < 0.6 {
                let mut problem_types = Vec::new();
                if *transfer_quality < 0.5 {
                    problem_types.push(LanguagePairProblemType::VoiceTransferFailure);
                }
                if *identity_preservation < 0.5 {
                    problem_types.push(LanguagePairProblemType::SpeakerIdentityLoss);
                }
                if *adaptation_quality < 0.5 {
                    problem_types.push(LanguagePairProblemType::AdaptationInadequacy);
                }
                if *consistency < 0.5 {
                    problem_types.push(LanguagePairProblemType::AcousticInconsistency);
                }
                if *similarity < 0.5 {
                    problem_types.push(LanguagePairProblemType::PerceptualDissimilarity);
                }
                problematic_pairs.push(ProblematicLanguagePair {
                    source_language: reference_language,
                    target_language: *language,
                    problem_severity: 1.0 - overall_quality,
                    problem_types,
                    problem_description: format!(
                        "Poor multilingual speaker model performance for {:?} -> {:?}",
                        reference_language, language
                    ),
                    improvement_suggestions: vec![
                        "Improve cross-linguistic training data".to_string(),
                        "Enhance language-specific adaptation mechanisms".to_string(),
                        "Implement better voice transfer techniques".to_string(),
                    ],
                });
            }
        }
        problematic_pairs
    }
    /// Calculate overall quality
    fn calculate_overall_quality(
        &self,
        voice_transfer_quality: &HashMap<LanguageCode, f32>,
        speaker_identity_preservation: &HashMap<LanguageCode, f32>,
        language_adaptation: &HashMap<LanguageCode, f32>,
        acoustic_consistency: &HashMap<LanguageCode, f32>,
        perceptual_similarity: &HashMap<LanguageCode, f32>,
    ) -> f32 {
        let mut total_weighted_score = 0.0;
        let mut total_weight = 0.0;
        for language in voice_transfer_quality.keys() {
            let transfer_quality = voice_transfer_quality.get(language).unwrap_or(&0.5);
            let identity_preservation = speaker_identity_preservation.get(language).unwrap_or(&0.5);
            let adaptation_quality = language_adaptation.get(language).unwrap_or(&0.5);
            let consistency = acoustic_consistency.get(language).unwrap_or(&0.5);
            let similarity = perceptual_similarity.get(language).unwrap_or(&0.5);
            let language_score = transfer_quality * self.config.voice_transfer_quality_weight
                + identity_preservation * self.config.speaker_identity_preservation_weight
                + adaptation_quality * self.config.language_adaptation_weight
                + consistency * self.config.acoustic_consistency_weight
                + similarity * self.config.perceptual_similarity_weight;
            total_weighted_score += language_score;
            total_weight += self.config.voice_transfer_quality_weight
                + self.config.speaker_identity_preservation_weight
                + self.config.language_adaptation_weight
                + self.config.acoustic_consistency_weight
                + self.config.perceptual_similarity_weight;
        }
        if total_weight > 0.0 {
            total_weighted_score / total_weight
        } else {
            0.5
        }
    }
    /// Calculate evaluation confidence
    fn calculate_evaluation_confidence(
        &self,
        voice_transfer_quality: &HashMap<LanguageCode, f32>,
        speaker_identity_preservation: &HashMap<LanguageCode, f32>,
        language_adaptation: &HashMap<LanguageCode, f32>,
        acoustic_consistency: &HashMap<LanguageCode, f32>,
        perceptual_similarity: &HashMap<LanguageCode, f32>,
    ) -> f32 {
        let mut all_scores = Vec::new();
        for language in voice_transfer_quality.keys() {
            all_scores.push(*voice_transfer_quality.get(language).unwrap_or(&0.5));
            all_scores.push(*speaker_identity_preservation.get(language).unwrap_or(&0.5));
            all_scores.push(*language_adaptation.get(language).unwrap_or(&0.5));
            all_scores.push(*acoustic_consistency.get(language).unwrap_or(&0.5));
            all_scores.push(*perceptual_similarity.get(language).unwrap_or(&0.5));
        }
        if all_scores.is_empty() {
            return 0.5;
        }
        let mean_score = all_scores.iter().sum::<f32>() / all_scores.len() as f32;
        let variance = all_scores
            .iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f32>()
            / all_scores.len() as f32;
        let consistency = 1.0 - variance.sqrt();
        let confidence = consistency * 0.6 + mean_score * 0.4;
        confidence.max(0.1).min(1.0)
    }
    /// Get supported languages
    pub fn get_supported_languages(&self) -> Vec<LanguageCode> {
        self.config.target_languages.clone()
    }
    /// Get speaker model
    pub fn get_speaker_model(&self, speaker_id: &str) -> Option<&SpeakerModel> {
        self.speaker_models.get(speaker_id)
    }
    /// Get all speaker models
    pub fn get_all_speaker_models(&self) -> &HashMap<String, SpeakerModel> {
        &self.speaker_models
    }
}
