use crate::{types::VoiceSample, Error, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex;
use scirs2_fft::RealFftPlanner;
use std::collections::HashMap;

use super::types::*;

impl Default for AgeGenderAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl AgeGenderAdapter {
    /// Create new age/gender adapter with default configuration
    pub fn new() -> Self {
        Self {
            config: AgeGenderAdaptationConfig::default(),
            model_cache: HashMap::new(),
            analysis_cache: HashMap::new(),
        }
    }

    /// Create adapter with custom configuration
    pub fn with_config(config: AgeGenderAdaptationConfig) -> Self {
        Self {
            config,
            model_cache: HashMap::new(),
            analysis_cache: HashMap::new(),
        }
    }

    /// Train age/gender adaptation model from voice samples
    pub async fn train_adaptation_model(
        &mut self,
        speaker_id: &str,
        source_samples: &[VoiceSample],
        target: VoiceAdaptationTarget,
    ) -> Result<AgeGenderModel> {
        if source_samples.is_empty() {
            return Err(Error::InsufficientData(
                "No source samples provided for adaptation".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        // Analyze source voice characteristics
        let source_characteristics = self.analyze_voice_characteristics(source_samples).await?;

        // Generate transformation matrices
        let transformation_matrices =
            self.generate_transformation_matrices(&source_characteristics, &target)?;

        // Create adaptation model
        let model = AgeGenderModel {
            source_characteristics: source_characteristics.clone(),
            target,
            transformation_matrices,
            training_stats: ModelTrainingStats {
                training_samples: source_samples.len(),
                training_accuracy: 0.85, // Placeholder
                cv_score: 0.82,          // Placeholder
                complexity_score: 0.3,   // Placeholder
            },
        };

        // Cache the model
        self.model_cache
            .insert(speaker_id.to_string(), model.clone());
        self.analysis_cache
            .insert(speaker_id.to_string(), source_characteristics);

        println!(
            "Trained age/gender adaptation model for speaker {} in {:?}",
            speaker_id,
            start_time.elapsed()
        );

        Ok(model)
    }

    /// Apply age/gender adaptation to voice samples
    pub async fn adapt_voice(
        &self,
        model: &AgeGenderModel,
        input_samples: &[VoiceSample],
    ) -> Result<AgeGenderAdaptationResult> {
        if input_samples.is_empty() {
            return Err(Error::Processing(
                "No input samples provided for adaptation".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();
        let mut adapted_samples = Vec::new();
        let mut total_frames = 0;

        // Process each input sample
        for sample in input_samples {
            let adapted_audio = self.apply_adaptation_to_audio(
                &model.transformation_matrices,
                &sample.get_normalized_audio(),
                sample.sample_rate,
            )?;

            let adapted_sample = VoiceSample::new(
                format!("adapted_{}", sample.id),
                adapted_audio,
                sample.sample_rate,
            );

            total_frames += sample.get_normalized_audio().len();
            adapted_samples.push(adapted_sample);
        }

        // Analyze adapted characteristics
        let adapted_characteristics = self.analyze_voice_characteristics(&adapted_samples).await?;

        // Compute quality metrics
        let quality_metrics = self.compute_adaptation_quality_metrics(
            &model.source_characteristics,
            &adapted_characteristics,
            &model.target,
        )?;

        // Compute confidence score
        let confidence = self.compute_adaptation_confidence(&quality_metrics)?;

        let processing_time = start_time.elapsed();
        let result = AgeGenderAdaptationResult {
            success: quality_metrics.target_achievement > 0.6,
            adapted_characteristics,
            confidence,
            quality_metrics,
            processing_stats: AdaptationProcessingStats {
                processing_time,
                frames_processed: total_frames,
                memory_usage: self.estimate_memory_usage(),
                converged: true,
            },
        };

        Ok(result)
    }

    /// Analyze voice characteristics from samples
    pub(super) async fn analyze_voice_characteristics(
        &self,
        samples: &[VoiceSample],
    ) -> Result<VoiceCharacteristics> {
        if samples.is_empty() {
            return Err(Error::Processing(
                "No samples provided for analysis".to_string(),
            ));
        }

        // Aggregate audio from all samples
        let mut combined_audio = Vec::new();
        let sample_rate = samples[0].sample_rate;

        for sample in samples {
            combined_audio.extend_from_slice(&sample.get_normalized_audio());
        }

        // Extract F0 statistics
        let f0_statistics = self.extract_f0_statistics(&combined_audio, sample_rate)?;

        // Extract formant frequencies
        let formant_frequencies = self.extract_formant_frequencies(&combined_audio, sample_rate)?;

        // Extract voice quality metrics
        let voice_quality = self.extract_voice_quality_metrics(&combined_audio, sample_rate)?;

        // Extract spectral characteristics
        let spectral_characteristics =
            self.extract_spectral_characteristics(&combined_audio, sample_rate)?;

        // Estimate apparent age and gender
        let apparent_age =
            self.estimate_apparent_age(&f0_statistics, &formant_frequencies, &voice_quality)?;
        let gender_score = self.estimate_gender_score(&f0_statistics, &formant_frequencies)?;

        Ok(VoiceCharacteristics {
            apparent_age,
            gender_score,
            f0_statistics,
            formant_frequencies,
            voice_quality,
            spectral_characteristics,
        })
    }

    /// Generate transformation matrices for adaptation
    fn generate_transformation_matrices(
        &self,
        source: &VoiceCharacteristics,
        target: &VoiceAdaptationTarget,
    ) -> Result<TransformationMatrices> {
        // Calculate target characteristics
        let target_f0 = self.calculate_target_f0(&source.f0_statistics, target)?;
        let target_formants =
            self.calculate_target_formants(&source.formant_frequencies, target)?;

        // Generate F0 transformation curve
        let f0_transform = self.generate_f0_transformation_curve(
            source.f0_statistics.mean_f0,
            target_f0,
            target.age_intensity,
        );

        // Generate formant transformation matrix
        let formant_transform = self.generate_formant_transformation_matrix(
            &source.formant_frequencies,
            &target_formants,
            target.gender_intensity,
        );

        // Generate spectral transformation
        let spectral_transform =
            self.generate_spectral_transformation(&source.spectral_characteristics, target);

        // Generate quality transformation
        let quality_transform = self.generate_quality_transformation(&source.voice_quality, target);

        Ok(TransformationMatrices {
            f0_transform,
            formant_transform,
            spectral_transform,
            quality_transform,
        })
    }

    /// Apply adaptation transformations to audio
    fn apply_adaptation_to_audio(
        &self,
        transforms: &TransformationMatrices,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let mut adapted_audio = audio.to_vec();

        // Apply F0 transformation
        adapted_audio =
            self.apply_f0_transformation(&adapted_audio, &transforms.f0_transform, sample_rate)?;

        // Apply formant transformation
        adapted_audio = self.apply_formant_transformation(
            &adapted_audio,
            &transforms.formant_transform,
            sample_rate,
        )?;

        // Apply spectral transformation
        adapted_audio = self.apply_spectral_transformation(
            &adapted_audio,
            &transforms.spectral_transform,
            sample_rate,
        )?;

        // Apply quality transformation
        adapted_audio = self.apply_quality_transformation(
            &adapted_audio,
            &transforms.quality_transform,
            sample_rate,
        )?;

        // Apply smoothing
        if self.config.smoothness_factor > 0.0 {
            adapted_audio = self.apply_smoothing(&adapted_audio, self.config.smoothness_factor);
        }

        Ok(adapted_audio)
    }

    /// Extract F0 statistics from audio
    pub(super) fn extract_f0_statistics(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<F0Statistics> {
        if audio.len() < sample_rate as usize / 10 {
            return Err(Error::Processing(
                "Audio too short for F0 analysis".to_string(),
            ));
        }

        // Simplified F0 extraction using autocorrelation
        let frame_size = (sample_rate as f32 * 0.025) as usize; // 25ms frames
        let hop_size = (sample_rate as f32 * 0.010) as usize; // 10ms hop
        let mut f0_values = Vec::new();

        for i in (0..audio.len().saturating_sub(frame_size)).step_by(hop_size) {
            let frame = &audio[i..i + frame_size];
            let f0 = self.estimate_frame_f0(frame, sample_rate);
            if f0 > 50.0 && f0 < 500.0 {
                f0_values.push(f0);
            }
        }

        if f0_values.is_empty() {
            return Ok(F0Statistics {
                mean_f0: 0.0,
                f0_std: 0.0,
                f0_range: 0.0,
                jitter: 0.0,
            });
        }

        let mean_f0 = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
        let variance =
            f0_values.iter().map(|f| (f - mean_f0).powi(2)).sum::<f32>() / f0_values.len() as f32;
        let f0_std = variance.sqrt();
        let f0_range = f0_values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("f0_values is non-empty")
            - f0_values
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("f0_values is non-empty");

        // Calculate jitter
        let mut jitter_sum = 0.0;
        let mut jitter_count = 0;
        for i in 1..f0_values.len() {
            if f0_values[i - 1] > 0.0 && f0_values[i] > 0.0 {
                jitter_sum += (f0_values[i] - f0_values[i - 1]).abs() / f0_values[i - 1];
                jitter_count += 1;
            }
        }
        let jitter = if jitter_count > 0 {
            (jitter_sum / jitter_count as f32) * 100.0
        } else {
            0.0
        };

        Ok(F0Statistics {
            mean_f0,
            f0_std,
            f0_range,
            jitter,
        })
    }

    /// Estimate F0 for a single frame
    fn estimate_frame_f0(&self, frame: &[f32], sample_rate: u32) -> f32 {
        let min_period = sample_rate / 500; // 500 Hz max
        let max_period = sample_rate / 50; // 50 Hz min

        let mut max_corr = 0.0;
        let mut best_period = min_period;

        for period in min_period..max_period.min(frame.len() as u32 / 2) {
            let mut correlation = 0.0;
            let period_samples = period as usize;
            let mut count = 0;

            for i in 0..(frame.len() - period_samples) {
                correlation += frame[i] * frame[i + period_samples];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f32;
            }

            if correlation > max_corr {
                max_corr = correlation;
                best_period = period;
            }
        }

        if max_corr > 0.3 {
            sample_rate as f32 / best_period as f32
        } else {
            0.0
        }
    }

    /// Extract formant frequencies from audio.
    ///
    /// Formants are estimated as the strongest peaks of the real FFT magnitude
    /// spectrum within the typical formant band (200-4000 Hz). The peaks are
    /// ordered by ascending frequency and assigned to F1-F4. Bands for which no
    /// peak is found fall back to canonical average-adult-voice values so that
    /// downstream ratio computations remain well-defined.
    pub(super) fn extract_formant_frequencies(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<[f32; 4]> {
        // Canonical default formant values for an average adult voice.
        let mut formants = [500.0, 1500.0, 2500.0, 3500.0];

        // FFT-based spectral peak picking restricted to the formant band.
        let spectral_peaks = self.find_spectral_peaks(audio, sample_rate, 200.0, 4000.0, 4);
        for (i, peak) in spectral_peaks.iter().take(4).enumerate() {
            if *peak > 200.0 && *peak < 4000.0 {
                formants[i] = *peak;
            }
        }

        Ok(formants)
    }

    /// Compute the Hann-windowed real FFT magnitude spectrum of `audio`.
    ///
    /// Returns the `N/2 + 1` non-redundant magnitude bins, where `N` is the
    /// next power of two at least as large as the input length (zero-padded).
    pub(super) fn rfft_magnitude_spectrum(audio: &[f32]) -> (Vec<f32>, usize) {
        if audio.is_empty() {
            return (Vec::new(), 0);
        }
        // Use a power-of-two transform size for efficiency and stable
        // frequency resolution, capped to avoid pathological allocations.
        let n = audio.len().next_power_of_two().clamp(256, 1 << 16);

        let windowed: Vec<f64> = (0..n)
            .map(|i| {
                let sample = if i < audio.len() { audio[i] } else { 0.0 };
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
                sample as f64 * w
            })
            .collect();

        let num_bins = n / 2 + 1;
        let spectrum = scirs2_fft::rfft(&windowed, None)
            .unwrap_or_else(|_| vec![scirs2_core::Complex::new(0.0, 0.0); num_bins]);
        let mags: Vec<f32> = spectrum
            .iter()
            .take(num_bins)
            .map(|c| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
            .collect();
        (mags, n)
    }

    /// Find the strongest spectral peaks of `audio` within `[min_hz, max_hz]`.
    ///
    /// A peak is a local maximum of the magnitude spectrum whose value exceeds
    /// a fraction of the in-band maximum (an adaptive noise floor). At most
    /// `max_peaks` peaks are returned, ordered by ascending frequency.
    pub(super) fn find_spectral_peaks(
        &self,
        audio: &[f32],
        sample_rate: u32,
        min_hz: f32,
        max_hz: f32,
        max_peaks: usize,
    ) -> Vec<f32> {
        let (mags, n) = Self::rfft_magnitude_spectrum(audio);
        if mags.len() < 3 {
            return Vec::new();
        }
        let bin_hz = sample_rate as f32 / n as f32;
        let min_bin = ((min_hz / bin_hz).floor() as usize).max(1);
        let max_bin = ((max_hz / bin_hz).ceil() as usize).min(mags.len() - 2);
        if min_bin >= max_bin {
            return Vec::new();
        }

        // Adaptive threshold: a fraction of the maximum in-band magnitude.
        let in_band_max = mags[min_bin..=max_bin]
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let threshold = in_band_max * 0.1;

        // Collect local maxima above the threshold as (frequency, magnitude).
        let mut candidates: Vec<(f32, f32)> = Vec::new();
        for k in min_bin..=max_bin {
            let m = mags[k];
            if m > threshold && m >= mags[k - 1] && m >= mags[k + 1] {
                // Parabolic interpolation for sub-bin frequency precision.
                let (a, b, c) = (mags[k - 1], mags[k], mags[k + 1]);
                let denom = a - 2.0 * b + c;
                let delta = if denom.abs() > 1e-12 {
                    0.5 * (a - c) / denom
                } else {
                    0.0
                };
                let freq = (k as f32 + delta) * bin_hz;
                candidates.push((freq, m));
            }
        }

        // Keep the strongest peaks, then re-order by ascending frequency.
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(max_peaks);
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.into_iter().map(|(f, _)| f).collect()
    }

    /// Extract voice quality metrics
    pub(super) fn extract_voice_quality_metrics(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<VoiceQualityMetrics> {
        // Calculate breathiness (high-frequency noise ratio)
        let breathiness = self.calculate_breathiness(audio, sample_rate);

        // Calculate roughness (low-frequency modulation)
        let roughness = self.calculate_roughness(audio, sample_rate);

        // Calculate harmonics-to-noise ratio
        let hnr = self.calculate_hnr(audio, sample_rate);

        // Calculate spectral tilt
        let spectral_tilt = self.calculate_spectral_tilt(audio, sample_rate);

        Ok(VoiceQualityMetrics {
            breathiness,
            roughness,
            hnr,
            spectral_tilt,
        })
    }

    /// Calculate breathiness metric
    fn calculate_breathiness(&self, audio: &[f32], _sample_rate: u32) -> f32 {
        // Simplified breathiness calculation
        // Real implementation would use spectral analysis to measure noise in higher frequencies
        let high_freq_energy = audio
            .iter()
            .skip(audio.len() / 2)
            .map(|x| x * x)
            .sum::<f32>();
        let total_energy = audio.iter().map(|x| x * x).sum::<f32>();

        if total_energy > 0.0 {
            (high_freq_energy / total_energy).min(1.0)
        } else {
            0.0
        }
    }

    /// Calculate roughness metric
    fn calculate_roughness(&self, audio: &[f32], sample_rate: u32) -> f32 {
        // Look for amplitude modulations in the 20-50 Hz range typical of roughness
        let modulation_freq = 30.0; // Hz
        let samples_per_cycle = sample_rate as f32 / modulation_freq;

        if audio.len() < samples_per_cycle as usize * 2 {
            return 0.0;
        }

        let mut modulation_strength = 0.0;
        let window_size = samples_per_cycle as usize;

        for i in 0..(audio.len() - window_size) {
            let current_energy = audio[i..i + window_size].iter().map(|x| x * x).sum::<f32>();
            if i >= window_size {
                let prev_energy = audio[i - window_size..i].iter().map(|x| x * x).sum::<f32>();
                if prev_energy > 0.0 {
                    modulation_strength += ((current_energy - prev_energy) / prev_energy).abs();
                }
            }
        }

        (modulation_strength / (audio.len() - window_size) as f32).min(1.0)
    }

    /// Calculate harmonics-to-noise ratio
    fn calculate_hnr(&self, audio: &[f32], _sample_rate: u32) -> f32 {
        // Simplified HNR calculation
        // Real implementation would use autocorrelation-based harmonic analysis
        let signal_energy = audio.iter().map(|x| x * x).sum::<f32>();
        let noise_estimate =
            audio.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / audio.len() as f32;

        if noise_estimate > 0.0 {
            10.0 * (signal_energy / (noise_estimate * noise_estimate)).log10()
        } else {
            20.0 // High HNR for clean signal
        }
    }

    /// Calculate spectral tilt
    fn calculate_spectral_tilt(&self, audio: &[f32], _sample_rate: u32) -> f32 {
        // Simplified spectral tilt calculation
        // Real implementation would use FFT to measure energy distribution across frequencies
        let low_freq_energy = audio
            .iter()
            .take(audio.len() / 4)
            .map(|x| x * x)
            .sum::<f32>();
        let high_freq_energy = audio
            .iter()
            .skip(3 * audio.len() / 4)
            .map(|x| x * x)
            .sum::<f32>();

        if high_freq_energy > 0.0 && low_freq_energy > 0.0 {
            -10.0 * (high_freq_energy / low_freq_energy).log10()
        } else {
            0.0
        }
    }

    /// Extract spectral characteristics
    pub(super) fn extract_spectral_characteristics(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<SpectralCharacteristics> {
        // Frequency-domain spectral analysis: each scalar feature is derived
        // from the Hann-windowed real FFT magnitude spectrum (see
        // `rfft_magnitude_spectrum`), so values are expressed in physical units
        // (Hz) rather than sample indices.
        let spectral_centroid = self.calculate_spectral_centroid(audio, sample_rate);
        let spectral_rolloff = self.calculate_spectral_rolloff(audio, sample_rate);
        let spectral_flux = self.calculate_spectral_flux(audio, sample_rate);
        let high_freq_ratio = self.calculate_high_freq_ratio(audio);

        Ok(SpectralCharacteristics {
            spectral_centroid,
            spectral_rolloff,
            spectral_flux,
            high_freq_ratio,
        })
    }

    /// Calculate the spectral centroid in Hz.
    ///
    /// The centroid is the magnitude-weighted mean frequency of the spectrum,
    /// `Σ(f_k·|X_k|) / Σ|X_k|`, where `f_k = k·sample_rate/N` is the centre
    /// frequency of bin `k` and `|X_k|` is its magnitude. This is the
    /// "brightness" of the signal and is a perceptually meaningful frequency,
    /// unlike a sample-index average.
    pub(super) fn calculate_spectral_centroid(&self, audio: &[f32], sample_rate: u32) -> f32 {
        let (mags, n) = Self::rfft_magnitude_spectrum(audio);
        if mags.is_empty() || n == 0 {
            return 0.0;
        }
        let bin_hz = sample_rate as f32 / n as f32;

        let mut weighted_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;
        for (k, &magnitude) in mags.iter().enumerate() {
            let freq = k as f32 * bin_hz;
            weighted_sum += freq * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    /// Calculate the spectral rolloff frequency in Hz.
    ///
    /// The rolloff is the frequency below which 85% of the cumulative magnitude
    /// energy of the spectrum is contained. Walking the magnitude spectrum from
    /// DC upward and stopping once the running magnitude-energy reaches the
    /// threshold yields a frequency strictly below Nyquist for band-limited
    /// signals.
    pub(super) fn calculate_spectral_rolloff(&self, audio: &[f32], sample_rate: u32) -> f32 {
        let (mags, n) = Self::rfft_magnitude_spectrum(audio);
        if mags.is_empty() || n == 0 {
            return 0.0;
        }
        let bin_hz = sample_rate as f32 / n as f32;

        let total_energy: f32 = mags.iter().map(|m| m * m).sum();
        if total_energy <= 0.0 {
            return 0.0;
        }
        let threshold = total_energy * 0.85; // 85% energy threshold

        let mut cumulative_energy = 0.0f32;
        for (k, &magnitude) in mags.iter().enumerate() {
            cumulative_energy += magnitude * magnitude;
            if cumulative_energy >= threshold {
                return k as f32 * bin_hz;
            }
        }

        // Fallback: the highest analysed bin (the Nyquist frequency).
        (mags.len().saturating_sub(1)) as f32 * bin_hz
    }

    /// Calculate the spectral flux of the signal.
    ///
    /// Flux measures the frame-to-frame change in spectral shape. For each pair
    /// of successive (Hann-windowed) short-time FFT frames the half-wave
    /// rectified magnitude difference `Σ max(0, |X_k(t)| − |X_k(t−1)|)` is
    /// accumulated, then averaged over the frame transitions. A spectrally
    /// stationary tone yields near-zero flux, whereas a signal whose spectrum
    /// evolves over time yields a larger value.
    pub(super) fn calculate_spectral_flux(&self, audio: &[f32], sample_rate: u32) -> f32 {
        let frame_mags = Self::stft_frame_magnitudes(audio, sample_rate);
        if frame_mags.len() < 2 {
            return 0.0;
        }

        let mut total_flux = 0.0f32;
        for pair in frame_mags.windows(2) {
            let prev = &pair[0];
            let curr = &pair[1];
            let bins = prev.len().min(curr.len());
            let mut frame_flux = 0.0f32;
            for k in 0..bins {
                let diff = curr[k] - prev[k];
                if diff > 0.0 {
                    frame_flux += diff;
                }
            }
            total_flux += frame_flux;
        }

        total_flux / (frame_mags.len() - 1) as f32
    }

    /// Compute Hann-windowed magnitude spectra for successive STFT frames.
    ///
    /// The signal is split into fixed-size frames (~25 ms) with 50% overlap.
    /// Each frame is Hann-windowed and transformed with `scirs2_fft::rfft`,
    /// mirroring [`rfft_magnitude_spectrum`]. Returns one magnitude vector per
    /// frame; an empty result indicates the signal was too short for a frame.
    pub(super) fn stft_frame_magnitudes(audio: &[f32], sample_rate: u32) -> Vec<Vec<f32>> {
        if audio.is_empty() {
            return Vec::new();
        }
        // ~25 ms frames, clamped to a sane power-of-two transform size.
        let target = (sample_rate as usize / 40).max(256);
        let frame_len = target.next_power_of_two().clamp(256, 1 << 14);
        let hop = (frame_len / 2).max(1);
        let num_bins = frame_len / 2 + 1;

        // Precompute the Hann window once for all frames.
        let window: Vec<f64> = (0..frame_len)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (frame_len - 1) as f64).cos()
            })
            .collect();

        let mut frames: Vec<Vec<f32>> = Vec::new();
        let mut start = 0usize;
        while start < audio.len() {
            let windowed: Vec<f64> = (0..frame_len)
                .map(|i| {
                    let idx = start + i;
                    let sample = if idx < audio.len() { audio[idx] } else { 0.0 };
                    sample as f64 * window[i]
                })
                .collect();

            let spectrum = scirs2_fft::rfft(&windowed, None)
                .unwrap_or_else(|_| vec![Complex::new(0.0, 0.0); num_bins]);
            let mags: Vec<f32> = spectrum
                .iter()
                .take(num_bins)
                .map(|c| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
                .collect();
            frames.push(mags);

            // Stop once a frame would start beyond the available samples; the
            // last partial frame above is zero-padded so the tail is covered.
            if start + frame_len >= audio.len() {
                break;
            }
            start += hop;
        }

        frames
    }

    /// Calculate high frequency ratio
    fn calculate_high_freq_ratio(&self, audio: &[f32]) -> f32 {
        let split_point = audio.len() / 2;
        let low_energy = audio.iter().take(split_point).map(|x| x * x).sum::<f32>();
        let high_energy = audio.iter().skip(split_point).map(|x| x * x).sum::<f32>();
        let total_energy = low_energy + high_energy;

        if total_energy > 0.0 {
            high_energy / total_energy
        } else {
            0.0
        }
    }

    /// Estimate apparent age from voice characteristics
    pub(super) fn estimate_apparent_age(
        &self,
        f0_stats: &F0Statistics,
        formants: &[f32; 4],
        quality: &VoiceQualityMetrics,
    ) -> Result<f32> {
        // Age estimation based on acoustic correlates
        let mut age_score = 0.0;
        let mut weight_sum = 0.0;

        // F0-based age estimation (higher F0 generally indicates younger age)
        if f0_stats.mean_f0 > 0.0 {
            let f0_age_factor = if f0_stats.mean_f0 > 200.0 {
                // Higher F0 suggests younger age
                25.0 + (300.0 - f0_stats.mean_f0) * 0.2
            } else {
                // Lower F0 suggests older age
                35.0 + (200.0 - f0_stats.mean_f0) * 0.3
            };
            age_score += f0_age_factor * 0.4;
            weight_sum += 0.4;
        }

        // Formant-based age estimation (higher formants suggest smaller vocal tract/younger age)
        let formant_age_factor = if formants[0] > 600.0 || formants[1] > 1800.0 {
            // Higher formants suggest younger age
            20.0 + (2000.0 - formants[1]) * 0.01
        } else {
            // Lower formants suggest older age
            40.0 + (600.0 - formants[0]) * 0.05
        };
        age_score += formant_age_factor * 0.3;
        weight_sum += 0.3;

        // Voice quality-based age estimation
        let quality_age_factor = 30.0 + quality.roughness * 30.0 - quality.hnr * 0.5;
        age_score += quality_age_factor * 0.3;
        weight_sum += 0.3;

        if weight_sum > 0.0 {
            Ok((age_score / weight_sum).clamp(5.0, 80.0))
        } else {
            Ok(35.0) // Default age
        }
    }

    /// Estimate gender score from voice characteristics
    pub(super) fn estimate_gender_score(
        &self,
        f0_stats: &F0Statistics,
        formants: &[f32; 4],
    ) -> Result<f32> {
        let mut gender_score = 0.0;
        let mut weight_sum = 0.0;

        // F0-based gender estimation
        if f0_stats.mean_f0 > 0.0 {
            let f0_gender = if f0_stats.mean_f0 > 165.0 {
                // Higher F0 suggests feminine voice
                ((f0_stats.mean_f0 - 165.0) / 100.0).min(1.0)
            } else {
                // Lower F0 suggests masculine voice
                -((165.0 - f0_stats.mean_f0) / 80.0).min(1.0)
            };
            gender_score += f0_gender * 0.5;
            weight_sum += 0.5;
        }

        // Formant-based gender estimation
        let formant_gender = if formants[1] > 1400.0 {
            // Higher F2 suggests feminine voice
            ((formants[1] - 1400.0) / 600.0).min(1.0)
        } else {
            // Lower F2 suggests masculine voice
            -((1400.0 - formants[1]) / 400.0).min(1.0)
        };
        gender_score += formant_gender * 0.5;
        weight_sum += 0.5;

        if weight_sum > 0.0 {
            Ok((gender_score / weight_sum).clamp(-1.0, 1.0))
        } else {
            Ok(0.0) // Neutral
        }
    }

    /// Calculate target F0 based on adaptation parameters
    pub(super) fn calculate_target_f0(
        &self,
        source_f0: &F0Statistics,
        target: &VoiceAdaptationTarget,
    ) -> Result<f32> {
        let mut target_f0 = source_f0.mean_f0;

        // Age-based F0 adjustment
        let age_adjustment = match target.age {
            AgeCategory::Child => 50.0 * target.age_intensity,
            AgeCategory::Teenager => 30.0 * target.age_intensity,
            AgeCategory::YoungAdult => 10.0 * target.age_intensity,
            AgeCategory::Adult => 0.0,
            AgeCategory::MiddleAged => -10.0 * target.age_intensity,
            AgeCategory::Senior => -20.0 * target.age_intensity,
        };

        // Gender-based F0 adjustment
        let gender_adjustment = match target.gender {
            GenderCategory::Feminine => 40.0 * target.gender_intensity,
            GenderCategory::Neutral => 0.0,
            GenderCategory::Masculine => -30.0 * target.gender_intensity,
        };

        target_f0 += age_adjustment + gender_adjustment;
        target_f0 = target_f0.clamp(80.0, 400.0); // Reasonable F0 range

        Ok(target_f0)
    }

    /// Calculate target formants based on adaptation parameters
    fn calculate_target_formants(
        &self,
        source_formants: &[f32; 4],
        target: &VoiceAdaptationTarget,
    ) -> Result<[f32; 4]> {
        let mut target_formants = *source_formants;

        // Age-based formant adjustments
        let age_factors = match target.age {
            AgeCategory::Child => [1.3, 1.25, 1.2, 1.15],
            AgeCategory::Teenager => [1.15, 1.1, 1.08, 1.05],
            AgeCategory::YoungAdult => [1.05, 1.03, 1.02, 1.01],
            AgeCategory::Adult => [1.0, 1.0, 1.0, 1.0],
            AgeCategory::MiddleAged => [0.98, 0.97, 0.98, 0.98],
            AgeCategory::Senior => [0.95, 0.93, 0.95, 0.96],
        };

        // Gender-based formant adjustments
        let gender_factors = match target.gender {
            GenderCategory::Feminine => [1.1, 1.15, 1.1, 1.05],
            GenderCategory::Neutral => [1.0, 1.0, 1.0, 1.0],
            GenderCategory::Masculine => [0.9, 0.85, 0.9, 0.95],
        };

        for i in 0..4 {
            let age_factor = 1.0 + (age_factors[i] - 1.0) * target.age_intensity;
            let gender_factor = 1.0 + (gender_factors[i] - 1.0) * target.gender_intensity;
            target_formants[i] *= age_factor * gender_factor;
        }

        Ok(target_formants)
    }

    /// Generate F0 transformation curve
    fn generate_f0_transformation_curve(
        &self,
        source_f0: f32,
        target_f0: f32,
        intensity: f32,
    ) -> Array1<f32> {
        let curve_length = 1024; // Number of points in transformation curve
        let mut curve = Array1::zeros(curve_length);

        let f0_ratio = if source_f0 > 0.0 {
            target_f0 / source_f0
        } else {
            1.0
        };
        let final_ratio = 1.0 + (f0_ratio - 1.0) * intensity;

        for i in 0..curve_length {
            curve[i] = final_ratio;
        }

        curve
    }

    /// Generate formant transformation matrix
    fn generate_formant_transformation_matrix(
        &self,
        source_formants: &[f32; 4],
        target_formants: &[f32; 4],
        intensity: f32,
    ) -> Array2<f32> {
        let mut transform = Array2::eye(4);

        for i in 0..4 {
            if source_formants[i] > 0.0 {
                let ratio = target_formants[i] / source_formants[i];
                let final_ratio = 1.0 + (ratio - 1.0) * intensity;
                transform[[i, i]] = final_ratio;
            }
        }

        transform
    }

    /// Generate a per-bin spectral transformation envelope from the source
    /// spectral characteristics and the adaptation target.
    ///
    /// The envelope is the multiplicative gain applied to each FFT magnitude
    /// bin (indexed `0..512`, mapping linearly to `0..Nyquist`). It encodes two
    /// effects derived from the *target* age/gender relative to the *source*:
    ///
    /// * A spectral tilt that brightens (boosts high frequencies) when the
    ///   target favours a higher spectral centroid / high-frequency ratio than
    ///   the source, and darkens it otherwise. Younger and feminine targets are
    ///   generally brighter; senior and masculine targets are darker.
    /// * A broadband balance so the overall energy shift is bounded.
    ///
    /// The intensities scale the magnitude of the effect; an intensity of zero
    /// yields a flat (unity) envelope.
    pub(super) fn generate_spectral_transformation(
        &self,
        source_spectral: &SpectralCharacteristics,
        target: &VoiceAdaptationTarget,
    ) -> Array1<f32> {
        let transform_length = 512;
        let mut transform = Array1::ones(transform_length);

        // Target brightness factor in [-1, 1]: positive => brighter target.
        // Age: children/teens/young adults are brighter; senior darker.
        let age_brightness = match target.age {
            AgeCategory::Child => 0.6,
            AgeCategory::Teenager => 0.4,
            AgeCategory::YoungAdult => 0.15,
            AgeCategory::Adult => 0.0,
            AgeCategory::MiddleAged => -0.2,
            AgeCategory::Senior => -0.45,
        } * target.age_intensity;

        // Gender: feminine voices carry relatively more high-frequency energy.
        let gender_brightness = match target.gender {
            GenderCategory::Feminine => 0.45,
            GenderCategory::Neutral => 0.0,
            GenderCategory::Masculine => -0.35,
        } * target.gender_intensity;

        // Couple the target shift to how bright the source already is: a source
        // that is already very bright (high_freq_ratio near 1) needs less
        // additional boost, an already-dark source needs more. high_freq_ratio
        // is in [0, 1]; map to a [-0.5, 0.5] correction around the 0.5 midpoint.
        let source_correction = 0.5 - source_spectral.high_freq_ratio.clamp(0.0, 1.0);

        let tilt = (age_brightness + gender_brightness + 0.5 * source_correction).clamp(-1.0, 1.0);

        // Build a smooth tilt envelope: low frequencies and high frequencies
        // move in opposite directions about unity, with a tilt slope of up to
        // +/- `max_tilt_db` decibels per spectrum edge.
        let max_tilt_db = 9.0_f32;
        for i in 0..transform_length {
            // Normalised frequency in [0, 1].
            let f = i as f32 / (transform_length - 1) as f32;
            // Symmetric tilt: -1 at DC, +1 at Nyquist.
            let tilt_shape = 2.0 * f - 1.0;
            let gain_db = tilt * tilt_shape * max_tilt_db;
            transform[i] = 10.0_f32.powf(gain_db / 20.0).clamp(0.1, 10.0);
        }

        transform
    }

    /// Generate quality transformation
    fn generate_quality_transformation(
        &self,
        _source_quality: &VoiceQualityMetrics,
        _target: &VoiceAdaptationTarget,
    ) -> Array1<f32> {
        // Quality transformation parameters
        Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]) // [breathiness, roughness, hnr, spectral_tilt]
    }

    /// Apply F0 transformation to audio
    fn apply_f0_transformation(
        &self,
        audio: &[f32],
        f0_transform: &Array1<f32>,
        _sample_rate: u32,
    ) -> Result<Vec<f32>> {
        // Simplified F0 transformation using time-domain pitch shifting
        let shift_factor = f0_transform[0];
        let mut result = audio.to_vec();

        if (shift_factor - 1.0).abs() > 0.01 {
            // Apply simple pitch shifting by resampling
            let new_length = (audio.len() as f32 / shift_factor) as usize;
            let mut shifted = Vec::with_capacity(new_length);

            for i in 0..new_length {
                let source_idx = (i as f32 * shift_factor) as usize;
                if source_idx < audio.len() {
                    shifted.push(audio[source_idx]);
                } else {
                    shifted.push(0.0);
                }
            }

            // Resize to original length with interpolation
            result = Vec::with_capacity(audio.len());
            for i in 0..audio.len() {
                let shifted_idx = (i as f32 * new_length as f32 / audio.len() as f32) as usize;
                if shifted_idx < shifted.len() {
                    result.push(shifted[shifted_idx]);
                } else {
                    result.push(0.0);
                }
            }
        }

        Ok(result)
    }

    /// Apply a formant transformation to audio via STFT spectral warping.
    ///
    /// The `formant_transform` is a diagonal matrix whose `i`-th entry is the
    /// frequency-scaling ratio for formant `Fi` (as produced by
    /// [`Self::generate_formant_transformation_matrix`]). Each analysis frame is
    /// transformed with a real FFT; every magnitude bin in a formant's
    /// frequency band is relocated to `bin / ratio` (a formant shift), the
    /// warped spectrum is inverted, and the frames are recombined with
    /// overlap-add.
    pub(super) fn apply_formant_transformation(
        &self,
        audio: &[f32],
        formant_transform: &Array2<f32>,
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        const FRAME: usize = 1024;
        const HOP: usize = 256;
        if audio.len() < FRAME {
            return Ok(audio.to_vec());
        }
        let n_bins = FRAME / 2 + 1;
        let nyquist = sample_rate as f32 / 2.0;

        // Per-formant frequency-scaling ratios (diagonal of the matrix).
        let f1_shift = formant_transform[[0, 0]].clamp(0.5, 2.0);
        let f2_shift = formant_transform[[1, 1]].clamp(0.5, 2.0);
        let f3_shift = formant_transform[[2, 2]].clamp(0.5, 2.0);
        let f4_shift = formant_transform[[3, 3]].clamp(0.5, 2.0);

        // Formant band boundaries (Hz).
        let bands = [
            (300.0_f32, 900.0_f32, f1_shift),
            (900.0_f32, 2500.0_f32, f2_shift),
            (2500.0_f32, 3500.0_f32, f3_shift),
            (3500.0_f32, 4500.0_f32, f4_shift),
        ];

        let mut output = vec![0.0f32; audio.len()];
        let mut norm = vec![0.0f32; audio.len()];
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(FRAME);
        let inv = planner.plan_fft_inverse(FRAME);
        let window = Self::hann_window(FRAME);

        let mut start = 0usize;
        while start + FRAME <= audio.len() {
            let frame: Vec<f32> = audio[start..start + FRAME]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| s * w)
                .collect();
            let mut spectrum = vec![Complex::new(0.0f32, 0.0f32); n_bins];
            fwd.process(&frame, &mut spectrum)
                .map_err(|e| Error::Processing(e.to_string()))?;

            // Warp each band toward its shifted frequency.
            let mut warped = vec![Complex::new(0.0f32, 0.0f32); n_bins];
            for (k, &c) in spectrum.iter().enumerate() {
                let freq = k as f32 * nyquist / (n_bins - 1) as f32;
                let mut shift = 1.0f32;
                for &(lo, hi, s) in &bands {
                    if freq >= lo && freq < hi {
                        shift = s;
                        break;
                    }
                }
                // `shift` is the target/source formant ratio: energy at bin `k`
                // (frequency `freq`) is relocated to `shift * freq`, i.e. bin
                // `k * shift`, so a ratio > 1 raises the formant frequency.
                let target_bin = ((k as f32 * shift).round() as usize).min(n_bins - 1);
                warped[target_bin].re += c.re;
                warped[target_bin].im += c.im;
            }

            let mut out_frame = vec![0.0f32; FRAME];
            inv.process(&warped, &mut out_frame)
                .map_err(|e| Error::Processing(e.to_string()))?;
            let scale = 1.0 / FRAME as f32;
            for (j, (&w, &s)) in window.iter().zip(out_frame.iter()).enumerate() {
                if start + j < output.len() {
                    output[start + j] += s * w * scale;
                    norm[start + j] += w * w;
                }
            }
            start += HOP;
        }

        let mut result = audio.to_vec();
        for (i, r) in result.iter_mut().enumerate() {
            if norm[i] > 1e-8 {
                *r = output[i] / norm[i];
            }
        }
        Ok(result)
    }

    /// Apply a spectral envelope transformation to audio via STFT.
    ///
    /// `spectral_transform` is a per-bin multiplicative gain envelope (as
    /// produced by [`Self::generate_spectral_transformation`]). It is linearly
    /// interpolated to the FFT bin count; each analysis frame's magnitudes are
    /// scaled by the envelope (preserving phase) and recombined with
    /// overlap-add.
    pub(super) fn apply_spectral_transformation(
        &self,
        audio: &[f32],
        spectral_transform: &Array1<f32>,
        _sample_rate: u32,
    ) -> Result<Vec<f32>> {
        const FRAME: usize = 1024;
        const HOP: usize = 256;
        if audio.len() < FRAME || spectral_transform.is_empty() {
            return Ok(audio.to_vec());
        }
        let n_bins = FRAME / 2 + 1;
        let env_len = spectral_transform.len();

        // Interpolate the envelope onto the FFT bin grid.
        let env_bins: Vec<f32> = (0..n_bins)
            .map(|k| {
                let t = if n_bins > 1 {
                    k as f32 * (env_len - 1) as f32 / (n_bins - 1) as f32
                } else {
                    0.0
                };
                let lo = t as usize;
                let hi = (lo + 1).min(env_len - 1);
                let f = t - lo as f32;
                (spectral_transform[lo] * (1.0 - f) + spectral_transform[hi] * f).clamp(0.1, 10.0)
            })
            .collect();

        let mut output = vec![0.0f32; audio.len()];
        let mut norm = vec![0.0f32; audio.len()];
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(FRAME);
        let inv = planner.plan_fft_inverse(FRAME);
        let window = Self::hann_window(FRAME);

        let mut start = 0usize;
        while start + FRAME <= audio.len() {
            let frame: Vec<f32> = audio[start..start + FRAME]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| s * w)
                .collect();
            let mut spectrum = vec![Complex::new(0.0f32, 0.0f32); n_bins];
            fwd.process(&frame, &mut spectrum)
                .map_err(|e| Error::Processing(e.to_string()))?;

            for (c, &g) in spectrum.iter_mut().zip(&env_bins) {
                c.re *= g;
                c.im *= g;
            }

            let mut out_frame = vec![0.0f32; FRAME];
            inv.process(&spectrum, &mut out_frame)
                .map_err(|e| Error::Processing(e.to_string()))?;
            let scale = 1.0 / FRAME as f32;
            for (j, (&w, &s)) in window.iter().zip(out_frame.iter()).enumerate() {
                if start + j < output.len() {
                    output[start + j] += s * w * scale;
                    norm[start + j] += w * w;
                }
            }
            start += HOP;
        }

        let mut result = audio.to_vec();
        for (i, r) in result.iter_mut().enumerate() {
            if norm[i] > 1e-8 {
                *r = output[i] / norm[i];
            }
        }
        Ok(result)
    }

    /// Construct a periodic Hann analysis/synthesis window of length `size`.
    pub(super) fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                if size > 1 {
                    0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos()
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// Apply quality transformation to audio
    fn apply_quality_transformation(
        &self,
        audio: &[f32],
        _quality_transform: &Array1<f32>,
        _sample_rate: u32,
    ) -> Result<Vec<f32>> {
        // Simplified quality transformation
        // Real implementation would apply breathiness, roughness, and spectral tilt changes
        Ok(audio.to_vec())
    }

    /// Apply smoothing to audio
    fn apply_smoothing(&self, audio: &[f32], smoothness: f32) -> Vec<f32> {
        if smoothness <= 0.0 || audio.len() < 3 {
            return audio.to_vec();
        }

        let kernel_size = ((smoothness * 10.0) as usize).clamp(3, 21);
        let mut smoothed = Vec::with_capacity(audio.len());

        for i in 0..audio.len() {
            let start = i.saturating_sub(kernel_size / 2);
            let end = (i + kernel_size / 2 + 1).min(audio.len());

            let sum: f32 = audio[start..end].iter().sum();
            let count = end - start;
            smoothed.push(sum / count as f32);
        }

        smoothed
    }

    /// Compute adaptation quality metrics
    fn compute_adaptation_quality_metrics(
        &self,
        source: &VoiceCharacteristics,
        adapted: &VoiceCharacteristics,
        target: &VoiceAdaptationTarget,
    ) -> Result<AdaptationQualityMetrics> {
        // Naturalness score (how natural the adapted voice sounds)
        let naturalness = self.compute_naturalness_score(adapted)?;

        // Identity preservation score
        let identity_preservation = self.compute_identity_preservation_score(source, adapted)?;

        // Target achievement score
        let target_achievement = self.compute_target_achievement_score(adapted, target)?;

        // Audio quality score
        let audio_quality = self.compute_audio_quality_score(adapted)?;

        Ok(AdaptationQualityMetrics {
            naturalness,
            identity_preservation,
            target_achievement,
            audio_quality,
        })
    }

    /// Compute naturalness score
    fn compute_naturalness_score(&self, characteristics: &VoiceCharacteristics) -> Result<f32> {
        let mut naturalness: f32 = 1.0;

        // Check F0 naturalness
        if characteristics.f0_statistics.mean_f0 < 60.0
            || characteristics.f0_statistics.mean_f0 > 400.0
        {
            naturalness *= 0.5;
        }

        // Check formant naturalness
        for &formant in &characteristics.formant_frequencies {
            if formant < 200.0 || formant > 4000.0 {
                naturalness *= 0.8;
            }
        }

        // Check voice quality naturalness (jitter is in F0Statistics, not VoiceQualityMetrics)
        if characteristics.f0_statistics.jitter > 5.0 {
            naturalness *= 0.7;
        }

        Ok(naturalness.clamp(0.0, 1.0))
    }

    /// Compute identity preservation score
    fn compute_identity_preservation_score(
        &self,
        source: &VoiceCharacteristics,
        adapted: &VoiceCharacteristics,
    ) -> Result<f32> {
        let mut preservation = 1.0;

        // F0 similarity
        let f0_diff = (source.f0_statistics.mean_f0 - adapted.f0_statistics.mean_f0).abs();
        let f0_similarity = 1.0 - (f0_diff / source.f0_statistics.mean_f0).min(1.0);
        preservation *= 0.3 + 0.7 * f0_similarity;

        // Formant similarity
        let mut formant_similarity = 0.0;
        for i in 0..4 {
            let diff = (source.formant_frequencies[i] - adapted.formant_frequencies[i]).abs();
            formant_similarity += 1.0 - (diff / source.formant_frequencies[i]).min(1.0);
        }
        formant_similarity /= 4.0;
        preservation *= 0.5 + 0.5 * formant_similarity;

        Ok(preservation.clamp(0.0, 1.0))
    }

    /// Compute target achievement score
    fn compute_target_achievement_score(
        &self,
        adapted: &VoiceCharacteristics,
        target: &VoiceAdaptationTarget,
    ) -> Result<f32> {
        let mut achievement = 0.0;

        // Age target achievement
        let target_age_numeric = match target.age {
            AgeCategory::Child => 8.0,
            AgeCategory::Teenager => 16.0,
            AgeCategory::YoungAdult => 25.0,
            AgeCategory::Adult => 40.0,
            AgeCategory::MiddleAged => 55.0,
            AgeCategory::Senior => 70.0,
        };

        let age_diff = (adapted.apparent_age - target_age_numeric).abs();
        let age_achievement = 1.0 - (age_diff / 30.0).min(1.0);
        achievement += age_achievement * 0.5;

        // Gender target achievement
        let target_gender_numeric = match target.gender {
            GenderCategory::Masculine => -1.0,
            GenderCategory::Neutral => 0.0,
            GenderCategory::Feminine => 1.0,
        };

        let gender_diff = (adapted.gender_score - target_gender_numeric).abs();
        let gender_achievement = 1.0 - gender_diff;
        achievement += gender_achievement * 0.5;

        Ok(achievement.clamp(0.0, 1.0))
    }

    /// Compute audio quality score
    fn compute_audio_quality_score(&self, characteristics: &VoiceCharacteristics) -> Result<f32> {
        let mut quality: f32 = 1.0;

        // HNR-based quality assessment
        if characteristics.voice_quality.hnr > 10.0 {
            quality *= 1.0;
        } else if characteristics.voice_quality.hnr > 5.0 {
            quality *= 0.8;
        } else {
            quality *= 0.6;
        }

        // Jitter-based quality assessment
        if characteristics.f0_statistics.jitter < 2.0 {
            quality *= 1.0;
        } else if characteristics.f0_statistics.jitter < 5.0 {
            quality *= 0.8;
        } else {
            quality *= 0.6;
        }

        Ok(quality.clamp(0.0, 1.0))
    }

    /// Compute overall adaptation confidence
    fn compute_adaptation_confidence(
        &self,
        quality_metrics: &AdaptationQualityMetrics,
    ) -> Result<f32> {
        let confidence = (quality_metrics.naturalness * 0.3
            + quality_metrics.identity_preservation * 0.2
            + quality_metrics.target_achievement * 0.3
            + quality_metrics.audio_quality * 0.2)
            .clamp(0.0, 1.0);

        Ok(confidence)
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> usize {
        // Rough estimate of memory usage
        let base_size = std::mem::size_of::<Self>();
        let cache_size = self.model_cache.len() * 1024; // Approximate
        let analysis_cache_size = self.analysis_cache.len() * 512; // Approximate

        base_size + cache_size + analysis_cache_size
    }

    /// Get cached model for speaker
    pub fn get_cached_model(&self, speaker_id: &str) -> Option<&AgeGenderModel> {
        self.model_cache.get(speaker_id)
    }

    /// Clear model cache
    pub fn clear_cache(&mut self) {
        self.model_cache.clear();
        self.analysis_cache.clear();
    }
}
