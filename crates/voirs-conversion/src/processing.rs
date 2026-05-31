//! Audio processing utilities for voice conversion

use crate::{core::AudioFeatures, Error, Result};
use scirs2_core::Complex;
use scirs2_fft::RealFftPlanner;
use tracing::{debug, trace};

/// Audio buffer for processing
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Audio samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Buffer capacity
    pub capacity: usize,
    /// Current write position
    write_pos: usize,
    /// Ring buffer mode
    ring_buffer: bool,
}

impl AudioBuffer {
    /// Create new buffer
    pub fn new(capacity: usize, sample_rate: u32) -> Self {
        Self {
            samples: vec![0.0; capacity],
            sample_rate,
            capacity,
            write_pos: 0,
            ring_buffer: false,
        }
    }

    /// Create ring buffer
    pub fn new_ring_buffer(capacity: usize, sample_rate: u32) -> Self {
        let mut buffer = Self::new(capacity, sample_rate);
        buffer.ring_buffer = true;
        buffer
    }

    /// Add samples to buffer
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        if self.ring_buffer {
            for &sample in samples {
                self.samples[self.write_pos] = sample;
                self.write_pos = (self.write_pos + 1) % self.capacity;
            }
        } else {
            if self.samples.len() + samples.len() > self.capacity {
                return Err(Error::buffer("Buffer overflow".to_string()));
            }
            self.samples.extend_from_slice(samples);
        }
        Ok(())
    }

    /// Get samples and clear buffer
    pub fn drain(&mut self) -> Vec<f32> {
        if self.ring_buffer {
            let mut result = Vec::with_capacity(self.capacity);
            for i in 0..self.capacity {
                let idx = (self.write_pos + i) % self.capacity;
                result.push(self.samples[idx]);
                self.samples[idx] = 0.0;
            }
            result
        } else {
            std::mem::take(&mut self.samples)
        }
    }

    /// Get current buffer level
    pub fn level(&self) -> f32 {
        if self.ring_buffer {
            1.0 // Ring buffer is always "full"
        } else {
            self.samples.len() as f32 / self.capacity as f32
        }
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        if self.ring_buffer {
            self.samples.fill(0.0);
            self.write_pos = 0;
        } else {
            self.samples.clear();
        }
    }
}

/// Processing pipeline for audio
#[derive(Debug, Clone)]
pub struct ProcessingPipeline {
    /// Pipeline stages
    pub stages: Vec<ProcessingStage>,
    /// Pipeline configuration
    pub config: PipelineConfig,
}

/// Configuration for processing pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable parallel processing
    pub parallel: bool,
    /// Maximum concurrent stages
    pub max_concurrent: usize,
    /// Enable stage caching
    pub enable_caching: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            max_concurrent: 4,
            enable_caching: true,
        }
    }
}

impl ProcessingPipeline {
    /// Create new pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            config: PipelineConfig::default(),
        }
    }

    /// Create pipeline with configuration
    pub fn with_config(config: PipelineConfig) -> Self {
        Self {
            stages: Vec::new(),
            config,
        }
    }

    /// Add processing stage
    pub fn add_stage(&mut self, stage: ProcessingStage) {
        self.stages.push(stage);
    }

    /// Process audio through pipeline
    pub async fn process(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = input.to_vec();

        if self.config.parallel && self.stages.len() > 1 {
            // Parallel processing for independent stages
            for stage in &self.stages {
                if stage.can_run_parallel() {
                    output = stage.process(&output).await?;
                }
            }
        } else {
            // Sequential processing
            for stage in &self.stages {
                output = stage.process(&output).await?;
            }
        }

        Ok(output)
    }

    /// Get pipeline latency estimate
    pub fn estimated_latency_ms(&self, sample_rate: u32) -> f32 {
        self.stages
            .iter()
            .map(|stage| stage.estimated_latency_ms(sample_rate))
            .sum()
    }
}

impl Default for ProcessingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual processing stage
#[derive(Debug, Clone)]
pub struct ProcessingStage {
    /// Stage name
    pub name: String,
    /// Stage type
    pub stage_type: StageType,
    /// Stage parameters
    pub parameters: std::collections::HashMap<String, f32>,
    /// Enables parallel execution
    pub parallel_capable: bool,
}

/// Types of processing stages
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageType {
    /// Normalization stage
    Normalize,
    /// Noise reduction
    NoiseReduction,
    /// Filtering
    Filter,
    /// Resampling
    Resample,
    /// Compression
    Compression,
    /// Custom processing
    Custom(String),
}

impl ProcessingStage {
    /// Create new stage
    pub fn new(name: String, stage_type: StageType) -> Self {
        Self {
            name,
            stage_type,
            parameters: std::collections::HashMap::new(),
            parallel_capable: true,
        }
    }

    /// Set parameter
    pub fn with_parameter(mut self, key: String, value: f32) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Set parallel capability
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel_capable = parallel;
        self
    }

    /// Check if stage can run in parallel
    pub fn can_run_parallel(&self) -> bool {
        self.parallel_capable
    }

    /// Process audio in this stage
    pub async fn process(&self, input: &[f32]) -> Result<Vec<f32>> {
        trace!(
            "Processing stage: {} with {} samples",
            self.name,
            input.len()
        );

        match self.stage_type {
            StageType::Normalize => self.normalize(input),
            StageType::NoiseReduction => self.noise_reduction(input),
            StageType::Filter => self.filter(input),
            StageType::Resample => self.resample(input),
            StageType::Compression => self.compression(input),
            StageType::Custom(_) => {
                // Custom processing - placeholder
                Ok(input.to_vec())
            }
        }
    }

    /// Estimate processing latency
    pub fn estimated_latency_ms(&self, _sample_rate: u32) -> f32 {
        match self.stage_type {
            StageType::Normalize => 0.1,
            StageType::NoiseReduction => 2.0,
            StageType::Filter => 0.5,
            StageType::Resample => 1.0,
            StageType::Compression => 0.3,
            StageType::Custom(_) => 1.0,
        }
    }

    // Stage-specific processing methods

    fn normalize(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(input.to_vec());
        }

        let max_val = input
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);
        if max_val == 0.0 {
            return Ok(input.to_vec());
        }

        let target_level = self.parameters.get("target_level").copied().unwrap_or(0.9);
        let scale = target_level / max_val;

        Ok(input.iter().map(|x| x * scale).collect())
    }

    fn noise_reduction(&self, input: &[f32]) -> Result<Vec<f32>> {
        let noise_threshold = self
            .parameters
            .get("noise_threshold")
            .copied()
            .unwrap_or(0.01);

        Ok(input
            .iter()
            .map(|&x| {
                if x.abs() < noise_threshold {
                    x * 0.1 // Reduce low-level noise
                } else {
                    x
                }
            })
            .collect())
    }

    fn filter(&self, input: &[f32]) -> Result<Vec<f32>> {
        let cutoff = self.parameters.get("cutoff").copied().unwrap_or(0.5);

        // Simple low-pass filter
        let mut output = Vec::with_capacity(input.len());
        let mut prev = 0.0;

        for &sample in input {
            let filtered = prev + cutoff * (sample - prev);
            output.push(filtered);
            prev = filtered;
        }

        Ok(output)
    }

    fn resample(&self, input: &[f32]) -> Result<Vec<f32>> {
        let ratio = self.parameters.get("ratio").copied().unwrap_or(1.0);

        if ratio == 1.0 {
            return Ok(input.to_vec());
        }

        let output_len = (input.len() as f32 * ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = (i as f32 / ratio) as usize;
            if src_idx < input.len() {
                output.push(input[src_idx]);
            } else {
                output.push(0.0);
            }
        }

        Ok(output)
    }

    fn compression(&self, input: &[f32]) -> Result<Vec<f32>> {
        let ratio = self.parameters.get("ratio").copied().unwrap_or(4.0);
        let threshold = self.parameters.get("threshold").copied().unwrap_or(0.7);

        Ok(input
            .iter()
            .map(|&x| {
                let abs_x = x.abs();
                if abs_x > threshold {
                    let excess = abs_x - threshold;
                    let compressed_excess = excess / ratio;
                    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                    sign * (threshold + compressed_excess)
                } else {
                    x
                }
            })
            .collect())
    }
}

/// Feature extractor for audio analysis
pub struct FeatureExtractor {
    /// Sample rate for processing
    sample_rate: u32,
    /// FFT planner
    #[allow(dead_code)]
    fft_planner: RealFftPlanner<f32>,
    /// Feature cache
    cache: std::collections::HashMap<String, AudioFeatures>,
}

impl std::fmt::Debug for FeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureExtractor")
            .field("sample_rate", &self.sample_rate)
            .field("cache", &self.cache)
            .finish()
    }
}

impl FeatureExtractor {
    /// Create new feature extractor
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            fft_planner: RealFftPlanner::<f32>::new(),
            cache: std::collections::HashMap::new(),
        }
    }

    /// Extract comprehensive audio features
    pub async fn extract_features(&self, audio: &[f32], sample_rate: u32) -> Result<AudioFeatures> {
        debug!(
            "Extracting features from {} samples at {} Hz",
            audio.len(),
            sample_rate
        );

        // Resample if necessary
        let processed_audio = if sample_rate != self.sample_rate {
            self.resample_audio(audio, sample_rate, self.sample_rate)?
        } else {
            audio.to_vec()
        };

        // Extract different feature types
        let spectral = self.extract_spectral_features(&processed_audio)?;
        let temporal = self.extract_temporal_features(&processed_audio)?;
        let prosodic = self.extract_prosodic_features(&processed_audio)?;
        let speaker_embedding = None; // Would require neural network

        let quality = self.compute_quality_features(&processed_audio)?;
        let formants = self.compute_formant_features(&processed_audio)?;
        let harmonics = self.compute_harmonic_features(&processed_audio)?;

        Ok(AudioFeatures {
            spectral,
            temporal,
            prosodic,
            speaker_embedding,
            quality,
            formants,
            harmonics,
        })
    }

    /// Compute a representative Hann-windowed FFT frame from audio.
    /// Returns the magnitude spectrum of the first full window.
    fn compute_representative_spectrum(&self, audio: &[f32]) -> Result<Vec<f32>> {
        const WINDOW_SIZE: usize = 1024;
        if audio.len() < WINDOW_SIZE {
            // Pad with zeros for short audio
            let mut padded = audio.to_vec();
            padded.resize(WINDOW_SIZE, 0.0);
            let windowed: Vec<f32> = padded
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let hann = 0.5
                        - 0.5
                            * (2.0 * std::f32::consts::PI * i as f32 / (WINDOW_SIZE - 1) as f32)
                                .cos();
                    x * hann
                })
                .collect();
            return self.compute_fft(&windowed);
        }
        let window = &audio[..WINDOW_SIZE];
        let windowed: Vec<f32> = window
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let hann = 0.5
                    - 0.5
                        * (2.0 * std::f32::consts::PI * i as f32 / (WINDOW_SIZE - 1) as f32).cos();
                x * hann
            })
            .collect();
        self.compute_fft(&windowed)
    }

    /// Compute SNR-based voice quality estimate as a 1-element Vec in [0, 1].
    fn compute_quality_features(&self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.is_empty() {
            return Ok(vec![0.0]);
        }
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        let spectrum = self.compute_representative_spectrum(audio)?;

        // Noise floor: median of bottom 20% of spectrum magnitudes
        let bottom_count = ((spectrum.len() as f32 * 0.2) as usize).max(1);
        let mut sorted_mags: Vec<f32> = spectrum[..bottom_count].to_vec();
        sorted_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_floor = sorted_mags[sorted_mags.len() / 2];

        let snr_db = 20.0 * (rms / (noise_floor + 1e-8)).log10();
        let snr_normalized = snr_db.clamp(0.0, 60.0) / 60.0;

        Ok(vec![snr_normalized])
    }

    /// Find peak-energy spectrum bin in a frequency band [low_hz, high_hz].
    /// Returns frequency in Hz of the peak bin.
    fn peak_freq_in_band(&self, spectrum: &[f32], low_hz: f32, high_hz: f32) -> f32 {
        let sr = self.sample_rate as f32;
        let n = spectrum.len() as f32;
        let low_bin = ((low_hz * n * 2.0 / sr) as usize).min(spectrum.len().saturating_sub(1));
        let high_bin = ((high_hz * n * 2.0 / sr) as usize).min(spectrum.len().saturating_sub(1));

        let mut peak_mag = 0.0_f32;
        let mut peak_bin = low_bin;

        for k in low_bin..=high_bin {
            if spectrum[k] > peak_mag {
                peak_mag = spectrum[k];
                peak_bin = k;
            }
        }

        peak_bin as f32 * sr / (2.0 * spectrum.len() as f32)
    }

    /// Compute formant frequencies [F1_hz, F2_hz, F3_hz] via spectral peak-picking.
    fn compute_formant_features(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let spectrum = self.compute_representative_spectrum(audio)?;

        let f1 = self.peak_freq_in_band(&spectrum, 300.0, 900.0);
        let f2 = self.peak_freq_in_band(&spectrum, 900.0, 2500.0);
        let f3 = self.peak_freq_in_band(&spectrum, 2500.0, 3500.0);

        Ok(vec![f1, f2, f3])
    }

    /// Compute HNR-proxy via autocorrelation of the first frame.
    /// Returns a 1-element Vec with the normalized peak correlation in [0, 1).
    fn compute_harmonic_features(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let sr = self.sample_rate as usize;
        let frame_len = (sr / 20).min(audio.len()); // 50 ms frame max
        if frame_len == 0 {
            return Ok(vec![0.0]);
        }
        let frame = &audio[..frame_len];

        let zero_lag: f32 = frame.iter().map(|x| x * x).sum();
        if zero_lag < 1e-12 {
            return Ok(vec![0.0]);
        }

        let lag_min = sr / 500; // min period ~ 500 Hz
        let lag_max = (sr / 50).min(frame_len / 2); // max period ~ 50 Hz

        if lag_min >= lag_max {
            return Ok(vec![0.0]);
        }

        let mut peak_corr = 0.0_f32;
        for lag in lag_min..lag_max {
            let corr: f32 = frame[..frame_len - lag]
                .iter()
                .zip(frame[lag..].iter())
                .map(|(a, b)| a * b)
                .sum();
            if corr > peak_corr {
                peak_corr = corr;
            }
        }

        let hnr = (peak_corr / zero_lag).clamp(0.0, 0.999);
        Ok(vec![hnr])
    }

    /// Extract spectral features
    fn extract_spectral_features(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        // Window parameters
        let window_size = 1024;
        let hop_size = 512;

        if audio.len() < window_size {
            return Ok(vec![0.0; 13]); // Return zero features for short audio
        }

        // Process windows
        let mut spectral_centroids = Vec::new();
        let mut spectral_rolloffs = Vec::new();
        let mut mfccs = Vec::new();

        for window_start in (0..audio.len() - window_size).step_by(hop_size) {
            let window = &audio[window_start..window_start + window_size];

            // Apply window function (Hann window)
            let windowed: Vec<f32> = window
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let hann = 0.5
                        - 0.5
                            * (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32)
                                .cos();
                    x * hann
                })
                .collect();

            // Compute FFT
            let spectrum = self.compute_fft(&windowed)?;

            // Extract spectral features
            spectral_centroids.push(self.compute_spectral_centroid(&spectrum));
            spectral_rolloffs.push(self.compute_spectral_rolloff(&spectrum, 0.85));

            // Compute MFCCs (simplified)
            let mel_spectrum = self.compute_mel_spectrum(&spectrum, 13);
            mfccs.extend(mel_spectrum);
        }

        // Aggregate features
        features.push(self.mean(&spectral_centroids)); // Spectral centroid mean
        features.push(self.std(&spectral_centroids)); // Spectral centroid std
        features.push(self.mean(&spectral_rolloffs)); // Spectral rolloff mean
        features.push(self.std(&spectral_rolloffs)); // Spectral rolloff std

        // Add MFCC statistics (first 13 coefficients)
        if !mfccs.is_empty() {
            let chunk_size = 13;
            for i in 0..chunk_size {
                let coeff_values: Vec<f32> =
                    mfccs.iter().skip(i).step_by(chunk_size).copied().collect();
                features.push(self.mean(&coeff_values));
            }
        } else {
            features.extend(vec![0.0; 13]);
        }

        Ok(features)
    }

    /// Extract temporal features
    fn extract_temporal_features(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        // RMS energy
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        features.push(rms);

        // Zero crossing rate
        let zcr = audio
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count() as f32
            / (audio.len() - 1) as f32;
        features.push(zcr);

        // Energy contour statistics
        let frame_size = self.sample_rate as usize / 100; // 10ms frames
        let mut energy_contour = Vec::new();

        for chunk in audio.chunks(frame_size) {
            let energy = chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32;
            energy_contour.push(energy.sqrt());
        }

        features.push(self.mean(&energy_contour));
        features.push(self.std(&energy_contour));

        // Spectral flux (simplified)
        let spectral_flux = self.compute_spectral_flux(audio)?;
        features.push(spectral_flux);

        Ok(features)
    }

    /// Extract prosodic features
    fn extract_prosodic_features(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        // Fundamental frequency estimation (simplified autocorrelation)
        let f0_values = self.estimate_f0_contour(audio)?;

        if !f0_values.is_empty() {
            features.push(self.mean(&f0_values)); // Mean F0
            features.push(self.std(&f0_values)); // F0 variance
            features.push(f0_values.iter().copied().reduce(f32::max).unwrap_or(0.0)); // Max F0
            features.push(f0_values.iter().copied().reduce(f32::min).unwrap_or(0.0));
        // Min F0
        } else {
            features.extend(vec![0.0; 4]);
        }

        // Intensity contour
        let intensity_values = self.compute_intensity_contour(audio);
        features.push(self.mean(&intensity_values));
        features.push(self.std(&intensity_values));

        // Speaking rate estimate (simplified)
        let speaking_rate = self.estimate_speaking_rate(audio)?;
        features.push(speaking_rate);

        Ok(features)
    }

    // Helper methods for feature extraction

    fn resample_audio(&self, audio: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(audio.to_vec());
        }

        let ratio = to_rate as f32 / from_rate as f32;
        let output_len = (audio.len() as f32 * ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f32 / ratio;
            let idx = src_idx as usize;

            if idx + 1 < audio.len() {
                // Linear interpolation
                let frac = src_idx - idx as f32;
                let sample = audio[idx] * (1.0 - frac) + audio[idx + 1] * frac;
                output.push(sample);
            } else if idx < audio.len() {
                output.push(audio[idx]);
            } else {
                output.push(0.0);
            }
        }

        Ok(output)
    }

    fn compute_fft(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(audio.len());

        let input = audio.to_vec();
        let mut output = vec![Complex::new(0.0, 0.0); audio.len() / 2 + 1];

        fft.process(&input, &mut output)
            .map_err(|e| Error::processing(e.to_string()))?;

        Ok(output.iter().map(|c| c.norm()).collect())
    }

    fn compute_spectral_centroid(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let freq = i as f32 * self.sample_rate as f32 / (2.0 * spectrum.len() as f32);
            weighted_sum += freq * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    fn compute_spectral_rolloff(&self, spectrum: &[f32], rolloff_point: f32) -> f32 {
        let total_energy: f32 = spectrum.iter().map(|x| x * x).sum();
        let target_energy = total_energy * rolloff_point;

        let mut cumulative_energy = 0.0;
        for (i, &magnitude) in spectrum.iter().enumerate() {
            cumulative_energy += magnitude * magnitude;
            if cumulative_energy >= target_energy {
                return i as f32 * self.sample_rate as f32 / (2.0 * spectrum.len() as f32);
            }
        }

        (spectrum.len() - 1) as f32 * self.sample_rate as f32 / (2.0 * spectrum.len() as f32)
    }

    fn compute_mel_spectrum(&self, spectrum: &[f32], num_coeffs: usize) -> Vec<f32> {
        // Real triangular mel filterbank MFCC pipeline (26 filters).
        const NUM_FILTERS: usize = 26;
        let f_low: f32 = 80.0;
        let f_high: f32 = self.sample_rate as f32 / 2.0;

        let mel_low = self.hz_to_mel(f_low);
        let mel_high = self.hz_to_mel(f_high);

        // Compute NUM_FILTERS + 2 center frequencies evenly spaced in mel domain.
        // Index 0 and NUM_FILTERS+1 are the outer edges; 1..=NUM_FILTERS are filter centers.
        let mut hz_centers = vec![0.0_f32; NUM_FILTERS + 2];
        for (i, center) in hz_centers.iter_mut().enumerate() {
            let mel_val = mel_low + i as f32 * (mel_high - mel_low) / (NUM_FILTERS + 1) as f32;
            *center = self.mel_to_hz(mel_val);
        }

        // Accumulate triangular filter energies.
        let mut filter_energies = vec![0.0_f32; NUM_FILTERS];
        let sr = self.sample_rate as f32;
        let num_bins = spectrum.len();

        for (k, &mag) in spectrum.iter().enumerate() {
            let freq_k = k as f32 * sr / (2.0 * num_bins as f32);

            // Filter i (1-indexed) uses hz_centers[i-1], hz_centers[i], hz_centers[i+1].
            for i in 1..=NUM_FILTERS {
                let left = hz_centers[i - 1];
                let center = hz_centers[i];
                let right = hz_centers[i + 1];

                let weight = if (left..=center).contains(&freq_k) {
                    let denom = center - left;
                    if denom > 0.0 {
                        (freq_k - left) / denom
                    } else {
                        0.0
                    }
                } else if (center..=right).contains(&freq_k) {
                    let denom = right - center;
                    if denom > 0.0 {
                        (right - freq_k) / denom
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                filter_energies[i - 1] += weight * mag;
            }
        }

        // Apply log compression.
        let log_energies: Vec<f32> = filter_energies.iter().map(|&e| (e + 1e-8).ln()).collect();

        // Apply DCT-II and return first num_coeffs coefficients.
        let dct_out = self.apply_dct(&log_energies);
        let take = num_coeffs.min(dct_out.len());
        let mut out = dct_out[..take].to_vec();
        // Pad with zeros if num_coeffs > NUM_FILTERS
        out.resize(num_coeffs, 0.0);
        out
    }

    fn hz_to_mel(&self, hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(&self, mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn apply_dct(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mut output = vec![0.0; n];

        for (k, output_value) in output.iter_mut().enumerate().take(n) {
            let mut sum = 0.0;
            for (i, &input_value) in input.iter().enumerate().take(n) {
                sum += input_value
                    * (std::f32::consts::PI * k as f32 * (i as f32 + 0.5) / n as f32).cos();
            }
            *output_value = sum;
        }

        output
    }

    fn compute_spectral_flux(&self, audio: &[f32]) -> Result<f32> {
        // Simplified spectral flux computation
        let window_size = 1024;
        let hop_size = 512;

        if audio.len() < window_size * 2 {
            return Ok(0.0);
        }

        let mut flux_values = Vec::new();

        for i in (hop_size..audio.len() - window_size).step_by(hop_size) {
            let window1 = &audio[i - hop_size..i - hop_size + window_size];
            let window2 = &audio[i..i + window_size];

            let spectrum1 = self.compute_fft(window1)?;
            let spectrum2 = self.compute_fft(window2)?;

            let flux: f32 = spectrum1
                .iter()
                .zip(spectrum2.iter())
                .map(|(s1, s2)| (s2 - s1).max(0.0))
                .sum();

            flux_values.push(flux);
        }

        Ok(self.mean(&flux_values))
    }

    fn estimate_f0_contour(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let frame_size = self.sample_rate as usize / 100; // 10ms frames
        let mut f0_values = Vec::new();

        for chunk in audio.chunks(frame_size) {
            if chunk.len() < frame_size / 2 {
                continue;
            }

            let f0 = self.estimate_f0_autocorrelation(chunk);
            f0_values.push(f0);
        }

        Ok(f0_values)
    }

    fn estimate_f0_autocorrelation(&self, frame: &[f32]) -> f32 {
        let min_period = self.sample_rate / 500; // 500 Hz max
        let max_period = self.sample_rate / 50; // 50 Hz min

        let mut max_correlation = 0.0;
        let mut best_period = min_period;

        for period in min_period..max_period.min(frame.len() as u32 / 2) {
            let mut correlation = 0.0;
            let period_samples = period as usize;

            for i in 0..(frame.len() - period_samples) {
                correlation += frame[i] * frame[i + period_samples];
            }

            if correlation > max_correlation {
                max_correlation = correlation;
                best_period = period;
            }
        }

        if max_correlation > 0.0 {
            self.sample_rate as f32 / best_period as f32
        } else {
            0.0
        }
    }

    fn compute_intensity_contour(&self, audio: &[f32]) -> Vec<f32> {
        let frame_size = self.sample_rate as usize / 100; // 10ms frames
        let mut intensity_values = Vec::new();

        for chunk in audio.chunks(frame_size) {
            let intensity = chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32;
            intensity_values.push(intensity.sqrt());
        }

        intensity_values
    }

    fn estimate_speaking_rate(&self, audio: &[f32]) -> Result<f32> {
        // Simple syllable counting based on energy peaks
        let intensity = self.compute_intensity_contour(audio);
        let threshold = self.mean(&intensity) * 1.2;

        let mut peak_count = 0;
        let mut in_peak = false;

        for &value in &intensity {
            if value > threshold && !in_peak {
                peak_count += 1;
                in_peak = true;
            } else if value <= threshold {
                in_peak = false;
            }
        }

        let duration_seconds = audio.len() as f32 / self.sample_rate as f32;
        Ok(peak_count as f32 / duration_seconds * 60.0) // Peaks per minute
    }

    fn mean(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }

    fn std(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = self.mean(values);
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;

        variance.sqrt()
    }
}

/// Signal processor for audio manipulation
#[derive(Debug)]
pub struct SignalProcessor {
    /// Buffer size for processing
    #[allow(dead_code)]
    buffer_size: usize,
    /// Processing cache
    #[allow(dead_code)]
    cache: std::collections::HashMap<String, Vec<f32>>,
}

impl SignalProcessor {
    /// Create new signal processor
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            cache: std::collections::HashMap::new(),
        }
    }

    /// Normalize audio to target level
    pub fn normalize(&self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.is_empty() {
            return Ok(audio.to_vec());
        }

        let max_val = audio
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);
        if max_val == 0.0 {
            return Ok(audio.to_vec());
        }

        let scale = 0.95 / max_val;
        Ok(audio.iter().map(|x| x * scale).collect())
    }

    /// Apply noise reduction
    pub fn denoise(&self, audio: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        // Simple spectral gating
        let noise_threshold = 0.02;

        Ok(audio
            .iter()
            .map(|&x| {
                if x.abs() < noise_threshold {
                    x * 0.1
                } else {
                    x
                }
            })
            .collect())
    }

    /// Resample audio
    pub fn resample(&self, audio: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(audio.to_vec());
        }

        let ratio = to_rate as f32 / from_rate as f32;
        let output_len = (audio.len() as f32 * ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f32 / ratio;
            let idx = src_idx as usize;

            if idx + 1 < audio.len() {
                let frac = src_idx - idx as f32;
                let sample = audio[idx] * (1.0 - frac) + audio[idx + 1] * frac;
                output.push(sample);
            } else if idx < audio.len() {
                output.push(audio[idx]);
            } else {
                output.push(0.0);
            }
        }

        Ok(output)
    }

    /// Apply smoothing filter
    pub fn smooth(&self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.len() < 3 {
            return Ok(audio.to_vec());
        }

        let mut output = Vec::with_capacity(audio.len());
        output.push(audio[0]);

        for i in 1..audio.len() - 1 {
            let smoothed = (audio[i - 1] + 2.0 * audio[i] + audio[i + 1]) / 4.0;
            output.push(smoothed);
        }

        output.push(audio[audio.len() - 1]);
        Ok(output)
    }

    /// Apply dynamic range compression
    pub fn compress(&self, audio: &[f32], ratio: f32) -> Result<Vec<f32>> {
        let threshold = 0.7;

        Ok(audio
            .iter()
            .map(|&x| {
                let abs_x = x.abs();
                if abs_x > threshold {
                    let excess = abs_x - threshold;
                    let compressed_excess = excess / ratio;
                    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                    sign * (threshold + compressed_excess)
                } else {
                    x
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests_mel {
    use super::*;

    fn make_extractor() -> FeatureExtractor {
        FeatureExtractor::new(22050)
    }

    /// Generate a pure sine wave at `freq_hz` Hz with `sample_rate` and `num_samples` samples.
    fn sine_wave(freq_hz: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_mel_spectrum_shape() {
        let extractor = make_extractor();
        // Use a simple flat spectrum as input.
        let spectrum = vec![1.0_f32; 513]; // 1024-point FFT → 513 bins
        for num_coeffs in [1_usize, 13, 26, 30] {
            let result = extractor.compute_mel_spectrum(&spectrum, num_coeffs);
            assert_eq!(
                result.len(),
                num_coeffs,
                "Expected output length {num_coeffs}, got {}",
                result.len()
            );
        }
    }

    #[test]
    fn test_formant_extraction_sine() {
        // A 1 kHz sine wave should place most energy in the F2 band (900–2500 Hz).
        let extractor = make_extractor();
        let audio = sine_wave(1000.0, extractor.sample_rate, 4096);
        let spectrum = extractor.compute_representative_spectrum(&audio).unwrap();

        // The F2 band (900–2500 Hz) peak should be near 1000 Hz.
        let f2 = extractor.peak_freq_in_band(&spectrum, 900.0, 2500.0);
        assert!(
            (f2 - 1000.0).abs() < 100.0,
            "Expected F2 near 1000 Hz, got {f2:.1} Hz"
        );

        // The formant extraction convenience method should return a 3-element Vec.
        let formants = extractor.compute_formant_features(&audio).unwrap();
        assert_eq!(formants.len(), 3, "Expected 3 formants");
    }

    #[test]
    fn test_quality_snr_positive() {
        // A loud sine wave should have strictly positive normalized SNR.
        let extractor = make_extractor();
        let audio = sine_wave(440.0, extractor.sample_rate, 4096);
        let quality = extractor.compute_quality_features(&audio).unwrap();
        assert_eq!(quality.len(), 1, "Expected 1-element quality Vec");
        let snr = quality[0];
        assert!(
            snr > 0.0,
            "Expected positive SNR for a pure sine wave, got {snr}"
        );
        assert!(snr <= 1.0, "SNR should be normalised to [0,1], got {snr}");
    }
}
