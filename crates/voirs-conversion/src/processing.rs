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
                // Custom stages carry no built-in DSP transform; the signal is
                // passed through unchanged (bespoke behaviour is supplied elsewhere).
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
            // Short audio yields no analysis windows. Return a zero vector whose
            // length matches the normal path (4 spectral statistics + 13 MFCC
            // means = 17) so the descriptor length is always stable.
            return Ok(vec![0.0; 17]);
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

            // Compute 13 MFCCs (real triangular mel-filterbank + log + DCT-II;
            // see `compute_mel_spectrum`).
            let mfcc = self.compute_mel_spectrum(&spectrum, 13);
            mfccs.extend(mfcc);
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

        // Spectral flux (real half-wave-rectified L2 flux over Hann-windowed
        // FFT frames; see `compute_spectral_flux`).
        let spectral_flux = self.compute_spectral_flux(audio)?;
        features.push(spectral_flux);

        Ok(features)
    }

    /// Extract prosodic features
    fn extract_prosodic_features(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        // Fundamental frequency contour via normalized autocorrelation with
        // voicing detection and octave-error guarding (see
        // `estimate_f0_autocorrelation`).
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

        // Speaking rate via energy-onset detection (see
        // `estimate_speaking_rate`).
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

    /// Compute the mean half-wave-rectified L2 spectral flux of an audio signal.
    ///
    /// The signal is divided into overlapping 1024-sample Hann-windowed frames
    /// (256-sample hop) whose magnitude spectra are computed with
    /// `scirs2_fft` (via [`Self::compute_fft`]). For each pair of consecutive
    /// frames the half-wave-rectified L2 spectral flux is computed:
    ///
    /// ```text
    /// flux_t = sqrt( Σ_k max(|X_t[k]| − |X_{t-1}[k]|, 0)² )
    /// ```
    ///
    /// Each frame's flux is normalized by the mean magnitude of that frame so
    /// the measure is invariant to overall signal level. The mean flux across
    /// all consecutive frame pairs is returned. A steady tone yields a value
    /// near zero, whereas signals with abrupt spectral changes yield larger
    /// values.
    fn compute_spectral_flux(&self, audio: &[f32]) -> Result<f32> {
        const WINDOW_SIZE: usize = 1024;
        const HOP_SIZE: usize = 256;

        if audio.len() < WINDOW_SIZE * 2 {
            return Ok(0.0);
        }

        // Precompute the Hann window once for all frames.
        let hann: Vec<f32> = (0..WINDOW_SIZE)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (WINDOW_SIZE - 1) as f32).cos()
            })
            .collect();

        let mut prev_spectrum: Option<Vec<f32>> = None;
        let mut flux_values = Vec::new();

        let mut start = 0;
        while start + WINDOW_SIZE <= audio.len() {
            let windowed: Vec<f32> = audio[start..start + WINDOW_SIZE]
                .iter()
                .zip(hann.iter())
                .map(|(&x, &w)| x * w)
                .collect();
            let spectrum = self.compute_fft(&windowed)?;

            if let Some(prev) = &prev_spectrum {
                // Half-wave-rectified L2 spectral flux: only spectral increases
                // contribute, emphasising note/phone onsets.
                let sum_sq: f32 = spectrum
                    .iter()
                    .zip(prev.iter())
                    .map(|(&cur, &old)| {
                        let diff = (cur - old).max(0.0);
                        diff * diff
                    })
                    .sum();
                let flux = sum_sq.sqrt();

                // Normalize by the mean magnitude of the current frame so the
                // measure does not scale with signal loudness.
                let mean_mag = self.mean(&spectrum).max(1e-8);
                flux_values.push(flux / mean_mag);
            }

            prev_spectrum = Some(spectrum);
            start += HOP_SIZE;
        }

        Ok(self.mean(&flux_values))
    }

    /// Estimate a frame-by-frame fundamental-frequency (F0) contour.
    ///
    /// The autocorrelation pitch estimator needs an analysis window long
    /// enough to retain ample overlap even at the lowest tracked pitch
    /// (~80 Hz, whose period is `sample_rate / 80` samples). A `sample_rate /
    /// 20` (~50 ms) window keeps the overlap at that longest lag well above
    /// 50 %, which suppresses the spurious high-lag (tiny-overlap)
    /// correlations that a too-short frame would otherwise produce — so noise
    /// reads as unvoiced rather than as a phantom low pitch. A `sample_rate /
    /// 100` (~10 ms) hop preserves the contour's time resolution through
    /// overlapping frames.
    ///
    /// `pub(crate)` so other modules (e.g. `crate::acoustic::types`) can
    /// reuse this real autocorrelation-based estimator instead of
    /// hardcoding a constant F0.
    pub(crate) fn estimate_f0_contour(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let window = (self.sample_rate as usize / 20).max(2); // ~50 ms
        let hop = (self.sample_rate as usize / 100).max(1); // ~10 ms
        let mut f0_values = Vec::new();

        if audio.len() < window {
            // Shorter than one full window: fall back to a single estimate over
            // the whole clip, but only when it still leaves the autocorrelation
            // enough overlap (>= half a window) to be reliable.
            if audio.len() >= window / 2 {
                f0_values.push(self.estimate_f0_autocorrelation(audio));
            }
            return Ok(f0_values);
        }

        let mut start = 0;
        while start + window <= audio.len() {
            let frame = &audio[start..start + window];
            f0_values.push(self.estimate_f0_autocorrelation(frame));
            start += hop;
        }

        Ok(f0_values)
    }

    /// Estimate the fundamental frequency (F0) of a frame using normalized
    /// autocorrelation with voicing detection, octave-error guarding and
    /// parabolic peak interpolation.
    ///
    /// Algorithm:
    /// 1. The frame is mean-removed (DC offset is irrelevant to pitch).
    /// 2. The normalized autocorrelation
    ///    `r(τ) = Σ x[i]·x[i+τ] / sqrt(Σ x[i]² · Σ x[i+τ]²)` is evaluated for
    ///    every lag `τ` in the range corresponding to 80–500 Hz. The
    ///    normalization keeps `r(τ)` in `[-1, 1]`, where it doubles as a
    ///    voicing-strength measure.
    /// 3. **Voicing decision**: if the global maximum correlation is below
    ///    [`VOICING_THRESHOLD`], the frame is treated as unvoiced and `0.0`
    ///    is returned (also covers silence and white noise).
    /// 4. **Octave-error guarding**: a sub-multiple of the best lag may be the
    ///    true (shortest) period. The shortest sub-multiple period whose
    ///    correlation is still at least [`OCTAVE_FACTOR`] × the global maximum
    ///    is preferred, avoiding sub-harmonic ("too low") octave errors.
    /// 5. **Parabolic interpolation** around the chosen lag yields sub-sample
    ///    period accuracy.
    ///
    /// Returns the estimated F0 in Hz, or `0.0` for unvoiced / too-short frames.
    fn estimate_f0_autocorrelation(&self, frame: &[f32]) -> f32 {
        /// Minimum normalized autocorrelation for a frame to be voiced.
        const VOICING_THRESHOLD: f32 = 0.3;
        /// Octave-guard factor: prefer the shortest sub-multiple period whose
        /// correlation is at least this fraction of the global maximum.
        const OCTAVE_FACTOR: f32 = 0.9;
        /// Highest fundamental frequency considered (Hz).
        const MAX_F0_HZ: f32 = 500.0;
        /// Lowest fundamental frequency considered (Hz).
        const MIN_F0_HZ: f32 = 80.0;

        let sr = self.sample_rate as f32;
        if sr <= 0.0 {
            return 0.0;
        }

        let min_lag = (sr / MAX_F0_HZ).floor() as usize;
        let max_lag_raw = (sr / MIN_F0_HZ).ceil() as usize;
        let n = frame.len();
        if min_lag < 1 || n <= 2 * min_lag {
            return 0.0;
        }
        let max_lag = max_lag_raw.min(n - 1);
        if max_lag <= min_lag {
            return 0.0;
        }

        // Mean removal: pitch periodicity is independent of any DC component.
        let mean = frame.iter().sum::<f32>() / n as f32;
        let centered: Vec<f32> = frame.iter().map(|&x| x - mean).collect();

        // Reject silent frames before the (more expensive) lag search.
        let energy: f32 = centered.iter().map(|&x| x * x).sum();
        if energy < 1e-10 {
            return 0.0;
        }

        // Normalized autocorrelation over the candidate lag range.
        let mut correlations = vec![0.0f32; max_lag + 1];
        let mut global_best_corr = -1.0f32;
        let mut global_best_lag = min_lag;
        for lag in min_lag..=max_lag {
            let mut cross = 0.0f32;
            let mut left_sq = 0.0f32;
            let mut right_sq = 0.0f32;
            for i in 0..(n - lag) {
                let a = centered[i];
                let b = centered[i + lag];
                cross += a * b;
                left_sq += a * a;
                right_sq += b * b;
            }
            let denom = (left_sq * right_sq).sqrt();
            let r = if denom > 1e-10 { cross / denom } else { 0.0 };
            correlations[lag] = r;
            if r > global_best_corr {
                global_best_corr = r;
                global_best_lag = lag;
            }
        }

        // Voicing decision.
        if global_best_corr < VOICING_THRESHOLD {
            return 0.0;
        }

        // Octave-error guarding: prefer the shortest sub-multiple period that
        // is still strongly periodic.
        let octave_threshold = global_best_corr * OCTAVE_FACTOR;
        let mut best_lag = global_best_lag;
        for divisor in 2..=4 {
            let candidate = global_best_lag / divisor;
            if candidate >= min_lag && correlations[candidate] >= octave_threshold {
                best_lag = candidate;
            }
        }

        // Parabolic interpolation around the chosen lag for sub-sample accuracy.
        let refined_lag = if best_lag > min_lag && best_lag < max_lag {
            let alpha = correlations[best_lag - 1];
            let beta = correlations[best_lag];
            let gamma = correlations[best_lag + 1];
            let denom = alpha - 2.0 * beta + gamma;
            if denom.abs() > 1e-10 {
                let offset = (0.5 * (alpha - gamma) / denom).clamp(-1.0, 1.0);
                best_lag as f32 + offset
            } else {
                best_lag as f32
            }
        } else {
            best_lag as f32
        };

        if refined_lag > 0.0 {
            (sr / refined_lag).clamp(MIN_F0_HZ, MAX_F0_HZ)
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

    /// Estimate the speaking rate of an audio signal in syllables per second.
    ///
    /// Syllables are marked acoustically by energy onsets (the vowel nucleus
    /// rising after a consonant). The estimator follows the standard
    /// onset-detection pipeline:
    /// 1. Compute a short-time energy envelope over ~10 ms frames and apply log
    ///    compression so soft and loud syllables contribute comparably.
    /// 2. Derive an onset-detection function as the half-wave-rectified first
    ///    difference of the log-energy envelope; it peaks at sudden energy
    ///    increases (syllable onsets) and is zero during steady or decaying
    ///    regions.
    /// 3. Pick onsets as local maxima of the onset function that exceed an
    ///    adaptive threshold (`mean + 0.5·std` of the onset function), while
    ///    enforcing a minimum inter-onset interval of ~120 ms so that the rise
    ///    of a single syllable is not counted more than once.
    /// 4. Divide the onset count by the signal duration in seconds.
    ///
    /// Returns syllables per second (`0.0` for empty or very short input).
    fn estimate_speaking_rate(&self, audio: &[f32]) -> Result<f32> {
        /// Minimum inter-onset interval in seconds (~max 8 syllables/sec).
        const MIN_INTER_ONSET_SEC: f32 = 0.120;

        let sr = self.sample_rate as f32;
        if audio.is_empty() || sr <= 0.0 {
            return Ok(0.0);
        }

        // ~10 ms analysis frames for the energy envelope.
        let frame_size = (self.sample_rate as usize / 100).max(1);
        let num_frames = audio.len() / frame_size;
        if num_frames < 2 {
            return Ok(0.0);
        }

        // Log-compressed short-time energy envelope.
        let mut energy_env = Vec::with_capacity(num_frames);
        for f in 0..num_frames {
            let start = f * frame_size;
            let frame = &audio[start..start + frame_size];
            let energy = frame.iter().map(|&x| x * x).sum::<f32>() / frame_size as f32;
            energy_env.push((energy + 1e-10).ln());
        }

        // Onset-detection function: half-wave-rectified first difference.
        let mut onset_env = Vec::with_capacity(num_frames);
        onset_env.push(0.0);
        for i in 1..energy_env.len() {
            onset_env.push((energy_env[i] - energy_env[i - 1]).max(0.0));
        }

        // Adaptive threshold derived from the onset function statistics.
        let threshold = self.mean(&onset_env) + 0.5 * self.std(&onset_env);

        // Minimum inter-onset interval expressed in frames.
        let frame_period = frame_size as f32 / sr; // seconds per frame
        let min_gap_frames = ((MIN_INTER_ONSET_SEC / frame_period).round() as usize).max(1);

        // Peak picking with a refractory period.
        let mut onset_count = 0usize;
        let mut last_onset: Option<usize> = None;
        for i in 1..onset_env.len().saturating_sub(1) {
            let is_peak = onset_env[i] > threshold
                && onset_env[i] >= onset_env[i - 1]
                && onset_env[i] > onset_env[i + 1];
            if is_peak {
                let far_enough = match last_onset {
                    Some(prev) => i - prev >= min_gap_frames,
                    None => true,
                };
                if far_enough {
                    onset_count += 1;
                    last_onset = Some(i);
                }
            }
        }

        let duration_seconds = audio.len() as f32 / sr;
        if duration_seconds > 0.0 {
            Ok(onset_count as f32 / duration_seconds)
        } else {
            Ok(0.0)
        }
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

    /// Deterministic pseudo-random noise in [-1, 1) via a 32-bit xorshift PRNG.
    /// Kept deterministic (seeded) so the voicing test never flakes.
    fn white_noise(num_samples: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1; // avoid the zero fixed point
        (0..num_samples)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_f0_autocorrelation_sine_200hz() {
        let extractor = make_extractor();
        // 200 Hz sine; the period is ~110.25 samples at 22050 Hz.
        let audio = sine_wave(200.0, extractor.sample_rate, 2048);
        let f0 = extractor.estimate_f0_autocorrelation(&audio);
        assert!(
            (f0 - 200.0).abs() < 5.0,
            "Expected F0 near 200 Hz, got {f0:.2} Hz"
        );
    }

    #[test]
    fn test_f0_autocorrelation_unvoiced() {
        let extractor = make_extractor();

        // Silence carries no periodicity and must be unvoiced.
        let silence = vec![0.0_f32; 2048];
        let f0_silence = extractor.estimate_f0_autocorrelation(&silence);
        assert!(
            f0_silence.abs() < 1e-6,
            "Silence must be unvoiced (0.0), got {f0_silence:.2} Hz"
        );

        // White noise has no strong periodicity and must be unvoiced.
        let noise = white_noise(2048, 0x1234_5678);
        let f0_noise = extractor.estimate_f0_autocorrelation(&noise);
        assert!(
            f0_noise.abs() < 1e-6,
            "White noise must be unvoiced (0.0), got {f0_noise:.2} Hz"
        );
    }

    #[test]
    fn test_spectral_flux_steady_vs_changing() {
        let extractor = make_extractor();

        // Steady tone: consecutive frames are near-identical → low flux.
        let steady = sine_wave(440.0, extractor.sample_rate, 8192);
        let flux_steady = extractor.compute_spectral_flux(&steady).unwrap();

        // Signal that abruptly switches spectral content every 1024 samples.
        let sr = extractor.sample_rate as f32;
        let changing: Vec<f32> = (0..8192)
            .map(|i| {
                let freq = if (i / 1024) % 2 == 0 { 300.0 } else { 3000.0 };
                (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin()
            })
            .collect();
        let flux_changing = extractor.compute_spectral_flux(&changing).unwrap();

        assert!(
            flux_changing > flux_steady,
            "Flux of abruptly-changing signal ({flux_changing:.4}) should exceed steady tone ({flux_steady:.4})"
        );
    }

    #[test]
    fn test_speaking_rate_evenly_spaced_bursts() {
        let extractor = make_extractor();
        let sr = extractor.sample_rate as f32;

        // Build N evenly-spaced energy bursts. Each segment is `spacing_sec`
        // long: silence followed by a short tone, giving one clean onset per
        // segment, so the expected rate is num_bursts / total_duration.
        let num_bursts = 8usize;
        let spacing_sec = 0.3_f32;
        let segment_len = (spacing_sec * sr) as usize;
        let tone_len = (0.08 * sr) as usize; // 80 ms tone at the end of a segment

        let mut audio: Vec<f32> = Vec::with_capacity(num_bursts * segment_len);
        for _ in 0..num_bursts {
            let silence_len = segment_len - tone_len;
            let new_len = audio.len() + silence_len;
            audio.resize(new_len, 0.0);
            for i in 0..tone_len {
                audio.push(0.8 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr).sin());
            }
        }

        let rate = extractor.estimate_speaking_rate(&audio).unwrap();
        let total_sec = audio.len() as f32 / sr;
        let expected = num_bursts as f32 / total_sec; // ≈ 1 / spacing_sec ≈ 3.33

        assert!(
            (rate - expected).abs() < 0.8,
            "Expected speaking rate ~{expected:.2} syll/s, got {rate:.2} syll/s"
        );
    }

    /// Euclidean distance between two equal-length feature vectors.
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn test_spectral_centroid_bright_vs_low() {
        // A high-frequency tone must have a higher spectral centroid than a
        // low-frequency tone (the defining property of the centroid).
        let extractor = make_extractor();
        let low = sine_wave(200.0, extractor.sample_rate, 4096);
        let bright = sine_wave(5000.0, extractor.sample_rate, 4096);

        let low_centroid = extractor
            .compute_spectral_centroid(&extractor.compute_representative_spectrum(&low).unwrap());
        let bright_centroid = extractor.compute_spectral_centroid(
            &extractor.compute_representative_spectrum(&bright).unwrap(),
        );

        assert!(
            bright_centroid > low_centroid,
            "Bright-tone centroid ({bright_centroid:.1} Hz) should exceed low-tone centroid ({low_centroid:.1} Hz)"
        );
        // Sanity bounds: each centroid should sit near its tone's frequency band.
        assert!(
            bright_centroid > 2500.0,
            "Bright-tone centroid should be high, got {bright_centroid:.1} Hz"
        );
        assert!(
            low_centroid < 2000.0,
            "Low-tone centroid should be low, got {low_centroid:.1} Hz"
        );
    }

    #[test]
    fn test_mfcc_distinct_timbres_differ() {
        // MFCCs encode timbre, so two signals with the same pitch but very
        // different spectral envelopes must yield clearly different MFCCs.
        let extractor = make_extractor();
        let sr = extractor.sample_rate;

        // Timbre A: a pure 500 Hz tone (energy in one narrow region).
        let pure = sine_wave(500.0, sr, 4096);

        // Timbre B: a 500 Hz tone rich in odd harmonics (square-wave-like),
        // spreading energy across a very different spectral envelope.
        let rich: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut s = 0.0;
                for k in [1.0_f32, 3.0, 5.0, 7.0, 9.0] {
                    s += (1.0 / k) * (2.0 * std::f32::consts::PI * 500.0 * k * t).sin();
                }
                s * 0.5
            })
            .collect();

        let pure_mfcc = extractor.compute_mel_spectrum(
            &extractor.compute_representative_spectrum(&pure).unwrap(),
            13,
        );
        let rich_mfcc = extractor.compute_mel_spectrum(
            &extractor.compute_representative_spectrum(&rich).unwrap(),
            13,
        );

        // The transform is deterministic: distance to itself is exactly zero.
        assert!(
            l2_distance(&pure_mfcc, &pure_mfcc) < 1e-6,
            "MFCC extraction must be deterministic"
        );
        // Distinct timbres are well separated (measured distance ~110).
        let dist = l2_distance(&pure_mfcc, &rich_mfcc);
        assert!(
            dist > 5.0,
            "Distinct timbres should give well-separated MFCCs, got distance {dist:.4}"
        );
    }

    #[test]
    fn test_f0_contour_periodic_vs_noise() {
        // A periodic signal yields a stable, voiced F0 contour; white noise
        // yields an unvoiced one. This guards against the short-frame
        // autocorrelation artifact that would otherwise read noise as a
        // phantom low pitch.
        let extractor = make_extractor();
        let sr = extractor.sample_rate;

        let tone = sine_wave(150.0, sr, sr as usize); // 1 second
        let tone_contour = extractor.estimate_f0_contour(&tone).unwrap();
        assert!(
            tone_contour.len() > 10,
            "Expected a multi-frame contour, got {}",
            tone_contour.len()
        );
        let tone_voiced: Vec<f32> = tone_contour.iter().copied().filter(|&f| f > 0.0).collect();
        let tone_frac = tone_voiced.len() as f32 / tone_contour.len() as f32;
        assert!(
            tone_frac > 0.9,
            "A periodic tone should be voiced in nearly every frame, got fraction {tone_frac:.2}"
        );

        let tone_mean = tone_voiced.iter().sum::<f32>() / tone_voiced.len() as f32;
        assert!(
            (tone_mean - 150.0).abs() < 10.0,
            "Stable contour mean should be near 150 Hz, got {tone_mean:.1} Hz"
        );
        let tone_std = (tone_voiced
            .iter()
            .map(|f| (f - tone_mean).powi(2))
            .sum::<f32>()
            / tone_voiced.len() as f32)
            .sqrt();
        assert!(
            tone_std < 8.0,
            "Periodic contour should be stable (low std), got {tone_std:.2} Hz"
        );

        // White noise has no periodicity: the contour must be essentially unvoiced.
        let noise = white_noise(sr as usize, 0x0BAD_F00D);
        let noise_contour = extractor.estimate_f0_contour(&noise).unwrap();
        let noise_voiced = noise_contour.iter().filter(|&&f| f > 0.0).count();
        let noise_frac = noise_voiced as f32 / noise_contour.len().max(1) as f32;
        assert!(
            noise_frac < 0.2,
            "Noise F0 contour should be mostly unvoiced, got voiced fraction {noise_frac:.2}"
        );
        assert!(
            noise_frac < tone_frac,
            "Noise ({noise_frac:.2}) must be less voiced than a periodic tone ({tone_frac:.2})"
        );
    }

    #[test]
    fn test_spectral_features_stable_length() {
        // The spectral descriptor length must be identical for short audio
        // (no analysis windows) and normal audio (4 spectral stats + 13 MFCCs).
        let extractor = make_extractor();
        let short = sine_wave(440.0, extractor.sample_rate, 512); // < 1024 window
        let long = sine_wave(440.0, extractor.sample_rate, 8192);

        let short_features = extractor.extract_spectral_features(&short).unwrap();
        let long_features = extractor.extract_spectral_features(&long).unwrap();

        assert_eq!(
            short_features.len(),
            long_features.len(),
            "Short ({}) and long ({}) audio must yield equal-length spectral descriptors",
            short_features.len(),
            long_features.len()
        );
        assert_eq!(
            long_features.len(),
            17,
            "Expected 17 spectral features (4 stats + 13 MFCC means), got {}",
            long_features.len()
        );
    }
}
