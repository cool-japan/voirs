//! Speaker verification for voice cloning

use crate::{
    embedding::{SpeakerEmbedding, SpeakerEmbeddingExtractor},
    preprocessing::AudioPreprocessor,
    types::{SpeakerProfile, VoiceSample},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Verification result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether verification passed
    pub verified: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Verification threshold used
    pub threshold: f32,
    /// Distance/similarity score
    pub score: f32,
    /// Additional verification metrics
    pub metrics: VerificationMetrics,
    /// Processing time
    pub processing_time: Duration,
    /// Verification method used
    pub method: VerificationMethod,
}

/// Additional verification metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationMetrics {
    /// Embedding similarity score
    pub embedding_similarity: f32,
    /// Acoustic similarity score
    pub acoustic_similarity: f32,
    /// Prosodic similarity score  
    pub prosodic_similarity: f32,
    /// Quality scores for both samples
    pub sample_qualities: (f32, f32),
    /// False acceptance rate estimate
    pub far_estimate: f32,
    /// False rejection rate estimate
    pub frr_estimate: f32,
}

/// Verification methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMethod {
    /// Embedding-based verification
    EmbeddingOnly,
    /// Audio sample comparison
    SampleComparison,
    /// Multi-modal verification (embedding + audio)
    MultiModal,
    /// Deep verification with neural networks
    DeepVerification,
}

impl VerificationResult {
    /// Create new verification result
    pub fn new(
        verified: bool,
        confidence: f32,
        threshold: f32,
        score: f32,
        metrics: VerificationMetrics,
        processing_time: Duration,
        method: VerificationMethod,
    ) -> Self {
        Self {
            verified,
            confidence,
            threshold,
            score,
            metrics,
            processing_time,
            method,
        }
    }

    /// Create a simple verification result (for backward compatibility)
    pub fn simple(verified: bool, confidence: f32, threshold: f32, score: f32) -> Self {
        Self {
            verified,
            confidence,
            threshold,
            score,
            metrics: VerificationMetrics::default(),
            processing_time: Duration::default(),
            method: VerificationMethod::EmbeddingOnly,
        }
    }
}

impl Default for VerificationMetrics {
    fn default() -> Self {
        Self {
            embedding_similarity: 0.0,
            acoustic_similarity: 0.0,
            prosodic_similarity: 0.0,
            sample_qualities: (0.0, 0.0),
            far_estimate: 0.01, // 1% false acceptance rate
            frr_estimate: 0.05, // 5% false rejection rate
        }
    }
}

/// Speaker verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Main verification threshold
    pub threshold: f32,
    /// Strict threshold for high-security applications
    pub strict_threshold: f32,
    /// Minimum sample duration (seconds)
    pub min_sample_duration: f32,
    /// Maximum acceptable noise level
    pub max_noise_level: f32,
    /// Enable multi-modal verification
    pub enable_multimodal: bool,
    /// Enable deep verification
    pub enable_deep_verification: bool,
    /// Verification method to use
    pub method: VerificationMethod,
    /// Cache verification results
    pub enable_caching: bool,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            strict_threshold: 0.9,
            min_sample_duration: 1.0,
            max_noise_level: 0.1,
            enable_multimodal: true,
            enable_deep_verification: false,
            method: VerificationMethod::MultiModal,
            enable_caching: true,
        }
    }
}

/// Speaker verifier with comprehensive verification capabilities
#[derive(Debug)]
pub struct SpeakerVerifier {
    /// Configuration
    config: VerificationConfig,
    /// Embedding extractor
    embedding_extractor: SpeakerEmbeddingExtractor,
    /// Audio preprocessor
    preprocessor: AudioPreprocessor,
    /// Verification cache
    cache: HashMap<String, VerificationResult>,
    /// Performance metrics
    metrics: VerificationSystemMetrics,
}

/// System-wide verification metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSystemMetrics {
    /// Total verification attempts
    pub total_verifications: u64,
    /// Successful verifications
    pub successful_verifications: u64,
    /// Failed verifications
    pub failed_verifications: u64,
    /// False acceptances (if ground truth available)
    pub false_acceptances: u64,
    /// False rejections (if ground truth available)
    pub false_rejections: u64,
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Average confidence scores
    pub avg_confidence: f32,
}

impl SpeakerVerifier {
    /// Create new verifier
    pub fn new(config: VerificationConfig) -> Result<Self> {
        let embedding_extractor =
            SpeakerEmbeddingExtractor::new(crate::embedding::EmbeddingConfig::default())?;
        let preprocessor = AudioPreprocessor::new(16000); // 16kHz sample rate

        Ok(Self {
            config,
            embedding_extractor,
            preprocessor,
            cache: HashMap::new(),
            metrics: VerificationSystemMetrics::new(),
        })
    }

    /// Create verifier with default configuration
    pub fn with_default_config() -> Result<Self> {
        Self::new(VerificationConfig::default())
    }

    /// Verify speaker from embeddings
    pub fn verify_embeddings(
        &self,
        reference: &SpeakerEmbedding,
        test: &SpeakerEmbedding,
    ) -> Result<VerificationResult> {
        let start_time = Instant::now();

        info!("Performing embedding-based speaker verification");

        // Validate embeddings
        if !reference.is_valid() || !test.is_valid() {
            return Err(Error::Verification(
                "Invalid speaker embeddings provided".to_string(),
            ));
        }

        if reference.dimension != test.dimension {
            return Err(Error::Verification(
                "Embedding dimensions don't match".to_string(),
            ));
        }

        // Compute similarity
        let similarity = reference.similarity(test);
        let verified = similarity >= self.config.threshold;

        // Build metrics
        let mut metrics = VerificationMetrics::default();
        metrics.embedding_similarity = similarity;
        metrics.sample_qualities = (reference.confidence, test.confidence);

        // Estimate error rates based on similarity and threshold
        metrics.far_estimate =
            self.estimate_false_acceptance_rate(similarity, self.config.threshold);
        metrics.frr_estimate =
            self.estimate_false_rejection_rate(similarity, self.config.threshold);

        let processing_time = start_time.elapsed();

        Ok(VerificationResult::new(
            verified,
            similarity,
            self.config.threshold,
            similarity,
            metrics,
            processing_time,
            VerificationMethod::EmbeddingOnly,
        ))
    }

    /// Verify speaker from voice samples with comprehensive analysis
    pub async fn verify_samples(
        &mut self,
        reference: &VoiceSample,
        test: &VoiceSample,
    ) -> Result<VerificationResult> {
        let start_time = Instant::now();

        info!("Performing comprehensive speaker verification from audio samples");

        // Validate samples
        self.validate_samples(reference, test)?;

        // Check cache if enabled
        if self.config.enable_caching {
            let cache_key = format!("{}_{}", reference.id, test.id);
            if let Some(cached_result) = self.cache.get(&cache_key) {
                debug!("Using cached verification result");
                return Ok(cached_result.clone());
            }
        }

        // Preprocess samples
        let processed_ref = self.preprocessor.preprocess(reference).await?;
        let processed_test = self.preprocessor.preprocess(test).await?;

        // Extract embeddings
        let ref_embedding = self.embedding_extractor.extract(&processed_ref).await?;
        let test_embedding = self.embedding_extractor.extract(&processed_test).await?;

        // Perform verification based on method
        let result = match self.config.method {
            VerificationMethod::EmbeddingOnly => {
                self.verify_embeddings(&ref_embedding, &test_embedding)?
            }
            VerificationMethod::SampleComparison => {
                self.verify_audio_comparison(&processed_ref, &processed_test)
                    .await?
            }
            VerificationMethod::MultiModal => {
                self.verify_multimodal(
                    &processed_ref,
                    &processed_test,
                    &ref_embedding,
                    &test_embedding,
                )
                .await?
            }
            VerificationMethod::DeepVerification => {
                self.verify_deep(&processed_ref, &processed_test).await?
            }
        };

        let processing_time = start_time.elapsed();

        // Update system metrics
        self.update_metrics(&result, processing_time);

        // Cache result if enabled
        if self.config.enable_caching {
            let cache_key = format!("{}_{}", reference.id, test.id);
            self.cache.insert(cache_key, result.clone());
        }

        info!(
            "Verification completed: verified={}, confidence={:.3}, time={:?}",
            result.verified, result.confidence, processing_time
        );

        Ok(result)
    }

    /// Validate input samples
    fn validate_samples(&self, reference: &VoiceSample, test: &VoiceSample) -> Result<()> {
        // Check sample duration
        if reference.duration < self.config.min_sample_duration {
            return Err(Error::Validation(format!(
                "Reference sample too short: {:.2}s, minimum: {:.2}s",
                reference.duration, self.config.min_sample_duration
            )));
        }

        if test.duration < self.config.min_sample_duration {
            return Err(Error::Validation(format!(
                "Test sample too short: {:.2}s, minimum: {:.2}s",
                test.duration, self.config.min_sample_duration
            )));
        }

        // Check sample quality (noise level)
        let ref_quality = self.assess_sample_quality(reference);
        let test_quality = self.assess_sample_quality(test);

        if ref_quality.noise_level > self.config.max_noise_level {
            warn!(
                "Reference sample has high noise level: {:.3}",
                ref_quality.noise_level
            );
        }

        if test_quality.noise_level > self.config.max_noise_level {
            warn!(
                "Test sample has high noise level: {:.3}",
                test_quality.noise_level
            );
        }

        Ok(())
    }

    /// Audio-based comparison verification
    async fn verify_audio_comparison(
        &self,
        reference: &VoiceSample,
        test: &VoiceSample,
    ) -> Result<VerificationResult> {
        let start_time = Instant::now();

        // Compute acoustic similarity
        let acoustic_similarity = self.compute_acoustic_similarity(reference, test)?;
        let prosodic_similarity = self.compute_prosodic_similarity(reference, test)?;

        // Combine similarities
        let combined_score = (acoustic_similarity + prosodic_similarity) / 2.0;
        let verified = combined_score >= self.config.threshold;

        let mut metrics = VerificationMetrics::default();
        metrics.acoustic_similarity = acoustic_similarity;
        metrics.prosodic_similarity = prosodic_similarity;

        let ref_quality = self.assess_sample_quality(reference);
        let test_quality = self.assess_sample_quality(test);
        metrics.sample_qualities = (ref_quality.overall_score, test_quality.overall_score);

        Ok(VerificationResult::new(
            verified,
            combined_score,
            self.config.threshold,
            combined_score,
            metrics,
            start_time.elapsed(),
            VerificationMethod::SampleComparison,
        ))
    }

    /// Multi-modal verification (embeddings + audio)
    async fn verify_multimodal(
        &mut self,
        reference: &VoiceSample,
        test: &VoiceSample,
        ref_embedding: &SpeakerEmbedding,
        test_embedding: &SpeakerEmbedding,
    ) -> Result<VerificationResult> {
        let start_time = Instant::now();

        // Get embedding similarity
        let embedding_similarity = ref_embedding.similarity(test_embedding);

        // Get acoustic similarity
        let acoustic_similarity = self.compute_acoustic_similarity(reference, test)?;
        let prosodic_similarity = self.compute_prosodic_similarity(reference, test)?;

        // Weighted combination of similarities
        let combined_score = (embedding_similarity * 0.5)
            + (acoustic_similarity * 0.3)
            + (prosodic_similarity * 0.2);

        let verified = combined_score >= self.config.threshold;

        let mut metrics = VerificationMetrics::default();
        metrics.embedding_similarity = embedding_similarity;
        metrics.acoustic_similarity = acoustic_similarity;
        metrics.prosodic_similarity = prosodic_similarity;

        let ref_quality = self.assess_sample_quality(reference);
        let test_quality = self.assess_sample_quality(test);
        metrics.sample_qualities = (ref_quality.overall_score, test_quality.overall_score);

        metrics.far_estimate =
            self.estimate_false_acceptance_rate(combined_score, self.config.threshold);
        metrics.frr_estimate =
            self.estimate_false_rejection_rate(combined_score, self.config.threshold);

        Ok(VerificationResult::new(
            verified,
            combined_score,
            self.config.threshold,
            combined_score,
            metrics,
            start_time.elapsed(),
            VerificationMethod::MultiModal,
        ))
    }

    /// Deep neural network-based verification (placeholder for future implementation)
    async fn verify_deep(
        &mut self,
        reference: &VoiceSample,
        test: &VoiceSample,
    ) -> Result<VerificationResult> {
        // For now, fall back to multimodal verification
        warn!("Deep verification not yet implemented, falling back to multimodal");

        let ref_embedding = self.embedding_extractor.extract(reference).await?;
        let test_embedding = self.embedding_extractor.extract(test).await?;

        self.verify_multimodal(reference, test, &ref_embedding, &test_embedding)
            .await
    }

    /// Compute acoustic similarity between samples
    fn compute_acoustic_similarity(
        &self,
        reference: &VoiceSample,
        test: &VoiceSample,
    ) -> Result<f32> {
        let ref_audio = reference.get_normalized_audio();
        let test_audio = test.get_normalized_audio();

        if ref_audio.is_empty() || test_audio.is_empty() {
            return Ok(0.0);
        }

        // Compute spectral features for both samples
        let ref_features = self.extract_acoustic_features(&ref_audio, reference.sample_rate)?;
        let test_features = self.extract_acoustic_features(&test_audio, test.sample_rate)?;

        // Compute cosine similarity
        let similarity = self.cosine_similarity(&ref_features, &test_features)?;
        Ok(similarity.clamp(0.0, 1.0))
    }

    /// Compute prosodic similarity between samples
    fn compute_prosodic_similarity(
        &self,
        reference: &VoiceSample,
        test: &VoiceSample,
    ) -> Result<f32> {
        let ref_audio = reference.get_normalized_audio();
        let test_audio = test.get_normalized_audio();

        if ref_audio.is_empty() || test_audio.is_empty() {
            return Ok(0.0);
        }

        // Extract prosodic features
        let ref_prosody = self.extract_prosodic_features(&ref_audio, reference.sample_rate)?;
        let test_prosody = self.extract_prosodic_features(&test_audio, test.sample_rate)?;

        // Compare fundamental frequency characteristics
        let f0_similarity = self.compare_f0_characteristics(&ref_prosody, &test_prosody);

        // Compare rhythm patterns
        let rhythm_similarity = self.compare_rhythm_patterns(&ref_prosody, &test_prosody);

        // Combine prosodic similarities
        let prosodic_similarity = (f0_similarity + rhythm_similarity) / 2.0;
        Ok(prosodic_similarity.clamp(0.0, 1.0))
    }

    /// Extract acoustic features from audio
    fn extract_acoustic_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        // Energy features
        let mut features = vec![
            self.compute_rms_energy(audio),
            self.compute_spectral_centroid(audio, sample_rate),
            self.compute_spectral_rolloff(audio, sample_rate),
            self.compute_zero_crossing_rate(audio),
        ];

        // MFCC-like features (simplified)
        features.extend(self.compute_mel_features(audio, sample_rate)?);

        Ok(features)
    }

    /// Extract prosodic features from audio
    fn extract_prosodic_features(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<ProsodicFeatures> {
        let f0_contour = self.extract_f0_contour(audio, sample_rate)?;
        let energy_contour = self.extract_energy_contour(audio, sample_rate)?;

        Ok(ProsodicFeatures {
            f0_mean: self.compute_mean(&f0_contour),
            f0_std: self.compute_std(&f0_contour),
            f0_range: self.compute_range(&f0_contour),
            energy_mean: self.compute_mean(&energy_contour),
            energy_std: self.compute_std(&energy_contour),
            speech_rate: self.estimate_speech_rate(audio, sample_rate),
        })
    }

    /// Assess sample quality
    fn assess_sample_quality(&self, sample: &VoiceSample) -> SampleQualityMetrics {
        let audio = sample.get_normalized_audio();

        let snr = self.estimate_snr(&audio);
        let noise_level = 1.0 - (snr / 20.0).clamp(0.0, 1.0); // Convert SNR to noise level
        let clarity = self.assess_spectral_clarity(&audio, sample.sample_rate);

        let overall_score =
            ((snr / 20.0) * 0.4 + clarity * 0.4 + (1.0 - noise_level) * 0.2).clamp(0.0, 1.0);

        SampleQualityMetrics {
            snr,
            noise_level,
            clarity,
            overall_score,
        }
    }

    // Helper methods for audio processing
    fn compute_rms_energy(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt()
    }

    /// Compute the power spectrum of an audio frame using a Hann window and RFFT.
    ///
    /// Returns a `Vec<f32>` of length `n_fft / 2 + 1` (the one-sided power spectrum).
    /// `n_fft` is clamped to `audio.len()` when the signal is shorter than `n_fft`.
    fn compute_power_spectrum(&self, audio: &[f32], n_fft: usize) -> Vec<f32> {
        if audio.is_empty() {
            return vec![0.0; n_fft / 2 + 1];
        }

        // Build the (windowed) frame as f64 for scirs2_fft
        let frame_len = n_fft.min(audio.len());
        let mut frame_f64: Vec<f64> = Vec::with_capacity(n_fft);
        for i in 0..frame_len {
            // Hann window: w(n) = 0.5 * (1 − cos(2π n / (N−1)))
            let w = if frame_len > 1 {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (frame_len - 1) as f64).cos())
            } else {
                1.0
            };
            frame_f64.push(audio[i] as f64 * w);
        }
        // Zero-pad to n_fft if needed
        frame_f64.resize(n_fft, 0.0);

        let n_out = n_fft / 2 + 1;
        match scirs2_fft::rfft(&frame_f64, Some(n_fft)) {
            Ok(spectrum) => spectrum
                .iter()
                .take(n_out)
                .map(|c| (c.re * c.re + c.im * c.im) as f32)
                .collect(),
            Err(_) => vec![0.0; n_out],
        }
    }

    fn compute_spectral_centroid(&self, audio: &[f32], sample_rate: u32) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let n_fft = 512.min(audio.len()).next_power_of_two().max(2);
        let power = self.compute_power_spectrum(audio, n_fft);
        let n_out = power.len(); // n_fft / 2 + 1

        let mut centroid = 0.0_f32;
        let mut magnitude_sum = 0.0_f32;
        for (k, &p) in power.iter().enumerate() {
            let magnitude = p.sqrt();
            let frequency = k as f32 * sample_rate as f32 / n_fft as f32;
            centroid += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            let nyquist = sample_rate as f32 / 2.0;
            (centroid / magnitude_sum / nyquist).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    fn compute_spectral_rolloff(&self, audio: &[f32], sample_rate: u32) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let n_fft = 512.min(audio.len()).next_power_of_two().max(2);
        let power = self.compute_power_spectrum(audio, n_fft);
        let total_energy: f32 = power.iter().sum();

        if total_energy <= 0.0 {
            return 0.0;
        }

        let rolloff_threshold = total_energy * 0.85;
        let mut cumulative = 0.0_f32;
        let n_out = power.len();
        for (k, &p) in power.iter().enumerate() {
            cumulative += p;
            if cumulative >= rolloff_threshold {
                // Normalize to [0, 1] relative to Nyquist
                return (k as f32 / (n_out - 1).max(1) as f32).clamp(0.0, 1.0);
            }
        }
        1.0
    }

    fn compute_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }
        let crossings = audio
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count();
        crossings as f32 / (audio.len() - 1) as f32
    }

    fn compute_mel_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        // Full MFCC pipeline:
        //   frame audio → Hann window → FFT → magnitude spectrum
        //   → area-normalised triangular mel filterbank (26 filters, 80 Hz–Nyquist)
        //   → log → orthonormal DCT-II → keep first 13 coefficients
        //   → average over all frames → return Vec<f32> of length 13
        const N_MFCC: usize = 13;
        const N_MEL: usize = 26;
        const F_MIN_HZ: f64 = 80.0;

        // Return zero vector for degenerate input rather than erroring.
        if audio.is_empty() {
            return Ok(vec![0.0_f32; N_MFCC]);
        }

        let sr = sample_rate as f64;

        // --- Frame parameters (standard speech processing defaults) ---
        // 25 ms frames, 10 ms hop, rounded to nearest sample
        let frame_len = ((sr * 0.025).round() as usize).max(2);
        let hop_len = ((sr * 0.010).round() as usize).max(1);

        // N_FFT = next power-of-two >= frame_len so zero-padding is correct
        let n_fft = frame_len.next_power_of_two();
        let n_out = n_fft / 2 + 1; // one-sided spectrum bins

        // --- Pre-compute Hann window ---
        // w[n] = 0.5 * (1 - cos(2π n / (N-1))), length = frame_len
        let hann: Vec<f64> = (0..frame_len)
            .map(|n| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (frame_len - 1) as f64).cos())
            })
            .collect();

        // --- Pre-compute area-normalised triangular mel filterbank ---
        //
        // Following Slaney (1993 / Auditory Toolbox):
        //   area = 2 / (f_right - f_left)     (area normalisation)
        //   rising  slope:  area * (k - f_left)  / (f_center - f_left)
        //   falling slope:  area * (f_right - k) / (f_right - f_center)
        let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);

        let nyquist = sr / 2.0;
        let f_max_hz = nyquist.min(8000.0); // cap at 8 kHz as specified
        let mel_min = hz_to_mel(F_MIN_HZ);
        let mel_max = hz_to_mel(f_max_hz);

        // N_MEL + 2 equally-spaced mel-axis points (edges + centres)
        let n_mel_pts = N_MEL + 2;
        let mel_pts: Vec<f64> = (0..n_mel_pts)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mel_pts - 1) as f64)
            .collect();

        // Convert mel points to FFT bin indices (real-valued, not rounded)
        let bin_f: Vec<f64> = mel_pts
            .iter()
            .map(|&m| mel_to_hz(m) / sr * n_fft as f64)
            .collect();

        // Build filterbank weights: filter_weights[m][k]  (N_MEL × n_out)
        // Pre-allocate as flat array for cache locality.
        let mut filter_weights = vec![0.0_f64; N_MEL * n_out];
        for m in 0..N_MEL {
            let f_left = bin_f[m];
            let f_center = bin_f[m + 1];
            let f_right = bin_f[m + 2];
            let bw = f_right - f_left;
            // area normalisation: peak of triangle = 2 / bw
            let area_norm = if bw > 1e-12 { 2.0 / bw } else { 1.0 };
            for k in 0..n_out {
                let kf = k as f64;
                let w = if kf >= f_left && kf <= f_center {
                    let rise = f_center - f_left;
                    if rise > 1e-12 {
                        area_norm * (kf - f_left) / rise
                    } else {
                        area_norm
                    }
                } else if kf > f_center && kf <= f_right {
                    let fall = f_right - f_center;
                    if fall > 1e-12 {
                        area_norm * (f_right - kf) / fall
                    } else {
                        area_norm
                    }
                } else {
                    0.0
                };
                filter_weights[m * n_out + k] = w;
            }
        }

        // --- Pre-compute DCT-II orthonormal basis coefficients ---
        // For coefficient k (0-indexed, k = 0..N_MFCC):
        //   k = 0:  sqrt(1 / N_MEL)
        //   k > 0:  sqrt(2 / N_MEL) * cos(π k (2n + 1) / (2 N_MEL))
        //
        // We keep coefficients k = 0..N_MFCC but return all N_MFCC of them.
        // (Dropping k=0 is optional; the task says return 13 coefficients from the
        // DCT; using k=1..=13 is common, but we follow the orthonormal definition
        // and return k=0..12 so the length is always exactly 13.)
        let dct_basis: Vec<f64> = (0..N_MFCC)
            .flat_map(|k| {
                let norm = if k == 0 {
                    (1.0_f64 / N_MEL as f64).sqrt()
                } else {
                    (2.0_f64 / N_MEL as f64).sqrt()
                };
                (0..N_MEL).map(move |n| {
                    norm * (std::f64::consts::PI * k as f64 * (2 * n + 1) as f64
                        / (2 * N_MEL) as f64)
                        .cos()
                })
            })
            .collect();

        // --- Frame the audio and accumulate MFCC sums ---
        let mut mfcc_sum = vec![0.0_f64; N_MFCC];
        let mut n_frames: usize = 0;

        // Slide the window; skip frames shorter than half of frame_len at end
        let mut frame_buf = vec![0.0_f64; n_fft];
        let mut start = 0_usize;
        while start + frame_len / 2 <= audio.len() {
            let end = (start + frame_len).min(audio.len());
            let actual_len = end - start;

            // Apply Hann window and zero-pad to n_fft
            for i in 0..n_fft {
                frame_buf[i] = if i < actual_len {
                    audio[start + i] as f64 * hann[i]
                } else {
                    0.0
                };
            }

            // RFFT → magnitude power spectrum (one-sided)
            let spectrum = scirs2_fft::rfft(&frame_buf, Some(n_fft))
                .map_err(|e| Error::Processing(format!("FFT failed: {e}")))?;

            // Power spectrum: |X[k]|²
            let power: Vec<f64> = spectrum
                .iter()
                .take(n_out)
                .map(|c| c.re * c.re + c.im * c.im)
                .collect();

            // Apply mel filterbank
            let mut mel_energy = [0.0_f64; N_MEL];
            for m in 0..N_MEL {
                let row_offset = m * n_out;
                let mut acc = 0.0_f64;
                for k in 0..n_out {
                    acc += power[k] * filter_weights[row_offset + k];
                }
                mel_energy[m] = acc;
            }

            // Log mel energies (floor at 1e-10 to avoid -inf)
            let log_mel: [f64; N_MEL] = std::array::from_fn(|m| (mel_energy[m] + 1e-10_f64).ln());

            // Orthonormal DCT-II: accumulate into mfcc_sum
            for k in 0..N_MFCC {
                let row_offset = k * N_MEL;
                let mut acc = 0.0_f64;
                for n in 0..N_MEL {
                    acc += dct_basis[row_offset + n] * log_mel[n];
                }
                mfcc_sum[k] += acc;
            }

            n_frames += 1;
            start += hop_len;
        }

        // Average over frames (guard against zero-frame edge case)
        let n_frames_f = if n_frames == 0 { 1 } else { n_frames } as f64;
        let mfcc: Vec<f32> = mfcc_sum.iter().map(|&s| (s / n_frames_f) as f32).collect();

        Ok(mfcc)
    }

    fn extract_f0_contour(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        // Simplified F0 extraction using autocorrelation
        let window_size = (sample_rate as f32 * 0.025) as usize; // 25ms windows
        let hop_size = window_size / 2;
        let mut f0_values = Vec::new();

        for start in (0..audio.len()).step_by(hop_size) {
            let end = (start + window_size).min(audio.len());
            if end - start < window_size / 2 {
                break;
            }

            let window = &audio[start..end];
            let f0 = self.estimate_f0_autocorr(window, sample_rate);
            f0_values.push(f0);
        }

        Ok(f0_values)
    }

    fn extract_energy_contour(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let window_size = (sample_rate as f32 * 0.025) as usize; // 25ms windows
        let hop_size = window_size / 2;
        let mut energy_values = Vec::new();

        for start in (0..audio.len()).step_by(hop_size) {
            let end = (start + window_size).min(audio.len());
            if end - start < window_size / 2 {
                break;
            }

            let window = &audio[start..end];
            let energy = self.compute_rms_energy(window);
            energy_values.push(energy);
        }

        Ok(energy_values)
    }

    fn estimate_f0_autocorr(&self, window: &[f32], sample_rate: u32) -> f32 {
        if window.len() < 64 {
            return 0.0;
        }

        // Compute autocorrelation
        let mut autocorr = vec![0.0; window.len()];
        for lag in 0..window.len() {
            for i in 0..(window.len() - lag) {
                autocorr[lag] += window[i] * window[i + lag];
            }
        }

        // Find peak in typical speech F0 range (80-400 Hz)
        let min_period = (sample_rate as f32 / 400.0) as usize;
        let max_period = (sample_rate as f32 / 80.0) as usize;
        let max_period = max_period.min(autocorr.len() - 1);

        if min_period >= max_period {
            return 0.0;
        }

        let mut best_period = min_period;
        let mut best_value = autocorr[min_period];

        for (offset, &value) in autocorr
            .iter()
            .enumerate()
            .skip(min_period)
            .take(max_period - min_period + 1)
        {
            if value > best_value {
                best_value = value;
                best_period = offset;
            }
        }

        // Check if peak is significant
        if best_value < autocorr[0] * 0.3 {
            return 0.0;
        }

        sample_rate as f32 / best_period as f32
    }

    fn compare_f0_characteristics(
        &self,
        ref_prosody: &ProsodicFeatures,
        test_prosody: &ProsodicFeatures,
    ) -> f32 {
        // Prevent division by zero with minimum threshold
        let min_threshold = 1e-6;

        let mean_diff =
            if ref_prosody.f0_mean > min_threshold && test_prosody.f0_mean > min_threshold {
                (ref_prosody.f0_mean - test_prosody.f0_mean).abs()
                    / ref_prosody.f0_mean.max(test_prosody.f0_mean)
            } else {
                0.5 // Default similarity when F0 values are too small
            };

        let std_diff = if ref_prosody.f0_std > min_threshold && test_prosody.f0_std > min_threshold
        {
            (ref_prosody.f0_std - test_prosody.f0_std).abs()
                / ref_prosody.f0_std.max(test_prosody.f0_std)
        } else {
            0.5
        };

        let range_diff =
            if ref_prosody.f0_range > min_threshold && test_prosody.f0_range > min_threshold {
                (ref_prosody.f0_range - test_prosody.f0_range).abs()
                    / ref_prosody.f0_range.max(test_prosody.f0_range)
            } else {
                0.5
            };

        let similarity = 1.0 - ((mean_diff + std_diff + range_diff) / 3.0);
        similarity.clamp(0.0, 1.0)
    }

    fn compare_rhythm_patterns(
        &self,
        ref_prosody: &ProsodicFeatures,
        test_prosody: &ProsodicFeatures,
    ) -> f32 {
        let min_threshold = 1e-6;

        let rate_diff = if ref_prosody.speech_rate > min_threshold
            && test_prosody.speech_rate > min_threshold
        {
            (ref_prosody.speech_rate - test_prosody.speech_rate).abs()
                / ref_prosody.speech_rate.max(test_prosody.speech_rate)
        } else {
            0.2 // Default difference when speech rate can't be determined
        };

        let energy_diff = if ref_prosody.energy_mean > min_threshold
            && test_prosody.energy_mean > min_threshold
        {
            (ref_prosody.energy_mean - test_prosody.energy_mean).abs()
                / ref_prosody.energy_mean.max(test_prosody.energy_mean)
        } else {
            0.2 // Default difference when energy can't be determined
        };

        let similarity = 1.0 - ((rate_diff + energy_diff) / 2.0);
        similarity.clamp(0.0, 1.0)
    }

    fn estimate_speech_rate(&self, _audio: &[f32], _sample_rate: u32) -> f32 {
        // Simplified speech rate estimation (syllables per second)
        5.0 // Typical speech rate
    }

    fn estimate_snr(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Estimate SNR using energy distribution
        let energies: Vec<f32> = audio.iter().map(|x| x * x).collect();
        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;

        // Estimate noise floor (bottom 10% of energies)
        let mut sorted_energies = energies.clone();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_samples = sorted_energies.len() / 10;
        let noise_energy =
            sorted_energies[..noise_samples].iter().sum::<f32>() / noise_samples as f32;

        if noise_energy > 0.0 {
            10.0 * (mean_energy / noise_energy).log10() // SNR in dB
        } else {
            20.0 // High SNR if no noise detected
        }
    }

    fn assess_spectral_clarity(&self, _audio: &[f32], _sample_rate: u32) -> f32 {
        // Simplified spectral clarity assessment
        0.8 // Placeholder
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(Error::Processing(
                "Feature vectors have different lengths".to_string(),
            ));
        }

        let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            let similarity = dot_product / (norm_a * norm_b);
            Ok((similarity + 1.0) / 2.0) // Map from [-1, 1] to [0, 1]
        } else {
            Ok(0.0)
        }
    }

    fn compute_mean(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f32>() / values.len() as f32
    }

    fn compute_std(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = self.compute_mean(values);
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
        variance.sqrt()
    }

    fn compute_range(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        max_val - min_val
    }

    fn estimate_false_acceptance_rate(&self, similarity: f32, threshold: f32) -> f32 {
        // Simplified FAR estimation based on similarity and threshold
        if similarity >= threshold {
            // Higher confidence (further above threshold) means lower FAR
            let excess = similarity - threshold;
            let max_excess = 1.0 - threshold;
            let normalized_excess = (excess / max_excess).clamp(0.0, 1.0);
            (0.1 * (1.0 - normalized_excess)).clamp(0.001, 0.1)
        } else {
            // Below threshold, we reject, so FAR is essentially 0
            0.001
        }
    }

    fn estimate_false_rejection_rate(&self, similarity: f32, threshold: f32) -> f32 {
        // Simplified FRR estimation
        if similarity < threshold {
            // Lower similarity means higher FRR for legitimate users
            (threshold - similarity).clamp(0.001, 0.2)
        } else {
            // Above threshold, FRR is low
            0.01
        }
    }

    fn update_metrics(&mut self, result: &VerificationResult, processing_time: Duration) {
        self.metrics.total_verifications += 1;

        if result.verified {
            self.metrics.successful_verifications += 1;
        } else {
            self.metrics.failed_verifications += 1;
        }

        // Update average processing time
        let total_time_ms = self.metrics.avg_processing_time.as_millis() as f32
            * (self.metrics.total_verifications - 1) as f32;
        let new_total_time_ms = total_time_ms + processing_time.as_millis() as f32;
        self.metrics.avg_processing_time = Duration::from_millis(
            (new_total_time_ms / self.metrics.total_verifications as f32) as u64,
        );

        // Update average confidence
        let total_confidence =
            self.metrics.avg_confidence * (self.metrics.total_verifications - 1) as f32;
        self.metrics.avg_confidence =
            (total_confidence + result.confidence) / self.metrics.total_verifications as f32;
    }

    /// Get system metrics
    pub fn get_metrics(&self) -> &VerificationSystemMetrics {
        &self.metrics
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        info!("Verification cache cleared");
    }

    /// Update verification thresholds
    pub fn update_threshold(&mut self, threshold: f32) {
        self.config.threshold = threshold.clamp(0.0, 1.0);
        info!(
            "Verification threshold updated to {}",
            self.config.threshold
        );
    }

    /// Set strict mode
    pub fn set_strict_mode(&mut self, enabled: bool) {
        if enabled {
            self.config.threshold = self.config.strict_threshold;
            info!(
                "Strict verification mode enabled (threshold: {})",
                self.config.threshold
            );
        } else {
            self.config.threshold = 0.8; // Default threshold
            info!(
                "Normal verification mode enabled (threshold: {})",
                self.config.threshold
            );
        }
    }
}

/// Prosodic features for voice comparison
#[derive(Debug, Clone)]
struct ProsodicFeatures {
    f0_mean: f32,
    f0_std: f32,
    f0_range: f32,
    energy_mean: f32,
    energy_std: f32,
    speech_rate: f32,
}

/// Sample quality metrics
#[derive(Debug, Clone)]
struct SampleQualityMetrics {
    snr: f32,
    noise_level: f32,
    clarity: f32,
    overall_score: f32,
}

impl VerificationSystemMetrics {
    fn new() -> Self {
        Self {
            total_verifications: 0,
            successful_verifications: 0,
            failed_verifications: 0,
            false_acceptances: 0,
            false_rejections: 0,
            avg_processing_time: Duration::default(),
            avg_confidence: 0.0,
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f32 {
        if self.total_verifications == 0 {
            return 0.0;
        }
        self.successful_verifications as f32 / self.total_verifications as f32
    }

    /// Get estimated FAR
    pub fn false_acceptance_rate(&self) -> f32 {
        if self.successful_verifications == 0 {
            return 0.0;
        }
        self.false_acceptances as f32 / self.successful_verifications as f32
    }

    /// Get estimated FRR
    pub fn false_rejection_rate(&self) -> f32 {
        if self.failed_verifications == 0 {
            return 0.0;
        }
        self.false_rejections as f32 / self.failed_verifications as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceSample;

    #[tokio::test]
    async fn test_speaker_verifier_creation() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config);
        assert!(verifier.is_ok());
    }

    #[test]
    fn test_verification_result_creation() {
        let metrics = VerificationMetrics::default();
        let result = VerificationResult::new(
            true,
            0.85,
            0.8,
            0.85,
            metrics,
            Duration::from_millis(100),
            VerificationMethod::EmbeddingOnly,
        );

        assert!(result.verified);
        assert_eq!(result.confidence, 0.85);
        assert_eq!(result.threshold, 0.8);
        assert_eq!(result.score, 0.85);
        assert_eq!(result.method, VerificationMethod::EmbeddingOnly);
    }

    #[test]
    fn test_verification_metrics_default() {
        let metrics = VerificationMetrics::default();

        assert_eq!(metrics.embedding_similarity, 0.0);
        assert_eq!(metrics.acoustic_similarity, 0.0);
        assert_eq!(metrics.prosodic_similarity, 0.0);
        assert_eq!(metrics.sample_qualities, (0.0, 0.0));
        assert_eq!(metrics.far_estimate, 0.01);
        assert_eq!(metrics.frr_estimate, 0.05);
    }

    #[tokio::test]
    async fn test_embedding_verification() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        // Create test embeddings
        let ref_embedding = SpeakerEmbedding {
            vector: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            dimension: 5,
            confidence: 0.9,
            metadata: crate::embedding::EmbeddingMetadata::default(),
        };

        let test_embedding_similar = SpeakerEmbedding {
            vector: vec![0.11, 0.21, 0.31, 0.41, 0.51], // Similar to reference
            dimension: 5,
            confidence: 0.85,
            metadata: crate::embedding::EmbeddingMetadata::default(),
        };

        let test_embedding_different = SpeakerEmbedding {
            vector: vec![0.9, -0.8, 0.1, -0.2, 0.3], // Very different from reference
            dimension: 5,
            confidence: 0.8,
            metadata: crate::embedding::EmbeddingMetadata::default(),
        };

        // Test similar embeddings (should verify)
        let result_similar = verifier
            .verify_embeddings(&ref_embedding, &test_embedding_similar)
            .unwrap();
        assert!(result_similar.verified); // Should pass verification
        assert!(result_similar.confidence > 0.8);

        // Test different embeddings (should not verify)
        let result_different = verifier
            .verify_embeddings(&ref_embedding, &test_embedding_different)
            .unwrap();
        assert!(!result_different.verified); // Should fail verification
        assert!(result_different.confidence < 0.8);
    }

    #[test]
    fn test_invalid_embeddings() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        // Create embeddings with different dimensions
        let ref_embedding = SpeakerEmbedding {
            vector: vec![0.1, 0.2, 0.3],
            dimension: 3,
            confidence: 0.9,
            metadata: crate::embedding::EmbeddingMetadata::default(),
        };

        let test_embedding_wrong_dim = SpeakerEmbedding {
            vector: vec![0.1, 0.2, 0.3, 0.4, 0.5], // Different dimension
            dimension: 5,
            confidence: 0.85,
            metadata: crate::embedding::EmbeddingMetadata::default(),
        };

        let result = verifier.verify_embeddings(&ref_embedding, &test_embedding_wrong_dim);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_sample_validation() {
        let config = VerificationConfig {
            min_sample_duration: 2.0, // Require at least 2 seconds
            max_noise_level: 0.05,    // Very low noise tolerance
            ..VerificationConfig::default()
        };
        let mut verifier = SpeakerVerifier::new(config).unwrap();

        // Create a too-short sample
        let short_sample = VoiceSample::new(
            "short_sample".to_string(),
            vec![0.1; 8000], // Only 0.5 seconds at 16kHz
            16000,
        );

        // Create a good sample
        let good_sample = VoiceSample::new(
            "good_sample".to_string(),
            vec![0.1; 48000], // 3 seconds at 16kHz
            16000,
        );

        // Test with too-short samples
        let result = verifier.verify_samples(&short_sample, &good_sample).await;
        assert!(result.is_err()); // Should fail due to short reference

        let result = verifier.verify_samples(&good_sample, &short_sample).await;
        assert!(result.is_err()); // Should fail due to short test sample
    }

    #[tokio::test]
    async fn test_audio_comparison_verification() {
        let config = VerificationConfig {
            method: VerificationMethod::SampleComparison,
            threshold: 0.6, // Lower threshold for test
            ..VerificationConfig::default()
        };
        let mut verifier = SpeakerVerifier::new(config).unwrap();

        // Create similar samples
        let ref_sample = VoiceSample::new(
            "ref_sample".to_string(),
            vec![0.1; 32000], // 2 seconds at 16kHz
            16000,
        );

        let test_sample = VoiceSample::new(
            "test_sample".to_string(),
            vec![0.11; 32000], // Similar to reference
            16000,
        );

        let result = verifier.verify_samples(&ref_sample, &test_sample).await;
        assert!(result.is_ok());

        let verification_result = result.unwrap();
        assert_eq!(
            verification_result.method,
            VerificationMethod::SampleComparison
        );
        assert!(verification_result.metrics.acoustic_similarity > 0.0);
        assert!(verification_result.metrics.prosodic_similarity > 0.0);
    }

    #[tokio::test]
    async fn test_multimodal_verification() {
        let config = VerificationConfig {
            method: VerificationMethod::MultiModal,
            threshold: 0.6,
            ..VerificationConfig::default()
        };
        let mut verifier = SpeakerVerifier::new(config).unwrap();

        let ref_sample = VoiceSample::new(
            "ref_sample".to_string(),
            vec![0.1; 32000], // 2 seconds at 16kHz
            16000,
        );

        let test_sample = VoiceSample::new(
            "test_sample".to_string(),
            vec![0.12; 32000], // Similar to reference
            16000,
        );

        let result = verifier.verify_samples(&ref_sample, &test_sample).await;
        assert!(result.is_ok());

        let verification_result = result.unwrap();
        assert_eq!(verification_result.method, VerificationMethod::MultiModal);
        assert!(verification_result.metrics.embedding_similarity > 0.0);
        assert!(verification_result.metrics.acoustic_similarity > 0.0);
        assert!(verification_result.metrics.prosodic_similarity > 0.0);
    }

    #[test]
    fn test_verification_system_metrics() {
        let mut metrics = VerificationSystemMetrics::new();

        assert_eq!(metrics.total_verifications, 0);
        assert_eq!(metrics.success_rate(), 0.0);

        // Simulate some verifications
        metrics.total_verifications = 10;
        metrics.successful_verifications = 8;
        metrics.failed_verifications = 2;

        assert_eq!(metrics.success_rate(), 0.8);
    }

    #[test]
    fn test_acoustic_feature_extraction() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        let audio = vec![0.1, 0.2, -0.1, -0.2, 0.15, -0.15]; // Simple test audio
        let sample_rate = 16000;

        let features = verifier
            .extract_acoustic_features(&audio, sample_rate)
            .unwrap();

        assert!(!features.is_empty());
        assert!(features.len() >= 4); // Basic features + mel features

        // Test RMS energy computation
        let rms = verifier.compute_rms_energy(&audio);
        assert!(rms > 0.0);
        assert!(rms < 1.0);

        // Test ZCR computation
        let zcr = verifier.compute_zero_crossing_rate(&audio);
        assert!(zcr > 0.0); // Should have some zero crossings
        assert!(zcr <= 1.0);
    }

    #[test]
    fn test_prosodic_features() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        // Create a simple sine wave for testing
        let sample_rate = 16000;
        let duration = 0.1; // 0.1 seconds
        let frequency = 200.0; // 200 Hz
        let num_samples = (sample_rate as f32 * duration) as usize;

        let audio: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let prosody_result = verifier.extract_prosodic_features(&audio, sample_rate);
        assert!(prosody_result.is_ok());

        let prosody = prosody_result.unwrap();
        assert!(prosody.f0_mean > 0.0);
        assert!(prosody.energy_mean > 0.0);
        assert!(prosody.speech_rate > 0.0);
    }

    #[test]
    fn test_f0_estimation() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        // Create a simple sine wave at 200 Hz
        let sample_rate = 16000;
        let frequency = 200.0;
        let duration_samples = sample_rate / 4; // 0.25 seconds

        let audio: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let f0 = verifier.estimate_f0_autocorr(&audio, sample_rate);

        // Should detect a frequency close to 200 Hz (within reasonable tolerance)
        assert!(f0 > 150.0);
        assert!(f0 < 250.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];

        let sim_ab = verifier.cosine_similarity(&a, &b).unwrap();
        let sim_ac = verifier.cosine_similarity(&a, &c).unwrap();

        assert!(sim_ac > sim_ab); // c is more similar to a than b
        assert!(sim_ac > 0.9); // Should be very high similarity (same vector)
        assert!(sim_ab < 0.6); // Should be low similarity (orthogonal vectors)
    }

    #[test]
    fn test_error_rate_estimation() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        // High similarity, above threshold
        let far_high = verifier.estimate_false_acceptance_rate(0.95, 0.8);
        let frr_high = verifier.estimate_false_rejection_rate(0.95, 0.8);

        // Low similarity, below threshold
        let far_low = verifier.estimate_false_acceptance_rate(0.3, 0.8);
        let frr_low = verifier.estimate_false_rejection_rate(0.3, 0.8);

        // High similarity should have higher FAR (might accept imposter) but lower FRR (won't reject genuine)
        // Low similarity should have lower FAR (will reject imposter) but higher FRR (might reject genuine)
        assert!(far_high > far_low);
        assert!(frr_high < frr_low);

        // All rates should be reasonable
        assert!(far_high >= 0.0 && far_high <= 1.0);
        assert!(frr_high >= 0.0 && frr_high <= 1.0);
        assert!(far_low >= 0.0 && far_low <= 1.0);
        assert!(frr_low >= 0.0 && frr_low <= 1.0);
    }

    #[tokio::test]
    async fn test_verification_caching() {
        let config = VerificationConfig {
            enable_caching: true,
            threshold: 0.6,
            ..VerificationConfig::default()
        };
        let mut verifier = SpeakerVerifier::new(config).unwrap();

        let ref_sample = VoiceSample::new("ref_cached".to_string(), vec![0.1; 32000], 16000);

        let test_sample = VoiceSample::new("test_cached".to_string(), vec![0.11; 32000], 16000);

        // First verification (should compute and cache)
        let start_time = std::time::Instant::now();
        let result1 = verifier
            .verify_samples(&ref_sample, &test_sample)
            .await
            .unwrap();
        let time1 = start_time.elapsed();

        // Second verification (should use cache)
        let start_time = std::time::Instant::now();
        let result2 = verifier
            .verify_samples(&ref_sample, &test_sample)
            .await
            .unwrap();
        let time2 = start_time.elapsed();

        // Results should be identical
        assert_eq!(result1.verified, result2.verified);
        assert_eq!(result1.confidence, result2.confidence);

        // Second call should be faster (using cache)
        // Note: This might not always be true in test environment, but cache should work
        assert!(time2 < time1 || time2.as_millis() < 10); // Either faster or very fast
    }

    #[test]
    fn test_threshold_updates() {
        let config = VerificationConfig::default();
        let mut verifier = SpeakerVerifier::new(config).unwrap();

        // Test normal threshold update
        verifier.update_threshold(0.9);
        assert_eq!(verifier.config.threshold, 0.9);

        // Test clamping (should clamp to 1.0)
        verifier.update_threshold(1.5);
        assert_eq!(verifier.config.threshold, 1.0);

        // Test clamping (should clamp to 0.0)
        verifier.update_threshold(-0.1);
        assert_eq!(verifier.config.threshold, 0.0);
    }

    #[test]
    fn test_strict_mode() {
        let config = VerificationConfig {
            threshold: 0.8,
            strict_threshold: 0.95,
            ..VerificationConfig::default()
        };
        let mut verifier = SpeakerVerifier::new(config).unwrap();

        assert_eq!(verifier.config.threshold, 0.8);

        // Enable strict mode
        verifier.set_strict_mode(true);
        assert_eq!(verifier.config.threshold, 0.95);

        // Disable strict mode
        verifier.set_strict_mode(false);
        assert_eq!(verifier.config.threshold, 0.8); // Should return to default
    }

    #[test]
    fn test_sample_quality_assessment() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        // Create a high-quality sample (mostly signal, some quiet periods)
        let mut high_quality_audio = vec![0.0; 16000];
        for (i, sample) in high_quality_audio.iter_mut().enumerate() {
            if i < 12000 {
                *sample = 0.7; // Strong signal
            } else {
                *sample = 0.05; // Low noise floor
            }
        }
        let high_quality_sample =
            VoiceSample::new("high_quality".to_string(), high_quality_audio, 16000);

        // Create a low-quality sample (mostly noise, some signal)
        let mut noisy_audio = vec![0.0; 16000];
        for (i, sample) in noisy_audio.iter_mut().enumerate() {
            if i < 8000 {
                *sample = 0.3; // Weaker signal
            } else {
                *sample = 0.6; // High noise floor
            }
        }
        let low_quality_sample = VoiceSample::new("low_quality".to_string(), noisy_audio, 16000);

        let quality_high = verifier.assess_sample_quality(&high_quality_sample);
        let quality_low = verifier.assess_sample_quality(&low_quality_sample);

        // High quality sample should have better scores
        assert!(quality_high.overall_score > quality_low.overall_score);
        assert!(quality_high.snr > quality_low.snr);
    }

    // --- MFCC / acoustic feature tests ---

    #[test]
    fn test_compute_mel_features_length() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        let sample_rate = 16000_u32;
        let audio: Vec<f32> = (0..8000)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.5
            })
            .collect();

        let mfcc = verifier.compute_mel_features(&audio, sample_rate).unwrap();
        assert_eq!(
            mfcc.len(),
            13,
            "compute_mel_features must return exactly 13 coefficients"
        );
    }

    #[test]
    fn test_compute_mel_features_not_constant() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        let sample_rate = 16000_u32;
        let n = 8000_usize;

        // Sine wave at 440 Hz
        let sine_audio: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.8
            })
            .collect();

        // Alternating noise-like signal (broadband)
        let noise_audio: Vec<f32> = (0..n)
            .map(|i| {
                // Deterministic pseudo-noise via simple LCG for reproducibility
                let v = ((i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    >> 33) as f32
                    / u32::MAX as f32;
                v * 2.0 - 1.0
            })
            .collect();

        let mfcc_sine = verifier
            .compute_mel_features(&sine_audio, sample_rate)
            .unwrap();
        let mfcc_noise = verifier
            .compute_mel_features(&noise_audio, sample_rate)
            .unwrap();

        // The two feature vectors must not be identical (real computation, not hardcoded)
        let all_equal = mfcc_sine
            .iter()
            .zip(mfcc_noise.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            !all_equal,
            "MFCC must differ for different input signals; got {:?} == {:?}",
            mfcc_sine, mfcc_noise
        );

        // Also confirm nothing is still the hardcoded placeholder [0.5; 13]
        let all_half = mfcc_sine.iter().all(|&v| (v - 0.5).abs() < 1e-6);
        assert!(
            !all_half,
            "compute_mel_features still returns hardcoded [0.5; 13]"
        );
    }

    #[test]
    fn test_extract_acoustic_features_length() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        let sample_rate = 16000_u32;
        let audio: Vec<f32> = (0..8000)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 300.0 * t).sin() * 0.6
            })
            .collect();

        let features = verifier
            .extract_acoustic_features(&audio, sample_rate)
            .unwrap();

        // 4 basic features + 13 MFCC = 17 minimum
        assert!(
            features.len() >= 17,
            "extract_acoustic_features must return at least 17 values (4 basic + 13 MFCC), got {}",
            features.len()
        );
    }

    // --- Required MFCC unit tests ---

    /// 440 Hz sine at 22050 Hz sample rate must return exactly 13 coefficients.
    #[test]
    fn test_mel_features_shape() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        let sample_rate = 22050_u32;
        // ~0.5 seconds of pure 440 Hz tone
        let n_samples = sample_rate as usize / 2;
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.7
            })
            .collect();

        let coeffs = verifier
            .compute_mel_features(&audio, sample_rate)
            .expect("compute_mel_features must not error on valid input");

        assert_eq!(
            coeffs.len(),
            13,
            "Expected exactly 13 MFCC coefficients, got {}",
            coeffs.len()
        );
    }

    /// All-zero input (silence) must return 13 finite coefficients without panicking.
    #[test]
    fn test_mel_features_silence() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        let sample_rate = 16000_u32;
        // One second of complete silence
        let audio = vec![0.0_f32; sample_rate as usize];

        let coeffs = verifier
            .compute_mel_features(&audio, sample_rate)
            .expect("compute_mel_features must not error on silence");

        assert_eq!(
            coeffs.len(),
            13,
            "Expected exactly 13 MFCC coefficients for silence, got {}",
            coeffs.len()
        );
        for (i, &c) in coeffs.iter().enumerate() {
            assert!(
                c.is_finite(),
                "MFCC coefficient[{i}] is not finite for silence input: {c}"
            );
        }
    }

    /// Two spectrally distinct tones must produce different MFCC vectors.
    #[test]
    fn test_mel_features_different() {
        let config = VerificationConfig::default();
        let verifier = SpeakerVerifier::new(config).unwrap();

        let sample_rate = 16000_u32;
        let n_samples = sample_rate as usize / 2; // 0.5 s

        // Low-frequency tone (200 Hz)
        let low_tone: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.8
            })
            .collect();

        // High-frequency tone (3500 Hz)
        let high_tone: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 3500.0 * t).sin() * 0.8
            })
            .collect();

        let mfcc_low = verifier
            .compute_mel_features(&low_tone, sample_rate)
            .expect("compute_mel_features must not error on low tone");
        let mfcc_high = verifier
            .compute_mel_features(&high_tone, sample_rate)
            .expect("compute_mel_features must not error on high tone");

        assert_eq!(mfcc_low.len(), 13);
        assert_eq!(mfcc_high.len(), 13);

        // The two vectors must be meaningfully different (L1 distance > small threshold)
        let l1: f32 = mfcc_low
            .iter()
            .zip(mfcc_high.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            l1 > 0.1,
            "MFCC vectors for 200 Hz and 3500 Hz tones must differ; L1 distance = {l1}"
        );
    }
}
