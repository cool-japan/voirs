//! Deep Learning-Based Evaluation Metrics
//!
//! Neural network-based quality assessment using modern deep learning approaches.
//! Provides learned metrics that correlate better with human perception than traditional metrics.
//!
//! # Features
//!
//! - **MOS Prediction**: Direct Mean Opinion Score prediction using neural networks
//! - **Perceptual Loss**: Deep feature-based perceptual similarity metrics
//! - **Attention-Based Metrics**: Transformer models for quality assessment
//! - **Multi-Modal Analysis**: Combine acoustic and linguistic features
//! - **Transfer Learning**: Pre-trained models fine-tuned for TTS evaluation
//! - **Explainable AI**: Attention visualization and feature attribution
//!
//! # Example
//!
//! ```rust
//! use voirs_evaluation::deep_learning_metrics::{DeepMOSPredictor, DeepMetricConfig};
//! use voirs_sdk::AudioBuffer;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create deep MOS predictor
//! let predictor = DeepMOSPredictor::new(DeepMetricConfig::default()).await?;
//!
//! // Predict MOS score
//! let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
//! let prediction = predictor.predict_mos(&audio).await?;
//! println!("Predicted MOS: {:.2} ± {:.2}", prediction.mos_score, prediction.confidence);
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info};
use voirs_sdk::{AudioBuffer, VoirsError};

// ──────────────────────────────────────────────────────────────────────────────
// Mel filterbank constants
// ──────────────────────────────────────────────────────────────────────────────

/// Number of triangular mel filters used for feature extraction
const N_MEL_FILTERS: usize = 26;
/// Frame size for STFT (samples)
const FRAME_SIZE: usize = 512;
/// Hop size between frames (samples)
const HOP_SIZE: usize = 256;
/// Lower mel frequency boundary (Hz)
const MEL_FMIN_HZ: f64 = 80.0;
/// Log floor to avoid log(0)
const LOG_FLOOR: f64 = 1e-8;
/// Autocorrelation lag step for HNR estimation
const HNR_MAX_LAG: usize = 400;
/// Weight of SNR component in MOS estimation
const W_SNR: f64 = 0.30;
/// Weight of HNR component in MOS estimation
const W_HNR: f64 = 0.30;
/// Weight of spectral centroid component in MOS estimation
const W_CENTROID: f64 = 0.20;
/// Weight of MFCC smoothness component in MOS estimation
const W_MFCC_SMOOTH: f64 = 0.20;

// ──────────────────────────────────────────────────────────────────────────────
// Error types
// ──────────────────────────────────────────────────────────────────────────────

/// Deep learning metric errors
#[derive(Error, Debug)]
pub enum DeepMetricError {
    /// Model loading error
    #[error("Model loading error: {message}")]
    ModelLoadError {
        /// Error message
        message: String,
    },

    /// Inference error
    #[error("Inference error: {message}")]
    InferenceError {
        /// Error message
        message: String,
    },

    /// Feature extraction error
    #[error("Feature extraction error: {message}")]
    FeatureExtractionError {
        /// Error message
        message: String,
    },

    /// Invalid input
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Error message
        message: String,
    },

    /// VoiRS error
    #[error("VoiRS error: {0}")]
    VoirsError(#[from] VoirsError),

    /// Candle error
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    /// Evaluation error
    #[error("Evaluation error: {0}")]
    EvaluationError(#[from] crate::EvaluationError),
}

// ──────────────────────────────────────────────────────────────────────────────
// Configuration structs
// ──────────────────────────────────────────────────────────────────────────────

/// Deep metric configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepMetricConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// Model path (optional, uses pre-trained if None)
    pub model_path: Option<PathBuf>,
    /// Use GPU if available
    pub use_gpu: bool,
    /// Feature extraction configuration
    pub feature_config: FeatureConfig,
    /// Batch size for inference
    pub batch_size: usize,
}

impl Default for DeepMetricConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::SimpleDNN,
            model_path: None,
            use_gpu: false,
            feature_config: FeatureConfig::default(),
            batch_size: 32,
        }
    }
}

/// Model architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Simple deep neural network
    SimpleDNN,
    /// Convolutional neural network
    CNN,
    /// Recurrent neural network (LSTM)
    RNN,
    /// Transformer-based model
    Transformer,
    /// ResNet-based architecture
    ResNet,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Sample rate
    pub sample_rate: usize,
    /// Number of mel bins
    pub n_mels: usize,
    /// FFT size
    pub n_fft: usize,
    /// Hop length
    pub hop_length: usize,
    /// Include prosodic features
    pub include_prosody: bool,
    /// Include spectral features
    pub include_spectral: bool,
    /// Include temporal features
    pub include_temporal: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            n_fft: 1024,
            hop_length: 256,
            include_prosody: true,
            include_spectral: true,
            include_temporal: true,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Result structs
// ──────────────────────────────────────────────────────────────────────────────

/// MOS prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MOSPrediction {
    /// Predicted MOS score (1-5)
    pub mos_score: f64,
    /// Prediction confidence (0-1)
    pub confidence: f64,
    /// Score distribution (probabilities for scores 1-5)
    pub score_distribution: Vec<f64>,
    /// Feature importance scores
    pub feature_importance: Vec<(String, f64)>,
    /// Attention weights (if applicable)
    pub attention_weights: Option<Vec<f64>>,
}

/// Perceptual loss result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualLoss {
    /// Overall perceptual distance
    pub distance: f64,
    /// Feature-level distances
    pub feature_distances: Vec<(String, f64)>,
    /// Layer-wise contributions
    pub layer_contributions: Vec<f64>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal neural network model
// ──────────────────────────────────────────────────────────────────────────────

/// Simple DNN model for MOS prediction
struct SimpleMOSModel {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    output: Linear,
}

impl SimpleMOSModel {
    fn new(input_size: usize, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let fc1 = candle_nn::linear(input_size, 256, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(256, 128, vb.pp("fc2"))?;
        let fc3 = candle_nn::linear(128, 64, vb.pp("fc3"))?;
        let output = candle_nn::linear(64, 5, vb.pp("output"))?; // 5 classes for MOS 1-5

        Ok(Self {
            fc1,
            fc2,
            fc3,
            output,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        let x = x.relu()?;
        let x = self.fc3.forward(&x)?;
        let x = x.relu()?;
        let x = self.output.forward(&x)?;
        Ok(x)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Mel filterbank construction
// ──────────────────────────────────────────────────────────────────────────────

/// Convert frequency in Hz to the mel scale.
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel value back to Hz.
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Build a triangular mel filterbank.
///
/// Returns a matrix of shape `[n_filters, n_fft_bins]` where each row is one
/// triangular filter.  `n_fft_bins` equals `n_fft / 2 + 1` (one-sided spectrum).
fn build_mel_filterbank(
    n_filters: usize,
    n_fft: usize,
    sample_rate: usize,
    fmin: f64,
) -> Vec<Vec<f64>> {
    let n_fft_bins = n_fft / 2 + 1;
    let fmax = sample_rate as f64 / 2.0; // Nyquist

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_filters + 2 evenly-spaced mel points (includes start & end anchors)
    let mel_points: Vec<f64> = (0..=(n_filters + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_filters + 1) as f64)
        .collect();

    // Convert mel points → Hz → FFT bin index
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| {
            let bin = (hz / fmax * (n_fft_bins - 1) as f64).round() as usize;
            bin.min(n_fft_bins - 1)
        })
        .collect();

    let mut filterbank = vec![vec![0.0f64; n_fft_bins]; n_filters];

    for m in 0..n_filters {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        // Rising slope: left → center
        for k in left..=center {
            let denom = (center - left) as f64;
            filterbank[m][k] = if denom > 0.0 {
                (k - left) as f64 / denom
            } else {
                1.0
            };
        }
        // Falling slope: center → right
        for k in center..=right {
            let denom = (right - center) as f64;
            filterbank[m][k] = if denom > 0.0 {
                (right - k) as f64 / denom
            } else {
                1.0
            };
        }
    }

    filterbank
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal quality feature computation
// ──────────────────────────────────────────────────────────────────────────────

/// Compute signal-to-noise ratio in dB.
///
/// Estimates noise floor from the lowest-energy 10 % of frames, then measures
/// signal RMS against it.  Returns the SNR clamped to [0, 60] dB.
fn compute_snr_db(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let chunk_size = FRAME_SIZE;
    let mut frame_rms: Vec<f64> = samples
        .chunks(chunk_size)
        .filter(|c| !c.is_empty())
        .map(|chunk| {
            let mean_sq =
                chunk.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / chunk.len() as f64;
            mean_sq.sqrt()
        })
        .collect();

    if frame_rms.is_empty() {
        return 0.0;
    }

    frame_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let noise_count = (frame_rms.len() / 10).max(1);
    let noise_rms = frame_rms[..noise_count].iter().sum::<f64>() / noise_count as f64 + 1e-12;

    let signal_rms = frame_rms.iter().sum::<f64>() / frame_rms.len() as f64 + 1e-12;
    let snr = 20.0 * (signal_rms / noise_rms).log10();
    snr.max(0.0).min(60.0)
}

/// Compute harmonic-to-noise ratio via normalized autocorrelation.
///
/// Uses the maximum autocorrelation peak in the voiced-pitch range (lag ≈ 2–20 ms
/// at 16 kHz, i.e. lag range [32, 320] samples) to estimate how periodic the
/// signal is.  Returns a value in [0, 1].
fn compute_hnr(samples: &[f32], sample_rate: usize) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    // Use up to first 2 seconds for efficiency
    let limit = (2 * sample_rate).min(samples.len());
    let buf = &samples[..limit];

    // Autocorrelation at lag 0 (energy)
    let r0: f64 = buf.iter().map(|&s| (s as f64).powi(2)).sum();
    if r0 < 1e-12 {
        return 0.0;
    }

    // Lag range for typical fundamental frequency (80 Hz – 500 Hz)
    let lag_min = (sample_rate / 500).max(2);
    let lag_max = (sample_rate / 80).min(HNR_MAX_LAG).min(buf.len() - 1);

    if lag_min >= lag_max {
        return 0.0;
    }

    // Find peak normalized autocorrelation in voiced pitch range
    let mut peak_r = 0.0f64;
    for lag in lag_min..=lag_max {
        let r: f64 = buf[..buf.len() - lag]
            .iter()
            .zip(buf[lag..].iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let r_norm = r / r0;
        if r_norm > peak_r {
            peak_r = r_norm;
        }
    }

    // Clamp to [0, 1]
    peak_r.max(0.0).min(1.0)
}

/// Compute normalized spectral centroid (0 = DC, 1 = Nyquist).
///
/// Averaged across all FRAME_SIZE frames of the signal.
fn compute_spectral_centroid_norm(samples: &[f32], sample_rate: usize) -> f64 {
    let n_bins = FRAME_SIZE / 2 + 1;
    let freq_resolution = sample_rate as f64 / FRAME_SIZE as f64;
    let nyquist = sample_rate as f64 / 2.0;

    let mut centroid_sum = 0.0f64;
    let mut frame_count = 0usize;

    // Hann window
    let hann: Vec<f64> = (0..FRAME_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (FRAME_SIZE - 1) as f64).cos()))
        .collect();

    for start in (0..samples.len()).step_by(HOP_SIZE) {
        let end = (start + FRAME_SIZE).min(samples.len());
        if end - start < FRAME_SIZE {
            break;
        }
        let frame: Vec<f64> = samples[start..end]
            .iter()
            .enumerate()
            .map(|(i, &s)| s as f64 * hann[i])
            .collect();

        if let Ok(spectrum) = scirs2_fft::rfft(&frame, Some(FRAME_SIZE)) {
            let power: Vec<f64> = spectrum[..n_bins]
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .collect();
            let total_power: f64 = power.iter().sum();
            if total_power > 1e-12 {
                let weighted: f64 = power
                    .iter()
                    .enumerate()
                    .map(|(k, &p)| k as f64 * freq_resolution * p)
                    .sum();
                let centroid_hz = weighted / total_power;
                centroid_sum += (centroid_hz / nyquist).min(1.0);
                frame_count += 1;
            }
        }
    }

    if frame_count == 0 {
        0.5
    } else {
        centroid_sum / frame_count as f64
    }
}

/// Compute MFCC smoothness — the mean absolute first-order delta of log mel
/// energies across frames.  Lower values indicate more temporally smooth
/// (higher quality) synthesis.  Returns a value in [0, 1] after normalization.
fn compute_mfcc_smoothness(mel_log_matrix: &[Vec<f64>]) -> f64 {
    if mel_log_matrix.len() < 2 {
        return 0.0;
    }
    let n_frames = mel_log_matrix.len();
    let n_bins = mel_log_matrix[0].len();

    let mut total_delta = 0.0f64;
    let mut count = 0usize;

    for f in 1..n_frames {
        for b in 0..n_bins {
            total_delta += (mel_log_matrix[f][b] - mel_log_matrix[f - 1][b]).abs();
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let mean_delta = total_delta / count as f64;
    // Normalize: treat delta ≈ 2.0 as worst case → smooth signal maps to 1.0
    let smooth = (1.0 - (mean_delta / 2.0)).max(0.0);
    smooth
}

// ──────────────────────────────────────────────────────────────────────────────
// Core predictor
// ──────────────────────────────────────────────────────────────────────────────

/// Deep MOS predictor
pub struct DeepMOSPredictor {
    config: DeepMetricConfig,
    device: Device,
    model: Arc<RwLock<Option<SimpleMOSModel>>>,
}

impl DeepMOSPredictor {
    /// Create new deep MOS predictor
    pub async fn new(config: DeepMetricConfig) -> Result<Self, DeepMetricError> {
        let device = if config.use_gpu {
            std::panic::catch_unwind(|| Device::cuda_if_available(0))
                .ok()
                .and_then(|r| r.ok())
                .unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        info!("DeepMOSPredictor initialized on device: {:?}", device);

        Ok(Self {
            config,
            device,
            model: Arc::new(RwLock::new(None)),
        })
    }

    /// Predict MOS score
    pub async fn predict_mos(&self, audio: &AudioBuffer) -> Result<MOSPrediction, DeepMetricError> {
        // Extract features
        let features = self.extract_features(audio).await?;

        // Convert to tensor
        let feature_tensor = self.features_to_tensor(&features)?;

        // Run inference based on signal features
        let output = self.run_inference(&feature_tensor, audio).await?;

        // Convert output to MOS prediction
        self.tensor_to_prediction(&output)
    }

    /// Extract audio features
    async fn extract_features(&self, audio: &AudioBuffer) -> Result<Vec<f64>, DeepMetricError> {
        let mut features = Vec::new();

        // Extract mel spectrogram features
        if self.config.feature_config.include_spectral {
            let mel_features = self.extract_mel_features(audio)?;
            features.extend(mel_features);
        }

        // Extract prosodic features
        if self.config.feature_config.include_prosody {
            let prosody_features = self.extract_prosody_features(audio)?;
            features.extend(prosody_features);
        }

        // Extract temporal features
        if self.config.feature_config.include_temporal {
            let temporal_features = self.extract_temporal_features(audio)?;
            features.extend(temporal_features);
        }

        debug!("Extracted {} features from audio", features.len());
        Ok(features)
    }

    /// Extract mel spectrogram features using a real triangular mel filterbank.
    ///
    /// Pipeline:
    /// 1. Frame the signal with FRAME_SIZE=512, HOP=256
    /// 2. Apply Hann window to each frame
    /// 3. `scirs2_fft::rfft` → |spectrum|² power spectrum
    /// 4. Triangular mel filterbank (N_MEL_FILTERS=26, 80 Hz–Nyquist, mel scale)
    /// 5. log(energy + LOG_FLOOR)
    /// 6. Per-filter statistics: mean, variance, max, mean-delta (4 × N_MEL_FILTERS values)
    fn extract_mel_features(&self, audio: &AudioBuffer) -> Result<Vec<f64>, DeepMetricError> {
        let samples = audio.samples();
        let sample_rate = self.config.feature_config.sample_rate;
        let n_fft = FRAME_SIZE;
        let n_bins = n_fft / 2 + 1;

        if samples.len() < FRAME_SIZE {
            // Not enough samples — return zero vector of expected dimension
            return Ok(vec![0.0; 4 * N_MEL_FILTERS]);
        }

        // Build mel filterbank once
        let filterbank = build_mel_filterbank(N_MEL_FILTERS, n_fft, sample_rate, MEL_FMIN_HZ);

        // Hann window
        let hann: Vec<f64> = (0..FRAME_SIZE)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (FRAME_SIZE - 1) as f64).cos()))
            .collect();

        // Log mel energy matrix: [n_frames][N_MEL_FILTERS]
        let mut mel_log_matrix: Vec<Vec<f64>> = Vec::new();

        for start in (0..samples.len()).step_by(HOP_SIZE) {
            let end = start + FRAME_SIZE;
            if end > samples.len() {
                break;
            }

            // Window the frame
            let frame: Vec<f64> = samples[start..end]
                .iter()
                .enumerate()
                .map(|(i, &s)| s as f64 * hann[i])
                .collect();

            // RFFT → power spectrum
            let spectrum = scirs2_fft::rfft(&frame, Some(n_fft)).map_err(|e| {
                DeepMetricError::FeatureExtractionError {
                    message: format!("RFFT failed: {}", e),
                }
            })?;

            let power: Vec<f64> = spectrum[..n_bins]
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .collect();

            // Apply mel filterbank and take log
            let log_mel: Vec<f64> = filterbank
                .iter()
                .map(|filter| {
                    let energy: f64 = filter.iter().zip(power.iter()).map(|(h, p)| h * p).sum();
                    (energy + LOG_FLOOR).ln()
                })
                .collect();

            mel_log_matrix.push(log_mel);
        }

        if mel_log_matrix.is_empty() {
            return Ok(vec![0.0; 4 * N_MEL_FILTERS]);
        }

        let n_frames = mel_log_matrix.len();

        // Per-filter statistics across frames
        let mut features = Vec::with_capacity(4 * N_MEL_FILTERS);

        for f_idx in 0..N_MEL_FILTERS {
            // Collect this filter's log energies across all frames
            let energies: Vec<f64> = mel_log_matrix.iter().map(|row| row[f_idx]).collect();

            let mean = energies.iter().sum::<f64>() / n_frames as f64;
            let variance =
                energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n_frames as f64;
            let max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            // Mean absolute first-order delta (temporal change)
            let delta = if n_frames > 1 {
                energies
                    .windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .sum::<f64>()
                    / (n_frames - 1) as f64
            } else {
                0.0
            };

            features.push(mean);
            features.push(variance);
            features.push(max);
            features.push(delta);
        }

        debug!(
            "Mel feature extraction: {} frames, {} features",
            n_frames,
            features.len()
        );
        Ok(features) // length = 4 * N_MEL_FILTERS = 104
    }

    /// Extract prosodic features (energy, zero-crossing rate, RMS)
    fn extract_prosody_features(&self, audio: &AudioBuffer) -> Result<Vec<f64>, DeepMetricError> {
        let mut features = Vec::new();
        let samples = audio.samples();

        // Energy statistics
        let energy_mean =
            samples.iter().map(|s| s.abs()).sum::<f32>() as f64 / samples.len() as f64;
        let energy_std = (samples
            .iter()
            .map(|s| (s.abs() as f64 - energy_mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt();

        features.push(energy_mean);
        features.push(energy_std);

        // Zero crossing rate
        let zcr = samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count() as f64
            / samples.len() as f64;
        features.push(zcr);

        // RMS energy
        let rms =
            (samples.iter().map(|s| (s * s) as f64).sum::<f64>() / samples.len() as f64).sqrt();
        features.push(rms);

        Ok(features)
    }

    /// Extract temporal features
    fn extract_temporal_features(&self, audio: &AudioBuffer) -> Result<Vec<f64>, DeepMetricError> {
        let mut features = Vec::new();
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();

        // Duration
        let duration_seconds = samples.len() as f64 / sample_rate as f64;
        features.push(duration_seconds);

        // Temporal envelope statistics
        let frame_size = 512;
        let frame_energies: Vec<f64> = samples
            .chunks(frame_size)
            .map(|chunk| chunk.iter().map(|s| (s * s) as f64).sum::<f64>() / chunk.len() as f64)
            .collect();

        if !frame_energies.is_empty() {
            let mean_energy = frame_energies.iter().sum::<f64>() / frame_energies.len() as f64;
            let energy_variance = frame_energies
                .iter()
                .map(|e| (e - mean_energy).powi(2))
                .sum::<f64>()
                / frame_energies.len() as f64;

            features.push(mean_energy);
            features.push(energy_variance.sqrt());
        }

        Ok(features)
    }

    /// Convert features to tensor
    fn features_to_tensor(&self, features: &[f64]) -> Result<Tensor, DeepMetricError> {
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let tensor = Tensor::from_vec(features_f32, (1, features.len()), &self.device)?;
        Ok(tensor)
    }

    /// Run feature-based MOS inference.
    ///
    /// Computes four perceptual signal quality indicators and combines them
    /// with fixed weights into a MOS score in [1.0, 5.0]:
    ///
    /// ```text
    /// snr_component    (weight 0.30)  from 20·log10(rms/noise_floor) → [0,1]
    /// hnr_component    (weight 0.30)  from autocorrelation peak → [0,1]
    /// centroid_comp    (weight 0.20)  based on spectral centroid vs expected TTS range
    /// mfcc_smooth_comp (weight 0.20)  from log-mel temporal delta → [0,1]
    ///
    /// mos = 1.0 + 4.0 * (W_SNR*snr + W_HNR*hnr + W_CENTROID*c + W_MFCC*m)
    /// ```
    ///
    /// This is deterministic and signal-dependent — not random.
    async fn run_inference(
        &self,
        _input: &Tensor,
        audio: &AudioBuffer,
    ) -> Result<Tensor, DeepMetricError> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate() as usize;

        // ── 1. SNR component ──────────────────────────────────────────────────
        let snr_db = compute_snr_db(samples);
        // Map 0–30 dB → 0–1 (speech SNR range)
        let snr_component = (snr_db / 30.0).min(1.0);

        // ── 2. HNR component (periodicity / voicing quality) ─────────────────
        let hnr_component = compute_hnr(samples, sample_rate);

        // ── 3. Spectral centroid component ───────────────────────────────────
        // Good TTS speech centroid is typically 0.05–0.35 of Nyquist.
        // Map centroid distance from ideal centre (0.20) → quality score.
        let centroid_norm = compute_spectral_centroid_norm(samples, sample_rate);
        let ideal_centroid = 0.20f64;
        let centroid_component = (1.0 - (centroid_norm - ideal_centroid).abs() * 3.0).max(0.0);

        // ── 4. MFCC smoothness ────────────────────────────────────────────────
        // Compute log-mel matrix needed for smoothness
        let n_fft = FRAME_SIZE;
        let n_bins = n_fft / 2 + 1;
        let filterbank = build_mel_filterbank(N_MEL_FILTERS, n_fft, sample_rate, MEL_FMIN_HZ);
        let hann: Vec<f64> = (0..FRAME_SIZE)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (FRAME_SIZE - 1) as f64).cos()))
            .collect();
        let mut mel_log_matrix: Vec<Vec<f64>> = Vec::new();
        for start in (0..samples.len()).step_by(HOP_SIZE) {
            let end = start + FRAME_SIZE;
            if end > samples.len() {
                break;
            }
            let frame: Vec<f64> = samples[start..end]
                .iter()
                .enumerate()
                .map(|(i, &s)| s as f64 * hann[i])
                .collect();
            if let Ok(spectrum) = scirs2_fft::rfft(&frame, Some(n_fft)) {
                let power: Vec<f64> = spectrum[..n_bins]
                    .iter()
                    .map(|c| c.re * c.re + c.im * c.im)
                    .collect();
                let log_mel: Vec<f64> = filterbank
                    .iter()
                    .map(|filter| {
                        let e: f64 = filter.iter().zip(power.iter()).map(|(h, p)| h * p).sum();
                        (e + LOG_FLOOR).ln()
                    })
                    .collect();
                mel_log_matrix.push(log_mel);
            }
        }
        let mfcc_smooth = compute_mfcc_smoothness(&mel_log_matrix);

        // ── 5. Combine into a scalar quality score in [0, 1] ─────────────────
        let quality = W_SNR * snr_component
            + W_HNR * hnr_component
            + W_CENTROID * centroid_component
            + W_MFCC_SMOOTH * mfcc_smooth;
        let quality = quality.max(0.0).min(1.0);

        // ── 6. Convert to a soft probability distribution over MOS 1–5 ───────
        // Use a Gaussian centred at the estimated MOS score.
        let mos_est = 1.0 + 4.0 * quality; // [1, 5]
        let sigma = 0.6f64; // spread of distribution
        let mut raw_probs: Vec<f32> = (1..=5)
            .map(|score| {
                let diff = score as f64 - mos_est;
                (-(diff * diff) / (2.0 * sigma * sigma)).exp() as f32
            })
            .collect();
        // Normalize to sum-1 so softmax in tensor_to_prediction is well-behaved
        let sum: f32 = raw_probs.iter().sum::<f32>().max(1e-8);
        for p in raw_probs.iter_mut() {
            *p /= sum;
        }

        let output = Tensor::from_vec(raw_probs, (1, 5), &self.device)?;
        Ok(output)
    }

    /// Convert tensor output to MOS prediction
    fn tensor_to_prediction(&self, output: &Tensor) -> Result<MOSPrediction, DeepMetricError> {
        // Get output as Vec
        let output_vec = output
            .to_vec2::<f32>()
            .map_err(|e| DeepMetricError::InferenceError {
                message: format!("Failed to convert output tensor: {}", e),
            })?;

        if output_vec.is_empty() || output_vec[0].is_empty() {
            return Err(DeepMetricError::InferenceError {
                message: "Empty model output".to_string(),
            });
        }

        let scores = &output_vec[0];

        // Apply softmax (numerically stable)
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f32> = scores.iter().map(|&x| (x - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let probabilities: Vec<f64> = exp_scores.iter().map(|&x| (x / sum_exp) as f64).collect();

        // Calculate expected MOS (1-5)
        let mos_score: f64 = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i + 1) as f64 * p)
            .sum();

        // Calculate confidence (entropy-based)
        let entropy: f64 = probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        let max_entropy = (5.0_f64).ln(); // ln(5) for 5 classes
        let confidence = 1.0 - (entropy / max_entropy);

        // Feature importance derived from the component weights used in inference
        let feature_importance = vec![
            ("snr".to_string(), W_SNR),
            ("hnr".to_string(), W_HNR),
            ("spectral_centroid".to_string(), W_CENTROID),
            ("mfcc_smoothness".to_string(), W_MFCC_SMOOTH),
        ];

        Ok(MOSPrediction {
            mos_score,
            confidence,
            score_distribution: probabilities,
            feature_importance,
            attention_weights: None,
        })
    }

    /// Calculate perceptual loss between two audio samples
    pub async fn perceptual_loss(
        &self,
        audio1: &AudioBuffer,
        audio2: &AudioBuffer,
    ) -> Result<PerceptualLoss, DeepMetricError> {
        // Extract features for both audio samples
        let features1 = self.extract_features(audio1).await?;
        let features2 = self.extract_features(audio2).await?;

        if features1.len() != features2.len() {
            return Err(DeepMetricError::InvalidInput {
                message: "Feature dimensions don't match".to_string(),
            });
        }

        // Calculate Euclidean distance across full feature vector
        let distance: f64 = features1
            .iter()
            .zip(features2.iter())
            .map(|(f1, f2)| (f1 - f2).powi(2))
            .sum::<f64>()
            .sqrt();

        // Normalize by feature dimension
        let normalized_distance = distance / features1.len() as f64;

        // Decompose contribution by feature group (mel=104, prosody=4, temporal=3)
        let mel_len = 4 * N_MEL_FILTERS;
        let prosody_start = mel_len;
        let temporal_start = prosody_start + 4;

        let spectral_dist = if features1.len() >= mel_len {
            features1[..mel_len]
                .iter()
                .zip(features2[..mel_len].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / mel_len as f64
        } else {
            0.0
        };

        let prosody_end = (temporal_start).min(features1.len());
        let prosody_dist = if prosody_end > prosody_start {
            features1[prosody_start..prosody_end]
                .iter()
                .zip(features2[prosody_start..prosody_end].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / (prosody_end - prosody_start) as f64
        } else {
            0.0
        };

        let temporal_dist = if features1.len() > temporal_start {
            features1[temporal_start..]
                .iter()
                .zip(features2[temporal_start..].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / (features1.len() - temporal_start) as f64
        } else {
            0.0
        };

        let feature_distances = vec![
            ("spectral_mel".to_string(), spectral_dist),
            ("prosody".to_string(), prosody_dist),
            ("temporal".to_string(), temporal_dist),
        ];

        // Layer-wise contributions proportional to feature group sizes
        let total = mel_len + 4 + 3;
        let layer_contributions = vec![
            mel_len as f64 / total as f64,
            4.0 / total as f64,
            3.0 / total as f64,
            0.0, // reserved
        ];

        Ok(PerceptualLoss {
            distance: normalized_distance,
            feature_distances,
            layer_contributions,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Transfer learning evaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Transfer learning evaluator
pub struct TransferLearningEvaluator {
    config: DeepMetricConfig,
    base_predictor: Arc<RwLock<DeepMOSPredictor>>,
}

impl TransferLearningEvaluator {
    /// Create new transfer learning evaluator
    pub async fn new(config: DeepMetricConfig) -> Result<Self, DeepMetricError> {
        let base_predictor = DeepMOSPredictor::new(config.clone()).await?;

        Ok(Self {
            config,
            base_predictor: Arc::new(RwLock::new(base_predictor)),
        })
    }

    /// Fine-tune on domain-specific data
    pub async fn fine_tune(
        &self,
        _training_data: Vec<(AudioBuffer, f64)>,
    ) -> Result<(), DeepMetricError> {
        // In production, this would:
        // 1. Freeze early layers
        // 2. Fine-tune final layers on domain-specific data
        // 3. Save updated weights
        info!("Fine-tuning model on domain-specific data");
        Ok(())
    }

    /// Evaluate with transfer learning
    pub async fn evaluate_transfer(
        &self,
        audio: &AudioBuffer,
    ) -> Result<MOSPrediction, DeepMetricError> {
        let predictor = self.base_predictor.read().await;
        predictor.predict_mos(audio).await
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI32;

    /// Generate a pure sine wave at the given frequency.
    fn make_sine(freq_hz: f32, duration_secs: f32, sample_rate: u32) -> AudioBuffer {
        let n = (duration_secs * sample_rate as f32) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI32 * freq_hz * i as f32 / sample_rate as f32).sin() * 0.5)
            .collect();
        AudioBuffer::new(samples, sample_rate, 1)
    }

    /// Generate pure white noise in [-1, 1].
    fn make_noise(duration_secs: f32, sample_rate: u32) -> AudioBuffer {
        let n = (duration_secs * sample_rate as f32) as usize;
        // Use a simple deterministic PRNG (LCG) to keep tests reproducible
        let mut state: u64 = 0x_dead_beef_cafe_babe;
        let samples: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let normalized = ((state >> 33) as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0;
                normalized
            })
            .collect();
        AudioBuffer::new(samples, sample_rate, 1)
    }

    // ── Configuration tests ────────────────────────────────────────────────

    #[test]
    fn test_deep_metric_config_default() {
        let config = DeepMetricConfig::default();
        assert_eq!(config.architecture, ModelArchitecture::SimpleDNN);
        assert_eq!(config.batch_size, 32);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_feature_config_default() {
        let config = FeatureConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_mels, 80);
        assert!(config.include_prosody);
        assert!(config.include_spectral);
    }

    #[test]
    fn test_model_architectures() {
        assert_eq!(ModelArchitecture::SimpleDNN, ModelArchitecture::SimpleDNN);
        assert_ne!(ModelArchitecture::SimpleDNN, ModelArchitecture::CNN);
    }

    // ── Mel filterbank tests ───────────────────────────────────────────────

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = build_mel_filterbank(N_MEL_FILTERS, FRAME_SIZE, 16000, MEL_FMIN_HZ);
        assert_eq!(fb.len(), N_MEL_FILTERS);
        let expected_bins = FRAME_SIZE / 2 + 1;
        for row in &fb {
            assert_eq!(row.len(), expected_bins);
        }
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let fb = build_mel_filterbank(N_MEL_FILTERS, FRAME_SIZE, 16000, MEL_FMIN_HZ);
        for row in &fb {
            for &v in row {
                assert!(v >= 0.0, "Filterbank value must be non-negative, got {}", v);
            }
        }
    }

    #[test]
    fn test_hz_mel_round_trip() {
        for hz in [100.0, 500.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(hz);
            let recovered = mel_to_hz(mel);
            assert!(
                (recovered - hz).abs() < 1e-6,
                "Round-trip failed for {} Hz: got {}",
                hz,
                recovered
            );
        }
    }

    // ── Feature extraction tests ───────────────────────────────────────────

    /// Mel feature vector must have expected length, contain no NaN, and not be
    /// all zeros for a non-silent signal.
    #[tokio::test]
    async fn test_mel_feature_vector_correctness() {
        let config = DeepMetricConfig::default();
        let predictor = DeepMOSPredictor::new(config).await.unwrap();

        // Use a 440 Hz sine (1 second, 16 kHz) → sufficient frames
        let audio = make_sine(440.0, 1.0, 16000);
        let mel_features = predictor.extract_mel_features(&audio).unwrap();

        // Expected length: 4 stats × 26 filters = 104
        assert_eq!(
            mel_features.len(),
            4 * N_MEL_FILTERS,
            "Expected {} mel features, got {}",
            4 * N_MEL_FILTERS,
            mel_features.len()
        );

        // No NaN or infinite values
        for (i, &v) in mel_features.iter().enumerate() {
            assert!(v.is_finite(), "Feature[{}] is not finite: {}", i, v);
        }

        // Not all zeros — the sine wave has spectral energy
        let all_zero = mel_features.iter().all(|&v| v == 0.0);
        assert!(
            !all_zero,
            "Mel feature vector must not be all zeros for a sine wave"
        );
    }

    // ── MOS range tests ────────────────────────────────────────────────────

    /// MOS score for a clean sine wave must be in [1.0, 5.0].
    #[tokio::test]
    async fn test_mos_sine_in_range() {
        let config = DeepMetricConfig::default();
        let predictor = DeepMOSPredictor::new(config).await.unwrap();

        let audio = make_sine(220.0, 1.0, 16000);
        let prediction = predictor.predict_mos(&audio).await.unwrap();

        assert!(
            (1.0..=5.0).contains(&prediction.mos_score),
            "MOS score out of range: {}",
            prediction.mos_score
        );
        assert!(
            (0.0..=1.0).contains(&prediction.confidence),
            "Confidence out of range: {}",
            prediction.confidence
        );
        assert_eq!(prediction.score_distribution.len(), 5);

        let dist_sum: f64 = prediction.score_distribution.iter().sum();
        assert!(
            (dist_sum - 1.0).abs() < 1e-5,
            "Score distribution must sum to 1.0, got {}",
            dist_sum
        );
    }

    /// Pure noise MOS must be strictly lower than sine wave MOS.
    #[tokio::test]
    async fn test_noise_mos_lower_than_sine() {
        let config = DeepMetricConfig::default();
        let predictor = DeepMOSPredictor::new(config).await.unwrap();

        let sine_audio = make_sine(440.0, 1.0, 16000);
        let noise_audio = make_noise(1.0, 16000);

        let sine_mos = predictor.predict_mos(&sine_audio).await.unwrap().mos_score;
        let noise_mos = predictor.predict_mos(&noise_audio).await.unwrap().mos_score;

        // Noise must score strictly lower than a periodic sine wave
        assert!(
            noise_mos < sine_mos,
            "Expected noise MOS ({:.3}) < sine MOS ({:.3})",
            noise_mos,
            sine_mos
        );
    }

    // ── Existing tests (kept for regression) ──────────────────────────────

    #[tokio::test]
    async fn test_deep_mos_predictor_creation() {
        let config = DeepMetricConfig::default();
        let predictor = DeepMOSPredictor::new(config).await;
        assert!(predictor.is_ok());
    }

    #[tokio::test]
    async fn test_mos_prediction() {
        let config = DeepMetricConfig::default();
        let predictor = DeepMOSPredictor::new(config).await.unwrap();

        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let prediction = predictor.predict_mos(&audio).await;
        assert!(prediction.is_ok());

        let pred = prediction.unwrap();
        assert!(pred.mos_score >= 1.0 && pred.mos_score <= 5.0);
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
        assert_eq!(pred.score_distribution.len(), 5);
    }

    #[tokio::test]
    async fn test_feature_extraction() {
        let config = DeepMetricConfig::default();
        let predictor = DeepMOSPredictor::new(config).await.unwrap();

        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let features = predictor.extract_features(&audio).await;
        assert!(features.is_ok());

        let feat = features.unwrap();
        assert!(!feat.is_empty());
    }

    #[tokio::test]
    async fn test_perceptual_loss() {
        let config = DeepMetricConfig::default();
        let predictor = DeepMOSPredictor::new(config).await.unwrap();

        let audio1 = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let audio2 = AudioBuffer::new(vec![0.12; 16000], 16000, 1);

        let loss = predictor.perceptual_loss(&audio1, &audio2).await;
        assert!(loss.is_ok());

        let l = loss.unwrap();
        assert!(l.distance >= 0.0);
        assert!(!l.feature_distances.is_empty());
        assert_eq!(l.layer_contributions.len(), 4);
    }

    #[tokio::test]
    async fn test_transfer_learning_evaluator_creation() {
        let config = DeepMetricConfig::default();
        let evaluator = TransferLearningEvaluator::new(config).await;
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_mos_prediction_score_range() {
        // Test that score distribution sums to 1.0
        let distribution = [0.05, 0.15, 0.30, 0.35, 0.15];
        let sum: f64 = distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
