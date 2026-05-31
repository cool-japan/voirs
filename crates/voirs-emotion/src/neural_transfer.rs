//! Neural Emotion Transfer System
//!
//! This module implements deep learning-based emotion transfer that can learn
//! emotion representations from reference audio and transfer them between speakers
//! while preserving speaker identity.
//!
//! ## Features
//!
//! - **Emotion Embedding Learning**: Learn compact emotion representations
//! - **Cross-speaker Transfer**: Transfer emotions between different speakers
//! - **Identity Preservation**: Maintain speaker characteristics during transfer
//! - **Fine-grained Control**: Attention-based emotion modulation
//!
//! ## Architecture
//!
//! The system uses a VAE (Variational Autoencoder) architecture with:
//! - Emotion encoder: Extracts emotion embeddings from audio
//! - Speaker encoder: Extracts speaker identity embeddings
//! - Decoder: Reconstructs audio with target emotion and speaker identity

use crate::{Error, Result};
use scirs2_core::numeric::Complex;
use scirs2_fft::{irfft, rfft};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;

/// Dimensionality of emotion embedding space
pub const EMOTION_EMBEDDING_DIM: usize = 64;

/// Dimensionality of speaker embedding space
pub const SPEAKER_EMBEDDING_DIM: usize = 256;

/// Neural emotion transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEmotionTransferConfig {
    /// Learning rate for training
    pub learning_rate: f32,
    /// Number of training iterations
    pub num_iterations: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Beta parameter for VAE KL divergence
    pub beta_kl: f32,
    /// Whether to use attention mechanism
    pub use_attention: bool,
}

impl Default for NeuralEmotionTransferConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            num_iterations: 1000,
            batch_size: 32,
            beta_kl: 0.5,
            use_attention: true,
        }
    }
}

/// Emotion embedding in latent space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionEmbedding {
    /// Embedding vector
    pub values: Vec<f32>,
    /// Confidence score
    pub confidence: f32,
}

impl EmotionEmbedding {
    /// Create a new emotion embedding
    pub fn new(values: Vec<f32>) -> Self {
        assert_eq!(
            values.len(),
            EMOTION_EMBEDDING_DIM,
            "Embedding dimension mismatch"
        );
        Self {
            values,
            confidence: 1.0,
        }
    }

    /// Create a zero embedding
    pub fn zero() -> Self {
        Self {
            values: vec![0.0; EMOTION_EMBEDDING_DIM],
            confidence: 0.0,
        }
    }

    /// Interpolate between two embeddings
    pub fn interpolate(&self, other: &EmotionEmbedding, alpha: f32) -> EmotionEmbedding {
        let alpha = alpha.clamp(0.0, 1.0);
        let values = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(&a, &b)| a * (1.0 - alpha) + b * alpha)
            .collect();

        EmotionEmbedding {
            values,
            confidence: self.confidence * (1.0 - alpha) + other.confidence * alpha,
        }
    }

    /// Compute cosine similarity with another embedding
    pub fn similarity(&self, other: &EmotionEmbedding) -> f32 {
        let dot_product: f32 = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(&a, &b)| a * b)
            .sum();

        let norm_a: f32 = self.values.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Speaker embedding representing speaker identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    /// Embedding vector
    pub values: Vec<f32>,
}

impl SpeakerEmbedding {
    /// Create a new speaker embedding
    pub fn new(values: Vec<f32>) -> Self {
        assert_eq!(
            values.len(),
            SPEAKER_EMBEDDING_DIM,
            "Speaker embedding dimension mismatch"
        );
        Self { values }
    }

    /// Create a zero embedding
    pub fn zero() -> Self {
        Self {
            values: vec![0.0; SPEAKER_EMBEDDING_DIM],
        }
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn frame_signal(signal: &[f32], frame_size: usize, hop: usize) -> Vec<Vec<f64>> {
    if signal.is_empty() || frame_size == 0 {
        return vec![vec![0.0f64; frame_size.max(1)]];
    }
    let mut frames = Vec::new();
    let mut start = 0usize;
    loop {
        let end = (start + frame_size).min(signal.len());
        let mut frame = vec![0.0f64; frame_size];
        for (i, &v) in signal[start..end].iter().enumerate() {
            frame[i] = v as f64;
        }
        frames.push(frame);
        if end >= signal.len() {
            break;
        }
        start += hop;
    }
    frames
}

fn apply_hann_f64(frame: &[f64], size: usize) -> Vec<f64> {
    frame
        .iter()
        .enumerate()
        .take(size)
        .map(|(i, &v)| {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos());
            v * w
        })
        .collect()
}

fn apply_hann_f32(frame: &[f32], size: usize) -> Vec<f32> {
    frame
        .iter()
        .enumerate()
        .take(size)
        .map(|(i, &v)| {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos());
            v * w
        })
        .collect()
}

fn rfft_magnitude_f64(signal: &[f64], n: usize) -> Vec<f64> {
    match rfft(signal, Some(n)) {
        Ok(spec) => spec
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect(),
        Err(_) => vec![0.0; n / 2 + 1],
    }
}

/// Neural emotion transfer model
#[derive(Debug)]
pub struct NeuralEmotionTransfer {
    /// Configuration
    config: NeuralEmotionTransferConfig,
    /// Learned emotion embeddings library
    emotion_library: HashMap<String, EmotionEmbedding>,
    /// Speaker embeddings library
    speaker_library: HashMap<String, SpeakerEmbedding>,
}

impl NeuralEmotionTransfer {
    /// Create a new neural emotion transfer model
    pub fn new(config: NeuralEmotionTransferConfig) -> Self {
        Self {
            config,
            emotion_library: HashMap::new(),
            speaker_library: HashMap::new(),
        }
    }

    /// Extract emotion embedding from audio features
    ///
    /// In a full implementation, this would use a trained neural network.
    /// This is a placeholder that demonstrates the interface.
    pub fn extract_emotion_embedding(&self, audio_features: &[f32]) -> Result<EmotionEmbedding> {
        // Placeholder: In production, this would use a trained encoder network
        // For now, we compute simple statistical features
        let embedding = self.compute_emotion_features(audio_features);
        Ok(EmotionEmbedding::new(embedding))
    }

    /// Extract speaker embedding from audio
    pub fn extract_speaker_embedding(&self, audio: &[f32]) -> Result<SpeakerEmbedding> {
        // Placeholder: Would use trained speaker encoder (e.g., x-vector, d-vector)
        let embedding = self.compute_speaker_features(audio);
        Ok(SpeakerEmbedding::new(embedding))
    }

    /// Transfer emotion from source to target while preserving target speaker identity
    pub fn transfer_emotion(
        &self,
        target_audio: &[f32],
        source_emotion: &EmotionEmbedding,
        target_speaker: &SpeakerEmbedding,
        intensity: f32,
    ) -> Result<Vec<f32>> {
        // Extract current emotion from target
        let target_features = self.extract_audio_features(target_audio);
        let current_emotion = self.extract_emotion_embedding(&target_features)?;

        // Interpolate emotions
        let mixed_emotion = current_emotion.interpolate(source_emotion, intensity);

        // Synthesize with mixed emotion and target speaker
        self.synthesize_audio(&mixed_emotion, target_speaker, target_audio.len())
    }

    /// Store an emotion embedding in the library
    pub fn store_emotion_embedding(&mut self, name: String, embedding: EmotionEmbedding) {
        self.emotion_library.insert(name, embedding);
    }

    /// Retrieve an emotion embedding from the library
    pub fn get_emotion_embedding(&self, name: &str) -> Option<&EmotionEmbedding> {
        self.emotion_library.get(name)
    }

    /// Store a speaker embedding in the library
    pub fn store_speaker_embedding(&mut self, name: String, embedding: SpeakerEmbedding) {
        self.speaker_library.insert(name, embedding);
    }

    /// Get a speaker embedding from the library
    pub fn get_speaker_embedding(&self, name: &str) -> Option<&SpeakerEmbedding> {
        self.speaker_library.get(name)
    }

    fn compute_emotion_features(&self, features: &[f32]) -> Vec<f32> {
        const FRAME_SIZE: usize = 256;
        const HOP: usize = 128;
        const N_FILTERS: usize = 10;
        const SR: f32 = 22050.0;

        let n_bins = FRAME_SIZE / 2 + 1;

        // --- STEP 1: per-frame triangular filterbank projection → 10 cepstral dims ---
        let frames: Vec<Vec<f64>> = frame_signal(features, FRAME_SIZE, HOP);
        let n_frames = frames.len();

        let mut filter_accum = vec![0.0f32; N_FILTERS];
        for frame in &frames {
            let windowed = apply_hann_f64(frame, FRAME_SIZE);
            let mag = rfft_magnitude_f64(&windowed, FRAME_SIZE);
            for k in 0..N_FILTERS {
                let lo = (k * n_bins) / N_FILTERS;
                let hi = ((k + 1) * n_bins) / N_FILTERS;
                let hi = hi.min(n_bins);
                let band: f64 = if hi > lo {
                    mag[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
                } else {
                    0.0
                };
                filter_accum[k] += (band as f32).ln_1p();
            }
        }
        let n_frames_f = n_frames.max(1) as f32;
        let step1: Vec<f32> = filter_accum.iter().map(|&v| v / n_frames_f).collect();

        // --- STEP 2: prosodic features (F0 via normalized autocorrelation) → 8 dims ---
        let tau_lo = FRAME_SIZE / 6;
        let tau_hi = FRAME_SIZE;
        let voiced_threshold = 0.35_f32;

        let mut voiced_f0: Vec<f32> = Vec::new();
        for frame in &frames {
            let frame_f32: Vec<f32> = frame.iter().map(|&x| x as f32).collect();
            let energy: f32 = frame_f32.iter().map(|&x| x * x).sum::<f32>();
            if energy < 1e-10 {
                continue;
            }
            let r0: f32 = frame_f32.iter().map(|&x| x * x).sum();
            if r0 < 1e-10 {
                continue;
            }
            let mut best_tau = 0usize;
            let mut best_r = -1.0f32;
            for tau in tau_lo..tau_hi {
                let r: f32 = frame_f32[..FRAME_SIZE - tau]
                    .iter()
                    .zip(&frame_f32[tau..])
                    .map(|(&a, &b)| a * b)
                    .sum();
                let norm_r = r / (r0 + 1e-10);
                if norm_r > best_r {
                    best_r = norm_r;
                    best_tau = tau;
                }
            }
            if best_r > voiced_threshold && best_tau > 0 {
                voiced_f0.push(SR / best_tau as f32);
            }
        }

        let step2 = if voiced_f0.len() < 2 {
            vec![0.0f32; 8]
        } else {
            let mean_f0 = voiced_f0.iter().sum::<f32>() / voiced_f0.len() as f32;
            let std_f0 = {
                let var = voiced_f0
                    .iter()
                    .map(|&x| (x - mean_f0).powi(2))
                    .sum::<f32>()
                    / voiced_f0.len() as f32;
                var.sqrt()
            };
            let max_f0 = voiced_f0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_f0 = voiced_f0.iter().cloned().fold(f32::INFINITY, f32::min);
            let voiced_frac = voiced_f0.len() as f32 / n_frames_f;
            let jitter = if mean_f0 > 1e-10 && voiced_f0.len() >= 2 {
                let diff_sum: f32 = voiced_f0.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
                (diff_sum / (voiced_f0.len() - 1) as f32) / mean_f0
            } else {
                0.0
            };
            vec![
                mean_f0 / 500.0,
                std_f0 / 200.0,
                (max_f0 - min_f0) / 400.0,
                voiced_frac,
                jitter,
                0.0,
                0.0,
                0.0,
            ]
        };

        // --- STEP 3: energy envelope → 8 dims ---
        const N_SEGS: usize = 8;
        let seg_len = features.len().max(1) / N_SEGS;
        let mut step3 = vec![0.0f32; N_SEGS];
        if seg_len > 0 {
            for k in 0..N_SEGS {
                let lo = k * seg_len;
                let hi = ((k + 1) * seg_len).min(features.len());
                let rms = if hi > lo {
                    let sq_sum: f32 = features[lo..hi].iter().map(|&x| x * x).sum();
                    (sq_sum / (hi - lo) as f32).sqrt()
                } else {
                    0.0
                };
                step3[k] = rms;
            }
            let max_e = step3.iter().cloned().fold(0.0f32, f32::max) + 1e-10;
            for v in step3.iter_mut() {
                *v /= max_e;
            }
        }

        // --- STEP 4: spectral shape (full-signal rfft → 16 bands) → 16 dims ---
        let pad_len = features.len().next_power_of_two();
        let mut padded_f64: Vec<f64> = features.iter().map(|&x| x as f64).collect();
        padded_f64.resize(pad_len, 0.0);
        let full_mag = rfft_magnitude_f64(&padded_f64, pad_len);
        let full_bins = full_mag.len();
        const N_BANDS: usize = 16;
        let mut step4 = vec![0.0f32; N_BANDS];
        for b in 0..N_BANDS {
            let lo = (b * full_bins) / N_BANDS;
            let hi = (((b + 1) * full_bins) / N_BANDS).min(full_bins);
            let mean_mag: f64 = if hi > lo {
                full_mag[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
            } else {
                0.0
            };
            step4[b] = (mean_mag as f32).ln_1p();
        }

        // --- STEP 5: temporal features → 8 dims (4 computed, 4 zero-padded) ---
        let mut zcr_sum = 0.0f32;
        let mut energy_vals: Vec<f32> = Vec::with_capacity(n_frames);
        let mut centroid_sum = 0.0f32;
        let mut rolloff_sum = 0.0f32;
        for frame in &frames {
            let frame_f32: Vec<f32> = frame.iter().map(|&x| x as f32).collect();
            let zcr: f32 = frame_f32
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count() as f32
                / FRAME_SIZE as f32;
            zcr_sum += zcr;

            let rms: f32 =
                (frame_f32.iter().map(|&x| x * x).sum::<f32>() / FRAME_SIZE as f32).sqrt();
            energy_vals.push(rms);

            let windowed = apply_hann_f32(&frame_f32, FRAME_SIZE);
            let mag: Vec<f32> = rfft_magnitude_f64(
                &windowed.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                FRAME_SIZE,
            )
            .iter()
            .map(|&x| x as f32)
            .collect();
            let n_b = mag.len();
            let total_mag: f32 = mag.iter().sum::<f32>() + 1e-10;
            let centroid: f32 = mag
                .iter()
                .enumerate()
                .map(|(i, &m)| i as f32 * m)
                .sum::<f32>()
                / (total_mag * n_b as f32);
            centroid_sum += centroid;

            let total_e: f32 = mag.iter().map(|&m| m * m).sum::<f32>();
            let rolloff_e = total_e * 0.85;
            let mut cumsum = 0.0f32;
            let mut rolloff_bin = n_b as f32;
            for (i, &m) in mag.iter().enumerate() {
                cumsum += m * m;
                if cumsum >= rolloff_e {
                    rolloff_bin = i as f32 / n_b as f32;
                    break;
                }
            }
            rolloff_sum += rolloff_bin;
        }
        let nf = n_frames_f;
        let zcr_mean = zcr_sum / nf;
        let energy_mean = energy_vals.iter().sum::<f32>() / nf;
        let energy_var = energy_vals
            .iter()
            .map(|&e| (e - energy_mean).powi(2))
            .sum::<f32>()
            / nf;
        let centroid_mean = centroid_sum / nf;
        let rolloff_mean = rolloff_sum / nf;
        let step5 = vec![
            zcr_mean,
            energy_var,
            centroid_mean,
            rolloff_mean,
            0.0,
            0.0,
            0.0,
            0.0,
        ];

        // --- STEP 6: concatenate 10+8+8+16+8=50 → zero-pad to 64 ---
        let mut result: Vec<f32> = Vec::with_capacity(EMOTION_EMBEDDING_DIM);
        result.extend_from_slice(&step1);
        result.extend_from_slice(&step2);
        result.extend_from_slice(&step3);
        result.extend_from_slice(&step4);
        result.extend_from_slice(&step5);
        result.resize(EMOTION_EMBEDDING_DIM, 0.0);
        result
    }

    fn compute_speaker_features(&self, audio: &[f32]) -> Vec<f32> {
        const FRAME_SIZE: usize = 512;
        const HOP: usize = 256;
        const N_MFCC: usize = 26;
        const N_MEL: usize = 26;
        const SR: f32 = 22050.0;
        const F_LO: f32 = 80.0;
        const F_HI: f32 = 8000.0;

        let n_bins = FRAME_SIZE / 2 + 1;
        let frames = frame_signal(audio, FRAME_SIZE, HOP);
        let n_frames = frames.len().max(1);

        // --- STEP 1: MFCC statistics (52 dims: mean-26 + std-26) ---
        let mel_centers: Vec<f32> = {
            let mel_lo = hz_to_mel(F_LO);
            let mel_hi = hz_to_mel(F_HI);
            (0..N_MEL)
                .map(|k| mel_to_hz(mel_lo + (mel_hi - mel_lo) * k as f32 / (N_MEL + 1) as f32))
                .collect()
        };
        let mel_widths: Vec<f32> = {
            let mel_lo = hz_to_mel(F_LO);
            let mel_hi = hz_to_mel(F_HI);
            let step = (mel_hi - mel_lo) / (N_MEL + 1) as f32;
            let step_hz = mel_to_hz(mel_lo + step) - mel_to_hz(mel_lo);
            vec![step_hz; N_MEL]
        };

        let mut mfcc_frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);
        for frame in &frames {
            let windowed = apply_hann_f64(frame, FRAME_SIZE);
            let mag = rfft_magnitude_f64(&windowed, FRAME_SIZE);

            // Triangular mel filterbank with log
            let log_mel: Vec<f32> = (0..N_MEL)
                .map(|k| {
                    let center_bin = (mel_centers[k] / (SR / 2.0) * (n_bins - 1) as f32) as usize;
                    let width_bins =
                        ((mel_widths[k] / (SR / 2.0)) * (n_bins - 1) as f32).max(1.0) as usize;
                    let lo = center_bin.saturating_sub(width_bins);
                    let hi = (center_bin + width_bins + 1).min(n_bins);
                    let energy: f64 = mag[lo..hi].iter().sum::<f64>() / (hi - lo).max(1) as f64;
                    (energy as f32 + 1e-10).ln()
                })
                .collect();

            // DCT-II orthonormal projection onto N_MFCC coefficients
            let mfcc: Vec<f32> = (0..N_MFCC)
                .map(|n| {
                    let scale = if n == 0 {
                        (1.0 / N_MEL as f32).sqrt()
                    } else {
                        (2.0 / N_MEL as f32).sqrt()
                    };
                    scale
                        * log_mel
                            .iter()
                            .enumerate()
                            .map(|(m, &v)| {
                                v * (PI * n as f32 * (2 * m + 1) as f32 / (2 * N_MEL) as f32).cos()
                            })
                            .sum::<f32>()
                })
                .collect();
            mfcc_frames.push(mfcc);
        }

        let mut mfcc_mean = vec![0.0f32; N_MFCC];
        for frame in &mfcc_frames {
            for (a, &v) in mfcc_mean.iter_mut().zip(frame.iter()) {
                *a += v;
            }
        }
        for v in mfcc_mean.iter_mut() {
            *v /= n_frames as f32;
        }
        let mut mfcc_std = vec![0.0f32; N_MFCC];
        for frame in &mfcc_frames {
            for (s, (&v, &m)) in mfcc_std.iter_mut().zip(frame.iter().zip(mfcc_mean.iter())) {
                *s += (v - m).powi(2);
            }
        }
        for v in mfcc_std.iter_mut() {
            *v = (*v / n_frames as f32).sqrt();
        }
        let mut step1: Vec<f32> = Vec::with_capacity(52);
        step1.extend_from_slice(&mfcc_mean);
        step1.extend_from_slice(&mfcc_std);

        // --- STEP 2: sub-band energies (32 dims) from full rfft ---
        let pad_len = audio.len().next_power_of_two();
        let mut padded_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();
        padded_f64.resize(pad_len, 0.0);
        let full_mag = rfft_magnitude_f64(&padded_f64, pad_len);
        let full_bins = full_mag.len();
        const N_SUBBANDS: usize = 32;
        let mut step2 = vec![0.0f32; N_SUBBANDS];
        for b in 0..N_SUBBANDS {
            let lo = (b * full_bins) / N_SUBBANDS;
            let hi = (((b + 1) * full_bins) / N_SUBBANDS).min(full_bins);
            let mean_m: f64 = if hi > lo {
                full_mag[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
            } else {
                0.0
            };
            step2[b] = (mean_m as f32 + 1e-10).ln();
        }

        // --- STEP 3: spectral statistics (8 dims) ---
        // centroid, spread, skewness, kurtosis, rolloff85, flatness, zcr, rms
        let total_mag_full: f64 = full_mag.iter().sum::<f64>() + 1e-10;
        let centroid: f64 = full_mag
            .iter()
            .enumerate()
            .map(|(i, &m)| i as f64 * m)
            .sum::<f64>()
            / total_mag_full;
        let spread: f64 = (full_mag
            .iter()
            .enumerate()
            .map(|(i, &m)| (i as f64 - centroid).powi(2) * m)
            .sum::<f64>()
            / total_mag_full)
            .sqrt();
        let skewness: f64 = if spread > 1e-10 {
            full_mag
                .iter()
                .enumerate()
                .map(|(i, &m)| (i as f64 - centroid).powi(3) * m)
                .sum::<f64>()
                / (total_mag_full * spread.powi(3))
        } else {
            0.0
        };
        let kurtosis: f64 = if spread > 1e-10 {
            full_mag
                .iter()
                .enumerate()
                .map(|(i, &m)| (i as f64 - centroid).powi(4) * m)
                .sum::<f64>()
                / (total_mag_full * spread.powi(4))
        } else {
            0.0
        };
        let total_e_full: f64 = full_mag.iter().map(|&m| m * m).sum::<f64>() + 1e-10;
        let rolloff_thr = total_e_full * 0.85;
        let mut cumsum = 0.0f64;
        let mut rolloff85 = 0.0f64;
        for (i, &m) in full_mag.iter().enumerate() {
            cumsum += m * m;
            if cumsum >= rolloff_thr {
                rolloff85 = i as f64 / full_bins as f64;
                break;
            }
        }
        let flatness: f64 = {
            let log_mean =
                full_mag.iter().map(|&m| (m + 1e-10).ln()).sum::<f64>() / full_bins as f64;
            log_mean.exp() / (total_mag_full / full_bins as f64)
        };
        let zcr_global: f32 = if audio.len() > 1 {
            audio
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count() as f32
                / audio.len() as f32
        } else {
            0.0
        };
        let rms_global: f32 =
            (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len().max(1) as f32).sqrt();
        let max_c = centroid / full_bins as f64;
        let max_s = spread / full_bins as f64;
        let step3 = vec![
            max_c.clamp(0.0, 1.0) as f32,
            max_s.clamp(0.0, 1.0) as f32,
            (skewness / 10.0).tanh() as f32,
            ((kurtosis - 3.0) / 10.0).tanh() as f32,
            rolloff85 as f32,
            flatness.clamp(0.0, 1.0) as f32,
            zcr_global,
            rms_global.clamp(0.0, 1.0),
        ];

        // --- STEP 4: F0 statistics (8 dims) ---
        let tau_lo = FRAME_SIZE / 6;
        let tau_hi = FRAME_SIZE;
        let voiced_thr = 0.35f32;
        let mut voiced_f0: Vec<f32> = Vec::new();
        for frame in &frames {
            let frame_f32: Vec<f32> = frame.iter().map(|&x| x as f32).collect();
            let r0: f32 = frame_f32.iter().map(|&x| x * x).sum();
            if r0 < 1e-10 {
                continue;
            }
            let mut best_tau = 0usize;
            let mut best_r = -1.0f32;
            for tau in tau_lo..tau_hi {
                let r: f32 = frame_f32[..FRAME_SIZE - tau]
                    .iter()
                    .zip(&frame_f32[tau..])
                    .map(|(&a, &b)| a * b)
                    .sum();
                let norm_r = r / (r0 + 1e-10);
                if norm_r > best_r {
                    best_r = norm_r;
                    best_tau = tau;
                }
            }
            if best_r > voiced_thr && best_tau > 0 {
                voiced_f0.push(SR / best_tau as f32);
            }
        }
        let step4 = if voiced_f0.len() < 2 {
            vec![0.0f32; 8]
        } else {
            let mean_f0 = voiced_f0.iter().sum::<f32>() / voiced_f0.len() as f32;
            let std_f0 = {
                let var = voiced_f0
                    .iter()
                    .map(|&x| (x - mean_f0).powi(2))
                    .sum::<f32>()
                    / voiced_f0.len() as f32;
                var.sqrt()
            };
            let max_f0 = voiced_f0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_f0 = voiced_f0.iter().cloned().fold(f32::INFINITY, f32::min);
            let voiced_frac = voiced_f0.len() as f32 / n_frames as f32;
            let lags = [0.1f32, 0.2, 0.3, 0.4];
            let acf_vals: Vec<f32> = lags
                .iter()
                .map(|&lag| {
                    let tau = (lag * FRAME_SIZE as f32) as usize;
                    if tau == 0 || n_frames < 2 {
                        return 0.0;
                    }
                    let mean_sq: f32 =
                        voiced_f0.iter().map(|&x| x * x).sum::<f32>() / voiced_f0.len() as f32;
                    if mean_sq < 1e-10 {
                        return 0.0;
                    }
                    let pairs = voiced_f0.len().saturating_sub(tau);
                    if pairs == 0 {
                        return 0.0;
                    }
                    voiced_f0[..pairs]
                        .iter()
                        .zip(&voiced_f0[tau..])
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                        / (pairs as f32 * mean_sq)
                })
                .collect();
            vec![
                mean_f0 / 500.0,
                std_f0 / 200.0,
                (max_f0 - min_f0) / 400.0,
                voiced_frac,
                acf_vals[0],
                acf_vals[1],
                acf_vals[2],
                acf_vals[3],
            ]
        };

        // --- STEP 5: MFCC delta (26 dims) ---
        let step5: Vec<f32> = (0..N_MFCC)
            .map(|c| {
                if mfcc_frames.len() < 2 {
                    return 0.0;
                }
                let diff_sum: f32 = mfcc_frames
                    .windows(2)
                    .map(|w| (w[1][c] - w[0][c]).abs())
                    .sum();
                diff_sum / (mfcc_frames.len() - 1) as f32
            })
            .collect();

        // --- STEP 6: formant positions (8 dims: 4 freqs + 4 bandwidths) ---
        let lifter_cutoff = (FRAME_SIZE / 16).max(1);
        let formant_bands: [(f32, f32); 4] = [
            (200.0, 900.0),
            (900.0, 2500.0),
            (2500.0, 3500.0),
            (3500.0, 5000.0),
        ];
        let mut formant_dims = vec![0.0f32; 8];
        if !frames.is_empty() {
            let first_frame = &frames[0];
            let windowed = apply_hann_f64(first_frame, FRAME_SIZE);
            let mag = rfft_magnitude_f64(&windowed, FRAME_SIZE);
            let mut cepstrum = vec![0.0f64; FRAME_SIZE];
            for (i, &m) in mag.iter().enumerate() {
                cepstrum[i] = (m + 1e-10).ln();
            }
            // Zero cepstrum above lifter_cutoff (spectral envelope extraction)
            for v in cepstrum[lifter_cutoff..].iter_mut() {
                *v = 0.0;
            }
            // Reconstruct spectral envelope via rfft of cepstrum slice
            let env_mag: Vec<f64> = {
                let cep_slice = &cepstrum[..n_bins];
                cep_slice.iter().map(|&x| x.exp()).collect()
            };
            let nyquist = SR / 2.0;
            for (k, &(lo_hz, hi_hz)) in formant_bands.iter().enumerate() {
                if lo_hz >= nyquist {
                    break;
                }
                let hi_hz = hi_hz.min(nyquist);
                let lo_bin = ((lo_hz / nyquist) * (n_bins - 1) as f32) as usize;
                let hi_bin = ((hi_hz / nyquist) * (n_bins - 1) as f32) as usize;
                let hi_bin = hi_bin.min(n_bins - 1);
                let mut peak_val = 0.0f64;
                let mut peak_bin = lo_bin;
                for b in lo_bin..=hi_bin {
                    if env_mag[b] > peak_val {
                        peak_val = env_mag[b];
                        peak_bin = b;
                    }
                }
                formant_dims[k] = (peak_bin as f32 / (n_bins - 1) as f32).clamp(0.0, 1.0);
                // Bandwidth ≈ 100 Hz, normalized by sr/2
                formant_dims[4 + k] = (100.0 / nyquist).clamp(0.0, 1.0);
            }
        }

        // --- Concatenate: 52+32+8+8+26+8=134 → zero-pad to 256 ---
        let mut result: Vec<f32> = Vec::with_capacity(SPEAKER_EMBEDDING_DIM);
        result.extend_from_slice(&step1);
        result.extend_from_slice(&step2);
        result.extend_from_slice(&step3);
        result.extend_from_slice(&step4);
        result.extend_from_slice(&step5);
        result.extend_from_slice(&formant_dims);
        result.resize(SPEAKER_EMBEDDING_DIM, 0.0);
        result
    }

    /// Extract audio features for emotion analysis
    fn extract_audio_features(&self, audio: &[f32]) -> Vec<f32> {
        // Placeholder: Would compute spectral features, prosody, etc.
        audio.to_vec()
    }

    fn synthesize_audio(
        &self,
        emotion: &EmotionEmbedding,
        speaker: &SpeakerEmbedding,
        length: usize,
    ) -> Result<Vec<f32>> {
        if length == 0 {
            return Ok(Vec::new());
        }
        const SR: f32 = 22050.0;

        // STEP 1: extract F0 from speaker embedding[0]
        let spk_nonzero = speaker.values.iter().any(|&v| v != 0.0);
        let f0 = if spk_nonzero {
            (speaker.values[0].abs() * 500.0).clamp(80.0, 400.0)
        } else {
            120.0
        };

        // STEP 2: energy contour from emotion.values[2..10]
        let anchor_slice = &emotion.values[2..10.min(emotion.values.len())];
        let anchors: Vec<f32> = if anchor_slice.is_empty() {
            vec![1.0; 8]
        } else {
            anchor_slice.to_vec()
        };
        let a_min = anchors.iter().cloned().fold(f32::INFINITY, f32::min);
        let a_max = anchors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let a_range = (a_max - a_min) + 1e-10;
        let anchors_norm: Vec<f32> = anchors.iter().map(|&v| (v - a_min) / a_range).collect();
        let n_anchors = anchors_norm.len();
        let envelope: Vec<f32> = (0..length)
            .map(|i| {
                let pos =
                    i as f32 / length.saturating_sub(1).max(1) as f32 * (n_anchors - 1) as f32;
                let lo_idx = (pos as usize).min(n_anchors - 1);
                let hi_idx = (lo_idx + 1).min(n_anchors - 1);
                let frac = pos - lo_idx as f32;
                anchors_norm[lo_idx] * (1.0 - frac) + anchors_norm[hi_idx] * frac
            })
            .collect();

        // STEP 3: formant centres from emotion.values[10..14]
        let formant_centers: Vec<f32> = (0..4)
            .map(|k| {
                let idx = 10 + k;
                if idx < emotion.values.len() {
                    emotion.values[idx] * 500.0 + 300.0
                } else {
                    300.0 + k as f32 * 125.0
                }
            })
            .collect();

        // STEP 4: harmonic stack
        let mut audio = vec![0.0f32; length];
        for h in 1u32..=6 {
            let freq = f0 * h as f32;
            if freq >= SR / 2.0 {
                break;
            }
            let amplitude = 1.0 / h as f32;
            for (i, s) in audio.iter_mut().enumerate() {
                *s += amplitude * (2.0 * PI * freq * i as f32 / SR).sin();
            }
        }
        // Apply energy envelope
        for (s, &env) in audio.iter_mut().zip(envelope.iter()) {
            *s *= env;
        }

        // Apply formant shaping in FFT domain
        let pad_len = length.next_power_of_two();
        let mut padded_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();
        padded_f64.resize(pad_len, 0.0);
        if let Ok(mut spectrum) = rfft(&padded_f64, Some(pad_len)) {
            let formant_bw = 100.0f32;
            for (bin_idx, c) in spectrum.iter_mut().enumerate() {
                let bin_hz = bin_idx as f32 * SR / pad_len as f32;
                let mut boost = 1.0f64;
                for &center in &formant_centers {
                    let delta = bin_hz - center;
                    let g = (-delta * delta / (2.0 * formant_bw * formant_bw)).exp();
                    boost += 0.5 * g as f64;
                }
                *c = Complex::new(c.re * boost, c.im * boost);
            }
            if let Ok(out) = irfft(&spectrum, Some(length)) {
                audio = out.iter().map(|&x| x as f32).collect();
                audio.resize(length, 0.0);
            } else {
                audio.resize(length, 0.0);
            }
            let peak = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            if peak > 1e-6 {
                let scale = 0.95 / peak;
                for s in audio.iter_mut() {
                    *s *= scale;
                }
            }
            for (s, &env) in audio.iter_mut().zip(envelope.iter()) {
                *s *= env;
            }
        }

        // STEP 5: scale by intensity
        let intensity = emotion.confidence.clamp(0.0, 1.0);
        for s in audio.iter_mut() {
            *s *= intensity;
        }

        // STEP 6: soft clip
        for s in audio.iter_mut() {
            *s = s.clamp(-0.95, 0.95);
        }

        Ok(audio)
    }
}

/// Attention-based emotion modulation
///
/// Applies attention weights to different parts of the emotion embedding
/// for fine-grained control.
#[derive(Debug, Clone)]
pub struct EmotionAttention {
    /// Attention weights for each embedding dimension
    weights: Vec<f32>,
}

impl EmotionAttention {
    /// Create uniform attention
    pub fn uniform() -> Self {
        Self {
            weights: vec![1.0; EMOTION_EMBEDDING_DIM],
        }
    }

    /// Create attention focusing on specific dimensions
    pub fn focused(focus_indices: &[usize], focus_strength: f32) -> Self {
        let mut weights = vec![1.0; EMOTION_EMBEDDING_DIM];

        for &idx in focus_indices {
            if idx < EMOTION_EMBEDDING_DIM {
                weights[idx] = focus_strength;
            }
        }

        Self { weights }
    }

    /// Apply attention to an emotion embedding
    pub fn apply(&self, embedding: &mut EmotionEmbedding) {
        for (value, &weight) in embedding.values.iter_mut().zip(&self.weights) {
            *value *= weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_embedding_creation() {
        let embedding = EmotionEmbedding::new(vec![0.5; EMOTION_EMBEDDING_DIM]);
        assert_eq!(embedding.values.len(), EMOTION_EMBEDDING_DIM);
        assert!((embedding.confidence - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_emotion_embedding_interpolation() {
        let emb1 = EmotionEmbedding::new(vec![0.0; EMOTION_EMBEDDING_DIM]);
        let emb2 = EmotionEmbedding::new(vec![1.0; EMOTION_EMBEDDING_DIM]);

        let mid = emb1.interpolate(&emb2, 0.5);
        assert!((mid.values[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_emotion_similarity() {
        let emb1 = EmotionEmbedding::new(vec![1.0; EMOTION_EMBEDDING_DIM]);
        let emb2 = EmotionEmbedding::new(vec![1.0; EMOTION_EMBEDDING_DIM]);
        let emb3 = EmotionEmbedding::new(vec![-1.0; EMOTION_EMBEDDING_DIM]);

        let sim_same = emb1.similarity(&emb2);
        let sim_opposite = emb1.similarity(&emb3);

        assert!((sim_same - 1.0).abs() < 1e-5);
        assert!((sim_opposite + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_neural_transfer_creation() {
        let config = NeuralEmotionTransferConfig::default();
        let transfer = NeuralEmotionTransfer::new(config);

        assert!(transfer.emotion_library.is_empty());
        assert!(transfer.speaker_library.is_empty());
    }

    #[test]
    fn test_emotion_library() {
        let config = NeuralEmotionTransferConfig::default();
        let mut transfer = NeuralEmotionTransfer::new(config);

        let embedding = EmotionEmbedding::new(vec![0.5; EMOTION_EMBEDDING_DIM]);
        transfer.store_emotion_embedding("happy".to_string(), embedding.clone());

        let retrieved = transfer.get_emotion_embedding("happy").unwrap();
        assert_eq!(retrieved.values.len(), EMOTION_EMBEDDING_DIM);
    }

    #[test]
    fn test_speaker_library() {
        let config = NeuralEmotionTransferConfig::default();
        let mut transfer = NeuralEmotionTransfer::new(config);

        let embedding = SpeakerEmbedding::new(vec![0.5; SPEAKER_EMBEDDING_DIM]);
        transfer.store_speaker_embedding("speaker1".to_string(), embedding);

        let retrieved = transfer.get_speaker_embedding("speaker1").unwrap();
        assert_eq!(retrieved.values.len(), SPEAKER_EMBEDDING_DIM);
    }

    #[test]
    fn test_emotion_attention() {
        let attention = EmotionAttention::uniform();
        assert_eq!(attention.weights.len(), EMOTION_EMBEDDING_DIM);

        let mut embedding = EmotionEmbedding::new(vec![1.0; EMOTION_EMBEDDING_DIM]);
        attention.apply(&mut embedding);

        // Uniform attention shouldn't change values
        assert!((embedding.values[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_focused_attention() {
        let attention = EmotionAttention::focused(&[0, 1], 2.0);
        let mut embedding = EmotionEmbedding::new(vec![1.0; EMOTION_EMBEDDING_DIM]);

        attention.apply(&mut embedding);

        // First dimensions should be amplified
        assert!((embedding.values[0] - 2.0).abs() < 1e-5);
        assert!((embedding.values[1] - 2.0).abs() < 1e-5);
        // Other dimensions unchanged
        assert!((embedding.values[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_extract_emotion_embedding() {
        let config = NeuralEmotionTransferConfig::default();
        let transfer = NeuralEmotionTransfer::new(config);

        let audio_features = vec![0.5; 1000];
        let embedding = transfer.extract_emotion_embedding(&audio_features).unwrap();

        assert_eq!(embedding.values.len(), EMOTION_EMBEDDING_DIM);
    }

    #[test]
    fn test_extract_speaker_embedding() {
        let config = NeuralEmotionTransferConfig::default();
        let transfer = NeuralEmotionTransfer::new(config);

        let audio = vec![0.5; 44100]; // 1 second at 44.1kHz
        let embedding = transfer.extract_speaker_embedding(&audio).unwrap();

        assert_eq!(embedding.values.len(), SPEAKER_EMBEDDING_DIM);
    }

    #[test]
    fn test_emotion_features_shape() {
        let config = NeuralEmotionTransferConfig::default();
        let transfer = NeuralEmotionTransfer::new(config);

        let input: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.1).sin()).collect();
        let features = transfer.compute_emotion_features(&input);
        assert_eq!(features.len(), EMOTION_EMBEDDING_DIM);
    }

    #[test]
    fn test_speaker_features_shape() {
        let config = NeuralEmotionTransferConfig::default();
        let transfer = NeuralEmotionTransfer::new(config);

        let input: Vec<f32> = (0..8192).map(|i| (i as f32 * 0.1).sin()).collect();
        let features = transfer.compute_speaker_features(&input);
        assert_eq!(features.len(), SPEAKER_EMBEDDING_DIM);
    }

    #[test]
    fn test_synthesize_creates_audio() {
        let config = NeuralEmotionTransferConfig::default();
        let transfer = NeuralEmotionTransfer::new(config);

        let mut emotion_vals = vec![0.0f32; EMOTION_EMBEDDING_DIM];
        emotion_vals[0] = 0.8;
        emotion_vals[2] = 0.5;
        emotion_vals[3] = 0.7;
        let emotion = EmotionEmbedding {
            values: emotion_vals,
            confidence: 0.9,
        };

        let mut speaker_vals = vec![0.0f32; SPEAKER_EMBEDDING_DIM];
        speaker_vals[0] = 0.4;
        let speaker = SpeakerEmbedding {
            values: speaker_vals,
        };

        let length = 2048;
        let audio = transfer
            .synthesize_audio(&emotion, &speaker, length)
            .unwrap();

        assert_eq!(audio.len(), length);
        assert!(audio.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn test_synthesize_silence_from_zero_embedding() {
        let config = NeuralEmotionTransferConfig::default();
        let transfer = NeuralEmotionTransfer::new(config);

        let emotion = EmotionEmbedding::zero();
        let speaker = SpeakerEmbedding::zero();

        let length = 1024;
        let audio = transfer
            .synthesize_audio(&emotion, &speaker, length)
            .unwrap();

        assert_eq!(audio.len(), length);
        let max_abs = audio.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 0.01,
            "Expected near-silence, got max_abs={max_abs}"
        );
    }
}
