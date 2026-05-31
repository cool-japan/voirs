//! Voice cloning module for VITS
//!
//! Provides speaker-adaptive voice cloning capabilities including:
//! - Few-shot voice cloning from audio samples
//! - Speaker embedding extraction and caching
//! - Voice quality analysis with real signal processing
//! - Fine-tuning with transcripts
//!
//! # Signal Processing Details
//!
//! ## Mel Spectrogram Conversion
//! Audio → frames (FRAME=1024, HOP=256) → Hann window → rfft → magnitude
//! → triangular mel filterbank (80 filters, 80 Hz – Nyquist) → log(energy + 1e-8)
//! → shape [n_frames, n_mels]
//!
//! ## L2 Normalisation
//! Per-vector: `x / sqrt(sum(x_i^2) + epsilon)` where epsilon = 1e-12.

use candle_core::{Device, Tensor};
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

use crate::{AcousticError, Result};

use super::utils::LinearLayer;

// SciRS2 FFT for real mel-spectrogram computation
use scirs2_fft::rfft;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Voice cloning configuration
#[derive(Debug, Clone)]
pub struct VoiceCloningConfig {
    /// Number of speaker embedding dimensions
    pub speaker_embedding_dim: usize,
    /// Number of adaptation samples required
    pub adaptation_samples: usize,
    /// Fine-tuning learning rate
    pub fine_tuning_rate: f32,
    /// Number of fine-tuning epochs
    pub fine_tuning_epochs: usize,
    /// Use few-shot learning approach
    pub few_shot_learning: bool,
    /// Voice similarity threshold
    pub similarity_threshold: f32,
    /// Sample rate assumed for raw audio tensors passed to the encoder
    pub assumed_sample_rate: u32,
}

impl Default for VoiceCloningConfig {
    fn default() -> Self {
        Self {
            speaker_embedding_dim: 512,
            adaptation_samples: 10,
            fine_tuning_rate: 0.0001,
            fine_tuning_epochs: 50,
            few_shot_learning: true,
            similarity_threshold: 0.85,
            assumed_sample_rate: 22050,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/// Speaker embedding with metadata
#[derive(Debug, Clone)]
pub struct SpeakerEmbedding {
    /// Speaker embedding vector
    pub embedding: Tensor,
    /// Voice quality metrics
    pub quality_metrics: VoiceQualityMetrics,
    /// Number of samples used for training
    pub sample_count: usize,
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
}

/// Voice quality metrics derived from real signal processing
#[derive(Debug, Clone)]
pub struct VoiceQualityMetrics {
    /// Pitch characteristics (fundamental frequency estimate, Hz)
    pub pitch_mean: f32,
    pub pitch_std: f32,
    /// Formant peak frequencies (Hz)
    pub formant_frequencies: Vec<f32>,
    /// Spectral centroid (Hz)
    pub spectral_centroid: f32,
    /// Spectral roll-off (Hz) at 85 % energy threshold
    pub spectral_rolloff: f32,
    /// Rough speaking-rate estimate (syllables/min proxy)
    pub speaking_rate: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel spectrogram DSP (pure Rust, uses scirs2_fft::rfft)
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for the internal mel spectrogram computation.
#[derive(Debug, Clone, Copy)]
pub struct MelParams {
    /// FFT window size (power of 2)
    pub n_fft: usize,
    /// Hop size between successive frames
    pub hop_length: usize,
    /// Number of mel filter banks
    pub n_mels: usize,
    /// Minimum frequency for mel filters (Hz)
    pub f_min: f32,
    /// Maximum frequency for mel filters (Hz) — typically Nyquist
    pub f_max: f32,
    /// Log floor to avoid log(0)
    pub log_floor: f32,
}

impl MelParams {
    /// Default parameters matching the VITS speaker encoder input: 80 mel bins,
    /// 1024-point FFT, 256-sample hop, 22 050 Hz sample rate.
    pub fn vits_default(sample_rate: u32) -> Self {
        Self {
            n_fft: 1024,
            hop_length: 256,
            n_mels: 80,
            f_min: 80.0,
            f_max: sample_rate as f32 / 2.0,
            log_floor: 1e-8,
        }
    }
}

/// Convert Hz to mel scale using the standard formula.
///
/// `mel = 2595 * log10(1 + hz / 700)`
#[inline]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale back to Hz.
///
/// `hz = 700 * (10^(mel / 2595) - 1)`
#[inline]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Build a triangular mel filterbank.
///
/// Returns a matrix of shape `[n_mels][n_fft/2+1]` where each row is one
/// triangular filter whose weights sum the FFT magnitude bins that fall inside
/// it.  Filters are normalised by their bandwidth (Slaney-style) so that
/// higher-frequency, wider filters do not dominate.
///
/// # Arguments
/// * `params`   – mel configuration
/// * `n_freqs`  – number of FFT frequency bins = `n_fft / 2 + 1`
fn build_mel_filterbank(params: &MelParams, n_freqs: usize) -> Vec<Vec<f32>> {
    let n_mels = params.n_mels;
    let f_min = params.f_min;
    let f_max = params.f_max;

    // n_mels + 2 evenly spaced mel points, then convert back to Hz
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=(n_mels + 1) {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_to_hz(mel));
    }

    // Map Hz frequencies to FFT bin indices (float, for interpolation)
    // bin = freq / (sample_rate / n_fft)  — but we only know f_max ~ nyquist
    // so: bin = freq * (n_fft / (2 * f_max)) * (n_freqs - 1) / (n_fft / 2)
    //        = freq * (n_freqs - 1) / f_max
    let hz_to_bin = |hz: f32| -> f32 { hz * (n_freqs - 1) as f32 / f_max };

    let bin_points: Vec<f32> = mel_points.iter().map(|&hz| hz_to_bin(hz)).collect();

    // Build filterbank matrix [n_mels][n_freqs]
    let mut filterbank = vec![vec![0.0_f32; n_freqs]; n_mels];

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        // Slaney normalisation: divide by the filter bandwidth in Hz
        let bandwidth = mel_points[m + 2] - mel_points[m];
        let norm = if bandwidth > 1e-6 {
            2.0 / bandwidth
        } else {
            1.0
        };

        for (k, slot) in filterbank[m].iter_mut().enumerate() {
            let k_f = k as f32;
            let weight = if k_f >= left && k_f <= center {
                (k_f - left) / (center - left + 1e-10)
            } else if k_f > center && k_f <= right {
                (right - k_f) / (right - center + 1e-10)
            } else {
                0.0
            };
            *slot = weight * norm;
        }
    }

    filterbank
}

/// Generate a Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0_f32; n];
    }
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos()))
        .collect()
}

/// Compute a mel spectrogram from a flat f32 audio slice.
///
/// Returns a matrix of shape `[n_frames][n_mels]` where every value is
/// `log(mel_energy + log_floor)`.
///
/// # Errors
/// Propagates any FFT errors as `AcousticError::ProcessingError`.
pub fn compute_mel_spectrogram(audio: &[f32], params: &MelParams) -> Result<Vec<Vec<f32>>> {
    let n_fft = params.n_fft;
    let hop = params.hop_length;
    let n_mels = params.n_mels;
    let n_freqs = n_fft / 2 + 1;

    if audio.is_empty() {
        return Err(AcousticError::ProcessingError {
            message: "Cannot compute mel spectrogram from empty audio".to_string(),
        });
    }

    // Build analysis window and filterbank once
    let window = hann_window(n_fft);
    let filterbank = build_mel_filterbank(params, n_freqs);

    // Determine frame positions
    let n_frames = if audio.len() >= n_fft {
        (audio.len() - n_fft) / hop + 1
    } else {
        1
    };

    let mut mel_frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);

    let mut frame_buf = vec![0.0_f64; n_fft]; // f64 for scirs2_fft

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;

        // Fill frame buffer with windowed audio; zero-pad if near the end
        for i in 0..n_fft {
            let sample_idx = start + i;
            frame_buf[i] = if sample_idx < audio.len() {
                audio[sample_idx] as f64 * window[i] as f64
            } else {
                0.0
            };
        }

        // Real FFT via scirs2_fft — returns n_fft/2+1 complex values
        let spectrum =
            rfft(&frame_buf, Some(n_fft)).map_err(|e| AcousticError::ProcessingError {
                message: format!("rfft failed on frame {frame_idx}: {e:?}"),
            })?;

        // Magnitude spectrum  (|Z|)
        let magnitude: Vec<f32> = spectrum
            .iter()
            .take(n_freqs)
            .map(|c| (c.re as f32).hypot(c.im as f32))
            .collect();

        // Apply mel filterbank and apply log compression
        let mut mel_frame = Vec::with_capacity(n_mels);
        for filter_row in filterbank.iter().take(n_mels) {
            let energy: f32 = filter_row
                .iter()
                .zip(magnitude.iter())
                .map(|(&w, &mag)| w * mag)
                .sum();
            mel_frame.push((energy + params.log_floor).ln());
        }
        mel_frames.push(mel_frame);
    }

    Ok(mel_frames)
}

// ─────────────────────────────────────────────────────────────────────────────
// L2 normalisation (epsilon-stabilised, applied per-vector on a 1-D Tensor)
// ─────────────────────────────────────────────────────────────────────────────

/// L2-normalise a flat embedding vector stored in a `Tensor`.
///
/// `x_normed = x / sqrt(sum(x_i^2) + epsilon)`
///
/// Applied per-vector (last dimension), stable against near-zero vectors.
/// Uses `affine(1.0, EPSILON)` for the scalar addition to avoid candle's
/// strict same-shape requirement on binary `add`.
///
/// # Errors
/// Forwards `candle_core` tensor errors as `AcousticError::ProcessingError`.
pub fn l2_normalize_embedding(x: &Tensor) -> Result<Tensor> {
    const EPSILON: f64 = 1e-12;

    // sum(x^2) keepdim along last axis  → shape [..., 1]
    let sq_sum = x.sqr()?.sum_keepdim(candle_core::D::Minus1)?;

    // Add scalar epsilon via affine (avoids same-shape binary-op requirement):
    //   affine(mul=1.0, add=EPSILON) computes element-wise 1.0 * sq_sum + EPSILON
    let sq_sum_eps = sq_sum.affine(1.0, EPSILON)?;

    // sqrt(sum_sq + epsilon)
    let norm = sq_sum_eps.sqrt()?;

    // Broadcast-divide: x / norm  (norm has shape [..., 1] and broadcasts)
    let x_normed = x.broadcast_div(&norm)?;
    Ok(x_normed)
}

/// Perform a linear layer forward pass with broadcasting bias addition.
///
/// Candle's binary `add` (`Tensor::add`) requires identical shapes, so we
/// call `broadcast_add` explicitly for the bias term.  This handles the
/// common case of input `[batch, in_features]` + bias `[out_features]`.
fn linear_forward_broadcast(weight: &Tensor, bias: &Tensor, input: &Tensor) -> Result<Tensor> {
    let linear_out = input.matmul(&weight.transpose(0, 1)?)?;
    let out = linear_out.broadcast_add(bias)?;
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Voice quality analysis (real signal processing via scirs2_fft)
// ─────────────────────────────────────────────────────────────────────────────

/// Analyse voice quality from raw audio (f32 slice) using FFT-based features.
///
/// Computes:
/// - Spectral centroid  (energy-weighted mean frequency, Hz)
/// - Spectral roll-off  (frequency below which 85 % of energy falls, Hz)
/// - Pitch mean / std   (zero-crossing-rate proxy for fundamental frequency)
/// - Formant peaks      (top-3 magnitude peaks in the 200 – 3500 Hz range)
/// - Speaking rate      (energy-envelope fluctuation proxy)
fn analyse_voice_quality_from_samples(
    audio: &[f32],
    sample_rate: u32,
) -> Result<VoiceQualityMetrics> {
    let n = audio.len();
    if n == 0 {
        return Ok(VoiceQualityMetrics {
            pitch_mean: 0.0,
            pitch_std: 0.0,
            formant_frequencies: vec![],
            spectral_centroid: 0.0,
            spectral_rolloff: 0.0,
            speaking_rate: 0.0,
        });
    }

    // ── Step 1: Whole-signal magnitude spectrum for spectral shape features ──
    // Use the largest power-of-2 that fits the signal (up to 16384)
    let fft_size = {
        let mut sz = 1_usize;
        while sz * 2 <= n.min(16384) {
            sz *= 2;
        }
        sz
    };

    let frame_f64: Vec<f64> = audio.iter().take(fft_size).map(|&s| s as f64).collect();
    let spectrum =
        rfft(&frame_f64, Some(fft_size)).map_err(|e| AcousticError::ProcessingError {
            message: format!("rfft failed in quality analysis: {e:?}"),
        })?;

    let n_freqs = fft_size / 2 + 1;
    let magnitudes: Vec<f32> = spectrum
        .iter()
        .take(n_freqs)
        .map(|c| (c.re as f32).hypot(c.im as f32))
        .collect();

    let freq_resolution = sample_rate as f32 / fft_size as f32;

    // ── Step 2: Spectral centroid ──
    let total_magnitude: f32 = magnitudes.iter().sum();
    let spectral_centroid = if total_magnitude > 1e-10 {
        magnitudes
            .iter()
            .enumerate()
            .map(|(k, &mag)| k as f32 * freq_resolution * mag)
            .sum::<f32>()
            / total_magnitude
    } else {
        0.0
    };

    // ── Step 3: Spectral roll-off at 85 % energy ──
    let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
    let rolloff_threshold = 0.85 * total_energy;
    let mut cumulative_energy = 0.0_f32;
    let mut spectral_rolloff = 0.0_f32;
    for (k, &mag) in magnitudes.iter().enumerate() {
        cumulative_energy += mag * mag;
        if cumulative_energy >= rolloff_threshold {
            spectral_rolloff = k as f32 * freq_resolution;
            break;
        }
    }

    // ── Step 4: Formant peaks — top-3 local maxima in 200–3500 Hz ──
    let bin_low = (200.0 / freq_resolution).ceil() as usize;
    let bin_high = ((3500.0 / freq_resolution) as usize).min(n_freqs.saturating_sub(2));

    let mut formant_candidates: Vec<(usize, f32)> = Vec::new();
    if bin_low + 1 < bin_high {
        for k in (bin_low + 1)..bin_high {
            if magnitudes[k] > magnitudes[k - 1] && magnitudes[k] > magnitudes[k + 1] {
                formant_candidates.push((k, magnitudes[k]));
            }
        }
    }
    // sort descending by magnitude, keep top-3
    formant_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut formant_frequencies: Vec<f32> = formant_candidates
        .iter()
        .take(3)
        .map(|(k, _)| *k as f32 * freq_resolution)
        .collect();
    formant_frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // ── Step 5: Pitch estimate via zero-crossing rate ──
    // ZCR (crossings per sample) ≈ 2 * F0 / sample_rate  →  F0 ≈ ZCR * sample_rate / 2
    let zcr_frame = 1024_usize.min(n);
    let zero_crossings = audio
        .windows(2)
        .take(zcr_frame - 1)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    let pitch_mean =
        (zero_crossings as f32 * sample_rate as f32) / (2.0 * zcr_frame as f32).max(1.0);

    // Estimate std from short-time ZCR variance across 8 non-overlapping sub-frames
    let sub_frame_size = zcr_frame / 8;
    let mut sub_pitches: Vec<f32> = Vec::with_capacity(8);
    if sub_frame_size > 1 {
        for sf in 0..8 {
            let s = sf * sub_frame_size;
            let e = (s + sub_frame_size).min(n);
            if e <= s + 1 {
                break;
            }
            let zc = audio[s..e]
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count();
            let p = (zc as f32 * sample_rate as f32) / (2.0 * (e - s) as f32);
            sub_pitches.push(p);
        }
    }
    let pitch_std = if sub_pitches.len() > 1 {
        let mean = sub_pitches.iter().sum::<f32>() / sub_pitches.len() as f32;
        let var =
            sub_pitches.iter().map(|&p| (p - mean).powi(2)).sum::<f32>() / sub_pitches.len() as f32;
        var.sqrt()
    } else {
        0.0
    };

    // ── Step 6: Speaking-rate proxy — energy envelope fluctuation rate ──
    // Count energy envelope peaks (local maxima in 10-ms windows)
    let env_window = (sample_rate as usize / 100).max(1); // ~10 ms
    let envelope: Vec<f32> = audio
        .chunks(env_window)
        .map(|chunk| chunk.iter().map(|s| s * s).sum::<f32>().sqrt())
        .collect();
    let env_peaks = envelope
        .windows(3)
        .filter(|w| w[1] > w[0] && w[1] > w[2])
        .count();
    let duration_secs = n as f32 / sample_rate as f32;
    let speaking_rate = if duration_secs > 0.0 {
        env_peaks as f32 / duration_secs * 60.0 // peaks-per-minute
    } else {
        0.0
    };

    Ok(VoiceQualityMetrics {
        pitch_mean,
        pitch_std,
        formant_frequencies,
        spectral_centroid,
        spectral_rolloff,
        speaking_rate,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Speaker encoder (mel spectrogram → embedding)
// ─────────────────────────────────────────────────────────────────────────────

/// Speaker encoder network
pub(crate) struct SpeakerEncoder {
    layers: Vec<LinearLayer>,
    device: Device,
    mel_params: MelParams,
}

impl SpeakerEncoder {
    pub(crate) fn new(embedding_dim: usize, device: Device) -> Result<Self> {
        let layers = vec![
            LinearLayer::new(80, 512, device.clone())?, // Mel-spec input (80 mel bands)
            LinearLayer::new(512, 512, device.clone())?,
            LinearLayer::new(512, embedding_dim, device.clone())?,
        ];

        // Default assumed sample rate; updated at call time if needed
        let mel_params = MelParams::vits_default(22050);

        Ok(Self {
            layers,
            device,
            mel_params,
        })
    }

    /// Encode a raw audio tensor into an L2-normalised speaker embedding.
    ///
    /// The audio tensor must be 1-D (samples) or 2-D (1 × samples) containing
    /// f32 values.  Internally the audio is:
    ///   1. Flattened to a sample slice.
    ///   2. Converted to a real mel spectrogram [n_frames × 80].
    ///   3. Fed through three linear layers with ReLU activations.
    ///   4. Global-average-pooled across the frame dimension.
    ///   5. L2-normalised per-vector.
    pub(crate) fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        // ── 1. Extract f32 samples from the tensor ──
        let audio_f32: Vec<f32> = match audio.dims() {
            [_n] => audio
                .to_vec1::<f32>()
                .map_err(|e| AcousticError::ProcessingError {
                    message: format!("Cannot read 1-D audio tensor: {e}"),
                })?,
            [1, _n] => audio
                .squeeze(0)
                .map_err(|e| AcousticError::ProcessingError {
                    message: format!("Squeeze failed: {e}"),
                })?
                .to_vec1::<f32>()
                .map_err(|e| AcousticError::ProcessingError {
                    message: format!("Cannot read 2-D audio tensor: {e}"),
                })?,
            dims => {
                // Fallback: flatten whatever shape we received
                let flat = audio
                    .flatten_all()
                    .map_err(|e| AcousticError::ProcessingError {
                        message: format!("Cannot flatten audio tensor with dims {dims:?}: {e}"),
                    })?;
                flat.to_vec1::<f32>()
                    .map_err(|e| AcousticError::ProcessingError {
                        message: format!("Cannot convert flattened tensor to vec: {e}"),
                    })?
            }
        };

        // ── 2. Real mel spectrogram [n_frames × 80] ──
        let mel_frames = compute_mel_spectrogram(&audio_f32, &self.mel_params)?;
        let n_frames = mel_frames.len();
        let n_mels = self.mel_params.n_mels;

        // Flatten to [n_frames × n_mels] row-major
        let flat_mel: Vec<f32> = mel_frames.into_iter().flatten().collect();

        // Build a Tensor of shape [n_frames, n_mels]
        let mut x = Tensor::from_vec(flat_mel, (n_frames, n_mels), &self.device).map_err(|e| {
            AcousticError::ProcessingError {
                message: format!("Failed to create mel tensor: {e}"),
            }
        })?;

        // ── 3. Apply linear layers with ReLU activations ──
        // Use broadcast_add for bias to handle [n_frames, out_dim] + [out_dim]
        for (i, layer) in self.layers.iter().enumerate() {
            x = linear_forward_broadcast(&layer.weight, &layer.bias, &x)?;
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }
        // x: [n_frames, embedding_dim]

        // ── 4. Global average pooling across the frame dimension → [1, embedding_dim] ──
        x = x.mean(0)?.unsqueeze(0)?;
        // x: [1, embedding_dim]

        // ── 5. L2 normalisation per vector (epsilon-stabilised) ──
        x = l2_normalize_embedding(&x)?;

        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptation network
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptation network for fine-tuning
pub(crate) struct AdaptationNetwork {
    #[allow(dead_code)]
    layers: Vec<LinearLayer>,
    #[allow(dead_code)]
    device: Device,
}

impl AdaptationNetwork {
    pub(crate) fn new(embedding_dim: usize, device: Device) -> Result<Self> {
        let layers = vec![
            LinearLayer::new(embedding_dim, 256, device.clone())?,
            LinearLayer::new(256, 256, device.clone())?,
            LinearLayer::new(256, embedding_dim, device.clone())?,
        ];

        Ok(Self { layers, device })
    }

    #[allow(dead_code)]
    fn adapt(&self, speaker_embedding: &Tensor) -> Result<Tensor> {
        let mut x = speaker_embedding.clone();

        // Apply adaptation layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }

        // Residual connection
        x = (x + speaker_embedding)?;

        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VoiceCloning — main interface
// ─────────────────────────────────────────────────────────────────────────────

/// Voice cloning module for VITS
pub struct VoiceCloning {
    config: VoiceCloningConfig,
    /// Speaker encoder for voice characteristics
    speaker_encoder: Arc<SpeakerEncoder>,
    /// Voice adaptation network
    #[allow(dead_code)]
    adaptation_network: Arc<AdaptationNetwork>,
    /// Speaker embedding cache
    speaker_cache: Arc<Mutex<std::collections::HashMap<String, SpeakerEmbedding>>>,
    /// Device for computation
    device: Device,
}

impl VoiceCloning {
    /// Create new voice cloning module
    pub fn new(config: VoiceCloningConfig, device: Device) -> Result<Self> {
        let speaker_encoder = Arc::new(SpeakerEncoder::new(
            config.speaker_embedding_dim,
            device.clone(),
        )?);

        let adaptation_network = Arc::new(AdaptationNetwork::new(
            config.speaker_embedding_dim,
            device.clone(),
        )?);

        Ok(Self {
            config,
            speaker_encoder,
            adaptation_network,
            speaker_cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
            device,
        })
    }

    /// Create voice clone from audio samples
    pub fn create_voice_clone(
        &self,
        speaker_id: String,
        audio_samples: &[Tensor],
        transcripts: Option<&[String]>,
    ) -> Result<SpeakerEmbedding> {
        // Validate minimum samples
        if audio_samples.len() < self.config.adaptation_samples {
            return Err(AcousticError::ProcessingError {
                message: format!(
                    "Need at least {} samples for voice cloning",
                    self.config.adaptation_samples
                ),
            });
        }

        // Extract speaker embeddings from all samples
        let mut embeddings = Vec::new();
        let mut quality_metrics = Vec::new();

        for sample in audio_samples.iter() {
            let embedding = self.speaker_encoder.encode(sample)?;
            let quality = self.analyze_voice_quality(sample)?;

            embeddings.push(embedding);
            quality_metrics.push(quality);
        }

        // Average embeddings and quality metrics
        let averaged_embedding = self.average_embeddings(&embeddings)?;
        let averaged_quality = self.average_quality_metrics(&quality_metrics)?;

        // Fine-tune adaptation network if transcripts are provided
        let adapted_embedding = if let Some(transcripts) = transcripts {
            self.fine_tune_with_transcripts(&averaged_embedding, audio_samples, transcripts)?
        } else {
            averaged_embedding
        };

        let speaker_embedding = SpeakerEmbedding {
            embedding: adapted_embedding,
            quality_metrics: averaged_quality,
            sample_count: audio_samples.len(),
            created_at: std::time::SystemTime::now(),
        };

        // Cache the speaker embedding
        self.cache_speaker_embedding(speaker_id, speaker_embedding.clone())?;

        Ok(speaker_embedding)
    }

    /// Synthesize with cloned voice
    pub fn synthesize_with_cloned_voice(
        &self,
        text: &str,
        speaker_embedding: &SpeakerEmbedding,
    ) -> Result<Tensor> {
        // Convert text to phonemes
        let phonemes = self.text_to_phonemes(text)?;

        // Apply speaker conditioning
        let conditioned_phonemes =
            self.apply_speaker_conditioning(&phonemes, &speaker_embedding.embedding)?;

        // Generate audio with voice characteristics
        let audio = self.generate_audio_with_voice(&conditioned_phonemes, speaker_embedding)?;

        Ok(audio)
    }

    /// Get cached speaker embedding
    pub fn get_speaker_embedding(&self, speaker_id: &str) -> Result<Option<SpeakerEmbedding>> {
        let cache = self
            .speaker_cache
            .lock()
            .map_err(|_| AcousticError::ProcessingError {
                message: "Failed to lock speaker cache".to_string(),
            })?;

        Ok(cache.get(speaker_id).cloned())
    }

    /// Update existing voice clone with new samples
    pub fn update_voice_clone(
        &self,
        speaker_id: &str,
        new_samples: &[Tensor],
        transcripts: Option<&[String]>,
    ) -> Result<SpeakerEmbedding> {
        // Get existing embedding
        let existing_embedding = self.get_speaker_embedding(speaker_id)?.ok_or_else(|| {
            AcousticError::ProcessingError {
                message: format!("Speaker '{speaker_id}' not found"),
            }
        })?;

        // Create new embedding from samples
        let new_embedding =
            self.create_voice_clone(format!("{speaker_id}_temp"), new_samples, transcripts)?;

        // Combine existing and new embeddings
        let combined_embedding = self.combine_embeddings(&existing_embedding, &new_embedding)?;

        // Update cache
        self.cache_speaker_embedding(speaker_id.to_string(), combined_embedding.clone())?;

        Ok(combined_embedding)
    }

    /// Analyse voice quality from an audio tensor using real signal processing.
    fn analyze_voice_quality(&self, audio: &Tensor) -> Result<VoiceQualityMetrics> {
        // Extract f32 samples from the tensor (best-effort flatten)
        let audio_f32 = match audio.dims() {
            [_n] => audio.to_vec1::<f32>().unwrap_or_default(),
            [1, _n] => audio
                .squeeze(0)
                .ok()
                .and_then(|t| t.to_vec1::<f32>().ok())
                .unwrap_or_default(),
            _ => audio
                .flatten_all()
                .ok()
                .and_then(|t| t.to_vec1::<f32>().ok())
                .unwrap_or_default(),
        };

        analyse_voice_quality_from_samples(&audio_f32, self.config.assumed_sample_rate)
    }

    /// Average multiple embeddings
    fn average_embeddings(&self, embeddings: &[Tensor]) -> Result<Tensor> {
        if embeddings.is_empty() {
            return Err(AcousticError::ProcessingError {
                message: "No embeddings to average".to_string(),
            });
        }

        let mut sum = embeddings[0].clone();
        for embedding in embeddings.iter().skip(1) {
            sum = (sum + embedding)?;
        }

        let count_tensor = Tensor::new(&[embeddings.len() as f32], sum.device())?;
        let averaged = (sum / count_tensor)?;

        // Re-normalise after averaging
        l2_normalize_embedding(&averaged)
    }

    /// Average quality metrics
    fn average_quality_metrics(
        &self,
        metrics: &[VoiceQualityMetrics],
    ) -> Result<VoiceQualityMetrics> {
        if metrics.is_empty() {
            return Err(AcousticError::ProcessingError {
                message: "No quality metrics to average".to_string(),
            });
        }

        let count = metrics.len() as f32;
        Ok(VoiceQualityMetrics {
            pitch_mean: metrics.iter().map(|m| m.pitch_mean).sum::<f32>() / count,
            pitch_std: metrics.iter().map(|m| m.pitch_std).sum::<f32>() / count,
            formant_frequencies: metrics[0].formant_frequencies.clone(),
            spectral_centroid: metrics.iter().map(|m| m.spectral_centroid).sum::<f32>() / count,
            spectral_rolloff: metrics.iter().map(|m| m.spectral_rolloff).sum::<f32>() / count,
            speaking_rate: metrics.iter().map(|m| m.speaking_rate).sum::<f32>() / count,
        })
    }

    /// Fine-tune with transcripts (stub — requires supervised training pipeline)
    fn fine_tune_with_transcripts(
        &self,
        embedding: &Tensor,
        _audio_samples: &[Tensor],
        _transcripts: &[String],
    ) -> Result<Tensor> {
        // Real fine-tuning would train the adaptation network on audio-transcript
        // pairs using gradient descent. The stub returns the averaged embedding as-is
        // so that the rest of the pipeline remains functional.
        Ok(embedding.clone())
    }

    /// Convert text to phonemes
    fn text_to_phonemes(&self, text: &str) -> Result<Tensor> {
        // Simplified phoneme conversion (downstream G2P module handles the real path)
        let phoneme_embedding = Tensor::randn(0f32, 1f32, &[text.len(), 256], &self.device)?;
        Ok(phoneme_embedding)
    }

    /// Apply speaker conditioning
    fn apply_speaker_conditioning(
        &self,
        phonemes: &Tensor,
        speaker_embedding: &Tensor,
    ) -> Result<Tensor> {
        // Broadcast speaker embedding to match phoneme sequence length
        let speaker_broadcast = speaker_embedding.broadcast_as(phonemes.shape())?;

        // Combine phonemes with speaker characteristics
        let scale_tensor = Tensor::new(&[0.3f32], speaker_broadcast.device())?;
        let speaker_scaled = (speaker_broadcast * scale_tensor)?;
        let conditioned = (phonemes + speaker_scaled)?;

        Ok(conditioned)
    }

    /// Generate audio with voice characteristics
    fn generate_audio_with_voice(
        &self,
        phonemes: &Tensor,
        speaker_embedding: &SpeakerEmbedding,
    ) -> Result<Tensor> {
        // Apply voice quality characteristics
        let quality_factor = (speaker_embedding.quality_metrics.pitch_mean / 150.0).min(2.0);
        let quality_tensor = Tensor::new(&[quality_factor], phonemes.device())?;
        let audio = (phonemes * quality_tensor)?;

        Ok(audio)
    }

    /// Cache speaker embedding
    fn cache_speaker_embedding(
        &self,
        speaker_id: String,
        embedding: SpeakerEmbedding,
    ) -> Result<()> {
        let mut cache = self
            .speaker_cache
            .lock()
            .map_err(|_| AcousticError::ProcessingError {
                message: "Failed to lock speaker cache".to_string(),
            })?;

        cache.insert(speaker_id, embedding);
        Ok(())
    }

    /// Combine two speaker embeddings weighted by sample count
    fn combine_embeddings(
        &self,
        existing: &SpeakerEmbedding,
        new: &SpeakerEmbedding,
    ) -> Result<SpeakerEmbedding> {
        let total_samples = existing.sample_count + new.sample_count;
        let existing_weight = existing.sample_count as f32 / total_samples as f32;
        let new_weight = new.sample_count as f32 / total_samples as f32;

        let existing_weight_tensor = Tensor::new(&[existing_weight], existing.embedding.device())?;
        let new_weight_tensor = Tensor::new(&[new_weight], new.embedding.device())?;
        let existing_weighted = (existing.embedding.clone() * existing_weight_tensor)?;
        let new_weighted = (new.embedding.clone() * new_weight_tensor)?;
        let combined_embedding = (existing_weighted + new_weighted)?;

        // Re-normalise after weighted combination
        let combined_normalised = l2_normalize_embedding(&combined_embedding)?;

        Ok(SpeakerEmbedding {
            embedding: combined_normalised,
            quality_metrics: existing.quality_metrics.clone(),
            sample_count: total_samples,
            created_at: std::time::SystemTime::now(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use std::f32::consts::PI;

    // ── Helper: generate a pure sine wave ──
    fn sine_wave(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    // ────────────────────────────────────────────────────────────────
    // Test 1: L2-normalised vector has unit norm (within tolerance)
    // ────────────────────────────────────────────────────────────────
    #[test]
    fn test_l2_normalise_produces_unit_norm() {
        let device = Device::Cpu;

        // Non-unit vector: [3.0, 4.0] → expected norm 5.0 → normalised [0.6, 0.8]
        let v = Tensor::from_vec(vec![3.0_f32, 4.0], (1, 2), &device).unwrap();
        let v_norm = l2_normalize_embedding(&v).unwrap();

        let vals: Vec<f32> = v_norm.squeeze(0).unwrap().to_vec1().unwrap();
        let norm_sq: f32 = vals.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();

        assert!((norm - 1.0).abs() < 1e-5, "Expected unit norm, got {norm}");
    }

    // ────────────────────────────────────────────────────────────────
    // Test 2: L2 normalisation handles near-zero vector without NaN
    // ────────────────────────────────────────────────────────────────
    #[test]
    fn test_l2_normalise_near_zero_no_nan() {
        let device = Device::Cpu;
        let v = Tensor::from_vec(vec![0.0_f32, 0.0, 0.0, 0.0], (1, 4), &device).unwrap();
        let v_norm = l2_normalize_embedding(&v).unwrap();
        let vals: Vec<f32> = v_norm.squeeze(0).unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|x| x.is_finite()),
            "Expected finite values for near-zero input, got {vals:?}"
        );
    }

    // ────────────────────────────────────────────────────────────────
    // Test 3: L2 normalisation of a batch — every row has unit norm
    // ────────────────────────────────────────────────────────────────
    #[test]
    fn test_l2_normalise_batch_all_unit_norm() {
        let device = Device::Cpu;
        // 4 vectors in R^3
        let data = vec![
            1.0_f32, 0.0, 0.0, // already unit
            0.0, 2.0, 0.0, // scale 2
            1.0, 1.0, 1.0, // scale sqrt(3)
            3.0, 4.0, 0.0, // scale 5
        ];
        let mat = Tensor::from_vec(data, (4, 3), &device).unwrap();
        let normed = l2_normalize_embedding(&mat).unwrap();
        let rows: Vec<Vec<f32>> = (0..4)
            .map(|i| normed.get(i).unwrap().to_vec1().unwrap())
            .collect();

        for (idx, row) in rows.iter().enumerate() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Row {idx} has norm {norm} (expected 1.0)"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Test 4: Mel spectrogram output has correct shape [n_frames, n_mels]
    // ────────────────────────────────────────────────────────────────
    #[test]
    fn test_mel_spectrogram_shape() {
        let sample_rate = 22050_u32;
        let params = MelParams::vits_default(sample_rate);

        // 0.5 s sine wave at 440 Hz
        let audio = sine_wave(440.0, sample_rate, 0.5);
        let mel = compute_mel_spectrogram(&audio, &params).unwrap();

        // Every frame must have exactly n_mels columns
        assert_eq!(params.n_mels, 80);
        assert!(!mel.is_empty(), "Expected at least one frame");
        for (frame_idx, frame) in mel.iter().enumerate() {
            assert_eq!(
                frame.len(),
                params.n_mels,
                "Frame {frame_idx} has {} mel bins (expected {})",
                frame.len(),
                params.n_mels
            );
        }

        // Expected n_frames ≈ (n_samples - n_fft) / hop_length + 1
        let expected_frames = (audio.len().saturating_sub(params.n_fft)) / params.hop_length + 1;
        assert_eq!(
            mel.len(),
            expected_frames,
            "Expected {expected_frames} frames, got {}",
            mel.len()
        );
    }

    // ────────────────────────────────────────────────────────────────
    // Test 5: Mel filterbank sums to reasonable values (not zero, no NaN)
    // ────────────────────────────────────────────────────────────────
    #[test]
    fn test_mel_filterbank_reasonable_values() {
        let params = MelParams::vits_default(22050);
        let n_freqs = params.n_fft / 2 + 1;
        let filterbank = build_mel_filterbank(&params, n_freqs);

        assert_eq!(
            filterbank.len(),
            params.n_mels,
            "Wrong number of mel filters"
        );

        let mut any_positive = false;
        for (m, filter) in filterbank.iter().enumerate() {
            assert_eq!(filter.len(), n_freqs, "Filter {m} has wrong length");
            let filter_sum: f32 = filter.iter().sum();
            assert!(
                filter.iter().all(|w| w.is_finite()),
                "Filter {m} contains non-finite weights"
            );
            assert!(
                filter_sum >= 0.0,
                "Filter {m} has negative sum {filter_sum}"
            );
            if filter_sum > 1e-6 {
                any_positive = true;
            }
        }
        assert!(
            any_positive,
            "All mel filters have near-zero sums — filterbank is degenerate"
        );
    }

    // ────────────────────────────────────────────────────────────────
    // Test 6: Mel spectrogram values are finite (no NaN / Inf)
    // ────────────────────────────────────────────────────────────────
    #[test]
    fn test_mel_spectrogram_finite_values() {
        let sample_rate = 22050_u32;
        let params = MelParams::vits_default(sample_rate);
        let audio = sine_wave(880.0, sample_rate, 0.3);
        let mel = compute_mel_spectrogram(&audio, &params).unwrap();

        for (frame_idx, frame) in mel.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Non-finite mel[{frame_idx}][{mel_idx}] = {val}"
                );
            }
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Test 7: Voice quality analysis returns finite, non-negative values
    // ────────────────────────────────────────────────────────────────
    #[test]
    fn test_voice_quality_analysis_finite() {
        let sample_rate = 22050_u32;
        let audio = sine_wave(220.0, sample_rate, 0.5);
        let metrics = analyse_voice_quality_from_samples(&audio, sample_rate).unwrap();

        assert!(
            metrics.spectral_centroid.is_finite() && metrics.spectral_centroid >= 0.0,
            "spectral_centroid invalid: {}",
            metrics.spectral_centroid
        );
        assert!(
            metrics.spectral_rolloff.is_finite() && metrics.spectral_rolloff >= 0.0,
            "spectral_rolloff invalid: {}",
            metrics.spectral_rolloff
        );
        assert!(
            metrics.pitch_mean.is_finite() && metrics.pitch_mean >= 0.0,
            "pitch_mean invalid: {}",
            metrics.pitch_mean
        );
        assert!(
            metrics.pitch_std.is_finite() && metrics.pitch_std >= 0.0,
            "pitch_std invalid: {}",
            metrics.pitch_std
        );
    }

    // ────────────────────────────────────────────────────────────────
    // Test 8: SpeakerEncoder produces correct embedding shape and unit norm
    // ────────────────────────────────────────────────────────────────
    #[test]
    fn test_speaker_encoder_embedding_shape_and_norm() {
        let device = Device::Cpu;
        let embedding_dim = 512_usize;
        let encoder = SpeakerEncoder::new(embedding_dim, device.clone()).unwrap();

        // 0.5 s sine wave at 440 Hz
        let sample_rate = 22050_u32;
        let audio_data = sine_wave(440.0, sample_rate, 0.5);
        let n_samples = audio_data.len();
        let audio_t = Tensor::from_vec(audio_data, n_samples, &device).unwrap();

        let embedding = encoder.encode(&audio_t).unwrap();

        // Shape: [1, embedding_dim]
        assert_eq!(
            embedding.dims(),
            &[1, embedding_dim],
            "Expected embedding shape [1, {embedding_dim}], got {:?}",
            embedding.dims()
        );

        // L2 norm of the embedding row ≈ 1.0
        let vals: Vec<f32> = embedding.squeeze(0).unwrap().to_vec1().unwrap();
        let norm: f32 = vals.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Embedding norm should be ~1.0, got {norm}"
        );
    }
}
