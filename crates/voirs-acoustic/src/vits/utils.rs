//! Utility functions and helpers for VITS model
//!
//! This module contains shared utility functions including:
//! - Linear layer implementation
//! - Phoneme characteristic lookups
//! - Tensor conversion utilities
//! - Mel-frequency conversion helpers

use candle_core::{Device, Tensor};

use crate::{AcousticError, MelSpectrogram, Result};

/// Linear layer implementation
pub(crate) struct LinearLayer {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
}

impl LinearLayer {
    pub(crate) fn new(in_features: usize, out_features: usize, device: Device) -> Result<Self> {
        let weight = Tensor::randn(0f32, 1f32, &[out_features, in_features], &device)?;
        let bias = Tensor::randn(0f32, 1f32, &[out_features], &device)?;

        Ok(Self { weight, bias })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = (input.matmul(&self.weight.transpose(0, 1)?)? + &self.bias)?;
        Ok(output)
    }
}

/// Get phoneme-specific acoustic characteristics
/// Returns (fundamental_frequency, formant_frequencies, energy_level)
#[allow(dead_code)]
pub(crate) fn get_phoneme_characteristics(phoneme: &str) -> (f32, Vec<f32>, f32) {
    match phoneme {
        // Vowels - have clear formant structure
        "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY" | "OW"
        | "OY" | "UH" | "UW" => {
            let (f1, f2, f3) = match phoneme {
                "AA" => (730.0, 1090.0, 2440.0), // father
                "AE" => (660.0, 1720.0, 2410.0), // cat
                "AH" => (520.0, 1190.0, 2390.0), // but
                "AO" => (570.0, 840.0, 2410.0),  // thought
                "EH" => (530.0, 1840.0, 2480.0), // bed
                "ER" => (490.0, 1350.0, 1690.0), // bird
                "EY" => (400.0, 2000.0, 2550.0), // bait
                "IH" => (390.0, 1990.0, 2550.0), // bit
                "IY" => (270.0, 2290.0, 3010.0), // beat
                "OW" => (490.0, 910.0, 2200.0),  // boat
                "UH" => (440.0, 1020.0, 2240.0), // book
                "UW" => (300.0, 870.0, 2240.0),  // boot
                _ => (500.0, 1500.0, 2500.0),    // default vowel
            };
            (150.0, vec![f1, f2, f3], 0.8) // High energy for vowels
        }

        // Fricatives - high frequency energy
        "F" | "TH" | "S" | "SH" | "HH" | "V" | "DH" | "Z" | "ZH" => {
            let center_freq = match phoneme {
                "S" => 7000.0,
                "SH" => 4000.0,
                "F" | "TH" => 6000.0,
                "HH" => 3000.0,
                _ => 5000.0,
            };
            (0.0, vec![center_freq], 0.6) // No fundamental, moderate energy
        }

        // Stops - brief bursts
        "P" | "B" | "T" | "D" | "K" | "G" => {
            let burst_freq = match phoneme {
                "P" | "B" => 1500.0,
                "T" | "D" => 4000.0,
                "K" | "G" => 2500.0,
                _ => 2000.0,
            };
            (0.0, vec![burst_freq], 0.5) // No fundamental, brief energy
        }

        // Nasals - low frequency resonance
        "M" | "N" | "NG" => {
            let nasal_freq = match phoneme {
                "M" => 1000.0,
                "N" => 1500.0,
                "NG" => 1200.0,
                _ => 1200.0,
            };
            (120.0, vec![nasal_freq, 2500.0], 0.7) // Low fundamental, good energy
        }

        // Liquids - formant-like structure
        "L" | "R" => {
            let formants = match phoneme {
                "L" => vec![350.0, 1200.0, 2900.0],
                "R" => vec![350.0, 1200.0, 1700.0],
                _ => vec![400.0, 1200.0, 2800.0],
            };
            (140.0, formants, 0.75)
        }

        // Glides - vowel-like but transitional
        "W" | "Y" => {
            let formants = match phoneme {
                "W" => vec![300.0, 600.0, 2200.0],
                "Y" => vec![300.0, 2200.0, 3000.0],
                _ => vec![300.0, 1400.0, 2600.0],
            };
            (140.0, formants, 0.6)
        }

        // Affricates
        "CH" | "JH" => (0.0, vec![2500.0, 4000.0], 0.5),

        // Silence and special tokens
        " " | "<pad>" | "<unk>" | "<bos>" | "<eos>" => (0.0, vec![], 0.0),

        // Default for unknown phonemes
        _ => (130.0, vec![500.0, 1500.0, 2500.0], 0.5),
    }
}

/// Convert decoder output tensor to MelSpectrogram
pub(crate) fn tensor_to_mel_spectrogram(
    tensor: &candle_core::Tensor,
    sample_rate: u32,
    hop_length: u32,
) -> Result<MelSpectrogram> {
    // Get tensor dimensions [batch_size, n_mel_channels, n_frames]
    let dims = tensor.dims();
    if dims.len() != 3 {
        return Err(AcousticError::ModelError {
            message: format!("Expected 3D tensor [batch, mel_channels, frames], got {dims:?}"),
        });
    }

    let (_batch_size, n_mel_channels, _n_frames) =
        tensor.dims3().map_err(|e| AcousticError::ModelError {
            message: format!("Failed to get tensor dimensions: {e}"),
        })?;

    // For now, take the first batch
    let mel_tensor = tensor.get(0).map_err(|e| AcousticError::ModelError {
        message: format!("Failed to get first batch: {e}"),
    })?;

    // Convert to Vec<Vec<f32>> format
    let mut data = Vec::with_capacity(n_mel_channels);

    for mel_idx in 0..n_mel_channels {
        let mel_channel = mel_tensor
            .get(mel_idx)
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to get mel channel {mel_idx}: {e}"),
            })?;

        let channel_data = mel_channel
            .to_vec1::<f32>()
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to convert channel {mel_idx} to vec: {e}"),
            })?;

        data.push(channel_data);
    }

    let mel = MelSpectrogram::new(data, sample_rate, hop_length);

    tracing::debug!(
        "Converted tensor {:?} to MelSpectrogram: {}x{} frames",
        dims,
        mel.n_mels,
        mel.n_frames
    );

    Ok(mel)
}

/// Convert mel bin index to frequency in Hz
#[allow(dead_code)]
pub(crate) fn mel_idx_to_frequency(mel_idx: usize, total_mel_bins: usize) -> f32 {
    // Mel scale conversion: mel = 2595 * log10(1 + freq/700)
    // Inverse: freq = 700 * (10^(mel/2595) - 1)

    let mel_max = 2595.0 * (1.0_f32 + 8000.0 / 700.0).log10(); // Max mel for 8kHz
    let mel_value = (mel_idx as f32 / total_mel_bins as f32) * mel_max;

    700.0 * (10.0_f32.powf(mel_value / 2595.0) - 1.0)
}

/// Estimate duration for a phoneme based on its type
pub(crate) fn estimate_phoneme_duration(phoneme_symbol: &str) -> f32 {
    match phoneme_symbol {
        // Vowels - longer duration
        "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY" | "OW"
        | "OY" | "UH" | "UW" => 0.12, // 120ms

        // Consonants - shorter duration
        "B" | "CH" | "D" | "DH" | "F" | "G" | "HH" | "JH" | "K" | "L" | "M" | "N" | "NG" | "P"
        | "R" | "S" | "SH" | "T" | "TH" | "V" | "W" | "Y" | "Z" | "ZH" => 0.08, // 80ms

        // Silence and special tokens
        " " | "<pad>" | "<unk>" | "<bos>" | "<eos>" => 0.05, // 50ms

        // Default for unknown phonemes
        _ => 0.09, // 90ms
    }
}
