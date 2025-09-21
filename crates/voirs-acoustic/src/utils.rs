//! Utility functions for acoustic modeling.

use crate::{MelSpectrogram, Phoneme, SynthesisConfig};
use std::collections::HashMap;

/// Mel spectrogram processing utilities
pub fn normalize_mel_spectrogram(mel: &mut MelSpectrogram) {
    // Compute global statistics across all mel channels and frames
    let mut all_values: Vec<f32> = Vec::new();
    for mel_channel in &mel.data {
        all_values.extend_from_slice(mel_channel);
    }

    if all_values.is_empty() {
        return;
    }

    // Compute mean and standard deviation
    let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
    let variance =
        all_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / all_values.len() as f32;
    let std_dev = variance.sqrt().max(1e-8); // Avoid division by zero

    // Apply Z-score normalization to all channels
    for mel_channel in &mut mel.data {
        for value in mel_channel {
            *value = (*value - mean) / std_dev;
        }
    }

    // Apply dynamic range compression (tanh normalization)
    for mel_channel in &mut mel.data {
        for value in mel_channel {
            *value = value.tanh(); // Compress to [-1, 1] range
        }
    }

    // Spectral smoothing using simple moving average
    if mel.n_frames > 2 {
        for mel_channel in &mut mel.data {
            let original = mel_channel.clone();
            for i in 1..mel_channel.len() - 1 {
                mel_channel[i] = (original[i - 1] + original[i] + original[i + 1]) / 3.0;
            }
        }
    }
}

/// Phoneme sequence processing
pub fn process_phoneme_sequence(phonemes: &[Phoneme], config: &SynthesisConfig) -> Vec<Phoneme> {
    let mut processed_phonemes = phonemes.to_vec();

    // Duration adjustment based on speaking rate
    for phoneme in &mut processed_phonemes {
        if let Some(duration) = phoneme.duration {
            phoneme.duration = Some(duration / config.speed);
        }
    }

    // Stress pattern modification
    for (i, phoneme) in processed_phonemes.iter_mut().enumerate() {
        // Add stress information if not present
        let features = phoneme.features.get_or_insert_with(HashMap::new);
        if !features.contains_key("stress") {
            // Simple heuristic: stress every 3rd phoneme in content words
            let stress_level = if is_content_phoneme(&phoneme.symbol) && i % 3 == 0 {
                "primary"
            } else {
                "unstressed"
            };
            features.insert("stress".to_string(), stress_level.to_string());
        }

        // Adjust duration based on stress
        if let Some(duration) = phoneme.duration {
            let stress_factor = if let Some(features) = &phoneme.features {
                match features.get("stress").map(|s| s.as_str()) {
                    Some("primary") => 1.2,
                    Some("secondary") => 1.1,
                    Some("unstressed") => 0.9,
                    _ => 1.0,
                }
            } else {
                1.0
            };
            phoneme.duration = Some(duration * stress_factor);
        }
    }

    // Apply energy adjustments based on synthesis config
    if config.energy != 1.0 {
        for phoneme in &mut processed_phonemes {
            let features = phoneme
                .features
                .get_or_insert_with(std::collections::HashMap::new);
            features.insert("energy_scale".to_string(), config.energy.to_string());
        }
    }

    // Apply pitch shift adjustments
    if config.pitch_shift != 0.0 {
        for phoneme in &mut processed_phonemes {
            let features = phoneme
                .features
                .get_or_insert_with(std::collections::HashMap::new);
            features.insert("pitch_shift".to_string(), config.pitch_shift.to_string());
        }
    }

    processed_phonemes
}

/// Duration prediction utilities
pub fn predict_phoneme_durations(phonemes: &[Phoneme]) -> Vec<f32> {
    let mut durations = Vec::with_capacity(phonemes.len());

    for (i, phoneme) in phonemes.iter().enumerate() {
        // Start with base duration based on phoneme type
        let base_duration = get_base_phoneme_duration(&phoneme.symbol);

        // Context-aware adjustments
        let context_factor = calculate_context_factor(phonemes, i);

        // Stress-based adjustments
        let stress_factor = if let Some(features) = &phoneme.features {
            match features.get("stress").map(|s| s.as_str()) {
                Some("primary") => 1.3,
                Some("secondary") => 1.15,
                Some("unstressed") => 0.85,
                _ => 1.0,
            }
        } else {
            1.0
        };

        // Position-based adjustments (phrase-initial and phrase-final lengthening)
        let position_factor = if i == 0 || i == phonemes.len() - 1 {
            1.2 // Lengthen initial and final phonemes
        } else {
            1.0
        };

        // Calculate final duration
        let final_duration = base_duration * context_factor * stress_factor * position_factor;
        durations.push(final_duration.clamp(20.0, 500.0)); // Clamp between 20ms and 500ms
    }

    durations
}

/// Prosody control utilities
pub fn apply_prosody_control(mel: &mut MelSpectrogram, pitch_shift: f32, speaking_rate: f32) {
    // Duration modification through resampling
    if speaking_rate != 1.0 && speaking_rate > 0.0 {
        let new_frame_count = (mel.n_frames as f32 / speaking_rate) as usize;

        for mel_channel in &mut mel.data {
            let original = mel_channel.clone();
            mel_channel.resize(new_frame_count, 0.0);

            // Linear interpolation for resampling
            #[allow(clippy::needless_range_loop)]
            for i in 0..new_frame_count {
                let src_idx = (i as f32 * speaking_rate) as usize;
                let frac = (i as f32 * speaking_rate) % 1.0;

                if src_idx < original.len() {
                    if src_idx + 1 < original.len() {
                        mel_channel[i] =
                            original[src_idx] * (1.0 - frac) + original[src_idx + 1] * frac;
                    } else {
                        mel_channel[i] = original[src_idx];
                    }
                }
            }
        }

        mel.n_frames = new_frame_count;
    }

    // Pitch shifting through spectral modification
    if pitch_shift != 0.0 {
        let _shift_factor = 2.0f32.powf(pitch_shift / 12.0); // Convert semitones to frequency ratio

        // Apply pitch shift by modifying lower mel channels more strongly
        for (mel_idx, mel_channel) in mel.data.iter_mut().enumerate() {
            let mel_factor = (-(mel_idx as f32) / mel.n_mels as f32).exp(); // Exponential decay with frequency
            let actual_shift = pitch_shift * mel_factor;

            for value in mel_channel {
                *value += actual_shift * 0.01; // Scale down the effect
            }
        }
    }

    // Energy adjustment (simple gain)
    let energy_factor = 1.0 + pitch_shift.abs() * 0.1; // Slight energy boost with pitch changes
    for mel_channel in &mut mel.data {
        for value in mel_channel {
            *value *= energy_factor;
        }
    }
}

/// Speaker embedding utilities
pub fn get_speaker_embedding(speaker_id: Option<u32>) -> Option<Vec<f32>> {
    speaker_id.map(|id| {
        // Generate deterministic embedding based on speaker ID
        let mut embedding = vec![0.0; 256];

        // Use speaker ID as seed for deterministic generation
        let mut state = id as u64;
        #[allow(clippy::needless_range_loop)]
        for i in 0..256 {
            // Simple linear congruential generator
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0; // Range [-1, 1]

            // Create speaker-specific characteristics
            embedding[i] = match i {
                // Fundamental frequency characteristics (0-31)
                0..=31 => normalized * 0.5 + get_speaker_f0_bias(id),
                // Formant characteristics (32-95)
                32..=95 => normalized * 0.3 + get_speaker_formant_bias(id, i - 32),
                // Voice quality characteristics (96-159)
                96..=159 => normalized * 0.4 + get_speaker_quality_bias(id, i - 96),
                // Prosody characteristics (160-223)
                160..=223 => normalized * 0.6 + get_speaker_prosody_bias(id, i - 160),
                // General characteristics (224-255)
                _ => normalized * 0.2,
            };
        }

        // Normalize the embedding vector
        let norm = embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        for value in &mut embedding {
            *value /= norm;
        }

        embedding
    })
}

/// Helper function to determine if a phoneme represents a content word
fn is_content_phoneme(symbol: &str) -> bool {
    // Simple heuristic: vowels and most consonants are content phonemes
    // Exclude silence and special symbols
    !matches!(symbol, "_" | " " | "<pad>" | "<unk>" | "<bos>" | "<eos>")
}

/// Get base duration for a phoneme based on its type
fn get_base_phoneme_duration(symbol: &str) -> f32 {
    match symbol {
        // Vowels - generally longer
        "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY" | "OW"
        | "OY" | "UH" | "UW" => 120.0, // 120ms

        // Fricatives - medium duration
        "F" | "TH" | "S" | "SH" | "HH" | "V" | "DH" | "Z" | "ZH" => 100.0, // 100ms

        // Stops - shorter
        "P" | "B" | "T" | "D" | "K" | "G" => 70.0, // 70ms

        // Nasals - medium
        "M" | "N" | "NG" => 90.0, // 90ms

        // Liquids - medium
        "L" | "R" => 85.0, // 85ms

        // Glides - shorter
        "W" | "Y" => 75.0, // 75ms

        // Affricates - medium
        "CH" | "JH" => 95.0, // 95ms

        // Silence and special tokens
        "_" | " " => 50.0,                             // 50ms
        "<pad>" | "<unk>" | "<bos>" | "<eos>" => 20.0, // 20ms

        // Default for unknown phonemes
        _ => 80.0, // 80ms
    }
}

/// Calculate context factor for duration prediction
fn calculate_context_factor(phonemes: &[Phoneme], index: usize) -> f32 {
    let mut factor = 1.0;

    // Check preceding phoneme
    if index > 0 {
        let prev_symbol = &phonemes[index - 1].symbol;
        if is_vowel(prev_symbol) {
            factor *= 0.95; // Slightly shorter after vowels
        }
    }

    // Check following phoneme
    if index + 1 < phonemes.len() {
        let next_symbol = &phonemes[index + 1].symbol;
        if is_consonant_cluster_start(next_symbol) {
            factor *= 1.1; // Longer before consonant clusters
        }
    }

    factor
}

/// Check if a phoneme is a vowel
fn is_vowel(symbol: &str) -> bool {
    matches!(
        symbol,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

/// Check if a phoneme starts a consonant cluster
fn is_consonant_cluster_start(symbol: &str) -> bool {
    matches!(
        symbol,
        "S" | "SH" | "TH" | "CH" | "P" | "T" | "K" | "B" | "D" | "G"
    )
}

/// Get speaker-specific F0 bias
fn get_speaker_f0_bias(speaker_id: u32) -> f32 {
    // Map speaker ID to F0 bias (-0.3 to +0.3)
    ((speaker_id % 100) as f32 / 100.0) * 0.6 - 0.3
}

/// Get speaker-specific formant bias
fn get_speaker_formant_bias(speaker_id: u32, formant_idx: usize) -> f32 {
    // Different speakers have different formant characteristics
    let speaker_factor = (speaker_id % 50) as f32 / 50.0;
    let formant_factor = (formant_idx % 16) as f32 / 16.0;
    (speaker_factor + formant_factor) * 0.2 - 0.1
}

/// Get speaker-specific voice quality bias
fn get_speaker_quality_bias(speaker_id: u32, quality_idx: usize) -> f32 {
    // Voice quality characteristics (breathiness, roughness, etc.)
    let quality_seed = speaker_id.wrapping_mul(37).wrapping_add(quality_idx as u32);
    let normalized = (quality_seed % 1000) as f32 / 1000.0;
    normalized * 0.4 - 0.2
}

/// Get speaker-specific prosody bias
fn get_speaker_prosody_bias(speaker_id: u32, prosody_idx: usize) -> f32 {
    // Prosodic characteristics (rhythm, stress patterns, etc.)
    let prosody_seed = speaker_id.wrapping_mul(73).wrapping_add(prosody_idx as u32);
    let normalized = (prosody_seed % 1000) as f32 / 1000.0;
    normalized * 0.6 - 0.3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Phoneme;

    #[test]
    fn test_predict_phoneme_durations() {
        let phonemes = vec![
            Phoneme::new("h"),
            Phoneme::new("ɛ"),
            Phoneme::new("l"),
            Phoneme::new("oʊ"),
        ];

        let durations = predict_phoneme_durations(&phonemes);
        assert_eq!(durations.len(), 4);
        assert!(durations.iter().all(|&d| d > 0.0));
    }

    #[test]
    fn test_speaker_embedding() {
        let embedding = get_speaker_embedding(Some(42));
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap().len(), 256);

        let no_embedding = get_speaker_embedding(None);
        assert!(no_embedding.is_none());
    }
}
