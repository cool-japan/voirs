//! Utility functions for singing synthesis

#![allow(
    clippy::needless_range_loop,
    clippy::uninlined_format_args,
    clippy::field_reassign_with_default
)]

/// Audio utility functions
pub mod audio {
    /// Convert samples to different sample rates using linear interpolation.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input audio samples to resample
    /// * `from_rate` - Original sample rate in Hz
    /// * `to_rate` - Target sample rate in Hz
    ///
    /// # Returns
    ///
    /// A new vector of resampled audio samples at the target sample rate
    pub fn resample(samples: &[f32], from_rate: f32, to_rate: f32) -> Vec<f32> {
        if from_rate == to_rate {
            return samples.to_vec();
        }

        let ratio = to_rate / from_rate;
        let new_len = (samples.len() as f32 * ratio) as usize;
        let mut output = vec![0.0; new_len];

        for i in 0..new_len {
            let source_index = i as f32 / ratio;
            let index = source_index as usize;

            if index < samples.len() - 1 {
                let fraction = source_index - index as f32;
                output[i] = samples[index] * (1.0 - fraction) + samples[index + 1] * fraction;
            } else if index < samples.len() {
                output[i] = samples[index];
            }
        }

        output
    }

    /// Apply fade in/out to audio samples to reduce clicks and pops.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples to modify in-place
    /// * `fade_in_samples` - Number of samples for the fade-in duration
    /// * `fade_out_samples` - Number of samples for the fade-out duration
    pub fn apply_fade(samples: &mut [f32], fade_in_samples: usize, fade_out_samples: usize) {
        // Fade in
        for i in 0..fade_in_samples.min(samples.len()) {
            let factor = i as f32 / fade_in_samples as f32;
            samples[i] *= factor;
        }

        // Fade out
        let start_fade_out = samples.len().saturating_sub(fade_out_samples);
        for i in start_fade_out..samples.len() {
            let factor = (samples.len() - i) as f32 / fade_out_samples as f32;
            samples[i] *= factor;
        }
    }

    /// Normalize audio to a target peak amplitude.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples to normalize in-place
    /// * `target_peak` - Target peak amplitude (typically 1.0 for full scale)
    pub fn normalize(samples: &mut [f32], target_peak: f32) {
        let max_sample = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);
        if max_sample > 0.0 {
            let scale = target_peak / max_sample;
            for sample in samples {
                *sample *= scale;
            }
        }
    }

    /// Calculate the Root Mean Square (RMS) amplitude of audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples to analyze
    ///
    /// # Returns
    ///
    /// RMS value representing the average signal power
    pub fn rms(samples: &[f32]) -> f32 {
        let sum: f32 = samples.iter().map(|x| x * x).sum();
        (sum / samples.len() as f32).sqrt()
    }
}

/// Music theory utility functions
pub mod music {

    /// Convert note name and octave to frequency using equal temperament tuning.
    ///
    /// # Arguments
    ///
    /// * `note` - Note name (e.g., "C", "C#", "Db", "D", etc.)
    /// * `octave` - Octave number (typically 0-8, where A4 = 440 Hz)
    ///
    /// # Returns
    ///
    /// Frequency in Hz corresponding to the given note and octave
    pub fn note_to_frequency(note: &str, octave: u8) -> f32 {
        let note_values = [
            ("C", 0),
            ("C#", 1),
            ("Db", 1),
            ("D", 2),
            ("D#", 3),
            ("Eb", 3),
            ("E", 4),
            ("F", 5),
            ("F#", 6),
            ("Gb", 6),
            ("G", 7),
            ("G#", 8),
            ("Ab", 8),
            ("A", 9),
            ("A#", 10),
            ("Bb", 10),
            ("B", 11),
        ];

        let semitone_offset = note_values
            .iter()
            .find(|(n, _)| n == &note)
            .map(|(_, offset)| *offset)
            .unwrap_or(9); // Default to A

        // A4 = 440 Hz is at octave 4, semitone 9
        let semitones_from_a4 = (octave as i32 - 4) * 12 + semitone_offset - 9;
        440.0 * 2.0_f32.powf(semitones_from_a4 as f32 / 12.0)
    }

    /// Convert frequency to the nearest note name and octave.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz to convert
    ///
    /// # Returns
    ///
    /// A tuple containing the note name (String) and octave number (u8)
    pub fn frequency_to_note(frequency: f32) -> (String, u8) {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        let a4_freq = 440.0;
        let semitones_from_a4 = (frequency / a4_freq).log2() * 12.0;
        // A4 is note 9 (A) in octave 4, which is semitone 57 from C0
        let total_semitones = (semitones_from_a4 + 57.0) as i32;

        let octave = (total_semitones / 12) as u8;
        let note_index = (total_semitones % 12) as usize;

        (note_names[note_index].to_string(), octave)
    }

    /// Get scale notes for a given root note and scale type.
    ///
    /// # Arguments
    ///
    /// * `root` - Root note of the scale (e.g., "C", "D", "F#")
    /// * `scale_type` - Type of scale ("major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian")
    ///
    /// # Returns
    ///
    /// Vector of note names in the specified scale
    pub fn get_scale_notes(root: &str, scale_type: &str) -> Vec<String> {
        let chromatic = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let root_index = chromatic.iter().position(|&n| n == root).unwrap_or(0);

        let intervals = match scale_type {
            "major" => vec![0, 2, 4, 5, 7, 9, 11],
            "minor" => vec![0, 2, 3, 5, 7, 8, 10],
            "dorian" => vec![0, 2, 3, 5, 7, 9, 10],
            "phrygian" => vec![0, 1, 3, 5, 7, 8, 10],
            "lydian" => vec![0, 2, 4, 6, 7, 9, 11],
            "mixolydian" => vec![0, 2, 4, 5, 7, 9, 10],
            "locrian" => vec![0, 1, 3, 5, 6, 8, 10],
            _ => vec![0, 2, 4, 5, 7, 9, 11], // Default to major
        };

        intervals
            .iter()
            .map(|&interval| chromatic[(root_index + interval) % 12].to_string())
            .collect()
    }

    /// Calculate the interval between two notes in semitones.
    ///
    /// # Arguments
    ///
    /// * `from_note` - Starting note name
    /// * `from_octave` - Starting octave number
    /// * `to_note` - Ending note name
    /// * `to_octave` - Ending octave number
    ///
    /// # Returns
    ///
    /// Number of semitones between the two notes (positive for ascending, negative for descending)
    pub fn interval_semitones(
        from_note: &str,
        from_octave: u8,
        to_note: &str,
        to_octave: u8,
    ) -> i32 {
        let from_freq = note_to_frequency(from_note, from_octave);
        let to_freq = note_to_frequency(to_note, to_octave);
        ((to_freq / from_freq).log2() * 12.0).round() as i32
    }
}

/// DSP utility functions
pub mod dsp {
    use std::f32::consts::PI;

    /// Apply a window function to a signal for spectral analysis.
    ///
    /// # Arguments
    ///
    /// * `signal` - Signal samples to modify in-place
    /// * `window_type` - Type of window function to apply
    pub fn apply_window(signal: &mut [f32], window_type: WindowType) {
        let n = signal.len();
        for (i, sample) in signal.iter_mut().enumerate() {
            let window_value = match window_type {
                WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos(),
                WindowType::Hanning => 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos()),
                WindowType::Blackman => {
                    let angle = 2.0 * PI * i as f32 / (n - 1) as f32;
                    0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos()
                }
                WindowType::Rectangular => 1.0,
            };
            *sample *= window_value;
        }
    }

    /// Window function types for spectral analysis.
    ///
    /// Different window types provide different trade-offs between
    /// frequency resolution and spectral leakage reduction.
    #[derive(Debug, Clone, Copy)]
    pub enum WindowType {
        /// Hamming window (raised cosine)
        Hamming,
        /// Hanning window (cosine squared)
        Hanning,
        /// Blackman window (three-term cosine)
        Blackman,
        /// Rectangular window (no tapering)
        Rectangular,
    }

    /// Apply a simple first-order low-pass filter to remove high frequencies.
    ///
    /// # Arguments
    ///
    /// * `signal` - Signal samples to filter in-place
    /// * `cutoff` - Cutoff frequency in Hz (frequencies above this are attenuated)
    /// * `sample_rate` - Sample rate of the signal in Hz
    pub fn low_pass_filter(signal: &mut [f32], cutoff: f32, sample_rate: f32) {
        let rc = 1.0 / (2.0 * PI * cutoff);
        let dt = 1.0 / sample_rate;
        let alpha = dt / (rc + dt);

        for i in 1..signal.len() {
            signal[i] = signal[i - 1] + alpha * (signal[i] - signal[i - 1]);
        }
    }

    /// Apply a simple first-order high-pass filter to remove low frequencies.
    ///
    /// # Arguments
    ///
    /// * `signal` - Signal samples to filter in-place
    /// * `cutoff` - Cutoff frequency in Hz (frequencies below this are attenuated)
    /// * `sample_rate` - Sample rate of the signal in Hz
    pub fn high_pass_filter(signal: &mut [f32], cutoff: f32, sample_rate: f32) {
        let rc = 1.0 / (2.0 * PI * cutoff);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        let mut prev_input = signal[0];
        let mut prev_output = signal[0];

        for i in 1..signal.len() {
            let current_input = signal[i];
            let current_output = alpha * (prev_output + current_input - prev_input);
            signal[i] = current_output;
            prev_input = current_input;
            prev_output = current_output;
        }
    }
}

/// Text processing utilities
pub mod text {
    /// Convert text to phonemes using a simple grapheme-to-phoneme mapping.
    ///
    /// This is a basic G2P implementation for demonstration purposes.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to convert to phonemes
    ///
    /// # Returns
    ///
    /// Vector of phoneme strings in IPA notation
    pub fn text_to_phonemes(text: &str) -> Vec<String> {
        // Very simple phoneme mapping
        let phoneme_map = [
            ("a", "æ"),
            ("e", "ɛ"),
            ("i", "ɪ"),
            ("o", "ɔ"),
            ("u", "ʊ"),
            ("th", "θ"),
            ("sh", "ʃ"),
            ("ch", "tʃ"),
            ("ng", "ŋ"),
        ];

        let mut result = Vec::new();
        let text_lower = text.to_lowercase();
        let mut chars = text_lower.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch.is_alphabetic() {
                let mut phoneme = ch.to_string();

                // Check for digraphs
                if let Some(&next_ch) = chars.peek() {
                    let digraph = format!("{ch}{next_ch}");
                    if phoneme_map.iter().any(|(key, _)| *key == digraph) {
                        phoneme = digraph;
                        chars.next(); // Consume the next character
                    }
                }

                // Map to phoneme
                let mapped = phoneme_map
                    .iter()
                    .find(|(key, _)| *key == phoneme)
                    .map(|(_, value)| value.to_string())
                    .unwrap_or(phoneme);

                result.push(mapped);
            }
        }

        result
    }

    /// Extract syllables from text using vowel-based segmentation.
    ///
    /// This is a simple heuristic-based syllabification algorithm.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to syllabify
    ///
    /// # Returns
    ///
    /// Vector of syllable strings
    pub fn text_to_syllables(text: &str) -> Vec<String> {
        // Simple syllable extraction
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut syllables = Vec::new();

        for word in words {
            // Very basic syllable counting based on vowels
            let vowels = ['a', 'e', 'i', 'o', 'u'];
            let mut word_syllables = Vec::new();
            let mut current_syllable = String::new();

            for ch in word.chars() {
                current_syllable.push(ch);
                if vowels.contains(&ch.to_lowercase().next().unwrap_or('x')) {
                    word_syllables.push(std::mem::take(&mut current_syllable));
                }
            }

            if !current_syllable.is_empty() {
                if let Some(last) = word_syllables.last_mut() {
                    last.push_str(&current_syllable);
                } else {
                    word_syllables.push(current_syllable);
                }
            }

            if word_syllables.is_empty() {
                word_syllables.push(word.to_string());
            }

            syllables.extend(word_syllables);
        }

        syllables
    }
}

/// Voice utilities
pub mod voice {
    use crate::types::VoiceType;

    /// Get typical vocal characteristics for a given voice type.
    ///
    /// # Arguments
    ///
    /// * `voice_type` - The voice type (Soprano, Alto, Tenor, Bass, etc.)
    ///
    /// # Returns
    ///
    /// VoiceCharacteristics structure with typical values for the voice type
    pub fn get_voice_type_characteristics(
        voice_type: VoiceType,
    ) -> crate::types::VoiceCharacteristics {
        let mut characteristics = crate::types::VoiceCharacteristics::default();
        characteristics.voice_type = voice_type;
        characteristics.range = voice_type.frequency_range();
        characteristics.f0_mean = voice_type.f0_mean();

        // Adjust other characteristics based on voice type
        match voice_type {
            VoiceType::Soprano => {
                characteristics.vibrato_frequency = 6.5;
                characteristics.vibrato_depth = 0.4;
                characteristics.vocal_power = 0.9;
            }
            VoiceType::Alto => {
                characteristics.vibrato_frequency = 6.0;
                characteristics.vibrato_depth = 0.3;
                characteristics.vocal_power = 0.8;
            }
            VoiceType::Tenor => {
                characteristics.vibrato_frequency = 5.5;
                characteristics.vibrato_depth = 0.35;
                characteristics.vocal_power = 0.85;
            }
            VoiceType::Bass => {
                characteristics.vibrato_frequency = 5.0;
                characteristics.vibrato_depth = 0.3;
                characteristics.vocal_power = 0.9;
            }
            _ => {}
        }

        characteristics
    }

    /// Blend two voice characteristics using linear interpolation.
    ///
    /// # Arguments
    ///
    /// * `voice1` - First voice characteristics (weight = 1.0 - factor)
    /// * `voice2` - Second voice characteristics (weight = factor)
    /// * `factor` - Blending factor (0.0 = voice1, 1.0 = voice2), clamped to [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// New VoiceCharacteristics interpolated between the two inputs
    pub fn blend_voice_characteristics(
        voice1: &crate::types::VoiceCharacteristics,
        voice2: &crate::types::VoiceCharacteristics,
        factor: f32,
    ) -> crate::types::VoiceCharacteristics {
        let factor = factor.clamp(0.0, 1.0);
        let mut result = voice1.clone();

        result.f0_mean = voice1.f0_mean * (1.0 - factor) + voice2.f0_mean * factor;
        result.f0_std = voice1.f0_std * (1.0 - factor) + voice2.f0_std * factor;
        result.vibrato_frequency =
            voice1.vibrato_frequency * (1.0 - factor) + voice2.vibrato_frequency * factor;
        result.vibrato_depth =
            voice1.vibrato_depth * (1.0 - factor) + voice2.vibrato_depth * factor;
        result.vocal_power = voice1.vocal_power * (1.0 - factor) + voice2.vocal_power * factor;

        result
    }
}

#[cfg(test)]
mod tests {
    use super::{audio, music, text, voice};
    use crate::types::VoiceType;

    #[test]
    fn test_note_to_frequency() {
        assert!((music::note_to_frequency("A", 4) - 440.0).abs() < 0.1);
        assert!((music::note_to_frequency("C", 4) - 261.63).abs() < 0.1);
    }

    #[test]
    fn test_frequency_to_note() {
        let (note, octave) = music::frequency_to_note(440.0);
        assert_eq!(note, "A");
        assert_eq!(octave, 4);
    }

    #[test]
    fn test_audio_normalization() {
        let mut samples = vec![0.5, -0.8, 0.3, -0.2];
        audio::normalize(&mut samples, 1.0);
        assert_eq!(samples.iter().map(|&x| x.abs()).fold(0.0, f32::max), 1.0);
    }

    #[test]
    fn test_text_to_phonemes() {
        let phonemes = text::text_to_phonemes("hello");
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_get_scale_notes() {
        let c_major = music::get_scale_notes("C", "major");
        assert_eq!(c_major.len(), 7);
        assert_eq!(c_major[0], "C");
        assert_eq!(c_major[1], "D");
        assert_eq!(c_major[2], "E");
    }

    #[test]
    fn test_voice_type_characteristics() {
        let soprano = voice::get_voice_type_characteristics(VoiceType::Soprano);
        let bass = voice::get_voice_type_characteristics(VoiceType::Bass);

        assert!(soprano.f0_mean > bass.f0_mean);
        assert_eq!(soprano.voice_type, VoiceType::Soprano);
        assert_eq!(bass.voice_type, VoiceType::Bass);
    }
}
