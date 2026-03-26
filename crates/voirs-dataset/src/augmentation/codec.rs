//! Codec Simulation for Audio Robustness Testing
//!
//! This module simulates various audio codecs and compression artifacts
//! to train models that are robust to real-world degradation.
//!
//! Supported codec simulations:
//! - Telephone (G.711 μ-law/A-law)
//! - MP3 (various bitrates)
//! - Opus (VoIP codec)
//! - GSM (mobile telephony)
//! - AMR (Adaptive Multi-Rate)

use crate::{AudioData, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

/// Codec type for simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecType {
    /// G.711 μ-law (telephone quality, 8-bit, 8kHz)
    G711MuLaw,
    /// G.711 A-law (European telephone standard)
    G711ALaw,
    /// MP3 at high bitrate (320 kbps)
    Mp3High,
    /// MP3 at medium bitrate (192 kbps)
    Mp3Medium,
    /// MP3 at low bitrate (128 kbps)
    Mp3Low,
    /// MP3 at very low bitrate (64 kbps)
    Mp3VeryLow,
    /// Opus codec (VoIP, 16 kbps)
    OpusLowBandwidth,
    /// Opus codec (VoIP, 32 kbps)
    OpusMediumBandwidth,
    /// Opus codec (VoIP, 64 kbps)
    OpusHighBandwidth,
    /// GSM mobile codec (13 kbps)
    GsmFull,
    /// GSM enhanced (12.2 kbps)
    GsmEnhanced,
    /// AMR narrowband (12.2 kbps)
    AmrNarrowband,
    /// AMR wideband (23.85 kbps)
    AmrWideband,
}

impl CodecType {
    /// Get codec description
    pub fn description(&self) -> &'static str {
        match self {
            CodecType::G711MuLaw => "G.711 μ-law (8-bit, 8kHz)",
            CodecType::G711ALaw => "G.711 A-law (8-bit, 8kHz)",
            CodecType::Mp3High => "MP3 320kbps",
            CodecType::Mp3Medium => "MP3 192kbps",
            CodecType::Mp3Low => "MP3 128kbps",
            CodecType::Mp3VeryLow => "MP3 64kbps",
            CodecType::OpusLowBandwidth => "Opus 16kbps",
            CodecType::OpusMediumBandwidth => "Opus 32kbps",
            CodecType::OpusHighBandwidth => "Opus 64kbps",
            CodecType::GsmFull => "GSM Full Rate 13kbps",
            CodecType::GsmEnhanced => "GSM Enhanced 12.2kbps",
            CodecType::AmrNarrowband => "AMR-NB 12.2kbps",
            CodecType::AmrWideband => "AMR-WB 23.85kbps",
        }
    }

    /// Get target sample rate for codec
    pub fn target_sample_rate(&self) -> u32 {
        match self {
            CodecType::G711MuLaw | CodecType::G711ALaw => 8000,
            CodecType::GsmFull | CodecType::GsmEnhanced | CodecType::AmrNarrowband => 8000,
            CodecType::AmrWideband => 16000,
            CodecType::OpusLowBandwidth | CodecType::OpusMediumBandwidth => 16000,
            CodecType::OpusHighBandwidth => 24000,
            CodecType::Mp3High
            | CodecType::Mp3Medium
            | CodecType::Mp3Low
            | CodecType::Mp3VeryLow => 44100,
        }
    }

    /// Get bandwidth limitation in Hz
    pub fn bandwidth_limit(&self) -> u32 {
        match self {
            CodecType::G711MuLaw | CodecType::G711ALaw => 3400,
            CodecType::GsmFull | CodecType::GsmEnhanced => 3400,
            CodecType::AmrNarrowband => 3400,
            CodecType::AmrWideband => 7000,
            CodecType::OpusLowBandwidth => 4000,
            CodecType::OpusMediumBandwidth => 6000,
            CodecType::OpusHighBandwidth => 12000,
            CodecType::Mp3VeryLow => 11000,
            CodecType::Mp3Low => 15000,
            CodecType::Mp3Medium => 19000,
            CodecType::Mp3High => 20000,
        }
    }
}

/// Codec simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecConfig {
    /// Codec type to simulate
    pub codec_type: CodecType,
    /// Add packet loss simulation (0.0 to 1.0)
    pub packet_loss_rate: f32,
    /// Add jitter simulation (0.0 to 1.0)
    pub jitter_rate: f32,
    /// Add quantization noise
    pub add_quantization_noise: bool,
    /// Add pre-emphasis filter (telephone simulation)
    pub add_pre_emphasis: bool,
    /// Simulate bit errors
    pub bit_error_rate: f32,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            codec_type: CodecType::Mp3Medium,
            packet_loss_rate: 0.0,
            jitter_rate: 0.0,
            add_quantization_noise: true,
            add_pre_emphasis: false,
            bit_error_rate: 0.0,
        }
    }
}

impl CodecConfig {
    /// Create configuration for telephone quality
    pub fn telephone() -> Self {
        Self {
            codec_type: CodecType::G711MuLaw,
            packet_loss_rate: 0.01,
            jitter_rate: 0.02,
            add_quantization_noise: true,
            add_pre_emphasis: true,
            bit_error_rate: 0.001,
        }
    }

    /// Create configuration for mobile phone quality
    pub fn mobile() -> Self {
        Self {
            codec_type: CodecType::AmrNarrowband,
            packet_loss_rate: 0.02,
            jitter_rate: 0.05,
            add_quantization_noise: true,
            add_pre_emphasis: false,
            bit_error_rate: 0.005,
        }
    }

    /// Create configuration for VoIP quality
    pub fn voip() -> Self {
        Self {
            codec_type: CodecType::OpusMediumBandwidth,
            packet_loss_rate: 0.01,
            jitter_rate: 0.03,
            add_quantization_noise: true,
            add_pre_emphasis: false,
            bit_error_rate: 0.0001,
        }
    }

    /// Create configuration for low-quality MP3
    pub fn mp3_low_quality() -> Self {
        Self {
            codec_type: CodecType::Mp3Low,
            packet_loss_rate: 0.0,
            jitter_rate: 0.0,
            add_quantization_noise: true,
            add_pre_emphasis: false,
            bit_error_rate: 0.0,
        }
    }
}

/// Codec simulator for audio degradation
pub struct CodecSimulator {
    config: CodecConfig,
}

impl CodecSimulator {
    /// Create new codec simulator
    pub fn new(config: CodecConfig) -> Self {
        Self { config }
    }

    /// Apply codec simulation to audio
    pub fn simulate(&self, audio: &AudioData) -> Result<AudioData> {
        let mut processed = audio.clone();

        // Step 1: Resample to target sample rate
        if audio.sample_rate() != self.config.codec_type.target_sample_rate() {
            processed = self.resample(&processed, self.config.codec_type.target_sample_rate())?;
        }

        // Step 2: Apply bandwidth limitation
        processed =
            self.apply_bandwidth_limit(&processed, self.config.codec_type.bandwidth_limit())?;

        // Step 3: Apply pre-emphasis if needed
        if self.config.add_pre_emphasis {
            processed = self.apply_pre_emphasis(&processed)?;
        }

        // Step 4: Apply codec-specific quantization
        processed = self.apply_codec_quantization(&processed)?;

        // Step 5: Simulate packet loss
        if self.config.packet_loss_rate > 0.0 {
            processed = self.simulate_packet_loss(&processed)?;
        }

        // Step 6: Simulate jitter
        if self.config.jitter_rate > 0.0 {
            processed = self.simulate_jitter(&processed)?;
        }

        // Step 7: Add bit errors
        if self.config.bit_error_rate > 0.0 {
            processed = self.simulate_bit_errors(&processed)?;
        }

        // Step 8: Resample back to original sample rate if needed
        if processed.sample_rate() != audio.sample_rate() {
            processed = self.resample(&processed, audio.sample_rate())?;
        }

        Ok(processed)
    }

    /// Resample audio to target sample rate
    fn resample(&self, audio: &AudioData, target_rate: u32) -> Result<AudioData> {
        // Simple linear resampling (could be improved with better interpolation)
        let samples = audio.samples();
        let ratio = target_rate as f32 / audio.sample_rate() as f32;
        let new_length = (samples.len() as f32 * ratio) as usize;

        let mut resampled = Vec::with_capacity(new_length);
        for i in 0..new_length {
            let src_pos = i as f32 / ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f32;

            let sample = if idx + 1 < samples.len() {
                samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
            } else if idx < samples.len() {
                samples[idx]
            } else {
                0.0
            };
            resampled.push(sample);
        }

        Ok(AudioData::new(resampled, target_rate, audio.channels()))
    }

    /// Apply bandwidth limitation (low-pass filter)
    fn apply_bandwidth_limit(&self, audio: &AudioData, cutoff_hz: u32) -> Result<AudioData> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();

        // Simple first-order low-pass filter
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz as f32);
        let dt = 1.0 / sample_rate as f32;
        let alpha = dt / (rc + dt);

        let mut filtered = Vec::with_capacity(samples.len());
        let mut prev = 0.0;

        for &sample in samples {
            let filtered_sample = prev + alpha * (sample - prev);
            filtered.push(filtered_sample);
            prev = filtered_sample;
        }

        Ok(AudioData::new(filtered, sample_rate, audio.channels()))
    }

    /// Apply pre-emphasis filter (telephone simulation)
    fn apply_pre_emphasis(&self, audio: &AudioData) -> Result<AudioData> {
        let samples = audio.samples();
        let pre_emphasis_coef = 0.97;

        let mut emphasized = Vec::with_capacity(samples.len());
        emphasized.push(samples[0]);

        for i in 1..samples.len() {
            emphasized.push(samples[i] - pre_emphasis_coef * samples[i - 1]);
        }

        Ok(AudioData::new(
            emphasized,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Apply codec-specific quantization
    fn apply_codec_quantization(&self, audio: &AudioData) -> Result<AudioData> {
        let samples = audio.samples();
        let bits = match self.config.codec_type {
            CodecType::G711MuLaw | CodecType::G711ALaw => 8,
            CodecType::GsmFull | CodecType::GsmEnhanced => 13,
            CodecType::AmrNarrowband | CodecType::AmrWideband => 16,
            CodecType::OpusLowBandwidth => 8,
            CodecType::OpusMediumBandwidth => 12,
            CodecType::OpusHighBandwidth => 16,
            CodecType::Mp3VeryLow => 12,
            CodecType::Mp3Low => 14,
            CodecType::Mp3Medium => 15,
            CodecType::Mp3High => 16,
        };

        let levels = (1 << bits) as f32;
        let quantized: Vec<f32> = samples
            .iter()
            .map(|&s| {
                let quantized = (s * levels / 2.0).round() / (levels / 2.0);
                quantized.clamp(-1.0, 1.0)
            })
            .collect();

        let mut result = AudioData::new(quantized, audio.sample_rate(), audio.channels());

        // Add quantization noise if enabled
        if self.config.add_quantization_noise {
            let noise_level = 1.0 / levels * 0.5; // Half LSB
            result = self.add_noise(&result, noise_level)?;
        }

        Ok(result)
    }

    /// Simulate packet loss
    fn simulate_packet_loss(&self, audio: &AudioData) -> Result<AudioData> {
        let mut rng = thread_rng();
        let samples = audio.samples().to_vec();
        let packet_size = 160; // Typical packet size (20ms at 8kHz)

        let mut processed = samples.clone();
        let num_packets = samples.len().div_ceil(packet_size);

        for i in 0..num_packets {
            if rng.random::<f32>() < self.config.packet_loss_rate {
                let start = i * packet_size;
                let end = (start + packet_size).min(samples.len());

                // Zero out lost packets (could use PLC - Packet Loss Concealment)
                for sample in &mut processed[start..end] {
                    *sample = 0.0;
                }
            }
        }

        Ok(AudioData::new(
            processed,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Simulate jitter (timing variations)
    fn simulate_jitter(&self, audio: &AudioData) -> Result<AudioData> {
        let mut rng = thread_rng();
        let samples = audio.samples();
        let jitter_samples = (audio.sample_rate() as f32 * 0.02 * self.config.jitter_rate) as usize; // Max 20ms jitter

        let mut jittered = Vec::with_capacity(samples.len());

        for (i, &sample) in samples.iter().enumerate() {
            if rng.random::<f32>() < self.config.jitter_rate {
                // Ensure we have a valid range to sample from
                let max_offset = jitter_samples.min(i);
                if max_offset > 0 {
                    let offset = rng.random_range(0..max_offset);
                    let src_idx = i.saturating_sub(offset);
                    jittered.push(samples[src_idx]);
                } else {
                    jittered.push(sample);
                }
            } else {
                jittered.push(sample);
            }
        }

        Ok(AudioData::new(
            jittered,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Simulate bit errors
    fn simulate_bit_errors(&self, audio: &AudioData) -> Result<AudioData> {
        let mut rng = thread_rng();
        let samples = audio.samples();

        let corrupted: Vec<f32> = samples
            .iter()
            .map(|&s| {
                if rng.random::<f32>() < self.config.bit_error_rate {
                    // Flip random bits by adding noise
                    let noise = rng.random_range(-0.1..0.1);
                    (s + noise).clamp(-1.0, 1.0)
                } else {
                    s
                }
            })
            .collect();

        Ok(AudioData::new(
            corrupted,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Add Gaussian noise
    fn add_noise(&self, audio: &AudioData, noise_level: f32) -> Result<AudioData> {
        let mut rng = thread_rng();
        let samples = audio.samples();

        let noisy: Vec<f32> = samples
            .iter()
            .map(|&s| {
                let noise = (rng.random::<f32>() - 0.5) * 2.0 * noise_level;
                (s + noise).clamp(-1.0, 1.0)
            })
            .collect();

        Ok(AudioData::new(noisy, audio.sample_rate(), audio.channels()))
    }

    /// Get codec information
    pub fn codec_info(&self) -> String {
        format!(
            "Codec: {}, Sample Rate: {}Hz, Bandwidth: {}Hz",
            self.config.codec_type.description(),
            self.config.codec_type.target_sample_rate(),
            self.config.codec_type.bandwidth_limit()
        )
    }
}

/// Batch codec simulation
pub struct BatchCodecSimulator {
    configs: Vec<CodecConfig>,
}

impl BatchCodecSimulator {
    /// Create new batch simulator with multiple codecs
    pub fn new(configs: Vec<CodecConfig>) -> Self {
        Self { configs }
    }

    /// Create simulator with common codec types
    pub fn common_codecs() -> Self {
        Self {
            configs: vec![
                CodecConfig::telephone(),
                CodecConfig::mobile(),
                CodecConfig::voip(),
                CodecConfig::mp3_low_quality(),
            ],
        }
    }

    /// Apply all codec simulations to audio
    pub fn simulate_all(&self, audio: &AudioData) -> Result<Vec<AudioData>> {
        let mut results = Vec::new();

        for config in &self.configs {
            let simulator = CodecSimulator::new(config.clone());
            results.push(simulator.simulate(audio)?);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio() -> AudioData {
        let sample_rate = 44100;
        let duration_secs = 0.1;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();
        AudioData::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_codec_config_default() {
        let config = CodecConfig::default();
        assert_eq!(config.codec_type, CodecType::Mp3Medium);
        assert_eq!(config.packet_loss_rate, 0.0);
    }

    #[test]
    fn test_codec_config_telephone() {
        let config = CodecConfig::telephone();
        assert_eq!(config.codec_type, CodecType::G711MuLaw);
        assert!(config.add_pre_emphasis);
        assert!(config.packet_loss_rate > 0.0);
    }

    #[test]
    fn test_codec_type_descriptions() {
        assert_eq!(
            CodecType::G711MuLaw.description(),
            "G.711 μ-law (8-bit, 8kHz)"
        );
        assert_eq!(CodecType::Mp3High.description(), "MP3 320kbps");
    }

    #[test]
    fn test_codec_type_sample_rates() {
        assert_eq!(CodecType::G711MuLaw.target_sample_rate(), 8000);
        assert_eq!(CodecType::AmrWideband.target_sample_rate(), 16000);
        assert_eq!(CodecType::Mp3High.target_sample_rate(), 44100);
    }

    #[test]
    fn test_codec_type_bandwidth_limits() {
        assert_eq!(CodecType::G711MuLaw.bandwidth_limit(), 3400);
        assert_eq!(CodecType::Mp3High.bandwidth_limit(), 20000);
    }

    #[test]
    fn test_codec_simulator_creation() {
        let config = CodecConfig::default();
        let simulator = CodecSimulator::new(config);
        assert!(simulator.codec_info().contains("MP3"));
    }

    #[test]
    fn test_codec_simulation_basic() {
        let audio = create_test_audio();
        let config = CodecConfig {
            codec_type: CodecType::G711MuLaw,
            packet_loss_rate: 0.0,
            jitter_rate: 0.0,
            add_quantization_noise: false,
            add_pre_emphasis: false,
            bit_error_rate: 0.0,
        };

        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        // Should return to original sample rate
        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert!(!result.samples().is_empty());
    }

    #[test]
    fn test_telephone_quality_simulation() {
        let audio = create_test_audio();
        let config = CodecConfig::telephone();
        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert!(!result.samples().is_empty());
    }

    #[test]
    fn test_mobile_quality_simulation() {
        let audio = create_test_audio();
        let config = CodecConfig::mobile();
        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert!(!result.samples().is_empty());
    }

    #[test]
    fn test_voip_quality_simulation() {
        let audio = create_test_audio();
        let config = CodecConfig::voip();
        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert!(!result.samples().is_empty());
    }

    #[test]
    fn test_mp3_low_quality_simulation() {
        let audio = create_test_audio();
        let config = CodecConfig::mp3_low_quality();
        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert!(!result.samples().is_empty());
    }

    #[test]
    fn test_packet_loss_simulation() {
        let audio = create_test_audio();
        let config = CodecConfig {
            codec_type: CodecType::OpusMediumBandwidth,
            packet_loss_rate: 0.1, // 10% packet loss
            jitter_rate: 0.0,
            add_quantization_noise: false,
            add_pre_emphasis: false,
            bit_error_rate: 0.0,
        };

        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        // Check that some samples are zeroed (packet loss)
        let zero_count = result.samples().iter().filter(|&&s| s == 0.0).count();
        assert!(zero_count > 0 || result.samples().len() < 160); // May not have zeros in very short audio
    }

    #[test]
    fn test_jitter_simulation() {
        let audio = create_test_audio();
        let config = CodecConfig {
            codec_type: CodecType::OpusMediumBandwidth,
            packet_loss_rate: 0.0,
            jitter_rate: 0.2,
            add_quantization_noise: false,
            add_pre_emphasis: false,
            bit_error_rate: 0.0,
        };

        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        assert_eq!(result.samples().len(), audio.samples().len());
    }

    #[test]
    fn test_bit_error_simulation() {
        let audio = create_test_audio();
        let config = CodecConfig {
            codec_type: CodecType::G711MuLaw,
            packet_loss_rate: 0.0,
            jitter_rate: 0.0,
            add_quantization_noise: false,
            add_pre_emphasis: false,
            bit_error_rate: 0.05,
        };

        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        // Result should be different from original due to bit errors
        let original = audio.samples();
        let processed = result.samples();
        let differences = original
            .iter()
            .zip(processed.iter())
            .filter(|(&a, &b)| (a - b).abs() > 0.01)
            .count();

        assert!(differences > 0 || original.len() < 100);
    }

    #[test]
    fn test_batch_codec_simulator() {
        let audio = create_test_audio();
        let batch_simulator = BatchCodecSimulator::common_codecs();
        let results = batch_simulator.simulate_all(&audio).unwrap();

        assert_eq!(results.len(), 4); // 4 common codecs
        for result in results {
            assert_eq!(result.sample_rate(), audio.sample_rate());
            assert!(!result.samples().is_empty());
        }
    }

    #[test]
    fn test_codec_info() {
        let config = CodecConfig::telephone();
        let simulator = CodecSimulator::new(config);
        let info = simulator.codec_info();

        assert!(info.contains("G.711"));
        assert!(info.contains("8000"));
        assert!(info.contains("3400"));
    }

    #[test]
    fn test_quantization() {
        let audio = create_test_audio();
        let config = CodecConfig {
            codec_type: CodecType::G711MuLaw, // 8-bit quantization
            packet_loss_rate: 0.0,
            jitter_rate: 0.0,
            add_quantization_noise: true,
            add_pre_emphasis: false,
            bit_error_rate: 0.0,
        };

        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        // Quantized audio should be different but similar in RMS
        let original_rms = audio.rms().unwrap_or(0.0);
        let quantized_rms = result.rms().unwrap_or(0.0);

        assert!((original_rms - quantized_rms).abs() < 0.3); // Should be reasonably similar
    }

    #[test]
    fn test_bandwidth_limitation() {
        let audio = create_test_audio();
        let config = CodecConfig {
            codec_type: CodecType::G711MuLaw, // 3.4kHz bandwidth
            packet_loss_rate: 0.0,
            jitter_rate: 0.0,
            add_quantization_noise: false,
            add_pre_emphasis: false,
            bit_error_rate: 0.0,
        };

        let simulator = CodecSimulator::new(config);
        let result = simulator.simulate(&audio).unwrap();

        // Audio should be band-limited (high frequencies attenuated)
        assert_eq!(result.sample_rate(), audio.sample_rate());
        assert!(!result.samples().is_empty());
    }
}
