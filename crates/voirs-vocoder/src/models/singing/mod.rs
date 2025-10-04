//! Singing voice vocoder implementation with enhanced musical features.

pub mod artifact_reduction;
pub mod breath_sound;
pub mod config;
pub mod harmonics;
pub mod pitch_stability;
pub mod processor;
pub mod quality_metrics;
pub mod vibrato;

pub use artifact_reduction::*;
pub use breath_sound::*;
pub use config::*;
pub use harmonics::*;
pub use pitch_stability::*;
pub use processor::*;
pub use quality_metrics::*;
pub use vibrato::*;

use crate::hifigan::HiFiGanVocoder;
use crate::{MelSpectrogram, Vocoder};
use anyhow::Result;
use scirs2_core::ndarray::Array2;

/// Singing voice vocoder for enhanced musical content processing
pub struct SingingVocoder {
    /// Base vocoder configuration
    pub config: SingingVocoderConfig,
    /// Base HiFi-GAN vocoder for audio generation
    pub base_vocoder: Option<HiFiGanVocoder>,
    /// Pitch stability processor
    pub pitch_processor: PitchStabilityProcessor,
    /// Vibrato enhancement processor
    pub vibrato_processor: VibratoProcessor,
    /// Harmonic enhancement processor
    pub harmonic_processor: HarmonicProcessor,
    /// Breath sound processor
    pub breath_processor: BreathSoundProcessor,
    /// Artifact reduction processor
    pub artifact_reducer: ArtifactReductionProcessor,
    /// Quality metrics calculator
    pub quality_metrics: SingingQualityMetrics,
}

impl SingingVocoder {
    /// Create new singing vocoder with configuration
    pub fn new(config: SingingVocoderConfig) -> Result<Self> {
        Ok(Self {
            base_vocoder: None, // Will be set via set_base_vocoder
            pitch_processor: PitchStabilityProcessor::new(&config.pitch_stability)?,
            vibrato_processor: VibratoProcessor::new(&config.vibrato)?,
            harmonic_processor: HarmonicProcessor::new(&config.harmonic_enhancement)?,
            breath_processor: BreathSoundProcessor::new(&config.breath_sound)?,
            artifact_reducer: ArtifactReductionProcessor::new(&config.artifact_reduction)?,
            quality_metrics: SingingQualityMetrics::new(),
            config,
        })
    }

    /// Create new singing vocoder with base vocoder
    pub fn with_base_vocoder(
        config: SingingVocoderConfig,
        base_vocoder: HiFiGanVocoder,
    ) -> Result<Self> {
        Ok(Self {
            base_vocoder: Some(base_vocoder),
            pitch_processor: PitchStabilityProcessor::new(&config.pitch_stability)?,
            vibrato_processor: VibratoProcessor::new(&config.vibrato)?,
            harmonic_processor: HarmonicProcessor::new(&config.harmonic_enhancement)?,
            breath_processor: BreathSoundProcessor::new(&config.breath_sound)?,
            artifact_reducer: ArtifactReductionProcessor::new(&config.artifact_reduction)?,
            quality_metrics: SingingQualityMetrics::new(),
            config,
        })
    }

    /// Set the base vocoder for audio generation
    pub fn set_base_vocoder(&mut self, base_vocoder: HiFiGanVocoder) {
        self.base_vocoder = Some(base_vocoder);
    }

    /// Process singing voice with enhanced features
    pub fn process_singing_voice(&mut self, mel_spectrogram: &Array2<f32>) -> Result<Vec<f32>> {
        // Apply pitch stability processing
        let mut processed_mel = self.pitch_processor.process(mel_spectrogram)?;

        // Apply vibrato enhancement
        processed_mel = self.vibrato_processor.process(&processed_mel)?;

        // Apply harmonic enhancement
        processed_mel = self.harmonic_processor.process(&processed_mel)?;

        // Apply breath sound processing
        processed_mel = self.breath_processor.process(&processed_mel)?;

        // Apply artifact reduction
        processed_mel = self.artifact_reducer.process(&processed_mel)?;

        // Generate audio from processed mel spectrogram
        let audio = self.generate_audio(&processed_mel)?;

        // Calculate quality metrics
        let quality = self.quality_metrics.calculate(&audio)?;
        tracing::info!("Singing vocoder quality: {:.3}", quality);

        Ok(audio)
    }

    /// Generate audio from processed mel spectrogram
    fn generate_audio(&self, mel_spectrogram: &Array2<f32>) -> Result<Vec<f32>> {
        if let Some(base_vocoder) = &self.base_vocoder {
            // Use the actual HiFi-GAN vocoder for high-quality audio generation
            // Convert ndarray to MelSpectrogram format with pre-allocated vectors
            let num_rows = mel_spectrogram.nrows();
            let num_cols = mel_spectrogram.ncols();
            let mut mel_data = Vec::with_capacity(num_rows);

            for i in 0..num_rows {
                let mut row = Vec::with_capacity(num_cols);
                row.extend_from_slice(mel_spectrogram.row(i).as_slice().unwrap());
                mel_data.push(row);
            }
            let mel = MelSpectrogram::new(
                mel_data,
                self.config.sample_rate,
                self.config.hop_size as u32,
            );

            // Use async runtime to call the vocoder
            let rt = tokio::runtime::Runtime::new()?;
            let audio_buffer = rt.block_on(async { base_vocoder.vocode(&mel, None).await })?;

            Ok(audio_buffer.samples)
        } else {
            // Enhanced singing vocoder with built-in mel-to-audio synthesis
            tracing::info!("Using built-in singing vocoder for mel-to-audio synthesis");

            let frames = mel_spectrogram.shape()[1];
            let _mel_bins = mel_spectrogram.shape()[0];
            let audio_length = frames * self.config.hop_size;

            // Generate high-quality singing audio from mel spectrogram
            let mut audio =
                self.synthesize_singing_audio_from_mel(mel_spectrogram, audio_length)?;

            // Apply singing-specific processing pipeline
            audio = self.apply_singing_processing_pipeline(&audio)?;

            Ok(audio)
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &SingingVocoderConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SingingVocoderConfig) -> Result<()> {
        self.config = config;
        self.pitch_processor
            .update_config(&self.config.pitch_stability)?;
        self.vibrato_processor.update_config(&self.config.vibrato)?;
        self.harmonic_processor
            .update_config(&self.config.harmonic_enhancement)?;
        self.breath_processor
            .update_config(&self.config.breath_sound)?;
        self.artifact_reducer
            .update_config(&self.config.artifact_reduction)?;
        Ok(())
    }

    /// Synthesize singing audio from mel spectrogram using vocal-specific techniques
    fn synthesize_singing_audio_from_mel(
        &self,
        mel_spectrogram: &Array2<f32>,
        audio_length: usize,
    ) -> Result<Vec<f32>> {
        let frames = mel_spectrogram.shape()[1];
        let mel_bins = mel_spectrogram.shape()[0];
        let mut audio = vec![0.0; audio_length];

        // Enhanced singing synthesis with formant modeling and harmonic generation
        for (i, sample_ref) in audio.iter_mut().enumerate().take(audio_length) {
            let time = i as f32 / self.config.sample_rate as f32;
            let frame_idx = (i / self.config.hop_size).min(frames - 1);

            // Extract fundamental frequency from mel spectrogram
            let _f0 = self.extract_f0_from_mel(mel_spectrogram, frame_idx);

            // Generate harmonic series for singing voice
            let mut sample = 0.0;
            for (bin_idx, &magnitude) in mel_spectrogram.column(frame_idx).iter().enumerate() {
                // Convert mel bin to frequency with better mel-to-linear mapping
                let freq = self.mel_bin_to_frequency(bin_idx, mel_bins);

                // Apply singing-specific formant enhancement
                let formant_boost = self.apply_vocal_formants(freq);
                let enhanced_magnitude = magnitude * formant_boost;

                // Generate harmonics with vibrato modulation
                let vibrato_freq = self.apply_vibrato_modulation(freq, time);
                let phase = 2.0 * std::f32::consts::PI * vibrato_freq * time;

                // Add harmonic with appropriate amplitude
                sample += enhanced_magnitude * phase.sin() * 0.005; // Reduced amplitude for clarity

                // Add sub-harmonics for richer vocal texture
                if freq > 160.0 && bin_idx % 2 == 0 {
                    let sub_harmonic_phase = std::f32::consts::PI * vibrato_freq * time;
                    sample += enhanced_magnitude * 0.3 * sub_harmonic_phase.sin() * 0.002;
                }
            }

            // Apply breath noise if configured
            if self.config.breath_sound.enable_processing {
                sample += self.generate_breath_noise(time) * 0.01;
            }

            // Apply vocal dynamics
            sample = self.apply_vocal_dynamics(sample, time);

            *sample_ref = sample.tanh(); // Soft clipping for vocal character
        }

        Ok(audio)
    }

    /// Apply singing-specific processing pipeline
    fn apply_singing_processing_pipeline(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut processed_audio = audio.to_vec();

        // Apply audio-level processing instead of mel spectrogram processing
        // Note: These processors work on mel spectrograms, so we skip direct audio processing here
        // The enhancement is already applied in the mel-to-audio synthesis stage

        // 6. Final vocal post-processing
        self.apply_final_vocal_processing(&mut processed_audio);

        Ok(processed_audio)
    }

    /// Extract fundamental frequency from mel spectrogram
    fn extract_f0_from_mel(&self, mel_spectrogram: &Array2<f32>, frame_idx: usize) -> f32 {
        let frame = mel_spectrogram.column(frame_idx);
        let mut max_magnitude = 0.0;
        let mut f0_bin = 0;

        // Find the bin with maximum magnitude in the fundamental frequency range
        for (bin_idx, &magnitude) in frame.iter().enumerate() {
            let freq = self.mel_bin_to_frequency(bin_idx, mel_spectrogram.shape()[0]);
            if (80.0..=800.0).contains(&freq) && magnitude > max_magnitude {
                max_magnitude = magnitude;
                f0_bin = bin_idx;
            }
        }

        self.mel_bin_to_frequency(f0_bin, mel_spectrogram.shape()[0])
    }

    /// Convert mel bin to frequency
    fn mel_bin_to_frequency(&self, bin_idx: usize, total_bins: usize) -> f32 {
        let mel_min = 0.0;
        let mel_max = 2595.0 * (1.0f32 + 8000.0 / 700.0).ln(); // ~8kHz max
        let mel = mel_min + (bin_idx as f32 / total_bins as f32) * (mel_max - mel_min);
        700.0 * (mel / 2595.0).exp() - 700.0
    }

    /// Apply vocal formants (resonances)
    fn apply_vocal_formants(&self, frequency: f32) -> f32 {
        // Typical vocal formants for singing: F1=500Hz, F2=1500Hz, F3=2500Hz
        let formants = [500.0, 1500.0, 2500.0];
        let mut boost = 1.0;

        for &formant in &formants {
            let distance = (frequency - formant).abs();
            let bandwidth = 100.0; // Formant bandwidth

            if distance < bandwidth {
                let formant_strength = 1.0 - (distance / bandwidth);
                boost += formant_strength * 0.5; // Boost formant frequencies
            }
        }

        boost.min(2.0) // Limit boost to prevent distortion
    }

    /// Apply vibrato modulation
    fn apply_vibrato_modulation(&self, frequency: f32, time: f32) -> f32 {
        if !self.config.vibrato.enable_enhancement {
            return frequency;
        }

        let vibrato_rate = 5.0; // Default Hz rate
        let vibrato_depth = self.config.vibrato.enhancement_strength * 50.0; // Convert to cents

        let modulation = (2.0 * std::f32::consts::PI * vibrato_rate * time).sin();
        let freq_modulation = frequency * (vibrato_depth / 1200.0) * modulation; // Cents to frequency ratio

        frequency + freq_modulation
    }

    /// Generate breath noise
    fn generate_breath_noise(&self, _time: f32) -> f32 {
        use fastrand::f32;

        // Generate filtered noise for breath sounds
        let noise = f32() * 2.0 - 1.0; // White noise

        // Apply simple high-pass filtering for breath character
        let cutoff = 2000.0; // Hz
        let sample_rate = self.config.sample_rate as f32;
        let alpha = 1.0 - (-2.0 * std::f32::consts::PI * cutoff / sample_rate).exp();

        noise * alpha * self.config.breath_sound.enhancement_strength
    }

    /// Apply vocal dynamics
    fn apply_vocal_dynamics(&self, sample: f32, time: f32) -> f32 {
        // Simple envelope for natural vocal dynamics
        let envelope_freq = 0.5; // Hz - slow modulation
        let envelope = 0.8 + 0.2 * (2.0 * std::f32::consts::PI * envelope_freq * time).sin();

        sample * envelope
    }

    /// Apply final vocal post-processing
    fn apply_final_vocal_processing(&self, audio: &mut [f32]) {
        // Apply gentle compression for vocal consistency
        let threshold = 0.7;
        let ratio = 3.0;

        for sample in audio.iter_mut() {
            if sample.abs() > threshold {
                let excess = sample.abs() - threshold;
                let compressed_excess = excess / ratio;
                *sample = (*sample).signum() * (threshold + compressed_excess);
            }
        }

        // Apply gentle high-frequency emphasis for vocal clarity
        let mut prev_sample = 0.0;
        for sample in audio.iter_mut() {
            let high_freq = *sample - prev_sample * 0.9;
            *sample += high_freq * 0.1;
            prev_sample = *sample;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singing_vocoder_creation() {
        let config = SingingVocoderConfig::default();
        let vocoder = SingingVocoder::new(config);
        assert!(vocoder.is_ok());
    }

    #[test]
    fn test_singing_vocoder_processing() {
        let config = SingingVocoderConfig::default();
        let mut vocoder = SingingVocoder::new(config).unwrap();

        // Create sample mel spectrogram
        let mel = Array2::ones((80, 100));
        let result = vocoder.process_singing_voice(&mel);
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_update() {
        let config = SingingVocoderConfig::default();
        let mut vocoder = SingingVocoder::new(config).unwrap();

        let new_config = SingingVocoderConfig {
            pitch_stability: PitchStabilityConfig {
                stability_threshold: 0.1,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = vocoder.update_config(new_config);
        assert!(result.is_ok());
    }
}
