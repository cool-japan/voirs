//! Harmonic enhancement processor for singing voice vocoder.

use crate::models::singing::config::HarmonicEnhancementConfig;
use anyhow::Result;
use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::Complex;
use scirs2_fft::{FftPlanner, RealFftPlanner};
use std::collections::HashMap;

/// Processor for harmonic enhancement in singing voices
pub struct HarmonicProcessor {
    /// Configuration
    config: HarmonicEnhancementConfig,
    /// Window size for analysis
    window_size: usize,
    /// Hop size for analysis
    #[allow(dead_code)] // Reserved for future overlapping analysis
    hop_size: usize,
    /// Sample rate
    sample_rate: u32,
    /// Harmonic templates for different voice types
    harmonic_templates: HashMap<VoiceType, Vec<f32>>,
}

/// Voice type for adaptive harmonic enhancement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoiceType {
    /// Soprano voice
    Soprano,
    /// Alto voice
    Alto,
    /// Tenor voice
    Tenor,
    /// Bass voice
    Bass,
    /// Generic voice
    Generic,
}

/// Harmonic analysis result
#[derive(Debug, Clone)]
pub struct HarmonicAnalysis {
    /// Fundamental frequency
    pub fundamental_freq: f32,
    /// Harmonic frequencies
    pub harmonic_freqs: Vec<f32>,
    /// Harmonic magnitudes
    pub harmonic_magnitudes: Vec<f32>,
    /// Harmonic phases
    pub harmonic_phases: Vec<f32>,
    /// Detected voice type
    pub voice_type: VoiceType,
}

impl HarmonicProcessor {
    /// Create new harmonic processor
    pub fn new(config: &HarmonicEnhancementConfig) -> Result<Self> {
        let mut processor = Self {
            config: config.clone(),
            window_size: 4096,
            hop_size: 1024,
            sample_rate: 22050,
            harmonic_templates: HashMap::new(),
        };

        // Initialize harmonic templates
        processor.initialize_harmonic_templates();

        Ok(processor)
    }

    /// Initialize harmonic templates for different voice types
    fn initialize_harmonic_templates(&mut self) {
        // Soprano: emphasize higher harmonics
        self.harmonic_templates.insert(
            VoiceType::Soprano,
            vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        );

        // Alto: balanced harmonic content
        self.harmonic_templates.insert(
            VoiceType::Alto,
            vec![1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        );

        // Tenor: emphasize mid-range harmonics
        self.harmonic_templates.insert(
            VoiceType::Tenor,
            vec![1.0, 0.9, 0.8, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05],
        );

        // Bass: emphasize lower harmonics
        self.harmonic_templates.insert(
            VoiceType::Bass,
            vec![1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02],
        );

        // Generic: standard harmonic distribution
        self.harmonic_templates.insert(
            VoiceType::Generic,
            vec![1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02],
        );
    }

    /// Process mel spectrogram for harmonic enhancement
    pub fn process(&mut self, mel_spectrogram: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.config.enable_enhancement {
            return Ok(mel_spectrogram.clone());
        }

        let mut processed = mel_spectrogram.clone();
        let frames = mel_spectrogram.shape()[1];

        // Process each frame
        for frame_idx in 0..frames {
            let frame = mel_spectrogram.column(frame_idx);
            let analysis = self.analyze_harmonics(&frame)?;

            // Apply harmonic enhancement
            self.apply_harmonic_enhancement(&mut processed, frame_idx, &analysis)?;
        }

        Ok(processed)
    }

    /// Analyze harmonic content in a frame
    fn analyze_harmonics(&mut self, frame: &ArrayView1<f32>) -> Result<HarmonicAnalysis> {
        // Convert mel frame to linear spectrum
        let spectrum = self.mel_to_linear_spectrum(frame)?;

        // Apply windowing
        let windowed_spectrum = self.apply_hann_window(&spectrum);

        // Perform FFT
        let fft_input: Vec<Complex<f32>> = windowed_spectrum
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        let fft_output_f64 = scirs2_fft::fft(&fft_input, None)?;

        // Convert f64 output to f32 for processing
        let fft_output: Vec<Complex<f32>> = fft_output_f64
            .into_iter()
            .map(|c| Complex::new(c.re as f32, c.im as f32))
            .collect();

        // Detect fundamental frequency
        let fundamental_freq = self.detect_fundamental_frequency(&fft_output)?;

        // Extract harmonic information
        let (harmonic_freqs, harmonic_magnitudes, harmonic_phases) =
            self.extract_harmonics(&fft_output, fundamental_freq)?;

        // Detect voice type
        let voice_type = self.detect_voice_type(fundamental_freq, &harmonic_magnitudes)?;

        Ok(HarmonicAnalysis {
            fundamental_freq,
            harmonic_freqs,
            harmonic_magnitudes,
            harmonic_phases,
            voice_type,
        })
    }

    /// Detect fundamental frequency from spectrum
    fn detect_fundamental_frequency(&mut self, spectrum: &[Complex<f32>]) -> Result<f32> {
        let mut max_magnitude = 0.0;
        let mut peak_bin = 0;

        // Look for peaks in the fundamental frequency range
        let min_freq = self.config.frequency_range.0;
        let max_freq = self.config.frequency_range.1;
        let min_bin = (min_freq * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;
        let max_bin = (max_freq * spectrum.len() as f32 * 2.0 / self.sample_rate as f32) as usize;

        for (bin, &spectrum_val) in spectrum
            .iter()
            .enumerate()
            .take(max_bin.min(spectrum.len()))
            .skip(min_bin)
        {
            let magnitude = spectrum_val.norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                peak_bin = bin;
            }
        }

        // Convert bin to frequency
        let fundamental_freq =
            (peak_bin as f32 * self.sample_rate as f32) / (2.0 * spectrum.len() as f32);
        Ok(fundamental_freq)
    }

    /// Extract harmonic frequencies, magnitudes, and phases
    fn extract_harmonics(
        &mut self,
        spectrum: &[Complex<f32>],
        fundamental_freq: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut harmonic_freqs = Vec::new();
        let mut harmonic_magnitudes = Vec::new();
        let mut harmonic_phases = Vec::new();

        // Extract harmonics up to the configured count
        for harmonic_idx in 1..=self.config.harmonic_count {
            let harmonic_freq = fundamental_freq * harmonic_idx as f32;
            let harmonic_bin =
                (harmonic_freq * 2.0 * spectrum.len() as f32 / self.sample_rate as f32) as usize;

            if harmonic_bin < spectrum.len() {
                let complex = spectrum[harmonic_bin];
                let magnitude = complex.norm();
                let phase = complex.arg();

                harmonic_freqs.push(harmonic_freq);
                harmonic_magnitudes.push(magnitude);
                harmonic_phases.push(phase);
            }
        }

        Ok((harmonic_freqs, harmonic_magnitudes, harmonic_phases))
    }

    /// Detect voice type based on fundamental frequency and harmonic content
    fn detect_voice_type(
        &mut self,
        fundamental_freq: f32,
        harmonic_magnitudes: &[f32],
    ) -> Result<VoiceType> {
        // Simple voice type detection based on fundamental frequency ranges
        let voice_type = match fundamental_freq {
            f if (440.0..=1046.0).contains(&f) => VoiceType::Soprano,
            f if (262.0..440.0).contains(&f) => VoiceType::Alto,
            f if (165.0..262.0).contains(&f) => VoiceType::Tenor,
            f if (82.0..165.0).contains(&f) => VoiceType::Bass,
            _ => VoiceType::Generic,
        };

        // Refine detection based on harmonic content
        let refined_type = if self.config.adaptive_enhancement {
            self.refine_voice_type_by_harmonics(voice_type, harmonic_magnitudes)?
        } else {
            voice_type
        };

        Ok(refined_type)
    }

    /// Refine voice type detection based on harmonic content
    fn refine_voice_type_by_harmonics(
        &mut self,
        initial_type: VoiceType,
        harmonic_magnitudes: &[f32],
    ) -> Result<VoiceType> {
        let mut best_type = initial_type;
        let mut best_score = 0.0;

        // Compare against all voice type templates
        for (&voice_type, template) in &self.harmonic_templates {
            let score = self.calculate_harmonic_similarity(harmonic_magnitudes, template)?;
            if score > best_score {
                best_score = score;
                best_type = voice_type;
            }
        }

        Ok(best_type)
    }

    /// Calculate similarity between harmonic magnitudes and template
    fn calculate_harmonic_similarity(&self, magnitudes: &[f32], template: &[f32]) -> Result<f32> {
        let min_len = magnitudes.len().min(template.len());
        if min_len == 0 {
            return Ok(0.0);
        }

        let mut correlation = 0.0;
        let mut magnitude_norm = 0.0;
        let mut template_norm = 0.0;

        for i in 0..min_len {
            correlation += magnitudes[i] * template[i];
            magnitude_norm += magnitudes[i] * magnitudes[i];
            template_norm += template[i] * template[i];
        }

        let similarity = if magnitude_norm > 0.0 && template_norm > 0.0 {
            correlation / (magnitude_norm.sqrt() * template_norm.sqrt())
        } else {
            0.0
        };

        Ok(similarity)
    }

    /// Apply harmonic enhancement to frame
    fn apply_harmonic_enhancement(
        &mut self,
        mel_spectrogram: &mut Array2<f32>,
        frame_idx: usize,
        analysis: &HarmonicAnalysis,
    ) -> Result<()> {
        if analysis.fundamental_freq == 0.0 {
            return Ok(());
        }

        let mel_bins = mel_spectrogram.shape()[0];
        let template = self
            .harmonic_templates
            .get(&analysis.voice_type)
            .unwrap_or(&self.harmonic_templates[&VoiceType::Generic])
            .clone();

        // Enhance each harmonic
        for (harmonic_idx, &harmonic_freq) in analysis.harmonic_freqs.iter().enumerate() {
            let mel_bin = self.hz_to_mel_bin(harmonic_freq, mel_bins);

            if mel_bin < mel_bins {
                // Get enhancement strength for this harmonic
                let enhancement_strength = if harmonic_idx < self.config.enhancement_strengths.len()
                {
                    self.config.enhancement_strengths[harmonic_idx]
                } else if harmonic_idx < template.len() {
                    template[harmonic_idx]
                } else {
                    0.1 // Default fallback
                };

                // Apply enhancement
                let current_value = mel_spectrogram[[mel_bin, frame_idx]];
                let enhanced_value = current_value * (1.0 + enhancement_strength);

                mel_spectrogram[[mel_bin, frame_idx]] = enhanced_value;

                // Apply enhancement to neighboring bins for smoother result
                for offset in -2..=2 {
                    let neighbor_bin = (mel_bin as i32 + offset) as usize;
                    if neighbor_bin < mel_bins && neighbor_bin != mel_bin {
                        let neighbor_value = mel_spectrogram[[neighbor_bin, frame_idx]];
                        let neighbor_enhancement =
                            enhancement_strength * 0.3 * (1.0 - offset.abs() as f32 / 2.0);
                        mel_spectrogram[[neighbor_bin, frame_idx]] =
                            neighbor_value * (1.0 + neighbor_enhancement);
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert mel frame to linear spectrum
    fn mel_to_linear_spectrum(&mut self, frame: &ArrayView1<f32>) -> Result<Vec<f32>> {
        let mel_bins = frame.len();
        let mut spectrum = vec![0.0; self.window_size / 2 + 1];

        for (mel_idx, &mel_value) in frame.iter().enumerate() {
            let mel_freq =
                (mel_idx as f32 / mel_bins as f32) * self.hz_to_mel(self.sample_rate as f32 / 2.0);
            let hz_freq = self.mel_to_hz(mel_freq);
            let spec_bin =
                (hz_freq / (self.sample_rate as f32 / 2.0) * spectrum.len() as f32) as usize;

            if spec_bin < spectrum.len() {
                spectrum[spec_bin] = mel_value;
            }
        }

        Ok(spectrum)
    }

    /// Apply Hann window to spectrum
    fn apply_hann_window(&mut self, spectrum: &[f32]) -> Vec<f32> {
        let n = spectrum.len();
        spectrum
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let window =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
                value * window
            })
            .collect()
    }

    /// Convert mel frequency to Hz
    fn mel_to_hz(&mut self, mel: f32) -> f32 {
        700.0 * (mel / 1127.0).exp() - 700.0
    }

    /// Convert Hz to mel frequency
    fn hz_to_mel(&mut self, hz: f32) -> f32 {
        1127.0 * (1.0 + hz / 700.0).ln()
    }

    /// Convert Hz to mel bin index
    fn hz_to_mel_bin(&mut self, hz: f32, mel_bins: usize) -> usize {
        let mel_freq = self.hz_to_mel(hz);
        let max_mel = self.hz_to_mel(self.sample_rate as f32 / 2.0);
        let bin_idx = (mel_freq / max_mel * mel_bins as f32) as usize;
        bin_idx.min(mel_bins - 1)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &HarmonicEnhancementConfig) -> Result<()> {
        self.config = config.clone();
        Ok(())
    }

    /// Get harmonic statistics
    pub fn get_harmonic_stats(&mut self, analysis: &HarmonicAnalysis) -> HarmonicStats {
        let total_energy = analysis.harmonic_magnitudes.iter().sum::<f32>();
        let harmonic_ratio = if total_energy > 0.0 {
            analysis.harmonic_magnitudes[0] / total_energy
        } else {
            0.0
        };

        let spectral_centroid = if total_energy > 0.0 {
            analysis
                .harmonic_freqs
                .iter()
                .zip(analysis.harmonic_magnitudes.iter())
                .map(|(freq, mag)| freq * mag)
                .sum::<f32>()
                / total_energy
        } else {
            0.0
        };

        HarmonicStats {
            fundamental_freq: analysis.fundamental_freq,
            harmonic_count: analysis.harmonic_freqs.len() as u32,
            harmonic_ratio,
            spectral_centroid,
            total_energy,
            voice_type: analysis.voice_type,
        }
    }
}

/// Statistics for harmonic analysis
#[derive(Debug, Clone)]
pub struct HarmonicStats {
    /// Fundamental frequency
    pub fundamental_freq: f32,
    /// Number of detected harmonics
    pub harmonic_count: u32,
    /// Ratio of fundamental to total energy
    pub harmonic_ratio: f32,
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Total harmonic energy
    pub total_energy: f32,
    /// Detected voice type
    pub voice_type: VoiceType,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_harmonic_processor_creation() {
        let config = HarmonicEnhancementConfig::default();
        let processor = HarmonicProcessor::new(&config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_harmonic_analysis() {
        let config = HarmonicEnhancementConfig::default();
        let mut processor = HarmonicProcessor::new(&config).unwrap();

        // Create sample mel frame
        let frame = Array1::from_vec(vec![0.1, 0.5, 0.8, 0.3, 0.1]);
        let frame_view = frame.view();

        let analysis = processor.analyze_harmonics(&frame_view);
        assert!(analysis.is_ok());
    }

    #[test]
    fn test_fundamental_frequency_detection() {
        let config = HarmonicEnhancementConfig::default();
        let mut processor = HarmonicProcessor::new(&config).unwrap();

        // Create sample spectrum with peak at 440 Hz
        let spectrum: Vec<Complex<f32>> = (0..1024)
            .map(|i| {
                let freq = i as f32 * 22050.0 / 2048.0;
                let magnitude = if (freq - 440.0).abs() < 10.0 {
                    1.0
                } else {
                    0.1
                };
                Complex::new(magnitude, 0.0)
            })
            .collect();

        let fundamental = processor.detect_fundamental_frequency(&spectrum);
        assert!(fundamental.is_ok());
    }

    #[test]
    fn test_voice_type_detection() {
        let config = HarmonicEnhancementConfig {
            adaptive_enhancement: false,
            ..Default::default()
        };
        let mut processor = HarmonicProcessor::new(&config).unwrap();

        // Test different fundamental frequencies
        let soprano_voice = processor.detect_voice_type(500.0, &[1.0, 0.9, 0.8]);
        assert!(soprano_voice.is_ok());
        assert_eq!(soprano_voice.unwrap(), VoiceType::Soprano);

        let bass_voice = processor.detect_voice_type(120.0, &[1.0, 0.9, 0.7]);
        assert!(bass_voice.is_ok());
        assert_eq!(bass_voice.unwrap(), VoiceType::Bass);
    }

    #[test]
    fn test_harmonic_similarity() {
        let config = HarmonicEnhancementConfig::default();
        let processor = HarmonicProcessor::new(&config).unwrap();

        let magnitudes = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        let template = vec![1.0, 0.8, 0.6, 0.4, 0.2];

        let similarity = processor.calculate_harmonic_similarity(&magnitudes, &template);
        assert!(similarity.is_ok());
        assert!(similarity.unwrap() > 0.8); // Should be high similarity
    }

    #[test]
    fn test_harmonic_extraction() {
        let config = HarmonicEnhancementConfig::default();
        let mut processor = HarmonicProcessor::new(&config).unwrap();

        // Create spectrum with harmonics
        let spectrum: Vec<Complex<f32>> = (0..1024).map(|_i| Complex::new(0.1, 0.0)).collect();

        let result = processor.extract_harmonics(&spectrum, 220.0);
        assert!(result.is_ok());
        let (freqs, mags, phases) = result.unwrap();
        assert!(!freqs.is_empty());
        assert!(!mags.is_empty());
        assert!(!phases.is_empty());
    }

    #[test]
    fn test_process_mel_spectrogram() {
        let config = HarmonicEnhancementConfig::default();
        let mut processor = HarmonicProcessor::new(&config).unwrap();

        // Create sample mel spectrogram
        let mel = Array2::ones((80, 100));
        let result = processor.process(&mel);
        assert!(result.is_ok());

        let processed = result.unwrap();
        assert_eq!(processed.shape(), mel.shape());
    }

    #[test]
    fn test_config_update() {
        let config = HarmonicEnhancementConfig::default();
        let mut processor = HarmonicProcessor::new(&config).unwrap();

        let new_config = HarmonicEnhancementConfig {
            harmonic_count: 8,
            ..Default::default()
        };

        let result = processor.update_config(&new_config);
        assert!(result.is_ok());
        assert_eq!(processor.config.harmonic_count, 8);
    }

    #[test]
    fn test_harmonic_stats() {
        let config = HarmonicEnhancementConfig::default();
        let mut processor = HarmonicProcessor::new(&config).unwrap();

        let analysis = HarmonicAnalysis {
            fundamental_freq: 440.0,
            harmonic_freqs: vec![440.0, 880.0, 1320.0],
            harmonic_magnitudes: vec![1.0, 0.8, 0.6],
            harmonic_phases: vec![0.0, 0.5, 1.0],
            voice_type: VoiceType::Soprano,
        };

        let stats = processor.get_harmonic_stats(&analysis);
        assert_eq!(stats.fundamental_freq, 440.0);
        assert_eq!(stats.harmonic_count, 3);
        assert_eq!(stats.voice_type, VoiceType::Soprano);
    }

    #[test]
    fn test_mel_to_linear_spectrum() {
        let config = HarmonicEnhancementConfig::default();
        let mut processor = HarmonicProcessor::new(&config).unwrap();

        let frame = Array1::from_vec(vec![0.1, 0.5, 0.8, 0.3, 0.1]);
        let frame_view = frame.view();

        let spectrum = processor.mel_to_linear_spectrum(&frame_view);
        assert!(spectrum.is_ok());
        assert!(!spectrum.unwrap().is_empty());
    }
}
