//! Audio processing module for voirs-vocoder.
//!
//! This module provides comprehensive audio processing capabilities including:
//! - Audio buffer operations and format conversions
//! - Audio I/O (WAV, FLAC, MP3, etc.)
//! - Audio analysis and quality metrics
//! - Real-time audio processing utilities

pub mod ops;
pub mod io;
pub mod analysis;

pub use ops::*;
pub use io::*;
pub use analysis::*;

use crate::{AudioBuffer, Result};

/// Audio format types supported by the vocoder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// 32-bit floating point
    F32,
    /// 16-bit signed integer
    I16,
    /// 24-bit signed integer
    I24,
    /// 32-bit signed integer
    I32,
}

/// Audio quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioQuality {
    /// Low quality (fast processing)
    Low,
    /// Medium quality (balanced)
    Medium,
    /// High quality (slow processing)
    High,
    /// Ultra quality (very slow processing)
    Ultra,
}

/// Audio output configuration
#[derive(Debug, Clone)]
pub struct AudioOutputConfig {
    /// Output sample rate
    pub sample_rate: u32,
    /// Number of output channels
    pub channels: u32,
    /// Audio format
    pub format: AudioFormat,
    /// Quality setting
    pub quality: AudioQuality,
    /// Enable real-time processing
    pub realtime: bool,
}

impl Default for AudioOutputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            format: AudioFormat::F32,
            quality: AudioQuality::Medium,
            realtime: false,
        }
    }
}

/// Audio processing context
pub struct AudioProcessor {
    config: AudioOutputConfig,
}

impl AudioProcessor {
    /// Create new audio processor
    pub fn new(config: AudioOutputConfig) -> Self {
        Self { config }
    }

    /// Process audio buffer with current configuration
    pub fn process(&self, audio: &mut AudioBuffer) -> Result<()> {
        // Apply quality-based processing
        match self.config.quality {
            AudioQuality::Low => {
                // Basic processing only
            }
            AudioQuality::Medium => {
                // Standard processing
                post_process_audio(audio)?;
            }
            AudioQuality::High => {
                // Enhanced processing
                post_process_audio(audio)?;
                apply_enhancement_filters(audio)?;
            }
            AudioQuality::Ultra => {
                // Maximum quality processing
                post_process_audio(audio)?;
                apply_enhancement_filters(audio)?;
                apply_advanced_processing(audio)?;
            }
        }

        Ok(())
    }
}

/// Apply post-processing to audio
pub fn post_process_audio(audio: &mut AudioBuffer) -> Result<()> {
    // DC offset removal
    remove_dc_offset(audio);
    
    // Normalize audio levels
    normalize_audio(audio);
    
    Ok(())
}

/// Apply enhancement filters
pub fn apply_enhancement_filters(audio: &mut AudioBuffer) -> Result<()> {
    // High-frequency enhancement
    enhance_high_frequencies(audio);
    
    // Dynamic range optimization
    optimize_dynamic_range(audio);
    
    Ok(())
}

/// Apply advanced processing
pub fn apply_advanced_processing(audio: &mut AudioBuffer) -> Result<()> {
    // Spectral enhancement
    enhance_spectral_quality(audio);
    
    // Harmonic enhancement
    enhance_harmonics(audio);
    
    Ok(())
}

/// Remove DC offset from audio
pub fn remove_dc_offset(audio: &mut AudioBuffer) {
    let samples = audio.samples_mut();
    if samples.is_empty() {
        return;
    }
    
    // Calculate DC offset
    let dc_offset: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    
    // Remove DC offset
    for sample in samples.iter_mut() {
        *sample -= dc_offset;
    }
}

/// Normalize audio levels
pub fn normalize_audio(audio: &mut AudioBuffer) {
    let samples = audio.samples_mut();
    if samples.is_empty() {
        return;
    }
    
    // Find peak level
    let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    
    if peak > 0.0 && peak != 1.0 {
        // Normalize to prevent clipping (leave some headroom)
        let scale = 0.95 / peak;
        for sample in samples.iter_mut() {
            *sample *= scale;
        }
    }
}

/// Enhance high frequencies
pub fn enhance_high_frequencies(audio: &mut AudioBuffer) {
    // Simple high-frequency enhancement (placeholder)
    // In a real implementation, this would use proper DSP filters
    let samples = audio.samples_mut();
    for sample in samples.iter_mut() {
        // Apply subtle high-frequency boost
        *sample *= 1.01;
    }
}

/// Optimize dynamic range
pub fn optimize_dynamic_range(audio: &mut AudioBuffer) {
    // Simple dynamic range optimization (placeholder)
    // In a real implementation, this would use proper compression algorithms
    let samples = audio.samples_mut();
    for sample in samples.iter_mut() {
        // Apply soft compression
        *sample = sample.signum() * (sample.abs().powf(0.9));
    }
}

/// Enhance spectral quality
pub fn enhance_spectral_quality(audio: &mut AudioBuffer) {
    // Placeholder for spectral enhancement
    // In a real implementation, this would use FFT-based processing
    let _ = audio;
}

/// Enhance harmonics
pub fn enhance_harmonics(audio: &mut AudioBuffer) {
    // Placeholder for harmonic enhancement
    // In a real implementation, this would use harmonic analysis
    let _ = audio;
}

/// Extensions for AudioBuffer
impl AudioBuffer {
    /// Get mutable samples
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.samples
    }

    /// Convert to different format
    pub fn convert_format(&self, format: AudioFormat) -> Result<Vec<u8>> {
        match format {
            AudioFormat::F32 => Ok(self.to_f32_bytes()),
            AudioFormat::I16 => Ok(self.to_i16_bytes()),
            AudioFormat::I24 => Ok(self.to_i24_bytes()),
            AudioFormat::I32 => Ok(self.to_i32_bytes()),
        }
    }

    /// Convert to F32 bytes
    pub fn to_f32_bytes(&self) -> Vec<u8> {
        self.samples.iter()
            .flat_map(|&x| x.to_le_bytes().to_vec())
            .collect()
    }

    /// Convert to I16 bytes
    pub fn to_i16_bytes(&self) -> Vec<u8> {
        self.samples.iter()
            .map(|&x| (x * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect()
    }

    /// Convert to I24 bytes
    pub fn to_i24_bytes(&self) -> Vec<u8> {
        self.samples.iter()
            .map(|&x| (x * 8388607.0).clamp(-8388608.0, 8388607.0) as i32)
            .flat_map(|x| x.to_le_bytes()[..3].to_vec())
            .collect()
    }

    /// Convert to I32 bytes
    pub fn to_i32_bytes(&self) -> Vec<u8> {
        self.samples.iter()
            .map(|&x| (x * 2147483647.0).clamp(-2147483648.0, 2147483647.0) as i32)
            .flat_map(|x| x.to_le_bytes().to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_processor() {
        let config = AudioOutputConfig::default();
        let processor = AudioProcessor::new(config);
        
        let mut audio = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 22050, 1);
        let result = processor.process(&mut audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_remove_dc_offset() {
        let mut audio = AudioBuffer::new(vec![1.0, 1.0, 1.0, 1.0], 22050, 1);
        remove_dc_offset(&mut audio);
        
        // All samples should be 0 after DC removal
        for &sample in audio.samples() {
            assert_eq!(sample, 0.0);
        }
    }

    #[test]
    fn test_normalize_audio() {
        let mut audio = AudioBuffer::new(vec![2.0, -2.0, 1.0, -1.0], 22050, 1);
        normalize_audio(&mut audio);
        
        // Peak should be around 0.95 (95% of max)
        let peak = audio.samples().iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((peak - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_format_conversion() {
        let audio = AudioBuffer::new(vec![0.5, -0.5, 0.25, -0.25], 22050, 1);
        
        // Test I16 conversion
        let i16_bytes = audio.to_i16_bytes();
        assert_eq!(i16_bytes.len(), 8); // 4 samples * 2 bytes per sample
        
        // Test F32 conversion
        let f32_bytes = audio.to_f32_bytes();
        assert_eq!(f32_bytes.len(), 16); // 4 samples * 4 bytes per sample
    }
}