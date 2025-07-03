//! Audio processing module with modular architecture.
//!
//! This module provides comprehensive audio processing capabilities organized into
//! modular components:
//!
//! - [`buffer`] - Core AudioBuffer struct and basic operations
//! - [`processing`] - Audio processing functions (gain, normalize, mix, etc.)
//! - [`io`] - Audio I/O operations (save, load, format conversion)
//!
//! # Example
//!
//! ```no_run
//! use voirs_sdk::audio::{AudioBuffer, AudioFormat};
//!
//! // Create a sine wave
//! let buffer = AudioBuffer::sine_wave(440.0, 2.0, 44100, 0.5);
//!
//! // Apply some processing
//! let mut processed = buffer.clone();
//! processed.apply_gain(6.0).unwrap();
//! processed.normalize(0.8).unwrap();
//!
//! // Save to file
//! processed.save("output.wav", AudioFormat::Wav).unwrap();
//! ```

pub mod buffer;
pub mod processing;
pub mod io;

// Re-export the main types for convenience
pub use buffer::{AudioBuffer, AudioMetadata, BufferFormat};
pub use io::{AudioInfo, RawFormat};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AudioFormat;
    use tempfile::NamedTempFile;

    #[test]
    fn test_full_audio_pipeline() {
        // Create a test audio buffer
        let mut buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        
        // Apply some processing
        buffer.apply_gain(3.0).unwrap();
        buffer.normalize(0.8).unwrap();
        buffer.fade_in(0.1).unwrap();
        buffer.fade_out(0.1).unwrap();
        
        // Mix with another buffer
        let other = AudioBuffer::sine_wave(880.0, 1.0, 44100, 0.3);
        buffer.mix(&other, 0.5).unwrap();
        
        // Split and append
        let (first, second) = buffer.split(0.5).unwrap();
        let mut combined = first;
        combined.append(&second).unwrap();
        
        // Save to file with proper extension
        let temp_file = NamedTempFile::with_suffix(".wav").unwrap();
        combined.save(temp_file.path(), AudioFormat::Wav).unwrap();
        
        // Load back and verify
        let loaded = AudioBuffer::load(temp_file.path()).unwrap();
        assert_eq!(loaded.sample_rate(), combined.sample_rate());
        assert_eq!(loaded.channels(), combined.channels());
        assert!((loaded.duration() - combined.duration()).abs() < 0.01);
    }

    #[test]
    fn test_buffer_format_compatibility() {
        let format1 = BufferFormat::mono(44100);
        let format2 = BufferFormat::stereo(44100);
        let format3 = BufferFormat::mono(22050);
        
        assert!(!format1.is_compatible(&format2)); // Different channels
        assert!(!format1.is_compatible(&format3)); // Different sample rate
        assert!(format1.is_compatible(&format1)); // Same format
    }

    #[test]
    fn test_metadata_accuracy() {
        let samples = vec![0.5, -0.3, 0.8, -0.1, 0.0];
        let buffer = AudioBuffer::mono(samples.clone(), 44100);
        
        let metadata = buffer.metadata();
        
        // Check peak amplitude
        let expected_peak = samples.iter().map(|&s| s.abs()).fold(0.0, f32::max);
        assert_eq!(metadata.peak_amplitude, expected_peak);
        
        // Check duration
        let expected_duration = samples.len() as f32 / 44100.0;
        assert!((metadata.duration - expected_duration).abs() < 0.001);
        
        // Check RMS
        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        let expected_rms = (sum_squares / samples.len() as f32).sqrt();
        assert!((metadata.rms_amplitude - expected_rms).abs() < 0.001);
    }

    #[test]
    fn test_advanced_processing() {
        let mut buffer = AudioBuffer::sine_wave(440.0, 2.0, 44100, 0.7);
        
        // Apply advanced processing
        buffer.lowpass_filter(2000.0).unwrap();
        buffer.compress(0.5, 4.0, 10.0, 100.0).unwrap();
        buffer.reverb(0.3, 0.2, 0.1).unwrap();
        
        // Test time stretching
        let stretched = buffer.time_stretch(1.5).unwrap();
        assert!((stretched.duration() - buffer.duration() / 1.5).abs() < 0.1);
        
        // Test pitch shifting
        let shifted = buffer.pitch_shift(12.0).unwrap();
        assert!((shifted.duration() - buffer.duration()).abs() < 0.1);
    }

    #[test]
    fn test_io_operations() {
        let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        
        // Test byte conversion
        let wav_bytes = buffer.to_wav_bytes().unwrap();
        assert!(!wav_bytes.is_empty());
        
        // Test streaming
        let mut chunk_count = 0;
        buffer.stream_to_callback(1024, |_chunk| {
            chunk_count += 1;
            Ok(())
        }).unwrap();
        
        let expected_chunks = (buffer.len() + 1023) / 1024; // Ceiling division
        assert_eq!(chunk_count, expected_chunks);
        
        // Test metadata export
        let metadata_json = buffer.export_metadata().unwrap();
        assert!(metadata_json.contains("duration"));
        assert!(metadata_json.contains("peak_amplitude"));
    }

    #[test]
    fn test_error_handling() {
        let buffer1 = AudioBuffer::mono(vec![0.1, 0.2], 44100);
        let buffer2 = AudioBuffer::mono(vec![0.3, 0.4], 22050); // Different sample rate
        
        // Mixing different sample rates should fail
        let mut buffer1_copy = buffer1.clone();
        assert!(buffer1_copy.mix(&buffer2, 0.5).is_err());
        
        // Appending different formats should fail
        assert!(buffer1_copy.append(&buffer2).is_err());
        
        // Invalid split time should fail
        assert!(buffer1.split(10.0).is_err()); // 10 seconds > 2 samples at 44100 Hz
        
        // Invalid extract should fail
        assert!(buffer1.extract(10.0, 1.0).is_err());
    }
}