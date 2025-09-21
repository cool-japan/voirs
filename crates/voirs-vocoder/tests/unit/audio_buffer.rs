//! Unit tests for AudioBuffer operations
//!
//! Tests audio buffer manipulation, format conversion, and audio processing
//! operations to ensure correctness and performance.

use voirs_vocoder::{AudioBuffer, VocoderError};
use std::f32::consts::PI;

#[test]
fn test_audio_buffer_creation() {
    let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let buffer = AudioBuffer::new(samples.clone(), 44100, 1).unwrap();
    
    assert_eq!(buffer.samples(), &samples);
    assert_eq!(buffer.sample_rate(), 44100);
    assert_eq!(buffer.channels(), 1);
    assert_eq!(buffer.duration(), 5.0 / 44100.0);
}

#[test]
fn test_audio_buffer_validation() {
    // Test invalid sample rate
    let samples = vec![0.1, 0.2, 0.3];
    let result = AudioBuffer::new(samples.clone(), 0, 1);
    assert!(result.is_err());
    
    // Test invalid channel count
    let result = AudioBuffer::new(samples.clone(), 44100, 0);
    assert!(result.is_err());
    
    // Test empty samples
    let result = AudioBuffer::new(vec![], 44100, 1);
    assert!(result.is_err());
}

#[test]
fn test_sine_wave_generation() {
    let frequency = 440.0;
    let duration = 1.0;
    let sample_rate = 44100;
    let amplitude = 0.5;
    
    let buffer = AudioBuffer::sine_wave(frequency, duration, sample_rate, amplitude);
    
    assert_eq!(buffer.sample_rate(), sample_rate);
    assert_eq!(buffer.channels(), 1);
    assert!((buffer.duration() - duration).abs() < 0.001);
    
    // Verify sine wave properties
    let samples = buffer.samples();
    assert!(!samples.is_empty());
    
    // Check that samples are within expected amplitude range
    for &sample in samples {
        assert!(sample >= -amplitude && sample <= amplitude);
    }
    
    // Verify first few samples match sine wave formula
    let expected_first = (2.0 * PI * frequency / sample_rate as f32).sin() * amplitude;
    assert!((samples[1] - expected_first).abs() < 0.001);
}

#[test]
fn test_silence_generation() {
    let duration = 0.5;
    let sample_rate = 48000;
    let channels = 2;
    
    let buffer = AudioBuffer::silence(duration, sample_rate, channels).unwrap();
    
    assert_eq!(buffer.sample_rate(), sample_rate);
    assert_eq!(buffer.channels(), channels);
    assert!((buffer.duration() - duration).abs() < 0.001);
    
    // Verify all samples are zero
    for &sample in buffer.samples() {
        assert_eq!(sample, 0.0);
    }
}

#[test]
fn test_sample_rate_conversion() {
    let original_rate = 22050;
    let target_rate = 44100;
    let buffer = AudioBuffer::sine_wave(440.0, 1.0, original_rate, 0.5);
    
    let converted = buffer.resample(target_rate).unwrap();
    
    assert_eq!(converted.sample_rate(), target_rate);
    assert_eq!(converted.channels(), buffer.channels());
    assert!((converted.duration() - buffer.duration()).abs() < 0.01);
    
    // Verify the number of samples approximately doubled
    let expected_samples = (buffer.samples().len() as f32 * target_rate as f32 / original_rate as f32) as usize;
    assert!((converted.samples().len() as i32 - expected_samples as i32).abs() <= 1);
}

#[test]
fn test_channel_conversion() {
    // Test mono to stereo
    let mono = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
    let stereo = mono.to_stereo().unwrap();
    
    assert_eq!(stereo.channels(), 2);
    assert_eq!(stereo.sample_rate(), mono.sample_rate());
    assert_eq!(stereo.samples().len(), mono.samples().len() * 2);
    
    // Test stereo to mono
    let back_to_mono = stereo.to_mono().unwrap();
    assert_eq!(back_to_mono.channels(), 1);
    assert_eq!(back_to_mono.sample_rate(), stereo.sample_rate());
    assert_eq!(back_to_mono.samples().len(), stereo.samples().len() / 2);
}

#[test]
fn test_format_conversion() {
    let buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.8);
    
    // Test f32 to i16 conversion
    let i16_data = buffer.to_i16();
    assert_eq!(i16_data.len(), buffer.samples().len());
    
    // Verify conversion accuracy
    for (i, &sample) in buffer.samples().iter().enumerate() {
        let expected = (sample * i16::MAX as f32) as i16;
        assert!((i16_data[i] - expected).abs() <= 1);
    }
    
    // Test i16 to f32 conversion
    let back_to_f32 = AudioBuffer::from_i16(&i16_data, buffer.sample_rate(), buffer.channels()).unwrap();
    assert_eq!(back_to_f32.samples().len(), buffer.samples().len());
    
    // Verify conversion accuracy (with some tolerance for quantization)
    for (i, &original) in buffer.samples().iter().enumerate() {
        let converted = back_to_f32.samples()[i];
        assert!((original - converted).abs() < 0.01);
    }
}

#[test]
fn test_audio_normalization() {
    let mut samples = vec![0.1, -0.3, 0.7, -0.9, 0.2];
    let mut buffer = AudioBuffer::new(samples.clone(), 44100, 1).unwrap();
    
    buffer.normalize();
    
    let normalized = buffer.samples();
    
    // Find peak value
    let peak = normalized.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    assert!((peak - 1.0).abs() < 0.001);
    
    // Verify relative proportions are maintained
    let original_peak = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    let scale_factor = 1.0 / original_peak;
    
    for (i, &original) in samples.iter().enumerate() {
        let expected = original * scale_factor;
        assert!((normalized[i] - expected).abs() < 0.001);
    }
}

#[test]
fn test_gain_application() {
    let original_samples = vec![0.1, -0.2, 0.3, -0.4];
    let mut buffer = AudioBuffer::new(original_samples.clone(), 44100, 1).unwrap();
    let gain = 2.0;
    
    buffer.apply_gain(gain);
    
    let gained = buffer.samples();
    
    for (i, &original) in original_samples.iter().enumerate() {
        let expected = original * gain;
        assert!((gained[i] - expected).abs() < 0.001);
    }
}

#[test]
fn test_dc_offset_removal() {
    let dc_offset = 0.3;
    let samples = vec![0.1 + dc_offset, -0.2 + dc_offset, 0.3 + dc_offset, -0.4 + dc_offset];
    let mut buffer = AudioBuffer::new(samples, 44100, 1).unwrap();
    
    buffer.remove_dc_offset();
    
    let corrected = buffer.samples();
    
    // Calculate mean (should be close to zero)
    let mean = corrected.iter().sum::<f32>() / corrected.len() as f32;
    assert!(mean.abs() < 0.001);
}

#[test]
fn test_audio_concatenation() {
    let buffer1 = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
    let buffer2 = AudioBuffer::sine_wave(880.0, 0.5, 44100, 0.5);
    
    let concatenated = buffer1.concatenate(&buffer2).unwrap();
    
    assert_eq!(concatenated.sample_rate(), buffer1.sample_rate());
    assert_eq!(concatenated.channels(), buffer1.channels());
    assert_eq!(concatenated.samples().len(), buffer1.samples().len() + buffer2.samples().len());
    assert!((concatenated.duration() - (buffer1.duration() + buffer2.duration())).abs() < 0.001);
    
    // Verify samples are correctly concatenated
    let samples = concatenated.samples();
    let split_point = buffer1.samples().len();
    
    for (i, &sample) in buffer1.samples().iter().enumerate() {
        assert_eq!(samples[i], sample);
    }
    
    for (i, &sample) in buffer2.samples().iter().enumerate() {
        assert_eq!(samples[split_point + i], sample);
    }
}

#[test]
fn test_audio_chunking() {
    let duration = 2.0;
    let chunk_duration = 0.5;
    let buffer = AudioBuffer::sine_wave(440.0, duration, 44100, 0.5);
    
    let chunks = buffer.into_chunks(chunk_duration).unwrap();
    
    assert_eq!(chunks.len(), 4); // 2.0 / 0.5 = 4 chunks
    
    for chunk in &chunks {
        assert_eq!(chunk.sample_rate(), buffer.sample_rate());
        assert_eq!(chunk.channels(), buffer.channels());
        assert!((chunk.duration() - chunk_duration).abs() < 0.01);
    }
    
    // Verify total duration is preserved
    let total_duration: f32 = chunks.iter().map(|c| c.duration()).sum();
    assert!((total_duration - duration).abs() < 0.01);
}

#[test]
fn test_audio_mixing() {
    let buffer1 = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.3);
    let buffer2 = AudioBuffer::sine_wave(880.0, 1.0, 44100, 0.3);
    
    let mixed = buffer1.mix(&buffer2, 0.5).unwrap();
    
    assert_eq!(mixed.sample_rate(), buffer1.sample_rate());
    assert_eq!(mixed.channels(), buffer1.channels());
    assert_eq!(mixed.samples().len(), buffer1.samples().len());
    
    // Verify mixing formula
    let samples = mixed.samples();
    for (i, &sample) in samples.iter().enumerate() {
        let expected = buffer1.samples()[i] * 0.5 + buffer2.samples()[i] * 0.5;
        assert!((sample - expected).abs() < 0.001);
    }
}

#[test]
fn test_silence_trimming() {
    let mut samples = vec![0.0; 100]; // 100 samples of silence
    
    // Add some signal in the middle
    for i in 40..60 {
        samples[i] = 0.5 * (i as f32 / 10.0).sin();
    }
    
    let buffer = AudioBuffer::new(samples, 44100, 1).unwrap();
    let trimmed = buffer.trim_silence(0.01).unwrap();
    
    // Should trim leading and trailing silence
    assert!(trimmed.samples().len() < buffer.samples().len());
    assert!(trimmed.samples().len() >= 20); // At least the signal portion
    
    // Verify no leading/trailing silence
    let samples = trimmed.samples();
    assert!(samples[0].abs() > 0.01 || samples[1].abs() > 0.01);
    assert!(samples[samples.len()-1].abs() > 0.01 || samples[samples.len()-2].abs() > 0.01);
}

#[test]
fn test_peak_and_rms_calculations() {
    let samples = vec![0.1, -0.5, 0.8, -0.3, 0.2];
    let buffer = AudioBuffer::new(samples.clone(), 44100, 1).unwrap();
    
    let peak = buffer.peak();
    let rms = buffer.rms();
    
    // Peak should be the maximum absolute value
    let expected_peak = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    assert!((peak - expected_peak).abs() < 0.001);
    
    // RMS calculation
    let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
    let expected_rms = (sum_squares / samples.len() as f32).sqrt();
    assert!((rms - expected_rms).abs() < 0.001);
}

#[test]
fn test_audio_buffer_clone() {
    let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
    let cloned = buffer.clone();
    
    assert_eq!(buffer.samples(), cloned.samples());
    assert_eq!(buffer.sample_rate(), cloned.sample_rate());
    assert_eq!(buffer.channels(), cloned.channels());
    assert_eq!(buffer.duration(), cloned.duration());
}

#[test]
fn test_audio_buffer_debug() {
    let buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5);
    let debug_str = format!("{:?}", buffer);
    
    assert!(debug_str.contains("AudioBuffer"));
    assert!(debug_str.contains("44100"));
    assert!(debug_str.contains("1"));
}

#[test]
fn test_invalid_operations() {
    let buffer = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
    
    // Test invalid resample rate
    assert!(buffer.resample(0).is_err());
    
    // Test concatenation with different sample rates
    let different_rate = AudioBuffer::sine_wave(440.0, 0.5, 48000, 0.5);
    assert!(buffer.concatenate(&different_rate).is_err());
    
    // Test concatenation with different channel counts
    let different_channels = AudioBuffer::silence(0.5, 44100, 2).unwrap();
    assert!(buffer.concatenate(&different_channels).is_err());
    
    // Test invalid chunk duration
    assert!(buffer.into_chunks(0.0).is_err());
    assert!(buffer.into_chunks(-1.0).is_err());
}

#[test]
fn test_edge_cases() {
    // Test very short buffer
    let short_buffer = AudioBuffer::sine_wave(440.0, 0.001, 44100, 0.5);
    assert!(short_buffer.samples().len() > 0);
    
    // Test very low amplitude
    let quiet_buffer = AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.001);
    assert!(quiet_buffer.peak() < 0.002);
    
    // Test high frequency
    let high_freq = AudioBuffer::sine_wave(20000.0, 0.1, 44100, 0.5);
    assert!(high_freq.samples().len() > 0);
}