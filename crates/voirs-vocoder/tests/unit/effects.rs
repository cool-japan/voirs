//! Unit tests for audio effects system
//!
//! Tests audio effects processing, effect chains, and audio validation
//! to ensure high-quality audio enhancement capabilities.

use voirs_vocoder::{
    AudioBuffer, VocoderError,
    effects::{
        AudioEffect, EffectChain, EffectConfig,
        Compressor, NoiseGate, Limiter, AutomaticGainControl,
        ParametricEQ, PresenceEnhancer, DeEsser,
        SchroederReverb, StereoWidthControl, DelayEffect,
        AudioValidator, ClippingDetector, DCOffsetRemover
    }
};
use std::sync::Arc;

#[test]
fn test_compressor_creation() {
    let compressor = Compressor::new(4.0, 10.0, 50.0, 100.0);
    
    assert!(compressor.is_ok());
    let compressor = compressor.unwrap();
    
    assert_eq!(compressor.ratio(), 4.0);
    assert_eq!(compressor.threshold_db(), -10.0);
    assert_eq!(compressor.attack_ms(), 10.0);
    assert_eq!(compressor.release_ms(), 50.0);
    assert_eq!(compressor.knee_width_db(), 2.0); // Default
}

#[test]
fn test_compressor_audio_processing() {
    let mut compressor = Compressor::new(4.0, -10.0, 10.0, 50.0).unwrap();
    
    // Create test audio with varying levels
    let mut samples = Vec::new();
    for i in 0..44100 {
        let amplitude = if i < 22050 { 0.8 } else { 0.2 }; // High then low level
        samples.push(amplitude * (440.0 * 2.0 * std::f32::consts::PI * i as f32 / 44100.0).sin());
    }
    
    let mut audio = AudioBuffer::new(samples.clone(), 44100, 1).unwrap();
    let original_peak = audio.peak();
    
    compressor.process(&mut audio).unwrap();
    
    let compressed_peak = audio.peak();
    
    // Compressor should reduce peaks above threshold
    assert!(compressed_peak <= original_peak);
    
    // Should maintain relative dynamics but reduce overall dynamic range
    let original_rms = AudioBuffer::new(samples, 44100, 1).unwrap().rms();
    let compressed_rms = audio.rms();
    
    // RMS should be closer to peak (reduced dynamic range)
    let original_crest = original_peak / original_rms;
    let compressed_crest = compressed_peak / compressed_rms;
    assert!(compressed_crest < original_crest);
}

#[test]
fn test_noise_gate_creation() {
    let gate = NoiseGate::new(-40.0, 4.0, 10.0, 100.0, 500.0);
    
    assert!(gate.is_ok());
    let gate = gate.unwrap();
    
    assert_eq!(gate.threshold_db(), -40.0);
    assert_eq!(gate.ratio(), 4.0);
    assert_eq!(gate.attack_ms(), 10.0);
    assert_eq!(gate.release_ms(), 100.0);
    assert_eq!(gate.hold_ms(), 500.0);
}

#[test]
fn test_noise_gate_audio_processing() {
    let mut gate = NoiseGate::new(-40.0, 10.0, 1.0, 50.0, 100.0).unwrap();
    
    // Create test audio with quiet and loud sections
    let mut samples = Vec::new();
    for i in 0..44100 {
        let amplitude = if i % 8820 < 4410 { 0.001 } else { 0.5 }; // Alternating quiet/loud
        samples.push(amplitude * (440.0 * 2.0 * std::f32::consts::PI * i as f32 / 44100.0).sin());
    }
    
    let mut audio = AudioBuffer::new(samples, 44100, 1).unwrap();
    gate.process(&mut audio).unwrap();
    
    // Gate should significantly reduce quiet sections
    let processed_samples = audio.samples();
    
    // Find a quiet section (first quarter)
    let quiet_section = &processed_samples[0..11025];
    let quiet_rms = (quiet_section.iter().map(|x| x * x).sum::<f32>() / quiet_section.len() as f32).sqrt();
    
    // Find a loud section (second quarter)
    let loud_section = &processed_samples[11025..22050];
    let loud_rms = (loud_section.iter().map(|x| x * x).sum::<f32>() / loud_section.len() as f32).sqrt();
    
    // Gate should make quiet sections much quieter relative to loud sections
    assert!(loud_rms > quiet_rms * 10.0);
}

#[test]
fn test_limiter_creation() {
    let limiter = Limiter::new(-1.0, 1.0, 10.0);
    
    assert!(limiter.is_ok());
    let limiter = limiter.unwrap();
    
    assert_eq!(limiter.threshold_db(), -1.0);
    assert_eq!(limiter.release_ms(), 10.0);
    assert_eq!(limiter.lookahead_ms(), 5.0); // Default
}

#[test]
fn test_limiter_audio_processing() {
    let mut limiter = Limiter::new(-1.0, 1.0, 10.0).unwrap();
    
    // Create test audio that would clip
    let samples: Vec<f32> = (0..44100)
        .map(|i| 1.5 * (440.0 * 2.0 * std::f32::consts::PI * i as f32 / 44100.0).sin())
        .collect();
    
    let mut audio = AudioBuffer::new(samples, 44100, 1).unwrap();
    let original_peak = audio.peak();
    
    limiter.process(&mut audio).unwrap();
    
    let limited_peak = audio.peak();
    
    // Limiter should prevent clipping
    assert!(limited_peak <= 1.0);
    assert!(limited_peak < original_peak);
    
    // Should preserve loudness while preventing clipping
    let limited_rms = audio.rms();
    assert!(limited_rms > 0.5); // Should still be loud
}

#[test]
fn test_parametric_eq_creation() {
    let eq = ParametricEQ::new(1000.0, 3.0, 1.0, 44100);
    
    assert!(eq.is_ok());
    let eq = eq.unwrap();
    
    assert_eq!(eq.frequency(), 1000.0);
    assert_eq!(eq.gain_db(), 3.0);
    assert_eq!(eq.q_factor(), 1.0);
    assert_eq!(eq.sample_rate(), 44100);
}

#[test]
fn test_parametric_eq_frequency_response() {
    let mut eq = ParametricEQ::new(1000.0, 6.0, 2.0, 44100).unwrap();
    
    // Create test tones at different frequencies
    let test_frequencies = vec![100.0, 1000.0, 10000.0];
    let mut responses = Vec::new();
    
    for freq in test_frequencies {
        let samples: Vec<f32> = (0..44100)
            .map(|i| 0.5 * (freq * 2.0 * std::f32::consts::PI * i as f32 / 44100.0).sin())
            .collect();
        
        let mut audio = AudioBuffer::new(samples, 44100, 1).unwrap();
        let original_rms = audio.rms();
        
        eq.process(&mut audio).unwrap();
        
        let processed_rms = audio.rms();
        let gain = 20.0 * (processed_rms / original_rms).log10();
        responses.push(gain);
    }
    
    // EQ should boost 1000Hz the most
    assert!(responses[1] > responses[0]); // 1000Hz > 100Hz
    assert!(responses[1] > responses[2]); // 1000Hz > 10000Hz
    
    // 1000Hz should have approximately 6dB gain
    assert!((responses[1] - 6.0).abs() < 2.0);
}

#[test]
fn test_schroeder_reverb_creation() {
    let reverb = SchroederReverb::new(0.5, 2.0, 0.8, 44100);
    
    assert!(reverb.is_ok());
    let reverb = reverb.unwrap();
    
    assert_eq!(reverb.room_size(), 0.5);
    assert_eq!(reverb.decay_time(), 2.0);
    assert_eq!(reverb.wet_level(), 0.8);
    assert_eq!(reverb.sample_rate(), 44100);
}

#[test]
fn test_reverb_audio_processing() {
    let mut reverb = SchroederReverb::new(0.7, 1.5, 0.5, 44100).unwrap();
    
    // Create a short impulse
    let mut samples = vec![0.0; 44100];
    samples[0] = 1.0; // Single impulse
    
    let mut audio = AudioBuffer::new(samples, 44100, 1).unwrap();
    reverb.process(&mut audio).unwrap();
    
    let reverbed = audio.samples();
    
    // Reverb should extend the impulse response
    let tail_energy: f32 = reverbed[22050..].iter().map(|x| x * x).sum();
    assert!(tail_energy > 0.01); // Should have significant tail
    
    // Peak should be reduced due to wet/dry mix
    assert!(reverbed[0].abs() < 1.0);
    
    // Should not clip
    assert!(audio.peak() <= 1.0);
}

#[test]
fn test_effect_chain_creation() {
    let mut chain = EffectChain::new();
    
    // Add effects to chain
    let compressor = Box::new(Compressor::new(4.0, -12.0, 10.0, 50.0).unwrap());
    let eq = Box::new(ParametricEQ::new(1000.0, 3.0, 1.0, 44100).unwrap());
    let reverb = Box::new(SchroederReverb::new(0.5, 1.0, 0.3, 44100).unwrap());
    
    chain.add_effect(compressor);
    chain.add_effect(eq);
    chain.add_effect(reverb);
    
    assert_eq!(chain.effect_count(), 3);
    assert!(chain.is_enabled());
}

#[test]
fn test_effect_chain_processing() {
    let mut chain = EffectChain::new();
    
    // Create a simple chain: compressor -> EQ
    let compressor = Box::new(Compressor::new(4.0, -12.0, 10.0, 50.0).unwrap());
    let eq = Box::new(ParametricEQ::new(2000.0, 6.0, 2.0, 44100).unwrap());
    
    chain.add_effect(compressor);
    chain.add_effect(eq);
    
    // Create test audio
    let samples: Vec<f32> = (0..44100)
        .map(|i| 0.8 * (2000.0 * 2.0 * std::f32::consts::PI * i as f32 / 44100.0).sin())
        .collect();
    
    let mut audio = AudioBuffer::new(samples.clone(), 44100, 1).unwrap();
    let original_peak = audio.peak();
    let original_rms = audio.rms();
    
    chain.process(&mut audio).unwrap();
    
    let processed_peak = audio.peak();
    let processed_rms = audio.rms();
    
    // Chain should have modified the audio
    assert_ne!(processed_peak, original_peak);
    assert_ne!(processed_rms, original_rms);
    
    // Should not clip
    assert!(processed_peak <= 1.0);
}

#[test]
fn test_effect_chain_bypass() {
    let mut chain = EffectChain::new();
    
    let compressor = Box::new(Compressor::new(4.0, -12.0, 10.0, 50.0).unwrap());
    chain.add_effect(compressor);
    
    let samples = vec![0.8, -0.6, 0.9, -0.7]; // Some test samples
    let original_audio = AudioBuffer::new(samples.clone(), 44100, 1).unwrap();
    let mut test_audio = original_audio.clone();
    
    // Test with bypass enabled
    chain.set_bypass(true);
    chain.process(&mut test_audio).unwrap();
    
    // Audio should be unchanged when bypassed
    for (original, processed) in original_audio.samples().iter().zip(test_audio.samples().iter()) {
        assert_eq!(original, processed);
    }
    
    // Test with bypass disabled
    let mut test_audio2 = original_audio.clone();
    chain.set_bypass(false);
    chain.process(&mut test_audio2).unwrap();
    
    // Audio should be changed when not bypassed
    assert_ne!(original_audio.samples(), test_audio2.samples());
}

#[test]
fn test_audio_validator_creation() {
    let validator = AudioValidator::new();
    
    // Should create successfully
    assert!(true);
    
    // Test with custom settings
    let custom_validator = AudioValidator::with_thresholds(-1.0, 0.01, 0.05);
    assert!(true);
}

#[test]
fn test_clipping_detection() {
    let validator = AudioValidator::new();
    
    // Test non-clipping audio
    let normal_samples = vec![0.5, -0.3, 0.8, -0.9, 0.2];
    let normal_audio = AudioBuffer::new(normal_samples, 44100, 1).unwrap();
    
    let clipping_report = validator.detect_clipping(&normal_audio);
    assert!(!clipping_report.has_clipping);
    assert_eq!(clipping_report.clipped_samples, 0);
    
    // Test clipping audio
    let clipping_samples = vec![0.5, -1.0, 1.0, -0.9, 1.0]; // Contains ±1.0
    let clipping_audio = AudioBuffer::new(clipping_samples, 44100, 1).unwrap();
    
    let clipping_report = validator.detect_clipping(&clipping_audio);
    assert!(clipping_report.has_clipping);
    assert_eq!(clipping_report.clipped_samples, 3); // Three ±1.0 samples
    assert!((clipping_report.clipping_percentage - 60.0).abs() < 1.0); // 3/5 = 60%
}

#[test]
fn test_dc_offset_detection() {
    let validator = AudioValidator::new();
    
    // Test audio without DC offset
    let centered_samples = vec![0.1, -0.1, 0.2, -0.2, 0.0];
    let centered_audio = AudioBuffer::new(centered_samples, 44100, 1).unwrap();
    
    let dc_report = validator.detect_dc_offset(&centered_audio);
    assert!(!dc_report.has_dc_offset);
    assert!(dc_report.dc_level.abs() < 0.01);
    
    // Test audio with DC offset
    let offset = 0.1;
    let offset_samples = vec![0.1 + offset, -0.1 + offset, 0.2 + offset, -0.2 + offset, 0.0 + offset];
    let offset_audio = AudioBuffer::new(offset_samples, 44100, 1).unwrap();
    
    let dc_report = validator.detect_dc_offset(&offset_audio);
    assert!(dc_report.has_dc_offset);
    assert!((dc_report.dc_level - offset).abs() < 0.01);
}

#[test]
fn test_phase_coherency_check() {
    let validator = AudioValidator::new();
    
    // Test mono audio (should be coherent)
    let mono_samples = vec![0.1, -0.2, 0.3, -0.4];
    let mono_audio = AudioBuffer::new(mono_samples, 44100, 1).unwrap();
    
    let phase_report = validator.check_phase_coherency(&mono_audio);
    assert!(phase_report.is_coherent); // Mono is always coherent
    
    // Test stereo audio
    let stereo_samples = vec![0.1, 0.1, -0.2, -0.2, 0.3, 0.3, -0.4, -0.4]; // In-phase stereo
    let stereo_audio = AudioBuffer::new(stereo_samples, 44100, 2).unwrap();
    
    let phase_report = validator.check_phase_coherency(&stereo_audio);
    assert!(phase_report.is_coherent);
    assert!(phase_report.correlation > 0.9); // High correlation for in-phase
    
    // Test out-of-phase stereo
    let out_of_phase = vec![0.1, -0.1, -0.2, 0.2, 0.3, -0.3, -0.4, 0.4];
    let out_of_phase_audio = AudioBuffer::new(out_of_phase, 44100, 2).unwrap();
    
    let phase_report = validator.check_phase_coherency(&out_of_phase_audio);
    assert!(!phase_report.is_coherent);
    assert!(phase_report.correlation < -0.9); // High negative correlation
}

#[test]
fn test_quality_metrics_calculation() {
    let validator = AudioValidator::new();
    
    // Create test audio with known characteristics
    let samples: Vec<f32> = (0..44100)
        .map(|i| 0.5 * (440.0 * 2.0 * std::f32::consts::PI * i as f32 / 44100.0).sin())
        .collect();
    
    let audio = AudioBuffer::new(samples, 44100, 1).unwrap();
    let metrics = validator.calculate_quality_metrics(&audio);
    
    // Test basic metrics
    assert!(metrics.peak_level > 0.0);
    assert!(metrics.rms_level > 0.0);
    assert!(metrics.dynamic_range > 0.0);
    assert!(metrics.snr_db > 20.0); // Pure sine wave should have high SNR
    
    // Test THD+N calculation
    assert!(metrics.thd_n_percent < 5.0); // Pure sine should have low THD+N
    
    // Test loudness metrics
    assert!(metrics.lufs < 0.0); // LUFS is typically negative
    assert!(metrics.peak_lufs < metrics.lufs); // Peak should be higher (less negative)
}

#[test]
fn test_effect_parameter_validation() {
    // Test compressor parameter validation
    assert!(Compressor::new(0.0, -10.0, 10.0, 50.0).is_err()); // Invalid ratio
    assert!(Compressor::new(4.0, 10.0, 10.0, 50.0).is_err());  // Positive threshold
    assert!(Compressor::new(4.0, -10.0, 0.0, 50.0).is_err());  // Zero attack
    assert!(Compressor::new(4.0, -10.0, 10.0, 0.0).is_err());  // Zero release
    
    // Test EQ parameter validation
    assert!(ParametricEQ::new(0.0, 3.0, 1.0, 44100).is_err());    // Zero frequency
    assert!(ParametricEQ::new(1000.0, 3.0, 0.0, 44100).is_err()); // Zero Q
    assert!(ParametricEQ::new(1000.0, 3.0, 1.0, 0).is_err());     // Zero sample rate
    
    // Test reverb parameter validation
    assert!(SchroederReverb::new(-0.1, 2.0, 0.8, 44100).is_err()); // Negative room size
    assert!(SchroederReverb::new(0.5, 0.0, 0.8, 44100).is_err());  // Zero decay time
    assert!(SchroederReverb::new(0.5, 2.0, -0.1, 44100).is_err()); // Negative wet level
    assert!(SchroederReverb::new(0.5, 2.0, 0.8, 0).is_err());      // Zero sample rate
}

#[test]
fn test_effect_performance() {
    // Test that effects don't introduce excessive latency
    let mut compressor = Compressor::new(4.0, -12.0, 1.0, 10.0).unwrap(); // Fast attack/release
    
    let samples = vec![0.8; 1024]; // 1024 samples
    let mut audio = AudioBuffer::new(samples, 44100, 1).unwrap();
    
    let start = std::time::Instant::now();
    compressor.process(&mut audio).unwrap();
    let duration = start.elapsed();
    
    // Should process 1024 samples very quickly (< 1ms)
    assert!(duration.as_millis() < 1);
}

#[test]
fn test_effect_memory_usage() {
    // Test that effects don't use excessive memory
    let reverb = SchroederReverb::new(0.9, 5.0, 0.8, 44100).unwrap(); // Large reverb
    
    // Memory usage should be reasonable even for large reverbs
    let memory_estimate = reverb.estimated_memory_usage();
    assert!(memory_estimate < 10 * 1024 * 1024); // < 10MB
}

#[test]
fn test_effect_stability() {
    // Test that effects remain stable with extreme inputs
    let mut limiter = Limiter::new(-6.0, 1.0, 10.0).unwrap();
    
    // Test with very loud input
    let loud_samples = vec![10.0; 1024]; // Way above 0dBFS
    let mut loud_audio = AudioBuffer::new(loud_samples, 44100, 1).unwrap();
    
    limiter.process(&mut loud_audio).unwrap();
    
    // Should not produce NaN or infinite values
    for &sample in loud_audio.samples() {
        assert!(sample.is_finite());
        assert!(!sample.is_nan());
        assert!(sample.abs() <= 1.0); // Should be limited
    }
    
    // Test with very quiet input
    let quiet_samples = vec![0.0001; 1024];
    let mut quiet_audio = AudioBuffer::new(quiet_samples, 44100, 1).unwrap();
    
    limiter.process(&mut quiet_audio).unwrap();
    
    // Should still be finite and reasonable
    for &sample in quiet_audio.samples() {
        assert!(sample.is_finite());
        assert!(!sample.is_nan());
    }
}