//! Perceptual audio quality tests
//!
//! Tests perceptual quality metrics like PESQ, STOI, and MOS prediction
//! to ensure vocoder output meets quality standards.

use voirs_vocoder::{AudioBuffer, MelSpectrogram, DummyVocoder, Vocoder, SynthesisConfig};
use std::sync::Arc;

/// Perceptual quality metrics
#[derive(Debug, Clone)]
pub struct PerceptualMetrics {
    /// PESQ score (1.0 - 5.0, higher is better)
    pub pesq_score: f32,
    
    /// STOI score (0.0 - 1.0, higher is better)
    pub stoi_score: f32,
    
    /// Predicted MOS (1.0 - 5.0, higher is better)
    pub predicted_mos: f32,
    
    /// Loudness (LUFS)
    pub lufs: f32,
    
    /// Peak loudness (LUFS)
    pub peak_lufs: f32,
    
    /// Spectral centroid (Hz)
    pub spectral_centroid: f32,
    
    /// Spectral rolloff (Hz)
    pub spectral_rolloff: f32,
    
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
}

/// Quality assessment engine
pub struct QualityAssessor {
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
}

impl QualityAssessor {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            frame_size: 2048,
            hop_size: 512,
        }
    }
    
    /// Calculate perceptual quality metrics for audio
    pub fn assess_quality(&self, audio: &AudioBuffer, reference: Option<&AudioBuffer>) -> PerceptualMetrics {
        let samples = audio.samples();
        
        // Calculate basic metrics
        let lufs = self.calculate_lufs(samples);
        let peak_lufs = self.calculate_peak_lufs(samples);
        let spectral_centroid = self.calculate_spectral_centroid(samples);
        let spectral_rolloff = self.calculate_spectral_rolloff(samples);
        let zero_crossing_rate = self.calculate_zero_crossing_rate(samples);
        
        // Calculate PESQ and STOI if reference is available
        let (pesq_score, stoi_score) = if let Some(ref_audio) = reference {
            let pesq = self.calculate_pesq(samples, ref_audio.samples());
            let stoi = self.calculate_stoi(samples, ref_audio.samples());
            (pesq, stoi)
        } else {
            // Estimate quality without reference
            let estimated_pesq = self.estimate_pesq_no_reference(samples);
            let estimated_stoi = self.estimate_stoi_no_reference(samples);
            (estimated_pesq, estimated_stoi)
        };
        
        // Predict MOS based on available metrics
        let predicted_mos = self.predict_mos(pesq_score, stoi_score, &audio);
        
        PerceptualMetrics {
            pesq_score,
            stoi_score,
            predicted_mos,
            lufs,
            peak_lufs,
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate,
        }
    }
    
    /// Calculate LUFS (Loudness Units Full Scale)
    fn calculate_lufs(&self, samples: &[f32]) -> f32 {
        // Simplified LUFS calculation
        // Real implementation would use proper K-weighting filter
        
        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        
        // Convert to LUFS (approximate)
        if rms > 0.0 {
            -23.0 + 20.0 * rms.log10()
        } else {
            -80.0 // Very quiet
        }
    }
    
    /// Calculate peak LUFS
    fn calculate_peak_lufs(&self, samples: &[f32]) -> f32 {
        // Calculate short-term peak loudness
        let window_size = self.sample_rate as usize / 10; // 100ms windows
        let mut max_lufs = -80.0;
        
        for chunk in samples.chunks(window_size) {
            let chunk_lufs = self.calculate_lufs(chunk);
            max_lufs = max_lufs.max(chunk_lufs);
        }
        
        max_lufs
    }
    
    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, samples: &[f32]) -> f32 {
        use scirs2_fft::FftPlanner;
        use scirs2_core::Complex;
        
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.frame_size);
        
        let mut total_centroid = 0.0;
        let mut frame_count = 0;
        
        for chunk in samples.chunks(self.hop_size) {
            if chunk.len() < self.frame_size {
                break;
            }
            
            // Prepare FFT input
            let mut buffer: Vec<Complex<f32>> = chunk[..self.frame_size]
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect();
            
            // Apply window
            for (i, sample) in buffer.iter_mut().enumerate() {
                let window = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / self.frame_size as f32).cos();
                *sample *= window;
            }
            
            fft.process(&mut buffer);
            
            // Calculate magnitude spectrum
            let magnitudes: Vec<f32> = buffer.iter()
                .take(self.frame_size / 2)
                .map(|c| c.norm())
                .collect();
            
            // Calculate centroid
            let mut weighted_sum = 0.0;
            let mut magnitude_sum = 0.0;
            
            for (i, &magnitude) in magnitudes.iter().enumerate() {
                let frequency = i as f32 * self.sample_rate as f32 / self.frame_size as f32;
                weighted_sum += frequency * magnitude;
                magnitude_sum += magnitude;
            }
            
            if magnitude_sum > 0.0 {
                total_centroid += weighted_sum / magnitude_sum;
                frame_count += 1;
            }
        }
        
        if frame_count > 0 {
            total_centroid / frame_count as f32
        } else {
            self.sample_rate as f32 / 4.0 // Default to quarter of sample rate
        }
    }
    
    /// Calculate spectral rolloff
    fn calculate_spectral_rolloff(&self, samples: &[f32]) -> f32 {
        use scirs2_fft::FftPlanner;
        use scirs2_core::Complex;
        
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.frame_size);
        
        let mut total_rolloff = 0.0;
        let mut frame_count = 0;
        
        for chunk in samples.chunks(self.hop_size) {
            if chunk.len() < self.frame_size {
                break;
            }
            
            // Prepare FFT input
            let mut buffer: Vec<Complex<f32>> = chunk[..self.frame_size]
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect();
            
            fft.process(&mut buffer);
            
            // Calculate magnitude spectrum
            let magnitudes: Vec<f32> = buffer.iter()
                .take(self.frame_size / 2)
                .map(|c| c.norm())
                .collect();
            
            // Calculate rolloff (frequency below which 85% of energy is contained)
            let total_energy: f32 = magnitudes.iter().map(|x| x * x).sum();
            let threshold = total_energy * 0.85;
            
            let mut cumulative_energy = 0.0;
            let mut rolloff_bin = magnitudes.len() - 1;
            
            for (i, &magnitude) in magnitudes.iter().enumerate() {
                cumulative_energy += magnitude * magnitude;
                if cumulative_energy >= threshold {
                    rolloff_bin = i;
                    break;
                }
            }
            
            let rolloff_freq = rolloff_bin as f32 * self.sample_rate as f32 / self.frame_size as f32;
            total_rolloff += rolloff_freq;
            frame_count += 1;
        }
        
        if frame_count > 0 {
            total_rolloff / frame_count as f32
        } else {
            self.sample_rate as f32 / 2.0 // Nyquist frequency
        }
    }
    
    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        
        let mut zero_crossings = 0;
        
        for i in 1..samples.len() {
            if (samples[i-1] >= 0.0) != (samples[i] >= 0.0) {
                zero_crossings += 1;
            }
        }
        
        zero_crossings as f32 / (samples.len() - 1) as f32
    }
    
    /// Calculate PESQ score (simplified implementation)
    fn calculate_pesq(&self, degraded: &[f32], reference: &[f32]) -> f32 {
        // Simplified PESQ implementation
        // Real PESQ requires complex perceptual modeling
        
        let min_len = degraded.len().min(reference.len());
        if min_len == 0 {
            return 1.0; // Minimum PESQ score
        }
        
        // Calculate mean squared error
        let mse: f32 = degraded[..min_len].iter()
            .zip(&reference[..min_len])
            .map(|(&d, &r)| (d - r).powi(2))
            .sum::<f32>() / min_len as f32;
        
        // Convert MSE to PESQ-like score (rough approximation)
        if mse > 0.0 {
            (4.5 - 2.0 * mse.log10()).clamp(1.0, 4.5)
        } else {
            4.5 // Perfect score
        }
    }
    
    /// Calculate STOI score (simplified implementation)
    fn calculate_stoi(&self, degraded: &[f32], reference: &[f32]) -> f32 {
        // Simplified STOI implementation
        // Real STOI requires short-time analysis and correlation computation
        
        let min_len = degraded.len().min(reference.len());
        if min_len == 0 {
            return 0.0;
        }
        
        // Calculate correlation coefficient
        let deg_mean = degraded[..min_len].iter().sum::<f32>() / min_len as f32;
        let ref_mean = reference[..min_len].iter().sum::<f32>() / min_len as f32;
        
        let mut numerator = 0.0;
        let mut deg_var = 0.0;
        let mut ref_var = 0.0;
        
        for i in 0..min_len {
            let deg_diff = degraded[i] - deg_mean;
            let ref_diff = reference[i] - ref_mean;
            
            numerator += deg_diff * ref_diff;
            deg_var += deg_diff * deg_diff;
            ref_var += ref_diff * ref_diff;
        }
        
        let correlation = if deg_var > 0.0 && ref_var > 0.0 {
            numerator / (deg_var * ref_var).sqrt()
        } else {
            0.0
        };
        
        correlation.abs().clamp(0.0, 1.0)
    }
    
    /// Estimate PESQ without reference (no-reference quality assessment)
    fn estimate_pesq_no_reference(&self, samples: &[f32]) -> f32 {
        // Estimate quality based on signal characteristics
        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let peak = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);
        
        let crest_factor = if rms > 0.0 { peak / rms } else { 1.0 };
        let zcr = self.calculate_zero_crossing_rate(samples);
        
        // Estimate based on signal properties
        let mut score = 3.0; // Base score
        
        // Penalize for clipping
        if peak >= 0.99 {
            score -= 1.0;
        }
        
        // Penalize for very low or very high crest factor
        if crest_factor < 1.2 || crest_factor > 8.0 {
            score -= 0.5;
        }
        
        // Penalize for unusual zero crossing rate
        if zcr < 0.01 || zcr > 0.5 {
            score -= 0.3;
        }
        
        score.clamp(1.0, 4.5)
    }
    
    /// Estimate STOI without reference
    fn estimate_stoi_no_reference(&self, samples: &[f32]) -> f32 {
        // Estimate intelligibility based on signal characteristics
        let zcr = self.calculate_zero_crossing_rate(samples);
        let spectral_centroid = self.calculate_spectral_centroid(samples);
        
        // Voice-like signals typically have:
        // - Moderate zero crossing rate (speech-like)
        // - Spectral centroid in speech range
        
        let mut score = 0.7; // Base score
        
        // Optimal ZCR for speech is around 0.1-0.3
        if zcr >= 0.1 && zcr <= 0.3 {
            score += 0.2;
        } else {
            score -= 0.1;
        }
        
        // Optimal spectral centroid for speech is around 1-3 kHz
        if spectral_centroid >= 1000.0 && spectral_centroid <= 3000.0 {
            score += 0.1;
        }
        
        score.clamp(0.0, 1.0)
    }
    
    /// Predict MOS score based on available metrics
    fn predict_mos(&self, pesq: f32, stoi: f32, audio: &AudioBuffer) -> f32 {
        // Simplified MOS prediction model
        // Real models would use machine learning with extensive training data
        
        let mut mos = 3.0; // Base MOS
        
        // Weight PESQ contribution (30%)
        mos += (pesq - 3.0) * 0.3;
        
        // Weight STOI contribution (20%)
        mos += (stoi - 0.7) * 2.0 * 0.2;
        
        // Weight audio characteristics (50%)
        let peak = audio.peak();
        let rms = audio.rms();
        
        // Penalize clipping
        if peak >= 0.99 {
            mos -= 1.0;
        }
        
        // Penalize very low or high levels
        if rms < 0.01 {
            mos -= 0.5; // Too quiet
        } else if rms > 0.8 {
            mos -= 0.3; // Too loud
        }
        
        // Check for artifacts
        let artifacts_score = self.detect_artifacts(audio);
        mos -= artifacts_score;
        
        mos.clamp(1.0, 5.0)
    }
    
    /// Detect audio artifacts
    fn detect_artifacts(&self, audio: &AudioBuffer) -> f32 {
        let samples = audio.samples();
        let mut artifact_score = 0.0;
        
        // Check for sudden level changes (clicks/pops)
        for window in samples.windows(2) {
            let diff = (window[1] - window[0]).abs();
            if diff > 0.1 {
                artifact_score += 0.01;
            }
        }
        
        // Check for DC offset
        let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;
        if dc_offset.abs() > 0.01 {
            artifact_score += 0.1;
        }
        
        // Check for silence
        let silent_samples = samples.iter().filter(|&&x| x.abs() < 0.001).count();
        let silence_ratio = silent_samples as f32 / samples.len() as f32;
        if silence_ratio > 0.5 {
            artifact_score += 0.2;
        }
        
        artifact_score.min(2.0) // Cap at 2.0 points deduction
    }
}

#[tokio::test]
async fn test_perceptual_quality_assessment() {
    let vocoder = Arc::new(DummyVocoder::new());
    let assessor = QualityAssessor::new(22050);
    
    // Generate test audio
    let mel_data = create_test_mel_data(100, 80);
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    let audio = vocoder.vocode(&mel, None).await.unwrap();
    
    // Assess quality
    let metrics = assessor.assess_quality(&audio, None);
    
    // Verify metrics are reasonable
    assert!(metrics.pesq_score >= 1.0 && metrics.pesq_score <= 5.0);
    assert!(metrics.stoi_score >= 0.0 && metrics.stoi_score <= 1.0);
    assert!(metrics.predicted_mos >= 1.0 && metrics.predicted_mos <= 5.0);
    
    // LUFS should be negative (typical for audio)
    assert!(metrics.lufs < 0.0);
    assert!(metrics.peak_lufs >= metrics.lufs);
    
    // Spectral features should be reasonable
    assert!(metrics.spectral_centroid > 0.0);
    assert!(metrics.spectral_centroid < 22050.0 / 2.0); // Below Nyquist
    assert!(metrics.spectral_rolloff > metrics.spectral_centroid);
    
    // Zero crossing rate should be reasonable for audio
    assert!(metrics.zero_crossing_rate >= 0.0);
    assert!(metrics.zero_crossing_rate <= 1.0);
}

#[tokio::test]
async fn test_quality_comparison_with_reference() {
    let vocoder = Arc::new(DummyVocoder::new());
    let assessor = QualityAssessor::new(22050);
    
    // Create reference audio (ground truth)
    let reference = AudioBuffer::sine_wave(440.0, 2.0, 22050, 0.5);
    
    // Create test audio from vocoder
    let mel_data = create_test_mel_data(100, 80);
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    let test_audio = vocoder.vocode(&mel, None).await.unwrap();
    
    // Assess quality with reference
    let metrics_with_ref = assessor.assess_quality(&test_audio, Some(&reference));
    let metrics_without_ref = assessor.assess_quality(&test_audio, None);
    
    // With reference should provide different (potentially more accurate) scores
    assert_ne!(metrics_with_ref.pesq_score, metrics_without_ref.pesq_score);
    assert_ne!(metrics_with_ref.stoi_score, metrics_without_ref.stoi_score);
    
    // Both should be valid
    assert!(metrics_with_ref.pesq_score >= 1.0);
    assert!(metrics_with_ref.stoi_score >= 0.0);
    assert!(metrics_with_ref.predicted_mos >= 1.0);
}

#[tokio::test]
async fn test_quality_degradation_detection() {
    let vocoder = Arc::new(DummyVocoder::new());
    let assessor = QualityAssessor::new(22050);
    
    // Generate clean audio
    let mel_data = create_test_mel_data(100, 80);
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    let clean_audio = vocoder.vocode(&mel, None).await.unwrap();
    
    // Create degraded version (add clipping)
    let mut degraded_samples = clean_audio.samples().clone();
    for sample in degraded_samples.iter_mut() {
        *sample = (*sample * 2.0).clamp(-1.0, 1.0); // Introduce clipping
    }
    let degraded_audio = AudioBuffer::new(degraded_samples, 22050, 1).unwrap();
    
    // Assess both versions
    let clean_metrics = assessor.assess_quality(&clean_audio, None);
    let degraded_metrics = assessor.assess_quality(&degraded_audio, None);
    
    // Degraded audio should have lower quality scores
    assert!(degraded_metrics.predicted_mos < clean_metrics.predicted_mos);
    assert!(degraded_metrics.pesq_score <= clean_metrics.pesq_score);
    
    // Degraded audio should show artifacts
    assert!(degraded_audio.peak() >= 0.99); // Should be clipped
}

#[tokio::test]
async fn test_quality_across_synthesis_configs() {
    let vocoder = Arc::new(DummyVocoder::new());
    let assessor = QualityAssessor::new(22050);
    
    let mel_data = create_test_mel_data(100, 80);
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    
    // Test different synthesis configurations
    let configs = vec![
        SynthesisConfig { speed: 1.0, pitch: 1.0, energy: 1.0, ..Default::default() },
        SynthesisConfig { speed: 1.2, pitch: 1.0, energy: 1.0, ..Default::default() },
        SynthesisConfig { speed: 1.0, pitch: 1.2, energy: 1.0, ..Default::default() },
        SynthesisConfig { speed: 1.0, pitch: 1.0, energy: 1.2, ..Default::default() },
    ];
    
    let mut quality_scores = Vec::new();
    
    for config in configs {
        let audio = vocoder.vocode(&mel, Some(&config)).await.unwrap();
        let metrics = assessor.assess_quality(&audio, None);
        quality_scores.push(metrics.predicted_mos);
    }
    
    // All configurations should produce reasonable quality
    for &score in &quality_scores {
        assert!(score >= 2.0 && score <= 5.0);
    }
    
    // Quality should be consistent across reasonable parameter variations
    let score_variance = calculate_variance(&quality_scores);
    assert!(score_variance < 1.0); // Should not vary too much
}

#[tokio::test]
async fn test_spectral_analysis_accuracy() {
    let assessor = QualityAssessor::new(44100);
    
    // Create pure tone at known frequency
    let frequency = 1000.0;
    let audio = AudioBuffer::sine_wave(frequency, 1.0, 44100, 0.5);
    
    let metrics = assessor.assess_quality(&audio, None);
    
    // Spectral centroid should be close to the sine wave frequency
    let centroid_error = (metrics.spectral_centroid - frequency).abs();
    assert!(centroid_error < 100.0); // Within 100 Hz
    
    // Zero crossing rate should match the frequency
    let expected_zcr = frequency / 44100.0 * 2.0; // Two crossings per cycle
    let zcr_error = (metrics.zero_crossing_rate - expected_zcr).abs();
    assert!(zcr_error < 0.01); // Within reasonable tolerance
}

// Helper functions

fn create_test_mel_data(frames: usize, mel_bins: usize) -> Vec<Vec<f32>> {
    let mut mel_data = Vec::new();
    
    for frame in 0..frames {
        let mut mel_frame = Vec::new();
        
        for bin in 0..mel_bins {
            // Create realistic mel spectrogram with harmonic structure
            let fundamental = 440.0;
            let frequency = fundamental * (1.0 + bin as f32 / mel_bins as f32);
            let amplitude = 1.0 / (1.0 + bin as f32 / 10.0); // Harmonic decay
            let phase = frame as f32 * frequency / 22050.0 * 2.0 * std::f32::consts::PI;
            
            let magnitude = amplitude * phase.sin().abs();
            let mel_value = -10.0 + 8.0 * magnitude; // Convert to mel scale
            
            mel_frame.push(mel_value);
        }
        
        mel_data.push(mel_frame);
    }
    
    mel_data
}

fn calculate_variance(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    variance
}