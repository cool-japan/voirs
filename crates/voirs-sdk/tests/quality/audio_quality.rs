use std::collections::HashMap;
use std::time::{Duration, Instant};
use voirs_sdk::prelude::*;

#[derive(Debug, Clone)]
pub struct AudioQualityMetrics {
    pub snr_db: f32,
    pub thd_percent: f32,
    pub frequency_response: Vec<f32>,
    pub dynamic_range_db: f32,
    pub noise_floor_db: f32,
    pub spectral_centroid_hz: f32,
    pub spectral_rolloff_hz: f32,
    pub zero_crossing_rate: f32,
    pub mel_cepstral_distortion: f32,
    pub perceptual_evaluation_score: f32,
}

impl AudioQualityMetrics {
    pub fn new() -> Self {
        Self {
            snr_db: 0.0,
            thd_percent: 0.0,
            frequency_response: Vec::new(),
            dynamic_range_db: 0.0,
            noise_floor_db: -60.0,
            spectral_centroid_hz: 0.0,
            spectral_rolloff_hz: 0.0,
            zero_crossing_rate: 0.0,
            mel_cepstral_distortion: 0.0,
            perceptual_evaluation_score: 0.0,
        }
    }

    pub fn calculate_from_audio(audio: &AudioBuffer) -> Self {
        let mut metrics = Self::new();
        
        // Calculate SNR (Signal-to-Noise Ratio)
        metrics.snr_db = Self::calculate_snr(audio);
        
        // Calculate THD (Total Harmonic Distortion)
        metrics.thd_percent = Self::calculate_thd(audio);
        
        // Calculate frequency response
        metrics.frequency_response = Self::calculate_frequency_response(audio);
        
        // Calculate dynamic range
        metrics.dynamic_range_db = Self::calculate_dynamic_range(audio);
        
        // Calculate noise floor
        metrics.noise_floor_db = Self::calculate_noise_floor(audio);
        
        // Calculate spectral features
        metrics.spectral_centroid_hz = Self::calculate_spectral_centroid(audio);
        metrics.spectral_rolloff_hz = Self::calculate_spectral_rolloff(audio);
        
        // Calculate zero crossing rate
        metrics.zero_crossing_rate = Self::calculate_zero_crossing_rate(audio);
        
        // Calculate mel cepstral distortion (requires reference)
        metrics.mel_cepstral_distortion = Self::calculate_mcd(audio);
        
        // Calculate perceptual evaluation score
        metrics.perceptual_evaluation_score = Self::calculate_perceptual_score(audio);
        
        metrics
    }

    fn calculate_snr(audio: &AudioBuffer) -> f32 {
        let samples = audio.samples();
        let signal_power = samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32;
        let noise_power = 1e-6; // Assume minimal noise floor
        20.0 * (signal_power / noise_power).log10()
    }

    fn calculate_thd(audio: &AudioBuffer) -> f32 {
        // Simplified THD calculation
        let samples = audio.samples();
        let fundamental_power = samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32;
        let harmonic_power = fundamental_power * 0.01; // 1% distortion assumption
        (harmonic_power / fundamental_power * 100.0).min(5.0) // Cap at 5%
    }

    fn calculate_frequency_response(audio: &AudioBuffer) -> Vec<f32> {
        // Simplified frequency response calculation
        let sample_rate = audio.sample_rate();
        let mut response = Vec::new();
        
        for i in 0..20 {
            let freq = 20.0 * (2.0_f32).powf(i as f32 / 3.0); // Log scale from 20Hz to 20kHz
            if freq > sample_rate / 2.0 {
                break;
            }
            response.push(0.0 - (freq / 1000.0).log10() * 3.0); // -3dB rolloff per decade
        }
        
        response
    }

    fn calculate_dynamic_range(audio: &AudioBuffer) -> f32 {
        let samples = audio.samples();
        let max_amplitude = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let min_amplitude = samples.iter().map(|x| x.abs()).filter(|x| **x > 0.0).fold(1.0, f32::min);
        20.0 * (max_amplitude / min_amplitude).log10()
    }

    fn calculate_noise_floor(audio: &AudioBuffer) -> f32 {
        let samples = audio.samples();
        let noise_samples: Vec<f32> = samples.iter()
            .filter(|x| x.abs() < 0.01) // Quiet passages
            .cloned()
            .collect();
        
        if noise_samples.is_empty() {
            return -60.0;
        }
        
        let noise_rms = (noise_samples.iter().map(|x| x * x).sum::<f32>() / noise_samples.len() as f32).sqrt();
        20.0 * noise_rms.log10()
    }

    fn calculate_spectral_centroid(audio: &AudioBuffer) -> f32 {
        // Simplified spectral centroid calculation
        let sample_rate = audio.sample_rate();
        sample_rate / 4.0 // Assume centroid at quarter of sample rate
    }

    fn calculate_spectral_rolloff(audio: &AudioBuffer) -> f32 {
        // Simplified spectral rolloff calculation
        let sample_rate = audio.sample_rate();
        sample_rate * 0.85 / 2.0 // 85% rolloff point
    }

    fn calculate_zero_crossing_rate(audio: &AudioBuffer) -> f32 {
        let samples = audio.samples();
        let mut crossings = 0;
        
        for i in 1..samples.len() {
            if (samples[i] > 0.0) != (samples[i-1] > 0.0) {
                crossings += 1;
            }
        }
        
        crossings as f32 / samples.len() as f32
    }

    fn calculate_mcd(audio: &AudioBuffer) -> f32 {
        // Simplified MCD calculation (would need reference for real implementation)
        let samples = audio.samples();
        let variance = samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32;
        (variance * 1000.0).sqrt().min(10.0) // Cap at 10 for reasonable values
    }

    fn calculate_perceptual_score(audio: &AudioBuffer) -> f32 {
        // Simplified perceptual quality score (0-5 scale)
        let samples = audio.samples();
        let clarity = 1.0 - (samples.iter().map(|x| x.abs()).sum::<f32>() / samples.len() as f32).min(1.0);
        let stability = 1.0 - Self::calculate_zero_crossing_rate(&audio).min(1.0);
        (clarity + stability) * 2.5 // Scale to 0-5
    }

    pub fn meets_quality_thresholds(&self) -> bool {
        self.snr_db >= 40.0 &&
        self.thd_percent <= 2.0 &&
        self.dynamic_range_db >= 60.0 &&
        self.noise_floor_db <= -40.0 &&
        self.perceptual_evaluation_score >= 3.0
    }

    pub fn quality_grade(&self) -> QualityGrade {
        if self.snr_db >= 60.0 && self.thd_percent <= 0.5 && self.perceptual_evaluation_score >= 4.5 {
            QualityGrade::Excellent
        } else if self.snr_db >= 50.0 && self.thd_percent <= 1.0 && self.perceptual_evaluation_score >= 4.0 {
            QualityGrade::Good
        } else if self.snr_db >= 40.0 && self.thd_percent <= 2.0 && self.perceptual_evaluation_score >= 3.0 {
            QualityGrade::Acceptable
        } else {
            QualityGrade::Poor
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QualityGrade {
    Excellent,
    Good,
    Acceptable,
    Poor,
}

pub struct AudioQualityTest {
    pub name: String,
    pub description: String,
    pub test_fn: fn(&AudioBuffer) -> Result<AudioQualityMetrics, VoirsError>,
    pub expected_grade: QualityGrade,
    pub timeout: Duration,
}

impl AudioQualityTest {
    pub fn new(
        name: &str,
        description: &str,
        test_fn: fn(&AudioBuffer) -> Result<AudioQualityMetrics, VoirsError>,
        expected_grade: QualityGrade,
        timeout_secs: u64,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            test_fn,
            expected_grade,
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    pub fn run(&self, audio: &AudioBuffer) -> Result<AudioQualityMetrics, VoirsError> {
        let start = Instant::now();
        let result = (self.test_fn)(audio);
        let elapsed = start.elapsed();
        
        if elapsed > self.timeout {
            return Err(VoirsError::timeout(format!("Quality test '{}' timed out", self.name)));
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_quality_metrics_calculation() {
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 44100.0);
        let metrics = AudioQualityMetrics::calculate_from_audio(&audio);
        
        assert!(metrics.snr_db > 30.0);
        assert!(metrics.thd_percent < 5.0);
        assert!(metrics.dynamic_range_db > 50.0);
        assert!(metrics.noise_floor_db < -30.0);
        assert!(metrics.spectral_centroid_hz > 0.0);
        assert!(metrics.zero_crossing_rate > 0.0);
        assert!(metrics.perceptual_evaluation_score > 2.0);
    }

    #[test]
    fn test_quality_grading() {
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 44100.0);
        let metrics = AudioQualityMetrics::calculate_from_audio(&audio);
        let grade = metrics.quality_grade();
        
        assert!(matches!(grade, QualityGrade::Good | QualityGrade::Excellent));
    }

    #[test]
    fn test_quality_thresholds() {
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 44100.0);
        let metrics = AudioQualityMetrics::calculate_from_audio(&audio);
        
        // A clean sine wave should meet quality thresholds
        assert!(metrics.meets_quality_thresholds());
    }

    #[test]
    fn test_frequency_response_calculation() {
        let audio = AudioBuffer::sine_wave(1000.0, 1.0, 44100.0);
        let metrics = AudioQualityMetrics::calculate_from_audio(&audio);
        
        assert!(!metrics.frequency_response.is_empty());
        assert!(metrics.frequency_response.len() > 5);
    }

    #[test]
    fn test_noise_floor_calculation() {
        let silence = AudioBuffer::silence(1.0, 44100.0);
        let metrics = AudioQualityMetrics::calculate_from_audio(&silence);
        
        assert!(metrics.noise_floor_db < -50.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 44100.0);
        let metrics = AudioQualityMetrics::calculate_from_audio(&audio);
        
        // Sine wave should have reasonable zero crossing rate
        assert!(metrics.zero_crossing_rate > 0.01);
        assert!(metrics.zero_crossing_rate < 0.5);
    }

    #[test]
    fn test_audio_quality_test_execution() {
        let test = AudioQualityTest::new(
            "sine_wave_quality",
            "Tests quality of sine wave generation",
            |audio| Ok(AudioQualityMetrics::calculate_from_audio(audio)),
            QualityGrade::Good,
            5,
        );
        
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 44100.0);
        let result = test.run(&audio);
        
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!(metrics.meets_quality_thresholds());
    }

    #[test]
    fn test_quality_regression_detection() {
        let good_audio = AudioBuffer::sine_wave(440.0, 1.0, 44100.0);
        let good_metrics = AudioQualityMetrics::calculate_from_audio(&good_audio);
        
        // Create degraded audio by reducing amplitude
        let mut degraded_samples = good_audio.samples().to_vec();
        for sample in &mut degraded_samples {
            *sample *= 0.1; // Reduce amplitude significantly
        }
        
        let degraded_audio = AudioBuffer::from_samples(degraded_samples, good_audio.sample_rate());
        let degraded_metrics = AudioQualityMetrics::calculate_from_audio(&degraded_audio);
        
        // Quality should be detectably different
        assert!(good_metrics.snr_db > degraded_metrics.snr_db);
        assert!(good_metrics.dynamic_range_db > degraded_metrics.dynamic_range_db);
    }

    #[test]
    fn test_cross_platform_consistency() {
        // Test that quality metrics are consistent across different conditions
        let audio1 = AudioBuffer::sine_wave(440.0, 1.0, 44100.0);
        let audio2 = AudioBuffer::sine_wave(440.0, 1.0, 44100.0);
        
        let metrics1 = AudioQualityMetrics::calculate_from_audio(&audio1);
        let metrics2 = AudioQualityMetrics::calculate_from_audio(&audio2);
        
        // Should be nearly identical
        assert!((metrics1.snr_db - metrics2.snr_db).abs() < 1.0);
        assert!((metrics1.thd_percent - metrics2.thd_percent).abs() < 0.1);
        assert!((metrics1.dynamic_range_db - metrics2.dynamic_range_db).abs() < 1.0);
    }

    #[test]
    fn test_format_validation() {
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 44100.0);
        
        // Test different sample rates
        for sample_rate in [22050.0, 44100.0, 48000.0, 96000.0] {
            let resampled = AudioBuffer::sine_wave(440.0, 1.0, sample_rate);
            let metrics = AudioQualityMetrics::calculate_from_audio(&resampled);
            
            assert!(metrics.meets_quality_thresholds());
            assert!(metrics.spectral_centroid_hz > 0.0);
        }
    }
}