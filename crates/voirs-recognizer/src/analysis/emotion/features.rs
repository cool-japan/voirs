//! Emotion feature extraction
//!
//! Extracts acoustic features relevant for emotion and sentiment recognition
//! including prosodic, spectral, and voice quality features.

use crate::RecognitionError;
use voirs_sdk::AudioBuffer;
use std::collections::HashMap;

/// Emotion feature extractor
pub struct EmotionFeatureExtractor {
    /// Sample rate for feature extraction
    sample_rate: u32,
    /// Frame size for analysis
    frame_size: usize,
    /// Hop size for overlapping frames
    hop_size: usize,
}

impl EmotionFeatureExtractor {
    /// Create new feature extractor
    pub fn new() -> Self {
        Self {
            sample_rate: 16000,
            frame_size: 1024,
            hop_size: 512,
        }
    }

    /// Extract comprehensive emotion features from audio
    pub async fn extract_emotion_features(&self, audio: &AudioBuffer) -> Result<HashMap<String, f32>, RecognitionError> {
        let samples = audio.samples();
        let mut features = HashMap::new();

        // Prosodic features
        let prosodic_features = self.extract_prosodic_features(samples)?;
        features.extend(prosodic_features);

        // Spectral features
        let spectral_features = self.extract_spectral_features(samples)?;
        features.extend(spectral_features);

        // Voice quality features
        let voice_quality_features = self.extract_voice_quality_features(samples)?;
        features.extend(voice_quality_features);

        // Temporal features
        let temporal_features = self.extract_temporal_features(samples)?;
        features.extend(temporal_features);

        Ok(features)
    }

    /// Extract prosodic features (pitch, intensity, rhythm)
    fn extract_prosodic_features(&self, samples: &[f32]) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut features = HashMap::new();

        // Fundamental frequency (F0) features
        let f0_contour = self.extract_f0_contour(samples)?;
        features.insert("f0_mean".to_string(), self.mean(&f0_contour));
        features.insert("f0_std".to_string(), self.std_dev(&f0_contour));
        features.insert("f0_min".to_string(), self.min(&f0_contour));
        features.insert("f0_max".to_string(), self.max(&f0_contour));
        features.insert("f0_range".to_string(), self.max(&f0_contour) - self.min(&f0_contour));

        // Pitch variance and trends
        features.insert("pitch_variance".to_string(), self.variance(&f0_contour));
        features.insert("pitch_slope".to_string(), self.linear_trend(&f0_contour));

        // Energy/intensity features
        let energy_contour = self.extract_energy_contour(samples)?;
        features.insert("energy_mean".to_string(), self.mean(&energy_contour));
        features.insert("energy_std".to_string(), self.std_dev(&energy_contour));
        features.insert("energy_variance".to_string(), self.variance(&energy_contour));
        features.insert("energy_range".to_string(), self.max(&energy_contour) - self.min(&energy_contour));

        // Speaking rate estimation
        features.insert("speaking_rate".to_string(), self.estimate_speaking_rate(samples)?);

        // Rhythm and timing features
        let pause_info = self.analyze_pauses(samples)?;
        features.insert("pause_frequency".to_string(), pause_info.0);
        features.insert("pause_duration_mean".to_string(), pause_info.1);

        Ok(features)
    }

    /// Extract spectral features
    fn extract_spectral_features(&self, samples: &[f32]) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut features = HashMap::new();

        // Spectral centroid
        let spectral_centroid = self.compute_spectral_centroid(samples)?;
        features.insert("spectral_centroid".to_string(), spectral_centroid);

        // Spectral rolloff
        let spectral_rolloff = self.compute_spectral_rolloff(samples)?;
        features.insert("spectral_rolloff".to_string(), spectral_rolloff);

        // Spectral bandwidth
        let spectral_bandwidth = self.compute_spectral_bandwidth(samples)?;
        features.insert("spectral_bandwidth".to_string(), spectral_bandwidth);

        // Zero crossing rate
        let zcr = self.compute_zero_crossing_rate(samples);
        features.insert("zero_crossing_rate".to_string(), zcr);

        // MFCC features (first 13 coefficients)
        let mfcc = self.compute_mfcc(samples)?;
        for (i, &coeff) in mfcc.iter().take(13).enumerate() {
            features.insert(format!("mfcc_{}", i), coeff);
        }

        // Formant frequencies
        let formants = self.extract_formants(samples)?;
        for (i, &formant) in formants.iter().take(3).enumerate() {
            features.insert(format!("formant_{}", i + 1), formant);
        }

        Ok(features)
    }

    /// Extract voice quality features
    fn extract_voice_quality_features(&self, samples: &[f32]) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut features = HashMap::new();

        // Jitter (pitch period irregularity)
        let jitter = self.compute_jitter(samples)?;
        features.insert("jitter".to_string(), jitter);

        // Shimmer (amplitude irregularity)
        let shimmer = self.compute_shimmer(samples)?;
        features.insert("shimmer".to_string(), shimmer);

        // Harmonic-to-noise ratio
        let hnr = self.compute_hnr(samples)?;
        features.insert("hnr".to_string(), hnr);

        // Breathiness measure
        let breathiness = self.compute_breathiness(samples)?;
        features.insert("breathiness".to_string(), breathiness);

        // Vocal fry detection
        let vocal_fry = self.detect_vocal_fry(samples)?;
        features.insert("vocal_fry".to_string(), vocal_fry);

        // Creakiness measure
        let creakiness = self.compute_creakiness(samples)?;
        features.insert("creakiness".to_string(), creakiness);

        Ok(features)
    }

    /// Extract temporal features
    fn extract_temporal_features(&self, samples: &[f32]) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut features = HashMap::new();

        // Total duration
        let duration = samples.len() as f32 / self.sample_rate as f32;
        features.insert("duration".to_string(), duration);

        // Speech/silence ratio
        let speech_ratio = self.compute_speech_ratio(samples)?;
        features.insert("speech_ratio".to_string(), speech_ratio);

        // Articulation rate
        let articulation_rate = self.compute_articulation_rate(samples)?;
        features.insert("articulation".to_string(), articulation_rate);

        // Tempo variation
        let tempo_variation = self.compute_tempo_variation(samples)?;
        features.insert("tempo_variation".to_string(), tempo_variation);

        Ok(features)
    }

    /// Extract F0 contour (simplified implementation)
    fn extract_f0_contour(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        let mut f0_values = Vec::new();
        let frame_count = (samples.len() - self.frame_size) / self.hop_size + 1;

        for i in 0..frame_count {
            let start = i * self.hop_size;
            let end = (start + self.frame_size).min(samples.len());
            let frame = &samples[start..end];

            // Simplified autocorrelation-based F0 estimation
            let f0 = self.estimate_f0_autocorr(frame);
            f0_values.push(f0);
        }

        Ok(f0_values)
    }

    /// Estimate F0 using autocorrelation (simplified)
    fn estimate_f0_autocorr(&self, frame: &[f32]) -> f32 {
        if frame.len() < 100 {
            return 0.0;
        }

        let mut max_corr = 0.0;
        let mut best_lag = 0;

        // Search for fundamental period
        for lag in 50..300 { // Typical F0 range: 50-400 Hz at 16kHz
            if lag >= frame.len() {
                break;
            }

            let mut corr = 0.0;
            for i in 0..(frame.len() - lag) {
                corr += frame[i] * frame[i + lag];
            }

            if corr > max_corr {
                max_corr = corr;
                best_lag = lag;
            }
        }

        if best_lag > 0 {
            self.sample_rate as f32 / best_lag as f32
        } else {
            0.0
        }
    }

    /// Extract energy contour
    fn extract_energy_contour(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        let mut energy_values = Vec::new();
        let frame_count = (samples.len() - self.frame_size) / self.hop_size + 1;

        for i in 0..frame_count {
            let start = i * self.hop_size;
            let end = (start + self.frame_size).min(samples.len());
            let frame = &samples[start..end];

            let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
            energy_values.push(energy.sqrt());
        }

        Ok(energy_values)
    }

    /// Estimate speaking rate (simplified)
    fn estimate_speaking_rate(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Count energy peaks as syllable approximation
        let energy_contour = self.extract_energy_contour(samples)?;
        let mean_energy = self.mean(&energy_contour);
        let threshold = mean_energy * 0.7;

        let mut peak_count = 0;
        let mut in_peak = false;

        for &energy in &energy_contour {
            if energy > threshold && !in_peak {
                peak_count += 1;
                in_peak = true;
            } else if energy <= threshold {
                in_peak = false;
            }
        }

        let duration = samples.len() as f32 / self.sample_rate as f32;
        Ok(peak_count as f32 / duration) // Syllables per second
    }

    /// Analyze pauses in speech
    fn analyze_pauses(&self, samples: &[f32]) -> Result<(f32, f32), RecognitionError> {
        let energy_contour = self.extract_energy_contour(samples)?;
        let mean_energy = self.mean(&energy_contour);
        let silence_threshold = mean_energy * 0.1;

        let mut pause_durations = Vec::new();
        let mut current_pause_length = 0;
        let frame_duration = self.hop_size as f32 / self.sample_rate as f32;

        for &energy in &energy_contour {
            if energy < silence_threshold {
                current_pause_length += 1;
            } else if current_pause_length > 0 {
                let pause_duration = current_pause_length as f32 * frame_duration;
                if pause_duration > 0.1 { // Minimum pause duration
                    pause_durations.push(pause_duration);
                }
                current_pause_length = 0;
            }
        }

        let duration = samples.len() as f32 / self.sample_rate as f32;
        let pause_frequency = pause_durations.len() as f32 / duration;
        let mean_pause_duration = if pause_durations.is_empty() {
            0.0
        } else {
            self.mean(&pause_durations)
        };

        Ok((pause_frequency, mean_pause_duration))
    }

    /// Compute spectral centroid (simplified)
    fn compute_spectral_centroid(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified implementation - would use FFT in practice
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += (i as f32) * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum)
        } else {
            Ok(0.0)
        }
    }

    /// Compute spectral rolloff (simplified)
    fn compute_spectral_rolloff(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified implementation
        let total_energy: f32 = samples.iter().map(|x| x * x).sum();
        let rolloff_threshold = total_energy * 0.85;

        let mut cumulative_energy = 0.0;
        for (i, &sample) in samples.iter().enumerate() {
            cumulative_energy += sample * sample;
            if cumulative_energy >= rolloff_threshold {
                return Ok(i as f32 / samples.len() as f32);
            }
        }

        Ok(1.0)
    }

    /// Compute spectral bandwidth (simplified)
    fn compute_spectral_bandwidth(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let centroid = self.compute_spectral_centroid(samples)?;
        let mut weighted_variance = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let magnitude = sample.abs();
            let freq_deviation = (i as f32) - centroid;
            weighted_variance += freq_deviation * freq_deviation * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok((weighted_variance / magnitude_sum).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Compute zero crossing rate
    fn compute_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i-1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / (samples.len() - 1) as f32
    }

    /// Compute MFCC features (simplified)
    fn compute_mfcc(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        // Simplified MFCC computation - would use proper mel filterbank in practice
        let mut mfcc = Vec::new();
        
        for i in 0..13 {
            let mut coeff = 0.0;
            for (j, &sample) in samples.iter().enumerate() {
                let cosine_arg = std::f32::consts::PI * (i + 1) as f32 * (j as f32 + 0.5) / samples.len() as f32;
                coeff += sample * cosine_arg.cos();
            }
            mfcc.push(coeff / samples.len() as f32);
        }
        
        Ok(mfcc)
    }

    /// Extract formant frequencies (simplified)
    fn extract_formants(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        // Simplified formant estimation - would use LPC analysis in practice
        Ok(vec![800.0, 1200.0, 2500.0]) // Typical formant values
    }

    /// Compute jitter (pitch irregularity)
    fn compute_jitter(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let f0_contour = self.extract_f0_contour(samples)?;
        if f0_contour.len() < 2 {
            return Ok(0.0);
        }

        let mut period_diffs = Vec::new();
        for i in 1..f0_contour.len() {
            if f0_contour[i] > 0.0 && f0_contour[i-1] > 0.0 {
                let period1 = 1.0 / f0_contour[i-1];
                let period2 = 1.0 / f0_contour[i];
                period_diffs.push((period2 - period1).abs());
            }
        }

        if period_diffs.is_empty() {
            Ok(0.0)
        } else {
            Ok(self.mean(&period_diffs))
        }
    }

    /// Compute shimmer (amplitude irregularity)
    fn compute_shimmer(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let energy_contour = self.extract_energy_contour(samples)?;
        if energy_contour.len() < 2 {
            return Ok(0.0);
        }

        let mut amplitude_diffs = Vec::new();
        for i in 1..energy_contour.len() {
            let diff = (energy_contour[i] - energy_contour[i-1]).abs();
            amplitude_diffs.push(diff);
        }

        Ok(self.mean(&amplitude_diffs))
    }

    /// Compute harmonic-to-noise ratio (simplified)
    fn compute_hnr(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified HNR computation
        let signal_power: f32 = samples.iter().map(|x| x * x).sum();
        let noise_estimate = self.std_dev(samples).powi(2);
        
        if noise_estimate > 0.0 {
            Ok(10.0 * (signal_power / noise_estimate).log10())
        } else {
            Ok(40.0) // High HNR if no noise detected
        }
    }

    /// Compute breathiness measure
    fn compute_breathiness(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified breathiness computation based on noise content
        let hnr = self.compute_hnr(samples)?;
        Ok((20.0 - hnr).max(0.0) / 20.0) // Inverse of HNR, normalized
    }

    /// Detect vocal fry
    fn detect_vocal_fry(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified vocal fry detection based on low frequency irregularities
        let f0_contour = self.extract_f0_contour(samples)?;
        let low_f0_ratio = f0_contour.iter()
            .filter(|&&f0| f0 > 0.0 && f0 < 80.0) // Very low F0
            .count() as f32 / f0_contour.len() as f32;
        
        Ok(low_f0_ratio)
    }

    /// Compute creakiness measure
    fn compute_creakiness(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified creakiness based on amplitude irregularities at low frequencies
        let shimmer = self.compute_shimmer(samples)?;
        let vocal_fry = self.detect_vocal_fry(samples)?;
        
        Ok((shimmer + vocal_fry) / 2.0)
    }

    /// Compute speech ratio (speech vs silence)
    fn compute_speech_ratio(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let energy_contour = self.extract_energy_contour(samples)?;
        let mean_energy = self.mean(&energy_contour);
        let threshold = mean_energy * 0.1;

        let speech_frames = energy_contour.iter()
            .filter(|&&energy| energy > threshold)
            .count();

        Ok(speech_frames as f32 / energy_contour.len() as f32)
    }

    /// Compute articulation rate
    fn compute_articulation_rate(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simplified articulation rate based on spectral changes
        let zcr = self.compute_zero_crossing_rate(samples);
        let spectral_centroid = self.compute_spectral_centroid(samples)?;
        
        // Higher ZCR and spectral centroid suggest clearer articulation
        Ok((zcr + spectral_centroid / 8000.0).min(1.0))
    }

    /// Compute tempo variation
    fn compute_tempo_variation(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        let energy_contour = self.extract_energy_contour(samples)?;
        if energy_contour.len() < 3 {
            return Ok(0.0);
        }

        // Compute tempo based on energy peaks
        let mut intervals = Vec::new();
        let threshold = self.mean(&energy_contour) * 0.7;
        let mut last_peak = 0;

        for (i, &energy) in energy_contour.iter().enumerate() {
            if energy > threshold && i > last_peak + 3 { // Avoid double peaks
                if last_peak > 0 {
                    intervals.push((i - last_peak) as f32);
                }
                last_peak = i;
            }
        }

        if intervals.is_empty() {
            Ok(0.0)
        } else {
            Ok(self.std_dev(&intervals) / self.mean(&intervals))
        }
    }

    // Utility functions for statistical calculations

    fn mean(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }

    fn std_dev(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = self.mean(values);
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (values.len() - 1) as f32;
        
        variance.sqrt()
    }

    fn variance(&self, values: &[f32]) -> f32 {
        self.std_dev(values).powi(2)
    }

    fn min(&self, values: &[f32]) -> f32 {
        values.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }

    fn max(&self, values: &[f32]) -> f32 {
        values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    fn linear_trend(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f32;
        let sum_x = (0..values.len()).sum::<usize>() as f32;
        let sum_y = values.iter().sum::<f32>();
        let sum_xy: f32 = values.iter().enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum();
        let sum_x2: f32 = (0..values.len())
            .map(|i| (i as f32).powi(2))
            .sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }
}

impl Default for EmotionFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_feature_extraction() {
        let extractor = EmotionFeatureExtractor::new();
        
        // Create test audio
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio = AudioBuffer::mono(samples, 16000);
        
        let features = extractor.extract_emotion_features(&audio).await;
        assert!(features.is_ok());
        
        let features = features.unwrap();
        assert!(features.contains_key("f0_mean"));
        assert!(features.contains_key("energy_mean"));
        assert!(features.contains_key("spectral_centroid"));
        assert!(features.contains_key("jitter"));
    }

    #[test]
    fn test_prosodic_features() {
        let extractor = EmotionFeatureExtractor::new();
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();
        
        let features = extractor.extract_prosodic_features(&samples);
        assert!(features.is_ok());
        
        let features = features.unwrap();
        assert!(features.contains_key("f0_mean"));
        assert!(features.contains_key("energy_mean"));
        assert!(features.contains_key("speaking_rate"));
    }

    #[test]
    fn test_spectral_features() {
        let extractor = EmotionFeatureExtractor::new();
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();
        
        let features = extractor.extract_spectral_features(&samples);
        assert!(features.is_ok());
        
        let features = features.unwrap();
        assert!(features.contains_key("spectral_centroid"));
        assert!(features.contains_key("zero_crossing_rate"));
        assert!(features.contains_key("mfcc_0"));
    }

    #[test]
    fn test_statistical_functions() {
        let extractor = EmotionFeatureExtractor::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(extractor.mean(&values), 3.0);
        assert!(extractor.std_dev(&values) > 0.0);
        assert_eq!(extractor.min(&values), 1.0);
        assert_eq!(extractor.max(&values), 5.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let extractor = EmotionFeatureExtractor::new();
        let samples = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let zcr = extractor.compute_zero_crossing_rate(&samples);
        assert!(zcr > 0.0);
    }
}