//! Naturalness testing for human-like singing quality

use super::reports::{NaturalnessProfile, NaturalnessReport};
use crate::types::{VoiceCharacteristics, VoiceType};
use crate::{Error, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::HashMap;

/// Naturalness testing for human-like singing quality
pub struct NaturalnessTester {
    reference_profiles: HashMap<VoiceType, NaturalnessProfile>,
}

impl NaturalnessTester {
    /// Create a new naturalness tester
    pub fn new() -> Self {
        let mut reference_profiles = HashMap::new();

        // Initialize reference profiles for different voice types
        reference_profiles.insert(VoiceType::Soprano, NaturalnessProfile::soprano_default());
        reference_profiles.insert(VoiceType::Alto, NaturalnessProfile::alto_default());
        reference_profiles.insert(VoiceType::Tenor, NaturalnessProfile::tenor_default());
        reference_profiles.insert(VoiceType::Bass, NaturalnessProfile::bass_default());

        Self { reference_profiles }
    }

    /// Evaluate naturalness of singing synthesis
    pub fn evaluate_naturalness(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
        voice_characteristics: &VoiceCharacteristics,
    ) -> Result<NaturalnessReport> {
        let voice_type = voice_characteristics.voice_type.clone();
        let reference_profile = self.reference_profiles.get(&voice_type).ok_or_else(|| {
            Error::Validation(format!(
                "No reference profile for voice type: {:?}",
                voice_type
            ))
        })?;

        // Analyze various naturalness aspects
        let breath_naturalness = self.analyze_breath_naturalness(audio_samples, sample_rate)?;
        let vibrato_naturalness = self.analyze_vibrato_naturalness(audio_samples, sample_rate)?;
        let formant_naturalness =
            self.analyze_formant_naturalness(audio_samples, sample_rate, reference_profile)?;
        let transition_naturalness =
            self.analyze_transition_naturalness(audio_samples, sample_rate)?;
        let timbre_consistency = self.analyze_timbre_consistency(audio_samples, sample_rate)?;

        let overall_score = breath_naturalness * 0.25
            + vibrato_naturalness * 0.20
            + formant_naturalness * 0.25
            + transition_naturalness * 0.20
            + timbre_consistency * 0.10;

        Ok(NaturalnessReport {
            overall_score,
            breath_naturalness,
            vibrato_naturalness,
            formant_naturalness,
            transition_naturalness,
            timbre_consistency,
            reference_voice_type: voice_type,
            recommendations: self.generate_naturalness_recommendations(
                breath_naturalness,
                vibrato_naturalness,
                formant_naturalness,
                transition_naturalness,
                timbre_consistency,
            ),
        })
    }

    /// Analyze breath pattern naturalness
    fn analyze_breath_naturalness(&self, audio_samples: &[f32], sample_rate: u32) -> Result<f64> {
        // Detect breath patterns and naturalness
        let window_size = (sample_rate as f64 * 0.1) as usize; // 100ms windows
        let mut breath_scores = Vec::new();

        for chunk in audio_samples.chunks(window_size) {
            let rms = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            let zero_crossings = chunk.windows(2).filter(|w| w[0] * w[1] < 0.0).count();

            // Natural breath patterns have low RMS and moderate zero crossings
            let breath_score = if rms < 0.1 && zero_crossings > 10 && zero_crossings < 100 {
                1.0 - (rms - 0.05).abs() * 10.0
            } else {
                0.5
            };

            breath_scores.push(breath_score.clamp(0.0, 1.0));
        }

        Ok(breath_scores.iter().map(|&x| x as f64).sum::<f64>() / breath_scores.len() as f64)
    }

    /// Analyze vibrato naturalness
    fn analyze_vibrato_naturalness(&self, audio_samples: &[f32], sample_rate: u32) -> Result<f64> {
        // Analyze vibrato characteristics for naturalness
        let window_size = sample_rate as usize / 2; // 0.5 second windows
        let mut vibrato_scores = Vec::new();

        for chunk in audio_samples.chunks(window_size) {
            if chunk.len() < window_size {
                continue;
            }

            // Perform FFT to detect vibrato frequency
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(chunk.len());
            let mut buffer: Vec<Complex<f32>> =
                chunk.iter().map(|&x| Complex::new(x, 0.0)).collect();
            fft.process(&mut buffer);

            // Analyze frequency modulation for vibrato
            let vibrato_rate = self.detect_vibrato_rate(&buffer, sample_rate as f32);
            let vibrato_depth = self.detect_vibrato_depth(&buffer);

            // Natural vibrato: 4-7 Hz rate, moderate depth
            let rate_score = if vibrato_rate >= 4.0 && vibrato_rate <= 7.0 {
                1.0 - ((vibrato_rate - 5.5).abs() / 1.5)
            } else {
                0.3
            };

            let depth_score = if vibrato_depth >= 0.02 && vibrato_depth <= 0.08 {
                1.0 - ((vibrato_depth - 0.05).abs() / 0.03)
            } else {
                0.3
            };

            vibrato_scores.push((rate_score * 0.6 + depth_score * 0.4).clamp(0.0, 1.0));
        }

        if vibrato_scores.is_empty() {
            Ok(0.5) // Neutral score if no vibrato detected
        } else {
            Ok(vibrato_scores.iter().map(|&x| x as f64).sum::<f64>() / vibrato_scores.len() as f64)
        }
    }

    /// Detect vibrato rate from FFT buffer
    fn detect_vibrato_rate(&self, fft_buffer: &[Complex<f32>], sample_rate: f32) -> f32 {
        // Simple vibrato detection - in practice this would be more sophisticated
        let bin_resolution = sample_rate / fft_buffer.len() as f32;

        // Look for peak in the 3-8 Hz range (typical vibrato range)
        let start_bin = (3.0 / bin_resolution) as usize;
        let end_bin = (8.0 / bin_resolution) as usize;

        if start_bin < fft_buffer.len() && end_bin < fft_buffer.len() {
            let mut max_magnitude = 0.0;
            let mut peak_bin = start_bin;

            for i in start_bin..=end_bin {
                let magnitude = fft_buffer[i].norm();
                if magnitude > max_magnitude {
                    max_magnitude = magnitude;
                    peak_bin = i;
                }
            }

            peak_bin as f32 * bin_resolution
        } else {
            5.0 // Default vibrato rate
        }
    }

    /// Detect vibrato depth from FFT buffer
    fn detect_vibrato_depth(&self, fft_buffer: &[Complex<f32>]) -> f32 {
        // Simplified vibrato depth calculation
        let total_energy: f32 = fft_buffer.iter().map(|c| c.norm_sqr()).sum();
        let modulation_energy: f32 = fft_buffer[1..fft_buffer.len() / 4]
            .iter()
            .map(|c| c.norm_sqr())
            .sum();

        if total_energy > 0.0 {
            (modulation_energy / total_energy).sqrt()
        } else {
            0.0
        }
    }

    /// Analyze formant naturalness against reference profile
    fn analyze_formant_naturalness(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
        reference_profile: &NaturalnessProfile,
    ) -> Result<f64> {
        // Extract formant frequencies using LPC or other formant analysis
        let formant_scores =
            self.extract_and_evaluate_formants(audio_samples, sample_rate, reference_profile)?;

        // Calculate average formant naturalness
        if formant_scores.is_empty() {
            Ok(0.5) // Neutral score if no formants detected
        } else {
            Ok(formant_scores.iter().sum::<f64>() / formant_scores.len() as f64)
        }
    }

    /// Extract and evaluate formant frequencies
    fn extract_and_evaluate_formants(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
        reference_profile: &NaturalnessProfile,
    ) -> Result<Vec<f64>> {
        let frame_size = sample_rate as usize / 20; // 50ms frames
        let mut formant_scores = Vec::new();

        for chunk in audio_samples.chunks(frame_size) {
            if chunk.len() < frame_size {
                continue;
            }

            // Simplified formant extraction using peak detection in FFT
            let (f1, f2) = self.extract_formants_from_frame(chunk, sample_rate as f32);

            // Evaluate against reference profile
            let f1_score =
                if f1 >= reference_profile.f1_range.0 && f1 <= reference_profile.f1_range.1 {
                    1.0
                } else {
                    let deviation = if f1 < reference_profile.f1_range.0 {
                        reference_profile.f1_range.0 - f1
                    } else {
                        f1 - reference_profile.f1_range.1
                    };
                    (1.0 - deviation / 500.0).clamp(0.0, 1.0)
                };

            let f2_score =
                if f2 >= reference_profile.f2_range.0 && f2 <= reference_profile.f2_range.1 {
                    1.0
                } else {
                    let deviation = if f2 < reference_profile.f2_range.0 {
                        reference_profile.f2_range.0 - f2
                    } else {
                        f2 - reference_profile.f2_range.1
                    };
                    (1.0 - deviation / 1000.0).clamp(0.0, 1.0)
                };

            formant_scores.push(((f1_score + f2_score) / 2.0) as f64);
        }

        Ok(formant_scores)
    }

    /// Extract formant frequencies from a frame
    fn extract_formants_from_frame(&self, frame: &[f32], sample_rate: f32) -> (f32, f32) {
        // Simplified formant extraction using FFT peaks
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame.len());
        let mut buffer: Vec<Complex<f32>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buffer);

        let bin_resolution = sample_rate / frame.len() as f32;
        let spectrum: Vec<f32> = buffer.iter().map(|c| c.norm()).collect();

        // Find first two prominent peaks (simplified formant detection)
        let mut peaks = Vec::new();
        for i in 2..spectrum.len() / 2 - 2 {
            if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.1 {
                peaks.push((i, spectrum[i]));
            }
        }

        // Sort by magnitude and take top 2
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let f1 = if peaks.len() > 0 {
            peaks[0].0 as f32 * bin_resolution
        } else {
            500.0 // Default F1
        };

        let f2 = if peaks.len() > 1 {
            peaks[1].0 as f32 * bin_resolution
        } else {
            1500.0 // Default F2
        };

        (f1, f2)
    }

    /// Analyze transition naturalness between notes
    fn analyze_transition_naturalness(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> Result<f64> {
        // Analyze pitch and amplitude transitions
        let frame_size = sample_rate as usize / 50; // 20ms frames
        let mut transition_scores = Vec::new();

        let mut prev_pitch = 0.0;
        let mut prev_amplitude = 0.0;

        for chunk in audio_samples.chunks(frame_size) {
            if chunk.len() < frame_size {
                continue;
            }

            let pitch = self.estimate_pitch(chunk, sample_rate as f32);
            let amplitude = chunk.iter().map(|&x| x.abs()).sum::<f32>() / chunk.len() as f32;

            if prev_pitch > 0.0 && pitch > 0.0 {
                // Calculate transition smoothness
                let pitch_change = (pitch - prev_pitch).abs() / prev_pitch;
                let amplitude_change =
                    (amplitude - prev_amplitude).abs() / prev_amplitude.max(0.001);

                // Natural transitions are gradual
                let pitch_score = if pitch_change < 0.1 {
                    1.0
                } else {
                    (1.0 - pitch_change).clamp(0.0, 1.0)
                };

                let amplitude_score = if amplitude_change < 0.3 {
                    1.0
                } else {
                    (1.0 - amplitude_change).clamp(0.0, 1.0)
                };

                transition_scores.push(((pitch_score + amplitude_score) / 2.0) as f64);
            }

            prev_pitch = pitch;
            prev_amplitude = amplitude;
        }

        if transition_scores.is_empty() {
            Ok(0.5)
        } else {
            Ok(transition_scores.iter().sum::<f64>() / transition_scores.len() as f64)
        }
    }

    /// Estimate pitch using autocorrelation
    fn estimate_pitch(&self, frame: &[f32], sample_rate: f32) -> f32 {
        let min_period = (sample_rate / 800.0) as usize; // 800 Hz max
        let max_period = (sample_rate / 80.0) as usize; // 80 Hz min

        let mut best_correlation = 0.0;
        let mut best_period = 0;

        for period in min_period..max_period.min(frame.len() / 2) {
            let mut correlation = 0.0;
            for i in 0..frame.len() - period {
                correlation += frame[i] * frame[i + period];
            }

            if correlation > best_correlation {
                best_correlation = correlation;
                best_period = period;
            }
        }

        if best_period > 0 && best_correlation > 0.1 {
            sample_rate / best_period as f32
        } else {
            0.0
        }
    }

    /// Analyze timbre consistency throughout the audio
    fn analyze_timbre_consistency(&self, audio_samples: &[f32], sample_rate: u32) -> Result<f64> {
        let frame_size = sample_rate as usize / 10; // 100ms frames
        let mut spectral_centroids = Vec::new();

        for chunk in audio_samples.chunks(frame_size) {
            if chunk.len() < frame_size {
                continue;
            }

            let centroid = self.calculate_spectral_centroid(chunk, sample_rate as f32);
            spectral_centroids.push(centroid);
        }

        if spectral_centroids.len() < 2 {
            return Ok(0.5);
        }

        // Calculate consistency as inverse of variance
        let mean: f32 = spectral_centroids.iter().sum::<f32>() / spectral_centroids.len() as f32;
        let variance: f32 = spectral_centroids
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / spectral_centroids.len() as f32;

        let std_dev = variance.sqrt();
        let consistency = if mean > 0.0 {
            1.0 - (std_dev / mean).min(1.0)
        } else {
            0.0
        };

        Ok(consistency as f64)
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, frame: &[f32], sample_rate: f32) -> f32 {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(frame.len());
        let mut buffer: Vec<Complex<f32>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buffer);

        let bin_resolution = sample_rate / frame.len() as f32;
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, complex_bin) in buffer.iter().enumerate().take(buffer.len() / 2) {
            let magnitude = complex_bin.norm();
            let frequency = i as f32 * bin_resolution;

            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    /// Generate naturalness improvement recommendations
    fn generate_naturalness_recommendations(
        &self,
        breath_naturalness: f64,
        vibrato_naturalness: f64,
        formant_naturalness: f64,
        transition_naturalness: f64,
        timbre_consistency: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if breath_naturalness < 0.7 {
            recommendations.push(
                "Improve breath modeling for more natural pauses and breathing patterns"
                    .to_string(),
            );
        }

        if vibrato_naturalness < 0.6 {
            recommendations.push(
                "Adjust vibrato parameters - aim for 4-7 Hz rate with moderate depth".to_string(),
            );
        }

        if formant_naturalness < 0.7 {
            recommendations.push(
                "Optimize formant frequencies to better match target voice type characteristics"
                    .to_string(),
            );
        }

        if transition_naturalness < 0.6 {
            recommendations.push(
                "Smooth pitch and amplitude transitions between notes for more natural legato"
                    .to_string(),
            );
        }

        if timbre_consistency < 0.7 {
            recommendations
                .push("Improve timbre consistency across different notes and dynamics".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Naturalness evaluation shows good overall quality".to_string());
        }

        recommendations
    }
}

impl Default for NaturalnessTester {
    fn default() -> Self {
        Self::new()
    }
}
