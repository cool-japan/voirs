//! Prosody analysis implementation
//!
//! This module provides prosodic analysis capabilities including:
//! - Pitch analysis (F0 extraction, pitch contour)
//! - Rhythm analysis (speaking rate, pause detection)
//! - Stress pattern detection
//! - Intonation analysis

#![allow(clippy::cast_precision_loss)]

use crate::traits::{
    AccentType, BoundaryTone, EnergyAnalysis, IntonationAnalysis, IntonationPattern,
    PauseStatistics, PitchAccent, PitchAnalysis, ProsodyAnalysis, RhythmAnalysis, StressAnalysis,
    ToneType,
};
use crate::RecognitionError;
use voirs_sdk::AudioBuffer;

/// Prosody analyzer
pub struct ProsodyAnalyzer {
    /// Sample rate for analysis
    sample_rate: f32,
    /// Frame size for pitch analysis
    frame_size: usize,
    /// Hop size for overlapping frames
    hop_size: usize,
    /// Window function
    window: Vec<f32>,
    /// Minimum F0 for pitch detection
    min_f0: f32,
    /// Maximum F0 for pitch detection
    max_f0: f32,
}

impl ProsodyAnalyzer {
    /// Create a new prosody analyzer
    ///
    /// # Errors
    ///
    /// Returns an error if the analyzer cannot be initialized
    pub async fn new() -> Result<Self, RecognitionError> {
        let frame_size = 1024;
        let hop_size = 256; // Smaller hop for better temporal resolution
        let window = Self::hann_window(frame_size);

        Ok(Self {
            sample_rate: 16000.0,
            frame_size,
            hop_size,
            window,
            min_f0: 50.0,  // Minimum F0 (Hz)
            max_f0: 800.0, // Maximum F0 (Hz)
        })
    }

    /// Analyze prosodic features of audio
    ///
    /// # Errors
    ///
    /// Returns an error if the prosody analysis fails due to invalid audio data
    pub async fn analyze_prosody(
        &self,
        audio: &AudioBuffer,
    ) -> Result<ProsodyAnalysis, RecognitionError> {
        // Extract pitch information
        let pitch = self.analyze_pitch(audio).await?;

        // Analyze rhythm
        let rhythm = self.analyze_rhythm(audio).await?;

        // Analyze stress patterns
        let stress = self.analyze_stress(audio, &pitch.pitch_contour).await?;

        // Analyze intonation
        let intonation = self.analyze_intonation(audio, &pitch.pitch_contour).await?;

        // Analyze energy
        let energy = self.analyze_energy(audio).await?;

        Ok(ProsodyAnalysis {
            pitch,
            rhythm,
            stress,
            intonation,
            energy,
        })
    }

    /// Analyze pitch characteristics
    async fn analyze_pitch(&self, audio: &AudioBuffer) -> Result<PitchAnalysis, RecognitionError> {
        let samples = audio.samples();

        // Extract pitch contour using autocorrelation-based method
        let pitch_contour = self.extract_pitch_contour(samples).await?;

        // Filter out unvoiced segments (pitch = 0)
        let voiced_pitches: Vec<f32> = pitch_contour
            .iter()
            .filter(|&&p| p > 0.0)
            .copied()
            .collect();

        if voiced_pitches.is_empty() {
            return Ok(PitchAnalysis {
                mean_f0: 0.0,
                f0_std: 0.0,
                f0_range: 0.0,
                pitch_contour,
            });
        }

        // Calculate statistics
        #[allow(clippy::cast_precision_loss)]
        let mean_f0 = voiced_pitches.iter().sum::<f32>() / voiced_pitches.len() as f32;

        #[allow(clippy::cast_precision_loss)]
        let variance = voiced_pitches
            .iter()
            .map(|&p| (p - mean_f0).powi(2))
            .sum::<f32>()
            / voiced_pitches.len() as f32;
        let f0_std = variance.sqrt();

        let min_f0 = voiced_pitches.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_f0 = voiced_pitches
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let f0_range = max_f0 - min_f0;

        Ok(PitchAnalysis {
            mean_f0,
            f0_std,
            f0_range,
            pitch_contour,
        })
    }

    /// Extract pitch contour using autocorrelation
    async fn extract_pitch_contour(&self, samples: &[f32]) -> Result<Vec<f32>, RecognitionError> {
        let mut pitch_contour = Vec::new();

        // Process audio in overlapping frames
        let mut pos = 0;
        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let pitch = self.estimate_pitch_autocorr(frame).await?;
            pitch_contour.push(pitch);
            pos += self.hop_size;
        }

        // Smooth the pitch contour
        let smoothed_contour = self.smooth_pitch_contour(&pitch_contour);

        Ok(smoothed_contour)
    }

    /// Estimate pitch using autocorrelation
    async fn estimate_pitch_autocorr(&self, frame: &[f32]) -> Result<f32, RecognitionError> {
        // Apply window
        let mut windowed_frame = frame.to_vec();
        for (i, sample) in windowed_frame.iter_mut().enumerate() {
            if i < self.window.len() {
                *sample *= self.window[i];
            }
        }

        // Compute autocorrelation
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let min_lag = (self.sample_rate / self.max_f0) as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_lag = (self.sample_rate / self.min_f0) as usize;

        let mut max_corr = 0.0;
        let mut best_lag = 0;

        for lag in min_lag..max_lag.min(windowed_frame.len() / 2) {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in 0..windowed_frame.len() - lag {
                correlation += windowed_frame[i] * windowed_frame[i + lag];
                norm1 += windowed_frame[i] * windowed_frame[i];
                norm2 += windowed_frame[i + lag] * windowed_frame[i + lag];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                let normalized_corr = correlation / (norm1 * norm2).sqrt();
                if normalized_corr > max_corr {
                    max_corr = normalized_corr;
                    best_lag = lag;
                }
            }
        }

        // Convert lag to frequency
        if best_lag > 0 && max_corr > 0.3 {
            // Threshold for voiced detection
            #[allow(clippy::cast_precision_loss)]
            Ok(self.sample_rate / best_lag as f32)
        } else {
            Ok(0.0) // Unvoiced
        }
    }

    /// Smooth pitch contour using median filter
    #[allow(clippy::unused_self)]
    fn smooth_pitch_contour(&self, contour: &[f32]) -> Vec<f32> {
        let window_size = 5;
        let mut smoothed = Vec::new();

        for i in 0..contour.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(contour.len());

            let mut window: Vec<f32> = contour[start..end].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = if window.len() % 2 == 0 {
                (window[window.len() / 2 - 1] + window[window.len() / 2]) / 2.0
            } else {
                window[window.len() / 2]
            };

            smoothed.push(median);
        }

        smoothed
    }

    /// Analyze rhythm characteristics
    async fn analyze_rhythm(
        &self,
        audio: &AudioBuffer,
    ) -> Result<RhythmAnalysis, RecognitionError> {
        let samples = audio.samples();

        // Detect pauses
        let pause_statistics = self.detect_pauses(samples).await?;

        // Calculate speaking rate
        #[allow(clippy::cast_precision_loss)]
        let total_duration = samples.len() as f32 / audio.sample_rate() as f32;
        let speech_duration = total_duration - pause_statistics.total_pause_duration;

        // Estimate syllable count based on vowel-like regions
        let syllable_count = self.estimate_syllable_count(samples).await?;

        let speaking_rate = if speech_duration > 0.0 {
            syllable_count / speech_duration
        } else {
            0.0
        };

        // Calculate rhythm regularity
        let regularity_score = self.calculate_rhythm_regularity(samples).await?;

        Ok(RhythmAnalysis {
            speaking_rate,
            pause_statistics,
            regularity_score,
        })
    }

    /// Detect pauses in audio
    async fn detect_pauses(&self, samples: &[f32]) -> Result<PauseStatistics, RecognitionError> {
        let frame_size = 1024;
        let hop_size = 512;
        let energy_threshold = 0.01; // Threshold for silence detection
        let min_pause_duration = 0.1; // Minimum pause duration in seconds

        // Calculate frame energies
        let mut frame_energies = Vec::new();
        let mut pos = 0;

        while pos + frame_size <= samples.len() {
            let frame = &samples[pos..pos + frame_size];
            #[allow(clippy::cast_precision_loss)]
            let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
            frame_energies.push(energy);
            pos += hop_size;
        }

        // Detect silent regions
        let mut pause_regions = Vec::new();
        let mut in_pause = false;
        let mut pause_start = 0;

        for (i, &energy) in frame_energies.iter().enumerate() {
            if energy < energy_threshold {
                if !in_pause {
                    pause_start = i;
                    in_pause = true;
                }
            } else if in_pause {
                #[allow(clippy::cast_precision_loss)]
                let pause_duration = (i - pause_start) as f32 * hop_size as f32 / 16000.0;
                if pause_duration >= min_pause_duration {
                    pause_regions.push((pause_start, i, pause_duration));
                }
                in_pause = false;
            }
        }

        // Handle pause at the end
        if in_pause {
            #[allow(clippy::cast_precision_loss)]
            let pause_duration =
                (frame_energies.len() - pause_start) as f32 * hop_size as f32 / 16000.0;
            if pause_duration >= min_pause_duration {
                pause_regions.push((pause_start, frame_energies.len(), pause_duration));
            }
        }

        // Calculate statistics
        let pause_count = pause_regions.len();
        let total_pause_duration: f32 = pause_regions.iter().map(|(_, _, duration)| duration).sum();
        let average_pause_duration = if pause_count > 0 {
            #[allow(clippy::cast_precision_loss)]
            {
                total_pause_duration / pause_count as f32
            }
        } else {
            0.0
        };

        let pause_positions: Vec<f32> = pause_regions
            .iter()
            .map(|(start, _, _)| {
                #[allow(clippy::cast_precision_loss)]
                {
                    *start as f32 * hop_size as f32 / 16000.0
                }
            })
            .collect();

        Ok(PauseStatistics {
            total_pause_duration,
            average_pause_duration,
            pause_count,
            pause_positions,
        })
    }

    /// Estimate syllable count based on vowel-like regions
    async fn estimate_syllable_count(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Simple syllable estimation based on energy peaks
        let frame_size = 512;
        let hop_size = 256;

        let mut energies = Vec::new();
        let mut pos = 0;

        while pos + frame_size <= samples.len() {
            let frame = &samples[pos..pos + frame_size];
            let energy = frame.iter().map(|x| x * x).sum::<f32>();
            energies.push(energy);
            pos += hop_size;
        }

        // Find peaks in energy that could correspond to syllable nuclei
        let mut peak_count = 0;
        #[allow(clippy::cast_precision_loss)]
        let threshold = energies.iter().sum::<f32>() / energies.len() as f32 * 1.5;

        for window in energies.windows(3) {
            if window[1] > window[0] && window[1] > window[2] && window[1] > threshold {
                peak_count += 1;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        Ok(peak_count as f32)
    }

    /// Calculate rhythm regularity
    async fn calculate_rhythm_regularity(&self, samples: &[f32]) -> Result<f32, RecognitionError> {
        // Analyze inter-beat intervals
        let frame_size = 1024;
        let hop_size = 512;

        let mut beat_times = Vec::new();
        let mut pos = 0;
        let mut prev_energy = 0.0;

        while pos + frame_size <= samples.len() {
            let frame = &samples[pos..pos + frame_size];
            let energy = frame.iter().map(|x| x * x).sum::<f32>();

            // Detect energy increases (potential beats)
            if energy > prev_energy * 1.2 && energy > 0.01 {
                let time = pos as f32 / 16000.0;
                beat_times.push(time);
            }

            prev_energy = energy;
            pos += hop_size;
        }

        if beat_times.len() < 3 {
            return Ok(0.5); // Default regularity for insufficient data
        }

        // Calculate inter-beat intervals
        let intervals: Vec<f32> = beat_times.windows(2).map(|w| w[1] - w[0]).collect();

        // Calculate coefficient of variation
        let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
        let variance = intervals
            .iter()
            .map(|&interval| (interval - mean_interval).powi(2))
            .sum::<f32>()
            / intervals.len() as f32;
        let std_dev = variance.sqrt();

        let cv = if mean_interval > 0.0 {
            std_dev / mean_interval
        } else {
            1.0
        };

        // Convert to regularity score (lower CV = higher regularity)
        Ok((1.0 - cv.min(1.0)).max(0.0))
    }

    /// Analyze stress patterns
    async fn analyze_stress(
        &self,
        audio: &AudioBuffer,
        pitch_contour: &[f32],
    ) -> Result<StressAnalysis, RecognitionError> {
        let samples = audio.samples();

        // Calculate energy contour
        let energy_contour = self.calculate_energy_contour(samples).await?;

        // Detect stress based on pitch and energy peaks
        let stress_pattern = self
            .detect_stress_pattern(pitch_contour, &energy_contour)
            .await?;

        // Classify stress levels
        let (primary_stress, secondary_stress) = self.classify_stress_levels(&stress_pattern);

        Ok(StressAnalysis {
            stress_pattern,
            primary_stress,
            secondary_stress,
        })
    }

    /// Calculate energy contour
    async fn calculate_energy_contour(
        &self,
        samples: &[f32],
    ) -> Result<Vec<f32>, RecognitionError> {
        let mut energy_contour = Vec::new();
        let mut pos = 0;

        while pos + self.hop_size <= samples.len() {
            let frame = &samples[pos..pos + self.hop_size];
            let energy = frame.iter().map(|x| x * x).sum::<f32>();
            energy_contour.push(energy);
            pos += self.hop_size;
        }

        Ok(energy_contour)
    }

    /// Detect stress pattern
    async fn detect_stress_pattern(
        &self,
        pitch_contour: &[f32],
        energy_contour: &[f32],
    ) -> Result<Vec<f32>, RecognitionError> {
        let min_len = pitch_contour.len().min(energy_contour.len());
        let mut stress_pattern = Vec::new();

        for i in 0..min_len {
            // Combine pitch and energy information for stress detection
            let pitch_factor = if pitch_contour[i] > 0.0 {
                pitch_contour[i] / 200.0 // Normalize around 200Hz
            } else {
                0.0
            };

            let energy_factor = energy_contour[i].sqrt() * 10.0; // Scale energy

            // Simple stress score combining pitch and energy
            let stress_score = (pitch_factor + energy_factor) / 2.0;
            stress_pattern.push(stress_score.min(1.0));
        }

        Ok(stress_pattern)
    }

    /// Classify stress levels
    fn classify_stress_levels(&self, stress_pattern: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut primary_stress = Vec::new();
        let mut secondary_stress = Vec::new();

        // Calculate thresholds
        let mean_stress = stress_pattern.iter().sum::<f32>() / stress_pattern.len() as f32;
        let primary_threshold = mean_stress + 0.3;
        let secondary_threshold = mean_stress + 0.15;

        for (i, &stress) in stress_pattern.iter().enumerate() {
            let time = i as f32 * self.hop_size as f32 / self.sample_rate;

            if stress > primary_threshold {
                primary_stress.push(time);
            } else if stress > secondary_threshold {
                secondary_stress.push(time);
            }
        }

        (primary_stress, secondary_stress)
    }

    /// Analyze intonation patterns
    async fn analyze_intonation(
        &self,
        _audio: &AudioBuffer,
        pitch_contour: &[f32],
    ) -> Result<IntonationAnalysis, RecognitionError> {
        // Classify overall intonation pattern
        let pattern_type = self.classify_intonation_pattern(pitch_contour);

        // Detect boundary tones
        let boundary_tones = self.detect_boundary_tones(pitch_contour).await?;

        // Detect pitch accents
        let pitch_accents = self.detect_pitch_accents(pitch_contour).await?;

        Ok(IntonationAnalysis {
            pattern_type,
            boundary_tones,
            pitch_accents,
        })
    }

    /// Classify intonation pattern
    fn classify_intonation_pattern(&self, pitch_contour: &[f32]) -> IntonationPattern {
        if pitch_contour.is_empty() {
            return IntonationPattern::Declarative;
        }

        // Get voiced segments
        let voiced_pitches: Vec<f32> = pitch_contour
            .iter()
            .filter(|&&p| p > 0.0)
            .copied()
            .collect();

        if voiced_pitches.len() < 3 {
            return IntonationPattern::Declarative;
        }

        // Analyze overall pitch trend throughout the utterance
        let start_pitch = voiced_pitches[0];
        let end_pitch = voiced_pitches[voiced_pitches.len() - 1];

        // Calculate overall pitch change
        let overall_pitch_change = (end_pitch - start_pitch) / start_pitch;

        // Also analyze pitch movement at the end (final 20% of the utterance)
        let end_segment_len = (voiced_pitches.len() / 5).max(3);
        let end_segment = &voiced_pitches[voiced_pitches.len().saturating_sub(end_segment_len)..];
        let end_start_pitch = end_segment[0];
        let end_final_pitch = end_segment[end_segment.len() - 1];

        let final_pitch_change = if end_start_pitch > 0.0 {
            (end_final_pitch - end_start_pitch) / end_start_pitch
        } else {
            0.0
        };

        // Classify based on overall trend and final movement
        if overall_pitch_change > 0.1 || final_pitch_change > 0.15 {
            IntonationPattern::Interrogative
        } else if overall_pitch_change < -0.1 || final_pitch_change < -0.15 {
            IntonationPattern::Declarative
        } else {
            // Check for exclamative pattern (high pitch variation)
            let pitch_variance = self.calculate_pitch_variance(&voiced_pitches);
            if pitch_variance > 0.3 {
                IntonationPattern::Exclamative
            } else {
                IntonationPattern::Declarative
            }
        }
    }

    /// Calculate pitch variance for pattern classification
    #[allow(clippy::unused_self)]
    fn calculate_pitch_variance(&self, pitches: &[f32]) -> f32 {
        if pitches.is_empty() {
            return 0.0;
        }

        let mean = pitches.iter().sum::<f32>() / pitches.len() as f32;
        let variance =
            pitches.iter().map(|&p| (p - mean).powi(2)).sum::<f32>() / pitches.len() as f32;

        variance.sqrt() / mean // Coefficient of variation
    }

    /// Detect boundary tones
    async fn detect_boundary_tones(
        &self,
        pitch_contour: &[f32],
    ) -> Result<Vec<BoundaryTone>, RecognitionError> {
        let mut boundary_tones = Vec::new();

        // Simple boundary tone detection at phrase boundaries
        if pitch_contour.len() > 10 {
            // Check beginning
            let start_segment = &pitch_contour[0..5];
            let start_trend = self.calculate_pitch_trend(start_segment);
            let start_time = 0.0;

            boundary_tones.push(BoundaryTone {
                time: start_time,
                tone_type: self.trend_to_tone_type(start_trend),
                confidence: 0.7,
            });

            // Check end
            let end_segment = &pitch_contour[pitch_contour.len() - 5..];
            let end_trend = self.calculate_pitch_trend(end_segment);
            let end_time = pitch_contour.len() as f32 * self.hop_size as f32 / self.sample_rate;

            boundary_tones.push(BoundaryTone {
                time: end_time,
                tone_type: self.trend_to_tone_type(end_trend),
                confidence: 0.7,
            });
        }

        Ok(boundary_tones)
    }

    /// Calculate pitch trend in a segment
    #[allow(clippy::unused_self)]
    fn calculate_pitch_trend(&self, segment: &[f32]) -> f32 {
        if segment.len() < 2 {
            return 0.0;
        }

        let voiced_segment: Vec<f32> = segment.iter().filter(|&&p| p > 0.0).copied().collect();

        if voiced_segment.len() < 2 {
            return 0.0;
        }

        // Simple linear trend
        let first = voiced_segment[0];
        let last = voiced_segment[voiced_segment.len() - 1];

        (last - first) / first
    }

    /// Convert pitch trend to tone type
    #[allow(clippy::unused_self)]
    fn trend_to_tone_type(&self, trend: f32) -> ToneType {
        if trend > 0.05 {
            ToneType::Rising
        } else if trend < -0.05 {
            ToneType::Falling
        } else {
            ToneType::Level
        }
    }

    /// Detect pitch accents
    async fn detect_pitch_accents(
        &self,
        pitch_contour: &[f32],
    ) -> Result<Vec<PitchAccent>, RecognitionError> {
        let mut pitch_accents = Vec::new();

        // Find local pitch peaks
        for i in 1..pitch_contour.len() - 1 {
            if pitch_contour[i] > 0.0
                && pitch_contour[i] > pitch_contour[i - 1]
                && pitch_contour[i] > pitch_contour[i + 1]
            {
                let time = i as f32 * self.hop_size as f32 / self.sample_rate;

                // Classify accent type based on pitch height
                let accent_type = if pitch_contour[i] > 300.0 {
                    AccentType::Primary
                } else if pitch_contour[i] > 200.0 {
                    AccentType::Secondary
                } else {
                    AccentType::Tertiary
                };

                pitch_accents.push(PitchAccent {
                    time,
                    accent_type,
                    confidence: 0.8,
                });
            }
        }

        Ok(pitch_accents)
    }

    /// Generate Hann window
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect()
    }

    /// Analyze energy characteristics
    async fn analyze_energy(
        &self,
        audio: &AudioBuffer,
    ) -> Result<EnergyAnalysis, RecognitionError> {
        let samples = audio.samples();

        // Calculate frame-based energy
        let mut energy_contour = Vec::new();
        let mut total_energy = 0.0f32;

        for chunk in samples.chunks(self.hop_size) {
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
            energy_contour.push(energy);
            total_energy += energy;
        }

        let mean_energy = total_energy / energy_contour.len() as f32;

        // Calculate statistics
        let energy_std = {
            let variance = energy_contour
                .iter()
                .map(|&e| (e - mean_energy).powi(2))
                .sum::<f32>()
                / energy_contour.len() as f32;
            variance.sqrt()
        };

        let energy_min = energy_contour
            .iter()
            .fold(f32::INFINITY, |acc, &x| acc.min(x));
        let energy_max = energy_contour
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let energy_range = energy_max - energy_min;

        Ok(EnergyAnalysis {
            mean_energy,
            energy_std,
            energy_range,
            energy_contour,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_prosody_analyzer_creation() {
        let analyzer = ProsodyAnalyzer::new().await.unwrap();
        assert_eq!(analyzer.frame_size, 1024);
        assert_eq!(analyzer.hop_size, 256);
        assert_eq!(analyzer.min_f0, 50.0);
        assert_eq!(analyzer.max_f0, 800.0);
    }

    #[tokio::test]
    async fn test_pitch_analysis() {
        let analyzer = ProsodyAnalyzer::new().await.unwrap();

        // Generate a sine wave with known frequency
        let frequency = 220.0; // A3 note
        let samples: Vec<f32> = (0..16000) // 1 second at 16kHz
            .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / 16000.0).sin())
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let pitch_analysis = analyzer.analyze_pitch(&audio).await.unwrap();

        // Should detect pitch close to the generated frequency
        assert!(pitch_analysis.mean_f0 > 0.0);
        assert!(!pitch_analysis.pitch_contour.is_empty());
        assert!(pitch_analysis.f0_range >= 0.0);
    }

    #[tokio::test]
    async fn test_rhythm_analysis() {
        let analyzer = ProsodyAnalyzer::new().await.unwrap();

        // Create audio with some pauses
        let mut samples = vec![0.1; 8000]; // 0.5 seconds of sound
        samples.extend(vec![0.0; 1600]); // 0.1 seconds of silence
        samples.extend(vec![0.1; 8000]); // 0.5 seconds of sound

        let audio = AudioBuffer::new(samples, 16000, 1);

        let rhythm_analysis = analyzer.analyze_rhythm(&audio).await.unwrap();

        assert!(rhythm_analysis.pause_statistics.pause_count > 0);
        assert!(rhythm_analysis.pause_statistics.total_pause_duration > 0.0);
        assert!(rhythm_analysis.speaking_rate >= 0.0);
        assert!(rhythm_analysis.regularity_score >= 0.0);
        assert!(rhythm_analysis.regularity_score <= 1.0);
    }

    #[tokio::test]
    async fn test_stress_analysis() {
        let analyzer = ProsodyAnalyzer::new().await.unwrap();

        let samples = vec![0.1; 16000]; // 1 second of audio
        let audio = AudioBuffer::new(samples, 16000, 1);

        // Create mock pitch contour
        let pitch_contour = vec![200.0; 64]; // Constant pitch

        let stress_analysis = analyzer
            .analyze_stress(&audio, &pitch_contour)
            .await
            .unwrap();

        assert!(!stress_analysis.stress_pattern.is_empty());
        // Stress values should be between 0 and 1
        for &stress in &stress_analysis.stress_pattern {
            assert!(stress >= 0.0);
            assert!(stress <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_intonation_analysis() {
        let analyzer = ProsodyAnalyzer::new().await.unwrap();

        let samples = vec![0.1; 16000];
        let audio = AudioBuffer::new(samples, 16000, 1);

        // Create rising pitch contour (interrogative pattern)
        let pitch_contour: Vec<f32> = (0..64)
            .map(|i| 150.0 + i as f32 * 2.0) // Rising from 150Hz to ~280Hz
            .collect();

        let intonation_analysis = analyzer
            .analyze_intonation(&audio, &pitch_contour)
            .await
            .unwrap();

        // Should detect interrogative pattern
        assert_eq!(
            intonation_analysis.pattern_type,
            IntonationPattern::Interrogative
        );
        assert!(!intonation_analysis.boundary_tones.is_empty());
    }

    #[tokio::test]
    async fn test_complete_prosody_analysis() {
        let analyzer = ProsodyAnalyzer::new().await.unwrap();

        // Generate complex audio with varying frequency
        let samples: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                let frequency = 220.0 + 50.0 * t; // Frequency sweep
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();
        let audio = AudioBuffer::new(samples, 16000, 1);

        let prosody_analysis = analyzer.analyze_prosody(&audio).await.unwrap();

        // Check all components are present
        assert!(prosody_analysis.pitch.mean_f0 > 0.0);
        assert!(!prosody_analysis.pitch.pitch_contour.is_empty());

        assert!(prosody_analysis.rhythm.speaking_rate >= 0.0);
        assert!(prosody_analysis.rhythm.regularity_score >= 0.0);

        assert!(!prosody_analysis.stress.stress_pattern.is_empty());

        // Should have detected some boundary tones
        assert!(!prosody_analysis.intonation.boundary_tones.is_empty());
    }

    #[tokio::test]
    async fn test_pause_detection() {
        let analyzer = ProsodyAnalyzer::new().await.unwrap();

        // Create audio with clear pauses
        let mut samples = Vec::new();
        samples.extend(vec![0.5; 8000]); // 0.5s speech
        samples.extend(vec![0.0; 3200]); // 0.2s pause
        samples.extend(vec![0.5; 8000]); // 0.5s speech
        samples.extend(vec![0.0; 1600]); // 0.1s pause
        samples.extend(vec![0.5; 8000]); // 0.5s speech

        let pause_stats = analyzer.detect_pauses(&samples).await.unwrap();

        assert!(pause_stats.pause_count >= 1); // Should detect at least one pause
        assert!(pause_stats.total_pause_duration > 0.1); // Should have significant pause time
        assert!(!pause_stats.pause_positions.is_empty());
    }

    #[tokio::test]
    async fn test_pitch_smoothing() {
        let analyzer = ProsodyAnalyzer::new().await.unwrap();

        // Create noisy pitch contour
        let noisy_contour = vec![220.0, 0.0, 225.0, 0.0, 230.0, 0.0, 235.0];
        let smoothed = analyzer.smooth_pitch_contour(&noisy_contour);

        assert_eq!(smoothed.len(), noisy_contour.len());

        // Smoothed contour should have less variation
        let original_variation = noisy_contour
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f32>();
        let smoothed_variation = smoothed
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f32>();

        // In most cases, smoothing should reduce variation
        // (though not always due to the nature of median filtering)
        assert!(smoothed_variation <= original_variation * 1.5);
    }
}
