//! Voice Activity Detection (VAD) implementation
//!
//! This module provides voice activity detection capabilities for identifying
//! speech segments in audio streams.

use crate::{RecognitionError, RecognitionResult};
use scirs2_fft::RealFftPlanner;
use voirs_sdk::AudioBuffer;

/// Voice Activity Detection configuration
#[derive(Debug, Clone)]
pub struct VADConfig {
    /// Energy threshold for speech detection
    pub energy_threshold: f32,
    /// Minimum speech segment duration (seconds)
    pub min_speech_duration: f32,
    /// Minimum silence duration (seconds)
    pub min_silence_duration: f32,
    /// Window size for analysis (samples)
    pub window_size: usize,
    /// Hop size for analysis (samples)
    pub hop_size: usize,
    /// Enable adaptive threshold adjustment
    pub adaptive_threshold: bool,
    /// Spectral centroid threshold for speech detection
    pub spectral_centroid_threshold: f32,
    /// Zero crossing rate threshold for speech detection
    pub zcr_threshold: f32,
    /// Smoothing factor for energy (0.0 to 1.0)
    pub energy_smoothing: f32,
    /// Noise floor estimation window (seconds)
    pub noise_estimation_window: f32,
    /// Spectral rolloff threshold (0.0 to 1.0)
    pub spectral_rolloff_threshold: f32,
}

impl Default for VADConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.01,
            min_speech_duration: 0.1,
            min_silence_duration: 0.1,
            window_size: 1024,
            hop_size: 512,
            adaptive_threshold: true,
            spectral_centroid_threshold: 1500.0, // Hz
            zcr_threshold: 0.1,
            energy_smoothing: 0.2,
            noise_estimation_window: 2.0, // seconds
            spectral_rolloff_threshold: 0.85,
        }
    }
}

/// Voice activity detection result
#[derive(Debug, Clone)]
pub struct VADResult {
    /// Whether voice activity was detected
    pub voice_detected: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Energy level
    pub energy: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Spectral centroid (Hz)
    pub spectral_centroid: f32,
    /// Spectral rolloff frequency (Hz)
    pub spectral_rolloff: f32,
    /// Adaptive threshold used for this frame
    pub adaptive_threshold: f32,
}

/// Speech segment with timing information
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Duration in seconds
    pub duration: f32,
    /// Confidence score
    pub confidence: f32,
}

/// Voice Activity Detector implementation
pub struct VoiceActivityDetector {
    /// Configuration
    config: VADConfig,
    /// Previous frame energy for smoothing
    prev_energy: f32,
    /// Speech state tracking
    in_speech: bool,
    /// Current speech segment start
    speech_start: Option<f32>,
    /// FFT planner for spectral analysis
    fft_planner: RealFftPlanner<f32>,
    /// Noise floor estimate for adaptive thresholding
    noise_floor: f32,
    /// Noise samples buffer for floor estimation
    noise_samples: Vec<f32>,
    /// Current adaptive threshold
    current_threshold: f32,
}

impl VoiceActivityDetector {
    /// Create new VAD with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(VADConfig::default())
    }

    /// Create new VAD with custom configuration
    #[must_use]
    pub fn with_config(config: VADConfig) -> Self {
        let planner = RealFftPlanner::<f32>::new();
        Self {
            current_threshold: config.energy_threshold,
            config,
            prev_energy: 0.0,
            in_speech: false,
            speech_start: None,
            fft_planner: planner,
            noise_floor: 0.001, // Initialize with small value
            noise_samples: Vec::new(),
        }
    }

    /// Detect voice activity in audio frame
    ///
    /// # Errors
    /// Returns `RecognitionError` if audio processing fails
    pub fn detect_frame(
        &mut self,
        audio: &AudioBuffer,
        _timestamp: f32,
    ) -> RecognitionResult<VADResult> {
        let samples = audio.samples();

        if samples.is_empty() {
            return Ok(VADResult {
                voice_detected: false,
                confidence: 0.0,
                energy: 0.0,
                zero_crossing_rate: 0.0,
                spectral_centroid: 0.0,
                spectral_rolloff: 0.0,
                adaptive_threshold: self.current_threshold,
            });
        }

        // Calculate energy
        let energy = Self::calculate_energy(samples);

        // Calculate zero crossing rate
        let zcr = Self::calculate_zero_crossing_rate(samples);

        // Calculate spectral features
        let (spectral_centroid, spectral_rolloff) =
            self.calculate_spectral_features(samples, audio.sample_rate())?;

        // Update adaptive threshold if enabled
        if self.config.adaptive_threshold {
            self.update_adaptive_threshold(energy);
        }

        // Enhanced voice activity detection combining multiple features
        let energy_vote = energy > self.current_threshold;
        let zcr_vote = zcr < self.config.zcr_threshold && zcr > 0.001; // Lower ZCR typically indicates speech, but not pure silence
        let spectral_vote = spectral_centroid > self.config.spectral_centroid_threshold;
        let rolloff_vote = spectral_rolloff
            < self.config.spectral_rolloff_threshold * audio.sample_rate() as f32 / 2.0
            && spectral_rolloff > 100.0;

        // Energy must be above threshold as mandatory condition, plus at least one other feature
        // For backwards compatibility, if energy is significantly above threshold, accept it even without other features
        let voice_detected = energy_vote
            && (energy > self.current_threshold * 2.0 || // Strong energy signal
            zcr_vote || spectral_vote || rolloff_vote);

        // Enhanced confidence calculation using multiple features
        let confidence = if voice_detected {
            let energy_conf = (energy / self.current_threshold).min(1.0);
            let zcr_conf = if zcr < self.config.zcr_threshold {
                1.0 - (zcr / self.config.zcr_threshold)
            } else {
                0.0
            };
            let spectral_conf = if spectral_centroid > self.config.spectral_centroid_threshold {
                (spectral_centroid / (self.config.spectral_centroid_threshold * 2.0)).min(1.0)
            } else {
                0.0
            };

            // Weighted average of confidence scores
            (0.4 * energy_conf + 0.3 * zcr_conf + 0.3 * spectral_conf).min(1.0)
        } else {
            0.0
        };

        // Smooth energy with configurable smoothing factor
        self.prev_energy = (1.0 - self.config.energy_smoothing) * self.prev_energy
            + self.config.energy_smoothing * energy;

        Ok(VADResult {
            voice_detected,
            confidence,
            energy,
            zero_crossing_rate: zcr,
            spectral_centroid,
            spectral_rolloff,
            adaptive_threshold: self.current_threshold,
        })
    }

    /// Detect speech segments in complete audio
    ///
    /// # Errors
    /// Returns `RecognitionError` if audio processing fails
    pub fn detect_segments(
        &mut self,
        audio: &AudioBuffer,
    ) -> RecognitionResult<Vec<SpeechSegment>> {
        let mut segments = Vec::new();
        let samples = audio.samples();
        #[allow(clippy::cast_precision_loss)]
        let sample_rate = audio.sample_rate() as f32;

        if samples.is_empty() {
            return Ok(segments);
        }

        let mut current_start: Option<f32> = None;
        let mut confidence_sum = 0.0;
        let mut confidence_count = 0;
        let mut i = 0;

        while i < samples.len() {
            let end_idx = (i + self.config.window_size).min(samples.len());
            let window = &samples[i..end_idx];

            // Create temporary audio buffer for this window
            let window_audio =
                AudioBuffer::new(window.to_vec(), audio.sample_rate(), audio.channels());

            #[allow(clippy::cast_precision_loss)]
            let timestamp = i as f32 / sample_rate;
            let result = self.detect_frame(&window_audio, timestamp)?;

            if result.voice_detected {
                if current_start.is_none() {
                    current_start = Some(timestamp);
                    confidence_sum = 0.0;
                    confidence_count = 0;
                }
                confidence_sum += result.confidence;
                confidence_count += 1;
            } else if let Some(start) = current_start {
                // End of speech segment
                let duration = timestamp - start;
                if duration >= self.config.min_speech_duration {
                    #[allow(clippy::cast_precision_loss)]
                    let avg_confidence = if confidence_count > 0 {
                        confidence_sum / confidence_count as f32
                    } else {
                        0.5 // Default confidence
                    };

                    segments.push(SpeechSegment {
                        start_time: start,
                        end_time: timestamp,
                        duration,
                        confidence: avg_confidence.max(0.1), // Ensure confidence is always > 0
                    });
                }
                current_start = None;
            }

            i += self.config.hop_size;
        }

        // Handle case where audio ends during speech
        if let Some(start) = current_start {
            #[allow(clippy::cast_precision_loss)]
            let end_time = samples.len() as f32 / sample_rate;
            let duration = end_time - start;
            if duration >= self.config.min_speech_duration {
                #[allow(clippy::cast_precision_loss)]
                let avg_confidence = if confidence_count > 0 {
                    confidence_sum / confidence_count as f32
                } else {
                    0.5 // Default confidence
                };

                segments.push(SpeechSegment {
                    start_time: start,
                    end_time,
                    duration,
                    confidence: avg_confidence.max(0.1), // Ensure confidence is always > 0
                });
            }
        }

        Ok(segments)
    }

    /// Reset VAD state
    pub fn reset(&mut self) {
        self.prev_energy = 0.0;
        self.in_speech = false;
        self.speech_start = None;
        self.noise_floor = 0.001;
        self.noise_samples.clear();
        self.current_threshold = self.config.energy_threshold;
    }

    /// Calculate spectral features (centroid and rolloff)
    fn calculate_spectral_features(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> RecognitionResult<(f32, f32)> {
        if samples.len() < 64 {
            return Ok((0.0, 0.0));
        }

        // Pad or truncate to power of 2 for FFT efficiency
        let fft_size = samples.len().next_power_of_two().min(2048);
        let mut input = vec![0.0f32; fft_size];
        let copy_len = samples.len().min(fft_size);
        input[..copy_len].copy_from_slice(&samples[..copy_len]);

        // Apply Hamming window
        for (i, sample) in input.iter_mut().enumerate() {
            let window_val =
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos();
            *sample *= window_val;
        }

        // Convert input to f64 for FFT computation
        let input_f64: Vec<f64> = input.iter().map(|&x| x as f64).collect();

        // Perform FFT using functional API
        let spectrum = scirs2_fft::rfft(&input_f64, None).map_err(|e| {
            RecognitionError::AudioProcessingError {
                message: format!("FFT computation failed: {e}"),
                source: None,
            }
        })?;

        // Calculate magnitude spectrum and convert to f32
        let mut magnitudes: Vec<f32> = spectrum
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f32)
            .collect();

        // Calculate spectral centroid
        let mut weighted_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;

        for (i, &magnitude) in magnitudes.iter().enumerate() {
            let freq = i as f32 * sample_rate as f32 / fft_size as f32;
            weighted_sum += freq * magnitude;
            magnitude_sum += magnitude;
        }

        let spectral_centroid = if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        };

        // Calculate spectral rolloff (frequency below which 85% of energy is contained)
        let total_energy: f32 = magnitudes.iter().map(|&m| m * m).sum();
        let rolloff_threshold = total_energy * self.config.spectral_rolloff_threshold;

        let mut cumulative_energy = 0.0f32;
        let mut rolloff_freq = 0.0f32;

        for (i, &magnitude) in magnitudes.iter().enumerate() {
            cumulative_energy += magnitude * magnitude;
            if cumulative_energy >= rolloff_threshold {
                rolloff_freq = i as f32 * sample_rate as f32 / fft_size as f32;
                break;
            }
        }

        Ok((spectral_centroid, rolloff_freq))
    }

    /// Update adaptive threshold based on current energy
    fn update_adaptive_threshold(&mut self, energy: f32) {
        // Add energy sample to noise estimation buffer
        if !self.in_speech {
            self.noise_samples.push(energy);

            // Limit buffer size based on noise estimation window
            let max_samples = (self.config.noise_estimation_window * 16000.0
                / self.config.hop_size as f32) as usize;
            if self.noise_samples.len() > max_samples {
                self.noise_samples.remove(0);
            }

            // Update noise floor estimate (median of recent quiet samples)
            if self.noise_samples.len() > 10 {
                let mut sorted_samples = self.noise_samples.clone();
                sorted_samples
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median_idx = sorted_samples.len() / 2;
                self.noise_floor = sorted_samples[median_idx];

                // Adapt threshold to be 3-5x above noise floor
                let adaptive_factor = 4.0;
                self.current_threshold =
                    (self.noise_floor * adaptive_factor).max(self.config.energy_threshold * 0.1);
            }
        }
    }

    /// Calculate energy of audio samples
    #[allow(clippy::cast_precision_loss)]
    fn calculate_energy(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Calculate zero crossing rate
    #[allow(clippy::cast_precision_loss)]
    fn calculate_zero_crossing_rate(samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0i32;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (samples.len() - 1) as f32
    }
}

impl Default for VoiceActivityDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_creation() {
        let vad = VoiceActivityDetector::new();
        assert!(!vad.in_speech);
        assert_eq!(vad.prev_energy, 0.0);
    }

    #[test]
    fn test_vad_with_config() {
        let config = VADConfig {
            energy_threshold: 0.05,
            min_speech_duration: 0.2,
            ..Default::default()
        };
        let vad = VoiceActivityDetector::with_config(config.clone());
        assert_eq!(vad.config.energy_threshold, 0.05);
        assert_eq!(vad.config.min_speech_duration, 0.2);
    }

    #[test]
    fn test_energy_calculation() {
        let _vad = VoiceActivityDetector::new();

        // Test with silence
        let silence = vec![0.0; 1000];
        let energy = VoiceActivityDetector::calculate_energy(&silence);
        assert_eq!(energy, 0.0);

        // Test with signal
        let signal = vec![0.5; 1000];
        let energy = VoiceActivityDetector::calculate_energy(&signal);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let _vad = VoiceActivityDetector::new();

        // Test with constant signal (no crossings)
        let constant = vec![0.5; 1000];
        let zcr = VoiceActivityDetector::calculate_zero_crossing_rate(&constant);
        assert_eq!(zcr, 0.0);

        // Test with alternating signal (maximum crossings)
        let alternating: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let zcr = VoiceActivityDetector::calculate_zero_crossing_rate(&alternating);
        assert!(zcr > 0.5); // Should be close to 1.0
    }

    #[tokio::test]
    async fn test_frame_detection() {
        let mut vad = VoiceActivityDetector::new();

        // Test with silence
        let silence_audio = AudioBuffer::new(vec![0.0; 1000], 16000, 1);
        let result = vad.detect_frame(&silence_audio, 0.0).unwrap();
        assert!(!result.voice_detected);
        assert_eq!(result.energy, 0.0);

        // Test with signal
        let signal_audio = AudioBuffer::new(vec![0.1; 1000], 16000, 1);
        let result = vad.detect_frame(&signal_audio, 0.0).unwrap();
        assert!(result.voice_detected);
        assert!(result.energy > 0.0);
    }

    #[tokio::test]
    async fn test_segment_detection() {
        let mut vad = VoiceActivityDetector::with_config(VADConfig {
            energy_threshold: 0.05,
            min_speech_duration: 0.1,
            ..Default::default()
        });

        // Create audio with speech segments
        let mut samples = Vec::new();

        // Silence
        samples.extend(vec![0.0; 1600]); // 0.1s at 16kHz

        // Speech
        samples.extend(vec![0.1; 3200]); // 0.2s at 16kHz

        // Silence
        samples.extend(vec![0.0; 1600]); // 0.1s at 16kHz

        let audio = AudioBuffer::new(samples, 16000, 1);
        let segments = vad.detect_segments(&audio).unwrap();

        // Should detect one speech segment
        assert!(!segments.is_empty());

        if !segments.is_empty() {
            let segment = &segments[0];
            assert!(segment.duration >= 0.1);
            assert!(segment.confidence > 0.0);
        }
    }
}
