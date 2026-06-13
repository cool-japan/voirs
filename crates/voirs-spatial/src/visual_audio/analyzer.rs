//! Audio analysis components for visual effect generation

use super::mapping::AudioVisualMapping;
use super::types::{ColorRGBA, FrequencyBand, VisualEvent, VisualEventType};
use crate::{types::AudioChannel, Result};
use std::time::{Duration, Instant};

/// Audio analysis for visual effect generation
pub(crate) struct VisualAudioAnalyzer {
    /// FFT analysis for frequency content
    fft_analyzer: FftAnalyzer,

    /// Onset detection for visual triggers
    onset_detector: OnsetDetector,

    /// Beat detection for rhythm visuals
    beat_detector: VisualBeatDetector,

    /// Spectral analysis for color mapping
    spectral_analyzer: SpectralAnalyzer,

    /// Amplitude tracking
    amplitude_tracker: AmplitudeTracker,
}

/// FFT analysis for frequency-based visuals
struct FftAnalyzer {
    /// FFT window size
    window_size: usize,

    /// Frequency bins
    frequency_bins: Vec<f32>,

    /// Previous frame for comparison
    previous_frame: Vec<f32>,

    /// Smoothing factor
    smoothing_factor: f32,
}

/// Audio onset detection for visual triggers
struct OnsetDetector {
    /// Spectral flux history
    flux_history: Vec<Vec<f32>>,

    /// Detection threshold
    threshold: f32,

    /// Peak picking parameters
    peak_picking: PeakPickingParams,

    /// Last onset time
    last_onset: Instant,
}

/// Peak picking parameters for onset detection
#[derive(Debug, Clone)]
struct PeakPickingParams {
    /// Minimum time between onsets
    min_interval: Duration,

    /// Threshold adaptation rate
    adaptation_rate: f32,

    /// Pre/post-roll for peak validation
    validation_window: usize,
}

/// Beat detection for rhythm-based visuals
struct VisualBeatDetector {
    /// Energy-based beat tracker
    energy_tracker: Vec<f32>,

    /// Tempo estimation
    tempo_estimator: TempoEstimator,

    /// Beat phase tracking
    phase_tracker: PhaseTracker,

    /// Confidence scoring
    confidence_tracker: ConfidenceTracker,
}

/// Tempo estimation for beat detection
#[derive(Debug)]
struct TempoEstimator {
    /// Autocorrelation buffer
    autocorr_buffer: Vec<f32>,

    /// Current tempo estimate (BPM)
    current_tempo: f32,

    /// Tempo stability measure
    stability: f32,

    /// Valid tempo range
    tempo_range: (f32, f32),
}

/// Phase tracking for beat alignment
#[derive(Debug)]
struct PhaseTracker {
    /// Current beat phase (0.0-1.0)
    current_phase: f32,

    /// Phase prediction
    predicted_phase: f32,

    /// Phase error tracking
    phase_error: f32,

    /// Phase correction factor
    correction_factor: f32,
}

/// Confidence tracking for beat detection
#[derive(Debug)]
struct ConfidenceTracker {
    /// Beat detection confidence
    beat_confidence: f32,

    /// Tempo confidence
    tempo_confidence: f32,

    /// Overall confidence
    overall_confidence: f32,

    /// Confidence history
    confidence_history: Vec<f32>,
}

/// Spectral analysis for advanced visual effects
struct SpectralAnalyzer {
    /// Spectral centroid tracking
    centroid_tracker: Vec<f32>,

    /// Spectral rolloff tracking
    rolloff_tracker: Vec<f32>,

    /// Spectral flux calculation
    flux_calculator: FluxCalculator,

    /// Harmonic analysis
    harmonic_analyzer: HarmonicAnalyzer,
}

/// Spectral flux calculation
#[derive(Debug)]
struct FluxCalculator {
    /// Previous spectrum
    previous_spectrum: Vec<f32>,

    /// Flux values
    flux_values: Vec<f32>,

    /// Smoothing window
    smoothing_window: usize,
}

/// Harmonic content analysis
#[derive(Debug)]
struct HarmonicAnalyzer {
    /// Fundamental frequency tracker
    f0_tracker: Vec<f32>,

    /// Harmonic strength
    harmonic_strength: Vec<f32>,

    /// Inharmonicity measure
    inharmonicity: f32,
}

/// Amplitude tracking for visual scaling
struct AmplitudeTracker {
    /// RMS amplitude history
    rms_history: Vec<f32>,

    /// Peak amplitude history
    peak_history: Vec<f32>,

    /// Dynamic range tracking
    dynamic_range: f32,

    /// Envelope following
    envelope_follower: EnvelopeFollower,
}

/// Envelope follower for smooth amplitude tracking
#[derive(Debug)]
struct EnvelopeFollower {
    /// Attack time constant
    attack_time: f32,

    /// Release time constant
    release_time: f32,

    /// Current envelope value
    current_value: f32,

    /// Sample rate
    sample_rate: f32,
}

/// Audio onset event
#[derive(Debug)]
pub(crate) struct OnsetEvent {
    pub(crate) strength: f32,
    pub(crate) flux_value: f32,
}

/// Beat detection event
#[derive(Debug)]
pub(crate) struct BeatEvent {
    pub(crate) strength: f32,
    pub(crate) is_downbeat: bool,
    pub(crate) confidence: f32,
}

impl VisualAudioAnalyzer {
    pub(crate) fn new() -> Self {
        Self {
            fft_analyzer: FftAnalyzer::new(1024),
            onset_detector: OnsetDetector::new(),
            beat_detector: VisualBeatDetector::new(),
            spectral_analyzer: SpectralAnalyzer::new(),
            amplitude_tracker: AmplitudeTracker::new(),
        }
    }

    pub(crate) fn analyze_frame(
        &mut self,
        audio_samples: &[f32],
        audio_channel_type: AudioChannel,
        mapping: &AudioVisualMapping,
    ) -> Result<Vec<VisualEvent>> {
        let mut events = Vec::new();

        // Perform FFT analysis
        self.fft_analyzer.analyze(audio_samples)?;

        // Detect onsets
        if let Some(onset) = self
            .onset_detector
            .detect(&self.fft_analyzer.frequency_bins)?
        {
            events.push(VisualEvent {
                source_id: format!("audio_{audio_channel_type:?}"),
                event_type: VisualEventType::Onset,
                intensity: onset.strength,
                color_hint: None,
                timestamp: Instant::now(),
            });
        }

        // Detect beats
        if let Some(beat) = self
            .beat_detector
            .detect(&self.fft_analyzer.frequency_bins)?
        {
            let event_type = if beat.is_downbeat {
                VisualEventType::Downbeat
            } else {
                VisualEventType::Beat
            };

            events.push(VisualEvent {
                source_id: format!("audio_{audio_channel_type:?}"),
                event_type,
                intensity: beat.strength,
                color_hint: None,
                timestamp: Instant::now(),
            });
        }

        // Analyze frequency bands
        self.analyze_frequency_bands(mapping, &audio_channel_type, &mut events)?;

        Ok(events)
    }

    fn analyze_frequency_bands(
        &self,
        mapping: &AudioVisualMapping,
        audio_channel_type: &AudioChannel,
        events: &mut Vec<VisualEvent>,
    ) -> Result<()> {
        let mappings = [
            (&mapping.low_freq_mapping, FrequencyBand::Low),
            (&mapping.mid_freq_mapping, FrequencyBand::Mid),
            (&mapping.high_freq_mapping, FrequencyBand::High),
        ];

        for (frequency_mapping, band) in mappings {
            let band_energy = self
                .fft_analyzer
                .calculate_band_energy(&frequency_mapping.frequency_range);

            if band_energy > 0.1 {
                // Threshold
                events.push(VisualEvent {
                    source_id: format!("audio_{audio_channel_type:?}"),
                    event_type: VisualEventType::FrequencyBand(band),
                    intensity: band_energy * frequency_mapping.intensity_scale,
                    color_hint: Some(frequency_mapping.base_color),
                    timestamp: Instant::now(),
                });
            }
        }

        Ok(())
    }
}

impl FftAnalyzer {
    fn new(window_size: usize) -> Self {
        Self {
            window_size,
            frequency_bins: vec![0.0; window_size / 2],
            previous_frame: vec![0.0; window_size / 2],
            smoothing_factor: 0.7,
        }
    }

    fn analyze(&mut self, samples: &[f32]) -> Result<()> {
        let n = self.window_size;

        // Store previous frame (used by spectral-flux based onset detection).
        self.previous_frame.copy_from_slice(&self.frequency_bins);

        // Real-FFT magnitude spectrum with a Hann window.
        //
        // The samples are windowed to reduce spectral leakage, transformed with
        // `rfft` (yielding `n / 2 + 1` complex bins), and the per-bin magnitude
        // `|X[k]|` is smoothed into the `n / 2` frequency-bin buffer (the Nyquist
        // bin is dropped to match the downstream bin count).
        let denom = n.saturating_sub(1).max(1) as f64;
        let mut windowed = vec![0.0_f64; n];
        for (i, (slot, &sample)) in windowed.iter_mut().zip(samples.iter()).enumerate() {
            let hann = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / denom).cos());
            *slot = sample as f64 * hann;
        }

        let spectrum = scirs2_fft::rfft(&windowed, Some(n))
            .map_err(|e| crate::Error::LegacyProcessing(format!("FFT error: {e}")))?;

        for (bin, value) in self.frequency_bins.iter_mut().zip(spectrum.iter()) {
            let magnitude = value.norm() as f32;
            *bin = self.smoothing_factor * (*bin) + (1.0 - self.smoothing_factor) * magnitude;
        }

        Ok(())
    }

    fn calculate_band_energy(&self, frequency_range: &(f32, f32)) -> f32 {
        let start_bin = (frequency_range.0 / 20000.0 * self.frequency_bins.len() as f32) as usize;
        let end_bin = (frequency_range.1 / 20000.0 * self.frequency_bins.len() as f32) as usize;

        self.frequency_bins[start_bin..end_bin.min(self.frequency_bins.len())]
            .iter()
            .sum::<f32>()
            / (end_bin - start_bin) as f32
    }
}

impl OnsetDetector {
    fn new() -> Self {
        Self {
            flux_history: Vec::with_capacity(50),
            threshold: 0.3,
            peak_picking: PeakPickingParams {
                min_interval: Duration::from_millis(100),
                adaptation_rate: 0.9,
                validation_window: 3,
            },
            last_onset: Instant::now(),
        }
    }

    fn detect(&mut self, frequency_bins: &[f32]) -> Result<Option<OnsetEvent>> {
        // Calculate spectral flux
        if !self.flux_history.is_empty() {
            let last_frame = &self.flux_history[self.flux_history.len() - 1];
            let flux: f32 = frequency_bins
                .iter()
                .zip(last_frame.iter())
                .map(|(current, previous)| (current - previous).max(0.0))
                .sum();

            // Adaptive threshold - calculate average across all frames
            let total_bins: usize = self.flux_history.iter().map(|frame| frame.len()).sum();
            let total_flux: f32 = self.flux_history.iter().flatten().sum();
            let avg_flux = if total_bins > 0 {
                total_flux / total_bins as f32
            } else {
                0.0
            };
            let adaptive_threshold = self.threshold + avg_flux * 0.5;

            if flux > adaptive_threshold {
                let now = Instant::now();
                if now.duration_since(self.last_onset) >= self.peak_picking.min_interval {
                    self.last_onset = now;
                    return Ok(Some(OnsetEvent {
                        strength: flux.min(1.0),
                        flux_value: flux,
                    }));
                }
            }
        }

        // Store current frame for next comparison
        self.flux_history.push(frequency_bins.to_vec());
        if self.flux_history.len() > 50 {
            self.flux_history.remove(0);
        }

        Ok(None)
    }
}

impl VisualBeatDetector {
    fn new() -> Self {
        Self {
            energy_tracker: Vec::with_capacity(100),
            tempo_estimator: TempoEstimator {
                autocorr_buffer: vec![0.0; 200],
                current_tempo: 120.0,
                stability: 0.0,
                tempo_range: (60.0, 180.0),
            },
            phase_tracker: PhaseTracker {
                current_phase: 0.0,
                predicted_phase: 0.0,
                phase_error: 0.0,
                correction_factor: 0.1,
            },
            confidence_tracker: ConfidenceTracker {
                beat_confidence: 0.0,
                tempo_confidence: 0.0,
                overall_confidence: 0.0,
                confidence_history: Vec::with_capacity(50),
            },
        }
    }

    fn detect(&mut self, frequency_bins: &[f32]) -> Result<Option<BeatEvent>> {
        // Calculate energy
        let energy = frequency_bins.iter().sum::<f32>() / frequency_bins.len() as f32;
        self.energy_tracker.push(energy);

        if self.energy_tracker.len() > 100 {
            self.energy_tracker.remove(0);
        }

        // Simple beat detection based on energy peaks
        if self.energy_tracker.len() >= 5 {
            let len = self.energy_tracker.len();
            let current = self.energy_tracker[len - 1];
            let recent_avg = self.energy_tracker[len - 5..].iter().sum::<f32>() / 5.0;

            if current > recent_avg * 1.3 && current > 0.3 {
                return Ok(Some(BeatEvent {
                    strength: current,
                    is_downbeat: self.phase_tracker.current_phase < 0.25,
                    confidence: self.confidence_tracker.overall_confidence,
                }));
            }
        }

        Ok(None)
    }
}

impl SpectralAnalyzer {
    fn new() -> Self {
        Self {
            centroid_tracker: Vec::with_capacity(50),
            rolloff_tracker: Vec::with_capacity(50),
            flux_calculator: FluxCalculator {
                previous_spectrum: vec![0.0; 512],
                flux_values: Vec::with_capacity(50),
                smoothing_window: 5,
            },
            harmonic_analyzer: HarmonicAnalyzer {
                f0_tracker: Vec::with_capacity(50),
                harmonic_strength: vec![0.0; 10],
                inharmonicity: 0.0,
            },
        }
    }
}

impl AmplitudeTracker {
    fn new() -> Self {
        Self {
            rms_history: Vec::with_capacity(100),
            peak_history: Vec::with_capacity(100),
            dynamic_range: 0.0,
            envelope_follower: EnvelopeFollower {
                attack_time: 0.01,
                release_time: 0.1,
                current_value: 0.0,
                sample_rate: 44100.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magnitude_spectrum_peaks_at_tone_frequency() {
        let window_size = 1024;
        let mut analyzer = FftAnalyzer::new(window_size);

        // Pure sinusoid completing exactly `k0` cycles across the window, so its
        // energy lands on bin `k0` of the FFT.
        let k0 = 32_usize;
        let samples: Vec<f32> = (0..window_size)
            .map(|i| (2.0 * std::f32::consts::PI * k0 as f32 * i as f32 / window_size as f32).sin())
            .collect();

        analyzer.analyze(&samples).expect("analysis should succeed");

        // The dominant bin must be at (or immediately adjacent to) k0; Hann
        // windowing spreads a little energy into neighbouring bins.
        let (peak_bin, _) = analyzer
            .frequency_bins
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("frequency bins must be non-empty");

        assert!(
            (peak_bin as i64 - k0 as i64).abs() <= 1,
            "spectral peak at bin {peak_bin}, expected ~{k0}"
        );
    }

    #[test]
    fn test_magnitude_spectrum_distinguishes_low_and_high_tones() {
        let window_size = 1024;

        let analyze_tone = |k0: usize| -> Vec<f32> {
            let mut analyzer = FftAnalyzer::new(window_size);
            let samples: Vec<f32> = (0..window_size)
                .map(|i| {
                    (2.0 * std::f32::consts::PI * k0 as f32 * i as f32 / window_size as f32).sin()
                })
                .collect();
            analyzer.analyze(&samples).expect("analysis should succeed");
            analyzer.frequency_bins.clone()
        };

        let low = analyze_tone(16);
        let high = analyze_tone(200);

        let argmax = |bins: &[f32]| -> usize {
            bins.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        assert!(argmax(&low) < argmax(&high));
    }
}
