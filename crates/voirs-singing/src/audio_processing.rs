//! Advanced audio processing for singing synthesis
//!
//! This module provides professional-grade audio processing including
//! high-quality resampling, phase coherence, stereo imaging, and dynamic range processing.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// High-quality resampling processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighQualityResampler {
    /// Source sample rate
    pub source_rate: f32,
    /// Target sample rate
    pub target_rate: f32,
    /// Interpolation method
    pub interpolation: InterpolationMethod,
    /// Anti-aliasing filter configuration
    pub anti_aliasing: AntiAliasingConfig,
    /// Quality vs speed tradeoff
    pub quality_level: QualityLevel,
    /// Internal buffer for processing
    #[serde(skip)]
    internal_buffer: VecDeque<f32>,
    /// Filter coefficients
    #[serde(skip)]
    filter_coefficients: Vec<f32>,
}

/// Phase coherence processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseCoherenceProcessor {
    /// Reference phase tracking
    pub reference_phase: f32,
    /// Phase correction strength (0.0-1.0)
    pub correction_strength: f32,
    /// Frequency-dependent phase correction
    pub frequency_weights: Vec<(f32, f32)>, // (frequency, weight) pairs
    /// Phase alignment method
    pub alignment_method: PhaseAlignmentMethod,
    /// Harmonic phase relationships
    pub harmonic_relationships: HarmonicPhaseConfig,
    /// Cross-correlation analysis
    pub correlation_analysis: CorrelationConfig,
}

/// Stereo imaging processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StereoImagingProcessor {
    /// Stereo width (0.0 = mono, 1.0 = normal, >1.0 = enhanced)
    pub stereo_width: f32,
    /// Pan law configuration
    pub pan_law: PanLaw,
    /// Multi-voice positioning
    pub voice_positioning: VoicePositioning,
    /// Ambience and space simulation
    pub spatial_simulation: SpatialSimulation,
    /// Stereo enhancement
    pub stereo_enhancement: StereoEnhancement,
    /// HRTF processing for binaural
    pub hrtf_processing: Option<HrtfConfig>,
}

/// Dynamic range processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRangeProcessor {
    /// Compressor configuration
    pub compressor: CompressorConfig,
    /// Limiter configuration
    pub limiter: LimiterConfig,
    /// Gate configuration
    pub gate: GateConfig,
    /// Expander configuration
    pub expander: ExpanderConfig,
    /// Multi-band processing
    pub multiband: MultibandConfig,
    /// Preserve natural dynamics
    pub natural_preservation: NaturalDynamicsConfig,
}

// === Supporting Types ===

/// Interpolation method for resampling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Linear interpolation (fast, lower quality)
    Linear,
    /// Cubic interpolation (balanced quality and speed)
    Cubic,
    /// Sinc interpolation (high quality)
    Sinc,
    /// Lanczos resampling (high quality with windowing)
    Lanczos,
    /// Kaiser windowed sinc (excellent quality, configurable)
    Kaiser,
    /// Blackman windowed sinc (very high quality)
    Blackman,
}

/// Anti-aliasing filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiAliasingConfig {
    /// Cutoff frequency as ratio of Nyquist frequency
    pub cutoff_ratio: f32,
    /// Filter order
    pub filter_order: u32,
    /// Filter type
    pub filter_type: FilterType,
    /// Transition band width
    pub transition_width: f32,
    /// Stop-band attenuation (dB)
    pub stopband_attenuation: f32,
}

/// Digital filter type for anti-aliasing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterType {
    /// Butterworth filter (maximally flat passband)
    Butterworth,
    /// Chebyshev Type I filter (passband ripple)
    Chebyshev1,
    /// Chebyshev Type II filter (stopband ripple)
    Chebyshev2,
    /// Elliptic filter (ripple in both bands, steepest rolloff)
    Elliptic,
    /// Bessel filter (maximally flat group delay, linear phase)
    Bessel,
    /// Kaiser window filter (configurable characteristics)
    Kaiser,
}

/// Quality vs speed tradeoff level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Lower quality, higher speed
    Fast,
    /// Good balance of quality and speed
    Balanced,
    /// High quality, slower processing
    High,
    /// Maximum quality, slowest processing
    Maximum,
}

/// Method for aligning phase between audio channels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseAlignmentMethod {
    /// Cross-correlation based alignment (time domain)
    CrossCorrelation,
    /// Phase vocoder based alignment (frequency domain)
    PhaseVocoder,
    /// Simple time delay alignment
    TimeDelay,
    /// Frequency domain analysis and correction
    FrequencyDomain,
    /// Hybrid time and frequency domain approach
    Hybrid,
}

/// Configuration for maintaining harmonic phase relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicPhaseConfig {
    /// Maintain harmonic phase relationships
    pub maintain_harmonics: bool,
    /// Fundamental frequency tracking
    pub fundamental_tracking: bool,
    /// Phase coherence across partials
    pub partial_coherence: f32,
    /// Harmonic phase offsets
    pub phase_offsets: Vec<f32>,
}

/// Configuration for cross-correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    /// Window size for correlation analysis
    pub window_size: u32,
    /// Overlap factor
    pub overlap_factor: f32,
    /// Correlation threshold for alignment
    pub correlation_threshold: f32,
    /// Maximum delay search range (samples)
    pub max_delay_range: u32,
}

/// Pan law for stereo positioning
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PanLaw {
    /// -3dB pan law (most common)
    Minus3dB,
    /// -4.5dB pan law
    Minus4_5dB,
    /// -6dB pan law (linear)
    Minus6dB,
    /// Equal power panning
    EqualPower,
    /// Custom pan law
    Custom(f32),
}

/// Configuration for positioning multiple voices in stereo field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicePositioning {
    /// Automatic voice spacing
    pub auto_spacing: bool,
    /// Voice positions (-1.0 to 1.0)
    pub voice_positions: Vec<f32>,
    /// Distance simulation
    pub distance_simulation: DistanceSimulation,
    /// Voice spread (apparent size of each voice)
    pub voice_spread: f32,
    /// Movement and modulation
    pub position_modulation: PositionModulation,
}

/// Configuration for simulating distance cues in audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceSimulation {
    /// Enable distance simulation
    pub enabled: bool,
    /// Reference distance (meters)
    pub reference_distance: f32,
    /// Distance per voice
    pub voice_distances: Vec<f32>,
    /// Air absorption simulation
    pub air_absorption: bool,
    /// Distance-dependent filtering
    pub distance_filtering: bool,
}

/// Configuration for dynamic position modulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionModulation {
    /// Enable position movement
    pub enabled: bool,
    /// Movement speed
    pub movement_speed: f32,
    /// Movement pattern
    pub movement_pattern: MovementPattern,
    /// Modulation depth
    pub modulation_depth: f32,
}

/// Pattern for spatial position movement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MovementPattern {
    /// Circular movement pattern
    Circular,
    /// Linear back-and-forth movement
    Linear,
    /// Figure-8 movement pattern
    Figure8,
    /// Random wandering movement
    Random,
    /// Slight movement like choir breathing
    Breathing,
}

/// Configuration for spatial room simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialSimulation {
    /// Room size simulation (width, depth, height in meters)
    pub room_size: (f32, f32, f32),
    /// Early reflections
    pub early_reflections: EarlyReflectionConfig,
    /// Late reverb
    pub late_reverb: LateReverbConfig,
    /// Source positions in room (x, y, z coordinates)
    pub source_positions: Vec<(f32, f32, f32)>,
    /// Listener position (x, y, z coordinates)
    pub listener_position: (f32, f32, f32),
}

/// Configuration for early reflections in room simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyReflectionConfig {
    /// Enable early reflections
    pub enabled: bool,
    /// Reflection density
    pub density: f32,
    /// Reflection level
    pub level: f32,
    /// Reflection delay spread (seconds)
    pub delay_spread: f32,
    /// High frequency damping
    pub hf_damping: f32,
}

/// Configuration for late reverb tail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateReverbConfig {
    /// Reverb time (RT60 in seconds)
    pub reverb_time: f32,
    /// High frequency decay ratio
    pub hf_decay_ratio: f32,
    /// Diffusion amount
    pub diffusion: f32,
    /// Echo density
    pub density: f32,
    /// Wet/dry mix level
    pub wet_level: f32,
}

/// Configuration for stereo field enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StereoEnhancement {
    /// Bass enhancement for center content
    pub bass_enhancement: f32,
    /// Mid-side processing configuration
    pub mid_side_processing: MidSideConfig,
    /// Harmonic enhancement amount
    pub harmonic_enhancement: f32,
    /// Stereo exciter amount
    pub exciter_amount: f32,
}

/// Configuration for mid-side stereo processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidSideConfig {
    /// Enable M/S processing
    pub enabled: bool,
    /// Mid channel gain
    pub mid_gain: f32,
    /// Side channel gain
    pub side_gain: f32,
    /// Mid channel EQ
    pub mid_eq: EqConfig,
    /// Side channel EQ
    pub side_eq: EqConfig,
}

/// Configuration for parametric equalizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqConfig {
    /// Enable EQ
    pub enabled: bool,
    /// EQ bands
    pub bands: Vec<EqBand>,
}

/// Single band in a parametric equalizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqBand {
    /// Center frequency (Hz)
    pub frequency: f32,
    /// Gain in dB
    pub gain: f32,
    /// Q factor (bandwidth)
    pub q: f32,
    /// Band type
    pub band_type: EqBandType,
}

/// Type of equalizer band filter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EqBandType {
    /// Peak/notch filter (bell curve)
    Peak,
    /// High shelf filter (boost/cut high frequencies)
    HighShelf,
    /// Low shelf filter (boost/cut low frequencies)
    LowShelf,
    /// High-pass filter (attenuate low frequencies)
    HighPass,
    /// Low-pass filter (attenuate high frequencies)
    LowPass,
    /// Band-pass filter (only pass certain frequency range)
    BandPass,
    /// Notch filter (reject narrow frequency range)
    Notch,
}

/// Configuration for Head-Related Transfer Function (HRTF) processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrtfConfig {
    /// HRTF database selection
    pub database: HrtfDatabase,
    /// Head size adjustment factor
    pub head_size: f32,
    /// Ear spacing adjustment (meters)
    pub ear_spacing: f32,
    /// Individual HRTF customization
    pub custom_hrtf: Option<CustomHrtfData>,
}

/// HRTF database selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HrtfDatabase {
    /// MIT KEMAR database
    MIT,
    /// UC Davis CIPIC database
    CIPIC,
    /// Austrian Research Institute database
    ARI,
    /// Generic/average HRTF
    Generic,
    /// User-provided HRTF
    Custom,
}

/// Custom HRTF impulse response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomHrtfData {
    /// Left ear impulse responses for each direction
    pub left_ear_irs: Vec<Vec<f32>>,
    /// Right ear impulse responses for each direction
    pub right_ear_irs: Vec<Vec<f32>>,
    /// Azimuth angles (degrees)
    pub azimuth_angles: Vec<f32>,
    /// Elevation angles (degrees)
    pub elevation_angles: Vec<f32>,
}

/// Configuration for dynamic range compressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorConfig {
    /// Enable compressor
    pub enabled: bool,
    /// Threshold level (dB)
    pub threshold: f32,
    /// Compression ratio (e.g., 4.0 = 4:1)
    pub ratio: f32,
    /// Attack time (milliseconds)
    pub attack: f32,
    /// Release time (milliseconds)
    pub release: f32,
    /// Knee width (dB)
    pub knee_width: f32,
    /// Makeup gain (dB)
    pub makeup_gain: f32,
    /// Sidechain filtering configuration
    pub sidechain_filter: Option<EqConfig>,
}

/// Configuration for peak limiter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimiterConfig {
    /// Enable limiter
    pub enabled: bool,
    /// Ceiling level (dB)
    pub ceiling: f32,
    /// Release time (milliseconds)
    pub release: f32,
    /// Lookahead time (milliseconds)
    pub lookahead: f32,
    /// ISR (Inter-Sample Peak) detection
    pub isr_detection: bool,
}

/// Configuration for noise gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateConfig {
    /// Enable gate
    pub enabled: bool,
    /// Threshold level (dB)
    pub threshold: f32,
    /// Ratio (for expander-style gating)
    pub ratio: f32,
    /// Attack time (milliseconds)
    pub attack: f32,
    /// Hold time (milliseconds)
    pub hold: f32,
    /// Release time (milliseconds)
    pub release: f32,
}

/// Configuration for dynamic range expander
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpanderConfig {
    /// Enable expander
    pub enabled: bool,
    /// Threshold level (dB)
    pub threshold: f32,
    /// Expansion ratio
    pub ratio: f32,
    /// Attack time (milliseconds)
    pub attack: f32,
    /// Release time (milliseconds)
    pub release: f32,
    /// Knee width (dB)
    pub knee_width: f32,
}

/// Configuration for multiband dynamic processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultibandConfig {
    /// Enable multiband processing
    pub enabled: bool,
    /// Crossover frequencies (Hz)
    pub crossover_frequencies: Vec<f32>,
    /// Per-band compressors
    pub band_compressors: Vec<CompressorConfig>,
    /// Per-band EQs
    pub band_eqs: Vec<EqConfig>,
    /// Crossover slopes (dB/octave)
    pub crossover_slopes: Vec<f32>,
}

/// Configuration for preserving natural vocal dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalDynamicsConfig {
    /// Preserve micro-dynamics
    pub preserve_micro_dynamics: bool,
    /// Transient preservation amount (0.0-1.0)
    pub transient_preservation: f32,
    /// Breath dynamics preservation amount (0.0-1.0)
    pub breath_preservation: f32,
    /// Musical phrase shaping amount (0.0-1.0)
    pub phrase_shaping: f32,
    /// Dynamic range target (dB)
    pub target_dynamic_range: f32,
}

// === Implementation ===

impl HighQualityResampler {
    /// Create a new high-quality resampler
    pub fn new(source_rate: f32, target_rate: f32) -> Self {
        let mut resampler = Self {
            source_rate,
            target_rate,
            interpolation: InterpolationMethod::Sinc,
            anti_aliasing: AntiAliasingConfig::default(),
            quality_level: QualityLevel::High,
            internal_buffer: VecDeque::new(),
            filter_coefficients: Vec::new(),
        };

        resampler.initialize_filter();
        resampler
    }

    /// Create a fast resampler for real-time use
    pub fn fast(source_rate: f32, target_rate: f32) -> Self {
        Self {
            source_rate,
            target_rate,
            interpolation: InterpolationMethod::Linear,
            quality_level: QualityLevel::Fast,
            ..Self::new(source_rate, target_rate)
        }
    }

    /// Create maximum quality resampler
    pub fn maximum_quality(source_rate: f32, target_rate: f32) -> Self {
        Self {
            source_rate,
            target_rate,
            interpolation: InterpolationMethod::Kaiser,
            quality_level: QualityLevel::Maximum,
            anti_aliasing: AntiAliasingConfig {
                cutoff_ratio: 0.95,
                filter_order: 64,
                filter_type: FilterType::Kaiser,
                transition_width: 0.05,
                stopband_attenuation: 120.0,
            },
            ..Self::new(source_rate, target_rate)
        }
    }

    /// Initialize the anti-aliasing filter
    fn initialize_filter(&mut self) {
        let nyquist =
            (self.source_rate.min(self.target_rate) / 2.0) * self.anti_aliasing.cutoff_ratio;
        let normalized_cutoff = nyquist / (self.source_rate / 2.0);

        self.filter_coefficients = match self.anti_aliasing.filter_type {
            FilterType::Kaiser => self.design_kaiser_filter(normalized_cutoff),
            FilterType::Butterworth => self.design_butterworth_filter(normalized_cutoff),
            _ => self.design_sinc_filter(normalized_cutoff),
        };
    }

    /// Design Kaiser filter
    fn design_kaiser_filter(&self, cutoff: f32) -> Vec<f32> {
        let order = self.anti_aliasing.filter_order as usize;
        let mut coefficients = Vec::with_capacity(order);

        let beta = self.kaiser_beta_from_attenuation(self.anti_aliasing.stopband_attenuation);

        for n in 0..order {
            let x = 2.0 * n as f32 / (order - 1) as f32 - 1.0;
            let kaiser_window = self.kaiser_window(x, beta);
            let sinc_value = if n == order / 2 {
                2.0 * cutoff
            } else {
                let arg = 2.0 * std::f32::consts::PI * cutoff * (n as f32 - order as f32 / 2.0);
                2.0 * cutoff * (arg.sin() / arg)
            };
            coefficients.push(sinc_value * kaiser_window);
        }

        // Normalize coefficients
        let sum: f32 = coefficients.iter().sum();
        coefficients.iter_mut().for_each(|c| *c /= sum);

        coefficients
    }

    /// Design Butterworth filter
    fn design_butterworth_filter(&self, cutoff: f32) -> Vec<f32> {
        // Simplified Butterworth filter design
        // In practice, this would be more sophisticated
        let order = self.anti_aliasing.filter_order.min(8) as usize;
        let mut coefficients = Vec::with_capacity(order + 1);

        for n in 0..=order {
            let angle = std::f32::consts::PI * cutoff * n as f32;
            let coeff = if n == 0 {
                cutoff
            } else {
                cutoff * angle.sin() / angle
            };
            coefficients.push(coeff);
        }

        coefficients
    }

    /// Design sinc filter
    fn design_sinc_filter(&self, cutoff: f32) -> Vec<f32> {
        let order = self.anti_aliasing.filter_order as usize;
        let mut coefficients = Vec::with_capacity(order);

        for n in 0..order {
            let x = n as f32 - order as f32 / 2.0;
            let coeff = if x == 0.0 {
                2.0 * cutoff
            } else {
                let arg = 2.0 * std::f32::consts::PI * cutoff * x;
                2.0 * cutoff * (arg.sin() / arg)
            };
            coefficients.push(coeff);
        }

        coefficients
    }

    /// Calculate Kaiser beta parameter from stopband attenuation
    fn kaiser_beta_from_attenuation(&self, attenuation_db: f32) -> f32 {
        if attenuation_db >= 50.0 {
            0.1102 * (attenuation_db - 8.7)
        } else if attenuation_db >= 21.0 {
            0.5842 * (attenuation_db - 21.0).powf(0.4) + 0.07886 * (attenuation_db - 21.0)
        } else {
            0.0
        }
    }

    /// Kaiser window function
    fn kaiser_window(&self, x: f32, beta: f32) -> f32 {
        let arg = beta * (1.0 - x.powi(2)).sqrt();
        self.modified_bessel_i0(arg) / self.modified_bessel_i0(beta)
    }

    /// Modified Bessel function of the first kind, order 0
    fn modified_bessel_i0(&self, x: f32) -> f32 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x_half_squared = (x / 2.0).powi(2);

        for k in 1..=50 {
            term *= x_half_squared / (k as f32).powi(2);
            sum += term;
            if term < 1e-10 * sum {
                break;
            }
        }

        sum
    }

    /// Resample audio buffer
    pub fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        let ratio = self.target_rate / self.source_rate;
        let output_length = (input.len() as f32 * ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_length);

        match self.interpolation {
            InterpolationMethod::Linear => self.linear_interpolate(input, &mut output, ratio),
            InterpolationMethod::Cubic => self.cubic_interpolate(input, &mut output, ratio),
            InterpolationMethod::Sinc | InterpolationMethod::Kaiser => {
                self.sinc_interpolate(input, &mut output, ratio)
            }
            _ => self.linear_interpolate(input, &mut output, ratio), // Fallback
        }

        output
    }

    /// Linear interpolation resampling
    fn linear_interpolate(&self, input: &[f32], output: &mut Vec<f32>, ratio: f32) {
        let output_length = (input.len() as f32 * ratio).ceil() as usize;

        for i in 0..output_length {
            let source_index = i as f32 / ratio;
            let index = source_index as usize;
            let fraction = source_index - index as f32;

            let sample = if index + 1 < input.len() {
                input[index] * (1.0 - fraction) + input[index + 1] * fraction
            } else if index < input.len() {
                input[index]
            } else {
                0.0
            };

            output.push(sample);
        }
    }

    /// Cubic interpolation resampling
    fn cubic_interpolate(&self, input: &[f32], output: &mut Vec<f32>, ratio: f32) {
        let output_length = (input.len() as f32 * ratio).ceil() as usize;

        for i in 0..output_length {
            let source_index = i as f32 / ratio;
            let index = source_index as usize;
            let fraction = source_index - index as f32;

            if index >= 1 && index + 2 < input.len() {
                let y0 = input[index - 1];
                let y1 = input[index];
                let y2 = input[index + 1];
                let y3 = input[index + 2];

                // Cubic Hermite interpolation
                let a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
                let b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                let c = -0.5 * y0 + 0.5 * y2;
                let d = y1;

                let sample = a * fraction.powi(3) + b * fraction.powi(2) + c * fraction + d;
                output.push(sample);
            } else {
                // Fallback to linear interpolation at boundaries
                let sample = if index + 1 < input.len() {
                    input[index] * (1.0 - fraction) + input[index + 1] * fraction
                } else if index < input.len() {
                    input[index]
                } else {
                    0.0
                };
                output.push(sample);
            }
        }
    }

    /// Sinc interpolation resampling (high quality)
    fn sinc_interpolate(&self, input: &[f32], output: &mut Vec<f32>, ratio: f32) {
        let filter_half_width = self.filter_coefficients.len() / 2;
        let output_length = (input.len() as f32 * ratio) as usize;

        for i in 0..output_length {
            let source_index = i as f32 / ratio;
            let mut sample = 0.0;

            for (j, &coeff) in self.filter_coefficients.iter().enumerate() {
                let input_index = source_index as i32 + j as i32 - filter_half_width as i32;
                if input_index >= 0 && (input_index as usize) < input.len() {
                    sample += input[input_index as usize] * coeff;
                }
            }

            output.push(sample);
        }
    }
}

impl Default for PhaseCoherenceProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseCoherenceProcessor {
    /// Create a new phase coherence processor
    pub fn new() -> Self {
        Self {
            reference_phase: 0.0,
            correction_strength: 0.8,
            frequency_weights: vec![
                (100.0, 1.0),  // Low frequencies
                (1000.0, 0.8), // Mid frequencies
                (5000.0, 0.6), // High frequencies
            ],
            alignment_method: PhaseAlignmentMethod::CrossCorrelation,
            harmonic_relationships: HarmonicPhaseConfig::default(),
            correlation_analysis: CorrelationConfig::default(),
        }
    }

    /// Process multiple audio channels for phase coherence
    pub fn process_channels(&mut self, channels: &mut [Vec<f32>]) {
        if channels.len() < 2 {
            return;
        }

        match self.alignment_method {
            PhaseAlignmentMethod::CrossCorrelation => self.cross_correlation_alignment(channels),
            PhaseAlignmentMethod::TimeDelay => self.time_delay_alignment(channels),
            _ => self.basic_phase_alignment(channels),
        }
    }

    /// Cross-correlation based phase alignment
    fn cross_correlation_alignment(&mut self, channels: &mut [Vec<f32>]) {
        let reference_channel = &channels[0].clone();

        for channel in channels.iter_mut().skip(1) {
            let delay = self.find_optimal_delay(reference_channel, channel);
            self.apply_delay_compensation(channel, delay);
        }
    }

    /// Find optimal delay using cross-correlation
    fn find_optimal_delay(&self, reference: &[f32], signal: &[f32]) -> i32 {
        let max_delay = self.correlation_analysis.max_delay_range as i32;
        let mut max_correlation = 0.0;
        let mut optimal_delay = 0;

        for delay in -max_delay..=max_delay {
            let correlation = self.compute_correlation(reference, signal, delay);
            if correlation > max_correlation {
                max_correlation = correlation;
                optimal_delay = delay;
            }
        }

        if max_correlation > self.correlation_analysis.correlation_threshold {
            optimal_delay
        } else {
            0 // No significant correlation found
        }
    }

    /// Compute cross-correlation at specific delay
    fn compute_correlation(&self, reference: &[f32], signal: &[f32], delay: i32) -> f32 {
        let window_size = self.correlation_analysis.window_size as usize;
        let start = delay.max(0) as usize;
        let end = (reference
            .len()
            .min(signal.len().saturating_sub(delay.unsigned_abs() as usize)))
        .min(start + window_size);

        if start >= end {
            return 0.0;
        }

        let mut correlation = 0.0;
        let mut ref_power = 0.0;
        let mut sig_power = 0.0;

        for i in start..end {
            let ref_idx = i;
            let sig_idx = if delay >= 0 {
                i
            } else {
                i + delay.unsigned_abs() as usize
            };

            if ref_idx < reference.len() && sig_idx < signal.len() {
                let ref_sample = reference[ref_idx];
                let sig_sample = signal[sig_idx];

                correlation += ref_sample * sig_sample;
                ref_power += ref_sample * ref_sample;
                sig_power += sig_sample * sig_sample;
            }
        }

        let normalization = (ref_power * sig_power).sqrt();
        if normalization > 0.0 {
            correlation / normalization
        } else {
            0.0
        }
    }

    /// Apply delay compensation to align phases
    fn apply_delay_compensation(&self, channel: &mut Vec<f32>, delay: i32) {
        if delay == 0 {
            return;
        }

        let correction = delay as f32 * self.correction_strength;
        let samples_delay = correction.round() as i32;

        if samples_delay > 0 {
            // Positive delay - insert zeros at beginning
            channel.splice(0..0, vec![0.0; samples_delay as usize]);
            channel.truncate(channel.len() - samples_delay as usize);
        } else if samples_delay < 0 {
            // Negative delay - remove samples from beginning
            let remove_count = (-samples_delay) as usize;
            if remove_count < channel.len() {
                channel.drain(0..remove_count);
                channel.extend(vec![0.0; remove_count]);
            }
        }
    }

    /// Time delay based phase alignment
    fn time_delay_alignment(&mut self, channels: &mut [Vec<f32>]) {
        // Simple time-based delay alignment
        for (i, channel) in channels.iter_mut().enumerate().skip(1) {
            let delay_samples = (i as f32 * 0.001 * 44100.0) as usize; // 1ms per channel
            if delay_samples < channel.len() {
                channel.drain(0..delay_samples);
                channel.extend(vec![0.0; delay_samples]);
            }
        }
    }

    /// Basic phase alignment
    fn basic_phase_alignment(&mut self, channels: &mut [Vec<f32>]) {
        // Apply simple phase correction based on reference phase
        for channel in channels.iter_mut().skip(1) {
            for sample in channel.iter_mut() {
                *sample *= self.correction_strength;
            }
        }
    }
}

impl Default for StereoImagingProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoImagingProcessor {
    /// Create a new stereo imaging processor
    pub fn new() -> Self {
        Self {
            stereo_width: 1.0,
            pan_law: PanLaw::Minus3dB,
            voice_positioning: VoicePositioning::default(),
            spatial_simulation: SpatialSimulation::default(),
            stereo_enhancement: StereoEnhancement::default(),
            hrtf_processing: None,
        }
    }

    /// Create wide stereo processor
    pub fn wide_stereo() -> Self {
        Self {
            stereo_width: 1.5,
            stereo_enhancement: StereoEnhancement {
                harmonic_enhancement: 0.3,
                exciter_amount: 0.2,
                ..StereoEnhancement::default()
            },
            ..Self::new()
        }
    }

    /// Process stereo audio with imaging
    pub fn process_stereo(&self, left: &mut [f32], right: &mut [f32]) {
        self.apply_stereo_width(left, right);
        self.apply_stereo_enhancement(left, right);

        if self.spatial_simulation.early_reflections.enabled {
            self.apply_spatial_simulation(left, right);
        }
    }

    /// Apply stereo width adjustment
    fn apply_stereo_width(&self, left: &mut [f32], right: &mut [f32]) {
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let mid = (*l + *r) * 0.5;
            let side = (*l - *r) * 0.5;

            let enhanced_side = side * self.stereo_width;

            *l = mid + enhanced_side;
            *r = mid - enhanced_side;
        }
    }

    /// Apply stereo enhancement
    fn apply_stereo_enhancement(&self, left: &mut [f32], right: &mut [f32]) {
        let enhancement = &self.stereo_enhancement;

        if enhancement.harmonic_enhancement > 0.0 {
            for (l, r) in left.iter_mut().zip(right.iter_mut()) {
                let harmonic_l = self.generate_harmonic(*l) * enhancement.harmonic_enhancement;
                let harmonic_r = self.generate_harmonic(*r) * enhancement.harmonic_enhancement;

                *l += harmonic_l;
                *r += harmonic_r;
            }
        }
    }

    /// Generate harmonic content for enhancement
    fn generate_harmonic(&self, sample: f32) -> f32 {
        // Simple harmonic generation using soft clipping
        let drive = 2.0;
        let driven = sample * drive;
        let clipped = driven.tanh();
        clipped - sample
    }

    /// Apply spatial simulation
    fn apply_spatial_simulation(&self, left: &mut [f32], right: &mut [f32]) {
        let early_refs = &self.spatial_simulation.early_reflections;

        if early_refs.enabled && early_refs.level > 0.0 {
            // Simple early reflection simulation
            let delay_samples = (early_refs.delay_spread * 44100.0) as usize;

            for i in delay_samples..left.len() {
                left[i] += left[i - delay_samples] * early_refs.level * early_refs.density;
                right[i] += right[i - delay_samples] * early_refs.level * early_refs.density;
            }
        }
    }

    /// Position voices in stereo field
    pub fn position_voices(&self, voices: &mut [Vec<f32>]) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut stereo_voices = Vec::new();

        for (i, voice) in voices.iter().enumerate() {
            let position = if i < self.voice_positioning.voice_positions.len() {
                self.voice_positioning.voice_positions[i]
            } else {
                // Auto-distribute voices across stereo field
                if voices.len() > 1 {
                    -1.0 + 2.0 * i as f32 / (voices.len() - 1) as f32
                } else {
                    0.0
                }
            };

            let (left, right) = self.pan_voice(voice, position);
            stereo_voices.push((left, right));
        }

        stereo_voices
    }

    /// Pan a single voice to stereo position
    fn pan_voice(&self, voice: &[f32], position: f32) -> (Vec<f32>, Vec<f32>) {
        let position_clamped = position.clamp(-1.0, 1.0);
        let (left_gain, right_gain) = self.calculate_pan_gains(position_clamped);

        let left: Vec<f32> = voice.iter().map(|&sample| sample * left_gain).collect();
        let right: Vec<f32> = voice.iter().map(|&sample| sample * right_gain).collect();

        (left, right)
    }

    /// Calculate pan gains based on pan law
    fn calculate_pan_gains(&self, position: f32) -> (f32, f32) {
        let angle = (position + 1.0) * 0.5 * std::f32::consts::PI * 0.5; // 0 to π/2

        match self.pan_law {
            PanLaw::Minus3dB => {
                let left = angle.cos();
                let right = (std::f32::consts::PI * 0.5 - angle).cos();
                (left, right)
            }
            PanLaw::EqualPower => {
                let left = ((1.0 - position) * 0.5).sqrt();
                let right = ((1.0 + position) * 0.5).sqrt();
                (left, right)
            }
            PanLaw::Minus6dB => {
                let left = (1.0 - position) * 0.5;
                let right = (1.0 + position) * 0.5;
                (left, right)
            }
            PanLaw::Custom(factor) => {
                let left = ((1.0 - position) * 0.5).powf(factor);
                let right = ((1.0 + position) * 0.5).powf(factor);
                (left, right)
            }
            _ => (0.5, 0.5), // Fallback
        }
    }
}

impl Default for DynamicRangeProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicRangeProcessor {
    /// Create a new dynamic range processor
    pub fn new() -> Self {
        Self {
            compressor: CompressorConfig::default(),
            limiter: LimiterConfig::default(),
            gate: GateConfig::default(),
            expander: ExpanderConfig::default(),
            multiband: MultibandConfig::default(),
            natural_preservation: NaturalDynamicsConfig::default(),
        }
    }

    /// Create a natural dynamics processor for singing
    pub fn natural_singing() -> Self {
        Self {
            compressor: CompressorConfig {
                enabled: true,
                threshold: -12.0,
                ratio: 3.0,
                attack: 10.0,
                release: 100.0,
                knee_width: 2.0,
                makeup_gain: 3.0,
                sidechain_filter: None,
            },
            natural_preservation: NaturalDynamicsConfig {
                preserve_micro_dynamics: true,
                transient_preservation: 0.8,
                breath_preservation: 0.9,
                phrase_shaping: 0.7,
                target_dynamic_range: 20.0,
            },
            ..Self::new()
        }
    }

    /// Process audio with dynamic range control
    pub fn process(&self, audio: &mut [f32]) {
        if self.gate.enabled {
            self.apply_gate(audio);
        }

        if self.expander.enabled {
            self.apply_expander(audio);
        }

        if self.compressor.enabled {
            self.apply_compressor(audio);
        }

        if self.limiter.enabled {
            self.apply_limiter(audio);
        }
    }

    /// Apply noise gate
    fn apply_gate(&self, audio: &mut [f32]) {
        let threshold_linear = self.db_to_linear(self.gate.threshold);
        let attack_coeff = self.time_to_coefficient(self.gate.attack);
        let release_coeff = self.time_to_coefficient(self.gate.release);

        let mut envelope = 0.0;
        let mut gate_state = 0.0;

        for sample in audio.iter_mut() {
            let input_level = sample.abs();

            // Envelope following
            if input_level > envelope {
                envelope = input_level * attack_coeff + envelope * (1.0 - attack_coeff);
            } else {
                envelope = input_level * release_coeff + envelope * (1.0 - release_coeff);
            }

            // Gate decision
            let target_gain = if envelope > threshold_linear {
                1.0
            } else {
                0.0
            };

            // Smooth gate state
            gate_state = target_gain * 0.01 + gate_state * 0.99;

            *sample *= gate_state;
        }
    }

    /// Apply expander
    fn apply_expander(&self, audio: &mut [f32]) {
        let threshold_linear = self.db_to_linear(self.expander.threshold);
        let ratio = self.expander.ratio;

        for sample in audio.iter_mut() {
            let input_level = sample.abs();

            if input_level < threshold_linear && input_level > 0.0 {
                let expansion_db =
                    (self.linear_to_db(input_level) - self.expander.threshold) * (ratio - 1.0);
                let gain = self.db_to_linear(expansion_db);
                *sample *= gain;
            }
        }
    }

    /// Apply compressor
    fn apply_compressor(&self, audio: &mut [f32]) {
        let threshold_linear = self.db_to_linear(self.compressor.threshold);
        let ratio = self.compressor.ratio;
        let makeup_gain = self.db_to_linear(self.compressor.makeup_gain);
        let attack_coeff = self.time_to_coefficient(self.compressor.attack);
        let release_coeff = self.time_to_coefficient(self.compressor.release);

        let mut envelope = 0.0;

        for sample in audio.iter_mut() {
            let input_level = sample.abs();

            // Envelope following
            if input_level > envelope {
                envelope = input_level * attack_coeff + envelope * (1.0 - attack_coeff);
            } else {
                envelope = input_level * release_coeff + envelope * (1.0 - release_coeff);
            }

            // Compression calculation
            if envelope > threshold_linear {
                let over_threshold_db = self.linear_to_db(envelope) - self.compressor.threshold;
                let compressed_db = over_threshold_db / ratio;
                let reduction_db = over_threshold_db - compressed_db;
                let gain = self.db_to_linear(-reduction_db);
                *sample *= gain;
            }

            *sample *= makeup_gain;
        }
    }

    /// Apply limiter
    fn apply_limiter(&self, audio: &mut [f32]) {
        let ceiling_linear = self.db_to_linear(self.limiter.ceiling);

        for sample in audio.iter_mut() {
            if sample.abs() > ceiling_linear {
                *sample = sample.signum() * ceiling_linear;
            }
        }
    }

    /// Convert dB to linear scale
    fn db_to_linear(&self, db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Convert linear to dB scale
    fn linear_to_db(&self, linear: f32) -> f32 {
        if linear > 0.0 {
            20.0 * linear.log10()
        } else {
            -100.0 // Very quiet floor
        }
    }

    /// Convert time in milliseconds to coefficient
    fn time_to_coefficient(&self, time_ms: f32) -> f32 {
        1.0 - (-1.0 / (time_ms * 44.1)).exp() // Assuming 44.1kHz sample rate
    }
}

// Default implementations
impl Default for AntiAliasingConfig {
    fn default() -> Self {
        Self {
            cutoff_ratio: 0.9,
            filter_order: 32,
            filter_type: FilterType::Kaiser,
            transition_width: 0.1,
            stopband_attenuation: 80.0,
        }
    }
}

impl Default for HarmonicPhaseConfig {
    fn default() -> Self {
        Self {
            maintain_harmonics: true,
            fundamental_tracking: true,
            partial_coherence: 0.8,
            phase_offsets: vec![0.0; 16], // 16 harmonics
        }
    }
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            window_size: 1024,
            overlap_factor: 0.5,
            correlation_threshold: 0.5,
            max_delay_range: 100,
        }
    }
}

impl Default for VoicePositioning {
    fn default() -> Self {
        Self {
            auto_spacing: true,
            voice_positions: vec![-0.5, -0.2, 0.2, 0.5], // Default SATB positioning
            distance_simulation: DistanceSimulation::default(),
            voice_spread: 0.1,
            position_modulation: PositionModulation::default(),
        }
    }
}

impl Default for DistanceSimulation {
    fn default() -> Self {
        Self {
            enabled: false,
            reference_distance: 3.0,
            voice_distances: vec![3.0, 3.2, 2.8, 3.1],
            air_absorption: false,
            distance_filtering: false,
        }
    }
}

impl Default for PositionModulation {
    fn default() -> Self {
        Self {
            enabled: false,
            movement_speed: 0.1,
            movement_pattern: MovementPattern::Breathing,
            modulation_depth: 0.05,
        }
    }
}

impl Default for SpatialSimulation {
    fn default() -> Self {
        Self {
            room_size: (10.0, 15.0, 4.0),
            early_reflections: EarlyReflectionConfig::default(),
            late_reverb: LateReverbConfig::default(),
            source_positions: vec![(0.0, 2.0, 0.0)],
            listener_position: (0.0, -5.0, 1.5),
        }
    }
}

impl Default for EarlyReflectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            density: 0.7,
            level: 0.3,
            delay_spread: 0.05,
            hf_damping: 0.5,
        }
    }
}

impl Default for LateReverbConfig {
    fn default() -> Self {
        Self {
            reverb_time: 1.8,
            hf_decay_ratio: 0.7,
            diffusion: 0.8,
            density: 0.9,
            wet_level: 0.2,
        }
    }
}

impl Default for StereoEnhancement {
    fn default() -> Self {
        Self {
            bass_enhancement: 0.0,
            mid_side_processing: MidSideConfig::default(),
            harmonic_enhancement: 0.0,
            exciter_amount: 0.0,
        }
    }
}

impl Default for MidSideConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mid_gain: 1.0,
            side_gain: 1.0,
            mid_eq: EqConfig::default(),
            side_eq: EqConfig::default(),
        }
    }
}

impl Default for EqConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bands: vec![],
        }
    }
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: -20.0,
            ratio: 4.0,
            attack: 5.0,
            release: 50.0,
            knee_width: 1.0,
            makeup_gain: 0.0,
            sidechain_filter: None,
        }
    }
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ceiling: -0.1,
            release: 5.0,
            lookahead: 1.0,
            isr_detection: true,
        }
    }
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: -40.0,
            ratio: 10.0,
            attack: 1.0,
            hold: 10.0,
            release: 100.0,
        }
    }
}

impl Default for ExpanderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: -30.0,
            ratio: 2.0,
            attack: 5.0,
            release: 50.0,
            knee_width: 1.0,
        }
    }
}

impl Default for MultibandConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            crossover_frequencies: vec![200.0, 2000.0],
            band_compressors: vec![],
            band_eqs: vec![],
            crossover_slopes: vec![12.0, 12.0],
        }
    }
}

impl Default for NaturalDynamicsConfig {
    fn default() -> Self {
        Self {
            preserve_micro_dynamics: true,
            transient_preservation: 1.0,
            breath_preservation: 1.0,
            phrase_shaping: 0.5,
            target_dynamic_range: 24.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_creation() {
        let resampler = HighQualityResampler::new(44100.0, 48000.0);
        assert_eq!(resampler.source_rate, 44100.0);
        assert_eq!(resampler.target_rate, 48000.0);

        let fast_resampler = HighQualityResampler::fast(44100.0, 48000.0);
        assert_eq!(fast_resampler.interpolation, InterpolationMethod::Linear);
        assert_eq!(fast_resampler.quality_level, QualityLevel::Fast);
    }

    #[test]
    fn test_resampler_ratios() {
        let mut upsampler = HighQualityResampler::new(44100.0, 48000.0);
        // Force linear interpolation for testing
        upsampler.interpolation = InterpolationMethod::Linear;
        let input = vec![1.0, 0.0, -1.0, 0.0]; // Simple test signal
        let output = upsampler.resample(&input);

        // Output should be longer than input for upsampling
        assert!(output.len() > input.len());
    }

    #[test]
    fn test_phase_coherence() {
        let mut processor = PhaseCoherenceProcessor::new();
        let mut channels = vec![
            vec![1.0, 0.5, -0.5, -1.0],
            vec![0.9, 0.6, -0.4, -0.8], // Slightly different
        ];

        processor.process_channels(&mut channels);

        // Channels should still have the same length
        assert_eq!(channels[0].len(), channels[1].len());
    }

    #[test]
    fn test_stereo_imaging() {
        let processor = StereoImagingProcessor::new();
        let mut left = vec![1.0, 0.5, -0.5, -1.0];
        let mut right = vec![0.8, 0.4, -0.4, -0.8];

        processor.process_stereo(&mut left, &mut right);

        // Should preserve length
        assert_eq!(left.len(), 4);
        assert_eq!(right.len(), 4);
    }

    #[test]
    fn test_voice_positioning() {
        let processor = StereoImagingProcessor::new();
        let voices = vec![vec![1.0, 0.5, -0.5, -1.0], vec![0.8, 0.4, -0.4, -0.8]];
        let mut voices_mut = voices.clone();

        let positioned = processor.position_voices(&mut voices_mut);

        assert_eq!(positioned.len(), 2);
        assert_eq!(positioned[0].0.len(), 4); // Left channel
        assert_eq!(positioned[0].1.len(), 4); // Right channel
    }

    #[test]
    fn test_dynamic_range_processor() {
        let processor = DynamicRangeProcessor::natural_singing();
        let mut audio = vec![0.1, 0.8, 1.2, -0.9, 0.3]; // Mixed levels

        processor.process(&mut audio);

        // Should preserve length
        assert_eq!(audio.len(), 5);

        // Loud peaks should be reduced if limiter is enabled
        if processor.limiter.enabled {
            assert!(audio
                .iter()
                .all(|&x| x.abs() <= processor.db_to_linear(processor.limiter.ceiling)));
        }
    }

    #[test]
    fn test_pan_law_calculations() {
        let processor = StereoImagingProcessor::new();

        // Center position should give equal gains
        let (left_gain, right_gain) = processor.calculate_pan_gains(0.0);
        assert!((left_gain - right_gain).abs() < 0.1);

        // Hard left should favor left channel
        let (left_gain, right_gain) = processor.calculate_pan_gains(-1.0);
        assert!(left_gain > right_gain);

        // Hard right should favor right channel
        let (left_gain, right_gain) = processor.calculate_pan_gains(1.0);
        assert!(right_gain > left_gain);
    }

    #[test]
    fn test_db_linear_conversion() {
        let processor = DynamicRangeProcessor::new();

        // 0 dB should be 1.0 linear
        assert!((processor.db_to_linear(0.0) - 1.0).abs() < 0.001);

        // -6 dB should be approximately 0.5 linear
        assert!((processor.db_to_linear(-6.0) - 0.5).abs() < 0.01);

        // Conversion should be reversible
        let db_value = -12.0;
        let linear = processor.db_to_linear(db_value);
        let back_to_db = processor.linear_to_db(linear);
        assert!((db_value - back_to_db).abs() < 0.01);
    }
}
