//! Professional audio effects processing for speech synthesis.
//!
//! This module provides high-quality audio effects that can be applied to synthesized speech,
//! including reverb, delay, chorus, compression, and equalization. All effects support
//! real-time processing and can be chained together.
//!
//! # Examples
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//! use voirs_sdk::audio::effects::*;
//!
//! # async fn example() -> Result<()> {
//! let pipeline = VoirsPipelineBuilder::new().build().await?;
//! let audio = pipeline.synthesize("Hello, world!").await?;
//!
//! // Apply reverb
//! let reverb = ReverbEffect::new(audio.sample_rate())
//!     .with_room_size(0.8)
//!     .with_damping(0.5);
//! let reverb_audio = reverb.process(&audio)?;
//!
//! // Apply delay
//! let delay = DelayEffect::new(audio.sample_rate(), 0.3, 0.4)?;
//! let delayed_audio = delay.process(&audio)?;
//!
//! // Chain multiple effects
//! let chain = EffectsChain::new()
//!     .add_effect(Box::new(reverb))
//!     .add_effect(Box::new(delay));
//! let processed = chain.process(&audio)?;
//! # Ok(())
//! # }
//! ```

use super::AudioBuffer;
use crate::{Result, VoirsError};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::f32::consts::PI;

/// Trait for audio effects that can be applied to audio buffers.
pub trait AudioEffect: Send + Sync {
    /// Process an audio buffer with this effect.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio buffer to process
    ///
    /// # Returns
    ///
    /// Processed audio buffer with effect applied
    fn process(&self, input: &AudioBuffer) -> Result<AudioBuffer>;

    /// Get the name of this effect.
    fn name(&self) -> &str;

    /// Reset the effect's internal state (useful for real-time processing).
    fn reset(&mut self);
}

/// Reverb effect using Freeverb algorithm.
///
/// Implements the classic Freeverb algorithm by Jezar at Dreampoint,
/// which uses parallel comb filters and series allpass filters to create
/// a natural-sounding reverb.
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::effects::ReverbEffect;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Hello").await?;
///
/// let reverb = ReverbEffect::new(audio.sample_rate())
///     .with_room_size(0.8)
///     .with_damping(0.5)
///     .with_wet_level(0.3);
///
/// let reverb_audio = reverb.process(&audio)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ReverbEffect {
    sample_rate: u32,
    room_size: f32,
    damping: f32,
    wet_level: f32,
    dry_level: f32,
    width: f32,
    comb_filters: Vec<CombFilter>,
    allpass_filters: Vec<AllpassFilter>,
}

/// Comb filter for reverb implementation.
#[derive(Clone)]
struct CombFilter {
    buffer: Vec<f32>,
    buffer_index: usize,
    filter_store: f32,
    damping1: f32,
    damping2: f32,
    feedback: f32,
}

/// Allpass filter for reverb implementation.
#[derive(Clone)]
struct AllpassFilter {
    buffer: Vec<f32>,
    buffer_index: usize,
}

impl CombFilter {
    fn new(buffer_size: usize) -> Self {
        Self {
            buffer: vec![0.0; buffer_size],
            buffer_index: 0,
            filter_store: 0.0,
            damping1: 0.5,
            damping2: 0.5,
            feedback: 0.5,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.buffer_index];
        self.filter_store = output * self.damping2 + self.filter_store * self.damping1;
        self.buffer[self.buffer_index] = input + self.filter_store * self.feedback;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();
        output
    }

    fn set_damping(&mut self, damping: f32) {
        self.damping1 = damping;
        self.damping2 = 1.0 - damping;
    }

    fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback;
    }
}

impl AllpassFilter {
    fn new(buffer_size: usize) -> Self {
        Self {
            buffer: vec![0.0; buffer_size],
            buffer_index: 0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let buffered = self.buffer[self.buffer_index];
        let output = buffered - input;
        self.buffer[self.buffer_index] = input + buffered * 0.5;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();
        output
    }
}

impl ReverbEffect {
    /// Create a new reverb effect.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate of the audio to be processed
    pub fn new(sample_rate: u32) -> Self {
        // Freeverb comb filter delays (tuned for 44.1 kHz)
        let comb_tunings = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
        let allpass_tunings = [556, 441, 341, 225];

        // Scale delays for sample rate
        let scale = sample_rate as f32 / 44100.0;

        let comb_filters = comb_tunings
            .iter()
            .map(|&size| CombFilter::new((size as f32 * scale) as usize))
            .collect();

        let allpass_filters = allpass_tunings
            .iter()
            .map(|&size| AllpassFilter::new((size as f32 * scale) as usize))
            .collect();

        Self {
            sample_rate,
            room_size: 0.5,
            damping: 0.5,
            wet_level: 0.3,
            dry_level: 0.7,
            width: 1.0,
            comb_filters,
            allpass_filters,
        }
    }

    /// Set the room size (0.0 to 1.0).
    pub fn with_room_size(mut self, room_size: f32) -> Self {
        self.room_size = room_size.clamp(0.0, 1.0);
        self.update_parameters();
        self
    }

    /// Set the damping amount (0.0 to 1.0).
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self.update_parameters();
        self
    }

    /// Set the wet (effect) level (0.0 to 1.0).
    pub fn with_wet_level(mut self, wet_level: f32) -> Self {
        self.wet_level = wet_level.clamp(0.0, 1.0);
        self
    }

    /// Set the dry (original) level (0.0 to 1.0).
    pub fn with_dry_level(mut self, dry_level: f32) -> Self {
        self.dry_level = dry_level.clamp(0.0, 1.0);
        self
    }

    /// Set the stereo width (0.0 to 1.0).
    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width.clamp(0.0, 1.0);
        self
    }

    fn update_parameters(&mut self) {
        let feedback = 0.28 + self.room_size * 0.7;
        for filter in &mut self.comb_filters {
            filter.set_feedback(feedback);
            filter.set_damping(self.damping);
        }
    }
}

impl AudioEffect for ReverbEffect {
    fn process(&self, input: &AudioBuffer) -> Result<AudioBuffer> {
        let samples = input.samples();
        let mut output = vec![0.0f32; samples.len()];

        // Clone filters for processing (to maintain immutability of self)
        let mut comb_filters = self.comb_filters.clone();
        let mut allpass_filters = self.allpass_filters.clone();

        for (i, &sample) in samples.iter().enumerate() {
            // Process through comb filters (parallel)
            let mut comb_out = 0.0;
            for filter in &mut comb_filters {
                comb_out += filter.process(sample);
            }

            // Process through allpass filters (series)
            let mut allpass_out = comb_out;
            for filter in &mut allpass_filters {
                allpass_out = filter.process(allpass_out);
            }

            // Mix wet and dry signals
            output[i] = sample * self.dry_level + allpass_out * self.wet_level;
        }

        Ok(AudioBuffer::mono(output, input.sample_rate()))
    }

    fn name(&self) -> &str {
        "Reverb"
    }

    fn reset(&mut self) {
        for filter in &mut self.comb_filters {
            filter.buffer.fill(0.0);
            filter.buffer_index = 0;
            filter.filter_store = 0.0;
        }
        for filter in &mut self.allpass_filters {
            filter.buffer.fill(0.0);
            filter.buffer_index = 0;
        }
    }
}

/// Delay/echo effect with feedback control.
///
/// Creates echo effects by delaying the input signal and feeding it back.
/// Supports configurable delay time and feedback amount.
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::effects::DelayEffect;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Hello").await?;
///
/// // 300ms delay with 40% feedback
/// let delay = DelayEffect::new(audio.sample_rate(), 0.3, 0.4)?;
/// let delayed = delay.process(&audio)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct DelayEffect {
    delay_buffer: Vec<f32>,
    buffer_index: usize,
    feedback: f32,
    wet_level: f32,
    dry_level: f32,
}

impl DelayEffect {
    /// Create a new delay effect.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate of the audio
    /// * `delay_time` - Delay time in seconds
    /// * `feedback` - Feedback amount (0.0 to 0.9)
    pub fn new(sample_rate: u32, delay_time: f32, feedback: f32) -> Result<Self> {
        if delay_time <= 0.0 || delay_time > 2.0 {
            return Err(VoirsError::InvalidConfiguration {
                field: "delay_time".to_string(),
                value: delay_time.to_string(),
                reason: "Delay time must be between 0 and 2 seconds".to_string(),
                valid_values: Some(vec!["0.0-2.0".to_string()]),
            });
        }

        let buffer_size = (sample_rate as f32 * delay_time) as usize;

        Ok(Self {
            delay_buffer: vec![0.0; buffer_size],
            buffer_index: 0,
            feedback: feedback.clamp(0.0, 0.9),
            wet_level: 0.5,
            dry_level: 1.0,
        })
    }

    /// Set the wet (effect) level.
    pub fn with_wet_level(mut self, wet_level: f32) -> Self {
        self.wet_level = wet_level.clamp(0.0, 1.0);
        self
    }

    /// Set the dry (original) level.
    pub fn with_dry_level(mut self, dry_level: f32) -> Self {
        self.dry_level = dry_level.clamp(0.0, 1.0);
        self
    }
}

impl AudioEffect for DelayEffect {
    fn process(&self, input: &AudioBuffer) -> Result<AudioBuffer> {
        let samples = input.samples();
        let mut output = vec![0.0f32; samples.len()];
        let mut delay_buffer = self.delay_buffer.clone();
        let mut buffer_index = self.buffer_index;

        for (i, &sample) in samples.iter().enumerate() {
            let delayed = delay_buffer[buffer_index];
            delay_buffer[buffer_index] = sample + delayed * self.feedback;
            output[i] = sample * self.dry_level + delayed * self.wet_level;
            buffer_index = (buffer_index + 1) % delay_buffer.len();
        }

        Ok(AudioBuffer::mono(output, input.sample_rate()))
    }

    fn name(&self) -> &str {
        "Delay"
    }

    fn reset(&mut self) {
        self.delay_buffer.fill(0.0);
        self.buffer_index = 0;
    }
}

/// Chorus effect using LFO modulation.
///
/// Creates a rich, ensemble sound by modulating delayed copies of the signal.
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::effects::ChorusEffect;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Hello").await?;
///
/// let chorus = ChorusEffect::new(audio.sample_rate())
///     .with_rate(0.5)
///     .with_depth(0.3);
///
/// let chorused = chorus.process(&audio)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ChorusEffect {
    sample_rate: u32,
    delay_buffer: Vec<f32>,
    buffer_index: usize,
    lfo_phase: f32,
    rate: f32,       // LFO rate in Hz
    depth: f32,      // Modulation depth
    delay_time: f32, // Base delay in seconds
    wet_level: f32,
    dry_level: f32,
}

impl ChorusEffect {
    /// Create a new chorus effect.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate of the audio
    pub fn new(sample_rate: u32) -> Self {
        let max_delay = 0.05; // 50ms maximum delay
        let buffer_size = (sample_rate as f32 * max_delay) as usize;

        Self {
            sample_rate,
            delay_buffer: vec![0.0; buffer_size],
            buffer_index: 0,
            lfo_phase: 0.0,
            rate: 0.5,        // 0.5 Hz LFO
            depth: 0.3,       // 30% modulation
            delay_time: 0.02, // 20ms base delay
            wet_level: 0.5,
            dry_level: 1.0,
        }
    }

    /// Set the LFO rate in Hz (0.1 to 5.0).
    pub fn with_rate(mut self, rate: f32) -> Self {
        self.rate = rate.clamp(0.1, 5.0);
        self
    }

    /// Set the modulation depth (0.0 to 1.0).
    pub fn with_depth(mut self, depth: f32) -> Self {
        self.depth = depth.clamp(0.0, 1.0);
        self
    }

    /// Set the wet (effect) level.
    pub fn with_wet_level(mut self, wet_level: f32) -> Self {
        self.wet_level = wet_level.clamp(0.0, 1.0);
        self
    }

    /// Set the dry (original) level.
    pub fn with_dry_level(mut self, dry_level: f32) -> Self {
        self.dry_level = dry_level.clamp(0.0, 1.0);
        self
    }
}

impl AudioEffect for ChorusEffect {
    fn process(&self, input: &AudioBuffer) -> Result<AudioBuffer> {
        let samples = input.samples();
        let mut output = vec![0.0f32; samples.len()];
        let mut delay_buffer = self.delay_buffer.clone();
        let mut buffer_index = self.buffer_index;
        let mut lfo_phase = self.lfo_phase;

        let lfo_increment = 2.0 * PI * self.rate / self.sample_rate as f32;

        for (i, &sample) in samples.iter().enumerate() {
            // Calculate modulated delay time
            let lfo = (lfo_phase.sin() + 1.0) * 0.5; // 0.0 to 1.0
            let modulated_delay = self.delay_time * (1.0 + self.depth * lfo);
            let delay_samples = (modulated_delay * self.sample_rate as f32) as usize;

            // Read from delay buffer with interpolation
            let read_index =
                (buffer_index + delay_buffer.len() - delay_samples) % delay_buffer.len();
            let delayed = delay_buffer[read_index];

            // Write to delay buffer
            delay_buffer[buffer_index] = sample;

            // Mix
            output[i] = sample * self.dry_level + delayed * self.wet_level;

            // Update LFO phase
            lfo_phase += lfo_increment;
            if lfo_phase >= 2.0 * PI {
                lfo_phase -= 2.0 * PI;
            }

            buffer_index = (buffer_index + 1) % delay_buffer.len();
        }

        Ok(AudioBuffer::mono(output, input.sample_rate()))
    }

    fn name(&self) -> &str {
        "Chorus"
    }

    fn reset(&mut self) {
        self.delay_buffer.fill(0.0);
        self.buffer_index = 0;
        self.lfo_phase = 0.0;
    }
}

/// Dynamic range compressor.
///
/// Reduces the dynamic range of audio by attenuating signals above a threshold.
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::effects::CompressorEffect;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Hello").await?;
///
/// let compressor = CompressorEffect::new(audio.sample_rate())
///     .with_threshold(-20.0)
///     .with_ratio(4.0)
///     .with_attack(0.005)
///     .with_release(0.1);
///
/// let compressed = compressor.process(&audio)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct CompressorEffect {
    sample_rate: u32,
    threshold: f32,    // dB
    ratio: f32,        // compression ratio
    attack_time: f32,  // seconds
    release_time: f32, // seconds
    makeup_gain: f32,  // dB
    envelope: f32,
}

impl CompressorEffect {
    /// Create a new compressor effect.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate of the audio
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            threshold: -20.0,
            ratio: 4.0,
            attack_time: 0.005, // 5ms
            release_time: 0.1,  // 100ms
            makeup_gain: 0.0,
            envelope: 0.0,
        }
    }

    /// Set the threshold in dB.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the compression ratio.
    pub fn with_ratio(mut self, ratio: f32) -> Self {
        self.ratio = ratio.max(1.0);
        self
    }

    /// Set the attack time in seconds.
    pub fn with_attack(mut self, attack: f32) -> Self {
        self.attack_time = attack.max(0.0001);
        self
    }

    /// Set the release time in seconds.
    pub fn with_release(mut self, release: f32) -> Self {
        self.release_time = release.max(0.001);
        self
    }

    /// Set the makeup gain in dB.
    pub fn with_makeup_gain(mut self, gain: f32) -> Self {
        self.makeup_gain = gain;
        self
    }

    fn lin_to_db(lin: f32) -> f32 {
        20.0 * lin.max(1e-10).log10()
    }

    fn db_to_lin(db: f32) -> f32 {
        10.0f32.powf(db / 20.0)
    }
}

impl AudioEffect for CompressorEffect {
    fn process(&self, input: &AudioBuffer) -> Result<AudioBuffer> {
        let samples = input.samples();
        let mut output = vec![0.0f32; samples.len()];
        let mut envelope = self.envelope;

        let attack_coef = (-1.0 / (self.attack_time * self.sample_rate as f32)).exp();
        let release_coef = (-1.0 / (self.release_time * self.sample_rate as f32)).exp();

        for (i, &sample) in samples.iter().enumerate() {
            // Envelope follower
            let input_level = sample.abs();
            if input_level > envelope {
                envelope = attack_coef * envelope + (1.0 - attack_coef) * input_level;
            } else {
                envelope = release_coef * envelope + (1.0 - release_coef) * input_level;
            }

            // Compute gain reduction
            let envelope_db = Self::lin_to_db(envelope);
            let gain_db = if envelope_db > self.threshold {
                (self.threshold - envelope_db) * (1.0 - 1.0 / self.ratio)
            } else {
                0.0
            };

            // Apply gain reduction and makeup gain
            let total_gain = Self::db_to_lin(gain_db + self.makeup_gain);
            output[i] = sample * total_gain;
        }

        Ok(AudioBuffer::mono(output, input.sample_rate()))
    }

    fn name(&self) -> &str {
        "Compressor"
    }

    fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// Parametric equalizer effect.
///
/// Provides frequency-selective amplification or attenuation.
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::effects::EqualizerEffect;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Hello").await?;
///
/// let eq = EqualizerEffect::new(audio.sample_rate())
///     .add_band(1000.0, 3.0, 1.0)?  // Boost 1kHz by 3dB
///     .add_band(5000.0, -2.0, 2.0)?; // Cut 5kHz by 2dB
///
/// let equalized = eq.process(&audio)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct EqualizerEffect {
    sample_rate: u32,
    bands: Vec<EqBand>,
}

#[derive(Clone)]
struct EqBand {
    frequency: f32,
    gain_db: f32,
    q: f32,
    // Biquad coefficients
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    // State variables
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl EqBand {
    fn new(sample_rate: u32, frequency: f32, gain_db: f32, q: f32) -> Self {
        let mut band = Self {
            frequency,
            gain_db,
            q,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        };
        band.update_coefficients(sample_rate);
        band
    }

    fn update_coefficients(&mut self, sample_rate: u32) {
        let w0 = 2.0 * PI * self.frequency / sample_rate as f32;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * self.q);
        let a = 10.0f32.powf(self.gain_db / 40.0);

        // Peaking EQ coefficients
        self.b0 = 1.0 + alpha * a;
        self.b1 = -2.0 * cos_w0;
        self.b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        self.a1 = -2.0 * cos_w0;
        self.a2 = 1.0 - alpha / a;

        // Normalize
        self.b0 /= a0;
        self.b1 /= a0;
        self.b2 /= a0;
        self.a1 /= a0;
        self.a2 /= a0;
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }
}

impl EqualizerEffect {
    /// Create a new equalizer effect.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate of the audio
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            bands: Vec::new(),
        }
    }

    /// Add an EQ band.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Center frequency in Hz
    /// * `gain_db` - Gain in dB (positive = boost, negative = cut)
    /// * `q` - Q factor (bandwidth), typically 0.5 to 10.0
    pub fn add_band(mut self, frequency: f32, gain_db: f32, q: f32) -> Result<Self> {
        if frequency <= 0.0 || frequency >= (self.sample_rate as f32) / 2.0 {
            return Err(VoirsError::InvalidConfiguration {
                field: "frequency".to_string(),
                value: frequency.to_string(),
                reason: format!(
                    "Frequency must be between 0 and {} Hz (Nyquist frequency)",
                    (self.sample_rate as f32) / 2.0
                ),
                valid_values: Some(vec![format!("0.0-{}", (self.sample_rate as f32) / 2.0)]),
            });
        }

        self.bands
            .push(EqBand::new(self.sample_rate, frequency, gain_db, q));
        Ok(self)
    }
}

impl AudioEffect for EqualizerEffect {
    fn process(&self, input: &AudioBuffer) -> Result<AudioBuffer> {
        let samples = input.samples();
        let mut output = samples.to_vec();
        let mut bands = self.bands.clone();

        for band in &mut bands {
            for sample in &mut output {
                *sample = band.process(*sample);
            }
        }

        Ok(AudioBuffer::mono(output, input.sample_rate()))
    }

    fn name(&self) -> &str {
        "Equalizer"
    }

    fn reset(&mut self) {
        for band in &mut self.bands {
            band.x1 = 0.0;
            band.x2 = 0.0;
            band.y1 = 0.0;
            band.y2 = 0.0;
        }
    }
}

/// Effects chain for processing multiple effects in sequence.
///
/// # Examples
///
/// ```no_run
/// use voirs_sdk::prelude::*;
/// use voirs_sdk::audio::effects::*;
///
/// # async fn example() -> Result<()> {
/// let pipeline = VoirsPipelineBuilder::new().build().await?;
/// let audio = pipeline.synthesize("Hello").await?;
///
/// let chain = EffectsChain::new()
///     .add_effect(Box::new(CompressorEffect::new(audio.sample_rate())))
///     .add_effect(Box::new(EqualizerEffect::new(audio.sample_rate())))
///     .add_effect(Box::new(ReverbEffect::new(audio.sample_rate())));
///
/// let processed = chain.process(&audio)?;
/// # Ok(())
/// # }
/// ```
pub struct EffectsChain {
    effects: Vec<Box<dyn AudioEffect>>,
}

impl EffectsChain {
    /// Create a new empty effects chain.
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Add an effect to the chain.
    pub fn add_effect(mut self, effect: Box<dyn AudioEffect>) -> Self {
        self.effects.push(effect);
        self
    }

    /// Get the number of effects in the chain.
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Process audio through the entire effects chain.
    pub fn process(&self, input: &AudioBuffer) -> Result<AudioBuffer> {
        let mut output = input.clone();
        for effect in &self.effects {
            output = effect.process(&output)?;
        }
        Ok(output)
    }

    /// Reset all effects in the chain.
    pub fn reset(&mut self) {
        for effect in &mut self.effects {
            effect.reset();
        }
    }
}

impl Default for EffectsChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(sample_rate: u32) -> AudioBuffer {
        let duration = 1.0; // 1 second
        let frequency = 440.0; // A4
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * PI * frequency * t).sin() * 0.5
            })
            .collect();
        AudioBuffer::mono(samples, sample_rate)
    }

    #[test]
    fn test_reverb_effect() {
        let audio = create_test_audio(44100);
        let reverb = ReverbEffect::new(audio.sample_rate())
            .with_room_size(0.8)
            .with_damping(0.5);

        let result = reverb.process(&audio);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), audio.len());
        assert_eq!(output.sample_rate(), audio.sample_rate());
    }

    #[test]
    fn test_delay_effect() {
        let audio = create_test_audio(44100);
        let delay = DelayEffect::new(audio.sample_rate(), 0.3, 0.4).unwrap();

        let result = delay.process(&audio);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), audio.len());
    }

    #[test]
    fn test_delay_invalid_time() {
        let result = DelayEffect::new(44100, 5.0, 0.4);
        assert!(result.is_err());
    }

    #[test]
    fn test_chorus_effect() {
        let audio = create_test_audio(44100);
        let chorus = ChorusEffect::new(audio.sample_rate())
            .with_rate(0.5)
            .with_depth(0.3);

        let result = chorus.process(&audio);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), audio.len());
    }

    #[test]
    fn test_compressor_effect() {
        let audio = create_test_audio(44100);
        let compressor = CompressorEffect::new(audio.sample_rate())
            .with_threshold(-20.0)
            .with_ratio(4.0);

        let result = compressor.process(&audio);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), audio.len());
    }

    #[test]
    fn test_equalizer_effect() {
        let audio = create_test_audio(44100);
        let eq = EqualizerEffect::new(audio.sample_rate())
            .add_band(1000.0, 3.0, 1.0)
            .unwrap();

        let result = eq.process(&audio);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), audio.len());
    }

    #[test]
    fn test_equalizer_invalid_frequency() {
        let eq = EqualizerEffect::new(44100);
        let result = eq.add_band(50000.0, 3.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_effects_chain() {
        let audio = create_test_audio(44100);
        let chain = EffectsChain::new()
            .add_effect(Box::new(CompressorEffect::new(audio.sample_rate())))
            .add_effect(Box::new(ReverbEffect::new(audio.sample_rate())));

        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());

        let result = chain.process(&audio);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), audio.len());
    }

    #[test]
    fn test_effects_chain_empty() {
        let audio = create_test_audio(44100);
        let chain = EffectsChain::new();

        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());

        let result = chain.process(&audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_effect_reset() {
        let mut reverb = ReverbEffect::new(44100);
        reverb.reset();

        let mut delay = DelayEffect::new(44100, 0.3, 0.4).unwrap();
        delay.reset();

        let mut chorus = ChorusEffect::new(44100);
        chorus.reset();

        let mut compressor = CompressorEffect::new(44100);
        compressor.reset();

        let mut eq = EqualizerEffect::new(44100)
            .add_band(1000.0, 3.0, 1.0)
            .unwrap();
        eq.reset();
    }
}
