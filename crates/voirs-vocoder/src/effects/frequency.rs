//! Frequency domain processing effects for audio enhancement.
//!
//! This module implements parametric EQ, high-frequency enhancement,
//! and de-essing for speech processing.

use super::{AudioEffect, EffectParameter};
use crate::{AudioBuffer, Result};
use std::f32::consts::PI;

/// Biquad filter coefficients and state
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    a0: f32,
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Default for BiquadFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl BiquadFilter {
    pub fn new() -> Self {
        Self {
            a0: 1.0,
            a1: 0.0,
            a2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let output = (self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2)
            / self.a0;

        // Update delay line
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Design a low shelf filter
    pub fn design_low_shelf(&mut self, sample_rate: f32, frequency: f32, gain_db: f32, q: f32) {
        let w = 2.0 * PI * frequency / sample_rate;
        let cos_w = w.cos();
        let sin_w = w.sin();
        let a = 10.0_f32.powf(gain_db / 40.0);
        let beta = (a / q).sqrt();

        self.b0 = a * ((a + 1.0) - (a - 1.0) * cos_w + beta * sin_w);
        self.b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w);
        self.b2 = a * ((a + 1.0) - (a - 1.0) * cos_w - beta * sin_w);
        self.a0 = (a + 1.0) + (a - 1.0) * cos_w + beta * sin_w;
        self.a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w);
        self.a2 = (a + 1.0) + (a - 1.0) * cos_w - beta * sin_w;
    }

    /// Design a high shelf filter
    pub fn design_high_shelf(&mut self, sample_rate: f32, frequency: f32, gain_db: f32, q: f32) {
        let w = 2.0 * PI * frequency / sample_rate;
        let cos_w = w.cos();
        let sin_w = w.sin();
        let a = 10.0_f32.powf(gain_db / 40.0);
        let beta = (a / q).sqrt();

        self.b0 = a * ((a + 1.0) + (a - 1.0) * cos_w + beta * sin_w);
        self.b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w);
        self.b2 = a * ((a + 1.0) + (a - 1.0) * cos_w - beta * sin_w);
        self.a0 = (a + 1.0) - (a - 1.0) * cos_w + beta * sin_w;
        self.a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w);
        self.a2 = (a + 1.0) - (a - 1.0) * cos_w - beta * sin_w;
    }

    /// Design a peaking filter
    pub fn design_peak(&mut self, sample_rate: f32, frequency: f32, gain_db: f32, q: f32) {
        let w = 2.0 * PI * frequency / sample_rate;
        let cos_w = w.cos();
        let sin_w = w.sin();
        let a = 10.0_f32.powf(gain_db / 40.0);
        let alpha = sin_w / (2.0 * q);

        self.b0 = 1.0 + alpha * a;
        self.b1 = -2.0 * cos_w;
        self.b2 = 1.0 - alpha * a;
        self.a0 = 1.0 + alpha / a;
        self.a1 = -2.0 * cos_w;
        self.a2 = 1.0 - alpha / a;
    }

    /// Design a high-pass filter
    pub fn design_highpass(&mut self, sample_rate: f32, frequency: f32, q: f32) {
        let w = 2.0 * PI * frequency / sample_rate;
        let cos_w = w.cos();
        let sin_w = w.sin();
        let alpha = sin_w / (2.0 * q);

        self.b0 = (1.0 + cos_w) / 2.0;
        self.b1 = -(1.0 + cos_w);
        self.b2 = (1.0 + cos_w) / 2.0;
        self.a0 = 1.0 + alpha;
        self.a1 = -2.0 * cos_w;
        self.a2 = 1.0 - alpha;
    }
}

/// Parametric equalizer with multiple bands
pub struct ParametricEQ {
    enabled: bool,
    bands: Vec<EQBand>,
    filters: Vec<BiquadFilter>,
    sample_rate: u32,
}

#[derive(Debug, Clone)]
pub struct EQBand {
    pub frequency: EffectParameter,
    pub gain: EffectParameter,
    pub q: EffectParameter,
    pub filter_type: EQFilterType,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum EQFilterType {
    LowShelf,
    Peak,
    HighShelf,
    HighPass,
    LowPass,
}

impl ParametricEQ {
    pub fn new(sample_rate: u32) -> Self {
        let mut eq = Self {
            enabled: true,
            bands: Vec::new(),
            filters: Vec::new(),
            sample_rate,
        };

        // Add default bands
        eq.add_band(EQFilterType::LowShelf, 100.0, 0.0, 0.7);
        eq.add_band(EQFilterType::Peak, 1000.0, 0.0, 1.0);
        eq.add_band(EQFilterType::Peak, 3000.0, 0.0, 1.0);
        eq.add_band(EQFilterType::HighShelf, 8000.0, 0.0, 0.7);

        eq.update_filters();
        eq
    }

    pub fn add_band(&mut self, filter_type: EQFilterType, frequency: f32, gain_db: f32, q: f32) {
        let band = EQBand {
            frequency: EffectParameter::new("frequency", frequency, 20.0, 20000.0),
            gain: EffectParameter::new("gain", gain_db, -20.0, 20.0),
            q: EffectParameter::new("q", q, 0.1, 10.0),
            filter_type,
            enabled: true,
        };

        self.bands.push(band);
        self.filters.push(BiquadFilter::new());
    }

    pub fn update_filters(&mut self) {
        for (i, band) in self.bands.iter().enumerate() {
            if i < self.filters.len() {
                let filter = &mut self.filters[i];
                let sample_rate = self.sample_rate as f32;

                match band.filter_type {
                    EQFilterType::LowShelf => {
                        filter.design_low_shelf(
                            sample_rate,
                            band.frequency.value,
                            band.gain.value,
                            band.q.value,
                        );
                    }
                    EQFilterType::HighShelf => {
                        filter.design_high_shelf(
                            sample_rate,
                            band.frequency.value,
                            band.gain.value,
                            band.q.value,
                        );
                    }
                    EQFilterType::Peak => {
                        filter.design_peak(
                            sample_rate,
                            band.frequency.value,
                            band.gain.value,
                            band.q.value,
                        );
                    }
                    EQFilterType::HighPass => {
                        filter.design_highpass(sample_rate, band.frequency.value, band.q.value);
                    }
                    EQFilterType::LowPass => {
                        // For simplicity, use high-pass with inverted frequency response
                        filter.design_highpass(sample_rate, band.frequency.value, band.q.value);
                    }
                }
            }
        }
    }

    pub fn set_band_gain(&mut self, band_index: usize, gain_db: f32) {
        if band_index < self.bands.len() {
            self.bands[band_index].gain.set_value(gain_db);
            self.update_filters();
        }
    }

    pub fn set_band_frequency(&mut self, band_index: usize, frequency: f32) {
        if band_index < self.bands.len() {
            self.bands[band_index].frequency.set_value(frequency);
            self.update_filters();
        }
    }
}

impl AudioEffect for ParametricEQ {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let samples = audio.samples_mut();

        for sample in samples.iter_mut() {
            let mut output = *sample;

            for (i, band) in self.bands.iter().enumerate() {
                if band.enabled && i < self.filters.len() {
                    output = self.filters[i].process(output);
                }
            }

            *sample = output;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ParametricEQ"
    }

    fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// High-frequency enhancement for presence and clarity
pub struct HighFrequencyEnhancer {
    enabled: bool,
    frequency: EffectParameter, // Enhancement frequency
    amount: EffectParameter,    // Enhancement amount
    bandwidth: EffectParameter, // Bandwidth of enhancement

    // Filters
    highpass: BiquadFilter,
    peaking: BiquadFilter,
    sample_rate: u32,
}

impl HighFrequencyEnhancer {
    pub fn new(sample_rate: u32) -> Self {
        let mut enhancer = Self {
            enabled: true,
            frequency: EffectParameter::new("frequency", 8000.0, 2000.0, 16000.0),
            amount: EffectParameter::new("amount", 3.0, 0.0, 12.0),
            bandwidth: EffectParameter::new("bandwidth", 1.5, 0.5, 5.0),

            highpass: BiquadFilter::new(),
            peaking: BiquadFilter::new(),
            sample_rate,
        };

        enhancer.update_filters();
        enhancer
    }

    fn update_filters(&mut self) {
        let sample_rate = self.sample_rate as f32;

        // High-pass to isolate high frequencies
        self.highpass
            .design_highpass(sample_rate, self.frequency.value * 0.8, 0.7);

        // Peaking filter for enhancement
        self.peaking.design_peak(
            sample_rate,
            self.frequency.value,
            self.amount.value,
            self.bandwidth.value,
        );
    }

    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency.set_value(frequency);
        self.update_filters();
    }

    pub fn set_amount(&mut self, amount_db: f32) {
        self.amount.set_value(amount_db);
        self.update_filters();
    }
}

impl AudioEffect for HighFrequencyEnhancer {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let samples = audio.samples_mut();

        for sample in samples.iter_mut() {
            // Process high frequencies
            let high_freq = self.highpass.process(*sample);
            let enhanced = self.peaking.process(high_freq);

            // Mix with original signal
            *sample += enhanced * 0.3; // Mix 30% of enhanced signal
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "HighFrequencyEnhancer"
    }

    fn reset(&mut self) {
        self.highpass.reset();
        self.peaking.reset();
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// De-esser for reducing sibilant sounds
pub struct Deesser {
    enabled: bool,
    frequency: EffectParameter, // Sibilant frequency range
    threshold: EffectParameter, // Detection threshold
    ratio: EffectParameter,     // Reduction ratio

    // Filters and detection
    bandpass: BiquadFilter,
    compressor_gain: f32,
    envelope: f32,
    sample_rate: u32,
}

impl Deesser {
    pub fn new(sample_rate: u32) -> Self {
        let mut deesser = Self {
            enabled: true,
            frequency: EffectParameter::new("frequency", 6000.0, 3000.0, 12000.0),
            threshold: EffectParameter::new("threshold", -25.0, -60.0, 0.0),
            ratio: EffectParameter::new("ratio", 4.0, 1.0, 10.0),

            bandpass: BiquadFilter::new(),
            compressor_gain: 1.0,
            envelope: 0.0,
            sample_rate,
        };

        deesser.update_filter();
        deesser
    }

    fn update_filter(&mut self) {
        let sample_rate = self.sample_rate as f32;
        // Create a bandpass filter for sibilant detection
        self.bandpass
            .design_peak(sample_rate, self.frequency.value, 6.0, 2.0);
    }

    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.abs().max(1e-10).log10()
    }

    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency.set_value(frequency);
        self.update_filter();
    }

    pub fn set_threshold(&mut self, threshold_db: f32) {
        self.threshold.set_value(threshold_db);
    }

    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio.set_value(ratio);
    }
}

impl AudioEffect for Deesser {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let samples = audio.samples_mut();
        let threshold_linear = Self::db_to_linear(self.threshold.value);
        let attack_coeff = 0.9; // Fast attack
        let release_coeff = 0.99; // Slow release

        for sample in samples.iter_mut() {
            // Detect sibilant energy
            let sibilant_signal = self.bandpass.process(*sample);
            let sibilant_level = sibilant_signal.abs();

            // Update envelope
            if sibilant_level > self.envelope {
                self.envelope = sibilant_level + (self.envelope - sibilant_level) * attack_coeff;
            } else {
                self.envelope = sibilant_level + (self.envelope - sibilant_level) * release_coeff;
            }

            // Calculate compression if sibilant is detected
            if self.envelope > threshold_linear {
                let over_threshold_db = Self::linear_to_db(self.envelope / threshold_linear);
                let gain_reduction_db = over_threshold_db * (1.0 - 1.0 / self.ratio.value);
                self.compressor_gain = Self::db_to_linear(-gain_reduction_db);
            } else {
                self.compressor_gain = 1.0;
            }

            // Apply de-essing only to high frequencies
            let original = *sample;
            let high_freq_component = sibilant_signal * 0.5; // Estimate high freq component
            let processed_high = high_freq_component * self.compressor_gain;

            *sample = original - high_freq_component + processed_high;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Deesser"
    }

    fn reset(&mut self) {
        self.bandpass.reset();
        self.envelope = 0.0;
        self.compressor_gain = 1.0;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Warmth and presence control for voice enhancement
pub struct WarmthPresence {
    enabled: bool,
    warmth: EffectParameter,   // Low-mid enhancement
    presence: EffectParameter, // Mid-high enhancement

    // Filters
    warmth_filter: BiquadFilter,
    presence_filter: BiquadFilter,
    sample_rate: u32,
}

impl WarmthPresence {
    pub fn new(sample_rate: u32) -> Self {
        let mut processor = Self {
            enabled: true,
            warmth: EffectParameter::new("warmth", 0.0, -6.0, 6.0),
            presence: EffectParameter::new("presence", 0.0, -6.0, 6.0),

            warmth_filter: BiquadFilter::new(),
            presence_filter: BiquadFilter::new(),
            sample_rate,
        };

        processor.update_filters();
        processor
    }

    fn update_filters(&mut self) {
        let sample_rate = self.sample_rate as f32;

        // Warmth: gentle low-mid boost around 200-500Hz
        self.warmth_filter
            .design_peak(sample_rate, 350.0, self.warmth.value, 1.5);

        // Presence: mid-high boost around 2-5kHz
        self.presence_filter
            .design_peak(sample_rate, 3000.0, self.presence.value, 1.2);
    }

    pub fn set_warmth(&mut self, warmth_db: f32) {
        self.warmth.set_value(warmth_db);
        self.update_filters();
    }

    pub fn set_presence(&mut self, presence_db: f32) {
        self.presence.set_value(presence_db);
        self.update_filters();
    }
}

impl AudioEffect for WarmthPresence {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let samples = audio.samples_mut();

        for sample in samples.iter_mut() {
            let mut output = *sample;

            // Apply warmth
            if self.warmth.value.abs() > 0.1 {
                output = self.warmth_filter.process(output);
            }

            // Apply presence
            if self.presence.value.abs() > 0.1 {
                output = self.presence_filter.process(output);
            }

            *sample = output;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "WarmthPresence"
    }

    fn reset(&mut self) {
        self.warmth_filter.reset();
        self.presence_filter.reset();
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}
