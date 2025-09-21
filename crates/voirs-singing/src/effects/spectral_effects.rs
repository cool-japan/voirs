//! Spectral audio effects for singing voice

use super::core::SingingEffect;
use super::filters::{HighShelfFilter, LowShelfFilter, PeakingFilter};
use super::helpers::{
    AntiFormantFilter, FormantFilter, InterpolationType, MorphType, PhaseAlignment,
};
use std::collections::HashMap;

/// EQ effect with multiple bands
#[derive(Debug, Clone)]
pub struct EQEffect {
    name: String,
    parameters: HashMap<String, f32>,
    filters: Vec<PeakingFilter>,
    low_shelf: LowShelfFilter,
    high_shelf: HighShelfFilter,
}

impl EQEffect {
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "eq".to_string(),
            parameters: HashMap::new(),
            filters: Vec::new(),
            low_shelf: LowShelfFilter::new(100.0, 0.0),
            high_shelf: HighShelfFilter::new(8000.0, 0.0),
        };

        // Initialize default parameters
        effect
            .parameters
            .insert("low_shelf_freq".to_string(), 100.0);
        effect.parameters.insert("low_shelf_gain".to_string(), 0.0);
        effect
            .parameters
            .insert("high_shelf_freq".to_string(), 8000.0);
        effect.parameters.insert("high_shelf_gain".to_string(), 0.0);
        effect.parameters.insert("band_count".to_string(), 3.0);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        // Initialize parametric EQ bands
        effect.initialize_bands();
        effect
    }

    fn initialize_bands(&mut self) {
        let band_count = *self.parameters.get("band_count").unwrap_or(&3.0) as usize;
        self.filters.clear();

        for i in 0..band_count {
            let freq = 200.0 * (i + 1) as f32;
            let gain = 0.0;
            let q = 1.0;
            self.filters.push(PeakingFilter::new(freq, gain, q));

            self.parameters.insert(format!("band_{}_freq", i), freq);
            self.parameters.insert(format!("band_{}_gain", i), gain);
            self.parameters.insert(format!("band_{}_q", i), q);
        }
    }
}

impl SingingEffect for EQEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        for sample in audio.iter_mut() {
            // Process through low shelf
            *sample = self.low_shelf.process(*sample, sample_rate);

            // Process through parametric bands
            for filter in &mut self.filters {
                *sample = filter.process(*sample, sample_rate);
            }

            // Process through high shelf
            *sample = self.high_shelf.process(*sample, sample_rate);
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            "low_shelf_freq" => {
                self.low_shelf.set_freq(value);
            }
            "low_shelf_gain" => {
                self.low_shelf.set_gain(value);
            }
            "high_shelf_freq" => {
                self.high_shelf.set_freq(value);
            }
            "high_shelf_gain" => {
                self.high_shelf.set_gain(value);
            }
            "band_count" => {
                self.initialize_bands();
            }
            name if name.starts_with("band_") => {
                if let Some(band_part) = name.strip_prefix("band_") {
                    if let Some(underscore_pos) = band_part.find('_') {
                        if let Ok(band_index) = band_part[..underscore_pos].parse::<usize>() {
                            if band_index < self.filters.len() {
                                let param_type = &band_part[underscore_pos + 1..];
                                match param_type {
                                    "freq" => self.filters[band_index].set_freq(value),
                                    "gain" => self.filters[band_index].set_gain(value),
                                    "q" => self.filters[band_index].set_q(value),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        self.low_shelf.reset();
        self.high_shelf.reset();
        for filter in &mut self.filters {
            filter.reset();
        }
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

/// Formant control effect for direct formant frequency manipulation
#[derive(Debug, Clone)]
pub struct FormantControlEffect {
    name: String,
    parameters: HashMap<String, f32>,
    formant_filters: Vec<FormantFilter>,
    anti_formant_filters: Vec<AntiFormantFilter>,
}

impl FormantControlEffect {
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "formant".to_string(),
            parameters: HashMap::new(),
            formant_filters: Vec::new(),
            anti_formant_filters: Vec::new(),
        };

        // Initialize default parameters for vowel formants
        effect.parameters.insert("f1_freq".to_string(), 700.0);
        effect.parameters.insert("f1_gain".to_string(), 6.0);
        effect.parameters.insert("f1_bandwidth".to_string(), 80.0);
        effect.parameters.insert("f2_freq".to_string(), 1220.0);
        effect.parameters.insert("f2_gain".to_string(), 4.0);
        effect.parameters.insert("f2_bandwidth".to_string(), 100.0);
        effect.parameters.insert("f3_freq".to_string(), 2600.0);
        effect.parameters.insert("f3_gain".to_string(), 2.0);
        effect.parameters.insert("f3_bandwidth".to_string(), 150.0);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect.initialize_formants();
        effect
    }

    fn initialize_formants(&mut self) {
        self.formant_filters.clear();

        // Create formant filters
        let f1_freq = *self.parameters.get("f1_freq").unwrap_or(&700.0);
        let f1_gain = *self.parameters.get("f1_gain").unwrap_or(&6.0);
        let f1_bandwidth = *self.parameters.get("f1_bandwidth").unwrap_or(&80.0);
        self.formant_filters
            .push(FormantFilter::new(f1_freq, f1_bandwidth, f1_gain));

        let f2_freq = *self.parameters.get("f2_freq").unwrap_or(&1220.0);
        let f2_gain = *self.parameters.get("f2_gain").unwrap_or(&4.0);
        let f2_bandwidth = *self.parameters.get("f2_bandwidth").unwrap_or(&100.0);
        self.formant_filters
            .push(FormantFilter::new(f2_freq, f2_bandwidth, f2_gain));

        let f3_freq = *self.parameters.get("f3_freq").unwrap_or(&2600.0);
        let f3_gain = *self.parameters.get("f3_gain").unwrap_or(&2.0);
        let f3_bandwidth = *self.parameters.get("f3_bandwidth").unwrap_or(&150.0);
        self.formant_filters
            .push(FormantFilter::new(f3_freq, f3_bandwidth, f3_gain));
    }
}

impl SingingEffect for FormantControlEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        for sample in audio.iter_mut() {
            for filter in &mut self.formant_filters {
                *sample = filter.process(*sample, sample_rate);
            }
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            name if name.starts_with("f1_") => {
                if let Some(formant) = self.formant_filters.get_mut(0) {
                    match &name[3..] {
                        "freq" => formant.set_parameters(
                            value,
                            *self.parameters.get("f1_bandwidth").unwrap_or(&80.0),
                            *self.parameters.get("f1_gain").unwrap_or(&6.0),
                        ),
                        "bandwidth" => formant.set_parameters(
                            *self.parameters.get("f1_freq").unwrap_or(&700.0),
                            value,
                            *self.parameters.get("f1_gain").unwrap_or(&6.0),
                        ),
                        "gain" => formant.set_parameters(
                            *self.parameters.get("f1_freq").unwrap_or(&700.0),
                            *self.parameters.get("f1_bandwidth").unwrap_or(&80.0),
                            value,
                        ),
                        _ => {}
                    }
                }
            }
            name if name.starts_with("f2_") => {
                if let Some(formant) = self.formant_filters.get_mut(1) {
                    match &name[3..] {
                        "freq" => formant.set_parameters(
                            value,
                            *self.parameters.get("f2_bandwidth").unwrap_or(&100.0),
                            *self.parameters.get("f2_gain").unwrap_or(&4.0),
                        ),
                        "bandwidth" => formant.set_parameters(
                            *self.parameters.get("f2_freq").unwrap_or(&1220.0),
                            value,
                            *self.parameters.get("f2_gain").unwrap_or(&4.0),
                        ),
                        "gain" => formant.set_parameters(
                            *self.parameters.get("f2_freq").unwrap_or(&1220.0),
                            *self.parameters.get("f2_bandwidth").unwrap_or(&100.0),
                            value,
                        ),
                        _ => {}
                    }
                }
            }
            name if name.starts_with("f3_") => {
                if let Some(formant) = self.formant_filters.get_mut(2) {
                    match &name[3..] {
                        "freq" => formant.set_parameters(
                            value,
                            *self.parameters.get("f3_bandwidth").unwrap_or(&150.0),
                            *self.parameters.get("f3_gain").unwrap_or(&2.0),
                        ),
                        "bandwidth" => formant.set_parameters(
                            *self.parameters.get("f3_freq").unwrap_or(&2600.0),
                            value,
                            *self.parameters.get("f3_gain").unwrap_or(&2.0),
                        ),
                        "gain" => formant.set_parameters(
                            *self.parameters.get("f3_freq").unwrap_or(&2600.0),
                            *self.parameters.get("f3_bandwidth").unwrap_or(&150.0),
                            value,
                        ),
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        for filter in &mut self.formant_filters {
            filter.reset();
        }
        for filter in &mut self.anti_formant_filters {
            filter.reset();
        }
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

/// Spectral morphing effect for voice transformation
#[derive(Debug, Clone)]
pub struct SpectralMorphingEffect {
    name: String,
    parameters: HashMap<String, f32>,
    morph_type: MorphType,
    interpolation: InterpolationType,
    phase_alignment: PhaseAlignment,
    morph_amount: f32,
}

impl SpectralMorphingEffect {
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "spectral_morph".to_string(),
            parameters: HashMap::new(),
            morph_type: MorphType::Linear,
            interpolation: InterpolationType::Linear,
            phase_alignment: PhaseAlignment::None,
            morph_amount: 0.5,
        };

        // Initialize default parameters
        effect.parameters.insert("morph_amount".to_string(), 0.5);
        effect.parameters.insert("morph_type".to_string(), 0.0); // Linear
        effect.parameters.insert("interpolation".to_string(), 0.0); // Linear
        effect.parameters.insert("phase_alignment".to_string(), 0.0); // None

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect
    }
}

impl SingingEffect for SpectralMorphingEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], _sample_rate: f32) -> crate::Result<()> {
        // Simplified spectral morphing - in practice this would involve FFT
        let morph_amount = *self.parameters.get("morph_amount").unwrap_or(&0.5);

        for sample in audio.iter_mut() {
            // Apply simple morphing by scaling
            *sample *= 1.0 + (morph_amount - 0.5) * 0.2;
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            "morph_amount" => {
                self.morph_amount = value.clamp(0.0, 1.0);
            }
            "morph_type" => {
                self.morph_type = match value as i32 {
                    0 => MorphType::Linear,
                    1 => MorphType::CrossFade,
                    2 => MorphType::SpectralEnvelope,
                    3 => MorphType::HarmonicMorph,
                    4 => MorphType::FormantMorph,
                    5 => MorphType::TimbreTransfer,
                    _ => MorphType::Linear,
                };
            }
            "interpolation" => {
                self.interpolation = match value as i32 {
                    0 => InterpolationType::Linear,
                    1 => InterpolationType::Cubic,
                    2 => InterpolationType::Spline,
                    _ => InterpolationType::Linear,
                };
            }
            "phase_alignment" => {
                self.phase_alignment = match value as i32 {
                    0 => PhaseAlignment::None,
                    1 => PhaseAlignment::Linear,
                    2 => PhaseAlignment::CrossCorrelation,
                    3 => PhaseAlignment::PhaseLock,
                    _ => PhaseAlignment::None,
                };
            }
            _ => {}
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        // Reset any internal state
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}
