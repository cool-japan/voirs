//! Rhythm and timing processing

#![allow(dead_code)]

use crate::score::{MusicalNote, MusicalScore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Rhythm generator for creating rhythmic patterns
#[derive(Debug, Clone)]
pub struct RhythmGenerator {
    /// Base tempo in BPM
    tempo: f32,
    /// Time signature (numerator, denominator)
    time_signature: (u8, u8),
    /// Swing factor (0.0 = straight, 1.0 = maximum swing)
    swing_factor: f32,
    /// Humanization amount (0.0 = perfect, 1.0 = maximum variation)
    humanization: f32,
    /// Groove patterns
    groove_patterns: HashMap<String, GroovePattern>,
    /// Current pattern
    current_pattern: Option<String>,
    /// Rhythm complexity (0.0 = simple, 1.0 = complex)
    complexity: f32,
    /// Accent patterns
    accent_pattern: Vec<f32>,
}

/// Rhythm processor for timing modifications
#[derive(Debug, Clone)]
pub struct RhythmProcessor {
    /// Quantization strength (0.0 = no quantization, 1.0 = perfect quantization)
    quantization: f32,
    /// Swing amount (0.0 = straight, 1.0 = maximum swing)
    swing: f32,
    /// Humanization level (0.0 = perfect timing, 1.0 = maximum variation)
    humanization: f32,
    /// Groove template
    groove_template: Option<GrooveTemplate>,
    /// Timing adjustments per beat position
    beat_adjustments: HashMap<u32, f32>,
    /// Velocity adjustments per beat position
    velocity_adjustments: HashMap<u32, f32>,
}

/// Timing controller for precise timing control
#[derive(Debug, Clone)]
pub struct TimingController {
    /// Base tempo in BPM
    base_tempo: f32,
    /// Tempo variations over time (time_in_beats, tempo)
    tempo_variations: Vec<(f32, f32)>,
    /// Fine timing adjustments (time_in_beats, adjustment_in_seconds)
    timing_adjustments: Vec<(f32, f32)>,
    /// Rubato settings
    rubato_settings: RubatoSettings,
    /// Ritardando/Accelerando curves
    tempo_curves: Vec<TempoCurve>,
    /// Current time tracking
    current_time: f32,
    /// Sample rate for precise timing
    sample_rate: f32,
}

/// Groove pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroovePattern {
    /// Pattern name
    pub name: String,
    /// Beat subdivisions (relative timing offsets)
    pub subdivisions: Vec<f32>,
    /// Accent pattern (0.0-1.0 for each subdivision)
    pub accents: Vec<f32>,
    /// Velocity variations
    pub velocities: Vec<f32>,
    /// Duration modifications
    pub durations: Vec<f32>,
    /// Pattern length in beats
    pub length: f32,
}

/// Groove template for applying timing feel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrooveTemplate {
    /// Template name
    pub name: String,
    /// Timing offsets for each 16th note position
    pub timing_offsets: Vec<f32>,
    /// Velocity multipliers for each position
    pub velocity_multipliers: Vec<f32>,
    /// Duration multipliers for each position
    pub duration_multipliers: Vec<f32>,
    /// Resolution (16th notes, 32nd notes, etc.)
    pub resolution: u32,
}

/// Rubato settings for expressive timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RubatoSettings {
    /// Rubato intensity (0.0-1.0)
    pub intensity: f32,
    /// Rubato frequency (how often tempo changes occur)
    pub frequency: f32,
    /// Rubato range (maximum tempo deviation as percentage)
    pub range: f32,
    /// Rubato curve type
    pub curve_type: RubatoCurve,
    /// Enable automatic rubato
    pub auto_rubato: bool,
}

/// Tempo curve for smooth tempo changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoCurve {
    /// Start time in beats
    pub start_time: f32,
    /// End time in beats
    pub end_time: f32,
    /// Start tempo
    pub start_tempo: f32,
    /// End tempo
    pub end_tempo: f32,
    /// Curve type
    pub curve_type: TempoCurveType,
}

/// Rubato curve types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RubatoCurve {
    /// Linear rubato
    Linear,
    /// Exponential rubato
    Exponential,
    /// Sinusoidal rubato
    Sinusoidal,
    /// Random rubato
    Random,
}

/// Tempo curve types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TempoCurveType {
    /// Linear tempo change
    Linear,
    /// Exponential tempo change
    Exponential,
    /// Logarithmic tempo change
    Logarithmic,
    /// Sinusoidal tempo change
    Sinusoidal,
}

impl RhythmGenerator {
    /// Create a new rhythm generator with the specified tempo
    ///
    /// # Arguments
    ///
    /// * `tempo` - Base tempo in BPM (beats per minute)
    ///
    /// # Returns
    ///
    /// A new `RhythmGenerator` instance with default settings and built-in groove patterns
    pub fn new(tempo: f32) -> Self {
        let mut generator = Self {
            tempo,
            time_signature: (4, 4),
            swing_factor: 0.0,
            humanization: 0.0,
            groove_patterns: HashMap::new(),
            current_pattern: None,
            complexity: 0.5,
            accent_pattern: vec![1.0, 0.7, 0.8, 0.6], // Default 4/4 pattern
        };

        // Add default groove patterns
        generator.add_default_patterns();
        generator
    }

    /// Set the time signature
    ///
    /// # Arguments
    ///
    /// * `numerator` - The number of beats per measure
    /// * `denominator` - The note value that represents one beat (e.g., 4 for quarter note)
    pub fn set_time_signature(&mut self, numerator: u8, denominator: u8) {
        self.time_signature = (numerator, denominator);
        self.update_accent_pattern();
    }

    /// Set the swing factor (0.0 = no swing, 1.0 = maximum swing)
    ///
    /// # Arguments
    ///
    /// * `swing` - Swing factor in range 0.0-1.0, where 0.0 is straight timing and 1.0 is maximum swing
    pub fn set_swing_factor(&mut self, swing: f32) {
        self.swing_factor = swing.clamp(0.0, 1.0);
    }

    /// Set the humanization level (0.0 = perfect timing, 1.0 = maximum variation)
    ///
    /// # Arguments
    ///
    /// * `humanization` - Humanization level in range 0.0-1.0, where 0.0 is perfect timing and 1.0 is maximum random variation
    pub fn set_humanization(&mut self, humanization: f32) {
        self.humanization = humanization.clamp(0.0, 1.0);
    }

    /// Set rhythm complexity
    ///
    /// # Arguments
    ///
    /// * `complexity` - Complexity level in range 0.0-1.0, where 0.0 is simple (quarter notes) and 1.0 is complex (sixteenth notes)
    pub fn set_complexity(&mut self, complexity: f32) {
        self.complexity = complexity.clamp(0.0, 1.0);
    }

    /// Add a groove pattern
    ///
    /// # Arguments
    ///
    /// * `pattern` - The groove pattern to add to the internal collection
    pub fn add_groove_pattern(&mut self, pattern: GroovePattern) {
        self.groove_patterns.insert(pattern.name.clone(), pattern);
    }

    /// Set current groove pattern
    ///
    /// # Arguments
    ///
    /// * `pattern_name` - Name of the groove pattern to activate (must exist in the collection)
    ///
    /// # Returns
    ///
    /// `Ok(())` if the pattern was found and set, or an error if the pattern doesn't exist
    ///
    /// # Errors
    ///
    /// Returns `Error::Processing` if the specified pattern name is not found in the collection
    pub fn set_groove_pattern(&mut self, pattern_name: &str) -> crate::Result<()> {
        if self.groove_patterns.contains_key(pattern_name) {
            self.current_pattern = Some(pattern_name.to_string());
            Ok(())
        } else {
            Err(crate::Error::Processing(format!(
                "Groove pattern '{}' not found",
                pattern_name
            )))
        }
    }

    /// Generate timing values for the given musical score
    ///
    /// # Arguments
    ///
    /// * `score` - The musical score to process
    ///
    /// # Returns
    ///
    /// A vector of timing values in beats, one for each note in the score, with swing, groove, and humanization applied
    pub fn generate_timing(&self, score: &MusicalScore) -> Vec<f32> {
        let mut timings = Vec::new();

        for note in &score.notes {
            let mut timing = note.start_time;

            // Apply swing
            if self.swing_factor > 0.0 {
                timing = self.apply_swing(timing);
            }

            // Apply groove pattern
            if let Some(pattern_name) = &self.current_pattern {
                if let Some(pattern) = self.groove_patterns.get(pattern_name) {
                    timing = self.apply_groove_pattern(timing, pattern);
                }
            }

            // Apply humanization
            if self.humanization > 0.0 {
                timing = self.apply_humanization(timing);
            }

            timings.push(timing);
        }

        timings
    }

    /// Generate rhythmic pattern for given duration
    ///
    /// # Arguments
    ///
    /// * `duration_beats` - Duration in beats for which to generate the pattern
    ///
    /// # Returns
    ///
    /// A vector of timing positions in beats where notes should occur, with density based on complexity setting
    pub fn generate_pattern(&self, duration_beats: f32) -> Vec<f32> {
        let mut pattern = Vec::new();
        let subdivision = match self.complexity {
            x if x < 0.3 => 1.0, // Quarter notes
            x if x < 0.7 => 0.5, // Eighth notes
            _ => 0.25,           // Sixteenth notes
        };

        let mut current_time = 0.0;
        while current_time < duration_beats {
            // Add some randomness based on complexity
            if scirs2_core::random::random::<f32>() < 0.5 + self.complexity * 0.3 {
                pattern.push(current_time);
            }
            current_time += subdivision;
        }

        pattern
    }

    /// Apply swing to timing
    fn apply_swing(&self, timing: f32) -> f32 {
        let beat_position = timing % 1.0;

        if beat_position >= 0.5 {
            // Delay off-beats (notes after the half-beat)
            let swing_offset = (beat_position - 0.5) * self.swing_factor * 0.2;
            timing + swing_offset
        } else {
            timing
        }
    }

    /// Apply groove pattern to timing
    fn apply_groove_pattern(&self, timing: f32, pattern: &GroovePattern) -> f32 {
        let pattern_position = timing % pattern.length;
        let subdivision_index =
            (pattern_position / (pattern.length / pattern.subdivisions.len() as f32)) as usize;

        if subdivision_index < pattern.subdivisions.len() {
            timing + pattern.subdivisions[subdivision_index] * 0.05 // Small timing adjustment
        } else {
            timing
        }
    }

    /// Apply humanization to timing
    fn apply_humanization(&self, timing: f32) -> f32 {
        let variation = (scirs2_core::random::random::<f32>() - 0.5) * self.humanization * 0.05;
        timing + variation
    }

    /// Add default groove patterns
    fn add_default_patterns(&mut self) {
        // Straight pattern
        let straight = GroovePattern {
            name: "straight".to_string(),
            subdivisions: vec![0.0, 0.0, 0.0, 0.0],
            accents: vec![1.0, 0.7, 0.8, 0.6],
            velocities: vec![1.0, 0.8, 0.9, 0.7],
            durations: vec![1.0, 1.0, 1.0, 1.0],
            length: 1.0,
        };

        // Swing pattern
        let swing = GroovePattern {
            name: "swing".to_string(),
            subdivisions: vec![0.0, 0.02, 0.0, 0.02],
            accents: vec![1.0, 0.6, 0.8, 0.5],
            velocities: vec![1.0, 0.7, 0.9, 0.6],
            durations: vec![1.1, 0.9, 1.1, 0.9],
            length: 1.0,
        };

        // Shuffle pattern
        let shuffle = GroovePattern {
            name: "shuffle".to_string(),
            subdivisions: vec![0.0, 0.05, -0.02, 0.03],
            accents: vec![1.0, 0.5, 0.9, 0.4],
            velocities: vec![1.0, 0.6, 0.95, 0.5],
            durations: vec![1.2, 0.8, 1.1, 0.8],
            length: 1.0,
        };

        self.groove_patterns
            .insert("straight".to_string(), straight);
        self.groove_patterns.insert("swing".to_string(), swing);
        self.groove_patterns.insert("shuffle".to_string(), shuffle);
    }

    /// Update accent pattern based on time signature
    fn update_accent_pattern(&mut self) {
        match self.time_signature {
            (4, 4) => self.accent_pattern = vec![1.0, 0.7, 0.8, 0.6],
            (3, 4) => self.accent_pattern = vec![1.0, 0.6, 0.7],
            (2, 4) => self.accent_pattern = vec![1.0, 0.7],
            (6, 8) => self.accent_pattern = vec![1.0, 0.5, 0.6, 0.8, 0.5, 0.6],
            _ => self.accent_pattern = vec![1.0; self.time_signature.0 as usize],
        }
    }
}

impl RhythmProcessor {
    /// Create a new rhythm processor
    ///
    /// # Returns
    ///
    /// A new `RhythmProcessor` instance with default settings (no quantization, swing, or humanization)
    pub fn new() -> Self {
        Self {
            quantization: 0.0,
            swing: 0.0,
            humanization: 0.0,
            groove_template: None,
            beat_adjustments: HashMap::new(),
            velocity_adjustments: HashMap::new(),
        }
    }

    /// Set the quantization level
    ///
    /// # Arguments
    ///
    /// * `quantization` - Quantization strength in range 0.0-1.0, where 0.0 is no quantization and 1.0 is perfect grid alignment
    pub fn set_quantization(&mut self, quantization: f32) {
        self.quantization = quantization.clamp(0.0, 1.0);
    }

    /// Set the swing amount
    ///
    /// # Arguments
    ///
    /// * `swing` - Swing amount in range 0.0-1.0, where 0.0 is straight timing and 1.0 is maximum swing
    pub fn set_swing(&mut self, swing: f32) {
        self.swing = swing.clamp(0.0, 1.0);
    }

    /// Set the humanization level
    ///
    /// # Arguments
    ///
    /// * `humanization` - Humanization level in range 0.0-1.0, where 0.0 is perfect timing and 1.0 is maximum random variation
    pub fn set_humanization(&mut self, humanization: f32) {
        self.humanization = humanization.clamp(0.0, 1.0);
    }

    /// Set groove template
    ///
    /// # Arguments
    ///
    /// * `template` - The groove template to apply to timing processing
    pub fn set_groove_template(&mut self, template: GrooveTemplate) {
        self.groove_template = Some(template);
    }

    /// Add beat timing adjustment
    ///
    /// # Arguments
    ///
    /// * `beat_position` - The beat position (in 16th note resolution) to adjust
    /// * `adjustment` - Timing adjustment in beats to add at this position
    pub fn add_beat_adjustment(&mut self, beat_position: u32, adjustment: f32) {
        self.beat_adjustments.insert(beat_position, adjustment);
    }

    /// Add velocity adjustment
    ///
    /// # Arguments
    ///
    /// * `beat_position` - The beat position (in 16th note resolution) to adjust
    /// * `adjustment` - Velocity multiplier in range 0.0-1.0 to apply at this position
    pub fn add_velocity_adjustment(&mut self, beat_position: u32, adjustment: f32) {
        self.velocity_adjustments.insert(beat_position, adjustment);
    }

    /// Process timing values
    ///
    /// # Arguments
    ///
    /// * `timings` - Mutable slice of timing values in beats to process in-place
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    ///
    /// # Errors
    ///
    /// Currently does not return errors, but uses Result for API consistency
    pub fn process_timing(&self, timings: &mut [f32]) -> crate::Result<()> {
        for timing in timings.iter_mut() {
            *timing = self.process_single_timing(*timing);
        }
        Ok(())
    }

    /// Process musical score timing
    ///
    /// # Arguments
    ///
    /// * `score` - Mutable reference to the musical score to process
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    ///
    /// # Errors
    ///
    /// Currently does not return errors, but uses Result for API consistency
    pub fn process_score(&self, score: &mut MusicalScore) -> crate::Result<()> {
        for note in &mut score.notes {
            // Process timing
            note.start_time = self.process_single_timing(note.start_time);

            // Apply velocity adjustments
            if let Some(adjustment) = self.get_velocity_adjustment(note.start_time) {
                note.event.velocity = (note.event.velocity * adjustment).clamp(0.0, 1.0);
            }

            // Apply groove template if available
            if let Some(template) = &self.groove_template {
                self.apply_groove_template_to_note(note, template);
            }
        }
        Ok(())
    }

    /// Process single timing value
    fn process_single_timing(&self, timing: f32) -> f32 {
        let mut processed_timing = timing;

        // Apply quantization
        if self.quantization > 0.0 {
            processed_timing = self.apply_quantization(processed_timing);
        }

        // Apply swing
        if self.swing > 0.0 {
            processed_timing = self.apply_swing(processed_timing);
        }

        // Apply humanization
        if self.humanization > 0.0 {
            processed_timing = self.apply_humanization(processed_timing);
        }

        // Apply beat adjustments
        if let Some(adjustment) = self.get_beat_adjustment(processed_timing) {
            processed_timing += adjustment;
        }

        processed_timing
    }

    /// Apply quantization to timing
    fn apply_quantization(&self, timing: f32) -> f32 {
        let grid_size = 0.25; // 16th note grid
        let quantized = (timing / grid_size).round() * grid_size;
        timing * (1.0 - self.quantization) + quantized * self.quantization
    }

    /// Apply swing to timing
    fn apply_swing(&self, timing: f32) -> f32 {
        let beat_position = timing % 1.0;

        if beat_position >= 0.5 {
            // Delay off-beats (notes after the half-beat)
            let swing_offset = (beat_position - 0.5) * self.swing * 0.2;
            timing + swing_offset
        } else {
            timing
        }
    }

    /// Apply humanization to timing
    fn apply_humanization(&self, timing: f32) -> f32 {
        let variation = (scirs2_core::random::random::<f32>() - 0.5) * self.humanization * 0.03;
        timing + variation
    }

    /// Get beat adjustment for timing
    fn get_beat_adjustment(&self, timing: f32) -> Option<f32> {
        let beat_position = (timing * 4.0) as u32; // 16th note resolution
        self.beat_adjustments.get(&beat_position).copied()
    }

    /// Get velocity adjustment for timing
    fn get_velocity_adjustment(&self, timing: f32) -> Option<f32> {
        let beat_position = (timing * 4.0) as u32; // 16th note resolution
        self.velocity_adjustments.get(&beat_position).copied()
    }

    /// Apply groove template to note
    fn apply_groove_template_to_note(&self, note: &mut MusicalNote, template: &GrooveTemplate) {
        let position_in_template =
            (note.start_time * template.resolution as f32) as usize % template.timing_offsets.len();

        if position_in_template < template.timing_offsets.len() {
            // Apply timing offset
            note.start_time += template.timing_offsets[position_in_template];

            // Apply velocity multiplier
            if position_in_template < template.velocity_multipliers.len() {
                note.event.velocity *= template.velocity_multipliers[position_in_template];
                note.event.velocity = note.event.velocity.clamp(0.0, 1.0);
            }

            // Apply duration multiplier
            if position_in_template < template.duration_multipliers.len() {
                note.duration *= template.duration_multipliers[position_in_template];
            }
        }
    }
}

impl TimingController {
    /// Create a new timing controller with base tempo
    ///
    /// # Arguments
    ///
    /// * `base_tempo` - Base tempo in BPM (beats per minute)
    ///
    /// # Returns
    ///
    /// A new `TimingController` instance with default sample rate of 44100 Hz
    pub fn new(base_tempo: f32) -> Self {
        Self {
            base_tempo,
            tempo_variations: Vec::new(),
            timing_adjustments: Vec::new(),
            rubato_settings: RubatoSettings::default(),
            tempo_curves: Vec::new(),
            current_time: 0.0,
            sample_rate: 44100.0,
        }
    }

    /// Create with sample rate
    ///
    /// # Arguments
    ///
    /// * `base_tempo` - Base tempo in BPM (beats per minute)
    /// * `sample_rate` - Sample rate in Hz for precise timing calculations
    ///
    /// # Returns
    ///
    /// A new `TimingController` instance with the specified sample rate
    pub fn with_sample_rate(base_tempo: f32, sample_rate: f32) -> Self {
        let mut controller = Self::new(base_tempo);
        controller.sample_rate = sample_rate;
        controller
    }

    /// Set rubato settings
    ///
    /// # Arguments
    ///
    /// * `settings` - Rubato settings to apply for expressive timing variations
    pub fn set_rubato_settings(&mut self, settings: RubatoSettings) {
        self.rubato_settings = settings;
    }

    /// Add a tempo variation at a specific time
    ///
    /// # Arguments
    ///
    /// * `time` - Time in beats where the tempo change occurs
    /// * `tempo` - New tempo in BPM to apply from this point forward
    pub fn add_tempo_variation(&mut self, time: f32, tempo: f32) {
        self.tempo_variations.push((time, tempo));
        self.tempo_variations
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    /// Add a timing adjustment at a specific time
    ///
    /// # Arguments
    ///
    /// * `time` - Time in beats where the adjustment occurs
    /// * `adjustment` - Fine timing adjustment in seconds to add at this point
    pub fn add_timing_adjustment(&mut self, time: f32, adjustment: f32) {
        self.timing_adjustments.push((time, adjustment));
        self.timing_adjustments
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    /// Add tempo curve
    ///
    /// # Arguments
    ///
    /// * `curve` - Tempo curve defining a smooth tempo change over a time range
    pub fn add_tempo_curve(&mut self, curve: TempoCurve) {
        self.tempo_curves.push(curve);
        self.tempo_curves
            .sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
    }

    /// Get the tempo at a specific time
    ///
    /// # Arguments
    ///
    /// * `time` - Time in beats at which to query the tempo
    ///
    /// # Returns
    ///
    /// The effective tempo in BPM at the specified time, including all curves, variations, and rubato
    pub fn get_tempo_at_time(&self, time: f32) -> f32 {
        let mut tempo = self.base_tempo;

        // Apply tempo curves
        for curve in &self.tempo_curves {
            if time >= curve.start_time && time <= curve.end_time {
                let progress = (time - curve.start_time) / (curve.end_time - curve.start_time);
                let curve_progress = self.apply_tempo_curve(progress, curve.curve_type);
                tempo = curve.start_tempo + (curve.end_tempo - curve.start_tempo) * curve_progress;
                break;
            }
        }

        // Apply tempo variations
        if let Some((_, variation_tempo)) = self
            .tempo_variations
            .iter()
            .rev()
            .find(|(var_time, _)| *var_time <= time)
        {
            tempo = *variation_tempo;
        }

        // Apply rubato if enabled
        if self.rubato_settings.auto_rubato {
            tempo = self.apply_rubato(tempo, time);
        }

        tempo
    }

    /// Get the timing adjustment at a specific time
    ///
    /// # Arguments
    ///
    /// * `time` - Time in beats at which to query the adjustment
    ///
    /// # Returns
    ///
    /// The timing adjustment in seconds at the specified time, or 0.0 if none is defined
    pub fn get_timing_adjustment(&self, time: f32) -> f32 {
        self.timing_adjustments
            .iter()
            .rev()
            .find(|(adj_time, _)| *adj_time <= time)
            .map(|(_, adjustment)| *adjustment)
            .unwrap_or(0.0)
    }

    /// Convert beats to seconds at given time
    ///
    /// # Arguments
    ///
    /// * `beats` - Number of beats to convert
    /// * `start_time` - Starting time in beats from which to begin the conversion
    ///
    /// # Returns
    ///
    /// The duration in seconds, accounting for tempo changes over the beat range
    pub fn beats_to_seconds(&self, beats: f32, start_time: f32) -> f32 {
        if beats == 0.0 {
            return 0.0;
        }

        let subdivision = 0.01; // Small subdivision for accurate tempo following
        let mut total_time = 0.0;
        let mut current_beat = start_time;

        while current_beat < start_time + beats {
            let current_tempo = self.get_tempo_at_time(current_beat);
            let beat_duration = 60.0 / current_tempo;
            total_time += beat_duration * subdivision;
            current_beat += subdivision;
        }

        total_time
    }

    /// Convert seconds to beats at given time
    ///
    /// # Arguments
    ///
    /// * `seconds` - Duration in seconds to convert
    /// * `start_time` - Starting time in beats from which to begin the conversion
    ///
    /// # Returns
    ///
    /// The number of beats equivalent to the given duration, accounting for tempo changes
    pub fn seconds_to_beats(&self, seconds: f32, start_time: f32) -> f32 {
        if seconds == 0.0 {
            return 0.0;
        }

        let subdivision = 0.001; // Small time subdivision
        let mut total_beats = 0.0;
        let mut current_time = 0.0;

        while current_time < seconds {
            let current_beat = start_time + total_beats;
            let current_tempo = self.get_tempo_at_time(current_beat);
            let beats_per_second = current_tempo / 60.0;
            total_beats += beats_per_second * subdivision;
            current_time += subdivision;
        }

        total_beats
    }

    /// Process score timing with tempo control
    ///
    /// # Arguments
    ///
    /// * `score` - Mutable reference to the musical score to process
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    ///
    /// # Errors
    ///
    /// Currently does not return errors, but uses Result for API consistency
    pub fn process_score_timing(&self, score: &mut MusicalScore) -> crate::Result<()> {
        for note in &mut score.notes {
            // Apply tempo variations to note timing
            let adjusted_start = note.start_time + self.get_timing_adjustment(note.start_time);
            note.start_time = adjusted_start;

            // Adjust duration based on tempo changes
            let start_seconds = self.beats_to_seconds(note.start_time, 0.0);
            let end_seconds = self.beats_to_seconds(note.start_time + note.duration, 0.0);
            let duration_seconds = end_seconds - start_seconds;

            // Convert back to beats using average tempo
            let avg_tempo = (self.get_tempo_at_time(note.start_time)
                + self.get_tempo_at_time(note.start_time + note.duration))
                / 2.0;
            note.duration = duration_seconds * avg_tempo / 60.0;
        }

        Ok(())
    }

    /// Apply tempo curve interpolation
    fn apply_tempo_curve(&self, progress: f32, curve_type: TempoCurveType) -> f32 {
        match curve_type {
            TempoCurveType::Linear => progress,
            TempoCurveType::Exponential => progress * progress,
            TempoCurveType::Logarithmic => 1.0 - (1.0 - progress) * (1.0 - progress),
            TempoCurveType::Sinusoidal => (progress * std::f32::consts::PI / 2.0).sin(),
        }
    }

    /// Apply rubato to tempo
    fn apply_rubato(&self, base_tempo: f32, time: f32) -> f32 {
        let rubato_value = match self.rubato_settings.curve_type {
            RubatoCurve::Linear => (time * self.rubato_settings.frequency) % 1.0,
            RubatoCurve::Exponential => ((time * self.rubato_settings.frequency) % 1.0).powf(2.0),
            RubatoCurve::Sinusoidal => {
                (time * self.rubato_settings.frequency * 2.0 * std::f32::consts::PI).sin()
            }
            RubatoCurve::Random => scirs2_core::random::random::<f32>(),
        };

        let rubato_offset = (rubato_value - 0.5)
            * 2.0
            * self.rubato_settings.intensity
            * self.rubato_settings.range
            / 100.0;
        base_tempo * (1.0 + rubato_offset)
    }

    /// Update current time (for real-time processing)
    ///
    /// # Arguments
    ///
    /// * `delta_seconds` - Time increment in seconds to advance the internal clock
    pub fn update_time(&mut self, delta_seconds: f32) {
        let beats_per_second = self.get_tempo_at_time(self.current_time) / 60.0;
        self.current_time += delta_seconds * beats_per_second;
    }

    /// Reset timing controller
    pub fn reset(&mut self) {
        self.current_time = 0.0;
    }
}

/// Default implementation for RhythmProcessor
///
/// Creates a rhythm processor with no quantization, swing, or humanization applied.
impl Default for RhythmProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Default implementation for RubatoSettings
///
/// Creates rubato settings with moderate intensity (0.3), low frequency (0.1),
/// 5% tempo variation range, sinusoidal curve, and auto-rubato disabled.
impl Default for RubatoSettings {
    fn default() -> Self {
        Self {
            intensity: 0.3,
            frequency: 0.1,
            range: 5.0, // 5% tempo variation
            curve_type: RubatoCurve::Sinusoidal,
            auto_rubato: false,
        }
    }
}

/// Default implementation for GrooveTemplate
///
/// Creates a straight groove template with no timing, velocity, or duration modifications,
/// using 16th note resolution (16 positions per beat).
impl Default for GrooveTemplate {
    fn default() -> Self {
        Self {
            name: "straight".to_string(),
            timing_offsets: vec![0.0; 16], // 16th note resolution
            velocity_multipliers: vec![1.0; 16],
            duration_multipliers: vec![1.0; 16],
            resolution: 16,
        }
    }
}

impl GrooveTemplate {
    /// Create swing groove template
    ///
    /// # Returns
    ///
    /// A `GrooveTemplate` configured for swing timing with delayed off-beats and dynamic accents
    pub fn swing() -> Self {
        let mut template = Self::default();
        template.name = "swing".to_string();

        // Apply swing timing - delay off-beats
        for i in (1..16).step_by(2) {
            template.timing_offsets[i] = 0.05; // Delay off-beats
        }

        // Accent on-beats more
        for i in (0..16).step_by(2) {
            template.velocity_multipliers[i] = 1.1;
        }
        for i in (1..16).step_by(2) {
            template.velocity_multipliers[i] = 0.8;
        }

        template
    }

    /// Create shuffle groove template
    ///
    /// # Returns
    ///
    /// A `GrooveTemplate` configured for shuffle timing with characteristic triplet-based feel
    pub fn shuffle() -> Self {
        let mut template = Self::default();
        template.name = "shuffle".to_string();

        // Shuffle timing pattern
        for i in 0..16 {
            match i % 4 {
                0 => template.timing_offsets[i] = 0.0,
                1 => template.timing_offsets[i] = 0.08,
                2 => template.timing_offsets[i] = -0.02,
                3 => template.timing_offsets[i] = 0.05,
                _ => {}
            }
        }

        template
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::MusicalNote;
    use crate::types::NoteEvent;

    #[test]
    fn test_rhythm_generator_creation() {
        let generator = RhythmGenerator::new(120.0);
        assert_eq!(generator.tempo, 120.0);
        assert_eq!(generator.time_signature, (4, 4));
        assert!(generator.groove_patterns.contains_key("straight"));
        assert!(generator.groove_patterns.contains_key("swing"));
    }

    #[test]
    fn test_rhythm_generator_groove_patterns() {
        let mut generator = RhythmGenerator::new(120.0);

        assert!(generator.set_groove_pattern("swing").is_ok());
        assert_eq!(generator.current_pattern, Some("swing".to_string()));

        assert!(generator.set_groove_pattern("nonexistent").is_err());
    }

    #[test]
    fn test_rhythm_generator_timing() {
        let generator = RhythmGenerator::new(120.0);

        let mut score = MusicalScore::new("Test".to_string(), "Test".to_string());
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let note = MusicalNote::new(event, 0.0, 1.0);
        score.add_note(note);

        let timings = generator.generate_timing(&score);
        assert_eq!(timings.len(), 1);
        assert!(timings[0] >= 0.0); // Should be non-negative
    }

    #[test]
    fn test_rhythm_generator_swing() {
        let mut generator = RhythmGenerator::new(120.0);
        generator.set_swing_factor(1.0); // Maximum swing for clearer test

        let straight_timing = 1.75; // Clearly off-beat (3/4 position)
        let swung_timing = generator.apply_swing(straight_timing);

        // With max swing, off-beat should be noticeably delayed
        assert!(
            swung_timing > straight_timing,
            "Swung timing {} should be greater than straight timing {}",
            swung_timing,
            straight_timing
        );
    }

    #[test]
    fn test_rhythm_processor() {
        let mut processor = RhythmProcessor::new();
        processor.set_quantization(0.5);
        processor.set_swing(0.3);
        processor.set_humanization(0.1);

        let mut timings = vec![0.1, 0.6, 1.1, 1.6];
        processor.process_timing(&mut timings).unwrap();

        // Timings should be modified
        assert_ne!(timings, vec![0.1, 0.6, 1.1, 1.6]);
    }

    #[test]
    fn test_timing_controller() {
        let mut controller = TimingController::new(120.0);

        // Add tempo variation
        controller.add_tempo_variation(2.0, 140.0);

        assert_eq!(controller.get_tempo_at_time(0.0), 120.0);
        assert_eq!(controller.get_tempo_at_time(3.0), 140.0);
    }

    #[test]
    fn test_timing_controller_curves() {
        let mut controller = TimingController::new(120.0);

        let curve = TempoCurve {
            start_time: 0.0,
            end_time: 4.0,
            start_tempo: 120.0,
            end_tempo: 140.0,
            curve_type: TempoCurveType::Linear,
        };
        controller.add_tempo_curve(curve);

        let mid_tempo = controller.get_tempo_at_time(2.0);
        assert!((mid_tempo - 130.0).abs() < 1.0); // Should be approximately halfway
    }

    #[test]
    fn test_beats_to_seconds_conversion() {
        let controller = TimingController::new(120.0); // 120 BPM = 2 beats per second
        let seconds = controller.beats_to_seconds(4.0, 0.0); // 4 beats
        assert!((seconds - 2.0).abs() < 0.1); // Should be approximately 2 seconds
    }

    #[test]
    fn test_groove_template_swing() {
        let swing_template = GrooveTemplate::swing();
        assert_eq!(swing_template.name, "swing");

        // Check that off-beats have timing delays
        for i in (1..16).step_by(2) {
            assert!(swing_template.timing_offsets[i] > 0.0);
        }
    }

    #[test]
    fn test_rubato_settings() {
        let settings = RubatoSettings::default();
        assert_eq!(settings.intensity, 0.3);
        assert_eq!(settings.curve_type, RubatoCurve::Sinusoidal);
        assert!(!settings.auto_rubato);
    }

    #[test]
    fn test_processor_score_processing() {
        let processor = RhythmProcessor::new();

        let mut score = MusicalScore::new("Test".to_string(), "Test".to_string());
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let note = MusicalNote::new(event, 1.0, 1.0);
        score.add_note(note);

        let original_start = score.notes[0].start_time;
        processor.process_score(&mut score).unwrap();

        // With default settings, timing should be unchanged
        assert_eq!(score.notes[0].start_time, original_start);
    }

    #[test]
    fn test_timing_controller_update() {
        let mut controller = TimingController::new(120.0);
        let initial_time = controller.current_time;

        controller.update_time(1.0); // 1 second
        assert!(controller.current_time > initial_time);

        controller.reset();
        assert_eq!(controller.current_time, 0.0);
    }
}
