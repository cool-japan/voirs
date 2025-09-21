//! Advanced singing techniques and vocal processing
//!
//! This module implements sophisticated vocal techniques including vocal runs,
//! pitch bends, advanced dynamics control, and enhanced articulation.

#![allow(dead_code, clippy::derivable_impls)]

use crate::types::{Articulation, NoteEvent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Advanced vocal techniques processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTechniques {
    /// Vocal runs and ornament processor
    pub vocal_runs: VocalRunProcessor,
    /// Pitch bend processor
    pub pitch_bends: PitchBendProcessor,
    /// Advanced dynamics controller
    pub dynamics: AdvancedDynamicsProcessor,
    /// Enhanced articulation processor
    pub articulation: AdvancedArticulationProcessor,
    /// Melismatic processing
    pub melisma: MelismaProcessor,
    /// Grace note processor
    pub grace_notes: GraceNoteProcessor,
}

/// Vocal run patterns and ornaments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalRunProcessor {
    /// Run generation settings
    pub settings: VocalRunSettings,
    /// Current run state
    pub run_state: RunState,
    /// Generated run patterns
    pub patterns: HashMap<String, RunPattern>,
}

/// Vocal run configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalRunSettings {
    /// Run density (notes per second)
    pub density: f32,
    /// Run complexity (0.0-1.0)
    pub complexity: f32,
    /// Run range in semitones
    pub range: f32,
    /// Run direction preference (-1.0 to 1.0, negative = down, positive = up)
    pub direction_bias: f32,
    /// Run rhythm pattern
    pub rhythm_pattern: RunRhythm,
    /// Enable automatic run insertion
    pub auto_insertion: bool,
    /// Run probability (0.0-1.0)
    pub probability: f32,
    /// Preferred run scales
    pub scales: Vec<MusicScale>,
    /// Run velocity curve
    pub velocity_curve: VelocityCurve,
    /// Run legato factor
    pub legato_factor: f32,
}

/// Current run state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunState {
    /// Currently executing run
    pub active_run: Option<RunPattern>,
    /// Run position (0.0-1.0)
    pub position: f32,
    /// Run phase
    pub phase: RunPhase,
    /// Time since run start
    pub time: f32,
    /// Next run trigger time
    pub next_trigger: f32,
}

/// Run pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunPattern {
    /// Pattern name
    pub name: String,
    /// Note sequence (semitone offsets)
    pub notes: Vec<f32>,
    /// Note durations (as fractions of beat)
    pub durations: Vec<f32>,
    /// Note velocities (0.0-1.0)
    pub velocities: Vec<f32>,
    /// Pattern length in beats
    pub length: f32,
    /// Pattern complexity rating
    pub complexity: f32,
    /// Musical style
    pub style: String,
}

/// Pitch bend processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchBendProcessor {
    /// Bend settings
    pub settings: PitchBendSettings,
    /// Current bend state
    pub bend_state: BendState,
    /// Bend automation curves
    pub curves: HashMap<String, BendCurve>,
}

/// Pitch bend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchBendSettings {
    /// Maximum bend range in semitones
    pub max_range: f32,
    /// Bend sensitivity
    pub sensitivity: f32,
    /// Bend curve type
    pub curve_type: BendCurveType,
    /// Bend speed (transitions per second)
    pub speed: f32,
    /// Enable automatic pitch bends
    pub auto_bends: bool,
    /// Bend probability for auto mode
    pub probability: f32,
    /// Preferred bend directions
    pub direction_preference: BendDirection,
    /// Bend return behavior
    pub return_behavior: ReturnBehavior,
}

/// Current bend state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BendState {
    /// Current bend amount in semitones
    pub current_bend: f32,
    /// Target bend amount
    pub target_bend: f32,
    /// Bend start time
    pub start_time: f32,
    /// Bend duration
    pub duration: f32,
    /// Bend phase
    pub phase: BendPhase,
    /// Source note frequency
    pub source_frequency: f32,
}

/// Advanced dynamics processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDynamicsProcessor {
    /// Dynamics settings
    pub settings: AdvancedDynamicsSettings,
    /// Current dynamics state
    pub state: DynamicsState,
    /// Dynamics automation
    pub automation: DynamicsAutomation,
    /// Envelope followers
    pub envelopes: Vec<EnvelopeFollower>,
}

/// Advanced dynamics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDynamicsSettings {
    /// Multi-band dynamics (frequency ranges)
    pub bands: Vec<DynamicsBand>,
    /// Master dynamics
    pub master: MasterDynamics,
    /// Automation settings
    pub automation: AutomationSettings,
    /// Dynamics response curve
    pub response_curve: DynamicsCurve,
    /// Enable dynamics smoothing
    pub smoothing: bool,
    /// Smoothing time constant
    pub smoothing_time: f32,
}

/// Dynamics band (frequency range)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsBand {
    /// Band name
    pub name: String,
    /// Frequency range (low, high) in Hz
    pub frequency_range: (f32, f32),
    /// Compression ratio
    pub compression: f32,
    /// Threshold in dB
    pub threshold: f32,
    /// Attack time in seconds
    pub attack: f32,
    /// Release time in seconds
    pub release: f32,
    /// Make-up gain in dB
    pub makeup_gain: f32,
    /// Band weight in mix
    pub weight: f32,
}

/// Enhanced articulation processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedArticulationProcessor {
    /// Articulation settings
    pub settings: AdvancedArticulationSettings,
    /// Current articulation state
    pub state: ArticulationState,
    /// Articulation templates
    pub templates: HashMap<Articulation, ArticulationTemplate>,
}

/// Advanced articulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedArticulationSettings {
    /// Articulation sensitivity
    pub sensitivity: f32,
    /// Transition smoothness
    pub transition_smoothness: f32,
    /// Enable articulation blending
    pub blending: bool,
    /// Blend time in seconds
    pub blend_time: f32,
    /// Articulation strength multiplier
    pub strength_multiplier: f32,
    /// Enable articulation automation
    pub automation: bool,
    /// Context awareness
    pub context_aware: bool,
}

/// Articulation template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulationTemplate {
    /// Articulation name
    pub name: String,
    /// Attack curve
    pub attack_curve: Vec<f32>,
    /// Sustain level
    pub sustain_level: f32,
    /// Release curve
    pub release_curve: Vec<f32>,
    /// Pitch modulation
    pub pitch_modulation: Vec<f32>,
    /// Velocity scaling
    pub velocity_scaling: f32,
    /// Duration factor
    pub duration_factor: f32,
}

/// Melismatic processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelismaProcessor {
    /// Melisma settings
    pub settings: MelismaSettings,
    /// Current melisma state
    pub state: MelismaState,
    /// Melisma patterns
    pub patterns: Vec<MelismaPattern>,
}

/// Grace note processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraceNoteProcessor {
    /// Grace note settings
    pub settings: GraceNoteSettings,
    /// Current grace state
    pub state: GraceState,
    /// Grace note types
    pub types: HashMap<String, GraceNoteType>,
}

// Supporting enums and structs

/// Rhythm patterns for musical runs and ornamentation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunRhythm {
    /// Even rhythm with equal note durations
    Even,
    /// Dotted rhythm with alternating long and short notes
    Dotted,
    /// Triplet subdivisions within beats
    Triplets,
    /// Syncopated rhythm emphasizing off-beats
    Syncopated,
    /// Gradually accelerating tempo
    Accelerando,
    /// Gradually slowing tempo
    Ritardando,
    /// Custom rhythm pattern with specified subdivisions
    Custom(u32),
}

/// Musical scales for run and ornamentation patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicScale {
    /// Major scale (Ionian mode)
    Major,
    /// Minor scale (natural minor/Aeolian mode)
    Minor,
    /// Dorian mode (minor with raised 6th)
    Dorian,
    /// Mixolydian mode (major with lowered 7th)
    Mixolydian,
    /// Pentatonic scale (5-note scale)
    Pentatonic,
    /// Blues scale with characteristic blue notes
    Blues,
    /// Chromatic scale using all 12 semitones
    Chromatic,
    /// Custom scale pattern
    Custom,
}

/// Velocity curve shapes for dynamic expression in runs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VelocityCurve {
    /// Linear velocity progression
    Linear,
    /// Exponential velocity curve (rapid change)
    Exponential,
    /// Logarithmic velocity curve (gradual change)
    Logarithmic,
    /// Bell-shaped velocity curve (crescendo then diminuendo)
    Bell,
    /// Ramp velocity curve (sudden change then steady)
    Ramp,
    /// Constant velocity throughout
    Constant,
}

/// Phases of musical run execution for state tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunPhase {
    /// Run is idle and waiting to start
    Idle,
    /// Run is beginning (attack phase)
    Starting,
    /// Run is actively playing (sustain phase)
    Active,
    /// Run is concluding (release phase)
    Ending,
    /// Run has completed execution
    Finished,
}

/// Curve shapes for pitch bend transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BendCurveType {
    /// Linear pitch bend transition
    Linear,
    /// Exponential pitch bend curve (rapid initial change)
    Exponential,
    /// Logarithmic pitch bend curve (gradual initial change)
    Logarithmic,
    /// S-shaped sigmoid curve for smooth transitions
    Sigmoid,
    /// Bezier curve for custom transition shapes
    Bezier,
    /// Custom curve implementation
    Custom,
}

/// Direction of pitch bends and glissandos
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BendDirection {
    /// Bend pitch upward (higher frequency)
    Up,
    /// Bend pitch downward (lower frequency)
    Down,
    /// Bend in both directions (oscillation)
    Both,
    /// Direction determined by musical context
    Contextual,
}

/// Behavior for returning to original pitch after bend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReturnBehavior {
    /// Return immediately to original pitch
    Immediate,
    /// Gradually return to original pitch
    Gradual,
    /// Hold the bent pitch without returning
    Hold,
    /// No return behavior (stays at bent pitch)
    None,
}

/// Phases of pitch bend execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BendPhase {
    /// Bend is inactive
    Idle,
    /// Pitch is actively bending to target
    Bending,
    /// Holding the bent pitch at target
    Holding,
    /// Returning from bent pitch to original
    Returning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BendCurve {
    pub name: String,
    pub points: Vec<(f32, f32)>, // (time, bend_amount)
    pub interpolation: CurveInterpolation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveInterpolation {
    Linear,
    Cubic,
    Spline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsState {
    pub current_level: f32,
    pub target_level: f32,
    pub envelope_position: f32,
    pub compression_state: Vec<f32>, // Per-band compression
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsAutomation {
    pub enabled: bool,
    pub curve: Vec<(f32, f32)>, // (time, level)
    pub position: f32,
    pub loop_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeFollower {
    pub name: String,
    pub attack: f32,
    pub release: f32,
    pub current_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterDynamics {
    pub gain: f32,
    pub limit: f32,
    pub compression: f32,
    pub saturation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationSettings {
    pub enabled: bool,
    pub speed: f32,
    pub depth: f32,
    pub pattern: AutomationPattern,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutomationPattern {
    Sine,
    Triangle,
    Sawtooth,
    Square,
    Random,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynamicsCurve {
    Linear,
    Exponential,
    Logarithmic,
    Sigmoid,
    Musical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulationState {
    pub current_articulation: Articulation,
    pub transition_progress: f32,
    pub envelope_phase: f32,
    pub modulation_phase: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelismaSettings {
    pub complexity: f32,
    pub note_density: f32,
    pub range_semitones: f32,
    pub legato_factor: f32,
    pub auto_enable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelismaState {
    pub active: bool,
    pub position: f32,
    pub current_pattern: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelismaPattern {
    pub name: String,
    pub notes: Vec<MelismaNoteData>,
    pub style: String,
    pub difficulty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelismaNoteData {
    pub pitch_offset: f32,
    pub duration: f32,
    pub velocity: f32,
    pub articulation: Articulation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraceNoteSettings {
    pub probability: f32,
    pub timing_offset: f32,
    pub velocity_scale: f32,
    pub duration_scale: f32,
    pub types_enabled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraceState {
    pub active: bool,
    pub grace_type: Option<String>,
    pub timing: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraceNoteType {
    pub name: String,
    pub pitch_offset: f32,
    pub timing_offset: f32,
    pub duration_scale: f32,
    pub velocity_scale: f32,
}

// Implementation methods

impl AdvancedTechniques {
    /// Create new advanced techniques processor
    pub fn new() -> Self {
        Self {
            vocal_runs: VocalRunProcessor::new(),
            pitch_bends: PitchBendProcessor::new(),
            dynamics: AdvancedDynamicsProcessor::new(),
            articulation: AdvancedArticulationProcessor::new(),
            melisma: MelismaProcessor::new(),
            grace_notes: GraceNoteProcessor::new(),
        }
    }

    /// Process a note with advanced techniques
    pub fn process_note(
        &mut self,
        note: &mut NoteEvent,
        sample_rate: f32,
        delta_time: f32,
    ) -> crate::Result<()> {
        // Process vocal runs
        self.vocal_runs.process_note(note, delta_time)?;

        // Process pitch bends
        self.pitch_bends.process_note(note, delta_time)?;

        // Process advanced dynamics
        self.dynamics.process_note(note, delta_time)?;

        // Process advanced articulation
        self.articulation.process_note(note, delta_time)?;

        // Process melisma
        self.melisma.process_note(note, delta_time)?;

        // Process grace notes
        self.grace_notes.process_note(note, delta_time)?;

        Ok(())
    }

    /// Generate vocal run pattern
    pub fn generate_vocal_run(
        &mut self,
        base_note: &NoteEvent,
        complexity: f32,
    ) -> crate::Result<Vec<NoteEvent>> {
        self.vocal_runs.generate_run(base_note, complexity)
    }

    /// Apply pitch bend to frequency
    pub fn apply_pitch_bend(&mut self, frequency: f32, bend_amount: f32) -> f32 {
        self.pitch_bends.apply_bend(frequency, bend_amount)
    }

    /// Set dynamics automation curve
    pub fn set_dynamics_curve(&mut self, curve: Vec<(f32, f32)>) {
        self.dynamics.set_automation_curve(curve);
    }

    /// Enable technique by name
    pub fn enable_technique(&mut self, technique: &str, enabled: bool) -> crate::Result<()> {
        match technique {
            "vocal_runs" => self.vocal_runs.settings.auto_insertion = enabled,
            "pitch_bends" => self.pitch_bends.settings.auto_bends = enabled,
            "advanced_dynamics" => self.dynamics.automation.enabled = enabled,
            "melisma" => self.melisma.settings.auto_enable = enabled,
            _ => {
                return Err(crate::Error::UnsupportedFeature(format!(
                    "Unknown technique: {technique}"
                )))
            }
        }
        Ok(())
    }
}

impl Default for VocalRunProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl VocalRunProcessor {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // Add some basic run patterns
        patterns.insert(
            "ascending_scale".to_string(),
            RunPattern {
                name: "Ascending Scale".to_string(),
                notes: vec![0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0, 12.0],
                durations: vec![0.125; 8],
                velocities: vec![0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.85],
                length: 1.0,
                complexity: 0.6,
                style: "classical".to_string(),
            },
        );

        patterns.insert(
            "blues_run".to_string(),
            RunPattern {
                name: "Blues Run".to_string(),
                notes: vec![0.0, 3.0, 5.0, 6.0, 7.0, 10.0],
                durations: vec![0.25, 0.125, 0.125, 0.25, 0.125, 0.125],
                velocities: vec![0.9, 0.7, 0.8, 0.85, 0.75, 0.8],
                length: 1.0,
                complexity: 0.7,
                style: "blues".to_string(),
            },
        );

        Self {
            settings: VocalRunSettings::default(),
            run_state: RunState::default(),
            patterns,
        }
    }

    pub fn process_note(&mut self, note: &mut NoteEvent, delta_time: f32) -> crate::Result<()> {
        if !self.settings.auto_insertion {
            return Ok(());
        }

        self.run_state.time += delta_time;

        // Check if we should trigger a new run
        if self.run_state.phase == RunPhase::Idle
            && self.run_state.time >= self.run_state.next_trigger
            && rand::random::<f32>() < self.settings.probability
        {
            self.trigger_run(note)?;
        }

        // Process active run
        if let Some(pattern) = self.run_state.active_run.clone() {
            self.process_active_run(note, &pattern, delta_time)?;
        }

        Ok(())
    }

    pub fn generate_run(
        &mut self,
        base_note: &NoteEvent,
        complexity: f32,
    ) -> crate::Result<Vec<NoteEvent>> {
        let mut run_notes = Vec::new();
        let num_notes = (complexity * 8.0 + 2.0) as usize;

        for i in 0..num_notes {
            let mut new_note = base_note.clone();
            let progress = i as f32 / num_notes as f32;

            // Generate pitch based on scale and direction
            let pitch_offset = self.calculate_run_pitch(progress, complexity);
            new_note.frequency *= (pitch_offset / 12.0).exp2();

            // Calculate duration
            new_note.duration = base_note.duration / num_notes as f32;

            // Calculate velocity with curve
            new_note.velocity = self.calculate_run_velocity(progress);

            run_notes.push(new_note);
        }

        Ok(run_notes)
    }

    fn trigger_run(&mut self, note: &NoteEvent) -> crate::Result<()> {
        // Select appropriate pattern based on note and settings
        let pattern_name = self.select_run_pattern(note);
        if let Some(pattern) = self.patterns.get(&pattern_name).cloned() {
            self.run_state.active_run = Some(pattern);
            self.run_state.phase = RunPhase::Starting;
            self.run_state.position = 0.0;
        }
        Ok(())
    }

    fn process_active_run(
        &mut self,
        note: &mut NoteEvent,
        pattern: &RunPattern,
        delta_time: f32,
    ) -> crate::Result<()> {
        let progress_speed = self.settings.density * delta_time;
        self.run_state.position += progress_speed;

        if self.run_state.position >= 1.0 {
            self.run_state.phase = RunPhase::Finished;
            self.run_state.active_run = None;
            self.run_state.next_trigger = self.run_state.time + 2.0; // Wait before next run
        } else {
            // Apply run to current note
            let note_index = (self.run_state.position * pattern.notes.len() as f32) as usize;
            if note_index < pattern.notes.len() {
                let pitch_offset = pattern.notes[note_index];
                note.frequency *= (pitch_offset / 12.0).exp2();
                note.velocity *= pattern.velocities.get(note_index).copied().unwrap_or(0.8);
            }
        }

        Ok(())
    }

    fn select_run_pattern(&self, _note: &NoteEvent) -> String {
        // Simple pattern selection - could be made more sophisticated
        let patterns: Vec<&String> = self.patterns.keys().collect();
        {
            use rand::{thread_rng, Rng};
            let mut rng = thread_rng();
            patterns[rng.gen_range(0..patterns.len())].clone()
        }
    }

    fn calculate_run_pitch(&self, progress: f32, complexity: f32) -> f32 {
        let scale_degree = (progress * 8.0 * complexity) as i32 % 12;
        let direction = if self.settings.direction_bias > 0.0 {
            1.0
        } else {
            -1.0
        };
        scale_degree as f32 * direction * self.settings.range / 12.0
    }

    fn calculate_run_velocity(&self, progress: f32) -> f32 {
        match self.settings.velocity_curve {
            VelocityCurve::Linear => progress,
            VelocityCurve::Exponential => progress.powi(2),
            VelocityCurve::Logarithmic => progress.sqrt(),
            VelocityCurve::Bell => (-4.0 * (progress - 0.5).powi(2)).exp(),
            VelocityCurve::Ramp => {
                if progress < 0.5 {
                    progress * 2.0
                } else {
                    1.0
                }
            }
            VelocityCurve::Constant => 0.8,
        }
    }
}

impl Default for PitchBendProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl PitchBendProcessor {
    pub fn new() -> Self {
        Self {
            settings: PitchBendSettings::default(),
            bend_state: BendState::default(),
            curves: HashMap::new(),
        }
    }

    pub fn process_note(&mut self, note: &mut NoteEvent, delta_time: f32) -> crate::Result<()> {
        self.update_bend_state(delta_time);

        if self.bend_state.current_bend != 0.0 {
            note.frequency = self.apply_bend(note.frequency, self.bend_state.current_bend);
        }

        // Auto-bend logic
        if self.settings.auto_bends
            && rand::random::<f32>() < self.settings.probability * delta_time
        {
            self.trigger_auto_bend(note);
        }

        Ok(())
    }

    pub fn apply_bend(&self, frequency: f32, bend_amount: f32) -> f32 {
        frequency * (bend_amount / 12.0).exp2()
    }

    fn update_bend_state(&mut self, delta_time: f32) {
        match self.bend_state.phase {
            BendPhase::Bending => {
                let progress = (self.bend_state.start_time + delta_time) / self.bend_state.duration;
                if progress >= 1.0 {
                    self.bend_state.current_bend = self.bend_state.target_bend;
                    self.bend_state.phase = BendPhase::Holding;
                } else {
                    self.bend_state.current_bend = self.interpolate_bend(progress);
                }
            }
            BendPhase::Returning => {
                let return_speed = self.settings.speed * delta_time;
                self.bend_state.current_bend *= 1.0 - return_speed;
                if self.bend_state.current_bend.abs() < 0.01 {
                    self.bend_state.current_bend = 0.0;
                    self.bend_state.phase = BendPhase::Idle;
                }
            }
            _ => {}
        }
        self.bend_state.start_time += delta_time;
    }

    fn trigger_auto_bend(&mut self, note: &NoteEvent) {
        let bend_range = self.settings.max_range;
        let bend_amount = match self.settings.direction_preference {
            BendDirection::Up => rand::random::<f32>() * bend_range,
            BendDirection::Down => -rand::random::<f32>() * bend_range,
            BendDirection::Both => (rand::random::<f32>() - 0.5) * 2.0 * bend_range,
            BendDirection::Contextual => {
                // Context-based bend direction based on note position
                if note.frequency > 440.0 {
                    -bend_range * 0.5
                } else {
                    bend_range * 0.5
                }
            }
        };

        self.bend_state.target_bend = bend_amount;
        self.bend_state.phase = BendPhase::Bending;
        self.bend_state.duration = 1.0 / self.settings.speed;
        self.bend_state.start_time = 0.0;
    }

    fn interpolate_bend(&self, progress: f32) -> f32 {
        let t = match self.settings.curve_type {
            BendCurveType::Linear => progress,
            BendCurveType::Exponential => progress.powi(2),
            BendCurveType::Logarithmic => progress.sqrt(),
            BendCurveType::Sigmoid => 1.0 / (1.0 + (-6.0 * (progress - 0.5)).exp()),
            BendCurveType::Bezier => {
                // Simple bezier approximation
                let t2 = progress * progress;
                let t3 = t2 * progress;
                3.0 * t2 - 2.0 * t3
            }
            BendCurveType::Custom => progress, // Would use custom curve from curves HashMap
        };

        self.bend_state.current_bend
            + t * (self.bend_state.target_bend - self.bend_state.current_bend)
    }
}

impl Default for AdvancedDynamicsProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedDynamicsProcessor {
    pub fn new() -> Self {
        Self {
            settings: AdvancedDynamicsSettings::default(),
            state: DynamicsState::default(),
            automation: DynamicsAutomation::default(),
            envelopes: vec![EnvelopeFollower::default()],
        }
    }

    pub fn process_note(&mut self, note: &mut NoteEvent, delta_time: f32) -> crate::Result<()> {
        // Update automation
        if self.automation.enabled {
            self.update_automation(delta_time);
        }

        // Process multi-band dynamics
        for (i, band) in self.settings.bands.iter().enumerate() {
            if self.note_in_frequency_range(note, band) {
                let compression_amount = self.calculate_compression(note.velocity, band);
                note.velocity = self.apply_compression(note.velocity, compression_amount, band);

                if i < self.state.compression_state.len() {
                    self.state.compression_state[i] = compression_amount;
                }
            }
        }

        // Apply master dynamics
        note.velocity *= self.settings.master.gain;
        note.velocity = note.velocity.min(self.settings.master.limit);

        // Update envelope followers
        for envelope in &mut self.envelopes {
            envelope.update(note.velocity, delta_time);
        }

        Ok(())
    }

    pub fn set_automation_curve(&mut self, curve: Vec<(f32, f32)>) {
        self.automation.curve = curve;
        self.automation.position = 0.0;
    }

    fn update_automation(&mut self, delta_time: f32) {
        self.automation.position += delta_time;

        if !self.automation.curve.is_empty() {
            let total_time = self.automation.curve.last().map(|(t, _)| *t).unwrap_or(1.0);

            if self.automation.position >= total_time {
                if self.automation.loop_enabled {
                    self.automation.position = 0.0;
                } else {
                    self.automation.position = total_time;
                }
            }

            // Interpolate current level from curve
            self.state.target_level = self.interpolate_automation_curve();
        }
    }

    fn note_in_frequency_range(&self, note: &NoteEvent, band: &DynamicsBand) -> bool {
        note.frequency >= band.frequency_range.0 && note.frequency <= band.frequency_range.1
    }

    fn calculate_compression(&self, velocity: f32, band: &DynamicsBand) -> f32 {
        let velocity_db = 20.0 * velocity.log10();
        if velocity_db > band.threshold {
            let over_threshold = velocity_db - band.threshold;
            over_threshold * (1.0 - 1.0 / band.compression)
        } else {
            0.0
        }
    }

    fn apply_compression(
        &self,
        velocity: f32,
        compression_amount: f32,
        band: &DynamicsBand,
    ) -> f32 {
        let compressed_db = 20.0 * velocity.log10() - compression_amount + band.makeup_gain;
        (10.0_f32).powf(compressed_db / 20.0).clamp(0.0, 1.0)
    }

    fn interpolate_automation_curve(&self) -> f32 {
        // Simple linear interpolation between curve points
        for window in self.automation.curve.windows(2) {
            let (t1, v1) = window[0];
            let (t2, v2) = window[1];

            if self.automation.position >= t1 && self.automation.position <= t2 {
                let t = (self.automation.position - t1) / (t2 - t1);
                return v1 + t * (v2 - v1);
            }
        }

        // Return first or last value if outside range
        if self.automation.position <= self.automation.curve[0].0 {
            self.automation.curve[0].1
        } else {
            self.automation.curve.last().map(|(_, v)| *v).unwrap_or(0.0)
        }
    }
}

impl Default for AdvancedArticulationProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedArticulationProcessor {
    pub fn new() -> Self {
        let mut templates = HashMap::new();

        // Add articulation templates
        templates.insert(
            Articulation::Staccato,
            ArticulationTemplate {
                name: "Staccato".to_string(),
                attack_curve: vec![0.0, 1.0],
                sustain_level: 0.0,
                release_curve: vec![1.0, 0.0],
                pitch_modulation: vec![0.0, 0.0],
                velocity_scaling: 0.9,
                duration_factor: 0.3,
            },
        );

        templates.insert(
            Articulation::Legato,
            ArticulationTemplate {
                name: "Legato".to_string(),
                attack_curve: vec![0.0, 0.3, 0.8, 1.0],
                sustain_level: 0.9,
                release_curve: vec![1.0, 0.8, 0.2, 0.0],
                pitch_modulation: vec![0.0, 0.0, 0.0, 0.0],
                velocity_scaling: 1.0,
                duration_factor: 1.1,
            },
        );

        Self {
            settings: AdvancedArticulationSettings::default(),
            state: ArticulationState::default(),
            templates,
        }
    }

    pub fn process_note(&mut self, note: &mut NoteEvent, delta_time: f32) -> crate::Result<()> {
        if let Some(template) = self.templates.get(&note.articulation) {
            // Apply articulation template
            note.velocity *= template.velocity_scaling * self.settings.strength_multiplier;
            note.duration *= template.duration_factor;

            // Update articulation state
            self.state.current_articulation = note.articulation;
            self.state.envelope_phase = 0.0;
        }

        // Update envelope phase
        self.state.envelope_phase += delta_time;

        Ok(())
    }
}

impl Default for MelismaProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl MelismaProcessor {
    pub fn new() -> Self {
        Self {
            settings: MelismaSettings::default(),
            state: MelismaState::default(),
            patterns: Vec::new(),
        }
    }

    pub fn process_note(&mut self, note: &mut NoteEvent, delta_time: f32) -> crate::Result<()> {
        if !self.settings.auto_enable {
            return Ok(());
        }

        // Simple melisma logic - would be expanded for full implementation
        if rand::random::<f32>() < 0.1 * delta_time {
            // Apply slight pitch variation for melismatic effect
            let variation = (rand::random::<f32>() - 0.5) * self.settings.range_semitones;
            note.frequency *= (variation / 12.0).exp2();
        }

        Ok(())
    }
}

impl Default for GraceNoteProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl GraceNoteProcessor {
    pub fn new() -> Self {
        let mut types = HashMap::new();

        types.insert(
            "acciaccatura".to_string(),
            GraceNoteType {
                name: "Acciaccatura".to_string(),
                pitch_offset: 1.0,   // Semitone above
                timing_offset: -0.1, // Slightly before main note
                duration_scale: 0.1,
                velocity_scale: 0.7,
            },
        );

        types.insert(
            "appoggiatura".to_string(),
            GraceNoteType {
                name: "Appoggiatura".to_string(),
                pitch_offset: 2.0, // Whole tone above
                timing_offset: -0.2,
                duration_scale: 0.3,
                velocity_scale: 0.8,
            },
        );

        Self {
            settings: GraceNoteSettings::default(),
            state: GraceState::default(),
            types,
        }
    }

    pub fn process_note(&mut self, note: &mut NoteEvent, _delta_time: f32) -> crate::Result<()> {
        if rand::random::<f32>() < self.settings.probability {
            // Apply grace note effect
            note.velocity *= self.settings.velocity_scale;
            note.duration *= self.settings.duration_scale;
        }

        Ok(())
    }
}

// Default implementations

impl Default for VocalRunSettings {
    fn default() -> Self {
        Self {
            density: 8.0,
            complexity: 0.5,
            range: 12.0,
            direction_bias: 0.0,
            rhythm_pattern: RunRhythm::Even,
            auto_insertion: false,
            probability: 0.1,
            scales: vec![MusicScale::Major, MusicScale::Minor],
            velocity_curve: VelocityCurve::Bell,
            legato_factor: 0.8,
        }
    }
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            active_run: None,
            position: 0.0,
            phase: RunPhase::Idle,
            time: 0.0,
            next_trigger: 2.0,
        }
    }
}

impl Default for PitchBendSettings {
    fn default() -> Self {
        Self {
            max_range: 2.0,
            sensitivity: 1.0,
            curve_type: BendCurveType::Sigmoid,
            speed: 2.0,
            auto_bends: false,
            probability: 0.05,
            direction_preference: BendDirection::Both,
            return_behavior: ReturnBehavior::Gradual,
        }
    }
}

impl Default for BendState {
    fn default() -> Self {
        Self {
            current_bend: 0.0,
            target_bend: 0.0,
            start_time: 0.0,
            duration: 0.5,
            phase: BendPhase::Idle,
            source_frequency: 440.0,
        }
    }
}

impl Default for AdvancedDynamicsSettings {
    fn default() -> Self {
        Self {
            bands: vec![
                DynamicsBand {
                    name: "Low".to_string(),
                    frequency_range: (20.0, 250.0),
                    compression: 3.0,
                    threshold: -20.0,
                    attack: 0.01,
                    release: 0.1,
                    makeup_gain: 2.0,
                    weight: 1.0,
                },
                DynamicsBand {
                    name: "Mid".to_string(),
                    frequency_range: (250.0, 2000.0),
                    compression: 2.0,
                    threshold: -15.0,
                    attack: 0.005,
                    release: 0.05,
                    makeup_gain: 1.0,
                    weight: 1.0,
                },
                DynamicsBand {
                    name: "High".to_string(),
                    frequency_range: (2000.0, 20000.0),
                    compression: 4.0,
                    threshold: -10.0,
                    attack: 0.001,
                    release: 0.03,
                    makeup_gain: 3.0,
                    weight: 1.0,
                },
            ],
            master: MasterDynamics::default(),
            automation: AutomationSettings::default(),
            response_curve: DynamicsCurve::Musical,
            smoothing: true,
            smoothing_time: 0.05,
        }
    }
}

impl Default for DynamicsState {
    fn default() -> Self {
        Self {
            current_level: 0.8,
            target_level: 0.8,
            envelope_position: 0.0,
            compression_state: vec![0.0; 3], // For 3 bands
        }
    }
}

impl Default for DynamicsAutomation {
    fn default() -> Self {
        Self {
            enabled: false,
            curve: Vec::new(),
            position: 0.0,
            loop_enabled: false,
        }
    }
}

impl Default for EnvelopeFollower {
    fn default() -> Self {
        Self {
            name: "Master".to_string(),
            attack: 0.01,
            release: 0.1,
            current_level: 0.0,
        }
    }
}

impl Default for MasterDynamics {
    fn default() -> Self {
        Self {
            gain: 1.0,
            limit: 1.0,
            compression: 1.0,
            saturation: 0.0,
        }
    }
}

impl Default for AutomationSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            speed: 1.0,
            depth: 0.5,
            pattern: AutomationPattern::Sine,
        }
    }
}

impl Default for AdvancedArticulationSettings {
    fn default() -> Self {
        Self {
            sensitivity: 1.0,
            transition_smoothness: 0.8,
            blending: true,
            blend_time: 0.1,
            strength_multiplier: 1.0,
            automation: false,
            context_aware: true,
        }
    }
}

impl Default for ArticulationState {
    fn default() -> Self {
        Self {
            current_articulation: Articulation::Normal,
            transition_progress: 0.0,
            envelope_phase: 0.0,
            modulation_phase: 0.0,
        }
    }
}

impl Default for MelismaSettings {
    fn default() -> Self {
        Self {
            complexity: 0.5,
            note_density: 4.0,
            range_semitones: 3.0,
            legato_factor: 0.9,
            auto_enable: false,
        }
    }
}

impl Default for MelismaState {
    fn default() -> Self {
        Self {
            active: false,
            position: 0.0,
            current_pattern: None,
        }
    }
}

impl Default for GraceNoteSettings {
    fn default() -> Self {
        Self {
            probability: 0.05,
            timing_offset: -0.1,
            velocity_scale: 0.8,
            duration_scale: 0.2,
            types_enabled: vec!["acciaccatura".to_string(), "appoggiatura".to_string()],
        }
    }
}

impl Default for GraceState {
    fn default() -> Self {
        Self {
            active: false,
            grace_type: None,
            timing: 0.0,
        }
    }
}

impl EnvelopeFollower {
    pub fn update(&mut self, input: f32, delta_time: f32) {
        let target = input.abs();
        let rate = if target > self.current_level {
            1.0 / self.attack
        } else {
            1.0 / self.release
        };

        let step = (rate * delta_time).min(1.0); // Clamp step to prevent overshoot
        self.current_level += (target - self.current_level) * step;
        self.current_level = self.current_level.max(0.0); // Ensure non-negative
    }
}

impl Default for AdvancedTechniques {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_techniques_creation() {
        let techniques = AdvancedTechniques::new();
        assert!(!techniques.vocal_runs.settings.auto_insertion);
        assert!(!techniques.pitch_bends.settings.auto_bends);
        assert!(!techniques.dynamics.automation.enabled);
    }

    #[test]
    fn test_vocal_run_generation() {
        let mut processor = VocalRunProcessor::new();
        let base_note = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);

        let run = processor.generate_run(&base_note, 0.5).unwrap();
        assert!(!run.is_empty());
        assert!(run.len() >= 2); // Should generate multiple notes
    }

    #[test]
    fn test_pitch_bend_application() {
        let processor = PitchBendProcessor::new();
        let frequency = 440.0;
        let bend_amount = 2.0; // Two semitones up

        let bent_frequency = processor.apply_bend(frequency, bend_amount);
        let expected = frequency * (2.0_f32 / 12.0).exp2();

        assert!((bent_frequency - expected).abs() < 0.01);
    }

    #[test]
    fn test_dynamics_compression() {
        let mut processor = AdvancedDynamicsProcessor::new();
        let mut note = NoteEvent::new("C".to_string(), 4, 1.0, 0.9); // High velocity

        processor.process_note(&mut note, 0.01).unwrap();

        // Note should be compressed (velocity reduced)
        assert!(note.velocity < 0.9);
        assert!(note.velocity > 0.0);
    }

    #[test]
    fn test_articulation_processing() {
        let mut processor = AdvancedArticulationProcessor::new();
        let mut note = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        note.articulation = Articulation::Staccato;

        let original_duration = note.duration;
        processor.process_note(&mut note, 0.01).unwrap();

        // Staccato should shorten the note
        assert!(note.duration < original_duration);
    }

    #[test]
    fn test_envelope_follower() {
        let mut envelope = EnvelopeFollower::default();

        // Attack: rise to high level
        envelope.update(1.0, 0.01);
        assert!(envelope.current_level > 0.0);
        let peak = envelope.current_level;

        // Release: should decay when input goes to zero
        envelope.update(0.0, 0.01);
        let after_first_decay = envelope.current_level;
        envelope.update(0.0, 0.01);
        let after_second_decay = envelope.current_level;

        // Should be decaying
        assert!(after_second_decay <= after_first_decay);
        assert!(after_first_decay <= peak);
    }
}
