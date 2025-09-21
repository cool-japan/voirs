//! Musical score processing and representation

#![allow(clippy::uninlined_format_args)]

use crate::types::{Articulation, BreathInfo, Dynamics, Expression, NoteEvent, PitchBend};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Musical score representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalScore {
    /// Title of the score
    pub title: String,
    /// Composer
    pub composer: String,
    /// Key signature
    pub key_signature: KeySignature,
    /// Time signature
    pub time_signature: TimeSignature,
    /// Tempo in BPM
    pub tempo: f32,
    /// Musical notes
    pub notes: Vec<MusicalNote>,
    /// Lyrics
    pub lyrics: Option<Lyrics>,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Total duration
    pub duration: Duration,
    /// Sections
    pub sections: Vec<Section>,
    /// Markers
    pub markers: Vec<Marker>,
    /// Breath marks
    pub breath_marks: Vec<BreathInfo>,
    /// Dynamics
    pub dynamics: Vec<DynamicMarking>,
    /// Expression markings
    pub expressions: Vec<ExpressionMarking>,
}

/// Musical note representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalNote {
    /// Note event
    pub event: NoteEvent,
    /// Start time in beats
    pub start_time: f32,
    /// Duration in beats
    pub duration: f32,
    /// Pitch bend information
    pub pitch_bend: Option<PitchBend>,
    /// Articulation
    pub articulation: Articulation,
    /// Dynamics
    pub dynamics: Dynamics,
    /// Tie to next note
    pub tie_next: bool,
    /// Tie from previous note
    pub tie_prev: bool,
    /// Tuplet information
    pub tuplet: Option<Tuplet>,
    /// Ornaments
    pub ornaments: Vec<Ornament>,
    /// Chord information (if part of chord)
    pub chord: Option<ChordInfo>,
}

/// Key signature
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeySignature {
    /// Root note
    pub root: Note,
    /// Mode (major/minor)
    pub mode: Mode,
    /// Number of sharps/flats
    pub accidentals: i8,
}

/// Time signature
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSignature {
    /// Numerator (beats per measure)
    pub numerator: u8,
    /// Denominator (note value)
    pub denominator: u8,
}

/// Musical note names
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Note {
    /// C note
    C,
    /// D note
    D,
    /// E note
    E,
    /// F note
    F,
    /// G note
    G,
    /// A note
    A,
    /// B note
    B,
}

/// Musical modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    /// Major mode (Ionian)
    Major,
    /// Minor mode (Natural minor)
    Minor,
    /// Dorian mode (minor with raised 6th)
    Dorian,
    /// Phrygian mode (minor with flattened 2nd)
    Phrygian,
    /// Lydian mode (major with raised 4th)
    Lydian,
    /// Mixolydian mode (major with flattened 7th)
    Mixolydian,
    /// Aeolian mode (natural minor)
    Aeolian,
    /// Locrian mode (diminished)
    Locrian,
}

/// Lyrics information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lyrics {
    /// Text lines
    pub lines: Vec<String>,
    /// Syllable timing
    pub syllables: Vec<Syllable>,
    /// Language
    pub language: String,
    /// Phoneme transcription
    pub phonemes: Option<Vec<String>>,
}

/// Syllable information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Syllable {
    /// Syllable text
    pub text: String,
    /// Start time in beats
    pub start_time: f32,
    /// Duration in beats
    pub duration: f32,
    /// Stress level (0-3)
    pub stress: u8,
    /// Phoneme sequence
    pub phonemes: Vec<String>,
}

/// Musical section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// Section name
    pub name: String,
    /// Start time in beats
    pub start_time: f32,
    /// End time in beats
    pub end_time: f32,
    /// Tempo change
    pub tempo_change: Option<f32>,
    /// Key change
    pub key_change: Option<KeySignature>,
    /// Time signature change
    pub time_signature_change: Option<TimeSignature>,
    /// Repeat count
    pub repeat_count: u8,
}

/// Score marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    /// Marker name
    pub name: String,
    /// Position in beats
    pub position: f32,
    /// Marker type
    pub marker_type: MarkerType,
    /// Description
    pub description: String,
}

/// Marker types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkerType {
    /// Rehearsal mark
    Rehearsal,
    /// Coda
    Coda,
    /// Segno
    Segno,
    /// Fine
    Fine,
    /// Da Capo
    DaCapo,
    /// Dal Segno
    DalSegno,
    /// Breath mark
    Breath,
    /// Fermata
    Fermata,
}

/// Dynamic marking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicMarking {
    /// Position in beats
    pub position: f32,
    /// Dynamics level
    pub dynamics: Dynamics,
    /// Duration (for crescendo/diminuendo)
    pub duration: Option<f32>,
    /// Target dynamics (for crescendo/diminuendo)
    pub target: Option<Dynamics>,
}

/// Expression marking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionMarking {
    /// Position in beats
    pub position: f32,
    /// Expression type
    pub expression: Expression,
    /// Duration
    pub duration: f32,
    /// Intensity (0.0-1.0)
    pub intensity: f32,
}

/// Tuplet information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tuplet {
    /// Number of notes in tuplet
    pub notes: u8,
    /// Note value
    pub note_value: u8,
    /// Time modification
    pub time_modification: f32,
}

/// Ornament types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ornament {
    /// Trill
    Trill,
    /// Mordent
    Mordent,
    /// Turn
    Turn,
    /// Appoggiatura
    Appoggiatura,
    /// Acciaccatura
    Acciaccatura,
    /// Glissando
    Glissando,
    /// Slide
    Slide,
}

/// Chord information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChordInfo {
    /// Chord root
    pub root: Note,
    /// Chord quality
    pub quality: ChordQuality,
    /// Chord extensions
    pub extensions: Vec<u8>,
    /// Bass note
    pub bass: Option<Note>,
    /// Inversion
    pub inversion: u8,
}

/// Chord quality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChordQuality {
    /// Major
    Major,
    /// Minor
    Minor,
    /// Diminished
    Diminished,
    /// Augmented
    Augmented,
    /// Dominant
    Dominant,
    /// Sus2
    Sus2,
    /// Sus4
    Sus4,
}

/// Score processor for manipulating musical scores
pub struct ScoreProcessor {
    /// Current score
    score: MusicalScore,
    /// Processing options
    options: ProcessingOptions,
}

/// Processing options
#[derive(Debug, Clone)]
pub struct ProcessingOptions {
    /// Quantization resolution
    pub quantization: f32,
    /// Swing factor
    pub swing: f32,
    /// Humanization amount
    pub humanization: f32,
    /// Tempo variations
    pub tempo_variations: bool,
    /// Dynamic variations
    pub dynamic_variations: bool,
    /// Articulation variations
    pub articulation_variations: bool,
}

impl MusicalScore {
    /// Create new empty score
    pub fn new(title: String, composer: String) -> Self {
        Self {
            title,
            composer,
            key_signature: KeySignature::default(),
            time_signature: TimeSignature::default(),
            tempo: 120.0,
            notes: Vec::new(),
            lyrics: None,
            metadata: HashMap::new(),
            duration: Duration::from_secs(0),
            sections: Vec::new(),
            markers: Vec::new(),
            breath_marks: Vec::new(),
            dynamics: Vec::new(),
            expressions: Vec::new(),
        }
    }

    /// Add note to score
    pub fn add_note(&mut self, note: MusicalNote) {
        let end_time = note.start_time + note.duration;
        if end_time > self.duration_in_beats() {
            self.duration = Duration::from_secs_f32(end_time * 60.0 / self.tempo);
        }
        self.notes.push(note);
    }

    /// Add section to score
    pub fn add_section(&mut self, section: Section) {
        self.sections.push(section);
    }

    /// Add marker to score
    pub fn add_marker(&mut self, marker: Marker) {
        self.markers.push(marker);
    }

    /// Add breath mark
    pub fn add_breath_mark(&mut self, breath: BreathInfo) {
        self.breath_marks.push(breath);
    }

    /// Add dynamic marking
    pub fn add_dynamic_marking(&mut self, marking: DynamicMarking) {
        self.dynamics.push(marking);
    }

    /// Add expression marking
    pub fn add_expression_marking(&mut self, marking: ExpressionMarking) {
        self.expressions.push(marking);
    }

    /// Get duration in beats
    pub fn duration_in_beats(&self) -> f32 {
        self.duration.as_secs_f32() * self.tempo / 60.0
    }

    /// Get notes in time range
    pub fn notes_in_range(&self, start: f32, end: f32) -> Vec<&MusicalNote> {
        self.notes
            .iter()
            .filter(|note| {
                let note_end = note.start_time + note.duration;
                note.start_time < end && note_end > start
            })
            .collect()
    }

    /// Get active dynamics at time
    pub fn dynamics_at_time(&self, time: f32) -> Dynamics {
        self.dynamics
            .iter()
            .filter(|d| d.position <= time)
            .max_by(|a, b| a.position.partial_cmp(&b.position).unwrap())
            .map(|d| d.dynamics)
            .unwrap_or(Dynamics::MezzoForte)
    }

    /// Get active expression at time
    pub fn expression_at_time(&self, time: f32) -> Option<&ExpressionMarking> {
        self.expressions
            .iter()
            .find(|e| e.position <= time && e.position + e.duration > time)
    }

    /// Transpose score
    pub fn transpose(&mut self, semitones: i8) {
        for note in &mut self.notes {
            note.event.frequency *= 2.0_f32.powf(semitones as f32 / 12.0);
        }

        // Update key signature
        self.key_signature.accidentals += semitones;
        if self.key_signature.accidentals > 7 {
            self.key_signature.accidentals -= 12;
        } else if self.key_signature.accidentals < -7 {
            self.key_signature.accidentals += 12;
        }
    }

    /// Change tempo
    pub fn set_tempo(&mut self, new_tempo: f32) {
        let scale_factor = self.tempo / new_tempo;
        self.tempo = new_tempo;

        // Scale all time values
        for note in &mut self.notes {
            note.start_time *= scale_factor;
            note.duration *= scale_factor;
        }

        for section in &mut self.sections {
            section.start_time *= scale_factor;
            section.end_time *= scale_factor;
        }

        for marker in &mut self.markers {
            marker.position *= scale_factor;
        }

        for breath in &mut self.breath_marks {
            breath.position *= scale_factor;
        }

        for dynamic in &mut self.dynamics {
            dynamic.position *= scale_factor;
            if let Some(ref mut duration) = dynamic.duration {
                *duration *= scale_factor;
            }
        }

        for expression in &mut self.expressions {
            expression.position *= scale_factor;
            expression.duration *= scale_factor;
        }
    }

    /// Quantize score
    pub fn quantize(&mut self, resolution: f32) {
        for note in &mut self.notes {
            note.start_time = (note.start_time / resolution).round() * resolution;
            note.duration = (note.duration / resolution).round() * resolution;
        }
    }

    /// Add swing to score
    pub fn add_swing(&mut self, factor: f32) {
        for note in &mut self.notes {
            let beat_position = note.start_time % 1.0;
            if beat_position >= 0.5 {
                note.start_time += (beat_position - 0.5) * factor;
            }
        }
    }

    /// Validate score
    pub fn validate(&self) -> Result<(), String> {
        if self.notes.is_empty() {
            return Err("Score has no notes".to_string());
        }

        if self.tempo <= 0.0 {
            return Err("Invalid tempo".to_string());
        }

        if self.time_signature.numerator == 0 || self.time_signature.denominator == 0 {
            return Err("Invalid time signature".to_string());
        }

        // Check for overlapping notes (if monophonic)
        let mut sorted_notes = self.notes.clone();
        sorted_notes.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

        for i in 0..sorted_notes.len() - 1 {
            let current_end = sorted_notes[i].start_time + sorted_notes[i].duration;
            let next_start = sorted_notes[i + 1].start_time;

            if current_end > next_start {
                return Err(format!("Overlapping notes at time {}", next_start));
            }
        }

        Ok(())
    }
}

impl ScoreProcessor {
    /// Create new score processor
    pub fn new(score: MusicalScore) -> Self {
        Self {
            score,
            options: ProcessingOptions::default(),
        }
    }

    /// Set processing options
    pub fn set_options(&mut self, options: ProcessingOptions) {
        self.options = options;
    }

    /// Process score
    pub fn process(&mut self) -> crate::Result<MusicalScore> {
        let mut processed_score = self.score.clone();

        // Apply quantization
        if self.options.quantization > 0.0 {
            processed_score.quantize(self.options.quantization);
        }

        // Apply swing
        if self.options.swing > 0.0 {
            processed_score.add_swing(self.options.swing);
        }

        // Apply humanization
        if self.options.humanization > 0.0 {
            self.apply_humanization(&mut processed_score)?;
        }

        // Apply tempo variations
        if self.options.tempo_variations {
            self.apply_tempo_variations(&mut processed_score)?;
        }

        // Apply dynamic variations
        if self.options.dynamic_variations {
            self.apply_dynamic_variations(&mut processed_score)?;
        }

        // Apply articulation variations
        if self.options.articulation_variations {
            self.apply_articulation_variations(&mut processed_score)?;
        }

        Ok(processed_score)
    }

    /// Apply humanization to score
    fn apply_humanization(&self, score: &mut MusicalScore) -> crate::Result<()> {
        for note in &mut score.notes {
            let timing_variation = (rand::random::<f32>() - 0.5) * self.options.humanization * 0.1;
            let velocity_variation =
                (rand::random::<f32>() - 0.5) * self.options.humanization * 0.2;

            note.start_time += timing_variation;
            note.event.velocity = (note.event.velocity + velocity_variation).clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// Apply tempo variations
    fn apply_tempo_variations(&self, score: &mut MusicalScore) -> crate::Result<()> {
        // Add subtle tempo variations for more natural feel
        for section in &mut score.sections {
            if section.tempo_change.is_none() {
                let variation = (rand::random::<f32>() - 0.5) * 0.1;
                section.tempo_change = Some(score.tempo * (1.0 + variation));
            }
        }
        Ok(())
    }

    /// Apply dynamic variations
    fn apply_dynamic_variations(&self, score: &mut MusicalScore) -> crate::Result<()> {
        for note in &mut score.notes {
            let variation = (rand::random::<f32>() - 0.5) * 0.1;
            note.event.velocity = (note.event.velocity + variation).clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// Apply articulation variations
    fn apply_articulation_variations(&self, score: &mut MusicalScore) -> crate::Result<()> {
        for note in &mut score.notes {
            // Add subtle articulation variations
            match note.articulation {
                Articulation::Legato => {
                    if rand::random::<f32>() < 0.1 {
                        note.articulation = Articulation::Tenuto;
                    }
                }
                Articulation::Staccato => {
                    if rand::random::<f32>() < 0.1 {
                        note.articulation = Articulation::Accent;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Get processed score
    pub fn get_score(&self) -> &MusicalScore {
        &self.score
    }
}

impl Default for KeySignature {
    fn default() -> Self {
        Self {
            root: Note::C,
            mode: Mode::Major,
            accidentals: 0,
        }
    }
}

impl Default for TimeSignature {
    fn default() -> Self {
        Self {
            numerator: 4,
            denominator: 4,
        }
    }
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            quantization: 0.0,
            swing: 0.0,
            humanization: 0.0,
            tempo_variations: false,
            dynamic_variations: false,
            articulation_variations: false,
        }
    }
}

impl MusicalNote {
    /// Create new musical note
    pub fn new(event: NoteEvent, start_time: f32, duration: f32) -> Self {
        Self {
            event,
            start_time,
            duration,
            pitch_bend: None,
            articulation: Articulation::Normal,
            dynamics: Dynamics::MezzoForte,
            tie_next: false,
            tie_prev: false,
            tuplet: None,
            ornaments: Vec::new(),
            chord: None,
        }
    }

    /// Set articulation
    pub fn with_articulation(mut self, articulation: Articulation) -> Self {
        self.articulation = articulation;
        self
    }

    /// Set dynamics
    pub fn with_dynamics(mut self, dynamics: Dynamics) -> Self {
        self.dynamics = dynamics;
        self
    }

    /// Add ornament
    pub fn with_ornament(mut self, ornament: Ornament) -> Self {
        self.ornaments.push(ornament);
        self
    }

    /// Set pitch bend
    pub fn with_pitch_bend(mut self, pitch_bend: PitchBend) -> Self {
        self.pitch_bend = Some(pitch_bend);
        self
    }

    /// Set tie to next note
    pub fn with_tie_next(mut self, tie: bool) -> Self {
        self.tie_next = tie;
        self
    }

    /// Set chord information
    pub fn with_chord(mut self, chord: ChordInfo) -> Self {
        self.chord = Some(chord);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_creation() {
        let score = MusicalScore::new("Test Song".to_string(), "Test Composer".to_string());
        assert_eq!(score.title, "Test Song");
        assert_eq!(score.composer, "Test Composer");
        assert_eq!(score.tempo, 120.0);
    }

    #[test]
    fn test_note_addition() {
        let mut score = MusicalScore::new("Test".to_string(), "Test".to_string());
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let note = MusicalNote::new(event, 0.0, 1.0);

        score.add_note(note);
        assert_eq!(score.notes.len(), 1);
        assert!(score.duration_in_beats() > 0.0);
    }

    #[test]
    fn test_transposition() {
        let mut score = MusicalScore::new("Test".to_string(), "Test".to_string());
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let note = MusicalNote::new(event, 0.0, 1.0);
        let original_frequency = note.event.frequency;

        score.add_note(note);
        score.transpose(12); // Transpose up an octave

        assert!((score.notes[0].event.frequency / original_frequency - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_tempo_change() {
        let mut score = MusicalScore::new("Test".to_string(), "Test".to_string());
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let note = MusicalNote::new(event, 2.0, 1.0);

        score.add_note(note);
        score.set_tempo(240.0); // Double tempo

        assert_eq!(score.tempo, 240.0);
        assert!((score.notes[0].start_time - 1.0).abs() < 0.01); // Should be halved
    }

    #[test]
    fn test_quantization() {
        let mut score = MusicalScore::new("Test".to_string(), "Test".to_string());
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let note = MusicalNote::new(event, 0.3, 0.7);

        score.add_note(note);
        score.quantize(0.5);

        assert!((score.notes[0].start_time - 0.5).abs() < 0.01);
        assert!((score.notes[0].duration - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_score_validation() {
        let score = MusicalScore::new("Test".to_string(), "Test".to_string());
        assert!(score.validate().is_err()); // No notes

        let mut score_with_notes = score.clone();
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let note = MusicalNote::new(event, 0.0, 1.0);
        score_with_notes.add_note(note);

        assert!(score_with_notes.validate().is_ok());
    }

    #[test]
    fn test_score_processor() {
        let mut score = MusicalScore::new("Test".to_string(), "Test".to_string());
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let note = MusicalNote::new(event, 0.0, 1.0);
        score.add_note(note);

        let mut processor = ScoreProcessor::new(score);
        let options = ProcessingOptions {
            quantization: 0.5,
            humanization: 0.1,
            ..Default::default()
        };
        processor.set_options(options);

        let processed_score = processor.process().unwrap();
        assert_eq!(processed_score.notes.len(), 1);
    }
}
