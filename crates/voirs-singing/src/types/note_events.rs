//! Note event types and related structures for singing synthesis

use super::core_types::{Articulation, BendCurve, BreathType, Expression};
use serde::{Deserialize, Serialize};

/// Note event for singing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteEvent {
    /// Note name (C, D, E, F, G, A, B)
    pub note: String,
    /// Octave (0-8)
    pub octave: u8,
    /// Pitch in Hz
    pub frequency: f32,
    /// Note duration in beats
    pub duration: f32,
    /// Note velocity (0.0-1.0)
    pub velocity: f32,
    /// Vibrato intensity (0.0-1.0)
    pub vibrato: f32,
    /// Lyric text for this note
    pub lyric: Option<String>,
    /// Phoneme sequence for this note
    pub phonemes: Vec<String>,
    /// Expression for this note
    pub expression: Expression,
    /// Timing offset in seconds
    pub timing_offset: f32,
    /// Breath before note (0.0-1.0)
    pub breath_before: f32,
    /// Legato connection to next note
    pub legato: bool,
    /// Articulation type
    pub articulation: Articulation,
}

/// Pitch bend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchBend {
    /// Bend start time in beats
    pub start_time: f32,
    /// Bend duration in beats
    pub duration: f32,
    /// Bend amount in semitones
    pub amount: f32,
    /// Bend curve type
    pub curve: BendCurve,
}

/// Breath control information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathInfo {
    /// Breath position in beats
    pub position: f32,
    /// Breath duration in seconds
    pub duration: f32,
    /// Breath type
    pub breath_type: BreathType,
    /// Breath intensity (0.0-1.0)
    pub intensity: f32,
}

impl NoteEvent {
    /// Create a new note event
    pub fn new(note: String, octave: u8, duration: f32, velocity: f32) -> Self {
        Self {
            frequency: Self::note_to_frequency(&note, octave),
            note,
            octave,
            duration,
            velocity,
            vibrato: 0.5,
            lyric: None,
            phonemes: Vec::new(),
            expression: Expression::Neutral,
            timing_offset: 0.0,
            breath_before: 0.0,
            legato: false,
            articulation: Articulation::Normal,
        }
    }

    /// Convert note name and octave to frequency
    pub fn note_to_frequency(note: &str, octave: u8) -> f32 {
        let base_frequencies = [
            ("C", 16.35),
            ("C#", 17.32),
            ("Db", 17.32),
            ("D", 18.35),
            ("D#", 19.45),
            ("Eb", 19.45),
            ("E", 20.60),
            ("F", 21.83),
            ("F#", 23.12),
            ("Gb", 23.12),
            ("G", 24.50),
            ("G#", 25.96),
            ("Ab", 25.96),
            ("A", 27.50),
            ("A#", 29.14),
            ("Bb", 29.14),
            ("B", 30.87),
        ];

        let base_freq = base_frequencies
            .iter()
            .find(|(n, _)| n == &note)
            .map(|(_, f)| f)
            .copied()
            .unwrap_or(27.50); // Default to A0

        base_freq * 2.0_f32.powi(octave as i32)
    }

    /// Set lyric for this note
    pub fn with_lyric(mut self, lyric: String) -> Self {
        self.lyric = Some(lyric);
        self
    }

    /// Set phonemes for this note
    pub fn with_phonemes(mut self, phonemes: Vec<String>) -> Self {
        self.phonemes = phonemes;
        self
    }

    /// Set expression for this note
    pub fn with_expression(mut self, expression: Expression) -> Self {
        self.expression = expression;
        self
    }

    /// Set vibrato intensity
    pub fn with_vibrato(mut self, vibrato: f32) -> Self {
        self.vibrato = vibrato.clamp(0.0, 1.0);
        self
    }

    /// Set breath before note
    pub fn with_breath_before(mut self, breath: f32) -> Self {
        self.breath_before = breath.clamp(0.0, 1.0);
        self
    }

    /// Set legato connection
    pub fn with_legato(mut self, legato: bool) -> Self {
        self.legato = legato;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_to_frequency() {
        // Test A4 = 440 Hz
        assert!((NoteEvent::note_to_frequency("A", 4) - 440.0).abs() < 0.1);

        // Test C4 = 261.63 Hz
        assert!((NoteEvent::note_to_frequency("C", 4) - 261.63).abs() < 0.1);

        // Test octave relationship
        let c4 = NoteEvent::note_to_frequency("C", 4);
        let c5 = NoteEvent::note_to_frequency("C", 5);
        assert!((c5 / c4 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_note_event_creation() {
        let note = NoteEvent::new("C".to_string(), 4, 1.0, 0.8)
            .with_lyric("Hello".to_string())
            .with_expression(Expression::Happy)
            .with_vibrato(0.7);

        assert_eq!(note.note, "C");
        assert_eq!(note.octave, 4);
        assert_eq!(note.lyric, Some("Hello".to_string()));
        assert_eq!(note.expression, Expression::Happy);
        assert_eq!(note.vibrato, 0.7);
    }
}
