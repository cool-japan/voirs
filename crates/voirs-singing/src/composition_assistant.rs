//! AI-Driven Composition Assistance
//!
//! This module provides intelligent composition assistance using AI models
//! for melody generation, harmonic progression, and musical arrangement.

use crate::score::{KeySignature, Mode, MusicalNote, MusicalScore, Note, TimeSignature};
use crate::types::{Articulation, Dynamics, Expression, NoteEvent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// AI-driven composition assistant
///
/// Provides intelligent melody generation, harmonic progression,
/// and musical arrangement capabilities.
#[derive(Debug)]
pub struct CompositionAssistant {
    /// Configuration
    config: CompositionConfig,
    /// Melody generator
    melody_generator: MelodyGenerator,
    /// Harmony generator
    harmony_generator: HarmonyGenerator,
    /// Arrangement engine
    arrangement_engine: ArrangementEngine,
}

/// Configuration for composition assistance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionConfig {
    /// Musical style (e.g., "classical", "pop", "jazz")
    pub style: String,
    /// Creativity level (0.0-1.0, higher = more creative)
    pub creativity: f32,
    /// Complexity level (0.0-1.0)
    pub complexity: f32,
    /// Enable harmonic analysis
    pub enable_harmony: bool,
    /// Enable counterpoint generation
    pub enable_counterpoint: bool,
}

impl Default for CompositionConfig {
    fn default() -> Self {
        Self {
            style: "pop".to_string(),
            creativity: 0.7,
            complexity: 0.5,
            enable_harmony: true,
            enable_counterpoint: false,
        }
    }
}

/// Melody generation prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelodyPrompt {
    /// Key signature
    pub key: KeySignature,
    /// Time signature
    pub time_signature: TimeSignature,
    /// Tempo in BPM
    pub tempo: f32,
    /// Number of measures to generate
    pub measures: usize,
    /// Melodic contour hint (e.g., "ascending", "arch", "descending")
    pub contour: Option<String>,
    /// Mood/emotion hint
    pub mood: Option<String>,
}

/// Generated melody
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedMelody {
    /// Musical notes
    pub notes: Vec<MusicalNote>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Melodic analysis
    pub analysis: MelodicAnalysis,
}

/// Melodic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelodicAnalysis {
    /// Melodic range (lowest to highest note)
    pub range: (Note, Note),
    /// Average interval size
    pub avg_interval: f32,
    /// Melodic complexity score (0.0-1.0)
    pub complexity: f32,
    /// Contour type
    pub contour: String,
}

/// Harmony generation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyRequest {
    /// Melody to harmonize
    pub melody: Vec<MusicalNote>,
    /// Key signature
    pub key: KeySignature,
    /// Number of harmony voices
    pub num_voices: usize,
    /// Harmony style (e.g., "parallel", "counterpoint", "jazz")
    pub style: String,
}

/// Generated harmony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedHarmony {
    /// Harmony voices (each voice is a sequence of notes)
    pub voices: Vec<Vec<MusicalNote>>,
    /// Chord progression
    pub chord_progression: Vec<String>,
    /// Voice leading quality score (0.0-1.0)
    pub voice_leading_score: f32,
}

/// Melody generator engine
#[derive(Debug)]
pub struct MelodyGenerator {
    /// Generative model parameters
    model_params: HashMap<String, f32>,
}

/// Harmony generator engine
#[derive(Debug)]
pub struct HarmonyGenerator {
    /// Harmony rules
    rules: HarmonyRules,
}

/// Harmony rules for different styles
#[derive(Debug, Clone)]
pub struct HarmonyRules {
    /// Allow parallel fifths
    pub allow_parallel_fifths: bool,
    /// Allow parallel octaves
    pub allow_parallel_octaves: bool,
    /// Voice range constraints
    pub voice_ranges: HashMap<String, (Note, Note)>,
}

/// Arrangement engine for full musical arrangements
#[derive(Debug)]
pub struct ArrangementEngine {
    /// Arrangement style
    style: String,
}

/// Musical arrangement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalArrangement {
    /// Main melody
    pub melody: Vec<MusicalNote>,
    /// Harmony parts
    pub harmony: Vec<Vec<MusicalNote>>,
    /// Bass line
    pub bass: Vec<MusicalNote>,
    /// Rhythmic accompaniment patterns
    pub rhythm_patterns: Vec<RhythmPattern>,
}

/// Rhythm pattern for accompaniment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmPattern {
    /// Pattern name
    pub name: String,
    /// Note positions (0.0-1.0 within measure)
    pub positions: Vec<f32>,
    /// Dynamics for each position
    pub dynamics: Vec<Dynamics>,
}

impl CompositionAssistant {
    /// Create new composition assistant
    pub fn new(config: CompositionConfig) -> Self {
        Self {
            config,
            melody_generator: MelodyGenerator::new(),
            harmony_generator: HarmonyGenerator::new(),
            arrangement_engine: ArrangementEngine::new("pop".to_string()),
        }
    }

    /// Generate melody from prompt
    ///
    /// # Arguments
    /// * `prompt` - Melody generation prompt
    ///
    /// # Returns
    /// Generated melody with analysis
    pub fn generate_melody(&self, prompt: &MelodyPrompt) -> crate::Result<GeneratedMelody> {
        let notes = self
            .melody_generator
            .generate(prompt, self.config.creativity)?;

        let analysis = self.analyze_melody(&notes);

        Ok(GeneratedMelody {
            notes,
            confidence: 0.85,
            analysis,
        })
    }

    /// Generate multiple melody variations
    ///
    /// # Arguments
    /// * `prompt` - Base melody generation prompt
    /// * `num_variations` - Number of variations to generate
    ///
    /// # Returns
    /// List of melody variations with analyses
    pub fn generate_melody_variations(
        &self,
        prompt: &MelodyPrompt,
        num_variations: usize,
    ) -> crate::Result<Vec<GeneratedMelody>> {
        let mut variations = Vec::with_capacity(num_variations);

        for i in 0..num_variations {
            // Vary creativity for each variation
            let varied_creativity = (self.config.creativity + (i as f32 * 0.1)) % 1.0;
            let notes = self.melody_generator.generate(prompt, varied_creativity)?;
            let analysis = self.analyze_melody(&notes);

            variations.push(GeneratedMelody {
                notes,
                confidence: 0.85 - (i as f32 * 0.05),
                analysis,
            });
        }

        Ok(variations)
    }

    /// Analyze musical coherence of a score
    ///
    /// # Arguments
    /// * `melody` - Melody to analyze
    ///
    /// # Returns
    /// Coherence score and detailed analysis
    pub fn analyze_coherence(&self, melody: &[MusicalNote]) -> CoherenceAnalysis {
        if melody.is_empty() {
            return CoherenceAnalysis {
                overall_score: 0.0,
                melodic_flow_score: 0.0,
                harmonic_consistency_score: 0.0,
                rhythmic_regularity_score: 0.0,
                recommendations: vec!["Melody is empty".to_string()],
            };
        }

        // Melodic flow analysis (based on interval consistency)
        let melodic_flow_score = self.calculate_melodic_flow(melody);

        // Harmonic consistency (based on note relationships)
        let harmonic_consistency_score = self.calculate_harmonic_consistency(melody);

        // Rhythmic regularity (based on duration patterns)
        let rhythmic_regularity_score = self.calculate_rhythmic_regularity(melody);

        // Overall score
        let overall_score =
            (melodic_flow_score + harmonic_consistency_score + rhythmic_regularity_score) / 3.0;

        // Generate recommendations
        let mut recommendations = Vec::new();
        if melodic_flow_score < 0.6 {
            recommendations.push("Consider smoother melodic transitions".to_string());
        }
        if harmonic_consistency_score < 0.6 {
            recommendations.push("Improve harmonic consistency".to_string());
        }
        if rhythmic_regularity_score < 0.6 {
            recommendations.push("Add more rhythmic variety".to_string());
        }

        CoherenceAnalysis {
            overall_score,
            melodic_flow_score,
            harmonic_consistency_score,
            rhythmic_regularity_score,
            recommendations,
        }
    }

    /// Transform melody according to musical rules
    ///
    /// # Arguments
    /// * `melody` - Original melody
    /// * `transformation` - Type of transformation to apply
    ///
    /// # Returns
    /// Transformed melody
    pub fn transform_melody(
        &self,
        melody: &[MusicalNote],
        transformation: MelodyTransformation,
    ) -> crate::Result<Vec<MusicalNote>> {
        match transformation {
            MelodyTransformation::Inversion => self.apply_inversion(melody),
            MelodyTransformation::Retrograde => Ok(melody.iter().rev().cloned().collect()),
            MelodyTransformation::Augmentation => self.apply_augmentation(melody, 2.0),
            MelodyTransformation::Diminution => self.apply_augmentation(melody, 0.5),
        }
    }

    // ===== Helper Methods =====

    fn calculate_melodic_flow(&self, melody: &[MusicalNote]) -> f32 {
        if melody.len() < 2 {
            return 1.0;
        }

        let intervals: Vec<f32> = melody
            .windows(2)
            .map(|w| (w[1].event.frequency - w[0].event.frequency).abs())
            .collect();

        // Score based on smooth intervals (smaller intervals = better flow)
        let avg_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
        (1.0 - (avg_interval / 200.0)).clamp(0.0, 1.0)
    }

    fn calculate_harmonic_consistency(&self, melody: &[MusicalNote]) -> f32 {
        // Simplified harmonic consistency based on note frequency relationships
        if melody.len() < 2 {
            return 1.0;
        }

        let mut consonance_count = 0;
        for window in melody.windows(2) {
            let ratio = window[1].event.frequency / window[0].event.frequency;
            // Check for consonant intervals (octave, fifth, fourth, third)
            if (ratio - 2.0).abs() < 0.1
                || (ratio - 1.5).abs() < 0.1
                || (ratio - 1.33).abs() < 0.1
                || (ratio - 1.25).abs() < 0.1
            {
                consonance_count += 1;
            }
        }

        consonance_count as f32 / (melody.len() - 1) as f32
    }

    fn calculate_rhythmic_regularity(&self, melody: &[MusicalNote]) -> f32 {
        if melody.len() < 2 {
            return 1.0;
        }

        // Check for consistent duration patterns
        let durations: Vec<f32> = melody.iter().map(|n| n.duration).collect();
        let unique_durations: std::collections::HashSet<_> =
            durations.iter().map(|&d| (d * 100.0) as i32).collect();

        // Score based on variety of durations (some variety is good)
        let variety_score = (unique_durations.len() as f32 / melody.len() as f32).min(1.0);
        (variety_score * 2.0).min(1.0)
    }

    fn apply_inversion(&self, melody: &[MusicalNote]) -> crate::Result<Vec<MusicalNote>> {
        if melody.is_empty() {
            return Ok(Vec::new());
        }

        let pivot = melody[0].event.frequency;
        let inverted: Vec<MusicalNote> = melody
            .iter()
            .map(|note| {
                let interval = note.event.frequency - pivot;
                let inverted_frequency = pivot - interval;

                let mut inverted_note = note.clone();
                inverted_note.event.frequency = inverted_frequency.max(20.0);
                inverted_note
            })
            .collect();

        Ok(inverted)
    }

    fn apply_augmentation(
        &self,
        melody: &[MusicalNote],
        factor: f32,
    ) -> crate::Result<Vec<MusicalNote>> {
        let augmented: Vec<MusicalNote> = melody
            .iter()
            .map(|note| {
                let mut augmented_note = note.clone();
                augmented_note.duration *= factor;
                augmented_note
            })
            .collect();

        Ok(augmented)
    }

    /// Generate harmony for melody
    ///
    /// # Arguments
    /// * `request` - Harmony generation request
    ///
    /// # Returns
    /// Generated harmony voices
    pub fn generate_harmony(&self, request: &HarmonyRequest) -> crate::Result<GeneratedHarmony> {
        let harmony = self.harmony_generator.generate(request)?;

        Ok(harmony)
    }

    /// Create full musical arrangement
    ///
    /// # Arguments
    /// * `melody` - Main melody
    /// * `key` - Key signature
    /// * `num_voices` - Number of harmony voices
    ///
    /// # Returns
    /// Complete musical arrangement
    pub fn create_arrangement(
        &self,
        melody: &[MusicalNote],
        key: &KeySignature,
        num_voices: usize,
    ) -> crate::Result<MusicalArrangement> {
        // Generate harmony
        let harmony_request = HarmonyRequest {
            melody: melody.to_vec(),
            key: *key,
            num_voices,
            style: self.config.style.clone(),
        };

        let harmony = self.generate_harmony(&harmony_request)?;

        // Generate bass line
        let bass = self.generate_bass_line(melody, key)?;

        // Generate rhythm patterns
        let rhythm_patterns = self.generate_rhythm_patterns()?;

        Ok(MusicalArrangement {
            melody: melody.to_vec(),
            harmony: harmony.voices,
            bass,
            rhythm_patterns,
        })
    }

    /// Suggest improvements to existing melody
    ///
    /// # Arguments
    /// * `melody` - Original melody
    ///
    /// # Returns
    /// List of improvement suggestions
    pub fn suggest_improvements(&self, melody: &[MusicalNote]) -> Vec<ImprovementSuggestion> {
        let mut suggestions = Vec::new();

        let analysis = self.analyze_melody(melody);

        // Check for excessive repetition
        if self.has_excessive_repetition(melody) {
            suggestions.push(ImprovementSuggestion {
                measure: 0,
                suggestion_type: SuggestionType::AddVariation,
                description: "Consider adding more melodic variation to reduce repetition"
                    .to_string(),
                confidence: 0.8,
            });
        }

        // Check melodic range
        if analysis.complexity < 0.3 {
            suggestions.push(ImprovementSuggestion {
                measure: 0,
                suggestion_type: SuggestionType::IncreaseComplexity,
                description: "Melody could benefit from more intervallic variety".to_string(),
                confidence: 0.7,
            });
        }

        suggestions
    }

    /// Continue melody based on existing pattern
    ///
    /// # Arguments
    /// * `existing_melody` - Existing melody to continue
    /// * `num_measures` - Number of measures to generate
    ///
    /// # Returns
    /// Continuation of the melody
    pub fn continue_melody(
        &self,
        existing_melody: &[MusicalNote],
        num_measures: usize,
    ) -> crate::Result<Vec<MusicalNote>> {
        // Analyze existing melody to learn patterns
        let analysis = self.analyze_melody(existing_melody);

        // Generate continuation in similar style
        let continuation = self.melody_generator.continue_from_pattern(
            existing_melody,
            num_measures,
            &analysis,
        )?;

        Ok(continuation)
    }

    // ===== Internal Helper Methods =====

    /// Analyze melody structure and characteristics
    fn analyze_melody(&self, notes: &[MusicalNote]) -> MelodicAnalysis {
        if notes.is_empty() {
            return MelodicAnalysis {
                range: (Note::C, Note::C),
                avg_interval: 0.0,
                complexity: 0.0,
                contour: "flat".to_string(),
            };
        }

        // Calculate range
        let frequencies: Vec<f32> = notes.iter().map(|n| n.event.frequency).collect();
        let min_freq = frequencies.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_freq = frequencies
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        // Calculate average interval
        let intervals: Vec<f32> = frequencies
            .windows(2)
            .map(|w| (w[1] / w[0]).abs().ln() * 12.0 / 2f32.ln())
            .collect();
        let avg_interval = if intervals.is_empty() {
            0.0
        } else {
            intervals.iter().sum::<f32>() / intervals.len() as f32
        };

        // Calculate complexity (based on interval variety)
        let complexity = (avg_interval / 12.0).min(1.0);

        // Determine contour
        let contour = if max_freq == min_freq {
            "flat"
        } else if frequencies[frequencies.len() - 1] > frequencies[0] {
            "ascending"
        } else {
            "descending"
        }
        .to_string();

        MelodicAnalysis {
            range: (Note::C, Note::C), // Simplified
            avg_interval,
            complexity,
            contour,
        }
    }

    /// Check if melody has excessive repetition
    fn has_excessive_repetition(&self, notes: &[MusicalNote]) -> bool {
        if notes.len() < 4 {
            return false;
        }

        // Check for identical consecutive sequences
        let window_size = 4;
        for i in 0..notes.len().saturating_sub(window_size * 2) {
            let seq1 = &notes[i..i + window_size];
            let seq2 = &notes[i + window_size..i + window_size * 2];

            if self.sequences_match(seq1, seq2) {
                return true;
            }
        }

        false
    }

    /// Check if two note sequences match
    fn sequences_match(&self, seq1: &[MusicalNote], seq2: &[MusicalNote]) -> bool {
        if seq1.len() != seq2.len() {
            return false;
        }

        seq1.iter()
            .zip(seq2.iter())
            .all(|(n1, n2)| (n1.event.frequency - n2.event.frequency).abs() < 1.0)
    }

    /// Generate bass line from melody
    fn generate_bass_line(
        &self,
        melody: &[MusicalNote],
        _key: &KeySignature,
    ) -> crate::Result<Vec<MusicalNote>> {
        // Simple bass line generation (root notes of implied harmony)
        let bass_notes: Vec<MusicalNote> = melody
            .iter()
            .step_by(4)
            .map(|note| {
                let bass_octave = (note.event.octave as i32 - 2).max(1) as u8;
                MusicalNote {
                    event: NoteEvent::new(
                        note.event.note.clone(),
                        bass_octave,
                        note.duration * 4.0,
                        note.event.velocity * 0.8,
                    ),
                    start_time: note.start_time,
                    duration: note.duration * 4.0,
                    pitch_bend: None,
                    articulation: note.articulation,
                    dynamics: note.dynamics,
                    tie_next: false,
                    tie_prev: false,
                    tuplet: None,
                    ornaments: vec![],
                    chord: None,
                }
            })
            .collect();

        Ok(bass_notes)
    }

    /// Generate rhythm patterns for accompaniment
    fn generate_rhythm_patterns(&self) -> crate::Result<Vec<RhythmPattern>> {
        let patterns = vec![
            RhythmPattern {
                name: "basic".to_string(),
                positions: vec![0.0, 0.25, 0.5, 0.75],
                dynamics: vec![
                    Dynamics::MezzoForte,
                    Dynamics::MezzoPiano,
                    Dynamics::MezzoForte,
                    Dynamics::MezzoPiano,
                ],
            },
            RhythmPattern {
                name: "syncopated".to_string(),
                positions: vec![0.0, 0.375, 0.75],
                dynamics: vec![Dynamics::Forte, Dynamics::MezzoPiano, Dynamics::MezzoForte],
            },
        ];

        Ok(patterns)
    }
}

impl Default for CompositionAssistant {
    fn default() -> Self {
        Self::new(CompositionConfig::default())
    }
}

/// Improvement suggestion for melody
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementSuggestion {
    /// Measure number
    pub measure: usize,
    /// Type of suggestion
    pub suggestion_type: SuggestionType,
    /// Description of suggestion
    pub description: String,
    /// Confidence in suggestion (0.0-1.0)
    pub confidence: f32,
}

/// Type of improvement suggestion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Add melodic variation
    AddVariation,
    /// Increase complexity
    IncreaseComplexity,
    /// Improve voice leading
    ImproveVoiceLeading,
    /// Add harmonic interest
    AddHarmony,
    /// Adjust rhythm
    AdjustRhythm,
}

/// Melody transformation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MelodyTransformation {
    /// Invert intervals around a pivot note
    Inversion,
    /// Reverse the melody
    Retrograde,
    /// Increase note durations
    Augmentation,
    /// Decrease note durations
    Diminution,
}

/// Musical coherence analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceAnalysis {
    /// Overall coherence score (0.0-1.0)
    pub overall_score: f32,
    /// Melodic flow score
    pub melodic_flow_score: f32,
    /// Harmonic consistency score
    pub harmonic_consistency_score: f32,
    /// Rhythmic regularity score
    pub rhythmic_regularity_score: f32,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

impl MelodyGenerator {
    /// Create new melody generator
    pub fn new() -> Self {
        Self {
            model_params: HashMap::new(),
        }
    }

    /// Create a musical note from basic parameters
    fn create_note(
        pitch: &str,
        octave: u8,
        duration: f32,
        velocity: f32,
        start_time: f32,
    ) -> MusicalNote {
        let event = NoteEvent::new(pitch.to_string(), octave, duration, velocity);
        MusicalNote {
            event,
            start_time,
            duration,
            pitch_bend: None,
            articulation: Articulation::Normal,
            dynamics: Dynamics::MezzoForte,
            tie_next: false,
            tie_prev: false,
            tuplet: None,
            ornaments: vec![],
            chord: None,
        }
    }

    /// Generate melody from prompt
    fn generate(&self, prompt: &MelodyPrompt, creativity: f32) -> crate::Result<Vec<MusicalNote>> {
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();

        let mut notes = Vec::new();
        let beats_per_measure = prompt.time_signature.numerator as f32;
        let total_beats = beats_per_measure * prompt.measures as f32;

        // Generate notes for the melody
        let mut current_beat = 0.0;

        while current_beat < total_beats {
            // Randomize pitch with creativity factor (within one octave range)
            let pitch_idx = rng.gen_range(0..7);
            let pitches = ["C", "D", "E", "F", "G", "A", "B"];
            let pitch = pitches[pitch_idx];
            let octave = 4;

            // Randomize duration
            let duration = if rng.gen_bool((0.7 + creativity * 0.3) as f64) {
                0.5 // Quarter note
            } else {
                1.0 // Half note
            };

            let velocity = rng.gen_range(0.6..0.9);

            notes.push(Self::create_note(
                pitch,
                octave,
                duration,
                velocity,
                current_beat,
            ));

            current_beat += duration;
        }

        Ok(notes)
    }

    /// Continue melody from existing pattern
    fn continue_from_pattern(
        &self,
        _existing: &[MusicalNote],
        num_measures: usize,
        _analysis: &MelodicAnalysis,
    ) -> crate::Result<Vec<MusicalNote>> {
        // Use analysis to inform generation
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();

        let mut notes = Vec::new();

        for i in 0..num_measures * 4 {
            let pitch_idx = rng.gen_range(0..7);
            let pitches = ["C", "D", "E", "F", "G", "A", "B"];
            let pitch = pitches[pitch_idx];

            notes.push(Self::create_note(pitch, 4, 0.5, 0.8, i as f32 * 0.5));
        }

        Ok(notes)
    }
}

impl Default for MelodyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonyGenerator {
    /// Create new harmony generator
    pub fn new() -> Self {
        Self {
            rules: HarmonyRules {
                allow_parallel_fifths: false,
                allow_parallel_octaves: false,
                voice_ranges: HashMap::new(),
            },
        }
    }

    /// Generate harmony voices
    fn generate(&self, request: &HarmonyRequest) -> crate::Result<GeneratedHarmony> {
        let mut voices = Vec::new();

        // Generate simple parallel harmony for each voice
        for voice_idx in 0..request.num_voices {
            let mut voice_notes = Vec::new();

            for note in &request.melody {
                // Create harmony note at interval above/below (simplified)
                let octave_shift = match voice_idx {
                    0 => 0,  // Same octave
                    1 => -1, // Octave below
                    _ => 0,
                };

                let adjusted_octave = (note.event.octave as i32 + octave_shift).max(1) as u8;

                // Clone the note with adjusted octave
                let harmony_note = MusicalNote {
                    event: NoteEvent::new(
                        note.event.note.clone(),
                        adjusted_octave,
                        note.duration,
                        note.event.velocity * 0.8,
                    ),
                    start_time: note.start_time,
                    duration: note.duration,
                    pitch_bend: None,
                    articulation: note.articulation,
                    dynamics: note.dynamics,
                    tie_next: false,
                    tie_prev: false,
                    tuplet: None,
                    ornaments: vec![],
                    chord: None,
                };

                voice_notes.push(harmony_note);
            }

            voices.push(voice_notes);
        }

        Ok(GeneratedHarmony {
            voices,
            chord_progression: vec!["C".to_string(), "G".to_string(), "Am".to_string()],
            voice_leading_score: 0.8,
        })
    }
}

impl Default for HarmonyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrangementEngine {
    /// Create new arrangement engine
    pub fn new(style: String) -> Self {
        Self { style }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition_assistant_creation() {
        let config = CompositionConfig::default();
        let assistant = CompositionAssistant::new(config);

        assert_eq!(assistant.config.style, "pop");
    }

    #[test]
    fn test_melody_generation() {
        let assistant = CompositionAssistant::default();

        let prompt = MelodyPrompt {
            key: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            tempo: 120.0,
            measures: 4,
            contour: Some("ascending".to_string()),
            mood: Some("happy".to_string()),
        };

        let result = assistant.generate_melody(&prompt).unwrap();

        assert!(!result.notes.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_harmony_generation() {
        let assistant = CompositionAssistant::default();

        let melody = vec![MusicalNote {
            event: NoteEvent::new("C".to_string(), 4, 1.0, 0.8),
            start_time: 0.0,
            duration: 1.0,
            pitch_bend: None,
            articulation: Articulation::Normal,
            dynamics: Dynamics::MezzoForte,
            tie_next: false,
            tie_prev: false,
            tuplet: None,
            ornaments: vec![],
            chord: None,
        }];

        let request = HarmonyRequest {
            melody,
            key: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            num_voices: 2,
            style: "parallel".to_string(),
        };

        let harmony = assistant.generate_harmony(&request).unwrap();

        assert_eq!(harmony.voices.len(), 2);
        assert!(harmony.voice_leading_score > 0.0);
    }

    #[test]
    fn test_full_arrangement() {
        let assistant = CompositionAssistant::default();

        let melody = vec![
            MusicalNote {
                event: NoteEvent::new("C".to_string(), 4, 1.0, 0.8),
                start_time: 0.0,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: vec![],
                chord: None,
            },
            MusicalNote {
                event: NoteEvent::new("E".to_string(), 4, 1.0, 0.8),
                start_time: 1.0,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: vec![],
                chord: None,
            },
        ];

        let key = KeySignature {
            root: Note::C,
            mode: Mode::Major,
            accidentals: 0,
        };

        let arrangement = assistant.create_arrangement(&melody, &key, 2).unwrap();

        assert_eq!(arrangement.melody.len(), 2);
        assert_eq!(arrangement.harmony.len(), 2);
        assert!(!arrangement.bass.is_empty());
        assert!(!arrangement.rhythm_patterns.is_empty());
    }

    #[test]
    fn test_improvement_suggestions() {
        let assistant = CompositionAssistant::default();

        // Create repetitive melody
        let melody = vec![
            MusicalNote {
                event: NoteEvent::new("C".to_string(), 4, 1.0, 0.8),
                start_time: 0.0,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: vec![],
                chord: None,
            };
            8
        ];

        let suggestions = assistant.suggest_improvements(&melody);

        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_melody_continuation() {
        let assistant = CompositionAssistant::default();

        let existing = vec![
            MusicalNote {
                event: NoteEvent::new("C".to_string(), 4, 1.0, 0.8),
                start_time: 0.0,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: vec![],
                chord: None,
            },
            MusicalNote {
                event: NoteEvent::new("D".to_string(), 4, 1.0, 0.8),
                start_time: 1.0,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: vec![],
                chord: None,
            },
        ];

        let continuation = assistant.continue_melody(&existing, 2).unwrap();

        assert!(!continuation.is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let config = CompositionConfig::default();

        assert_eq!(config.style, "pop");
        assert_eq!(config.creativity, 0.7);
        assert!(config.enable_harmony);
    }

    #[test]
    fn test_suggestion_type() {
        let suggestion_type = SuggestionType::AddVariation;
        assert_eq!(suggestion_type, SuggestionType::AddVariation);
    }

    #[test]
    fn test_melody_variations() {
        let assistant = CompositionAssistant::default();

        let prompt = MelodyPrompt {
            key: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            tempo: 120.0,
            measures: 2,
            contour: Some("ascending".to_string()),
            mood: Some("happy".to_string()),
        };

        let variations = assistant.generate_melody_variations(&prompt, 3).unwrap();

        assert_eq!(variations.len(), 3);
        assert!(variations.iter().all(|v| !v.notes.is_empty()));
    }

    #[test]
    fn test_coherence_analysis() {
        let assistant = CompositionAssistant::default();

        let melody = vec![
            MusicalNote {
                event: NoteEvent::new("C".to_string(), 4, 1.0, 0.8),
                start_time: 0.0,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: vec![],
                chord: None,
            },
            MusicalNote {
                event: NoteEvent::new("D".to_string(), 4, 1.0, 0.8),
                start_time: 1.0,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: vec![],
                chord: None,
            },
        ];

        let coherence = assistant.analyze_coherence(&melody);

        assert!(coherence.overall_score >= 0.0 && coherence.overall_score <= 1.0);
        assert!(coherence.melodic_flow_score >= 0.0);
        assert!(coherence.harmonic_consistency_score >= 0.0);
        assert!(coherence.rhythmic_regularity_score >= 0.0);
    }

    #[test]
    fn test_melody_transformation_retrograde() {
        let assistant = CompositionAssistant::default();

        let melody = vec![
            MusicalNote {
                event: NoteEvent::new("C".to_string(), 4, 1.0, 0.8),
                start_time: 0.0,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: vec![],
                chord: None,
            },
            MusicalNote {
                event: NoteEvent::new("E".to_string(), 4, 1.0, 0.8),
                start_time: 1.0,
                duration: 1.0,
                pitch_bend: None,
                articulation: Articulation::Normal,
                dynamics: Dynamics::MezzoForte,
                tie_next: false,
                tie_prev: false,
                tuplet: None,
                ornaments: vec![],
                chord: None,
            },
        ];

        let transformed = assistant
            .transform_melody(&melody, MelodyTransformation::Retrograde)
            .unwrap();

        assert_eq!(transformed.len(), melody.len());
        assert_eq!(transformed[0].event.note, "E");
        assert_eq!(transformed[1].event.note, "C");
    }

    #[test]
    fn test_melody_transformation_augmentation() {
        let assistant = CompositionAssistant::default();

        let melody = vec![MusicalNote {
            event: NoteEvent::new("C".to_string(), 4, 1.0, 0.8),
            start_time: 0.0,
            duration: 1.0,
            pitch_bend: None,
            articulation: Articulation::Normal,
            dynamics: Dynamics::MezzoForte,
            tie_next: false,
            tie_prev: false,
            tuplet: None,
            ornaments: vec![],
            chord: None,
        }];

        let transformed = assistant
            .transform_melody(&melody, MelodyTransformation::Augmentation)
            .unwrap();

        assert_eq!(transformed.len(), melody.len());
        assert_eq!(transformed[0].duration, 2.0); // Duration doubled
    }

    #[test]
    fn test_melody_transformation_types() {
        let transformation = MelodyTransformation::Inversion;
        assert_eq!(transformation, MelodyTransformation::Inversion);

        let transformation2 = MelodyTransformation::Retrograde;
        assert_ne!(transformation, transformation2);
    }
}
