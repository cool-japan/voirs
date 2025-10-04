//! Custom optimized score format for VoiRS singing synthesis
//!
//! This module provides an optimized internal score representation with
//! advanced features for efficient singing synthesis processing.

#![allow(dead_code, clippy::derivable_impls)]

use crate::score::{KeySignature, MusicalNote, MusicalScore, TimeSignature};
use crate::types::{Dynamics, Expression, NoteEvent};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

/// Optimized custom score format with time-indexed timeline and caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedScore {
    /// Score metadata including title, composer, tempo, etc.
    pub metadata: ScoreMetadata,
    /// Time-indexed note events for efficient lookup
    pub timeline: Timeline,
    /// Voice-specific data for polyphonic scores
    pub voices: HashMap<String, VoiceTrack>,
    /// Performance hints for optimization
    pub performance_hints: PerformanceHints,
    /// Cached calculations for expensive operations
    pub cache: ScoreCache,
    /// Format version for compatibility
    pub version: u32,
}

/// Score metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreMetadata {
    /// Title
    pub title: String,
    /// Composer
    pub composer: String,
    /// Key signature
    pub key_signature: KeySignature,
    /// Time signature
    pub time_signature: TimeSignature,
    /// Base tempo (BPM)
    pub base_tempo: f32,
    /// Total duration in beats
    pub duration_beats: f32,
    /// Total duration in seconds
    pub duration_seconds: f32,
    /// Score complexity rating (0.0-1.0)
    pub complexity: f32,
    /// Creation timestamp
    pub created: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp
    pub modified: chrono::DateTime<chrono::Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom properties
    pub properties: HashMap<String, String>,
}

/// Timeline structure for efficient time-based operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    /// Time grid resolution in beats per grid unit
    pub resolution: f32,
    /// Grid-based note events indexed by time position
    pub grid: BTreeMap<u64, GridCell>,
    /// Continuous events like tempo and dynamic changes
    pub continuous_events: BTreeMap<u64, Vec<ContinuousEvent>>,
    /// Index mapping note IDs to their grid positions
    pub note_index: HashMap<String, NoteReference>,
}

/// Grid cell containing events at a specific time point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridCell {
    /// Grid time position in grid units
    pub time_position: u64,
    /// Notes starting at this time point
    pub note_starts: Vec<OptimizedNote>,
    /// References to notes ending at this time point
    pub note_ends: Vec<NoteReference>,
    /// Tempo changes occurring at this time
    pub tempo_changes: Vec<TempoChange>,
    /// Dynamic markings at this time
    pub dynamic_changes: Vec<DynamicChange>,
    /// Expression changes at this time
    pub expression_changes: Vec<ExpressionChange>,
}

/// Optimized note representation with caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedNote {
    /// Unique note identifier
    pub id: String,
    /// Note event data with pitch, duration, and expression
    pub event: NoteEvent,
    /// Start time in grid units
    pub start_grid: u64,
    /// Duration in grid units
    pub duration_grid: u64,
    /// Voice track assignment
    pub voice_id: String,
    /// Polyphony layer (0 = lowest)
    pub layer: u8,
    /// Processing priority (0-10, higher = more important)
    pub priority: u8,
    /// Optimization flags for caching and effects
    pub flags: NoteFlags,
    /// Precomputed frequency curve for vibrato and pitch bends
    pub frequency_cache: Vec<f32>,
    /// Phoneme timing information for lyrics
    pub phoneme_timing: Vec<PhonemeSegment>,
}

/// Note flags for performance optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NoteFlags {
    /// Note can be cached for reuse
    pub cacheable: bool,
    /// Note requires real-time processing priority
    pub realtime: bool,
    /// Note has complex effects requiring extra processing
    pub complex_effects: bool,
    /// Note is part of a harmonic chord
    pub harmony_note: bool,
    /// Note has pitch bend modulation
    pub has_pitch_bend: bool,
    /// Note has vibrato effect
    pub has_vibrato: bool,
    /// Note has breath sound effects
    pub has_breath: bool,
}

/// Voice track for polyphonic scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceTrack {
    /// Voice track name
    pub name: String,
    /// Voice characteristics (range, timbre, etc.)
    pub characteristics: crate::types::VoiceCharacteristics,
    /// Default singing technique for this voice
    pub default_technique: crate::techniques::SingingTechnique,
    /// Note references in chronological order
    pub notes: Vec<NoteReference>,
    /// Voice-specific effects applied
    pub effects: Vec<String>,
    /// Voice volume level (0.0-1.0)
    pub volume: f32,
    /// Voice pan position (-1.0 = left, 1.0 = right)
    pub pan: f32,
}

/// Reference to a note in the timeline grid
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NoteReference {
    /// Grid position in timeline
    pub grid_position: u64,
    /// Index within the grid cell's note array
    pub cell_index: usize,
}

/// Continuous event (tempo, dynamics, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContinuousEvent {
    /// Tempo change
    Tempo(TempoChange),
    /// Dynamic marking
    Dynamic(DynamicChange),
    /// Expression change
    Expression(ExpressionChange),
    /// Key change
    KeyChange(KeySignature),
    /// Time signature change
    TimeSignature(TimeSignature),
}

/// Tempo change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoChange {
    /// New tempo in beats per minute (BPM)
    pub tempo: f32,
    /// Transition duration in beats for gradual changes
    pub transition_duration: f32,
    /// Type of tempo change (immediate, accelerando, etc.)
    pub change_type: TempoChangeType,
}

/// Dynamic level change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicChange {
    /// New dynamics level (pp, p, mf, f, ff, etc.)
    pub dynamics: Dynamics,
    /// Transition duration in beats for gradual changes
    pub transition_duration: f32,
    /// Voice tracks affected by this change
    pub voices: Vec<String>,
}

/// Musical expression change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionChange {
    /// New expression type (happy, sad, excited, etc.)
    pub expression: Expression,
    /// Duration in beats for this expression
    pub duration: f32,
    /// Expression intensity (0.0-1.0)
    pub intensity: f32,
    /// Voice tracks affected by this change
    pub voices: Vec<String>,
}

/// Tempo change types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TempoChangeType {
    /// Immediate change
    Immediate,
    /// Gradual acceleration
    Accelerando,
    /// Gradual deceleration
    Ritardando,
    /// Return to original tempo
    ATempo,
}

/// Phoneme segment timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeSegment {
    /// Phoneme symbol (IPA or X-SAMPA)
    pub phoneme: String,
    /// Start time within note normalized (0.0-1.0)
    pub start: f32,
    /// Duration within note normalized (0.0-1.0)
    pub duration: f32,
    /// Phoneme intensity/emphasis (0.0-1.0)
    pub intensity: f32,
}

/// Performance optimization hints for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHints {
    /// Recommended buffer size in samples
    pub buffer_size: usize,
    /// Whether parallel processing is recommended
    pub parallel_processing: bool,
    /// Note IDs that should be precomputed
    pub precompute_notes: Vec<String>,
    /// Estimated memory usage in bytes
    pub memory_estimate: usize,
    /// Processing complexity score (0.0-1.0)
    pub complexity_score: f32,
    /// Recommended quality level (0-10)
    pub recommended_quality: u8,
    /// Whether real-time synthesis is feasible
    pub realtime_feasible: bool,
}

/// Score cache for expensive calculations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreCache {
    /// Precomputed frequency curves indexed by note ID
    pub frequency_curves: HashMap<String, Vec<f32>>,
    /// Precomputed amplitude envelopes indexed by note ID
    pub amplitude_envelopes: HashMap<String, Vec<f32>>,
    /// Harmony analysis results at each grid position
    pub harmony_analysis: HashMap<u64, HarmonyInfo>,
    /// Detected phrase boundaries
    pub phrase_boundaries: Vec<PhraseBoundary>,
    /// Suggested breath placement locations
    pub breath_suggestions: Vec<BreathSuggestion>,
}

/// Harmony information at a specific time point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyInfo {
    /// Root note of the chord
    pub root: String,
    /// Chord quality (major, minor, diminished, etc.)
    pub quality: String,
    /// Chord extensions (7th, 9th, etc.)
    pub extensions: Vec<String>,
    /// Chord inversion (0 = root position)
    pub inversion: u8,
    /// Harmonic stability score (0.0-1.0)
    pub stability: f32,
}

/// Phrase boundary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseBoundary {
    /// Position in beats from score start
    pub position: f32,
    /// Type of boundary (minor, major, section, etc.)
    pub boundary_type: BoundaryType,
    /// Whether a breath is recommended at this boundary
    pub breath_recommended: bool,
    /// Boundary strength score (0.0-1.0)
    pub strength: f32,
}

/// Breath placement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathSuggestion {
    /// Position in beats from score start
    pub position: f32,
    /// Type of breath (natural, deep, quick)
    pub breath_type: crate::types::BreathType,
    /// Breath duration in seconds
    pub duration: f32,
    /// Breath priority (0-10, higher = more important)
    pub priority: u8,
}

/// Boundary types for phrase analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Minor phrase boundary
    Minor,
    /// Major phrase boundary
    Major,
    /// Section boundary
    Section,
    /// Movement boundary
    Movement,
}

/// Score optimizer for creating optimized scores
pub struct ScoreOptimizer {
    /// Grid resolution in beats per grid unit
    grid_resolution: f32,
    /// Whether to enable precomputation caching
    enable_caching: bool,
    /// Maximum cache size in bytes
    max_cache_size: usize,
}

impl OptimizedScore {
    /// Create optimized score from standard musical score
    pub fn from_musical_score(
        score: MusicalScore,
        optimizer: &ScoreOptimizer,
    ) -> crate::Result<Self> {
        let mut optimized = Self::new(score.title.clone(), score.composer.clone());

        // Copy metadata
        optimized.metadata.key_signature = score.key_signature;
        optimized.metadata.time_signature = score.time_signature;
        optimized.metadata.base_tempo = score.tempo;
        optimized.metadata.duration_beats = score.duration_in_beats();
        optimized.metadata.duration_seconds = score.duration.as_secs_f32();

        // Build timeline
        optimized.timeline.resolution = optimizer.grid_resolution;
        optimized.build_timeline_from_notes(&score.notes, optimizer)?;

        // Analyze and cache
        if optimizer.enable_caching {
            optimized.analyze_and_cache()?;
        }

        // Generate performance hints
        optimized.generate_performance_hints();

        Ok(optimized)
    }

    /// Create new optimized score
    pub fn new(title: String, composer: String) -> Self {
        Self {
            metadata: ScoreMetadata {
                title,
                composer,
                key_signature: KeySignature::default(),
                time_signature: TimeSignature::default(),
                base_tempo: 120.0,
                duration_beats: 0.0,
                duration_seconds: 0.0,
                complexity: 0.0,
                created: chrono::Utc::now(),
                modified: chrono::Utc::now(),
                tags: Vec::new(),
                properties: HashMap::new(),
            },
            timeline: Timeline {
                resolution: 0.25, // 16th note resolution
                grid: BTreeMap::new(),
                continuous_events: BTreeMap::new(),
                note_index: HashMap::new(),
            },
            voices: HashMap::new(),
            performance_hints: PerformanceHints::default(),
            cache: ScoreCache::default(),
            version: 1,
        }
    }

    /// Build timeline from musical notes
    fn build_timeline_from_notes(
        &mut self,
        notes: &[MusicalNote],
        optimizer: &ScoreOptimizer,
    ) -> crate::Result<()> {
        for (i, note) in notes.iter().enumerate() {
            let note_id = format!("note_{i}");
            let start_grid = (note.start_time / self.timeline.resolution) as u64;
            let duration_grid = (note.duration / self.timeline.resolution).ceil() as u64;

            // Create optimized note
            let optimized_note = OptimizedNote {
                id: note_id.clone(),
                event: note.event.clone(),
                start_grid,
                duration_grid,
                voice_id: "main".to_string(), // Default voice
                layer: 0,
                priority: 5,
                flags: self.analyze_note_flags(note),
                frequency_cache: if optimizer.enable_caching {
                    self.precompute_frequency_curve(note)
                } else {
                    Vec::new()
                },
                phoneme_timing: self.analyze_phoneme_timing(note),
            };

            // Add to grid
            let grid_cell = self.timeline.grid.entry(start_grid).or_insert(GridCell {
                time_position: start_grid,
                note_starts: Vec::new(),
                note_ends: Vec::new(),
                tempo_changes: Vec::new(),
                dynamic_changes: Vec::new(),
                expression_changes: Vec::new(),
            });

            let cell_index = grid_cell.note_starts.len();
            grid_cell.note_starts.push(optimized_note);

            // Add to note index
            let note_ref = NoteReference {
                grid_position: start_grid,
                cell_index,
            };
            self.timeline.note_index.insert(note_id, note_ref);

            // Add note end reference
            let end_grid = start_grid + duration_grid;
            let end_cell = self.timeline.grid.entry(end_grid).or_insert(GridCell {
                time_position: end_grid,
                note_starts: Vec::new(),
                note_ends: Vec::new(),
                tempo_changes: Vec::new(),
                dynamic_changes: Vec::new(),
                expression_changes: Vec::new(),
            });
            end_cell.note_ends.push(note_ref);
        }

        Ok(())
    }

    /// Analyze note flags for optimization
    fn analyze_note_flags(&self, note: &MusicalNote) -> NoteFlags {
        NoteFlags {
            cacheable: note.duration > 0.5,      // Cache longer notes
            realtime: note.event.velocity > 0.8, // High velocity = realtime priority
            complex_effects: !note.ornaments.is_empty(),
            harmony_note: note.chord.is_some(),
            has_pitch_bend: note.pitch_bend.is_some(),
            has_vibrato: note.event.vibrato > 0.1,
            has_breath: note.event.breath_before > 0.0,
        }
    }

    /// Precompute frequency curve for a note
    fn precompute_frequency_curve(&self, note: &MusicalNote) -> Vec<f32> {
        // Simple frequency curve - would be more sophisticated in practice
        let num_points = (note.duration * 100.0) as usize; // 100 points per beat
        let mut curve = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let t = i as f32 / num_points as f32;
            let mut freq = note.event.frequency;

            // Apply vibrato
            if note.event.vibrato > 0.0 {
                let vibrato_freq = 6.0; // Hz
                let vibrato_amount = note.event.vibrato * 0.05; // 5% max
                freq *=
                    1.0 + vibrato_amount * (2.0 * std::f32::consts::PI * vibrato_freq * t).sin();
            }

            curve.push(freq);
        }

        curve
    }

    /// Analyze phoneme timing
    fn analyze_phoneme_timing(&self, note: &MusicalNote) -> Vec<PhonemeSegment> {
        if note.event.phonemes.is_empty() {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let phoneme_count = note.event.phonemes.len();

        for (i, phoneme) in note.event.phonemes.iter().enumerate() {
            let start = i as f32 / phoneme_count as f32;
            let duration = 1.0 / phoneme_count as f32;

            segments.push(PhonemeSegment {
                phoneme: phoneme.clone(),
                start,
                duration,
                intensity: 1.0, // Default intensity
            });
        }

        segments
    }

    /// Analyze score and populate cache
    fn analyze_and_cache(&mut self) -> crate::Result<()> {
        // Analyze harmony
        self.analyze_harmony();

        // Detect phrase boundaries
        self.detect_phrase_boundaries();

        // Generate breath suggestions
        self.generate_breath_suggestions();

        Ok(())
    }

    /// Analyze harmony at different time points
    fn analyze_harmony(&mut self) {
        for (&grid_pos, cell) in &self.timeline.grid {
            if !cell.note_starts.is_empty() {
                let harmony = self.analyze_harmony_at_position(&cell.note_starts);
                self.cache.harmony_analysis.insert(grid_pos, harmony);
            }
        }
    }

    /// Analyze harmony at a specific position
    fn analyze_harmony_at_position(&self, notes: &[OptimizedNote]) -> HarmonyInfo {
        if notes.is_empty() {
            return HarmonyInfo {
                root: "C".to_string(),
                quality: "major".to_string(),
                extensions: Vec::new(),
                inversion: 0,
                stability: 0.5,
            };
        }

        // Simple harmony analysis - extract frequencies and determine chord
        let mut frequencies: Vec<f32> = notes.iter().map(|n| n.event.frequency).collect();
        frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Basic chord recognition (simplified)
        HarmonyInfo {
            root: notes[0].event.note.clone(),
            quality: if notes.len() >= 3 {
                "triad"
            } else {
                "interval"
            }
            .to_string(),
            extensions: Vec::new(),
            inversion: 0,
            stability: if notes.len() >= 3 { 0.8 } else { 0.6 },
        }
    }

    /// Detect phrase boundaries
    fn detect_phrase_boundaries(&mut self) {
        let mut boundaries = Vec::new();
        let mut last_note_end = 0.0;

        for cell in self.timeline.grid.values() {
            for note in &cell.note_starts {
                let note_start = note.start_grid as f32 * self.timeline.resolution;
                let gap = note_start - last_note_end;

                // Large gaps suggest phrase boundaries
                if gap > 1.0 {
                    // More than 1 beat gap
                    boundaries.push(PhraseBoundary {
                        position: note_start,
                        boundary_type: if gap > 2.0 {
                            BoundaryType::Major
                        } else {
                            BoundaryType::Minor
                        },
                        breath_recommended: gap > 0.5,
                        strength: (gap / 4.0).min(1.0), // Normalize to 0-1
                    });
                }

                let note_end = note_start + (note.duration_grid as f32 * self.timeline.resolution);
                last_note_end = last_note_end.max(note_end);
            }
        }

        self.cache.phrase_boundaries = boundaries;
    }

    /// Generate breath suggestions
    fn generate_breath_suggestions(&mut self) {
        let mut suggestions = Vec::new();

        for boundary in &self.cache.phrase_boundaries {
            if boundary.breath_recommended {
                suggestions.push(BreathSuggestion {
                    position: boundary.position,
                    breath_type: match boundary.boundary_type {
                        BoundaryType::Major | BoundaryType::Section => {
                            crate::types::BreathType::Deep
                        }
                        BoundaryType::Minor => crate::types::BreathType::Quick,
                        BoundaryType::Movement => crate::types::BreathType::Natural,
                    },
                    duration: match boundary.boundary_type {
                        BoundaryType::Major => 0.5,
                        BoundaryType::Minor => 0.2,
                        BoundaryType::Section => 0.8,
                        BoundaryType::Movement => 1.0,
                    },
                    priority: (boundary.strength * 10.0) as u8,
                });
            }
        }

        self.cache.breath_suggestions = suggestions;
    }

    /// Generate performance hints
    fn generate_performance_hints(&mut self) {
        let note_count = self.timeline.note_index.len();
        let complexity = self.calculate_complexity_score();

        self.performance_hints = PerformanceHints {
            buffer_size: if complexity > 0.7 { 1024 } else { 512 },
            parallel_processing: note_count > 50,
            precompute_notes: self.identify_precompute_candidates(),
            memory_estimate: note_count * 1024 + self.cache.frequency_curves.len() * 4096,
            complexity_score: complexity,
            recommended_quality: if complexity > 0.8 {
                8
            } else if complexity > 0.5 {
                6
            } else {
                4
            },
            realtime_feasible: complexity < 0.6 && note_count < 200,
        };
    }

    /// Calculate complexity score
    fn calculate_complexity_score(&self) -> f32 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        for cell in self.timeline.grid.values() {
            for note in &cell.note_starts {
                let note_complexity = (if note.flags.complex_effects { 0.3 } else { 0.0 })
                    + (if note.flags.has_pitch_bend { 0.2 } else { 0.0 })
                    + (if note.flags.has_vibrato { 0.1 } else { 0.0 })
                    + (if note.flags.harmony_note { 0.2 } else { 0.0 })
                    + (note.event.velocity * 0.2);

                score += note_complexity;
                total_weight += 1.0;
            }
        }

        if total_weight > 0.0 {
            score / total_weight
        } else {
            0.0
        }
    }

    /// Identify notes that would benefit from precomputation
    fn identify_precompute_candidates(&self) -> Vec<String> {
        let mut candidates = Vec::new();

        for cell in self.timeline.grid.values() {
            for note in &cell.note_starts {
                if note.flags.cacheable && (note.flags.complex_effects || note.flags.has_vibrato) {
                    candidates.push(note.id.clone());
                }
            }
        }

        candidates
    }

    /// Get notes in time range (optimized lookup)
    pub fn get_notes_in_range(&self, start_beats: f32, end_beats: f32) -> Vec<&OptimizedNote> {
        let start_grid = (start_beats / self.timeline.resolution) as u64;
        let end_grid = (end_beats / self.timeline.resolution) as u64;

        let mut notes = Vec::new();

        for grid_pos in start_grid..=end_grid {
            if let Some(cell) = self.timeline.grid.get(&grid_pos) {
                for note in &cell.note_starts {
                    let note_end_grid = note.start_grid + note.duration_grid;
                    if note.start_grid <= end_grid && note_end_grid >= start_grid {
                        notes.push(note);
                    }
                }
            }
        }

        notes
    }

    /// Convert back to standard musical score
    pub fn to_musical_score(&self) -> MusicalScore {
        let mut score =
            MusicalScore::new(self.metadata.title.clone(), self.metadata.composer.clone());

        score.key_signature = self.metadata.key_signature;
        score.time_signature = self.metadata.time_signature;
        score.tempo = self.metadata.base_tempo;
        score.duration = Duration::from_secs_f32(self.metadata.duration_seconds);

        // Convert optimized notes back to musical notes
        for cell in self.timeline.grid.values() {
            for opt_note in &cell.note_starts {
                let start_beats = opt_note.start_grid as f32 * self.timeline.resolution;
                let duration_beats = opt_note.duration_grid as f32 * self.timeline.resolution;

                let musical_note =
                    MusicalNote::new(opt_note.event.clone(), start_beats, duration_beats);

                score.add_note(musical_note);
            }
        }

        score
    }
}

impl Default for PerformanceHints {
    fn default() -> Self {
        Self {
            buffer_size: 512,
            parallel_processing: false,
            precompute_notes: Vec::new(),
            memory_estimate: 1024,
            complexity_score: 0.5,
            recommended_quality: 6,
            realtime_feasible: true,
        }
    }
}

impl ScoreOptimizer {
    /// Create new score optimizer
    pub fn new() -> Self {
        Self {
            grid_resolution: 0.25, // 16th note resolution
            enable_caching: true,
            max_cache_size: 1024 * 1024, // 1MB cache
        }
    }

    /// Set grid resolution
    pub fn with_resolution(mut self, resolution: f32) -> Self {
        self.grid_resolution = resolution;
        self
    }

    /// Enable/disable caching
    pub fn with_caching(mut self, enable: bool) -> Self {
        self.enable_caching = enable;
        self
    }

    /// Set maximum cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }
}

impl Default for ScoreOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NoteEvent;

    #[test]
    fn test_optimized_score_creation() {
        let score = OptimizedScore::new("Test Score".to_string(), "Test Composer".to_string());
        assert_eq!(score.metadata.title, "Test Score");
        assert_eq!(score.metadata.composer, "Test Composer");
        assert_eq!(score.version, 1);
    }

    #[test]
    fn test_score_optimizer() {
        let optimizer = ScoreOptimizer::new()
            .with_resolution(0.125)
            .with_caching(true)
            .with_cache_size(2048);

        assert_eq!(optimizer.grid_resolution, 0.125);
        assert!(optimizer.enable_caching);
        assert_eq!(optimizer.max_cache_size, 2048);
    }

    #[test]
    fn test_from_musical_score() {
        let mut musical_score = MusicalScore::new("Test".to_string(), "Test".to_string());
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let note = MusicalNote::new(event, 0.0, 1.0);
        musical_score.add_note(note);

        let optimizer = ScoreOptimizer::default();
        let optimized = OptimizedScore::from_musical_score(musical_score, &optimizer);
        assert!(optimized.is_ok());

        let score = optimized.unwrap();
        assert_eq!(score.timeline.grid.len(), 2); // Start and end positions
    }

    #[test]
    fn test_note_flags_analysis() {
        let score = OptimizedScore::new("Test".to_string(), "Test".to_string());
        let mut note = MusicalNote::new(
            NoteEvent::new("C".to_string(), 4, 1.0, 0.9),
            0.0,
            2.0, // Long duration
        );
        note.event.vibrato = 0.5;

        let flags = score.analyze_note_flags(&note);
        assert!(flags.cacheable); // Long note should be cacheable
        assert!(flags.realtime); // High velocity
        assert!(flags.has_vibrato);
    }

    #[test]
    fn test_complexity_calculation() {
        let mut score = OptimizedScore::new("Test".to_string(), "Test".to_string());

        // Add a simple note
        let mut grid_cell = GridCell {
            time_position: 0,
            note_starts: vec![OptimizedNote {
                id: "test1".to_string(),
                event: NoteEvent::new("C".to_string(), 4, 1.0, 0.5),
                start_grid: 0,
                duration_grid: 4,
                voice_id: "main".to_string(),
                layer: 0,
                priority: 5,
                flags: NoteFlags {
                    cacheable: false,
                    realtime: false,
                    complex_effects: false,
                    harmony_note: false,
                    has_pitch_bend: false,
                    has_vibrato: false,
                    has_breath: false,
                },
                frequency_cache: Vec::new(),
                phoneme_timing: Vec::new(),
            }],
            note_ends: Vec::new(),
            tempo_changes: Vec::new(),
            dynamic_changes: Vec::new(),
            expression_changes: Vec::new(),
        };

        score.timeline.grid.insert(0, grid_cell);
        let complexity = score.calculate_complexity_score();
        assert!(complexity < 0.5); // Simple note should have low complexity
    }
}
