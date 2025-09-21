//! Common types and data structures for musical intelligence

/// Chord quality types
#[derive(Debug, Clone, PartialEq)]
pub enum ChordQuality {
    Major,
    Minor,
    Diminished,
    Augmented,
    Dominant7,
    Major7,
    Minor7,
    MinorMajor7,
    Diminished7,
    HalfDiminished7,
    Suspended2,
    Suspended4,
}

/// Key mode types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyMode {
    Major,
    Minor,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Locrian,
}

/// Template for chord recognition
#[derive(Debug, Clone)]
pub struct ChordTemplate {
    /// Chord name (e.g., "Cmaj7", "Am", "G7")
    pub name: String,
    /// Root note (0-11, C=0)
    pub root: u8,
    /// Chord intervals from root
    pub intervals: Vec<u8>,
    /// Chord quality (major, minor, diminished, etc.)
    pub quality: ChordQuality,
    /// Extensions (7th, 9th, 11th, 13th)
    pub extensions: Vec<u8>,
    /// Recognition weight
    pub weight: f32,
}

/// Result of chord recognition
#[derive(Debug, Clone)]
pub struct ChordResult {
    /// Recognized chord name
    pub chord_name: String,
    /// Root note
    pub root_note: String,
    /// Chord quality
    pub quality: ChordQuality,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Inversion (0=root position, 1=first inversion, etc.)
    pub inversion: u8,
    /// Bass note if different from root
    pub bass_note: Option<String>,
    /// Extensions present
    pub extensions: Vec<String>,
}

/// Profile for key detection using Krumhansl-Schmuckler algorithm
#[derive(Debug, Clone)]
pub struct KeyProfile {
    /// Profile weights for each pitch class
    pub weights: [f32; 12],
    /// Key mode
    pub mode: KeyMode,
    /// Profile name
    pub name: String,
}

/// Result of key detection
#[derive(Debug, Clone)]
pub struct KeyResult {
    /// Detected key (e.g., "C major", "A minor")
    pub key_name: String,
    /// Root note
    pub root_note: String,
    /// Key mode
    pub mode: KeyMode,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Alternative key candidates
    pub alternatives: Vec<(String, f32)>,
}

/// Musical scale pattern definition
#[derive(Debug, Clone)]
pub struct ScalePattern {
    /// Scale name
    pub name: String,
    /// Interval pattern (semitones from root)
    pub intervals: Vec<u8>,
    /// Scale characteristics
    pub characteristics: ScaleCharacteristics,
}

/// Scale characteristics and properties
#[derive(Debug, Clone)]
pub struct ScaleCharacteristics {
    /// Number of notes in scale
    pub note_count: u8,
    /// Modal brightness (relative major/minor character)
    pub brightness: f32,
    /// Tension level
    pub tension: f32,
    /// Common usage contexts
    pub contexts: Vec<String>,
}

/// Result of scale analysis
#[derive(Debug, Clone)]
pub struct ScaleResult {
    /// Detected scale name
    pub scale_name: String,
    /// Root note
    pub root_note: String,
    /// Scale pattern intervals
    pub intervals: Vec<u8>,
    /// Confidence score
    pub confidence: f32,
    /// Scale characteristics
    pub characteristics: ScaleCharacteristics,
    /// Notes present in the scale
    pub scale_notes: Vec<String>,
}

/// Rhythm pattern definition
#[derive(Debug, Clone)]
pub struct RhythmPattern {
    /// Pattern name
    pub name: String,
    /// Time signature
    pub time_signature: (u8, u8),
    /// Pattern as onset times (0.0-1.0 within measure)
    pub onset_pattern: Vec<f32>,
    /// Accent pattern (0.0-1.0 intensity)
    pub accent_pattern: Vec<f32>,
    /// Groove characteristics
    pub groove_type: String,
}

/// Result of rhythm analysis
#[derive(Debug, Clone)]
pub struct RhythmResult {
    /// Detected tempo (BPM)
    pub tempo: f32,
    /// Time signature
    pub time_signature: (u8, u8),
    /// Detected rhythm pattern
    pub pattern_name: String,
    /// Groove characteristics
    pub groove: GrooveCharacteristics,
    /// Confidence score
    pub confidence: f32,
    /// Swing ratio (if applicable)
    pub swing_ratio: Option<f32>,
}

/// Groove characteristics
#[derive(Debug, Clone)]
pub struct GrooveCharacteristics {
    /// Groove type (e.g., "straight", "swing", "shuffle")
    pub groove_type: String,
    /// Microtiming variations
    pub microtiming: Vec<f32>,
    /// Dynamic accents
    pub dynamics: Vec<f32>,
    /// Rhythmic density
    pub density: f32,
    /// Syncopation level
    pub syncopation: f32,
}

/// Comprehensive musical analysis result
#[derive(Debug, Clone)]
pub struct MusicalAnalysis {
    /// Chord progression analysis
    pub chord_analysis: Vec<ChordResult>,
    /// Key detection result
    pub key_analysis: KeyResult,
    /// Scale analysis results
    pub scale_analysis: Vec<ScaleResult>,
    /// Rhythm analysis result
    pub rhythm_analysis: RhythmResult,
    /// Overall analysis confidence
    pub overall_confidence: f32,
    /// Analysis metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl ChordTemplate {
    /// Create a new chord template
    pub fn new(name: String, root: u8, intervals: Vec<u8>, quality: ChordQuality) -> Self {
        Self {
            name,
            root,
            intervals,
            quality,
            extensions: Vec::new(),
            weight: 1.0,
        }
    }

    /// Add extensions to the chord
    pub fn with_extensions(mut self, extensions: Vec<u8>) -> Self {
        self.extensions = extensions;
        self
    }

    /// Set recognition weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

impl KeyProfile {
    /// Create major key profile
    pub fn major() -> Self {
        Self {
            weights: [
                6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
            ],
            mode: KeyMode::Major,
            name: "Major".to_string(),
        }
    }

    /// Create minor key profile
    pub fn minor() -> Self {
        Self {
            weights: [
                6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
            ],
            mode: KeyMode::Minor,
            name: "Minor".to_string(),
        }
    }
}

impl ScaleCharacteristics {
    /// Create characteristics for major scale
    pub fn major_scale() -> Self {
        Self {
            note_count: 7,
            brightness: 0.8,
            tension: 0.2,
            contexts: vec![
                "classical".to_string(),
                "pop".to_string(),
                "folk".to_string(),
            ],
        }
    }

    /// Create characteristics for minor scale
    pub fn minor_scale() -> Self {
        Self {
            note_count: 7,
            brightness: 0.3,
            tension: 0.6,
            contexts: vec![
                "classical".to_string(),
                "folk".to_string(),
                "blues".to_string(),
            ],
        }
    }
}
