//! Common types and data structures for musical intelligence

/// Chord quality types representing various harmonic structures
#[derive(Debug, Clone, PartialEq)]
pub enum ChordQuality {
    /// Major triad (root, major third, perfect fifth)
    Major,
    /// Minor triad (root, minor third, perfect fifth)
    Minor,
    /// Diminished triad (root, minor third, diminished fifth)
    Diminished,
    /// Augmented triad (root, major third, augmented fifth)
    Augmented,
    /// Dominant seventh chord (root, major third, perfect fifth, minor seventh)
    Dominant7,
    /// Major seventh chord (root, major third, perfect fifth, major seventh)
    Major7,
    /// Minor seventh chord (root, minor third, perfect fifth, minor seventh)
    Minor7,
    /// Minor-major seventh chord (root, minor third, perfect fifth, major seventh)
    MinorMajor7,
    /// Fully diminished seventh chord (root, minor third, diminished fifth, diminished seventh)
    Diminished7,
    /// Half-diminished seventh chord (root, minor third, diminished fifth, minor seventh)
    HalfDiminished7,
    /// Suspended second chord (root, major second, perfect fifth)
    Suspended2,
    /// Suspended fourth chord (root, perfect fourth, perfect fifth)
    Suspended4,
}

/// Key mode types representing different modal scales
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyMode {
    /// Major mode (Ionian) with intervals: W-W-H-W-W-W-H
    Major,
    /// Natural minor mode (Aeolian) with intervals: W-H-W-W-H-W-W
    Minor,
    /// Dorian mode with intervals: W-H-W-W-W-H-W
    Dorian,
    /// Phrygian mode with intervals: H-W-W-W-H-W-W
    Phrygian,
    /// Lydian mode with intervals: W-W-W-H-W-W-H
    Lydian,
    /// Mixolydian mode with intervals: W-W-H-W-W-H-W
    Mixolydian,
    /// Locrian mode with intervals: H-W-W-H-W-W-W
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
    ///
    /// # Arguments
    ///
    /// * `name` - Chord name (e.g., "Cmaj7", "Am")
    /// * `root` - Root note pitch class (0-11, where C=0)
    /// * `intervals` - Semitone intervals from root note
    /// * `quality` - Harmonic quality of the chord
    ///
    /// # Returns
    ///
    /// New chord template instance with default weight of 1.0
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
    ///
    /// # Arguments
    ///
    /// * `extensions` - Extension intervals (7th, 9th, 11th, 13th)
    ///
    /// # Returns
    ///
    /// Modified chord template with specified extensions
    pub fn with_extensions(mut self, extensions: Vec<u8>) -> Self {
        self.extensions = extensions;
        self
    }

    /// Set recognition weight
    ///
    /// # Arguments
    ///
    /// * `weight` - Recognition weight for pattern matching (higher values prioritize this chord)
    ///
    /// # Returns
    ///
    /// Modified chord template with specified weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

impl KeyProfile {
    /// Create major key profile using Krumhansl-Schmuckler weights
    ///
    /// # Returns
    ///
    /// Key profile optimized for major key detection with empirically-derived weights
    pub fn major() -> Self {
        Self {
            weights: [
                6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
            ],
            mode: KeyMode::Major,
            name: "Major".to_string(),
        }
    }

    /// Create minor key profile using Krumhansl-Schmuckler weights
    ///
    /// # Returns
    ///
    /// Key profile optimized for minor key detection with empirically-derived weights
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
    ///
    /// # Returns
    ///
    /// Scale characteristics with high brightness, low tension, suitable for classical/pop/folk contexts
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
    ///
    /// # Returns
    ///
    /// Scale characteristics with low brightness, moderate tension, suitable for classical/folk/blues contexts
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
