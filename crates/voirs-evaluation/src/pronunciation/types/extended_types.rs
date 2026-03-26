//! Extended types for pronunciation evaluation: emotional prosody, cross-linguistic analysis,
//! phonetic features, and prosodic structures.

use crate::traits::WordPronunciationScore;
use voirs_sdk::LanguageCode;

/// Emotional dynamics over time
#[derive(Debug, Clone)]
pub struct EmotionalDynamics {
    /// Emotional trajectory over time
    pub emotion_trajectory: Vec<(f32, EmotionalState, f32)>,
    /// Emotional stability (low values indicate rapid changes)
    pub emotional_stability: f32,
    /// Peak emotional intensity
    pub peak_intensity: f32,
    /// Emotional transitions
    pub emotion_transitions: Vec<EmotionalTransition>,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum VowelHeight {
    High = 0,
    Mid = 1,
    Low = 2,
}
/// Phonetic features for similarity analysis
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PhoneticFeatures {
    pub(crate) is_vowel: bool,
    pub(crate) height: VowelHeight,
    pub(crate) backness: VowelBackness,
    pub(crate) rounded: bool,
    pub(crate) place: PlaceOfArticulation,
    pub(crate) manner: MannerOfArticulation,
    pub(crate) voiced: bool,
}
impl PhoneticFeatures {
    pub(crate) fn vowel(height: VowelHeight, backness: VowelBackness, rounded: bool) -> Self {
        Self {
            is_vowel: true,
            height,
            backness,
            rounded,
            place: PlaceOfArticulation::Glottal,
            manner: MannerOfArticulation::Approximant,
            voiced: true,
        }
    }
    pub(crate) fn consonant(
        place: PlaceOfArticulation,
        manner: MannerOfArticulation,
        voiced: bool,
    ) -> Self {
        Self {
            is_vowel: false,
            height: VowelHeight::Mid,
            backness: VowelBackness::Central,
            rounded: false,
            place,
            manner,
            voiced,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PlaceOfArticulation {
    Bilabial = 0,
    Labiodental = 1,
    Dental = 2,
    Alveolar = 3,
    PostAlveolar = 4,
    Palatal = 5,
    Velar = 6,
    Glottal = 7,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MannerOfArticulation {
    Stop = 0,
    Fricative = 1,
    Affricate = 2,
    Nasal = 3,
    Lateral = 4,
    Approximant = 5,
}
/// Prosodic timing types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimingType {
    /// Stress-timed languages (English, German)
    StressTimed,
    /// Syllable-timed languages (Spanish, French, Italian)
    SyllableTimed,
    /// Mora-timed languages (Japanese)
    MoraTimed,
    /// Mixed timing
    Mixed,
}
/// Complete emotional prosody analysis result
#[derive(Debug, Clone)]
pub struct EmotionalProsodyScore {
    /// Detected primary emotional state
    pub detected_emotion: EmotionalState,
    /// How well the detected emotion matches expected emotion
    pub emotional_appropriateness: f32,
    /// Intensity of emotional expression (0.0 = flat, 1.0 = very expressive)
    pub emotional_intensity: f32,
    /// Consistency of emotional expression throughout
    pub emotional_consistency: f32,
    /// Emotional dynamics analysis
    pub emotional_dynamics: EmotionalDynamics,
    /// Underlying prosodic features
    pub prosodic_features: EmotionalProsodicFeatures,
    /// Confidence in emotion detection
    pub confidence: f32,
}
/// Language-specific rhythm norms
#[derive(Debug, Clone)]
pub struct RhythmNorms {
    /// Variability coefficient norms for syllables
    pub syllable_variability: (f32, f32),
    /// Variability coefficient norms for vowels
    pub vowel_variability: (f32, f32),
    /// Stress pattern regularity
    pub stress_regularity: f32,
    /// Typical rhythm class index
    pub rhythm_class_index: f32,
}
/// Emotional state classification for prosody analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmotionalState {
    /// Neutral emotional state
    Neutral,
    /// Positive emotions
    Happy,
    /// High-energy positive emotion
    Excited,
    /// Unexpected positive reaction
    Surprised,
    /// Negative emotions
    Sad,
    /// High-energy negative emotion
    Angry,
    /// Persistent negative emotion
    Frustrated,
    /// Low-energy negative emotion
    Disappointed,
    /// Complex emotions
    Anxious,
    /// Self-assured emotional state
    Confident,
    /// Hesitant or doubtful emotional state
    Uncertain,
    /// Deliberate emphasis or stress
    Emphatic,
}
/// Prosodic features for adaptation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProsodicFeature {
    /// Fundamental frequency patterns
    F0Range,
    /// Speaking rate
    SpeakingRate,
    /// Stress patterns
    StressPattern,
    /// Rhythm timing
    RhythmTiming,
    /// Pause patterns
    PausePattern,
    /// Intonation contours
    IntonationContour,
}
/// Language-specific prosodic parameters for cross-linguistic comparison
#[derive(Debug, Clone)]
pub struct LanguageProsodicProfile {
    /// Language code
    pub language: LanguageCode,
    /// Typical F0 range for this language
    pub f0_range: (f32, f32),
    /// Typical speaking rate for this language
    pub speaking_rate_range: (f32, f32),
    /// Stress timing vs syllable timing preference
    pub timing_preference: TimingType,
    /// Intonation pattern characteristics
    pub intonation_patterns: Vec<IntonationPattern>,
    /// Typical pause patterns
    pub pause_characteristics: PauseCharacteristics,
    /// Rhythm metrics norms
    pub rhythm_norms: RhythmNorms,
}
/// Pronunciation dictionary entry for multiple pronunciations
#[derive(Debug, Clone)]
pub struct PronunciationEntry {
    /// Word text
    pub word: String,
    /// Multiple possible pronunciations
    pub pronunciations: Vec<WordPronunciation>,
}
/// Emotional transition between states
#[derive(Debug, Clone)]
pub struct EmotionalTransition {
    /// Start time of transition
    pub start_time: f32,
    /// End time of transition
    pub end_time: f32,
    /// From emotional state
    pub from_emotion: EmotionalState,
    /// To emotional state
    pub to_emotion: EmotionalState,
    /// Transition smoothness (0.0 = abrupt, 1.0 = smooth)
    pub smoothness: f32,
}
/// Syllable-level accuracy score
#[derive(Debug, Clone)]
pub struct SyllableAccuracyScore {
    /// Expected syllable
    pub expected_syllable: SyllableInfo,
    /// Actual syllable (if detected)
    pub actual_syllable: Option<SyllableInfo>,
    /// Onset accuracy
    pub onset_accuracy: f32,
    /// Nucleus accuracy
    pub nucleus_accuracy: f32,
    /// Coda accuracy
    pub coda_accuracy: f32,
    /// Stress accuracy
    pub stress_accuracy: f32,
    /// Overall syllable score
    pub overall_accuracy: f32,
}
/// Cross-linguistic prosody comparison result
#[derive(Debug, Clone)]
pub struct CrossLinguisticProsodyScore {
    /// Source language
    pub source_language: LanguageCode,
    /// Target language for comparison
    pub target_language: LanguageCode,
    /// Prosodic transfer score (how well source patterns fit target)
    pub transfer_score: f32,
    /// Language distance metric
    pub language_distance: f32,
    /// Specific comparison results
    pub comparison_details: LanguageComparisonDetails,
    /// Adaptation recommendations
    pub adaptation_recommendations: Vec<ProsodyAdaptation>,
    /// Overall cross-linguistic intelligibility
    pub cross_linguistic_intelligibility: f32,
}
/// Enhanced syllable information
#[derive(Debug, Clone)]
pub struct SyllableInfo {
    /// Syllable text
    pub text: String,
    /// Phonemes in this syllable
    pub phonemes: Vec<String>,
    /// Stress level (0=unstressed, 1=secondary, 2=primary)
    pub stress_level: u8,
    /// Onset phonemes
    pub onset: Vec<String>,
    /// Nucleus (vowel core)
    pub nucleus: Vec<String>,
    /// Coda phonemes
    pub coda: Vec<String>,
    /// Start time in alignment
    pub start_time: f32,
    /// End time in alignment
    pub end_time: f32,
    /// Confidence score
    pub confidence: f32,
}
/// Language-specific intonation patterns
#[derive(Debug, Clone)]
pub struct IntonationPattern {
    /// Pattern name
    pub name: String,
    /// Typical F0 contour points (time, `relative_f0`)
    pub contour: Vec<(f32, f32)>,
    /// Usage frequency in the language
    pub frequency: f32,
    /// Semantic/pragmatic function
    pub function: IntonationFunction,
}
/// Prosodic features for emotional analysis
#[derive(Debug, Clone)]
pub struct EmotionalProsodicFeatures {
    /// Average pitch (F0) in Hz
    pub mean_f0: f32,
    /// Pitch standard deviation
    pub f0_std: f32,
    /// Pitch range (max - min)
    pub f0_range: f32,
    /// Speaking rate (syllables per second)
    pub speaking_rate: f32,
    /// Energy/intensity measures
    pub mean_energy: f32,
    /// Standard deviation of energy values
    pub energy_std: f32,
    /// Timing features
    pub pause_frequency: f32,
    /// Average duration of pauses
    pub pause_duration_mean: f32,
    /// Voice quality indicators
    pub jitter: f32,
    /// Amplitude variation between periods
    pub shimmer: f32,
    /// Rhythmic features
    pub rhythm_regularity: f32,
    /// Strength of stress pattern contrasts
    pub stress_pattern_strength: f32,
}
/// Pause placement preferences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PausePlacement {
    /// Between phrases
    Phrasal,
    /// Between sentences
    Sentential,
    /// Within phrases (rare)
    Intraphrasal,
    /// Breath groups
    Respiratory,
}
/// Detailed comparison between language prosodic patterns
#[derive(Debug, Clone)]
pub struct LanguageComparisonDetails {
    /// F0 pattern similarity
    pub f0_similarity: f32,
    /// Rhythm pattern similarity
    pub rhythm_similarity: f32,
    /// Stress pattern similarity
    pub stress_similarity: f32,
    /// Timing pattern similarity
    pub timing_similarity: f32,
    /// Intonation pattern similarity
    pub intonation_similarity: f32,
    /// Pause pattern similarity
    pub pause_similarity: f32,
}
/// Enhanced word-level assessment with stress and syllables
#[derive(Debug, Clone)]
pub struct EnhancedWordScore {
    /// Base word pronunciation score
    pub base_score: WordPronunciationScore,
    /// Lexical stress accuracy
    pub lexical_stress_accuracy: f32,
    /// Syllable boundary accuracy
    pub syllable_boundary_accuracy: f32,
    /// Syllable structure correctness
    pub syllable_structure_score: f32,
    /// Individual syllable scores
    pub syllable_scores: Vec<SyllableAccuracyScore>,
}
/// Single pronunciation of a word
#[derive(Debug, Clone)]
pub struct WordPronunciation {
    /// Phoneme sequence
    pub phonemes: Vec<String>,
    /// Syllable breakdown
    pub syllables: Vec<SyllableInfo>,
    /// Frequency/likelihood of this pronunciation
    pub frequency: f32,
    /// Language-specific notes
    pub notes: Option<String>,
}
/// Language-specific pause characteristics
#[derive(Debug, Clone)]
pub struct PauseCharacteristics {
    /// Typical pause frequency (pauses per second)
    pub typical_frequency: f32,
    /// Typical pause duration distribution
    pub duration_distribution: Vec<(f32, f32)>,
    /// Pause placement preferences
    pub placement_preferences: Vec<PausePlacement>,
}
/// Functions of intonation patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntonationFunction {
    /// Declarative statements
    Statement,
    /// Yes/no questions
    YesNoQuestion,
    /// Wh-questions
    WhQuestion,
    /// Commands/imperatives
    Command,
    /// Emotional expression
    Emotional,
    /// Focus/emphasis
    Focus,
    /// Continuation/listing
    Continuation,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum VowelBackness {
    Front = 0,
    Central = 1,
    Back = 2,
}
/// Prosody adaptation recommendation
#[derive(Debug, Clone)]
pub struct ProsodyAdaptation {
    /// Prosodic feature to adapt
    pub feature: ProsodicFeature,
    /// Current value/pattern
    pub current_value: f32,
    /// Target value/pattern for better cross-linguistic performance
    pub target_value: f32,
    /// Importance of this adaptation
    pub importance: f32,
    /// Specific recommendation text
    pub recommendation: String,
}
