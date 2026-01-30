//! Style characteristics types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Style characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleCharacteristics {
    /// Speaking style category
    pub speaking_style: SpeakingStyleCategory,

    /// Emotional characteristics
    pub emotional_characteristics: EmotionalCharacteristics,

    /// Prosodic characteristics
    pub prosodic_characteristics: ProsodicCharacteristics,

    /// Articulation characteristics
    pub articulation_characteristics: ArticulationCharacteristics,

    /// Voice quality characteristics
    pub voice_quality_characteristics: VoiceQualityCharacteristics,

    /// Cultural/regional characteristics
    pub cultural_characteristics: CulturalCharacteristics,
}

/// Speaking style category
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeakingStyleCategory {
    /// Conversational style
    Conversational,

    /// Formal presentation style
    Formal,

    /// Storytelling style
    Storytelling,

    /// News reading style
    NewsReading,

    /// Dramatic style
    Dramatic,

    /// Whispering style
    Whispering,

    /// Singing style
    Singing,

    /// Custom style
    Custom(String),
}

/// Emotional characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalCharacteristics {
    /// Primary emotion
    pub primary_emotion: EmotionType,

    /// Emotional intensity (0.0 to 1.0)
    pub intensity: f32,

    /// Emotional stability
    pub stability: f32,

    /// Emotional range
    pub emotional_range: Vec<EmotionType>,

    /// Emotional transitions
    pub transition_patterns: Vec<EmotionalTransition>,
}

/// Emotion type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmotionType {
    /// Neutral emotion
    Neutral,

    /// Happy emotion
    Happy,

    /// Sad emotion
    Sad,

    /// Angry emotion
    Angry,

    /// Fearful emotion
    Fearful,

    /// Surprised emotion
    Surprised,

    /// Disgusted emotion
    Disgusted,

    /// Excited emotion
    Excited,

    /// Calm emotion
    Calm,

    /// Confident emotion
    Confident,
}

/// Emotional transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalTransition {
    /// Source emotion
    pub from_emotion: EmotionType,

    /// Target emotion
    pub to_emotion: EmotionType,

    /// Transition duration (seconds)
    pub duration: f32,

    /// Transition curve
    pub transition_curve: TransitionCurve,
}

/// Transition curve type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionCurve {
    /// Linear transition
    Linear,

    /// Exponential transition
    Exponential,

    /// Sigmoid transition
    Sigmoid,

    /// Custom curve
    Custom,
}

/// Prosodic characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicCharacteristics {
    /// Fundamental frequency characteristics
    pub f0_characteristics: F0Characteristics,

    /// Rhythm characteristics
    pub rhythm_characteristics: RhythmCharacteristics,

    /// Stress characteristics
    pub stress_characteristics: StressCharacteristics,

    /// Intonation patterns
    pub intonation_patterns: Vec<IntonationPattern>,
}

/// F0 characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0Characteristics {
    /// Mean F0 (Hz)
    pub mean_f0: f32,

    /// F0 range (Hz)
    pub f0_range: (f32, f32),

    /// F0 variability
    pub f0_variability: f32,

    /// F0 contour patterns
    pub contour_patterns: Vec<F0ContourPattern>,

    /// Pitch accent patterns
    pub pitch_accent_patterns: Vec<PitchAccentPattern>,
}

/// F0 contour pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0ContourPattern {
    /// Pattern name
    pub name: String,

    /// Contour points (normalized time, normalized F0)
    pub contour_points: Vec<(f32, f32)>,

    /// Pattern frequency
    pub frequency: f32,

    /// Context conditions
    pub context_conditions: Vec<String>,
}

/// Pitch accent pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchAccentPattern {
    /// Accent type
    pub accent_type: AccentType,

    /// Accent strength
    pub strength: f32,

    /// Timing characteristics
    pub timing: AccentTiming,

    /// Frequency characteristics
    pub frequency_characteristics: AccentFrequencyCharacteristics,
}

/// Accent type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccentType {
    /// Rising accent
    Rising,

    /// Falling accent
    Falling,

    /// High accent
    High,

    /// Low accent
    Low,

    /// Complex accent
    Complex,
}

/// Accent timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccentTiming {
    /// Onset time (relative to syllable)
    pub onset: f32,

    /// Peak time (relative to syllable)
    pub peak: f32,

    /// Duration (relative to syllable)
    pub duration: f32,
}

/// Accent frequency characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccentFrequencyCharacteristics {
    /// Peak frequency (Hz)
    pub peak_frequency: f32,

    /// Frequency excursion (Hz)
    pub frequency_excursion: f32,

    /// Frequency slope (Hz/s)
    pub frequency_slope: f32,
}

/// Rhythm characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmCharacteristics {
    /// Speaking rate (syllables per second)
    pub speaking_rate: f32,

    /// Rate variability
    pub rate_variability: f32,

    /// Pause patterns
    pub pause_patterns: Vec<PausePattern>,

    /// Rhythmic patterns
    pub rhythmic_patterns: Vec<RhythmicPattern>,

    /// Tempo characteristics
    pub tempo_characteristics: TempoCharacteristics,
}

/// Pause pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PausePattern {
    /// Pause type
    pub pause_type: PauseType,

    /// Average duration (seconds)
    pub average_duration: f32,

    /// Duration variability
    pub duration_variability: f32,

    /// Frequency of occurrence
    pub frequency: f32,

    /// Context conditions
    pub context_conditions: Vec<String>,
}

/// Pause type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauseType {
    /// Breath pause
    Breath,

    /// Hesitation pause
    Hesitation,

    /// Syntactic pause
    Syntactic,

    /// Emphatic pause
    Emphatic,

    /// Silent pause
    Silent,
}

/// Rhythmic pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmicPattern {
    /// Pattern name
    pub name: String,

    /// Beat pattern
    pub beat_pattern: Vec<f32>,

    /// Pattern strength
    pub strength: f32,

    /// Pattern regularity
    pub regularity: f32,
}

/// Tempo characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoCharacteristics {
    /// Base tempo (BPM)
    pub base_tempo: f32,

    /// Tempo variations
    pub tempo_variations: Vec<TempoVariation>,

    /// Acceleration patterns
    pub acceleration_patterns: Vec<AccelerationPattern>,

    /// Rubato characteristics
    pub rubato_characteristics: RubatoCharacteristics,
}

/// Tempo variation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoVariation {
    /// Variation type
    pub variation_type: TempoVariationType,

    /// Variation amount (percentage)
    pub amount: f32,

    /// Duration (seconds)
    pub duration: f32,

    /// Context conditions
    pub context_conditions: Vec<String>,
}

/// Tempo variation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TempoVariationType {
    /// Gradual acceleration
    Accelerando,

    /// Gradual deceleration
    Ritardando,

    /// Sudden speed change
    Sudden,

    /// Cyclical variation
    Cyclical,
}

/// Acceleration pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerationPattern {
    /// Pattern name
    pub name: String,

    /// Acceleration curve
    pub curve: Vec<(f32, f32)>, // (time, acceleration)

    /// Pattern frequency
    pub frequency: f32,
}

/// Rubato characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RubatoCharacteristics {
    /// Rubato strength
    pub strength: f32,

    /// Rubato patterns
    pub patterns: Vec<RubatoPattern>,

    /// Musical context sensitivity
    pub context_sensitivity: f32,
}

/// Rubato pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RubatoPattern {
    /// Pattern name
    pub name: String,

    /// Timing adjustments
    pub timing_adjustments: Vec<f32>,

    /// Pattern scope
    pub scope: RubatoScope,
}

/// Rubato scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RubatoScope {
    /// Note-level rubato
    Note,

    /// Phrase-level rubato
    Phrase,

    /// Sentence-level rubato
    Sentence,

    /// Global rubato
    Global,
}

/// Stress characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressCharacteristics {
    /// Stress patterns
    pub stress_patterns: Vec<StressPattern>,

    /// Stress marking methods
    pub stress_marking: Vec<StressMarkingMethod>,

    /// Stress hierarchy
    pub stress_hierarchy: StressHierarchy,
}

/// Stress pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressPattern {
    /// Pattern name
    pub name: String,

    /// Stress levels
    pub stress_levels: Vec<StressLevel>,

    /// Pattern regularity
    pub regularity: f32,

    /// Context dependency
    pub context_dependency: f32,
}

/// Stress level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StressLevel {
    /// No stress
    Unstressed,

    /// Secondary stress
    Secondary,

    /// Primary stress
    Primary,

    /// Emphatic stress
    Emphatic,
}

/// Stress marking method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StressMarkingMethod {
    /// Pitch-based stress
    Pitch,

    /// Duration-based stress
    Duration,

    /// Intensity-based stress
    Intensity,

    /// Combined marking
    Combined,
}

/// Stress hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressHierarchy {
    /// Hierarchical levels
    pub levels: Vec<StressHierarchyLevel>,

    /// Interaction patterns
    pub interaction_patterns: Vec<StressInteractionPattern>,
}

/// Stress hierarchy level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressHierarchyLevel {
    /// Level name
    pub name: String,

    /// Level importance
    pub importance: f32,

    /// Acoustic correlates
    pub acoustic_correlates: Vec<AcousticCorrelate>,
}

/// Acoustic correlate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticCorrelate {
    /// Correlate type
    pub correlate_type: AcousticCorrelateType,

    /// Correlate strength
    pub strength: f32,

    /// Correlate direction
    pub direction: CorrelateDirection,
}

/// Acoustic correlate type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcousticCorrelateType {
    /// Fundamental frequency
    F0,

    /// Duration
    Duration,

    /// Intensity
    Intensity,

    /// Formant frequency
    Formant,

    /// Spectral tilt
    SpectralTilt,
}

/// Correlate direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrelateDirection {
    /// Positive correlation
    Positive,

    /// Negative correlation
    Negative,

    /// Non-linear correlation
    NonLinear,
}

/// Stress interaction pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressInteractionPattern {
    /// Pattern name
    pub name: String,

    /// Interacting levels
    pub levels: Vec<String>,

    /// Interaction type
    pub interaction_type: InteractionType,

    /// Interaction strength
    pub strength: f32,
}

/// Interaction type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    /// Additive interaction
    Additive,

    /// Multiplicative interaction
    Multiplicative,

    /// Competitive interaction
    Competitive,

    /// Cooperative interaction
    Cooperative,
}

/// Intonation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationPattern {
    /// Pattern name
    pub name: String,

    /// Pattern type
    pub pattern_type: IntonationPatternType,

    /// F0 contour
    pub f0_contour: Vec<(f32, f32)>, // (time, F0)

    /// Pattern frequency
    pub frequency: f32,

    /// Context conditions
    pub context_conditions: Vec<String>,

    /// Communicative function
    pub communicative_function: CommunicativeFunction,
}

/// Intonation pattern type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntonationPatternType {
    /// Declarative pattern
    Declarative,

    /// Interrogative pattern
    Interrogative,

    /// Exclamatory pattern
    Exclamatory,

    /// Imperative pattern
    Imperative,

    /// Continuation pattern
    Continuation,

    /// Final pattern
    Final,
}

/// Communicative function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunicativeFunction {
    /// Statement function
    Statement,

    /// Question function
    Question,

    /// Command function
    Command,

    /// Emphasis function
    Emphasis,

    /// Contrast function
    Contrast,

    /// Focus function
    Focus,
}

/// Articulation characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulationCharacteristics {
    /// Consonant articulation
    pub consonant_articulation: ConsonantArticulation,

    /// Vowel articulation
    pub vowel_articulation: VowelArticulation,

    /// Coarticulation patterns
    pub coarticulation_patterns: Vec<CoarticulationPattern>,

    /// Articulatory precision
    pub articulatory_precision: ArticulatoryPrecision,
}

/// Consonant articulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsonantArticulation {
    /// Place of articulation preferences
    pub place_preferences: HashMap<String, f32>,

    /// Manner of articulation preferences
    pub manner_preferences: HashMap<String, f32>,

    /// Voicing characteristics
    pub voicing_characteristics: VoicingCharacteristics,

    /// Consonant cluster handling
    pub cluster_handling: ConsonantClusterHandling,
}

/// Voicing characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicingCharacteristics {
    /// Voice onset time patterns
    pub vot_patterns: HashMap<String, VOTPattern>,

    /// Voicing assimilation patterns
    pub assimilation_patterns: Vec<VoicingAssimilationPattern>,

    /// Devoicing patterns
    pub devoicing_patterns: Vec<DevoicingPattern>,
}

/// Voice onset time pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VOTPattern {
    /// Mean VOT (ms)
    pub mean_vot: f32,

    /// VOT variability
    pub variability: f32,

    /// Context dependencies
    pub context_dependencies: Vec<VOTContext>,
}

/// VOT context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VOTContext {
    /// Context description
    pub context: String,

    /// VOT adjustment (ms)
    pub adjustment: f32,
}

/// Voicing assimilation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicingAssimilationPattern {
    /// Pattern name
    pub name: String,

    /// Source voicing
    pub source_voicing: bool,

    /// Target voicing
    pub target_voicing: bool,

    /// Assimilation strength
    pub strength: f32,

    /// Context conditions
    pub conditions: Vec<String>,
}

/// Devoicing pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevoicingPattern {
    /// Pattern name
    pub name: String,

    /// Affected phonemes
    pub affected_phonemes: Vec<String>,

    /// Devoicing strength
    pub strength: f32,

    /// Context conditions
    pub conditions: Vec<String>,
}

/// Consonant cluster handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsonantClusterHandling {
    /// Cluster simplification patterns
    pub simplification_patterns: Vec<ClusterSimplificationPattern>,

    /// Epenthesis patterns
    pub epenthesis_patterns: Vec<EpenthesisPattern>,

    /// Deletion patterns
    pub deletion_patterns: Vec<DeletionPattern>,
}

/// Cluster simplification pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSimplificationPattern {
    /// Pattern name
    pub name: String,

    /// Input cluster
    pub input_cluster: Vec<String>,

    /// Output cluster
    pub output_cluster: Vec<String>,

    /// Simplification probability
    pub probability: f32,

    /// Context conditions
    pub conditions: Vec<String>,
}

/// Epenthesis pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpenthesisPattern {
    /// Pattern name
    pub name: String,

    /// Epenthetic segment
    pub epenthetic_segment: String,

    /// Insertion position
    pub position: InsertionPosition,

    /// Insertion probability
    pub probability: f32,

    /// Context conditions
    pub conditions: Vec<String>,
}

/// Insertion position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsertionPosition {
    /// Before cluster
    Before,

    /// Within cluster
    Within,

    /// After cluster
    After,
}

/// Deletion pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionPattern {
    /// Pattern name
    pub name: String,

    /// Deleted segment
    pub deleted_segment: String,

    /// Deletion position
    pub position: DeletionPosition,

    /// Deletion probability
    pub probability: f32,

    /// Context conditions
    pub conditions: Vec<String>,
}

/// Deletion position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeletionPosition {
    /// Initial position
    Initial,

    /// Medial position
    Medial,

    /// Final position
    Final,
}

/// Vowel articulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VowelArticulation {
    /// Vowel space characteristics
    pub vowel_space: VowelSpaceCharacteristics,

    /// Vowel reduction patterns
    pub reduction_patterns: Vec<VowelReductionPattern>,

    /// Vowel harmony patterns
    pub harmony_patterns: Vec<VowelHarmonyPattern>,

    /// Diphthongization patterns
    pub diphthongization_patterns: Vec<DiphthongizationPattern>,
}

/// Vowel space characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VowelSpaceCharacteristics {
    /// Formant space mapping
    pub formant_space: HashMap<String, FormantValues>,

    /// Vowel dispersion
    pub dispersion: f32,

    /// Vowel centralization tendency
    pub centralization_tendency: f32,

    /// Dynamic range
    pub dynamic_range: f32,
}

/// Formant values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantValues {
    /// F1 frequency (Hz)
    pub f1: f32,

    /// F2 frequency (Hz)
    pub f2: f32,

    /// F3 frequency (Hz)
    pub f3: f32,

    /// F4 frequency (Hz)
    pub f4: Option<f32>,
}

/// Vowel reduction pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VowelReductionPattern {
    /// Pattern name
    pub name: String,

    /// Source vowel
    pub source_vowel: String,

    /// Target vowel
    pub target_vowel: String,

    /// Reduction probability
    pub probability: f32,

    /// Context conditions
    pub conditions: Vec<String>,
}

/// Vowel harmony pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VowelHarmonyPattern {
    /// Pattern name
    pub name: String,

    /// Harmony type
    pub harmony_type: VowelHarmonyType,

    /// Feature spreading
    pub feature_spreading: FeatureSpreading,

    /// Harmony strength
    pub strength: f32,
}

/// Vowel harmony type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VowelHarmonyType {
    /// Front-back harmony
    FrontBack,

    /// High-low harmony
    HighLow,

    /// Round-unround harmony
    RoundUnround,

    /// Advanced tongue root harmony
    ATR,
}

/// Feature spreading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSpreading {
    /// Spreading direction
    pub direction: SpreadingDirection,

    /// Spreading distance
    pub distance: usize,

    /// Blocking segments
    pub blocking_segments: Vec<String>,
}

/// Spreading direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpreadingDirection {
    /// Left-to-right spreading
    LeftToRight,

    /// Right-to-left spreading
    RightToLeft,

    /// Bidirectional spreading
    Bidirectional,
}

/// Diphthongization pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiphthongizationPattern {
    /// Pattern name
    pub name: String,

    /// Source monophthong
    pub source_monophthong: String,

    /// Target diphthong
    pub target_diphthong: String,

    /// Diphthongization probability
    pub probability: f32,

    /// Context conditions
    pub conditions: Vec<String>,
}

/// Coarticulation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoarticulationPattern {
    /// Pattern name
    pub name: String,

    /// Coarticulation type
    pub coarticulation_type: CoarticulationType,

    /// Affected segments
    pub affected_segments: Vec<String>,

    /// Coarticulation strength
    pub strength: f32,

    /// Temporal extent
    pub temporal_extent: TemporalExtent,
}

/// Coarticulation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoarticulationType {
    /// Anticipatory coarticulation
    Anticipatory,

    /// Carryover coarticulation
    Carryover,

    /// Bidirectional coarticulation
    Bidirectional,
}

/// Temporal extent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalExtent {
    /// Extent in milliseconds
    pub extent_ms: f32,

    /// Extent in segments
    pub extent_segments: usize,

    /// Extent variability
    pub variability: f32,
}

/// Articulatory precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulatoryPrecision {
    /// Overall precision score
    pub overall_precision: f32,

    /// Consonant precision
    pub consonant_precision: f32,

    /// Vowel precision
    pub vowel_precision: f32,

    /// Precision variability
    pub precision_variability: f32,

    /// Context effects on precision
    pub context_effects: Vec<PrecisionContextEffect>,
}

/// Precision context effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionContextEffect {
    /// Context description
    pub context: String,

    /// Precision adjustment
    pub adjustment: f32,

    /// Effect strength
    pub strength: f32,
}

/// Voice quality characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityCharacteristics {
    /// Phonation type
    pub phonation_type: PhonationType,

    /// Breathiness characteristics
    pub breathiness: BreathinessCharacteristics,

    /// Roughness characteristics
    pub roughness: RoughnessCharacteristics,

    /// Creakiness characteristics
    pub creakiness: CreakynessCharacteristics,

    /// Tenseness characteristics
    pub tenseness: TensenessCharacteristics,

    /// Resonance characteristics
    pub resonance: ResonanceCharacteristics,
}

/// Phonation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhonationType {
    /// Modal phonation
    Modal,

    /// Breathy phonation
    Breathy,

    /// Creaky phonation
    Creaky,

    /// Harsh phonation
    Harsh,

    /// Falsetto phonation
    Falsetto,

    /// Mixed phonation
    Mixed,
}

/// Breathiness characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathinessCharacteristics {
    /// Breathiness level (0.0 to 1.0)
    pub level: f32,

    /// Breathiness variability
    pub variability: f32,

    /// Context dependencies
    pub context_dependencies: Vec<BreathinessContext>,

    /// Acoustic correlates
    pub acoustic_correlates: BreathinessAcousticCorrelates,
}

/// Breathiness context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathinessContext {
    /// Context description
    pub context: String,

    /// Breathiness adjustment
    pub adjustment: f32,
}

/// Breathiness acoustic correlates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathinessAcousticCorrelates {
    /// Harmonics-to-noise ratio
    pub hnr: f32,

    /// Spectral tilt
    pub spectral_tilt: f32,

    /// First formant bandwidth
    pub f1_bandwidth: f32,

    /// Aspiration noise level
    pub aspiration_noise: f32,
}

/// Roughness characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoughnessCharacteristics {
    /// Roughness level (0.0 to 1.0)
    pub level: f32,

    /// Roughness variability
    pub variability: f32,

    /// Roughness type
    pub roughness_type: RoughnessType,

    /// Acoustic correlates
    pub acoustic_correlates: RoughnessAcousticCorrelates,
}

/// Roughness type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoughnessType {
    /// Periodic roughness
    Periodic,

    /// Aperiodic roughness
    Aperiodic,

    /// Mixed roughness
    Mixed,
}

/// Roughness acoustic correlates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoughnessAcousticCorrelates {
    /// Jitter
    pub jitter: f32,

    /// Shimmer
    pub shimmer: f32,

    /// Noise-to-harmonics ratio
    pub nhr: f32,

    /// Fundamental frequency irregularity
    pub f0_irregularity: f32,
}

/// Creakiness characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreakynessCharacteristics {
    /// Creakiness level (0.0 to 1.0)
    pub level: f32,

    /// Creakiness variability
    pub variability: f32,

    /// Creak distribution
    pub distribution: CreakDistribution,

    /// Acoustic correlates
    pub acoustic_correlates: CreakyAcousticCorrelates,
}

/// Creak distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreakDistribution {
    /// Phrase-initial creak
    pub phrase_initial: f32,

    /// Phrase-final creak
    pub phrase_final: f32,

    /// Stressed syllable creak
    pub stressed_syllable: f32,

    /// Vowel-specific creak
    pub vowel_specific: HashMap<String, f32>,
}

/// Creaky acoustic correlates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreakyAcousticCorrelates {
    /// Fundamental frequency
    pub f0_characteristics: CreakyF0Characteristics,

    /// Spectral characteristics
    pub spectral_characteristics: CreakySpectralCharacteristics,

    /// Temporal characteristics
    pub temporal_characteristics: CreakyTemporalCharacteristics,
}

/// Creaky F0 characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreakyF0Characteristics {
    /// Mean F0 in creak (Hz)
    pub mean_f0: f32,

    /// F0 irregularity
    pub f0_irregularity: f32,

    /// Subharmonics presence
    pub subharmonics: f32,
}

/// Creaky spectral characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreakySpectralCharacteristics {
    /// Spectral tilt
    pub spectral_tilt: f32,

    /// High-frequency energy
    pub high_frequency_energy: f32,

    /// Formant damping
    pub formant_damping: f32,
}

/// Creaky temporal characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreakyTemporalCharacteristics {
    /// Pulse irregularity
    pub pulse_irregularity: f32,

    /// Inter-pulse intervals
    pub inter_pulse_intervals: Vec<f32>,

    /// Creak duration patterns
    pub duration_patterns: Vec<f32>,
}

/// Tenseness characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensenessCharacteristics {
    /// Tenseness level (0.0 to 1.0)
    pub level: f32,

    /// Tenseness variability
    pub variability: f32,

    /// Tenseness distribution
    pub distribution: TensenessDistribution,

    /// Acoustic correlates
    pub acoustic_correlates: TensenessAcousticCorrelates,
}

/// Tenseness distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensenessDistribution {
    /// Context-dependent tenseness
    pub context_tenseness: HashMap<String, f32>,

    /// Emotion-dependent tenseness
    pub emotion_tenseness: HashMap<String, f32>,

    /// Stress-dependent tenseness
    pub stress_tenseness: HashMap<String, f32>,
}

/// Tenseness acoustic correlates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensenessAcousticCorrelates {
    /// Fundamental frequency elevation
    pub f0_elevation: f32,

    /// Formant frequency shifts
    pub formant_shifts: HashMap<String, f32>,

    /// Spectral energy distribution
    pub spectral_energy: f32,

    /// Voice source characteristics
    pub voice_source: VoiceSourceCharacteristics,
}

/// Voice source characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSourceCharacteristics {
    /// Open quotient
    pub open_quotient: f32,

    /// Closing quotient
    pub closing_quotient: f32,

    /// Spectral tilt
    pub spectral_tilt: f32,

    /// Glottal flow derivative
    pub flow_derivative: f32,
}

/// Resonance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceCharacteristics {
    /// Vocal tract length
    pub vocal_tract_length: f32,

    /// Formant frequencies
    pub formant_frequencies: HashMap<String, f32>,

    /// Formant bandwidths
    pub formant_bandwidths: HashMap<String, f32>,

    /// Resonance coupling
    pub resonance_coupling: ResonanceCoupling,

    /// Nasality characteristics
    pub nasality: NasalityCharacteristics,
}

/// Resonance coupling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceCoupling {
    /// Oral-nasal coupling
    pub oral_nasal_coupling: f32,

    /// Pharyngeal coupling
    pub pharyngeal_coupling: f32,

    /// Coupling variability
    pub coupling_variability: f32,
}

/// Nasality characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NasalityCharacteristics {
    /// Nasality level (0.0 to 1.0)
    pub level: f32,

    /// Nasality variability
    pub variability: f32,

    /// Nasality distribution
    pub distribution: NasalityDistribution,

    /// Acoustic correlates
    pub acoustic_correlates: NasalityAcousticCorrelates,
}

/// Nasality distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NasalityDistribution {
    /// Consonant nasality
    pub consonant_nasality: HashMap<String, f32>,

    /// Vowel nasality
    pub vowel_nasality: HashMap<String, f32>,

    /// Context effects
    pub context_effects: Vec<NasalityContextEffect>,
}

/// Nasality context effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NasalityContextEffect {
    /// Context description
    pub context: String,

    /// Nasality adjustment
    pub adjustment: f32,
}

/// Nasality acoustic correlates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NasalityAcousticCorrelates {
    /// Nasal formant frequencies
    pub nasal_formants: Vec<f32>,

    /// Anti-formant frequencies
    pub anti_formants: Vec<f32>,

    /// Nasal coupling bandwidth
    pub coupling_bandwidth: f32,

    /// Spectral zeros
    pub spectral_zeros: Vec<f32>,
}

/// Cultural characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalCharacteristics {
    /// Regional dialect features
    pub regional_features: Vec<RegionalFeature>,

    /// Sociolinguistic markers
    pub sociolinguistic_markers: Vec<SociolinguisticMarker>,

    /// Cultural speaking norms
    pub speaking_norms: SpeakingNorms,

    /// Code-switching patterns
    pub code_switching: CodeSwitchingPatterns,
}

/// Regional feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalFeature {
    /// Feature name
    pub name: String,

    /// Feature type
    pub feature_type: RegionalFeatureType,

    /// Feature strength
    pub strength: f32,

    /// Regional distribution
    pub distribution: Vec<String>,
}

/// Regional feature type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionalFeatureType {
    /// Phonological feature
    Phonological,

    /// Lexical feature
    Lexical,

    /// Prosodic feature
    Prosodic,

    /// Pragmatic feature
    Pragmatic,
}

/// Sociolinguistic marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SociolinguisticMarker {
    /// Marker name
    pub name: String,

    /// Social dimension
    pub social_dimension: SocialDimension,

    /// Marker salience
    pub salience: f32,

    /// Usage contexts
    pub contexts: Vec<String>,
}

/// Social dimension
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SocialDimension {
    /// Age-related variation
    Age,

    /// Gender-related variation
    Gender,

    /// Class-related variation
    Class,

    /// Ethnicity-related variation
    Ethnicity,

    /// Education-related variation
    Education,

    /// Occupation-related variation
    Occupation,
}

/// Speaking norms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakingNorms {
    /// Turn-taking patterns
    pub turn_taking: TurnTakingPatterns,

    /// Politeness strategies
    pub politeness_strategies: Vec<PolitenessStrategy>,

    /// Discourse markers
    pub discourse_markers: Vec<DiscourseMarker>,

    /// Cultural taboos
    pub cultural_taboos: Vec<CulturalTaboo>,
}

/// Turn-taking patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnTakingPatterns {
    /// Overlap tolerance
    pub overlap_tolerance: f32,

    /// Pause expectations
    pub pause_expectations: Vec<PauseExpectation>,

    /// Interruption patterns
    pub interruption_patterns: Vec<InterruptionPattern>,
}

/// Pause expectation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauseExpectation {
    /// Context
    pub context: String,

    /// Expected pause duration (ms)
    pub expected_duration: f32,

    /// Tolerance range (ms)
    pub tolerance: f32,
}

/// Interruption pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptionPattern {
    /// Pattern name
    pub name: String,

    /// Interruption type
    pub interruption_type: InterruptionType,

    /// Acceptability
    pub acceptability: f32,

    /// Context conditions
    pub conditions: Vec<String>,
}

/// Interruption type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterruptionType {
    /// Cooperative interruption
    Cooperative,

    /// Competitive interruption
    Competitive,

    /// Supportive interruption
    Supportive,

    /// Corrective interruption
    Corrective,
}

/// Politeness strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolitenessStrategy {
    /// Strategy name
    pub name: String,

    /// Politeness type
    pub politeness_type: PolitenessType,

    /// Usage frequency
    pub frequency: f32,

    /// Context appropriateness
    pub appropriateness: HashMap<String, f32>,
}

/// Politeness type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolitenessType {
    /// Positive politeness
    Positive,

    /// Negative politeness
    Negative,

    /// Bald on-record
    BaldOnRecord,

    /// Off-record
    OffRecord,
}

/// Discourse marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseMarker {
    /// Marker text
    pub text: String,

    /// Discourse function
    pub function: DiscourseFunction,

    /// Usage frequency
    pub frequency: f32,

    /// Prosodic characteristics
    pub prosody: DiscourseMarkerProsody,
}

/// Discourse function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscourseFunction {
    /// Topic shift
    TopicShift,

    /// Emphasis
    Emphasis,

    /// Hesitation
    Hesitation,

    /// Confirmation
    Confirmation,

    /// Elaboration
    Elaboration,

    /// Contrast
    Contrast,
}

/// Discourse marker prosody
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseMarkerProsody {
    /// Typical F0 pattern
    pub f0_pattern: Vec<f32>,

    /// Duration characteristics
    pub duration: f32,

    /// Intensity characteristics
    pub intensity: f32,

    /// Pause patterns
    pub pause_patterns: Vec<f32>,
}

/// Cultural taboo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalTaboo {
    /// Taboo description
    pub description: String,

    /// Taboo strength
    pub strength: f32,

    /// Context specificity
    pub context_specificity: Vec<String>,

    /// Violation consequences
    pub consequences: Vec<String>,
}

/// Code-switching patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSwitchingPatterns {
    /// Languages involved
    pub languages: Vec<String>,

    /// Switching triggers
    pub triggers: Vec<SwitchingTrigger>,

    /// Switching points
    pub switching_points: Vec<SwitchingPoint>,

    /// Switching strategies
    pub strategies: Vec<SwitchingStrategy>,
}

/// Switching trigger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchingTrigger {
    /// Trigger type
    pub trigger_type: TriggerType,

    /// Trigger strength
    pub strength: f32,

    /// Context conditions
    pub conditions: Vec<String>,
}

/// Trigger type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerType {
    /// Topic change
    TopicChange,

    /// Emotional state
    EmotionalState,

    /// Audience change
    AudienceChange,

    /// Emphasis
    Emphasis,

    /// Quotation
    Quotation,
}

/// Switching point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchingPoint {
    /// Linguistic level
    pub level: LinguisticLevel,

    /// Switching frequency
    pub frequency: f32,

    /// Constraints
    pub constraints: Vec<SwitchingConstraint>,
}

/// Linguistic level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinguisticLevel {
    /// Phoneme level
    Phoneme,

    /// Morpheme level
    Morpheme,

    /// Word level
    Word,

    /// Phrase level
    Phrase,

    /// Clause level
    Clause,

    /// Sentence level
    Sentence,
}

/// Switching constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchingConstraint {
    /// Constraint name
    pub name: String,

    /// Constraint type
    pub constraint_type: ConstraintType,

    /// Constraint strength
    pub strength: f32,
}

/// Constraint type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Syntactic constraint
    Syntactic,

    /// Phonological constraint
    Phonological,

    /// Semantic constraint
    Semantic,

    /// Pragmatic constraint
    Pragmatic,
}

/// Switching strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchingStrategy {
    /// Strategy name
    pub name: String,

    /// Strategy type
    pub strategy_type: StrategyType,

    /// Usage frequency
    pub frequency: f32,

    /// Effectiveness
    pub effectiveness: f32,
}

/// Strategy type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyType {
    /// Insertion strategy
    Insertion,

    /// Alternation strategy
    Alternation,

    /// Congruent lexicalization
    CongruentLexicalization,

    /// Flagged switching
    FlaggedSwitching,
}
