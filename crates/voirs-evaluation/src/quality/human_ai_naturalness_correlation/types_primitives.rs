//! Primitive and leaf types for human-AI naturalness correlation
//!
//! This module contains simple enums, leaf structs, and low-level types
//! that have no cross-dependencies on other custom types in this crate.

use crate::integration::{RecommendationPriority, RecommendationType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Experience level with audio/speech
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperienceLevel {
    /// Beginner
    Beginner,
    /// Intermediate
    Intermediate,
    /// Advanced
    Advanced,
    /// Expert
    Expert,
}

/// Calibration curve type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationCurveType {
    /// Linear calibration
    Linear,
    /// Quadratic calibration
    Quadratic,
    /// Sigmoid calibration
    Sigmoid,
    /// Piecewise linear calibration
    PiecewiseLinear,
}

/// Temporal pattern type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalPatternType {
    /// Increasing correlation over time
    Increasing,
    /// Decreasing correlation over time
    Decreasing,
    /// Cyclical pattern
    Cyclical,
    /// Stable pattern
    Stable,
    /// Irregular pattern
    Irregular,
}

/// Gender categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Gender {
    /// Male
    Male,
    /// Female
    Female,
    /// Non-binary
    NonBinary,
    /// Prefer not to say
    Other,
}

/// Age group categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgeGroup {
    /// 18-25 years
    Young,
    /// 26-40 years
    MiddleAged,
    /// 41-60 years
    Mature,
    /// 60+ years
    Senior,
}

/// Drift direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDirection {
    /// Positive drift
    Positive,
    /// Negative drift
    Negative,
    /// No significant drift
    NoSignificantDrift,
    /// Oscillating drift
    Oscillating,
}

/// Cultural background
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CulturalBackground {
    /// Western
    Western,
    /// East Asian
    EastAsian,
    /// South Asian
    SouthAsian,
    /// Middle Eastern
    MiddleEastern,
    /// African
    African,
    /// Latin American
    LatinAmerican,
    /// Nordic
    Nordic,
    /// Mediterranean
    Mediterranean,
    /// Other
    Other,
}

/// Hearing ability level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HearingAbility {
    /// Normal hearing
    Normal,
    /// Mild hearing loss
    MildLoss,
    /// Moderate hearing loss
    ModerateLoss,
    /// Severe hearing loss
    SevereLoss,
    /// Uses hearing aids
    HearingAids,
}

/// Data quality indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityIndicators {
    /// Data completeness
    pub completeness: f32,
    /// Data consistency
    pub consistency: f32,
    /// Data accuracy
    pub accuracy: f32,
    /// Data representativeness
    pub representativeness: f32,
    /// Missing data percentage
    pub missing_data_percentage: f32,
}

/// Model quality indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQualityIndicators {
    /// Model fit quality
    pub fit_quality: f32,
    /// Model robustness
    pub robustness: f32,
    /// Model generalizability
    pub generalizability: f32,
    /// Model interpretability
    pub interpretability: f32,
}

/// Reliability indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityIndicators {
    /// Test-retest reliability
    pub test_retest_reliability: f32,
    /// Internal consistency
    pub internal_consistency: f32,
    /// Inter-rater reliability
    pub inter_rater_reliability: f32,
    /// Predictive validity
    pub predictive_validity: f32,
}

/// Cluster quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterQualityMetrics {
    /// Silhouette score
    pub silhouette_score: f32,
    /// Calinski-Harabasz index
    pub calinski_harabasz_index: f32,
    /// Davies-Bouldin index
    pub davies_bouldin_index: f32,
    /// Within-cluster sum of squares
    pub within_cluster_ss: f32,
    /// Between-cluster sum of squares
    pub between_cluster_ss: f32,
}

/// Dimension reduction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionReductionResults {
    /// Original dimensionality
    pub original_dimensions: usize,
    /// Reduced dimensionality
    pub reduced_dimensions: usize,
    /// Information retention
    pub information_retention: f32,
    /// Reduction quality
    pub reduction_quality: f32,
    /// Optimal dimensions recommendation
    pub optimal_dimensions: usize,
}

/// Factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factor {
    /// Factor index
    pub factor_index: usize,
    /// Factor name
    pub factor_name: String,
    /// Factor interpretation
    pub interpretation: String,
    /// Factor reliability
    pub reliability: f32,
}

/// Principal component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipalComponent {
    /// Component index
    pub component_index: usize,
    /// Explained variance
    pub explained_variance: f32,
    /// Component interpretation
    pub interpretation: String,
}

/// Timed correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedCorrelation {
    /// Time window start
    pub time_start: f32,
    /// Time window end
    pub time_end: f32,
    /// Correlation coefficient
    pub correlation: f32,
    /// Confidence level
    pub confidence: f32,
}

/// F0 naturalness features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0NaturalnessFeatures {
    /// F0 smoothness
    pub f0_smoothness: f32,
    /// F0 variability
    pub f0_variability: f32,
    /// F0 range appropriateness
    pub f0_range_appropriateness: f32,
    /// F0 transition naturalness
    pub f0_transition_naturalness: f32,
    /// F0 outlier detection
    pub f0_outlier_score: f32,
}

/// Formant naturalness features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantNaturalnessFeatures {
    /// Formant trajectory smoothness
    pub formant_smoothness: f32,
    /// Formant frequency appropriateness
    pub formant_frequency_appropriateness: f32,
    /// Formant bandwidth naturalness
    pub formant_bandwidth_naturalness: f32,
    /// Formant transition quality
    pub formant_transition_quality: f32,
}

/// Energy naturalness features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyNaturalnessFeatures {
    /// Energy envelope smoothness
    pub energy_smoothness: f32,
    /// Energy variability
    pub energy_variability: f32,
    /// Energy distribution naturalness
    pub energy_distribution_naturalness: f32,
    /// Energy transition quality
    pub energy_transition_quality: f32,
}

/// Rhythm naturalness features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmNaturalnessFeatures {
    /// Rhythm regularity
    pub rhythm_regularity: f32,
    /// Stress pattern naturalness
    pub stress_pattern_naturalness: f32,
    /// Syllable timing variability
    pub syllable_timing_variability: f32,
    /// Pause pattern appropriateness
    pub pause_pattern_appropriateness: f32,
}

/// Voice quality naturalness features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityNaturalnessFeatures {
    /// Jitter naturalness
    pub jitter_naturalness: f32,
    /// Shimmer naturalness
    pub shimmer_naturalness: f32,
    /// Harmonic-to-noise ratio naturalness
    pub hnr_naturalness: f32,
    /// Breathiness naturalness
    pub breathiness_naturalness: f32,
    /// Roughness naturalness
    pub roughness_naturalness: f32,
}

/// Spectral envelope naturalness features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralEnvelopeNaturalnessFeatures {
    /// Spectral smoothness
    pub spectral_smoothness: f32,
    /// Spectral balance
    pub spectral_balance: f32,
    /// Spectral tilt naturalness
    pub spectral_tilt_naturalness: f32,
    /// Spectral peak naturalness
    pub spectral_peak_naturalness: f32,
}

/// Cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Cluster index
    pub cluster_index: usize,
    /// Cluster centroid
    pub centroid: Vec<f32>,
    /// Cluster size
    pub size: usize,
    /// Cluster characteristics
    pub characteristics: String,
}

/// Calibration curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCurve {
    /// Curve type
    pub curve_type: CalibrationCurveType,
    /// Curve parameters
    pub parameters: Vec<f32>,
    /// Curve fit quality
    pub fit_quality: f32,
}

/// Drift analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAnalysis {
    /// Overall drift magnitude
    pub drift_magnitude: f32,
    /// Drift direction
    pub drift_direction: DriftDirection,
    /// Drift significance
    pub drift_significance: f32,
    /// Drift correction recommendations
    pub drift_recommendations: Vec<String>,
}

/// Temporal pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    /// Pattern type
    pub pattern_type: TemporalPatternType,
    /// Pattern strength
    pub strength: f32,
    /// Pattern duration
    pub duration: f32,
    /// Pattern description
    pub description: String,
}

/// Statistical significance results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignificanceResults {
    /// P-values for each correlation
    pub p_values: HashMap<String, f32>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f32, f32)>,
    /// Sample sizes
    pub sample_sizes: HashMap<String, usize>,
    /// Effect sizes
    pub effect_sizes: HashMap<String, f32>,
    /// Statistical power
    pub statistical_power: HashMap<String, f32>,
}

/// Bias analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasAnalysisResults {
    /// Demographic bias analysis
    pub demographic_bias: HashMap<String, f32>,
    /// Experience bias analysis
    pub experience_bias: HashMap<String, f32>,
    /// Cultural bias analysis
    pub cultural_bias: HashMap<String, f32>,
    /// Systematic bias detection
    pub systematic_bias: f32,
    /// Bias correction factors
    pub bias_correction_factors: HashMap<String, f32>,
    /// Bias-corrected correlations
    pub bias_corrected_correlations: HashMap<String, f32>,
}

/// Principal component analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaAnalysis {
    /// Principal components
    pub principal_components: Vec<PrincipalComponent>,
    /// Explained variance ratios
    pub explained_variance_ratios: Vec<f32>,
    /// Cumulative explained variance
    pub cumulative_variance: Vec<f32>,
    /// Component loadings
    pub component_loadings: HashMap<String, Vec<f32>>,
}

/// Factor analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAnalysis {
    /// Factors
    pub factors: Vec<Factor>,
    /// Factor loadings
    pub factor_loadings: HashMap<String, Vec<f32>>,
    /// Communalities
    pub communalities: HashMap<String, f32>,
    /// Factor correlations
    pub factor_correlations: Vec<Vec<f32>>,
}

/// Cluster analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAnalysis {
    /// Clusters
    pub clusters: Vec<Cluster>,
    /// Cluster quality metrics
    pub cluster_quality: ClusterQualityMetrics,
    /// Cluster assignments
    pub cluster_assignments: Vec<usize>,
}

/// Detailed naturalness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedNaturalnessMetrics {
    /// F0 naturalness features
    pub f0_features: F0NaturalnessFeatures,
    /// Formant naturalness features
    pub formant_features: FormantNaturalnessFeatures,
    /// Energy naturalness features
    pub energy_features: EnergyNaturalnessFeatures,
    /// Rhythm naturalness features
    pub rhythm_features: RhythmNaturalnessFeatures,
    /// Voice quality features
    pub voice_quality_features: VoiceQualityNaturalnessFeatures,
    /// Spectral envelope features
    pub spectral_envelope_features: SpectralEnvelopeNaturalnessFeatures,
}

/// Recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommendation priority
    pub priority: RecommendationPriority,
    /// Recommendation description
    pub description: String,
    /// Expected impact
    pub expected_impact: f32,
    /// Implementation difficulty
    pub implementation_difficulty: f32,
    /// Specific actions
    pub specific_actions: Vec<String>,
}

/// Temporal dynamics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDynamicsAnalysis {
    /// Time-based correlation changes
    pub temporal_correlations: Vec<TimedCorrelation>,
    /// Correlation stability
    pub correlation_stability: f32,
    /// Temporal patterns
    pub temporal_patterns: Vec<TemporalPattern>,
    /// Drift analysis
    pub drift_analysis: DriftAnalysis,
}

/// AI naturalness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiNaturalnessMetrics {
    /// Overall AI naturalness score [0.0, 1.0]
    pub overall_naturalness: f32,
    /// Prosodic naturalness score [0.0, 1.0]
    pub prosodic_naturalness: f32,
    /// Acoustic naturalness score [0.0, 1.0]
    pub acoustic_naturalness: f32,
    /// Temporal naturalness score [0.0, 1.0]
    pub temporal_naturalness: f32,
    /// Spectral naturalness score [0.0, 1.0]
    pub spectral_naturalness: f32,
    /// Confidence in metrics [0.0, 1.0]
    pub confidence: f32,
    /// Feature importance weights
    pub feature_importance: HashMap<String, f32>,
    /// Detailed metrics
    pub detailed_metrics: DetailedNaturalnessMetrics,
}
