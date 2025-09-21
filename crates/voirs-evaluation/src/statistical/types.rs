//! Common types used across statistical analysis modules

use serde::{Deserialize, Serialize};

/// Result of a statistical test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value of the test
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: Option<usize>,
    /// Test type identifier
    pub test_type: String,
    /// Whether the result is statistically significant
    pub is_significant: bool,
    /// Significance level used
    pub alpha: f64,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
    /// Confidence interval (lower, upper)
    pub confidence_interval: Option<(f64, f64)>,
    /// Human-readable interpretation of the test result
    pub interpretation: String,
    /// Confidence level used
    pub confidence_level: f64,
}

/// Descriptive statistics for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    /// Sample mean
    pub mean: f64,
    /// Sample variance
    pub variance: f64,
    /// Sample standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Sample size
    pub n: usize,
    /// Median value
    pub median: f64,
    /// First quartile (25th percentile)
    pub q1: f64,
    /// Third quartile (75th percentile)
    pub q3: f64,
    /// Skewness measure
    pub skewness: f64,
    /// Kurtosis measure
    pub kurtosis: f64,
}

/// Statistical distribution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionParams {
    /// Distribution type
    pub distribution_type: DistributionType,
    /// Distribution parameters
    pub parameters: Vec<f64>,
    /// Goodness of fit measures
    pub goodness_of_fit: Option<GoodnessOfFit>,
}

/// Supported statistical distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    /// Normal distribution (mean, std_dev)
    Normal,
    /// Student's t-distribution (degrees_of_freedom)
    StudentT,
    /// Chi-squared distribution (degrees_of_freedom)
    ChiSquared,
    /// F-distribution (df1, df2)
    F,
    /// Beta distribution (alpha, beta)
    Beta,
    /// Gamma distribution (shape, scale)
    Gamma,
    /// Uniform distribution (min, max)
    Uniform,
    /// Exponential distribution (rate)
    Exponential,
}

/// Goodness of fit measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoodnessOfFit {
    /// Kolmogorov-Smirnov test statistic
    pub ks_statistic: f64,
    /// Kolmogorov-Smirnov p-value
    pub ks_p_value: f64,
    /// Anderson-Darling test statistic
    pub ad_statistic: f64,
    /// Log-likelihood value
    pub log_likelihood: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
}

/// Hypothesis test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestConfig {
    /// Significance level
    pub alpha: f64,
    /// Test type
    pub test_type: TestType,
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Whether to apply continuity correction
    pub continuity_correction: bool,
}

/// Types of statistical tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestType {
    /// One-sample t-test
    OneSampleT,
    /// Two-sample t-test
    TwoSampleT,
    /// Paired t-test
    PairedT,
    /// Welch's t-test (unequal variances)
    WelchT,
    /// Chi-squared test
    ChiSquared,
    /// Fisher's exact test
    FisherExact,
    /// Mann-Whitney U test
    MannWhitneyU,
    /// Wilcoxon signed-rank test
    WilcoxonSignedRank,
    /// Kruskal-Wallis test
    KruskalWallis,
    /// ANOVA
    Anova,
}

/// Alternative hypothesis types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Alternative {
    /// Two-sided test
    TwoSided,
    /// Less than (left-tailed)
    Less,
    /// Greater than (right-tailed)
    Greater,
}

/// Multiple comparison correction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultipleComparisonCorrection {
    /// No correction
    None,
    /// Bonferroni correction
    Bonferroni,
    /// Benjamini-Hochberg (FDR)
    BenjaminiHochberg,
    /// Holm-Bonferroni method
    HolmBonferroni,
    /// Sidak correction
    Sidak,
}

/// Bootstrap configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Confidence level
    pub confidence_level: f64,
    /// Bootstrap method
    pub method: BootstrapMethod,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

/// Bootstrap methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BootstrapMethod {
    /// Standard percentile bootstrap
    Percentile,
    /// Bias-corrected and accelerated (BCa)
    BCA,
    /// Student bootstrap
    Student,
}

/// Regression analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Regression coefficients
    pub coefficients: Vec<f64>,
    /// R-squared value
    pub r_squared: f64,
    /// Adjusted R-squared value
    pub adjusted_r_squared: f64,
    /// Standard errors of coefficients
    pub standard_errors: Vec<f64>,
    /// T-statistics for coefficients
    pub t_statistics: Vec<f64>,
    /// P-values for coefficients
    pub p_values: Vec<f64>,
    /// Residual standard error
    pub residual_standard_error: f64,
    /// F-statistic
    pub f_statistic: f64,
    /// P-value for F-statistic
    pub f_p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: (usize, usize),
    /// Slope coefficient (for simple linear regression)
    pub slope: f64,
    /// Intercept coefficient (for simple linear regression)
    pub intercept: f64,
    /// Overall p-value for the regression
    pub p_value: f64,
    /// Standard error
    pub standard_error: f64,
}

/// A/B testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    /// Sample size for group A
    pub sample_size_a: usize,
    /// Sample size for group B
    pub sample_size_b: usize,
    /// Expected effect size
    pub effect_size: f64,
    /// Significance level (alpha)
    pub alpha: f64,
    /// Statistical power (1 - beta)
    pub power: f64,
    /// Whether to use two-tailed test
    pub two_tailed: bool,
    /// Expected effect size (alias for effect_size)
    pub expected_effect_size: f64,
    /// Minimum detectable difference
    pub minimum_detectable_difference: f64,
}

/// Power analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysisResult {
    /// Achieved statistical power
    pub achieved_power: f64,
    /// Effect size used in calculation
    pub effect_size: f64,
    /// Sample size used in calculation
    pub sample_size: usize,
    /// Significance level (alpha)
    pub alpha: f64,
}

/// Multiple comparison correction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleComparisonResult {
    /// Adjusted p-values after correction
    pub adjusted_p_values: Vec<f32>,
    /// Original p-values
    pub original_p_values: Vec<f32>,
    /// Correction method used
    pub method: MultipleComparisonCorrection,
}
