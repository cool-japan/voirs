//! Dataset experiment tracking and management

use crate::{DatasetError, Result as DatasetResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Dataset version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetVersion {
    pub version: String,
    pub hash: String,
    pub created_at: DateTime<Utc>,
    pub source: String,
    pub num_samples: usize,
    pub total_duration: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Processing parameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub name: String,
    pub version: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub dependencies: Vec<String>,
    pub environment: HashMap<String, String>,
}

/// Experiment results with detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    pub metrics: HashMap<String, f64>,
    pub artifacts: Vec<PathBuf>,
    pub logs: Vec<String>,
    pub status: ExperimentStatus,
    pub error_message: Option<String>,
    pub execution_time: Duration,
    pub resource_usage: ResourceUsage,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub peak_memory_mb: f64,
    pub cpu_utilization: f64,
    pub disk_io_mb: f64,
    pub network_io_mb: Option<f64>,
}

/// Experiment execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Enhanced experiment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetadata {
    pub id: String,
    pub name: String,
    pub description: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub tags: Vec<String>,
    pub dataset_version: DatasetVersion,
    pub processing_config: ProcessingConfig,
    pub results: Option<ExperimentResults>,
    pub parent_experiment: Option<String>,
    pub reproducibility_seed: Option<u64>,
}

/// Experiment comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub experiment_ids: Vec<String>,
    pub metric_comparisons: HashMap<String, Vec<f64>>,
    pub statistical_significance: HashMap<String, f64>,
    pub ranking: Vec<String>,
    pub summary: String,
}

/// Experiment tracker with advanced functionality
#[derive(Debug, Default)]
pub struct ExperimentTracker {
    experiments: HashMap<String, ExperimentMetadata>,
    storage_path: Option<PathBuf>,
}

impl ExperimentTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_storage<P: AsRef<Path>>(path: P) -> Self {
        Self {
            experiments: HashMap::new(),
            storage_path: Some(path.as_ref().to_path_buf()),
        }
    }

    /// Create new experiment with dataset version tracking
    pub fn create_experiment(
        &mut self,
        name: String,
        description: String,
        dataset_version: DatasetVersion,
        processing_config: ProcessingConfig,
        tags: Vec<String>,
        seed: Option<u64>,
    ) -> DatasetResult<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();

        let experiment = ExperimentMetadata {
            id: id.clone(),
            name,
            description,
            created_at: now,
            updated_at: now,
            tags,
            dataset_version,
            processing_config,
            results: None,
            parent_experiment: None,
            reproducibility_seed: seed,
        };

        self.experiments.insert(id.clone(), experiment);
        self.save_to_disk()?;

        Ok(id)
    }

    /// Update experiment with results
    pub fn update_experiment_results(
        &mut self,
        experiment_id: &str,
        results: ExperimentResults,
    ) -> DatasetResult<()> {
        let experiment = self
            .experiments
            .get_mut(experiment_id)
            .ok_or_else(|| DatasetError::LoadError(format!("Experiment {experiment_id}")))?;

        experiment.results = Some(results);
        experiment.updated_at = Utc::now();

        self.save_to_disk()?;
        Ok(())
    }

    /// Get experiment by ID
    pub fn get_experiment(&self, experiment_id: &str) -> Option<&ExperimentMetadata> {
        self.experiments.get(experiment_id)
    }

    /// Get all experiments
    pub fn get_experiments(&self) -> Vec<&ExperimentMetadata> {
        self.experiments.values().collect()
    }

    /// Find experiments by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<&ExperimentMetadata> {
        self.experiments
            .values()
            .filter(|exp| exp.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Find experiments by dataset version
    pub fn find_by_dataset_version(&self, version: &str) -> Vec<&ExperimentMetadata> {
        self.experiments
            .values()
            .filter(|exp| exp.dataset_version.version == version)
            .collect()
    }

    /// Compare experiments by metrics with statistical analysis
    pub fn compare_experiments(
        &self,
        experiment_ids: &[String],
    ) -> DatasetResult<ComparisonResult> {
        let experiments: Vec<&ExperimentMetadata> = experiment_ids
            .iter()
            .map(|id| self.get_experiment(id))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                DatasetError::LoadError("One or more experiments not found".to_string())
            })?;

        let mut metric_comparisons = HashMap::new();
        let mut all_metrics = std::collections::HashSet::new();

        // Collect all metrics
        for exp in &experiments {
            if let Some(results) = &exp.results {
                for metric in results.metrics.keys() {
                    all_metrics.insert(metric.clone());
                }
            }
        }

        // Compare each metric
        for metric in all_metrics {
            let values: Vec<f64> = experiments
                .iter()
                .filter_map(|exp| exp.results.as_ref()?.metrics.get(&metric).copied())
                .collect();

            if !values.is_empty() {
                metric_comparisons.insert(metric, values);
            }
        }

        // Calculate statistical significance using t-tests
        let mut statistical_significance = HashMap::new();
        for (metric, values) in &metric_comparisons {
            if values.len() >= 2 {
                let p_value = self.calculate_statistical_significance(values);
                statistical_significance.insert(metric.clone(), p_value);
            }
        }

        // Advanced ranking considering multiple metrics with weights
        let ranking = self.rank_experiments_advanced(&experiments, &metric_comparisons);

        let summary = self.generate_comparison_summary(
            &experiments,
            &metric_comparisons,
            &statistical_significance,
        );

        Ok(ComparisonResult {
            experiment_ids: experiment_ids.to_vec(),
            metric_comparisons,
            statistical_significance,
            ranking,
            summary,
        })
    }

    /// Calculate statistical significance using Welch's t-test
    fn calculate_statistical_significance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 1.0; // No significance with less than 2 values
        }

        // Split into two groups for comparison (first half vs second half)
        let mid = values.len() / 2;
        let group1 = &values[..mid];
        let group2 = &values[mid..];

        if group1.is_empty() || group2.is_empty() {
            return 1.0;
        }

        let mean1 = group1.iter().sum::<f64>() / group1.len() as f64;
        let mean2 = group2.iter().sum::<f64>() / group2.len() as f64;

        let var1 = group1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>()
            / (group1.len() - 1).max(1) as f64;
        let var2 = group2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>()
            / (group2.len() - 1).max(1) as f64;

        let n1 = group1.len() as f64;
        let n2 = group2.len() as f64;

        // Squared standard errors of each group mean.
        let se1_sq = var1 / n1;
        let se2_sq = var2 / n2;
        let se = (se1_sq + se2_sq).sqrt();

        if se == 0.0 {
            return if mean1 == mean2 { 1.0 } else { 0.0 };
        }

        let t_stat = (mean1 - mean2) / se;

        // Welch-Satterthwaite approximation for the effective degrees of freedom.
        // df = (s1^2/n1 + s2^2/n2)^2 / [ (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) ]
        let denom =
            (se1_sq * se1_sq) / (n1 - 1.0).max(1.0) + (se2_sq * se2_sq) / (n2 - 1.0).max(1.0);
        let df = if denom > 0.0 {
            (se1_sq + se2_sq).powi(2) / denom
        } else {
            (n1 + n2 - 2.0).max(1.0)
        };

        // Real two-sided Student's t p-value.
        students_t_two_sided_p(t_stat, df).clamp(0.0, 1.0)
    }

    /// Advanced ranking using multiple criteria
    fn rank_experiments_advanced(
        &self,
        experiments: &[&ExperimentMetadata],
        metric_comparisons: &HashMap<String, Vec<f64>>,
    ) -> Vec<String> {
        let mut experiment_scores: Vec<(String, f64)> = Vec::new();

        for exp in experiments.iter() {
            let mut total_score = 0.0;
            let mut metric_count = 0;

            if let Some(results) = &exp.results {
                // Score based on metrics (normalized)
                for (metric, values) in metric_comparisons {
                    if let Some(value) = results.metrics.get(metric) {
                        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

                        if max_val != min_val {
                            let normalized_score = (value - min_val) / (max_val - min_val);
                            total_score += normalized_score;
                            metric_count += 1;
                        }
                    }
                }

                // Bonus for lower execution time
                let execution_penalty = results.execution_time.as_secs_f64() / 3600.0; // Hours
                total_score -= execution_penalty * 0.1;

                // Bonus for successful completion
                if matches!(results.status, ExperimentStatus::Completed) {
                    total_score += 1.0;
                }
            }

            if metric_count > 0 {
                total_score /= metric_count as f64;
            }

            experiment_scores.push((exp.id.clone(), total_score));
        }

        // Sort by score (descending)
        experiment_scores
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        experiment_scores.into_iter().map(|(id, _)| id).collect()
    }

    /// Generate detailed comparison summary
    fn generate_comparison_summary(
        &self,
        experiments: &[&ExperimentMetadata],
        metric_comparisons: &HashMap<String, Vec<f64>>,
        statistical_significance: &HashMap<String, f64>,
    ) -> String {
        let mut summary = Vec::new();

        summary.push("## Experiment Comparison Report".to_string());
        summary.push(format!("**Experiments analyzed**: {}", experiments.len()));
        summary.push(format!(
            "**Metrics compared**: {}",
            metric_comparisons.len()
        ));
        summary.push("".to_string());

        // Best performing experiment by metric
        for (metric, values) in metric_comparisons {
            let max_idx = values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);

            if let Some(idx) = max_idx {
                if let Some(exp) = experiments.get(idx) {
                    let p_value = statistical_significance.get(metric).unwrap_or(&1.0);
                    let significance = if *p_value < 0.05 {
                        "significant"
                    } else {
                        "not significant"
                    };

                    summary.push(format!(
                        "**{}**: Best = {} ({:.4}) [{}]",
                        metric, exp.name, values[idx], significance
                    ));
                }
            }
        }

        summary.push("".to_string());
        summary.push("*Statistical significance threshold: p < 0.05*".to_string());

        summary.join("\n")
    }

    /// Check reproducibility of experiment
    pub fn check_reproducibility(&self, experiment_id: &str) -> DatasetResult<bool> {
        let experiment = self
            .get_experiment(experiment_id)
            .ok_or_else(|| DatasetError::LoadError(format!("Experiment {experiment_id}")))?;

        // Check if experiment has reproducibility seed and deterministic parameters
        Ok(experiment.reproducibility_seed.is_some()
            && experiment
                .processing_config
                .parameters
                .contains_key("deterministic"))
    }

    /// Save experiments to disk
    fn save_to_disk(&self) -> DatasetResult<()> {
        if let Some(path) = &self.storage_path {
            let json = serde_json::to_string_pretty(&self.experiments)
                .map_err(|e| DatasetError::IoError(std::io::Error::other(e)))?;

            std::fs::write(path, json).map_err(DatasetError::IoError)?;
        }
        Ok(())
    }

    /// Load experiments from disk
    pub fn load_from_disk(&mut self) -> DatasetResult<()> {
        if let Some(path) = &self.storage_path {
            if path.exists() {
                let content = std::fs::read_to_string(path).map_err(DatasetError::IoError)?;

                self.experiments = serde_json::from_str(&content)
                    .map_err(|e| DatasetError::IoError(std::io::Error::other(e)))?;
            }
        }
        Ok(())
    }

    /// Get experiment statistics
    pub fn get_statistics(&self) -> ExperimentStatistics {
        let total = self.experiments.len();
        let completed = self
            .experiments
            .values()
            .filter(|exp| {
                matches!(
                    exp.results.as_ref().map(|r| &r.status),
                    Some(ExperimentStatus::Completed)
                )
            })
            .count();
        let failed = self
            .experiments
            .values()
            .filter(|exp| {
                matches!(
                    exp.results.as_ref().map(|r| &r.status),
                    Some(ExperimentStatus::Failed)
                )
            })
            .count();

        let avg_execution_time = if completed > 0 {
            let total_time: std::time::Duration = self
                .experiments
                .values()
                .filter_map(|exp| exp.results.as_ref())
                .filter(|r| matches!(r.status, ExperimentStatus::Completed))
                .map(|r| r.execution_time)
                .sum();

            Some(total_time / completed as u32)
        } else {
            None
        };

        ExperimentStatistics {
            total_experiments: total,
            completed_experiments: completed,
            failed_experiments: failed,
            average_execution_time: avg_execution_time,
        }
    }

    /// Create child experiment based on parent
    pub fn create_child_experiment(
        &mut self,
        parent_id: &str,
        name: String,
        description: String,
        processing_config: ProcessingConfig,
        tags: Vec<String>,
    ) -> DatasetResult<String> {
        let parent = self
            .get_experiment(parent_id)
            .ok_or_else(|| DatasetError::LoadError(format!("Parent experiment {parent_id}")))?;

        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();

        let experiment = ExperimentMetadata {
            id: id.clone(),
            name,
            description,
            created_at: now,
            updated_at: now,
            tags,
            dataset_version: parent.dataset_version.clone(),
            processing_config,
            results: None,
            parent_experiment: Some(parent_id.to_string()),
            reproducibility_seed: parent.reproducibility_seed,
        };

        self.experiments.insert(id.clone(), experiment);
        self.save_to_disk()?;

        Ok(id)
    }

    /// Validate experiment reproducibility by re-running
    pub fn validate_reproducibility(
        &self,
        experiment_id: &str,
    ) -> DatasetResult<ReproducibilityReport> {
        let experiment = self
            .get_experiment(experiment_id)
            .ok_or_else(|| DatasetError::LoadError(format!("Experiment {experiment_id}")))?;

        let has_seed = experiment.reproducibility_seed.is_some();
        let has_deterministic = experiment
            .processing_config
            .parameters
            .contains_key("deterministic");
        let has_version_pinning = !experiment.processing_config.dependencies.is_empty();

        let reproducibility_score = match (has_seed, has_deterministic, has_version_pinning) {
            (true, true, true) => 1.0,
            (true, true, false) | (true, false, true) | (false, true, true) => 0.67,
            (true, false, false) | (false, true, false) | (false, false, true) => 0.33,
            (false, false, false) => 0.0,
        };

        let mut issues = Vec::new();
        if !has_seed {
            issues.push("Missing reproducibility seed".to_string());
        }
        if !has_deterministic {
            issues.push("Deterministic processing not enabled".to_string());
        }
        if !has_version_pinning {
            issues.push("Dependencies not version-pinned".to_string());
        }

        let recommendations = if reproducibility_score < 1.0 {
            vec![
                "Add reproducibility seed to experiment configuration".to_string(),
                "Enable deterministic processing in parameters".to_string(),
                "Pin exact versions of all dependencies".to_string(),
                "Document hardware and environment specifications".to_string(),
            ]
        } else {
            vec!["Experiment meets reproducibility best practices".to_string()]
        };

        Ok(ReproducibilityReport {
            experiment_id: experiment_id.to_string(),
            reproducibility_score,
            has_seed,
            has_deterministic_config: has_deterministic,
            has_version_pinning,
            issues,
            recommendations,
        })
    }

    /// Archive old experiments
    pub fn archive_experiments_older_than(&mut self, days: u32) -> DatasetResult<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(days as i64);
        let to_archive: Vec<String> = self
            .experiments
            .iter()
            .filter(|(_, exp)| exp.created_at < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        let archived_count = to_archive.len();

        // In a real implementation, we'd move these to an archive storage
        for id in to_archive {
            self.experiments.remove(&id);
        }

        self.save_to_disk()?;
        Ok(archived_count)
    }
}

/// Experiment statistics summary
#[derive(Debug, Clone)]
pub struct ExperimentStatistics {
    pub total_experiments: usize,
    pub completed_experiments: usize,
    pub failed_experiments: usize,
    pub average_execution_time: Option<std::time::Duration>,
}

/// Reproducibility report for experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityReport {
    pub experiment_id: String,
    pub reproducibility_score: f64,
    pub has_seed: bool,
    pub has_deterministic_config: bool,
    pub has_version_pinning: bool,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

// ---------------------------------------------------------------------------
// Student's t-distribution helpers
//
// The two-sided p-value of a t-statistic with `df` degrees of freedom is
// derived from the cumulative distribution function (CDF) of Student's
// t-distribution, which has a closed form in terms of the regularized
// incomplete beta function `I_x(a, b)`:
//
//     P(|T| >= |t|) = I_x(df/2, 1/2),   with   x = df / (df + t^2)
//
// (see e.g. Press, Teukolsky, Vetterling & Flannery, "Numerical Recipes",
// 3rd ed., §6.4 "Incomplete Beta Function" and §6.14 "Statistical Functions").
//
// We implement everything in pure Rust with no external special-function
// dependency:
//   * `ln_gamma`  - the natural log of the gamma function via the Lanczos
//                   approximation (Numerical Recipes §6.1).
//   * `betacf`    - the continued-fraction expansion of the incomplete beta
//                   function evaluated with the modified Lentz algorithm.
//   * `betai`     - the regularized incomplete beta function `I_x(a, b)`.
// ---------------------------------------------------------------------------

/// Natural logarithm of the gamma function, `ln(Γ(x))`, for `x > 0`.
///
/// Uses the Lanczos approximation with `g = 5` and 6 coefficients, which is
/// accurate to roughly 15 significant digits over the positive real axis
/// (Numerical Recipes, 3rd ed., §6.1).
fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients (g = 5, n = 6).
    const COEFFS: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];

    let mut y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000_000_000_190_015;
    for c in COEFFS.iter() {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Continued-fraction expansion used by [`betai`], evaluated with the modified
/// Lentz algorithm (Numerical Recipes, 3rd ed., §6.4, function `betacf`).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3.0e-12;
    const FP_MIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FP_MIN {
        d = FP_MIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        // Even step of the recurrence.
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FP_MIN {
            d = FP_MIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FP_MIN {
            c = FP_MIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step of the recurrence.
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FP_MIN {
            d = FP_MIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FP_MIN {
            c = FP_MIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() <= EPS {
            break;
        }
    }

    h
}

/// Regularized incomplete beta function `I_x(a, b)` for `0 <= x <= 1`.
///
/// Implementation follows Numerical Recipes (3rd ed., §6.4, function `betai`),
/// choosing the continued fraction that converges fastest for the given `x`.
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Prefactor `x^a (1-x)^b / B(a, b)`, evaluated in log space for stability.
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        // Use the continued fraction directly.
        bt * betacf(a, b, x) / a
    } else {
        // Use the symmetry relation `I_x(a,b) = 1 - I_{1-x}(b,a)`.
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Two-sided p-value of a Student's t-statistic with `df` degrees of freedom.
///
/// Returns `P(|T| >= |t|)` where `T` follows Student's t-distribution. Computed
/// from the regularized incomplete beta function as
/// `I_x(df/2, 1/2)` with `x = df / (df + t^2)`.
///
/// Degenerate inputs (`df <= 0`, or non-finite `t`) collapse to a p-value of
/// `1.0`, i.e. "no evidence against the null hypothesis".
fn students_t_two_sided_p(t: f64, df: f64) -> f64 {
    if df <= 0.0 || !t.is_finite() {
        return 1.0;
    }
    let x = df / (df + t * t);
    betai(0.5 * df, 0.5, x).clamp(0.0, 1.0)
}

#[cfg(test)]
mod stats_tests {
    use super::*;

    /// Deterministic linear-congruential generator so that the test data is
    /// fixed and reproducible without pulling in any RNG dependency.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            // Numerical Recipes LCG constants.
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Use the top 53 bits for a uniform value in [0, 1).
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    #[test]
    fn t_two_sided_p_matches_reference_table() {
        // Classic t-table critical value: t(0.025, df=10) = 2.228, so the
        // two-sided p-value should be ~0.05.
        let p = students_t_two_sided_p(2.228, 10.0);
        assert!(
            (p - 0.05).abs() < 1.0e-3,
            "expected ~0.05 for t=2.228, df=10; got {p}"
        );

        // Sign symmetry: the two-sided p-value depends only on |t|.
        let p_neg = students_t_two_sided_p(-2.228, 10.0);
        assert!((p - p_neg).abs() < 1.0e-12);
    }

    #[test]
    fn t_two_sided_p_at_zero_is_one() {
        let p = students_t_two_sided_p(0.0, 10.0);
        assert!((p - 1.0).abs() < 1.0e-9, "expected ~1.0 for t=0; got {p}");
    }

    #[test]
    fn t_two_sided_p_large_t_is_tiny() {
        // A very large t-statistic yields a vanishingly small p-value.
        let p = students_t_two_sided_p(20.0, 30.0);
        assert!(p < 1.0e-6, "expected tiny p for t=20, df=30; got {p}");
    }

    #[test]
    fn t_two_sided_p_normal_limit() {
        // As df -> infinity the t-distribution approaches the standard normal,
        // for which P(|Z| >= 1.96) ~ 0.05.
        let p = students_t_two_sided_p(1.96, 1_000_000.0);
        assert!(
            (p - 0.05).abs() < 5.0e-3,
            "expected ~0.05 in the normal limit; got {p}"
        );
    }

    #[test]
    fn significance_helper_separates_distinct_groups() {
        // Build two clearly-separated groups with a deterministic LCG so that
        // the Welch t-test reports strong significance (small p-value).
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let mut values = Vec::new();
        for _ in 0..20 {
            values.push(1.0 + 0.01 * rng.next_f64()); // group 1 ~ 1.0
        }
        for _ in 0..20 {
            values.push(5.0 + 0.01 * rng.next_f64()); // group 2 ~ 5.0
        }
        let tracker = ExperimentTracker::new();
        let p = tracker.calculate_statistical_significance(&values);
        assert!(
            p < 0.01,
            "well-separated groups should be significant; got {p}"
        );
    }

    #[test]
    fn significance_helper_identical_groups_not_significant() {
        // Identical halves => zero standard error, equal means => p = 1.0.
        let values = vec![2.0; 20];
        let tracker = ExperimentTracker::new();
        let p = tracker.calculate_statistical_significance(&values);
        assert!(
            (p - 1.0).abs() < 1.0e-12,
            "identical data => p=1.0; got {p}"
        );
    }
}
