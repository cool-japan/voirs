//! Causal inference methods: difference-in-differences (DiD) and simple
//! nearest-neighbor covariate matching.
//!
//! These are standard, well-established quasi-experimental estimators for
//! settings where a randomized controlled trial is not available (e.g.
//! comparing a metric's before/after behavior across a treated group — such
//! as samples processed by a new codec/model version — against a control
//! group over the same period). Both estimators here compute real statistics
//! from the caller's data; there is no synthetic/seeded "effect" injected.

use super::utils::{mean, pearson_correlation_f64, std_dev};
use crate::{EvaluationError, EvaluationResult};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Result of a difference-in-differences estimate (see
/// [`difference_in_differences`]).
#[derive(Debug, Clone, PartialEq)]
pub struct DidResult {
    /// The DiD estimate itself:
    /// `(treated_after - treated_before) - (control_after - control_before)`
    pub estimate: f64,
    /// Standard error of the estimate (from the four group variances)
    pub standard_error: f64,
    /// t-statistic of the estimate against a null of no effect
    pub t_statistic: f64,
    /// Two-sided p-value (via `statrs`'s exact Student's-t CDF, matching the
    /// pattern established in [`super::basic_tests`])
    pub p_value: f64,
    /// Mean of the treated group before the intervention
    pub treated_before_mean: f64,
    /// Mean of the treated group after the intervention
    pub treated_after_mean: f64,
    /// Mean of the control group before the intervention
    pub control_before_mean: f64,
    /// Mean of the control group after the intervention
    pub control_after_mean: f64,
}

/// Difference-in-differences estimate of a treatment effect.
///
/// `treated_before`/`treated_after` are the treated group's outcome
/// observations before/after the intervention; `control_before`/
/// `control_after` are the same for a comparison group that did *not*
/// receive the intervention over the same period. The two "before" samples
/// need not be paired with the two "after" samples (this is the standard
/// repeated-cross-section DiD, not paired-panel DiD), so the four vectors
/// may have different lengths from each other.
///
/// The core identifying assumption (parallel trends: absent treatment, the
/// treated and control groups would have moved together) is a property of
/// the caller's experimental design, not something this function can verify
/// from the four samples alone — callers should ensure it holds.
///
/// Returns [`EvaluationError::InvalidInput`] when any group has fewer than 2
/// observations (no variance to estimate a standard error from).
pub fn difference_in_differences(
    treated_before: &[f64],
    treated_after: &[f64],
    control_before: &[f64],
    control_after: &[f64],
) -> EvaluationResult<DidResult> {
    for (name, group) in [
        ("treated_before", treated_before),
        ("treated_after", treated_after),
        ("control_before", control_before),
        ("control_after", control_after),
    ] {
        if group.len() < 2 {
            return Err(EvaluationError::InvalidInput {
                message: format!(
                    "difference_in_differences requires at least 2 observations per group; \
                     '{name}' has {}",
                    group.len()
                ),
            }
            .into());
        }
    }

    let treated_before_mean = mean(treated_before);
    let treated_after_mean = mean(treated_after);
    let control_before_mean = mean(control_before);
    let control_after_mean = mean(control_after);

    let treated_diff = treated_after_mean - treated_before_mean;
    let control_diff = control_after_mean - control_before_mean;
    let estimate = treated_diff - control_diff;

    // Standard error of the DiD estimate: the four group means are
    // independent, so the variance of their linear combination is the sum
    // of each group mean's own variance (group variance / group size).
    let var_of_mean = |group: &[f64]| -> f64 {
        let n = group.len() as f64;
        std_dev(group).powi(2) / n
    };
    let se_squared = var_of_mean(treated_before)
        + var_of_mean(treated_after)
        + var_of_mean(control_before)
        + var_of_mean(control_after);
    let standard_error = se_squared.sqrt();

    let (t_statistic, p_value) = if standard_error > 1e-12 {
        let t = estimate / standard_error;
        // Welch-Satterthwaite-style conservative degrees of freedom: the
        // minimum group size minus 1, a standard conservative choice when
        // combining four independent group variances without computing the
        // full Welch-Satterthwaite approximation.
        let df = ([treated_before, treated_after, control_before, control_after]
            .iter()
            .map(|g| g.len())
            .min()
            .unwrap_or(2)
            - 1) as f64;
        let p = match StudentsT::new(0.0, 1.0, df.max(1.0)) {
            Ok(dist) => (2.0 * (1.0 - dist.cdf(t.abs()))).clamp(0.0, 1.0),
            Err(_) => 1.0,
        };
        (t, p)
    } else {
        (0.0, 1.0)
    };

    Ok(DidResult {
        estimate,
        standard_error,
        t_statistic,
        p_value,
        treated_before_mean,
        treated_after_mean,
        control_before_mean,
        control_after_mean,
    })
}

/// Result of nearest-neighbor covariate matching (see
/// [`nearest_neighbor_matching`]).
#[derive(Debug, Clone, PartialEq)]
pub struct MatchingResult {
    /// Average treatment effect on the treated (ATT): mean over matched
    /// pairs of `treated_outcome - matched_control_outcome`
    pub att: f64,
    /// Standard error of the ATT (over the per-pair differences)
    pub standard_error: f64,
    /// Indices into `control_covariates`/`control_outcomes` of each treated
    /// unit's matched control (same order as the treated inputs)
    pub matched_control_indices: Vec<usize>,
    /// Mean absolute covariate distance across matched pairs (a matching
    /// quality diagnostic — large values indicate poor covariate overlap
    /// between treated and control groups)
    pub mean_match_distance: f64,
}

/// Estimate the average treatment effect on the treated (ATT) via
/// one-nearest-neighbor matching on a single real-valued covariate (e.g. a
/// propensity score, or any confounding covariate believed to drive
/// selection into treatment), with replacement (a control unit may be
/// matched to more than one treated unit).
///
/// For each treated unit, finds the control unit with the closest covariate
/// value and computes the outcome difference; the ATT is the mean of these
/// differences. This is the simplest standard matching estimator — a
/// starting point for causal comparison when random assignment isn't
/// available, not a substitute for careful propensity-score modeling with
/// multiple covariates.
///
/// Returns [`EvaluationError::InvalidInput`] when either group is empty or
/// the covariate/outcome vectors within a group have mismatched lengths.
pub fn nearest_neighbor_matching(
    treated_covariates: &[f64],
    treated_outcomes: &[f64],
    control_covariates: &[f64],
    control_outcomes: &[f64],
) -> EvaluationResult<MatchingResult> {
    if treated_covariates.len() != treated_outcomes.len() {
        return Err(EvaluationError::InvalidInput {
            message: "treated covariates and outcomes must have equal length".to_string(),
        }
        .into());
    }
    if control_covariates.len() != control_outcomes.len() {
        return Err(EvaluationError::InvalidInput {
            message: "control covariates and outcomes must have equal length".to_string(),
        }
        .into());
    }
    if treated_covariates.is_empty() || control_covariates.is_empty() {
        return Err(EvaluationError::InvalidInput {
            message: "nearest_neighbor_matching requires at least one treated and one control unit"
                .to_string(),
        }
        .into());
    }

    let mut matched_control_indices = Vec::with_capacity(treated_covariates.len());
    let mut differences = Vec::with_capacity(treated_covariates.len());
    let mut distances = Vec::with_capacity(treated_covariates.len());

    for (i, &treated_cov) in treated_covariates.iter().enumerate() {
        let mut best_idx = 0usize;
        let mut best_dist = f64::MAX;
        for (j, &control_cov) in control_covariates.iter().enumerate() {
            let dist = (treated_cov - control_cov).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = j;
            }
        }
        matched_control_indices.push(best_idx);
        distances.push(best_dist);
        differences.push(treated_outcomes[i] - control_outcomes[best_idx]);
    }

    let att = mean(&differences);
    let standard_error = if differences.len() > 1 {
        std_dev(&differences) / (differences.len() as f64).sqrt()
    } else {
        0.0
    };
    let mean_match_distance = mean(&distances);

    Ok(MatchingResult {
        att,
        standard_error,
        matched_control_indices,
        mean_match_distance,
    })
}

/// Simple covariate-balance check: the Pearson correlation between a
/// treatment indicator (`1.0` for treated, `0.0` for control, encoded by the
/// caller) and a candidate confounding covariate.
///
/// A DiD or matching estimate is more credible when this correlation is
/// small in magnitude for the covariates that plausibly drive both treatment
/// assignment and the outcome (i.e. treatment is not strongly predicted by
/// the covariate) — large correlations are a warning sign of confounding
/// that the simple estimators here do not adjust for.
pub fn covariate_balance(treatment_indicator: &[f64], covariate: &[f64]) -> EvaluationResult<f64> {
    pearson_correlation_f64(treatment_indicator, covariate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difference_in_differences_known_effect() {
        // Treated group: 0.5 -> 0.8 (rise of 0.3). Control group: 0.5 -> 0.55
        // (rise of 0.05, ambient trend). True DiD effect: 0.3 - 0.05 = 0.25.
        let treated_before = vec![0.48, 0.50, 0.52, 0.49, 0.51];
        let treated_after = vec![0.78, 0.80, 0.82, 0.79, 0.81];
        let control_before = vec![0.49, 0.51, 0.50, 0.48, 0.52];
        let control_after = vec![0.54, 0.56, 0.55, 0.53, 0.57];

        let result = difference_in_differences(
            &treated_before,
            &treated_after,
            &control_before,
            &control_after,
        )
        .unwrap();

        assert!(
            (result.estimate - 0.25).abs() < 0.02,
            "expected DiD estimate near 0.25, got {}",
            result.estimate
        );
        assert!(
            result.p_value < 0.05,
            "effect should be significant, p = {}",
            result.p_value
        );
    }

    #[test]
    fn test_difference_in_differences_no_effect() {
        // Both groups rise identically: true DiD effect is 0.
        let treated_before = vec![0.5, 0.5, 0.5, 0.5];
        let treated_after = vec![0.6, 0.6, 0.6, 0.6];
        let control_before = vec![0.5, 0.5, 0.5, 0.5];
        let control_after = vec![0.6, 0.6, 0.6, 0.6];

        let result = difference_in_differences(
            &treated_before,
            &treated_after,
            &control_before,
            &control_after,
        )
        .unwrap();
        assert!(result.estimate.abs() < 1e-9);
    }

    #[test]
    fn test_difference_in_differences_requires_min_size() {
        let too_small = vec![0.5];
        let ok = vec![0.5, 0.6];
        assert!(difference_in_differences(&too_small, &ok, &ok, &ok).is_err());
    }

    #[test]
    fn test_nearest_neighbor_matching_known_effect() {
        // Treated units with covariate values matched closely to controls
        // that differ from treated outcomes by exactly +0.2.
        let treated_covariates = vec![1.0, 2.0, 3.0];
        let treated_outcomes = vec![0.7, 0.8, 0.9];
        let control_covariates = vec![1.05, 1.95, 3.1, 10.0]; // extra unrelated control
        let control_outcomes = vec![0.5, 0.6, 0.7, 5.0];

        let result = nearest_neighbor_matching(
            &treated_covariates,
            &treated_outcomes,
            &control_covariates,
            &control_outcomes,
        )
        .unwrap();

        assert!(
            (result.att - 0.2).abs() < 1e-6,
            "expected ATT near 0.2, got {}",
            result.att
        );
        assert_eq!(result.matched_control_indices, vec![0, 1, 2]);
        assert!(result.mean_match_distance < 0.2);
    }

    #[test]
    fn test_nearest_neighbor_matching_empty_group_errors() {
        assert!(nearest_neighbor_matching(&[], &[], &[1.0], &[1.0]).is_err());
        assert!(nearest_neighbor_matching(&[1.0], &[1.0], &[], &[]).is_err());
    }

    #[test]
    fn test_covariate_balance_detects_confounding() {
        // Treatment indicator perfectly correlated with the covariate: a
        // clear confounding warning sign.
        let treatment = vec![0.0, 0.0, 1.0, 1.0];
        let confounded_covariate = vec![1.0, 2.0, 8.0, 9.0];
        let balance = covariate_balance(&treatment, &confounded_covariate).unwrap();
        assert!(
            balance > 0.9,
            "expected strong confounding correlation, got {balance}"
        );

        let balanced_covariate = vec![5.0, 4.0, 5.0, 4.0];
        let balance2 = covariate_balance(&treatment, &balanced_covariate).unwrap();
        assert!(
            balance2.abs() < balance,
            "a covariate unrelated to treatment should show much weaker correlation"
        );
    }
}
