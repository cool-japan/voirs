//! Experimental design support: sample-size calculation for a target
//! statistical power, and minimum-detectable-effect calculation for a fixed
//! sample size.
//!
//! [`super::basic_tests::StatisticalAnalyzer::power_analysis_t_test`] already
//! answers "given n and an effect size, what power do I have?"; this module
//! answers the two complementary planning questions an experimenter actually
//! needs *before* collecting data: "how many samples do I need for a target
//! power?" ([`required_sample_size`]) and "what effect size can I even detect
//! with the n I can afford?" ([`minimum_detectable_effect`]). Both use the
//! same two-sample, two-tailed t-test power model (central-t-shifted-by-
//! non-centrality-parameter approximation) as `power_analysis_t_test`, for
//! consistency across the crate.

use crate::{EvaluationError, EvaluationResult};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Achieved power of a two-sample, two-tailed t-test with `n` observations
/// per group, standardized effect size `effect_size` (Cohen's d), and
/// significance level `alpha`.
///
/// This is the same non-centrality-parameter approximation as
/// [`super::basic_tests::StatisticalAnalyzer::power_analysis_t_test`]
/// (`df = 2n - 2`, `ncp = d·√(n/2)`), exposed as a free function here so
/// [`required_sample_size`]/[`minimum_detectable_effect`] can search over it
/// without needing a `StatisticalAnalyzer` instance.
fn two_sample_power(effect_size: f64, n_per_group: f64, alpha: f64) -> f64 {
    if n_per_group < 2.0 {
        return 0.0;
    }
    let df = (2.0 * n_per_group - 2.0).max(1.0);
    let ncp = effect_size * (n_per_group / 2.0).sqrt();
    let dist = match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => dist,
        Err(_) => return 0.0,
    };
    let critical_t = dist.inverse_cdf(1.0 - alpha / 2.0);
    let upper = 1.0 - dist.cdf(critical_t - ncp);
    let lower = dist.cdf(-critical_t - ncp);
    (upper + lower).clamp(0.0, 1.0)
}

/// Required per-group sample size to achieve `target_power` for a two-sample,
/// two-tailed t-test with standardized effect size `effect_size` at
/// significance level `alpha`.
///
/// Uses binary search over `n` (the power function is monotonically
/// increasing in `n`) rather than a closed-form normal approximation, so it
/// remains accurate for small samples where the t-distribution's heavier
/// tails matter. Returns [`EvaluationError::InvalidInput`] for a non-positive
/// effect size (no finite `n` can achieve a target power against a truly
/// zero effect), or `target_power`/`alpha` outside `(0, 1)`.
pub fn required_sample_size(
    effect_size: f64,
    target_power: f64,
    alpha: f64,
) -> EvaluationResult<usize> {
    if effect_size <= 0.0 || !effect_size.is_finite() {
        return Err(EvaluationError::InvalidInput {
            message: "required_sample_size requires a positive, finite effect size".to_string(),
        }
        .into());
    }
    if !(0.0..1.0).contains(&target_power) {
        return Err(EvaluationError::InvalidInput {
            message: "target_power must be in (0, 1)".to_string(),
        }
        .into());
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(EvaluationError::InvalidInput {
            message: "alpha must be in (0, 1)".to_string(),
        }
        .into());
    }

    // Binary search over per-group n in [2, upper_bound], expanding
    // upper_bound until it brackets the target power (guards against
    // unreasonably large searches for a vanishingly small effect size while
    // still terminating for any legitimate input).
    let mut lo = 2.0f64;
    let mut hi = 100.0f64;
    while two_sample_power(effect_size, hi, alpha) < target_power {
        hi *= 2.0;
        if hi > 10_000_000.0 {
            return Err(EvaluationError::InvalidInput {
                message: "required sample size exceeds a practical search bound (effect size \
                          too small relative to target power)"
                    .to_string(),
            }
            .into());
        }
    }

    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if two_sample_power(effect_size, mid, alpha) >= target_power {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo < 0.5 {
            break;
        }
    }

    Ok(hi.ceil() as usize)
}

/// The smallest standardized effect size (Cohen's d) detectable with
/// `target_power` at significance level `alpha`, given a fixed per-group
/// sample size `n_per_group` — the practical "what can I even measure with
/// the data I can afford" question.
///
/// Uses binary search over the effect size (power is monotonically
/// increasing in effect size for fixed n). Returns
/// [`EvaluationError::InvalidInput`] for `n_per_group < 2` (no meaningful
/// power calculation is possible).
pub fn minimum_detectable_effect(
    n_per_group: usize,
    target_power: f64,
    alpha: f64,
) -> EvaluationResult<f64> {
    if n_per_group < 2 {
        return Err(EvaluationError::InvalidInput {
            message: "minimum_detectable_effect requires at least 2 observations per group"
                .to_string(),
        }
        .into());
    }
    if !(0.0..1.0).contains(&target_power) {
        return Err(EvaluationError::InvalidInput {
            message: "target_power must be in (0, 1)".to_string(),
        }
        .into());
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(EvaluationError::InvalidInput {
            message: "alpha must be in (0, 1)".to_string(),
        }
        .into());
    }

    let n = n_per_group as f64;
    let mut lo = 0.0f64;
    let mut hi = 5.0f64; // Cohen's d of 5 is an enormous effect; ample upper bound.
    if two_sample_power(hi, n, alpha) < target_power {
        return Err(EvaluationError::InvalidInput {
            message: format!(
                "target power {target_power} is not achievable at n = {n_per_group} even for \
                 a very large effect size (d = 5); increase the sample size"
            ),
        }
        .into());
    }

    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if two_sample_power(mid, n, alpha) >= target_power {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo < 1e-4 {
            break;
        }
    }

    Ok(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_sample_size_larger_effect_needs_fewer_samples() {
        let n_small_effect = required_sample_size(0.2, 0.8, 0.05).unwrap();
        let n_large_effect = required_sample_size(0.8, 0.8, 0.05).unwrap();
        assert!(
            n_large_effect < n_small_effect,
            "a larger effect size ({n_large_effect}) should need fewer samples than a small \
             one ({n_small_effect})"
        );
        // Classic textbook value: d=0.5, power=0.8, alpha=0.05 needs roughly
        // 64 per group (standard reference tables give 63-64); allow a
        // reasonable tolerance around that since the implementation here
        // uses the non-centrality-shift approximation, not an exact
        // noncentral-t integral.
        let n_medium = required_sample_size(0.5, 0.8, 0.05).unwrap();
        assert!(
            (40..=100).contains(&n_medium),
            "expected a textbook-plausible sample size for d=0.5, got {n_medium}"
        );
    }

    #[test]
    fn test_required_sample_size_higher_power_needs_more_samples() {
        let n_low_power = required_sample_size(0.5, 0.6, 0.05).unwrap();
        let n_high_power = required_sample_size(0.5, 0.95, 0.05).unwrap();
        assert!(n_high_power > n_low_power);
    }

    #[test]
    fn test_required_sample_size_achieves_target_power() {
        let effect_size = 0.5;
        let target_power = 0.8;
        let alpha = 0.05;
        let n = required_sample_size(effect_size, target_power, alpha).unwrap();
        let achieved = two_sample_power(effect_size, n as f64, alpha);
        assert!(
            achieved >= target_power - 0.01,
            "computed sample size {n} should achieve at least the target power {target_power}, \
             achieved {achieved}"
        );
    }

    #[test]
    fn test_required_sample_size_invalid_inputs() {
        assert!(required_sample_size(0.0, 0.8, 0.05).is_err());
        assert!(required_sample_size(-0.5, 0.8, 0.05).is_err());
        assert!(required_sample_size(0.5, 1.5, 0.05).is_err());
        assert!(required_sample_size(0.5, 0.8, 1.5).is_err());
    }

    #[test]
    fn test_minimum_detectable_effect_larger_n_detects_smaller_effect() {
        let mde_small_n = minimum_detectable_effect(20, 0.8, 0.05).unwrap();
        let mde_large_n = minimum_detectable_effect(500, 0.8, 0.05).unwrap();
        assert!(
            mde_large_n < mde_small_n,
            "a larger sample size ({mde_large_n}) should detect a smaller minimum effect than \
             a small one ({mde_small_n})"
        );
    }

    #[test]
    fn test_minimum_detectable_effect_round_trips_with_sample_size() {
        // The MDE at n should be (approximately) the effect size for which
        // required_sample_size returns ~n.
        let n = 64usize;
        let mde = minimum_detectable_effect(n, 0.8, 0.05).unwrap();
        let achieved_power = two_sample_power(mde, n as f64, 0.05);
        assert!(
            achieved_power >= 0.79 && achieved_power <= 0.82,
            "power at the computed MDE should be very close to the 0.8 target, got {achieved_power}"
        );
    }

    #[test]
    fn test_minimum_detectable_effect_invalid_inputs() {
        assert!(minimum_detectable_effect(1, 0.8, 0.05).is_err());
        assert!(minimum_detectable_effect(20, 1.5, 0.05).is_err());
    }
}
