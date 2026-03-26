//! Float comparison utilities with epsilon-based comparisons.
//!
//! This module provides robust float comparison functions that avoid
//! strict equality checks, which are unreliable for floating-point numbers.

use std::cmp::Ordering;

/// Default epsilon for f32 comparisons (approximately 6 decimal places of precision)
pub const F32_EPSILON: f32 = 1e-6;

/// Default epsilon for f64 comparisons (approximately 10 decimal places of precision)
pub const F64_EPSILON: f64 = 1e-10;

/// Relative epsilon for comparing numbers of different magnitudes
pub const RELATIVE_EPSILON: f32 = 1e-5;

/// Compare two f32 values for approximate equality.
///
/// # Arguments
///
/// * `a` - First value
/// * `b` - Second value
/// * `epsilon` - Maximum allowed difference
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_eq_f32;
///
/// assert!(approx_eq_f32(1.0, 1.0000001, 1e-6));
/// assert!(!approx_eq_f32(1.0, 1.001, 1e-6));
/// ```
#[inline]
#[must_use]
pub fn approx_eq_f32(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

/// Compare two f64 values for approximate equality.
///
/// # Arguments
///
/// * `a` - First value
/// * `b` - Second value
/// * `epsilon` - Maximum allowed difference
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_eq_f64;
///
/// assert!(approx_eq_f64(1.0, 1.000000000001, 1e-10));
/// assert!(!approx_eq_f64(1.0, 1.0000001, 1e-10));
/// ```
#[inline]
#[must_use]
pub fn approx_eq_f64(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

/// Compare two f32 values with default epsilon.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_eq;
///
/// assert!(approx_eq(1.0_f32, 1.0000001_f32));
/// assert!(!approx_eq(1.0_f32, 1.001_f32));
/// ```
#[inline]
#[must_use]
pub fn approx_eq(a: f32, b: f32) -> bool {
    approx_eq_f32(a, b, F32_EPSILON)
}

/// Compare two f32 values for approximate inequality.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_ne;
///
/// assert!(approx_ne(1.0_f32, 1.001_f32));
/// assert!(!approx_ne(1.0_f32, 1.0000001_f32));
/// ```
#[inline]
#[must_use]
pub fn approx_ne(a: f32, b: f32) -> bool {
    !approx_eq(a, b)
}

/// Check if a is approximately greater than b.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_gt;
///
/// assert!(approx_gt(1.001_f32, 1.0_f32));
/// assert!(!approx_gt(1.0000001_f32, 1.0_f32)); // Within epsilon
/// ```
#[inline]
#[must_use]
pub fn approx_gt(a: f32, b: f32) -> bool {
    a - b > F32_EPSILON
}

/// Check if a is approximately less than b.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_lt;
///
/// assert!(approx_lt(1.0_f32, 1.001_f32));
/// assert!(!approx_lt(1.0_f32, 1.0000001_f32)); // Within epsilon
/// ```
#[inline]
#[must_use]
pub fn approx_lt(a: f32, b: f32) -> bool {
    b - a > F32_EPSILON
}

/// Check if a is approximately greater than or equal to b.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_ge;
///
/// assert!(approx_ge(1.001_f32, 1.0_f32));
/// assert!(approx_ge(1.0000001_f32, 1.0_f32)); // Within epsilon
/// ```
#[inline]
#[must_use]
pub fn approx_ge(a: f32, b: f32) -> bool {
    a - b > -F32_EPSILON
}

/// Check if a is approximately less than or equal to b.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_le;
///
/// assert!(approx_le(1.0_f32, 1.001_f32));
/// assert!(approx_le(1.0000001_f32, 1.0_f32)); // Within epsilon
/// ```
#[inline]
#[must_use]
pub fn approx_le(a: f32, b: f32) -> bool {
    b - a > -F32_EPSILON
}

/// Compare two floats with ordering, accounting for epsilon.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_cmp;
/// use std::cmp::Ordering;
///
/// assert_eq!(approx_cmp(1.001_f32, 1.0_f32), Ordering::Greater);
/// assert_eq!(approx_cmp(1.0000001_f32, 1.0_f32), Ordering::Equal);
/// assert_eq!(approx_cmp(0.999_f32, 1.0_f32), Ordering::Less);
/// ```
#[must_use]
pub fn approx_cmp(a: f32, b: f32) -> Ordering {
    if approx_eq(a, b) {
        Ordering::Equal
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}

/// Relative comparison for values of different magnitudes.
///
/// Uses relative epsilon scaled by the magnitude of the values.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_eq_relative;
///
/// assert!(approx_eq_relative(1000.0, 1000.005, 1e-5));
/// assert!(approx_eq_relative(0.001, 0.0010000001, 1e-5));
/// ```
#[must_use]
pub fn approx_eq_relative(a: f32, b: f32, relative_epsilon: f32) -> bool {
    let abs_a = a.abs();
    let abs_b = b.abs();
    let magnitude = abs_a.max(abs_b);

    if magnitude < F32_EPSILON {
        // Both values are very small, use absolute comparison
        approx_eq_f32(a, b, F32_EPSILON)
    } else {
        // Use relative comparison
        (a - b).abs() < magnitude * relative_epsilon
    }
}

/// Check if a float is approximately zero.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_zero;
///
/// assert!(approx_zero(0.0000001_f32));
/// assert!(!approx_zero(0.001_f32));
/// ```
#[inline]
#[must_use]
pub fn approx_zero(value: f32) -> bool {
    value.abs() < F32_EPSILON
}

/// Check if a float is approximately one.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_one;
///
/// assert!(approx_one(0.9999999_f32));
/// assert!(!approx_one(0.999_f32));
/// ```
#[inline]
#[must_use]
pub fn approx_one(value: f32) -> bool {
    approx_eq(value, 1.0)
}

/// Clamp a value to a range with epsilon tolerance.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::clamp_with_epsilon;
///
/// assert_eq!(clamp_with_epsilon(1.5, 0.0, 1.0), 1.0);
/// assert_eq!(clamp_with_epsilon(0.9999999, 0.0, 1.0), 1.0); // Treated as 1.0
/// ```
#[must_use]
pub fn clamp_with_epsilon(value: f32, min: f32, max: f32) -> f32 {
    if approx_le(value, min) {
        min
    } else if approx_ge(value, max) {
        max
    } else {
        value
    }
}

/// Check if a value is within a range (inclusive) with epsilon tolerance.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::in_range;
///
/// assert!(in_range(0.5, 0.0, 1.0));
/// assert!(in_range(0.9999999, 0.0, 1.0)); // Within epsilon of 1.0
/// assert!(!in_range(1.001, 0.0, 1.0));
/// ```
#[must_use]
pub fn in_range(value: f32, min: f32, max: f32) -> bool {
    approx_ge(value, min) && approx_le(value, max)
}

/// Find the maximum of two floats with epsilon comparison.
///
/// If values are equal within epsilon, returns the first value.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_max;
///
/// assert_eq!(approx_max(1.0, 0.5), 1.0);
/// assert_eq!(approx_max(1.0, 1.0000001), 1.0); // Equal within epsilon
/// ```
#[must_use]
pub fn approx_max(a: f32, b: f32) -> f32 {
    if approx_eq(a, b) {
        a
    } else if a > b {
        a
    } else {
        b
    }
}

/// Find the minimum of two floats with epsilon comparison.
///
/// If values are equal within epsilon, returns the first value.
///
/// # Examples
///
/// ```
/// use voirs_feedback::float_utils::approx_min;
///
/// assert_eq!(approx_min(1.0, 0.5), 0.5);
/// assert_eq!(approx_min(1.0, 1.0000001), 1.0); // Equal within epsilon
/// ```
#[must_use]
pub fn approx_min(a: f32, b: f32) -> f32 {
    if approx_eq(a, b) {
        a
    } else if a < b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_eq_f32() {
        assert!(approx_eq_f32(1.0, 1.0, F32_EPSILON));
        assert!(approx_eq_f32(1.0, 1.0 + F32_EPSILON / 2.0, F32_EPSILON));
        assert!(!approx_eq_f32(1.0, 1.0 + F32_EPSILON * 2.0, F32_EPSILON));
    }

    #[test]
    fn test_approx_comparisons() {
        assert!(approx_gt(1.001, 1.0));
        assert!(!approx_gt(1.0000001, 1.0));

        assert!(approx_lt(0.999, 1.0));
        assert!(!approx_lt(0.9999999, 1.0));

        assert!(approx_ge(1.0, 1.0));
        assert!(approx_le(1.0, 1.0));
    }

    #[test]
    fn test_approx_cmp() {
        assert_eq!(approx_cmp(1.001, 1.0), Ordering::Greater);
        assert_eq!(approx_cmp(1.0000001, 1.0), Ordering::Equal);
        assert_eq!(approx_cmp(0.999, 1.0), Ordering::Less);
    }

    #[test]
    fn test_relative_comparison() {
        assert!(approx_eq_relative(1000.0, 1000.01, 1e-4));
        assert!(!approx_eq_relative(1000.0, 1000.01, 1e-6));
    }

    #[test]
    fn test_special_values() {
        assert!(approx_zero(0.0));
        assert!(approx_zero(F32_EPSILON / 2.0));
        assert!(!approx_zero(0.001));

        assert!(approx_one(1.0));
        assert!(approx_one(1.0 + F32_EPSILON / 2.0));
        assert!(!approx_one(0.999));
    }

    #[test]
    fn test_clamp_with_epsilon() {
        assert_eq!(clamp_with_epsilon(0.5, 0.0, 1.0), 0.5);
        assert_eq!(clamp_with_epsilon(1.5, 0.0, 1.0), 1.0);
        assert_eq!(clamp_with_epsilon(-0.5, 0.0, 1.0), 0.0);
        assert_eq!(clamp_with_epsilon(0.9999999, 0.0, 1.0), 1.0);
    }

    #[test]
    fn test_in_range() {
        assert!(in_range(0.5, 0.0, 1.0));
        assert!(in_range(0.0, 0.0, 1.0));
        assert!(in_range(1.0, 0.0, 1.0));
        assert!(in_range(0.9999999, 0.0, 1.0));
        assert!(!in_range(1.001, 0.0, 1.0));
    }

    #[test]
    fn test_approx_max_min() {
        assert_eq!(approx_max(1.0, 0.5), 1.0);
        assert_eq!(approx_max(0.5, 1.0), 1.0);
        assert_eq!(approx_max(1.0, 1.0000001), 1.0);

        assert_eq!(approx_min(1.0, 0.5), 0.5);
        assert_eq!(approx_min(0.5, 1.0), 0.5);
        assert_eq!(approx_min(1.0, 1.0000001), 1.0);
    }
}
