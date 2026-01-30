//! Data validation helpers for ensuring data integrity.
//!
//! This module provides comprehensive validation functions for feedback data,
//! user inputs, and system configuration.

use crate::float_utils::{approx_ge, approx_le, in_range};

/// Validation result with detailed error information.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// List of validation errors
    pub errors: Vec<String>,
    /// List of validation warnings (non-critical)
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a successful validation result.
    #[must_use]
    pub fn success() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a failed validation result with an error.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            valid: false,
            errors: vec![message.into()],
            warnings: Vec::new(),
        }
    }

    /// Add an error to the result.
    pub fn add_error(&mut self, message: impl Into<String>) {
        self.valid = false;
        self.errors.push(message.into());
    }

    /// Add a warning to the result.
    pub fn add_warning(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }

    /// Check if the result has any errors.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Check if the result has any warnings.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Merge another validation result into this one.
    pub fn merge(&mut self, other: ValidationResult) {
        self.valid = self.valid && other.valid;
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

/// Validate that a score is within the valid range [0.0, 1.0].
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_score;
///
/// assert!(validate_score(0.5).valid);
/// assert!(!validate_score(1.5).valid);
/// ```
#[must_use]
pub fn validate_score(score: f32) -> ValidationResult {
    let mut result = ValidationResult::success();

    if !in_range(score, 0.0, 1.0) {
        result.add_error(format!("Score {score} is outside valid range [0.0, 1.0]"));
    }

    if score.is_nan() {
        result.add_error("Score is NaN (Not a Number)");
    }

    if score.is_infinite() {
        result.add_error("Score is infinite");
    }

    result
}

/// Validate that a percentage is within [0.0, 100.0].
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_percentage;
///
/// assert!(validate_percentage(50.0).valid);
/// assert!(!validate_percentage(150.0).valid);
/// ```
#[must_use]
pub fn validate_percentage(percentage: f32) -> ValidationResult {
    let mut result = ValidationResult::success();

    if !in_range(percentage, 0.0, 100.0) {
        result.add_error(format!(
            "Percentage {percentage} is outside valid range [0.0, 100.0]"
        ));
    }

    result
}

/// Validate that a confidence value is within [0.0, 1.0].
///
/// Also checks for suspiciously low confidence values.
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_confidence;
///
/// assert!(validate_confidence(0.8).valid);
/// assert!(validate_confidence(0.3).has_warnings());
/// ```
#[must_use]
pub fn validate_confidence(confidence: f32) -> ValidationResult {
    let mut result = validate_score(confidence);

    if result.valid && approx_le(confidence, 0.5) {
        result.add_warning(format!(
            "Confidence {confidence} is low (≤ 0.5), results may be unreliable"
        ));
    }

    result
}

/// Validate that a sample rate is reasonable for audio processing.
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_sample_rate;
///
/// assert!(validate_sample_rate(16_000).valid);
/// assert!(validate_sample_rate(44_100).valid);
/// assert!(!validate_sample_rate(1000).valid);
/// ```
#[must_use]
pub fn validate_sample_rate(sample_rate: u32) -> ValidationResult {
    let mut result = ValidationResult::success();

    const MIN_SAMPLE_RATE: u32 = 8_000;
    const MAX_SAMPLE_RATE: u32 = 192_000;
    const COMMON_RATES: &[u32] = &[
        8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000, 192_000,
    ];

    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        result.add_error(format!(
            "Sample rate {sample_rate} is outside valid range [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]"
        ));
    }

    if result.valid && !COMMON_RATES.contains(&sample_rate) {
        result.add_warning(format!(
            "Sample rate {sample_rate} is non-standard, common rates are: {COMMON_RATES:?}"
        ));
    }

    result
}

/// Validate text input for feedback processing.
///
/// Checks for empty strings, excessive length, and suspicious patterns.
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_text_input;
///
/// assert!(validate_text_input("Hello world").valid);
/// assert!(!validate_text_input("").valid);
/// ```
#[must_use]
pub fn validate_text_input(text: &str) -> ValidationResult {
    let mut result = ValidationResult::success();

    const MAX_TEXT_LENGTH: usize = 10_000;
    const MIN_TEXT_LENGTH: usize = 1;

    if text.is_empty() {
        result.add_error("Text input is empty");
    } else if text.len() < MIN_TEXT_LENGTH {
        result.add_error(format!(
            "Text input is too short (minimum {MIN_TEXT_LENGTH} characters)"
        ));
    } else if text.len() > MAX_TEXT_LENGTH {
        result.add_error(format!(
            "Text input exceeds maximum length of {MAX_TEXT_LENGTH} characters"
        ));
    }

    // Check for suspicious patterns
    if text.trim().is_empty() && !text.is_empty() {
        result.add_warning("Text contains only whitespace characters");
    }

    let control_chars = text
        .chars()
        .filter(|c| c.is_control() && *c != '\n' && *c != '\r' && *c != '\t')
        .count();
    if control_chars > 0 {
        result.add_warning(format!("Text contains {control_chars} control characters"));
    }

    result
}

/// Validate audio buffer parameters.
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_audio_buffer;
///
/// assert!(validate_audio_buffer(16_000, 1, 16_000).valid);
/// assert!(!validate_audio_buffer(16_000, 0, 16_000).valid);
/// ```
#[must_use]
pub fn validate_audio_buffer(
    sample_rate: u32,
    channels: usize,
    sample_count: usize,
) -> ValidationResult {
    let mut result = ValidationResult::success();

    // Validate sample rate
    result.merge(validate_sample_rate(sample_rate));

    // Validate channels
    if channels == 0 {
        result.add_error("Audio buffer has zero channels");
    } else if channels > 8 {
        result.add_warning(format!(
            "Audio buffer has {channels} channels, which is unusual"
        ));
    }

    // Validate sample count
    if sample_count == 0 {
        result.add_error("Audio buffer has zero samples");
    }

    const MAX_SAMPLES: usize = 100_000_000; // ~1 hour at 48kHz stereo
    if sample_count > MAX_SAMPLES {
        result.add_warning(format!(
            "Audio buffer has {sample_count} samples, which is very large"
        ));
    }

    result
}

/// Validate email address format.
///
/// Basic validation for email addresses.
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_email;
///
/// assert!(validate_email("user@example.com").valid);
/// assert!(!validate_email("invalid").valid);
/// ```
#[must_use]
pub fn validate_email(email: &str) -> ValidationResult {
    let mut result = ValidationResult::success();

    if email.is_empty() {
        result.add_error("Email address is empty");
        return result;
    }

    if email.contains('@') {
        let parts: Vec<&str> = email.split('@').collect();
        if parts.len() == 2 {
            if parts[0].is_empty() {
                result.add_error("Email address has empty local part");
            }
            if parts[1].is_empty() {
                result.add_error("Email address has empty domain part");
            }
            if !parts[1].contains('.') {
                result.add_warning("Email domain doesn't contain a dot (may be invalid)");
            }
        } else {
            result.add_error("Email address has invalid format");
        }
    } else {
        result.add_error("Email address must contain '@' symbol");
    }

    result
}

/// Validate username format.
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_username;
///
/// assert!(validate_username("john_doe123").valid);
/// assert!(!validate_username("ab").valid);
/// ```
#[must_use]
pub fn validate_username(username: &str) -> ValidationResult {
    let mut result = ValidationResult::success();

    const MIN_LENGTH: usize = 3;
    const MAX_LENGTH: usize = 50;

    if username.len() < MIN_LENGTH {
        result.add_error(format!("Username must be at least {MIN_LENGTH} characters"));
    }

    if username.len() > MAX_LENGTH {
        result.add_error(format!("Username must not exceed {MAX_LENGTH} characters"));
    }

    if !username
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        result
            .add_error("Username contains invalid characters (only alphanumeric, _, -, . allowed)");
    }

    if username.starts_with(|c: char| !c.is_alphanumeric()) {
        result.add_warning("Username should start with an alphanumeric character");
    }

    result
}

/// Validate a collection is not empty.
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_not_empty;
///
/// assert!(validate_not_empty(&vec![1, 2, 3], "numbers").valid);
/// assert!(!validate_not_empty(&Vec::<i32>::new(), "numbers").valid);
/// ```
pub fn validate_not_empty<T>(collection: &[T], name: &str) -> ValidationResult {
    if collection.is_empty() {
        ValidationResult::error(format!("{name} collection is empty"))
    } else {
        ValidationResult::success()
    }
}

/// Validate that a value is within a specified range.
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_range;
///
/// assert!(validate_range(5.0, 0.0, 10.0, "value").valid);
/// assert!(!validate_range(15.0, 0.0, 10.0, "value").valid);
/// ```
#[must_use]
pub fn validate_range(value: f32, min: f32, max: f32, name: &str) -> ValidationResult {
    if in_range(value, min, max) {
        ValidationResult::success()
    } else {
        ValidationResult::error(format!(
            "{name} ({value}) is outside valid range [{min}, {max}]"
        ))
    }
}

/// Validate duration is positive and reasonable.
///
/// # Examples
///
/// ```
/// use voirs_feedback::validation_helpers::validate_duration;
/// use std::time::Duration;
///
/// assert!(validate_duration(Duration::from_secs(10)).valid);
/// assert!(!validate_duration(Duration::from_secs(0)).valid);
/// ```
#[must_use]
pub fn validate_duration(duration: std::time::Duration) -> ValidationResult {
    let mut result = ValidationResult::success();

    if duration.is_zero() {
        result.add_error("Duration is zero");
    }

    const MAX_DURATION_SECS: u64 = 3_600 * 24; // 24 hours
    if duration.as_secs() > MAX_DURATION_SECS {
        result.add_warning(format!(
            "Duration ({} seconds) exceeds typical maximum of {} seconds",
            duration.as_secs(),
            MAX_DURATION_SECS
        ));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::success();
        assert!(result.valid);
        assert!(!result.has_errors());

        result.add_error("Test error");
        assert!(!result.valid);
        assert!(result.has_errors());

        result.add_warning("Test warning");
        assert!(result.has_warnings());
    }

    #[test]
    fn test_validation_result_merge() {
        let mut result1 = ValidationResult::success();
        result1.add_warning("Warning 1");

        let mut result2 = ValidationResult::success();
        result2.add_error("Error 2");

        result1.merge(result2);
        assert!(!result1.valid);
        assert_eq!(result1.errors.len(), 1);
        assert_eq!(result1.warnings.len(), 1);
    }

    #[test]
    fn test_validate_score() {
        assert!(validate_score(0.5).valid);
        assert!(validate_score(0.0).valid);
        assert!(validate_score(1.0).valid);
        assert!(!validate_score(-0.1).valid);
        assert!(!validate_score(1.1).valid);
        assert!(!validate_score(f32::NAN).valid);
        assert!(!validate_score(f32::INFINITY).valid);
    }

    #[test]
    fn test_validate_percentage() {
        assert!(validate_percentage(50.0).valid);
        assert!(validate_percentage(0.0).valid);
        assert!(validate_percentage(100.0).valid);
        assert!(!validate_percentage(-10.0).valid);
        assert!(!validate_percentage(150.0).valid);
    }

    #[test]
    fn test_validate_confidence() {
        assert!(validate_confidence(0.8).valid);
        assert!(!validate_confidence(0.8).has_warnings());

        let low_confidence = validate_confidence(0.3);
        assert!(low_confidence.valid);
        assert!(low_confidence.has_warnings());
    }

    #[test]
    fn test_validate_sample_rate() {
        assert!(validate_sample_rate(16_000).valid);
        assert!(validate_sample_rate(44_100).valid);
        assert!(!validate_sample_rate(1_000).valid);
        assert!(!validate_sample_rate(200_000).valid);

        let non_standard = validate_sample_rate(32_768);
        assert!(non_standard.valid);
        assert!(non_standard.has_warnings());
    }

    #[test]
    fn test_validate_text_input() {
        assert!(validate_text_input("Hello world").valid);
        assert!(!validate_text_input("").valid);

        let whitespace_only = validate_text_input("   ");
        assert!(whitespace_only.has_warnings());

        let too_long = validate_text_input(&"a".repeat(10_001));
        assert!(!too_long.valid);
    }

    #[test]
    fn test_validate_audio_buffer() {
        assert!(validate_audio_buffer(16_000, 1, 16_000).valid);
        assert!(!validate_audio_buffer(16_000, 0, 16_000).valid);
        assert!(!validate_audio_buffer(16_000, 1, 0).valid);
    }

    #[test]
    fn test_validate_email() {
        assert!(validate_email("user@example.com").valid);
        assert!(!validate_email("invalid").valid);
        assert!(!validate_email("@example.com").valid);
        assert!(!validate_email("user@").valid);

        let no_dot = validate_email("user@localhost");
        assert!(no_dot.has_warnings());
    }

    #[test]
    fn test_validate_username() {
        assert!(validate_username("john_doe123").valid);
        assert!(!validate_username("ab").valid);
        assert!(!validate_username("a".repeat(51).as_str()).valid);
        assert!(!validate_username("user@name").valid);

        let starts_with_underscore = validate_username("_username");
        assert!(starts_with_underscore.has_warnings());
    }

    #[test]
    fn test_validate_not_empty() {
        assert!(validate_not_empty(&vec![1, 2, 3], "test").valid);
        assert!(!validate_not_empty(&Vec::<i32>::new(), "test").valid);
    }

    #[test]
    fn test_validate_range() {
        assert!(validate_range(5.0, 0.0, 10.0, "test").valid);
        assert!(!validate_range(15.0, 0.0, 10.0, "test").valid);
    }

    #[test]
    fn test_validate_duration() {
        use std::time::Duration;

        assert!(validate_duration(Duration::from_secs(10)).valid);
        assert!(!validate_duration(Duration::from_secs(0)).valid);

        let very_long = validate_duration(Duration::from_secs(100_000));
        assert!(very_long.has_warnings());
    }
}
