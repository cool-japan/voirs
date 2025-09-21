/// Enhanced error message generation for better user experience
///
/// This module provides utilities for creating informative, actionable error messages
/// that help users understand what went wrong and how to fix it. It also includes
/// comprehensive error code documentation and automatic retry mechanisms.
use std::time::Duration;
use tokio::time::sleep;

/// Error categories for systematic error handling
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    /// Input validation errors (E100-E199)
    InputValidation,
    /// Audio processing errors (E200-E299)
    AudioProcessing,
    /// Resource errors (E300-E399)
    Resource,
    /// Configuration errors (E400-E499)
    Configuration,
    /// Network/IO errors (E500-E599)
    NetworkIO,
    /// Internal errors (E600-E699)
    Internal,
}

/// Comprehensive error code documentation
#[derive(Debug, Clone)]
pub struct ErrorCode {
    /// Numeric error code
    pub code: u16,
    /// Error category
    pub category: ErrorCategory,
    /// Short error name
    pub name: String,
    /// Detailed description
    pub description: String,
    /// Common causes
    pub common_causes: Vec<String>,
    /// Suggested solutions
    pub solutions: Vec<String>,
    /// Whether this error is retryable
    pub retryable: bool,
    /// Maximum retry attempts if retryable
    pub max_retries: u8,
    /// Base delay between retries in milliseconds
    pub retry_delay_ms: u64,
}

impl ErrorCode {
    /// Create a new error code
    pub fn new(code: u16, category: ErrorCategory, name: &str, description: &str) -> Self {
        Self {
            code,
            category,
            name: name.to_string(),
            description: description.to_string(),
            common_causes: Vec::new(),
            solutions: Vec::new(),
            retryable: false,
            max_retries: 0,
            retry_delay_ms: 1000,
        }
    }

    /// Add common causes
    pub fn with_causes(mut self, causes: &[&str]) -> Self {
        self.common_causes = causes.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add solutions
    pub fn with_solutions(mut self, solutions: &[&str]) -> Self {
        self.solutions = solutions.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Make this error retryable
    pub fn retryable(mut self, max_retries: u8, delay_ms: u64) -> Self {
        self.retryable = true;
        self.max_retries = max_retries;
        self.retry_delay_ms = delay_ms;
        self
    }

    /// Get formatted error message
    pub fn format_message(&self, context: &str) -> String {
        let mut message = format!(
            "Error {code}: {name} in {context}\n\nDescription: {description}",
            code = self.code,
            name = self.name,
            context = context,
            description = self.description
        );

        if !self.common_causes.is_empty() {
            message.push_str("\n\nCommon Causes:");
            for (i, cause) in self.common_causes.iter().enumerate() {
                message.push_str(&format!("\n  {}. {}", i + 1, cause));
            }
        }

        if !self.solutions.is_empty() {
            message.push_str("\n\nSuggested Solutions:");
            for (i, solution) in self.solutions.iter().enumerate() {
                message.push_str(&format!("\n  {}. {}", i + 1, solution));
            }
        }

        if self.retryable {
            message.push_str(&format!(
                "\n\nRetry Information: This operation can be retried up to {} times with {}ms delays",
                self.max_retries, self.retry_delay_ms
            ));
        }

        message
    }
}

/// Error code registry with comprehensive documentation
pub struct ErrorCodeRegistry {
    codes: std::collections::HashMap<u16, ErrorCode>,
}

impl ErrorCodeRegistry {
    /// Create new registry with all documented error codes
    pub fn new() -> Self {
        let mut registry = Self {
            codes: std::collections::HashMap::new(),
        };
        registry.register_all_codes();
        registry
    }

    /// Register an error code
    pub fn register(&mut self, error_code: ErrorCode) {
        self.codes.insert(error_code.code, error_code);
    }

    /// Get error code by number
    pub fn get(&self, code: u16) -> Option<&ErrorCode> {
        self.codes.get(&code)
    }

    /// Get all error codes in a category
    pub fn get_by_category(&self, category: &ErrorCategory) -> Vec<&ErrorCode> {
        self.codes
            .values()
            .filter(|code| &code.category == category)
            .collect()
    }

    /// Register all standard error codes
    fn register_all_codes(&mut self) {
        // Input Validation Errors (E100-E199)
        self.register(
            ErrorCode::new(
                101,
                ErrorCategory::InputValidation,
                "INVALID_SAMPLE_RATE",
                "Audio sample rate does not match evaluator requirements",
            )
            .with_causes(&[
                "Audio file has different sample rate than expected",
                "Incorrect audio loading configuration",
                "Missing sample rate conversion in pipeline",
            ])
            .with_solutions(&[
                "Resample audio to match evaluator requirements",
                "Use evaluator configured for the audio sample rate",
                "Check audio loading pipeline configuration",
            ]),
        );

        self.register(
            ErrorCode::new(
                102,
                ErrorCategory::InputValidation,
                "INVALID_CHANNEL_COUNT",
                "Audio channel count does not match evaluator requirements",
            )
            .with_causes(&[
                "Stereo audio provided when mono required",
                "Mono audio provided when stereo required",
                "Multi-channel audio not supported",
            ])
            .with_solutions(&[
                "Convert audio to required channel count",
                "Use audio conversion utilities",
                "Check evaluator channel requirements",
            ]),
        );

        self.register(
            ErrorCode::new(
                103,
                ErrorCategory::InputValidation,
                "INSUFFICIENT_AUDIO_LENGTH",
                "Audio is too short for reliable evaluation",
            )
            .with_causes(&[
                "Audio file shorter than minimum required duration",
                "Audio truncated during loading",
                "Silent audio detected as empty",
            ])
            .with_solutions(&[
                "Provide longer audio samples",
                "Check audio file integrity",
                "Verify audio loading process",
            ]),
        );

        self.register(
            ErrorCode::new(
                104,
                ErrorCategory::InputValidation,
                "EMPTY_AUDIO_BUFFER",
                "Audio buffer contains no data",
            )
            .with_causes(&[
                "Audio file failed to load",
                "Corrupted audio file",
                "Incorrect file path",
            ])
            .with_solutions(&[
                "Verify audio file exists and is readable",
                "Check audio file format support",
                "Validate file path and permissions",
            ]),
        );

        // Audio Processing Errors (E200-E299)
        self.register(
            ErrorCode::new(
                201,
                ErrorCategory::AudioProcessing,
                "FFT_PROCESSING_FAILED",
                "Fast Fourier Transform processing failed",
            )
            .with_causes(&[
                "Audio buffer size incompatible with FFT",
                "NaN or infinite values in audio",
                "FFT library internal error",
            ])
            .with_solutions(&[
                "Check audio for NaN or infinite values",
                "Ensure audio buffer size is valid",
                "Try different FFT frame size",
            ])
            .retryable(3, 500),
        );

        self.register(
            ErrorCode::new(
                202,
                ErrorCategory::AudioProcessing,
                "FEATURE_EXTRACTION_FAILED",
                "Audio feature extraction failed",
            )
            .with_causes(&[
                "Insufficient audio quality",
                "Audio contains only silence",
                "Unsupported audio characteristics",
            ])
            .with_solutions(&[
                "Check audio quality and content",
                "Verify audio is not silent",
                "Try different preprocessing parameters",
            ])
            .retryable(2, 1000),
        );

        self.register(
            ErrorCode::new(
                203,
                ErrorCategory::AudioProcessing,
                "ALIGNMENT_FAILED",
                "Audio alignment between reference and test signals failed",
            )
            .with_causes(&[
                "Audio signals too different for alignment",
                "One signal contains only noise",
                "Insufficient overlap between signals",
            ])
            .with_solutions(&[
                "Check that audio signals are related",
                "Verify both signals contain speech",
                "Try manual time alignment",
            ]),
        );

        // Resource Errors (E300-E399)
        self.register(
            ErrorCode::new(
                301,
                ErrorCategory::Resource,
                "INSUFFICIENT_MEMORY",
                "Not enough memory available for processing",
            )
            .with_causes(&[
                "Large audio files exceed available memory",
                "Memory leak in processing pipeline",
                "System under memory pressure",
            ])
            .with_solutions(&[
                "Process audio in smaller chunks",
                "Free memory by closing other applications",
                "Use streaming processing mode",
            ])
            .retryable(2, 2000),
        );

        self.register(
            ErrorCode::new(
                302,
                ErrorCategory::Resource,
                "PROCESSING_TIMEOUT",
                "Processing operation exceeded time limit",
            )
            .with_causes(&[
                "Audio file too large for available processing power",
                "System under high load",
                "Inefficient processing configuration",
            ])
            .with_solutions(&[
                "Process smaller audio segments",
                "Increase timeout limits",
                "Optimize processing configuration",
            ])
            .retryable(3, 5000),
        );

        // Configuration Errors (E400-E499)
        self.register(
            ErrorCode::new(
                401,
                ErrorCategory::Configuration,
                "INVALID_CONFIGURATION",
                "Evaluator configuration parameters are invalid",
            )
            .with_causes(&[
                "Parameter values outside valid ranges",
                "Incompatible parameter combinations",
                "Missing required configuration",
            ])
            .with_solutions(&[
                "Check parameter documentation",
                "Use default configuration",
                "Validate all parameters before use",
            ]),
        );

        // Network/IO Errors (E500-E599)
        self.register(
            ErrorCode::new(
                501,
                ErrorCategory::NetworkIO,
                "FILE_IO_ERROR",
                "File input/output operation failed",
            )
            .with_causes(&[
                "File permission denied",
                "Disk space insufficient",
                "File corrupted or inaccessible",
            ])
            .with_solutions(&[
                "Check file permissions",
                "Verify disk space availability",
                "Test file integrity",
            ])
            .retryable(3, 1000),
        );

        // Internal Errors (E600-E699)
        self.register(
            ErrorCode::new(
                601,
                ErrorCategory::Internal,
                "UNEXPECTED_INTERNAL_ERROR",
                "An unexpected internal error occurred",
            )
            .with_causes(&[
                "Software bug or logic error",
                "Corrupted internal state",
                "Unhandled edge case",
            ])
            .with_solutions(&[
                "Restart the evaluation process",
                "Report this error to developers",
                "Try with different input data",
            ])
            .retryable(1, 3000),
        );
    }
}

/// Enhanced error message builder for invalid input errors
pub struct ErrorMessageBuilder {
    context: String,
    expected: Option<String>,
    actual: Option<String>,
    suggestions: Vec<String>,
}

impl ErrorMessageBuilder {
    /// Create a new error message builder
    pub fn new(context: &str) -> Self {
        Self {
            context: context.to_string(),
            expected: None,
            actual: None,
            suggestions: Vec::new(),
        }
    }

    /// Set what was expected
    pub fn expected(mut self, expected: &str) -> Self {
        self.expected = Some(expected.to_string());
        self
    }

    /// Set what was actually received
    pub fn actual(mut self, actual: &str) -> Self {
        self.actual = Some(actual.to_string());
        self
    }

    /// Add a suggestion for how to fix the issue
    pub fn suggestion(mut self, suggestion: &str) -> Self {
        self.suggestions.push(suggestion.to_string());
        self
    }

    /// Add multiple suggestions
    pub fn suggestions(mut self, suggestions: &[&str]) -> Self {
        for suggestion in suggestions {
            self.suggestions.push(suggestion.to_string());
        }
        self
    }

    /// Build the final error message
    pub fn build(self) -> String {
        let mut message = format!("Invalid input for {context}", context = self.context);

        if let (Some(expected), Some(actual)) = (&self.expected, &self.actual) {
            message.push_str(&format!("\n  Expected: {expected}\n  Actual: {actual}"));
        } else if let Some(expected) = &self.expected {
            message.push_str(&format!("\n  Expected: {expected}"));
        } else if let Some(actual) = &self.actual {
            message.push_str(&format!("\n  Received: {actual}"));
        }

        if !self.suggestions.is_empty() {
            message.push_str("\n\nSuggestions:");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                message.push_str(&format!("\n  {}. {}", i + 1, suggestion));
            }
        }

        message
    }
}

/// Helper functions for common error scenarios

/// Create a sample rate mismatch error message
pub fn sample_rate_mismatch_error(
    component: &str,
    expected_rate: u32,
    actual_rate: u32,
    audio_type: &str,
) -> String {
    ErrorMessageBuilder::new(&format!("{component} evaluation"))
        .expected(&format!("{expected_rate} Hz sample rate"))
        .actual(&format!(
            "{} Hz sample rate in {} audio",
            actual_rate, audio_type
        ))
        .suggestions(&[
            &format!(
                "Resample your {} audio to {} Hz before evaluation",
                audio_type, expected_rate
            ),
            &format!(
                "Use a {} evaluator configured for {} Hz",
                component, actual_rate
            ),
            "Check your audio loading pipeline for sample rate conversion issues",
        ])
        .build()
}

/// Create a channel count mismatch error message
pub fn channel_mismatch_error(
    component: &str,
    expected_channels: u32,
    actual_channels: u32,
    audio_type: &str,
) -> String {
    let channel_name = |count: u32| match count {
        1 => String::from("mono"),
        2 => String::from("stereo"),
        n => format!("{n}-channel"),
    };

    ErrorMessageBuilder::new(&format!("{component} evaluation"))
        .expected(&format!("{} audio", channel_name(expected_channels)))
        .actual(&format!(
            "{} {} audio",
            channel_name(actual_channels),
            audio_type
        ))
        .suggestions(&[
            &format!(
                "Convert your {} audio to {} using audio conversion tools",
                audio_type,
                channel_name(expected_channels)
            ),
            "Use the audio::conversion module for automatic channel conversion",
            "Check your audio loading pipeline for proper channel handling",
        ])
        .build()
}

/// Create an audio length mismatch error message
pub fn length_mismatch_error(
    component: &str,
    min_length_seconds: f32,
    actual_length_seconds: f32,
    audio_type: &str,
) -> String {
    ErrorMessageBuilder::new(&format!("{component} evaluation"))
        .expected(&format!(
            "at least {:.2} seconds of audio",
            min_length_seconds
        ))
        .actual(&format!(
            "{:.2} seconds in {} audio",
            actual_length_seconds, audio_type
        ))
        .suggestions(&[
            &format!(
                "Provide {} audio with at least {:.2} seconds of content",
                audio_type, min_length_seconds
            ),
            "Check if your audio file was loaded correctly",
            "Verify that the audio file is not corrupted or truncated",
        ])
        .build()
}

/// Create an empty audio buffer error message
pub fn empty_audio_error(component: &str, audio_type: &str) -> String {
    ErrorMessageBuilder::new(&format!("{component} evaluation"))
        .expected("non-empty audio buffer")
        .actual(&format!("empty {} audio buffer", audio_type))
        .suggestions(&[
            &format!(
                "Ensure your {} audio file contains actual audio data",
                audio_type
            ),
            "Check if the audio file loaded correctly",
            "Verify that the audio file path is correct and the file exists",
            "Check for audio loading errors in your pipeline",
        ])
        .build()
}

/// Create a feature extraction error message
pub fn feature_extraction_error(component: &str, feature_type: &str, cause: &str) -> String {
    ErrorMessageBuilder::new(&format!("{} feature extraction", component))
        .actual(&format!(
            "failed to extract {} features: {}",
            feature_type, cause
        ))
        .suggestions(&[
            "Check if your audio has sufficient quality for feature extraction",
            "Verify that the audio is not silent or corrupted",
            "Try using different audio preprocessing parameters",
            "Check if the audio format is supported",
        ])
        .build()
}

/// Create a configuration error message
pub fn configuration_error(component: &str, parameter: &str, issue: &str) -> String {
    ErrorMessageBuilder::new(&format!("{} configuration", component))
        .actual(&format!("invalid {} parameter: {}", parameter, issue))
        .suggestions(&[
            &format!("Check the {} parameter documentation", parameter),
            "Use the default configuration if unsure",
            "Validate configuration parameters before creating the evaluator",
            "Check for parameter range and type requirements",
        ])
        .build()
}

/// Create a compatibility error message
pub fn compatibility_error(component: &str, reference_info: &str, degraded_info: &str) -> String {
    ErrorMessageBuilder::new(&format!("{component} evaluation"))
        .expected("compatible reference and degraded audio")
        .actual(&format!(
            "reference: {}, degraded: {}",
            reference_info, degraded_info
        ))
        .suggestions(&[
            "Ensure both audio files have the same sample rate",
            "Ensure both audio files have the same number of channels",
            "Use audio conversion tools to make the files compatible",
            "Check the audio loading pipeline for consistency",
        ])
        .build()
}

/// Create a resource error message
pub fn resource_error(component: &str, resource: &str, issue: &str) -> String {
    ErrorMessageBuilder::new(&format!("{component} evaluation"))
        .actual(&format!("resource problem with {}: {}", resource, issue))
        .suggestions(&[
            "Check system memory availability",
            "Reduce batch size or audio length if processing large files",
            "Close other applications to free up resources",
            "Check disk space if working with large audio files",
        ])
        .build()
}

/// Automatic retry mechanism for error recovery
pub struct RetryMechanism {
    registry: ErrorCodeRegistry,
}

impl RetryMechanism {
    /// Create new retry mechanism with error code registry
    pub fn new() -> Self {
        Self {
            registry: ErrorCodeRegistry::new(),
        }
    }

    /// Execute operation with automatic retry based on error type
    pub async fn execute_with_retry<T, E, F, Fut>(
        &self,
        operation: F,
        context: &str,
    ) -> Result<T, E>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        let mut attempts = 0;
        let max_attempts = 5; // Global maximum

        loop {
            attempts += 1;

            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    // Try to extract error code from error message
                    let error_str = error.to_string();
                    let error_code = self.extract_error_code(&error_str);

                    if let Some(code) = error_code {
                        if let Some(error_info) = self.registry.get(code) {
                            if error_info.retryable
                                && attempts < error_info.max_retries.min(max_attempts as u8)
                            {
                                // Calculate exponential backoff delay
                                let delay_ms =
                                    error_info.retry_delay_ms * (2_u64.pow((attempts - 1) as u32));

                                eprintln!(
                                    "Operation failed with retryable error {} in {}, retrying in {}ms (attempt {}/{})",
                                    code, context, delay_ms, attempts, error_info.max_retries
                                );

                                sleep(Duration::from_millis(delay_ms)).await;
                                continue;
                            }
                        }
                    }

                    // Error is not retryable or max attempts reached
                    if attempts > 1 {
                        eprintln!(
                            "Operation failed in {} after {} attempts: {}",
                            context, attempts, error
                        );
                    }
                    return Err(error);
                }
            }
        }
    }

    /// Execute operation with custom retry configuration
    pub async fn execute_with_custom_retry<T, E, F, Fut>(
        &self,
        operation: F,
        context: &str,
        max_retries: u8,
        base_delay_ms: u64,
        use_exponential_backoff: bool,
    ) -> Result<T, E>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        let mut attempts = 0;

        loop {
            attempts += 1;

            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if attempts <= max_retries {
                        let delay_ms = if use_exponential_backoff {
                            base_delay_ms * (2_u64.pow((attempts - 1) as u32))
                        } else {
                            base_delay_ms
                        };

                        eprintln!(
                            "Operation failed in {}, retrying in {}ms (attempt {}/{}): {}",
                            context, delay_ms, attempts, max_retries, error
                        );

                        sleep(Duration::from_millis(delay_ms)).await;
                        continue;
                    }

                    eprintln!(
                        "Operation failed in {} after {} attempts: {}",
                        context, attempts, error
                    );
                    return Err(error);
                }
            }
        }
    }

    /// Extract error code from error message
    fn extract_error_code(&self, error_message: &str) -> Option<u16> {
        // Look for patterns like "Error 201:" or "E201:"
        if let Some(start) = error_message.find("Error ") {
            let after_error = &error_message[start + 6..];
            if let Some(colon_pos) = after_error.find(':') {
                let code_str = &after_error[..colon_pos];
                if let Ok(code) = code_str.parse::<u16>() {
                    return Some(code);
                }
            }
        }

        // Try alternative pattern "E201:"
        if let Some(start) = error_message.find("E") {
            let after_e = &error_message[start + 1..];
            if let Some(colon_pos) = after_e.find(':') {
                let code_str = &after_e[..colon_pos];
                if let Ok(code) = code_str.parse::<u16>() {
                    return Some(code);
                }
            }
        }

        None
    }

    /// Get error code information
    pub fn get_error_info(&self, code: u16) -> Option<&ErrorCode> {
        self.registry.get(code)
    }

    /// Generate documentation for all error codes
    pub fn generate_error_documentation(&self) -> String {
        let mut doc = String::from("# VoiRS Evaluation Error Code Documentation\n\n");

        for category in [
            ErrorCategory::InputValidation,
            ErrorCategory::AudioProcessing,
            ErrorCategory::Resource,
            ErrorCategory::Configuration,
            ErrorCategory::NetworkIO,
            ErrorCategory::Internal,
        ] {
            let codes = self.registry.get_by_category(&category);
            if !codes.is_empty() {
                doc.push_str(&format!("## {:?} Errors\n\n", category));

                for code in codes {
                    doc.push_str(&format!("### Error {}: {}\n\n", code.code, code.name));
                    doc.push_str(&format!("**Description:** {}\n\n", code.description));

                    if !code.common_causes.is_empty() {
                        doc.push_str("**Common Causes:**\n");
                        for cause in &code.common_causes {
                            doc.push_str(&format!("- {}\n", cause));
                        }
                        doc.push('\n');
                    }

                    if !code.solutions.is_empty() {
                        doc.push_str("**Solutions:**\n");
                        for solution in &code.solutions {
                            doc.push_str(&format!("- {}\n", solution));
                        }
                        doc.push('\n');
                    }

                    if code.retryable {
                        doc.push_str(&format!(
                            "**Retry Policy:** Retryable up to {} times with {}ms base delay\n\n",
                            code.max_retries, code.retry_delay_ms
                        ));
                    } else {
                        doc.push_str("**Retry Policy:** Not retryable\n\n");
                    }

                    doc.push_str("---\n\n");
                }
            }
        }

        doc
    }
}

impl Default for RetryMechanism {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_message_builder() {
        let message = ErrorMessageBuilder::new("PESQ evaluation")
            .expected("16000 Hz sample rate")
            .actual("44100 Hz sample rate")
            .suggestion("Resample your audio to 16000 Hz")
            .suggestion("Use a different evaluator")
            .build();

        assert!(message.contains("Invalid input for PESQ evaluation"));
        assert!(message.contains("Expected: 16000 Hz sample rate"));
        assert!(message.contains("Actual: 44100 Hz sample rate"));
        assert!(message.contains("Suggestions:"));
        assert!(message.contains("1. Resample your audio"));
        assert!(message.contains("2. Use a different evaluator"));
    }

    #[test]
    fn test_sample_rate_mismatch_error() {
        let message = sample_rate_mismatch_error("PESQ", 16000, 44100, "reference");

        assert!(message.contains("PESQ evaluation"));
        assert!(message.contains("16000 Hz"));
        assert!(message.contains("44100 Hz"));
        assert!(message.contains("reference audio"));
        assert!(message.contains("Resample"));
    }

    #[test]
    fn test_channel_mismatch_error() {
        let message = channel_mismatch_error("PESQ", 1, 2, "degraded");

        assert!(message.contains("PESQ evaluation"));
        assert!(message.contains("mono"));
        assert!(message.contains("stereo"));
        assert!(message.contains("degraded audio"));
    }

    #[test]
    fn test_empty_audio_error() {
        let message = empty_audio_error("MCD", "reference");

        assert!(message.contains("MCD evaluation"));
        assert!(message.contains("empty reference audio"));
        assert!(message.contains("non-empty audio buffer"));
    }
}
