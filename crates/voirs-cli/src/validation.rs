//! Input validation utilities for VoiRS CLI
//!
//! This module provides comprehensive validation functions for user inputs,
//! configuration values, and command parameters. It ensures data integrity
//! and provides helpful error messages for invalid inputs.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Validation result with helpful error messages
pub type ValidationResult<T> = Result<T>;

/// Voice name validation
pub struct VoiceValidator;

impl VoiceValidator {
    /// Validate voice name format
    ///
    /// Voice names must:
    /// - Be 1-64 characters long
    /// - Contain only alphanumeric characters, hyphens, and underscores
    /// - Start with an alphabetic character
    pub fn validate_name(name: &str) -> ValidationResult<()> {
        if name.is_empty() {
            bail!("Voice name cannot be empty");
        }

        if name.len() > 64 {
            bail!("Voice name too long (max 64 characters): {}", name.len());
        }

        if !name.chars().next().is_some_and(|c| c.is_alphabetic()) {
            bail!("Voice name must start with a letter: '{}'", name);
        }

        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            bail!(
                "Voice name contains invalid characters: '{}'.\nAllowed: letters, numbers, hyphens, underscores",
                name
            );
        }

        Ok(())
    }

    /// Validate language code (ISO 639-1 or 639-3 format)
    pub fn validate_language(lang: &str) -> ValidationResult<()> {
        let len = lang.len();
        if len != 2 && len != 3 {
            bail!(
                "Invalid language code '{}'. Expected 2-letter (ISO 639-1) or 3-letter (ISO 639-3) code",
                lang
            );
        }

        if !lang.chars().all(|c| c.is_ascii_lowercase()) {
            bail!("Language code must be lowercase: '{}'", lang);
        }

        Ok(())
    }
}

/// Audio parameter validation
pub struct AudioValidator;

impl AudioValidator {
    /// Validate sample rate (must be positive and within reasonable range)
    pub fn validate_sample_rate(rate: u32) -> ValidationResult<u32> {
        const MIN_RATE: u32 = 8000;
        const MAX_RATE: u32 = 192000;

        if rate < MIN_RATE {
            bail!(
                "Sample rate too low: {} Hz (minimum: {} Hz)",
                rate,
                MIN_RATE
            );
        }

        if rate > MAX_RATE {
            bail!(
                "Sample rate too high: {} Hz (maximum: {} Hz)",
                rate,
                MAX_RATE
            );
        }

        // Warn if not a standard rate
        const STANDARD_RATES: &[u32] = &[8000, 11025, 16000, 22050, 44100, 48000, 96000, 192000];
        if !STANDARD_RATES.contains(&rate) {
            tracing::warn!(
                "Non-standard sample rate: {} Hz. Standard rates: {:?}",
                rate,
                STANDARD_RATES
            );
        }

        Ok(rate)
    }

    /// Validate speed factor (0.25 - 4.0)
    pub fn validate_speed(speed: f32) -> ValidationResult<f32> {
        const MIN_SPEED: f32 = 0.25;
        const MAX_SPEED: f32 = 4.0;

        if !speed.is_finite() {
            bail!("Speed must be a finite number");
        }

        if speed < MIN_SPEED {
            bail!("Speed too low: {:.2} (minimum: {:.2})", speed, MIN_SPEED);
        }

        if speed > MAX_SPEED {
            bail!("Speed too high: {:.2} (maximum: {:.2})", speed, MAX_SPEED);
        }

        Ok(speed)
    }

    /// Validate pitch adjustment (-12.0 to +12.0 semitones)
    pub fn validate_pitch(pitch: f32) -> ValidationResult<f32> {
        const MIN_PITCH: f32 = -12.0;
        const MAX_PITCH: f32 = 12.0;

        if !pitch.is_finite() {
            bail!("Pitch must be a finite number");
        }

        if pitch < MIN_PITCH {
            bail!(
                "Pitch too low: {:.1} semitones (minimum: {:.1})",
                pitch,
                MIN_PITCH
            );
        }

        if pitch > MAX_PITCH {
            bail!(
                "Pitch too high: {:.1} semitones (maximum: {:.1})",
                pitch,
                MAX_PITCH
            );
        }

        Ok(pitch)
    }

    /// Validate volume level (0.0 - 2.0)
    pub fn validate_volume(volume: f32) -> ValidationResult<f32> {
        const MIN_VOLUME: f32 = 0.0;
        const MAX_VOLUME: f32 = 2.0;

        if !volume.is_finite() {
            bail!("Volume must be a finite number");
        }

        if volume < MIN_VOLUME {
            bail!("Volume cannot be negative: {:.2}", volume);
        }

        if volume > MAX_VOLUME {
            bail!(
                "Volume too high: {:.2} (maximum: {:.2})",
                volume,
                MAX_VOLUME
            );
        }

        if volume > 1.0 {
            tracing::warn!("Volume above 1.0 may cause clipping: {:.2}", volume);
        }

        Ok(volume)
    }
}

/// File path validation
pub struct PathValidator;

impl PathValidator {
    /// Validate input file exists and is readable
    pub fn validate_input_file(path: &Path) -> ValidationResult<PathBuf> {
        if !path.exists() {
            bail!("Input file not found: {}", path.display());
        }

        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }

        // Check if file is readable by attempting to get metadata
        std::fs::metadata(path)
            .with_context(|| format!("Cannot access file: {}", path.display()))?;

        Ok(path.to_path_buf())
    }

    /// Validate output path is writable
    pub fn validate_output_path(path: &Path) -> ValidationResult<PathBuf> {
        // Check parent directory exists and is writable
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                bail!("Output directory does not exist: {}", parent.display());
            }

            if !parent.is_dir() {
                bail!("Output parent is not a directory: {}", parent.display());
            }

            // Try to check write permissions
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let metadata = std::fs::metadata(parent)
                    .with_context(|| format!("Cannot access directory: {}", parent.display()))?;
                let permissions = metadata.permissions();
                if permissions.mode() & 0o200 == 0 {
                    bail!("Output directory is not writable: {}", parent.display());
                }
            }
        }

        // Warn if file already exists
        if path.exists() {
            tracing::warn!(
                "Output file already exists and will be overwritten: {}",
                path.display()
            );
        }

        Ok(path.to_path_buf())
    }

    /// Validate directory exists and is accessible
    pub fn validate_directory(path: &Path) -> ValidationResult<PathBuf> {
        if !path.exists() {
            bail!("Directory not found: {}", path.display());
        }

        if !path.is_dir() {
            bail!("Path is not a directory: {}", path.display());
        }

        Ok(path.to_path_buf())
    }

    /// Validate and create output directory if it doesn't exist
    pub fn ensure_output_directory(path: &Path) -> ValidationResult<PathBuf> {
        if !path.exists() {
            std::fs::create_dir_all(path)
                .with_context(|| format!("Failed to create directory: {}", path.display()))?;
            tracing::info!("Created output directory: {}", path.display());
        } else if !path.is_dir() {
            bail!("Path exists but is not a directory: {}", path.display());
        }

        Ok(path.to_path_buf())
    }
}

/// Text input validation
pub struct TextValidator;

impl TextValidator {
    /// Validate text is not empty after trimming
    pub fn validate_non_empty(text: &str) -> ValidationResult<()> {
        if text.trim().is_empty() {
            bail!("Text input cannot be empty");
        }
        Ok(())
    }

    /// Validate text length is within limits
    pub fn validate_length(text: &str, max_len: usize) -> ValidationResult<()> {
        if text.len() > max_len {
            bail!(
                "Text too long: {} characters (maximum: {})",
                text.len(),
                max_len
            );
        }
        Ok(())
    }

    /// Validate SSML structure (basic check)
    pub fn validate_ssml_basic(text: &str) -> ValidationResult<()> {
        if !text.contains("<speak>") {
            bail!("SSML must contain <speak> root element");
        }

        if !text.contains("</speak>") {
            bail!("SSML <speak> element is not closed");
        }

        // Check for balanced tags (simplified)
        let open_count = text.matches('<').count();
        let close_count = text.matches('>').count();

        if open_count != close_count {
            bail!(
                "Unbalanced XML tags in SSML (found {} '<' and {} '>')",
                open_count,
                close_count
            );
        }

        Ok(())
    }
}

/// Batch processing validation
pub struct BatchValidator;

impl BatchValidator {
    /// Validate batch size is reasonable
    pub fn validate_batch_size(size: usize) -> ValidationResult<usize> {
        const MAX_BATCH_SIZE: usize = 10000;

        if size == 0 {
            bail!("Batch size must be greater than 0");
        }

        if size > MAX_BATCH_SIZE {
            bail!(
                "Batch size too large: {} (maximum: {})",
                size,
                MAX_BATCH_SIZE
            );
        }

        Ok(size)
    }

    /// Validate worker count
    pub fn validate_workers(workers: usize) -> ValidationResult<usize> {
        let max_workers = num_cpus::get() * 2;

        if workers == 0 {
            bail!("Worker count must be greater than 0");
        }

        if workers > max_workers {
            tracing::warn!(
                "Worker count {} exceeds recommended maximum {} (2x CPU cores)",
                workers,
                max_workers
            );
        }

        Ok(workers)
    }
}

/// Model validation
pub struct ModelValidator;

impl ModelValidator {
    /// Validate model file exists and has correct extension
    pub fn validate_model_file(path: &Path) -> ValidationResult<()> {
        if !path.exists() {
            bail!("Model file not found: {}", path.display());
        }

        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }

        // Check extension
        let ext = path.extension().and_then(|e| e.to_str());
        match ext {
            Some("safetensors") | Some("st") | Some("pt") | Some("pth") | Some("bin")
            | Some("onnx") => Ok(()),
            Some(e) => bail!(
                "Unsupported model format: '{}'. Supported: safetensors, pt, pth, bin, onnx",
                e
            ),
            None => bail!("Model file has no extension: {}", path.display()),
        }
    }

    /// Validate model file size is reasonable
    pub fn validate_model_size(path: &Path, max_size_mb: Option<u64>) -> ValidationResult<u64> {
        let metadata = std::fs::metadata(path)
            .with_context(|| format!("Cannot access file: {}", path.display()))?;

        let size_bytes = metadata.len();
        let size_mb = size_bytes / 1_048_576;

        // Check minimum size (models should be at least 1KB)
        if size_bytes < 1024 {
            bail!(
                "Model file is suspiciously small: {} bytes. May be corrupted.",
                size_bytes
            );
        }

        // Check maximum size if specified
        if let Some(max_mb) = max_size_mb {
            if size_mb > max_mb {
                bail!(
                    "Model file too large: {} MB (maximum: {} MB)",
                    size_mb,
                    max_mb
                );
            }
        }

        // Warn if very large
        if size_mb > 1000 {
            tracing::warn!(
                "Model file is very large: {} MB. Loading may take significant time.",
                size_mb
            );
        }

        Ok(size_bytes)
    }

    /// Validate model type matches expected type
    pub fn validate_model_type(model_path: &Path, expected_type: &str) -> ValidationResult<()> {
        let filename = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid filename"))?
            .to_lowercase();

        let matches = match expected_type {
            "vocoder" => {
                filename.contains("vocoder")
                    || filename.contains("hifigan")
                    || filename.contains("diffwave")
            }
            "acoustic" => {
                filename.contains("acoustic")
                    || filename.contains("vits")
                    || filename.contains("fastspeech")
            }
            "g2p" => filename.contains("g2p") || filename.contains("phoneme"),
            _ => true, // Unknown type, allow any
        };

        if !matches {
            tracing::warn!(
                "Model filename '{}' may not match expected type '{}'",
                filename,
                expected_type
            );
        }

        Ok(())
    }

    /// Validate model directory structure
    pub fn validate_model_directory(path: &Path) -> ValidationResult<()> {
        if !path.exists() {
            bail!("Model directory not found: {}", path.display());
        }

        if !path.is_dir() {
            bail!("Path is not a directory: {}", path.display());
        }

        // Check for config files
        let config_json = path.join("config.json");
        let has_config = config_json.exists();

        if !has_config {
            tracing::warn!(
                "No config.json found in model directory: {}",
                path.display()
            );
        }

        Ok(())
    }
}

/// Configuration validation
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate port number
    pub fn validate_port(port: u16) -> ValidationResult<u16> {
        const MIN_PORT: u16 = 1024;
        const MAX_PORT: u16 = 65535;

        if port < MIN_PORT {
            bail!(
                "Port number too low: {} (minimum: {} to avoid privileged ports)",
                port,
                MIN_PORT
            );
        }

        if port > MAX_PORT {
            bail!("Port number too high: {} (maximum: {})", port, MAX_PORT);
        }

        Ok(port)
    }

    /// Validate timeout duration (in seconds)
    pub fn validate_timeout(timeout_secs: u64) -> ValidationResult<u64> {
        const MAX_TIMEOUT: u64 = 3600; // 1 hour

        if timeout_secs == 0 {
            bail!("Timeout must be greater than 0");
        }

        if timeout_secs > MAX_TIMEOUT {
            bail!(
                "Timeout too long: {} seconds (maximum: {} seconds / 1 hour)",
                timeout_secs,
                MAX_TIMEOUT
            );
        }

        Ok(timeout_secs)
    }

    /// Validate buffer size
    pub fn validate_buffer_size(size: usize) -> ValidationResult<usize> {
        const MIN_BUFFER: usize = 64;
        const MAX_BUFFER: usize = 8192;

        if size < MIN_BUFFER {
            bail!("Buffer size too small: {} (minimum: {})", size, MIN_BUFFER);
        }

        if size > MAX_BUFFER {
            bail!("Buffer size too large: {} (maximum: {})", size, MAX_BUFFER);
        }

        // Check if power of 2
        if size & (size - 1) != 0 {
            tracing::warn!(
                "Buffer size {} is not a power of 2, may affect performance",
                size
            );
        }

        Ok(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_voice_name_validation() {
        assert!(VoiceValidator::validate_name("myvoice").is_ok());
        assert!(VoiceValidator::validate_name("my_voice").is_ok());
        assert!(VoiceValidator::validate_name("my-voice-123").is_ok());

        assert!(VoiceValidator::validate_name("").is_err());
        assert!(VoiceValidator::validate_name("123voice").is_err());
        assert!(VoiceValidator::validate_name("my voice").is_err());
        assert!(VoiceValidator::validate_name("my.voice").is_err());
    }

    #[test]
    fn test_language_validation() {
        assert!(VoiceValidator::validate_language("en").is_ok());
        assert!(VoiceValidator::validate_language("ja").is_ok());
        assert!(VoiceValidator::validate_language("eng").is_ok());

        assert!(VoiceValidator::validate_language("EN").is_err());
        assert!(VoiceValidator::validate_language("e").is_err());
        assert!(VoiceValidator::validate_language("english").is_err());
    }

    #[test]
    fn test_sample_rate_validation() {
        assert!(AudioValidator::validate_sample_rate(44100).is_ok());
        assert!(AudioValidator::validate_sample_rate(48000).is_ok());

        assert!(AudioValidator::validate_sample_rate(1000).is_err());
        assert!(AudioValidator::validate_sample_rate(300000).is_err());
    }

    #[test]
    fn test_speed_validation() {
        assert!(AudioValidator::validate_speed(1.0).is_ok());
        assert!(AudioValidator::validate_speed(0.5).is_ok());
        assert!(AudioValidator::validate_speed(2.0).is_ok());

        assert!(AudioValidator::validate_speed(0.1).is_err());
        assert!(AudioValidator::validate_speed(5.0).is_err());
        assert!(AudioValidator::validate_speed(f32::NAN).is_err());
        assert!(AudioValidator::validate_speed(f32::INFINITY).is_err());
    }

    #[test]
    fn test_pitch_validation() {
        assert!(AudioValidator::validate_pitch(0.0).is_ok());
        assert!(AudioValidator::validate_pitch(6.0).is_ok());
        assert!(AudioValidator::validate_pitch(-6.0).is_ok());

        assert!(AudioValidator::validate_pitch(15.0).is_err());
        assert!(AudioValidator::validate_pitch(-15.0).is_err());
    }

    #[test]
    fn test_volume_validation() {
        assert!(AudioValidator::validate_volume(1.0).is_ok());
        assert!(AudioValidator::validate_volume(0.5).is_ok());

        assert!(AudioValidator::validate_volume(-0.1).is_err());
        assert!(AudioValidator::validate_volume(3.0).is_err());
    }

    #[test]
    fn test_text_validation() {
        assert!(TextValidator::validate_non_empty("Hello").is_ok());
        assert!(TextValidator::validate_non_empty("  Hello  ").is_ok());

        assert!(TextValidator::validate_non_empty("").is_err());
        assert!(TextValidator::validate_non_empty("   ").is_err());
    }

    #[test]
    fn test_text_length_validation() {
        assert!(TextValidator::validate_length("Hello", 10).is_ok());
        assert!(TextValidator::validate_length("Hello", 5).is_ok());

        assert!(TextValidator::validate_length("Hello World", 5).is_err());
    }

    #[test]
    fn test_ssml_validation() {
        assert!(TextValidator::validate_ssml_basic("<speak>Hello</speak>").is_ok());

        assert!(TextValidator::validate_ssml_basic("Hello").is_err());
        assert!(TextValidator::validate_ssml_basic("<speak>Hello").is_err());
        assert!(TextValidator::validate_ssml_basic("<speak>Hello<").is_err());
    }

    #[test]
    fn test_batch_size_validation() {
        assert!(BatchValidator::validate_batch_size(100).is_ok());
        assert!(BatchValidator::validate_batch_size(1).is_ok());

        assert!(BatchValidator::validate_batch_size(0).is_err());
        assert!(BatchValidator::validate_batch_size(20000).is_err());
    }

    #[test]
    fn test_worker_validation() {
        assert!(BatchValidator::validate_workers(1).is_ok());
        assert!(BatchValidator::validate_workers(4).is_ok());

        assert!(BatchValidator::validate_workers(0).is_err());
    }

    #[test]
    fn test_port_validation() {
        assert!(ConfigValidator::validate_port(8080).is_ok());
        assert!(ConfigValidator::validate_port(3000).is_ok());
        assert!(ConfigValidator::validate_port(65535).is_ok());

        assert!(ConfigValidator::validate_port(80).is_err());
        assert!(ConfigValidator::validate_port(1000).is_err());
    }

    #[test]
    fn test_timeout_validation() {
        assert!(ConfigValidator::validate_timeout(30).is_ok());
        assert!(ConfigValidator::validate_timeout(300).is_ok());

        assert!(ConfigValidator::validate_timeout(0).is_err());
        assert!(ConfigValidator::validate_timeout(5000).is_err());
    }

    #[test]
    fn test_buffer_size_validation() {
        assert!(ConfigValidator::validate_buffer_size(512).is_ok());
        assert!(ConfigValidator::validate_buffer_size(1024).is_ok());

        assert!(ConfigValidator::validate_buffer_size(32).is_err());
        assert!(ConfigValidator::validate_buffer_size(10000).is_err());
    }

    #[test]
    fn test_input_file_validation() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_input.txt");
        std::fs::write(&test_file, "test").unwrap();

        assert!(PathValidator::validate_input_file(&test_file).is_ok());
        assert!(PathValidator::validate_input_file(Path::new("/nonexistent/file.txt")).is_err());

        std::fs::remove_file(&test_file).ok();
    }

    #[test]
    fn test_directory_validation() {
        let temp_dir = env::temp_dir();

        assert!(PathValidator::validate_directory(&temp_dir).is_ok());
        assert!(PathValidator::validate_directory(Path::new("/nonexistent/dir")).is_err());
    }

    #[test]
    fn test_model_file_validation() {
        let temp_dir = env::temp_dir();

        // Test valid model file
        let valid_model = temp_dir.join("test_model_file_validation.safetensors");
        std::fs::write(&valid_model, "test").unwrap();
        assert!(ModelValidator::validate_model_file(&valid_model).is_ok());

        // Test invalid extension
        let invalid_model = temp_dir.join("test_model.txt");
        std::fs::write(&invalid_model, "test").unwrap();
        assert!(ModelValidator::validate_model_file(&invalid_model).is_err());

        // Test nonexistent file
        assert!(
            ModelValidator::validate_model_file(Path::new("/nonexistent/model.safetensors"))
                .is_err()
        );

        // Cleanup
        std::fs::remove_file(&valid_model).ok();
        std::fs::remove_file(&invalid_model).ok();
    }

    #[test]
    fn test_model_size_validation() {
        let temp_dir = env::temp_dir();
        let model_file = temp_dir.join("test_model_size_validation.safetensors");

        // Test reasonable size
        std::fs::write(&model_file, vec![0u8; 10240]).unwrap(); // 10KB
        assert!(ModelValidator::validate_model_size(&model_file, Some(100)).is_ok());

        // Test too small
        std::fs::write(&model_file, vec![0u8; 512]).unwrap(); // 512 bytes
        assert!(ModelValidator::validate_model_size(&model_file, None).is_err());

        // Cleanup
        std::fs::remove_file(&model_file).ok();
    }

    #[test]
    fn test_model_type_validation() {
        let temp_dir = env::temp_dir();

        let vocoder_model = temp_dir.join("hifigan_vocoder.safetensors");
        std::fs::write(&vocoder_model, "test").unwrap();
        assert!(ModelValidator::validate_model_type(&vocoder_model, "vocoder").is_ok());

        let acoustic_model = temp_dir.join("vits_acoustic.safetensors");
        std::fs::write(&acoustic_model, "test").unwrap();
        assert!(ModelValidator::validate_model_type(&acoustic_model, "acoustic").is_ok());

        // Cleanup
        std::fs::remove_file(&vocoder_model).ok();
        std::fs::remove_file(&acoustic_model).ok();
    }

    #[test]
    fn test_model_directory_validation() {
        let temp_dir = env::temp_dir();
        let model_dir = temp_dir.join("test_model_dir");
        std::fs::create_dir_all(&model_dir).unwrap();

        assert!(ModelValidator::validate_model_directory(&model_dir).is_ok());
        assert!(
            ModelValidator::validate_model_directory(Path::new("/nonexistent/model_dir")).is_err()
        );

        // Cleanup
        std::fs::remove_dir_all(&model_dir).ok();
    }
}
