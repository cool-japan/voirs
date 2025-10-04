//! Batch processing commands.
//!
//! This module provides commands for batch processing multiple text inputs,
//! supporting various input formats (TXT, CSV, JSON) and parallel processing.

use crate::GlobalOptions;
use std::path::PathBuf;
use voirs_sdk::config::AppConfig;
use voirs_sdk::{AudioFormat, QualityLevel, Result};

pub mod files;
pub mod parallel;
pub mod resume;

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Input file or directory
    pub input_path: PathBuf,
    /// Output directory
    pub output_dir: PathBuf,
    /// Number of parallel workers
    pub workers: usize,
    /// Quality level for synthesis
    pub quality: QualityLevel,
    /// Speaking rate
    pub speaking_rate: f32,
    /// Pitch adjustment
    pub pitch: f32,
    /// Volume gain
    pub volume: f32,
    /// Audio output format
    pub format: AudioFormat,
    /// Enable resume functionality
    pub enable_resume: bool,
    /// Maximum retries for failed items
    pub max_retries: u32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            input_path: PathBuf::new(),
            output_dir: PathBuf::new(),
            workers: num_cpus::get(),
            quality: QualityLevel::High,
            speaking_rate: 1.0,
            pitch: 0.0,
            volume: 0.0,
            format: AudioFormat::Wav,
            enable_resume: true,
            max_retries: 3,
        }
    }
}

/// Run batch processing command
pub async fn run_batch_process(
    input: &PathBuf,
    output_dir: Option<&PathBuf>,
    workers: Option<usize>,
    quality: QualityLevel,
    rate: f32,
    pitch: f32,
    volume: f32,
    resume: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    // Create batch configuration
    let mut batch_config = BatchConfig {
        input_path: input.clone(),
        output_dir: output_dir.cloned().unwrap_or_else(|| {
            input
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf()
        }),
        workers: workers.unwrap_or_else(num_cpus::get),
        quality,
        speaking_rate: rate,
        pitch,
        volume,
        format: AudioFormat::Wav, // Default to WAV format
        enable_resume: resume,
        max_retries: 3,
    };

    // Ensure output directory exists
    std::fs::create_dir_all(&batch_config.output_dir)?;

    if !global.quiet {
        println!("Batch Processing Configuration:");
        println!("==============================");
        println!("Input: {}", batch_config.input_path.display());
        println!("Output: {}", batch_config.output_dir.display());
        println!("Workers: {}", batch_config.workers);
        println!("Quality: {:?}", batch_config.quality);
        println!("Resume: {}", batch_config.enable_resume);
        println!();
    }

    // Detect input format and process
    if batch_config.input_path.is_file() {
        files::process_file(&batch_config, config, global).await
    } else if batch_config.input_path.is_dir() {
        files::process_directory(&batch_config, config, global).await
    } else {
        Err(voirs_sdk::VoirsError::config_error(&format!(
            "Input path does not exist: {}",
            batch_config.input_path.display()
        )))
    }
}

/// Get supported input file extensions
pub fn get_supported_extensions() -> Vec<&'static str> {
    vec!["txt", "csv", "json", "jsonl"]
}

/// Check if file extension is supported
pub fn is_supported_extension(path: &PathBuf) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        get_supported_extensions().contains(&ext.to_lowercase().as_str())
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert!(config.workers > 0);
        assert_eq!(config.quality, QualityLevel::High);
        assert!(config.enable_resume);
    }

    #[test]
    fn test_get_supported_extensions() {
        let extensions = get_supported_extensions();
        assert!(extensions.contains(&"txt"));
        assert!(extensions.contains(&"csv"));
        assert!(extensions.contains(&"json"));
    }

    #[test]
    fn test_is_supported_extension() {
        assert!(is_supported_extension(&PathBuf::from("test.txt")));
        assert!(is_supported_extension(&PathBuf::from("data.csv")));
        assert!(is_supported_extension(&PathBuf::from("config.json")));
        assert!(!is_supported_extension(&PathBuf::from("image.png")));
        assert!(!is_supported_extension(&PathBuf::from("noextension")));
    }
}
