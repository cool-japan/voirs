//! Export utilities for various formats
//!
//! This module provides export functionality for different machine learning
//! frameworks and data formats.

pub mod huggingface;
pub mod pytorch;
pub mod tensorflow;
pub mod generic;

use crate::{DatasetSample, DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Export format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// HuggingFace Datasets format
    HuggingFace,
    /// PyTorch format
    PyTorch,
    /// TensorFlow format
    TensorFlow,
    /// JSON Lines format
    JsonLines,
    /// CSV format
    Csv,
    /// Parquet format
    Parquet,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Export format
    pub format: ExportFormat,
    /// Output path
    pub output_path: String,
    /// Include audio files
    pub include_audio: bool,
    /// Audio format for export
    pub audio_format: Option<crate::AudioFormat>,
    /// Compression level (0-9)
    pub compression_level: Option<u8>,
    /// Split into chunks
    pub chunk_size: Option<usize>,
    /// Generate manifest file
    pub generate_manifest: bool,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::JsonLines,
            output_path: "output".to_string(),
            include_audio: true,
            audio_format: Some(crate::AudioFormat::Wav),
            compression_level: None,
            chunk_size: None,
            generate_manifest: true,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Dataset exporter
pub struct DatasetExporter {
    config: ExportConfig,
}

impl DatasetExporter {
    /// Create new exporter with configuration
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }
    
    /// Export dataset samples
    pub fn export_samples(&self, samples: &[DatasetSample]) -> Result<()> {
        match self.config.format {
            ExportFormat::JsonLines => self.export_jsonlines(samples),
            ExportFormat::Csv => self.export_csv(samples),
            ExportFormat::HuggingFace => self.export_huggingface(samples),
            ExportFormat::PyTorch => self.export_pytorch(samples),
            ExportFormat::TensorFlow => self.export_tensorflow(samples),
            ExportFormat::Parquet => self.export_parquet(samples),
        }
    }
    
    /// Export as JSON Lines format
    fn export_jsonlines(&self, samples: &[DatasetSample]) -> Result<()> {
        use std::io::Write;
        
        let output_path = Path::new(&self.config.output_path);
        std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))?;
        
        let mut file = std::fs::File::create(output_path.with_extension("jsonl"))?;
        
        for sample in samples {
            let export_sample = ExportSample::from_dataset_sample(sample, &self.config);
            let json_line = serde_json::to_string(&export_sample)
                .map_err(|e| DatasetError::FormatError(format!("JSON serialization failed: {}", e)))?;
            
            writeln!(file, "{}", json_line)?;
        }
        
        Ok(())
    }
    
    /// Export as CSV format
    fn export_csv(&self, samples: &[DatasetSample]) -> Result<()> {
        let output_path = Path::new(&self.config.output_path);
        std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))?;
        
        let mut writer = csv::Writer::from_path(output_path.with_extension("csv"))?;
        
        // Write header
        writer.write_record(&["id", "text", "audio_path", "speaker_id", "language", "duration"])?;
        
        for sample in samples {
            let audio_path = if self.config.include_audio {
                format!("{}/{}.wav", self.config.output_path, sample.id)
            } else {
                String::new()
            };
            
            writer.write_record(&[
                &sample.id,
                &sample.text,
                &audio_path,
                &sample.speaker_id().unwrap_or("").to_string(),
                sample.language.as_str(),
                &sample.duration().to_string(),
            ])?;
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// Export as HuggingFace format (placeholder)
    fn export_huggingface(&self, _samples: &[DatasetSample]) -> Result<()> {
        // TODO: Implement HuggingFace Datasets export
        Err(DatasetError::FormatError("HuggingFace export not implemented".to_string()))
    }
    
    /// Export as PyTorch format (placeholder)
    fn export_pytorch(&self, _samples: &[DatasetSample]) -> Result<()> {
        // TODO: Implement PyTorch export
        Err(DatasetError::FormatError("PyTorch export not implemented".to_string()))
    }
    
    /// Export as TensorFlow format (placeholder)
    fn export_tensorflow(&self, _samples: &[DatasetSample]) -> Result<()> {
        // TODO: Implement TensorFlow export
        Err(DatasetError::FormatError("TensorFlow export not implemented".to_string()))
    }
    
    /// Export as Parquet format (placeholder)
    fn export_parquet(&self, _samples: &[DatasetSample]) -> Result<()> {
        // TODO: Implement Parquet export
        Err(DatasetError::FormatError("Parquet export not implemented".to_string()))
    }
}

/// Export sample representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSample {
    /// Sample ID
    pub id: String,
    /// Text content
    pub text: String,
    /// Audio file path (if included)
    pub audio_path: Option<String>,
    /// Speaker ID
    pub speaker_id: Option<String>,
    /// Language code
    pub language: String,
    /// Duration in seconds
    pub duration: f32,
    /// Quality metrics
    pub quality: Option<crate::QualityMetrics>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl ExportSample {
    /// Create export sample from dataset sample
    pub fn from_dataset_sample(sample: &DatasetSample, config: &ExportConfig) -> Self {
        let audio_path = if config.include_audio {
            Some(format!("{}/{}.wav", config.output_path, sample.id))
        } else {
            None
        };
        
        Self {
            id: sample.id.clone(),
            text: sample.text.clone(),
            audio_path,
            speaker_id: sample.speaker_id().map(|s| s.to_string()),
            language: sample.language.as_str().to_string(),
            duration: sample.duration(),
            quality: Some(sample.quality.clone()),
            metadata: sample.metadata.clone(),
        }
    }
}

/// Export progress tracking
#[derive(Debug, Clone)]
pub struct ExportProgress {
    /// Total samples to export
    pub total_samples: usize,
    /// Samples exported
    pub exported_samples: usize,
    /// Export start time
    pub start_time: std::time::Instant,
    /// Current operation
    pub current_operation: String,
}

impl ExportProgress {
    /// Create new progress tracker
    pub fn new(total_samples: usize) -> Self {
        Self {
            total_samples,
            exported_samples: 0,
            start_time: std::time::Instant::now(),
            current_operation: "Starting export".to_string(),
        }
    }
    
    /// Update progress
    pub fn update(&mut self, exported_samples: usize, operation: String) {
        self.exported_samples = exported_samples;
        self.current_operation = operation;
    }
    
    /// Get completion percentage
    pub fn completion_percentage(&self) -> f32 {
        if self.total_samples == 0 {
            return 100.0;
        }
        (self.exported_samples as f32 / self.total_samples as f32) * 100.0
    }
    
    /// Get export rate
    pub fn export_rate(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.exported_samples as f32 / elapsed
        } else {
            0.0
        }
    }
}
