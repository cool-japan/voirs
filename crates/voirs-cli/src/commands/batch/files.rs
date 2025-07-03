//! File processing for batch operations.

use std::path::PathBuf;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use voirs::config::AppConfig;
use voirs::error::Result;
use voirs::types::{AudioFormat, SynthesisConfig};
use voirs::VoirsPipeline;
use crate::GlobalOptions;
use super::{BatchConfig, parallel};

/// Input item for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInput {
    /// Unique identifier for this item
    pub id: String,
    /// Text to synthesize
    pub text: String,
    /// Optional output filename (without extension)
    pub filename: Option<String>,
    /// Optional voice override
    pub voice: Option<String>,
    /// Optional speaking rate override
    pub rate: Option<f32>,
    /// Optional pitch override
    pub pitch: Option<f32>,
    /// Optional volume override
    pub volume: Option<f32>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Process a single file
pub async fn process_file(
    batch_config: &BatchConfig,
    app_config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    let extension = batch_config.input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    let inputs = match extension.as_str() {
        "txt" => parse_txt_file(&batch_config.input_path)?,
        "csv" => parse_csv_file(&batch_config.input_path)?,
        "json" => parse_json_file(&batch_config.input_path)?,
        "jsonl" => parse_jsonl_file(&batch_config.input_path)?,
        _ => {
            return Err(voirs::VoirsError::config_error(
                &format!("Unsupported file format: {}", extension)
            ));
        }
    };
    
    if !global.quiet {
        println!("Loaded {} inputs from {}", inputs.len(), batch_config.input_path.display());
    }
    
    // Process inputs in parallel
    parallel::process_inputs_parallel(&inputs, batch_config, app_config, global).await
}

/// Process a directory of files
pub async fn process_directory(
    batch_config: &BatchConfig,
    app_config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    // Process all supported files in directory
    for entry in std::fs::read_dir(&batch_config.input_path)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && super::is_supported_extension(&path) {
            if !global.quiet {
                println!("Processing file: {}", path.display());
            }
            
            let mut file_config = batch_config.clone();
            file_config.input_path = path.clone();
            
            let inputs = process_file(&file_config, app_config, global).await;
            match inputs {
                Ok(_) => {
                    // Individual file processing succeeded
                    continue;
                }
                Err(e) => {
                    tracing::warn!("Failed to process file {}: {}", path.display(), e);
                    continue;
                }
            }
        }
    }
    
    Ok(())
}

/// Parse TXT file (one text per line)
fn parse_txt_file(path: &PathBuf) -> Result<Vec<BatchInput>> {
    let content = std::fs::read_to_string(path)?;
    let mut inputs = Vec::new();
    
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            inputs.push(BatchInput {
                id: format!("txt_line_{}", i + 1),
                text: line.to_string(),
                filename: Some(format!("line_{:04}", i + 1)),
                voice: None,
                rate: None,
                pitch: None,
                volume: None,
                metadata: HashMap::new(),
            });
        }
    }
    
    Ok(inputs)
}

/// Parse CSV file
fn parse_csv_file(path: &PathBuf) -> Result<Vec<BatchInput>> {
    let content = std::fs::read_to_string(path)?;
    let mut inputs = Vec::new();
    let mut reader = csv::Reader::from_reader(content.as_bytes());
    
    for (i, result) in reader.records().enumerate() {
        let record = result.map_err(|e| voirs::VoirsError::config_error(&e.to_string()))?;
        
        // Expect at least text column, optionally id, filename, voice, rate, pitch, volume
        if record.len() == 0 {
            continue;
        }
        
        let text = record.get(0).unwrap_or("").trim();
        if text.is_empty() {
            continue;
        }
        
        let id = record.get(1)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("csv_row_{}", i + 1));
        
        let filename = record.get(2)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        
        let voice = record.get(3)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        
        let rate = record.get(4)
            .and_then(|s| s.trim().parse::<f32>().ok());
        
        let pitch = record.get(5)
            .and_then(|s| s.trim().parse::<f32>().ok());
        
        let volume = record.get(6)
            .and_then(|s| s.trim().parse::<f32>().ok());
        
        inputs.push(BatchInput {
            id,
            text: text.to_string(),
            filename,
            voice,
            rate,
            pitch,
            volume,
            metadata: HashMap::new(),
        });
    }
    
    Ok(inputs)
}

/// Parse JSON file (array of BatchInput objects)
fn parse_json_file(path: &PathBuf) -> Result<Vec<BatchInput>> {
    let content = std::fs::read_to_string(path)?;
    let inputs: Vec<BatchInput> = serde_json::from_str(&content)
        .map_err(|e| voirs::VoirsError::config_error(&e.to_string()))?;
    Ok(inputs)
}

/// Parse JSONL file (one BatchInput object per line)
fn parse_jsonl_file(path: &PathBuf) -> Result<Vec<BatchInput>> {
    let content = std::fs::read_to_string(path)?;
    let mut inputs = Vec::new();
    
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        
        match serde_json::from_str::<BatchInput>(line) {
            Ok(input) => inputs.push(input),
            Err(e) => {
                tracing::warn!("Failed to parse line {} in JSONL file: {}", i + 1, e);
                continue;
            }
        }
    }
    
    Ok(inputs)
}

/// Generate output filename for batch input
pub fn generate_output_filename(
    input: &BatchInput,
    index: usize,
    format: AudioFormat,
) -> String {
    if let Some(filename) = &input.filename {
        format!("{}.{}", filename, format.extension())
    } else {
        // Generate from text or use index
        let safe_text = input.text
            .chars()
            .take(30)
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .replace(' ', "_")
            .to_lowercase();
        
        if safe_text.is_empty() {
            format!("batch_{:04}.{}", index + 1, format.extension())
        } else {
            format!("{}_{:04}.{}", safe_text, index + 1, format.extension())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_parse_txt_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Hello world").unwrap();
        writeln!(temp_file, "This is a test").unwrap();
        writeln!(temp_file, "# This is a comment").unwrap();
        writeln!(temp_file, "").unwrap();
        writeln!(temp_file, "Another line").unwrap();
        
        let path = temp_file.path().to_path_buf();
        let inputs = parse_txt_file(&path).unwrap();
        
        assert_eq!(inputs.len(), 3);
        assert_eq!(inputs[0].text, "Hello world");
        assert_eq!(inputs[1].text, "This is a test");
        assert_eq!(inputs[2].text, "Another line");
        assert_eq!(inputs[0].id, "txt_line_1");
    }
    
    #[test]
    fn test_generate_output_filename() {
        let input = BatchInput {
            id: "test".to_string(),
            text: "Hello world!".to_string(),
            filename: Some("custom_name".to_string()),
            voice: None,
            rate: None,
            pitch: None,
            volume: None,
            metadata: HashMap::new(),
        };
        
        let filename = generate_output_filename(&input, 0, AudioFormat::Wav);
        assert_eq!(filename, "custom_name.wav");
        
        let input_no_filename = BatchInput {
            id: "test".to_string(),
            text: "Hello world!".to_string(),
            filename: None,
            voice: None,
            rate: None,
            pitch: None,
            volume: None,
            metadata: HashMap::new(),
        };
        
        let filename = generate_output_filename(&input_no_filename, 5, AudioFormat::Mp3);
        assert_eq!(filename, "hello_world_0006.mp3");
    }
}