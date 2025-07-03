//! Utility functions and helpers for dataset operations
//!
//! This module provides common utility functions for file operations,
//! text processing, and other helper functionality.

use crate::{DatasetError, Result};
use std::path::{Path, PathBuf};
use std::collections::HashMap;

/// File system utilities
pub struct FileUtils;

impl FileUtils {
    /// Get all files with specific extensions in a directory
    pub fn find_files_with_extensions<P: AsRef<Path>>(
        dir: P,
        extensions: &[&str],
        recursive: bool,
    ) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        let dir = dir.as_ref();
        
        if !dir.exists() {
            return Err(DatasetError::IoError(
                std::io::Error::new(std::io::ErrorKind::NotFound, "Directory does not exist")
            ));
        }
        
        if recursive {
            for entry in walkdir::WalkDir::new(dir) {
                let entry = entry.map_err(|e| DatasetError::IoError(e.into()))?;
                let path = entry.path();
                
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if extensions.contains(&ext.to_lowercase().as_str()) {
                            files.push(path.to_path_buf());
                        }
                    }
                }
            }
        } else {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if extensions.contains(&ext.to_lowercase().as_str()) {
                            files.push(path);
                        }
                    }
                }
            }
        }
        
        files.sort();
        Ok(files)
    }
    
    /// Create directory structure
    pub fn create_dirs<P: AsRef<Path>>(path: P) -> Result<()> {
        std::fs::create_dir_all(path)?;
        Ok(())
    }
    
    /// Copy file with progress callback
    pub fn copy_file_with_progress<P1: AsRef<Path>, P2: AsRef<Path>, F>(
        src: P1,
        dst: P2,
        mut progress_callback: F,
    ) -> Result<()>
    where
        F: FnMut(u64, u64),
    {
        use std::io::{Read, Write};
        
        let src_path = src.as_ref();
        let dst_path = dst.as_ref();
        
        let mut src_file = std::fs::File::open(src_path)?;
        let mut dst_file = std::fs::File::create(dst_path)?;
        
        let file_size = src_file.metadata()?.len();
        let mut buffer = vec![0u8; 8192];
        let mut bytes_copied = 0u64;
        
        loop {
            let bytes_read = src_file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            
            dst_file.write_all(&buffer[..bytes_read])?;
            bytes_copied += bytes_read as u64;
            progress_callback(bytes_copied, file_size);
        }
        
        Ok(())
    }
    
    /// Get file size in bytes
    pub fn get_file_size<P: AsRef<Path>>(path: P) -> Result<u64> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len())
    }
    
    /// Check if path is safe (no directory traversal)
    pub fn is_safe_path<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        !path.components().any(|component| {
            matches!(component, std::path::Component::ParentDir)
        })
    }
}

/// Text processing utilities
pub struct TextUtils;

impl TextUtils {
    /// Normalize whitespace in text
    pub fn normalize_whitespace(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }
    
    /// Remove special characters
    pub fn remove_special_chars(text: &str, keep_chars: &[char]) -> String {
        text.chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || keep_chars.contains(c))
            .collect()
    }
    
    /// Convert to lowercase and normalize
    pub fn normalize_case(text: &str) -> String {
        text.to_lowercase()
    }
    
    /// Extract language from text (simple heuristic)
    pub fn detect_language_simple(text: &str) -> Option<String> {
        // Very basic language detection
        if text.chars().any(|c| matches!(c, '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}' | '\u{4E00}'..='\u{9FAF}')) {
            Some("ja".to_string())
        } else if text.chars().any(|c| matches!(c, '\u{4E00}'..='\u{9FAF}')) {
            Some("zh".to_string())
        } else if text.chars().any(|c| matches!(c, '\u{AC00}'..='\u{D7AF}')) {
            Some("ko".to_string())
        } else {
            Some("en".to_string()) // Default to English
        }
    }
    
    /// Count characters, words, and sentences
    pub fn text_statistics(text: &str) -> TextStatistics {
        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();
        let sentence_count = text.split(&['.', '!', '?']).filter(|s| !s.trim().is_empty()).count();
        
        TextStatistics {
            char_count,
            word_count,
            sentence_count,
        }
    }
}

/// Text statistics structure
#[derive(Debug, Clone)]
pub struct TextStatistics {
    pub char_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
}

/// Math utilities
pub struct MathUtils;

impl MathUtils {
    /// Calculate percentile
    pub fn percentile(data: &[f32], percentile: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile / 100.0 * (sorted_data.len() - 1) as f32) as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }
    
    /// Calculate standard deviation
    pub fn std_dev(data: &[f32]) -> f32 {
        if data.len() <= 1 {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (data.len() - 1) as f32;
        
        variance.sqrt()
    }
    
    /// Linear interpolation
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }
    
    /// Clamp value to range
    pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
        value.max(min).min(max)
    }
}

/// Progress reporting utilities
pub struct ProgressReporter {
    total: usize,
    current: usize,
    start_time: std::time::Instant,
    last_report: std::time::Instant,
    report_interval: std::time::Duration,
}

impl ProgressReporter {
    /// Create new progress reporter
    pub fn new(total: usize) -> Self {
        let now = std::time::Instant::now();
        Self {
            total,
            current: 0,
            start_time: now,
            last_report: now,
            report_interval: std::time::Duration::from_secs(1),
        }
    }
    
    /// Update progress
    pub fn update(&mut self, current: usize) -> Option<ProgressUpdate> {
        self.current = current;
        let now = std::time::Instant::now();
        
        if now.duration_since(self.last_report) >= self.report_interval {
            self.last_report = now;
            Some(self.get_progress_update())
        } else {
            None
        }
    }
    
    /// Force progress update
    pub fn force_update(&mut self) -> ProgressUpdate {
        self.last_report = std::time::Instant::now();
        self.get_progress_update()
    }
    
    /// Get current progress update
    fn get_progress_update(&self) -> ProgressUpdate {
        let elapsed = self.start_time.elapsed();
        let percentage = if self.total > 0 {
            (self.current as f32 / self.total as f32) * 100.0
        } else {
            0.0
        };
        
        let rate = if elapsed.as_secs_f32() > 0.0 {
            self.current as f32 / elapsed.as_secs_f32()
        } else {
            0.0
        };
        
        let eta = if rate > 0.0 && self.current < self.total {
            Some(std::time::Duration::from_secs_f32(
                (self.total - self.current) as f32 / rate
            ))
        } else {
            None
        };
        
        ProgressUpdate {
            current: self.current,
            total: self.total,
            percentage,
            rate,
            elapsed,
            eta,
        }
    }
}

/// Progress update information
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub current: usize,
    pub total: usize,
    pub percentage: f32,
    pub rate: f32,
    pub elapsed: std::time::Duration,
    pub eta: Option<std::time::Duration>,
}

impl ProgressUpdate {
    /// Format as string
    pub fn format(&self) -> String {
        let eta_str = if let Some(eta) = self.eta {
            format!(" ETA: {:?}", eta)
        } else {
            String::new()
        };
        
        format!(
            "{}/{} ({:.1}%) - {:.1} items/s - Elapsed: {:?}{}",
            self.current, self.total, self.percentage, self.rate, self.elapsed, eta_str
        )
    }
}

/// Configuration utilities
pub struct ConfigUtils;

impl ConfigUtils {
    /// Load configuration from TOML file
    pub fn load_toml_config<T, P>(path: P) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: AsRef<Path>,
    {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content)
            .map_err(|e| DatasetError::ConfigError(format!("TOML parsing failed: {}", e)))
    }
    
    /// Save configuration to TOML file
    pub fn save_toml_config<T, P>(config: &T, path: P) -> Result<()>
    where
        T: serde::Serialize,
        P: AsRef<Path>,
    {
        let content = toml::to_string_pretty(config)
            .map_err(|e| DatasetError::ConfigError(format!("TOML serialization failed: {}", e)))?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Merge two configuration hashmaps
    pub fn merge_configs(
        base: HashMap<String, serde_json::Value>,
        override_config: HashMap<String, serde_json::Value>,
    ) -> HashMap<String, serde_json::Value> {
        let mut merged = base;
        for (key, value) in override_config {
            merged.insert(key, value);
        }
        merged
    }
}
