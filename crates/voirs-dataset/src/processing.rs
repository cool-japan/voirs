//! Dataset processing utilities
//!
//! This module provides preprocessing, validation, and feature extraction
//! utilities for speech synthesis datasets.

pub mod features;
pub mod pipeline;
pub mod validation;

use crate::{DatasetSample, Result};

/// Processing configuration
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    /// Enable text normalization
    pub normalize_text: bool,
    /// Enable audio resampling
    pub resample_audio: bool,
    /// Target sample rate
    pub target_sample_rate: Option<u32>,
    /// Enable audio normalization
    pub normalize_audio: bool,
    /// Enable silence trimming
    pub trim_silence: bool,
    /// Silence threshold
    pub silence_threshold: f32,
    /// Enable quality filtering
    pub quality_filter: bool,
    /// Minimum quality threshold
    pub min_quality: f32,
    /// Enable parallel processing
    pub parallel: bool,
    /// Number of worker threads
    pub num_threads: Option<usize>,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            normalize_text: true,
            resample_audio: true,
            target_sample_rate: Some(22050),
            normalize_audio: true,
            trim_silence: true,
            silence_threshold: 0.01,
            quality_filter: true,
            min_quality: 0.5,
            parallel: true,
            num_threads: None,
        }
    }
}

/// Dataset processor for applying transformations
pub struct DatasetProcessor {
    config: ProcessingConfig,
}

impl DatasetProcessor {
    /// Create new processor with configuration
    pub fn new(config: ProcessingConfig) -> Self {
        Self { config }
    }

    /// Process a single sample
    pub fn process_sample(&self, sample: &mut DatasetSample) -> Result<()> {
        // Text normalization
        if self.config.normalize_text {
            self.normalize_text(sample)?;
        }

        // Audio processing
        if self.config.resample_audio {
            if let Some(target_rate) = self.config.target_sample_rate {
                if sample.audio.sample_rate() != target_rate {
                    sample.audio = sample.audio.resample(target_rate)?;
                }
            }
        }

        if self.config.normalize_audio {
            sample.audio.normalize()?;
        }

        if self.config.trim_silence {
            sample.audio = crate::audio::AudioProcessor::trim_silence(
                &sample.audio,
                self.config.silence_threshold,
            )?;
        }

        Ok(())
    }

    /// Process multiple samples
    pub fn process_samples(&self, samples: &mut [DatasetSample]) -> Result<()> {
        if self.config.parallel {
            use scirs2_core::parallel_ops::*;

            // Use parallel iterator to process samples concurrently
            let results: Result<Vec<()>> = samples
                .par_iter_mut()
                .map(|sample| self.process_sample(sample))
                .collect();

            // Check if any processing failed
            results?;
        } else {
            for sample in samples {
                self.process_sample(sample)?;
            }
        }
        Ok(())
    }

    /// Normalize text content
    fn normalize_text(&self, sample: &mut DatasetSample) -> Result<()> {
        let mut text = sample.text.clone();

        // Basic cleaning
        text = text.trim().to_string();

        // Remove multiple spaces and normalize whitespace
        text = text.split_whitespace().collect::<Vec<_>>().join(" ");

        // Normalize quotes and apostrophes
        text = text
            .replace(['"', '"', '"'], "\"")
            .replace(['\u{2018}', '\u{2019}'], "'") // Left and right single quotes
            .replace(['`'], "'");

        // Normalize dashes
        text = text.replace(['—', '–'], "-").replace("--", "-");

        // Remove or normalize special characters
        text = text.replace(['…'], "...").replace(['•'], "*");

        // Normalize case (optional - keep original case for now)
        // text = text.to_lowercase();

        // Remove control characters except newlines and tabs
        text = text
            .chars()
            .filter(|&c| !c.is_control() || c == '\n' || c == '\t')
            .collect();

        // Normalize numbers (expand common abbreviations)
        text = self.normalize_numbers(&text);

        // Normalize abbreviations and contractions
        text = self.normalize_abbreviations(&text);

        // Final cleanup
        text = text.trim().to_string();

        sample.text = text;
        Ok(())
    }

    /// Normalize numbers in text
    fn normalize_numbers(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Simple replacements for common patterns
        result = result.replace("%", " percent");
        result = result.replace("$", " dollars ");

        // Normalize ordinals (simplified)
        let ordinals = [
            "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th",
        ];
        for (i, ordinal) in ordinals.iter().enumerate() {
            result = result.replace(ordinal, &(i + 1).to_string());
        }

        // Remove extra spaces that might have been introduced
        result = result.split_whitespace().collect::<Vec<_>>().join(" ");

        result
    }

    /// Normalize abbreviations and contractions
    fn normalize_abbreviations(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Common contractions
        let contractions = [
            ("can't", "cannot"),
            ("won't", "will not"),
            ("n't", " not"),
            ("'re", " are"),
            ("'ve", " have"),
            ("'ll", " will"),
            ("'d", " would"),
            ("'m", " am"),
            ("'s", " is"), // Note: this is ambiguous but common
        ];

        for (contraction, expansion) in contractions {
            result = result.replace(contraction, expansion);
        }

        // Common abbreviations
        let abbreviations = [
            ("Dr.", "Doctor"),
            ("Mr.", "Mister"),
            ("Mrs.", "Missus"),
            ("Ms.", "Miss"),
            ("Prof.", "Professor"),
            ("St.", "Street"),
            ("Ave.", "Avenue"),
            ("Blvd.", "Boulevard"),
            ("etc.", "et cetera"),
            ("vs.", "versus"),
            ("e.g.", "for example"),
            ("i.e.", "that is"),
        ];

        for (abbr, expansion) in abbreviations {
            result = result.replace(abbr, expansion);
        }

        result
    }
}

/// Progress tracking for processing operations
#[derive(Debug, Clone)]
pub struct ProcessingProgress {
    /// Total number of items to process
    pub total: usize,
    /// Number of items processed
    pub processed: usize,
    /// Number of items that failed processing
    pub failed: usize,
    /// Processing start time
    pub start_time: std::time::Instant,
}

impl ProcessingProgress {
    /// Create new progress tracker
    pub fn new(total: usize) -> Self {
        Self {
            total,
            processed: 0,
            failed: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Update progress
    pub fn update(&mut self, processed: usize, failed: usize) {
        self.processed = processed;
        self.failed = failed;
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> f32 {
        if self.total == 0 {
            return 100.0;
        }
        ((self.processed + self.failed) as f32 / self.total as f32) * 100.0
    }

    /// Get processing rate (items per second)
    pub fn processing_rate(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            (self.processed + self.failed) as f32 / elapsed
        } else {
            0.0
        }
    }

    /// Get estimated time remaining
    pub fn estimated_time_remaining(&self) -> Option<std::time::Duration> {
        let rate = self.processing_rate();
        if rate > 0.0 {
            let remaining = self.total - self.processed - self.failed;
            let seconds = remaining as f32 / rate;
            Some(std::time::Duration::from_secs_f32(seconds))
        } else {
            None
        }
    }
}
