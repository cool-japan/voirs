//! Audio preprocessing for voice cloning

use crate::{types::VoiceSample, Result};

/// Audio preprocessor for preparing samples
#[derive(Debug, Clone)]
pub struct AudioPreprocessor {
    /// Target sample rate
    pub target_sample_rate: u32,
    /// Normalize audio
    pub normalize: bool,
}

impl AudioPreprocessor {
    /// Create new preprocessor
    pub fn new(target_sample_rate: u32) -> Self {
        Self {
            target_sample_rate,
            normalize: true,
        }
    }

    /// Preprocess voice sample
    pub async fn preprocess(&self, sample: &VoiceSample) -> Result<VoiceSample> {
        let mut processed = sample.clone();

        // Resample if needed
        if sample.sample_rate != self.target_sample_rate {
            processed = processed.resample(self.target_sample_rate)?;
        }

        // Normalize if enabled
        if self.normalize {
            processed.audio = processed.get_normalized_audio();
        }

        Ok(processed)
    }
}

/// Preprocessing pipeline
#[derive(Debug, Clone)]
pub struct PreprocessingPipeline {
    /// Individual preprocessors
    pub processors: Vec<AudioPreprocessor>,
}

impl PreprocessingPipeline {
    /// Create new pipeline
    pub fn new() -> Self {
        Self {
            processors: vec![AudioPreprocessor::new(22050)],
        }
    }

    /// Process sample through pipeline
    pub async fn process(&self, sample: &VoiceSample) -> Result<VoiceSample> {
        let mut result = sample.clone();
        for processor in &self.processors {
            result = processor.preprocess(&result).await?;
        }
        Ok(result)
    }
}

impl Default for PreprocessingPipeline {
    fn default() -> Self {
        Self::new()
    }
}
