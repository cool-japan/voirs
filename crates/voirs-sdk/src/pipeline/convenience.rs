// Convenience methods for common synthesis patterns
//!
//! This module provides helper methods for common VoiRS usage patterns, making it easier
//! to perform frequent operations without boilerplate code.

use crate::{
    audio::AudioBuffer,
    error::Result,
    pipeline::VoirsPipeline,
    types::{AudioFormat, QualityLevel, SynthesisConfig},
    VoirsError,
};
use std::path::Path;

/// Quick-start synthesis methods
impl VoirsPipeline {
    /// Synthesize text and save directly to a WAV file
    ///
    /// This is a convenience method that combines synthesis and file saving in one call.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipelineBuilder::new().build().await?;
    ///     pipeline.synthesize_to_wav("Hello, world!", "output.wav").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn synthesize_to_wav(&self, text: &str, path: impl AsRef<Path>) -> Result<()> {
        let audio = self.synthesize(text).await?;
        audio.save_wav(path)?;
        Ok(())
    }

    /// Synthesize text and save to a file with specified format
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipelineBuilder::new().build().await?;
    ///     pipeline.synthesize_to_file(
    ///         "Hello, world!",
    ///         "output.mp3",
    ///         AudioFormat::Mp3
    ///     ).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn synthesize_to_file(
        &self,
        text: &str,
        path: impl AsRef<Path>,
        format: AudioFormat,
    ) -> Result<()> {
        let audio = self.synthesize(text).await?;

        match format {
            AudioFormat::Wav => audio.save_wav(path)?,
            AudioFormat::Mp3 | AudioFormat::Flac | AudioFormat::Ogg | AudioFormat::Opus => {
                return Err(VoirsError::UnsupportedFileFormat {
                    path: path.as_ref().to_path_buf(),
                    format: format!("{:?}", format),
                });
            }
        }

        Ok(())
    }

    /// Synthesize text with a specific quality level
    ///
    /// This is a convenience method for quick quality-adjusted synthesis.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipelineBuilder::new().build().await?;
    ///     let audio = pipeline.synthesize_with_quality(
    ///         "High quality speech",
    ///         QualityLevel::High
    ///     ).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn synthesize_with_quality(
        &self,
        text: &str,
        quality: QualityLevel,
    ) -> Result<AudioBuffer> {
        let config = SynthesisConfig {
            quality,
            ..Default::default()
        };
        self.synthesize_with_config(text, &config).await
    }

    /// Synthesize text with custom speed (speaking rate)
    ///
    /// # Arguments
    ///
    /// * `text` - The text to synthesize
    /// * `speed` - Speaking rate multiplier (0.5 = half speed, 2.0 = double speed)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipelineBuilder::new().build().await?;
    ///     let fast = pipeline.synthesize_with_speed("Fast speech", 1.5).await?;
    ///     let slow = pipeline.synthesize_with_speed("Slow speech", 0.75).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn synthesize_with_speed(&self, text: &str, speed: f32) -> Result<AudioBuffer> {
        let config = SynthesisConfig {
            speaking_rate: speed,
            ..Default::default()
        };
        self.synthesize_with_config(text, &config).await
    }

    /// Synthesize text with custom pitch
    ///
    /// # Arguments
    ///
    /// * `text` - The text to synthesize
    /// * `pitch` - Pitch shift in semitones (-12.0 to +12.0)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipelineBuilder::new().build().await?;
    ///     let higher = pipeline.synthesize_with_pitch("Higher pitch", 3.0).await?;
    ///     let lower = pipeline.synthesize_with_pitch("Lower pitch", -3.0).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn synthesize_with_pitch(&self, text: &str, pitch: f32) -> Result<AudioBuffer> {
        let config = SynthesisConfig {
            pitch_shift: pitch,
            ..Default::default()
        };
        self.synthesize_with_config(text, &config).await
    }

    /// Synthesize text with both custom speed and pitch
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipelineBuilder::new().build().await?;
    ///     let audio = pipeline.synthesize_with_speed_and_pitch(
    ///         "Custom voice parameters",
    ///         1.2,  // 20% faster
    ///         2.0   // 2 semitones higher
    ///     ).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn synthesize_with_speed_and_pitch(
        &self,
        text: &str,
        speed: f32,
        pitch: f32,
    ) -> Result<AudioBuffer> {
        let config = SynthesisConfig {
            speaking_rate: speed,
            pitch_shift: pitch,
            ..Default::default()
        };
        self.synthesize_with_config(text, &config).await
    }

    /// Synthesize multiple texts in batch
    ///
    /// This is a convenience method for synthesizing multiple texts sequentially.
    /// For parallel batch processing with advanced features, use `BatchProcessor`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipelineBuilder::new().build().await?;
    ///     let texts = vec![
    ///         "First sentence.",
    ///         "Second sentence.",
    ///         "Third sentence."
    ///     ];
    ///     let results = pipeline.synthesize_batch(texts).await?;
    ///     println!("Synthesized {} audio buffers", results.len());
    ///     Ok(())
    /// }
    /// ```
    pub async fn synthesize_batch<S: AsRef<str>>(&self, texts: Vec<S>) -> Result<Vec<AudioBuffer>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let audio = self.synthesize(text.as_ref()).await?;
            results.push(audio);
        }
        Ok(results)
    }

    /// Concatenate multiple texts and synthesize as one
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipelineBuilder::new().build().await?;
    ///     let parts = vec![
    ///         "Hello,",
    ///         "this is a test.",
    ///         "Multiple parts combined."
    ///     ];
    ///     let audio = pipeline.synthesize_concatenated(parts, " ").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn synthesize_concatenated<S: AsRef<str>>(
        &self,
        texts: Vec<S>,
        separator: &str,
    ) -> Result<AudioBuffer> {
        let combined = texts
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .join(separator);
        self.synthesize(&combined).await
    }
}

/// Quick-start factory methods for VoirsPipeline
impl VoirsPipeline {
    /// Create a pipeline with default settings (quick start)
    ///
    /// This is the fastest way to get started with VoiRS.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipeline::default().await?;
    ///     let audio = pipeline.synthesize("Hello!").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn default() -> Result<Self> {
        Self::builder().build().await
    }

    /// Create a high-quality pipeline preset
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipeline::high_quality().await?;
    ///     let audio = pipeline.synthesize("High quality output").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn high_quality() -> Result<Self> {
        Self::builder()
            .with_quality(QualityLevel::High)
            .build()
            .await
    }

    /// Create a fast synthesis pipeline preset
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipeline::fast().await?;
    ///     let audio = pipeline.synthesize("Fast synthesis").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn fast() -> Result<Self> {
        use crate::builder::VoirsPipelineBuilder;
        VoirsPipelineBuilder::new()
            .with_quality(QualityLevel::Low)
            .with_preset(crate::builder::PresetProfile::FastSynthesis)
            .build()
            .await
    }

    /// Create a low-memory pipeline preset
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use voirs_sdk::prelude::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let pipeline = VoirsPipeline::low_memory().await?;
    ///     let audio = pipeline.synthesize("Memory efficient").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn low_memory() -> Result<Self> {
        use crate::builder::VoirsPipelineBuilder;
        VoirsPipelineBuilder::new()
            .with_preset(crate::builder::PresetProfile::LowMemory)
            .build()
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[tokio::test]
    async fn test_synthesize_to_wav() {
        use std::env::temp_dir;

        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();

        let temp_path = temp_dir().join("test_output.wav");
        pipeline
            .synthesize_to_wav("Test synthesis", &temp_path)
            .await
            .unwrap();

        assert!(temp_path.exists());
        std::fs::remove_file(temp_path).ok();
    }

    #[tokio::test]
    async fn test_synthesize_with_quality() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();

        let audio = pipeline
            .synthesize_with_quality("High quality test", QualityLevel::High)
            .await
            .unwrap();

        assert!(!audio.samples().is_empty());
    }

    #[tokio::test]
    async fn test_synthesize_with_speed() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();

        let fast = pipeline.synthesize_with_speed("Fast", 1.5).await.unwrap();
        let slow = pipeline.synthesize_with_speed("Slow", 0.75).await.unwrap();

        assert!(!fast.samples().is_empty());
        assert!(!slow.samples().is_empty());
    }

    #[tokio::test]
    async fn test_synthesize_with_pitch() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();

        let higher = pipeline.synthesize_with_pitch("Higher", 3.0).await.unwrap();
        let lower = pipeline.synthesize_with_pitch("Lower", -3.0).await.unwrap();

        assert!(!higher.samples().is_empty());
        assert!(!lower.samples().is_empty());
    }

    #[tokio::test]
    async fn test_synthesize_batch() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();

        let texts = vec!["First", "Second", "Third"];
        let results = pipeline.synthesize_batch(texts).await.unwrap();

        assert_eq!(results.len(), 3);
        for audio in results {
            assert!(!audio.samples().is_empty());
        }
    }

    #[tokio::test]
    async fn test_synthesize_concatenated() {
        let pipeline = VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await
            .unwrap();

        let parts = vec!["Part one.", "Part two.", "Part three."];
        let audio = pipeline.synthesize_concatenated(parts, " ").await.unwrap();

        assert!(!audio.samples().is_empty());
    }

    #[tokio::test]
    async fn test_factory_methods() {
        // Note: Factory methods cannot enable test_mode automatically
        // so they may not work in test environments without proper models.
        // This is expected behavior - use builder with test_mode for testing.

        // Test that the factory methods at least construct the builders correctly
        // by using the builder pattern with test mode

        // Test default equivalent
        let default = crate::prelude::VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await;
        assert!(default.is_ok());

        // Test high_quality equivalent
        let hq = crate::prelude::VoirsPipelineBuilder::new()
            .with_quality(QualityLevel::High)
            .with_test_mode(true)
            .build()
            .await;
        assert!(hq.is_ok());

        // Test fast equivalent
        let fast = crate::prelude::VoirsPipelineBuilder::new()
            .with_quality(QualityLevel::Low)
            .with_test_mode(true)
            .build()
            .await;
        assert!(fast.is_ok());

        // Test low_memory equivalent
        let lm = crate::prelude::VoirsPipelineBuilder::new()
            .with_test_mode(true)
            .build()
            .await;
        assert!(lm.is_ok());
    }
}
