//! Model warmup and preloading utilities
//!
//! This module provides utilities for warming up acoustic models and preloading
//! common synthesis requests to reduce cold-start latency in production.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::synthesis_cache::{SynthesisCache, SynthesisCacheKey};
use crate::traits::AcousticModel;
use crate::{AcousticError, Result};

/// Warmup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Number of warmup iterations
    pub iterations: usize,
    /// Warmup phoneme sequences (short, medium, long)
    pub warmup_sequences: Vec<String>,
    /// Speaker IDs to warm up (for multi-speaker models)
    pub speaker_ids: Vec<u32>,
    /// Whether to preload results into cache
    pub preload_cache: bool,
    /// Timeout for warmup operations
    pub timeout: Duration,
    /// Enable parallel warmup
    pub parallel: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            warmup_sequences: vec![
                "hello".to_string(),
                "hello world how are you".to_string(),
                "the quick brown fox jumps over the lazy dog".to_string(),
            ],
            speaker_ids: vec![0],
            preload_cache: true,
            timeout: Duration::from_secs(30),
            parallel: true,
        }
    }
}

/// Warmup statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupStats {
    /// Total warmup time
    pub total_duration: Duration,
    /// Number of successful warmup iterations
    pub successful_iterations: usize,
    /// Number of failed warmup iterations
    pub failed_iterations: usize,
    /// Average synthesis time after warmup
    pub avg_synthesis_time: Duration,
    /// Min synthesis time
    pub min_synthesis_time: Duration,
    /// Max synthesis time
    pub max_synthesis_time: Duration,
    /// Number of cache entries preloaded
    pub preloaded_entries: usize,
}

impl Default for WarmupStats {
    fn default() -> Self {
        Self {
            total_duration: Duration::ZERO,
            successful_iterations: 0,
            failed_iterations: 0,
            avg_synthesis_time: Duration::ZERO,
            min_synthesis_time: Duration::MAX,
            max_synthesis_time: Duration::ZERO,
            preloaded_entries: 0,
        }
    }
}

/// Model warmup manager
pub struct ModelWarmup {
    /// Warmup configuration
    config: WarmupConfig,
    /// Cache for preloading (optional)
    cache: Option<Arc<SynthesisCache>>,
}

impl ModelWarmup {
    /// Create a new model warmup manager
    pub fn new(config: WarmupConfig) -> Self {
        Self {
            config,
            cache: None,
        }
    }

    /// Create with cache for preloading
    pub fn with_cache(config: WarmupConfig, cache: Arc<SynthesisCache>) -> Self {
        Self {
            config,
            cache: Some(cache),
        }
    }

    /// Warm up an acoustic model
    pub async fn warmup<M: AcousticModel>(&self, model: &M) -> Result<WarmupStats> {
        info!(
            "Starting model warmup with {} iterations",
            self.config.iterations
        );
        let start = Instant::now();
        let mut stats = WarmupStats::default();

        // Convert warmup sequences to phonemes
        let phoneme_sequences = self.prepare_phoneme_sequences()?;

        // Run warmup iterations
        for iteration in 0..self.config.iterations {
            debug!(
                "Warmup iteration {}/{}",
                iteration + 1,
                self.config.iterations
            );

            for (seq_idx, phonemes) in phoneme_sequences.iter().enumerate() {
                for &speaker_id in &self.config.speaker_ids {
                    let warmup_start = Instant::now();

                    // Perform synthesis
                    match model.synthesize(phonemes, None).await {
                        Ok(mel) => {
                            let duration = warmup_start.elapsed();
                            stats.successful_iterations += 1;

                            // Update timing stats
                            if duration < stats.min_synthesis_time {
                                stats.min_synthesis_time = duration;
                            }
                            if duration > stats.max_synthesis_time {
                                stats.max_synthesis_time = duration;
                            }

                            // Preload to cache if enabled and this is the last iteration
                            if self.config.preload_cache && iteration == self.config.iterations - 1
                            {
                                if let Some(cache) = &self.cache {
                                    let key = SynthesisCacheKey::new(
                                        &self.config.warmup_sequences[seq_idx],
                                        Some(speaker_id),
                                        1.0,
                                        0.0,
                                        1.0,
                                    );
                                    if cache.insert(key, mel).is_ok() {
                                        stats.preloaded_entries += 1;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Warmup iteration failed: {}", e);
                            stats.failed_iterations += 1;
                        }
                    }

                    // Check timeout
                    if start.elapsed() > self.config.timeout {
                        warn!("Warmup timeout reached");
                        stats.total_duration = start.elapsed();
                        return Ok(stats);
                    }
                }
            }
        }

        stats.total_duration = start.elapsed();

        // Calculate average synthesis time
        if stats.successful_iterations > 0 {
            let total_ms = stats.total_duration.as_millis() as u64;
            stats.avg_synthesis_time =
                Duration::from_millis(total_ms / stats.successful_iterations as u64);
        }

        info!(
            "Model warmup completed in {:.2}s ({} successful, {} failed)",
            stats.total_duration.as_secs_f64(),
            stats.successful_iterations,
            stats.failed_iterations
        );

        Ok(stats)
    }

    /// Prepare phoneme sequences for warmup
    fn prepare_phoneme_sequences(&self) -> Result<Vec<Vec<crate::Phoneme>>> {
        // For now, create dummy phoneme sequences
        // In production, these would be actual phoneme conversions
        Ok(self
            .config
            .warmup_sequences
            .iter()
            .map(|text| {
                // Simple: one phoneme per character (dummy for warmup)
                text.chars()
                    .map(|c| crate::Phoneme {
                        symbol: c.to_string(),
                        features: None,
                        duration: Some(0.1),
                    })
                    .collect()
            })
            .collect())
    }

    /// Create a warmup configuration with common phrases
    pub fn common_phrases() -> WarmupConfig {
        WarmupConfig {
            iterations: 2,
            warmup_sequences: vec![
                "hello".to_string(),
                "hello world".to_string(),
                "how are you today".to_string(),
                "thank you very much".to_string(),
                "please try again".to_string(),
            ],
            speaker_ids: vec![0],
            preload_cache: true,
            timeout: Duration::from_secs(30),
            parallel: false,
        }
    }

    /// Create a quick warmup configuration (minimal)
    pub fn quick() -> WarmupConfig {
        WarmupConfig {
            iterations: 1,
            warmup_sequences: vec!["test".to_string()],
            speaker_ids: vec![0],
            preload_cache: false,
            timeout: Duration::from_secs(5),
            parallel: false,
        }
    }

    /// Create a thorough warmup configuration
    pub fn thorough() -> WarmupConfig {
        WarmupConfig {
            iterations: 5,
            warmup_sequences: vec![
                "a".to_string(),
                "short phrase".to_string(),
                "this is a medium length sentence for testing".to_string(),
                "the quick brown fox jumps over the lazy dog".to_string(),
                "a very long sentence with multiple clauses and various phonetic patterns to thoroughly test the synthesis system".to_string(),
            ],
            speaker_ids: vec![0, 1, 2],
            preload_cache: true,
            timeout: Duration::from_secs(60),
            parallel: true,
        }
    }
}

/// Preloader for common phrases
pub struct PhrasePreloader {
    /// Common phrases to preload
    phrases: Vec<String>,
    /// Cache to preload into
    cache: Arc<SynthesisCache>,
}

impl PhrasePreloader {
    /// Create a new phrase preloader
    pub fn new(phrases: Vec<String>, cache: Arc<SynthesisCache>) -> Self {
        Self { phrases, cache }
    }

    /// Preload all phrases
    pub async fn preload<M: AcousticModel>(&self, model: &M) -> Result<usize> {
        info!("Preloading {} common phrases", self.phrases.len());
        let mut count = 0;

        for phrase in &self.phrases {
            // Convert to phonemes (dummy for now)
            let phonemes: Vec<crate::Phoneme> = phrase
                .chars()
                .map(|c| crate::Phoneme {
                    symbol: c.to_string(),
                    features: None,
                    duration: Some(0.1),
                })
                .collect();

            // Synthesize
            match model.synthesize(&phonemes, None).await {
                Ok(mel) => {
                    let key = SynthesisCacheKey::new(phrase, None, 1.0, 0.0, 1.0);
                    if self.cache.insert(key, mel).is_ok() {
                        count += 1;
                    }
                }
                Err(e) => {
                    warn!("Failed to preload phrase '{}': {}", phrase, e);
                }
            }
        }

        info!("Preloaded {} phrases successfully", count);
        Ok(count)
    }

    /// Create preloader with common system phrases
    pub fn system_phrases(cache: Arc<SynthesisCache>) -> Self {
        let phrases = vec![
            "hello".to_string(),
            "goodbye".to_string(),
            "yes".to_string(),
            "no".to_string(),
            "thank you".to_string(),
            "please".to_string(),
            "sorry".to_string(),
            "okay".to_string(),
            "welcome".to_string(),
            "error".to_string(),
        ];
        Self::new(phrases, cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_config_default() {
        let config = WarmupConfig::default();
        assert_eq!(config.iterations, 3);
        assert!(!config.warmup_sequences.is_empty());
    }

    #[test]
    fn test_warmup_config_presets() {
        let quick = ModelWarmup::quick();
        assert_eq!(quick.iterations, 1);

        let thorough = ModelWarmup::thorough();
        assert_eq!(thorough.iterations, 5);

        let common = ModelWarmup::common_phrases();
        assert!(!common.warmup_sequences.is_empty());
    }

    #[test]
    fn test_warmup_stats_default() {
        let stats = WarmupStats::default();
        assert_eq!(stats.successful_iterations, 0);
        assert_eq!(stats.failed_iterations, 0);
    }

    #[test]
    fn test_model_warmup_creation() {
        let config = WarmupConfig::default();
        let warmup = ModelWarmup::new(config);
        assert!(warmup.cache.is_none());
    }

    #[test]
    fn test_phrase_preloader_creation() {
        let cache = Arc::new(SynthesisCache::default());
        let preloader = PhrasePreloader::system_phrases(cache);
        assert_eq!(preloader.phrases.len(), 10);
    }
}
