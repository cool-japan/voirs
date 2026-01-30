//! # MobileOptimizedVocoder - Trait Implementations
//!
//! This module contains trait implementations for `MobileOptimizedVocoder`.
//!
//! ## Implemented Traits
//!
//! - `Vocoder`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MobileOptimizedVocoder;

impl Vocoder for MobileOptimizedVocoder {
    fn synthesize(&self, spectrogram: &[Vec<f32>]) -> Result<Vec<f32>> {
        let optimized_spectrogram = self.apply_mobile_optimizations(spectrogram);
        #[cfg(feature = "candle")]
        {
            if let Some(ref generator) = self.generator {
                match self.synthesize_with_hifigan(generator, &optimized_spectrogram) {
                    Ok(audio) => return Ok(audio),
                    Err(_) => {
                        tracing::warn!(
                            "HiFi-GAN synthesis failed, falling back to basic synthesis"
                        );
                    }
                }
            }
        }
        self.synthesize_basic(&optimized_spectrogram)
    }
}

