//! Memory access pattern optimization for cache efficiency
//!
//! This module provides optimizations for different memory access patterns
//! commonly used in audio processing operations.

use crate::cache::{CacheAlignedBuffer, CacheError, CacheOptimizer};

/// Common memory access patterns in audio processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access (streaming through arrays)
    Sequential,
    /// Strided access (e.g., interleaved audio channels)
    Strided { stride: usize },
    /// Random access (sparse data access)
    Random,
    /// Block-wise access (tiled algorithms)
    Blocked { block_size: usize },
    /// Transpose access (column-major to row-major)
    Transpose,
}

/// Memory access pattern analyzer
pub struct PatternAnalyzer {
    optimizer: CacheOptimizer,
}

impl PatternAnalyzer {
    /// Create new pattern analyzer
    pub fn new(optimizer: CacheOptimizer) -> Self {
        Self { optimizer }
    }

    /// Analyze access pattern and suggest optimizations
    pub fn analyze_pattern(&self, pattern: AccessPattern, data_size: usize) -> PatternOptimization {
        match pattern {
            AccessPattern::Sequential => PatternOptimization {
                pattern,
                prefetch_strategy: PrefetchStrategy::Linear { distance: 4 },
                blocking_strategy: BlockingStrategy::None,
                suggested_block_size: 0,
                cache_locality: CacheLocality::High,
            },
            AccessPattern::Strided { stride } => {
                let locality = if stride <= 8 {
                    CacheLocality::High
                } else if stride <= 64 {
                    CacheLocality::Medium
                } else {
                    CacheLocality::Low
                };

                PatternOptimization {
                    pattern,
                    prefetch_strategy: PrefetchStrategy::Strided {
                        stride,
                        distance: 2,
                    },
                    blocking_strategy: if stride > 32 {
                        BlockingStrategy::StrideTiling { stride }
                    } else {
                        BlockingStrategy::None
                    },
                    suggested_block_size: self.optimizer.optimal_block_size(data_size),
                    cache_locality: locality,
                }
            }
            AccessPattern::Random => PatternOptimization {
                pattern,
                prefetch_strategy: PrefetchStrategy::None,
                blocking_strategy: BlockingStrategy::DataReorganization,
                suggested_block_size: 0,
                cache_locality: CacheLocality::Low,
            },
            AccessPattern::Blocked { block_size } => {
                let optimal_size = self.optimizer.optimal_block_size(data_size);
                let adjusted_block_size = if block_size > optimal_size {
                    optimal_size
                } else {
                    block_size
                };

                PatternOptimization {
                    pattern: AccessPattern::Blocked {
                        block_size: adjusted_block_size,
                    },
                    prefetch_strategy: PrefetchStrategy::Block {
                        block_size: adjusted_block_size,
                    },
                    blocking_strategy: BlockingStrategy::SquareTiling {
                        size: adjusted_block_size,
                    },
                    suggested_block_size: adjusted_block_size,
                    cache_locality: CacheLocality::High,
                }
            }
            AccessPattern::Transpose => PatternOptimization {
                pattern,
                prefetch_strategy: PrefetchStrategy::None,
                blocking_strategy: BlockingStrategy::TransposeTiling {
                    tile_size: self.optimizer.optimal_block_size(data_size),
                },
                suggested_block_size: self.optimizer.optimal_block_size(data_size),
                cache_locality: CacheLocality::Medium,
            },
        }
    }

    /// Suggest optimal access pattern for given operation
    pub fn suggest_pattern(
        &self,
        operation: AudioOperation,
        data_characteristics: DataCharacteristics,
    ) -> AccessPattern {
        match operation {
            AudioOperation::Convolution => {
                if data_characteristics.channels > 1 {
                    AccessPattern::Strided {
                        stride: data_characteristics.channels,
                    }
                } else {
                    AccessPattern::Sequential
                }
            }
            AudioOperation::FFT => {
                if data_characteristics.size > 8192 {
                    AccessPattern::Blocked {
                        block_size: self.optimizer.optimal_block_size(data_characteristics.size),
                    }
                } else {
                    AccessPattern::Sequential
                }
            }
            AudioOperation::MatrixMultiply => AccessPattern::Blocked {
                block_size: self.optimizer.optimal_block_size(data_characteristics.size),
            },
            AudioOperation::Transpose => AccessPattern::Transpose,
            AudioOperation::Filtering => {
                if data_characteristics.filter_order > 32 {
                    AccessPattern::Blocked { block_size: 64 }
                } else {
                    AccessPattern::Sequential
                }
            }
            AudioOperation::Resampling => AccessPattern::Strided {
                stride: data_characteristics.interpolation_factor,
            },
        }
    }
}

/// Pattern optimization recommendations
#[derive(Debug, Clone)]
pub struct PatternOptimization {
    pub pattern: AccessPattern,
    pub prefetch_strategy: PrefetchStrategy,
    pub blocking_strategy: BlockingStrategy,
    pub suggested_block_size: usize,
    pub cache_locality: CacheLocality,
}

/// Prefetch strategies for different access patterns
#[derive(Debug, Clone)]
pub enum PrefetchStrategy {
    None,
    Linear { distance: usize },
    Strided { stride: usize, distance: usize },
    Block { block_size: usize },
}

/// Blocking strategies for cache optimization
#[derive(Debug, Clone)]
pub enum BlockingStrategy {
    None,
    SquareTiling { size: usize },
    StrideTiling { stride: usize },
    TransposeTiling { tile_size: usize },
    DataReorganization,
}

/// Cache locality assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLocality {
    High,   // Good cache reuse, minimal misses
    Medium, // Some cache misses, room for improvement
    Low,    // Poor cache utilization, needs optimization
}

/// Audio processing operations
#[derive(Debug, Clone, Copy)]
pub enum AudioOperation {
    Convolution,
    FFT,
    MatrixMultiply,
    Transpose,
    Filtering,
    Resampling,
}

/// Data characteristics for pattern analysis
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub size: usize,
    pub channels: usize,
    pub sample_rate: u32,
    pub filter_order: usize,
    pub interpolation_factor: usize,
}

impl Default for DataCharacteristics {
    fn default() -> Self {
        Self {
            size: 1024,
            channels: 1,
            sample_rate: 44100,
            filter_order: 4,
            interpolation_factor: 1,
        }
    }
}

/// Cache-friendly data structure for audio processing
pub struct CacheFriendlyAudioBuffer {
    data: CacheAlignedBuffer<f32>,
    channels: usize,
    samples_per_channel: usize,
    interleaved: bool,
}

impl CacheFriendlyAudioBuffer {
    /// Create new cache-friendly audio buffer
    pub fn new(
        channels: usize,
        samples_per_channel: usize,
        interleaved: bool,
    ) -> Result<Self, CacheError> {
        let total_samples = channels * samples_per_channel;
        let data = CacheAlignedBuffer::new(total_samples)?;

        Ok(Self {
            data,
            channels,
            samples_per_channel,
            interleaved,
        })
    }

    /// Get sample at (channel, sample_index)
    pub fn get_sample(&self, channel: usize, sample_index: usize) -> Option<f32> {
        if channel >= self.channels || sample_index >= self.samples_per_channel {
            return None;
        }

        let index = if self.interleaved {
            sample_index * self.channels + channel
        } else {
            channel * self.samples_per_channel + sample_index
        };

        self.data.as_slice().get(index).copied()
    }

    /// Set sample at (channel, sample_index)
    pub fn set_sample(&mut self, channel: usize, sample_index: usize, value: f32) -> bool {
        if channel >= self.channels || sample_index >= self.samples_per_channel {
            return false;
        }

        let index = if self.interleaved {
            sample_index * self.channels + channel
        } else {
            channel * self.samples_per_channel + sample_index
        };

        if let Some(sample) = self.data.as_mut_slice().get_mut(index) {
            *sample = value;
            true
        } else {
            false
        }
    }

    /// Get channel data as slice (only for non-interleaved format)
    pub fn channel_slice(&self, channel: usize) -> Option<&[f32]> {
        if self.interleaved || channel >= self.channels {
            return None;
        }

        let start = channel * self.samples_per_channel;
        let end = start + self.samples_per_channel;
        Some(&self.data.as_slice()[start..end])
    }

    /// Get mutable channel data as slice (only for non-interleaved format)
    pub fn channel_slice_mut(&mut self, channel: usize) -> Option<&mut [f32]> {
        if self.interleaved || channel >= self.channels {
            return None;
        }

        let start = channel * self.samples_per_channel;
        let end = start + self.samples_per_channel;
        Some(&mut self.data.as_mut_slice()[start..end])
    }

    /// Convert between interleaved and non-interleaved formats
    pub fn convert_layout(&mut self) -> Result<(), CacheError> {
        let mut new_data = CacheAlignedBuffer::new(self.data.len())?;

        if self.interleaved {
            // Convert from interleaved to non-interleaved
            for channel in 0..self.channels {
                for sample in 0..self.samples_per_channel {
                    let src_index = sample * self.channels + channel;
                    let dst_index = channel * self.samples_per_channel + sample;
                    new_data.as_mut_slice()[dst_index] = self.data.as_slice()[src_index];
                }
            }
        } else {
            // Convert from non-interleaved to interleaved
            for channel in 0..self.channels {
                for sample in 0..self.samples_per_channel {
                    let src_index = channel * self.samples_per_channel + sample;
                    let dst_index = sample * self.channels + channel;
                    new_data.as_mut_slice()[dst_index] = self.data.as_slice()[src_index];
                }
            }
        }

        self.data = new_data;
        self.interleaved = !self.interleaved;
        Ok(())
    }

    /// Get buffer properties
    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn samples_per_channel(&self) -> usize {
        self.samples_per_channel
    }
    pub fn is_interleaved(&self) -> bool {
        self.interleaved
    }
    pub fn total_samples(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_analyzer() {
        let optimizer = CacheOptimizer::default();
        let analyzer = PatternAnalyzer::new(optimizer);

        // Test sequential pattern analysis
        let optimization = analyzer.analyze_pattern(AccessPattern::Sequential, 1024);
        assert_eq!(optimization.pattern, AccessPattern::Sequential);
        assert_eq!(optimization.cache_locality, CacheLocality::High);

        // Test strided pattern analysis
        let optimization = analyzer.analyze_pattern(AccessPattern::Strided { stride: 2 }, 1024);
        if let AccessPattern::Strided { stride } = optimization.pattern {
            assert_eq!(stride, 2);
        } else {
            panic!("Expected strided pattern");
        }
    }

    #[test]
    fn test_pattern_suggestion() {
        let optimizer = CacheOptimizer::default();
        let analyzer = PatternAnalyzer::new(optimizer);

        let characteristics = DataCharacteristics {
            size: 1024,
            channels: 2,
            ..Default::default()
        };

        let pattern = analyzer.suggest_pattern(AudioOperation::Convolution, characteristics);
        if let AccessPattern::Strided { stride } = pattern {
            assert_eq!(stride, 2);
        } else {
            panic!("Expected strided pattern for multi-channel convolution");
        }
    }

    #[test]
    fn test_cache_friendly_buffer() {
        let mut buffer = CacheFriendlyAudioBuffer::new(2, 1024, true).unwrap();

        // Test sample access
        assert!(buffer.set_sample(0, 100, 1.0));
        assert_eq!(buffer.get_sample(0, 100), Some(1.0));

        // Test bounds checking
        assert!(!buffer.set_sample(2, 100, 1.0)); // Invalid channel
        assert!(!buffer.set_sample(0, 1024, 1.0)); // Invalid sample index

        // Test layout conversion
        let was_interleaved = buffer.is_interleaved();
        buffer.convert_layout().unwrap();
        assert_ne!(buffer.is_interleaved(), was_interleaved);
    }

    #[test]
    fn test_data_characteristics() {
        let characteristics = DataCharacteristics::default();
        assert_eq!(characteristics.channels, 1);
        assert_eq!(characteristics.size, 1024);
        assert_eq!(characteristics.sample_rate, 44100);
    }
}
