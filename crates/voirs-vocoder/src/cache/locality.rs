//! Data locality optimization for improved cache performance
//!
//! Provides algorithms and data structures that optimize for spatial and temporal locality.

use super::{CacheConfig, CacheOptimizer, CACHE_LINE_SIZE};

/// Locality optimizer for audio processing data access patterns
pub struct LocalityOptimizer {
    cache_optimizer: CacheOptimizer,
}

impl LocalityOptimizer {
    /// Create new locality optimizer
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache_optimizer: CacheOptimizer::new(config),
        }
    }

    /// Optimize data layout for sequential access pattern
    pub fn optimize_sequential_layout<T: Copy>(&self, data: &[T]) -> Vec<T> {
        // For sequential access, data is already optimally laid out
        // Just ensure proper alignment and blocking if needed
        let block_size = self.cache_optimizer.optimal_block_size(data.len());

        if data.len() <= block_size {
            data.to_vec()
        } else {
            // Reorganize into cache-friendly blocks
            let mut optimized = Vec::with_capacity(data.len());

            for chunk in data.chunks(block_size) {
                optimized.extend_from_slice(chunk);
            }

            optimized
        }
    }

    /// Optimize 2D data layout for better spatial locality
    pub fn optimize_2d_layout<T: Copy + Default>(
        &self,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Vec<T> {
        if data.len() != rows * cols {
            return data.to_vec();
        }

        let block_size = self.cache_optimizer.optimal_block_size(data.len());
        let tile_rows = (block_size as f32).sqrt() as usize;
        let tile_cols = tile_rows;

        if tile_rows >= rows || tile_cols >= cols {
            // Data is small enough, no tiling needed
            return data.to_vec();
        }

        // Apply tiling for better locality
        let mut optimized = vec![T::default(); data.len()];

        for tile_r in (0..rows).step_by(tile_rows) {
            for tile_c in (0..cols).step_by(tile_cols) {
                let end_r = (tile_r + tile_rows).min(rows);
                let end_c = (tile_c + tile_cols).min(cols);

                for r in tile_r..end_r {
                    for c in tile_c..end_c {
                        let src_idx = r * cols + c;
                        let dst_idx = self.tiled_index(r, c, tile_rows, tile_cols, rows, cols);
                        optimized[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        optimized
    }

    /// Calculate tiled index for cache-friendly 2D layout
    fn tiled_index(
        &self,
        row: usize,
        col: usize,
        tile_rows: usize,
        tile_cols: usize,
        _total_rows: usize,
        total_cols: usize,
    ) -> usize {
        let tile_r = row / tile_rows;
        let tile_c = col / tile_cols;
        let in_tile_r = row % tile_rows;
        let in_tile_c = col % tile_cols;

        let tiles_per_row = (total_cols + tile_cols - 1) / tile_cols;
        let tile_idx = tile_r * tiles_per_row + tile_c;
        let tile_offset = tile_idx * tile_rows * tile_cols;
        let in_tile_offset = in_tile_r * tile_cols + in_tile_c;

        tile_offset + in_tile_offset
    }

    /// Optimize convolution data layout for better cache utilization
    pub fn optimize_convolution_layout<T: Copy + Default>(
        &self,
        input: &[T],
        input_height: usize,
        input_width: usize,
        _kernel_size: usize,
    ) -> Vec<T> {
        // For convolution, optimize for sliding window access pattern
        let block_size = self.cache_optimizer.optimal_block_size(input.len());

        // If input is small, no optimization needed
        if input.len() <= block_size {
            return input.to_vec();
        }

        // Create blocked layout optimized for convolution sliding window
        let mut optimized = vec![T::default(); input.len()];
        let block_height = (block_size as f32).sqrt() as usize;
        let block_width = block_height;

        for block_r in (0..input_height).step_by(block_height) {
            for block_c in (0..input_width).step_by(block_width) {
                let end_r = (block_r + block_height).min(input_height);
                let end_c = (block_c + block_width).min(input_width);

                for r in block_r..end_r {
                    for c in block_c..end_c {
                        let src_idx = r * input_width + c;
                        let dst_idx = self.convolution_index(
                            r,
                            c,
                            block_height,
                            block_width,
                            input_height,
                            input_width,
                        );
                        optimized[dst_idx] = input[src_idx];
                    }
                }
            }
        }

        optimized
    }

    /// Calculate index for convolution-optimized layout
    fn convolution_index(
        &self,
        row: usize,
        col: usize,
        block_height: usize,
        block_width: usize,
        _total_height: usize,
        total_width: usize,
    ) -> usize {
        let block_r = row / block_height;
        let block_c = col / block_width;
        let in_block_r = row % block_height;
        let in_block_c = col % block_width;

        let blocks_per_row = (total_width + block_width - 1) / block_width;
        let block_idx = block_r * blocks_per_row + block_c;
        let block_offset = block_idx * block_height * block_width;
        let in_block_offset = in_block_r * block_width + in_block_c;

        block_offset + in_block_offset
    }

    /// Optimize audio buffer layout for streaming access
    pub fn optimize_streaming_layout<T: Copy>(&self, data: &[T], chunk_size: usize) -> Vec<T> {
        // Ensure chunks are aligned to cache lines for optimal streaming
        let elements_per_cache_line = CACHE_LINE_SIZE / std::mem::size_of::<T>();
        let aligned_chunk_size = ((chunk_size + elements_per_cache_line - 1)
            / elements_per_cache_line)
            * elements_per_cache_line;

        let mut optimized = Vec::with_capacity(data.len());

        for chunk in data.chunks(chunk_size) {
            optimized.extend_from_slice(chunk);

            // Pad to aligned chunk size if necessary
            let padding_needed = aligned_chunk_size.saturating_sub(chunk.len());
            if padding_needed > 0 && optimized.len() + padding_needed <= optimized.capacity() {
                // Note: We can't add padding for generic T without Default
                // This would need to be handled at a higher level
            }
        }

        optimized
    }

    /// Analyze data access pattern for locality optimization
    pub fn analyze_access_pattern<T>(
        &self,
        data: &[T],
        access_indices: &[usize],
    ) -> LocalityAnalysis {
        if access_indices.is_empty() {
            return LocalityAnalysis::default();
        }

        let mut sequential_accesses = 0;
        let mut cache_line_hits = 0;
        let mut total_distance = 0i64;
        let elements_per_cache_line = CACHE_LINE_SIZE / std::mem::size_of::<T>();

        for window in access_indices.windows(2) {
            let curr = window[0];
            let next = window[1];

            // Check for sequential access
            if next == curr + 1 {
                sequential_accesses += 1;
            }

            // Check for cache line locality
            let curr_line = curr / elements_per_cache_line;
            let next_line = next / elements_per_cache_line;
            if curr_line == next_line {
                cache_line_hits += 1;
            }

            // Calculate access distance
            total_distance += (next as i64 - curr as i64).abs();
        }

        let total_accesses = access_indices.len() - 1;
        let sequential_ratio = sequential_accesses as f32 / total_accesses as f32;
        let cache_locality_ratio = cache_line_hits as f32 / total_accesses as f32;
        let average_distance = if total_accesses > 0 {
            total_distance as f32 / total_accesses as f32
        } else {
            0.0
        };

        LocalityAnalysis {
            total_accesses,
            sequential_accesses,
            cache_line_hits,
            sequential_ratio,
            cache_locality_ratio,
            average_distance,
            data_size: data.len(),
        }
    }
}

impl Default for LocalityOptimizer {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Analysis results for data access locality
#[derive(Debug, Clone)]
pub struct LocalityAnalysis {
    /// Total number of memory accesses analyzed
    pub total_accesses: usize,

    /// Number of sequential accesses (addr[i+1] = addr[i] + 1)
    pub sequential_accesses: usize,

    /// Number of accesses within the same cache line
    pub cache_line_hits: usize,

    /// Ratio of sequential accesses (0.0 - 1.0)
    pub sequential_ratio: f32,

    /// Ratio of cache line locality (0.0 - 1.0)
    pub cache_locality_ratio: f32,

    /// Average distance between consecutive accesses
    pub average_distance: f32,

    /// Total data size analyzed
    pub data_size: usize,
}

impl Default for LocalityAnalysis {
    fn default() -> Self {
        Self {
            total_accesses: 0,
            sequential_accesses: 0,
            cache_line_hits: 0,
            sequential_ratio: 0.0,
            cache_locality_ratio: 0.0,
            average_distance: 0.0,
            data_size: 0,
        }
    }
}

impl LocalityAnalysis {
    /// Check if access pattern has good locality
    pub fn has_good_locality(&self) -> bool {
        self.cache_locality_ratio > 0.7 || self.sequential_ratio > 0.5
    }

    /// Get locality score (0.0 - 1.0, higher is better)
    pub fn locality_score(&self) -> f32 {
        let sequential_weight = 0.6;
        let cache_line_weight = 0.4;

        sequential_weight * self.sequential_ratio + cache_line_weight * self.cache_locality_ratio
    }

    /// Suggest optimization strategy based on analysis
    pub fn optimization_suggestion(&self) -> LocalityOptimization {
        if self.sequential_ratio > 0.8 {
            LocalityOptimization::Sequential
        } else if self.cache_locality_ratio > 0.6 {
            LocalityOptimization::CacheBlocking
        } else if self.average_distance < 100.0 {
            LocalityOptimization::Tiling
        } else {
            LocalityOptimization::Restructure
        }
    }
}

/// Suggested locality optimization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum LocalityOptimization {
    /// Data already has good sequential locality
    Sequential,

    /// Use cache-aware blocking
    CacheBlocking,

    /// Apply tiling/blocking transformation
    Tiling,

    /// Restructure data layout completely
    Restructure,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locality_optimizer_creation() {
        let optimizer = LocalityOptimizer::default();
        assert!(optimizer.cache_optimizer.config().l1_cache_size > 0);
    }

    #[test]
    fn test_sequential_layout_optimization() {
        let optimizer = LocalityOptimizer::default();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let optimized = optimizer.optimize_sequential_layout(&data);
        assert_eq!(optimized.len(), data.len());

        // For small data, should be identical
        assert_eq!(optimized, data);
    }

    #[test]
    fn test_2d_layout_optimization() {
        let optimizer = LocalityOptimizer::default();
        let data = (0..16).map(|i| i as f32).collect::<Vec<_>>();

        let optimized = optimizer.optimize_2d_layout(&data, 4, 4);
        assert_eq!(optimized.len(), data.len());
    }

    #[test]
    fn test_access_pattern_analysis() {
        let optimizer = LocalityOptimizer::default();
        let data = vec![1.0f32; 1000];

        // Sequential access pattern
        let sequential_indices: Vec<usize> = (0..100).collect();
        let analysis = optimizer.analyze_access_pattern(&data, &sequential_indices);

        assert!(analysis.sequential_ratio > 0.9);
        assert!(analysis.has_good_locality());
        assert_eq!(
            analysis.optimization_suggestion(),
            LocalityOptimization::Sequential
        );
    }

    #[test]
    fn test_random_access_pattern() {
        let optimizer = LocalityOptimizer::default();
        let data = vec![1.0f32; 1000];

        // Random access pattern
        let random_indices = vec![0, 500, 100, 800, 50, 900, 200];
        let analysis = optimizer.analyze_access_pattern(&data, &random_indices);

        assert!(analysis.sequential_ratio < 0.2);
        assert!(analysis.average_distance > 100.0);
        assert_eq!(
            analysis.optimization_suggestion(),
            LocalityOptimization::Restructure
        );
    }

    #[test]
    fn test_locality_analysis_score() {
        let analysis = LocalityAnalysis {
            total_accesses: 100,
            sequential_accesses: 80,
            cache_line_hits: 90,
            sequential_ratio: 0.8,
            cache_locality_ratio: 0.9,
            average_distance: 5.0,
            data_size: 1000,
        };

        let score = analysis.locality_score();
        assert!(score > 0.8); // Should be high for good locality
        assert!(analysis.has_good_locality());
    }
}
