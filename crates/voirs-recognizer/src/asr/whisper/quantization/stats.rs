//! Statistics and tracking for quantization operations
//!
//! This module contains all statistics structures for tracking quantization
//! performance, pruning results, and model compression metrics.

use candle_core::DType;

/// Quantization statistics for a tensor
#[derive(Debug, Clone)]
/// Quantization Stats
pub struct QuantizationStats {
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i8,
    /// Minimum value
    pub min_val: f32,
    /// Maximum value
    pub max_val: f32,
    /// Data type
    pub dtype: DType,
}

/// 4-bit quantization statistics
#[derive(Debug, Clone)]
/// Quantization4 Bit Stats
pub struct Quantization4BitStats {
    /// Scale factors per group
    pub scales: Vec<f32>,
    /// Zero points per group
    pub zero_points: Vec<i8>,
    /// Group size
    pub group_size: usize,
    /// Number of groups
    pub num_groups: usize,
}

/// Pruning statistics
#[derive(Debug, Clone)]
/// Pruning Stats
pub struct PruningStats {
    /// Original number of parameters
    pub original_params: usize,
    /// Number of pruned parameters
    pub pruned_params: usize,
    /// Sparsity ratio achieved
    pub sparsity_ratio: f32,
    /// Pruning mask (true = keep, false = prune)
    pub mask: Vec<bool>,
}

/// Overall pruning statistics
#[derive(Debug, Clone)]
/// Overall Pruning Stats
pub struct OverallPruningStats {
    /// Total original parameters across all layers
    pub total_original_params: usize,
    /// Total pruned parameters across all layers
    pub total_pruned_params: usize,
    /// Overall sparsity ratio
    pub overall_sparsity_ratio: f32,
    /// Number of layers that were pruned
    pub layers_pruned: usize,
}

/// Quantization savings metrics
#[derive(Debug, Clone)]
/// Quantization Savings
pub struct QuantizationSavings {
    /// Original model size in MB
    pub original_size_mb: f32,
    /// Quantized model size in MB
    pub quantized_size_mb: f32,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Memory saved in MB
    pub memory_saved_mb: f32,
}

/// Knowledge distillation loss computation
#[derive(Debug, Clone)]
/// Distillation Loss
pub struct DistillationLoss {
    /// Distillation loss value
    pub distillation_loss: f32,
    /// Hard target loss value
    pub hard_loss: f32,
    /// Combined total loss
    pub total_loss: f32,
    /// Temperature used
    pub temperature: f32,
}

/// Moving average tracker for dynamic quantization
#[derive(Debug, Clone)]
/// Moving Average Tracker
pub struct MovingAverageTracker {
    /// Recent activation statistics
    pub recent_mins: Vec<f32>,
    /// Recent activation statistics  
    pub recent_maxs: Vec<f32>,
    /// Window size for moving average
    pub window_size: usize,
    /// Current position in circular buffer
    pub position: usize,
    /// Number of samples collected
    pub samples: usize,
}

impl MovingAverageTracker {
    #[must_use]
    /// new
    pub fn new(window_size: usize) -> Self {
        Self {
            recent_mins: vec![0.0; window_size],
            recent_maxs: vec![0.0; window_size],
            window_size,
            position: 0,
            samples: 0,
        }
    }

    /// update
    pub fn update(&mut self, min_val: f32, max_val: f32) {
        self.recent_mins[self.position] = min_val;
        self.recent_maxs[self.position] = max_val;
        self.position = (self.position + 1) % self.window_size;
        self.samples = (self.samples + 1).min(self.window_size);
    }

    #[must_use]
    /// get averaged range
    pub fn get_averaged_range(&self) -> (f32, f32) {
        if self.samples == 0 {
            return (0.0, 1.0);
        }

        let avg_min = self.recent_mins[..self.samples].iter().sum::<f32>() / self.samples as f32;
        let avg_max = self.recent_maxs[..self.samples].iter().sum::<f32>() / self.samples as f32;
        (avg_min, avg_max)
    }
}
