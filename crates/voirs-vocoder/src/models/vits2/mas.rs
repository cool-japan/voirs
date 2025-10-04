//! Monotonic Alignment Search (MAS) for VITS2
//!
//! This module implements the Monotonic Alignment Search algorithm which is crucial
//! for learning alignments between text and audio in VITS2 without requiring
//! external alignment data.

use crate::{Result, VocoderError};
use serde::{Deserialize, Serialize};

/// Configuration for Monotonic Alignment Search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MASConfig {
    /// Maximum path length factor
    pub max_path_factor: f32,
    /// Noise scale for exploration
    pub noise_scale: f32,
    /// Noise scale decay factor
    pub noise_scale_decay: f32,
    /// Minimum noise scale
    pub min_noise_scale: f32,
    /// Use Viterbi algorithm for alignment
    pub use_viterbi: bool,
    /// Temperature for soft alignment
    pub temperature: f32,
    /// Use hard alignment during inference
    pub hard_alignment: bool,
}

impl Default for MASConfig {
    fn default() -> Self {
        Self {
            max_path_factor: 2.0,
            noise_scale: 1.0,
            noise_scale_decay: 0.999,
            min_noise_scale: 0.01,
            use_viterbi: true,
            temperature: 1.0,
            hard_alignment: true,
        }
    }
}

/// Alignment matrix representing the correspondence between text and audio
#[derive(Debug, Clone)]
pub struct AlignmentMatrix {
    /// Alignment weights [text_len, audio_len]
    pub weights: Vec<Vec<f32>>,
    /// Text sequence length
    pub text_len: usize,
    /// Audio sequence length
    pub audio_len: usize,
    /// Whether this is a hard alignment (0/1) or soft (probabilities)
    pub is_hard: bool,
}

impl AlignmentMatrix {
    /// Create new alignment matrix
    pub fn new(text_len: usize, audio_len: usize, is_hard: bool) -> Self {
        let weights = vec![vec![0.0; audio_len]; text_len];
        Self {
            weights,
            text_len,
            audio_len,
            is_hard,
        }
    }

    /// Create uniform alignment as initialization
    pub fn uniform(text_len: usize, audio_len: usize) -> Self {
        let mut matrix = Self::new(text_len, audio_len, false);
        let uniform_weight = 1.0 / audio_len as f32;

        for i in 0..text_len {
            for j in 0..audio_len {
                matrix.weights[i][j] = uniform_weight;
            }
        }

        matrix
    }

    /// Get alignment path (for hard alignments)
    pub fn get_path(&self) -> Result<Vec<usize>> {
        if !self.is_hard {
            return Err(VocoderError::VocodingError(
                "Cannot extract path from soft alignment".to_string(),
            ));
        }

        let mut path = Vec::new();

        for i in 0..self.text_len {
            let mut max_j = 0;
            let mut max_weight = self.weights[i][0];

            for j in 1..self.audio_len {
                if self.weights[i][j] > max_weight {
                    max_weight = self.weights[i][j];
                    max_j = j;
                }
            }

            if max_weight > 0.0 {
                path.push(max_j);
            }
        }

        Ok(path)
    }

    /// Convert soft alignment to hard alignment
    pub fn to_hard(&self) -> AlignmentMatrix {
        let mut hard_matrix = AlignmentMatrix::new(self.text_len, self.audio_len, true);

        for i in 0..self.text_len {
            let mut max_j = 0;
            let mut max_weight = self.weights[i][0];

            for j in 1..self.audio_len {
                if self.weights[i][j] > max_weight {
                    max_weight = self.weights[i][j];
                    max_j = j;
                }
            }

            hard_matrix.weights[i][max_j] = 1.0;
        }

        hard_matrix
    }

    /// Validate alignment (monotonicity check)
    pub fn validate_monotonic(&self) -> Result<()> {
        if !self.is_hard {
            return Ok(()); // Only check hard alignments
        }

        let path = self.get_path()?;

        for i in 1..path.len() {
            if path[i] < path[i - 1] {
                return Err(VocoderError::VocodingError(
                    "Alignment is not monotonic".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Calculate alignment cost/loss
    pub fn calculate_cost(&self, log_probs: &[Vec<f32>]) -> Result<f32> {
        if log_probs.len() != self.text_len {
            return Err(VocoderError::VocodingError(
                "Log probabilities length mismatch".to_string(),
            ));
        }

        let mut total_cost = 0.0;
        let mut count = 0;

        for (i, log_prob_row) in log_probs.iter().enumerate() {
            if log_prob_row.len() != self.audio_len {
                return Err(VocoderError::VocodingError(
                    "Log probabilities dimension mismatch".to_string(),
                ));
            }

            for (j, &log_prob_val) in log_prob_row.iter().enumerate() {
                if self.weights[i][j] > 0.0 {
                    total_cost += self.weights[i][j] * log_prob_val;
                    count += 1;
                }
            }
        }

        Ok(if count > 0 {
            total_cost / count as f32
        } else {
            0.0
        })
    }
}

/// Monotonic Alignment Search implementation
#[derive(Debug, Clone)]
pub struct MonotonicAlignmentSearch {
    /// Configuration
    pub config: MASConfig,
    /// Current training step (for noise decay)
    pub step: u64,
}

impl MonotonicAlignmentSearch {
    /// Create new MAS instance
    pub fn new(config: MASConfig) -> Self {
        Self { config, step: 0 }
    }

    /// Find optimal alignment using dynamic programming
    pub fn find_alignment(
        &mut self,
        log_probs: &[Vec<f32>],
        text_len: usize,
        audio_len: usize,
        training: bool,
    ) -> Result<AlignmentMatrix> {
        if log_probs.len() != text_len {
            return Err(VocoderError::VocodingError(
                "Log probabilities length mismatch".to_string(),
            ));
        }

        if text_len == 0 || audio_len == 0 {
            return Err(VocoderError::VocodingError(
                "Invalid sequence lengths".to_string(),
            ));
        }

        // Check maximum path constraint
        let max_audio_len = (text_len as f32 * self.config.max_path_factor) as usize;
        if audio_len > max_audio_len {
            return Err(VocoderError::VocodingError(format!(
                "Audio length {} exceeds maximum allowed {} for text length {}",
                audio_len, max_audio_len, text_len
            )));
        }

        if training && self.config.use_viterbi {
            self.viterbi_alignment(log_probs, text_len, audio_len)
        } else {
            self.forward_algorithm(log_probs, text_len, audio_len)
        }
    }

    /// Viterbi algorithm for finding best alignment path
    fn viterbi_alignment(
        &mut self,
        log_probs: &[Vec<f32>],
        text_len: usize,
        audio_len: usize,
    ) -> Result<AlignmentMatrix> {
        // Initialize DP table
        let mut dp = vec![vec![f32::NEG_INFINITY; audio_len]; text_len];
        let mut backtrack = vec![vec![0usize; audio_len]; text_len];

        // Add noise for exploration during training
        let current_noise_scale = self.get_current_noise_scale();

        // Initialize first row
        for j in 0..audio_len {
            let noise = if current_noise_scale > 0.0 {
                (fastrand::f32() - 0.5) * current_noise_scale
            } else {
                0.0
            };
            dp[0][j] = log_probs[0][j] + noise;
        }

        // Fill DP table
        for i in 1..text_len {
            for j in 0..audio_len {
                if log_probs[i].len() != audio_len {
                    return Err(VocoderError::VocodingError(
                        "Log probabilities dimension mismatch".to_string(),
                    ));
                }

                let noise = if current_noise_scale > 0.0 {
                    (fastrand::f32() - 0.5) * current_noise_scale
                } else {
                    0.0
                };

                // Consider transitions from previous time step
                for prev_j in 0..=j {
                    let score = dp[i - 1][prev_j] + log_probs[i][j] + noise;
                    if score > dp[i][j] {
                        dp[i][j] = score;
                        backtrack[i][j] = prev_j;
                    }
                }
            }
        }

        // Backtrack to find best path
        let mut alignment = AlignmentMatrix::new(text_len, audio_len, true);

        // Find best ending position
        let mut best_j = 0;
        let mut best_score = dp[text_len - 1][0];
        for j in 1..audio_len {
            if dp[text_len - 1][j] > best_score {
                best_score = dp[text_len - 1][j];
                best_j = j;
            }
        }

        // Backtrack path
        let mut current_j = best_j;
        for i in (0..text_len).rev() {
            alignment.weights[i][current_j] = 1.0;
            if i > 0 {
                current_j = backtrack[i][current_j];
            }
        }

        // Validate monotonicity
        alignment.validate_monotonic()?;

        self.step += 1;
        Ok(alignment)
    }

    /// Forward algorithm for soft alignment
    fn forward_algorithm(
        &self,
        log_probs: &[Vec<f32>],
        text_len: usize,
        audio_len: usize,
    ) -> Result<AlignmentMatrix> {
        // Initialize forward table
        let mut alpha = vec![vec![f32::NEG_INFINITY; audio_len]; text_len];

        // Initialize first row
        for j in 0..audio_len {
            alpha[0][j] = log_probs[0][j];
        }

        // Forward pass
        for i in 1..text_len {
            for j in 0..audio_len {
                if log_probs[i].len() != audio_len {
                    return Err(VocoderError::VocodingError(
                        "Log probabilities dimension mismatch".to_string(),
                    ));
                }

                let mut log_sum = f32::NEG_INFINITY;

                // Sum over all previous positions (with monotonicity constraint)
                for prev_j in 0..=j {
                    let score = alpha[i - 1][prev_j];
                    if score > f32::NEG_INFINITY {
                        log_sum = log_sum_exp(log_sum, score);
                    }
                }

                alpha[i][j] = log_sum + log_probs[i][j];
            }
        }

        // Convert to probabilities
        let mut alignment = AlignmentMatrix::new(text_len, audio_len, false);

        for (i, alpha_row) in alpha.iter().enumerate() {
            // Compute normalizing constant for this time step
            let mut log_norm = f32::NEG_INFINITY;
            for &alpha_val in alpha_row.iter() {
                log_norm = log_sum_exp(log_norm, alpha_val);
            }

            // Normalize to get probabilities
            for (j, &alpha_val) in alpha_row.iter().enumerate() {
                alignment.weights[i][j] = (alpha_val - log_norm).exp();
            }
        }

        // Convert to hard alignment if requested
        if self.config.hard_alignment {
            Ok(alignment.to_hard())
        } else {
            Ok(alignment)
        }
    }

    /// Get current noise scale (with decay)
    fn get_current_noise_scale(&self) -> f32 {
        let decayed_scale =
            self.config.noise_scale * self.config.noise_scale_decay.powi(self.step as i32);
        decayed_scale.max(self.config.min_noise_scale)
    }

    /// Update training step
    pub fn step(&mut self) {
        self.step += 1;
    }

    /// Reset step counter
    pub fn reset_step(&mut self) {
        self.step = 0;
    }

    /// Get training statistics
    pub fn get_stats(&self) -> MASStats {
        MASStats {
            step: self.step,
            current_noise_scale: self.get_current_noise_scale(),
        }
    }
}

/// Training statistics for MAS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MASStats {
    /// Current training step
    pub step: u64,
    /// Current noise scale
    pub current_noise_scale: f32,
}

/// Utility function for log-sum-exp computation
fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY && b == f32::NEG_INFINITY {
        f32::NEG_INFINITY
    } else if a == f32::NEG_INFINITY {
        b
    } else if b == f32::NEG_INFINITY {
        a
    } else {
        let max_val = a.max(b);
        max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
    }
}

/// Batch alignment processing
pub struct BatchAlignmentProcessor {
    /// MAS instance
    pub mas: MonotonicAlignmentSearch,
    /// Maximum batch size
    pub max_batch_size: usize,
}

impl BatchAlignmentProcessor {
    /// Create new batch processor
    pub fn new(config: MASConfig, max_batch_size: usize) -> Self {
        Self {
            mas: MonotonicAlignmentSearch::new(config),
            max_batch_size,
        }
    }

    /// Process batch of alignments
    pub fn process_batch(
        &mut self,
        batch_log_probs: &[Vec<Vec<f32>>],
        text_lengths: &[usize],
        audio_lengths: &[usize],
        training: bool,
    ) -> Result<Vec<AlignmentMatrix>> {
        if batch_log_probs.len() != text_lengths.len() || text_lengths.len() != audio_lengths.len()
        {
            return Err(VocoderError::VocodingError(
                "Batch size mismatch".to_string(),
            ));
        }

        if batch_log_probs.len() > self.max_batch_size {
            return Err(VocoderError::VocodingError(format!(
                "Batch size {} exceeds maximum {}",
                batch_log_probs.len(),
                self.max_batch_size
            )));
        }

        let mut alignments = Vec::new();

        for (i, log_probs) in batch_log_probs.iter().enumerate() {
            let alignment =
                self.mas
                    .find_alignment(log_probs, text_lengths[i], audio_lengths[i], training)?;
            alignments.push(alignment);
        }

        Ok(alignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_matrix() {
        let mut matrix = AlignmentMatrix::new(3, 5, true);
        matrix.weights[0][0] = 1.0;
        matrix.weights[1][2] = 1.0;
        matrix.weights[2][4] = 1.0;

        let path = matrix.get_path().unwrap();
        assert_eq!(path, vec![0, 2, 4]);
        assert!(matrix.validate_monotonic().is_ok());
    }

    #[test]
    fn test_uniform_alignment() {
        let matrix = AlignmentMatrix::uniform(2, 4);
        assert_eq!(matrix.text_len, 2);
        assert_eq!(matrix.audio_len, 4);
        assert!(!matrix.is_hard);

        // Check that weights sum to 1 for each text position
        for i in 0..matrix.text_len {
            let sum: f32 = matrix.weights[i].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_soft_to_hard_conversion() {
        let mut soft_matrix = AlignmentMatrix::new(2, 3, false);
        soft_matrix.weights[0] = vec![0.1, 0.7, 0.2];
        soft_matrix.weights[1] = vec![0.3, 0.2, 0.5];

        let hard_matrix = soft_matrix.to_hard();
        assert!(hard_matrix.is_hard);
        assert_eq!(hard_matrix.weights[0][1], 1.0); // Max at position 1
        assert_eq!(hard_matrix.weights[1][2], 1.0); // Max at position 2
    }

    #[test]
    fn test_monotonic_alignment_search() {
        let config = MASConfig::default();
        let mut mas = MonotonicAlignmentSearch::new(config);

        // Create simple log probabilities
        let log_probs = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-3.0, -1.0, -2.0],
            vec![-2.0, -3.0, -1.0],
        ];

        let alignment = mas.find_alignment(&log_probs, 3, 3, true).unwrap();
        assert_eq!(alignment.text_len, 3);
        assert_eq!(alignment.audio_len, 3);
        assert!(alignment.validate_monotonic().is_ok());
    }

    #[test]
    fn test_viterbi_alignment() {
        let config = MASConfig {
            use_viterbi: true,
            noise_scale: 0.0, // No noise for deterministic test
            ..Default::default()
        };
        let mut mas = MonotonicAlignmentSearch::new(config);

        let log_probs = vec![vec![-1.0, -2.0], vec![-2.0, -1.0]];

        let alignment = mas.viterbi_alignment(&log_probs, 2, 2).unwrap();
        let path = alignment.get_path().unwrap();
        assert_eq!(path, vec![0, 1]); // Should follow the higher probabilities
    }

    #[test]
    fn test_forward_algorithm() {
        let config = MASConfig {
            use_viterbi: false,
            hard_alignment: false,
            ..Default::default()
        };
        let mas = MonotonicAlignmentSearch::new(config);

        let log_probs = vec![vec![-1.0, -2.0], vec![-2.0, -1.0]];

        let alignment = mas.forward_algorithm(&log_probs, 2, 2).unwrap();
        assert!(!alignment.is_hard);

        // Check that probabilities sum to 1 for each text position
        for i in 0..alignment.text_len {
            let sum: f32 = alignment.weights[i].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_noise_scale_decay() {
        let config = MASConfig {
            noise_scale: 1.0,
            noise_scale_decay: 0.9,
            min_noise_scale: 0.1,
            ..Default::default()
        };
        let mut mas = MonotonicAlignmentSearch::new(config);

        let initial_noise = mas.get_current_noise_scale();
        assert_eq!(initial_noise, 1.0);

        mas.step();
        let decayed_noise = mas.get_current_noise_scale();
        assert!(decayed_noise < initial_noise);
        assert!(decayed_noise >= 0.1); // Should not go below minimum
    }

    #[test]
    fn test_batch_alignment_processor() {
        let config = MASConfig::default();
        let mut processor = BatchAlignmentProcessor::new(config, 10);

        let batch_log_probs = vec![
            vec![vec![-1.0, -2.0], vec![-2.0, -1.0]],
            vec![vec![-1.0, -3.0], vec![-3.0, -1.0]],
        ];
        let text_lengths = vec![2, 2];
        let audio_lengths = vec![2, 2];

        let alignments = processor
            .process_batch(&batch_log_probs, &text_lengths, &audio_lengths, false)
            .unwrap();

        assert_eq!(alignments.len(), 2);
        for alignment in &alignments {
            assert!(alignment.validate_monotonic().is_ok());
        }
    }

    #[test]
    fn test_log_sum_exp() {
        assert_eq!(
            log_sum_exp(f32::NEG_INFINITY, f32::NEG_INFINITY),
            f32::NEG_INFINITY
        );
        assert_eq!(log_sum_exp(0.0, f32::NEG_INFINITY), 0.0);
        assert_eq!(log_sum_exp(f32::NEG_INFINITY, 1.0), 1.0);

        let result = log_sum_exp(0.0, 0.0);
        assert!((result - 2.0_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_alignment_cost() {
        let mut matrix = AlignmentMatrix::new(2, 2, false);
        matrix.weights[0] = vec![0.6, 0.4];
        matrix.weights[1] = vec![0.3, 0.7];

        let log_probs = vec![vec![-1.0, -2.0], vec![-2.0, -1.0]];

        let cost = matrix.calculate_cost(&log_probs).unwrap();
        assert!(cost < 0.0); // Should be negative (log probabilities)
    }
}
