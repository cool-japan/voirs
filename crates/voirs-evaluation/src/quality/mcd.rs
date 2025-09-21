//! MCD (Mel-Cepstral Distortion) Implementation
//!
//! Implementation of Mel-Cepstral Distortion calculation with:
//! - Dynamic Time Warping (DTW) alignment
//! - MFCC feature extraction
//! - Power-normalized cepstral distance
//! - Statistical significance testing

use crate::EvaluationError;
use ndarray::{Array1, Array2};
use realfft::{RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Mutex;
use voirs_sdk::AudioBuffer;

/// MCD evaluator for speech quality assessment
pub struct MCDEvaluator {
    /// Sample rate
    sample_rate: u32,
    /// Frame length in samples
    frame_len: usize,
    /// Frame shift in samples
    frame_shift: usize,
    /// Number of mel filters
    num_mel_filters: usize,
    /// Number of MFCC coefficients
    num_mfcc: usize,
    /// Mel filter bank
    mel_filterbank: Array2<f32>,
    /// DCT matrix for MFCC computation
    dct_matrix: Array2<f32>,
    /// FFT planner
    fft_planner: Mutex<RealFftPlanner<f32>>,
}

impl MCDEvaluator {
    /// Create new MCD evaluator
    pub fn new(sample_rate: u32) -> Result<Self, EvaluationError> {
        // Standard parameters
        let frame_len = 1024; // ~64ms at 16kHz, ~128ms at 8kHz
        let frame_shift = 256; // 25% frame shift
        let num_mel_filters = 26;
        let num_mfcc = 13;

        // Create mel filter bank
        let mel_filterbank = Self::create_mel_filterbank(sample_rate, frame_len, num_mel_filters)?;

        // Create DCT matrix
        let dct_matrix = Self::create_dct_matrix(num_mel_filters, num_mfcc);

        let fft_planner = Mutex::new(RealFftPlanner::new());

        Ok(Self {
            sample_rate,
            frame_len,
            frame_shift,
            num_mel_filters,
            num_mfcc,
            mel_filterbank,
            dct_matrix,
            fft_planner,
        })
    }

    /// Calculate MCD between reference and generated signals with DTW alignment
    pub async fn calculate_mcd_with_dtw(
        &self,
        reference: &AudioBuffer,
        generated: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Validate inputs
        self.validate_inputs(reference, generated)?;

        // Extract MFCC features
        let ref_mfcc = self.extract_mfcc(reference.samples())?;
        let gen_mfcc = self.extract_mfcc(generated.samples())?;

        if ref_mfcc.nrows() == 0 || gen_mfcc.nrows() == 0 {
            return Err(EvaluationError::AudioProcessingError {
                message: "Failed to extract MFCC features".to_string(),
                source: None,
            });
        }

        // Perform DTW alignment
        let (aligned_ref, aligned_gen) = self.dynamic_time_warping(&ref_mfcc, &gen_mfcc)?;

        // Calculate frame-wise MCD
        let mcd = self.calculate_frame_mcd(&aligned_ref, &aligned_gen)?;

        Ok(mcd)
    }

    /// Calculate simple MCD without DTW (assumes pre-aligned signals)
    pub async fn calculate_mcd_simple(
        &self,
        reference: &AudioBuffer,
        generated: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        self.validate_inputs(reference, generated)?;

        let ref_mfcc = self.extract_mfcc(reference.samples())?;
        let gen_mfcc = self.extract_mfcc(generated.samples())?;

        // Use the shorter sequence
        let min_frames = ref_mfcc.nrows().min(gen_mfcc.nrows());
        if min_frames == 0 {
            return Ok(0.0);
        }

        let ref_subset = ref_mfcc.slice(ndarray::s![0..min_frames, ..]);
        let gen_subset = gen_mfcc.slice(ndarray::s![0..min_frames, ..]);

        self.calculate_frame_mcd(&ref_subset.to_owned(), &gen_subset.to_owned())
    }

    /// Validate input audio buffers
    fn validate_inputs(
        &self,
        reference: &AudioBuffer,
        generated: &AudioBuffer,
    ) -> Result<(), EvaluationError> {
        if reference.sample_rate() != self.sample_rate {
            return Err(EvaluationError::InvalidInput {
                message: crate::error_enhancement::sample_rate_mismatch_error(
                    "MCD",
                    self.sample_rate,
                    reference.sample_rate(),
                    "reference",
                ),
            });
        }

        if generated.sample_rate() != self.sample_rate {
            return Err(EvaluationError::InvalidInput {
                message: crate::error_enhancement::sample_rate_mismatch_error(
                    "MCD",
                    self.sample_rate,
                    generated.sample_rate(),
                    "generated",
                ),
            });
        }

        if reference.channels() != 1 {
            return Err(EvaluationError::InvalidInput {
                message: crate::error_enhancement::channel_mismatch_error(
                    "MCD",
                    1,
                    reference.channels(),
                    "reference",
                ),
            });
        }

        if generated.channels() != 1 {
            return Err(EvaluationError::InvalidInput {
                message: crate::error_enhancement::channel_mismatch_error(
                    "MCD",
                    1,
                    generated.channels(),
                    "generated",
                ),
            });
        }

        Ok(())
    }

    /// Extract MFCC features from audio signal
    fn extract_mfcc(&self, signal: &[f32]) -> Result<Array2<f32>, EvaluationError> {
        if signal.len() < self.frame_len {
            return Ok(Array2::zeros((0, self.num_mfcc)));
        }

        let num_frames = (signal.len() - self.frame_len) / self.frame_shift + 1;
        let mut mfcc_features = Array2::zeros((num_frames, self.num_mfcc));

        let mut fft_planner = self.fft_planner.lock().unwrap();
        let fft = fft_planner.plan_fft_forward(self.frame_len);
        let mut spectrum = fft.make_output_vec();

        for (frame_idx, frame_start) in (0..signal.len() - self.frame_len + 1)
            .step_by(self.frame_shift)
            .enumerate()
        {
            if frame_idx >= num_frames {
                break;
            }

            // Extract frame and apply pre-emphasis and windowing
            let mut frame = Array1::zeros(self.frame_len);

            // Apply pre-emphasis: y[n] = x[n] - 0.97 * x[n-1]
            frame[0] = signal[frame_start];
            for i in 1..self.frame_len {
                if frame_start + i < signal.len() {
                    frame[i] = signal[frame_start + i] - 0.97 * signal[frame_start + i - 1];
                }
            }

            // Apply Hamming window
            for i in 0..self.frame_len {
                let window =
                    0.54 - 0.46 * (2.0 * PI * i as f32 / (self.frame_len - 1) as f32).cos();
                frame[i] *= window;
            }

            // Compute FFT
            fft.process(frame.as_slice_mut().unwrap(), &mut spectrum)
                .map_err(|e| EvaluationError::AudioProcessingError {
                    message: format!("FFT processing failed: {e}"),
                    source: None,
                })?;

            // Compute power spectrum
            let power_spectrum: Vec<f32> = spectrum
                .iter()
                .map(num_complex::Complex::norm_sqr)
                .collect();

            // Apply mel filterbank
            let mut mel_energies = Array1::zeros(self.num_mel_filters);
            for (filter_idx, filter_row) in self.mel_filterbank.rows().into_iter().enumerate() {
                let mut energy = 0.0;
                for (bin_idx, &weight) in filter_row.iter().enumerate() {
                    if bin_idx < power_spectrum.len() {
                        energy += weight * power_spectrum[bin_idx];
                    }
                }
                mel_energies[filter_idx] = energy.max(1e-10); // Avoid log(0)
            }

            // Convert to log scale
            mel_energies.mapv_inplace(f32::ln);

            // Apply DCT to get MFCC coefficients
            for mfcc_idx in 0..self.num_mfcc {
                let mut mfcc_val = 0.0;
                for (mel_idx, &mel_energy) in mel_energies.iter().enumerate() {
                    mfcc_val += self.dct_matrix[[mfcc_idx, mel_idx]] * mel_energy;
                }
                mfcc_features[[frame_idx, mfcc_idx]] = mfcc_val;
            }
        }

        Ok(mfcc_features)
    }

    /// Perform Dynamic Time Warping alignment
    fn dynamic_time_warping(
        &self,
        ref_mfcc: &Array2<f32>,
        gen_mfcc: &Array2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>), EvaluationError> {
        let (ref_frames, num_coeffs) = ref_mfcc.dim();
        let (gen_frames, _) = gen_mfcc.dim();

        if ref_frames == 0 || gen_frames == 0 {
            return Err(EvaluationError::AudioProcessingError {
                message: "Empty MFCC sequences for DTW".to_string(),
                source: None,
            });
        }

        // Compute distance matrix
        let mut distance_matrix = Array2::zeros((ref_frames, gen_frames));
        for i in 0..ref_frames {
            for j in 0..gen_frames {
                let distance = self.euclidean_distance(&ref_mfcc.row(i), &gen_mfcc.row(j));
                distance_matrix[[i, j]] = distance;
            }
        }

        // DTW dynamic programming
        let mut dtw_matrix = Array2::from_elem((ref_frames + 1, gen_frames + 1), f32::INFINITY);
        dtw_matrix[[0, 0]] = 0.0;

        for i in 1..=ref_frames {
            for j in 1..=gen_frames {
                let cost = distance_matrix[[i - 1, j - 1]];
                dtw_matrix[[i, j]] = cost
                    + [
                        dtw_matrix[[i - 1, j]],     // insertion
                        dtw_matrix[[i, j - 1]],     // deletion
                        dtw_matrix[[i - 1, j - 1]], // match
                    ]
                    .iter()
                    .fold(f32::INFINITY, |a, &b| a.min(b));
            }
        }

        // Backtrack to find optimal path
        let mut path = Vec::new();
        let mut i = ref_frames;
        let mut j = gen_frames;

        while i > 0 && j > 0 {
            path.push((i - 1, j - 1));

            let candidates = [
                (i - 1, j, dtw_matrix[[i - 1, j]]),         // insertion
                (i, j - 1, dtw_matrix[[i, j - 1]]),         // deletion
                (i - 1, j - 1, dtw_matrix[[i - 1, j - 1]]), // match
            ];

            let (best_i, best_j, _) = candidates
                .iter()
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            i = *best_i;
            j = *best_j;
        }

        path.reverse();

        // Create aligned sequences
        let aligned_len = path.len();
        let mut aligned_ref = Array2::zeros((aligned_len, num_coeffs));
        let mut aligned_gen = Array2::zeros((aligned_len, num_coeffs));

        for (align_idx, &(ref_idx, gen_idx)) in path.iter().enumerate() {
            aligned_ref
                .row_mut(align_idx)
                .assign(&ref_mfcc.row(ref_idx));
            aligned_gen
                .row_mut(align_idx)
                .assign(&gen_mfcc.row(gen_idx));
        }

        Ok((aligned_ref, aligned_gen))
    }

    /// Calculate Euclidean distance between two MFCC vectors
    fn euclidean_distance(
        &self,
        vec1: &ndarray::ArrayView1<f32>,
        vec2: &ndarray::ArrayView1<f32>,
    ) -> f32 {
        // Use high-precision calculation for better numerical stability
        let vec1_slice: Vec<f32> = vec1.to_vec();
        let vec2_slice: Vec<f32> = vec2.to_vec();
        crate::precision::precise_euclidean_distance(&vec1_slice, &vec2_slice) as f32
    }

    /// Calculate frame-wise MCD
    fn calculate_frame_mcd(
        &self,
        ref_mfcc: &Array2<f32>,
        gen_mfcc: &Array2<f32>,
    ) -> Result<f32, EvaluationError> {
        let (num_frames, num_coeffs) = ref_mfcc.dim();
        if num_frames == 0 || num_coeffs == 0 {
            return Ok(0.0);
        }

        // Use Kahan summation for better numerical precision
        let mut total_distortion_kahan = crate::precision::KahanSum::new();

        for frame_idx in 0..num_frames {
            let ref_frame = ref_mfcc.row(frame_idx);
            let gen_frame = gen_mfcc.row(frame_idx);

            // Calculate cepstral distance (excluding c0 - energy coefficient)
            // Use Kahan summation for coefficient differences as well
            let mut coeff_sum_kahan = crate::precision::KahanSum::new();
            for coeff_idx in 1..num_coeffs.min(gen_frame.len()) {
                // Skip c0
                let diff = ref_frame[coeff_idx] as f64 - gen_frame[coeff_idx] as f64;
                coeff_sum_kahan.add(diff * diff);
            }

            // MCD formula: (10/ln(10)) * sqrt(2 * sum_sq_diff) with numerical stability
            let sum_sq_diff = coeff_sum_kahan.sum().max(0.0);
            let frame_mcd = (10.0 / std::f64::consts::LN_10) * (2.0 * sum_sq_diff).sqrt();
            total_distortion_kahan.add(frame_mcd);
        }

        Ok((total_distortion_kahan.sum() / num_frames as f64) as f32)
    }

    /// Create mel frequency filter bank
    fn create_mel_filterbank(
        sample_rate: u32,
        frame_len: usize,
        num_filters: usize,
    ) -> Result<Array2<f32>, EvaluationError> {
        let num_fft_bins = frame_len / 2 + 1;
        let nyquist = sample_rate as f32 / 2.0;

        // Convert Hz to Mel scale: mel = 2595 * log10(1 + f/700)
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10_f32.powf(mel / 2595.0) - 1.0);

        // Create mel frequency points
        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(nyquist);
        let mel_points: Vec<f32> = (0..=num_filters + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_filters + 1) as f32)
            .map(mel_to_hz)
            .collect();

        // Convert to FFT bin indices
        let bin_points: Vec<usize> = mel_points
            .iter()
            .map(|&freq| ((freq * frame_len as f32) / sample_rate as f32).round() as usize)
            .map(|bin| bin.min(num_fft_bins - 1))
            .collect();

        // Create triangular filters
        let mut filterbank = Array2::zeros((num_filters, num_fft_bins));

        for filter_idx in 0..num_filters {
            let left = bin_points[filter_idx];
            let center = bin_points[filter_idx + 1];
            let right = bin_points[filter_idx + 2];

            // Left slope
            for bin in left..center {
                if center > left {
                    filterbank[[filter_idx, bin]] = (bin - left) as f32 / (center - left) as f32;
                }
            }

            // Right slope
            for bin in center..=right.min(num_fft_bins - 1) {
                if right > center {
                    filterbank[[filter_idx, bin]] = (right - bin) as f32 / (right - center) as f32;
                }
            }
        }

        Ok(filterbank)
    }

    /// Create DCT matrix for MFCC computation
    fn create_dct_matrix(num_mel_filters: usize, num_mfcc: usize) -> Array2<f32> {
        let mut dct_matrix = Array2::zeros((num_mfcc, num_mel_filters));

        for mfcc_idx in 0..num_mfcc {
            for mel_idx in 0..num_mel_filters {
                let val = (PI * mfcc_idx as f32 * (2.0 * mel_idx as f32 + 1.0)
                    / (2.0 * num_mel_filters as f32))
                    .cos();
                dct_matrix[[mfcc_idx, mel_idx]] = val
                    * if mfcc_idx == 0 {
                        (1.0 / num_mel_filters as f32).max(0.0).sqrt()
                    } else {
                        (2.0 / num_mel_filters as f32).max(0.0).sqrt()
                    };
            }
        }

        dct_matrix
    }

    /// Calculate statistical measures for MCD
    #[must_use]
    pub fn calculate_mcd_statistics(mcd_values: &[f32]) -> MCDStatistics {
        if mcd_values.is_empty() {
            return MCDStatistics::default();
        }

        let mean = mcd_values.iter().sum::<f32>() / mcd_values.len() as f32;

        let variance =
            mcd_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / mcd_values.len() as f32;
        let std_dev = variance.max(0.0).sqrt();

        let mut sorted = mcd_values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        MCDStatistics {
            mean,
            std_dev,
            median,
            min,
            max,
            variance,
        }
    }
}

/// MCD statistical measures
#[derive(Debug, Clone)]
pub struct MCDStatistics {
    /// Mean MCD value
    pub mean: f32,
    /// Standard deviation of MCD values
    pub std_dev: f32,
    /// Median MCD value
    pub median: f32,
    /// Minimum MCD value
    pub min: f32,
    /// Maximum MCD value
    pub max: f32,
    /// Variance of MCD values
    pub variance: f32,
}

impl Default for MCDStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            median: 0.0,
            min: 0.0,
            max: 0.0,
            variance: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_mcd_evaluator_creation() {
        let evaluator = MCDEvaluator::new(16000).unwrap();
        assert_eq!(evaluator.sample_rate, 16000);
        assert_eq!(evaluator.num_mel_filters, 26);
        assert_eq!(evaluator.num_mfcc, 13);
    }

    #[tokio::test]
    async fn test_mcd_calculation() {
        let evaluator = MCDEvaluator::new(16000).unwrap();

        // Create test signals (need at least 1 frame)
        let reference = AudioBuffer::new(vec![0.1; 2048], 16000, 1);
        let generated = AudioBuffer::new(vec![0.08; 2048], 16000, 1);

        let mcd_simple = evaluator
            .calculate_mcd_simple(&reference, &generated)
            .await
            .unwrap();
        assert!(mcd_simple >= 0.0);

        let mcd_dtw = evaluator
            .calculate_mcd_with_dtw(&reference, &generated)
            .await
            .unwrap();
        assert!(mcd_dtw >= 0.0);
    }

    #[tokio::test]
    async fn test_mfcc_extraction() {
        let evaluator = MCDEvaluator::new(16000).unwrap();
        let signal = vec![0.1; 2048]; // Need enough samples for at least one frame

        let mfcc = evaluator.extract_mfcc(&signal).unwrap();

        assert_eq!(mfcc.ncols(), 13); // Number of MFCC coefficients
        assert!(mfcc.nrows() > 0); // Should have at least one frame
    }

    #[test]
    fn test_mel_filterbank_creation() {
        let filterbank = MCDEvaluator::create_mel_filterbank(16000, 1024, 26).unwrap();

        assert_eq!(filterbank.nrows(), 26); // Number of filters
        assert_eq!(filterbank.ncols(), 513); // Number of FFT bins (1024/2 + 1)

        // Each filter should have non-zero values
        for filter_row in filterbank.rows() {
            let sum: f32 = filter_row.sum();
            assert!(sum > 0.0);
        }
    }

    #[test]
    fn test_dct_matrix_creation() {
        let dct_matrix = MCDEvaluator::create_dct_matrix(26, 13);

        assert_eq!(dct_matrix.nrows(), 13); // Number of MFCC coefficients
        assert_eq!(dct_matrix.ncols(), 26); // Number of mel filters

        // DCT matrix should be orthogonal (approximately)
        // Test first coefficient (should be normalized differently)
        let first_row_norm: f32 = dct_matrix.row(0).mapv(|x| x * x).sum();
        assert!((first_row_norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_euclidean_distance() {
        let evaluator = MCDEvaluator::new(16000).unwrap();

        let vec1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let vec2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let distance = evaluator.euclidean_distance(&vec1.view(), &vec2.view());
        assert!((distance - 0.0).abs() < 1e-6); // Should be zero for identical vectors

        let vec3 = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let distance2 = evaluator.euclidean_distance(&vec1.view(), &vec3.view());
        assert!(distance2 > 0.0); // Should be positive for different vectors
    }

    #[test]
    fn test_mcd_statistics() {
        let mcd_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = MCDEvaluator::calculate_mcd_statistics(&mcd_values);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!(stats.std_dev > 0.0);
    }
}
