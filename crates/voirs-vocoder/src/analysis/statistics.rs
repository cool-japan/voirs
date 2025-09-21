//! Statistical analysis tools for audio signals
//!
//! Provides comprehensive statistical measures including:
//! - Basic statistics (mean, variance, skewness, kurtosis)
//! - Information theory measures (entropy, complexity)
//! - Distribution analysis
//! - Temporal statistics

use crate::{Result, VocoderError};
use ndarray::{Array1, Array2, s};
use std::collections::HashMap;

/// Comprehensive statistical analysis result
#[derive(Debug, Clone)]
pub struct StatisticsAnalysis {
    /// Basic statistical measures
    pub basic_stats: BasicStatistics,
    
    /// Distribution analysis
    pub distribution: DistributionAnalysis,
    
    /// Information theory measures
    pub information: InformationMeasures,
    
    /// Temporal statistics
    pub temporal: TemporalStatistics,
    
    /// Complexity measures
    pub complexity: ComplexityMeasures,
}

/// Basic statistical measures
#[derive(Debug, Clone)]
pub struct BasicStatistics {
    /// Sample mean
    pub mean: f32,
    
    /// Sample variance
    pub variance: f32,
    
    /// Standard deviation
    pub std_dev: f32,
    
    /// Skewness (asymmetry)
    pub skewness: f32,
    
    /// Kurtosis (tail heaviness)
    pub kurtosis: f32,
    
    /// Minimum value
    pub min: f32,
    
    /// Maximum value
    pub max: f32,
    
    /// Range
    pub range: f32,
    
    /// Median
    pub median: f32,
    
    /// Mode (most frequent value)
    pub mode: f32,
    
    /// Interquartile range
    pub iqr: f32,
    
    /// Root mean square
    pub rms: f32,
    
    /// Peak-to-peak amplitude
    pub peak_to_peak: f32,
    
    /// Crest factor (peak/RMS ratio)
    pub crest_factor: f32,
}

/// Distribution analysis
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    /// Histogram bins
    pub histogram_bins: Vec<f32>,
    
    /// Histogram counts
    pub histogram_counts: Vec<u32>,
    
    /// Percentiles (5th, 25th, 75th, 95th)
    pub percentiles: [f32; 4],
    
    /// Probability density function estimate
    pub pdf_estimate: Vec<f32>,
    
    /// Cumulative distribution function
    pub cdf: Vec<f32>,
    
    /// Distribution shape measures
    pub shape_measures: ShapeMeasures,
}

/// Shape measures for distribution
#[derive(Debug, Clone)]
pub struct ShapeMeasures {
    /// Gaussianity measure (how close to normal distribution)
    pub gaussianity: f32,
    
    /// Bimodality coefficient
    pub bimodality: f32,
    
    /// Outlier ratio
    pub outlier_ratio: f32,
    
    /// Distribution symmetry
    pub symmetry: f32,
}

/// Information theory measures
#[derive(Debug, Clone)]
pub struct InformationMeasures {
    /// Shannon entropy
    pub shannon_entropy: f32,
    
    /// Differential entropy
    pub differential_entropy: f32,
    
    /// Kolmogorov complexity estimate
    pub kolmogorov_complexity: f32,
    
    /// Approximate entropy
    pub approximate_entropy: f32,
    
    /// Sample entropy
    pub sample_entropy: f32,
    
    /// Permutation entropy
    pub permutation_entropy: f32,
    
    /// Spectral entropy
    pub spectral_entropy: f32,
}

/// Temporal statistics
#[derive(Debug, Clone)]
pub struct TemporalStatistics {
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    
    /// Mean crossing rate
    pub mean_crossing_rate: f32,
    
    /// Autocorrelation function
    pub autocorrelation: Vec<f32>,
    
    /// Peak autocorrelation lag
    pub autocorr_peak_lag: usize,
    
    /// Periodicity strength
    pub periodicity_strength: f32,
    
    /// Stationarity measure
    pub stationarity: f32,
    
    /// Trend strength
    pub trend_strength: f32,
    
    /// Seasonality strength
    pub seasonality_strength: f32,
}

/// Complexity measures
#[derive(Debug, Clone)]
pub struct ComplexityMeasures {
    /// Fractal dimension
    pub fractal_dimension: f32,
    
    /// Hurst exponent
    pub hurst_exponent: f32,
    
    /// Lyapunov exponent estimate
    pub lyapunov_exponent: f32,
    
    /// Correlation dimension
    pub correlation_dimension: f32,
    
    /// Lempel-Ziv complexity
    pub lempel_ziv_complexity: f32,
    
    /// Multiscale entropy
    pub multiscale_entropy: Vec<f32>,
}

impl Default for StatisticsAnalysis {
    fn default() -> Self {
        Self {
            basic_stats: BasicStatistics::default(),
            distribution: DistributionAnalysis::default(),
            information: InformationMeasures::default(),
            temporal: TemporalStatistics::default(),
            complexity: ComplexityMeasures::default(),
        }
    }
}

impl Default for BasicStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            min: 0.0,
            max: 0.0,
            range: 0.0,
            median: 0.0,
            mode: 0.0,
            iqr: 0.0,
            rms: 0.0,
            peak_to_peak: 0.0,
            crest_factor: 0.0,
        }
    }
}

impl Default for DistributionAnalysis {
    fn default() -> Self {
        Self {
            histogram_bins: Vec::new(),
            histogram_counts: Vec::new(),
            percentiles: [0.0; 4],
            pdf_estimate: Vec::new(),
            cdf: Vec::new(),
            shape_measures: ShapeMeasures::default(),
        }
    }
}

impl Default for ShapeMeasures {
    fn default() -> Self {
        Self {
            gaussianity: 0.0,
            bimodality: 0.0,
            outlier_ratio: 0.0,
            symmetry: 0.0,
        }
    }
}

impl Default for InformationMeasures {
    fn default() -> Self {
        Self {
            shannon_entropy: 0.0,
            differential_entropy: 0.0,
            kolmogorov_complexity: 0.0,
            approximate_entropy: 0.0,
            sample_entropy: 0.0,
            permutation_entropy: 0.0,
            spectral_entropy: 0.0,
        }
    }
}

impl Default for TemporalStatistics {
    fn default() -> Self {
        Self {
            zero_crossing_rate: 0.0,
            mean_crossing_rate: 0.0,
            autocorrelation: Vec::new(),
            autocorr_peak_lag: 0,
            periodicity_strength: 0.0,
            stationarity: 0.0,
            trend_strength: 0.0,
            seasonality_strength: 0.0,
        }
    }
}

impl Default for ComplexityMeasures {
    fn default() -> Self {
        Self {
            fractal_dimension: 0.0,
            hurst_exponent: 0.5,
            lyapunov_exponent: 0.0,
            correlation_dimension: 0.0,
            lempel_ziv_complexity: 0.0,
            multiscale_entropy: Vec::new(),
        }
    }
}

/// Statistical analyzer
pub struct StatisticalAnalyzer {
    sample_rate: u32,
}

impl StatisticalAnalyzer {
    /// Create new statistical analyzer
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }
    
    /// Perform comprehensive statistical analysis
    pub fn analyze(&self, samples: &Array1<f32>) -> Result<StatisticsAnalysis> {
        if samples.is_empty() {
            return Ok(StatisticsAnalysis::default());
        }
        
        let basic_stats = self.compute_basic_statistics(samples);
        let distribution = self.analyze_distribution(samples)?;
        let information = self.compute_information_measures(samples)?;
        let temporal = self.compute_temporal_statistics(samples)?;
        let complexity = self.compute_complexity_measures(samples)?;
        
        Ok(StatisticsAnalysis {
            basic_stats,
            distribution,
            information,
            temporal,
            complexity,
        })
    }
    
    /// Compute basic statistical measures
    fn compute_basic_statistics(&self, samples: &Array1<f32>) -> BasicStatistics {
        let n = samples.len() as f32;
        
        // Basic moments
        let mean = samples.mean().unwrap_or(0.0);
        let variance = samples.var(1.0); // Using sample variance (n-1)
        let std_dev = variance.sqrt();
        
        // Skewness and kurtosis
        let skewness = self.compute_skewness(samples, mean, std_dev);
        let kurtosis = self.compute_kurtosis(samples, mean, std_dev);
        
        // Min, max, range
        let min = samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;
        let peak_to_peak = range;
        
        // Median and mode
        let median = self.compute_median(samples);
        let mode = self.compute_mode(samples);
        
        // Interquartile range
        let iqr = self.compute_iqr(samples);
        
        // RMS and crest factor
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / n).sqrt();
        let crest_factor = if rms > 1e-10 { max.abs().max((-min).abs()) / rms } else { 0.0 };
        
        BasicStatistics {
            mean,
            variance,
            std_dev,
            skewness,
            kurtosis,
            min,
            max,
            range,
            median,
            mode,
            iqr,
            rms,
            peak_to_peak,
            crest_factor,
        }
    }
    
    /// Compute skewness
    fn compute_skewness(&self, samples: &Array1<f32>, mean: f32, std_dev: f32) -> f32 {
        if std_dev <= 1e-10 {
            return 0.0;
        }
        
        let n = samples.len() as f32;
        let skew_sum: f32 = samples.iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum();
        
        skew_sum / n
    }
    
    /// Compute kurtosis
    fn compute_kurtosis(&self, samples: &Array1<f32>, mean: f32, std_dev: f32) -> f32 {
        if std_dev <= 1e-10 {
            return 0.0;
        }
        
        let n = samples.len() as f32;
        let kurt_sum: f32 = samples.iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum();
        
        kurt_sum / n - 3.0 // Excess kurtosis
    }
    
    /// Compute median
    fn compute_median(&self, samples: &Array1<f32>) -> f32 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n/2 - 1] + sorted[n/2]) / 2.0
        } else {
            sorted[n/2]
        }
    }
    
    /// Compute mode (most frequent value, approximated using histogram)
    fn compute_mode(&self, samples: &Array1<f32>) -> f32 {
        let bins = 100;
        let (hist_bins, hist_counts) = self.compute_histogram(samples, bins);
        
        // Find bin with maximum count
        let max_idx = hist_counts.iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        hist_bins.get(max_idx).copied().unwrap_or(0.0)
    }
    
    /// Compute interquartile range
    fn compute_iqr(&self, samples: &Array1<f32>) -> f32 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        
        if q3_idx < n && q1_idx < n {
            sorted[q3_idx] - sorted[q1_idx]
        } else {
            0.0
        }
    }
    
    /// Analyze distribution
    fn analyze_distribution(&self, samples: &Array1<f32>) -> Result<DistributionAnalysis> {
        let bins = 50;
        let (histogram_bins, histogram_counts) = self.compute_histogram(samples, bins);
        
        // Compute percentiles
        let percentiles = self.compute_percentiles(samples, &[5.0, 25.0, 75.0, 95.0]);
        
        // Estimate PDF and CDF
        let pdf_estimate = self.estimate_pdf(&histogram_counts);
        let cdf = self.compute_cdf(&pdf_estimate);
        
        // Shape measures
        let shape_measures = self.compute_shape_measures(samples, &histogram_counts)?;
        
        Ok(DistributionAnalysis {
            histogram_bins,
            histogram_counts,
            percentiles: [percentiles[0], percentiles[1], percentiles[2], percentiles[3]],
            pdf_estimate,
            cdf,
            shape_measures,
        })
    }
    
    /// Compute histogram
    fn compute_histogram(&self, samples: &Array1<f32>, bins: usize) -> (Vec<f32>, Vec<u32>) {
        let min = samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max - min).abs() < 1e-10 {
            return (vec![min], vec![samples.len() as u32]);
        }
        
        let bin_width = (max - min) / bins as f32;
        let mut hist_bins = Vec::with_capacity(bins);
        let mut hist_counts = vec![0u32; bins];
        
        // Create bin centers
        for i in 0..bins {
            hist_bins.push(min + (i as f32 + 0.5) * bin_width);
        }
        
        // Count samples in each bin
        for &sample in samples.iter() {
            let bin_idx = ((sample - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            hist_counts[bin_idx] += 1;
        }
        
        (hist_bins, hist_counts)
    }
    
    /// Compute percentiles
    fn compute_percentiles(&self, samples: &Array1<f32>, percentiles: &[f32]) -> Vec<f32> {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        percentiles.iter()
            .map(|&p| {
                let idx = ((p / 100.0) * (sorted.len() - 1) as f32) as usize;
                sorted[idx.min(sorted.len() - 1)]
            })
            .collect()
    }
    
    /// Estimate probability density function
    fn estimate_pdf(&self, histogram_counts: &[u32]) -> Vec<f32> {
        let total_count: u32 = histogram_counts.iter().sum();
        if total_count == 0 {
            return vec![0.0; histogram_counts.len()];
        }
        
        histogram_counts.iter()
            .map(|&count| count as f32 / total_count as f32)
            .collect()
    }
    
    /// Compute cumulative distribution function
    fn compute_cdf(&self, pdf: &[f32]) -> Vec<f32> {
        let mut cdf = Vec::with_capacity(pdf.len());
        let mut cumulative = 0.0;
        
        for &prob in pdf.iter() {
            cumulative += prob;
            cdf.push(cumulative);
        }
        
        cdf
    }
    
    /// Compute shape measures
    fn compute_shape_measures(&self, samples: &Array1<f32>, histogram_counts: &[u32]) -> Result<ShapeMeasures> {
        let gaussianity = self.compute_gaussianity(samples);
        let bimodality = self.compute_bimodality(histogram_counts);
        let outlier_ratio = self.compute_outlier_ratio(samples);
        let symmetry = self.compute_symmetry(samples);
        
        Ok(ShapeMeasures {
            gaussianity,
            bimodality,
            outlier_ratio,
            symmetry,
        })
    }
    
    /// Compute gaussianity measure
    fn compute_gaussianity(&self, samples: &Array1<f32>) -> f32 {
        // Simple gaussianity test based on skewness and kurtosis
        let mean = samples.mean().unwrap_or(0.0);
        let std_dev = samples.var(1.0).sqrt();
        let skewness = self.compute_skewness(samples, mean, std_dev);
        let kurtosis = self.compute_kurtosis(samples, mean, std_dev);
        
        // D'Agostino-Pearson test approximation
        let skew_component = skewness * skewness / 6.0;
        let kurt_component = kurtosis * kurtosis / 24.0;
        let test_statistic = skew_component + kurt_component;
        
        // Convert to 0-1 scale (lower values = more Gaussian)
        (-test_statistic).exp()
    }
    
    /// Compute bimodality coefficient
    fn compute_bimodality(&self, histogram_counts: &[u32]) -> f32 {
        if histogram_counts.len() < 3 {
            return 0.0;
        }
        
        let mut peaks = 0;
        for i in 1..histogram_counts.len()-1 {
            if histogram_counts[i] > histogram_counts[i-1] && histogram_counts[i] > histogram_counts[i+1] {
                peaks += 1;
            }
        }
        
        // Normalize by potential number of peaks
        peaks as f32 / (histogram_counts.len() / 2) as f32
    }
    
    /// Compute outlier ratio
    fn compute_outlier_ratio(&self, samples: &Array1<f32>) -> f32 {
        let q1 = self.compute_percentiles(samples, &[25.0])[0];
        let q3 = self.compute_percentiles(samples, &[75.0])[0];
        let iqr = q3 - q1;
        
        if iqr <= 1e-10 {
            return 0.0;
        }
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        let outliers = samples.iter()
            .filter(|&&x| x < lower_bound || x > upper_bound)
            .count();
        
        outliers as f32 / samples.len() as f32
    }
    
    /// Compute symmetry measure
    fn compute_symmetry(&self, samples: &Array1<f32>) -> f32 {
        let mean = samples.mean().unwrap_or(0.0);
        let median = self.compute_median(samples);
        let std_dev = samples.var(1.0).sqrt();
        
        if std_dev <= 1e-10 {
            return 1.0; // Perfect symmetry for constant signal
        }
        
        // Symmetry based on mean-median difference
        let asymmetry = ((mean - median) / std_dev).abs();
        
        // Convert to symmetry measure (1 = perfect symmetry, 0 = highly asymmetric)
        (-asymmetry).exp()
    }
    
    /// Compute information measures
    fn compute_information_measures(&self, samples: &Array1<f32>) -> Result<InformationMeasures> {
        let shannon_entropy = self.compute_shannon_entropy(samples);
        let differential_entropy = self.compute_differential_entropy(samples);
        let kolmogorov_complexity = self.estimate_kolmogorov_complexity(samples);
        let approximate_entropy = self.compute_approximate_entropy(samples, 2, 0.1)?;
        let sample_entropy = self.compute_sample_entropy(samples, 2, 0.1)?;
        let permutation_entropy = self.compute_permutation_entropy(samples, 3)?;
        let spectral_entropy = self.compute_spectral_entropy(samples)?;
        
        Ok(InformationMeasures {
            shannon_entropy,
            differential_entropy,
            kolmogorov_complexity,
            approximate_entropy,
            sample_entropy,
            permutation_entropy,
            spectral_entropy,
        })
    }
    
    /// Compute Shannon entropy from quantized samples
    fn compute_shannon_entropy(&self, samples: &Array1<f32>) -> f32 {
        let bins = 256;
        let (_, hist_counts) = self.compute_histogram(samples, bins);
        let total_samples = samples.len() as f32;
        
        let mut entropy = 0.0;
        for &count in hist_counts.iter() {
            if count > 0 {
                let prob = count as f32 / total_samples;
                entropy -= prob * prob.log2();
            }
        }
        
        entropy
    }
    
    /// Compute differential entropy (continuous version)
    fn compute_differential_entropy(&self, samples: &Array1<f32>) -> f32 {
        // Approximation using histogram with fine bins
        let bins = 1000;
        let (_, hist_counts) = self.compute_histogram(samples, bins);
        let total_samples = samples.len() as f32;
        let bin_width = if samples.len() > 1 {
            let min = samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            (max - min) / bins as f32
        } else {
            1.0
        };
        
        let mut entropy = 0.0;
        for &count in hist_counts.iter() {
            if count > 0 {
                let density = count as f32 / (total_samples * bin_width);
                if density > 1e-10 {
                    entropy -= density * bin_width * density.ln();
                }
            }
        }
        
        entropy
    }
    
    /// Estimate Kolmogorov complexity using compression
    fn estimate_kolmogorov_complexity(&self, samples: &Array1<f32>) -> f32 {
        // Convert to bytes and estimate compressibility
        let bytes: Vec<u8> = samples.iter()
            .map(|&x| ((x + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
            .collect();
        
        // Simple compression estimate using run-length encoding
        let mut compressed_size = 0;
        let mut i = 0;
        
        while i < bytes.len() {
            let current = bytes[i];
            let mut run_length = 1;
            
            while i + run_length < bytes.len() && bytes[i + run_length] == current {
                run_length += 1;
            }
            
            compressed_size += if run_length > 1 { 2 } else { 1 }; // (value, count) or value
            i += run_length;
        }
        
        compressed_size as f32 / bytes.len() as f32
    }
    
    /// Compute approximate entropy
    fn compute_approximate_entropy(&self, samples: &Array1<f32>, m: usize, r: f32) -> Result<f32> {
        if samples.len() < m + 1 {
            return Ok(0.0);
        }
        
        let phi_m = self.compute_phi(samples, m, r);
        let phi_m1 = self.compute_phi(samples, m + 1, r);
        
        Ok(phi_m - phi_m1)
    }
    
    /// Helper function for approximate entropy
    fn compute_phi(&self, samples: &Array1<f32>, m: usize, r: f32) -> f32 {
        let n = samples.len();
        if n < m {
            return 0.0;
        }
        
        let mut phi = 0.0;
        
        for i in 0..=n-m {
            let mut matches = 0;
            
            for j in 0..=n-m {
                let mut is_match = true;
                for k in 0..m {
                    if (samples[i + k] - samples[j + k]).abs() > r {
                        is_match = false;
                        break;
                    }
                }
                if is_match {
                    matches += 1;
                }
            }
            
            if matches > 0 {
                phi += (matches as f32 / (n - m + 1) as f32).ln();
            }
        }
        
        phi / (n - m + 1) as f32
    }
    
    /// Compute sample entropy
    fn compute_sample_entropy(&self, samples: &Array1<f32>, m: usize, r: f32) -> Result<f32> {
        if samples.len() < m + 1 {
            return Ok(0.0);
        }
        
        let mut a = 0.0; // matches of length m
        let mut b = 0.0; // matches of length m+1
        
        let n = samples.len();
        
        for i in 0..n-m {
            for j in i+1..n-m {
                // Check match of length m
                let mut match_m = true;
                for k in 0..m {
                    if (samples[i + k] - samples[j + k]).abs() > r {
                        match_m = false;
                        break;
                    }
                }
                
                if match_m {
                    a += 1.0;
                    
                    // Check match of length m+1
                    if i < n-m-1 && j < n-m-1 {
                        if (samples[i + m] - samples[j + m]).abs() <= r {
                            b += 1.0;
                        }
                    }
                }
            }
        }
        
        if a > 0.0 && b > 0.0 {
            Ok(-((b / a).ln()))
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute permutation entropy
    fn compute_permutation_entropy(&self, samples: &Array1<f32>, order: usize) -> Result<f32> {
        if samples.len() < order {
            return Ok(0.0);
        }
        
        let mut permutation_counts: HashMap<Vec<usize>, u32> = HashMap::new();
        let mut total_permutations = 0;
        
        for i in 0..=samples.len()-order {
            let window: Vec<f32> = samples.slice(s![i..i+order]).to_vec();
            let permutation = self.get_ordinal_pattern(&window);
            
            *permutation_counts.entry(permutation).or_insert(0) += 1;
            total_permutations += 1;
        }
        
        let mut entropy = 0.0;
        for &count in permutation_counts.values() {
            let prob = count as f32 / total_permutations as f32;
            entropy -= prob * prob.log2();
        }
        
        Ok(entropy)
    }
    
    /// Get ordinal pattern for permutation entropy
    fn get_ordinal_pattern(&self, window: &[f32]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..window.len()).collect();
        indices.sort_by(|&a, &b| window[a].partial_cmp(&window[b]).unwrap());
        indices
    }
    
    /// Compute spectral entropy
    fn compute_spectral_entropy(&self, samples: &Array1<f32>) -> Result<f32> {
        use realfft::RealFftPlanner;
        
        let fft_size = samples.len().min(1024);
        let mut planner = RealFftPlanner::new();
        let mut fft = planner.plan_fft_forward(fft_size);
        
        let mut input = vec![0.0; fft_size];
        for (i, &sample) in samples.iter().take(fft_size).enumerate() {
            input[i] = sample;
        }
        
        let mut output = fft.make_output_vec();
        fft.process(&mut input, &mut output)
            .map_err(|_| VocoderError::ProcessingError("FFT failed".to_string()))?;
        
        // Convert to power spectrum
        let power_spectrum: Vec<f32> = output.iter()
            .map(|c| c.norm_sqr())
            .collect();
        
        let total_power: f32 = power_spectrum.iter().sum();
        if total_power <= 1e-10 {
            return Ok(0.0);
        }
        
        let mut entropy = 0.0;
        for &power in power_spectrum.iter() {
            if power > 1e-10 {
                let prob = power / total_power;
                entropy -= prob * prob.log2();
            }
        }
        
        Ok(entropy)
    }
    
    /// Compute temporal statistics
    fn compute_temporal_statistics(&self, samples: &Array1<f32>) -> Result<TemporalStatistics> {
        let zero_crossing_rate = self.compute_zero_crossing_rate(samples);
        let mean_crossing_rate = self.compute_mean_crossing_rate(samples);
        let autocorrelation = self.compute_autocorrelation(samples, 100);
        let (autocorr_peak_lag, periodicity_strength) = self.find_autocorr_peak(&autocorrelation);
        let stationarity = self.compute_stationarity(samples);
        let trend_strength = self.compute_trend_strength(samples);
        let seasonality_strength = self.compute_seasonality_strength(samples);
        
        Ok(TemporalStatistics {
            zero_crossing_rate,
            mean_crossing_rate,
            autocorrelation,
            autocorr_peak_lag,
            periodicity_strength,
            stationarity,
            trend_strength,
            seasonality_strength,
        })
    }
    
    /// Compute zero crossing rate
    fn compute_zero_crossing_rate(&self, samples: &Array1<f32>) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        
        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i-1] >= 0.0) {
                crossings += 1;
            }
        }
        
        crossings as f32 / samples.len() as f32
    }
    
    /// Compute mean crossing rate
    fn compute_mean_crossing_rate(&self, samples: &Array1<f32>) -> f32 {
        let mean = samples.mean().unwrap_or(0.0);
        
        if samples.len() < 2 {
            return 0.0;
        }
        
        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= mean) != (samples[i-1] >= mean) {
                crossings += 1;
            }
        }
        
        crossings as f32 / samples.len() as f32
    }
    
    /// Compute autocorrelation function
    fn compute_autocorrelation(&self, samples: &Array1<f32>, max_lag: usize) -> Vec<f32> {
        let n = samples.len();
        let max_lag = max_lag.min(n - 1);
        let mut autocorr = Vec::with_capacity(max_lag + 1);
        
        let mean = samples.mean().unwrap_or(0.0);
        
        for lag in 0..=max_lag {
            let mut sum = 0.0;
            let mut count = 0;
            
            for i in 0..n-lag {
                sum += (samples[i] - mean) * (samples[i + lag] - mean);
                count += 1;
            }
            
            let correlation = if count > 0 { sum / count as f32 } else { 0.0 };
            autocorr.push(correlation);
        }
        
        // Normalize by variance
        if let Some(variance) = autocorr.get(0) {
            if *variance > 1e-10 {
                for corr in autocorr.iter_mut() {
                    *corr /= variance;
                }
            }
        }
        
        autocorr
    }
    
    /// Find peak in autocorrelation function
    fn find_autocorr_peak(&self, autocorr: &[f32]) -> (usize, f32) {
        if autocorr.len() < 2 {
            return (0, 0.0);
        }
        
        // Skip lag 0 and find maximum
        let mut max_val = 0.0;
        let mut max_lag = 0;
        
        for (lag, &val) in autocorr.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_lag = lag;
            }
        }
        
        (max_lag, max_val)
    }
    
    /// Compute stationarity measure
    fn compute_stationarity(&self, samples: &Array1<f32>) -> f32 {
        if samples.len() < 10 {
            return 1.0;
        }
        
        // Divide signal into segments and compare statistics
        let segment_size = samples.len() / 5;
        let mut segment_means = Vec::new();
        let mut segment_vars = Vec::new();
        
        for i in 0..5 {
            let start = i * segment_size;
            let end = if i == 4 { samples.len() } else { (i + 1) * segment_size };
            
            if end > start {
                let segment = samples.slice(s![start..end]);
                segment_means.push(segment.mean().unwrap_or(0.0));
                segment_vars.push(segment.var(1.0));
            }
        }
        
        if segment_means.len() < 2 {
            return 1.0;
        }
        
        // Compute variance of segment means and variances
        let mean_of_means: f32 = segment_means.iter().sum::<f32>() / segment_means.len() as f32;
        let mean_of_vars: f32 = segment_vars.iter().sum::<f32>() / segment_vars.len() as f32;
        
        let var_of_means: f32 = segment_means.iter()
            .map(|&x| (x - mean_of_means).powi(2))
            .sum::<f32>() / segment_means.len() as f32;
        
        let var_of_vars: f32 = segment_vars.iter()
            .map(|&x| (x - mean_of_vars).powi(2))
            .sum::<f32>() / segment_vars.len() as f32;
        
        // Stationarity: lower variation in statistics = higher stationarity
        let mean_variation = if mean_of_means.abs() > 1e-10 { 
            var_of_means.sqrt() / mean_of_means.abs() 
        } else { 
            0.0 
        };
        
        let var_variation = if mean_of_vars > 1e-10 { 
            var_of_vars.sqrt() / mean_of_vars 
        } else { 
            0.0 
        };
        
        let total_variation = mean_variation + var_variation;
        (-total_variation).exp()
    }
    
    /// Compute trend strength
    fn compute_trend_strength(&self, samples: &Array1<f32>) -> f32 {
        if samples.len() < 3 {
            return 0.0;
        }
        
        // Simple linear trend using least squares
        let n = samples.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = samples.mean().unwrap_or(0.0);
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (i, &y) in samples.iter().enumerate() {
            let x = i as f32;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }
        
        if denominator > 1e-10 {
            let slope = numerator / denominator;
            slope.abs()
        } else {
            0.0
        }
    }
    
    /// Compute seasonality strength (simplified)
    fn compute_seasonality_strength(&self, samples: &Array1<f32>) -> f32 {
        // Use autocorrelation to detect periodic patterns
        let autocorr = self.compute_autocorrelation(samples, samples.len() / 4);
        let (_, periodicity_strength) = self.find_autocorr_peak(&autocorr);
        periodicity_strength
    }
    
    /// Compute complexity measures
    fn compute_complexity_measures(&self, samples: &Array1<f32>) -> Result<ComplexityMeasures> {
        let fractal_dimension = self.compute_fractal_dimension(samples);
        let hurst_exponent = self.compute_hurst_exponent(samples);
        let lyapunov_exponent = self.estimate_lyapunov_exponent(samples);
        let correlation_dimension = self.estimate_correlation_dimension(samples);
        let lempel_ziv_complexity = self.compute_lempel_ziv_complexity(samples);
        let multiscale_entropy = self.compute_multiscale_entropy(samples)?;
        
        Ok(ComplexityMeasures {
            fractal_dimension,
            hurst_exponent,
            lyapunov_exponent,
            correlation_dimension,
            lempel_ziv_complexity,
            multiscale_entropy,
        })
    }
    
    /// Compute fractal dimension using box counting
    fn compute_fractal_dimension(&self, samples: &Array1<f32>) -> f32 {
        // Simplified fractal dimension estimate
        if samples.len() < 4 {
            return 1.0;
        }
        
        // Compute curve length at different scales
        let mut scales = Vec::new();
        let mut lengths = Vec::new();
        
        for scale in [1, 2, 4, 8] {
            if scale < samples.len() {
                let mut length = 0.0;
                for i in scale..samples.len() {
                    length += (samples[i] - samples[i - scale]).abs();
                }
                scales.push((scale as f32).ln());
                lengths.push(length.ln());
            }
        }
        
        if scales.len() < 2 {
            return 1.0;
        }
        
        // Linear regression to find slope
        let n = scales.len() as f32;
        let x_mean: f32 = scales.iter().sum::<f32>() / n;
        let y_mean: f32 = lengths.iter().sum::<f32>() / n;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (x, y) in scales.iter().zip(lengths.iter()) {
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }
        
        if denominator > 1e-10 {
            1.0 - (numerator / denominator) // Fractal dimension
        } else {
            1.0
        }
    }
    
    /// Compute Hurst exponent using R/S analysis
    fn compute_hurst_exponent(&self, samples: &Array1<f32>) -> f32 {
        if samples.len() < 8 {
            return 0.5;
        }
        
        // Simplified R/S analysis
        let mean = samples.mean().unwrap_or(0.0);
        let mut cumulative_deviations = Vec::with_capacity(samples.len());
        let mut sum = 0.0;
        
        for &sample in samples.iter() {
            sum += sample - mean;
            cumulative_deviations.push(sum);
        }
        
        let range = cumulative_deviations.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) -
                   cumulative_deviations.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        
        let std_dev = samples.var(1.0).sqrt();
        
        if std_dev > 1e-10 && range > 1e-10 {
            let rs_ratio = range / std_dev;
            let n = samples.len() as f32;
            rs_ratio.ln() / n.ln()
        } else {
            0.5
        }
    }
    
    /// Estimate Lyapunov exponent (simplified)
    fn estimate_lyapunov_exponent(&self, samples: &Array1<f32>) -> f32 {
        // Very simplified estimation - not a true Lyapunov exponent calculation
        if samples.len() < 10 {
            return 0.0;
        }
        
        let mut divergence_sum = 0.0;
        let mut count = 0;
        
        for i in 1..samples.len().min(100) {
            if i < samples.len() {
                let divergence = (samples[i] - samples[i-1]).abs();
                if divergence > 1e-10 {
                    divergence_sum += divergence.ln();
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            divergence_sum / count as f32
        } else {
            0.0
        }
    }
    
    /// Estimate correlation dimension (simplified)
    fn estimate_correlation_dimension(&self, samples: &Array1<f32>) -> f32 {
        // Simplified correlation dimension estimate
        if samples.len() < 10 {
            return 1.0;
        }
        
        let embedding_dim = 3;
        let mut correlation_sum = 0;
        let mut total_pairs = 0;
        
        let eps = samples.var(1.0).sqrt() * 0.1; // Threshold distance
        
        for i in 0..samples.len().saturating_sub(embedding_dim) {
            for j in i+1..samples.len().saturating_sub(embedding_dim) {
                let mut distance = 0.0;
                for k in 0..embedding_dim {
                    distance += (samples[i + k] - samples[j + k]).powi(2);
                }
                distance = distance.sqrt();
                
                if distance < eps {
                    correlation_sum += 1;
                }
                total_pairs += 1;
                
                if total_pairs > 1000 { // Limit computation
                    break;
                }
            }
        }
        
        if correlation_sum > 0 && total_pairs > 0 {
            let correlation_integral = correlation_sum as f32 / total_pairs as f32;
            if correlation_integral > 1e-10 && eps > 1e-10 {
                correlation_integral.ln() / eps.ln()
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
    
    /// Compute Lempel-Ziv complexity
    fn compute_lempel_ziv_complexity(&self, samples: &Array1<f32>) -> f32 {
        // Convert to binary sequence for LZ complexity
        let median = self.compute_median(samples);
        let binary_seq: Vec<u8> = samples.iter()
            .map(|&x| if x >= median { 1 } else { 0 })
            .collect();
        
        let mut complexity = 0;
        let mut i = 0;
        
        while i < binary_seq.len() {
            let mut j = 1;
            
            // Find longest match in previous subsequences
            while i + j <= binary_seq.len() {
                let current_substr = &binary_seq[i..i+j];
                let mut found = false;
                
                for start in 0..i {
                    if start + j <= i {
                        let prev_substr = &binary_seq[start..start+j];
                        if current_substr == prev_substr {
                            found = true;
                            break;
                        }
                    }
                }
                
                if !found {
                    break;
                }
                j += 1;
            }
            
            complexity += 1;
            i += j.max(1);
        }
        
        complexity as f32 / binary_seq.len() as f32
    }
    
    /// Compute multiscale entropy
    fn compute_multiscale_entropy(&self, samples: &Array1<f32>) -> Result<Vec<f32>> {
        let max_scale = 5;
        let mut mse = Vec::with_capacity(max_scale);
        
        for scale in 1..=max_scale {
            let coarse_grained = self.coarse_grain(samples, scale);
            let entropy = self.compute_sample_entropy(&coarse_grained, 2, 0.15)?;
            mse.push(entropy);
        }
        
        Ok(mse)
    }
    
    /// Coarse grain signal for multiscale entropy
    fn coarse_grain(&self, samples: &Array1<f32>, scale: usize) -> Array1<f32> {
        let n = samples.len() / scale;
        let mut coarse_grained = Array1::zeros(n);
        
        for i in 0..n {
            let start = i * scale;
            let end = ((i + 1) * scale).min(samples.len());
            let sum: f32 = samples.slice(s![start..end]).sum();
            coarse_grained[i] = sum / (end - start) as f32;
        }
        
        coarse_grained
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_statistical_analyzer_creation() {
        let analyzer = StatisticalAnalyzer::new(44100);
        assert_eq!(analyzer.sample_rate, 44100);
    }
    
    #[test]
    fn test_basic_statistics() {
        let analyzer = StatisticalAnalyzer::new(44100);
        let samples = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let stats = analyzer.compute_basic_statistics(&samples);
        
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.median - 3.0).abs() < 1e-6);
        assert!(stats.std_dev > 0.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.range, 4.0);
    }
    
    #[test]
    fn test_histogram_computation() {
        let analyzer = StatisticalAnalyzer::new(44100);
        let samples = Array1::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0]);
        
        let (bins, counts) = analyzer.compute_histogram(&samples, 3);
        
        assert_eq!(bins.len(), 3);
        assert_eq!(counts.len(), 3);
        assert_eq!(counts.iter().sum::<u32>(), 5);
    }
    
    #[test]
    fn test_zero_crossing_rate() {
        let analyzer = StatisticalAnalyzer::new(44100);
        
        // Alternating signal
        let samples = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0]);
        let zcr = analyzer.compute_zero_crossing_rate(&samples);
        
        assert!(zcr > 0.5); // High ZCR for alternating signal
    }
    
    #[test]
    fn test_shannon_entropy() {
        let analyzer = StatisticalAnalyzer::new(44100);
        
        // Uniform distribution should have high entropy
        let uniform_samples = Array1::from_vec((0..100).map(|i| i as f32).collect());
        let entropy_uniform = analyzer.compute_shannon_entropy(&uniform_samples);
        
        // Constant signal should have low entropy
        let constant_samples = Array1::from_vec(vec![1.0; 100]);
        let entropy_constant = analyzer.compute_shannon_entropy(&constant_samples);
        
        assert!(entropy_uniform > entropy_constant);
    }
    
    #[test]
    fn test_autocorrelation() {
        let analyzer = StatisticalAnalyzer::new(44100);
        
        // Periodic signal should show strong autocorrelation
        let samples: Array1<f32> = Array1::from_vec(
            (0..100).map(|i| (2.0 * std::f32::consts::PI * i as f32 / 10.0).sin()).collect()
        );
        
        let autocorr = analyzer.compute_autocorrelation(&samples, 20);
        
        assert_eq!(autocorr[0], 1.0); // Perfect self-correlation at lag 0
        assert!(autocorr.len() == 21); // 0 to 20 lags
    }
    
    #[test]
    fn test_percentiles() {
        let analyzer = StatisticalAnalyzer::new(44100);
        let samples = Array1::from_vec((1..=100).map(|i| i as f32).collect());
        
        let percentiles = analyzer.compute_percentiles(&samples, &[25.0, 50.0, 75.0]);
        
        assert!((percentiles[0] - 25.0).abs() < 5.0); // 25th percentile
        assert!((percentiles[1] - 50.0).abs() < 5.0); // Median
        assert!((percentiles[2] - 75.0).abs() < 5.0); // 75th percentile
    }
}