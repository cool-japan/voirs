//! Custom Metric Implementation Example
//!
//! This example demonstrates how to implement custom evaluation metrics
//! for the VoiRS evaluation framework, showing how to extend the existing
//! evaluation system with specialized audio analysis tools.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use voirs_evaluation::prelude::*;
use voirs_evaluation::{
    traits::{EvaluationResult, QualityScore},
    EvaluationError,
};
use voirs_sdk::AudioBuffer;

/// Custom spectral centroid metric for evaluating spectral brightness
#[derive(Debug, Clone)]
pub struct SpectralCentroidMetric {
    /// Window size for FFT analysis
    pub window_size: usize,
    /// Hop size for overlapping windows
    pub hop_size: usize,
    /// Whether to apply windowing function
    pub use_windowing: bool,
}

impl Default for SpectralCentroidMetric {
    fn default() -> Self {
        Self {
            window_size: 1024,
            hop_size: 512,
            use_windowing: true,
        }
    }
}

impl SpectralCentroidMetric {
    /// Calculate spectral centroid analysis for audio
    pub async fn analyze(&self, audio: &AudioBuffer) -> Result<QualityScore, EvaluationError> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate() as f32;

        // Calculate spectral centroid for each frame
        let mut centroids = Vec::new();
        let mut i = 0;

        while i + self.window_size <= samples.len() {
            let frame = &samples[i..i + self.window_size];
            let centroid = self.calculate_spectral_centroid(frame, sample_rate)?;
            centroids.push(centroid);
            i += self.hop_size;
        }

        if centroids.is_empty() {
            return Err(EvaluationError::MetricCalculationError {
                metric: "spectral_centroid".to_string(),
                message: "No frames to analyze".to_string(),
                source: None,
            });
        }

        // Calculate statistics
        let mean_centroid = centroids.iter().sum::<f32>() / centroids.len() as f32;
        let variance = centroids
            .iter()
            .map(|c| (c - mean_centroid).powi(2))
            .sum::<f32>()
            / centroids.len() as f32;
        let std_dev = variance.sqrt();

        // Create quality score based on spectral centroid
        // Higher centroid generally indicates brighter, more intelligible speech
        let normalized_centroid = (mean_centroid / (sample_rate / 2.0)).clamp(0.0, 1.0);
        let quality_score = if normalized_centroid > 0.1 && normalized_centroid < 0.6 {
            1.0 - (normalized_centroid - 0.35).abs() / 0.25
        } else {
            0.5
        };

        let mut component_scores = HashMap::new();
        component_scores.insert("mean_centroid_hz".to_string(), mean_centroid);
        component_scores.insert("centroid_std_dev".to_string(), std_dev);
        component_scores.insert("normalized_centroid".to_string(), normalized_centroid);
        component_scores.insert("num_frames".to_string(), centroids.len() as f32);

        let recommendations = if normalized_centroid < 0.1 {
            vec!["Audio appears too dark, consider enhancing high frequencies".to_string()]
        } else if normalized_centroid > 0.6 {
            vec!["Audio appears too bright, consider reducing high frequencies".to_string()]
        } else {
            vec!["Spectral balance appears good".to_string()]
        };

        Ok(QualityScore {
            overall_score: quality_score,
            component_scores,
            recommendations,
            confidence: if centroids.len() > 10 { 0.8 } else { 0.6 },
            processing_time: None,
        })
    }
}

impl SpectralCentroidMetric {
    /// Calculate spectral centroid for a single frame
    fn calculate_spectral_centroid(
        &self,
        frame: &[f32],
        sample_rate: f32,
    ) -> Result<f32, EvaluationError> {
        // Apply windowing if enabled
        let windowed: Vec<f32> = if self.use_windowing {
            frame
                .iter()
                .enumerate()
                .map(|(i, &sample)| {
                    let window_val = 0.5
                        - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / frame.len() as f32).cos();
                    sample * window_val
                })
                .collect()
        } else {
            frame.to_vec()
        };

        // Simple FFT-like magnitude spectrum calculation
        let magnitude_spectrum = self.calculate_magnitude_spectrum(&windowed)?;

        // Calculate spectral centroid
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &magnitude) in magnitude_spectrum.iter().enumerate() {
            let frequency = i as f32 * sample_rate / frame.len() as f32;
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum)
        } else {
            Err(EvaluationError::MetricCalculationError {
                metric: "spectral_centroid".to_string(),
                message: "Zero magnitude spectrum".to_string(),
                source: None,
            })
        }
    }

    /// Calculate magnitude spectrum (simplified implementation)
    fn calculate_magnitude_spectrum(&self, samples: &[f32]) -> Result<Vec<f32>, EvaluationError> {
        // Simplified magnitude spectrum calculation
        // In a real implementation, you would use a proper FFT library like rustfft
        let n = samples.len();
        let mut magnitude_spectrum = vec![0.0; n / 2];

        for k in 0..n / 2 {
            let mut real = 0.0;
            let mut imag = 0.0;

            for i in 0..n {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                real += samples[i] * angle.cos();
                imag += samples[i] * angle.sin();
            }

            magnitude_spectrum[k] = (real * real + imag * imag).sqrt();
        }

        Ok(magnitude_spectrum)
    }
}

/// Custom zero crossing rate metric for temporal analysis
#[derive(Debug, Clone)]
pub struct ZeroCrossingRateMetric {
    /// Frame size for analysis
    pub frame_size: usize,
    /// Hop size between frames
    pub hop_size: usize,
}

impl Default for ZeroCrossingRateMetric {
    fn default() -> Self {
        Self {
            frame_size: 1024,
            hop_size: 512,
        }
    }
}

impl ZeroCrossingRateMetric {
    /// Calculate zero crossing rate analysis for audio
    pub async fn analyze(&self, audio: &AudioBuffer) -> Result<QualityScore, EvaluationError> {
        let samples = audio.samples();

        // Calculate ZCR for each frame
        let mut zcr_values = Vec::new();
        let mut i = 0;

        while i + self.frame_size <= samples.len() {
            let frame = &samples[i..i + self.frame_size];
            let zcr = self.calculate_frame_zcr(frame);
            zcr_values.push(zcr);
            i += self.hop_size;
        }

        if zcr_values.is_empty() {
            return Err(EvaluationError::MetricCalculationError {
                metric: "zero_crossing_rate".to_string(),
                message: "No frames to analyze".to_string(),
                source: None,
            });
        }

        // Calculate statistics
        let mean_zcr = zcr_values.iter().sum::<f32>() / zcr_values.len() as f32;
        let variance = zcr_values
            .iter()
            .map(|z| (z - mean_zcr).powi(2))
            .sum::<f32>()
            / zcr_values.len() as f32;
        let std_dev = variance.sqrt();

        // Quality assessment based on ZCR
        // Moderate ZCR indicates good speech quality
        let quality_score = if mean_zcr > 0.02 && mean_zcr < 0.3 {
            1.0 - (mean_zcr - 0.1).abs() / 0.2
        } else {
            0.3
        };

        let mut component_scores = HashMap::new();
        component_scores.insert("mean_zcr".to_string(), mean_zcr);
        component_scores.insert("zcr_std_dev".to_string(), std_dev);
        component_scores.insert("zcr_variance".to_string(), variance);
        component_scores.insert(
            "min_zcr".to_string(),
            zcr_values.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        );
        component_scores.insert(
            "max_zcr".to_string(),
            zcr_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        );

        let recommendations = if mean_zcr < 0.02 {
            vec!["Very low zero crossing rate, audio may lack high frequency content".to_string()]
        } else if mean_zcr > 0.3 {
            vec!["Very high zero crossing rate, audio may be noisy or have excessive high frequencies".to_string()]
        } else {
            vec!["Zero crossing rate appears normal".to_string()]
        };

        Ok(QualityScore {
            overall_score: quality_score,
            component_scores,
            recommendations,
            confidence: 0.7,
            processing_time: None,
        })
    }
}

impl ZeroCrossingRateMetric {
    /// Calculate zero crossing rate for a single frame
    fn calculate_frame_zcr(&self, frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..frame.len() {
            if (frame[i] >= 0.0) != (frame[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (frame.len() - 1) as f32
    }
}

/// Custom composite metric combining multiple simple metrics
#[derive(Debug, Clone)]
pub struct CompositeQualityMetric {
    /// Individual metrics to combine
    pub spectral_centroid: SpectralCentroidMetric,
    pub zero_crossing_rate: ZeroCrossingRateMetric,
    /// Weights for combining metrics
    pub weights: HashMap<String, f32>,
}

impl Default for CompositeQualityMetric {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("spectral_centroid".to_string(), 0.6);
        weights.insert("zero_crossing_rate".to_string(), 0.4);

        Self {
            spectral_centroid: SpectralCentroidMetric::default(),
            zero_crossing_rate: ZeroCrossingRateMetric::default(),
            weights,
        }
    }
}

impl CompositeQualityMetric {
    /// Analyze audio using composite metrics
    pub async fn analyze(&self, audio: &AudioBuffer) -> Result<QualityScore, EvaluationError> {
        // Calculate individual metrics
        let spectral_result = self.spectral_centroid.analyze(audio).await?;
        let zcr_result = self.zero_crossing_rate.analyze(audio).await?;

        // Combine scores using weights
        let spectral_weight = self.weights.get("spectral_centroid").unwrap_or(&0.5);
        let zcr_weight = self.weights.get("zero_crossing_rate").unwrap_or(&0.5);
        let total_weight = spectral_weight + zcr_weight;

        let overall_score = if total_weight > 0.0 {
            (spectral_result.overall_score * spectral_weight
                + zcr_result.overall_score * zcr_weight)
                / total_weight
        } else {
            0.5
        };

        // Combine component scores
        let mut component_scores = HashMap::new();

        // Add prefixed individual scores
        for (key, value) in &spectral_result.component_scores {
            component_scores.insert(format!("spectral_{}", key), *value);
        }
        for (key, value) in &zcr_result.component_scores {
            component_scores.insert(format!("zcr_{}", key), *value);
        }

        // Add combined metrics
        component_scores.insert("spectral_score".to_string(), spectral_result.overall_score);
        component_scores.insert("zcr_score".to_string(), zcr_result.overall_score);
        component_scores.insert("spectral_weight".to_string(), *spectral_weight);
        component_scores.insert("zcr_weight".to_string(), *zcr_weight);

        // Calculate combined confidence
        let confidence = (spectral_result.confidence + zcr_result.confidence) / 2.0;

        // Combine recommendations
        let mut recommendations = spectral_result.recommendations.clone();
        recommendations.extend(zcr_result.recommendations);

        Ok(QualityScore {
            overall_score,
            component_scores,
            recommendations,
            confidence,
            processing_time: None,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 VoiRS Custom Metric Implementation Example");
    println!("===========================================");

    // Create sample audio buffer
    println!("\n🎧 Creating test audio buffer...");
    let sample_rate = 16000;
    let duration_samples = 2 * sample_rate; // 2 seconds

    // Generate test audio with varying characteristics
    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Combination of multiple frequencies for interesting spectral content
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
        })
        .collect();

    let audio = AudioBuffer::mono(samples, sample_rate as u32);

    // Test individual custom metrics
    println!("\n📊 Testing Spectral Centroid Metric:");
    println!("------------------------------------");

    let spectral_metric = SpectralCentroidMetric::default();
    let spectral_result = spectral_metric.analyze(&audio).await?;

    println!("Metric: Spectral Centroid");
    println!("Overall Score: {:.3}", spectral_result.overall_score);
    println!("Confidence: {:.3}", spectral_result.confidence);
    println!("Component Scores:");
    for (key, value) in &spectral_result.component_scores {
        println!("  {}: {:.3}", key, value);
    }

    println!("\n📈 Testing Zero Crossing Rate Metric:");
    println!("-------------------------------------");

    let zcr_metric = ZeroCrossingRateMetric::default();
    let zcr_result = zcr_metric.analyze(&audio).await?;

    println!("Metric: Zero Crossing Rate");
    println!("Overall Score: {:.3}", zcr_result.overall_score);
    println!("Confidence: {:.3}", zcr_result.confidence);
    println!("Component Scores:");
    for (key, value) in &zcr_result.component_scores {
        println!("  {}: {:.3}", key, value);
    }

    println!("\n🔄 Testing Composite Quality Metric:");
    println!("------------------------------------");

    let composite_metric = CompositeQualityMetric::default();
    let composite_result = composite_metric.analyze(&audio).await?;

    println!("Metric: Composite Quality");
    println!("Overall Score: {:.3}", composite_result.overall_score);
    println!("Confidence: {:.3}", composite_result.confidence);
    println!("Component Scores:");
    for (key, value) in &composite_result.component_scores {
        println!("  {}: {:.3}", key, value);
    }

    // Demonstrate metric customization
    println!("\n⚙️  Custom Metric Configuration:");
    println!("--------------------------------");

    let mut custom_spectral = SpectralCentroidMetric::default();
    custom_spectral.window_size = 2048; // Larger window for better frequency resolution
    custom_spectral.hop_size = 1024; // Less overlap
    custom_spectral.use_windowing = true;

    let custom_result = custom_spectral.analyze(&audio).await?;
    println!("Custom Spectral Centroid (2048 window):");
    println!("  Overall Score: {:.3}", custom_result.overall_score);
    println!(
        "  Mean Centroid: {:.1} Hz",
        custom_result
            .component_scores
            .get("mean_centroid_hz")
            .unwrap_or(&0.0)
    );

    // Demonstrate custom weight configuration for composite metric
    println!("\n⚖️  Custom Weight Configuration:");
    println!("-------------------------------");

    let mut custom_composite = CompositeQualityMetric::default();
    custom_composite
        .weights
        .insert("spectral_centroid".to_string(), 0.8);
    custom_composite
        .weights
        .insert("zero_crossing_rate".to_string(), 0.2);

    let weighted_result = custom_composite.analyze(&audio).await?;
    println!("Weighted Composite (80% spectral, 20% ZCR):");
    println!("  Overall Score: {:.3}", weighted_result.overall_score);
    println!(
        "  Spectral Weight: {:.1}",
        weighted_result
            .component_scores
            .get("spectral_weight")
            .unwrap_or(&0.0)
    );
    println!(
        "  ZCR Weight: {:.1}",
        weighted_result
            .component_scores
            .get("zcr_weight")
            .unwrap_or(&0.0)
    );

    // Implementation guidelines
    println!("\n💡 Custom Metric Implementation Guidelines:");
    println!("==========================================");

    println!("🔧 Key Implementation Steps:");
    println!("  1. Define your metric struct with configuration parameters");
    println!("  2. Implement an analyze() method that returns QualityScore");
    println!("  3. Handle error cases gracefully with descriptive messages");
    println!("  4. Return meaningful component scores and recommendations");
    println!("  5. Set appropriate confidence levels based on data quality");

    println!("\n📋 Best Practices:");
    println!("  • Use descriptive metric names and component score keys");
    println!("  • Include metadata for debugging and analysis");
    println!("  • Normalize scores to consistent ranges (e.g., 0-1)");
    println!("  • Handle edge cases (empty audio, invalid parameters)");
    println!("  • Consider computational efficiency for real-time use");

    println!("\n🎯 Metric Types:");
    println!("  • No-reference: Analyze single audio without comparison");
    println!("  • Full-reference: Compare with reference audio for accuracy");
    println!("  • Composite: Combine multiple metrics with configurable weights");

    println!("\n✅ Custom metric implementation example complete!");

    Ok(())
}
