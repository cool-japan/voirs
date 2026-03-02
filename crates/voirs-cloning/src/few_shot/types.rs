//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::embedding::SpeakerEmbedding;
use crate::types::{CloningMethod, SpeakerData, VoiceSample};
use crate::{Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

use super::functions::*;

/// Type alias for feature vectors with quality scores
pub type QualityFeaturePair = (Vec<f32>, SampleQuality);

/// Type alias for feature set (collection of feature-quality pairs)
pub type FeatureSet = Vec<QualityFeaturePair>;

/// Type alias for support and query set tuple
pub type SupportQuerySets = (FeatureSet, FeatureSet);

/// Configuration for few-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotConfig {
    /// Number of support samples (shots)
    pub num_shots: usize,
    /// Number of query samples for evaluation
    pub num_queries: usize,
    /// Feature dimension for speaker embeddings
    pub embedding_dim: usize,
    /// Hidden dimensions for meta-learner
    pub meta_hidden_dims: Vec<usize>,
    /// Learning rate for meta-learning
    pub meta_learning_rate: f32,
    /// Learning rate for adaptation
    pub adaptation_learning_rate: f32,
    /// Number of meta-learning episodes
    pub meta_episodes: usize,
    /// Number of adaptation steps during meta-learning
    pub adaptation_steps: usize,
    /// Temperature for prototypical networks
    pub prototype_temperature: f32,
    /// Quality threshold for sample inclusion
    pub quality_threshold: f32,
    /// Use quality-weighted averaging
    pub use_quality_weighting: bool,
    /// Enable cross-lingual learning
    pub enable_cross_lingual: bool,
    /// Distance metric for similarity
    pub distance_metric: DistanceMetric,
    /// Meta-learning algorithm
    pub meta_algorithm: MetaLearningAlgorithm,
}
/// Distance metrics for similarity computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine similarity
    Cosine,
    /// Manhattan distance
    Manhattan,
    /// Learned distance metric
    Learned,
}
/// Training episode for meta-learning
#[derive(Debug, Clone)]
pub(crate) struct TrainingEpisode {
    /// Episode ID
    id: String,
    /// Support samples
    support_samples: Vec<(VoiceSample, SampleQuality)>,
    /// Query samples
    query_samples: Vec<(VoiceSample, SampleQuality)>,
    /// Episode loss
    loss: f32,
    /// Adaptation accuracy
    accuracy: f32,
    /// Training time
    duration: Duration,
}
/// Few-shot learning result
#[derive(Debug, Clone)]
pub struct FewShotResult {
    /// Adapted speaker embedding
    pub speaker_embedding: Vec<f32>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Quality score of the adaptation
    pub quality_score: f32,
    /// Number of samples used
    pub samples_used: usize,
    /// Adaptation time
    pub adaptation_time: Duration,
    /// Meta-learning algorithm used
    pub algorithm: MetaLearningAlgorithm,
    /// Cross-lingual adaptation info
    pub cross_lingual_info: Option<CrossLingualInfo>,
}
/// Quality metrics for voice samples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleQuality {
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Spectral clarity score
    pub spectral_clarity: f32,
    /// Prosodic naturalness
    pub prosodic_naturalness: f32,
    /// Speaker consistency (if multiple samples)
    pub speaker_consistency: f32,
    /// Overall quality score
    pub overall_quality: f32,
}
impl SampleQuality {
    /// Compute quality from audio sample
    pub fn from_sample(sample: &VoiceSample) -> Self {
        let audio = sample.get_normalized_audio();
        let snr = Self::compute_snr(&audio);
        let spectral_clarity = Self::compute_spectral_clarity(&audio, sample.sample_rate);
        let prosodic_naturalness = Self::compute_prosodic_naturalness(&audio, sample.sample_rate);
        let speaker_consistency = 0.8;
        let overall_quality = (snr * 0.3
            + spectral_clarity * 0.3
            + prosodic_naturalness * 0.3
            + speaker_consistency * 0.1)
            .clamp(0.0, 1.0);
        Self {
            snr,
            spectral_clarity,
            prosodic_naturalness,
            speaker_consistency,
            overall_quality,
        }
    }
    fn compute_snr(audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let signal_energy: f32 = audio.iter().map(|x| x * x).sum();
        let mean_energy = signal_energy / audio.len() as f32;
        let mut sorted_energies: Vec<f32> = audio.iter().map(|x| x * x).collect();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_threshold = sorted_energies.len() / 10;
        let noise_energy =
            sorted_energies[..noise_threshold].iter().sum::<f32>() / noise_threshold as f32;
        if noise_energy > 0.0 {
            (mean_energy / noise_energy).log10() * 10.0
        } else {
            1.0
        }
    }
    fn compute_spectral_clarity(audio: &[f32], sample_rate: u32) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let nyquist = sample_rate as f32 / 2.0;
        let high_freq_energy = audio
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let freq = (i as f32 / audio.len() as f32) * nyquist;
                if freq > 1000.0 && freq < 8000.0 {
                    x * x
                } else {
                    0.0
                }
            })
            .sum::<f32>();
        let total_energy: f32 = audio.iter().map(|x| x * x).sum();
        if total_energy > 0.0 {
            (high_freq_energy / total_energy).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
    fn compute_prosodic_naturalness(audio: &[f32], sample_rate: u32) -> f32 {
        if audio.len() < sample_rate as usize {
            return 0.5;
        }
        let window_size = sample_rate as usize / 20;
        let mut energy_variations = Vec::new();
        for chunk in audio.chunks(window_size) {
            let energy: f32 = chunk.iter().map(|x| x * x).sum();
            energy_variations.push(energy / chunk.len() as f32);
        }
        if energy_variations.len() < 2 {
            return 0.5;
        }
        let mean_energy = energy_variations.iter().sum::<f32>() / energy_variations.len() as f32;
        let variance = energy_variations
            .iter()
            .map(|x| (x - mean_energy).powi(2))
            .sum::<f32>()
            / energy_variations.len() as f32;
        let std_dev = variance.sqrt();
        let cv = if mean_energy > 0.0 {
            std_dev / mean_energy
        } else {
            0.0
        };
        if cv > 0.3 && cv < 0.7 {
            1.0 - (cv - 0.5).abs() * 2.0
        } else {
            0.5
        }
    }
}
/// Meta-learning network
struct MetaNetwork {
    /// Embedding layers
    embedding_layers: Vec<Linear>,
    /// Meta-learning layers
    meta_layers: Vec<Linear>,
    /// Adaptation layers
    adaptation_layers: Vec<Linear>,
    /// Output layer
    output_layer: Linear,
}
/// Meta-learning algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// Prototypical Networks
    ProtoNet,
    /// Matching Networks
    MatchingNet,
    /// Relation Networks
    RelationNet,
    /// Meta-SGD
    MetaSGD,
}
/// Feature extractor for voice samples
pub(crate) struct FeatureExtractor {
    /// Configuration
    config: FewShotConfig,
    /// Device
    device: Device,
    /// Feature cache
    feature_cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}
impl FeatureExtractor {
    /// Create new feature extractor
    pub(crate) fn new(config: FewShotConfig, device: Device) -> Result<Self> {
        Ok(Self {
            config,
            device,
            feature_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    /// Extract features from voice sample
    pub(crate) async fn extract_features(&self, sample: &VoiceSample) -> Result<Vec<f32>> {
        {
            let cache = self.feature_cache.read().await;
            if let Some(cached_features) = cache.get(&sample.id) {
                return Ok(cached_features.clone());
            }
        }
        let audio = sample.get_normalized_audio();
        if audio.is_empty() {
            return Err(Error::Processing("Empty audio sample".to_string()));
        }
        let mut features = Vec::new();
        features.extend(self.extract_acoustic_features(&audio, sample.sample_rate)?);
        features.extend(self.extract_spectral_features(&audio, sample.sample_rate)?);
        features.extend(self.extract_prosodic_features(&audio, sample.sample_rate)?);
        features.extend(self.extract_speaker_features(&audio, sample.sample_rate)?);
        features.resize(self.config.embedding_dim, 0.0);
        {
            let mut cache = self.feature_cache.write().await;
            cache.insert(sample.id.clone(), features.clone());
        }
        Ok(features)
    }
    fn extract_acoustic_features(&self, audio: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        let features = vec![
            self.compute_rms_energy(audio),
            self.compute_peak_energy(audio),
            self.compute_energy_variance(audio),
            self.compute_mean(audio),
            self.compute_std(audio),
            self.compute_skewness(audio),
            self.compute_kurtosis(audio),
        ];
        Ok(features)
    }
    fn extract_spectral_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let features = vec![
            self.compute_spectral_centroid(audio, sample_rate),
            self.compute_spectral_rolloff(audio, sample_rate),
            self.compute_spectral_bandwidth(audio, sample_rate),
            self.compute_spectral_flux(audio),
            self.compute_zero_crossing_rate(audio),
            self.compute_high_frequency_energy(audio, sample_rate),
        ];
        Ok(features)
    }
    fn extract_prosodic_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let mut features = Vec::new();
        let f0_contour = self.extract_f0_contour(audio, sample_rate)?;
        features.push(self.compute_f0_mean(&f0_contour));
        features.push(self.compute_f0_std(&f0_contour));
        features.push(self.compute_f0_range(&f0_contour));
        features.push(self.compute_rhythm_strength(audio, sample_rate));
        features.push(self.compute_speech_rate(audio, sample_rate));
        Ok(features)
    }
    fn extract_speaker_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let mut features = Vec::new();
        features.push(self.compute_jitter(audio, sample_rate)?);
        features.push(self.compute_shimmer(audio, sample_rate)?);
        features.push(self.compute_harmonic_ratio(audio, sample_rate));
        features.extend(self.extract_formant_features(audio, sample_rate)?);
        Ok(features)
    }
    fn compute_rms_energy(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt()
    }
    fn compute_peak_energy(&self, audio: &[f32]) -> f32 {
        audio.iter().map(|x| x.abs()).fold(0.0, f32::max)
    }
    fn compute_energy_variance(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }
        let mean = self.compute_mean(audio);
        audio.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (audio.len() - 1) as f32
    }
    fn compute_mean(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        audio.iter().sum::<f32>() / audio.len() as f32
    }
    fn compute_std(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }
        let mean = self.compute_mean(audio);
        let variance =
            audio.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (audio.len() - 1) as f32;
        variance.sqrt()
    }
    fn compute_skewness(&self, audio: &[f32]) -> f32 {
        if audio.len() < 3 {
            return 0.0;
        }
        let mean = self.compute_mean(audio);
        let std = self.compute_std(audio);
        if std == 0.0 {
            return 0.0;
        }
        let n = audio.len() as f32;
        let skew_sum = audio
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f32>();
        (n / ((n - 1.0) * (n - 2.0))) * skew_sum
    }
    fn compute_kurtosis(&self, audio: &[f32]) -> f32 {
        if audio.len() < 4 {
            return 0.0;
        }
        let mean = self.compute_mean(audio);
        let std = self.compute_std(audio);
        if std == 0.0 {
            return 0.0;
        }
        let n = audio.len() as f32;
        let kurt_sum = audio
            .iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f32>();
        ((n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * kurt_sum
            - (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0))
    }
    fn compute_spectral_centroid(&self, audio: &[f32], sample_rate: u32) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let window_size = 1024.min(audio.len());
        let audio_slice = &audio[..window_size];
        let mut magnitude_spectrum = vec![0.0; window_size / 2];
        for (k, magnitude) in magnitude_spectrum.iter_mut().enumerate() {
            let mut real = 0.0;
            let mut imag = 0.0;
            for (n, &sample) in audio_slice.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / window_size as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            *magnitude = (real * real + imag * imag).sqrt();
        }
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        for (k, &magnitude) in magnitude_spectrum.iter().enumerate() {
            let frequency = k as f32 * sample_rate as f32 / window_size as f32;
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
        }
        if magnitude_sum > 0.0 {
            (weighted_sum / magnitude_sum) / 8000.0
        } else {
            0.5
        }
    }
    fn compute_spectral_rolloff(&self, audio: &[f32], sample_rate: u32) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let window_size = 1024.min(audio.len());
        let audio_slice = &audio[..window_size];
        let mut magnitude_spectrum = vec![0.0; window_size / 2];
        let mut total_energy = 0.0;
        for (k, magnitude) in magnitude_spectrum.iter_mut().enumerate() {
            let mut real = 0.0;
            let mut imag = 0.0;
            for (n, &sample) in audio_slice.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / window_size as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            let mag_value = (real * real + imag * imag).sqrt();
            *magnitude = mag_value;
            total_energy += mag_value;
        }
        let rolloff_threshold = total_energy * 0.85;
        let mut cumulative_energy = 0.0;
        for (k, &magnitude) in magnitude_spectrum.iter().enumerate() {
            cumulative_energy += magnitude;
            if cumulative_energy >= rolloff_threshold {
                let frequency = k as f32 * sample_rate as f32 / window_size as f32;
                return (frequency / 8000.0).clamp(0.0, 1.0);
            }
        }
        0.85
    }
    fn compute_spectral_bandwidth(&self, _audio: &[f32], _sample_rate: u32) -> f32 {
        0.5
    }
    fn compute_spectral_flux(&self, _audio: &[f32]) -> f32 {
        0.5
    }
    fn compute_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }
        let crossings = audio
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count();
        crossings as f32 / (audio.len() - 1) as f32
    }
    fn compute_high_frequency_energy(&self, _audio: &[f32], _sample_rate: u32) -> f32 {
        0.5
    }
    fn extract_f0_contour(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if audio.is_empty() {
            return Ok(vec![]);
        }
        let window_size = (sample_rate as f32 * 0.025) as usize;
        let hop_size = window_size / 2;
        let mut f0_values = Vec::new();
        for start in (0..audio.len()).step_by(hop_size) {
            let end = (start + window_size).min(audio.len());
            if end - start < window_size / 2 {
                break;
            }
            let window = &audio[start..end];
            let f0 = self.estimate_f0_autocorrelation(window, sample_rate)?;
            f0_values.push(f0);
        }
        self.smooth_f0_contour(&mut f0_values);
        Ok(f0_values)
    }
    /// Estimate F0 using autocorrelation method
    fn estimate_f0_autocorrelation(&self, window: &[f32], sample_rate: u32) -> Result<f32> {
        if window.len() < 64 {
            return Ok(0.0);
        }
        let mut windowed: Vec<f32> = window
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let w = 0.54
                    - 0.46
                        * (2.0 * std::f32::consts::PI * i as f32 / (window.len() - 1) as f32).cos();
                x * w
            })
            .collect();
        let mut autocorr = vec![0.0; window.len()];
        for lag in 0..window.len() {
            for i in 0..(window.len() - lag) {
                autocorr[lag] += windowed[i] * windowed[i + lag];
            }
        }
        let min_period = (sample_rate as f32 / 400.0) as usize;
        let max_period = (sample_rate as f32 / 80.0) as usize;
        let max_period = max_period.min(autocorr.len() - 1);
        if min_period >= max_period {
            return Ok(0.0);
        }
        let mut best_period = min_period;
        let mut best_value = autocorr[min_period];
        for (idx, &value) in autocorr[min_period..=max_period].iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_period = min_period + idx;
            }
        }
        let threshold = autocorr[0] * 0.3;
        if best_value < threshold {
            return Ok(0.0);
        }
        let f0 = sample_rate as f32 / best_period as f32;
        Ok(f0)
    }
    /// Smooth F0 contour to remove spurious values
    fn smooth_f0_contour(&self, f0_values: &mut [f32]) {
        if f0_values.len() < 3 {
            return;
        }
        for i in 1..f0_values.len() - 1 {
            let mut window = [f0_values[i - 1], f0_values[i], f0_values[i + 1]];
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            f0_values[i] = window[1];
        }
        let original = f0_values.to_vec();
        for i in 1..f0_values.len() - 1 {
            f0_values[i] = (original[i - 1] + original[i] + original[i + 1]) / 3.0;
        }
    }
    fn compute_f0_mean(&self, f0_contour: &[f32]) -> f32 {
        if f0_contour.is_empty() {
            return 0.0;
        }
        f0_contour.iter().sum::<f32>() / f0_contour.len() as f32
    }
    fn compute_f0_std(&self, f0_contour: &[f32]) -> f32 {
        if f0_contour.len() < 2 {
            return 0.0;
        }
        let mean = self.compute_f0_mean(f0_contour);
        let variance = f0_contour.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / (f0_contour.len() - 1) as f32;
        variance.sqrt()
    }
    fn compute_f0_range(&self, f0_contour: &[f32]) -> f32 {
        if f0_contour.is_empty() {
            return 0.0;
        }
        let min_f0 = f0_contour.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_f0 = f0_contour.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        max_f0 - min_f0
    }
    fn compute_rhythm_strength(&self, _audio: &[f32], _sample_rate: u32) -> f32 {
        0.5
    }
    fn compute_speech_rate(&self, _audio: &[f32], _sample_rate: u32) -> f32 {
        0.5
    }
    fn compute_jitter(&self, _audio: &[f32], _sample_rate: u32) -> Result<f32> {
        Ok(0.01)
    }
    fn compute_shimmer(&self, _audio: &[f32], _sample_rate: u32) -> Result<f32> {
        Ok(0.05)
    }
    fn compute_harmonic_ratio(&self, _audio: &[f32], _sample_rate: u32) -> f32 {
        0.7
    }
    fn extract_formant_features(&self, _audio: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        Ok(vec![800.0, 1200.0, 2400.0])
    }
}
/// Meta-learning model wrapper
pub(crate) struct MetaModel {
    /// Variable map for model parameters
    varmap: VarMap,
    /// Main network
    network: MetaNetwork,
    /// Optimizer
    optimizer: AdamW,
    /// Current parameters
    current_params: HashMap<String, Tensor>,
}
/// Advanced few-shot learner for voice cloning
pub struct FewShotLearner {
    /// Configuration
    pub(crate) config: FewShotConfig,
    /// Computation device
    pub(crate) device: Device,
    /// Meta-learning model
    pub(crate) meta_model: Option<MetaModel>,
    /// Feature extractor
    pub(crate) feature_extractor: FeatureExtractor,
    /// Training episodes history
    pub(crate) training_history: VecDeque<TrainingEpisode>,
    /// Performance metrics
    pub(crate) metrics: FewShotMetrics,
}
impl FewShotLearner {
    /// Create new few-shot learner
    pub fn new(config: FewShotConfig) -> Result<Self> {
        let device = Device::Cpu;
        let feature_extractor = FeatureExtractor::new(config.clone(), device.clone())?;
        Ok(Self {
            config,
            device,
            meta_model: None,
            feature_extractor,
            training_history: VecDeque::new(),
            metrics: FewShotMetrics::new(),
        })
    }
    /// Perform cross-lingual adaptation for a new speaker
    pub async fn adapt_speaker_cross_lingual(
        &mut self,
        speaker_id: &str,
        samples: &[VoiceSample],
        source_language: &str,
        target_language: &str,
    ) -> Result<FewShotResult> {
        let start_time = Instant::now();
        if !self.config.enable_cross_lingual {
            return Err(Error::Processing(
                "Cross-lingual learning is not enabled in configuration".to_string(),
            ));
        }
        if samples.len() < self.config.num_shots {
            return Err(Error::InsufficientData(format!(
                "Need at least {} samples for cross-lingual few-shot learning, got {}",
                self.config.num_shots,
                samples.len()
            )));
        }
        info!(
            "Starting cross-lingual adaptation for speaker {} from {} to {} with {} samples",
            speaker_id,
            source_language,
            target_language,
            samples.len()
        );
        let quality_samples: Vec<(VoiceSample, SampleQuality)> = samples
            .iter()
            .map(|sample| (sample.clone(), SampleQuality::from_sample(sample)))
            .collect();
        let filtered_samples = if self.config.use_quality_weighting {
            self.filter_by_quality(&quality_samples)?
        } else {
            quality_samples
        };
        if filtered_samples.len() < 2 {
            return Err(Error::Quality(
                "Insufficient high-quality samples for cross-lingual adaptation".to_string(),
            ));
        }
        let features = self
            .extract_cross_lingual_features(&filtered_samples, source_language, target_language)
            .await?;
        let phonetic_similarity =
            self.calculate_phonetic_similarity(source_language, target_language);
        let mut result = self
            .adapt_cross_lingual(&features, source_language, target_language)
            .await?;
        let adaptation_time = start_time.elapsed();
        result.cross_lingual_info = Some(CrossLingualInfo {
            source_language: source_language.to_string(),
            target_language: target_language.to_string(),
            language_adaptation_confidence: result.confidence * phonetic_similarity,
            phonetic_similarity,
            language_adaptation_applied: true,
        });
        self.metrics
            .update_adaptation_metrics(&result, adaptation_time);
        debug!(
            "Cross-lingual adaptation completed in {:?} with confidence {:.3}, phonetic similarity {:.3}",
            adaptation_time, result.confidence, phonetic_similarity
        );
        Ok(FewShotResult {
            adaptation_time,
            samples_used: filtered_samples.len(),
            ..result
        })
    }
    /// Extract cross-lingual features from samples
    async fn extract_cross_lingual_features(
        &self,
        samples: &[(VoiceSample, SampleQuality)],
        source_language: &str,
        target_language: &str,
    ) -> Result<Vec<(Vec<f32>, SampleQuality)>> {
        let mut features = Vec::new();
        for (sample, quality) in samples {
            let mut sample_features = self.feature_extractor.extract_features(sample).await?;
            self.apply_language_adaptation(&mut sample_features, source_language, target_language)?;
            features.push((sample_features, quality.clone()));
        }
        Ok(features)
    }
    /// Apply language-specific adaptations to feature vectors
    fn apply_language_adaptation(
        &self,
        features: &mut [f32],
        source_language: &str,
        target_language: &str,
    ) -> Result<()> {
        let adaptation_matrix =
            self.get_language_adaptation_matrix(source_language, target_language)?;
        for (i, &adaptation_factor) in adaptation_matrix.iter().enumerate() {
            if i < features.len() {
                features[i] *= adaptation_factor;
            }
        }
        self.apply_phonetic_mapping(features, source_language, target_language)?;
        Ok(())
    }
    /// Get language-specific adaptation matrix
    pub(crate) fn get_language_adaptation_matrix(
        &self,
        source_language: &str,
        target_language: &str,
    ) -> Result<Vec<f32>> {
        let mut matrix = vec![1.0; self.config.embedding_dim];
        let adaptation_factors = match (source_language, target_language) {
            ("en", "es") | ("es", "en") => self.get_english_spanish_adaptation(),
            ("en", "fr") | ("fr", "en") => self.get_english_french_adaptation(),
            ("en", "de") | ("de", "en") => self.get_english_german_adaptation(),
            ("en", "zh") | ("zh", "en") => self.get_english_chinese_adaptation(),
            ("en", "ja") | ("ja", "en") => self.get_english_japanese_adaptation(),
            ("en", "ko") | ("ko", "en") => self.get_english_korean_adaptation(),
            ("fr", "es") | ("es", "fr") => self.get_romance_language_adaptation(),
            ("de", "nl") | ("nl", "de") => self.get_germanic_language_adaptation(),
            _ => self.get_default_cross_lingual_adaptation(),
        };
        for (i, factor) in adaptation_factors.into_iter().enumerate() {
            if i < matrix.len() {
                matrix[i] = factor;
            }
        }
        Ok(matrix)
    }
    /// Apply phonetic mapping between languages
    fn apply_phonetic_mapping(
        &self,
        features: &mut [f32],
        source_language: &str,
        target_language: &str,
    ) -> Result<()> {
        let phonetic_shifts = self.get_phonetic_shifts(source_language, target_language);
        let formant_start = 50.min(features.len());
        let formant_end = 80.min(features.len());
        for (i, &shift) in phonetic_shifts.iter().enumerate() {
            let feature_idx = formant_start + i;
            if feature_idx < formant_end && feature_idx < features.len() {
                features[feature_idx] += shift;
            }
        }
        Ok(())
    }
    /// Calculate phonetic similarity between two languages
    pub(crate) fn calculate_phonetic_similarity(
        &self,
        source_language: &str,
        target_language: &str,
    ) -> f32 {
        if source_language == target_language {
            return 1.0;
        }
        match (source_language, target_language) {
            ("en", "de") | ("de", "en") => 0.75,
            ("en", "nl") | ("nl", "en") => 0.78,
            ("fr", "es") | ("es", "fr") => 0.85,
            ("fr", "it") | ("it", "fr") => 0.82,
            ("es", "it") | ("it", "es") => 0.84,
            ("es", "pt") | ("pt", "es") => 0.88,
            ("en", "fr") | ("fr", "en") => 0.65,
            ("en", "es") | ("es", "en") => 0.68,
            ("de", "fr") | ("fr", "de") => 0.62,
            ("en", "ru") | ("ru", "en") => 0.55,
            ("fr", "ru") | ("ru", "fr") => 0.52,
            ("de", "ru") | ("ru", "de") => 0.58,
            ("en", "zh") | ("zh", "en") => 0.35,
            ("en", "ja") | ("ja", "en") => 0.32,
            ("en", "ko") | ("ko", "en") => 0.30,
            ("en", "ar") | ("ar", "en") => 0.28,
            ("zh", "ja") | ("ja", "zh") => 0.45,
            ("zh", "ko") | ("ko", "zh") => 0.42,
            ("ja", "ko") | ("ko", "ja") => 0.48,
            _ => 0.40,
        }
    }
    /// Cross-lingual adaptation using language-aware prototypical networks
    async fn adapt_cross_lingual(
        &mut self,
        features: &[(Vec<f32>, SampleQuality)],
        source_language: &str,
        target_language: &str,
    ) -> Result<FewShotResult> {
        trace!(
            "Performing cross-lingual adaptation from {} to {}",
            source_language,
            target_language
        );
        let (support_features, query_features) = self.split_support_query(features);
        let mut prototype = vec![0.0; self.config.embedding_dim];
        let mut total_weight = 0.0;
        let phonetic_similarity =
            self.calculate_phonetic_similarity(source_language, target_language);
        for (feature, quality) in &support_features {
            let quality_weight = if self.config.use_quality_weighting {
                quality.overall_quality
            } else {
                1.0
            };
            let combined_weight = quality_weight * (0.5 + 0.5 * phonetic_similarity);
            for (i, &f) in feature.iter().enumerate() {
                if i < prototype.len() {
                    prototype[i] += f * combined_weight;
                }
            }
            total_weight += combined_weight;
        }
        if total_weight > 0.0 {
            for val in &mut prototype {
                *val /= total_weight;
            }
        }
        self.apply_final_cross_lingual_transform(&mut prototype, source_language, target_language)?;
        self.l2_normalize(&mut prototype);
        let base_confidence = self.evaluate_prototype(&prototype, &query_features)?;
        let confidence = base_confidence * (0.3 + 0.7 * phonetic_similarity);
        let quality_score = support_features
            .iter()
            .map(|(_, q)| q.overall_quality)
            .sum::<f32>()
            / support_features.len() as f32;
        Ok(FewShotResult {
            speaker_embedding: prototype,
            confidence,
            quality_score,
            samples_used: features.len(),
            adaptation_time: Duration::default(),
            algorithm: MetaLearningAlgorithm::ProtoNet,
            cross_lingual_info: None,
        })
    }
    /// Apply final cross-lingual transformation to the prototype
    fn apply_final_cross_lingual_transform(
        &self,
        prototype: &mut [f32],
        source_language: &str,
        target_language: &str,
    ) -> Result<()> {
        let bias_corrections = self.get_language_bias_corrections(source_language, target_language);
        for (i, &correction) in bias_corrections.iter().enumerate() {
            if i < prototype.len() {
                prototype[i] += correction;
            }
        }
        let phonetic_similarity =
            self.calculate_phonetic_similarity(source_language, target_language);
        let scaling_factor = 0.8 + 0.2 * phonetic_similarity;
        for val in prototype.iter_mut() {
            *val *= scaling_factor;
        }
        Ok(())
    }
    fn get_english_spanish_adaptation(&self) -> Vec<f32> {
        let mut factors = vec![1.0; 20];
        factors.extend(vec![1.1, 0.9, 1.2, 0.95, 1.05]);
        factors.resize(50, 1.0);
        factors
    }
    fn get_english_french_adaptation(&self) -> Vec<f32> {
        let mut factors = vec![1.0; 15];
        factors.extend(vec![0.85, 1.15, 0.9, 1.1, 0.95, 1.05]);
        factors.resize(50, 1.0);
        factors
    }
    fn get_english_german_adaptation(&self) -> Vec<f32> {
        let mut factors = vec![1.0; 18];
        factors.extend(vec![1.05, 0.98, 1.03, 0.97]);
        factors.resize(50, 1.0);
        factors
    }
    fn get_english_chinese_adaptation(&self) -> Vec<f32> {
        let mut factors = vec![0.7; 10];
        factors.extend(vec![1.5, 1.4, 1.3, 1.2, 1.1]);
        factors.resize(50, 0.9);
        factors
    }
    fn get_english_japanese_adaptation(&self) -> Vec<f32> {
        let mut factors = vec![0.8; 12];
        factors.extend(vec![1.3, 1.2, 0.9, 1.1, 0.85]);
        factors.resize(50, 0.85);
        factors
    }
    fn get_english_korean_adaptation(&self) -> Vec<f32> {
        let mut factors = vec![0.9; 15];
        factors.extend(vec![1.2, 0.8, 1.1, 0.95, 1.05]);
        factors.resize(50, 0.88);
        factors
    }
    fn get_romance_language_adaptation(&self) -> Vec<f32> {
        vec![1.02; 50]
    }
    fn get_germanic_language_adaptation(&self) -> Vec<f32> {
        vec![1.03; 50]
    }
    fn get_default_cross_lingual_adaptation(&self) -> Vec<f32> {
        vec![0.9; 50]
    }
    pub(crate) fn get_phonetic_shifts(
        &self,
        source_language: &str,
        target_language: &str,
    ) -> Vec<f32> {
        match (source_language, target_language) {
            ("en", "es") => vec![0.05, -0.02, 0.08, -0.04, 0.03],
            ("en", "fr") => vec![-0.03, 0.06, -0.05, 0.09, -0.02],
            ("en", "de") => vec![0.02, -0.01, 0.04, -0.03, 0.01],
            ("en", "zh") => vec![0.15, -0.12, 0.18, -0.15, 0.10],
            ("en", "ja") => vec![0.08, -0.06, 0.12, -0.09, 0.05],
            ("en", "ko") => vec![0.10, -0.08, 0.14, -0.11, 0.07],
            _ => vec![0.0; 5],
        }
    }
    fn get_language_bias_corrections(
        &self,
        source_language: &str,
        target_language: &str,
    ) -> Vec<f32> {
        let mut corrections = vec![0.0; self.config.embedding_dim];
        match (source_language, target_language) {
            ("en", "zh") | ("zh", "en") => {
                for i in 10..20 {
                    if i < corrections.len() {
                        corrections[i] = if source_language == "en" { 0.1 } else { -0.1 };
                    }
                }
            }
            ("en", "ar") | ("ar", "en") => {
                for i in 15..25 {
                    if i < corrections.len() {
                        corrections[i] = if source_language == "en" { 0.08 } else { -0.08 };
                    }
                }
            }
            _ => {
                for i in 5..10 {
                    if i < corrections.len() {
                        corrections[i] = 0.02;
                    }
                }
            }
        }
        corrections
    }
    /// Initialize meta-learning model
    pub fn initialize_meta_model(&mut self) -> Result<()> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        let network = self.create_meta_network(vs)?;
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: self.config.meta_learning_rate as f64,
                ..Default::default()
            },
        )?;
        self.meta_model = Some(MetaModel {
            varmap,
            network,
            optimizer,
            current_params: HashMap::new(),
        });
        info!(
            "Initialized meta-learning model with {} algorithm",
            format!("{:?}", self.config.meta_algorithm)
        );
        Ok(())
    }
    /// Perform few-shot adaptation for a new speaker
    pub async fn adapt_speaker(
        &mut self,
        speaker_id: &str,
        samples: &[VoiceSample],
    ) -> Result<FewShotResult> {
        let start_time = Instant::now();
        if samples.len() < self.config.num_shots {
            return Err(Error::InsufficientData(format!(
                "Need at least {} samples for few-shot learning, got {}",
                self.config.num_shots,
                samples.len()
            )));
        }
        info!(
            "Starting few-shot adaptation for speaker {} with {} samples",
            speaker_id,
            samples.len()
        );
        let quality_samples: Vec<(VoiceSample, SampleQuality)> = samples
            .iter()
            .map(|sample| (sample.clone(), SampleQuality::from_sample(sample)))
            .collect();
        let filtered_samples = if self.config.use_quality_weighting {
            self.filter_by_quality(&quality_samples)?
        } else {
            quality_samples
        };
        if filtered_samples.len() < 2 {
            return Err(Error::Quality(
                "Insufficient high-quality samples for adaptation".to_string(),
            ));
        }
        let features = self.extract_batch_features(&filtered_samples).await?;
        let result = match self.config.meta_algorithm {
            MetaLearningAlgorithm::MAML => self.adapt_maml(&features).await?,
            MetaLearningAlgorithm::ProtoNet => self.adapt_prototypical(&features).await?,
            MetaLearningAlgorithm::MatchingNet => self.adapt_matching(&features).await?,
            MetaLearningAlgorithm::RelationNet => self.adapt_relation(&features).await?,
            MetaLearningAlgorithm::MetaSGD => self.adapt_meta_sgd(&features).await?,
        };
        let adaptation_time = start_time.elapsed();
        self.metrics
            .update_adaptation_metrics(&result, adaptation_time);
        debug!(
            "Few-shot adaptation completed in {:?} with confidence {:.3}",
            adaptation_time, result.confidence
        );
        Ok(FewShotResult {
            adaptation_time,
            samples_used: filtered_samples.len(),
            ..result
        })
    }
    /// MAML (Model-Agnostic Meta-Learning) adaptation
    async fn adapt_maml(
        &mut self,
        features: &[(Vec<f32>, SampleQuality)],
    ) -> Result<FewShotResult> {
        trace!("Performing MAML adaptation");
        if self.meta_model.is_none() {
            self.initialize_meta_model()?;
        }
        let meta_model = self
            .meta_model
            .as_mut()
            .expect("meta_model was just initialized above");
        let (support_features, query_features) = self.split_support_query(features);
        let mut adapted_embedding = vec![0.0; self.config.embedding_dim];
        let mut total_weight = 0.0;
        for (feature, quality) in &support_features {
            let weight = if self.config.use_quality_weighting {
                quality.overall_quality
            } else {
                1.0
            };
            for (i, &f) in feature.iter().enumerate() {
                if i < adapted_embedding.len() {
                    adapted_embedding[i] += f * weight;
                }
            }
            total_weight += weight;
        }
        if total_weight > 0.0 {
            for val in &mut adapted_embedding {
                *val /= total_weight;
            }
        }
        let confidence = self.evaluate_embedding(&adapted_embedding, &query_features)?;
        let quality_score = support_features
            .iter()
            .map(|(_, q)| q.overall_quality)
            .sum::<f32>()
            / support_features.len() as f32;
        Ok(FewShotResult {
            speaker_embedding: adapted_embedding,
            confidence,
            quality_score,
            samples_used: features.len(),
            adaptation_time: Duration::default(),
            algorithm: MetaLearningAlgorithm::MAML,
            cross_lingual_info: None,
        })
    }
    /// Prototypical Networks adaptation
    async fn adapt_prototypical(
        &mut self,
        features: &[(Vec<f32>, SampleQuality)],
    ) -> Result<FewShotResult> {
        trace!("Performing Prototypical Networks adaptation");
        let (support_features, query_features) = self.split_support_query(features);
        let mut prototype = vec![0.0; self.config.embedding_dim];
        let mut total_weight = 0.0;
        for (feature, quality) in &support_features {
            let weight = if self.config.use_quality_weighting {
                quality.overall_quality
            } else {
                1.0
            };
            for (i, &f) in feature.iter().enumerate() {
                if i < prototype.len() {
                    prototype[i] += f * weight;
                }
            }
            total_weight += weight;
        }
        if total_weight > 0.0 {
            for val in &mut prototype {
                *val /= total_weight;
            }
        }
        self.l2_normalize(&mut prototype);
        let confidence = self.evaluate_prototype(&prototype, &query_features)?;
        let quality_score = support_features
            .iter()
            .map(|(_, q)| q.overall_quality)
            .sum::<f32>()
            / support_features.len() as f32;
        Ok(FewShotResult {
            speaker_embedding: prototype,
            confidence,
            quality_score,
            samples_used: features.len(),
            adaptation_time: Duration::default(),
            algorithm: MetaLearningAlgorithm::ProtoNet,
            cross_lingual_info: None,
        })
    }
    /// Matching Networks adaptation
    async fn adapt_matching(
        &mut self,
        features: &[(Vec<f32>, SampleQuality)],
    ) -> Result<FewShotResult> {
        trace!("Performing Matching Networks adaptation");
        let result = self.adapt_prototypical(features).await?;
        Ok(FewShotResult {
            algorithm: MetaLearningAlgorithm::MatchingNet,
            ..result
        })
    }
    /// Relation Networks adaptation
    async fn adapt_relation(
        &mut self,
        features: &[(Vec<f32>, SampleQuality)],
    ) -> Result<FewShotResult> {
        trace!("Performing Relation Networks adaptation");
        let result = self.adapt_prototypical(features).await?;
        Ok(FewShotResult {
            algorithm: MetaLearningAlgorithm::RelationNet,
            ..result
        })
    }
    /// Meta-SGD adaptation
    async fn adapt_meta_sgd(
        &mut self,
        features: &[(Vec<f32>, SampleQuality)],
    ) -> Result<FewShotResult> {
        trace!("Performing Meta-SGD adaptation");
        let mut result = self.adapt_maml(features).await?;
        result.algorithm = MetaLearningAlgorithm::MetaSGD;
        Ok(result)
    }
    /// Filter samples by quality threshold
    fn filter_by_quality(
        &self,
        samples: &[(VoiceSample, SampleQuality)],
    ) -> Result<Vec<(VoiceSample, SampleQuality)>> {
        let mut filtered: Vec<_> = samples
            .iter()
            .filter(|(_, quality)| quality.overall_quality >= self.config.quality_threshold)
            .cloned()
            .collect();
        filtered.sort_by(|a, b| {
            b.1.overall_quality
                .partial_cmp(&a.1.overall_quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if filtered.len() > self.config.num_shots * 2 {
            filtered.truncate(self.config.num_shots * 2);
        }
        debug!(
            "Filtered {} samples to {} high-quality samples",
            samples.len(),
            filtered.len()
        );
        Ok(filtered)
    }
    /// Extract features from batch of samples
    async fn extract_batch_features(
        &self,
        samples: &[(VoiceSample, SampleQuality)],
    ) -> Result<Vec<(Vec<f32>, SampleQuality)>> {
        let mut features = Vec::new();
        for (sample, quality) in samples {
            let sample_features = self.feature_extractor.extract_features(sample).await?;
            features.push((sample_features, quality.clone()));
        }
        Ok(features)
    }
    /// Split features into support and query sets
    fn split_support_query(&self, features: &[QualityFeaturePair]) -> SupportQuerySets {
        let num_support = self.config.num_shots.min(features.len() * 2 / 3);
        let support = features[..num_support].to_vec();
        let query = features[num_support..].to_vec();
        (support, query)
    }
    /// Evaluate embedding quality against query features
    fn evaluate_embedding(
        &self,
        embedding: &[f32],
        query_features: &[QualityFeaturePair],
    ) -> Result<f32> {
        if query_features.is_empty() {
            return Ok(0.8);
        }
        let mut total_similarity = 0.0;
        for (query_feature, _) in query_features {
            let similarity = self.compute_similarity(embedding, query_feature)?;
            total_similarity += similarity;
        }
        Ok((total_similarity / query_features.len() as f32).clamp(0.0, 1.0))
    }
    /// Evaluate prototype against query features
    fn evaluate_prototype(
        &self,
        prototype: &[f32],
        query_features: &[(Vec<f32>, SampleQuality)],
    ) -> Result<f32> {
        self.evaluate_embedding(prototype, query_features)
    }
    /// Compute similarity between two feature vectors
    pub(crate) fn compute_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(Error::Processing(
                "Feature vectors have different dimensions".to_string(),
            ));
        }
        match self.config.distance_metric {
            DistanceMetric::Cosine => {
                let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_a > 0.0 && norm_b > 0.0 {
                    let cosine_sim = dot_product / (norm_a * norm_b);
                    Ok((cosine_sim + 1.0) / 2.0)
                } else {
                    Ok(0.0)
                }
            }
            DistanceMetric::Euclidean => {
                let distance: f32 = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                Ok(1.0 / (1.0 + distance))
            }
            DistanceMetric::Manhattan => {
                let distance: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
                Ok(1.0 / (1.0 + distance))
            }
            DistanceMetric::Learned => self.compute_similarity_cosine(a, b),
        }
    }
    fn compute_similarity_cosine(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            let cosine_sim = dot_product / (norm_a * norm_b);
            Ok((cosine_sim + 1.0) / 2.0)
        } else {
            Ok(0.0)
        }
    }
    /// L2 normalize a vector
    fn l2_normalize(&self, vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector {
                *x /= norm;
            }
        }
    }
    /// Create meta-learning network
    fn create_meta_network(&self, vs: VarBuilder) -> Result<MetaNetwork> {
        let mut embedding_layers = Vec::new();
        let mut meta_layers = Vec::new();
        let mut adaptation_layers = Vec::new();
        let mut input_dim = self.config.embedding_dim;
        for (i, &hidden_dim) in self.config.meta_hidden_dims.iter().enumerate() {
            let layer = linear(input_dim, hidden_dim, vs.pp(format!("embed_{}", i)))?;
            embedding_layers.push(layer);
            input_dim = hidden_dim;
        }
        for (i, &hidden_dim) in self.config.meta_hidden_dims.iter().enumerate() {
            let layer = linear(input_dim, hidden_dim, vs.pp(format!("meta_{}", i)))?;
            meta_layers.push(layer);
            input_dim = hidden_dim;
        }
        for (i, &hidden_dim) in self.config.meta_hidden_dims.iter().enumerate() {
            let layer = linear(input_dim, hidden_dim, vs.pp(format!("adapt_{}", i)))?;
            adaptation_layers.push(layer);
            input_dim = hidden_dim;
        }
        let output_layer = linear(input_dim, self.config.embedding_dim, vs.pp("output"))?;
        Ok(MetaNetwork {
            embedding_layers,
            meta_layers,
            adaptation_layers,
            output_layer,
        })
    }
    /// Get few-shot learning metrics
    pub fn get_metrics(&self) -> &FewShotMetrics {
        &self.metrics
    }
    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = FewShotMetrics::new();
    }
}
/// Performance metrics for few-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotMetrics {
    /// Total episodes trained
    pub episodes_trained: u64,
    /// Average adaptation accuracy
    pub avg_accuracy: f32,
    /// Average adaptation time
    pub avg_adaptation_time: Duration,
    /// Success rate (confidence > threshold)
    pub success_rate: f32,
    /// Quality improvement ratio
    pub quality_improvement: f32,
}
impl FewShotMetrics {
    fn new() -> Self {
        Self {
            episodes_trained: 0,
            avg_accuracy: 0.0,
            avg_adaptation_time: Duration::default(),
            success_rate: 0.0,
            quality_improvement: 0.0,
        }
    }
    fn update_adaptation_metrics(&mut self, result: &FewShotResult, adaptation_time: Duration) {
        self.episodes_trained += 1;
        let alpha = 0.1;
        self.avg_accuracy = alpha * result.confidence + (1.0 - alpha) * self.avg_accuracy;
        let new_time_ms = adaptation_time.as_millis() as f32;
        let current_time_ms = self.avg_adaptation_time.as_millis() as f32;
        let avg_time_ms = alpha * new_time_ms + (1.0 - alpha) * current_time_ms;
        self.avg_adaptation_time = Duration::from_millis(avg_time_ms as u64);
        let success = if result.confidence > 0.7 { 1.0 } else { 0.0 };
        self.success_rate = alpha * success + (1.0 - alpha) * self.success_rate;
        self.quality_improvement =
            alpha * result.quality_score + (1.0 - alpha) * self.quality_improvement;
    }
}
/// Cross-lingual adaptation information
#[derive(Debug, Clone)]
pub struct CrossLingualInfo {
    /// Source language of training samples
    pub source_language: String,
    /// Target language for synthesis
    pub target_language: String,
    /// Language adaptation confidence
    pub language_adaptation_confidence: f32,
    /// Phonetic similarity score between languages
    pub phonetic_similarity: f32,
    /// Language-specific adaptation applied
    pub language_adaptation_applied: bool,
}
