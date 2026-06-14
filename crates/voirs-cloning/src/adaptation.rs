//! Speaker adaptation methods for voice cloning

use crate::{types::VoiceSample, Error, Result};
use candle_core::{Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Speaker adaptation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationMethod {
    /// Mean adaptation - simple averaging of features
    Mean,
    /// MAP (Maximum A Posteriori) adaptation
    MAP,
    /// MLLR (Maximum Likelihood Linear Regression)
    MLLR,
    /// Neural adaptation using speaker embeddings
    Neural,
    /// LoRA (Low-Rank Adaptation) for efficient fine-tuning
    LoRA,
    /// Feature-level adaptation
    FeatureLevel,
    /// Few-shot adaptation using limited samples
    FewShot,
}

/// Speaker adapter for performing adaptation
#[derive(Debug)]
pub struct SpeakerAdapter {
    /// Adaptation method to use
    method: AdaptationMethod,
    /// Neural network device
    device: Device,
    /// Adaptation configuration
    config: AdaptationConfig,
    /// Cached adaptation models
    adaptation_cache: HashMap<String, AdaptationModel>,
}

/// Configuration for speaker adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Hidden layer dimensions for neural adaptation
    pub hidden_dims: Vec<usize>,
    /// Learning rate for adaptation
    pub learning_rate: f32,
    /// Number of adaptation steps
    pub adaptation_steps: usize,
    /// Regularization strength
    pub regularization: f32,
    /// Minimum samples required for adaptation
    pub min_samples: usize,
    /// LoRA rank (for LoRA adaptation)
    pub lora_rank: usize,
    /// MLLR transform dimension
    pub mllr_transform_dim: usize,
}

/// Adaptation model containing learned parameters
#[derive(Debug, Clone)]
pub struct AdaptationModel {
    /// Adaptation method used
    pub method: AdaptationMethod,
    /// Speaker embedding
    pub speaker_embedding: Vec<f32>,
    /// Adaptation parameters (method-specific)
    pub parameters: AdaptationParameters,
    /// Training statistics
    pub training_stats: AdaptationStats,
}

/// Method-specific adaptation parameters
#[derive(Debug, Clone)]
pub enum AdaptationParameters {
    /// Mean adaptation parameters
    Mean {
        /// Mean feature vector
        mean_features: Vec<f32>,
        /// Feature covariance
        covariance: Array2<f32>,
    },
    /// MAP adaptation parameters
    MAP {
        /// Prior mean
        prior_mean: Vec<f32>,
        /// Posterior mean
        posterior_mean: Vec<f32>,
        /// Adaptation coefficient
        adaptation_coefficient: f32,
    },
    /// MLLR adaptation parameters
    MLLR {
        /// Transformation matrix
        transform_matrix: Array2<f32>,
        /// Bias vector
        bias_vector: Vec<f32>,
    },
    /// Neural adaptation parameters
    Neural {
        /// Learned speaker embedding
        speaker_embedding: Vec<f32>,
        /// Adaptation network weights
        network_weights: Vec<u8>, // Serialized neural network
    },
    /// LoRA adaptation parameters
    LoRA {
        /// Low-rank matrices A and B
        lora_a: Array2<f32>,
        lora_b: Array2<f32>,
        /// Scaling factor
        scaling: f32,
    },
    /// Feature-level adaptation parameters
    FeatureLevel {
        /// Feature transformation matrix
        feature_transform: Array2<f32>,
        /// Feature bias
        feature_bias: Vec<f32>,
    },
}

/// Adaptation training statistics
#[derive(Debug, Clone)]
pub struct AdaptationStats {
    /// Number of training samples used
    pub num_samples: usize,
    /// Training loss
    pub final_loss: f32,
    /// Training time
    pub training_time: std::time::Duration,
    /// Convergence achieved
    pub converged: bool,
}

/// Neural adaptation network
struct AdaptationNetwork {
    /// Input linear layer
    input_layer: Linear,
    /// Hidden layers
    hidden_layers: Vec<Linear>,
    /// Output layer
    output_layer: Linear,
    /// Device
    device: Device,
}

impl SpeakerAdapter {
    /// Create new speaker adapter
    pub fn new(method: AdaptationMethod) -> Result<Self> {
        let device = Device::Cpu; // Use CPU for now, could be GPU
        let config = AdaptationConfig::default();

        Ok(Self {
            method,
            device,
            config,
            adaptation_cache: HashMap::new(),
        })
    }

    /// Create adapter with custom configuration
    pub fn with_config(method: AdaptationMethod, config: AdaptationConfig) -> Result<Self> {
        let device = Device::Cpu;

        Ok(Self {
            method,
            device,
            config,
            adaptation_cache: HashMap::new(),
        })
    }

    /// Adapt speaker model from voice samples
    pub async fn adapt(
        &mut self,
        speaker_id: &str,
        samples: &[VoiceSample],
    ) -> Result<AdaptationModel> {
        if samples.len() < self.config.min_samples {
            return Err(Error::InsufficientData(format!(
                "Need at least {} samples for adaptation, got {}",
                self.config.min_samples,
                samples.len()
            )));
        }

        let start_time = std::time::Instant::now();

        // Extract features from samples
        let features = self.extract_features_batch(samples).await?;

        // Perform adaptation based on method
        let parameters = match self.method {
            AdaptationMethod::Mean => self.adapt_mean(&features)?,
            AdaptationMethod::MAP => self.adapt_map(&features)?,
            AdaptationMethod::MLLR => self.adapt_mllr(&features)?,
            AdaptationMethod::Neural => self.adapt_neural(&features).await?,
            AdaptationMethod::LoRA => self.adapt_lora(&features).await?,
            AdaptationMethod::FeatureLevel => self.adapt_feature_level(&features)?,
            AdaptationMethod::FewShot => self.adapt_few_shot(&features)?,
        };

        // Compute speaker embedding
        let speaker_embedding = self.compute_speaker_embedding(&features)?;

        let training_time = start_time.elapsed();
        let model = AdaptationModel {
            method: self.method,
            speaker_embedding: speaker_embedding.clone(),
            parameters,
            training_stats: AdaptationStats {
                num_samples: samples.len(),
                final_loss: 0.1, // Placeholder
                training_time,
                converged: true,
            },
        };

        // Cache the model
        self.adaptation_cache
            .insert(speaker_id.to_string(), model.clone());

        Ok(model)
    }

    /// Mean adaptation implementation
    fn adapt_mean(&self, features: &[Vec<f32>]) -> Result<AdaptationParameters> {
        if features.is_empty() {
            return Err(Error::Processing(
                "No features for mean adaptation".to_string(),
            ));
        }

        let feature_dim = features[0].len();
        let mut mean_features = vec![0.0; feature_dim];

        // Calculate mean
        for feature_vec in features {
            for (i, &value) in feature_vec.iter().enumerate() {
                mean_features[i] += value;
            }
        }

        for value in &mut mean_features {
            *value /= features.len() as f32;
        }

        // Calculate covariance matrix
        let mut covariance = Array2::zeros((feature_dim, feature_dim));
        for feature_vec in features {
            for i in 0..feature_dim {
                for j in 0..feature_dim {
                    let dev_i = feature_vec[i] - mean_features[i];
                    let dev_j = feature_vec[j] - mean_features[j];
                    covariance[[i, j]] += dev_i * dev_j;
                }
            }
        }

        covariance /= (features.len() - 1) as f32;

        Ok(AdaptationParameters::Mean {
            mean_features,
            covariance,
        })
    }

    /// MAP (Maximum A Posteriori) adaptation implementation
    fn adapt_map(&self, features: &[Vec<f32>]) -> Result<AdaptationParameters> {
        if features.is_empty() {
            return Err(Error::Processing(
                "No features for MAP adaptation".to_string(),
            ));
        }

        let feature_dim = features[0].len();

        // Use global mean as prior (in practice, this would come from a universal background model)
        let prior_mean = vec![0.0; feature_dim];

        // Calculate sample mean
        let mut sample_mean = vec![0.0; feature_dim];
        for feature_vec in features {
            for (i, &value) in feature_vec.iter().enumerate() {
                sample_mean[i] += value;
            }
        }
        for value in &mut sample_mean {
            *value /= features.len() as f32;
        }

        // MAP adaptation coefficient (simplified)
        let n_samples = features.len() as f32;
        let adaptation_coefficient = n_samples / (n_samples + self.config.regularization);

        // Compute MAP estimate
        let mut posterior_mean = vec![0.0; feature_dim];
        for i in 0..feature_dim {
            posterior_mean[i] = adaptation_coefficient * sample_mean[i]
                + (1.0 - adaptation_coefficient) * prior_mean[i];
        }

        Ok(AdaptationParameters::MAP {
            prior_mean,
            posterior_mean,
            adaptation_coefficient,
        })
    }

    /// MLLR (Maximum Likelihood Linear Regression) adaptation.
    ///
    /// Estimates the affine transform `W = [A | b]` that maps a speaker-independent
    /// reference distribution to the observed speaker (adaptation) features under the
    /// maximum-likelihood / least-squares criterion.
    ///
    /// The classical MLLR estimator needs, per regression class, a model mean `μ_m`
    /// and the adapted observation. This `adapt_mllr` call only receives the
    /// adaptation feature vectors, so the speaker-independent reference is built in a
    /// single pass from the pooled per-dimension mean `μ` and standard deviation `σ`
    /// of those vectors (a one-pass universal-background / standardisation basis).
    /// Each adaptation frame is then paired with its standardised counterpart
    /// `s_m = (x_m − μ) / σ`, and `W` is the maximum-likelihood affine map
    /// `x_m ≈ A·s_m + b`. With unit variances/occupancies the MLLR M-step reduces to
    /// the ordinary-least-squares normal equations, which is exactly what is solved
    /// here (the recovered `A ≈ diag(σ)`, `b ≈ μ` form the ML cepstral mean/variance
    /// transform). Because the alignment/occupancy is fixed (one regression class, no
    /// posteriors to re-estimate) the closed-form single pass is already the exact ML
    /// solution; no EM outer loop is required.
    fn adapt_mllr(&self, features: &[Vec<f32>]) -> Result<AdaptationParameters> {
        if features.is_empty() {
            return Err(Error::Processing(
                "No features for MLLR adaptation".to_string(),
            ));
        }

        let feature_dim = features[0].len();
        if feature_dim == 0 {
            return Err(Error::Processing(
                "Zero-dimensional features for MLLR adaptation".to_string(),
            ));
        }

        // Speaker-independent reference statistics (single-pass mean and std).
        let n = features.len() as f64;
        let mut mean = vec![0.0f64; feature_dim];
        for feature_vec in features {
            for (d, &value) in feature_vec.iter().enumerate() {
                if d < feature_dim {
                    mean[d] += value as f64;
                }
            }
        }
        mean.iter_mut().for_each(|m| *m /= n);

        let mut variance = vec![0.0f64; feature_dim];
        for feature_vec in features {
            for (d, &value) in feature_vec.iter().enumerate() {
                if d < feature_dim {
                    let deviation = value as f64 - mean[d];
                    variance[d] += deviation * deviation;
                }
            }
        }
        let std: Vec<f64> = variance.iter().map(|v| (v / n).sqrt()).collect();

        // Standardised speaker-independent sources s_m = (x_m − μ) / σ.
        let sources: Vec<Vec<f32>> = features
            .iter()
            .map(|feature_vec| {
                feature_vec
                    .iter()
                    .enumerate()
                    .map(|(d, &value)| {
                        let sigma = std[d];
                        if sigma > 1e-8 {
                            ((value as f64 - mean[d]) / sigma) as f32
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // Closed-form MLLR / OLS solve with unit variances and occupancies. The
        // normal matrix is ridge-regularised (scaled by the configured
        // regularisation strength) and singular systems fall back to identity.
        let ridge = (self.config.regularization as f64) * 1e-6;
        let (transform_matrix, bias_vector) =
            estimate_affine_transform(&sources, features, None, None, ridge);

        Ok(AdaptationParameters::MLLR {
            transform_matrix,
            bias_vector,
        })
    }

    /// Neural adaptation implementation
    async fn adapt_neural(&self, features: &[Vec<f32>]) -> Result<AdaptationParameters> {
        if features.is_empty() {
            return Err(Error::Processing(
                "No features for neural adaptation".to_string(),
            ));
        }

        let feature_dim = features[0].len();

        // Create adaptation network
        let varmap = candle_nn::VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &self.device);

        let network = self.create_adaptation_network(vs, feature_dim)?;

        // Convert features to tensors
        let feature_tensor = self.features_to_tensor(features)?;

        // Train the adaptation network (simplified)
        for _step in 0..self.config.adaptation_steps {
            let _output = network.forward(&feature_tensor)?;
            // In a real implementation, this would include loss computation and backpropagation
        }

        // Extract speaker embedding from network
        let speaker_embedding = self.extract_network_embedding(&network)?;

        // Serialize network weights (placeholder)
        let network_weights = vec![0u8; 1024]; // Placeholder serialized weights

        Ok(AdaptationParameters::Neural {
            speaker_embedding,
            network_weights,
        })
    }

    /// LoRA adaptation implementation
    async fn adapt_lora(&self, features: &[Vec<f32>]) -> Result<AdaptationParameters> {
        if features.is_empty() {
            return Err(Error::Processing(
                "No features for LoRA adaptation".to_string(),
            ));
        }

        let feature_dim = features[0].len();
        let rank = self.config.lora_rank;

        // Initialize LoRA matrices A and B
        let lora_a = Array2::zeros((rank, feature_dim));
        let lora_b = Array2::zeros((feature_dim, rank));
        let scaling = 1.0 / (rank as f32).sqrt();

        // In a real implementation, these would be trained using gradient descent
        // For now, initialize with small random values

        Ok(AdaptationParameters::LoRA {
            lora_a,
            lora_b,
            scaling,
        })
    }

    /// Feature-level adaptation implementation
    fn adapt_feature_level(&self, features: &[Vec<f32>]) -> Result<AdaptationParameters> {
        if features.is_empty() {
            return Err(Error::Processing(
                "No features for feature-level adaptation".to_string(),
            ));
        }

        let feature_dim = features[0].len();

        // Create identity transformation as baseline
        let mut feature_transform = Array2::eye(feature_dim);
        let feature_bias = vec![0.0; feature_dim];

        // Add small perturbations based on speaker characteristics
        for i in 0..feature_dim {
            feature_transform[[i, i]] += (scirs2_core::random::random::<f32>() - 0.5) * 0.1;
        }

        Ok(AdaptationParameters::FeatureLevel {
            feature_transform,
            feature_bias,
        })
    }

    /// Few-shot adaptation implementation
    fn adapt_few_shot(&self, features: &[Vec<f32>]) -> Result<AdaptationParameters> {
        if features.is_empty() {
            return Err(Error::Processing(
                "No features for few-shot adaptation".to_string(),
            ));
        }

        let feature_dim = features[0].len();

        // Few-shot learning using prototype-based approach
        // Compute prototype (mean) from few examples
        let mut prototype = vec![0.0; feature_dim];
        for feature_vec in features {
            for (i, &val) in feature_vec.iter().enumerate() {
                prototype[i] += val;
            }
        }

        // Normalize by number of samples
        for val in &mut prototype {
            *val /= features.len() as f32;
        }

        // Compute variance for adaptation strength
        let mut variance = vec![0.0; feature_dim];
        for feature_vec in features {
            for (i, &val) in feature_vec.iter().enumerate() {
                let diff = val - prototype[i];
                variance[i] += diff * diff;
            }
        }

        for val in &mut variance {
            *val /= features.len() as f32;
            *val = val.sqrt(); // Standard deviation
        }

        // Create adaptation transform based on prototype
        let mut feature_transform = Array2::eye(feature_dim);
        let feature_bias = prototype.clone();

        // Adjust transform based on few-shot variance
        for i in 0..feature_dim {
            let adaptation_strength = (variance[i] * 0.5).min(0.2); // Limit adaptation strength
            feature_transform[[i, i]] += adaptation_strength;
        }

        Ok(AdaptationParameters::FeatureLevel {
            feature_transform,
            feature_bias,
        })
    }

    /// Extract features from batch of voice samples
    async fn extract_features_batch(&self, samples: &[VoiceSample]) -> Result<Vec<Vec<f32>>> {
        let mut features = Vec::new();

        for sample in samples {
            let sample_features = self.extract_features_from_sample(sample).await?;
            features.push(sample_features);
        }

        Ok(features)
    }

    /// Extract features from a single voice sample
    async fn extract_features_from_sample(&self, sample: &VoiceSample) -> Result<Vec<f32>> {
        let audio = sample.get_normalized_audio();
        if audio.is_empty() {
            return Err(Error::Processing("Empty audio sample".to_string()));
        }

        // Extract basic acoustic features (placeholder)
        // In a real implementation, this would extract spectral features, MFCCs, etc.

        // Basic statistical features
        let mut features = vec![
            Self::compute_mean(&audio),
            Self::compute_std(&audio),
            Self::compute_skewness(&audio),
            Self::compute_kurtosis(&audio),
        ];

        // Spectral features (simplified)
        features.extend(self.extract_spectral_features(&audio)?);

        // Prosodic features
        features.extend(self.extract_prosodic_features(&audio, sample.sample_rate)?);

        // Pad or truncate to target dimension
        features.resize(self.config.embedding_dim, 0.0);

        Ok(features)
    }

    /// Extract spectral features from audio
    fn extract_spectral_features(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Simplified spectral features
        let features = vec![
            self.compute_spectral_centroid(audio),  // Spectral centroid
            self.compute_spectral_rolloff(audio),   // Spectral rolloff
            self.compute_zero_crossing_rate(audio), // Zero crossing rate
        ];

        Ok(features)
    }

    /// Extract prosodic features from audio
    fn extract_prosodic_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        // Fundamental frequency estimation (simplified)
        let f0 = self.estimate_f0(audio, sample_rate);
        features.push(f0);

        // Energy
        let energy = Self::compute_rms_energy(audio);
        features.push(energy);

        Ok(features)
    }

    /// Compute speaker embedding from features
    fn compute_speaker_embedding(&self, features: &[Vec<f32>]) -> Result<Vec<f32>> {
        if features.is_empty() {
            return Err(Error::Processing(
                "No features to compute embedding".to_string(),
            ));
        }

        let _feature_dim = features[0].len();
        let mut embedding = vec![0.0; self.config.embedding_dim];

        // Simple mean pooling of features
        for feature_vec in features {
            for (i, &value) in feature_vec.iter().enumerate() {
                if i < embedding.len() {
                    embedding[i] += value;
                }
            }
        }

        for value in &mut embedding {
            *value /= features.len() as f32;
        }

        // L2 normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut embedding {
                *value /= norm;
            }
        }

        Ok(embedding)
    }

    /// Create neural adaptation network
    fn create_adaptation_network(
        &self,
        vs: VarBuilder,
        input_dim: usize,
    ) -> Result<AdaptationNetwork> {
        let mut layers = Vec::new();
        let mut current_dim = input_dim;

        // Create hidden layers
        for &hidden_dim in &self.config.hidden_dims {
            let layer = linear(
                current_dim,
                hidden_dim,
                vs.pp(format!("hidden_{}", layers.len())),
            )?;
            layers.push(layer);
            current_dim = hidden_dim;
        }

        let input_layer = linear(input_dim, self.config.hidden_dims[0], vs.pp("input"))?;
        let output_layer = linear(current_dim, self.config.embedding_dim, vs.pp("output"))?;

        Ok(AdaptationNetwork {
            input_layer,
            hidden_layers: layers,
            output_layer,
            device: self.device.clone(),
        })
    }

    /// Convert features to tensor
    fn features_to_tensor(&self, features: &[Vec<f32>]) -> Result<Tensor> {
        let num_samples = features.len();
        let feature_dim = features[0].len();

        let mut data = Vec::with_capacity(num_samples * feature_dim);
        for feature_vec in features {
            data.extend_from_slice(feature_vec);
        }

        Tensor::from_vec(data, (num_samples, feature_dim), &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create tensor: {}", e)))
    }

    /// Extract embedding from neural network
    fn extract_network_embedding(&self, _network: &AdaptationNetwork) -> Result<Vec<f32>> {
        // Placeholder: extract learned embedding from network
        Ok(vec![0.0; self.config.embedding_dim])
    }

    /// Apply adaptation to new audio
    pub fn apply_adaptation(&self, model: &AdaptationModel, audio: &[f32]) -> Result<Vec<f32>> {
        match &model.parameters {
            AdaptationParameters::Mean { mean_features, .. } => {
                // Apply mean normalization
                let mut adapted = audio.to_vec();
                let audio_mean = Self::compute_mean(audio);
                let target_mean = Self::compute_mean(mean_features);
                let adjustment = target_mean - audio_mean;

                for sample in &mut adapted {
                    *sample += adjustment;
                }
                Ok(adapted)
            }
            AdaptationParameters::MLLR {
                transform_matrix: _,
                bias_vector,
            } => {
                // Apply MLLR transformation (simplified)
                let mut adapted = audio.to_vec();
                for (i, sample) in adapted.iter_mut().enumerate() {
                    if i < bias_vector.len() {
                        *sample += bias_vector[i];
                    }
                }
                Ok(adapted)
            }
            _ => {
                // For other methods, return original audio
                Ok(audio.to_vec())
            }
        }
    }

    /// Get cached adaptation model
    pub fn get_cached_model(&self, speaker_id: &str) -> Option<&AdaptationModel> {
        self.adaptation_cache.get(speaker_id)
    }

    /// Clear adaptation cache
    pub fn clear_cache(&mut self) {
        self.adaptation_cache.clear();
    }

    // Helper methods for feature extraction
    fn compute_mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            0.0
        } else {
            data.iter().sum::<f32>() / data.len() as f32
        }
    }

    fn compute_std(data: &[f32]) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean = Self::compute_mean(data);
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (data.len() - 1) as f32;
        variance.sqrt()
    }

    fn compute_skewness(data: &[f32]) -> f32 {
        if data.len() < 3 {
            return 0.0;
        }
        let mean = Self::compute_mean(data);
        let std = Self::compute_std(data);
        if std == 0.0 {
            return 0.0;
        }

        let n = data.len() as f32;
        let skew_sum = data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f32>();
        (n / ((n - 1.0) * (n - 2.0))) * skew_sum
    }

    fn compute_kurtosis(data: &[f32]) -> f32 {
        if data.len() < 4 {
            return 0.0;
        }
        let mean = Self::compute_mean(data);
        let std = Self::compute_std(data);
        if std == 0.0 {
            return 0.0;
        }

        let n = data.len() as f32;
        let kurt_sum = data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f32>();
        ((n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * kurt_sum
            - (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0))
    }

    fn compute_spectral_centroid(&self, audio: &[f32]) -> f32 {
        // Simplified spectral centroid calculation
        Self::compute_mean(audio)
    }

    fn compute_spectral_rolloff(&self, audio: &[f32]) -> f32 {
        // Simplified spectral rolloff calculation
        Self::compute_std(audio)
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

    fn estimate_f0(&self, audio: &[f32], sample_rate: u32) -> f32 {
        // Simplified F0 estimation using autocorrelation
        if audio.len() < 100 {
            return 0.0;
        }

        let min_period = sample_rate / 500; // 500 Hz max
        let max_period = sample_rate / 50; // 50 Hz min

        let mut max_corr = 0.0;
        let mut best_period = min_period;

        for period in min_period..max_period.min(audio.len() as u32 / 2) {
            let mut correlation = 0.0;
            let period_samples = period as usize;

            for i in 0..(audio.len() - period_samples) {
                correlation += audio[i] * audio[i + period_samples];
            }

            if correlation > max_corr {
                max_corr = correlation;
                best_period = period;
            }
        }

        if max_corr > 0.0 {
            sample_rate as f32 / best_period as f32
        } else {
            0.0
        }
    }

    fn compute_rms_energy(audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = audio.iter().map(|x| x * x).sum();
        (sum_squares / audio.len() as f32).sqrt()
    }
}

/// Build the affine identity transform `(A, b)` with `A` of shape
/// `out_dim × in_dim` (ones on the leading diagonal) and a zero bias.
///
/// Used as the safe fall-back whenever the MLLR system is degenerate or singular.
fn affine_identity(out_dim: usize, in_dim: usize) -> (Array2<f32>, Vec<f32>) {
    let mut a = Array2::<f32>::zeros((out_dim, in_dim));
    for i in 0..out_dim.min(in_dim) {
        a[[i, i]] = 1.0;
    }
    (a, vec![0.0f32; out_dim])
}

/// In-place LU factorisation with partial pivoting.
///
/// On success returns the combined lower/upper factors (unit lower triangle stored
/// below the diagonal) together with the row permutation. Returns `None` when a
/// pivot magnitude drops below `tol`, i.e. the matrix is (numerically) singular.
fn lu_factor(mut a: Array2<f64>, tol: f64) -> Option<(Array2<f64>, Vec<usize>)> {
    let n = a.nrows();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Partial pivoting: pick the largest-magnitude entry in column k.
        let mut pivot_row = k;
        let mut pivot_mag = a[[k, k]].abs();
        for i in (k + 1)..n {
            let mag = a[[i, k]].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = i;
            }
        }
        if pivot_mag < tol {
            return None;
        }
        if pivot_row != k {
            for j in 0..n {
                let tmp = a[[k, j]];
                a[[k, j]] = a[[pivot_row, j]];
                a[[pivot_row, j]] = tmp;
            }
            piv.swap(k, pivot_row);
        }

        let pivot = a[[k, k]];
        for i in (k + 1)..n {
            let factor = a[[i, k]] / pivot;
            a[[i, k]] = factor;
            for j in (k + 1)..n {
                let above = a[[k, j]];
                a[[i, j]] -= factor * above;
            }
        }
    }

    Some((a, piv))
}

/// Solve `A·x = rhs` from a precomputed [`lu_factor`] result.
fn lu_solve(lu: &Array2<f64>, piv: &[usize], rhs: &[f64]) -> Vec<f64> {
    let n = lu.nrows();
    // Apply the row permutation to the right-hand side.
    let mut x: Vec<f64> = piv.iter().map(|&p| rhs[p]).collect();

    // Forward substitution (unit lower triangle).
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..i {
            sum -= lu[[i, j]] * x[j];
        }
        x[i] = sum;
    }
    // Back substitution (upper triangle).
    for i in (0..n).rev() {
        let mut sum = x[i];
        for j in (i + 1)..n {
            sum -= lu[[i, j]] * x[j];
        }
        x[i] = sum / lu[[i, i]];
    }
    x
}

/// Estimate the affine transform `W = [A | b]` (returned as `(A, b)`) mapping each
/// `source` vector to its paired `target` vector under the maximum-likelihood /
/// least-squares criterion
///
/// ```text
/// minimise  Σ_m Σ_i (γ_m / σ²_{m,i}) · (a_i·s_m + b_i − t_{m,i})²
/// ```
///
/// This is the classical MLLR estimator. For every output dimension `i` the optimal
/// row `w_i = [a_i, b_i]` solves the normal equations `G_i · w_iᵀ = k_i`, where
///
/// ```text
/// G_i = Σ_m (γ_m / σ²_{m,i}) · ξ_m ξ_mᵀ      (per-dimension Gauss accumulator)
/// k_i = Σ_m (γ_m / σ²_{m,i}) · t_{m,i} · ξ_m  (cross-correlation)
/// ξ_m = [s_mᵀ, 1]ᵀ                            (homogeneous source vector)
/// ```
///
/// * `variances` — optional per-frame, per-output-dimension variances `σ²_{m,i}`.
///   When `None` every output dimension shares the same Gauss accumulator `G`, so it
///   is factorised once and reused (ordinary least squares = MLLR with unit
///   variance). When provided, each output dimension is solved independently with its
///   own precision-weighted accumulator.
/// * `occupancies` — optional per-frame occupation counts `γ_m` (default `1`).
/// * `ridge` — relative Tikhonov regularisation added to the diagonal of `G`
///   (scaled by its mean diagonal energy) for numerical stability.
///
/// Degenerate input (empty, mismatched lengths) and singular systems fall back to the
/// affine identity, so the function never panics.
fn estimate_affine_transform(
    sources: &[Vec<f32>],
    targets: &[Vec<f32>],
    variances: Option<&[Vec<f32>]>,
    occupancies: Option<&[f32]>,
    ridge: f64,
) -> (Array2<f32>, Vec<f32>) {
    let m = sources.len();
    let in_dim = sources.first().map(Vec::len).unwrap_or(0);
    let out_dim = targets.first().map(Vec::len).unwrap_or(0);

    // Validate shapes; any inconsistency yields a safe identity transform.
    if m == 0 || m != targets.len() || in_dim == 0 || out_dim == 0 {
        return affine_identity(out_dim, in_dim);
    }
    if sources.iter().any(|s| s.len() != in_dim) || targets.iter().any(|t| t.len() != out_dim) {
        return affine_identity(out_dim, in_dim);
    }
    if let Some(var) = variances {
        if var.len() != m || var.iter().any(|v| v.len() != out_dim) {
            return affine_identity(out_dim, in_dim);
        }
    }
    if let Some(occ) = occupancies {
        if occ.len() != m {
            return affine_identity(out_dim, in_dim);
        }
    }

    let ext = in_dim + 1; // homogeneous coordinate for the bias term
    let weight = |row: usize| -> f64 { occupancies.map_or(1.0, |occ| occ[row] as f64) };

    let mut a_mat = Array2::<f32>::zeros((out_dim, in_dim));
    let mut bias = vec![0.0f32; out_dim];

    match variances {
        // ---- Ordinary least squares: a single shared Gauss accumulator G. ----
        None => {
            let mut g = Array2::<f64>::zeros((ext, ext));
            let mut k = Array2::<f64>::zeros((ext, out_dim));

            for row in 0..m {
                let w = weight(row);
                let s = &sources[row];
                let t = &targets[row];
                for p in 0..ext {
                    let xp = if p < in_dim { s[p] as f64 } else { 1.0 };
                    let wxp = w * xp;
                    for q in p..ext {
                        let xq = if q < in_dim { s[q] as f64 } else { 1.0 };
                        g[[p, q]] += wxp * xq;
                    }
                    for (i, &ti) in t.iter().enumerate() {
                        k[[p, i]] += wxp * ti as f64;
                    }
                }
            }
            // Mirror the upper triangle into the lower one.
            for p in 0..ext {
                for q in 0..p {
                    g[[p, q]] = g[[q, p]];
                }
            }

            let mean_diag = (g.diag().iter().sum::<f64>() / ext as f64).max(1e-12);
            for d in g.diag_mut() {
                *d += ridge * mean_diag;
            }
            let tol = 1e-12 * mean_diag.max(1.0);

            match lu_factor(g, tol) {
                Some((lu, piv)) => {
                    for i in 0..out_dim {
                        let rhs: Vec<f64> = (0..ext).map(|p| k[[p, i]]).collect();
                        let w_row = lu_solve(&lu, &piv, &rhs);
                        for (j, slot) in a_mat.row_mut(i).iter_mut().enumerate() {
                            *slot = w_row[j] as f32;
                        }
                        bias[i] = w_row[in_dim] as f32;
                    }
                }
                None => return affine_identity(out_dim, in_dim),
            }
        }
        // ---- Precision-weighted MLLR: one accumulator per output dimension. ----
        Some(var) => {
            for i in 0..out_dim {
                let mut g = Array2::<f64>::zeros((ext, ext));
                let mut k = vec![0.0f64; ext];

                for row in 0..m {
                    let precision = weight(row) / (var[row][i] as f64).max(1e-8);
                    let s = &sources[row];
                    let ti = targets[row][i] as f64;
                    for p in 0..ext {
                        let xp = if p < in_dim { s[p] as f64 } else { 1.0 };
                        let pxp = precision * xp;
                        for q in p..ext {
                            let xq = if q < in_dim { s[q] as f64 } else { 1.0 };
                            g[[p, q]] += pxp * xq;
                        }
                        k[p] += pxp * ti;
                    }
                }
                for p in 0..ext {
                    for q in 0..p {
                        g[[p, q]] = g[[q, p]];
                    }
                }

                let mean_diag = (g.diag().iter().sum::<f64>() / ext as f64).max(1e-12);
                for d in g.diag_mut() {
                    *d += ridge * mean_diag;
                }
                let tol = 1e-12 * mean_diag.max(1.0);

                match lu_factor(g, tol) {
                    Some((lu, piv)) => {
                        let w_row = lu_solve(&lu, &piv, &k);
                        for (j, slot) in a_mat.row_mut(i).iter_mut().enumerate() {
                            *slot = w_row[j] as f32;
                        }
                        bias[i] = w_row[in_dim] as f32;
                    }
                    None => {
                        // Singular row: keep the identity mapping for this dimension.
                        if i < in_dim {
                            a_mat[[i, i]] = 1.0;
                        }
                    }
                }
            }
        }
    }

    (a_mat, bias)
}

impl AdaptationNetwork {
    /// Forward pass through the network
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.input_layer.forward(input)?;

        for layer in &self.hidden_layers {
            x = layer.forward(&x)?;
            x = x.relu()?; // ReLU activation
        }

        let output = self.output_layer.forward(&x)?;
        Ok(output)
    }
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 512,
            hidden_dims: vec![256, 128],
            learning_rate: 0.001,
            adaptation_steps: 100,
            regularization: 10.0,
            min_samples: 3,
            lora_rank: 8,
            mllr_transform_dim: 256,
        }
    }
}

impl Default for SpeakerAdapter {
    fn default() -> Self {
        Self::new(AdaptationMethod::Neural).expect("Failed to create default SpeakerAdapter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceSample;

    #[tokio::test]
    async fn test_mean_adaptation() {
        let mut adapter = SpeakerAdapter::new(AdaptationMethod::Mean).unwrap();

        // Create test samples
        let mut samples = Vec::new();
        for i in 0..5 {
            let audio = vec![0.1 * i as f32; 1000];
            samples.push(VoiceSample::new(format!("sample_{}", i), audio, 16000));
        }

        let model = adapter.adapt("test_speaker", &samples).await.unwrap();
        assert_eq!(model.method, AdaptationMethod::Mean);
        assert!(!model.speaker_embedding.is_empty());
    }

    #[tokio::test]
    async fn test_map_adaptation() {
        let mut adapter = SpeakerAdapter::new(AdaptationMethod::MAP).unwrap();

        let mut samples = Vec::new();
        for i in 0..5 {
            let audio = vec![0.1 * (i as f32 + 1.0); 1000];
            samples.push(VoiceSample::new(format!("sample_{}", i), audio, 16000));
        }

        let model = adapter.adapt("test_speaker", &samples).await.unwrap();
        assert_eq!(model.method, AdaptationMethod::MAP);

        if let AdaptationParameters::MAP {
            adaptation_coefficient,
            ..
        } = &model.parameters
        {
            assert!(*adaptation_coefficient > 0.0 && *adaptation_coefficient <= 1.0);
        } else {
            panic!("Expected MAP parameters");
        }
    }

    #[tokio::test]
    async fn test_insufficient_data() {
        let mut adapter = SpeakerAdapter::new(AdaptationMethod::FewShot).unwrap();

        // Only one sample, but config requires min_samples (default 3)
        let samples = vec![VoiceSample::new(
            "sample_1".to_string(),
            vec![0.1; 1000],
            16000,
        )];

        let result = adapter.adapt("test_speaker", &samples).await;
        assert!(result.is_err());

        if let Err(Error::InsufficientData(_)) = result {
            // Expected error type
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_feature_extraction_helpers() {
        let audio = vec![1.0, -1.0, 2.0, -2.0, 0.5];

        let mean = SpeakerAdapter::compute_mean(&audio);
        assert!((mean - 0.1).abs() < 0.01);

        let zcr = SpeakerAdapter::default().compute_zero_crossing_rate(&audio);
        assert!(zcr > 0.0);
    }

    #[test]
    fn test_adaptation_config() {
        let config = AdaptationConfig::default();
        assert_eq!(config.embedding_dim, 512);
        assert_eq!(config.min_samples, 3);
        assert!(!config.hidden_dims.is_empty());
    }

    /// Deterministic linear-congruential generator for reproducible test data
    /// (avoids any statistical RNG dependency in tests).
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed | 1)
        }

        /// Next pseudo-random value in `[0, 1)`.
        fn unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    /// Total squared error of the affine fit `A·s + b` against `targets`.
    fn fit_error(a: &Array2<f32>, b: &[f32], sources: &[Vec<f32>], targets: &[Vec<f32>]) -> f64 {
        let out_dim = b.len();
        let in_dim = a.ncols();
        let mut total = 0.0f64;
        for (s, t) in sources.iter().zip(targets.iter()) {
            for i in 0..out_dim {
                let mut pred = b[i] as f64;
                for j in 0..in_dim {
                    pred += a[[i, j]] as f64 * s[j] as f64;
                }
                let diff = pred - t[i] as f64;
                total += diff * diff;
            }
        }
        total
    }

    #[test]
    fn test_estimate_affine_recovers_known_map() {
        let in_dim = 4;
        let out_dim = 4;
        let m = 60;
        let mut lcg = Lcg::new(0x1234_5678);

        // Known, well-conditioned affine map (diagonally dominant + off-diagonal mix).
        let mut a_known = vec![vec![0.0f32; in_dim]; out_dim];
        let mut b_known = vec![0.0f32; out_dim];
        for i in 0..out_dim {
            for j in 0..in_dim {
                let base = if i == j { 1.0 } else { 0.0 };
                a_known[i][j] = base + 0.3 * (2.0 * lcg.unit() - 1.0);
            }
            b_known[i] = 2.0 * lcg.unit() - 1.0;
        }

        // Sources from a broad distribution => well-conditioned design matrix.
        let mut sources = Vec::with_capacity(m);
        let mut targets = Vec::with_capacity(m);
        for _ in 0..m {
            let s: Vec<f32> = (0..in_dim).map(|_| 4.0 * lcg.unit() - 2.0).collect();
            let t: Vec<f32> = (0..out_dim)
                .map(|i| {
                    let mut acc = b_known[i];
                    for j in 0..in_dim {
                        acc += a_known[i][j] * s[j];
                    }
                    acc
                })
                .collect();
            sources.push(s);
            targets.push(t);
        }

        let (a_hat, b_hat) = estimate_affine_transform(&sources, &targets, None, None, 1e-9);

        let err_recovered = fit_error(&a_hat, &b_hat, &sources, &targets);
        let (a_id, b_id) = affine_identity(out_dim, in_dim);
        let err_identity = fit_error(&a_id, &b_id, &sources, &targets);

        assert!(
            err_identity > 1.0,
            "identity fit error should be substantial: {err_identity}"
        );
        assert!(
            err_recovered < err_identity * 1e-3,
            "recovered error {err_recovered} should be far below identity error {err_identity}"
        );

        // Parameters are approximately recovered for the well-conditioned system.
        for i in 0..out_dim {
            for j in 0..in_dim {
                assert!(
                    (a_hat[[i, j]] - a_known[i][j]).abs() < 1e-2,
                    "A[{i},{j}] = {} expected {}",
                    a_hat[[i, j]],
                    a_known[i][j]
                );
            }
            assert!(
                (b_hat[i] - b_known[i]).abs() < 1e-2,
                "b[{i}] = {} expected {}",
                b_hat[i],
                b_known[i]
            );
        }
    }

    #[test]
    fn test_estimate_affine_weighted_mllr() {
        let in_dim = 3;
        let out_dim = 3;
        let m = 40;
        let mut lcg = Lcg::new(0x0bad_c0de);

        let a_known = [[1.2f32, -0.3, 0.1], [0.0, 0.8, 0.25], [-0.15, 0.05, 1.1]];
        let b_known = [0.5f32, -0.2, 0.3];

        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut variances = Vec::new();
        let mut occupancies = Vec::new();
        for _ in 0..m {
            let s: Vec<f32> = (0..in_dim).map(|_| 3.0 * lcg.unit() - 1.5).collect();
            let t: Vec<f32> = (0..out_dim)
                .map(|i| {
                    let mut acc = b_known[i];
                    for j in 0..in_dim {
                        acc += a_known[i][j] * s[j];
                    }
                    acc
                })
                .collect();
            // Non-uniform but strictly positive precisions/occupancies. As the map is
            // exact, the precision-weighted MLLR solve still recovers (A, b).
            let v: Vec<f32> = (0..out_dim).map(|_| 0.5 + lcg.unit()).collect();
            sources.push(s);
            targets.push(t);
            variances.push(v);
            occupancies.push(0.5 + lcg.unit());
        }

        let (a_hat, b_hat) = estimate_affine_transform(
            &sources,
            &targets,
            Some(&variances),
            Some(&occupancies),
            1e-9,
        );

        for i in 0..out_dim {
            for j in 0..in_dim {
                assert!(
                    (a_hat[[i, j]] - a_known[i][j]).abs() < 1e-2,
                    "weighted A[{i},{j}] = {} expected {}",
                    a_hat[[i, j]],
                    a_known[i][j]
                );
            }
            assert!(
                (b_hat[i] - b_known[i]).abs() < 1e-2,
                "weighted b[{i}] = {} expected {}",
                b_hat[i],
                b_known[i]
            );
        }
    }

    #[test]
    fn test_estimate_affine_singular_returns_identity() {
        // A single sample with no regularisation leaves the Gauss matrix rank-deficient
        // (ext = in_dim + 1 = 4 > rank 1) so the solve must fall back to identity.
        let sources = vec![vec![1.0f32, 2.0, 3.0]];
        let targets = vec![vec![4.0f32, 5.0, 6.0]];
        let (a_hat, b_hat) = estimate_affine_transform(&sources, &targets, None, None, 0.0);
        let (a_id, b_id) = affine_identity(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (a_hat[[i, j]] - a_id[[i, j]]).abs() < 1e-6,
                    "singular fallback should be identity"
                );
            }
            assert!((b_hat[i] - b_id[i]).abs() < 1e-6);
        }

        // Empty input must not panic and yields an empty transform.
        let empty: Vec<Vec<f32>> = Vec::new();
        let (a_empty, b_empty) = estimate_affine_transform(&empty, &empty, None, None, 1e-6);
        assert_eq!(a_empty.nrows(), 0);
        assert_eq!(a_empty.ncols(), 0);
        assert!(b_empty.is_empty());

        // Mismatched source/target counts fall back to identity safely.
        let src = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let tgt = vec![vec![1.0f32, 2.0]];
        let (a_mis, b_mis) = estimate_affine_transform(&src, &tgt, None, None, 1e-6);
        assert_eq!(a_mis.nrows(), 2);
        assert_eq!(a_mis.ncols(), 2);
        assert!((a_mis[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((a_mis[[1, 1]] - 1.0).abs() < 1e-6);
        assert_eq!(b_mis, vec![0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_mllr_adaptation_endtoend() {
        let mut adapter = SpeakerAdapter::new(AdaptationMethod::MLLR).unwrap();

        // Distinct samples so the pooled per-dimension std is non-zero.
        let mut samples = Vec::new();
        for i in 0..6 {
            let mut audio = vec![0.0f32; 2000];
            for (n, x) in audio.iter_mut().enumerate() {
                *x = 0.2 * ((i as f32 + 1.0) * 0.01 * n as f32).sin() + 0.05 * i as f32;
            }
            samples.push(VoiceSample::new(format!("s{i}"), audio, 16000));
        }

        let model = adapter.adapt("spk", &samples).await.unwrap();
        assert_eq!(model.method, AdaptationMethod::MLLR);

        if let AdaptationParameters::MLLR {
            transform_matrix,
            bias_vector,
        } = &model.parameters
        {
            let dim = adapter.config.embedding_dim;
            assert_eq!(transform_matrix.nrows(), dim);
            assert_eq!(transform_matrix.ncols(), dim);
            assert_eq!(bias_vector.len(), dim);
            // A real, data-driven transform: the bias carries the feature means and is
            // not the all-zero placeholder of the old identity-plus-noise stub.
            let bias_energy: f32 = bias_vector.iter().map(|b| b.abs()).sum();
            assert!(bias_energy > 0.0, "MLLR bias must be data-driven");
            // Numerically guarded solve => every entry is finite.
            assert!(transform_matrix.iter().all(|v| v.is_finite()));
            assert!(bias_vector.iter().all(|v| v.is_finite()));
        } else {
            panic!("expected MLLR parameters");
        }
    }
}
