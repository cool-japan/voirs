//! G2P model definitions, training, and loading.

use crate::{G2pError, LanguageCode, Phoneme, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// G2P model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// Rule-based model
    RuleBased,
    /// Statistical model
    Statistical,
    /// Neural network model
    Neural,
    /// Hybrid model combining multiple approaches
    Hybrid,
}

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model architecture parameters
    pub architecture: ArchitectureConfig,
    /// Training configuration
    pub training: TrainingConfig,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Neural network architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Input vocabulary size
    pub vocab_size: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Attention mechanism enabled
    pub use_attention: bool,
    /// Bidirectional processing
    pub bidirectional: bool,
    /// Activation function
    pub activation: String,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Validation split ratio
    pub validation_split: f32,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Optimizer type
    pub optimizer: String,
    /// Learning rate schedule
    pub lr_schedule: Option<LearningRateSchedule>,
    /// Regularization settings
    pub regularization: RegularizationConfig,
}

/// Learning rate schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRateSchedule {
    /// Schedule type (step, exponential, cosine)
    pub schedule_type: String,
    /// Decay rate
    pub decay_rate: f32,
    /// Decay steps
    pub decay_steps: usize,
    /// Minimum learning rate
    pub min_lr: f32,
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1: f32,
    /// L2 regularization strength
    pub l2: f32,
    /// Dropout rate
    pub dropout: f32,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f32>,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Description
    pub description: String,
    /// Target language
    pub language: LanguageCode,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Training duration
    pub training_duration: Option<Duration>,
    /// Training dataset info
    pub dataset_info: Option<DatasetInfo>,
    /// Model performance metrics
    pub performance_metrics: HashMap<String, f32>,
    /// Model size in bytes
    pub model_size: Option<u64>,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Number of training examples
    pub train_size: usize,
    /// Number of validation examples
    pub validation_size: usize,
    /// Number of test examples
    pub test_size: Option<usize>,
    /// Dataset source
    pub source: String,
    /// Dataset version
    pub version: String,
}

/// Training progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    /// Current epoch
    pub epoch: usize,
    /// Total epochs
    pub total_epochs: usize,
    /// Current step in epoch
    pub step: usize,
    /// Total steps in epoch
    pub total_steps: usize,
    /// Training loss
    pub train_loss: f32,
    /// Validation loss
    pub val_loss: Option<f32>,
    /// Training accuracy
    pub train_accuracy: f32,
    /// Validation accuracy
    pub val_accuracy: Option<f32>,
    /// Learning rate
    pub learning_rate: f32,
    /// Elapsed time
    pub elapsed_time: Duration,
    /// Estimated time remaining
    pub eta: Option<Duration>,
}

/// Model evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Phoneme-level accuracy
    pub phoneme_accuracy: f32,
    /// Word-level accuracy
    pub word_accuracy: f32,
    /// Edit distance (Levenshtein)
    pub edit_distance: f32,
    /// BLEU score
    pub bleu_score: Option<f32>,
    /// Perplexity
    pub perplexity: Option<f32>,
    /// Confidence score distribution
    pub confidence_stats: ConfidenceStats,
    /// Per-phoneme accuracy breakdown
    pub phoneme_breakdown: HashMap<String, f32>,
}

/// Confidence score statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceStats {
    /// Mean confidence score
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Minimum confidence
    pub min: f32,
    /// Maximum confidence
    pub max: f32,
    /// Median confidence
    pub median: f32,
}

/// G2P model with training capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pModel {
    /// Model configuration
    pub config: ModelConfig,
    /// Model weights and parameters
    pub parameters: ModelParameters,
    /// Training history
    pub training_history: Vec<TrainingProgress>,
    /// Evaluation metrics
    pub evaluation_metrics: Option<EvaluationMetrics>,
    /// Model file path
    pub model_path: Option<PathBuf>,
}

/// Model parameters storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Model weights as byte array
    pub weights: Vec<u8>,
    /// Vocabulary mapping
    pub vocabulary: HashMap<String, usize>,
    /// Phoneme mapping
    pub phoneme_mapping: HashMap<String, usize>,
    /// Additional parameters
    pub additional_params: HashMap<String, Vec<u8>>,
}

/// Training dataset representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataset {
    /// Training examples
    pub examples: Vec<TrainingExample>,
    /// Dataset metadata
    pub metadata: DatasetInfo,
    /// Language code
    pub language: LanguageCode,
}

/// Individual training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input text
    pub text: String,
    /// Target phonemes
    pub phonemes: Vec<Phoneme>,
    /// Additional context
    pub context: Option<String>,
    /// Example weight for training
    pub weight: f32,
}

/// Transfer learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningConfig {
    /// Source model path
    pub source_model_path: PathBuf,
    /// Layers to freeze during transfer
    pub freeze_layers: Vec<usize>,
    /// Fine-tuning learning rate
    pub fine_tune_lr: f32,
    /// Target language for transfer
    pub target_language: LanguageCode,
    /// Adaptation strategy
    pub adaptation_strategy: AdaptationStrategy,
}

/// Model adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Full fine-tuning
    FullFineTuning,
    /// Feature extraction (freeze base, train head)
    FeatureExtraction,
    /// Gradual unfreezing
    GradualUnfreezing,
    /// Domain adaptation
    DomainAdaptation,
}

/// Few-shot learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotConfig {
    /// Number of examples per phoneme
    pub examples_per_phoneme: usize,
    /// Meta-learning algorithm
    pub meta_learning_algorithm: String,
    /// Support set size
    pub support_set_size: usize,
    /// Query set size
    pub query_set_size: usize,
    /// Number of gradient steps
    pub gradient_steps: usize,
    /// Inner learning rate
    pub inner_lr: f32,
    /// Outer learning rate
    pub outer_lr: f32,
}

/// Pronunciation customization settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PronunciationCustomization {
    /// Custom pronunciation dictionary
    pub custom_dict: HashMap<String, Vec<Phoneme>>,
    /// Phoneme substitution rules
    pub substitution_rules: Vec<SubstitutionRule>,
    /// Regional accent modifications
    pub accent_modifications: HashMap<String, AccentModification>,
    /// Context-sensitive rules
    pub context_rules: Vec<ContextRule>,
}

/// Phoneme substitution rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstitutionRule {
    /// Source phoneme pattern
    pub source_pattern: String,
    /// Target phoneme pattern
    pub target_pattern: String,
    /// Context conditions
    pub context_conditions: Vec<String>,
    /// Rule priority
    pub priority: usize,
}

/// Regional accent modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccentModification {
    /// Accent name
    pub accent_name: String,
    /// Phoneme transformations
    pub transformations: Vec<SubstitutionRule>,
    /// Accent strength (0.0-1.0)
    pub strength: f32,
}

/// Context-sensitive pronunciation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRule {
    /// Word pattern
    pub word_pattern: String,
    /// Preceding context
    pub preceding_context: Option<String>,
    /// Following context
    pub following_context: Option<String>,
    /// Target pronunciation
    pub target_pronunciation: Vec<Phoneme>,
    /// Rule confidence
    pub confidence: f32,
}

impl G2pModel {
    /// Create new G2P model with configuration
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            parameters: ModelParameters {
                weights: Vec::new(),
                vocabulary: HashMap::new(),
                phoneme_mapping: HashMap::new(),
                additional_params: HashMap::new(),
            },
            training_history: Vec::new(),
            evaluation_metrics: None,
            model_path: None,
        }
    }

    /// Load model from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_content = std::fs::read(path.as_ref())
            .map_err(|e| G2pError::ModelError(format!("Failed to read model file: {e}")))?;

        let model: G2pModel = bincode::deserialize(&file_content)
            .map_err(|e| G2pError::ModelError(format!("Failed to deserialize model: {e}")))?;

        Ok(model)
    }

    /// Save model to file
    pub fn save_to_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.model_path = Some(path.as_ref().to_path_buf());

        let serialized = bincode::serialize(self)
            .map_err(|e| G2pError::ModelError(format!("Failed to serialize model: {e}")))?;

        std::fs::write(path.as_ref(), serialized)
            .map_err(|e| G2pError::ModelError(format!("Failed to write model file: {e}")))?;

        Ok(())
    }

    /// Get model size in bytes
    pub fn model_size(&self) -> usize {
        self.parameters.weights.len()
            + self.parameters.vocabulary.len() * 32 // rough estimate
            + self.parameters.phoneme_mapping.len() * 32
            + self.parameters.additional_params.values().map(|v| v.len()).sum::<usize>()
    }

    /// Add training progress entry
    pub fn add_training_progress(&mut self, progress: TrainingProgress) {
        self.training_history.push(progress);
    }

    /// Get latest training progress
    pub fn latest_training_progress(&self) -> Option<&TrainingProgress> {
        self.training_history.last()
    }

    /// Set evaluation metrics
    pub fn set_evaluation_metrics(&mut self, metrics: EvaluationMetrics) {
        self.evaluation_metrics = Some(metrics);
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        !self.parameters.weights.is_empty() && !self.training_history.is_empty()
    }

    /// Get supported language
    pub fn language(&self) -> LanguageCode {
        self.config.metadata.language
    }

    /// Get model performance summary
    pub fn performance_summary(&self) -> HashMap<String, f32> {
        let mut summary = HashMap::new();

        if let Some(metrics) = &self.evaluation_metrics {
            summary.insert("phoneme_accuracy".to_string(), metrics.phoneme_accuracy);
            summary.insert("word_accuracy".to_string(), metrics.word_accuracy);
            summary.insert("edit_distance".to_string(), metrics.edit_distance);

            if let Some(bleu) = metrics.bleu_score {
                summary.insert("bleu_score".to_string(), bleu);
            }

            if let Some(perplexity) = metrics.perplexity {
                summary.insert("perplexity".to_string(), perplexity);
            }
        }

        summary
    }
}

impl TrainingDataset {
    /// Create new training dataset
    pub fn new(language: LanguageCode, metadata: DatasetInfo) -> Self {
        Self {
            examples: Vec::new(),
            metadata,
            language,
        }
    }

    /// Add training example
    pub fn add_example(&mut self, example: TrainingExample) {
        self.examples.push(example);
    }

    /// Load dataset from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| G2pError::ModelError(format!("Failed to read dataset file: {e}")))?;

        let dataset: TrainingDataset = serde_json::from_str(&file_content)
            .map_err(|e| G2pError::ModelError(format!("Failed to parse dataset: {e}")))?;

        Ok(dataset)
    }

    /// Save dataset to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| G2pError::ModelError(format!("Failed to serialize dataset: {e}")))?;

        std::fs::write(path.as_ref(), serialized)
            .map_err(|e| G2pError::ModelError(format!("Failed to write dataset file: {e}")))?;

        Ok(())
    }

    /// Split dataset into train/validation/test sets
    pub fn split(
        &self,
        train_ratio: f32,
        val_ratio: f32,
    ) -> (TrainingDataset, TrainingDataset, TrainingDataset) {
        let total_size = self.examples.len();
        let train_size = (total_size as f32 * train_ratio) as usize;
        let val_size = (total_size as f32 * val_ratio) as usize;

        let mut train_examples = Vec::new();
        let mut val_examples = Vec::new();
        let mut test_examples = Vec::new();

        for (i, example) in self.examples.iter().enumerate() {
            if i < train_size {
                train_examples.push(example.clone());
            } else if i < train_size + val_size {
                val_examples.push(example.clone());
            } else {
                test_examples.push(example.clone());
            }
        }

        let train_dataset = TrainingDataset {
            examples: train_examples,
            metadata: DatasetInfo {
                name: format!("{}_train", self.metadata.name),
                train_size,
                validation_size: 0,
                test_size: None,
                source: self.metadata.source.clone(),
                version: self.metadata.version.clone(),
            },
            language: self.language,
        };

        let val_dataset = TrainingDataset {
            examples: val_examples,
            metadata: DatasetInfo {
                name: format!("{}_val", self.metadata.name),
                train_size: 0,
                validation_size: val_size,
                test_size: None,
                source: self.metadata.source.clone(),
                version: self.metadata.version.clone(),
            },
            language: self.language,
        };

        let test_dataset = TrainingDataset {
            examples: test_examples,
            metadata: DatasetInfo {
                name: format!("{}_test", self.metadata.name),
                train_size: 0,
                validation_size: 0,
                test_size: Some(total_size - train_size - val_size),
                source: self.metadata.source.clone(),
                version: self.metadata.version.clone(),
            },
            language: self.language,
        };

        (train_dataset, val_dataset, test_dataset)
    }

    /// Get dataset statistics
    pub fn statistics(&self) -> DatasetStatistics {
        let total_examples = self.examples.len();
        let total_phonemes: usize = self.examples.iter().map(|e| e.phonemes.len()).sum();
        let avg_phonemes_per_example = if total_examples > 0 {
            total_phonemes as f32 / total_examples as f32
        } else {
            0.0
        };

        let mut phoneme_counts = HashMap::new();
        for example in &self.examples {
            for phoneme in &example.phonemes {
                *phoneme_counts.entry(phoneme.symbol.clone()).or_insert(0) += 1;
            }
        }

        DatasetStatistics {
            total_examples,
            total_phonemes,
            avg_phonemes_per_example,
            phoneme_distribution: phoneme_counts,
            language: self.language,
        }
    }
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Total number of examples
    pub total_examples: usize,
    /// Total number of phonemes
    pub total_phonemes: usize,
    /// Average phonemes per example
    pub avg_phonemes_per_example: f32,
    /// Phoneme distribution
    pub phoneme_distribution: HashMap<String, usize>,
    /// Dataset language
    pub language: LanguageCode,
}

/// Default implementations
impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10000,
            hidden_dims: vec![256, 128],
            num_layers: 2,
            dropout: 0.1,
            use_attention: true,
            bidirectional: true,
            activation: "relu".to_string(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            validation_split: 0.2,
            early_stopping_patience: 10,
            optimizer: "adam".to_string(),
            lr_schedule: None,
            regularization: RegularizationConfig::default(),
        }
    }
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1: 0.0,
            l2: 0.0001,
            dropout: 0.1,
            gradient_clip: Some(1.0),
        }
    }
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            examples_per_phoneme: 5,
            meta_learning_algorithm: "maml".to_string(),
            support_set_size: 10,
            query_set_size: 5,
            gradient_steps: 1,
            inner_lr: 0.01,
            outer_lr: 0.001,
        }
    }
}
