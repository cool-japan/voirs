//! Comprehensive benchmarking suite for ASR models
//!
//! This module provides extensive benchmarking capabilities including:
//! - Word Error Rate (WER) and Character Error Rate (CER) calculation
//! - Real-time factor (RTF) measurement
//! - Memory usage profiling
//! - Accuracy vs speed trade-off analysis

use crate::traits::ASRModel;
use crate::RecognitionError;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Benchmarking configuration
#[derive(Debug, Clone)]
/// Benchmarking Config
pub struct BenchmarkingConfig {
    /// Test datasets to use
    pub datasets: Vec<Dataset>,
    /// Models to benchmark
    pub models: Vec<String>,
    /// Languages to test
    pub languages: Vec<LanguageCode>,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable real-time factor measurement
    pub enable_rtf_measurement: bool,
    /// Number of warmup runs
    pub warmup_runs: usize,
    /// Number of benchmark runs for averaging
    pub benchmark_runs: usize,
    /// Output detailed logs
    pub verbose_logging: bool,
}

impl Default for BenchmarkingConfig {
    fn default() -> Self {
        Self {
            datasets: vec![Dataset::LibriSpeech, Dataset::CommonVoice, Dataset::VCTK],
            models: vec!["whisper_base".to_string(), "whisper_tiny".to_string()],
            languages: vec![LanguageCode::EnUs, LanguageCode::EnGb],
            enable_memory_profiling: true,
            enable_rtf_measurement: true,
            warmup_runs: 3,
            benchmark_runs: 10,
            verbose_logging: true,
        }
    }
}

/// Test dataset enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Dataset
pub enum Dataset {
    /// Libri speech
    LibriSpeech,
    /// Common voice
    CommonVoice,
    /// V c t k
    VCTK,
    /// Custom
    Custom(String),
}

/// Test sample for benchmarking
#[derive(Debug, Clone)]
/// Test Sample
pub struct TestSample {
    /// Audio buffer
    pub audio: AudioBuffer,
    /// Ground truth transcription
    pub ground_truth: String,
    /// Sample identifier
    pub id: String,
    /// Language
    pub language: LanguageCode,
    /// Dataset source
    pub dataset: Dataset,
    /// Sample metadata
    pub metadata: HashMap<String, String>,
}

/// Word Error Rate calculation result
#[derive(Debug, Clone)]
/// W E R Result
pub struct WERResult {
    /// Word Error Rate (0.0 to 1.0+)
    pub wer: f32,
    /// Character Error Rate (0.0 to 1.0+)
    pub cer: f32,
    /// Number of substitutions
    pub substitutions: usize,
    /// Number of deletions
    pub deletions: usize,
    /// Number of insertions
    pub insertions: usize,
    /// Total number of reference words
    pub reference_words: usize,
    /// Total number of reference characters
    pub reference_chars: usize,
    /// Detailed alignment information
    pub alignment: Option<AlignmentResult>,
}

/// Alignment result for detailed analysis
#[derive(Debug, Clone)]
/// Alignment Result
pub struct AlignmentResult {
    /// Aligned reference words
    pub reference_aligned: Vec<String>,
    /// Aligned hypothesis words
    pub hypothesis_aligned: Vec<String>,
    /// Operations performed (S=substitution, D=deletion, I=insertion, C=correct)
    pub operations: Vec<char>,
}

/// Performance benchmark result
#[derive(Debug, Clone)]
/// Performance Benchmark
pub struct PerformanceBenchmark {
    /// Model identifier
    pub model_id: String,
    /// Dataset used
    pub dataset: Dataset,
    /// Language tested
    pub language: LanguageCode,
    /// Word Error Rate
    pub wer: f32,
    /// Character Error Rate
    pub cer: f32,
    /// Real-time factor (`processing_time` / `audio_duration`)
    pub rtf: f32,
    /// Average processing time per sample (seconds)
    pub avg_processing_time: f32,
    /// Peak memory usage (MB)
    pub peak_memory_mb: f32,
    /// Average memory usage (MB)
    pub avg_memory_mb: f32,
    /// Number of samples processed
    pub sample_count: usize,
    /// Detailed per-sample results
    pub sample_results: Vec<SampleResult>,
}

/// Individual sample benchmark result
#[derive(Debug, Clone)]
/// Sample Result
pub struct SampleResult {
    /// Sample ID
    pub sample_id: String,
    /// Processing time
    pub processing_time: Duration,
    /// Memory usage (MB)
    pub memory_usage_mb: f32,
    /// WER for this sample
    pub wer: f32,
    /// CER for this sample
    pub cer: f32,
    /// Confidence score
    pub confidence: f32,
    /// Transcription result
    pub transcription: String,
    /// Ground truth
    pub ground_truth: String,
}

/// Comprehensive benchmarking suite
pub struct ASRBenchmarkingSuite {
    /// Configuration
    config: BenchmarkingConfig,
    /// Available models
    models: Arc<RwLock<HashMap<String, Arc<dyn ASRModel>>>>,
    /// Test datasets
    datasets: Arc<RwLock<HashMap<Dataset, Vec<TestSample>>>>,
    /// Benchmark results
    results: Arc<RwLock<Vec<PerformanceBenchmark>>>,
}

impl ASRBenchmarkingSuite {
    /// Create a new benchmarking suite
    pub async fn new(config: BenchmarkingConfig) -> Result<Self, RecognitionError> {
        let models = Arc::new(RwLock::new(HashMap::new()));
        let datasets = Arc::new(RwLock::new(HashMap::new()));
        let results = Arc::new(RwLock::new(Vec::new()));

        let suite = Self {
            config,
            models,
            datasets,
            results,
        };

        // Load test datasets
        suite.load_datasets().await?;

        Ok(suite)
    }

    /// Add a model to benchmark
    pub async fn add_model(&self, model_id: String, model: Arc<dyn ASRModel>) {
        let mut models = self.models.write().await;
        models.insert(model_id, model);
    }

    /// Load test datasets
    async fn load_datasets(&self) -> Result<(), RecognitionError> {
        let mut datasets = self.datasets.write().await;

        for dataset in &self.config.datasets {
            match dataset {
                Dataset::LibriSpeech => {
                    let samples = self.load_librispeech_samples().await?;
                    datasets.insert(Dataset::LibriSpeech, samples);
                }
                Dataset::CommonVoice => {
                    let samples = self.load_commonvoice_samples().await?;
                    datasets.insert(Dataset::CommonVoice, samples);
                }
                Dataset::VCTK => {
                    let samples = self.load_vctk_samples().await?;
                    datasets.insert(Dataset::VCTK, samples);
                }
                Dataset::Custom(path) => {
                    let samples = self.load_custom_samples(path).await?;
                    datasets.insert(dataset.clone(), samples);
                }
            }
        }

        Ok(())
    }

    /// Load `LibriSpeech` test samples (mock implementation)
    async fn load_librispeech_samples(&self) -> Result<Vec<TestSample>, RecognitionError> {
        // Mock LibriSpeech samples
        let samples = vec![
            TestSample {
                audio: AudioBuffer::new(vec![0.1; 32000], 16000, 1), // 2 seconds
                ground_truth: "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG".to_string(),
                id: "librispeech_001".to_string(),
                language: LanguageCode::EnUs,
                dataset: Dataset::LibriSpeech,
                metadata: HashMap::new(),
            },
            TestSample {
                audio: AudioBuffer::new(vec![0.2; 48000], 16000, 1), // 3 seconds
                ground_truth: "SPEECH RECOGNITION TECHNOLOGY HAS ADVANCED SIGNIFICANTLY"
                    .to_string(),
                id: "librispeech_002".to_string(),
                language: LanguageCode::EnUs,
                dataset: Dataset::LibriSpeech,
                metadata: HashMap::new(),
            },
        ];

        tracing::info!("Loaded {} LibriSpeech test samples", samples.len());
        Ok(samples)
    }

    /// Load Common Voice test samples (mock implementation)
    async fn load_commonvoice_samples(&self) -> Result<Vec<TestSample>, RecognitionError> {
        // Mock Common Voice samples
        let samples = vec![TestSample {
            audio: AudioBuffer::new(vec![0.15; 24000], 16000, 1), // 1.5 seconds
            ground_truth: "Hello world, this is a test".to_string(),
            id: "commonvoice_001".to_string(),
            language: LanguageCode::EnUs,
            dataset: Dataset::CommonVoice,
            metadata: HashMap::new(),
        }];

        tracing::info!("Loaded {} Common Voice test samples", samples.len());
        Ok(samples)
    }

    /// Load VCTK test samples (enhanced mock implementation)
    async fn load_vctk_samples(&self) -> Result<Vec<TestSample>, RecognitionError> {
        // Enhanced mock VCTK samples with more realistic speaker diversity
        let samples = vec![
            TestSample {
                audio: AudioBuffer::new(vec![0.12; 40000], 16000, 1), // 2.5 seconds
                ground_truth: "Please call Stella".to_string(),
                id: "vctk_p225_001".to_string(),
                language: LanguageCode::EnUs,
                dataset: Dataset::VCTK,
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("speaker_id".to_string(), "p225".to_string());
                    metadata.insert("gender".to_string(), "female".to_string());
                    metadata.insert("accent".to_string(), "english".to_string());
                    metadata
                },
            },
            TestSample {
                audio: AudioBuffer::new(vec![0.08; 36000], 16000, 1), // 2.25 seconds
                ground_truth: "Ask her to bring these things with her from the store".to_string(),
                id: "vctk_p226_001".to_string(),
                language: LanguageCode::EnUs,
                dataset: Dataset::VCTK,
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("speaker_id".to_string(), "p226".to_string());
                    metadata.insert("gender".to_string(), "male".to_string());
                    metadata.insert("accent".to_string(), "english".to_string());
                    metadata
                },
            },
            TestSample {
                audio: AudioBuffer::new(vec![0.14; 44000], 16000, 1), // 2.75 seconds
                ground_truth: "Six spoons of fresh snow peas, five thick slabs of blue cheese"
                    .to_string(),
                id: "vctk_p227_001".to_string(),
                language: LanguageCode::EnUs,
                dataset: Dataset::VCTK,
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("speaker_id".to_string(), "p227".to_string());
                    metadata.insert("gender".to_string(), "male".to_string());
                    metadata.insert("accent".to_string(), "american".to_string());
                    metadata
                },
            },
        ];

        tracing::info!("Loaded {} VCTK test samples", samples.len());
        Ok(samples)
    }

    /// Load custom test samples (mock implementation)
    async fn load_custom_samples(&self, _path: &str) -> Result<Vec<TestSample>, RecognitionError> {
        Ok(Vec::new()) // Mock empty for now
    }

    /// Run comprehensive benchmark
    pub async fn run_benchmark(&self) -> Result<Vec<PerformanceBenchmark>, RecognitionError> {
        let models = self.models.read().await;
        let datasets = self.datasets.read().await;

        let mut all_results = Vec::new();

        for (model_id, model) in models.iter() {
            for (dataset, samples) in datasets.iter() {
                for language in &self.config.languages {
                    let language_samples: Vec<_> =
                        samples.iter().filter(|s| s.language == *language).collect();

                    if language_samples.is_empty() {
                        continue;
                    }

                    tracing::info!(
                        "Benchmarking {} on {:?} ({:?}) with {} samples",
                        model_id,
                        dataset,
                        language,
                        language_samples.len()
                    );

                    let benchmark = self
                        .benchmark_model_on_dataset(
                            model_id,
                            model.clone(),
                            dataset.clone(),
                            *language,
                            &language_samples,
                        )
                        .await?;

                    all_results.push(benchmark);
                }
            }
        }

        // Store results
        let mut results = self.results.write().await;
        results.extend(all_results.clone());

        Ok(all_results)
    }

    /// Benchmark a specific model on a dataset
    async fn benchmark_model_on_dataset(
        &self,
        model_id: &str,
        model: Arc<dyn ASRModel>,
        dataset: Dataset,
        language: LanguageCode,
        samples: &[&TestSample],
    ) -> Result<PerformanceBenchmark, RecognitionError> {
        let mut sample_results = Vec::new();
        let mut total_word_error_rate = 0.0;
        let mut total_char_error_rate = 0.0;
        let mut total_processing_time = Duration::ZERO;
        let mut peak_memory_mb: f32 = 0.0;
        let mut total_memory_mb = 0.0;

        // Warmup runs
        if self.config.warmup_runs > 0 {
            tracing::debug!("Running {} warmup iterations", self.config.warmup_runs);
            for _ in 0..self.config.warmup_runs {
                if let Some(sample) = samples.first() {
                    let _ = model.transcribe(&sample.audio, None).await;
                }
            }
        }

        // Benchmark runs
        for sample in samples {
            let mut sample_processing_times = Vec::new();
            let mut sample_transcriptions = Vec::new();
            let mut sample_memory_usages = Vec::new();

            // Multiple runs for averaging
            for _ in 0..self.config.benchmark_runs {
                let start_time = Instant::now();
                let start_memory = self.get_memory_usage();

                let transcript = model.transcribe(&sample.audio, None).await?;

                let processing_time = start_time.elapsed();
                let end_memory = self.get_memory_usage();
                let memory_usage = end_memory - start_memory;

                sample_processing_times.push(processing_time);
                sample_transcriptions.push(transcript.text);
                sample_memory_usages.push(memory_usage);

                peak_memory_mb = peak_memory_mb.max(end_memory);
            }

            // Average the results
            let avg_processing_time = sample_processing_times.iter().sum::<Duration>()
                / sample_processing_times.len() as u32;
            let avg_memory_usage =
                sample_memory_usages.iter().sum::<f32>() / sample_memory_usages.len() as f32;
            let best_transcription = &sample_transcriptions[0]; // Use first transcription for WER calculation

            // Calculate WER and CER
            let wer_result = self.calculate_wer(&sample.ground_truth, best_transcription);

            // Calculate confidence score based on WER and CER
            let confidence = self.calculate_confidence_score(wer_result.wer, wer_result.cer);

            let sample_result = SampleResult {
                sample_id: sample.id.clone(),
                processing_time: avg_processing_time,
                memory_usage_mb: avg_memory_usage,
                wer: wer_result.wer,
                cer: wer_result.cer,
                confidence,
                transcription: best_transcription.clone(),
                ground_truth: sample.ground_truth.clone(),
            };

            sample_results.push(sample_result);
            total_word_error_rate += wer_result.wer;
            total_char_error_rate += wer_result.cer;
            total_processing_time += avg_processing_time;
            total_memory_mb += avg_memory_usage;
        }

        let sample_count = samples.len();
        let average_word_error_rate = total_word_error_rate / sample_count as f32;
        let average_char_error_rate = total_char_error_rate / sample_count as f32;
        let avg_processing_time = total_processing_time.as_secs_f32() / sample_count as f32;
        let avg_memory_mb = total_memory_mb / sample_count as f32;

        // Calculate RTF
        let total_audio_duration: f32 = samples.iter().map(|s| s.audio.duration()).sum();
        let rtf = total_processing_time.as_secs_f32() / total_audio_duration;

        Ok(PerformanceBenchmark {
            model_id: model_id.to_string(),
            dataset,
            language,
            wer: average_word_error_rate,
            cer: average_char_error_rate,
            rtf,
            avg_processing_time,
            peak_memory_mb,
            avg_memory_mb,
            sample_count,
            sample_results,
        })
    }

    /// Calculate Word Error Rate and Character Error Rate
    #[must_use]
    /// calculate wer
    pub fn calculate_wer(&self, reference: &str, hypothesis: &str) -> WERResult {
        let ref_words: Vec<&str> = reference.split_whitespace().collect();
        let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

        // Calculate edit distance for words
        let (word_distance, word_ops) = self.edit_distance_with_ops(&ref_words, &hyp_words);

        // Calculate edit distance for characters
        let ref_chars: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();
        let hyp_chars: Vec<char> = hypothesis.chars().filter(|c| !c.is_whitespace()).collect();
        let (char_distance, _) = self.edit_distance_with_ops(&ref_chars, &hyp_chars);

        // Count operation types
        let mut substitutions = 0;
        let mut deletions = 0;
        let mut insertions = 0;

        for op in &word_ops {
            match op {
                'S' => substitutions += 1,
                'D' => deletions += 1,
                'I' => insertions += 1,
                _ => {} // 'C' for correct
            }
        }

        let wer = if ref_words.is_empty() {
            if hyp_words.is_empty() {
                0.0
            } else {
                1.0
            }
        } else {
            word_distance as f32 / ref_words.len() as f32
        };

        let cer = if ref_chars.is_empty() {
            if hyp_chars.is_empty() {
                0.0
            } else {
                1.0
            }
        } else {
            char_distance as f32 / ref_chars.len() as f32
        };

        WERResult {
            wer,
            cer,
            substitutions,
            deletions,
            insertions,
            reference_words: ref_words.len(),
            reference_chars: ref_chars.len(),
            alignment: Some(AlignmentResult {
                reference_aligned: ref_words.iter().map(|s| (*s).to_string()).collect(),
                hypothesis_aligned: hyp_words.iter().map(|s| (*s).to_string()).collect(),
                operations: word_ops,
            }),
        }
    }

    /// Calculate edit distance with operation tracking
    #[allow(clippy::many_single_char_names)]
    fn edit_distance_with_ops<T: PartialEq + Clone>(&self, a: &[T], b: &[T]) -> (usize, Vec<char>) {
        let m = a.len();
        let n = b.len();

        // DP table
        let mut dp = vec![vec![0; n + 1]; m + 1];
        let mut ops = vec![vec!['C'; n + 1]; m + 1];

        // Initialize base cases
        for i in 0..=m {
            dp[i][0] = i;
            if i > 0 {
                ops[i][0] = 'D';
            }
        }
        for j in 0..=n {
            dp[0][j] = j;
            if j > 0 {
                ops[0][j] = 'I';
            }
        }

        // Fill DP table
        for i in 1..=m {
            for j in 1..=n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                    ops[i][j] = 'C'; // Correct
                } else {
                    let substitute = dp[i - 1][j - 1] + 1;
                    let delete = dp[i - 1][j] + 1;
                    let insert = dp[i][j - 1] + 1;

                    if substitute <= delete && substitute <= insert {
                        dp[i][j] = substitute;
                        ops[i][j] = 'S'; // Substitution
                    } else if delete <= insert {
                        dp[i][j] = delete;
                        ops[i][j] = 'D'; // Deletion
                    } else {
                        dp[i][j] = insert;
                        ops[i][j] = 'I'; // Insertion
                    }
                }
            }
        }

        // Backtrack to get operations
        let mut operations = Vec::new();
        let mut i = m;
        let mut j = n;

        while i > 0 || j > 0 {
            operations.push(ops[i][j]);
            match ops[i][j] {
                'C' | 'S' => {
                    i -= 1;
                    j -= 1;
                }
                'D' => {
                    i -= 1;
                }
                'I' => {
                    j -= 1;
                }
                _ => break,
            }
        }

        operations.reverse();
        (dp[m][n], operations)
    }

    /// Calculate confidence score based on WER and CER
    fn calculate_confidence_score(&self, wer: f32, cer: f32) -> f32 {
        // Confidence estimation based on error rates
        // Lower error rates should result in higher confidence
        // Use a combination of WER and CER with heavier weight on WER

        let wer_score = (1.0 - wer.min(1.0)).max(0.0);
        let cer_score = (1.0 - cer.min(1.0)).max(0.0);

        // Weight WER more heavily than CER (70% WER, 30% CER)
        let combined_score = 0.7 * wer_score + 0.3 * cer_score;

        // Apply sigmoid-like function to make scores more realistic
        // Transform linear score to a more realistic confidence distribution
        let confidence = 1.0 / (1.0 + (-4.0 * (combined_score - 0.5)).exp());

        // Clamp to reasonable range (0.1 to 0.99)
        confidence.max(0.1).min(0.99)
    }

    /// Get current memory usage (real implementation using system APIs)
    fn get_memory_usage(&self) -> f32 {
        // Try to get memory usage from /proc/self/status on Linux
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(contents) = fs::read_to_string("/proc/self/status") {
                for line in contents.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(size_str) = line.split_whitespace().nth(1) {
                            if let Ok(size_kb) = size_str.parse::<f32>() {
                                return size_kb / 1024.0; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }

        // Try to get memory usage using sysinfo crate approach
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        {
            // For macOS and Windows, use a more sophisticated approach
            // This is a simplified implementation that could be enhanced with proper system calls
            if let Ok(output) = std::process::Command::new("ps")
                .args(["-o", "rss=", "-p"])
                .arg(std::process::id().to_string())
                .output()
            {
                if let Ok(memory_str) = String::from_utf8(output.stdout) {
                    if let Ok(memory_kb) = memory_str.trim().parse::<f32>() {
                        return memory_kb / 1024.0; // Convert KB to MB
                    }
                }
            }
        }

        // Fallback for unsupported platforms or if memory reading fails
        512.0 // MB (fallback value)
    }

    /// Generate benchmark report
    pub async fn generate_report(&self) -> BenchmarkReport {
        let results = self.results.read().await;

        let mut model_summaries = HashMap::new();
        let _dataset_summaries: HashMap<Dataset, ModelSummary> = HashMap::new();
        let _language_summaries: HashMap<LanguageCode, ModelSummary> = HashMap::new();

        for result in results.iter() {
            // Model summary
            let model_summary = model_summaries
                .entry(result.model_id.clone())
                .or_insert_with(|| ModelSummary {
                    model_id: result.model_id.clone(),
                    avg_wer: 0.0,
                    avg_cer: 0.0,
                    avg_rtf: 0.0,
                    avg_memory_mb: 0.0,
                    dataset_count: 0,
                });

            model_summary.avg_wer += result.wer;
            model_summary.avg_cer += result.cer;
            model_summary.avg_rtf += result.rtf;
            model_summary.avg_memory_mb += result.avg_memory_mb;
            model_summary.dataset_count += 1;
        }

        // Average the summaries
        for summary in model_summaries.values_mut() {
            summary.avg_wer /= summary.dataset_count as f32;
            summary.avg_cer /= summary.dataset_count as f32;
            summary.avg_rtf /= summary.dataset_count as f32;
            summary.avg_memory_mb /= summary.dataset_count as f32;
        }

        BenchmarkReport {
            total_benchmarks: results.len(),
            model_summaries: model_summaries.into_values().collect(),
            detailed_results: results.clone(),
            generation_time: std::time::SystemTime::now(),
        }
    }

    /// Get benchmark results
    pub async fn get_results(&self) -> Vec<PerformanceBenchmark> {
        self.results.read().await.clone()
    }
}

/// Model performance summary
#[derive(Debug, Clone)]
/// Model Summary
pub struct ModelSummary {
    /// Model identifier
    pub model_id: String,
    /// Average WER across all datasets
    pub avg_wer: f32,
    /// Average CER across all datasets
    pub avg_cer: f32,
    /// Average RTF across all datasets
    pub avg_rtf: f32,
    /// Average memory usage
    pub avg_memory_mb: f32,
    /// Number of datasets tested
    pub dataset_count: usize,
}

/// Comprehensive benchmark report
#[derive(Debug, Clone)]
/// Benchmark Report
pub struct BenchmarkReport {
    /// Total number of benchmarks run
    pub total_benchmarks: usize,
    /// Model performance summaries
    pub model_summaries: Vec<ModelSummary>,
    /// Detailed benchmark results
    pub detailed_results: Vec<PerformanceBenchmark>,
    /// Report generation time
    pub generation_time: std::time::SystemTime,
}

/// Accuracy validation framework for formal benchmarking
#[derive(Debug, Clone)]
/// Accuracy Validator
pub struct AccuracyValidator {
    /// Accuracy requirements
    pub requirements: Vec<AccuracyRequirement>,
}

/// Accuracy requirement specification
#[derive(Debug, Clone)]
/// Accuracy Requirement
pub struct AccuracyRequirement {
    /// Requirement identifier
    pub id: String,
    /// Description of the requirement
    pub description: String,
    /// Target dataset
    pub dataset: Dataset,
    /// Target language
    pub language: LanguageCode,
    /// Maximum acceptable WER
    pub max_wer: f32,
    /// Maximum acceptable CER
    pub max_cer: f32,
    /// Minimum phoneme alignment accuracy
    pub min_phoneme_accuracy: Option<f32>,
    /// Model identifier to test
    pub model_id: String,
}

/// Accuracy validation result
#[derive(Debug, Clone)]
/// Accuracy Validation Result
pub struct AccuracyValidationResult {
    /// Requirement that was tested
    pub requirement: AccuracyRequirement,
    /// Actual WER achieved
    pub actual_wer: f32,
    /// Actual CER achieved
    pub actual_cer: f32,
    /// Actual phoneme alignment accuracy
    pub actual_phoneme_accuracy: Option<f32>,
    /// Whether the requirement was met
    pub passed: bool,
    /// Detailed failure reason if failed
    pub failure_reason: Option<String>,
    /// Number of samples tested
    pub sample_count: usize,
}

/// Comprehensive accuracy validation report
#[derive(Debug, Clone)]
/// Accuracy Validation Report
pub struct AccuracyValidationReport {
    /// All validation results
    pub results: Vec<AccuracyValidationResult>,
    /// Overall pass/fail status
    pub overall_passed: bool,
    /// Total requirements tested
    pub total_requirements: usize,
    /// Number of requirements passed
    pub passed_requirements: usize,
    /// Validation timestamp
    pub validation_time: std::time::SystemTime,
}

impl AccuracyValidator {
    /// Create a new accuracy validator with standard VoiRS requirements
    pub fn new_standard() -> Self {
        let requirements = vec![
            AccuracyRequirement {
                id: "librispeech_clean".to_string(),
                description: "WER < 5% on LibriSpeech test-clean".to_string(),
                dataset: Dataset::LibriSpeech,
                language: LanguageCode::EnUs,
                max_wer: 0.05,
                max_cer: 0.02,
                min_phoneme_accuracy: Some(0.90),
                model_id: "whisper_base".to_string(),
            },
            AccuracyRequirement {
                id: "commonvoice_en".to_string(),
                description: "WER < 10% on CommonVoice en".to_string(),
                dataset: Dataset::CommonVoice,
                language: LanguageCode::EnUs,
                max_wer: 0.10,
                max_cer: 0.05,
                min_phoneme_accuracy: Some(0.85),
                model_id: "whisper_base".to_string(),
            },
            AccuracyRequirement {
                id: "vctk_alignment".to_string(),
                description: "Phoneme alignment accuracy > 90% on VCTK".to_string(),
                dataset: Dataset::VCTK,
                language: LanguageCode::EnGb,
                max_wer: 0.15,
                max_cer: 0.08,
                min_phoneme_accuracy: Some(0.90),
                model_id: "whisper_base".to_string(),
            },
        ];

        Self { requirements }
    }

    /// Create a custom accuracy validator
    pub fn new_custom(requirements: Vec<AccuracyRequirement>) -> Self {
        Self { requirements }
    }

    /// Add a new accuracy requirement
    pub fn add_requirement(&mut self, requirement: AccuracyRequirement) {
        self.requirements.push(requirement);
    }

    /// Validate accuracy against all requirements
    pub async fn validate_accuracy(
        &self,
        benchmark_suite: &ASRBenchmarkingSuite,
    ) -> Result<AccuracyValidationReport, RecognitionError> {
        let mut results = Vec::new();
        let mut passed_count = 0;

        for requirement in &self.requirements {
            let result = self
                .validate_single_requirement(requirement, benchmark_suite)
                .await?;
            if result.passed {
                passed_count += 1;
            }
            results.push(result);
        }

        let overall_passed = passed_count == self.requirements.len();

        Ok(AccuracyValidationReport {
            results,
            overall_passed,
            total_requirements: self.requirements.len(),
            passed_requirements: passed_count,
            validation_time: std::time::SystemTime::now(),
        })
    }

    /// Validate a single accuracy requirement
    async fn validate_single_requirement(
        &self,
        requirement: &AccuracyRequirement,
        benchmark_suite: &ASRBenchmarkingSuite,
    ) -> Result<AccuracyValidationResult, RecognitionError> {
        // Get benchmark results for this requirement
        let benchmark_results = benchmark_suite.get_results().await;

        // Find matching benchmark result
        let matching_result = benchmark_results.iter().find(|r| {
            r.model_id == requirement.model_id
                && r.dataset == requirement.dataset
                && r.language == requirement.language
        });

        if let Some(result) = matching_result {
            // Check WER requirement
            let wer_passed = result.wer <= requirement.max_wer;
            let cer_passed = result.cer <= requirement.max_cer;

            // Check phoneme alignment accuracy if specified
            let phoneme_passed = if let Some(min_accuracy) = requirement.min_phoneme_accuracy {
                // For now, we'll use a mock implementation that estimates phoneme accuracy
                // In a real implementation, this would use actual phoneme alignment results
                let estimated_phoneme_accuracy = self.estimate_phoneme_accuracy(result.wer);
                estimated_phoneme_accuracy >= min_accuracy
            } else {
                true
            };

            let passed = wer_passed && cer_passed && phoneme_passed;

            let failure_reason = if !passed {
                let mut reasons = Vec::new();
                if !wer_passed {
                    reasons.push(format!(
                        "WER {:.1}% > {:.1}%",
                        result.wer * 100.0,
                        requirement.max_wer * 100.0
                    ));
                }
                if !cer_passed {
                    reasons.push(format!(
                        "CER {:.1}% > {:.1}%",
                        result.cer * 100.0,
                        requirement.max_cer * 100.0
                    ));
                }
                if !phoneme_passed {
                    if let Some(min_accuracy) = requirement.min_phoneme_accuracy {
                        let estimated = self.estimate_phoneme_accuracy(result.wer);
                        reasons.push(format!(
                            "Phoneme accuracy {:.1}% < {:.1}%",
                            estimated * 100.0,
                            min_accuracy * 100.0
                        ));
                    }
                }
                Some(reasons.join(", "))
            } else {
                None
            };

            Ok(AccuracyValidationResult {
                requirement: requirement.clone(),
                actual_wer: result.wer,
                actual_cer: result.cer,
                actual_phoneme_accuracy: requirement
                    .min_phoneme_accuracy
                    .map(|_| self.estimate_phoneme_accuracy(result.wer)),
                passed,
                failure_reason,
                sample_count: result.sample_count,
            })
        } else {
            // No matching benchmark result found
            Ok(AccuracyValidationResult {
                requirement: requirement.clone(),
                actual_wer: 1.0, // Worst possible score
                actual_cer: 1.0,
                actual_phoneme_accuracy: None,
                passed: false,
                failure_reason: Some("No benchmark results found for this requirement".to_string()),
                sample_count: 0,
            })
        }
    }

    /// Estimate phoneme accuracy based on WER (simple heuristic)
    fn estimate_phoneme_accuracy(&self, wer: f32) -> f32 {
        // Simple heuristic: phoneme accuracy is typically higher than word accuracy
        // This is a rough approximation - in real implementation, use actual phoneme alignment
        let phoneme_accuracy = 1.0 - (wer * 0.7); // Assume phoneme errors are ~70% of word errors
        phoneme_accuracy.max(0.0).min(1.0)
    }

    /// Generate a summary report
    pub fn generate_summary_report(&self, report: &AccuracyValidationReport) -> String {
        let mut summary = String::new();

        summary.push_str("=== VoiRS Accuracy Validation Report ===\n");
        summary.push_str(&format!(
            "Overall Status: {}\n",
            if report.overall_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        ));
        summary.push_str(&format!(
            "Requirements Passed: {}/{}\n",
            report.passed_requirements, report.total_requirements
        ));
        summary.push_str(&format!(
            "Validation Time: {:?}\n\n",
            report.validation_time
        ));

        for result in &report.results {
            summary.push_str(&format!(
                "Requirement: {}\n",
                result.requirement.description
            ));
            summary.push_str(&format!(
                "  Status: {}\n",
                if result.passed { "PASSED" } else { "FAILED" }
            ));
            summary.push_str(&format!(
                "  WER: {:.2}% (max: {:.2}%)\n",
                result.actual_wer * 100.0,
                result.requirement.max_wer * 100.0
            ));
            summary.push_str(&format!(
                "  CER: {:.2}% (max: {:.2}%)\n",
                result.actual_cer * 100.0,
                result.requirement.max_cer * 100.0
            ));

            if let Some(phoneme_accuracy) = result.actual_phoneme_accuracy {
                if let Some(min_accuracy) = result.requirement.min_phoneme_accuracy {
                    summary.push_str(&format!(
                        "  Phoneme Accuracy: {:.1}% (min: {:.1}%)\n",
                        phoneme_accuracy * 100.0,
                        min_accuracy * 100.0
                    ));
                }
            }

            summary.push_str(&format!("  Samples: {}\n", result.sample_count));

            if let Some(failure_reason) = &result.failure_reason {
                summary.push_str(&format!("  Failure: {}\n", failure_reason));
            }

            summary.push_str("\n");
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wer_calculation() {
        let suite = ASRBenchmarkingSuite {
            config: BenchmarkingConfig::default(),
            models: Arc::new(RwLock::new(HashMap::new())),
            datasets: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(Vec::new())),
        };

        // Perfect match
        let result = suite.calculate_wer("hello world", "hello world");
        assert_eq!(result.wer, 0.0);
        assert_eq!(result.cer, 0.0);

        // One substitution
        let result = suite.calculate_wer("hello world", "hello earth");
        assert_eq!(result.wer, 0.5); // 1 error out of 2 words
        assert_eq!(result.substitutions, 1);

        // One deletion
        let result = suite.calculate_wer("hello world", "hello");
        assert_eq!(result.wer, 0.5); // 1 error out of 2 words
        assert_eq!(result.deletions, 1);

        // One insertion
        let result = suite.calculate_wer("hello world", "hello beautiful world");
        assert_eq!(result.wer, 0.5); // 1 error out of 2 words
        assert_eq!(result.insertions, 1);
    }

    #[test]
    fn test_edit_distance() {
        let suite = ASRBenchmarkingSuite {
            config: BenchmarkingConfig::default(),
            models: Arc::new(RwLock::new(HashMap::new())),
            datasets: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(Vec::new())),
        };

        let a = vec!["hello", "world"];
        let b = vec!["hello", "earth"];

        let (distance, ops) = suite.edit_distance_with_ops(&a, &b);
        assert_eq!(distance, 1);
        assert_eq!(ops, vec!['C', 'S']); // Correct, Substitution
    }

    #[tokio::test]
    async fn test_benchmark_suite_creation() {
        let config = BenchmarkingConfig::default();
        let suite = ASRBenchmarkingSuite::new(config).await.unwrap();

        let datasets = suite.datasets.read().await;
        assert!(datasets.contains_key(&Dataset::LibriSpeech));
        assert!(datasets.contains_key(&Dataset::CommonVoice));
        assert!(datasets.contains_key(&Dataset::VCTK));
    }

    #[test]
    fn test_accuracy_validator_creation() {
        let validator = AccuracyValidator::new_standard();
        assert_eq!(validator.requirements.len(), 3);

        // Check that the standard requirements are present
        let requirement_ids: Vec<_> = validator.requirements.iter().map(|r| &r.id).collect();
        assert!(requirement_ids.contains(&&"librispeech_clean".to_string()));
        assert!(requirement_ids.contains(&&"commonvoice_en".to_string()));
        assert!(requirement_ids.contains(&&"vctk_alignment".to_string()));
    }

    #[test]
    fn test_accuracy_requirement_configuration() {
        let validator = AccuracyValidator::new_standard();
        let librispeech_req = validator
            .requirements
            .iter()
            .find(|r| r.id == "librispeech_clean")
            .unwrap();

        assert_eq!(librispeech_req.max_wer, 0.05);
        assert_eq!(librispeech_req.max_cer, 0.02);
        assert_eq!(librispeech_req.min_phoneme_accuracy, Some(0.90));
        assert_eq!(librispeech_req.dataset, Dataset::LibriSpeech);
        assert_eq!(librispeech_req.language, LanguageCode::EnUs);
    }

    #[test]
    fn test_phoneme_accuracy_estimation() {
        let validator = AccuracyValidator::new_standard();

        // Test various WER values
        assert_eq!(validator.estimate_phoneme_accuracy(0.0), 1.0);
        assert_eq!(validator.estimate_phoneme_accuracy(0.1), 0.93);
        assert_eq!(validator.estimate_phoneme_accuracy(0.2), 0.86);
        assert_eq!(validator.estimate_phoneme_accuracy(1.0), 0.3);

        // Test boundary values
        assert!(validator.estimate_phoneme_accuracy(2.0) >= 0.0);
        assert!(validator.estimate_phoneme_accuracy(0.0) <= 1.0);
    }

    #[test]
    fn test_custom_accuracy_validator() {
        let custom_requirement = AccuracyRequirement {
            id: "custom_test".to_string(),
            description: "Custom test requirement".to_string(),
            dataset: Dataset::Custom("test_dataset".to_string()),
            language: LanguageCode::EnUs,
            max_wer: 0.15,
            max_cer: 0.08,
            min_phoneme_accuracy: Some(0.85),
            model_id: "custom_model".to_string(),
        };

        let validator = AccuracyValidator::new_custom(vec![custom_requirement]);
        assert_eq!(validator.requirements.len(), 1);
        assert_eq!(validator.requirements[0].id, "custom_test");
        assert_eq!(validator.requirements[0].max_wer, 0.15);
    }

    #[test]
    fn test_accuracy_validator_add_requirement() {
        let mut validator = AccuracyValidator::new_standard();
        let initial_count = validator.requirements.len();

        let new_requirement = AccuracyRequirement {
            id: "additional_test".to_string(),
            description: "Additional test requirement".to_string(),
            dataset: Dataset::VCTK,
            language: LanguageCode::EnGb,
            max_wer: 0.12,
            max_cer: 0.06,
            min_phoneme_accuracy: Some(0.88),
            model_id: "whisper_small".to_string(),
        };

        validator.add_requirement(new_requirement);
        assert_eq!(validator.requirements.len(), initial_count + 1);
        assert_eq!(validator.requirements.last().unwrap().id, "additional_test");
    }

    #[tokio::test]
    async fn test_accuracy_validation_no_results() {
        let validator = AccuracyValidator::new_standard();
        let config = BenchmarkingConfig::default();
        let suite = ASRBenchmarkingSuite::new(config).await.unwrap();

        // Test with empty benchmark suite (no results)
        let report = validator.validate_accuracy(&suite).await.unwrap();

        assert!(!report.overall_passed);
        assert_eq!(report.total_requirements, 3);
        assert_eq!(report.passed_requirements, 0);

        // All results should fail due to no benchmark data
        for result in &report.results {
            assert!(!result.passed);
            assert!(result.failure_reason.is_some());
            assert!(result
                .failure_reason
                .as_ref()
                .unwrap()
                .contains("No benchmark results found"));
        }
    }

    #[test]
    fn test_accuracy_validation_report_summary() {
        let validator = AccuracyValidator::new_standard();

        // Create a mock validation report
        let mock_result = AccuracyValidationResult {
            requirement: validator.requirements[0].clone(),
            actual_wer: 0.03,
            actual_cer: 0.015,
            actual_phoneme_accuracy: Some(0.92),
            passed: true,
            failure_reason: None,
            sample_count: 100,
        };

        let report = AccuracyValidationReport {
            results: vec![mock_result],
            overall_passed: true,
            total_requirements: 1,
            passed_requirements: 1,
            validation_time: std::time::SystemTime::now(),
        };

        let summary = validator.generate_summary_report(&report);

        assert!(summary.contains("Overall Status: PASSED"));
        assert!(summary.contains("Requirements Passed: 1/1"));
        assert!(summary.contains("WER: 3.00%"));
        assert!(summary.contains("CER: 1.50%"));
        assert!(summary.contains("Phoneme Accuracy: 92.0%"));
        assert!(summary.contains("Samples: 100"));
    }

    #[test]
    fn test_confidence_score_calculation() {
        let suite = ASRBenchmarkingSuite {
            config: BenchmarkingConfig::default(),
            models: Arc::new(RwLock::new(HashMap::new())),
            datasets: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(Vec::new())),
        };

        // Perfect transcription should have high confidence
        let perfect_confidence = suite.calculate_confidence_score(0.0, 0.0);
        assert!(perfect_confidence > 0.8);

        // High error rates should have low confidence
        let poor_confidence = suite.calculate_confidence_score(1.0, 1.0);
        assert!(poor_confidence < 0.3);

        // Medium error rates should have medium confidence
        let medium_confidence = suite.calculate_confidence_score(0.3, 0.3);
        assert!(medium_confidence > 0.3 && medium_confidence < 0.8);

        // Confidence should be bounded between 0.1 and 0.99
        let extreme_confidence = suite.calculate_confidence_score(10.0, 10.0);
        assert!(extreme_confidence >= 0.1);
    }
}
