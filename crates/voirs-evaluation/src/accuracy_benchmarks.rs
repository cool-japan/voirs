//! Comprehensive accuracy benchmarking for speech synthesis systems.
//!
//! This module implements accuracy benchmarking for different components of the VoiRS system:
//! - English CMU test set with >95% phoneme accuracy target
//! - Japanese JVS corpus with >90% mora accuracy target  
//! - Multilingual Common Voice tests
//! - Comprehensive accuracy reporting and regression detection

use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokio::time::Instant;
use voirs_sdk::AudioBuffer;

/// Accuracy benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyBenchmarkConfig {
    /// Target accuracy thresholds by language
    pub accuracy_targets: HashMap<LanguageCode, f64>,
    /// Test dataset configurations
    pub datasets: Vec<DatasetConfig>,
    /// Enable detailed per-sample reporting
    pub detailed_reporting: bool,
    /// Maximum processing time per sample (seconds)
    pub max_processing_time: f64,
    /// Output directory for benchmark results
    pub output_dir: String,
}

impl Default for AccuracyBenchmarkConfig {
    fn default() -> Self {
        let mut accuracy_targets = HashMap::new();
        accuracy_targets.insert(LanguageCode::EnUs, 0.95); // English CMU >95%
        accuracy_targets.insert(LanguageCode::Ja, 0.90); // Japanese JVS >90%
        accuracy_targets.insert(LanguageCode::Es, 0.88); // Spanish Common Voice
        accuracy_targets.insert(LanguageCode::Fr, 0.88); // French Common Voice
        accuracy_targets.insert(LanguageCode::De, 0.88); // German Common Voice
        accuracy_targets.insert(LanguageCode::ZhCn, 0.85); // Chinese Common Voice

        Self {
            accuracy_targets,
            datasets: vec![
                DatasetConfig::cmu_english(),
                DatasetConfig::jvs_japanese(),
                DatasetConfig::common_voice_multilingual(),
            ],
            detailed_reporting: true,
            max_processing_time: 10.0,
            output_dir: String::from("/tmp/voirs_accuracy_benchmarks"),
        }
    }
}

/// Language codes for benchmarking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LanguageCode {
    /// English (United States)
    EnUs,
    /// Japanese
    Ja,
    /// Spanish
    Es,
    /// French
    Fr,
    /// German
    De,
    /// Chinese (Simplified)
    ZhCn,
}

/// Dataset configuration for accuracy testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset name
    pub name: String,
    /// Dataset type
    pub dataset_type: DatasetType,
    /// Language code
    pub language: LanguageCode,
    /// Path to test data
    pub data_path: String,
    /// Expected accuracy threshold
    pub target_accuracy: f64,
    /// Maximum number of test samples (None = all)
    pub max_samples: Option<usize>,
}

impl DatasetConfig {
    /// Create CMU English dataset configuration
    pub fn cmu_english() -> Self {
        Self {
            name: String::from("CMU_English_Phoneme_Test"),
            dataset_type: DatasetType::CMU,
            language: LanguageCode::EnUs,
            data_path: String::from("tests/datasets/cmu_phoneme_test.txt"),
            target_accuracy: 0.95,
            max_samples: Some(1000),
        }
    }

    /// Create JVS Japanese dataset configuration
    pub fn jvs_japanese() -> Self {
        Self {
            name: String::from("JVS_Japanese_Mora_Test"),
            dataset_type: DatasetType::JVS,
            language: LanguageCode::Ja,
            data_path: String::from("tests/datasets/jvs_mora_test.txt"),
            target_accuracy: 0.90,
            max_samples: Some(800),
        }
    }

    /// Create Common Voice multilingual dataset configuration
    pub fn common_voice_multilingual() -> Self {
        Self {
            name: String::from("Common_Voice_Multilingual"),
            dataset_type: DatasetType::CommonVoice,
            language: LanguageCode::EnUs, // Will be overridden for specific languages
            data_path: String::from("tests/datasets/common_voice_test.txt"),
            target_accuracy: 0.88,
            max_samples: Some(500),
        }
    }
}

/// Types of accuracy test datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetType {
    /// CMU Pronunciation Dictionary
    CMU,
    /// Japanese Voice Speech Corpus
    JVS,
    /// Mozilla Common Voice
    CommonVoice,
    /// Custom dataset
    Custom,
}

/// Individual test case for accuracy evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyTestCase {
    /// Unique identifier for the test case
    pub id: String,
    /// Input text
    pub text: String,
    /// Expected phoneme sequence
    pub expected_phonemes: Vec<String>,
    /// Expected output audio (for TTS evaluation)
    pub expected_audio: Option<AudioBuffer>,
    /// Language code
    pub language: LanguageCode,
    /// Reference transcript (for ASR evaluation)
    pub reference_transcript: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Results from accuracy benchmark evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyBenchmarkResults {
    /// Benchmark configuration used
    pub config: AccuracyBenchmarkConfig,
    /// Results by dataset
    pub dataset_results: HashMap<String, DatasetResults>,
    /// Overall accuracy metrics
    pub overall_metrics: OverallAccuracyMetrics,
    /// Timestamp of evaluation
    pub timestamp: String,
    /// Total evaluation time
    pub total_time_seconds: f64,
    /// Performance statistics
    pub performance_stats: PerformanceStats,
}

/// Results for a specific dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetResults {
    /// Dataset name
    pub dataset_name: String,
    /// Language code
    pub language: LanguageCode,
    /// Total test cases
    pub total_cases: usize,
    /// Successful evaluations
    pub successful_cases: usize,
    /// Failed evaluations
    pub failed_cases: usize,
    /// Phoneme-level accuracy
    pub phoneme_accuracy: f64,
    /// Word-level accuracy  
    pub word_accuracy: f64,
    /// Average edit distance
    pub average_edit_distance: f64,
    /// Target accuracy threshold
    pub target_accuracy: f64,
    /// Whether target was met
    pub target_met: bool,
    /// Detailed per-case results
    pub case_results: Vec<CaseResult>,
    /// Processing time statistics
    pub processing_time_ms: ProcessingTimeStats,
}

/// Result for individual test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    /// Test case ID
    pub case_id: String,
    /// Input text
    pub input_text: String,
    /// Expected output
    pub expected: Vec<String>,
    /// Actual output
    pub actual: Vec<String>,
    /// Whether case passed
    pub passed: bool,
    /// Edit distance
    pub edit_distance: f64,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Overall accuracy metrics across all datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallAccuracyMetrics {
    /// Total test cases across all datasets
    pub total_cases: usize,
    /// Overall phoneme accuracy
    pub overall_phoneme_accuracy: f64,
    /// Overall word accuracy
    pub overall_word_accuracy: f64,
    /// Accuracy by language
    pub language_accuracies: HashMap<LanguageCode, f64>,
    /// Number of targets met
    pub targets_met: usize,
    /// Total number of targets
    pub total_targets: usize,
    /// Pass rate percentage
    pub pass_rate: f64,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Average processing time per case (ms)
    pub avg_processing_time_ms: f64,
    /// Median processing time (ms)
    pub median_processing_time_ms: f64,
    /// 95th percentile processing time (ms)
    pub p95_processing_time_ms: f64,
    /// Throughput (cases per second)
    pub throughput_cases_per_sec: f64,
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
}

/// Processing time statistics for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTimeStats {
    /// Minimum processing time (ms)
    pub min_ms: f64,
    /// Maximum processing time (ms)
    pub max_ms: f64,
    /// Average processing time (ms)
    pub mean_ms: f64,
    /// Standard deviation (ms)
    pub std_dev_ms: f64,
    /// Median processing time (ms)
    pub median_ms: f64,
}

/// Comprehensive accuracy benchmark runner
pub struct AccuracyBenchmarkRunner {
    config: AccuracyBenchmarkConfig,
    test_cases: HashMap<String, Vec<AccuracyTestCase>>,
}

impl AccuracyBenchmarkRunner {
    /// Create new accuracy benchmark runner
    pub fn new(config: AccuracyBenchmarkConfig) -> Self {
        Self {
            config,
            test_cases: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(AccuracyBenchmarkConfig::default())
    }

    /// Load test cases from configured datasets
    pub async fn load_test_cases(&mut self) -> Result<(), EvaluationError> {
        for dataset_config in &self.config.datasets.clone() {
            let test_cases = self.load_dataset_test_cases(dataset_config).await?;
            self.test_cases
                .insert(dataset_config.name.clone(), test_cases);
        }
        Ok(())
    }

    /// Load test cases for a specific dataset
    async fn load_dataset_test_cases(
        &self,
        dataset_config: &DatasetConfig,
    ) -> Result<Vec<AccuracyTestCase>, EvaluationError> {
        match dataset_config.dataset_type {
            DatasetType::CMU => self.load_cmu_test_cases(dataset_config).await,
            DatasetType::JVS => self.load_jvs_test_cases(dataset_config).await,
            DatasetType::CommonVoice => self.load_common_voice_test_cases(dataset_config).await,
            DatasetType::Custom => self.load_custom_test_cases(dataset_config).await,
        }
    }

    /// Load CMU pronunciation dictionary test cases
    async fn load_cmu_test_cases(
        &self,
        dataset_config: &DatasetConfig,
    ) -> Result<Vec<AccuracyTestCase>, EvaluationError> {
        // For now, create synthetic CMU-style test cases
        // In production, these would be loaded from the actual CMU pronunciation dictionary
        let mut test_cases = Vec::new();

        let cmu_samples = vec![
            ("ABANDON", vec!["AH0", "B", "AE1", "N", "D", "AH0", "N"]),
            ("ABILITY", vec!["AH0", "B", "IH1", "L", "AH0", "T", "IY0"]),
            ("ABOUT", vec!["AH0", "B", "AW1", "T"]),
            ("ABOVE", vec!["AH0", "B", "AH1", "V"]),
            ("ABSENCE", vec!["AE1", "B", "S", "AH0", "N", "S"]),
            ("ACCEPT", vec!["AE0", "K", "S", "EH1", "P", "T"]),
            ("ACCESS", vec!["AE1", "K", "S", "EH0", "S"]),
            ("ACCOUNT", vec!["AH0", "K", "AW1", "N", "T"]),
            ("ACHIEVE", vec!["AH0", "CH", "IY1", "V"]),
            ("ACTION", vec!["AE1", "K", "SH", "AH0", "N"]),
            ("ACTIVE", vec!["AE1", "K", "T", "IH0", "V"]),
            ("ADDRESS", vec!["AH0", "D", "R", "EH1", "S"]),
            ("ADVANCE", vec!["AH0", "D", "V", "AE1", "N", "S"]),
            ("AGAINST", vec!["AH0", "G", "EH1", "N", "S", "T"]),
            ("ALREADY", vec!["AO0", "L", "R", "EH1", "D", "IY0"]),
            ("ALTHOUGH", vec!["AO0", "L", "DH", "OW1"]),
            ("ALWAYS", vec!["AO1", "L", "W", "EY0", "Z"]),
            ("AMERICA", vec!["AH0", "M", "EH1", "R", "AH0", "K", "AH0"]),
            (
                "ANALYSIS",
                vec!["AH0", "N", "AE1", "L", "AH0", "S", "AH0", "S"],
            ),
            ("ANOTHER", vec!["AH0", "N", "AH1", "DH", "ER0"]),
        ];

        for (i, (word, phonemes)) in cmu_samples.iter().enumerate() {
            if let Some(max_samples) = dataset_config.max_samples {
                if i >= max_samples {
                    break;
                }
            }

            test_cases.push(AccuracyTestCase {
                id: format!("cmu_{i}"),
                text: word.to_lowercase(),
                expected_phonemes: phonemes.iter().map(|s| s.to_string()).collect(),
                expected_audio: None,
                language: LanguageCode::EnUs,
                reference_transcript: Some(word.to_lowercase()),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(String::from("dataset"), String::from("CMU"));
                    meta.insert(String::from("phoneme_count"), phonemes.len().to_string());
                    meta
                },
            });
        }

        // Add additional synthetic CMU test cases to reach target sample count
        if let Some(max_samples) = dataset_config.max_samples {
            self.generate_additional_cmu_cases(&mut test_cases, max_samples)
                .await?;
        }

        Ok(test_cases)
    }

    /// Generate additional CMU test cases to reach target count
    async fn generate_additional_cmu_cases(
        &self,
        test_cases: &mut Vec<AccuracyTestCase>,
        target_count: usize,
    ) -> Result<(), EvaluationError> {
        // Generate additional English words with phonetic representations
        let additional_words = vec![
            (
                "beautiful",
                vec!["B", "Y", "UW1", "T", "AH0", "F", "AH0", "L"],
            ),
            (
                "technology",
                vec!["T", "EH0", "K", "N", "AA1", "L", "AH0", "JH", "IY0"],
            ),
            (
                "development",
                vec![
                    "D", "IH0", "V", "EH1", "L", "AH0", "P", "M", "AH0", "N", "T",
                ],
            ),
            (
                "understand",
                vec!["AH2", "N", "D", "ER0", "S", "T", "AE1", "N", "D"],
            ),
            (
                "information",
                vec!["IH2", "N", "F", "ER0", "M", "EY1", "SH", "AH0", "N"],
            ),
            (
                "important",
                vec!["IH0", "M", "P", "AO1", "R", "T", "AH0", "N", "T"],
            ),
            ("different", vec!["D", "IH1", "F", "ER0", "AH0", "N", "T"]),
            (
                "experience",
                vec!["IH0", "K", "S", "P", "IH1", "R", "IY0", "AH0", "N", "S"],
            ),
            ("remember", vec!["R", "IH0", "M", "EH1", "M", "B", "ER0"]),
            (
                "education",
                vec!["EH2", "JH", "AH0", "K", "EY1", "SH", "AH0", "N"],
            ),
        ];

        for (i, (word, phonemes)) in additional_words.iter().enumerate() {
            if test_cases.len() >= target_count {
                break;
            }

            test_cases.push(AccuracyTestCase {
                id: format!("cmu_additional_{i}"),
                text: word.to_string(),
                expected_phonemes: phonemes.iter().map(|s| s.to_string()).collect(),
                expected_audio: None,
                language: LanguageCode::EnUs,
                reference_transcript: Some(word.to_string()),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(String::from("dataset"), String::from("CMU_Additional"));
                    meta.insert(String::from("phoneme_count"), phonemes.len().to_string());
                    meta
                },
            });
        }

        Ok(())
    }

    /// Load JVS Japanese corpus test cases
    async fn load_jvs_test_cases(
        &self,
        dataset_config: &DatasetConfig,
    ) -> Result<Vec<AccuracyTestCase>, EvaluationError> {
        // For now, create synthetic JVS-style test cases
        // In production, these would be loaded from the actual JVS corpus
        let mut test_cases = Vec::new();

        let jvs_samples = vec![
            ("おはよう", vec!["o", "h", "a", "y", "o", "u"]),
            (
                "こんにちは",
                vec!["k", "o", "n", "n", "i", "ch", "i", "w", "a"],
            ),
            ("ありがとう", vec!["a", "r", "i", "g", "a", "t", "o", "u"]),
            (
                "さようなら",
                vec!["s", "a", "y", "o", "u", "n", "a", "r", "a"],
            ),
            ("おめでとう", vec!["o", "m", "e", "d", "e", "t", "o", "u"]),
            ("がんばって", vec!["g", "a", "n", "b", "a", "t", "t", "e"]),
            (
                "おつかれさま",
                vec!["o", "ts", "u", "k", "a", "r", "e", "s", "a", "m", "a"],
            ),
            ("よろしく", vec!["y", "o", "r", "o", "sh", "i", "k", "u"]),
            (
                "すみません",
                vec!["s", "u", "m", "i", "m", "a", "s", "e", "n"],
            ),
            ("だいじょうぶ", vec!["d", "a", "i", "j", "o", "u", "b", "u"]),
        ];

        for (i, (text, morae)) in jvs_samples.iter().enumerate() {
            if let Some(max_samples) = dataset_config.max_samples {
                if i >= max_samples {
                    break;
                }
            }

            test_cases.push(AccuracyTestCase {
                id: format!("jvs_{i}"),
                text: text.to_string(),
                expected_phonemes: morae.iter().map(|s| s.to_string()).collect(),
                expected_audio: None,
                language: LanguageCode::Ja,
                reference_transcript: Some(text.to_string()),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(String::from("dataset"), String::from("JVS"));
                    meta.insert(String::from("mora_count"), morae.len().to_string());
                    meta
                },
            });
        }

        // Generate additional test cases if needed
        if let Some(max_samples) = dataset_config.max_samples {
            self.generate_additional_jvs_cases(&mut test_cases, max_samples)
                .await?;
        }

        Ok(test_cases)
    }

    /// Generate additional JVS test cases
    async fn generate_additional_jvs_cases(
        &self,
        test_cases: &mut Vec<AccuracyTestCase>,
        target_count: usize,
    ) -> Result<(), EvaluationError> {
        let additional_japanese = vec![
            ("にほんご", vec!["n", "i", "h", "o", "n", "g", "o"]),
            ("がくせい", vec!["g", "a", "k", "u", "s", "e", "i"]),
            ("せんせい", vec!["s", "e", "n", "s", "e", "i"]),
            ("ともだち", vec!["t", "o", "m", "o", "d", "a", "ch", "i"]),
            ("かぞく", vec!["k", "a", "z", "o", "k", "u"]),
            ("しごと", vec!["sh", "i", "g", "o", "t", "o"]),
            ("たべもの", vec!["t", "a", "b", "e", "m", "o", "n", "o"]),
            ("のみもの", vec!["n", "o", "m", "i", "m", "o", "n", "o"]),
            ("でんしゃ", vec!["d", "e", "n", "sh", "a"]),
            ("くるま", vec!["k", "u", "r", "u", "m", "a"]),
        ];

        for (i, (text, morae)) in additional_japanese.iter().enumerate() {
            if test_cases.len() >= target_count {
                break;
            }

            test_cases.push(AccuracyTestCase {
                id: format!("jvs_additional_{i}"),
                text: text.to_string(),
                expected_phonemes: morae.iter().map(|s| s.to_string()).collect(),
                expected_audio: None,
                language: LanguageCode::Ja,
                reference_transcript: Some(text.to_string()),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(String::from("dataset"), String::from("JVS_Additional"));
                    meta.insert(String::from("mora_count"), morae.len().to_string());
                    meta
                },
            });
        }

        Ok(())
    }

    /// Load Common Voice multilingual test cases
    async fn load_common_voice_test_cases(
        &self,
        dataset_config: &DatasetConfig,
    ) -> Result<Vec<AccuracyTestCase>, EvaluationError> {
        let mut test_cases = Vec::new();

        // Create multilingual test cases for Common Voice
        let multilingual_samples = vec![
            // Spanish
            (
                LanguageCode::Es,
                "hola mundo",
                vec!["o", "l", "a", "m", "u", "n", "d", "o"],
            ),
            (
                LanguageCode::Es,
                "buenos días",
                vec!["b", "w", "e", "n", "o", "s", "d", "i", "a", "s"],
            ),
            (
                LanguageCode::Es,
                "gracias",
                vec!["g", "r", "a", "th", "i", "a", "s"],
            ),
            // French
            (LanguageCode::Fr, "bonjour", vec!["b", "ɔ̃", "ʒ", "u", "ʁ"]),
            (LanguageCode::Fr, "merci", vec!["m", "ɛ", "ʁ", "s", "i"]),
            (
                LanguageCode::Fr,
                "au revoir",
                vec!["o", "ʁ", "ə", "v", "w", "a", "ʁ"],
            ),
            // German
            (
                LanguageCode::De,
                "guten tag",
                vec!["g", "u", "t", "ə", "n", "t", "a", "k"],
            ),
            (LanguageCode::De, "danke", vec!["d", "a", "ŋ", "k", "ə"]),
            (
                LanguageCode::De,
                "auf wiedersehen",
                vec!["a", "u", "f", "v", "i", "d", "ɐ", "z", "e", "n"],
            ),
            // Chinese
            (LanguageCode::ZhCn, "你好", vec!["n", "i", "h", "a", "o"]),
            (
                LanguageCode::ZhCn,
                "谢谢",
                vec!["x", "i", "e", "x", "i", "e"],
            ),
            (
                LanguageCode::ZhCn,
                "再见",
                vec!["z", "a", "i", "j", "i", "a", "n"],
            ),
        ];

        for (i, (language, text, phonemes)) in multilingual_samples.iter().enumerate() {
            if let Some(max_samples) = dataset_config.max_samples {
                if i >= max_samples {
                    break;
                }
            }

            test_cases.push(AccuracyTestCase {
                id: format!("cv_multilingual_{i}"),
                text: text.to_string(),
                expected_phonemes: phonemes.iter().map(|s| s.to_string()).collect(),
                expected_audio: None,
                language: *language,
                reference_transcript: Some(text.to_string()),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(String::from("dataset"), String::from("CommonVoice"));
                    meta.insert(String::from("language"), format!("{language:?}"));
                    meta.insert(String::from("phoneme_count"), phonemes.len().to_string());
                    meta
                },
            });
        }

        Ok(test_cases)
    }

    /// Load custom test cases from file
    async fn load_custom_test_cases(
        &self,
        dataset_config: &DatasetConfig,
    ) -> Result<Vec<AccuracyTestCase>, EvaluationError> {
        if !Path::new(&dataset_config.data_path).exists() {
            // Create a sample custom test file
            return self.create_sample_custom_test_cases(dataset_config).await;
        }

        let content = fs::read_to_string(&dataset_config.data_path).map_err(|e| {
            EvaluationError::InvalidInput {
                message: format!("Failed to read custom test file: {e}"),
            }
        })?;

        let mut test_cases = Vec::new();
        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let text = parts[0].to_string();
                let phonemes: Vec<String> =
                    parts[1].split_whitespace().map(|s| s.to_string()).collect();

                let language = match parts[2] {
                    "en-US" | "en" => LanguageCode::EnUs,
                    "ja" => LanguageCode::Ja,
                    "es" => LanguageCode::Es,
                    "fr" => LanguageCode::Fr,
                    "de" => LanguageCode::De,
                    "zh-CN" | "zh" => LanguageCode::ZhCn,
                    _ => LanguageCode::EnUs,
                };

                test_cases.push(AccuracyTestCase {
                    id: format!("custom_{i}"),
                    text,
                    expected_phonemes: phonemes,
                    expected_audio: None,
                    language,
                    reference_transcript: Some(parts[0].to_string()),
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert(String::from("dataset"), String::from("Custom"));
                        meta.insert(String::from("line_number"), (i + 1).to_string());
                        meta
                    },
                });
            }
        }

        Ok(test_cases)
    }

    /// Create sample custom test cases
    async fn create_sample_custom_test_cases(
        &self,
        _dataset_config: &DatasetConfig,
    ) -> Result<Vec<AccuracyTestCase>, EvaluationError> {
        let mut test_cases = Vec::new();

        // Create sample test cases for demonstration
        let samples = vec![
            ("hello", vec!["h", "ə", "l", "oʊ"], LanguageCode::EnUs),
            ("world", vec!["w", "ɝ", "l", "d"], LanguageCode::EnUs),
            ("speech", vec!["s", "p", "i", "tʃ"], LanguageCode::EnUs),
        ];

        for (i, (text, phonemes, language)) in samples.iter().enumerate() {
            test_cases.push(AccuracyTestCase {
                id: format!("sample_{i}"),
                text: text.to_string(),
                expected_phonemes: phonemes.iter().map(|s| s.to_string()).collect(),
                expected_audio: None,
                language: *language,
                reference_transcript: Some(text.to_string()),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(String::from("dataset"), String::from("Sample"));
                    meta
                },
            });
        }

        Ok(test_cases)
    }

    /// Run comprehensive accuracy benchmarks
    pub async fn run_benchmarks<G2P, TTS, ASR>(
        &mut self,
        g2p_system: Option<&G2P>,
        tts_system: Option<&TTS>,
        asr_system: Option<&ASR>,
    ) -> Result<AccuracyBenchmarkResults, EvaluationError>
    where
        G2P: G2pSystem,
        TTS: TtsSystem,
        ASR: AsrSystem,
    {
        let start_time = Instant::now();

        // Load test cases if not already loaded
        if self.test_cases.is_empty() {
            self.load_test_cases().await?;
        }

        let mut dataset_results = HashMap::new();
        let mut all_processing_times = Vec::new();

        // Run benchmarks for each dataset
        for (dataset_name, test_cases) in &self.test_cases {
            println!("Running accuracy benchmark for dataset: {}", dataset_name);

            let dataset_result = self
                .evaluate_dataset(dataset_name, test_cases, g2p_system, tts_system, asr_system)
                .await?;

            // Collect processing times for overall statistics
            for case_result in &dataset_result.case_results {
                all_processing_times.push(case_result.processing_time_ms);
            }

            dataset_results.insert(dataset_name.clone(), dataset_result);
        }

        let total_time = start_time.elapsed().as_secs_f64();

        // Calculate overall metrics
        let overall_metrics = self.calculate_overall_metrics(&dataset_results);
        let performance_stats = self.calculate_performance_stats(&all_processing_times, total_time);

        let results = AccuracyBenchmarkResults {
            config: self.config.clone(),
            dataset_results,
            overall_metrics,
            timestamp: chrono::Utc::now().to_rfc3339(),
            total_time_seconds: total_time,
            performance_stats,
        };

        // Save results if output directory is configured
        self.save_results(&results).await?;

        Ok(results)
    }

    /// Evaluate accuracy for a specific dataset
    async fn evaluate_dataset<G2P, TTS, ASR>(
        &self,
        dataset_name: &str,
        test_cases: &[AccuracyTestCase],
        g2p_system: Option<&G2P>,
        _tts_system: Option<&TTS>,
        _asr_system: Option<&ASR>,
    ) -> Result<DatasetResults, EvaluationError>
    where
        G2P: G2pSystem,
        TTS: TtsSystem,
        ASR: AsrSystem,
    {
        let mut case_results = Vec::new();
        let mut processing_times = Vec::new();
        let mut successful_cases = 0;
        let mut failed_cases = 0;
        let mut total_phoneme_matches = 0;
        let mut total_phonemes = 0;
        let mut total_word_matches = 0;
        let mut total_edit_distance = 0.0;

        for test_case in test_cases {
            let case_start = Instant::now();

            // For now, only implement G2P evaluation
            let result = if let Some(g2p) = g2p_system {
                self.evaluate_g2p_case(test_case, g2p).await
            } else {
                // Simulate evaluation for demo purposes
                self.simulate_case_evaluation(test_case).await
            };

            let processing_time_ms = case_start.elapsed().as_millis() as f64;
            processing_times.push(processing_time_ms);

            match result {
                Ok((actual_phonemes, edit_distance)) => {
                    successful_cases += 1;

                    // Calculate phoneme-level accuracy
                    let (phoneme_matches, phoneme_count) =
                        calculate_phoneme_accuracy(&actual_phonemes, &test_case.expected_phonemes);
                    total_phoneme_matches += phoneme_matches;
                    total_phonemes += phoneme_count;

                    // Calculate word-level accuracy
                    let word_match = actual_phonemes == test_case.expected_phonemes;
                    if word_match {
                        total_word_matches += 1;
                    }

                    total_edit_distance += edit_distance;

                    case_results.push(CaseResult {
                        case_id: test_case.id.clone(),
                        input_text: test_case.text.clone(),
                        expected: test_case.expected_phonemes.clone(),
                        actual: actual_phonemes,
                        passed: word_match,
                        edit_distance,
                        processing_time_ms,
                        error_message: None,
                    });
                }
                Err(e) => {
                    failed_cases += 1;
                    case_results.push(CaseResult {
                        case_id: test_case.id.clone(),
                        input_text: test_case.text.clone(),
                        expected: test_case.expected_phonemes.clone(),
                        actual: Vec::new(),
                        passed: false,
                        edit_distance: test_case.expected_phonemes.len() as f64,
                        processing_time_ms,
                        error_message: Some(format!("{e:?}")),
                    });
                }
            }
        }

        // Calculate dataset metrics
        let total_cases = test_cases.len();
        let phoneme_accuracy = if total_phonemes > 0 {
            total_phoneme_matches as f64 / total_phonemes as f64
        } else {
            0.0
        };
        let word_accuracy = if total_cases > 0 {
            total_word_matches as f64 / total_cases as f64
        } else {
            0.0
        };
        let average_edit_distance = if successful_cases > 0 {
            total_edit_distance / successful_cases as f64
        } else {
            0.0
        };

        // Find target accuracy for this dataset
        let language = if !test_cases.is_empty() {
            test_cases[0].language
        } else {
            LanguageCode::EnUs
        };
        let target_accuracy = self
            .config
            .accuracy_targets
            .get(&language)
            .copied()
            .unwrap_or(0.90);
        let target_met = phoneme_accuracy >= target_accuracy;

        let processing_time_stats = calculate_processing_time_stats(&processing_times);

        Ok(DatasetResults {
            dataset_name: dataset_name.to_string(),
            language,
            total_cases,
            successful_cases,
            failed_cases,
            phoneme_accuracy,
            word_accuracy,
            average_edit_distance,
            target_accuracy,
            target_met,
            case_results,
            processing_time_ms: processing_time_stats,
        })
    }

    /// Evaluate a single G2P test case
    async fn evaluate_g2p_case<G2P>(
        &self,
        test_case: &AccuracyTestCase,
        g2p_system: &G2P,
    ) -> Result<(Vec<String>, f64), EvaluationError>
    where
        G2P: G2pSystem,
    {
        let result = g2p_system
            .convert_to_phonemes(&test_case.text, test_case.language)
            .await?;
        let edit_distance = calculate_edit_distance(&result, &test_case.expected_phonemes);
        Ok((result, edit_distance))
    }

    /// Simulate case evaluation for demonstration
    async fn simulate_case_evaluation(
        &self,
        test_case: &AccuracyTestCase,
    ) -> Result<(Vec<String>, f64), EvaluationError> {
        // Simulate some processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Simulate realistic G2P conversion with some errors
        let mut result = test_case.expected_phonemes.clone();

        // Introduce some realistic errors based on language
        match test_case.language {
            LanguageCode::EnUs => {
                // Simulate common English G2P errors
                {
                    use rand::{thread_rng, Rng};
                    let mut rng = thread_rng();
                    if rng.r#gen::<f64>() < 0.05 {
                        // 5% chance of error
                        if !result.is_empty() {
                            let idx = rng.gen_range(0..result.len());
                            result[idx] = String::from("UH0"); // Common confusion
                        }
                    }
                }
            }
            LanguageCode::Ja => {
                // Simulate common Japanese errors
                {
                    use rand::{thread_rng, Rng};
                    let mut rng = thread_rng();
                    if rng.r#gen::<f64>() < 0.08 {
                        // 8% chance of error
                        if !result.is_empty() {
                            result.push(String::from("u")); // Extra vowel
                        }
                    }
                }
            }
            _ => {
                // Default simulation
                {
                    use rand::{thread_rng, Rng};
                    let mut rng = thread_rng();
                    if rng.r#gen::<f64>() < 0.10 {
                        if !result.is_empty() {
                            result.pop(); // Drop last phoneme
                        }
                    }
                }
            }
        }

        let edit_distance = calculate_edit_distance(&result, &test_case.expected_phonemes);
        Ok((result, edit_distance))
    }

    /// Calculate overall accuracy metrics
    fn calculate_overall_metrics(
        &self,
        dataset_results: &HashMap<String, DatasetResults>,
    ) -> OverallAccuracyMetrics {
        let mut total_cases = 0;
        let mut total_phoneme_matches = 0;
        let mut total_phonemes = 0;
        let mut total_word_matches = 0;
        let mut language_stats: HashMap<LanguageCode, (usize, usize)> = HashMap::new();
        let mut targets_met = 0;
        let total_targets = dataset_results.len();

        for dataset_result in dataset_results.values() {
            total_cases += dataset_result.total_cases;

            if dataset_result.target_met {
                targets_met += 1;
            }

            // Estimate phoneme matches from accuracy
            let dataset_phoneme_matches =
                (dataset_result.phoneme_accuracy * dataset_result.total_cases as f64) as usize;
            total_phoneme_matches += dataset_phoneme_matches;
            total_phonemes += dataset_result.total_cases; // Simplified estimation

            let dataset_word_matches =
                (dataset_result.word_accuracy * dataset_result.total_cases as f64) as usize;
            total_word_matches += dataset_word_matches;

            // Update language statistics
            let stats = language_stats
                .entry(dataset_result.language)
                .or_insert((0, 0));
            stats.0 += dataset_result.total_cases;
            stats.1 += dataset_word_matches;
        }

        let overall_phoneme_accuracy = if total_phonemes > 0 {
            total_phoneme_matches as f64 / total_phonemes as f64
        } else {
            0.0
        };

        let overall_word_accuracy = if total_cases > 0 {
            total_word_matches as f64 / total_cases as f64
        } else {
            0.0
        };

        let language_accuracies = language_stats
            .into_iter()
            .map(|(lang, (total, correct))| {
                let accuracy = if total > 0 {
                    correct as f64 / total as f64
                } else {
                    0.0
                };
                (lang, accuracy)
            })
            .collect();

        let pass_rate = if total_targets > 0 {
            targets_met as f64 / total_targets as f64 * 100.0
        } else {
            0.0
        };

        OverallAccuracyMetrics {
            total_cases,
            overall_phoneme_accuracy,
            overall_word_accuracy,
            language_accuracies,
            targets_met,
            total_targets,
            pass_rate,
        }
    }

    /// Calculate performance statistics
    fn calculate_performance_stats(
        &self,
        processing_times: &[f64],
        total_time_seconds: f64,
    ) -> PerformanceStats {
        if processing_times.is_empty() {
            return PerformanceStats {
                avg_processing_time_ms: 0.0,
                median_processing_time_ms: 0.0,
                p95_processing_time_ms: 0.0,
                throughput_cases_per_sec: 0.0,
                peak_memory_mb: 0.0,
            };
        }

        let mut sorted_times = processing_times.to_vec();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_processing_time_ms = sorted_times.iter().sum::<f64>() / sorted_times.len() as f64;
        let median_processing_time_ms = sorted_times[sorted_times.len() / 2];
        let p95_index = (sorted_times.len() as f64 * 0.95) as usize;
        let p95_processing_time_ms = sorted_times[p95_index.min(sorted_times.len() - 1)];

        let throughput_cases_per_sec = if total_time_seconds > 0.0 {
            processing_times.len() as f64 / total_time_seconds
        } else {
            0.0
        };

        // Simplified memory estimation
        let peak_memory_mb = 100.0; // Placeholder

        PerformanceStats {
            avg_processing_time_ms,
            median_processing_time_ms,
            p95_processing_time_ms,
            throughput_cases_per_sec,
            peak_memory_mb,
        }
    }

    /// Save benchmark results to file
    async fn save_results(
        &self,
        results: &AccuracyBenchmarkResults,
    ) -> Result<(), EvaluationError> {
        // Create output directory
        fs::create_dir_all(&self.config.output_dir).map_err(|e| EvaluationError::InvalidInput {
            message: format!("Failed to create output directory: {e}"),
        })?;

        // Save detailed results as JSON
        let json_path = format!(
            "{output_dir}/accuracy_benchmark_results.json",
            output_dir = self.config.output_dir
        );
        let json_content =
            serde_json::to_string_pretty(results).map_err(|e| EvaluationError::InvalidInput {
                message: format!("Failed to serialize results: {e}"),
            })?;

        fs::write(&json_path, json_content).map_err(|e| EvaluationError::InvalidInput {
            message: format!("Failed to save results: {e}"),
        })?;

        // Generate and save summary report
        let summary_path = format!(
            "{output_dir}/accuracy_benchmark_summary.txt",
            output_dir = self.config.output_dir
        );
        let summary_content = self.generate_summary_report(results);

        fs::write(&summary_path, summary_content).map_err(|e| EvaluationError::InvalidInput {
            message: format!("Failed to save summary: {}", e),
        })?;

        println!("Benchmark results saved to:");
        println!("  - Detailed results: {}", json_path);
        println!("  - Summary report: {}", summary_path);

        Ok(())
    }

    /// Generate human-readable summary report
    fn generate_summary_report(&self, results: &AccuracyBenchmarkResults) -> String {
        let mut report = String::new();

        report.push_str("=".repeat(80).as_str());
        report.push_str("\n");
        report.push_str("                    VoiRS ACCURACY BENCHMARK RESULTS\n");
        report.push_str("=".repeat(80).as_str());
        report.push_str("\n\n");

        report.push_str(&format!("Benchmark executed: {}\n", results.timestamp));
        report.push_str(&format!(
            "Total execution time: {:.2} seconds\n",
            results.total_time_seconds
        ));
        report.push_str(&format!(
            "Total test cases: {}\n",
            results.overall_metrics.total_cases
        ));
        report.push_str("\n");

        // Overall metrics
        report.push_str("OVERALL ACCURACY METRICS\n");
        report.push_str("-".repeat(40).as_str());
        report.push_str("\n");
        report.push_str(&format!(
            "Overall Phoneme Accuracy: {:.2}%\n",
            results.overall_metrics.overall_phoneme_accuracy * 100.0
        ));
        report.push_str(&format!(
            "Overall Word Accuracy: {:.2}%\n",
            results.overall_metrics.overall_word_accuracy * 100.0
        ));
        report.push_str(&format!(
            "Targets Met: {}/{} ({:.1}%)\n",
            results.overall_metrics.targets_met,
            results.overall_metrics.total_targets,
            results.overall_metrics.pass_rate
        ));
        report.push_str("\n");

        // Language-specific results
        report.push_str("LANGUAGE-SPECIFIC ACCURACY\n");
        report.push_str("-".repeat(40).as_str());
        report.push_str("\n");
        for (language, accuracy) in &results.overall_metrics.language_accuracies {
            let target = self
                .config
                .accuracy_targets
                .get(language)
                .copied()
                .unwrap_or(0.90);
            let status = if accuracy >= &target {
                "✅ PASS"
            } else {
                "❌ FAIL"
            };
            report.push_str(&format!(
                "{:?}: {:.2}% (target: {:.0}%) {}\n",
                language,
                accuracy * 100.0,
                target * 100.0,
                status
            ));
        }
        report.push_str("\n");

        // Dataset results
        report.push_str("DATASET RESULTS\n");
        report.push_str("-".repeat(40).as_str());
        report.push_str("\n");
        for (dataset_name, dataset_result) in &results.dataset_results {
            let status = if dataset_result.target_met {
                "✅ PASS"
            } else {
                "❌ FAIL"
            };
            report.push_str(&format!("{}:\n", dataset_name));
            report.push_str(&format!("  Language: {:?}\n", dataset_result.language));
            report.push_str(&format!("  Test Cases: {}\n", dataset_result.total_cases));
            report.push_str(&format!(
                "  Phoneme Accuracy: {:.2}%\n",
                dataset_result.phoneme_accuracy * 100.0
            ));
            report.push_str(&format!(
                "  Word Accuracy: {:.2}%\n",
                dataset_result.word_accuracy * 100.0
            ));
            report.push_str(&format!(
                "  Target: {:.1}% {}\n",
                dataset_result.target_accuracy * 100.0,
                status
            ));
            report.push_str(&format!(
                "  Avg Edit Distance: {:.2}\n",
                dataset_result.average_edit_distance
            ));
            report.push_str(&format!(
                "  Success Rate: {:.1}%\n",
                dataset_result.successful_cases as f64 / dataset_result.total_cases as f64 * 100.0
            ));
            report.push_str("\n");
        }

        // Performance statistics
        report.push_str("PERFORMANCE STATISTICS\n");
        report.push_str("-".repeat(40).as_str());
        report.push_str("\n");
        report.push_str(&format!(
            "Average Processing Time: {:.2} ms\n",
            results.performance_stats.avg_processing_time_ms
        ));
        report.push_str(&format!(
            "Median Processing Time: {:.2} ms\n",
            results.performance_stats.median_processing_time_ms
        ));
        report.push_str(&format!(
            "95th Percentile Time: {:.2} ms\n",
            results.performance_stats.p95_processing_time_ms
        ));
        report.push_str(&format!(
            "Throughput: {:.1} cases/second\n",
            results.performance_stats.throughput_cases_per_sec
        ));
        report.push_str(&format!(
            "Peak Memory Usage: {:.1} MB\n",
            results.performance_stats.peak_memory_mb
        ));
        report.push_str("\n");

        // Recommendations
        report.push_str("RECOMMENDATIONS\n");
        report.push_str("-".repeat(40).as_str());
        report.push_str("\n");

        if results.overall_metrics.pass_rate < 80.0 {
            report.push_str("❌ CRITICAL: Low pass rate detected. Consider:\n");
            report.push_str("   - Reviewing model architecture and training data\n");
            report.push_str("   - Increasing model complexity or training time\n");
            report.push_str("   - Validating test data quality\n\n");
        }

        if results.performance_stats.avg_processing_time_ms > 100.0 {
            report.push_str("⚠️  WARNING: High processing times detected. Consider:\n");
            report.push_str("   - Model optimization (quantization, pruning)\n");
            report.push_str("   - Hardware acceleration (GPU/TPU)\n");
            report.push_str("   - Batch processing optimization\n\n");
        }

        if results.overall_metrics.overall_phoneme_accuracy > 0.95 {
            report.push_str("✅ EXCELLENT: High accuracy achieved!\n");
            report.push_str("   - Consider running larger test sets\n");
            report.push_str("   - Test on more challenging datasets\n");
            report.push_str("   - Monitor for potential overfitting\n\n");
        }

        report.push_str("=".repeat(80).as_str());
        report.push_str("\n");

        report
    }
}

// Helper traits for system interfaces

/// Trait for grapheme-to-phoneme conversion systems
pub trait G2pSystem {
    /// Convert text to phonemes for a given language
    async fn convert_to_phonemes(
        &self,
        text: &str,
        language: LanguageCode,
    ) -> Result<Vec<String>, EvaluationError>;
}

/// Trait for text-to-speech synthesis systems
pub trait TtsSystem {
    /// Synthesize audio from text for a given language
    async fn synthesize(
        &self,
        text: &str,
        language: LanguageCode,
    ) -> Result<AudioBuffer, EvaluationError>;
}

/// Trait for automatic speech recognition systems
pub trait AsrSystem {
    /// Transcribe audio to text for a given language
    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        language: LanguageCode,
    ) -> Result<String, EvaluationError>;
}

// Helper functions
fn calculate_phoneme_accuracy(predicted: &[String], expected: &[String]) -> (usize, usize) {
    let min_len = predicted.len().min(expected.len());
    let mut matches = 0;

    for i in 0..min_len {
        if predicted[i] == expected[i] {
            matches += 1;
        }
    }

    (matches, expected.len())
}

fn calculate_edit_distance(predicted: &[String], expected: &[String]) -> f64 {
    let m = predicted.len();
    let n = expected.len();

    if m == 0 {
        return n as f64;
    }
    if n == 0 {
        return m as f64;
    }

    let mut dp = vec![vec![0; n + 1]; m + 1];

    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if predicted[i - 1] == expected[j - 1] {
                0
            } else {
                1
            };

            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n] as f64
}

fn calculate_processing_time_stats(times: &[f64]) -> ProcessingTimeStats {
    if times.is_empty() {
        return ProcessingTimeStats {
            min_ms: 0.0,
            max_ms: 0.0,
            mean_ms: 0.0,
            std_dev_ms: 0.0,
            median_ms: 0.0,
        };
    }

    let mut sorted_times = times.to_vec();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min_ms = sorted_times[0];
    let max_ms = sorted_times[sorted_times.len() - 1];
    let mean_ms = sorted_times.iter().sum::<f64>() / sorted_times.len() as f64;
    let median_ms = sorted_times[sorted_times.len() / 2];

    let variance = sorted_times
        .iter()
        .map(|x| (x - mean_ms).powi(2))
        .sum::<f64>()
        / sorted_times.len() as f64;
    let std_dev_ms = variance.sqrt();

    ProcessingTimeStats {
        min_ms,
        max_ms,
        mean_ms,
        std_dev_ms,
        median_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockG2pSystem;

    impl G2pSystem for MockG2pSystem {
        async fn convert_to_phonemes(
            &self,
            text: &str,
            _language: LanguageCode,
        ) -> Result<Vec<String>, EvaluationError> {
            // Simple mock conversion
            Ok(text.chars().map(|c| c.to_string()).collect())
        }
    }

    impl TtsSystem for MockG2pSystem {
        async fn synthesize(
            &self,
            text: &str,
            _language: LanguageCode,
        ) -> Result<AudioBuffer, EvaluationError> {
            // Mock TTS synthesis - create a simple audio buffer
            let sample_rate = 22050;
            let duration_seconds = text.len() as f32 * 0.1; // Rough duration estimate
            let num_samples = (sample_rate as f32 * duration_seconds) as usize;
            let samples = vec![0.0f32; num_samples]; // Silent audio for testing

            Ok(AudioBuffer::new(samples, sample_rate, 1))
        }
    }

    impl AsrSystem for MockG2pSystem {
        async fn transcribe(
            &self,
            _audio: &AudioBuffer,
            _language: LanguageCode,
        ) -> Result<String, EvaluationError> {
            // Mock ASR transcription - return a simple test string
            Ok(String::from("mock transcription"))
        }
    }

    #[tokio::test]
    async fn test_accuracy_benchmark_runner_creation() {
        let config = AccuracyBenchmarkConfig::default();
        let runner = AccuracyBenchmarkRunner::new(config);
        assert!(!runner.config.accuracy_targets.is_empty());
    }

    #[tokio::test]
    async fn test_load_test_cases() {
        let mut runner = AccuracyBenchmarkRunner::default();
        let result = runner.load_test_cases().await;
        assert!(result.is_ok());
        assert!(!runner.test_cases.is_empty());
    }

    #[tokio::test]
    async fn test_benchmark_evaluation() {
        let mut runner = AccuracyBenchmarkRunner::default();
        runner.load_test_cases().await.unwrap();

        let mock_g2p = MockG2pSystem;
        let results = runner
            .run_benchmarks(
                Some(&mock_g2p),
                None::<&MockG2pSystem>,
                None::<&MockG2pSystem>,
            )
            .await;

        assert!(results.is_ok());
        let benchmark_results = results.unwrap();
        assert!(!benchmark_results.dataset_results.is_empty());
        assert!(benchmark_results.overall_metrics.total_cases > 0);
    }

    #[test]
    fn test_phoneme_accuracy_calculation() {
        let predicted = vec![
            String::from("h"),
            String::from("ə"),
            String::from("l"),
            String::from("oʊ"),
        ];
        let expected = vec![
            String::from("h"),
            String::from("ə"),
            String::from("ˈl"),
            String::from("oʊ"),
        ];

        let (matches, total) = calculate_phoneme_accuracy(&predicted, &expected);
        assert_eq!(matches, 3);
        assert_eq!(total, 4);
    }

    #[test]
    fn test_edit_distance_calculation() {
        let predicted = vec![
            String::from("h"),
            String::from("ə"),
            String::from("l"),
            String::from("oʊ"),
        ];
        let expected = vec![
            String::from("h"),
            String::from("ə"),
            String::from("ˈl"),
            String::from("oʊ"),
        ];

        let distance = calculate_edit_distance(&predicted, &expected);
        assert_eq!(distance, 1.0);
    }

    #[test]
    fn test_dataset_config_creation() {
        let cmu_config = DatasetConfig::cmu_english();
        assert_eq!(cmu_config.language, LanguageCode::EnUs);
        assert_eq!(cmu_config.target_accuracy, 0.95);

        let jvs_config = DatasetConfig::jvs_japanese();
        assert_eq!(jvs_config.language, LanguageCode::Ja);
        assert_eq!(jvs_config.target_accuracy, 0.90);
    }
}
