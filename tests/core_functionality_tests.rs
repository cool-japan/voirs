//! Core functionality tests for VoiRS components
//!
//! This test suite validates the core functionality of all major VoiRS components
//! including G2P accuracy, acoustic model consistency, and vocoder reconstruction quality.

use std::path::Path;
use std::time::Instant;
use voirs_acoustic::{
    AcousticModel, DummyAcousticModel, MelSpectrogram as AcousticMel, Phoneme as AcousticPhoneme,
    SynthesisConfig as AcousticConfig,
};
use voirs_evaluation::quality::{
    mcd::MCDEvaluator, pesq::PESQEvaluator, si_sdr::SISdrEvaluator, stoi::STOIEvaluator,
};
use voirs_g2p::{
    accuracy::{AccuracyBenchmark, TestCase},
    rules::EnglishRuleG2p,
    DummyG2p, G2p, LanguageCode,
};
use voirs_vocoder::{
    audio::io::convenience::write_wav, AudioBuffer, DummyVocoder, MelSpectrogram as VocoderMel,
    SynthesisConfig as VocoderConfig, Vocoder,
};

/// Comprehensive core functionality test suite
pub struct CoreFunctionalityTests {
    g2p_benchmark: AccuracyBenchmark,
    mcd_evaluator: MCDEvaluator,
    pesq_evaluator: PESQEvaluator,
    si_sdr_evaluator: SISdrEvaluator,
    stoi_evaluator: STOIEvaluator,
}

impl CoreFunctionalityTests {
    /// Initialize the test suite with required components
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut g2p_benchmark = AccuracyBenchmark::new();

        // Add comprehensive test cases for G2P accuracy
        g2p_benchmark.add_test_case(TestCase {
            word: "hello".to_string(),
            expected_phonemes: vec![
                "h".to_string(),
                "ə".to_string(),
                "ˈl".to_string(),
                "oʊ".to_string(),
            ],
            language: LanguageCode::EnUs,
        });

        g2p_benchmark.add_test_case(TestCase {
            word: "world".to_string(),
            expected_phonemes: vec![
                "w".to_string(),
                "ɜː".to_string(),
                "r".to_string(),
                "l".to_string(),
                "d".to_string(),
            ],
            language: LanguageCode::EnUs,
        });

        g2p_benchmark.add_test_case(TestCase {
            word: "synthesis".to_string(),
            expected_phonemes: vec![
                "s".to_string(),
                "ɪ".to_string(),
                "n".to_string(),
                "θ".to_string(),
                "ə".to_string(),
                "s".to_string(),
                "ɪ".to_string(),
                "s".to_string(),
            ],
            language: LanguageCode::EnUs,
        });

        let sample_rate = 22050; // Standard sample rate for TTS
        Ok(Self {
            g2p_benchmark,
            mcd_evaluator: MCDEvaluator::new(sample_rate)?,
            pesq_evaluator: PESQEvaluator::new_wideband()?,
            si_sdr_evaluator: SISdrEvaluator::new(sample_rate),
            stoi_evaluator: STOIEvaluator::new(sample_rate)?,
        })
    }

    /// Run comprehensive core functionality tests
    pub async fn run_comprehensive_tests(
        &self,
    ) -> Result<CoreTestResults, Box<dyn std::error::Error>> {
        println!("🔍 Running Core Functionality Tests");
        println!("==================================");

        let mut results = CoreTestResults::new();

        // Test G2P accuracy with reference datasets
        results.g2p_results = self.test_g2p_accuracy().await?;

        // Test acoustic model output consistency
        results.acoustic_results = self.test_acoustic_model_consistency().await?;

        // Test vocoder reconstruction quality
        results.vocoder_results = self.test_vocoder_reconstruction_quality().await?;

        // Test full pipeline integration
        results.pipeline_results = self.test_full_pipeline_integration().await?;

        println!("\n✅ Core Functionality Tests Complete");
        results.print_summary();

        Ok(results)
    }

    /// Test G2P accuracy with reference datasets
    async fn test_g2p_accuracy(&self) -> Result<G2pTestResults, Box<dyn std::error::Error>> {
        println!("\n📊 Testing G2P Accuracy");
        println!("----------------------");

        let mut results = G2pTestResults::new();

        // Test DummyG2p (baseline)
        let dummy_g2p = DummyG2p::new();
        let start_time = Instant::now();
        let dummy_metrics = self.g2p_benchmark.evaluate(&dummy_g2p).await?;
        results.dummy_duration = start_time.elapsed();
        results.dummy_accuracy = dummy_metrics.phoneme_accuracy as f32;

        println!("DummyG2p Results:");
        println!(
            "  - Phoneme Accuracy: {:.2}%",
            dummy_metrics.phoneme_accuracy * 100.0
        );
        println!("  - Processing Time: {:?}", results.dummy_duration);

        // Test EnglishRuleG2p if available
        match EnglishRuleG2p::new() {
            Ok(rule_g2p) => {
                let start_time = Instant::now();
                let rule_metrics = self.g2p_benchmark.evaluate(&rule_g2p).await?;
                results.rule_duration = Some(start_time.elapsed());
                results.rule_accuracy = Some(rule_metrics.phoneme_accuracy as f32);

                println!("EnglishRuleG2p Results:");
                println!(
                    "  - Phoneme Accuracy: {:.2}%",
                    rule_metrics.phoneme_accuracy * 100.0
                );
                println!("  - Processing Time: {:?}", results.rule_duration.unwrap());

                // Validate that rule-based G2P performs better than dummy
                if rule_metrics.phoneme_accuracy > dummy_metrics.phoneme_accuracy {
                    println!("  ✅ Rule-based G2P outperforms baseline");
                    results.rule_better_than_dummy = true;
                } else {
                    println!("  ⚠️  Rule-based G2P does not outperform baseline");
                }
            }
            Err(e) => {
                println!("  ⚠️  Could not test EnglishRuleG2p: {e}");
            }
        }

        Ok(results)
    }

    /// Test acoustic model output consistency
    async fn test_acoustic_model_consistency(
        &self,
    ) -> Result<AcousticTestResults, Box<dyn std::error::Error>> {
        println!("\n🎵 Testing Acoustic Model Consistency");
        println!("-----------------------------------");

        let mut results = AcousticTestResults::new();
        let acoustic_model = DummyAcousticModel::new();

        // Test consistent output for same input
        let test_phonemes = vec![
            AcousticPhoneme::new("h"),
            AcousticPhoneme::new("ə"),
            AcousticPhoneme::new("l"),
            AcousticPhoneme::new("oʊ"),
        ];

        let config = AcousticConfig {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: Some(42), // Fixed seed for reproducibility
            emotion: None,
            voice_style: None,
        };

        // Generate mel spectrogram multiple times with same input
        let start_time = Instant::now();
        let mel1 = acoustic_model
            .synthesize(&test_phonemes, Some(&config))
            .await?;
        let mel2 = acoustic_model
            .synthesize(&test_phonemes, Some(&config))
            .await?;
        let mel3 = acoustic_model
            .synthesize(&test_phonemes, Some(&config))
            .await?;
        results.generation_time = start_time.elapsed();

        // Check consistency between generations
        results.mel_consistency = self.calculate_mel_consistency(&mel1, &mel2, &mel3);
        results.mel_dimensions = (mel1.n_mels, mel1.n_frames);
        results.mel_duration = mel1.duration();

        println!("Acoustic Model Results:");
        println!(
            "  - Mel Dimensions: {}x{}",
            results.mel_dimensions.0, results.mel_dimensions.1
        );
        println!("  - Mel Duration: {:.2}s", results.mel_duration);
        println!("  - Generation Time: {:?}", results.generation_time);
        println!("  - Consistency Score: {:.4}", results.mel_consistency);

        if results.mel_consistency > 0.95 {
            println!("  ✅ High consistency achieved");
        } else {
            println!("  ⚠️  Low consistency detected");
        }

        Ok(results)
    }

    /// Test vocoder reconstruction quality
    async fn test_vocoder_reconstruction_quality(
        &self,
    ) -> Result<VocoderTestResults, Box<dyn std::error::Error>> {
        println!("\n🔊 Testing Vocoder Reconstruction Quality");
        println!("---------------------------------------");

        let mut results = VocoderTestResults::new();
        let vocoder = DummyVocoder::new();

        // Create test mel spectrogram
        let mel_data = vec![vec![0.5f32; 128]; 80]; // 80 mel bins, 128 frames
        let test_mel = VocoderMel::new(mel_data, 22050, 256);

        let config = VocoderConfig {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: Some(42),
        };

        // Generate audio multiple times
        let start_time = Instant::now();
        let audio1 = vocoder.vocode(&test_mel, Some(&config)).await?;
        let audio2 = vocoder.vocode(&test_mel, Some(&config)).await?;
        results.vocoding_time = start_time.elapsed();

        // Test audio quality metrics
        results.audio_length = audio1.len();
        results.sample_rate = audio1.sample_rate();
        results.duration = audio1.len() as f32 / audio1.sample_rate() as f32;

        // Calculate consistency between generations
        results.audio_consistency = self.calculate_audio_consistency(&audio1, &audio2);

        // Test basic audio properties
        results.rms_level = self.calculate_rms(audio1.samples());
        results.peak_level = audio1
            .samples()
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);
        results.zero_crossings = self.count_zero_crossings(audio1.samples());

        println!("Vocoder Results:");
        println!("  - Audio Length: {} samples", results.audio_length);
        println!("  - Sample Rate: {} Hz", results.sample_rate);
        println!("  - Duration: {:.2}s", results.duration);
        println!("  - Vocoding Time: {:?}", results.vocoding_time);
        println!("  - Audio Consistency: {:.4}", results.audio_consistency);
        println!("  - RMS Level: {:.4}", results.rms_level);
        println!("  - Peak Level: {:.4}", results.peak_level);
        println!("  - Zero Crossings: {}", results.zero_crossings);

        if results.audio_consistency > 0.95 {
            println!("  ✅ High audio consistency achieved");
        } else {
            println!("  ⚠️  Audio consistency issues detected");
        }

        Ok(results)
    }

    /// Test full pipeline integration
    async fn test_full_pipeline_integration(
        &self,
    ) -> Result<PipelineTestResults, Box<dyn std::error::Error>> {
        println!("\n🔄 Testing Full Pipeline Integration");
        println!("----------------------------------");

        let mut results = PipelineTestResults::new();

        // Initialize components
        let g2p = DummyG2p::new();
        let acoustic_model = DummyAcousticModel::new();
        let vocoder = DummyVocoder::new();

        let test_texts = vec!["hello", "world", "synthesis", "test"];

        for text in test_texts {
            let start_time = Instant::now();

            // Full pipeline: text -> phonemes -> mel -> audio
            let phonemes = g2p.to_phonemes(text, Some(LanguageCode::EnUs)).await?;
            let acoustic_phonemes: Vec<AcousticPhoneme> = phonemes
                .iter()
                .map(|p| AcousticPhoneme::new(&p.symbol))
                .collect();

            let acoustic_config = AcousticConfig {
                speed: 1.0,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            };

            let mel = acoustic_model
                .synthesize(&acoustic_phonemes, Some(&acoustic_config))
                .await?;
            let vocoder_mel = VocoderMel::new(mel.data.clone(), mel.sample_rate, mel.hop_length);
            let vocoder_config = VocoderConfig {
                speed: acoustic_config.speed,
                pitch_shift: acoustic_config.pitch_shift,
                energy: acoustic_config.energy,
                speaker_id: acoustic_config.speaker_id,
                seed: acoustic_config.seed,
            };

            let audio = vocoder.vocode(&vocoder_mel, Some(&vocoder_config)).await?;

            let pipeline_time = start_time.elapsed();

            let result = PipelineResult {
                text: text.to_string(),
                phoneme_count: phonemes.len(),
                mel_dimensions: (mel.n_mels, mel.n_frames),
                audio_length: audio.len(),
                processing_time: pipeline_time,
            };

            println!(
                "  Text '{}': {} phonemes -> {}x{} mel -> {} samples ({:?})",
                result.text,
                result.phoneme_count,
                result.mel_dimensions.0,
                result.mel_dimensions.1,
                result.audio_length,
                result.processing_time
            );

            results.pipeline_results.push(result);
        }

        results.average_processing_time = results
            .pipeline_results
            .iter()
            .map(|r| r.processing_time)
            .sum::<std::time::Duration>()
            / results.pipeline_results.len() as u32;

        println!(
            "  Average Processing Time: {:?}",
            results.average_processing_time
        );
        println!("  ✅ Full pipeline integration successful");

        Ok(results)
    }

    /// Calculate mel spectrogram consistency between multiple generations
    fn calculate_mel_consistency(
        &self,
        mel1: &AcousticMel,
        mel2: &AcousticMel,
        mel3: &AcousticMel,
    ) -> f32 {
        if mel1.data.len() != mel2.data.len() || mel2.data.len() != mel3.data.len() {
            return 0.0;
        }

        let mut total_diff = 0.0f32;
        let mut count = 0;

        for (i, (row1, row2)) in mel1.data.iter().zip(mel2.data.iter()).enumerate() {
            if i < mel3.data.len() {
                let row3 = &mel3.data[i];
                for ((&val1, &val2), &val3) in row1.iter().zip(row2.iter()).zip(row3.iter()) {
                    let diff12 = (val1 - val2).abs();
                    let diff23 = (val2 - val3).abs();
                    let diff13 = (val1 - val3).abs();
                    total_diff += diff12 + diff23 + diff13;
                    count += 3;
                }
            }
        }

        if count > 0 {
            1.0 - (total_diff / count as f32).min(1.0)
        } else {
            0.0
        }
    }

    /// Calculate audio consistency between two audio buffers
    fn calculate_audio_consistency(&self, audio1: &AudioBuffer, audio2: &AudioBuffer) -> f32 {
        if audio1.len() != audio2.len() {
            return 0.0;
        }

        let mut total_diff = 0.0f32;
        for (&val1, &val2) in audio1.samples().iter().zip(audio2.samples().iter()) {
            total_diff += (val1 - val2).abs();
        }

        let avg_diff = total_diff / audio1.len() as f32;
        1.0 - avg_diff.min(1.0)
    }

    /// Calculate RMS level of audio
    fn calculate_rms(&self, audio: &[f32]) -> f32 {
        let sum_squares: f32 = audio.iter().map(|&x| x * x).sum();
        (sum_squares / audio.len() as f32).sqrt()
    }

    /// Count zero crossings in audio
    fn count_zero_crossings(&self, audio: &[f32]) -> usize {
        let mut count = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
                count += 1;
            }
        }
        count
    }
}

/// Results structure for core functionality tests
#[derive(Debug)]
pub struct CoreTestResults {
    pub g2p_results: G2pTestResults,
    pub acoustic_results: AcousticTestResults,
    pub vocoder_results: VocoderTestResults,
    pub pipeline_results: PipelineTestResults,
}

impl CoreTestResults {
    pub fn new() -> Self {
        Self {
            g2p_results: G2pTestResults::new(),
            acoustic_results: AcousticTestResults::new(),
            vocoder_results: VocoderTestResults::new(),
            pipeline_results: PipelineTestResults::new(),
        }
    }

    pub fn print_summary(&self) {
        println!("\n📋 Core Functionality Test Summary");
        println!("=================================");
        println!("G2P Tests:");
        println!(
            "  - Dummy Accuracy: {:.2}%",
            self.g2p_results.dummy_accuracy * 100.0
        );
        if let Some(rule_acc) = self.g2p_results.rule_accuracy {
            println!("  - Rule Accuracy: {:.2}%", rule_acc * 100.0);
        }

        println!("Acoustic Tests:");
        println!(
            "  - Consistency: {:.4}",
            self.acoustic_results.mel_consistency
        );
        println!("  - Duration: {:.2}s", self.acoustic_results.mel_duration);

        println!("Vocoder Tests:");
        println!(
            "  - Consistency: {:.4}",
            self.vocoder_results.audio_consistency
        );
        println!("  - RMS Level: {:.4}", self.vocoder_results.rms_level);

        println!("Pipeline Tests:");
        println!(
            "  - Tests Run: {}",
            self.pipeline_results.pipeline_results.len()
        );
        println!(
            "  - Avg Time: {:?}",
            self.pipeline_results.average_processing_time
        );
    }
}

#[derive(Debug)]
pub struct G2pTestResults {
    pub dummy_accuracy: f32,
    pub dummy_duration: std::time::Duration,
    pub rule_accuracy: Option<f32>,
    pub rule_duration: Option<std::time::Duration>,
    pub rule_better_than_dummy: bool,
}

impl G2pTestResults {
    pub fn new() -> Self {
        Self {
            dummy_accuracy: 0.0,
            dummy_duration: std::time::Duration::from_secs(0),
            rule_accuracy: None,
            rule_duration: None,
            rule_better_than_dummy: false,
        }
    }
}

#[derive(Debug)]
pub struct AcousticTestResults {
    pub mel_consistency: f32,
    pub mel_dimensions: (usize, usize),
    pub mel_duration: f32,
    pub generation_time: std::time::Duration,
}

impl AcousticTestResults {
    pub fn new() -> Self {
        Self {
            mel_consistency: 0.0,
            mel_dimensions: (0, 0),
            mel_duration: 0.0,
            generation_time: std::time::Duration::from_secs(0),
        }
    }
}

#[derive(Debug)]
pub struct VocoderTestResults {
    pub audio_consistency: f32,
    pub audio_length: usize,
    pub sample_rate: u32,
    pub duration: f32,
    pub vocoding_time: std::time::Duration,
    pub rms_level: f32,
    pub peak_level: f32,
    pub zero_crossings: usize,
}

impl VocoderTestResults {
    pub fn new() -> Self {
        Self {
            audio_consistency: 0.0,
            audio_length: 0,
            sample_rate: 0,
            duration: 0.0,
            vocoding_time: std::time::Duration::from_secs(0),
            rms_level: 0.0,
            peak_level: 0.0,
            zero_crossings: 0,
        }
    }
}

#[derive(Debug)]
pub struct PipelineTestResults {
    pub pipeline_results: Vec<PipelineResult>,
    pub average_processing_time: std::time::Duration,
}

impl PipelineTestResults {
    pub fn new() -> Self {
        Self {
            pipeline_results: Vec::new(),
            average_processing_time: std::time::Duration::from_secs(0),
        }
    }
}

#[derive(Debug)]
pub struct PipelineResult {
    pub text: String,
    pub phoneme_count: usize,
    pub mel_dimensions: (usize, usize),
    pub audio_length: usize,
    pub processing_time: std::time::Duration,
}

#[tokio::test]
async fn test_core_functionality_comprehensive() {
    let test_suite =
        CoreFunctionalityTests::new().expect("Failed to initialize core functionality tests");

    let results = test_suite
        .run_comprehensive_tests()
        .await
        .expect("Core functionality tests failed");

    // Validate minimum requirements
    assert!(
        results.g2p_results.dummy_accuracy > 0.0,
        "G2P should produce some output"
    );
    assert!(
        results.acoustic_results.mel_consistency > 0.5,
        "Acoustic model should be reasonably consistent"
    );
    assert!(
        results.vocoder_results.audio_consistency > 0.5,
        "Vocoder should be reasonably consistent"
    );
    assert!(
        !results.pipeline_results.pipeline_results.is_empty(),
        "Pipeline should process test cases"
    );
}
