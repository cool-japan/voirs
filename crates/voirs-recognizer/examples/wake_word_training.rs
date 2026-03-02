//! Custom Wake Word Training Example
//!
//! This example demonstrates how to implement custom wake word detection and training
//! using VoiRS Recognizer with keyword spotting capabilities.
//!
//! Usage:
//! ```bash
//! cargo run --example wake_word_training --features="whisper-pure"
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use voirs_recognizer::asr::{ASRBackend, FallbackConfig, WhisperModelSize};
use voirs_recognizer::prelude::*;
use voirs_recognizer::{PerformanceValidator, RecognitionError};

// Wake word detection configuration
#[derive(Debug, Clone)]
pub struct WakeWordConfig {
    pub wake_word: String,
    pub confidence_threshold: f32,
    pub energy_threshold: f32,
    pub max_detection_time: Duration,
    pub false_positive_reduction: bool,
    pub energy_efficient_mode: bool,
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        Self {
            wake_word: "hey assistant".to_string(),
            confidence_threshold: 0.8,
            energy_threshold: 0.1,
            max_detection_time: Duration::from_secs(3),
            false_positive_reduction: true,
            energy_efficient_mode: true,
        }
    }
}

// Wake word detection result
#[derive(Debug, Clone)]
pub struct WakeWordDetection {
    pub detected: bool,
    pub confidence: f32,
    pub timestamp: Duration,
    pub audio_segment: Option<AudioBuffer>,
}

// Wake word detector implementation
pub struct WakeWordDetector {
    config: WakeWordConfig,
    asr: IntelligentASRFallback,
    audio_analyzer: AudioAnalyzerImpl,
    detection_buffer: Vec<AudioBuffer>,
    performance_stats: HashMap<String, f32>,
}

impl WakeWordDetector {
    pub async fn new(config: WakeWordConfig) -> Result<Self, RecognitionError> {
        // Configure ASR for wake word detection (optimized for speed)
        let fallback_config = FallbackConfig {
            primary_backend: ASRBackend::Whisper {
                model_size: WhisperModelSize::Tiny,
                model_path: None,
            },
            quality_threshold: 0.5,
            max_processing_time_seconds: 1.0,
            adaptive_selection: true,
            ..Default::default()
        };

        let asr = IntelligentASRFallback::new(fallback_config).await?;

        // Configure audio analyzer for energy detection
        let analyzer_config = AudioAnalysisConfig {
            emotional_analysis: false,
            quality_metrics_list: vec![],
            frame_size: 1024,
            hop_size: 512,
            prosody_analysis: false,
            quality_metrics: false,
            speaker_analysis: false,
        };

        let audio_analyzer = AudioAnalyzerImpl::new(analyzer_config).await?;

        Ok(Self {
            config,
            asr,
            audio_analyzer,
            detection_buffer: Vec::new(),
            performance_stats: HashMap::new(),
        })
    }

    pub async fn detect_wake_word(
        &mut self,
        audio: &AudioBuffer,
    ) -> Result<WakeWordDetection, RecognitionError> {
        let detection_start = Instant::now();

        // Step 1: Energy-based pre-filtering (if enabled)
        if self.config.energy_efficient_mode {
            let analysis = self
                .audio_analyzer
                .analyze(audio, Some(&AudioAnalysisConfig::default()))
                .await?;

            if let Some(energy) = analysis.quality_metrics.get("energy") {
                if *energy < self.config.energy_threshold {
                    // Skip ASR processing for low-energy audio
                    return Ok(WakeWordDetection {
                        detected: false,
                        confidence: 0.0,
                        timestamp: detection_start.elapsed(),
                        audio_segment: None,
                    });
                }
            }
        }

        // Step 2: Perform ASR on the audio
        let asr_config = ASRConfig {
            language: Some(LanguageCode::EnUs),
            word_timestamps: true,
            confidence_threshold: 0.5,
            ..Default::default()
        };
        let recognition_result = self.asr.transcribe(audio, Some(&asr_config)).await?;

        // Step 3: Check for wake word in transcription
        let detected = self.check_wake_word_match(
            &recognition_result.transcript.text,
            recognition_result.transcript.confidence,
        );

        // Step 4: False positive reduction (if enabled)
        let final_detected = if self.config.false_positive_reduction && detected {
            self.reduce_false_positives(audio, &recognition_result.transcript.text)
                .await?
        } else {
            detected
        };

        // Step 5: Update performance statistics
        let detection_time = detection_start.elapsed();
        self.performance_stats.insert(
            "last_detection_time_ms".to_string(),
            detection_time.as_secs_f32() * 1000.0,
        );
        self.performance_stats.insert(
            "last_confidence".to_string(),
            recognition_result.transcript.confidence,
        );

        Ok(WakeWordDetection {
            detected: final_detected,
            confidence: recognition_result.transcript.confidence,
            timestamp: detection_time,
            audio_segment: if final_detected {
                Some(audio.clone())
            } else {
                None
            },
        })
    }

    fn check_wake_word_match(&self, transcription: &str, confidence: f32) -> bool {
        if confidence < self.config.confidence_threshold {
            return false;
        }

        // Normalize text for comparison
        let normalized_transcription = transcription.to_lowercase();
        let normalized_wake_word = self.config.wake_word.to_lowercase();

        // Check for exact match or substring match
        normalized_transcription.contains(&normalized_wake_word)
            || self.fuzzy_match(&normalized_transcription, &normalized_wake_word)
    }

    fn fuzzy_match(&self, text: &str, wake_word: &str) -> bool {
        // Simple fuzzy matching based on word similarity
        let text_words: Vec<&str> = text.split_whitespace().collect();
        let wake_words: Vec<&str> = wake_word.split_whitespace().collect();

        if wake_words.len() > text_words.len() {
            return false;
        }

        // Check if all wake words appear in order with reasonable gaps
        let mut wake_word_index = 0;
        for text_word in text_words {
            if wake_word_index < wake_words.len()
                && self.word_similarity(text_word, wake_words[wake_word_index]) > 0.7
            {
                wake_word_index += 1;
            }
        }

        wake_word_index >= wake_words.len()
    }

    fn word_similarity(&self, word1: &str, word2: &str) -> f32 {
        // Simple Levenshtein distance-based similarity
        let len1 = word1.len();
        let len2 = word2.len();

        if len1 == 0 {
            return if len2 == 0 { 1.0 } else { 0.0 };
        }
        if len2 == 0 {
            return 0.0;
        }

        let mut dp = vec![vec![0; len2 + 1]; len1 + 1];

        // 2D DP table: index-based access requires range loops
        #[allow(clippy::needless_range_loop)]
        for i in 0..=len1 {
            dp[i][0] = i;
        }
        for (j, cell) in dp[0].iter_mut().enumerate() {
            *cell = j;
        }

        #[allow(clippy::needless_range_loop)]
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if word1.chars().nth(i - 1) == word2.chars().nth(j - 1) {
                    0
                } else {
                    1
                };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        1.0 - (dp[len1][len2] as f32 / len1.max(len2) as f32)
    }

    async fn reduce_false_positives(
        &mut self,
        audio: &AudioBuffer,
        transcription: &str,
    ) -> Result<bool, RecognitionError> {
        // Add current audio to detection buffer
        self.detection_buffer.push(audio.clone());

        // Keep only recent detections (last 3 seconds)
        if self.detection_buffer.len() > 10 {
            self.detection_buffer.remove(0);
        }

        // Check if we have multiple recent detections (might indicate false positive)
        if self.detection_buffer.len() < 2 {
            return Ok(true); // First detection, accept it
        }

        // Analyze consistency of recent detections
        let mut recent_transcriptions = Vec::new();
        for buffer_audio in &self.detection_buffer {
            let asr_config = ASRConfig {
                language: Some(LanguageCode::EnUs),
                word_timestamps: true,
                confidence_threshold: 0.5,
                ..Default::default()
            };
            let result = self.asr.transcribe(buffer_audio, Some(&asr_config)).await?;
            recent_transcriptions.push(result.transcript.text.to_lowercase());
        }

        // Check for consistency
        let target_wake_word = self.config.wake_word.to_lowercase();
        let consistent_detections = recent_transcriptions
            .iter()
            .filter(|text| text.contains(&target_wake_word))
            .count();

        // Require at least 50% consistency to reduce false positives
        Ok(consistent_detections as f32 / recent_transcriptions.len() as f32 > 0.5)
    }

    pub fn get_performance_stats(&self) -> &HashMap<String, f32> {
        &self.performance_stats
    }
}

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("🎙️ Custom Wake Word Training Demo");
    println!("=================================\n");

    // Step 1: Configure wake word detection
    println!("🔧 Configuring wake word detection...");

    let wake_word_configs = vec![
        WakeWordConfig {
            wake_word: "hey assistant".to_string(),
            confidence_threshold: 0.8,
            energy_threshold: 0.1,
            max_detection_time: Duration::from_secs(3),
            false_positive_reduction: true,
            energy_efficient_mode: true,
        },
        WakeWordConfig {
            wake_word: "computer".to_string(),
            confidence_threshold: 0.85,
            energy_threshold: 0.15,
            max_detection_time: Duration::from_secs(2),
            false_positive_reduction: true,
            energy_efficient_mode: true,
        },
        WakeWordConfig {
            wake_word: "activate voice".to_string(),
            confidence_threshold: 0.75,
            energy_threshold: 0.12,
            max_detection_time: Duration::from_secs(4),
            false_positive_reduction: false,
            energy_efficient_mode: false,
        },
    ];

    println!("✅ Wake word configurations:");
    for (i, config) in wake_word_configs.iter().enumerate() {
        println!(
            "   {}. \"{}\" (confidence: {:.2}, energy: {:.2})",
            i + 1,
            config.wake_word,
            config.confidence_threshold,
            config.energy_threshold
        );
    }

    // Step 2: Initialize wake word detectors
    println!("\n🚀 Initializing wake word detectors...");

    let mut detectors = Vec::new();
    for config in wake_word_configs {
        let detector = WakeWordDetector::new(config.clone()).await?;
        detectors.push(detector);
    }

    println!("✅ {} wake word detectors initialized", detectors.len());

    // Step 3: Create test audio scenarios
    println!("\n🎵 Creating test audio scenarios...");
    let test_scenarios = create_wake_word_test_scenarios();

    println!("✅ Created {} test scenarios:", test_scenarios.len());
    for (i, (name, _, expected_detections)) in test_scenarios.iter().enumerate() {
        println!(
            "   {}. {}: {} expected detections",
            i + 1,
            name,
            expected_detections
        );
    }

    // Step 4: Performance validation setup
    println!("\n📊 Setting up performance validation...");
    let validator = PerformanceValidator::new().with_verbose(false);

    // Step 5: Run wake word detection tests
    println!("\n⚡ Running wake word detection tests...");
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Scenario              │ Wake Word      │ Detected │ Confidence │ Time(ms) │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    let mut total_detections = 0;
    let mut correct_detections = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    for (scenario_name, audio, expected_detections) in &test_scenarios {
        for (detector_idx, detector) in detectors.iter_mut().enumerate() {
            let detection_start = Instant::now();
            let detection = detector.detect_wake_word(audio).await?;
            let detection_time = detection_start.elapsed();

            // Validate detection latency
            let (latency_ms, _) = validator.validate_streaming_latency(detection_time);

            // Determine detection accuracy
            let expected_for_this_detector = *expected_detections > 0;
            let detection_correct = detection.detected == expected_for_this_detector;

            if detection.detected {
                total_detections += 1;
                if detection_correct {
                    correct_detections += 1;
                } else {
                    false_positives += 1;
                }
            } else if expected_for_this_detector {
                false_negatives += 1;
            }

            println!(
                "│ {:19} │ {:13} │ {:8} │ {:10.2} │ {:8.0} │",
                if detector_idx == 0 {
                    scenario_name.to_string()
                } else {
                    "".to_string()
                },
                &detector.config.wake_word,
                if detection.detected {
                    "✅ YES"
                } else {
                    "❌ NO"
                },
                detection.confidence,
                latency_ms
            );
        }
    }

    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    // Step 6: Calculate detection accuracy metrics
    println!("\n📈 Detection Accuracy Analysis:");

    let total_tests = test_scenarios.len() * detectors.len();
    let accuracy = if total_tests > 0 {
        (correct_detections as f32 / total_tests as f32) * 100.0
    } else {
        0.0
    };

    println!("   Overall Performance:");
    println!("   • Total tests: {}", total_tests);
    println!("   • Correct detections: {}", correct_detections);
    println!("   • False positives: {}", false_positives);
    println!("   • False negatives: {}", false_negatives);
    println!("   • Accuracy: {:.1}%", accuracy);

    // Calculate precision and recall
    let precision = if total_detections > 0 {
        (correct_detections as f32 / total_detections as f32) * 100.0
    } else {
        0.0
    };

    let recall = if (correct_detections + false_negatives) > 0 {
        (correct_detections as f32 / (correct_detections + false_negatives) as f32) * 100.0
    } else {
        0.0
    };

    println!("   • Precision: {:.1}%", precision);
    println!("   • Recall: {:.1}%", recall);

    if precision > 0.0 && recall > 0.0 {
        let f1_score = 2.0 * (precision * recall) / (precision + recall);
        println!("   • F1 Score: {:.1}%", f1_score);
    }

    // Step 7: Performance analysis per detector
    println!("\n🔍 Per-Detector Performance Analysis:");

    for (i, detector) in detectors.iter().enumerate() {
        let stats = detector.get_performance_stats();
        println!("   Detector {}: \"{}\"", i + 1, detector.config.wake_word);
        println!(
            "     • Confidence threshold: {:.2}",
            detector.config.confidence_threshold
        );
        println!(
            "     • Energy threshold: {:.2}",
            detector.config.energy_threshold
        );
        println!(
            "     • False positive reduction: {}",
            detector.config.false_positive_reduction
        );
        println!(
            "     • Energy efficient mode: {}",
            detector.config.energy_efficient_mode
        );

        if let Some(detection_time) = stats.get("last_detection_time_ms") {
            println!("     • Last detection time: {:.1}ms", detection_time);
        }
        if let Some(confidence) = stats.get("last_confidence") {
            println!("     • Last confidence: {:.2}", confidence);
        }
    }

    // Step 8: Energy efficiency analysis
    println!("\n⚡ Energy Efficiency Analysis:");

    let energy_efficient_detectors = detectors
        .iter()
        .filter(|d| d.config.energy_efficient_mode)
        .count();

    println!(
        "   • Energy efficient detectors: {}/{}",
        energy_efficient_detectors,
        detectors.len()
    );
    println!(
        "   • Energy savings estimate: {:.1}%",
        (energy_efficient_detectors as f32 / detectors.len() as f32) * 30.0
    ); // Estimated 30% savings

    // Simulate power consumption comparison
    let standard_power = 100.0; // mW
    let efficient_power = 70.0; // mW

    println!("   • Standard detection power: {:.0}mW", standard_power);
    println!("   • Efficient detection power: {:.0}mW", efficient_power);
    println!(
        "   • Power savings: {:.1}%",
        ((standard_power - efficient_power) / standard_power) * 100.0
    );

    // Step 9: Training recommendations
    println!("\n💡 Training Recommendations:");

    if false_positives > 0 {
        println!("   ⚠️ High False Positive Rate:");
        println!("   • Increase confidence threshold");
        println!("   • Enable false positive reduction");
        println!("   • Add more training examples");
        println!("   • Consider stricter energy filtering");
    }

    if false_negatives > 0 {
        println!("   ⚠️ High False Negative Rate:");
        println!("   • Decrease confidence threshold");
        println!("   • Improve audio quality");
        println!("   • Train with more accent variations");
        println!("   • Consider fuzzy matching improvements");
    }

    if accuracy > 90.0 {
        println!("   ✅ Excellent Performance:");
        println!("   • Current thresholds are well-tuned");
        println!("   • Consider enabling energy efficient mode");
        println!("   • Ready for production deployment");
    }

    // Step 10: Advanced features demonstration
    println!("\n🎯 Advanced Features Demonstration:");

    // Multi-wake-word support
    println!("   Multi-Wake-Word Support:");
    println!("   • {} different wake words configured", detectors.len());
    println!("   • Parallel detection capability");
    println!("   • Individual threshold tuning");

    // Adaptive thresholds
    println!("   Adaptive Threshold Suggestions:");
    for (i, detector) in detectors.iter().enumerate() {
        let suggested_threshold = if false_positives > false_negatives {
            detector.config.confidence_threshold + 0.05
        } else if false_negatives > false_positives {
            detector.config.confidence_threshold - 0.05
        } else {
            detector.config.confidence_threshold
        };

        println!(
            "   • Detector {}: {:.2} → {:.2}",
            i + 1,
            detector.config.confidence_threshold,
            suggested_threshold
        );
    }

    // Continuous learning potential
    println!("   Continuous Learning Potential:");
    println!("   • Collect detection feedback");
    println!("   • Adapt thresholds based on performance");
    println!("   • Personalize wake word sensitivity");
    println!("   • Update models with new training data");

    println!("\n✅ Custom wake word training demo completed!");
    println!("🎯 Key achievements:");
    println!("   • Multi-wake-word detection implemented");
    println!("   • Energy-efficient processing enabled");
    println!("   • False positive reduction active");
    println!("   • Performance metrics: {:.1}% accuracy", accuracy);

    println!("\n🚀 Next steps:");
    println!("   • Integrate with real-time audio streams");
    println!("   • Add voice biometric authentication");
    println!("   • Implement adaptive learning");
    println!("   • Deploy on edge devices");

    Ok(())
}

fn create_wake_word_test_scenarios() -> Vec<(String, AudioBuffer, usize)> {
    let mut scenarios = Vec::new();
    let sample_rate = 16000;

    // Scenario 1: Clear wake word
    let clear_wake_word = create_wake_word_audio(sample_rate, "hey assistant", 0.1, true);
    scenarios.push(("Clear_Wake_Word".to_string(), clear_wake_word, 1));

    // Scenario 2: Noisy wake word
    let noisy_wake_word = create_wake_word_audio(sample_rate, "hey assistant", 0.05, true);
    scenarios.push(("Noisy_Wake_Word".to_string(), noisy_wake_word, 1));

    // Scenario 3: Similar but not wake word
    let similar_phrase = create_wake_word_audio(sample_rate, "hey there", 0.1, false);
    scenarios.push(("Similar_Phrase".to_string(), similar_phrase, 0));

    // Scenario 4: Multiple wake words
    let multiple_wake_words = create_wake_word_audio(sample_rate, "computer activate", 0.08, true);
    scenarios.push(("Multiple_Wake_Words".to_string(), multiple_wake_words, 1));

    // Scenario 5: Background noise only
    let background_noise = create_wake_word_audio(sample_rate, "", 0.02, false);
    scenarios.push(("Background_Noise".to_string(), background_noise, 0));

    // Scenario 6: Partial wake word
    let partial_wake_word = create_wake_word_audio(sample_rate, "hey", 0.1, false);
    scenarios.push(("Partial_Wake_Word".to_string(), partial_wake_word, 0));

    // Scenario 7: Wake word with context
    let contextual_wake_word =
        create_wake_word_audio(sample_rate, "please hey assistant help", 0.09, true);
    scenarios.push(("Contextual_Wake_Word".to_string(), contextual_wake_word, 1));

    scenarios
}

fn create_wake_word_audio(
    sample_rate: u32,
    phrase: &str,
    amplitude: f32,
    is_wake_word: bool,
) -> AudioBuffer {
    let duration = if phrase.is_empty() {
        1.0
    } else {
        2.0 + phrase.len() as f32 * 0.1
    };
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        let mut sample = 0.0;

        if !phrase.is_empty() {
            // Create speech-like signal
            let base_freq = if is_wake_word { 150.0 } else { 180.0 };
            let freq_variation = 20.0 * (2.0 * t).sin();
            let f0 = base_freq + freq_variation;

            // Add formants for more realistic speech
            let formant1 = amplitude * 0.6 * (2.0 * std::f32::consts::PI * f0 * t).sin();
            let formant2 = amplitude * 0.3 * (2.0 * std::f32::consts::PI * f0 * 2.5 * t).sin();
            let formant3 = amplitude * 0.1 * (2.0 * std::f32::consts::PI * f0 * 3.5 * t).sin();

            // Add envelope to simulate word boundaries
            let envelope = 0.8 + 0.2 * (5.0 * t).sin();

            sample = (formant1 + formant2 + formant3) * envelope;
        }

        // Add background noise
        let noise = 0.01 * (i as f32 * 0.001).sin();
        sample += noise;

        samples.push(sample);
    }

    AudioBuffer::mono(samples, sample_rate)
}
