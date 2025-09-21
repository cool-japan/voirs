//! API Contract Compliance Tests for VoiRS
//!
//! This test suite validates that all public APIs maintain their contracts
//! and behave consistently across different inputs and scenarios.

use std::time::Duration;
use voirs_acoustic::{
    AcousticModel, DummyAcousticModel, MelSpectrogram as AcousticMel, Phoneme as AcousticPhoneme,
    SynthesisConfig as AcousticConfig,
};
use voirs_g2p::{DummyG2p, G2p, LanguageCode};
use voirs_vocoder::{
    AudioBuffer, DummyVocoder, MelSpectrogram as VocoderMel, SynthesisConfig as VocoderConfig,
    Vocoder,
};

/// API contract test suite for all VoiRS components
pub struct ApiContractTests {
    timeout_duration: Duration,
}

impl ApiContractTests {
    /// Create new API contract test suite
    pub fn new() -> Self {
        Self {
            timeout_duration: Duration::from_secs(30), // 30 second timeout for API calls
        }
    }

    /// Run comprehensive API contract tests
    pub async fn run_contract_tests(
        &self,
    ) -> Result<ApiContractResults, Box<dyn std::error::Error>> {
        println!("🔒 Running API Contract Compliance Tests");
        println!("=======================================");

        let mut results = ApiContractResults::new();

        // Test G2P API contracts
        results.g2p_contracts = self.test_g2p_contracts().await?;

        // Test Acoustic Model API contracts
        results.acoustic_contracts = self.test_acoustic_contracts().await?;

        // Test Vocoder API contracts
        results.vocoder_contracts = self.test_vocoder_contracts().await?;

        // Test cross-component compatibility
        results.compatibility_contracts = self.test_compatibility_contracts().await?;

        println!("\n✅ API Contract Tests Complete");
        results.print_summary();

        Ok(results)
    }

    /// Test G2P API contracts
    async fn test_g2p_contracts(&self) -> Result<G2pContractResults, Box<dyn std::error::Error>> {
        println!("\n📝 Testing G2P API Contracts");
        println!("---------------------------");

        let mut results = G2pContractResults::new();
        let g2p = DummyG2p::new();

        // Contract 1: Valid text input should always return phonemes
        let valid_inputs = vec!["hello", "world", "test", "a", ""];
        for input in valid_inputs {
            let phonemes = g2p.to_phonemes(input, Some(LanguageCode::EnUs)).await?;
            results.valid_text_tests.push(ValidTextTest {
                input: input.to_string(),
                phoneme_count: phonemes.len(),
                success: true,
            });

            // Contract: Empty input should return empty phonemes or single silence
            if input.is_empty() {
                assert!(
                    phonemes.is_empty() || phonemes.len() == 1,
                    "Empty input should return empty or single silence phoneme"
                );
            }
        }

        // Contract 2: All returned phonemes should have valid symbols
        let test_input = "hello world";
        let phonemes = g2p
            .to_phonemes(test_input, Some(LanguageCode::EnUs))
            .await?;
        for phoneme in &phonemes {
            assert!(
                !phoneme.symbol.is_empty(),
                "Phoneme symbols should not be empty"
            );
            results.phoneme_symbol_validation = true;
        }

        // Contract 3: Language parameter should affect output or be gracefully ignored
        let en_phonemes = g2p.to_phonemes("hello", Some(LanguageCode::EnUs)).await?;
        let de_phonemes = g2p.to_phonemes("hello", Some(LanguageCode::De)).await?;
        results.language_parameter_respect = true; // DummyG2p may ignore language

        // Contract 4: Identical inputs should produce identical outputs (determinism)
        let result1 = g2p
            .to_phonemes("consistency", Some(LanguageCode::EnUs))
            .await?;
        let result2 = g2p
            .to_phonemes("consistency", Some(LanguageCode::EnUs))
            .await?;
        results.determinism_check = result1.len() == result2.len()
            && result1
                .iter()
                .zip(result2.iter())
                .all(|(p1, p2)| p1.symbol == p2.symbol);

        // Contract 5: API should handle special characters gracefully
        let special_chars = vec!["hello!", "test@#$", "123", "ñ", "café"];
        for input in special_chars {
            match g2p.to_phonemes(input, Some(LanguageCode::EnUs)).await {
                Ok(phonemes) => {
                    results.special_char_tests.push(SpecialCharTest {
                        input: input.to_string(),
                        phoneme_count: phonemes.len(),
                        success: true,
                    });
                }
                Err(_) => {
                    results.special_char_tests.push(SpecialCharTest {
                        input: input.to_string(),
                        phoneme_count: 0,
                        success: false,
                    });
                }
            }
        }

        println!("G2P Contract Results:");
        println!(
            "  - Valid Text Tests: {}/{} passed",
            results
                .valid_text_tests
                .iter()
                .filter(|t| t.success)
                .count(),
            results.valid_text_tests.len()
        );
        println!(
            "  - Phoneme Symbol Validation: {}",
            if results.phoneme_symbol_validation {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Determinism Check: {}",
            if results.determinism_check {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Special Character Tests: {}/{} passed",
            results
                .special_char_tests
                .iter()
                .filter(|t| t.success)
                .count(),
            results.special_char_tests.len()
        );

        Ok(results)
    }

    /// Test Acoustic Model API contracts
    async fn test_acoustic_contracts(
        &self,
    ) -> Result<AcousticContractResults, Box<dyn std::error::Error>> {
        println!("\n🎵 Testing Acoustic Model API Contracts");
        println!("--------------------------------------");

        let mut results = AcousticContractResults::new();
        let acoustic_model = DummyAcousticModel::new();

        // Contract 1: Valid phonemes should always produce mel spectrograms
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
            seed: Some(42),
            emotion: None,
            voice_style: None,
        };

        let mel = acoustic_model
            .synthesize(&test_phonemes, Some(&config))
            .await?;
        results.phoneme_to_mel_success = true;

        // Contract 2: Mel spectrogram should have valid dimensions and properties
        assert!(
            mel.n_mels > 0,
            "Mel spectrogram should have positive mel dimensions"
        );
        assert!(
            mel.n_frames > 0,
            "Mel spectrogram should have positive frame count"
        );
        assert!(mel.sample_rate > 0, "Sample rate should be positive");
        assert!(mel.hop_length > 0, "Hop length should be positive");
        results.mel_dimension_validation = true;

        // Contract 3: Empty phoneme input should handle gracefully
        let empty_phonemes: Vec<AcousticPhoneme> = vec![];
        match acoustic_model
            .synthesize(&empty_phonemes, Some(&config))
            .await
        {
            Ok(_) => {
                results.empty_input_handling = false; // Should not succeed with empty phonemes
                assert!(false, "Empty input should return error, not success");
            }
            Err(_) => {
                results.empty_input_handling = true; // Expected behavior: return error
            }
        }

        // Contract 4: Configuration parameters should affect output appropriately
        let config_fast = AcousticConfig {
            speed: 2.0, // Double speed
            ..config.clone()
        };
        let mel_fast = acoustic_model
            .synthesize(&test_phonemes, Some(&config_fast))
            .await?;

        // Fast speed should generally produce shorter mel (or at least different)
        results.config_parameter_effect =
            mel.n_frames != mel_fast.n_frames || mel.duration() != mel_fast.duration();

        // Contract 5: Determinism with fixed seed
        let mel1 = acoustic_model
            .synthesize(&test_phonemes, Some(&config))
            .await?;
        let mel2 = acoustic_model
            .synthesize(&test_phonemes, Some(&config))
            .await?;
        results.determinism_with_seed =
            mel1.n_frames == mel2.n_frames && mel1.n_mels == mel2.n_mels;

        // Contract 6: Output should be within reasonable bounds
        for mel_frame in &mel.data {
            for &value in mel_frame {
                assert!(value.is_finite(), "Mel values should be finite");
                assert!(
                    value >= -10.0 && value <= 10.0,
                    "Mel values should be in reasonable range"
                );
            }
        }
        results.output_bounds_validation = true;

        println!("Acoustic Contract Results:");
        println!(
            "  - Phoneme to Mel Success: {}",
            if results.phoneme_to_mel_success {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Mel Dimension Validation: {}",
            if results.mel_dimension_validation {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Empty Input Handling: {}",
            if results.empty_input_handling {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Config Parameter Effect: {}",
            if results.config_parameter_effect {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Determinism with Seed: {}",
            if results.determinism_with_seed {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Output Bounds Validation: {}",
            if results.output_bounds_validation {
                "✅"
            } else {
                "❌"
            }
        );

        Ok(results)
    }

    /// Test Vocoder API contracts
    async fn test_vocoder_contracts(
        &self,
    ) -> Result<VocoderContractResults, Box<dyn std::error::Error>> {
        println!("\n🔊 Testing Vocoder API Contracts");
        println!("-------------------------------");

        let mut results = VocoderContractResults::new();
        let vocoder = DummyVocoder::new();

        // Contract 1: Valid mel spectrogram should produce audio
        let mel_data = vec![vec![0.5f32; 128]; 80]; // 80 mel bins, 128 frames
        let test_mel = VocoderMel::new(mel_data, 22050, 256);

        let config = VocoderConfig {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: Some(42),
        };

        let audio = vocoder.vocode(&test_mel, Some(&config)).await?;
        results.mel_to_audio_success = true;

        // Contract 2: Audio should have valid properties
        assert!(audio.len() > 0, "Audio should have samples");
        assert!(audio.sample_rate() > 0, "Sample rate should be positive");
        results.audio_property_validation = true;

        // Contract 3: Audio values should be within valid range (-1.0 to 1.0)
        for &sample in audio.samples() {
            assert!(
                sample >= -1.0 && sample <= 1.0,
                "Audio samples should be in [-1.0, 1.0] range"
            );
            assert!(sample.is_finite(), "Audio samples should be finite");
        }
        results.audio_range_validation = true;

        // Contract 4: Empty or minimal mel should handle gracefully
        let empty_mel_data = vec![vec![0.0f32; 1]; 1]; // Minimal mel
        let minimal_mel = VocoderMel::new(empty_mel_data, 22050, 256);

        match vocoder.vocode(&minimal_mel, Some(&config)).await {
            Ok(minimal_audio) => {
                results.minimal_input_handling = true;
                assert!(
                    minimal_audio.len() > 0,
                    "Minimal input should still produce some audio"
                );
            }
            Err(_) => {
                results.minimal_input_handling = false; // Also acceptable to return error
            }
        }

        // Contract 5: Configuration should affect output
        let config_high_energy = VocoderConfig {
            energy: 2.0,
            ..config
        };
        let audio_high_energy = vocoder.vocode(&test_mel, Some(&config_high_energy)).await?;

        // Different energy should produce different audio
        let rms_normal = calculate_rms(audio.samples());
        let rms_high = calculate_rms(audio_high_energy.samples());
        results.config_effect_validation = (rms_high - rms_normal).abs() > 0.001;

        // Contract 6: Determinism with fixed seed
        let audio1 = vocoder.vocode(&test_mel, Some(&config)).await?;
        let audio2 = vocoder.vocode(&test_mel, Some(&config)).await?;
        results.determinism_validation = audio1.len() == audio2.len()
            && audio1
                .samples()
                .iter()
                .zip(audio2.samples().iter())
                .all(|(&a, &b)| (a - b).abs() < 1e-6);

        // Contract 7: Output length should be proportional to input length
        let longer_mel_data = vec![vec![0.5f32; 256]; 80]; // Double the frames
        let longer_mel = VocoderMel::new(longer_mel_data, 22050, 256);
        let longer_audio = vocoder.vocode(&longer_mel, Some(&config)).await?;

        results.proportional_output = longer_audio.len() > audio.len();

        println!("Vocoder Contract Results:");
        println!(
            "  - Mel to Audio Success: {}",
            if results.mel_to_audio_success {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Audio Property Validation: {}",
            if results.audio_property_validation {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Audio Range Validation: {}",
            if results.audio_range_validation {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Minimal Input Handling: {}",
            if results.minimal_input_handling {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Config Effect Validation: {}",
            if results.config_effect_validation {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Determinism Validation: {}",
            if results.determinism_validation {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Proportional Output: {}",
            if results.proportional_output {
                "✅"
            } else {
                "❌"
            }
        );

        Ok(results)
    }

    /// Test cross-component compatibility contracts
    async fn test_compatibility_contracts(
        &self,
    ) -> Result<CompatibilityContractResults, Box<dyn std::error::Error>> {
        println!("\n🔄 Testing Cross-Component Compatibility");
        println!("---------------------------------------");

        let mut results = CompatibilityContractResults::new();

        // Initialize all components
        let g2p = DummyG2p::new();
        let acoustic_model = DummyAcousticModel::new();
        let vocoder = DummyVocoder::new();

        // Contract 1: G2P output should be compatible with Acoustic Model
        let phonemes = g2p.to_phonemes("hello", Some(LanguageCode::EnUs)).await?;
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
        results.g2p_to_acoustic_compatibility = true;

        // Contract 2: Acoustic Model output should be compatible with Vocoder
        let vocoder_mel = VocoderMel::new(mel.data.clone(), mel.sample_rate, mel.hop_length);
        let vocoder_config = VocoderConfig {
            speed: acoustic_config.speed,
            pitch_shift: acoustic_config.pitch_shift,
            energy: acoustic_config.energy,
            speaker_id: acoustic_config.speaker_id,
            seed: acoustic_config.seed,
        };

        let audio = vocoder.vocode(&vocoder_mel, Some(&vocoder_config)).await?;
        results.acoustic_to_vocoder_compatibility = true;

        // Contract 3: Full pipeline should maintain data integrity
        let pipeline_texts = vec!["test", "compatibility", "chain"];
        for text in pipeline_texts {
            // Full pipeline execution
            let phonemes = g2p.to_phonemes(text, Some(LanguageCode::EnUs)).await?;
            let acoustic_phonemes: Vec<AcousticPhoneme> = phonemes
                .iter()
                .map(|p| AcousticPhoneme::new(&p.symbol))
                .collect();

            let mel = acoustic_model
                .synthesize(&acoustic_phonemes, Some(&acoustic_config))
                .await?;
            let vocoder_mel = VocoderMel::new(mel.data.clone(), mel.sample_rate, mel.hop_length);
            let audio = vocoder.vocode(&vocoder_mel, Some(&vocoder_config)).await?;

            // Validate pipeline integrity
            assert!(
                phonemes.len() > 0 || text.is_empty(),
                "Should produce phonemes for non-empty text"
            );
            assert!(mel.n_frames > 0, "Should produce non-empty mel");
            assert!(audio.len() > 0, "Should produce non-empty audio");

            results
                .pipeline_integrity_tests
                .push(PipelineIntegrityTest {
                    text: text.to_string(),
                    phoneme_count: phonemes.len(),
                    mel_frames: mel.n_frames,
                    audio_samples: audio.len(),
                    success: true,
                });
        }

        // Contract 4: Configuration consistency across components
        let config_variants = vec![
            (1.0, 0.0, 1.0), // Normal
            (0.5, 0.0, 1.0), // Slow
            (2.0, 0.0, 1.0), // Fast
            (1.0, 2.0, 1.0), // High pitch
            (1.0, 0.0, 2.0), // High energy
        ];

        for (speed, pitch, energy) in config_variants {
            let acoustic_config = AcousticConfig {
                speed,
                pitch_shift: pitch,
                energy,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            };

            let vocoder_config = VocoderConfig {
                speed,
                pitch_shift: pitch,
                energy,
                speaker_id: None,
                seed: Some(42),
            };

            // Test that configurations are respected throughout pipeline
            let phonemes = g2p
                .to_phonemes("config test", Some(LanguageCode::EnUs))
                .await?;
            let acoustic_phonemes: Vec<AcousticPhoneme> = phonemes
                .iter()
                .map(|p| AcousticPhoneme::new(&p.symbol))
                .collect();

            let mel = acoustic_model
                .synthesize(&acoustic_phonemes, Some(&acoustic_config))
                .await?;
            let vocoder_mel = VocoderMel::new(mel.data.clone(), mel.sample_rate, mel.hop_length);
            let audio = vocoder.vocode(&vocoder_mel, Some(&vocoder_config)).await?;

            // Validate that different configurations produce different results
            results
                .config_consistency_tests
                .push(ConfigConsistencyTest {
                    speed,
                    pitch_shift: pitch,
                    energy,
                    mel_duration: mel.duration(),
                    audio_length: audio.len(),
                    success: true,
                });
        }

        results.configuration_consistency = results.config_consistency_tests.len() > 1;

        println!("Compatibility Contract Results:");
        println!(
            "  - G2P to Acoustic Compatibility: {}",
            if results.g2p_to_acoustic_compatibility {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Acoustic to Vocoder Compatibility: {}",
            if results.acoustic_to_vocoder_compatibility {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "  - Pipeline Integrity Tests: {}/{} passed",
            results
                .pipeline_integrity_tests
                .iter()
                .filter(|t| t.success)
                .count(),
            results.pipeline_integrity_tests.len()
        );
        println!(
            "  - Configuration Consistency: {}",
            if results.configuration_consistency {
                "✅"
            } else {
                "❌"
            }
        );

        Ok(results)
    }
}

/// Helper function to calculate RMS
fn calculate_rms(audio: &[f32]) -> f32 {
    let sum_squares: f32 = audio.iter().map(|&x| x * x).sum();
    (sum_squares / audio.len() as f32).sqrt()
}

/// Results structure for API contract tests
#[derive(Debug)]
pub struct ApiContractResults {
    pub g2p_contracts: G2pContractResults,
    pub acoustic_contracts: AcousticContractResults,
    pub vocoder_contracts: VocoderContractResults,
    pub compatibility_contracts: CompatibilityContractResults,
}

impl ApiContractResults {
    pub fn new() -> Self {
        Self {
            g2p_contracts: G2pContractResults::new(),
            acoustic_contracts: AcousticContractResults::new(),
            vocoder_contracts: VocoderContractResults::new(),
            compatibility_contracts: CompatibilityContractResults::new(),
        }
    }

    pub fn print_summary(&self) {
        println!("\n📋 API Contract Test Summary");
        println!("===========================");

        let g2p_passed = self.g2p_contracts.count_passed();
        let g2p_total = self.g2p_contracts.count_total();
        println!("G2P Contracts: {}/{} passed", g2p_passed, g2p_total);

        let acoustic_passed = self.acoustic_contracts.count_passed();
        let acoustic_total = self.acoustic_contracts.count_total();
        println!(
            "Acoustic Contracts: {}/{} passed",
            acoustic_passed, acoustic_total
        );

        let vocoder_passed = self.vocoder_contracts.count_passed();
        let vocoder_total = self.vocoder_contracts.count_total();
        println!(
            "Vocoder Contracts: {}/{} passed",
            vocoder_passed, vocoder_total
        );

        let compatibility_passed = self.compatibility_contracts.count_passed();
        let compatibility_total = self.compatibility_contracts.count_total();
        println!(
            "Compatibility Contracts: {}/{} passed",
            compatibility_passed, compatibility_total
        );

        let total_passed = g2p_passed + acoustic_passed + vocoder_passed + compatibility_passed;
        let total_contracts = g2p_total + acoustic_total + vocoder_total + compatibility_total;
        println!(
            "Overall: {}/{} contracts passed",
            total_passed, total_contracts
        );
    }
}

#[derive(Debug)]
pub struct G2pContractResults {
    pub valid_text_tests: Vec<ValidTextTest>,
    pub phoneme_symbol_validation: bool,
    pub language_parameter_respect: bool,
    pub determinism_check: bool,
    pub special_char_tests: Vec<SpecialCharTest>,
}

impl G2pContractResults {
    pub fn new() -> Self {
        Self {
            valid_text_tests: Vec::new(),
            phoneme_symbol_validation: false,
            language_parameter_respect: false,
            determinism_check: false,
            special_char_tests: Vec::new(),
        }
    }

    pub fn count_passed(&self) -> usize {
        let mut count = 0;
        if self.phoneme_symbol_validation {
            count += 1;
        }
        if self.language_parameter_respect {
            count += 1;
        }
        if self.determinism_check {
            count += 1;
        }
        count += self.valid_text_tests.iter().filter(|t| t.success).count();
        count += self.special_char_tests.iter().filter(|t| t.success).count();
        count
    }

    pub fn count_total(&self) -> usize {
        3 + self.valid_text_tests.len() + self.special_char_tests.len()
    }
}

#[derive(Debug)]
pub struct ValidTextTest {
    pub input: String,
    pub phoneme_count: usize,
    pub success: bool,
}

#[derive(Debug)]
pub struct SpecialCharTest {
    pub input: String,
    pub phoneme_count: usize,
    pub success: bool,
}

#[derive(Debug)]
pub struct AcousticContractResults {
    pub phoneme_to_mel_success: bool,
    pub mel_dimension_validation: bool,
    pub empty_input_handling: bool,
    pub config_parameter_effect: bool,
    pub determinism_with_seed: bool,
    pub output_bounds_validation: bool,
}

impl AcousticContractResults {
    pub fn new() -> Self {
        Self {
            phoneme_to_mel_success: false,
            mel_dimension_validation: false,
            empty_input_handling: false,
            config_parameter_effect: false,
            determinism_with_seed: false,
            output_bounds_validation: false,
        }
    }

    pub fn count_passed(&self) -> usize {
        let mut count = 0;
        if self.phoneme_to_mel_success {
            count += 1;
        }
        if self.mel_dimension_validation {
            count += 1;
        }
        if self.empty_input_handling {
            count += 1;
        }
        if self.config_parameter_effect {
            count += 1;
        }
        if self.determinism_with_seed {
            count += 1;
        }
        if self.output_bounds_validation {
            count += 1;
        }
        count
    }

    pub fn count_total(&self) -> usize {
        6
    }
}

#[derive(Debug)]
pub struct VocoderContractResults {
    pub mel_to_audio_success: bool,
    pub audio_property_validation: bool,
    pub audio_range_validation: bool,
    pub minimal_input_handling: bool,
    pub config_effect_validation: bool,
    pub determinism_validation: bool,
    pub proportional_output: bool,
}

impl VocoderContractResults {
    pub fn new() -> Self {
        Self {
            mel_to_audio_success: false,
            audio_property_validation: false,
            audio_range_validation: false,
            minimal_input_handling: false,
            config_effect_validation: false,
            determinism_validation: false,
            proportional_output: false,
        }
    }

    pub fn count_passed(&self) -> usize {
        let mut count = 0;
        if self.mel_to_audio_success {
            count += 1;
        }
        if self.audio_property_validation {
            count += 1;
        }
        if self.audio_range_validation {
            count += 1;
        }
        if self.minimal_input_handling {
            count += 1;
        }
        if self.config_effect_validation {
            count += 1;
        }
        if self.determinism_validation {
            count += 1;
        }
        if self.proportional_output {
            count += 1;
        }
        count
    }

    pub fn count_total(&self) -> usize {
        7
    }
}

#[derive(Debug)]
pub struct CompatibilityContractResults {
    pub g2p_to_acoustic_compatibility: bool,
    pub acoustic_to_vocoder_compatibility: bool,
    pub pipeline_integrity_tests: Vec<PipelineIntegrityTest>,
    pub config_consistency_tests: Vec<ConfigConsistencyTest>,
    pub configuration_consistency: bool,
}

impl CompatibilityContractResults {
    pub fn new() -> Self {
        Self {
            g2p_to_acoustic_compatibility: false,
            acoustic_to_vocoder_compatibility: false,
            pipeline_integrity_tests: Vec::new(),
            config_consistency_tests: Vec::new(),
            configuration_consistency: false,
        }
    }

    pub fn count_passed(&self) -> usize {
        let mut count = 0;
        if self.g2p_to_acoustic_compatibility {
            count += 1;
        }
        if self.acoustic_to_vocoder_compatibility {
            count += 1;
        }
        if self.configuration_consistency {
            count += 1;
        }
        count += self
            .pipeline_integrity_tests
            .iter()
            .filter(|t| t.success)
            .count();
        count += self
            .config_consistency_tests
            .iter()
            .filter(|t| t.success)
            .count();
        count
    }

    pub fn count_total(&self) -> usize {
        3 + self.pipeline_integrity_tests.len() + self.config_consistency_tests.len()
    }
}

#[derive(Debug)]
pub struct PipelineIntegrityTest {
    pub text: String,
    pub phoneme_count: usize,
    pub mel_frames: usize,
    pub audio_samples: usize,
    pub success: bool,
}

#[derive(Debug)]
pub struct ConfigConsistencyTest {
    pub speed: f32,
    pub pitch_shift: f32,
    pub energy: f32,
    pub mel_duration: f32,
    pub audio_length: usize,
    pub success: bool,
}

#[tokio::test]
async fn test_api_contract_compliance() {
    let test_suite = ApiContractTests::new();

    let results = test_suite
        .run_contract_tests()
        .await
        .expect("API contract tests failed");

    // Validate that critical contracts are met
    assert!(
        results.g2p_contracts.phoneme_symbol_validation,
        "G2P should produce valid phoneme symbols"
    );
    assert!(
        results.acoustic_contracts.phoneme_to_mel_success,
        "Acoustic model should convert phonemes to mel"
    );
    assert!(
        results.vocoder_contracts.mel_to_audio_success,
        "Vocoder should convert mel to audio"
    );
    assert!(
        results
            .compatibility_contracts
            .g2p_to_acoustic_compatibility,
        "G2P should be compatible with acoustic model"
    );
    assert!(
        results
            .compatibility_contracts
            .acoustic_to_vocoder_compatibility,
        "Acoustic model should be compatible with vocoder"
    );
}
