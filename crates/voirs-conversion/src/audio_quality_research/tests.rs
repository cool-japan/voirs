//! Tests for audio quality research module

#[cfg(test)]
mod audio_quality_research_tests {
    use super::super::config::ResearchConfig;
    use super::super::neural_model::NeuralQualityModel;
    use super::super::researcher::AudioQualityResearcher;

    #[test]
    fn test_research_config_creation() {
        let config = ResearchConfig::default();
        assert!(config.neural_models);
        assert_eq!(config.psychoacoustic_depth, 5);
        assert!(config.pesq_analysis);
        assert!(config.stoi_analysis);
        assert!(config.pemo_q_analysis);
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn test_research_config_builder() {
        let config = ResearchConfig::default()
            .with_neural_models(false)
            .with_psychoacoustic_depth(8)
            .with_sample_rate(44100);

        assert!(!config.neural_models);
        assert_eq!(config.psychoacoustic_depth, 8);
        assert_eq!(config.sample_rate, 44100);
    }

    #[test]
    fn test_audio_quality_researcher_creation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config);
        assert!(researcher.is_ok());
    }

    #[test]
    fn test_neural_quality_model_default() {
        let model = NeuralQualityModel::default();
        assert!(model.weights.contains_key("spectral_distortion"));
        assert!(model.weights.contains_key("temporal_coherence"));
        assert_eq!(model.hidden_layers.len(), 3);
    }

    #[test]
    fn test_comprehensive_analysis() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        let original = vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2];
        let processed = vec![0.09, 0.19, 0.29, 0.19, 0.09, 0.01, -0.09, -0.19];

        let result = researcher.comprehensive_analysis(&original, &processed, 16000);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.perceptual_quality >= 0.0 && analysis.perceptual_quality <= 1.0);
        assert!(analysis.neural_prediction >= 0.0 && analysis.neural_prediction <= 1.0);
        assert!(analysis.pesq_score >= 1.0 && analysis.pesq_score <= 5.0);
        assert!(analysis.stoi_score >= 0.0 && analysis.stoi_score <= 1.0);
        assert!(analysis.pemo_q_score >= 0.0 && analysis.pemo_q_score <= 1.0);
    }

    #[test]
    fn test_spectral_distortion_calculation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let original = vec![1.0, 0.5, 0.0, -0.5, -1.0];
        let processed = vec![0.9, 0.45, 0.0, -0.45, -0.9];

        let distortion = researcher.calculate_spectral_distortion(&original, &processed);
        assert!(distortion.is_ok());
        let distortion_value = distortion.unwrap();
        assert!(distortion_value > 0.0 && distortion_value < 1.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let c = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let correlation_perfect = researcher.calculate_correlation(&a, &b);
        let correlation_negative = researcher.calculate_correlation(&a, &c);

        assert!((correlation_perfect - 1.0).abs() < 1e-6);
        assert!(correlation_negative < 0.0);
    }

    #[test]
    fn test_temporal_coherence_calculation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let original = vec![0.1; 1024]; // Constant signal
        let processed = vec![0.1; 1024]; // Same signal

        let coherence = researcher.calculate_temporal_coherence(&original, &processed);
        assert!(coherence.is_ok());
        assert!(coherence.unwrap() > 0.9); // Should be very high for identical signals
    }

    #[test]
    fn test_envelope_calculation() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let audio = vec![0.5; 1000];
        let envelope = researcher.calculate_envelope(&audio);
        assert!(!envelope.is_empty());
        assert!(envelope.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_zero_crossing_rate() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let audio = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let zcr = researcher.calculate_zero_crossing_rate(&audio);
        assert!(zcr > 0.8); // High ZCR for alternating signal
    }

    #[test]
    fn test_spectral_flatness() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let audio = vec![1.0; 128]; // Constant signal (not flat spectrum)
        let flatness = researcher.calculate_spectral_flatness(&audio);
        assert!(flatness >= 0.0 && flatness <= 1.0);
    }

    #[test]
    fn test_magnitude_spectrum() {
        let config = ResearchConfig::default();
        let researcher = AudioQualityResearcher::new(config).unwrap();

        let audio = vec![1.0, 0.0, -1.0, 0.0]; // Simple sinusoid
        let spectrum = researcher.magnitude_spectrum(&audio);
        assert_eq!(spectrum.len(), audio.len() / 2 + 1);
        assert!(spectrum.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_empty_audio_handling() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        let empty_audio: Vec<f32> = vec![];
        let result = researcher.comprehensive_analysis(&empty_audio, &empty_audio, 16000);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_length_handling() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        let original = vec![0.1, 0.2, 0.3];
        let processed = vec![0.1, 0.2];

        let result = researcher.comprehensive_analysis(&original, &processed, 16000);
        assert!(result.is_err());
    }

    #[test]
    fn test_analysis_count_tracking() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        assert_eq!(researcher.get_analysis_count(), 0);

        let audio = vec![0.1; 1000];
        let _ = researcher.comprehensive_analysis(&audio, &audio, 16000);

        assert_eq!(researcher.get_analysis_count(), 1);
    }

    #[test]
    fn test_cache_functionality() {
        let config = ResearchConfig::default();
        let mut researcher = AudioQualityResearcher::new(config).unwrap();

        researcher.clear_cache();
        assert_eq!(researcher.analysis_cache.len(), 0);
    }

    // ── Psychoacoustic helpers (batch-17 real-DSP) ────────────────────────────

    /// Deterministic pure sine tone.
    fn pure_tone(freq_hz: f32, sample_rate: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    /// Deterministic white-ish noise from a fixed LCG seed (no rng crate).
    fn deterministic_noise(len: usize) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn researcher() -> AudioQualityResearcher {
        AudioQualityResearcher::new(ResearchConfig::default()).unwrap()
    }

    #[test]
    fn test_critical_bands_identical_is_zero() {
        let r = researcher();
        let audio = pure_tone(440.0, 16000.0, 1024);
        let analysis = r.analyze_critical_bands(&audio, &audio, 16000).unwrap();

        assert!(
            analysis.overall_distortion < 1e-4,
            "identical signals must have ~0 distortion, got {}",
            analysis.overall_distortion
        );
        // Preservation ratios should be ~1 for an unchanged signal.
        assert!((analysis.hf_preservation - 1.0).abs() < 1e-3);
        assert!((analysis.lf_preservation - 1.0).abs() < 1e-3);
        assert_eq!(analysis.band_deviations.len(), 24);
    }

    #[test]
    fn test_critical_bands_differ_for_different_inputs() {
        let r = researcher();
        let original = pure_tone(440.0, 16000.0, 1024);
        // A different-frequency tone redistributes band energy.
        let processed = pure_tone(2000.0, 16000.0, 1024);

        let analysis = r
            .analyze_critical_bands(&original, &processed, 16000)
            .unwrap();
        assert!(
            analysis.overall_distortion > 0.05,
            "different signals must show distortion, got {}",
            analysis.overall_distortion
        );
    }

    #[test]
    fn test_tonality_high_for_tone_low_for_noise() {
        let r = researcher();
        let tone = pure_tone(440.0, 16000.0, 1024);
        let noise = deterministic_noise(1024);

        let tone_tonality = r.analyze_tonality(&tone, &tone).unwrap();
        let noise_tonality = r.analyze_tonality(&noise, &noise).unwrap();

        assert!(
            tone_tonality.tonal_noise_ratio > noise_tonality.tonal_noise_ratio,
            "tone tonality {} should exceed noise tonality {}",
            tone_tonality.tonal_noise_ratio,
            noise_tonality.tonal_noise_ratio
        );
        assert!(
            tone_tonality.tonal_noise_ratio > 0.5,
            "pure tone should be highly tonal, got {}",
            tone_tonality.tonal_noise_ratio
        );
        // Identical inputs → near-perfect preservation.
        assert!(tone_tonality.tonal_preservation > 0.99);
        assert!(tone_tonality.spectral_peaks_preservation > 0.99);
    }

    #[test]
    fn test_sharpness_difference_zero_for_identical_positive_for_brighter() {
        let r = researcher();
        let audio = pure_tone(440.0, 16000.0, 1024);

        let same = r.calculate_sharpness_difference(&audio, &audio).unwrap();
        assert!(
            same < 1e-5,
            "identical signals must have ~0 sharpness difference, got {same}"
        );

        // Brighten by mixing in a high-frequency tone → higher sharpness.
        let high = pure_tone(6000.0, 16000.0, 1024);
        let brighter: Vec<f32> = audio
            .iter()
            .zip(high.iter())
            .map(|(&a, &h)| a + 0.8 * h)
            .collect();
        let diff = r.calculate_sharpness_difference(&audio, &brighter).unwrap();
        assert!(
            diff > 1e-3,
            "a brightened copy must have a positive sharpness difference, got {diff}"
        );
    }
}
