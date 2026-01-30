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
}
