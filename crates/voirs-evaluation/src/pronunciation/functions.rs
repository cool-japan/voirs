//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use voirs_recognizer::traits::{AlignedPhoneme, PhonemeAlignment};
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, SyllablePosition};

use super::types::{EmotionalProsodicFeatures, EmotionalState, PronunciationEvaluatorImpl};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::PronunciationEvaluator;
    use voirs_sdk::AudioBuffer;
    #[tokio::test]
    async fn test_pronunciation_evaluator_creation() {
        let evaluator = PronunciationEvaluatorImpl::new().await.unwrap();
        assert!(!evaluator.supported_metrics().is_empty());
        assert_eq!(evaluator.metadata().name, "VoiRS Pronunciation Evaluator");
    }
    #[tokio::test]
    async fn test_pronunciation_evaluation() {
        let evaluator = PronunciationEvaluatorImpl::new().await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let text = "Hello world";
        let result = evaluator
            .evaluate_pronunciation(&audio, text, None)
            .await
            .unwrap();
        assert!(result.overall_score >= 0.0);
        assert!(result.overall_score <= 1.0);
        assert!(result.confidence > 0.0);
        assert!(result.fluency_score >= 0.0);
        assert!(result.rhythm_score >= 0.0);
    }
    #[tokio::test]
    async fn test_phoneme_similarity() {
        let evaluator = PronunciationEvaluatorImpl::new().await.unwrap();
        assert_eq!(evaluator.calculate_phoneme_similarity("a", "a"), 1.0);
        assert_eq!(evaluator.calculate_phoneme_similarity("p", "b"), 0.9);
        assert_eq!(evaluator.calculate_phoneme_similarity("a", "e"), 0.6);
        assert_eq!(evaluator.calculate_phoneme_similarity("a", "p"), 0.1);
    }
    #[tokio::test]
    async fn test_batch_pronunciation_evaluation() {
        let evaluator = PronunciationEvaluatorImpl::new().await.unwrap();
        let samples = vec![
            (
                AudioBuffer::new(vec![0.1; 8000], 16000, 1),
                "Hello".to_string(),
            ),
            (
                AudioBuffer::new(vec![0.2; 8000], 16000, 1),
                "World".to_string(),
            ),
        ];
        let results = evaluator
            .evaluate_pronunciation_batch(&samples, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.overall_score >= 0.0);
            assert!(result.overall_score <= 1.0);
        }
    }
    #[tokio::test]
    async fn test_emotional_prosody_analysis() {
        let evaluator = PronunciationEvaluatorImpl::new().await.unwrap();
        let alignment = PhonemeAlignment {
            phonemes: vec![
                AlignedPhoneme {
                    phoneme: Phoneme {
                        symbol: "h".to_string(),
                        ipa_symbol: "h".to_string(),
                        stress: 1,
                        syllable_position: voirs_sdk::types::SyllablePosition::Onset,
                        duration_ms: Some(80.0),
                        confidence: 0.9,
                    },
                    start_time: 0.0,
                    end_time: 0.08,
                    confidence: 0.9,
                },
                AlignedPhoneme {
                    phoneme: Phoneme {
                        symbol: "æ".to_string(),
                        ipa_symbol: "æ".to_string(),
                        stress: 2,
                        syllable_position: voirs_sdk::types::SyllablePosition::Nucleus,
                        duration_ms: Some(200.0),
                        confidence: 0.9,
                    },
                    start_time: 0.08,
                    end_time: 0.28,
                    confidence: 0.9,
                },
                AlignedPhoneme {
                    phoneme: Phoneme {
                        symbol: "p".to_string(),
                        ipa_symbol: "p".to_string(),
                        stress: 0,
                        syllable_position: voirs_sdk::types::SyllablePosition::Coda,
                        duration_ms: Some(60.0),
                        confidence: 0.9,
                    },
                    start_time: 0.28,
                    end_time: 0.34,
                    confidence: 0.9,
                },
            ],
            total_duration: 0.34,
            alignment_confidence: 0.9,
            word_alignments: vec![],
        };
        let emotional_score = evaluator
            .calculate_emotional_prosody(&alignment, Some(EmotionalState::Happy))
            .await
            .unwrap();
        assert!(emotional_score.emotional_appropriateness >= 0.0);
        assert!(emotional_score.emotional_appropriateness <= 1.0);
        assert!(emotional_score.emotional_intensity >= 0.0);
        assert!(emotional_score.emotional_intensity <= 1.0);
        assert!(emotional_score.emotional_consistency >= 0.0);
        assert!(emotional_score.emotional_consistency <= 1.0);
        assert!(emotional_score.confidence >= 0.0);
        assert!(emotional_score.confidence <= 1.0);
        assert!(emotional_score.prosodic_features.mean_f0 > 0.0);
        assert!(emotional_score.prosodic_features.speaking_rate > 0.0);
        assert!(emotional_score.prosodic_features.mean_energy >= 0.0);
        assert!(emotional_score.prosodic_features.mean_energy <= 1.0);
    }
    #[tokio::test]
    async fn test_emotion_detection() {
        let evaluator = PronunciationEvaluatorImpl::new().await.unwrap();
        let happy_features = EmotionalProsodicFeatures {
            mean_f0: 200.0,
            f0_std: 35.0,
            f0_range: 80.0,
            speaking_rate: 6.5,
            mean_energy: 0.8,
            energy_std: 0.15,
            pause_frequency: 0.3,
            pause_duration_mean: 0.2,
            jitter: 0.02,
            shimmer: 0.06,
            rhythm_regularity: 0.7,
            stress_pattern_strength: 0.8,
        };
        let detected = evaluator.detect_emotional_state(&happy_features).unwrap();
        assert!(detected == EmotionalState::Excited || detected == EmotionalState::Happy);
        let sad_features = EmotionalProsodicFeatures {
            mean_f0: 120.0,
            f0_std: 15.0,
            f0_range: 30.0,
            speaking_rate: 3.5,
            mean_energy: 0.2,
            energy_std: 0.05,
            pause_frequency: 0.8,
            pause_duration_mean: 0.8,
            jitter: 0.01,
            shimmer: 0.04,
            rhythm_regularity: 0.6,
            stress_pattern_strength: 0.4,
        };
        let detected = evaluator.detect_emotional_state(&sad_features).unwrap();
        assert_eq!(detected, EmotionalState::Sad);
    }
    #[tokio::test]
    async fn test_emotional_accuracy_calculation() {
        let evaluator = PronunciationEvaluatorImpl::new().await.unwrap();
        let accuracy =
            evaluator.calculate_emotional_accuracy(&EmotionalState::Happy, &EmotionalState::Happy);
        assert_eq!(accuracy, 1.0);
        let accuracy = evaluator
            .calculate_emotional_accuracy(&EmotionalState::Happy, &EmotionalState::Excited);
        assert!(accuracy >= 0.7);
        let accuracy =
            evaluator.calculate_emotional_accuracy(&EmotionalState::Happy, &EmotionalState::Sad);
        assert!(accuracy <= 0.2);
        let accuracy = evaluator
            .calculate_emotional_accuracy(&EmotionalState::Neutral, &EmotionalState::Happy);
        assert_eq!(accuracy, 0.5);
    }
    #[tokio::test]
    async fn test_prosodic_feature_extraction() {
        let evaluator = PronunciationEvaluatorImpl::new().await.unwrap();
        let alignment = PhonemeAlignment {
            phonemes: vec![
                AlignedPhoneme {
                    phoneme: Phoneme {
                        symbol: "a".to_string(),
                        ipa_symbol: "a".to_string(),
                        stress: 2,
                        syllable_position: voirs_sdk::types::SyllablePosition::Nucleus,
                        duration_ms: Some(150.0),
                        confidence: 0.9,
                    },
                    start_time: 0.0,
                    end_time: 0.15,
                    confidence: 0.9,
                },
                AlignedPhoneme {
                    phoneme: Phoneme {
                        symbol: "e".to_string(),
                        ipa_symbol: "e".to_string(),
                        stress: 1,
                        syllable_position: voirs_sdk::types::SyllablePosition::Nucleus,
                        duration_ms: Some(120.0),
                        confidence: 0.9,
                    },
                    start_time: 0.2,
                    end_time: 0.32,
                    confidence: 0.9,
                },
            ],
            total_duration: 0.32,
            alignment_confidence: 0.9,
            word_alignments: vec![],
        };
        let features = evaluator
            .extract_emotional_prosodic_features(&alignment)
            .await
            .unwrap();
        assert!(features.mean_f0 > 100.0 && features.mean_f0 < 300.0);
        assert!(features.f0_std >= 0.0);
        assert!(features.f0_range >= 0.0);
        assert!(features.speaking_rate > 0.0);
        assert!(features.mean_energy >= 0.0 && features.mean_energy <= 1.0);
        assert!(features.energy_std >= 0.0);
        assert!(features.pause_frequency >= 0.0);
        assert!(features.pause_duration_mean >= 0.0);
        assert!(features.jitter >= 0.0 && features.jitter <= 1.0);
        assert!(features.shimmer >= 0.0 && features.shimmer <= 1.0);
        assert!(features.rhythm_regularity >= 0.0 && features.rhythm_regularity <= 1.0);
        assert!(features.stress_pattern_strength >= 0.0 && features.stress_pattern_strength <= 1.0);
    }
    #[tokio::test]
    async fn test_emotional_dynamics() {
        let evaluator = PronunciationEvaluatorImpl::new().await.unwrap();
        let mut phonemes = Vec::new();
        for i in 0..10 {
            phonemes.push(AlignedPhoneme {
                phoneme: Phoneme {
                    symbol: if i % 2 == 0 { "a" } else { "t" }.to_string(),
                    ipa_symbol: if i % 2 == 0 { "a" } else { "t" }.to_string(),
                    stress: if i < 5 { 2 } else { 1 },
                    syllable_position: voirs_sdk::types::SyllablePosition::Nucleus,
                    duration_ms: Some(100.0 + i as f32 * 10.0),
                    confidence: 0.9,
                },
                start_time: i as f32 * 0.15,
                end_time: (i + 1) as f32 * 0.15,
                confidence: 0.9,
            });
        }
        let alignment = PhonemeAlignment {
            phonemes,
            total_duration: 1.5,
            alignment_confidence: 0.9,
            word_alignments: vec![],
        };
        let features = evaluator
            .extract_emotional_prosodic_features(&alignment)
            .await
            .unwrap();
        let dynamics = evaluator
            .analyze_emotional_dynamics(&alignment, &features)
            .await
            .unwrap();
        assert!(!dynamics.emotion_trajectory.is_empty());
        assert!(dynamics.emotional_stability >= 0.0 && dynamics.emotional_stability <= 1.0);
        assert!(dynamics.peak_intensity >= 0.0 && dynamics.peak_intensity <= 1.0);
        for (time, _, confidence) in &dynamics.emotion_trajectory {
            assert!(*time >= 0.0 && *time <= alignment.total_duration);
            assert!(*confidence >= 0.0 && *confidence <= 1.0);
        }
    }
}
