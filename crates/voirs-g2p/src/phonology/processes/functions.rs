//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PhonologicalProcessor;
use super::{
    find_similar_phoneme, get_features, has_feature, Deserialize, PhonologicalFeature,
    PhonologicalProcess, ProcessConfig, ProcessType, Serialize,
};
use crate::{G2pError, LanguageCode, Phoneme, Result};
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_place_assimilation() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnUs);
        let input = vec![
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("n".to_string()),
            Phoneme::new("p".to_string()),
            Phoneme::new("ʊ".to_string()),
            Phoneme::new("t".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[1].symbol, "m");
    }
    #[test]
    fn test_no_assimilation_when_not_needed() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnUs);
        let input = vec![
            Phoneme::new("k".to_string()),
            Phoneme::new("æ".to_string()),
            Phoneme::new("t".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].symbol, "k");
        assert_eq!(output[1].symbol, "æ");
        assert_eq!(output[2].symbol, "t");
    }
    #[test]
    fn test_process_config() {
        let config = ProcessConfig {
            enable_place_assimilation: true,
            enable_vowel_reduction: false,
            language: LanguageCode::EnUs,
            ..Default::default()
        };
        let processor = PhonologicalProcessor::with_config(config);
        assert_eq!(processor.processes.len(), 1);
    }
    #[test]
    fn test_vowel_reduction() {
        let mut config = ProcessConfig::default();
        config.enable_place_assimilation = false;
        config.enable_vowel_reduction = true;
        config.aggressiveness = 0.8;
        let processor = PhonologicalProcessor::with_config(config);
        let mut input = vec![
            Phoneme::new("ə".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("aʊ".to_string()),
            Phoneme::new("t".to_string()),
        ];
        input[2].stress = 0;
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "ə");
    }
    #[test]
    fn test_voicing_assimilation_german() {
        let mut config = ProcessConfig::default();
        config.enable_place_assimilation = false;
        config.enable_voicing_assimilation = true;
        config.language = LanguageCode::De;
        config.aggressiveness = 0.8;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("t".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("aʊ".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "d");
        assert_eq!(output[1].symbol, "b");
    }
    #[test]
    fn test_voicing_assimilation_not_in_english() {
        let mut config = ProcessConfig::default();
        config.enable_voicing_assimilation = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.8;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![Phoneme::new("t".to_string()), Phoneme::new("b".to_string())];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "t");
    }
    #[test]
    fn test_elision_schwa_deletion() {
        let mut config = ProcessConfig::default();
        config.enable_place_assimilation = false;
        config.enable_elision = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.9;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("k".to_string()),
            Phoneme::new("ə".to_string()),
            Phoneme::new("m".to_string()),
            Phoneme::new("ə".to_string()),
            Phoneme::new("r".to_string()),
            Phoneme::new("ə".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert!(output.len() < input.len());
    }
    #[test]
    fn test_elision_preserves_word_boundaries() {
        let mut config = ProcessConfig::default();
        config.enable_elision = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.9;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![Phoneme::new("ə".to_string())];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].symbol, "ə");
    }
    #[test]
    fn test_combined_processes() {
        let mut config = ProcessConfig::default();
        config.enable_place_assimilation = true;
        config.enable_vowel_reduction = true;
        config.enable_elision = false;
        config.aggressiveness = 0.8;
        let processor = PhonologicalProcessor::with_config(config);
        let mut input = vec![
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("n".to_string()),
            Phoneme::new("p".to_string()),
            Phoneme::new("ʊ".to_string()),
            Phoneme::new("t".to_string()),
        ];
        input[3].stress = 0;
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[1].symbol, "m");
        assert_eq!(output[3].symbol, "ə");
    }
    #[test]
    fn test_nasalization_french() {
        let mut config = ProcessConfig::default();
        config.enable_nasalization = true;
        config.language = LanguageCode::Fr;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![Phoneme::new("a".to_string()), Phoneme::new("n".to_string())];
        let output = processor.apply_all_processes(&input).unwrap();
        assert!(output[0].symbol.contains('̃'));
    }
    #[test]
    fn test_nasalization_portuguese() {
        let mut config = ProcessConfig::default();
        config.enable_nasalization = true;
        config.language = LanguageCode::Pt;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![Phoneme::new("o".to_string()), Phoneme::new("m".to_string())];
        let output = processor.apply_all_processes(&input).unwrap();
        assert!(output[0].symbol.contains('̃'));
    }
    #[test]
    fn test_nasalization_not_applied_english() {
        let mut config = ProcessConfig::default();
        config.enable_nasalization = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![Phoneme::new("a".to_string()), Phoneme::new("n".to_string())];
        let output = processor.apply_all_processes(&input).unwrap();
        assert!(!output[0].symbol.contains('̃'));
    }
    #[test]
    fn test_palatalization_japanese() {
        let mut config = ProcessConfig::default();
        config.enable_palatalization = true;
        config.language = LanguageCode::Ja;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![Phoneme::new("t".to_string()), Phoneme::new("i".to_string())];
        let output = processor.apply_all_processes(&input).unwrap();
        assert!(output[0].symbol.contains('ʲ') || output[0].symbol == "tʃ");
    }
    #[test]
    fn test_palatalization_before_front_vowel() {
        let mut config = ProcessConfig::default();
        config.enable_palatalization = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.7;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![Phoneme::new("s".to_string()), Phoneme::new("i".to_string())];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "ʃ");
    }
    #[test]
    fn test_palatalization_not_applied_back_vowel() {
        let mut config = ProcessConfig::default();
        config.enable_palatalization = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.7;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![Phoneme::new("t".to_string()), Phoneme::new("u".to_string())];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "t");
    }
    #[test]
    fn test_lenition_spanish_intervocalic() {
        let mut config = ProcessConfig::default();
        config.enable_lenition = true;
        config.language = LanguageCode::Es;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("o".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[1].symbol, "β");
    }
    #[test]
    fn test_lenition_spanish_d_to_eth() {
        let mut config = ProcessConfig::default();
        config.enable_lenition = true;
        config.language = LanguageCode::Es;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("e".to_string()),
            Phoneme::new("d".to_string()),
            Phoneme::new("a".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[1].symbol, "ð");
    }
    #[test]
    fn test_lenition_not_initial_position() {
        let mut config = ProcessConfig::default();
        config.enable_lenition = true;
        config.language = LanguageCode::Es;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![Phoneme::new("b".to_string()), Phoneme::new("a".to_string())];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "b");
    }
    #[test]
    fn test_fortition_italian_gemination() {
        let mut config = ProcessConfig::default();
        config.enable_fortition = true;
        config.language = LanguageCode::It;
        config.aggressiveness = 0.7;
        let processor = PhonologicalProcessor::with_config(config);
        let mut input = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("t".to_string()),
            Phoneme::new("o".to_string()),
        ];
        input[0].stress = 1;
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[1].symbol, "tt");
    }
    #[test]
    fn test_fortition_japanese_gemination() {
        let mut config = ProcessConfig::default();
        config.enable_fortition = true;
        config.language = LanguageCode::Ja;
        config.aggressiveness = 0.7;
        let processor = PhonologicalProcessor::with_config(config);
        let mut input = vec![
            Phoneme::new("i".to_string()),
            Phoneme::new("k".to_string()),
            Phoneme::new("a".to_string()),
        ];
        input[0].stress = 1;
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[1].symbol, "kk");
    }
    #[test]
    fn test_fortition_not_applied_unstressed() {
        let mut config = ProcessConfig::default();
        config.enable_fortition = true;
        config.language = LanguageCode::It;
        config.aggressiveness = 0.7;
        let processor = PhonologicalProcessor::with_config(config);
        let mut input = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("t".to_string()),
            Phoneme::new("o".to_string()),
        ];
        input[0].stress = 0;
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[1].symbol, "t");
    }
    #[test]
    fn test_multiple_new_processes_combined() {
        let mut config = ProcessConfig::default();
        config.enable_lenition = true;
        config.enable_palatalization = true;
        config.language = LanguageCode::Es;
        config.aggressiveness = 0.7;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("o".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[1].symbol, "β");
    }
    #[test]
    fn test_process_type_nasalization() {
        let processor = PhonologicalProcessor::new(LanguageCode::Fr);
        let input = vec![Phoneme::new("a".to_string()), Phoneme::new("n".to_string())];
        let result = processor.apply_process(&input, ProcessType::Nasalization);
        assert!(result.is_ok());
    }
    #[test]
    fn test_process_type_palatalization() {
        let processor = PhonologicalProcessor::new(LanguageCode::Ja);
        let input = vec![Phoneme::new("t".to_string()), Phoneme::new("i".to_string())];
        let result = processor.apply_process(&input, ProcessType::Palatalization);
        assert!(result.is_ok());
    }
    #[test]
    fn test_process_type_lenition() {
        let processor = PhonologicalProcessor::new(LanguageCode::Es);
        let input = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("b".to_string()),
            Phoneme::new("o".to_string()),
        ];
        let result = processor.apply_process(&input, ProcessType::Lenition);
        assert!(result.is_ok());
    }
    #[test]
    fn test_process_type_fortition() {
        let processor = PhonologicalProcessor::new(LanguageCode::It);
        let mut input = vec![
            Phoneme::new("a".to_string()),
            Phoneme::new("t".to_string()),
            Phoneme::new("o".to_string()),
        ];
        input[0].stress = 1;
        let result = processor.apply_process(&input, ProcessType::Fortition);
        assert!(result.is_ok());
    }
    #[test]
    fn test_r_dropping_british_english() {
        let mut config = ProcessConfig::default();
        config.enable_r_dropping = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("k".to_string()),
            Phoneme::new("ɑ".to_string()),
            Phoneme::new("r".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].symbol, "k");
        assert_eq!(output[1].symbol, "ɑ");
    }
    #[test]
    fn test_r_dropping_linking_r() {
        let mut config = ProcessConfig::default();
        config.enable_r_dropping = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("k".to_string()),
            Phoneme::new("ɑ".to_string()),
            Phoneme::new("r".to_string()),
            Phoneme::new("ɪ".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output.len(), 4);
        assert_eq!(output[2].symbol, "r");
    }
    #[test]
    fn test_r_dropping_not_applied_american() {
        let mut config = ProcessConfig::default();
        config.enable_r_dropping = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("k".to_string()),
            Phoneme::new("ɑ".to_string()),
            Phoneme::new("r".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[2].symbol, "r");
    }
    #[test]
    fn test_t_flapping_american_english() {
        let mut config = ProcessConfig::default();
        config.enable_t_flapping = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("b".to_string()),
            Phoneme::new("ɛ".to_string()),
            Phoneme::new("t".to_string()),
            Phoneme::new("ɚ".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "ɾ");
    }
    #[test]
    fn test_t_flapping_water() {
        let mut config = ProcessConfig::default();
        config.enable_t_flapping = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("w".to_string()),
            Phoneme::new("ɔ".to_string()),
            Phoneme::new("t".to_string()),
            Phoneme::new("ɚ".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "ɾ");
    }
    #[test]
    fn test_t_flapping_not_initial() {
        let mut config = ProcessConfig::default();
        config.enable_t_flapping = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("t".to_string()),
            Phoneme::new("ɔ".to_string()),
            Phoneme::new("p".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "t");
    }
    #[test]
    fn test_final_devoicing_german() {
        let mut config = ProcessConfig::default();
        config.enable_final_devoicing = true;
        config.language = LanguageCode::De;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("h".to_string()),
            Phoneme::new("ʊ".to_string()),
            Phoneme::new("n".to_string()),
            Phoneme::new("d".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[3].symbol, "t");
    }
    #[test]
    fn test_final_devoicing_tag() {
        let mut config = ProcessConfig::default();
        config.enable_final_devoicing = true;
        config.language = LanguageCode::De;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("t".to_string()),
            Phoneme::new("a".to_string()),
            Phoneme::new("g".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "k");
    }
    #[test]
    fn test_final_devoicing_not_medial() {
        let mut config = ProcessConfig::default();
        config.enable_final_devoicing = true;
        config.language = LanguageCode::De;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("b".to_string()),
            Phoneme::new("a".to_string()),
            Phoneme::new("d".to_string()),
            Phoneme::new("ə".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "d");
    }
    #[test]
    fn test_vowel_devoicing_japanese() {
        let mut config = ProcessConfig::default();
        config.enable_vowel_devoicing = true;
        config.language = LanguageCode::Ja;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("s".to_string()),
            Phoneme::new("ɯ".to_string()),
            Phoneme::new("k".to_string()),
            Phoneme::new("i".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert!(output[1].symbol.contains('̥'));
    }
    #[test]
    fn test_vowel_devoicing_desu() {
        let mut config = ProcessConfig::default();
        config.enable_vowel_devoicing = true;
        config.language = LanguageCode::Ja;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("d".to_string()),
            Phoneme::new("e".to_string()),
            Phoneme::new("s".to_string()),
            Phoneme::new("ɯ".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert!(output[3].symbol.contains('̥'));
    }
    #[test]
    fn test_vowel_devoicing_not_low_vowel() {
        let mut config = ProcessConfig::default();
        config.enable_vowel_devoicing = true;
        config.language = LanguageCode::Ja;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);
        let input = vec![
            Phoneme::new("s".to_string()),
            Phoneme::new("a".to_string()),
            Phoneme::new("k".to_string()),
            Phoneme::new("i".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert!(!output[1].symbol.contains('̥'));
    }
    #[test]
    fn test_process_type_r_dropping() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnGb);
        let input = vec![
            Phoneme::new("k".to_string()),
            Phoneme::new("ɑ".to_string()),
            Phoneme::new("r".to_string()),
        ];
        let result = processor.apply_process(&input, ProcessType::RDropping);
        assert!(result.is_ok());
    }
    #[test]
    fn test_process_type_t_flapping() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnUs);
        let input = vec![
            Phoneme::new("b".to_string()),
            Phoneme::new("ɛ".to_string()),
            Phoneme::new("t".to_string()),
            Phoneme::new("ɚ".to_string()),
        ];
        let result = processor.apply_process(&input, ProcessType::TFlapping);
        assert!(result.is_ok());
    }
    #[test]
    fn test_process_type_final_devoicing() {
        let processor = PhonologicalProcessor::new(LanguageCode::De);
        let input = vec![
            Phoneme::new("h".to_string()),
            Phoneme::new("ʊ".to_string()),
            Phoneme::new("n".to_string()),
            Phoneme::new("d".to_string()),
        ];
        let result = processor.apply_process(&input, ProcessType::FinalDevoicing);
        assert!(result.is_ok());
    }
    #[test]
    fn test_process_type_vowel_devoicing() {
        let processor = PhonologicalProcessor::new(LanguageCode::Ja);
        let input = vec![
            Phoneme::new("s".to_string()),
            Phoneme::new("ɯ".to_string()),
            Phoneme::new("k".to_string()),
            Phoneme::new("i".to_string()),
        ];
        let result = processor.apply_process(&input, ProcessType::VowelDevoicing);
        assert!(result.is_ok());
    }

    // ==================== NEW DIALECT-SPECIFIC PROCESS TESTS (6 processes) ====================

    // H-dropping tests
    #[test]
    fn test_h_dropping_british_english() {
        let mut config = ProcessConfig::default();
        config.enable_h_dropping = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);

        // "house" /haʊs/ → /aʊs/
        let input = vec![
            Phoneme::new("h".to_string()),
            Phoneme::new("aʊ".to_string()),
            Phoneme::new("s".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        // H should be dropped
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].symbol, "aʊ");
    }

    #[test]
    fn test_h_dropping_no_apply_american() {
        let mut config = ProcessConfig::default();
        config.enable_h_dropping = true;
        config.language = LanguageCode::EnUs; // American English - H-dropping shouldn't apply
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);

        let input = vec![
            Phoneme::new("h".to_string()),
            Phoneme::new("aʊ".to_string()),
            Phoneme::new("s".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        // H should NOT be dropped in American English
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].symbol, "h");
    }

    #[test]
    fn test_h_dropping_low_aggressiveness() {
        let mut config = ProcessConfig::default();
        config.enable_h_dropping = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.2; // Too low for H-dropping
        let processor = PhonologicalProcessor::with_config(config);

        let input = vec![
            Phoneme::new("h".to_string()),
            Phoneme::new("aʊ".to_string()),
            Phoneme::new("s".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        // H should NOT be dropped with low aggressiveness
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].symbol, "h");
    }

    // TH-fronting tests
    #[test]
    fn test_th_fronting_voiceless() {
        let mut config = ProcessConfig::default();
        config.enable_th_fronting = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);

        // "think" /θɪŋk/ → /fɪŋk/
        let input = vec![
            Phoneme::new("θ".to_string()),
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("ŋ".to_string()),
            Phoneme::new("k".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "f"); // θ → f
    }

    #[test]
    fn test_th_fronting_voiced() {
        let mut config = ProcessConfig::default();
        config.enable_th_fronting = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);

        // "this" /ðɪs/ → /vɪs/
        let input = vec![
            Phoneme::new("ð".to_string()),
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("s".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "v"); // ð → v
    }

    #[test]
    fn test_th_fronting_american_english() {
        let mut config = ProcessConfig::default();
        config.enable_th_fronting = true;
        config.language = LanguageCode::EnUs; // Also applies to some US varieties
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);

        let input = vec![
            Phoneme::new("θ".to_string()),
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("ŋ".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "f"); // θ → f
    }

    // L-vocalization tests
    #[test]
    fn test_l_vocalization_coda() {
        let mut config = ProcessConfig::default();
        config.enable_l_vocalization = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);

        // "milk" /mɪlk/ → /mɪwk/
        let input = vec![
            Phoneme::new("m".to_string()),
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("l".to_string()),
            Phoneme::new("k".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "w"); // l → w in coda position
    }

    #[test]
    fn test_l_vocalization_word_final() {
        let mut config = ProcessConfig::default();
        config.enable_l_vocalization = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);

        // "feel" /fiːl/ → /fiːw/
        let input = vec![
            Phoneme::new("f".to_string()),
            Phoneme::new("iː".to_string()),
            Phoneme::new("l".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "w"); // l → w word-finally
    }

    #[test]
    fn test_l_vocalization_onset_no_apply() {
        let mut config = ProcessConfig::default();
        config.enable_l_vocalization = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);

        // "light" /laɪt/ - onset /l/ should NOT vocalize
        let input = vec![
            Phoneme::new("l".to_string()),
            Phoneme::new("aɪ".to_string()),
            Phoneme::new("t".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "l"); // l stays /l/ in onset
    }

    // G-dropping tests
    #[test]
    fn test_g_dropping_unstressed() {
        let mut config = ProcessConfig::default();
        config.enable_g_dropping = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.7;
        let processor = PhonologicalProcessor::with_config(config);

        // "running" /rʌnɪŋ/ → /rʌnɪn/
        let mut input = vec![
            Phoneme::new("r".to_string()),
            Phoneme::new("ʌ".to_string()),
            Phoneme::new("n".to_string()),
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("ŋ".to_string()),
        ];
        input[4].stress = 0; // Mark as unstressed

        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[4].symbol, "n"); // ŋ → n in unstressed syllable
    }

    #[test]
    fn test_g_dropping_british_english() {
        let mut config = ProcessConfig::default();
        config.enable_g_dropping = true;
        config.language = LanguageCode::EnGb; // Also applies to British English
        config.aggressiveness = 0.7;
        let processor = PhonologicalProcessor::with_config(config);

        let mut input = vec![
            Phoneme::new("w".to_string()),
            Phoneme::new("ɔː".to_string()),
            Phoneme::new("k".to_string()),
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("ŋ".to_string()),
        ];
        input[4].stress = 0;

        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[4].symbol, "n"); // ŋ → n
    }

    #[test]
    fn test_g_dropping_low_aggressiveness() {
        let mut config = ProcessConfig::default();
        config.enable_g_dropping = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.4; // Too low for G-dropping
        let processor = PhonologicalProcessor::with_config(config);

        let mut input = vec![Phoneme::new("ŋ".to_string())];
        input[0].stress = 0;

        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[0].symbol, "ŋ"); // Should NOT drop with low aggressiveness
    }

    // T-glottaling tests
    #[test]
    fn test_t_glottaling_coda() {
        let mut config = ProcessConfig::default();
        config.enable_t_glottaling = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);

        // "butter" /bʌtə/ → /bʌʔə/
        let input = vec![
            Phoneme::new("b".to_string()),
            Phoneme::new("ʌ".to_string()),
            Phoneme::new("t".to_string()),
            Phoneme::new("ə".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "ʔ"); // t → ʔ
    }

    #[test]
    fn test_t_glottaling_word_final() {
        let mut config = ProcessConfig::default();
        config.enable_t_glottaling = true;
        config.language = LanguageCode::EnGb;
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);

        // "cat" /kæt/ → /kæʔ/
        let input = vec![
            Phoneme::new("k".to_string()),
            Phoneme::new("æ".to_string()),
            Phoneme::new("t".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "ʔ"); // t → ʔ word-finally
    }

    #[test]
    fn test_t_glottaling_no_apply_american() {
        let mut config = ProcessConfig::default();
        config.enable_t_glottaling = true;
        config.language = LanguageCode::EnUs; // Shouldn't apply in American English
        config.aggressiveness = 0.6;
        let processor = PhonologicalProcessor::with_config(config);

        let input = vec![
            Phoneme::new("k".to_string()),
            Phoneme::new("æ".to_string()),
            Phoneme::new("t".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output[2].symbol, "t"); // Should stay /t/ in American English
    }

    // Yod-coalescence tests
    #[test]
    fn test_yod_coalescence_tune() {
        let mut config = ProcessConfig::default();
        config.enable_yod_coalescence = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);

        // "tune" /tjuːn/ → /tʃuːn/
        let input = vec![
            Phoneme::new("t".to_string()),
            Phoneme::new("j".to_string()),
            Phoneme::new("uː".to_string()),
            Phoneme::new("n".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output.len(), 3); // tj coalesced into single segment
        assert_eq!(output[0].symbol, "tʃ"); // tj → tʃ
    }

    #[test]
    fn test_yod_coalescence_dune() {
        let mut config = ProcessConfig::default();
        config.enable_yod_coalescence = true;
        config.language = LanguageCode::EnUs;
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);

        // "dune" /djuːn/ → /dʒuːn/
        let input = vec![
            Phoneme::new("d".to_string()),
            Phoneme::new("j".to_string()),
            Phoneme::new("uː".to_string()),
            Phoneme::new("n".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        assert_eq!(output.len(), 3); // dj coalesced
        assert_eq!(output[0].symbol, "dʒ"); // dj → dʒ
    }

    #[test]
    fn test_yod_coalescence_no_apply_british() {
        let mut config = ProcessConfig::default();
        config.enable_yod_coalescence = true;
        config.language = LanguageCode::EnGb; // Less common in British English
        config.aggressiveness = 0.5;
        let processor = PhonologicalProcessor::with_config(config);

        let input = vec![
            Phoneme::new("t".to_string()),
            Phoneme::new("j".to_string()),
            Phoneme::new("uː".to_string()),
        ];
        let output = processor.apply_all_processes(&input).unwrap();
        // Should NOT coalesce in British English
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].symbol, "t");
        assert_eq!(output[1].symbol, "j");
    }

    // Process type validation tests for new processes
    #[test]
    fn test_process_type_h_dropping() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnGb);
        let input = vec![
            Phoneme::new("h".to_string()),
            Phoneme::new("aʊ".to_string()),
        ];
        let result = processor.apply_process(&input, ProcessType::HDropping);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_type_th_fronting() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnGb);
        let input = vec![Phoneme::new("θ".to_string()), Phoneme::new("ɪ".to_string())];
        let result = processor.apply_process(&input, ProcessType::ThFronting);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_type_l_vocalization() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnGb);
        let input = vec![
            Phoneme::new("m".to_string()),
            Phoneme::new("ɪ".to_string()),
            Phoneme::new("l".to_string()),
        ];
        let result = processor.apply_process(&input, ProcessType::LVocalization);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_type_g_dropping() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnUs);
        let input = vec![Phoneme::new("ŋ".to_string())];
        let result = processor.apply_process(&input, ProcessType::GDropping);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_type_t_glottaling() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnGb);
        let input = vec![
            Phoneme::new("b".to_string()),
            Phoneme::new("ʌ".to_string()),
            Phoneme::new("t".to_string()),
        ];
        let result = processor.apply_process(&input, ProcessType::TGlottaling);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_type_yod_coalescence() {
        let processor = PhonologicalProcessor::new(LanguageCode::EnUs);
        let input = vec![
            Phoneme::new("t".to_string()),
            Phoneme::new("j".to_string()),
            Phoneme::new("uː".to_string()),
        ];
        let result = processor.apply_process(&input, ProcessType::YodCoalescence);
        assert!(result.is_ok());
    }
}
