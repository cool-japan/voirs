use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::types::SynthesisConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ModalityType {
    Text,
    Audio,
    Visual,
    Gesture,
    Facial,
    Prosody,
    Contextual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityConfig {
    pub modality_type: ModalityType,
    pub weight: f32, // 0.0 to 1.0
    pub synchronization_offset_ms: i32,
    pub duration_ms: Option<u32>,
    pub adaptive_weighting: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalAlignment {
    pub text_audio_alignment: f32,
    pub visual_audio_alignment: f32,
    pub gesture_audio_alignment: f32,
    pub prosody_text_alignment: f32,
    pub alignment_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveWeighting {
    pub confidence_threshold: f32,
    pub weight_adjustment_factor: f32,
    pub minimum_weight: f32,
    pub maximum_weight: f32,
    pub adaptation_speed: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalSynthesisConfig {
    pub base_config: SynthesisConfig,
    pub modality_configs: Vec<ModalityConfig>,
    pub cross_modal_alignment: CrossModalAlignment,
    pub adaptive_weighting: Option<AdaptiveWeighting>,
    pub synchronization_tolerance_ms: u32,
    pub fallback_modality: ModalityType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityData {
    pub modality_type: ModalityType,
    pub data: Vec<u8>,
    pub timestamp_ms: u64,
    pub confidence: f32,
    pub metadata: HashMap<String, String>,
}

pub struct MultimodalSynthesizer {
    current_modalities: HashMap<ModalityType, ModalityData>,
    modality_weights: HashMap<ModalityType, f32>,
    alignment_history: Vec<CrossModalAlignment>,
    adaptive_config: Option<AdaptiveWeighting>,
}

impl MultimodalSynthesizer {
    pub fn new() -> Self {
        Self {
            current_modalities: HashMap::new(),
            modality_weights: HashMap::new(),
            alignment_history: Vec::new(),
            adaptive_config: None,
        }
    }

    pub fn with_adaptive_weighting(mut self, config: AdaptiveWeighting) -> Self {
        self.adaptive_config = Some(config);
        self
    }

    pub fn add_modality_data(&mut self, modality_data: ModalityData) {
        let modality_type = modality_data.modality_type.clone();
        self.current_modalities.insert(modality_type, modality_data);
    }

    pub fn update_modality_weights(&mut self, weights: HashMap<ModalityType, f32>) {
        for (modality, weight) in weights {
            let clamped_weight = weight.clamp(0.0, 1.0);
            self.modality_weights.insert(modality, clamped_weight);
        }
    }

    pub fn calculate_cross_modal_alignment(&self) -> CrossModalAlignment {
        let text_audio = self.calculate_alignment_score(&ModalityType::Text, &ModalityType::Audio);
        let visual_audio =
            self.calculate_alignment_score(&ModalityType::Visual, &ModalityType::Audio);
        let gesture_audio =
            self.calculate_alignment_score(&ModalityType::Gesture, &ModalityType::Audio);
        let prosody_text =
            self.calculate_alignment_score(&ModalityType::Prosody, &ModalityType::Text);

        CrossModalAlignment {
            text_audio_alignment: text_audio,
            visual_audio_alignment: visual_audio,
            gesture_audio_alignment: gesture_audio,
            prosody_text_alignment: prosody_text,
            alignment_threshold: 0.7,
        }
    }

    fn calculate_alignment_score(&self, modality1: &ModalityType, modality2: &ModalityType) -> f32 {
        let data1 = self.current_modalities.get(modality1);
        let data2 = self.current_modalities.get(modality2);

        match (data1, data2) {
            (Some(d1), Some(d2)) => {
                let time_diff = (d1.timestamp_ms as i64 - d2.timestamp_ms as i64).abs();
                let confidence_product = d1.confidence * d2.confidence;
                let time_score = 1.0 - (time_diff as f32 / 1000.0).min(1.0);

                time_score * confidence_product
            }
            _ => 0.0,
        }
    }

    pub fn apply_adaptive_weighting(&mut self) {
        if let Some(config) = &self.adaptive_config {
            let alignment = self.calculate_cross_modal_alignment();

            for (modality_type, current_weight) in self.modality_weights.iter_mut() {
                let alignment_score = match modality_type {
                    ModalityType::Text => alignment.text_audio_alignment,
                    ModalityType::Audio => 1.0, // Audio is reference
                    ModalityType::Visual => alignment.visual_audio_alignment,
                    ModalityType::Gesture => alignment.gesture_audio_alignment,
                    ModalityType::Prosody => alignment.prosody_text_alignment,
                    _ => 0.5,
                };

                if alignment_score > config.confidence_threshold {
                    *current_weight = (*current_weight
                        + config.weight_adjustment_factor * alignment_score)
                        .clamp(config.minimum_weight, config.maximum_weight);
                } else {
                    *current_weight = (*current_weight
                        - config.weight_adjustment_factor * (1.0 - alignment_score))
                        .clamp(config.minimum_weight, config.maximum_weight);
                }
            }
        }
    }

    pub fn create_multimodal_synthesis_config(
        &self,
        base_config: SynthesisConfig,
        modality_configs: Vec<ModalityConfig>,
    ) -> MultimodalSynthesisConfig {
        let alignment = self.calculate_cross_modal_alignment();

        MultimodalSynthesisConfig {
            base_config,
            modality_configs,
            cross_modal_alignment: alignment,
            adaptive_weighting: self.adaptive_config.clone(),
            synchronization_tolerance_ms: 100,
            fallback_modality: ModalityType::Text,
        }
    }

    pub fn synchronize_modalities(&mut self, target_timestamp_ms: u64) {
        let tolerance_ms = 100;

        for (_, modality_data) in self.current_modalities.iter_mut() {
            let time_diff = (modality_data.timestamp_ms as i64 - target_timestamp_ms as i64).abs();

            if time_diff > tolerance_ms {
                modality_data.timestamp_ms = target_timestamp_ms;
            }
        }
    }

    pub fn get_active_modalities(&self) -> Vec<ModalityType> {
        self.current_modalities.keys().cloned().collect()
    }

    pub fn get_modality_weight(&self, modality: &ModalityType) -> f32 {
        self.modality_weights.get(modality).copied().unwrap_or(1.0)
    }

    pub fn clear_modality_data(&mut self) {
        self.current_modalities.clear();
    }
}

impl Default for MultimodalSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ModalityConfig {
    fn default() -> Self {
        Self {
            modality_type: ModalityType::Text,
            weight: 1.0,
            synchronization_offset_ms: 0,
            duration_ms: None,
            adaptive_weighting: false,
        }
    }
}

impl Default for CrossModalAlignment {
    fn default() -> Self {
        Self {
            text_audio_alignment: 0.8,
            visual_audio_alignment: 0.7,
            gesture_audio_alignment: 0.6,
            prosody_text_alignment: 0.9,
            alignment_threshold: 0.7,
        }
    }
}

impl Default for AdaptiveWeighting {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            weight_adjustment_factor: 0.1,
            minimum_weight: 0.1,
            maximum_weight: 1.0,
            adaptation_speed: 0.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_synthesizer_creation() {
        let synthesizer = MultimodalSynthesizer::new();
        assert!(synthesizer.current_modalities.is_empty());
        assert!(synthesizer.modality_weights.is_empty());
    }

    #[test]
    fn test_modality_data_addition() {
        let mut synthesizer = MultimodalSynthesizer::new();

        let text_data = ModalityData {
            modality_type: ModalityType::Text,
            data: b"Hello world".to_vec(),
            timestamp_ms: 1000,
            confidence: 0.9,
            metadata: HashMap::new(),
        };

        synthesizer.add_modality_data(text_data);
        assert_eq!(synthesizer.current_modalities.len(), 1);
        assert!(synthesizer
            .current_modalities
            .contains_key(&ModalityType::Text));
    }

    #[test]
    fn test_modality_weight_update() {
        let mut synthesizer = MultimodalSynthesizer::new();

        let mut weights = HashMap::new();
        weights.insert(ModalityType::Text, 0.8);
        weights.insert(ModalityType::Audio, 1.0);

        synthesizer.update_modality_weights(weights);
        assert_eq!(synthesizer.get_modality_weight(&ModalityType::Text), 0.8);
        assert_eq!(synthesizer.get_modality_weight(&ModalityType::Audio), 1.0);
    }

    #[test]
    fn test_cross_modal_alignment() {
        let mut synthesizer = MultimodalSynthesizer::new();

        let text_data = ModalityData {
            modality_type: ModalityType::Text,
            data: b"Hello".to_vec(),
            timestamp_ms: 1000,
            confidence: 0.9,
            metadata: HashMap::new(),
        };

        let audio_data = ModalityData {
            modality_type: ModalityType::Audio,
            data: vec![0u8; 1024],
            timestamp_ms: 1050,
            confidence: 0.8,
            metadata: HashMap::new(),
        };

        synthesizer.add_modality_data(text_data);
        synthesizer.add_modality_data(audio_data);

        let alignment = synthesizer.calculate_cross_modal_alignment();
        assert!(alignment.text_audio_alignment > 0.0);
        assert!(alignment.alignment_threshold > 0.0);
    }

    #[test]
    fn test_synchronization() {
        let mut synthesizer = MultimodalSynthesizer::new();

        let text_data = ModalityData {
            modality_type: ModalityType::Text,
            data: b"Hello".to_vec(),
            timestamp_ms: 1000,
            confidence: 0.9,
            metadata: HashMap::new(),
        };

        synthesizer.add_modality_data(text_data);
        synthesizer.synchronize_modalities(1500);

        let text_modality = synthesizer
            .current_modalities
            .get(&ModalityType::Text)
            .unwrap();
        assert_eq!(text_modality.timestamp_ms, 1500);
    }

    #[test]
    fn test_config_serialization() {
        let config = ModalityConfig {
            modality_type: ModalityType::Audio,
            weight: 0.8,
            synchronization_offset_ms: 100,
            duration_ms: Some(5000),
            adaptive_weighting: true,
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ModalityConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.weight, 0.8);
        assert_eq!(deserialized.synchronization_offset_ms, 100);
        assert_eq!(deserialized.adaptive_weighting, true);
    }
}
