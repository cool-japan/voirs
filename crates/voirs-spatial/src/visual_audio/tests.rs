//! Tests for visual audio integration

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::config::VisualAudioConfig;
    use super::super::processor::VisualAudioProcessor;
    use super::super::types::{
        AnimationParams, AnimationType, ColorRGBA, DirectionZone, EasingFunction, ShapeType,
        VisualEffect, VisualElement, VisualElementType,
    };
    use crate::Position3D;
    use std::time::Duration;

    #[test]
    fn test_visual_audio_config_creation() {
        let config = VisualAudioConfig::default();
        assert!(config.enabled);
        assert_eq!(config.master_intensity, 0.8);
    }

    #[test]
    fn test_visual_effect_creation() {
        let effect = VisualEffect {
            id: "test".to_string(),
            name: "Test Effect".to_string(),
            elements: vec![],
            duration: Duration::from_millis(100),
            looping: false,
            priority: 5,
            position: Position3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            audio_source_id: None,
        };
        assert_eq!(effect.id, "test");
        assert_eq!(effect.duration, Duration::from_millis(100));
    }

    #[test]
    fn test_color_rgba_creation() {
        let color = ColorRGBA {
            r: 1.0,
            g: 0.5,
            b: 0.0,
            a: 0.8,
        };
        assert_eq!(color.r, 1.0);
        assert_eq!(color.a, 0.8);
    }

    #[test]
    fn test_visual_processor_creation() {
        let config = VisualAudioConfig::default();
        let processor = VisualAudioProcessor::new(config);
        assert_eq!(processor.metrics().active_effects, 0);
    }

    #[test]
    fn test_direction_zone_calculation() {
        let config = VisualAudioConfig::default();
        let processor = VisualAudioProcessor::new(config);

        let listener = Position3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let front_source = Position3D {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let left_source = Position3D {
            x: -1.0,
            y: 0.0,
            z: 0.0,
        };

        let front_zone = processor.calculate_direction_zone(listener, front_source);
        let left_zone = processor.calculate_direction_zone(listener, left_source);

        assert_eq!(front_zone, DirectionZone::Front);
        assert_eq!(left_zone, DirectionZone::Left);
    }

    #[test]
    fn test_visual_distance_attenuation() {
        let config = VisualAudioConfig::default();
        let processor = VisualAudioProcessor::new(config);

        let close_attenuation = processor.calculate_visual_distance_attenuation(1.0);
        let far_attenuation = processor.calculate_visual_distance_attenuation(15.0);

        assert!(close_attenuation > far_attenuation);
        assert!(close_attenuation <= 1.0);
        assert!(far_attenuation >= 0.0);
    }

    #[test]
    fn test_visual_element_serialization() {
        let element = VisualElement {
            start_time: Duration::from_millis(0),
            duration: Duration::from_millis(100),
            element_type: VisualElementType::PointLight,
            color: ColorRGBA {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            intensity: 0.8,
            size: 1.0,
            animation: None,
            distance_attenuation: 1.0,
        };

        let serialized =
            serde_json::to_string(&element).expect("Should successfully serialize visual element");
        let deserialized: VisualElement = serde_json::from_str(&serialized)
            .expect("Should successfully deserialize visual element");
        assert_eq!(element.element_type, deserialized.element_type);
    }

    #[test]
    fn test_animation_params() {
        let animation = AnimationParams {
            animation_type: AnimationType::Pulse,
            speed: 1.0,
            amplitude: 0.5,
            phase: 0.0,
            easing: EasingFunction::EaseInOut,
        };

        assert_eq!(animation.animation_type, AnimationType::Pulse);
        assert_eq!(animation.easing, EasingFunction::EaseInOut);
    }

    #[test]
    fn test_accessibility_settings() {
        use super::super::config::VisualAccessibilitySettings;

        let accessibility = VisualAccessibilitySettings {
            high_contrast: true,
            color_blind_friendly: true,
            reduced_motion: false,
            indicator_scaling: 1.5,
            text_scaling: 1.2,
            audio_description_mode: false,
            screen_reader_compatible: true,
        };

        assert!(accessibility.high_contrast);
        assert!(accessibility.color_blind_friendly);
        assert_eq!(accessibility.indicator_scaling, 1.5);
    }
}
