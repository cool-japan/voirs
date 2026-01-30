//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::core::SingingEngine;
use crate::realtime::types::*;
use crate::score::MusicalScore;
use crate::techniques::SingingTechnique;
use crate::types::{NoteEvent, SingingRequest, VoiceCharacteristics};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SingingConfig;
    #[tokio::test]
    async fn test_realtime_engine_creation() {
        let config = SingingConfig::default();
        let core_engine = SingingEngine::new(config).await.unwrap();
        let rt_config = RealtimeConfig::default();
        let rt_engine = RealtimeEngine::new(core_engine, rt_config).await;
        assert!(rt_engine.is_ok());
    }
    #[tokio::test]
    async fn test_realtime_note_creation() {
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let rt_note = RealtimeNote::new(event)
            .with_priority(8)
            .with_latency_tolerance(Duration::from_millis(30));
        assert_eq!(rt_note.priority, 8);
        assert_eq!(rt_note.latency_tolerance, Duration::from_millis(30));
    }
    #[tokio::test]
    async fn test_live_session() {
        let config = SingingConfig::default();
        let core_engine = SingingEngine::new(config).await.unwrap();
        let rt_config = RealtimeConfig::default();
        let rt_engine = RealtimeEngine::new(core_engine, rt_config).await.unwrap();
        let mut session = rt_engine.create_session("test_session".to_string());
        assert_eq!(session.id, "test_session");
        assert_eq!(session.get_status(), SessionState::Idle);
        session.start_session().await.unwrap();
        assert_eq!(session.get_status(), SessionState::Performing);
    }
    #[tokio::test]
    async fn test_note_queuing() {
        let config = SingingConfig::default();
        let core_engine = SingingEngine::new(config).await.unwrap();
        let rt_config = RealtimeConfig::default();
        let rt_engine = RealtimeEngine::new(core_engine, rt_config).await.unwrap();
        let event = NoteEvent::new("C".to_string(), 4, 1.0, 0.8);
        let rt_note = RealtimeNote::new(event);
        rt_engine.queue_note(rt_note).await.unwrap();
        let queue_guard = rt_engine.note_queue.read().await;
        assert_eq!(queue_guard.len(), 1);
    }
    #[test]
    fn test_realtime_config_defaults() {
        let config = RealtimeConfig::default();
        assert_eq!(config.target_latency, 50.0);
        assert_eq!(config.sample_rate, 44100);
        assert!(config.low_latency_mode);
    }
    #[test]
    fn test_ultra_low_latency_config() {
        let config = RealtimeConfig::ultra_low_latency();
        assert_eq!(config.target_latency, 12.0);
        assert_eq!(config.buffer_size, 128);
        assert_eq!(config.sample_rate, 48000);
        assert!(config.ultra_low_latency_mode);
        assert!(config.midi_controller_support);
        assert!(config.expression_pedal_support);
        assert!(config.loop_station_enabled);
    }
    #[test]
    fn test_live_performance_config() {
        let config = RealtimeConfig::live_performance();
        assert_eq!(config.target_latency, 25.0);
        assert_eq!(config.buffer_size, 256);
        assert!(config.midi_controller_support);
        assert!(config.expression_pedal_support);
    }
    #[test]
    fn test_loop_station_config() {
        let config = RealtimeConfig::loop_station();
        assert_eq!(config.target_latency, 30.0);
        assert_eq!(config.quality_vs_speed, 0.8);
        assert!(config.loop_station_enabled);
    }
    #[test]
    fn test_midi_control_mapping() {
        let mapping = MidiControlMapping {
            cc_number: 7,
            parameter: ControlParameter::Volume,
            min_value: 0.0,
            max_value: 1.0,
            curve: MappingCurve::Linear,
        };
        assert_eq!(mapping.map_value(0.0), 0.0);
        assert_eq!(mapping.map_value(0.5), 0.5);
        assert_eq!(mapping.map_value(1.0), 1.0);
    }
    #[test]
    fn test_exponential_mapping_curve() {
        let mapping = MidiControlMapping {
            cc_number: 1,
            parameter: ControlParameter::VibratoDepth,
            min_value: 0.0,
            max_value: 1.0,
            curve: MappingCurve::Exponential(2.0),
        };
        let mapped_half = mapping.map_value(0.5);
        assert_eq!(mapped_half, 0.25);
    }
    #[test]
    fn test_expression_pedal_mapping() {
        let mapping = ExpressionMapping {
            pedal_id: "expression_1".to_string(),
            parameter: ControlParameter::BreathIntensity,
            curve: MappingCurve::Linear,
            sensitivity: 1.0,
        };
        assert_eq!(mapping.map_value(0.0), 0.0);
        assert_eq!(mapping.map_value(0.5), 0.5);
        assert_eq!(mapping.map_value(1.0), 1.0);
    }
    #[test]
    fn test_loop_station_creation() {
        let mut loop_station = LoopStation::new(Duration::from_secs(60));
        assert_eq!(loop_station.recording_state, RecordingState::Idle);
        assert_eq!(loop_station.playback_state, PlaybackState::Stopped);
        loop_station.start_recording("loop_1".to_string()).unwrap();
        assert_eq!(
            loop_station.recording_state,
            RecordingState::Recording("loop_1".to_string())
        );
        let audio_loop = loop_station.stop_recording().unwrap();
        assert_eq!(audio_loop.id, "loop_1");
        assert_eq!(loop_station.recording_state, RecordingState::Idle);
    }
    #[test]
    fn test_loop_station_playback() {
        let mut loop_station = LoopStation::new(Duration::from_secs(60));
        loop_station
            .start_recording("test_loop".to_string())
            .unwrap();
        loop_station.stop_recording().unwrap();
        loop_station.play_loop("test_loop").unwrap();
        assert_eq!(loop_station.playback_state, PlaybackState::Playing);
        loop_station.stop_loop("test_loop").unwrap();
        assert_eq!(loop_station.playback_state, PlaybackState::Stopped);
    }
    #[test]
    fn test_live_performance_controller() {
        let mut controller = LivePerformanceController::new();
        assert!(controller.midi_mappings.is_empty());
        assert!(controller.expression_mappings.is_empty());
        assert_eq!(controller.presets.len(), 2);
        let preset = controller.load_preset("Classical").unwrap();
        assert_eq!(preset.name, "Classical");
        assert!(preset.parameters.contains_key("vibrato_rate"));
    }
    #[test]
    fn test_control_parameter_display() {
        assert_eq!(ControlParameter::Volume.to_string(), "volume");
        assert_eq!(ControlParameter::PitchBend.to_string(), "pitch_bend");
        assert_eq!(ControlParameter::VibratoRate.to_string(), "vibrato_rate");
        assert_eq!(
            ControlParameter::VoiceCharacteristic("timbre".to_string()).to_string(),
            "voice_timbre"
        );
        assert_eq!(
            ControlParameter::EffectParameter("reverb".to_string(), "room_size".to_string())
                .to_string(),
            "effect_reverb_room_size"
        );
    }
    #[tokio::test]
    async fn test_live_performance_session_creation() {
        let config = SingingConfig::default();
        let core_engine = SingingEngine::new(config).await.unwrap();
        let rt_config = RealtimeConfig::live_performance();
        let rt_engine = RealtimeEngine::new(core_engine, rt_config).await.unwrap();
        let session = rt_engine.create_live_performance_session("live_test".to_string());
        assert_eq!(session.id, "live_test");
        assert!(session.controller.is_some());
        assert!(session.loop_station.is_some());
    }
    #[tokio::test]
    async fn test_ultra_low_latency_session() {
        let config = SingingConfig::default();
        let core_engine = SingingEngine::new(config).await.unwrap();
        let rt_config = RealtimeConfig::ultra_low_latency();
        let mut rt_engine = RealtimeEngine::new(core_engine, rt_config).await.unwrap();
        rt_engine.enable_ultra_low_latency().await.unwrap();
        assert_eq!(rt_engine.config.target_latency, 12.0);
        assert!(rt_engine.config.ultra_low_latency_mode);
    }
}
