//! Real-time singing performance system
//!
//! This module provides low-latency singing synthesis for live performances,
//! including streaming synthesis, real-time processing, and performance optimization.

#![allow(dead_code, clippy::derivable_impls)]

use crate::core::SingingEngine;
use crate::score::MusicalScore;
use crate::techniques::SingingTechnique;
use crate::types::{NoteEvent, SingingRequest, VoiceCharacteristics};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Real-time singing performance engine
pub struct RealtimeEngine {
    /// Core singing engine
    core_engine: Arc<RwLock<SingingEngine>>,
    /// Performance configuration
    config: RealtimeConfig,
    /// Audio buffer for streaming
    audio_buffer: Arc<RwLock<VecDeque<f32>>>,
    /// Processing state
    state: Arc<RwLock<RealtimeState>>,
    /// Input note queue
    note_queue: Arc<RwLock<VecDeque<RealtimeNote>>>,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
}

/// Real-time configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Target latency in milliseconds
    pub target_latency: f32,
    /// Buffer size in samples
    pub buffer_size: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of processing threads
    pub thread_count: usize,
    /// Enable low-latency mode
    pub low_latency_mode: bool,
    /// Pre-compute buffer size
    pub precompute_buffer: usize,
    /// Quality vs speed tradeoff (0.0-1.0, higher = better quality)
    pub quality_vs_speed: f32,
    /// Enable real-time effects processing
    pub realtime_effects: bool,
    /// Voice switching latency tolerance
    pub voice_switch_latency: f32,
    /// Ultra-low latency live performance mode (<15ms)
    pub ultra_low_latency_mode: bool,
    /// MIDI controller support
    pub midi_controller_support: bool,
    /// Expression pedal support
    pub expression_pedal_support: bool,
    /// Live loop station features
    pub loop_station_enabled: bool,
}

/// Real-time processing state
#[derive(Debug, Clone)]
struct RealtimeState {
    /// Is engine running
    is_running: bool,
    /// Current processing time
    current_time: f32,
    /// Next note to process
    next_note_time: f32,
    /// Current voice
    current_voice: VoiceCharacteristics,
    /// Current technique
    current_technique: SingingTechnique,
    /// Processing load (0.0-1.0)
    processing_load: f32,
    /// Buffer fill level (0.0-1.0)
    buffer_fill: f32,
}

/// Real-time note with timing information
#[derive(Debug, Clone)]
pub struct RealtimeNote {
    /// Note event
    pub event: NoteEvent,
    /// Scheduled time
    pub scheduled_time: Instant,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Latency tolerance
    pub latency_tolerance: Duration,
}

/// Performance metrics tracking
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Average latency in milliseconds
    pub avg_latency: f32,
    /// Maximum latency seen
    pub max_latency: f32,
    /// Buffer underruns
    pub underruns: u64,
    /// Processing time per note
    pub processing_time_per_note: f32,
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f32,
    /// Notes processed
    pub notes_processed: u64,
    /// Real-time factor (1.0 = real-time)
    pub realtime_factor: f32,
}

/// Real-time synthesis result
#[derive(Debug, Clone)]
pub struct RealtimeSynthesisResult {
    /// Synthesized audio samples
    pub audio: Vec<f32>,
    /// Actual latency experienced
    pub actual_latency: Duration,
    /// Processing time
    pub processing_time: Duration,
    /// Buffer status
    pub buffer_status: BufferStatus,
}

/// Buffer status information
#[derive(Debug, Clone)]
pub struct BufferStatus {
    /// Current fill level (0.0-1.0)
    pub fill_level: f32,
    /// Underrun occurred
    pub underrun: bool,
    /// Available samples
    pub available_samples: usize,
}

/// Live performance session
pub struct LiveSession {
    /// Session ID
    pub id: String,
    /// Realtime engine
    engine: Arc<RealtimeEngine>,
    /// Session start time
    start_time: Instant,
    /// Current score being performed
    current_score: Option<MusicalScore>,
    /// Performance state
    session_state: SessionState,
    /// Live performance controller
    controller: Option<LivePerformanceController>,
    /// Loop station
    loop_station: Option<LoopStation>,
}

/// Live performance controller for ultra-low latency
#[derive(Debug, Clone)]
pub struct LivePerformanceController {
    /// MIDI controller mappings
    pub midi_mappings: Vec<MidiControlMapping>,
    /// Expression pedal mappings
    pub expression_mappings: Vec<ExpressionMapping>,
    /// Real-time parameter controls
    pub parameter_controls: Vec<ParameterControl>,
    /// Performance presets
    pub presets: Vec<PerformancePreset>,
}

/// MIDI control mapping
#[derive(Debug, Clone)]
pub struct MidiControlMapping {
    /// MIDI CC number
    pub cc_number: u8,
    /// Parameter to control
    pub parameter: ControlParameter,
    /// Minimum value
    pub min_value: f32,
    /// Maximum value
    pub max_value: f32,
    /// Curve type for mapping
    pub curve: MappingCurve,
}

/// Expression pedal mapping
#[derive(Debug, Clone)]
pub struct ExpressionMapping {
    /// Pedal ID
    pub pedal_id: String,
    /// Parameter to control
    pub parameter: ControlParameter,
    /// Response curve
    pub curve: MappingCurve,
    /// Sensitivity
    pub sensitivity: f32,
}

/// Parameter control types
#[derive(Debug, Clone)]
pub enum ControlParameter {
    /// Volume control
    Volume,
    /// Pitch bend
    PitchBend,
    /// Vibrato rate
    VibratoRate,
    /// Vibrato depth
    VibratoDepth,
    /// Breath intensity
    BreathIntensity,
    /// Voice characteristic
    VoiceCharacteristic(String),
    /// Effect parameter
    EffectParameter(String, String),
}

/// Real-time parameter control
#[derive(Debug, Clone)]
pub struct ParameterControl {
    /// Parameter ID
    pub id: String,
    /// Current value
    pub current_value: f32,
    /// Target value
    pub target_value: f32,
    /// Smoothing factor (0.0-1.0)
    pub smoothing: f32,
    /// Last update time
    pub last_update: Instant,
}

/// Mapping curve types
#[derive(Debug, Clone)]
pub enum MappingCurve {
    /// Linear mapping
    Linear,
    /// Exponential mapping
    Exponential(f32),
    /// Logarithmic mapping
    Logarithmic,
    /// Custom curve with points
    Custom(Vec<(f32, f32)>),
}

/// Performance preset
#[derive(Debug, Clone)]
pub struct PerformancePreset {
    /// Preset name
    pub name: String,
    /// Voice characteristics
    pub voice: VoiceCharacteristics,
    /// Singing technique
    pub technique: SingingTechnique,
    /// Parameter values
    pub parameters: std::collections::HashMap<String, f32>,
    /// Effects settings
    pub effects: Vec<String>,
}

/// Loop station for live performance
#[derive(Debug)]
pub struct LoopStation {
    /// Audio loops
    loops: Vec<AudioLoop>,
    /// Recording state
    recording_state: RecordingState,
    /// Playback state
    playback_state: PlaybackState,
    /// Loop timing
    timing: LoopTiming,
    /// Maximum loop duration
    max_loop_duration: Duration,
}

/// Audio loop
#[derive(Debug, Clone)]
pub struct AudioLoop {
    /// Loop ID
    pub id: String,
    /// Audio data
    pub audio: Vec<f32>,
    /// Loop duration
    pub duration: Duration,
    /// Volume level
    pub volume: f32,
    /// Is playing
    pub is_playing: bool,
    /// Loop start time
    pub start_time: Option<Instant>,
}

/// Recording state
#[derive(Debug, Clone, PartialEq)]
pub enum RecordingState {
    /// Not recording
    Idle,
    /// Recording new loop
    Recording(String),
    /// Overdubbing existing loop
    Overdubbing(String),
}

/// Playback state
#[derive(Debug, Clone, PartialEq)]
pub enum PlaybackState {
    /// Stopped
    Stopped,
    /// Playing
    Playing,
    /// Paused
    Paused,
}

/// Loop timing information
#[derive(Debug, Clone)]
pub struct LoopTiming {
    /// Beats per minute
    pub bpm: f32,
    /// Time signature
    pub time_signature: (u32, u32),
    /// Loop sync mode
    pub sync_mode: SyncMode,
}

/// Synchronization modes
#[derive(Debug, Clone)]
pub enum SyncMode {
    /// Free timing
    Free,
    /// Beat sync
    BeatSync,
    /// Bar sync
    BarSync,
    /// Custom sync
    Custom(Duration),
}

/// Session state
#[derive(Debug, Clone, PartialEq)]
pub enum SessionState {
    /// Session not started
    Idle,
    /// Preparing to start
    Preparing,
    /// Currently performing
    Performing,
    /// Paused
    Paused,
    /// Finished
    Finished,
    /// Error state
    Error(String),
}

impl RealtimeEngine {
    /// Create new real-time engine
    pub async fn new(core_engine: SingingEngine, config: RealtimeConfig) -> crate::Result<Self> {
        let buffer_capacity = (config.sample_rate as f32 * config.target_latency / 1000.0) as usize;

        Ok(Self {
            core_engine: Arc::new(RwLock::new(core_engine)),
            config,
            audio_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_capacity))),
            state: Arc::new(RwLock::new(RealtimeState::new())),
            note_queue: Arc::new(RwLock::new(VecDeque::new())),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
        })
    }

    /// Start real-time processing
    pub async fn start(&self) -> crate::Result<()> {
        let mut state = self.state.write().await;
        if state.is_running {
            return Err(crate::Error::Processing(
                "Engine already running".to_string(),
            ));
        }

        state.is_running = true;
        state.current_time = 0.0;

        // Start processing thread
        self.spawn_processing_thread().await?;

        Ok(())
    }

    /// Stop real-time processing
    pub async fn stop(&self) -> crate::Result<()> {
        let mut state = self.state.write().await;
        state.is_running = false;
        Ok(())
    }

    /// Queue note for real-time synthesis
    pub async fn queue_note(&self, note: RealtimeNote) -> crate::Result<()> {
        let mut queue = self.note_queue.write().await;

        // Insert note in chronological order
        let mut insert_pos = queue.len();
        for (i, existing_note) in queue.iter().enumerate() {
            if note.scheduled_time < existing_note.scheduled_time {
                insert_pos = i;
                break;
            }
        }

        queue.insert(insert_pos, note);
        Ok(())
    }

    /// Get audio samples from buffer
    pub async fn get_audio(&self, num_samples: usize) -> Vec<f32> {
        let mut buffer = self.audio_buffer.write().await;
        let available = buffer.len().min(num_samples);

        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..available {
            samples.push(buffer.pop_front().unwrap_or(0.0));
        }

        // Pad with silence if needed
        while samples.len() < num_samples {
            samples.push(0.0);

            // Track underrun
            let mut metrics = self.metrics.write().await;
            metrics.underruns += 1;
        }

        samples
    }

    /// Set voice for real-time synthesis
    pub async fn set_realtime_voice(&self, voice: VoiceCharacteristics) -> crate::Result<()> {
        let mut state = self.state.write().await;
        state.current_voice = voice.clone();

        let core_engine = self.core_engine.read().await;
        core_engine.set_voice_characteristics(voice).await
    }

    /// Set technique for real-time synthesis
    pub async fn set_realtime_technique(&self, technique: SingingTechnique) -> crate::Result<()> {
        let mut state = self.state.write().await;
        state.current_technique = technique.clone();

        let core_engine = self.core_engine.read().await;
        core_engine.set_technique(technique).await
    }

    /// Get performance metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }

    /// Spawn processing thread
    async fn spawn_processing_thread(&self) -> crate::Result<()> {
        let engine_clone = Arc::clone(&self.core_engine);
        let state_clone = Arc::clone(&self.state);
        let queue_clone = Arc::clone(&self.note_queue);
        let buffer_clone = Arc::clone(&self.audio_buffer);
        let metrics_clone = Arc::clone(&self.metrics);
        let config = self.config.clone();

        tokio::spawn(async move {
            Self::processing_loop(
                engine_clone,
                state_clone,
                queue_clone,
                buffer_clone,
                metrics_clone,
                config,
            )
            .await;
        });

        Ok(())
    }

    /// Main processing loop
    async fn processing_loop(
        engine: Arc<RwLock<SingingEngine>>,
        state: Arc<RwLock<RealtimeState>>,
        queue: Arc<RwLock<VecDeque<RealtimeNote>>>,
        buffer: Arc<RwLock<VecDeque<f32>>>,
        metrics: Arc<RwLock<PerformanceMetrics>>,
        config: RealtimeConfig,
    ) {
        let mut last_process_time = Instant::now();

        while {
            let state_guard = state.read().await;
            state_guard.is_running
        } {
            let process_start = Instant::now();

            // Check for notes to process
            let note_to_process = {
                let mut queue_guard = queue.write().await;
                let now = Instant::now();

                if let Some(note) = queue_guard.front() {
                    if note.scheduled_time
                        <= now + Duration::from_millis(config.target_latency as u64)
                    {
                        Some(queue_guard.pop_front().unwrap())
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            if let Some(realtime_note) = note_to_process {
                let synthesis_start = Instant::now();

                // Create synthesis request
                let mut score = MusicalScore::new("Realtime".to_string(), "Live".to_string());
                let musical_note = crate::score::MusicalNote::new(
                    realtime_note.event.clone(),
                    0.0,
                    realtime_note.event.duration,
                );
                score.add_note(musical_note);

                let state_guard = state.read().await;
                let request = SingingRequest {
                    score,
                    voice: state_guard.current_voice.clone(),
                    technique: state_guard.current_technique.clone(),
                    effects: Vec::new(),
                    sample_rate: config.sample_rate,
                    target_duration: None,
                    quality: crate::types::QualitySettings {
                        quality_level: (config.quality_vs_speed * 10.0) as u8,
                        high_quality_pitch: config.quality_vs_speed > 0.7,
                        advanced_vibrato: config.quality_vs_speed > 0.5,
                        breath_modeling: config.quality_vs_speed > 0.3,
                        formant_modeling: config.quality_vs_speed > 0.6,
                        fft_size: if config.low_latency_mode { 1024 } else { 2048 },
                        hop_size: if config.low_latency_mode { 256 } else { 512 },
                    },
                };
                drop(state_guard);

                // Synthesize note
                match engine.read().await.synthesize(request).await {
                    Ok(response) => {
                        // Add to buffer
                        let mut buffer_guard = buffer.write().await;
                        for sample in response.audio {
                            buffer_guard.push_back(sample);
                        }

                        // Limit buffer size
                        while buffer_guard.len() > config.buffer_size * 2 {
                            buffer_guard.pop_front();
                        }

                        // Update metrics
                        let processing_time = synthesis_start.elapsed();
                        let mut metrics_guard = metrics.write().await;
                        metrics_guard.notes_processed += 1;
                        metrics_guard.processing_time_per_note =
                            processing_time.as_secs_f32() * 1000.0;

                        let actual_latency =
                            synthesis_start.duration_since(realtime_note.scheduled_time);
                        metrics_guard.avg_latency = (metrics_guard.avg_latency * 0.9)
                            + (actual_latency.as_secs_f32() * 1000.0 * 0.1);
                        metrics_guard.max_latency = metrics_guard
                            .max_latency
                            .max(actual_latency.as_secs_f32() * 1000.0);
                    }
                    Err(e) => {
                        tracing::warn!("Real-time synthesis error: {}", e);
                    }
                }
            }

            // Update processing load
            let processing_time = process_start.elapsed();
            let cycle_time = last_process_time.elapsed();
            last_process_time = Instant::now();

            let mut state_guard = state.write().await;
            state_guard.processing_load = processing_time.as_secs_f32() / cycle_time.as_secs_f32();

            let buffer_guard = buffer.read().await;
            state_guard.buffer_fill = buffer_guard.len() as f32 / config.buffer_size as f32;
            drop(buffer_guard);
            drop(state_guard);

            // Sleep to maintain target frame rate
            let target_cycle_time = Duration::from_micros(
                (1_000_000.0 / (config.sample_rate as f32 / config.buffer_size as f32)) as u64,
            );
            if processing_time < target_cycle_time {
                tokio::time::sleep(target_cycle_time - processing_time).await;
            }
        }
    }

    /// Create live session
    pub fn create_session(&self, session_id: String) -> LiveSession {
        LiveSession {
            id: session_id,
            engine: Arc::new(RealtimeEngine {
                core_engine: Arc::clone(&self.core_engine),
                config: self.config.clone(),
                audio_buffer: Arc::clone(&self.audio_buffer),
                state: Arc::clone(&self.state),
                note_queue: Arc::clone(&self.note_queue),
                metrics: Arc::clone(&self.metrics),
            }),
            start_time: Instant::now(),
            current_score: None,
            session_state: SessionState::Idle,
            controller: None,
            loop_station: None,
        }
    }

    /// Create live session with performance features
    pub fn create_live_performance_session(&self, session_id: String) -> LiveSession {
        let controller =
            if self.config.midi_controller_support || self.config.expression_pedal_support {
                Some(LivePerformanceController::new())
            } else {
                None
            };

        let loop_station = if self.config.loop_station_enabled {
            Some(LoopStation::new(Duration::from_secs(60))) // 60s max loop duration
        } else {
            None
        };

        LiveSession {
            id: session_id,
            engine: Arc::new(RealtimeEngine {
                core_engine: Arc::clone(&self.core_engine),
                config: self.config.clone(),
                audio_buffer: Arc::clone(&self.audio_buffer),
                state: Arc::clone(&self.state),
                note_queue: Arc::clone(&self.note_queue),
                metrics: Arc::clone(&self.metrics),
            }),
            start_time: Instant::now(),
            current_score: None,
            session_state: SessionState::Idle,
            controller,
            loop_station,
        }
    }

    /// Enable ultra-low latency mode
    pub async fn enable_ultra_low_latency(&mut self) -> crate::Result<()> {
        let mut config = self.config.clone();
        config.ultra_low_latency_mode = true;
        config.target_latency = 12.0;
        config.buffer_size = 128;
        config.thread_count = 4;
        config.quality_vs_speed = 0.4;

        self.config = config;
        Ok(())
    }
}

impl LiveSession {
    /// Start live performance session
    pub async fn start_session(&mut self) -> crate::Result<()> {
        self.session_state = SessionState::Preparing;
        self.engine.start().await?;
        self.session_state = SessionState::Performing;
        self.start_time = Instant::now();
        Ok(())
    }

    /// Stop live performance session
    pub async fn stop_session(&mut self) -> crate::Result<()> {
        self.engine.stop().await?;
        self.session_state = SessionState::Finished;
        Ok(())
    }

    /// Set score for performance
    pub fn set_score(&mut self, score: MusicalScore) {
        self.current_score = Some(score);
    }

    /// Queue note for immediate performance
    pub async fn perform_note(&self, note: NoteEvent) -> crate::Result<()> {
        let realtime_note = RealtimeNote {
            event: note,
            scheduled_time: Instant::now(),
            priority: 5,
            latency_tolerance: Duration::from_millis(50),
        };

        self.engine.queue_note(realtime_note).await
    }

    /// Get session status
    pub fn get_status(&self) -> SessionState {
        self.session_state.clone()
    }

    /// Get session duration
    pub fn get_duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Add MIDI controller mapping
    pub fn add_midi_mapping(&mut self, mapping: MidiControlMapping) -> crate::Result<()> {
        if let Some(controller) = &mut self.controller {
            controller.midi_mappings.push(mapping);
            Ok(())
        } else {
            Err(crate::Error::Config("No controller configured".to_string()))
        }
    }

    /// Handle MIDI control change
    pub async fn handle_midi_cc(&mut self, cc_number: u8, value: u8) -> crate::Result<()> {
        if let Some(controller) = &mut self.controller {
            let mapping = controller
                .midi_mappings
                .iter()
                .find(|m| m.cc_number == cc_number)
                .cloned();
            if let Some(mapping) = mapping {
                let normalized_value = value as f32 / 127.0;
                let mapped_value = mapping.map_value(normalized_value);
                controller
                    .update_parameter(&mapping.parameter, mapped_value)
                    .await?;
            }
        }
        Ok(())
    }

    /// Handle expression pedal input
    pub async fn handle_expression_pedal(
        &mut self,
        pedal_id: &str,
        value: f32,
    ) -> crate::Result<()> {
        if let Some(controller) = &mut self.controller {
            let mapping = controller
                .expression_mappings
                .iter()
                .find(|m| m.pedal_id == pedal_id)
                .cloned();
            if let Some(mapping) = mapping {
                let mapped_value = mapping.map_value(value);
                controller
                    .update_parameter(&mapping.parameter, mapped_value)
                    .await?;
            }
        }
        Ok(())
    }

    /// Start loop recording
    pub fn start_loop_recording(&mut self, loop_id: String) -> crate::Result<()> {
        if let Some(loop_station) = &mut self.loop_station {
            loop_station.start_recording(loop_id)
        } else {
            Err(crate::Error::Config(
                "No loop station configured".to_string(),
            ))
        }
    }

    /// Stop loop recording
    pub fn stop_loop_recording(&mut self) -> crate::Result<AudioLoop> {
        if let Some(loop_station) = &mut self.loop_station {
            loop_station.stop_recording()
        } else {
            Err(crate::Error::Config(
                "No loop station configured".to_string(),
            ))
        }
    }

    /// Play loop
    pub fn play_loop(&mut self, loop_id: &str) -> crate::Result<()> {
        if let Some(loop_station) = &mut self.loop_station {
            loop_station.play_loop(loop_id)
        } else {
            Err(crate::Error::Config(
                "No loop station configured".to_string(),
            ))
        }
    }

    /// Stop loop
    pub fn stop_loop(&mut self, loop_id: &str) -> crate::Result<()> {
        if let Some(loop_station) = &mut self.loop_station {
            loop_station.stop_loop(loop_id)
        } else {
            Err(crate::Error::Config(
                "No loop station configured".to_string(),
            ))
        }
    }
}

impl LivePerformanceController {
    /// Create new live performance controller
    pub fn new() -> Self {
        Self {
            midi_mappings: Vec::new(),
            expression_mappings: Vec::new(),
            parameter_controls: Vec::new(),
            presets: Self::default_presets(),
        }
    }

    /// Update parameter value
    pub async fn update_parameter(
        &mut self,
        parameter: &ControlParameter,
        value: f32,
    ) -> crate::Result<()> {
        // Find or create parameter control
        let param_id = parameter.to_string();
        if let Some(control) = self
            .parameter_controls
            .iter_mut()
            .find(|p| p.id == param_id)
        {
            control.target_value = value;
            control.last_update = Instant::now();
        } else {
            self.parameter_controls.push(ParameterControl {
                id: param_id,
                current_value: value,
                target_value: value,
                smoothing: 0.1,
                last_update: Instant::now(),
            });
        }
        Ok(())
    }

    /// Get current parameter value
    pub fn get_parameter_value(&self, parameter: &ControlParameter) -> f32 {
        let param_id = parameter.to_string();
        self.parameter_controls
            .iter()
            .find(|p| p.id == param_id)
            .map(|p| p.current_value)
            .unwrap_or(0.0)
    }

    /// Update all parameter smoothing
    pub fn update_smoothing(&mut self) {
        let now = Instant::now();
        for control in &mut self.parameter_controls {
            let delta_time = now.duration_since(control.last_update).as_secs_f32();
            let smoothing_factor = 1.0 - (-delta_time / control.smoothing).exp();
            control.current_value +=
                (control.target_value - control.current_value) * smoothing_factor;
        }
    }

    /// Load preset
    pub fn load_preset(&mut self, preset_name: &str) -> crate::Result<&PerformancePreset> {
        self.presets
            .iter()
            .find(|p| p.name == preset_name)
            .ok_or_else(|| crate::Error::Config(format!("Preset '{}' not found", preset_name)))
    }

    /// Create default presets
    fn default_presets() -> Vec<PerformancePreset> {
        vec![
            PerformancePreset {
                name: "Classical".to_string(),
                voice: VoiceCharacteristics::default(),
                technique: SingingTechnique::default(),
                parameters: [
                    ("vibrato_rate".to_string(), 5.0),
                    ("vibrato_depth".to_string(), 0.3),
                    ("breath_intensity".to_string(), 0.7),
                ]
                .into_iter()
                .collect(),
                effects: vec![],
            },
            PerformancePreset {
                name: "Pop".to_string(),
                voice: VoiceCharacteristics::default(),
                technique: SingingTechnique::default(),
                parameters: [
                    ("vibrato_rate".to_string(), 6.5),
                    ("vibrato_depth".to_string(), 0.4),
                    ("breath_intensity".to_string(), 0.5),
                ]
                .into_iter()
                .collect(),
                effects: vec![],
            },
        ]
    }
}

impl LoopStation {
    /// Create new loop station
    pub fn new(max_duration: Duration) -> Self {
        Self {
            loops: Vec::new(),
            recording_state: RecordingState::Idle,
            playback_state: PlaybackState::Stopped,
            timing: LoopTiming {
                bpm: 120.0,
                time_signature: (4, 4),
                sync_mode: SyncMode::Free,
            },
            max_loop_duration: max_duration,
        }
    }

    /// Start recording new loop
    pub fn start_recording(&mut self, loop_id: String) -> crate::Result<()> {
        if self.recording_state != RecordingState::Idle {
            return Err(crate::Error::Processing("Already recording".to_string()));
        }

        self.recording_state = RecordingState::Recording(loop_id);
        Ok(())
    }

    /// Stop recording and create loop
    pub fn stop_recording(&mut self) -> crate::Result<AudioLoop> {
        match &self.recording_state {
            RecordingState::Recording(loop_id) => {
                let audio_loop = AudioLoop {
                    id: loop_id.clone(),
                    audio: Vec::new(), // Would be filled with recorded audio
                    duration: Duration::from_secs(0), // Would be calculated
                    volume: 1.0,
                    is_playing: false,
                    start_time: None,
                };

                self.loops.push(audio_loop.clone());
                self.recording_state = RecordingState::Idle;
                Ok(audio_loop)
            }
            _ => Err(crate::Error::Processing("Not recording".to_string())),
        }
    }

    /// Play loop by ID
    pub fn play_loop(&mut self, loop_id: &str) -> crate::Result<()> {
        if let Some(audio_loop) = self.loops.iter_mut().find(|l| l.id == loop_id) {
            audio_loop.is_playing = true;
            audio_loop.start_time = Some(Instant::now());
            self.playback_state = PlaybackState::Playing;
            Ok(())
        } else {
            Err(crate::Error::Processing(format!(
                "Loop '{}' not found",
                loop_id
            )))
        }
    }

    /// Stop loop by ID
    pub fn stop_loop(&mut self, loop_id: &str) -> crate::Result<()> {
        if let Some(audio_loop) = self.loops.iter_mut().find(|l| l.id == loop_id) {
            audio_loop.is_playing = false;
            audio_loop.start_time = None;

            // Check if any loops are still playing
            if !self.loops.iter().any(|l| l.is_playing) {
                self.playback_state = PlaybackState::Stopped;
            }
            Ok(())
        } else {
            Err(crate::Error::Processing(format!(
                "Loop '{}' not found",
                loop_id
            )))
        }
    }

    /// Set BPM for loop synchronization
    pub fn set_bpm(&mut self, bpm: f32) {
        self.timing.bpm = bpm;
    }

    /// Set sync mode
    pub fn set_sync_mode(&mut self, mode: SyncMode) {
        self.timing.sync_mode = mode;
    }
}

impl MidiControlMapping {
    /// Map MIDI value to parameter range
    pub fn map_value(&self, midi_value: f32) -> f32 {
        let normalized = midi_value.clamp(0.0, 1.0);
        match self.curve {
            MappingCurve::Linear => self.min_value + (self.max_value - self.min_value) * normalized,
            MappingCurve::Exponential(factor) => {
                let mapped = normalized.powf(factor);
                self.min_value + (self.max_value - self.min_value) * mapped
            }
            MappingCurve::Logarithmic => {
                let mapped = (normalized * 9.0 + 1.0).log10();
                self.min_value + (self.max_value - self.min_value) * mapped
            }
            MappingCurve::Custom(ref points) => {
                // Linear interpolation between custom points
                if points.is_empty() {
                    return self.min_value;
                }

                for window in points.windows(2) {
                    if normalized >= window[0].0 && normalized <= window[1].0 {
                        let t = (normalized - window[0].0) / (window[1].0 - window[0].0);
                        let mapped = window[0].1 + (window[1].1 - window[0].1) * t;
                        return self.min_value + (self.max_value - self.min_value) * mapped;
                    }
                }

                // If not in range, use closest point
                if normalized <= points[0].0 {
                    self.min_value + (self.max_value - self.min_value) * points[0].1
                } else {
                    self.min_value + (self.max_value - self.min_value) * points.last().unwrap().1
                }
            }
        }
    }
}

impl ExpressionMapping {
    /// Map expression pedal value to parameter range
    pub fn map_value(&self, pedal_value: f32) -> f32 {
        let scaled_value = (pedal_value * self.sensitivity).clamp(0.0, 1.0);

        match self.curve {
            MappingCurve::Linear => scaled_value,
            MappingCurve::Exponential(factor) => scaled_value.powf(factor),
            MappingCurve::Logarithmic => (scaled_value * 9.0 + 1.0).log10(),
            MappingCurve::Custom(ref points) => {
                // Similar to MIDI mapping implementation
                if points.is_empty() {
                    return 0.0;
                }

                for window in points.windows(2) {
                    if scaled_value >= window[0].0 && scaled_value <= window[1].0 {
                        let t = (scaled_value - window[0].0) / (window[1].0 - window[0].0);
                        return window[0].1 + (window[1].1 - window[0].1) * t;
                    }
                }

                if scaled_value <= points[0].0 {
                    points[0].1
                } else {
                    points.last().unwrap().1
                }
            }
        }
    }
}

impl std::fmt::Display for ControlParameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ControlParameter::Volume => write!(f, "volume"),
            ControlParameter::PitchBend => write!(f, "pitch_bend"),
            ControlParameter::VibratoRate => write!(f, "vibrato_rate"),
            ControlParameter::VibratoDepth => write!(f, "vibrato_depth"),
            ControlParameter::BreathIntensity => write!(f, "breath_intensity"),
            ControlParameter::VoiceCharacteristic(name) => write!(f, "voice_{}", name),
            ControlParameter::EffectParameter(effect, param) => {
                write!(f, "effect_{}_{}", effect, param)
            }
        }
    }
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            target_latency: 50.0, // 50ms target latency
            buffer_size: 512,
            sample_rate: 44100,
            thread_count: 2,
            low_latency_mode: true,
            precompute_buffer: 2048,
            quality_vs_speed: 0.7, // Favor quality slightly
            realtime_effects: true,
            voice_switch_latency: 100.0, // 100ms for voice switching
            ultra_low_latency_mode: false,
            midi_controller_support: false,
            expression_pedal_support: false,
            loop_station_enabled: false,
        }
    }
}

impl RealtimeConfig {
    /// Create configuration for ultra-low latency live performance (<15ms)
    pub fn ultra_low_latency() -> Self {
        Self {
            target_latency: 12.0, // <15ms target
            buffer_size: 128,     // Smaller buffer for lower latency
            sample_rate: 48000,   // Higher sample rate for precision
            thread_count: 4,      // More threads for parallel processing
            low_latency_mode: true,
            precompute_buffer: 256, // Minimal precompute buffer
            quality_vs_speed: 0.4,  // Favor speed over quality
            realtime_effects: true,
            voice_switch_latency: 20.0, // Very fast voice switching
            ultra_low_latency_mode: true,
            midi_controller_support: true,
            expression_pedal_support: true,
            loop_station_enabled: true,
        }
    }

    /// Create configuration for live performance with controllers
    pub fn live_performance() -> Self {
        Self {
            target_latency: 25.0, // Balanced latency
            buffer_size: 256,
            sample_rate: 48000,
            thread_count: 4,
            low_latency_mode: true,
            precompute_buffer: 512,
            quality_vs_speed: 0.6, // Balanced quality/speed
            realtime_effects: true,
            voice_switch_latency: 50.0,
            ultra_low_latency_mode: false,
            midi_controller_support: true,
            expression_pedal_support: true,
            loop_station_enabled: true,
        }
    }

    /// Create configuration for loop station recording
    pub fn loop_station() -> Self {
        Self {
            target_latency: 30.0,
            buffer_size: 512,
            sample_rate: 44100,
            thread_count: 2,
            low_latency_mode: true,
            precompute_buffer: 1024,
            quality_vs_speed: 0.8, // Higher quality for recording
            realtime_effects: true,
            voice_switch_latency: 100.0,
            ultra_low_latency_mode: false,
            midi_controller_support: false,
            expression_pedal_support: false,
            loop_station_enabled: true,
        }
    }
}

impl RealtimeState {
    fn new() -> Self {
        Self {
            is_running: false,
            current_time: 0.0,
            next_note_time: 0.0,
            current_voice: VoiceCharacteristics::default(),
            current_technique: SingingTechnique::default(),
            processing_load: 0.0,
            buffer_fill: 0.0,
        }
    }
}

impl RealtimeNote {
    /// Create new real-time note
    pub fn new(event: NoteEvent) -> Self {
        Self {
            event,
            scheduled_time: Instant::now(),
            priority: 5,
            latency_tolerance: Duration::from_millis(50),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set latency tolerance
    pub fn with_latency_tolerance(mut self, tolerance: Duration) -> Self {
        self.latency_tolerance = tolerance;
        self
    }

    /// Schedule for specific time
    pub fn schedule_at(mut self, time: Instant) -> Self {
        self.scheduled_time = time;
        self
    }
}

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
        assert_eq!(mapped_half, 0.25); // 0.5^2 = 0.25
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

        // Test recording
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

        // Record a loop first
        loop_station
            .start_recording("test_loop".to_string())
            .unwrap();
        loop_station.stop_recording().unwrap();

        // Test playback
        loop_station.play_loop("test_loop").unwrap();
        assert_eq!(loop_station.playback_state, PlaybackState::Playing);

        // Test stopping
        loop_station.stop_loop("test_loop").unwrap();
        assert_eq!(loop_station.playback_state, PlaybackState::Stopped);
    }

    #[test]
    fn test_live_performance_controller() {
        let mut controller = LivePerformanceController::new();
        assert!(controller.midi_mappings.is_empty());
        assert!(controller.expression_mappings.is_empty());
        assert_eq!(controller.presets.len(), 2); // Classical and Pop presets

        // Test preset loading
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
