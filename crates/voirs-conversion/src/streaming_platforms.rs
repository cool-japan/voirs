//! Streaming platform integration for voice conversion
//!
//! This module provides comprehensive integration with major streaming platforms,
//! enabling real-time voice conversion for live streams, content creation,
//! and broadcast applications.
//!
//! ## Supported Platforms
//!
//! - **Twitch**: Live streaming with real-time voice conversion
//! - **YouTube Live**: Broadcasting with voice transformation
//! - **Discord**: Voice channel integration with custom voices
//! - **OBS Studio**: Plugin integration for streaming software
//! - **Streamlabs**: Direct integration with popular streaming tools
//! - **XSplit**: Professional broadcasting software integration
//! - **Custom RTMP**: Generic RTMP streaming support
//!
//! ## Features
//!
//! - **Real-time Voice Conversion**: Ultra-low latency for live streaming
//! - **Stream Quality Optimization**: Adaptive quality based on bandwidth
//! - **Multiple Voice Profiles**: Quick switching between character voices
//! - **Audience Interaction**: Voice conversion for donations/alerts
//! - **Content Creator Tools**: Voice effects and character voices
//! - **Moderation Support**: Voice masking and privacy protection
//!
//! ## Usage
//!
//! ```rust
//! # use voirs_conversion::streaming_platforms::{StreamingPlatform, StreamProcessor, StreamConfig};
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create processor for Twitch streaming
//! let config = StreamConfig::twitch_optimized();
//! let mut processor = StreamProcessor::new(StreamingPlatform::Twitch, config)?;
//!
//! // Process stream audio in real-time
//! let input_audio = vec![0.0f32; 1024]; // Sample audio data
//! let converted_stream = processor.process_stream_audio(&input_audio, "streamer_voice").await?;
//! # Ok(())
//! # }
//! ```

use crate::{
    config::ConversionConfig,
    core::VoiceConverter,
    realtime::{RealtimeConfig, RealtimeConverter},
    types::{ConversionRequest, ConversionTarget, ConversionType, VoiceCharacteristics},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Supported streaming platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamingPlatform {
    /// Twitch live streaming
    Twitch,
    /// YouTube Live streaming
    YouTubeLive,
    /// Discord voice channels
    Discord,
    /// OBS Studio integration
    OBSStudio,
    /// Streamlabs integration
    Streamlabs,
    /// XSplit broadcasting
    XSplit,
    /// Generic RTMP streaming
    RTMP,
    /// Facebook Live streaming
    FacebookLive,
    /// TikTok Live streaming
    TikTokLive,
}

impl StreamingPlatform {
    /// Get platform-specific streaming constraints
    pub fn streaming_constraints(&self) -> StreamingConstraints {
        match self {
            StreamingPlatform::Twitch => StreamingConstraints {
                max_latency_ms: 100.0, // Twitch buffer tolerance
                max_bitrate_kbps: 6000,
                sample_rate: 48000,
                channels: 2,
                recommended_buffer_ms: 50.0,
                bandwidth_adaptation: true,
                quality_levels: vec![
                    StreamQuality::Source,
                    StreamQuality::High,
                    StreamQuality::Medium,
                    StreamQuality::Low,
                ],
            },
            StreamingPlatform::YouTubeLive => StreamingConstraints {
                max_latency_ms: 200.0, // YouTube has higher tolerance
                max_bitrate_kbps: 8000,
                sample_rate: 48000,
                channels: 2,
                recommended_buffer_ms: 100.0,
                bandwidth_adaptation: true,
                quality_levels: vec![
                    StreamQuality::Source,
                    StreamQuality::High,
                    StreamQuality::Medium,
                ],
            },
            StreamingPlatform::Discord => StreamingConstraints {
                max_latency_ms: 40.0,  // Discord requires very low latency
                max_bitrate_kbps: 320, // Discord voice quality
                sample_rate: 48000,
                channels: 2,
                recommended_buffer_ms: 20.0,
                bandwidth_adaptation: false, // Discord handles this
                quality_levels: vec![StreamQuality::High, StreamQuality::Medium],
            },
            StreamingPlatform::OBSStudio => StreamingConstraints {
                max_latency_ms: 50.0,
                max_bitrate_kbps: 10000, // OBS can handle high quality
                sample_rate: 48000,
                channels: 2,
                recommended_buffer_ms: 25.0,
                bandwidth_adaptation: false, // OBS handles encoding
                quality_levels: vec![
                    StreamQuality::Source,
                    StreamQuality::High,
                    StreamQuality::Medium,
                    StreamQuality::Low,
                ],
            },
            StreamingPlatform::Streamlabs => StreamingConstraints {
                max_latency_ms: 75.0,
                max_bitrate_kbps: 8000,
                sample_rate: 48000,
                channels: 2,
                recommended_buffer_ms: 40.0,
                bandwidth_adaptation: true,
                quality_levels: vec![
                    StreamQuality::Source,
                    StreamQuality::High,
                    StreamQuality::Medium,
                ],
            },
            StreamingPlatform::XSplit => StreamingConstraints {
                max_latency_ms: 60.0,
                max_bitrate_kbps: 10000,
                sample_rate: 48000,
                channels: 2,
                recommended_buffer_ms: 30.0,
                bandwidth_adaptation: true,
                quality_levels: vec![
                    StreamQuality::Source,
                    StreamQuality::High,
                    StreamQuality::Medium,
                    StreamQuality::Low,
                ],
            },
            StreamingPlatform::RTMP => StreamingConstraints {
                max_latency_ms: 150.0, // Generic RTMP tolerance
                max_bitrate_kbps: 6000,
                sample_rate: 44100, // Standard for RTMP
                channels: 2,
                recommended_buffer_ms: 75.0,
                bandwidth_adaptation: true,
                quality_levels: vec![
                    StreamQuality::High,
                    StreamQuality::Medium,
                    StreamQuality::Low,
                ],
            },
            StreamingPlatform::FacebookLive => StreamingConstraints {
                max_latency_ms: 300.0, // Facebook has higher tolerance
                max_bitrate_kbps: 4000,
                sample_rate: 44100,
                channels: 2,
                recommended_buffer_ms: 150.0,
                bandwidth_adaptation: true,
                quality_levels: vec![
                    StreamQuality::High,
                    StreamQuality::Medium,
                    StreamQuality::Low,
                ],
            },
            StreamingPlatform::TikTokLive => StreamingConstraints {
                max_latency_ms: 200.0,
                max_bitrate_kbps: 3000, // TikTok mobile optimization
                sample_rate: 44100,
                channels: 2,
                recommended_buffer_ms: 100.0,
                bandwidth_adaptation: true,
                quality_levels: vec![StreamQuality::Medium, StreamQuality::Low],
            },
        }
    }

    /// Get platform name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            StreamingPlatform::Twitch => "Twitch",
            StreamingPlatform::YouTubeLive => "YouTube Live",
            StreamingPlatform::Discord => "Discord",
            StreamingPlatform::OBSStudio => "OBS Studio",
            StreamingPlatform::Streamlabs => "Streamlabs",
            StreamingPlatform::XSplit => "XSplit",
            StreamingPlatform::RTMP => "RTMP",
            StreamingPlatform::FacebookLive => "Facebook Live",
            StreamingPlatform::TikTokLive => "TikTok Live",
        }
    }
}

/// Stream quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamQuality {
    /// Source quality (no compression)
    Source,
    /// High quality (minimal compression)
    High,
    /// Medium quality (balanced)
    Medium,
    /// Low quality (high compression)
    Low,
}

impl StreamQuality {
    /// Get quality factor (0.0-1.0)
    pub fn quality_factor(&self) -> f32 {
        match self {
            StreamQuality::Source => 1.0,
            StreamQuality::High => 0.9,
            StreamQuality::Medium => 0.7,
            StreamQuality::Low => 0.5,
        }
    }
}

/// Platform-specific streaming constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamingConstraints {
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f32,
    /// Maximum audio bitrate in kbps
    pub max_bitrate_kbps: u32,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Recommended buffer size in milliseconds
    pub recommended_buffer_ms: f32,
    /// Enable bandwidth adaptation
    pub bandwidth_adaptation: bool,
    /// Supported quality levels
    pub quality_levels: Vec<StreamQuality>,
}

/// Stream-specific audio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Target streaming platform
    pub platform: StreamingPlatform,
    /// Audio buffer size for processing
    pub buffer_size: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Stream quality level
    pub quality: StreamQuality,
    /// Enable real-time monitoring
    pub enable_monitoring: bool,
    /// Enable bandwidth adaptation
    pub enable_adaptation: bool,
    /// Enable voice activity detection
    pub voice_activity_detection: bool,
    /// Enable automatic gain control
    pub automatic_gain_control: bool,
    /// Enable noise suppression for streaming
    pub noise_suppression: bool,
    /// Stream-specific optimizations
    pub platform_optimizations: HashMap<String, f32>,
}

impl StreamConfig {
    /// Create Twitch-optimized configuration
    pub fn twitch_optimized() -> Self {
        let mut platform_optimizations = HashMap::new();
        platform_optimizations.insert("twitch_chat_integration".to_string(), 1.0);
        platform_optimizations.insert("twitch_bitrate_optimization".to_string(), 0.9);
        platform_optimizations.insert("twitch_emote_response".to_string(), 0.8);

        Self {
            platform: StreamingPlatform::Twitch,
            buffer_size: 1024,
            sample_rate: 48000,
            channels: 2,
            quality: StreamQuality::High,
            enable_monitoring: true,
            enable_adaptation: true,
            voice_activity_detection: true,
            automatic_gain_control: true,
            noise_suppression: true,
            platform_optimizations,
        }
    }

    /// Create YouTube Live-optimized configuration
    pub fn youtube_optimized() -> Self {
        let mut platform_optimizations = HashMap::new();
        platform_optimizations.insert("youtube_quality_priority".to_string(), 1.0);
        platform_optimizations.insert("youtube_latency_tolerance".to_string(), 0.7);

        Self {
            platform: StreamingPlatform::YouTubeLive,
            buffer_size: 2048, // Higher buffer for quality
            sample_rate: 48000,
            channels: 2,
            quality: StreamQuality::Source,
            enable_monitoring: true,
            enable_adaptation: true,
            voice_activity_detection: true,
            automatic_gain_control: true,
            noise_suppression: true,
            platform_optimizations,
        }
    }

    /// Create Discord-optimized configuration
    pub fn discord_optimized() -> Self {
        let mut platform_optimizations = HashMap::new();
        platform_optimizations.insert("discord_voice_channel_optimization".to_string(), 1.0);
        platform_optimizations.insert("discord_push_to_talk_support".to_string(), 0.9);

        Self {
            platform: StreamingPlatform::Discord,
            buffer_size: 480, // Very small buffer for low latency
            sample_rate: 48000,
            channels: 2,
            quality: StreamQuality::High,
            enable_monitoring: true,
            enable_adaptation: false, // Discord handles this
            voice_activity_detection: true,
            automatic_gain_control: false, // Discord handles this
            noise_suppression: false,      // Discord handles this
            platform_optimizations,
        }
    }

    /// Create OBS Studio-optimized configuration
    pub fn obs_optimized() -> Self {
        let mut platform_optimizations = HashMap::new();
        platform_optimizations.insert("obs_plugin_integration".to_string(), 1.0);
        platform_optimizations.insert("obs_audio_filter_chain".to_string(), 0.9);

        Self {
            platform: StreamingPlatform::OBSStudio,
            buffer_size: 512,
            sample_rate: 48000,
            channels: 2,
            quality: StreamQuality::Source,
            enable_monitoring: true,
            enable_adaptation: false, // OBS handles encoding
            voice_activity_detection: true,
            automatic_gain_control: true,
            noise_suppression: true,
            platform_optimizations,
        }
    }
}

/// Stream voice processing modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamVoiceMode {
    /// Main streamer voice
    StreamerVoice,
    /// Guest/co-host voice
    GuestVoice,
    /// Character/roleplay voice
    CharacterVoice,
    /// Text-to-speech voice
    TextToSpeech,
    /// Donation/alert voice
    AlertVoice,
    /// Background narrator voice
    NarratorVoice,
}

/// Stream processor for real-time voice conversion in streaming
#[derive(Debug)]
pub struct StreamProcessor {
    /// Target streaming platform
    platform: StreamingPlatform,
    /// Stream configuration
    config: StreamConfig,
    /// Real-time voice converter
    realtime_converter: RealtimeConverter,
    /// Voice converter for complex transformations
    voice_converter: Arc<VoiceConverter>,
    /// Platform constraints
    constraints: StreamingConstraints,
    /// Voice presets for quick switching
    voice_presets: Arc<RwLock<HashMap<String, VoiceCharacteristics>>>,
    /// Active streaming sessions
    active_streams: Arc<RwLock<HashMap<String, StreamSession>>>,
    /// Stream performance monitor
    performance_monitor: StreamPerformanceMonitor,
    /// Current bandwidth adaptation state
    adaptation_state: BandwidthAdaptationState,
}

impl StreamProcessor {
    /// Create new stream processor
    pub fn new(platform: StreamingPlatform, config: StreamConfig) -> Result<Self> {
        let constraints = platform.streaming_constraints();

        // Create real-time converter with stream-optimized settings
        let realtime_config = RealtimeConfig {
            buffer_size: config.buffer_size,
            sample_rate: config.sample_rate,
            target_latency_ms: constraints.max_latency_ms,
            overlap_factor: 0.25,
            adaptive_buffering: config.enable_adaptation,
            max_threads: 2,
            enable_lookahead: false, // Disable for streaming
            lookahead_size: 0,
        };

        let realtime_converter = RealtimeConverter::new(realtime_config)?;
        let voice_converter = Arc::new(VoiceConverter::new()?);

        Ok(Self {
            platform,
            config,
            realtime_converter,
            voice_converter,
            constraints,
            voice_presets: Arc::new(RwLock::new(HashMap::new())),
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            performance_monitor: StreamPerformanceMonitor::new(),
            adaptation_state: BandwidthAdaptationState::new(),
        })
    }

    /// Process stream audio in real-time
    pub async fn process_stream_audio(
        &mut self,
        input_audio: &[f32],
        voice_preset: &str,
    ) -> Result<Vec<f32>> {
        let start_time = std::time::Instant::now();

        // Check bandwidth adaptation
        if self.config.enable_adaptation {
            self.update_bandwidth_adaptation().await?;
        }

        // Get voice characteristics
        let voice_characteristics = {
            let presets = self.voice_presets.read().await;
            presets.get(voice_preset).cloned().unwrap_or_default()
        };

        // Set conversion target
        let target = ConversionTarget::new(voice_characteristics);
        self.realtime_converter.set_conversion_target(target);

        // Process with real-time converter
        let mut result = self.realtime_converter.process_chunk(input_audio).await?;

        // Apply streaming-specific processing
        if self.config.noise_suppression {
            result = self.apply_streaming_noise_suppression(&result)?;
        }

        if self.config.automatic_gain_control {
            result = self.apply_streaming_agc(&result)?;
        }

        // Update performance metrics
        let processing_time = start_time.elapsed();
        self.performance_monitor.record_processing(
            processing_time,
            input_audio.len(),
            &self.constraints,
        );

        debug!(
            "Stream audio processed: {} samples in {:.2}ms for {}",
            input_audio.len(),
            processing_time.as_secs_f32() * 1000.0,
            self.platform.as_str()
        );

        Ok(result)
    }

    /// Process stream audio with specific mode
    pub async fn process_stream_with_mode(
        &mut self,
        input_audio: &[f32],
        voice_preset: &str,
        mode: StreamVoiceMode,
    ) -> Result<Vec<f32>> {
        // Apply mode-specific processing
        let processed_audio = match mode {
            StreamVoiceMode::StreamerVoice => {
                self.apply_streamer_processing(input_audio, voice_preset)
                    .await?
            }
            StreamVoiceMode::GuestVoice => {
                self.apply_guest_processing(input_audio, voice_preset)
                    .await?
            }
            StreamVoiceMode::CharacterVoice => {
                self.apply_character_processing(input_audio, voice_preset)
                    .await?
            }
            StreamVoiceMode::TextToSpeech => {
                self.apply_tts_processing(input_audio, voice_preset).await?
            }
            StreamVoiceMode::AlertVoice => {
                self.apply_alert_processing(input_audio, voice_preset)
                    .await?
            }
            StreamVoiceMode::NarratorVoice => {
                self.apply_narrator_processing(input_audio, voice_preset)
                    .await?
            }
        };

        Ok(processed_audio)
    }

    /// Register voice preset for quick switching
    pub async fn register_voice_preset(
        &self,
        preset_name: String,
        characteristics: VoiceCharacteristics,
    ) {
        let mut presets = self.voice_presets.write().await;
        presets.insert(preset_name.clone(), characteristics);
        info!("Registered voice preset: {}", preset_name);
    }

    /// Start streaming session
    pub async fn start_stream_session(
        &self,
        session_id: String,
        stream_title: String,
    ) -> Result<()> {
        let session = StreamSession {
            session_id: session_id.clone(),
            stream_title,
            start_time: std::time::Instant::now(),
            platform: self.platform,
            processed_frames: 0,
            total_latency_ms: 0.0,
            quality_drops: 0,
        };

        let mut streams = self.active_streams.write().await;
        streams.insert(session_id.clone(), session);

        info!(
            "Started streaming session: {} on {}",
            session_id,
            self.platform.as_str()
        );
        Ok(())
    }

    /// Stop streaming session
    pub async fn stop_stream_session(&self, session_id: &str) -> Result<StreamSession> {
        let mut streams = self.active_streams.write().await;
        streams
            .remove(session_id)
            .ok_or_else(|| Error::processing(format!("Stream session not found: {}", session_id)))
    }

    /// Get current stream performance metrics
    pub fn get_stream_metrics(&self) -> StreamPerformanceMetrics {
        self.performance_monitor.get_current_metrics()
    }

    /// Check if streaming performance is acceptable
    pub fn is_stream_performance_acceptable(&self) -> bool {
        self.performance_monitor
            .check_streaming_performance(&self.constraints)
    }

    /// Get platform integration information
    pub fn get_platform_integration(&self) -> PlatformIntegration {
        match self.platform {
            StreamingPlatform::Twitch => PlatformIntegration::Twitch(TwitchIntegration {
                api_version: "v1".to_string(),
                chat_integration: true,
                emote_support: true,
                subscriber_features: true,
                clip_integration: false,
            }),
            StreamingPlatform::YouTubeLive => PlatformIntegration::YouTube(YouTubeIntegration {
                api_version: "v3".to_string(),
                live_chat_integration: true,
                super_chat_support: true,
                recording_support: true,
                analytics_integration: true,
            }),
            StreamingPlatform::Discord => PlatformIntegration::Discord(DiscordIntegration {
                api_version: "v10".to_string(),
                voice_channel_integration: true,
                bot_integration: true,
                stage_channel_support: true,
                permission_system: true,
            }),
            StreamingPlatform::OBSStudio => PlatformIntegration::OBS(OBSIntegration {
                plugin_version: "1.0.0".to_string(),
                audio_filter_support: true,
                scene_switching: true,
                source_integration: true,
                hotkey_support: true,
            }),
            StreamingPlatform::Streamlabs => {
                PlatformIntegration::Streamlabs(StreamlabsIntegration {
                    api_version: "v1".to_string(),
                    donation_integration: true,
                    alert_system: true,
                    overlay_support: true,
                    chatbot_integration: true,
                })
            }
            StreamingPlatform::XSplit => PlatformIntegration::XSplit(XSplitIntegration {
                plugin_version: "1.0.0".to_string(),
                scene_management: true,
                audio_plugin_support: true,
                broadcast_profiles: true,
            }),
            StreamingPlatform::RTMP => PlatformIntegration::RTMP(RTMPIntegration {
                protocol_version: "1.0".to_string(),
                streaming_server_support: true,
                custom_endpoints: true,
                authentication_support: true,
            }),
            StreamingPlatform::FacebookLive => PlatformIntegration::Facebook(FacebookIntegration {
                api_version: "v15.0".to_string(),
                live_video_integration: true,
                comment_integration: true,
                reaction_support: true,
            }),
            StreamingPlatform::TikTokLive => PlatformIntegration::TikTok(TikTokIntegration {
                api_version: "v1".to_string(),
                live_stream_integration: true,
                gift_integration: true,
                comment_moderation: true,
            }),
        }
    }

    // Private helper methods

    async fn update_bandwidth_adaptation(&mut self) -> Result<()> {
        // Simple bandwidth adaptation based on performance
        let metrics = self.performance_monitor.get_current_metrics();

        if metrics.average_latency_ms > self.constraints.max_latency_ms * 1.2 {
            self.adaptation_state.decrease_quality();
        } else if metrics.average_latency_ms < self.constraints.max_latency_ms * 0.5 {
            self.adaptation_state.increase_quality();
        }

        Ok(())
    }

    async fn apply_streamer_processing(
        &mut self,
        input_audio: &[f32],
        voice_preset: &str,
    ) -> Result<Vec<f32>> {
        self.process_stream_audio(input_audio, voice_preset).await
    }

    async fn apply_guest_processing(
        &mut self,
        input_audio: &[f32],
        voice_preset: &str,
    ) -> Result<Vec<f32>> {
        // Similar to streamer but with guest-specific adjustments
        let mut result = self.process_stream_audio(input_audio, voice_preset).await?;

        // Apply guest normalization
        for sample in result.iter_mut() {
            *sample *= 0.9; // Slightly reduce volume for guests
        }

        Ok(result)
    }

    async fn apply_character_processing(
        &mut self,
        input_audio: &[f32],
        voice_preset: &str,
    ) -> Result<Vec<f32>> {
        self.process_stream_audio(input_audio, voice_preset).await
    }

    async fn apply_tts_processing(
        &mut self,
        input_audio: &[f32],
        voice_preset: &str,
    ) -> Result<Vec<f32>> {
        // Apply TTS-specific processing (clarity, pronunciation)
        let mut result = self.process_stream_audio(input_audio, voice_preset).await?;

        // Enhance clarity for TTS
        for sample in result.iter_mut() {
            *sample = (*sample * 1.1).clamp(-1.0, 1.0);
        }

        Ok(result)
    }

    async fn apply_alert_processing(
        &mut self,
        input_audio: &[f32],
        _voice_preset: &str,
    ) -> Result<Vec<f32>> {
        // Apply alert-specific processing (attention-grabbing)
        let mut result = input_audio.to_vec();

        // Add slight emphasis for alerts
        for sample in result.iter_mut() {
            *sample = (*sample * 1.3).clamp(-1.0, 1.0);
        }

        Ok(result)
    }

    async fn apply_narrator_processing(
        &mut self,
        input_audio: &[f32],
        voice_preset: &str,
    ) -> Result<Vec<f32>> {
        // Apply narrator-specific processing (authority, clarity)
        let mut result = self.process_stream_audio(input_audio, voice_preset).await?;

        // Apply narrator enhancement
        for sample in result.iter_mut() {
            *sample = (*sample * 1.05).clamp(-1.0, 1.0);
        }

        Ok(result)
    }

    fn apply_streaming_noise_suppression(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Streaming-optimized noise suppression
        let threshold = 0.02; // Higher threshold for streaming
        let mut processed = audio.to_vec();

        for sample in processed.iter_mut() {
            if sample.abs() < threshold {
                *sample *= 0.05; // More aggressive suppression
            }
        }

        Ok(processed)
    }

    fn apply_streaming_agc(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Streaming-optimized automatic gain control
        let target_level = 0.8; // Higher target for streaming
        let current_level = audio.iter().map(|&x| x.abs()).sum::<f32>() / audio.len() as f32;

        if current_level > 0.0 {
            let gain = target_level / current_level;
            let clamped_gain = gain.clamp(0.3, 3.0); // Wider range for streaming

            Ok(audio
                .iter()
                .map(|&x| (x * clamped_gain).clamp(-1.0, 1.0))
                .collect())
        } else {
            Ok(audio.to_vec())
        }
    }
}

/// Stream session for tracking streaming state and performance metrics
#[derive(Debug, Clone)]
pub struct StreamSession {
    /// Unique identifier for this streaming session
    pub session_id: String,
    /// Title of the stream for display purposes
    pub stream_title: String,
    /// Timestamp when the stream session started
    pub start_time: std::time::Instant,
    /// Target streaming platform for this session
    pub platform: StreamingPlatform,
    /// Total number of audio frames processed in this session
    pub processed_frames: u64,
    /// Cumulative latency across all processed frames in milliseconds
    pub total_latency_ms: f32,
    /// Number of quality degradation events during the session
    pub quality_drops: u32,
}

/// Bandwidth adaptation state for dynamic quality adjustment during streaming
#[derive(Debug, Clone)]
pub struct BandwidthAdaptationState {
    /// Current quality level based on bandwidth conditions
    pub current_quality: StreamQuality,
    /// Target bitrate in kbps for current quality level
    pub target_bitrate: u32,
    /// Historical record of quality adaptation events
    pub adaptation_history: Vec<AdaptationEvent>,
}

impl BandwidthAdaptationState {
    fn new() -> Self {
        Self {
            current_quality: StreamQuality::High,
            target_bitrate: 128,
            adaptation_history: Vec::new(),
        }
    }

    fn decrease_quality(&mut self) {
        self.current_quality = match self.current_quality {
            StreamQuality::Source => StreamQuality::High,
            StreamQuality::High => StreamQuality::Medium,
            StreamQuality::Medium => StreamQuality::Low,
            StreamQuality::Low => StreamQuality::Low,
        };

        self.adaptation_history.push(AdaptationEvent {
            timestamp: std::time::Instant::now(),
            direction: AdaptationDirection::Decrease,
            new_quality: self.current_quality,
        });
    }

    fn increase_quality(&mut self) {
        self.current_quality = match self.current_quality {
            StreamQuality::Low => StreamQuality::Medium,
            StreamQuality::Medium => StreamQuality::High,
            StreamQuality::High => StreamQuality::Source,
            StreamQuality::Source => StreamQuality::Source,
        };

        self.adaptation_history.push(AdaptationEvent {
            timestamp: std::time::Instant::now(),
            direction: AdaptationDirection::Increase,
            new_quality: self.current_quality,
        });
    }
}

/// Record of a bandwidth adaptation event
#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    /// Time when the adaptation occurred
    pub timestamp: std::time::Instant,
    /// Direction of quality change (increase or decrease)
    pub direction: AdaptationDirection,
    /// New quality level after adaptation
    pub new_quality: StreamQuality,
}

/// Direction of quality adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationDirection {
    /// Quality level increased due to improved conditions
    Increase,
    /// Quality level decreased due to bandwidth constraints
    Decrease,
}

/// Stream performance monitor for tracking real-time streaming metrics
#[derive(Debug)]
pub struct StreamPerformanceMonitor {
    processing_times: Vec<std::time::Duration>,
    bandwidth_samples: Vec<u32>,
    quality_drops: u32,
    last_check: std::time::Instant,
}

impl StreamPerformanceMonitor {
    fn new() -> Self {
        Self {
            processing_times: Vec::new(),
            bandwidth_samples: Vec::new(),
            quality_drops: 0,
            last_check: std::time::Instant::now(),
        }
    }

    fn record_processing(
        &mut self,
        processing_time: std::time::Duration,
        _sample_count: usize,
        constraints: &StreamingConstraints,
    ) {
        self.processing_times.push(processing_time);

        // Keep only recent samples
        if self.processing_times.len() > 100 {
            self.processing_times.drain(0..50);
        }

        // Check for quality drops
        let latency_ms = processing_time.as_secs_f32() * 1000.0;
        if latency_ms > constraints.max_latency_ms {
            self.quality_drops += 1;
        }
    }

    fn check_streaming_performance(&self, constraints: &StreamingConstraints) -> bool {
        if self.processing_times.is_empty() {
            return true;
        }

        let avg_latency_ms = self
            .processing_times
            .iter()
            .map(|d| d.as_secs_f32() * 1000.0)
            .sum::<f32>()
            / self.processing_times.len() as f32;

        avg_latency_ms <= constraints.max_latency_ms
    }

    fn get_current_metrics(&self) -> StreamPerformanceMetrics {
        let avg_latency_ms = if self.processing_times.is_empty() {
            0.0
        } else {
            self.processing_times
                .iter()
                .map(|d| d.as_secs_f32() * 1000.0)
                .sum::<f32>()
                / self.processing_times.len() as f32
        };

        StreamPerformanceMetrics {
            average_latency_ms: avg_latency_ms,
            current_bitrate_kbps: self.bandwidth_samples.last().copied().unwrap_or(0),
            quality_drops: self.quality_drops,
            uptime_seconds: self.last_check.elapsed().as_secs(),
            buffer_health_percent: 95.0, // Placeholder
        }
    }
}

/// Stream performance metrics for monitoring streaming quality and health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamPerformanceMetrics {
    /// Average processing latency across recent frames in milliseconds
    pub average_latency_ms: f32,
    /// Current audio bitrate in kilobits per second
    pub current_bitrate_kbps: u32,
    /// Total number of quality degradation events
    pub quality_drops: u32,
    /// Stream uptime duration in seconds
    pub uptime_seconds: u64,
    /// Current buffer health as percentage (0-100)
    pub buffer_health_percent: f32,
}

/// Platform-specific integration information with API details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlatformIntegration {
    /// Twitch live streaming integration
    Twitch(TwitchIntegration),
    /// YouTube Live streaming integration
    YouTube(YouTubeIntegration),
    /// Discord voice channel integration
    Discord(DiscordIntegration),
    /// OBS Studio plugin integration
    OBS(OBSIntegration),
    /// Streamlabs tools integration
    Streamlabs(StreamlabsIntegration),
    /// XSplit broadcaster integration
    XSplit(XSplitIntegration),
    /// Generic RTMP streaming integration
    RTMP(RTMPIntegration),
    /// Facebook Live streaming integration
    Facebook(FacebookIntegration),
    /// TikTok Live streaming integration
    TikTok(TikTokIntegration),
}

/// Twitch platform integration details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwitchIntegration {
    /// Twitch API version being used
    pub api_version: String,
    /// Whether chat integration is enabled
    pub chat_integration: bool,
    /// Whether Twitch emote support is available
    pub emote_support: bool,
    /// Whether subscriber-specific features are enabled
    pub subscriber_features: bool,
    /// Whether clip creation integration is available
    pub clip_integration: bool,
}

/// YouTube Live platform integration details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YouTubeIntegration {
    /// YouTube API version being used
    pub api_version: String,
    /// Whether live chat integration is enabled
    pub live_chat_integration: bool,
    /// Whether Super Chat support is available
    pub super_chat_support: bool,
    /// Whether stream recording is supported
    pub recording_support: bool,
    /// Whether analytics integration is available
    pub analytics_integration: bool,
}

/// Discord platform integration details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordIntegration {
    /// Discord API version being used
    pub api_version: String,
    /// Whether voice channel integration is enabled
    pub voice_channel_integration: bool,
    /// Whether bot integration is available
    pub bot_integration: bool,
    /// Whether stage channel support is available
    pub stage_channel_support: bool,
    /// Whether permission system integration is enabled
    pub permission_system: bool,
}

/// OBS Studio plugin integration details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OBSIntegration {
    /// OBS plugin version being used
    pub plugin_version: String,
    /// Whether audio filter support is available
    pub audio_filter_support: bool,
    /// Whether scene switching integration is enabled
    pub scene_switching: bool,
    /// Whether audio source integration is available
    pub source_integration: bool,
    /// Whether hotkey support is enabled
    pub hotkey_support: bool,
}

/// Streamlabs platform integration details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamlabsIntegration {
    /// Streamlabs API version being used
    pub api_version: String,
    /// Whether donation integration is enabled
    pub donation_integration: bool,
    /// Whether alert system integration is available
    pub alert_system: bool,
    /// Whether overlay support is enabled
    pub overlay_support: bool,
    /// Whether chatbot integration is available
    pub chatbot_integration: bool,
}

/// XSplit broadcaster integration details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XSplitIntegration {
    /// XSplit plugin version being used
    pub plugin_version: String,
    /// Whether scene management integration is enabled
    pub scene_management: bool,
    /// Whether audio plugin support is available
    pub audio_plugin_support: bool,
    /// Whether broadcast profile support is enabled
    pub broadcast_profiles: bool,
}

/// RTMP protocol integration details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RTMPIntegration {
    /// RTMP protocol version being used
    pub protocol_version: String,
    /// Whether custom streaming server support is enabled
    pub streaming_server_support: bool,
    /// Whether custom endpoint configuration is available
    pub custom_endpoints: bool,
    /// Whether authentication support is enabled
    pub authentication_support: bool,
}

/// Facebook Live platform integration details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacebookIntegration {
    /// Facebook API version being used
    pub api_version: String,
    /// Whether live video integration is enabled
    pub live_video_integration: bool,
    /// Whether comment integration is available
    pub comment_integration: bool,
    /// Whether reaction support is enabled
    pub reaction_support: bool,
}

/// TikTok Live platform integration details and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TikTokIntegration {
    /// TikTok API version being used
    pub api_version: String,
    /// Whether live stream integration is enabled
    pub live_stream_integration: bool,
    /// Whether gift integration is available
    pub gift_integration: bool,
    /// Whether comment moderation is enabled
    pub comment_moderation: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_platform_constraints() {
        let twitch_constraints = StreamingPlatform::Twitch.streaming_constraints();
        assert!(twitch_constraints.max_latency_ms <= 100.0);
        assert_eq!(twitch_constraints.sample_rate, 48000);

        let discord_constraints = StreamingPlatform::Discord.streaming_constraints();
        assert!(discord_constraints.max_latency_ms <= 40.0);
        assert!(!discord_constraints.bandwidth_adaptation);
    }

    #[test]
    fn test_stream_quality_levels() {
        assert_eq!(StreamQuality::Source.quality_factor(), 1.0);
        assert_eq!(StreamQuality::High.quality_factor(), 0.9);
        assert_eq!(StreamQuality::Medium.quality_factor(), 0.7);
        assert_eq!(StreamQuality::Low.quality_factor(), 0.5);
    }

    #[test]
    fn test_stream_config_creation() {
        let twitch_config = StreamConfig::twitch_optimized();
        assert_eq!(twitch_config.platform, StreamingPlatform::Twitch);
        assert!(twitch_config.enable_adaptation);

        let discord_config = StreamConfig::discord_optimized();
        assert_eq!(discord_config.platform, StreamingPlatform::Discord);
        assert!(!discord_config.enable_adaptation);
        assert_eq!(discord_config.buffer_size, 480);
    }

    #[tokio::test]
    async fn test_stream_processor_creation() {
        let config = StreamConfig::twitch_optimized();
        let processor = StreamProcessor::new(StreamingPlatform::Twitch, config);
        assert!(processor.is_ok());

        let processor = processor.unwrap();
        assert_eq!(processor.platform, StreamingPlatform::Twitch);
    }

    #[tokio::test]
    async fn test_voice_preset_registration() {
        let config = StreamConfig::twitch_optimized();
        let processor = StreamProcessor::new(StreamingPlatform::Twitch, config).unwrap();

        let characteristics = VoiceCharacteristics::default();
        processor
            .register_voice_preset("streamer_main".to_string(), characteristics)
            .await;

        let presets = processor.voice_presets.read().await;
        assert!(presets.contains_key("streamer_main"));
    }

    #[tokio::test]
    async fn test_stream_session_management() {
        let config = StreamConfig::twitch_optimized();
        let processor = StreamProcessor::new(StreamingPlatform::Twitch, config).unwrap();

        // Start session
        let result = processor
            .start_stream_session("stream1".to_string(), "Test Stream".to_string())
            .await;
        assert!(result.is_ok());

        // Check session exists
        let streams = processor.active_streams.read().await;
        assert!(streams.contains_key("stream1"));
    }

    #[test]
    fn test_bandwidth_adaptation() {
        let mut adaptation = BandwidthAdaptationState::new();

        let initial_quality = adaptation.current_quality;
        adaptation.decrease_quality();
        assert_ne!(adaptation.current_quality, initial_quality);

        adaptation.increase_quality();
        assert_eq!(adaptation.current_quality, initial_quality);
    }

    #[test]
    fn test_stream_voice_modes() {
        let modes = [
            StreamVoiceMode::StreamerVoice,
            StreamVoiceMode::GuestVoice,
            StreamVoiceMode::CharacterVoice,
            StreamVoiceMode::TextToSpeech,
            StreamVoiceMode::AlertVoice,
            StreamVoiceMode::NarratorVoice,
        ];

        for mode in &modes {
            // Test serialization
            let serialized = serde_json::to_string(mode).unwrap();
            let deserialized: StreamVoiceMode = serde_json::from_str(&serialized).unwrap();
            assert_eq!(*mode, deserialized);
        }
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = StreamPerformanceMonitor::new();
        let constraints = StreamingPlatform::Twitch.streaming_constraints();

        // Record some processing times
        monitor.record_processing(std::time::Duration::from_millis(50), 1024, &constraints);

        let metrics = monitor.get_current_metrics();
        assert!(metrics.average_latency_ms > 0.0);
        assert!(monitor.check_streaming_performance(&constraints));
    }

    #[test]
    fn test_platform_integration_info() {
        let config = StreamConfig::twitch_optimized();
        let processor = StreamProcessor::new(StreamingPlatform::Twitch, config).unwrap();

        let integration = processor.get_platform_integration();
        match integration {
            PlatformIntegration::Twitch(twitch) => {
                assert!(twitch.chat_integration);
                assert!(twitch.emote_support);
            }
            _ => panic!("Expected Twitch integration"),
        }
    }

    #[tokio::test]
    async fn test_streaming_audio_processing() {
        let config = StreamConfig::discord_optimized();
        let mut processor = StreamProcessor::new(StreamingPlatform::Discord, config).unwrap();

        // Register a voice preset
        let characteristics = VoiceCharacteristics::default();
        processor
            .register_voice_preset("test_voice".to_string(), characteristics)
            .await;

        // Process some audio
        let test_audio = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let result = processor
            .process_stream_audio(&test_audio, "test_voice")
            .await;

        assert!(result.is_ok());
        let processed = result.unwrap();
        assert!(!processed.is_empty());
    }

    #[tokio::test]
    async fn test_noise_suppression_streaming() {
        let config = StreamConfig::twitch_optimized();
        let processor = StreamProcessor::new(StreamingPlatform::Twitch, config).unwrap();

        let noisy_audio = vec![0.01, 0.5, 0.015, -0.7]; // Mix of noise and signal
        let processed = processor
            .apply_streaming_noise_suppression(&noisy_audio)
            .unwrap();

        // Check that small signals are more aggressively suppressed for streaming
        assert!(processed[0].abs() < noisy_audio[0].abs() * 0.1);
        assert!(processed[2].abs() < noisy_audio[2].abs() * 0.1);
    }

    #[tokio::test]
    async fn test_streaming_agc() {
        let config = StreamConfig::youtube_optimized();
        let processor = StreamProcessor::new(StreamingPlatform::YouTubeLive, config).unwrap();

        let quiet_audio = vec![0.1, -0.1, 0.05, -0.05];
        let processed = processor.apply_streaming_agc(&quiet_audio).unwrap();

        // Check that streaming AGC has a higher target level
        let original_level =
            quiet_audio.iter().map(|&x| x.abs()).sum::<f32>() / quiet_audio.len() as f32;
        let processed_level =
            processed.iter().map(|&x| x.abs()).sum::<f32>() / processed.len() as f32;
        assert!(processed_level > original_level * 2.0); // Should be significantly louder
    }
}
