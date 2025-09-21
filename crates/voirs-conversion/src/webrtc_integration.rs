//! WebRTC Integration for Real-time Voice Conversion
//!
//! This module provides comprehensive WebRTC integration for real-time voice conversion
//! in communication applications. It supports peer-to-peer voice conversion, real-time
//! audio streaming, and integration with WebRTC-based communication platforms.
//!
//! ## Key Features
//!
//! - **Real-time Voice Conversion**: Ultra-low latency voice conversion for WebRTC streams
//! - **P2P Voice Transformation**: Peer-to-peer voice conversion with minimal delay
//! - **Audio Stream Processing**: Real-time audio stream modification and enhancement
//! - **WebRTC API Integration**: Direct integration with WebRTC audio pipelines
//! - **Adaptive Quality Control**: Dynamic quality adjustment based on network conditions
//! - **Multi-party Conversion**: Support for group voice conversion in multi-party calls
//!
//! ## Performance Targets
//!
//! - **Ultra-low Latency**: <20ms additional latency for voice conversion
//! - **Real-time Processing**: Process audio in 10ms chunks for seamless communication
//! - **Network Adaptive**: Automatically adjust quality based on bandwidth
//! - **CPU Efficient**: <10% additional CPU usage over standard WebRTC
//!
//! ## Supported Features
//!
//! - **Voice Conversion**: Real-time speaker, age, and gender conversion
//! - **Audio Enhancement**: Noise reduction, echo cancellation, and quality improvement
//! - **Privacy Protection**: Voice anonymization for privacy-sensitive applications
//! - **Accessibility**: Voice modification for accessibility purposes
//! - **Entertainment**: Fun voice effects for gaming and social applications
//!
//! ## Usage
//!
//! ```rust,no_run
//! # use voirs_conversion::webrtc_integration::*;
//! # use voirs_conversion::types::*;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create WebRTC voice processor
//! let mut processor = WebRTCVoiceProcessor::new().await?;
//!
//! // Configure for real-time conversion
//! processor.set_conversion_mode(ConversionMode::RealTime).await?;
//!
//! // Set up voice conversion
//! let conversion_config = VoiceConversionConfig {
//!     conversion_type: ConversionType::SpeakerConversion,
//!     target: ConversionTarget::new(VoiceCharacteristics::default()),
//!     quality_mode: QualityMode::Balanced,
//!     enable_noise_reduction: true,
//!     enable_echo_cancellation: true,
//!     enable_agc: true,
//!     enable_vad: true,
//!     conversion_strength: 0.8,
//!     enable_adaptive_quality: true,
//! };
//!
//! processor.configure_conversion(conversion_config).await?;
//!
//! // Process audio chunk (typical WebRTC frame)
//! let audio_chunk = vec![0.1; 480]; // 10ms at 48kHz
//! let converted_chunk = processor.process_audio_chunk(&audio_chunk).await?;
//! # Ok(())
//! # }
//! ```

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};

/// WebRTC conversion modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConversionMode {
    /// Real-time conversion with minimal latency
    RealTime,
    /// High-quality conversion with acceptable latency
    HighQuality,
    /// Balanced quality and latency
    Balanced,
    /// Privacy-focused voice anonymization
    Privacy,
    /// Entertainment mode with creative effects
    Entertainment,
}

impl ConversionMode {
    /// Get target latency for mode in milliseconds
    pub fn target_latency_ms(&self) -> f64 {
        match self {
            ConversionMode::RealTime => 15.0,
            ConversionMode::HighQuality => 50.0,
            ConversionMode::Balanced => 25.0,
            ConversionMode::Privacy => 20.0,
            ConversionMode::Entertainment => 30.0,
        }
    }

    /// Get quality level for mode (0-100)
    pub fn quality_level(&self) -> u8 {
        match self {
            ConversionMode::RealTime => 70,
            ConversionMode::HighQuality => 95,
            ConversionMode::Balanced => 80,
            ConversionMode::Privacy => 75,
            ConversionMode::Entertainment => 85,
        }
    }
}

/// Quality modes for WebRTC processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityMode {
    /// Minimal quality for maximum performance
    Minimal,
    /// Low quality for bandwidth-constrained scenarios
    Low,
    /// Balanced quality and performance
    Balanced,
    /// High quality for premium applications
    High,
    /// Maximum quality regardless of performance
    Maximum,
}

/// Voice conversion configuration for WebRTC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConversionConfig {
    /// Type of conversion to apply
    pub conversion_type: ConversionType,
    /// Target voice characteristics
    pub target: ConversionTarget,
    /// Quality mode
    pub quality_mode: QualityMode,
    /// Enable noise reduction
    pub enable_noise_reduction: bool,
    /// Enable echo cancellation
    pub enable_echo_cancellation: bool,
    /// Enable automatic gain control
    pub enable_agc: bool,
    /// Enable voice activity detection
    pub enable_vad: bool,
    /// Conversion strength (0.0-1.0)
    pub conversion_strength: f32,
    /// Enable adaptive quality
    pub enable_adaptive_quality: bool,
}

impl Default for VoiceConversionConfig {
    fn default() -> Self {
        Self {
            conversion_type: ConversionType::SpeakerConversion,
            target: ConversionTarget::new(VoiceCharacteristics::default()),
            quality_mode: QualityMode::Balanced,
            enable_noise_reduction: true,
            enable_echo_cancellation: true,
            enable_agc: true,
            enable_vad: true,
            conversion_strength: 1.0,
            enable_adaptive_quality: true,
        }
    }
}

/// WebRTC audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebRTCAudioConfig {
    /// Sample rate (typical WebRTC rates: 8kHz, 16kHz, 32kHz, 48kHz)
    pub sample_rate: u32,
    /// Channels (1 for mono, 2 for stereo)
    pub channels: u32,
    /// Frame size in samples (typical: 80, 160, 320, 480, 960)
    pub frame_size: usize,
    /// Bit depth
    pub bit_depth: u16,
    /// Enable packet loss concealment
    pub enable_plc: bool,
    /// Enable jitter buffer adaptation
    pub enable_jitter_adaptation: bool,
    /// Target network delay in milliseconds
    pub target_network_delay_ms: u32,
    /// Maximum allowed delay in milliseconds
    pub max_delay_ms: u32,
}

impl Default for WebRTCAudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000, // High-quality WebRTC
            channels: 1,        // Mono for efficiency
            frame_size: 480,    // 10ms at 48kHz
            bit_depth: 16,
            enable_plc: true,
            enable_jitter_adaptation: true,
            target_network_delay_ms: 20,
            max_delay_ms: 150,
        }
    }
}

/// Network condition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    /// Available bandwidth in kbps
    pub bandwidth_kbps: u32,
    /// Round-trip time in milliseconds
    pub rtt_ms: u32,
    /// Packet loss percentage (0-100)
    pub packet_loss_percent: f32,
    /// Jitter in milliseconds
    pub jitter_ms: f32,
    /// Network quality score (0-100)
    pub quality_score: u8,
}

impl Default for NetworkConditions {
    fn default() -> Self {
        Self {
            bandwidth_kbps: 1000, // 1 Mbps
            rtt_ms: 50,
            packet_loss_percent: 0.0,
            jitter_ms: 5.0,
            quality_score: 100,
        }
    }
}

/// WebRTC voice processor
pub struct WebRTCVoiceProcessor {
    /// Conversion mode
    conversion_mode: ConversionMode,
    /// Voice conversion configuration
    conversion_config: VoiceConversionConfig,
    /// Audio configuration
    audio_config: WebRTCAudioConfig,
    /// Voice converter
    voice_converter: Arc<VoiceConverter>,
    /// Audio buffer for processing
    audio_buffer: Arc<Mutex<AudioBuffer>>,
    /// Network conditions
    network_conditions: Arc<RwLock<NetworkConditions>>,
    /// Processing statistics
    stats: Arc<WebRTCProcessingStats>,
    /// Quality controller
    quality_controller: Arc<Mutex<AdaptiveQualityController>>,
    /// Audio enhancements
    audio_enhancements: Arc<Mutex<AudioEnhancements>>,
    /// Jitter buffer
    jitter_buffer: Arc<Mutex<JitterBuffer>>,
    /// Packet loss concealer
    plc: Arc<Mutex<PacketLossConcealer>>,
    /// Initialized flag
    initialized: Arc<AtomicBool>,
}

impl WebRTCVoiceProcessor {
    /// Create new WebRTC voice processor
    pub async fn new() -> Result<Self> {
        let voice_converter = Arc::new(VoiceConverter::new()?);
        let audio_config = WebRTCAudioConfig::default();
        let audio_buffer = Arc::new(Mutex::new(AudioBuffer::new(audio_config.frame_size * 4)));
        let quality_controller = Arc::new(Mutex::new(AdaptiveQualityController::new()));
        let audio_enhancements = Arc::new(Mutex::new(AudioEnhancements::new(&audio_config)?));
        let jitter_buffer = Arc::new(Mutex::new(JitterBuffer::new(audio_config.frame_size)));
        let plc = Arc::new(Mutex::new(PacketLossConcealer::new(
            audio_config.sample_rate,
        )));

        Ok(Self {
            conversion_mode: ConversionMode::Balanced,
            conversion_config: VoiceConversionConfig::default(),
            audio_config,
            voice_converter,
            audio_buffer,
            network_conditions: Arc::new(RwLock::new(NetworkConditions::default())),
            stats: Arc::new(WebRTCProcessingStats::new()),
            quality_controller,
            audio_enhancements,
            jitter_buffer,
            plc,
            initialized: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Initialize the WebRTC voice processor
    pub async fn initialize(&mut self) -> Result<()> {
        if self.initialized.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Initialize audio enhancements
        self.audio_enhancements.lock().await.initialize().await?;

        // Initialize quality controller
        let config = self.conversion_config.clone();
        self.quality_controller
            .lock()
            .await
            .initialize(&config)
            .await?;

        // Start network monitoring
        self.start_network_monitoring().await?;

        self.initialized.store(true, Ordering::Relaxed);
        self.stats.record_initialization();

        Ok(())
    }

    /// Set conversion mode
    pub async fn set_conversion_mode(&mut self, mode: ConversionMode) -> Result<()> {
        self.conversion_mode = mode;

        // Adjust configuration based on mode
        self.adjust_config_for_mode(mode).await?;

        self.stats.record_mode_change(mode);
        Ok(())
    }

    /// Configure voice conversion
    pub async fn configure_conversion(&mut self, config: VoiceConversionConfig) -> Result<()> {
        self.conversion_config = config.clone();

        // Update quality controller
        self.quality_controller
            .lock()
            .await
            .update_config(&config)
            .await?;

        // Update audio enhancements
        self.audio_enhancements
            .lock()
            .await
            .configure(&config)
            .await?;

        Ok(())
    }

    /// Process audio chunk for WebRTC
    pub async fn process_audio_chunk(&self, audio_chunk: &[f32]) -> Result<Vec<f32>> {
        if !self.initialized.load(Ordering::Relaxed) {
            return Err(Error::runtime(
                "WebRTC processor not initialized".to_string(),
            ));
        }

        let start_time = Instant::now();

        // Validate chunk size
        if audio_chunk.len() != self.audio_config.frame_size {
            return Err(Error::audio(format!(
                "Invalid chunk size: expected {}, got {}",
                self.audio_config.frame_size,
                audio_chunk.len()
            )));
        }

        // Add to jitter buffer
        {
            let mut jitter_buffer = self.jitter_buffer.lock().await;
            jitter_buffer.add_frame(audio_chunk.to_vec());
        }

        // Get frame from jitter buffer
        let buffered_frame = {
            let mut jitter_buffer = self.jitter_buffer.lock().await;
            jitter_buffer.get_frame()?
        };

        // Apply audio enhancements
        let enhanced_audio = self
            .audio_enhancements
            .lock()
            .await
            .process(&buffered_frame)
            .await?;

        // Apply voice conversion if enabled
        let converted_audio = if self.conversion_config.conversion_strength > 0.0 {
            self.apply_voice_conversion(&enhanced_audio).await?
        } else {
            enhanced_audio
        };

        // Apply post-processing
        let final_audio = self.post_process_audio(&converted_audio).await?;

        let processing_time = start_time.elapsed();
        self.stats
            .record_processing(processing_time, audio_chunk.len());

        // Check if we're meeting latency targets
        if processing_time.as_millis() as f64 > self.conversion_mode.target_latency_ms() {
            self.quality_controller
                .lock()
                .await
                .report_latency_issue(processing_time)
                .await;
        }

        Ok(final_audio)
    }

    /// Process audio stream continuously
    pub async fn process_audio_stream<I, O>(&self, input_stream: I, output_stream: O) -> Result<()>
    where
        I: futures::Stream<Item = Vec<f32>> + Send + Unpin,
        O: futures::Sink<Vec<f32>> + Send + Unpin,
    {
        use futures::{SinkExt, StreamExt};

        let mut input = input_stream;
        let mut output = output_stream;

        while let Some(audio_chunk) = input.next().await {
            let processed_chunk = self.process_audio_chunk(&audio_chunk).await?;
            output
                .send(processed_chunk)
                .await
                .map_err(|_| Error::streaming("Failed to send processed audio".to_string()))?;
        }

        Ok(())
    }

    /// Update network conditions
    pub async fn update_network_conditions(&self, conditions: NetworkConditions) -> Result<()> {
        *self.network_conditions.write().await = conditions.clone();

        // Adjust quality based on network conditions
        if self.conversion_config.enable_adaptive_quality {
            self.quality_controller
                .lock()
                .await
                .adjust_for_network(&conditions)
                .await?;
        }

        self.stats.record_network_update(&conditions);
        Ok(())
    }

    /// Handle packet loss
    pub async fn handle_packet_loss(&self, lost_sequence_numbers: &[u32]) -> Result<Vec<Vec<f32>>> {
        let mut concealed_packets = Vec::new();

        for &seq_num in lost_sequence_numbers {
            let concealed_audio = self
                .plc
                .lock()
                .await
                .conceal_packet(seq_num, self.audio_config.frame_size)
                .await?;
            concealed_packets.push(concealed_audio);
        }

        self.stats.record_packet_loss(lost_sequence_numbers.len());
        Ok(concealed_packets)
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> WebRTCProcessingStatistics {
        self.stats.get_statistics()
    }

    /// Get current latency
    pub fn get_current_latency(&self) -> Duration {
        self.stats.get_average_latency()
    }

    /// Check if processing is real-time
    pub fn is_realtime(&self) -> bool {
        let current_latency = self.get_current_latency().as_millis() as f64;
        current_latency <= self.conversion_mode.target_latency_ms()
    }

    // Internal implementation methods

    async fn adjust_config_for_mode(&mut self, mode: ConversionMode) -> Result<()> {
        match mode {
            ConversionMode::RealTime => {
                self.conversion_config.quality_mode = QualityMode::Low;
                self.conversion_config.enable_adaptive_quality = true;
            }
            ConversionMode::HighQuality => {
                self.conversion_config.quality_mode = QualityMode::High;
                self.conversion_config.enable_adaptive_quality = false;
            }
            ConversionMode::Balanced => {
                self.conversion_config.quality_mode = QualityMode::Balanced;
                self.conversion_config.enable_adaptive_quality = true;
            }
            ConversionMode::Privacy => {
                self.conversion_config.conversion_strength = 1.0;
                self.conversion_config.enable_noise_reduction = true;
            }
            ConversionMode::Entertainment => {
                self.conversion_config.quality_mode = QualityMode::High;
                self.conversion_config.conversion_strength = 1.0;
            }
        }

        Ok(())
    }

    async fn apply_voice_conversion(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Create conversion request
        let request = ConversionRequest::new(
            format!("webrtc_{}", fastrand::u64(..)),
            audio.to_vec(),
            self.audio_config.sample_rate,
            self.conversion_config.conversion_type.clone(),
            self.conversion_config.target.clone(),
        );

        // Apply conversion
        let result = self.voice_converter.convert(request).await?;

        // Apply conversion strength
        let mut converted_audio = result.converted_audio;
        if self.conversion_config.conversion_strength < 1.0 {
            for (i, &original) in audio.iter().enumerate() {
                if i < converted_audio.len() {
                    converted_audio[i] = original
                        * (1.0 - self.conversion_config.conversion_strength)
                        + converted_audio[i] * self.conversion_config.conversion_strength;
                }
            }
        }

        Ok(converted_audio)
    }

    async fn post_process_audio(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut processed = audio.to_vec();

        // Ensure proper frame size
        processed.resize(self.audio_config.frame_size, 0.0);

        // Apply gain normalization
        let max_amplitude = processed.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        if max_amplitude > 1.0 {
            for sample in &mut processed {
                *sample /= max_amplitude;
            }
        }

        // Apply soft clipping
        for sample in &mut processed {
            *sample = sample.clamp(-0.95, 0.95);
        }

        Ok(processed)
    }

    async fn start_network_monitoring(&self) -> Result<()> {
        let network_conditions = Arc::clone(&self.network_conditions);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                // Simulate network monitoring (would use real network stats in practice)
                let conditions = NetworkConditions {
                    bandwidth_kbps: 500 + fastrand::u32(0..1000),
                    rtt_ms: 20 + fastrand::u32(0..100),
                    packet_loss_percent: fastrand::f32() * 5.0,
                    jitter_ms: fastrand::f32() * 10.0,
                    quality_score: 70 + fastrand::u8(0..30),
                };

                *network_conditions.write().await = conditions.clone();
                stats.record_network_update(&conditions);
            }
        });

        Ok(())
    }
}

/// Audio buffer for WebRTC processing
pub struct AudioBuffer {
    buffer: VecDeque<f32>,
    capacity: usize,
}

impl AudioBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, audio: &[f32]) {
        for &sample in audio {
            if self.buffer.len() >= self.capacity {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }
    }

    fn pop(&mut self, count: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(sample) = self.buffer.pop_front() {
                result.push(sample);
            } else {
                result.push(0.0);
            }
        }
        result
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// Adaptive quality controller for WebRTC
pub struct AdaptiveQualityController {
    current_quality: QualityMode,
    target_latency_ms: f64,
    latency_history: VecDeque<f64>,
}

impl AdaptiveQualityController {
    fn new() -> Self {
        Self {
            current_quality: QualityMode::Balanced,
            target_latency_ms: 25.0,
            latency_history: VecDeque::with_capacity(10),
        }
    }

    async fn initialize(&mut self, config: &VoiceConversionConfig) -> Result<()> {
        self.current_quality = config.quality_mode;
        Ok(())
    }

    async fn update_config(&mut self, config: &VoiceConversionConfig) -> Result<()> {
        self.current_quality = config.quality_mode;
        Ok(())
    }

    async fn adjust_for_network(&mut self, conditions: &NetworkConditions) -> Result<()> {
        // Adjust quality based on network conditions
        if conditions.quality_score < 50 || conditions.packet_loss_percent > 3.0 {
            self.current_quality = QualityMode::Low;
        } else if conditions.quality_score > 80 && conditions.packet_loss_percent < 1.0 {
            self.current_quality = QualityMode::High;
        } else {
            self.current_quality = QualityMode::Balanced;
        }

        Ok(())
    }

    async fn report_latency_issue(&mut self, latency: Duration) {
        let latency_ms = latency.as_millis() as f64;

        self.latency_history.push_back(latency_ms);
        if self.latency_history.len() > 10 {
            self.latency_history.pop_front();
        }

        // If consistently over target, reduce quality
        let avg_latency =
            self.latency_history.iter().sum::<f64>() / self.latency_history.len() as f64;
        if avg_latency > self.target_latency_ms * 1.5 {
            self.current_quality = match self.current_quality {
                QualityMode::Maximum => QualityMode::High,
                QualityMode::High => QualityMode::Balanced,
                QualityMode::Balanced => QualityMode::Low,
                QualityMode::Low => QualityMode::Minimal,
                QualityMode::Minimal => QualityMode::Minimal,
            };
        }
    }
}

/// Audio enhancements for WebRTC
pub struct AudioEnhancements {
    config: WebRTCAudioConfig,
    enable_noise_reduction: bool,
    enable_echo_cancellation: bool,
    enable_agc: bool,
}

impl AudioEnhancements {
    fn new(config: &WebRTCAudioConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            enable_noise_reduction: true,
            enable_echo_cancellation: true,
            enable_agc: true,
        })
    }

    async fn initialize(&self) -> Result<()> {
        // Initialize audio processing algorithms
        Ok(())
    }

    async fn configure(&mut self, config: &VoiceConversionConfig) -> Result<()> {
        self.enable_noise_reduction = config.enable_noise_reduction;
        self.enable_echo_cancellation = config.enable_echo_cancellation;
        self.enable_agc = config.enable_agc;
        Ok(())
    }

    async fn process(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut processed = audio.to_vec();

        if self.enable_noise_reduction {
            processed = self.apply_noise_reduction(&processed)?;
        }

        if self.enable_echo_cancellation {
            processed = self.apply_echo_cancellation(&processed)?;
        }

        if self.enable_agc {
            processed = self.apply_agc(&processed)?;
        }

        Ok(processed)
    }

    fn apply_noise_reduction(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Simple noise gate implementation
        let threshold = 0.01;
        let mut processed = audio.to_vec();

        for sample in &mut processed {
            if sample.abs() < threshold {
                *sample *= 0.1; // Reduce low-amplitude samples
            }
        }

        Ok(processed)
    }

    fn apply_echo_cancellation(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Simplified echo cancellation (would use adaptive filters in practice)
        Ok(audio.to_vec())
    }

    fn apply_agc(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Automatic gain control
        let target_rms = 0.1;
        let current_rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();

        if current_rms > 0.0 {
            let gain = target_rms / current_rms;
            let safe_gain = gain.clamp(0.1, 3.0); // Limit gain range
            Ok(audio.iter().map(|&x| x * safe_gain).collect())
        } else {
            Ok(audio.to_vec())
        }
    }
}

/// Jitter buffer for WebRTC
pub struct JitterBuffer {
    buffer: VecDeque<Vec<f32>>,
    frame_size: usize,
    target_size: usize,
}

impl JitterBuffer {
    fn new(frame_size: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            frame_size,
            target_size: 3, // Target 3 frames buffered
        }
    }

    fn add_frame(&mut self, frame: Vec<f32>) {
        self.buffer.push_back(frame);

        // Limit buffer size
        while self.buffer.len() > 10 {
            self.buffer.pop_front();
        }
    }

    fn get_frame(&mut self) -> Result<Vec<f32>> {
        if self.buffer.len() >= self.target_size {
            Ok(self.buffer.pop_front().unwrap())
        } else if !self.buffer.is_empty() {
            Ok(self.buffer.pop_front().unwrap())
        } else {
            // Generate silence if no frames available
            Ok(vec![0.0; self.frame_size])
        }
    }
}

/// Packet loss concealer
pub struct PacketLossConcealer {
    sample_rate: u32,
    last_frame: Vec<f32>,
}

impl PacketLossConcealer {
    fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            last_frame: Vec::new(),
        }
    }

    async fn conceal_packet(
        &mut self,
        _sequence_number: u32,
        frame_size: usize,
    ) -> Result<Vec<f32>> {
        if self.last_frame.is_empty() {
            // No previous frame, generate silence
            Ok(vec![0.0; frame_size])
        } else {
            // Simple repetition with fade-out
            let mut concealed = self.last_frame.clone();
            concealed.resize(frame_size, 0.0);

            // Apply fade-out to avoid discontinuities
            for (i, sample) in concealed.iter_mut().enumerate() {
                let fade_factor = 1.0 - (i as f32 / frame_size as f32) * 0.5;
                *sample *= fade_factor;
            }

            Ok(concealed)
        }
    }

    fn update_last_frame(&mut self, frame: &[f32]) {
        self.last_frame = frame.to_vec();
    }
}

/// WebRTC processing statistics
pub struct WebRTCProcessingStats {
    total_frames_processed: AtomicU64,
    total_processing_time: AtomicU64,
    packet_losses: AtomicU32,
    mode_changes: AtomicU32,
    initialization_count: AtomicU32,
    network_updates: AtomicU64,
    latency_violations: AtomicU32,
}

impl WebRTCProcessingStats {
    fn new() -> Self {
        Self {
            total_frames_processed: AtomicU64::new(0),
            total_processing_time: AtomicU64::new(0),
            packet_losses: AtomicU32::new(0),
            mode_changes: AtomicU32::new(0),
            initialization_count: AtomicU32::new(0),
            network_updates: AtomicU64::new(0),
            latency_violations: AtomicU32::new(0),
        }
    }

    fn record_processing(&self, duration: Duration, frame_samples: usize) {
        self.total_frames_processed.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);

        // Check for latency violations
        if duration.as_millis() > 20 {
            // 20ms threshold
            self.latency_violations.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_packet_loss(&self, lost_packets: usize) {
        self.packet_losses
            .fetch_add(lost_packets as u32, Ordering::Relaxed);
    }

    fn record_mode_change(&self, _mode: ConversionMode) {
        self.mode_changes.fetch_add(1, Ordering::Relaxed);
    }

    fn record_initialization(&self) {
        self.initialization_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_network_update(&self, _conditions: &NetworkConditions) {
        self.network_updates.fetch_add(1, Ordering::Relaxed);
    }

    fn get_statistics(&self) -> WebRTCProcessingStatistics {
        let total_frames = self.total_frames_processed.load(Ordering::Relaxed);
        let total_time_us = self.total_processing_time.load(Ordering::Relaxed);

        let average_latency_ms = if total_frames > 0 {
            (total_time_us as f64 / total_frames as f64) / 1000.0
        } else {
            0.0
        };

        WebRTCProcessingStatistics {
            total_frames_processed: total_frames,
            average_latency_ms,
            packet_losses: self.packet_losses.load(Ordering::Relaxed),
            mode_changes: self.mode_changes.load(Ordering::Relaxed),
            initialization_count: self.initialization_count.load(Ordering::Relaxed),
            network_updates: self.network_updates.load(Ordering::Relaxed),
            latency_violations: self.latency_violations.load(Ordering::Relaxed),
        }
    }

    fn get_average_latency(&self) -> Duration {
        let total_frames = self.total_frames_processed.load(Ordering::Relaxed);
        let total_time_us = self.total_processing_time.load(Ordering::Relaxed);

        if total_frames > 0 {
            Duration::from_micros(total_time_us / total_frames)
        } else {
            Duration::from_micros(0)
        }
    }
}

/// WebRTC processing statistics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebRTCProcessingStatistics {
    /// Total frames processed
    pub total_frames_processed: u64,
    /// Average latency in milliseconds
    pub average_latency_ms: f64,
    /// Number of packet losses handled
    pub packet_losses: u32,
    /// Number of mode changes
    pub mode_changes: u32,
    /// Number of initializations
    pub initialization_count: u32,
    /// Number of network condition updates
    pub network_updates: u64,
    /// Number of latency violations
    pub latency_violations: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_mode_properties() {
        assert!(
            ConversionMode::RealTime.target_latency_ms()
                < ConversionMode::HighQuality.target_latency_ms()
        );
        assert!(
            ConversionMode::HighQuality.quality_level() > ConversionMode::RealTime.quality_level()
        );
    }

    #[test]
    fn test_quality_mode_ordering() {
        let modes = vec![
            QualityMode::Minimal,
            QualityMode::Low,
            QualityMode::Balanced,
            QualityMode::High,
            QualityMode::Maximum,
        ];

        assert_eq!(modes.len(), 5);
    }

    #[tokio::test]
    async fn test_webrtc_processor_creation() {
        let processor = WebRTCVoiceProcessor::new().await;
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_webrtc_processor_initialization() {
        let mut processor = WebRTCVoiceProcessor::new().await.unwrap();
        assert!(processor.initialize().await.is_ok());
        assert!(processor.initialized.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_audio_chunk_processing() {
        let mut processor = WebRTCVoiceProcessor::new().await.unwrap();
        processor.initialize().await.unwrap();

        let audio_chunk = vec![0.1; 480]; // 10ms at 48kHz
        let result = processor.process_audio_chunk(&audio_chunk).await;
        assert!(result.is_ok());

        let processed_chunk = result.unwrap();
        assert_eq!(processed_chunk.len(), 480);
    }

    #[tokio::test]
    async fn test_conversion_mode_setting() {
        let mut processor = WebRTCVoiceProcessor::new().await.unwrap();
        processor.initialize().await.unwrap();

        assert!(processor
            .set_conversion_mode(ConversionMode::RealTime)
            .await
            .is_ok());
        assert_eq!(processor.conversion_mode, ConversionMode::RealTime);
    }

    #[tokio::test]
    async fn test_voice_conversion_config() {
        let mut processor = WebRTCVoiceProcessor::new().await.unwrap();
        processor.initialize().await.unwrap();

        let config = VoiceConversionConfig {
            conversion_type: ConversionType::PitchShift,
            quality_mode: QualityMode::High,
            enable_noise_reduction: true,
            conversion_strength: 0.8,
            ..VoiceConversionConfig::default()
        };

        assert!(processor.configure_conversion(config).await.is_ok());
    }

    #[tokio::test]
    async fn test_network_conditions_update() {
        let processor = WebRTCVoiceProcessor::new().await.unwrap();

        let conditions = NetworkConditions {
            bandwidth_kbps: 500,
            rtt_ms: 100,
            packet_loss_percent: 2.0,
            jitter_ms: 10.0,
            quality_score: 70,
        };

        assert!(processor
            .update_network_conditions(conditions)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn test_packet_loss_handling() {
        let processor = WebRTCVoiceProcessor::new().await.unwrap();

        let lost_packets = vec![1, 3, 5];
        let result = processor.handle_packet_loss(&lost_packets).await;
        assert!(result.is_ok());

        let concealed_packets = result.unwrap();
        assert_eq!(concealed_packets.len(), 3);
    }

    #[test]
    fn test_audio_buffer() {
        let mut buffer = AudioBuffer::new(100);
        let audio = vec![0.1, 0.2, 0.3];

        buffer.push(&audio);
        assert_eq!(buffer.len(), 3);

        let popped = buffer.pop(2);
        assert_eq!(popped.len(), 2);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_jitter_buffer() {
        let mut jitter_buffer = JitterBuffer::new(480);

        jitter_buffer.add_frame(vec![0.1; 480]);
        jitter_buffer.add_frame(vec![0.2; 480]);
        jitter_buffer.add_frame(vec![0.3; 480]);

        let frame = jitter_buffer.get_frame();
        assert!(frame.is_ok());
        assert_eq!(frame.unwrap().len(), 480);
    }

    #[tokio::test]
    async fn test_packet_loss_concealer() {
        let mut plc = PacketLossConcealer::new(48000);

        let result = plc.conceal_packet(1, 480).await;
        assert!(result.is_ok());

        let concealed = result.unwrap();
        assert_eq!(concealed.len(), 480);
    }

    #[test]
    fn test_webrtc_audio_config() {
        let config = WebRTCAudioConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.frame_size, 480);
        assert_eq!(config.channels, 1);
    }

    #[test]
    fn test_network_conditions() {
        let conditions = NetworkConditions::default();
        assert_eq!(conditions.bandwidth_kbps, 1000);
        assert_eq!(conditions.rtt_ms, 50);
        assert_eq!(conditions.quality_score, 100);
    }

    #[test]
    fn test_webrtc_stats() {
        let stats = WebRTCProcessingStats::new();
        stats.record_processing(Duration::from_millis(10), 480);
        stats.record_packet_loss(2);
        stats.record_mode_change(ConversionMode::RealTime);

        let statistics = stats.get_statistics();
        assert_eq!(statistics.total_frames_processed, 1);
        assert_eq!(statistics.packet_losses, 2);
        assert_eq!(statistics.mode_changes, 1);
    }
}
