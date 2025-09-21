//! WebSocket interface for real-time evaluation services
//!
//! This module provides WebSocket support for real-time audio evaluation,
//! enabling streaming quality assessment and live feedback capabilities.

use crate::quality::QualityEvaluator;
use crate::traits::{QualityEvaluationConfig, QualityEvaluator as QualityEvaluatorTrait};
use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use voirs_sdk::AudioBuffer;

/// WebSocket-specific errors
#[derive(Error, Debug)]
pub enum WebSocketError {
    /// Connection error
    #[error("WebSocket connection error: {0}")]
    ConnectionError(String),
    /// Invalid message format
    #[error("Invalid WebSocket message: {0}")]
    InvalidMessage(String),
    /// Authentication error
    #[error("WebSocket authentication failed: {0}")]
    AuthenticationError(String),
    /// Evaluation service error
    #[error("WebSocket evaluation error: {0}")]
    EvaluationError(String),
    /// Buffer overflow error
    #[error("Audio buffer overflow: {0}")]
    BufferOverflow(String),
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    /// Authentication message
    #[serde(rename = "auth")]
    Authentication {
        /// API key for authentication
        api_key: String,
        /// User identifier
        user_id: String,
        /// Session identifier
        session_id: String,
    },
    /// Start evaluation session
    #[serde(rename = "start_session")]
    StartSession {
        /// Session configuration
        config: SessionConfig,
    },
    /// Audio chunk for real-time evaluation
    #[serde(rename = "audio_chunk")]
    AudioChunk {
        /// Session identifier
        session_id: String,
        /// Chunk sequence number
        chunk_id: u64,
        /// Base64 encoded audio data
        audio_data: String,
        /// Timestamp in seconds
        timestamp: f64,
        /// Whether this is the final chunk
        is_final: bool,
    },
    /// Real-time evaluation result
    #[serde(rename = "evaluation_result")]
    EvaluationResult {
        /// Session identifier
        session_id: String,
        /// Chunk sequence number
        chunk_id: u64,
        /// Timestamp in seconds
        timestamp: f64,
        /// Overall quality score (0-1)
        quality_score: f64,
        /// Individual metric scores
        metrics: HashMap<String, f64>,
        /// Confidence level (0-1)
        confidence: f64,
        /// Detailed analysis results
        analysis: RealtimeAnalysis,
    },
    /// Session status update
    #[serde(rename = "session_status")]
    SessionStatus {
        /// Session identifier
        session_id: String,
        /// Current session status
        status: String,
        /// Status message
        message: String,
        /// Session statistics
        statistics: SessionStatistics,
    },
    /// Error message
    #[serde(rename = "error")]
    Error {
        /// Error code
        error_code: String,
        /// Error message
        message: String,
        /// Session identifier (if applicable)
        session_id: Option<String>,
    },
    /// Heartbeat/ping message
    #[serde(rename = "ping")]
    Ping {
        /// Timestamp in milliseconds
        timestamp: u64,
    },
    /// Heartbeat/pong response
    #[serde(rename = "pong")]
    Pong {
        /// Timestamp in milliseconds
        timestamp: u64,
    },
}

/// Session configuration for real-time evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Audio format configuration
    pub audio_format: AudioFormat,
    /// Evaluation metrics to compute
    pub metrics: Vec<String>,
    /// Chunk size in milliseconds
    pub chunk_size_ms: u32,
    /// Buffer size in chunks
    pub buffer_size: u32,
    /// Evaluation language
    pub language: Option<String>,
    /// Quality thresholds for alerts
    pub quality_thresholds: QualityThresholds,
    /// Real-time processing options
    pub processing_options: ProcessingOptions,
}

/// Audio format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormat {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Bit depth
    pub bit_depth: u8,
    /// Encoding format
    pub encoding: String,
}

/// Quality thresholds for real-time alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum acceptable quality score
    pub min_quality: f64,
    /// Warning threshold
    pub warning_threshold: f64,
    /// Alert threshold
    pub alert_threshold: f64,
    /// Confidence threshold for reliable results
    pub confidence_threshold: f64,
}

/// Processing options for real-time evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOptions {
    /// Enable adaptive quality adjustment
    pub adaptive_quality: bool,
    /// Enable noise reduction
    pub noise_reduction: bool,
    /// Enable automatic gain control
    pub auto_gain_control: bool,
    /// Buffer overlap percentage (0-50)
    pub buffer_overlap: f32,
    /// Enable quality prediction
    pub quality_prediction: bool,
}

/// Real-time analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAnalysis {
    /// Signal level analysis
    pub signal_level: SignalLevel,
    /// Spectral analysis results
    pub spectral_analysis: SpectralAnalysis,
    /// Quality trends
    pub quality_trend: QualityTrend,
    /// Detected issues
    pub detected_issues: Vec<DetectedIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Signal level analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalLevel {
    /// RMS level (dB)
    pub rms_db: f64,
    /// Peak level (dB)
    pub peak_db: f64,
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// Clipping percentage
    pub clipping_percent: f64,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,
}

/// Spectral analysis for real-time evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    /// Spectral centroid (Hz)
    pub spectral_centroid: f64,
    /// Spectral rolloff (Hz)
    pub spectral_rolloff: f64,
    /// Spectral flatness
    pub spectral_flatness: f64,
    /// Fundamental frequency (Hz)
    pub f0_hz: Option<f64>,
    /// Harmonic-to-noise ratio (dB)
    pub hnr_db: f64,
}

/// Quality trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrend {
    /// Current quality score
    pub current_score: f64,
    /// Moving average (last 10 chunks)
    pub moving_average: f64,
    /// Trend direction (-1: decreasing, 0: stable, 1: improving)
    pub trend_direction: i8,
    /// Trend strength (0-1)
    pub trend_strength: f64,
    /// Quality variance
    pub variance: f64,
}

/// Detected quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedIssue {
    /// Issue type
    pub issue_type: String,
    /// Severity level (1-5)
    pub severity: u8,
    /// Issue description
    pub description: String,
    /// Timestamp when detected
    pub timestamp: f64,
    /// Suggested action
    pub suggested_action: String,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Total chunks processed
    pub total_chunks: u64,
    /// Session duration (seconds)
    pub duration_seconds: f64,
    /// Average quality score
    pub average_quality: f64,
    /// Minimum quality score
    pub min_quality: f64,
    /// Maximum quality score
    pub max_quality: f64,
    /// Total processing time (ms)
    pub total_processing_time_ms: u64,
    /// Average processing time per chunk (ms)
    pub avg_processing_time_ms: f64,
    /// Number of quality alerts
    pub quality_alerts: u32,
    /// Data transfer statistics
    pub data_transfer: DataTransferStats,
}

/// Data transfer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransferStats {
    /// Total bytes received
    pub bytes_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Average throughput (bytes/sec)
    pub avg_throughput: f64,
    /// Number of dropped packets
    pub dropped_packets: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            audio_format: AudioFormat {
                sample_rate: 16000,
                channels: 1,
                bit_depth: 16,
                encoding: "pcm".to_string(),
            },
            metrics: vec!["pesq".to_string(), "stoi".to_string()],
            chunk_size_ms: 100,
            buffer_size: 10,
            language: None,
            quality_thresholds: QualityThresholds {
                min_quality: 0.5,
                warning_threshold: 0.7,
                alert_threshold: 0.3,
                confidence_threshold: 0.8,
            },
            processing_options: ProcessingOptions {
                adaptive_quality: true,
                noise_reduction: true,
                auto_gain_control: true,
                buffer_overlap: 20.0,
                quality_prediction: true,
            },
        }
    }
}

/// WebSocket session manager
pub struct WebSocketSessionManager {
    sessions: std::sync::Arc<std::sync::RwLock<HashMap<String, ActiveSession>>>,
    config: WebSocketConfig,
    quality_evaluator: QualityEvaluator,
}

/// Active WebSocket session
#[derive(Debug)]
struct ActiveSession {
    session_id: String,
    user_id: String,
    config: SessionConfig,
    start_time: std::time::Instant,
    last_activity: std::time::Instant,
    buffer: StreamingAudioBuffer,
    statistics: SessionStatistics,
    quality_history: Vec<f64>,
}

/// Audio buffer for real-time processing
#[derive(Debug)]
struct StreamingAudioBuffer {
    chunks: std::collections::VecDeque<AudioChunk>,
    max_size: usize,
    current_size: usize,
}

/// Audio chunk data
#[derive(Debug, Clone)]
struct AudioChunk {
    chunk_id: u64,
    data: Vec<f32>,
    timestamp: f64,
    sample_rate: u32,
}

/// WebSocket service configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// Maximum concurrent sessions
    pub max_sessions: usize,
    /// Session timeout (seconds)
    pub session_timeout: u64,
    /// Maximum chunk size (bytes)
    pub max_chunk_size: usize,
    /// Enable quality prediction
    pub enable_prediction: bool,
    /// Heartbeat interval (seconds)
    pub heartbeat_interval: u64,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            max_sessions: 100,
            session_timeout: 300,
            max_chunk_size: 8192,
            enable_prediction: true,
            heartbeat_interval: 30,
        }
    }
}

impl WebSocketSessionManager {
    /// Create a new WebSocket session manager
    pub async fn new(config: WebSocketConfig) -> Result<Self, WebSocketError> {
        let quality_evaluator = QualityEvaluator::new().await.map_err(|e| {
            WebSocketError::EvaluationError(format!(
                "Failed to initialize quality evaluator: {}",
                e
            ))
        })?;

        Ok(Self {
            sessions: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            config,
            quality_evaluator,
        })
    }

    /// Start a new evaluation session
    pub fn start_session(
        &self,
        session_id: String,
        user_id: String,
        config: SessionConfig,
    ) -> Result<(), WebSocketError> {
        let mut sessions = self.sessions.write().unwrap();

        if sessions.len() >= self.config.max_sessions {
            return Err(WebSocketError::ConnectionError(
                "Maximum number of sessions reached".to_string(),
            ));
        }

        let session = ActiveSession {
            session_id: session_id.clone(),
            user_id,
            config: config.clone(),
            start_time: std::time::Instant::now(),
            last_activity: std::time::Instant::now(),
            buffer: StreamingAudioBuffer {
                chunks: std::collections::VecDeque::new(),
                max_size: config.buffer_size as usize,
                current_size: 0,
            },
            statistics: SessionStatistics {
                total_chunks: 0,
                duration_seconds: 0.0,
                average_quality: 0.0,
                min_quality: f64::MAX,
                max_quality: f64::MIN,
                total_processing_time_ms: 0,
                avg_processing_time_ms: 0.0,
                quality_alerts: 0,
                data_transfer: DataTransferStats {
                    bytes_received: 0,
                    bytes_sent: 0,
                    avg_throughput: 0.0,
                    dropped_packets: 0,
                },
            },
            quality_history: Vec::new(),
        };

        sessions.insert(session_id, session);
        Ok(())
    }

    /// Process audio chunk for real-time evaluation
    pub async fn process_audio_chunk(
        &self,
        session_id: &str,
        chunk_id: u64,
        audio_data: Vec<f32>,
        timestamp: f64,
    ) -> Result<WebSocketMessage, WebSocketError> {
        let start_time = std::time::Instant::now();

        // Extract session config before the async operations
        let session_config = {
            let mut sessions = self.sessions.write().unwrap();
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| WebSocketError::InvalidMessage("Session not found".to_string()))?;

            // Update last activity
            session.last_activity = std::time::Instant::now();

            // Add chunk to buffer
            let chunk = AudioChunk {
                chunk_id,
                data: audio_data.clone(),
                timestamp,
                sample_rate: session.config.audio_format.sample_rate,
            };

            session.buffer.chunks.push_back(chunk);
            session.buffer.current_size += audio_data.len();

            // Remove old chunks if buffer is full
            while session.buffer.chunks.len() > session.buffer.max_size {
                if let Some(old_chunk) = session.buffer.chunks.pop_front() {
                    session.buffer.current_size -= old_chunk.data.len();
                }
            }

            // Clone config for use outside the lock
            session.config.clone()
        }; // Lock is dropped here

        // Perform real-time evaluation
        let quality_score = self.evaluate_chunk(&audio_data, &session_config).await?;
        let analysis = self.analyze_chunk(&audio_data, &session_config)?;

        // Update session statistics (acquire lock again)
        let (confidence, processing_time, analysis_result) = {
            let mut sessions = self.sessions.write().unwrap();
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| WebSocketError::InvalidMessage("Session not found".to_string()))?;

            let confidence = self.calculate_confidence(&session.quality_history, quality_score);

            session.statistics.total_chunks += 1;
            session.statistics.duration_seconds = session.start_time.elapsed().as_secs_f64();
            session.quality_history.push(quality_score);

            // Keep only last 100 quality scores for trend analysis
            if session.quality_history.len() > 100 {
                session.quality_history.remove(0);
            }

            session.statistics.average_quality =
                session.quality_history.iter().sum::<f64>() / session.quality_history.len() as f64;
            session.statistics.min_quality = session.statistics.min_quality.min(quality_score);
            session.statistics.max_quality = session.statistics.max_quality.max(quality_score);

            let processing_time = start_time.elapsed().as_millis() as u64;
            session.statistics.total_processing_time_ms += processing_time;
            session.statistics.avg_processing_time_ms = session.statistics.total_processing_time_ms
                as f64
                / session.statistics.total_chunks as f64;

            // Check quality thresholds
            if quality_score < session.config.quality_thresholds.alert_threshold {
                session.statistics.quality_alerts += 1;
            }

            (confidence, processing_time, analysis.clone())
        }; // Lock is dropped here

        // Create metrics map
        let mut metrics = HashMap::new();
        metrics.insert("quality".to_string(), quality_score);
        metrics.insert("rms_level".to_string(), analysis.signal_level.rms_db);
        metrics.insert(
            "spectral_centroid".to_string(),
            analysis.spectral_analysis.spectral_centroid,
        );

        Ok(WebSocketMessage::EvaluationResult {
            session_id: session_id.to_string(),
            chunk_id,
            timestamp,
            quality_score,
            metrics,
            confidence,
            analysis,
        })
    }

    /// Get session status
    pub fn get_session_status(&self, session_id: &str) -> Result<WebSocketMessage, WebSocketError> {
        let sessions = self.sessions.read().unwrap();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| WebSocketError::InvalidMessage("Session not found".to_string()))?;

        Ok(WebSocketMessage::SessionStatus {
            session_id: session_id.to_string(),
            status: "active".to_string(),
            message: "Session running normally".to_string(),
            statistics: session.statistics.clone(),
        })
    }

    /// End evaluation session
    pub fn end_session(&self, session_id: &str) -> Result<SessionStatistics, WebSocketError> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .remove(session_id)
            .ok_or_else(|| WebSocketError::InvalidMessage("Session not found".to_string()))?;

        Ok(session.statistics)
    }

    /// Start WebSocket server
    pub async fn start_server(&self, host: &str, port: u16) -> Result<(), WebSocketError> {
        use futures_util::{SinkExt, StreamExt};
        use warp::Filter;

        let manager = std::sync::Arc::new(self.clone());

        // WebSocket upgrade handler
        let websocket_route = warp::path("ws")
            .and(warp::path("evaluate"))
            .and(warp::ws())
            .map({
                let manager = manager.clone();
                move |ws: warp::ws::Ws| {
                    let manager = manager.clone();
                    ws.on_upgrade(move |websocket| handle_websocket_connection(websocket, manager))
                }
            });

        // Health check for WebSocket service
        let health_route = warp::path("ws")
            .and(warp::path("health"))
            .and(warp::get())
            .map(|| {
                warp::reply::json(&serde_json::json!({
                    "status": "healthy",
                    "service": "websocket",
                    "timestamp": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                }))
            });

        // WebSocket status endpoint
        let status_route = warp::path("ws")
            .and(warp::path("status"))
            .and(warp::get())
            .map({
                let manager = manager.clone();
                move || {
                    let sessions = manager.sessions.read().unwrap();
                    let status = serde_json::json!({
                        "active_sessions": sessions.len(),
                        "max_sessions": manager.config.max_sessions,
                        "uptime_seconds": std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    });
                    warp::reply::json(&status)
                }
            });

        let routes = websocket_route
            .or(health_route)
            .or(status_route)
            .with(warp::cors().allow_any_origin());

        println!("🔌 Starting WebSocket server on {}:{}", host, port);
        println!("   WebSocket endpoint: ws://{}:{}/ws/evaluate", host, port);
        println!("   Health check: http://{}:{}/ws/health", host, port);
        println!("   Status: http://{}:{}/ws/status", host, port);

        warp::serve(routes)
            .run((
                host.parse::<std::net::IpAddr>()
                    .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1))),
                port,
            ))
            .await;

        Ok(())
    }

    /// Cleanup expired sessions
    pub fn cleanup_expired_sessions(&self) {
        let mut sessions = self.sessions.write().unwrap();
        let now = std::time::Instant::now();
        let timeout_duration = std::time::Duration::from_secs(self.config.session_timeout);

        sessions.retain(|_, session| now.duration_since(session.last_activity) < timeout_duration);
    }

    /// Get active session count
    pub fn get_active_session_count(&self) -> usize {
        let sessions = self.sessions.read().unwrap();
        sessions.len()
    }

    /// Evaluate audio chunk using quality evaluator
    async fn evaluate_chunk(
        &self,
        audio_data: &[f32],
        config: &SessionConfig,
    ) -> Result<f64, WebSocketError> {
        // Create AudioBuffer from the chunk
        let audio_buffer = AudioBuffer::new(
            audio_data.to_vec(),
            config.audio_format.sample_rate,
            config.audio_format.channels.into(),
        );

        // Create evaluation config
        let eval_config = QualityEvaluationConfig::default();

        // Perform evaluation
        let result = self
            .quality_evaluator
            .evaluate_quality(&audio_buffer, None, Some(&eval_config))
            .await
            .map_err(|e| {
                WebSocketError::EvaluationError(format!("Quality evaluation failed: {}", e))
            })?;

        Ok(result.overall_score as f64)
    }

    /// Calculate confidence based on quality history
    fn calculate_confidence(&self, quality_history: &[f64], current_score: f64) -> f64 {
        if quality_history.len() < 3 {
            return 0.5; // Low confidence with insufficient data
        }

        // Calculate variance in recent quality scores
        let recent_scores: Vec<f64> = quality_history.iter().rev().take(10).copied().collect();
        let mean = recent_scores.iter().sum::<f64>() / recent_scores.len() as f64;
        let variance = recent_scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / recent_scores.len() as f64;

        // Higher variance means lower confidence
        let confidence = (1.0 - variance.min(1.0)).max(0.1);
        confidence
    }

    /// Analyze audio chunk for real-time feedback
    fn analyze_chunk(
        &self,
        audio_data: &[f32],
        config: &SessionConfig,
    ) -> Result<RealtimeAnalysis, WebSocketError> {
        // Calculate signal level metrics
        let rms = if !audio_data.is_empty() {
            let sum_squares: f32 = audio_data.iter().map(|x| x * x).sum();
            (sum_squares / audio_data.len() as f32).sqrt()
        } else {
            0.0
        };

        let peak = audio_data.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
        let rms_db = if rms > 0.0 { 20.0 * rms.log10() } else { -80.0 };
        let peak_db = if peak > 0.0 {
            20.0 * peak.log10()
        } else {
            -80.0
        };

        // Calculate spectral features (simplified)
        let spectral_centroid =
            calculate_spectral_centroid(audio_data, config.audio_format.sample_rate);
        let spectral_rolloff =
            calculate_spectral_rolloff(audio_data, config.audio_format.sample_rate);

        // Detect potential issues
        let mut detected_issues = Vec::new();

        if rms_db < -30.0 {
            detected_issues.push(DetectedIssue {
                issue_type: "low_level".to_string(),
                severity: 3,
                description: "Audio level is very low".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
                suggested_action: "Increase input gain".to_string(),
            });
        }

        if peak > 0.95 {
            detected_issues.push(DetectedIssue {
                issue_type: "clipping".to_string(),
                severity: 4,
                description: "Audio signal is clipping".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
                suggested_action: "Reduce input gain to prevent distortion".to_string(),
            });
        }

        // Generate recommendations
        let mut recommendations = Vec::new();
        if rms_db < -20.0 && rms_db > -40.0 {
            recommendations.push("Consider increasing audio level for better quality".to_string());
        }
        if spectral_centroid < 1000.0 {
            recommendations.push(
                "Audio appears to have low frequency content - check microphone positioning"
                    .to_string(),
            );
        }

        Ok(RealtimeAnalysis {
            signal_level: SignalLevel {
                rms_db: rms_db as f64,
                peak_db: peak_db as f64,
                dynamic_range: (peak_db - rms_db) as f64,
                clipping_percent: if peak > 0.95 {
                    audio_data.iter().filter(|&&x| x.abs() > 0.95).count() as f64
                        / audio_data.len() as f64
                        * 100.0
                } else {
                    0.0
                },
                snr_db: estimate_snr(audio_data) as f64,
            },
            spectral_analysis: SpectralAnalysis {
                spectral_centroid: spectral_centroid as f64,
                spectral_rolloff: spectral_rolloff as f64,
                spectral_flatness: calculate_spectral_flatness(audio_data),
                f0_hz: estimate_fundamental_frequency(audio_data, config.audio_format.sample_rate),
                hnr_db: estimate_hnr(audio_data) as f64,
            },
            quality_trend: QualityTrend {
                current_score: 0.8, // Would be calculated from actual quality metrics
                moving_average: 0.8,
                trend_direction: 0,
                trend_strength: 0.1,
                variance: 0.05,
            },
            detected_issues,
            recommendations,
        })
    }
}

impl Clone for WebSocketSessionManager {
    fn clone(&self) -> Self {
        Self {
            sessions: self.sessions.clone(),
            config: self.config.clone(),
            quality_evaluator: self.quality_evaluator.clone(),
        }
    }
}

/// Handle individual WebSocket connection
async fn handle_websocket_connection(
    websocket: warp::ws::WebSocket,
    manager: std::sync::Arc<WebSocketSessionManager>,
) {
    use futures_util::{SinkExt, StreamExt};

    let (mut ws_tx, mut ws_rx) = websocket.split();

    // Connection state
    let mut authenticated = false;
    let mut current_session_id: Option<String> = None;
    let mut last_heartbeat = std::time::Instant::now();

    // Send welcome message
    let welcome_msg = WebSocketMessage::SessionStatus {
        session_id: "connection".to_string(),
        status: "connected".to_string(),
        message: "WebSocket connection established. Please authenticate.".to_string(),
        statistics: SessionStatistics {
            total_chunks: 0,
            duration_seconds: 0.0,
            average_quality: 0.0,
            min_quality: 0.0,
            max_quality: 0.0,
            total_processing_time_ms: 0,
            avg_processing_time_ms: 0.0,
            quality_alerts: 0,
            data_transfer: DataTransferStats {
                bytes_received: 0,
                bytes_sent: 0,
                avg_throughput: 0.0,
                dropped_packets: 0,
            },
        },
    };

    if let Ok(msg_json) = serde_json::to_string(&welcome_msg) {
        let _ = ws_tx.send(warp::ws::Message::text(msg_json)).await;
    }

    // Main message loop
    while let Some(result) = ws_rx.next().await {
        match result {
            Ok(msg) => {
                if msg.is_text() {
                    let text = msg.to_str().unwrap_or("");
                    match serde_json::from_str::<WebSocketMessage>(text) {
                        Ok(ws_msg) => {
                            let response = handle_websocket_message(
                                ws_msg,
                                &manager,
                                &mut authenticated,
                                &mut current_session_id,
                            )
                            .await;

                            if let Ok(response_json) = serde_json::to_string(&response) {
                                let _ = ws_tx.send(warp::ws::Message::text(response_json)).await;
                            }
                        }
                        Err(e) => {
                            let error_msg = WebSocketMessage::Error {
                                error_code: "INVALID_MESSAGE".to_string(),
                                message: format!("Invalid message format: {}", e),
                                session_id: current_session_id.clone(),
                            };
                            if let Ok(error_json) = serde_json::to_string(&error_msg) {
                                let _ = ws_tx.send(warp::ws::Message::text(error_json)).await;
                            }
                        }
                    }
                } else if msg.is_ping() {
                    let _ = ws_tx.send(warp::ws::Message::pong(msg.into_bytes())).await;
                    last_heartbeat = std::time::Instant::now();
                }
            }
            Err(_) => {
                // Connection error - cleanup and exit
                break;
            }
        }

        // Check for heartbeat timeout
        if last_heartbeat.elapsed().as_secs() > manager.config.heartbeat_interval * 2 {
            break;
        }
    }

    // Cleanup session on disconnect
    if let Some(session_id) = current_session_id {
        let _ = manager.end_session(&session_id);
    }
}

/// Handle individual WebSocket message
async fn handle_websocket_message(
    message: WebSocketMessage,
    manager: &WebSocketSessionManager,
    authenticated: &mut bool,
    current_session_id: &mut Option<String>,
) -> WebSocketMessage {
    match message {
        WebSocketMessage::Authentication {
            api_key,
            user_id,
            session_id,
        } => {
            // Simplified authentication - in production, validate against database
            if api_key.len() >= 32 && !user_id.is_empty() {
                *authenticated = true;
                *current_session_id = Some(session_id.clone());

                WebSocketMessage::SessionStatus {
                    session_id: session_id.clone(),
                    status: "authenticated".to_string(),
                    message: "Authentication successful".to_string(),
                    statistics: SessionStatistics {
                        total_chunks: 0,
                        duration_seconds: 0.0,
                        average_quality: 0.0,
                        min_quality: 0.0,
                        max_quality: 0.0,
                        total_processing_time_ms: 0,
                        avg_processing_time_ms: 0.0,
                        quality_alerts: 0,
                        data_transfer: DataTransferStats {
                            bytes_received: 0,
                            bytes_sent: 0,
                            avg_throughput: 0.0,
                            dropped_packets: 0,
                        },
                    },
                }
            } else {
                WebSocketMessage::Error {
                    error_code: "AUTH_FAILED".to_string(),
                    message: "Invalid credentials".to_string(),
                    session_id: Some(session_id),
                }
            }
        }

        WebSocketMessage::StartSession { config } => {
            if !*authenticated {
                return WebSocketMessage::Error {
                    error_code: "NOT_AUTHENTICATED".to_string(),
                    message: "Authentication required".to_string(),
                    session_id: current_session_id.clone(),
                };
            }

            if let Some(session_id) = current_session_id {
                match manager.start_session(session_id.clone(), "user".to_string(), config) {
                    Ok(()) => WebSocketMessage::SessionStatus {
                        session_id: session_id.clone(),
                        status: "started".to_string(),
                        message: "Evaluation session started".to_string(),
                        statistics: SessionStatistics {
                            total_chunks: 0,
                            duration_seconds: 0.0,
                            average_quality: 0.0,
                            min_quality: 0.0,
                            max_quality: 0.0,
                            total_processing_time_ms: 0,
                            avg_processing_time_ms: 0.0,
                            quality_alerts: 0,
                            data_transfer: DataTransferStats {
                                bytes_received: 0,
                                bytes_sent: 0,
                                avg_throughput: 0.0,
                                dropped_packets: 0,
                            },
                        },
                    },
                    Err(e) => WebSocketMessage::Error {
                        error_code: "SESSION_START_FAILED".to_string(),
                        message: e.to_string(),
                        session_id: Some(session_id.clone()),
                    },
                }
            } else {
                WebSocketMessage::Error {
                    error_code: "NO_SESSION".to_string(),
                    message: "No active session".to_string(),
                    session_id: None,
                }
            }
        }

        WebSocketMessage::AudioChunk {
            session_id,
            chunk_id,
            audio_data,
            timestamp,
            is_final: _,
        } => {
            if !*authenticated {
                return WebSocketMessage::Error {
                    error_code: "NOT_AUTHENTICATED".to_string(),
                    message: "Authentication required".to_string(),
                    session_id: Some(session_id),
                };
            }

            // Decode base64 audio data
            match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &audio_data) {
                Ok(audio_bytes) => {
                    // Convert bytes to f32 samples (assuming 16-bit PCM)
                    let samples: Vec<f32> = audio_bytes
                        .chunks(2)
                        .map(|chunk| {
                            if chunk.len() == 2 {
                                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                                sample as f32 / 32768.0
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    match manager
                        .process_audio_chunk(&session_id, chunk_id, samples, timestamp)
                        .await
                    {
                        Ok(result) => result,
                        Err(e) => WebSocketMessage::Error {
                            error_code: "PROCESSING_FAILED".to_string(),
                            message: e.to_string(),
                            session_id: Some(session_id),
                        },
                    }
                }
                Err(e) => WebSocketMessage::Error {
                    error_code: "INVALID_AUDIO_DATA".to_string(),
                    message: format!("Failed to decode audio data: {}", e),
                    session_id: Some(session_id),
                },
            }
        }

        WebSocketMessage::Ping { timestamp } => WebSocketMessage::Pong { timestamp },

        _ => WebSocketMessage::Error {
            error_code: "UNSUPPORTED_MESSAGE".to_string(),
            message: "Message type not supported".to_string(),
            session_id: current_session_id.clone(),
        },
    }
}

// Helper functions for audio analysis

fn calculate_spectral_centroid(audio_data: &[f32], sample_rate: u32) -> f32 {
    // Simplified spectral centroid calculation
    if audio_data.is_empty() {
        return 0.0;
    }

    // This would typically use FFT, but for simplicity we'll estimate
    let mut weighted_sum = 0.0;
    let mut magnitude_sum = 0.0;

    for (i, &sample) in audio_data.iter().enumerate() {
        let frequency = (i as f32 * sample_rate as f32) / (2.0 * audio_data.len() as f32);
        let magnitude = sample.abs();
        weighted_sum += frequency * magnitude;
        magnitude_sum += magnitude;
    }

    if magnitude_sum > 0.0 {
        weighted_sum / magnitude_sum
    } else {
        0.0
    }
}

fn calculate_spectral_rolloff(audio_data: &[f32], sample_rate: u32) -> f32 {
    // Simplified spectral rolloff calculation
    if audio_data.is_empty() {
        return 0.0;
    }

    let energy_threshold = 0.85; // 85% energy threshold
    let total_energy: f32 = audio_data.iter().map(|x| x * x).sum();
    let threshold_energy = total_energy * energy_threshold;

    let mut cumulative_energy = 0.0;
    for (i, &sample) in audio_data.iter().enumerate() {
        cumulative_energy += sample * sample;
        if cumulative_energy >= threshold_energy {
            return (i as f32 * sample_rate as f32) / (2.0 * audio_data.len() as f32);
        }
    }

    sample_rate as f32 / 2.0 // Nyquist frequency
}

fn calculate_spectral_flatness(audio_data: &[f32]) -> f64 {
    // Simplified spectral flatness calculation
    if audio_data.is_empty() {
        return 0.0;
    }

    let magnitudes: Vec<f32> = audio_data.iter().map(|x| x.abs() + 1e-10).collect();
    let geometric_mean = magnitudes.iter().map(|x| x.ln()).sum::<f32>() / magnitudes.len() as f32;
    let arithmetic_mean = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;

    (geometric_mean.exp() / arithmetic_mean) as f64
}

fn estimate_fundamental_frequency(audio_data: &[f32], sample_rate: u32) -> Option<f64> {
    // Simplified F0 estimation using autocorrelation
    if audio_data.len() < 100 {
        return None;
    }

    let min_f0 = 50.0; // Minimum F0 in Hz
    let max_f0 = 800.0; // Maximum F0 in Hz
    let min_period = (sample_rate as f64 / max_f0) as usize;
    let max_period = (sample_rate as f64 / min_f0) as usize;

    let mut max_correlation = 0.0;
    let mut best_period = 0;

    for period in min_period..max_period.min(audio_data.len() / 2) {
        let mut correlation = 0.0;
        let samples_to_check = audio_data.len() - period;

        for i in 0..samples_to_check {
            correlation += audio_data[i] * audio_data[i + period];
        }

        correlation /= samples_to_check as f32;

        if correlation > max_correlation {
            max_correlation = correlation;
            best_period = period;
        }
    }

    if max_correlation > 0.3 && best_period > 0 {
        Some(sample_rate as f64 / best_period as f64)
    } else {
        None
    }
}

fn estimate_snr(audio_data: &[f32]) -> f32 {
    // Simplified SNR estimation
    if audio_data.is_empty() {
        return 0.0;
    }

    let signal_power: f32 = audio_data.iter().map(|x| x * x).sum();
    let signal_rms = (signal_power / audio_data.len() as f32).sqrt();

    // Estimate noise as the minimum RMS in short windows
    let window_size = 256;
    let mut min_rms = f32::MAX;

    for window in audio_data.chunks(window_size) {
        let window_power: f32 = window.iter().map(|x| x * x).sum();
        let window_rms = (window_power / window.len() as f32).sqrt();
        min_rms = min_rms.min(window_rms);
    }

    if min_rms > 0.0 && signal_rms > min_rms {
        20.0 * (signal_rms / min_rms).log10()
    } else {
        0.0
    }
}

fn estimate_hnr(audio_data: &[f32]) -> f32 {
    // Simplified Harmonic-to-Noise Ratio estimation
    if audio_data.is_empty() {
        return 0.0;
    }

    // This is a very simplified estimation
    // In practice, this would require more sophisticated signal processing
    let total_energy: f32 = audio_data.iter().map(|x| x * x).sum();
    let mean_amplitude = audio_data.iter().map(|x| x.abs()).sum::<f32>() / audio_data.len() as f32;

    if mean_amplitude > 0.0 {
        20.0 * (total_energy.sqrt() / mean_amplitude).log10()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.audio_format.sample_rate, 16000);
        assert_eq!(config.chunk_size_ms, 100);
        assert_eq!(config.buffer_size, 10);
    }

    #[tokio::test]
    async fn test_websocket_session_manager_creation() {
        let config = WebSocketConfig::default();
        let manager = WebSocketSessionManager::new(config).await.unwrap();
        assert_eq!(manager.sessions.read().unwrap().len(), 0);
    }

    #[test]
    fn test_websocket_message_serialization() {
        let message = WebSocketMessage::Authentication {
            api_key: "test_key".to_string(),
            user_id: "test_user".to_string(),
            session_id: "test_session".to_string(),
        };

        let serialized = serde_json::to_string(&message).unwrap();
        assert!(serialized.contains("auth"));

        let deserialized: WebSocketMessage = serde_json::from_str(&serialized).unwrap();
        match deserialized {
            WebSocketMessage::Authentication { api_key, .. } => {
                assert_eq!(api_key, "test_key");
            }
            _ => panic!("Wrong message type"),
        }
    }
}
