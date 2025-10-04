//! WebSocket support for real-time speech recognition streaming.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Extension, Query,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::types::*;
use super::SharedPipeline;
use crate::integration::UnifiedVoirsPipeline;

/// WebSocket connection parameters
#[derive(Debug, Deserialize)]
/// Web Socket Params
pub struct WebSocketParams {
    /// Session ID (optional, will be generated if not provided)
    pub session_id: Option<String>,
    /// Model to use for recognition
    pub model: Option<String>,
    /// Language code
    pub language: Option<String>,
    /// Enable interim results
    pub interim_results: Option<bool>,
}

/// Streaming session state
#[derive(Debug)]
/// Streaming Session
pub struct StreamingSession {
    /// session id
    pub session_id: String,
    /// config
    pub config: StreamingRecognitionRequest,
    /// tx
    pub tx: mpsc::UnboundedSender<WebSocketMessage>,
    /// active
    pub active: bool,
    /// created at
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// last activity
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Global streaming session manager
pub type SessionManager = Arc<RwLock<HashMap<String, StreamingSession>>>;

/// Create WebSocket routes
pub fn create_websocket_routes() -> Router {
    Router::new()
        .route("/ws", get(websocket_handler))
        .route("/ws/sessions", get(list_active_sessions))
        .route("/ws/sessions/:session_id", get(get_session_info))
}

/// WebSocket upgrade handler
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    Query(params): Query<WebSocketParams>,
    Extension(pipeline): Extension<SharedPipeline>,
    Extension(session_manager): Extension<SessionManager>,
) -> impl IntoResponse {
    let session_id = params
        .session_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    info!("WebSocket connection request for session: {}", session_id);

    ws.on_upgrade(move |socket| {
        handle_websocket(socket, session_id, params, pipeline, session_manager)
    })
}

/// Handle WebSocket connection
async fn handle_websocket(
    socket: WebSocket,
    session_id: String,
    params: WebSocketParams,
    pipeline: SharedPipeline,
    session_manager: SessionManager,
) {
    info!(
        "WebSocket connection established for session: {}",
        session_id
    );

    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel();

    // Create initial session configuration
    let config = StreamingRecognitionRequest {
        config: Some(StreamingConfigRequest {
            chunk_duration: Some(1.0),
            overlap_duration: Some(0.1),
            vad_threshold: Some(0.5),
            silence_duration: Some(2.0),
            max_chunk_size: Some(16384),
            enable_interim_results: params.interim_results,
        }),
        recognition_config: Some(RecognitionConfigRequest {
            model: params.model,
            language: params.language,
            enable_vad: Some(true),
            confidence_threshold: Some(0.3),
            beam_size: Some(5),
            temperature: Some(0.0),
            suppress_blank: Some(true),
            suppress_tokens: None,
        }),
        audio_format: Some(AudioFormatRequest {
            sample_rate: Some(16000),
            channels: Some(1),
            bits_per_sample: Some(16),
            format: Some("wav".to_string()),
        }),
    };

    // Register session
    {
        let mut sessions = session_manager.write().await;
        sessions.insert(
            session_id.clone(),
            StreamingSession {
                session_id: session_id.clone(),
                config: config.clone(),
                tx: tx.clone(),
                active: true,
                created_at: chrono::Utc::now(),
                last_activity: chrono::Utc::now(),
            },
        );
    }

    // Send initial session status
    let _ = tx.send(WebSocketMessage::SessionStatus {
        session_id: session_id.clone(),
        status: "ready".to_string(),
        message: Some("Session initialized and ready for audio data".to_string()),
    });

    // Spawn task to handle outgoing messages
    let session_id_clone = session_id.clone();
    let outgoing_task = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            match serde_json::to_string(&message) {
                Ok(json) => {
                    if sender.send(Message::Text(json)).await.is_err() {
                        error!(
                            "Failed to send WebSocket message for session: {}",
                            session_id_clone
                        );
                        break;
                    }
                }
                Err(e) => {
                    error!("Failed to serialize WebSocket message: {}", e);
                }
            }
        }
    });

    // Handle incoming messages
    let session_id_clone = session_id.clone();
    let session_manager_clone = session_manager.clone();
    let tx_clone = tx.clone();

    let incoming_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = handle_text_message(
                        &text,
                        &session_id_clone,
                        &pipeline,
                        &session_manager_clone,
                        &tx_clone,
                    )
                    .await
                    {
                        error!("Error handling text message: {}", e);
                        let _ = tx_clone.send(WebSocketMessage::Error {
                            session_id: Some(session_id_clone.clone()),
                            error: ErrorResponse {
                                code: "MESSAGE_PROCESSING_ERROR".to_string(),
                                message: format!("Failed to process message: {}", e),
                                details: None,
                                request_id: Uuid::new_v4().to_string(),
                                timestamp: chrono::Utc::now(),
                            },
                        });
                    }
                }
                Ok(Message::Binary(data)) => {
                    if let Err(e) = handle_binary_message(
                        &data,
                        &session_id_clone,
                        &pipeline,
                        &session_manager_clone,
                        &tx_clone,
                    )
                    .await
                    {
                        error!("Error handling binary message: {}", e);
                    }
                }
                Ok(Message::Close(_)) => {
                    info!(
                        "WebSocket connection closed for session: {}",
                        session_id_clone
                    );
                    break;
                }
                Ok(Message::Ping(data)) => {
                    debug!("Received ping for session: {}", session_id_clone);
                    // WebSocket implementation typically handles pong automatically
                }
                Ok(Message::Pong(_)) => {
                    debug!("Received pong for session: {}", session_id_clone);
                }
                Err(e) => {
                    error!("WebSocket error for session {}: {}", session_id_clone, e);
                    break;
                }
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = incoming_task => {
            info!("Incoming message task completed for session: {}", session_id);
        }
        _ = outgoing_task => {
            info!("Outgoing message task completed for session: {}", session_id);
        }
    }

    // Clean up session
    {
        let mut sessions = session_manager.write().await;
        sessions.remove(&session_id);
    }

    info!(
        "WebSocket connection cleanup completed for session: {}",
        session_id
    );
}

/// Handle text message (typically JSON commands)
async fn handle_text_message(
    text: &str,
    session_id: &str,
    pipeline: &SharedPipeline,
    session_manager: &SessionManager,
    tx: &mpsc::UnboundedSender<WebSocketMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let message: WebSocketMessage = serde_json::from_str(text)?;

    match message {
        WebSocketMessage::StartStreaming {
            session_id: msg_session_id,
            config,
        } => {
            if msg_session_id != session_id {
                return Err("Session ID mismatch".into());
            }

            info!("Starting streaming for session: {}", session_id);

            // Update session configuration
            {
                let mut sessions = session_manager.write().await;
                if let Some(session) = sessions.get_mut(session_id) {
                    session.config = config;
                    session.last_activity = chrono::Utc::now();
                }
            }

            tx.send(WebSocketMessage::SessionStatus {
                session_id: session_id.to_string(),
                status: "streaming".to_string(),
                message: Some("Ready to receive audio data".to_string()),
            })?;
        }

        WebSocketMessage::AudioChunk {
            session_id: msg_session_id,
            chunk_data,
            sequence_number,
        } => {
            if msg_session_id != session_id {
                return Err("Session ID mismatch".into());
            }

            debug!(
                "Received audio chunk {} for session: {}",
                sequence_number, session_id
            );

            // Decode base64 audio data
            use base64::{engine::general_purpose, Engine as _};
            let audio_data = general_purpose::STANDARD.decode(&chunk_data)?;

            // Process audio chunk
            let result = process_audio_chunk(
                &audio_data,
                session_id,
                sequence_number,
                pipeline,
                session_manager,
            )
            .await?;

            tx.send(WebSocketMessage::RecognitionResult {
                session_id: session_id.to_string(),
                result,
            })?;
        }

        WebSocketMessage::StopStreaming {
            session_id: msg_session_id,
        } => {
            if msg_session_id != session_id {
                return Err("Session ID mismatch".into());
            }

            info!("Stopping streaming for session: {}", session_id);

            // Mark session as inactive
            {
                let mut sessions = session_manager.write().await;
                if let Some(session) = sessions.get_mut(session_id) {
                    session.active = false;
                    session.last_activity = chrono::Utc::now();
                }
            }

            tx.send(WebSocketMessage::SessionStatus {
                session_id: session_id.to_string(),
                status: "stopped".to_string(),
                message: Some("Streaming stopped".to_string()),
            })?;
        }

        _ => {
            warn!(
                "Unhandled WebSocket message type for session: {}",
                session_id
            );
        }
    }

    Ok(())
}

/// Handle binary message (typically raw audio data)
async fn handle_binary_message(
    data: &[u8],
    session_id: &str,
    pipeline: &SharedPipeline,
    session_manager: &SessionManager,
    tx: &mpsc::UnboundedSender<WebSocketMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    debug!(
        "Received binary audio data ({} bytes) for session: {}",
        data.len(),
        session_id
    );

    // Process binary audio data
    let result = process_audio_chunk(
        data,
        session_id,
        0, // No sequence number for binary data
        pipeline,
        session_manager,
    )
    .await?;

    tx.send(WebSocketMessage::RecognitionResult {
        session_id: session_id.to_string(),
        result,
    })?;

    Ok(())
}

/// Process audio chunk and return recognition result
async fn process_audio_chunk(
    audio_data: &[u8],
    session_id: &str,
    sequence_number: u64,
    pipeline: &SharedPipeline,
    session_manager: &SessionManager,
) -> Result<StreamingRecognitionResponse, Box<dyn std::error::Error + Send + Sync>> {
    let start_time = std::time::Instant::now();

    // Update session activity
    {
        let mut sessions = session_manager.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.last_activity = chrono::Utc::now();
        }
    }

    // Get session configuration
    let session_config = {
        let sessions = session_manager.read().await;
        sessions.get(session_id).map(|s| s.config.clone())
    };

    match session_config {
        Some(config) => {
            // Try to process audio with the pipeline
            match process_with_pipeline(audio_data, &config, pipeline).await {
                Ok((text, confidence, is_final)) => {
                    let processing_time = start_time.elapsed().as_millis() as f64;

                    Ok(StreamingRecognitionResponse {
                        session_id: session_id.to_string(),
                        is_interim: !is_final,
                        is_final,
                        text: text.clone(),
                        confidence,
                        segment: if is_final {
                            Some(SegmentResponse {
                                start_time: 0.0,
                                end_time: 1.0, // Approximate chunk duration
                                text: text.clone(),
                                confidence,
                                no_speech_prob: 1.0 - confidence,
                                tokens: None,
                            })
                        } else {
                            None
                        },
                        processing_time_ms: processing_time,
                        sequence_number,
                    })
                }
                Err(e) => {
                    warn!(
                        "Pipeline processing failed for session {}: {}",
                        session_id, e
                    );
                    // Fall back to mock response with error indication
                    let processing_time = start_time.elapsed().as_millis() as f64;

                    Ok(StreamingRecognitionResponse {
                        session_id: session_id.to_string(),
                        is_interim: true,
                        is_final: false,
                        text: "[Processing temporarily unavailable]".to_string(),
                        confidence: 0.1,
                        segment: None,
                        processing_time_ms: processing_time,
                        sequence_number,
                    })
                }
            }
        }
        None => {
            // No session config found, return error response
            let processing_time = start_time.elapsed().as_millis() as f64;

            Ok(StreamingRecognitionResponse {
                session_id: session_id.to_string(),
                is_interim: true,
                is_final: false,
                text: "[Session not found]".to_string(),
                confidence: 0.0,
                segment: None,
                processing_time_ms: processing_time,
                sequence_number,
            })
        }
    }
}

/// Process audio data using the VoiRS pipeline
async fn process_with_pipeline(
    audio_data: &[u8],
    config: &StreamingRecognitionRequest,
    pipeline: &SharedPipeline,
) -> Result<(String, f32, bool), Box<dyn std::error::Error + Send + Sync>> {
    // Convert audio bytes to AudioBuffer
    let audio_buffer = convert_audio_bytes_to_buffer(audio_data, config)?;

    // Try to get pipeline access
    match pipeline.try_read() {
        Ok(pipeline_guard) => {
            // Use the pipeline to process audio
            match pipeline_guard.process(&audio_buffer).await {
                Ok(result) => {
                    // Extract text and confidence from the result
                    let text = result.transcript.text;
                    let confidence = result.transcript.confidence;

                    // Determine if this is a final result based on confidence and silence detection
                    let is_final = confidence > 0.7
                        || text.ends_with('.')
                        || text.ends_with('?')
                        || text.ends_with('!');

                    Ok((text, confidence, is_final))
                }
                Err(e) => Err(format!("Pipeline processing error: {}", e).into()),
            }
        }
        Err(_) => {
            // Pipeline is locked, return interim result
            Ok(("[Processing...]".to_string(), 0.5, false))
        }
    }
}

/// Convert raw audio bytes to AudioBuffer
fn convert_audio_bytes_to_buffer(
    audio_data: &[u8],
    config: &StreamingRecognitionRequest,
) -> Result<crate::AudioBuffer, Box<dyn std::error::Error + Send + Sync>> {
    // Get audio format from config or use defaults
    let sample_rate = config
        .audio_format
        .as_ref()
        .and_then(|f| f.sample_rate)
        .unwrap_or(16000) as u32;

    let channels = config
        .audio_format
        .as_ref()
        .and_then(|f| f.channels)
        .unwrap_or(1) as u16;

    let bits_per_sample = config
        .audio_format
        .as_ref()
        .and_then(|f| f.bits_per_sample)
        .unwrap_or(16) as u16;

    // Convert bytes to f32 samples based on bit depth
    let samples = match bits_per_sample {
        16 => {
            // Convert 16-bit PCM to f32
            let sample_count = audio_data.len() / 2;
            let mut samples = Vec::with_capacity(sample_count);

            for i in 0..sample_count {
                let sample_bytes = [audio_data[i * 2], audio_data[i * 2 + 1]];
                let sample_i16 = i16::from_le_bytes(sample_bytes);
                let sample_f32 = sample_i16 as f32 / i16::MAX as f32;
                samples.push(sample_f32);
            }
            samples
        }
        32 => {
            // Convert 32-bit float to f32
            let sample_count = audio_data.len() / 4;
            let mut samples = Vec::with_capacity(sample_count);

            for i in 0..sample_count {
                let sample_bytes = [
                    audio_data[i * 4],
                    audio_data[i * 4 + 1],
                    audio_data[i * 4 + 2],
                    audio_data[i * 4 + 3],
                ];
                let sample_f32 = f32::from_le_bytes(sample_bytes);
                samples.push(sample_f32);
            }
            samples
        }
        _ => {
            return Err(format!("Unsupported bit depth: {}", bits_per_sample).into());
        }
    };

    // Create AudioBuffer (assuming mono for simplicity, could be extended for stereo)
    if channels == 1 {
        Ok(crate::AudioBuffer::mono(samples, sample_rate))
    } else {
        // For multi-channel, we'll take the first channel for now
        // In a full implementation, you'd want to handle stereo properly
        let mono_samples: Vec<f32> = samples.iter().step_by(channels as usize).cloned().collect();
        Ok(crate::AudioBuffer::mono(mono_samples, sample_rate))
    }
}

/// List active streaming sessions
async fn list_active_sessions(
    Extension(session_manager): Extension<SessionManager>,
) -> axum::response::Json<ApiResponse<Vec<HashMap<String, serde_json::Value>>>> {
    let sessions = session_manager.read().await;

    let active_sessions: Vec<_> = sessions
        .values()
        .filter(|session| session.active)
        .map(|session| {
            let mut info = HashMap::new();
            info.insert(
                "session_id".to_string(),
                serde_json::Value::String(session.session_id.clone()),
            );
            info.insert(
                "active".to_string(),
                serde_json::Value::Bool(session.active),
            );
            info.insert(
                "created_at".to_string(),
                serde_json::Value::String(session.created_at.to_rfc3339()),
            );
            info.insert(
                "last_activity".to_string(),
                serde_json::Value::String(session.last_activity.to_rfc3339()),
            );
            info
        })
        .collect();

    axum::response::Json(ApiResponse::success(active_sessions))
}

/// Get specific session information
async fn get_session_info(
    axum::extract::Path(session_id): axum::extract::Path<String>,
    Extension(session_manager): Extension<SessionManager>,
) -> Result<
    axum::response::Json<ApiResponse<HashMap<String, serde_json::Value>>>,
    axum::http::StatusCode,
> {
    let sessions = session_manager.read().await;

    if let Some(session) = sessions.get(&session_id) {
        let mut info = HashMap::new();
        info.insert(
            "session_id".to_string(),
            serde_json::Value::String(session.session_id.clone()),
        );
        info.insert(
            "active".to_string(),
            serde_json::Value::Bool(session.active),
        );
        info.insert(
            "created_at".to_string(),
            serde_json::Value::String(session.created_at.to_rfc3339()),
        );
        info.insert(
            "last_activity".to_string(),
            serde_json::Value::String(session.last_activity.to_rfc3339()),
        );
        info.insert(
            "config".to_string(),
            serde_json::to_value(&session.config).unwrap_or(serde_json::Value::Null),
        );

        Ok(axum::response::Json(ApiResponse::success(info)))
    } else {
        Err(axum::http::StatusCode::NOT_FOUND)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_websocket_routes() {
        let session_manager: SessionManager = Arc::new(RwLock::new(HashMap::new()));

        let app = create_websocket_routes().layer(Extension(session_manager));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ws/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_session_manager() {
        let session_manager: SessionManager = Arc::new(RwLock::new(HashMap::new()));
        let (tx, _rx) = mpsc::unbounded_channel();

        // Add a test session
        {
            let mut sessions = session_manager.write().await;
            sessions.insert(
                "test-session".to_string(),
                StreamingSession {
                    session_id: "test-session".to_string(),
                    config: StreamingRecognitionRequest {
                        config: None,
                        recognition_config: None,
                        audio_format: None,
                    },
                    tx,
                    active: true,
                    created_at: chrono::Utc::now(),
                    last_activity: chrono::Utc::now(),
                },
            );
        }

        // Verify session exists
        {
            let sessions = session_manager.read().await;
            assert!(sessions.contains_key("test-session"));
            assert!(sessions.get("test-session").unwrap().active);
        }
    }
}
