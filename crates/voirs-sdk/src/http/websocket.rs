use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Extension, Query,
    },
    response::Response,
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Instant};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

use super::SharedPipeline;
use crate::{
    config::PipelineConfig,
    streaming::{StreamingConfig, StreamingState},
    types::{LanguageCode, QualityLevel, VoiceConfig},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    pub session_id: String,
    pub voice_id: Option<String>,
    pub language: Option<LanguageCode>,
    pub quality: Option<QualityLevel>,
    pub streaming_config: StreamingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    Connect {
        config: WebSocketConfig,
    },
    StreamText {
        text: String,
        sequence_id: Option<u64>,
    },
    Configure {
        config: Box<PipelineConfig>,
    },
    SwitchVoice {
        voice_id: String,
    },
    GetVoices,
    GetStatus,
    Ping,
    Disconnect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    Connected {
        session_id: String,
        capabilities: HashMap<String, serde_json::Value>,
    },
    AudioChunk {
        data: Vec<u8>,
        sequence_id: Option<u64>,
        metadata: AudioChunkMetadata,
    },
    TextProcessed {
        text: String,
        sequence_id: Option<u64>,
        processing_time: f64,
    },
    StreamingComplete {
        total_chunks: usize,
        total_duration: f64,
        quality_metrics: QualityMetrics,
    },
    VoicesList {
        voices: Vec<VoiceConfig>,
    },
    VoiceSwitched {
        voice_id: String,
    },
    Status {
        streaming_active: bool,
        current_voice: Option<String>,
        queue_size: usize,
        performance: PerformanceMetrics,
    },
    Error {
        message: String,
        code: u16,
    },
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunkMetadata {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration: f64,
    pub format: String,
    pub chunk_index: usize,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub latency_ms: f64,
    pub throughput_mbps: f64,
    pub quality_score: f32,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f64,
    pub processing_latency: f64,
    pub queue_latency: f64,
}

#[derive(Debug, Deserialize)]
pub struct WebSocketQuery {
    pub session_id: Option<String>,
    pub voice_id: Option<String>,
    pub language: Option<LanguageCode>,
}

pub struct WebSocketSession {
    pub id: String,
    pub config: WebSocketConfig,
    pub sender: mpsc::UnboundedSender<ServerMessage>,
    pub pipeline: SharedPipeline,
    pub streaming_state: Arc<RwLock<StreamingState>>,
    pub start_time: Instant,
}

pub type SessionMap = Arc<RwLock<HashMap<String, WebSocketSession>>>;

pub fn create_websocket_routes() -> Router {
    Router::new().route("/ws", get(websocket_handler))
}

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    Extension(pipeline): Extension<SharedPipeline>,
    Query(query): Query<WebSocketQuery>,
) -> Response {
    let session_id = query
        .session_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    ws.on_upgrade(move |socket| handle_socket(socket, pipeline, session_id, query))
}

async fn handle_socket(
    socket: WebSocket,
    pipeline: SharedPipeline,
    session_id: String,
    query: WebSocketQuery,
) {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<ServerMessage>();

    let session = WebSocketSession {
        id: session_id.clone(),
        config: WebSocketConfig {
            session_id: session_id.clone(),
            voice_id: query.voice_id,
            language: query.language,
            quality: None,
            streaming_config: StreamingConfig::default(),
        },
        sender: tx,
        pipeline: pipeline.clone(),
        streaming_state: Arc::new(RwLock::new(StreamingState::default())),
        start_time: Instant::now(),
    };

    // Send initial connection message
    let capabilities = get_capabilities();
    let _ = session.sender.send(ServerMessage::Connected {
        session_id: session_id.clone(),
        capabilities,
    });

    // Spawn task to handle outgoing messages
    let mut send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match serde_json::to_string(&msg) {
                Ok(json) => {
                    if sender.send(Message::Text(json)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to serialize message: {}", e);
                    break;
                }
            }
        }
    });

    // Spawn task to handle incoming messages
    let session_arc = Arc::new(RwLock::new(session));
    let session_clone = session_arc.clone();

    let mut recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = handle_client_message(text, session_clone.clone()).await {
                        tracing::error!("Error handling message: {}", e);
                        break;
                    }
                }
                Ok(Message::Binary(data)) => {
                    tracing::debug!("Received binary data: {} bytes", data.len());
                }
                Ok(Message::Close(_)) => {
                    tracing::info!("WebSocket connection closed");
                    break;
                }
                Err(e) => {
                    tracing::error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = (&mut send_task) => {
            recv_task.abort();
        }
        _ = (&mut recv_task) => {
            send_task.abort();
        }
    }

    tracing::info!("WebSocket connection ended for session: {}", session_id);
}

async fn handle_client_message(
    text: String,
    session: Arc<RwLock<WebSocketSession>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let message: ClientMessage = serde_json::from_str(&text)?;

    match message {
        ClientMessage::Connect { config } => {
            let mut session_guard = session.write().await;
            session_guard.config = config;

            // Initialize streaming if configured
            if let Some(voice_id) = &session_guard.config.voice_id {
                let pipeline = session_guard.pipeline.read().await;
                if let Err(e) = pipeline.set_voice(voice_id).await {
                    let _ = session_guard.sender.send(ServerMessage::Error {
                        message: format!("Failed to switch voice: {e}"),
                        code: 400,
                    });
                }
            }
        }

        ClientMessage::StreamText { text, sequence_id } => {
            let session_guard = session.read().await;
            let pipeline = session_guard.pipeline.read().await;

            let start_time = Instant::now();

            match pipeline.synthesize(&text).await {
                Ok(audio_buffer) => {
                    let processing_time = start_time.elapsed().as_secs_f64();

                    // Send text processed confirmation
                    let _ = session_guard.sender.send(ServerMessage::TextProcessed {
                        text: text.clone(),
                        sequence_id,
                        processing_time,
                    });

                    // Send audio as single chunk
                    let chunk = &audio_buffer;
                    let index = 0;
                    let metadata = AudioChunkMetadata {
                        sample_rate: chunk.sample_rate(),
                        channels: chunk.channels() as u16,
                        duration: chunk.duration() as f64,
                        format: "wav".to_string(),
                        chunk_index: index,
                        timestamp: start_time.elapsed().as_secs_f64(),
                    };

                    let audio_data = chunk.to_wav_bytes().unwrap_or_else(|_| vec![]);
                    let _ = session_guard.sender.send(ServerMessage::AudioChunk {
                        data: audio_data,
                        sequence_id,
                        metadata,
                    });

                    // Send completion message
                    let quality_metrics = QualityMetrics {
                        latency_ms: processing_time * 1000.0,
                        throughput_mbps: 0.0, // Calculate actual throughput
                        quality_score: 0.95,
                        compression_ratio: 0.8,
                    };

                    let _ = session_guard.sender.send(ServerMessage::StreamingComplete {
                        total_chunks: 1,
                        total_duration: audio_buffer.duration() as f64,
                        quality_metrics,
                    });
                }
                Err(e) => {
                    let _ = session_guard.sender.send(ServerMessage::Error {
                        message: format!("Streaming failed: {e}"),
                        code: 500,
                    });
                }
            }
        }

        ClientMessage::Configure { config } => {
            let session_guard = session.read().await;
            let pipeline = session_guard.pipeline.write().await;

            match pipeline.update_config(*config).await {
                Ok(()) => {
                    // Configuration updated successfully
                }
                Err(e) => {
                    let _ = session_guard.sender.send(ServerMessage::Error {
                        message: format!("Configuration update failed: {e}"),
                        code: 400,
                    });
                }
            }
        }

        ClientMessage::SwitchVoice { voice_id } => {
            let session_guard = session.read().await;
            let pipeline = session_guard.pipeline.write().await;

            match pipeline.set_voice(&voice_id).await {
                Ok(()) => {
                    let _ = session_guard
                        .sender
                        .send(ServerMessage::VoiceSwitched { voice_id });
                }
                Err(e) => {
                    let _ = session_guard.sender.send(ServerMessage::Error {
                        message: format!("Voice switch failed: {e}"),
                        code: 400,
                    });
                }
            }
        }

        ClientMessage::GetVoices => {
            let session_guard = session.read().await;
            let pipeline = session_guard.pipeline.read().await;

            match pipeline.list_voices().await {
                Ok(voices) => {
                    let _ = session_guard
                        .sender
                        .send(ServerMessage::VoicesList { voices });
                }
                Err(e) => {
                    let _ = session_guard.sender.send(ServerMessage::Error {
                        message: format!("Failed to get voices: {e}"),
                        code: 500,
                    });
                }
            }
        }

        ClientMessage::GetStatus => {
            let session_guard = session.read().await;
            let streaming_state = session_guard.streaming_state.read().await;

            let performance = PerformanceMetrics {
                cpu_usage: 0.0,    // Would need actual CPU monitoring
                memory_usage: 0.0, // Would need actual memory monitoring
                processing_latency: 0.0,
                queue_latency: 0.0,
            };

            let _ = session_guard.sender.send(ServerMessage::Status {
                streaming_active: streaming_state.is_realtime(),
                current_voice: None, // Would get from pipeline
                queue_size: 0,
                performance,
            });
        }

        ClientMessage::Ping => {
            let session_guard = session.read().await;
            let _ = session_guard.sender.send(ServerMessage::Pong);
        }

        ClientMessage::Disconnect => {
            // Client requesting disconnection
            tracing::info!("Client requested disconnection");
        }
    }

    Ok(())
}

fn get_capabilities() -> HashMap<String, serde_json::Value> {
    let mut capabilities = HashMap::new();

    capabilities.insert("streaming".to_string(), serde_json::json!(true));
    capabilities.insert("voice_switching".to_string(), serde_json::json!(true));
    capabilities.insert("real_time".to_string(), serde_json::json!(true));
    capabilities.insert(
        "audio_formats".to_string(),
        serde_json::json!(["wav", "raw"]),
    );
    capabilities.insert("max_text_length".to_string(), serde_json::json!(10000));
    capabilities.insert("max_chunk_size".to_string(), serde_json::json!(1024));
    capabilities.insert(
        "languages".to_string(),
        serde_json::json!(["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]),
    );

    capabilities
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VoirsPipelineBuilder;
    use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

    #[tokio::test]
    async fn test_websocket_capabilities() {
        let capabilities = get_capabilities();
        assert!(capabilities.contains_key("streaming"));
        assert!(capabilities.contains_key("voice_switching"));
        assert!(capabilities.contains_key("real_time"));
    }

    #[tokio::test]
    async fn test_client_message_serialization() {
        let msg = ClientMessage::Connect {
            config: WebSocketConfig {
                session_id: "test-session".to_string(),
                voice_id: Some("test-voice".to_string()),
                language: Some(LanguageCode::English),
                quality: Some(Quality::High),
                streaming_config: StreamingConfig::default(),
            },
        };

        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ClientMessage = serde_json::from_str(&json).unwrap();

        match parsed {
            ClientMessage::Connect { config } => {
                assert_eq!(config.session_id, "test-session");
                assert_eq!(config.voice_id, Some("test-voice".to_string()));
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[tokio::test]
    async fn test_server_message_serialization() {
        let msg = ServerMessage::Connected {
            session_id: "test-session".to_string(),
            capabilities: get_capabilities(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();

        match parsed {
            ServerMessage::Connected {
                session_id,
                capabilities,
            } => {
                assert_eq!(session_id, "test-session");
                assert!(!capabilities.is_empty());
            }
            _ => panic!("Wrong message type"),
        }
    }
}
