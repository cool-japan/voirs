//! WebSocket Server Example for Real-time Audio Evaluation
//!
//! This example demonstrates how to use the VoiRS WebSocket server for real-time
//! speech synthesis quality evaluation with streaming audio processing.

use std::collections::HashMap;
use tokio;
use voirs_evaluation::websocket::{
    AudioFormat, ProcessingOptions, QualityThresholds, SessionConfig, WebSocketConfig,
    WebSocketMessage, WebSocketSessionManager,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔌 VoiRS WebSocket Real-time Evaluation Server Example");
    println!("=====================================================");

    // Create WebSocket service configuration
    let mut ws_config = WebSocketConfig::default();
    ws_config.max_sessions = 100;
    ws_config.session_timeout = 600; // 10 minutes
    ws_config.max_chunk_size = 16384; // 16KB
    ws_config.enable_prediction = true;
    ws_config.heartbeat_interval = 30; // 30 seconds

    println!("📋 WebSocket Configuration:");
    println!("  Max Sessions: {}", ws_config.max_sessions);
    println!("  Session Timeout: {}s", ws_config.session_timeout);
    println!("  Max Chunk Size: {} bytes", ws_config.max_chunk_size);
    println!("  Heartbeat Interval: {}s", ws_config.heartbeat_interval);

    // Create session manager
    println!("\n🔧 Initializing WebSocket session manager...");
    let session_manager = WebSocketSessionManager::new(ws_config).await?;

    // Demonstrate session configuration
    let session_config = SessionConfig {
        audio_format: AudioFormat {
            sample_rate: 16000,
            channels: 1,
            bit_depth: 16,
            encoding: "pcm".to_string(),
        },
        metrics: vec![
            "pesq".to_string(),
            "stoi".to_string(),
            "overall".to_string(),
            "naturalness".to_string(),
        ],
        chunk_size_ms: 100, // 100ms chunks
        buffer_size: 20,    // 2 second buffer
        language: Some("en-US".to_string()),
        quality_thresholds: QualityThresholds {
            min_quality: 0.6,
            warning_threshold: 0.7,
            alert_threshold: 0.4,
            confidence_threshold: 0.8,
        },
        processing_options: ProcessingOptions {
            adaptive_quality: true,
            noise_reduction: true,
            auto_gain_control: true,
            buffer_overlap: 25.0, // 25% overlap
            quality_prediction: true,
        },
    };

    println!("\n⚙️  Session Configuration:");
    println!(
        "  Audio Format: {}Hz, {} channels, {}-bit {}",
        session_config.audio_format.sample_rate,
        session_config.audio_format.channels,
        session_config.audio_format.bit_depth,
        session_config.audio_format.encoding
    );
    println!("  Chunk Size: {}ms", session_config.chunk_size_ms);
    println!("  Buffer Size: {} chunks", session_config.buffer_size);
    println!(
        "  Language: {}",
        session_config
            .language
            .as_ref()
            .unwrap_or(&"None".to_string())
    );
    println!(
        "  Quality Thresholds: min={}, warn={}, alert={}",
        session_config.quality_thresholds.min_quality,
        session_config.quality_thresholds.warning_threshold,
        session_config.quality_thresholds.alert_threshold
    );

    // Test session management
    println!("\n🧪 Testing session management...");

    // Start a test session
    let session_id = "test_session_001";
    let user_id = "demo_user";

    match session_manager.start_session(
        session_id.to_string(),
        user_id.to_string(),
        session_config.clone(),
    ) {
        Ok(()) => {
            println!("✅ Test session started successfully");
            println!("   Session ID: {}", session_id);
            println!("   User ID: {}", user_id);
        }
        Err(e) => {
            println!("❌ Failed to start test session: {}", e);
            return Err(e.into());
        }
    }

    // Test audio chunk processing
    println!("\n🎵 Testing audio chunk processing...");

    // Generate test audio data (sine wave)
    let sample_rate = session_config.audio_format.sample_rate;
    let chunk_duration = session_config.chunk_size_ms as f32 / 1000.0;
    let samples_per_chunk = (sample_rate as f32 * chunk_duration) as usize;

    let frequency = 440.0; // A4 note
    let mut test_chunks = Vec::new();

    // Generate 5 chunks of test audio
    for chunk_num in 0..5 {
        let mut chunk_samples = Vec::with_capacity(samples_per_chunk);

        for i in 0..samples_per_chunk {
            let t = (chunk_num * samples_per_chunk + i) as f32 / sample_rate as f32;
            let amplitude = 0.3 * (1.0 - chunk_num as f32 * 0.1); // Gradually decrease amplitude
            let sample = amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();
            chunk_samples.push(sample);
        }

        test_chunks.push(chunk_samples);
    }

    // Process each audio chunk
    for (chunk_id, chunk_data) in test_chunks.iter().enumerate() {
        let timestamp = chunk_id as f64 * chunk_duration as f64;

        match session_manager
            .process_audio_chunk(session_id, chunk_id as u64, chunk_data.clone(), timestamp)
            .await
        {
            Ok(WebSocketMessage::EvaluationResult {
                quality_score,
                confidence,
                analysis,
                ..
            }) => {
                println!("✅ Chunk {} processed:", chunk_id);
                println!("   Quality Score: {:.3}", quality_score);
                println!("   Confidence: {:.3}", confidence);
                println!("   RMS Level: {:.1} dB", analysis.signal_level.rms_db);
                println!("   Peak Level: {:.1} dB", analysis.signal_level.peak_db);

                if !analysis.detected_issues.is_empty() {
                    println!("   Issues Detected:");
                    for issue in &analysis.detected_issues {
                        println!(
                            "     - {} (severity {}): {}",
                            issue.issue_type, issue.severity, issue.description
                        );
                    }
                }

                if !analysis.recommendations.is_empty() {
                    println!("   Recommendations:");
                    for rec in &analysis.recommendations {
                        println!("     - {}", rec);
                    }
                }
            }
            Ok(msg) => {
                println!("❓ Unexpected message type: {:?}", msg);
            }
            Err(e) => {
                println!("❌ Failed to process chunk {}: {}", chunk_id, e);
            }
        }
    }

    // Test session status
    println!("\n📊 Testing session status...");
    match session_manager.get_session_status(session_id) {
        Ok(WebSocketMessage::SessionStatus {
            status, statistics, ..
        }) => {
            println!("✅ Session status retrieved:");
            println!("   Status: {}", status);
            println!("   Total Chunks: {}", statistics.total_chunks);
            println!("   Duration: {:.1}s", statistics.duration_seconds);
            println!("   Average Quality: {:.3}", statistics.average_quality);
            println!("   Min Quality: {:.3}", statistics.min_quality);
            println!("   Max Quality: {:.3}", statistics.max_quality);
            println!(
                "   Avg Processing Time: {:.1}ms",
                statistics.avg_processing_time_ms
            );
            println!("   Quality Alerts: {}", statistics.quality_alerts);
        }
        Ok(msg) => {
            println!("❓ Unexpected message type: {:?}", msg);
        }
        Err(e) => {
            println!("❌ Failed to get session status: {}", e);
        }
    }

    // Test WebSocket message serialization
    println!("\n🔄 Testing WebSocket message serialization...");

    let auth_message = WebSocketMessage::Authentication {
        api_key: "voirs_api_key_demo_123456789012345678901234".to_string(),
        user_id: "demo_user".to_string(),
        session_id: session_id.to_string(),
    };

    match serde_json::to_string(&auth_message) {
        Ok(json) => {
            println!("✅ Authentication message serialized:");
            println!("   JSON: {}", json);

            match serde_json::from_str::<WebSocketMessage>(&json) {
                Ok(_) => println!("✅ Message deserialization successful"),
                Err(e) => println!("❌ Message deserialization failed: {}", e),
            }
        }
        Err(e) => {
            println!("❌ Message serialization failed: {}", e);
        }
    }

    // Clean up test session
    println!("\n🧹 Cleaning up test session...");
    match session_manager.end_session(session_id) {
        Ok(final_stats) => {
            println!("✅ Session ended successfully");
            println!("   Final Statistics:");
            println!("     Total Chunks: {}", final_stats.total_chunks);
            println!("     Duration: {:.1}s", final_stats.duration_seconds);
            println!("     Average Quality: {:.3}", final_stats.average_quality);
            println!(
                "     Total Processing Time: {}ms",
                final_stats.total_processing_time_ms
            );
        }
        Err(e) => {
            println!("❌ Failed to end session: {}", e);
        }
    }

    println!("\n🌐 Starting WebSocket server...");
    println!("   WebSocket URL: ws://127.0.0.1:8081/ws/evaluate");
    println!("   Health Check: http://127.0.0.1:8081/ws/health");
    println!("   Status: http://127.0.0.1:8081/ws/status");
    println!("💡 Press Ctrl+C to stop the server\n");

    // Show example client usage
    println!("📝 Example WebSocket Client Usage:");
    println!("   1. Connect to: ws://127.0.0.1:8081/ws/evaluate");
    println!("   2. Authenticate:");
    println!(
        "      {}",
        serde_json::to_string(&WebSocketMessage::Authentication {
            api_key: "your_api_key_here".to_string(),
            user_id: "your_user_id".to_string(),
            session_id: "unique_session_id".to_string(),
        })
        .unwrap()
    );
    println!("   3. Start session:");
    println!(
        "      {}",
        serde_json::to_string(&WebSocketMessage::StartSession {
            config: session_config.clone(),
        })
        .unwrap()
    );
    println!("   4. Send audio chunks:");
    println!(
        "      {}",
        serde_json::to_string(&WebSocketMessage::AudioChunk {
            session_id: "unique_session_id".to_string(),
            chunk_id: 1,
            audio_data: "base64_encoded_audio_data".to_string(),
            timestamp: 0.1,
            is_final: false,
        })
        .unwrap()
    );

    // Start the WebSocket server (this will block)
    session_manager.start_server("127.0.0.1", 8081).await?;

    Ok(())
}

/// Helper function to demonstrate client-side WebSocket connection
#[cfg(feature = "client_demo")]
async fn demonstrate_websocket_client() -> Result<(), Box<dyn std::error::Error>> {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

    println!("🔌 Connecting to WebSocket server...");

    let url = "ws://127.0.0.1:8081/ws/evaluate";
    let (ws_stream, _) = connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();

    // Authenticate
    let auth_msg = WebSocketMessage::Authentication {
        api_key: "voirs_demo_api_key_123456789012345678901234".to_string(),
        user_id: "demo_client_user".to_string(),
        session_id: "client_demo_session".to_string(),
    };

    let auth_json = serde_json::to_string(&auth_msg)?;
    write.send(Message::Text(auth_json)).await?;

    // Wait for authentication response
    if let Some(msg) = read.next().await {
        match msg? {
            Message::Text(text) => {
                println!("📨 Server response: {}", text);
            }
            _ => {}
        }
    }

    // Start evaluation session
    let session_config = SessionConfig::default();
    let start_msg = WebSocketMessage::StartSession {
        config: session_config,
    };

    let start_json = serde_json::to_string(&start_msg)?;
    write.send(Message::Text(start_json)).await?;

    // Generate and send test audio chunk
    let test_audio = vec![0.1, 0.2, 0.3, 0.4, 0.5]; // Simple test data
    let audio_bytes: Vec<u8> = test_audio
        .iter()
        .flat_map(|&sample| {
            let sample_i16 = (sample * 32767.0) as i16;
            sample_i16.to_le_bytes()
        })
        .collect();

    let audio_base64 =
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &audio_bytes);

    let chunk_msg = WebSocketMessage::AudioChunk {
        session_id: "client_demo_session".to_string(),
        chunk_id: 1,
        audio_data: audio_base64,
        timestamp: 0.1,
        is_final: false,
    };

    let chunk_json = serde_json::to_string(&chunk_msg)?;
    write.send(Message::Text(chunk_json)).await?;

    // Wait for evaluation result
    if let Some(msg) = read.next().await {
        match msg? {
            Message::Text(text) => {
                println!("📊 Evaluation result: {}", text);
            }
            _ => {}
        }
    }

    println!("✅ WebSocket client demo completed");
    Ok(())
}
