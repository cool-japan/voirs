//! # Web Server Integration Example
//!
//! This example demonstrates how to integrate VoiRS SDK into a web server
//! to provide text-to-speech API endpoints. This is a common use case for
//! building speech synthesis services.

use voirs_sdk::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

// Mock HTTP framework types (in a real app, you'd use axum, warp, etc.)
type StatusCode = u16;
type Response = (StatusCode, Vec<u8>, String);

#[derive(Debug, Serialize, Deserialize)]
struct SynthesisRequest {
    text: String,
    voice: Option<String>,
    quality: Option<f32>,
    speed: Option<f32>,
    format: Option<AudioFormat>,
}

#[derive(Debug, Serialize, Deserialize)]
enum AudioFormat {
    Wav,
    Mp3,
    Ogg,
}

#[derive(Debug, Serialize, Deserialize)]
struct SynthesisResponse {
    success: bool,
    message: String,
    audio_url: Option<String>,
    duration: Option<f64>,
    size_bytes: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VoiceInfo {
    name: String,
    language: String,
    gender: Option<String>,
    quality_score: f32,
    description: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct StatusResponse {
    status: String,
    version: String,
    available_voices: usize,
    cache_usage: CacheStats,
    performance: PerformanceStats,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheStats {
    models_cached: usize,
    cache_size_mb: f64,
    hit_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceStats {
    requests_processed: u64,
    average_latency_ms: f64,
    current_load: f32,
}

/// Web server state containing the VoiRS pipeline and statistics
pub struct ServerState {
    pipeline: Arc<VoirsPipeline>,
    stats: Arc<RwLock<ServerStats>>,
    config: ServerConfig,
}

struct ServerStats {
    requests_processed: u64,
    total_latency_ms: f64,
    errors: u64,
    concurrent_requests: u32,
}

#[derive(Clone)]
struct ServerConfig {
    max_text_length: usize,
    max_concurrent_requests: u32,
    cache_directory: String,
    allowed_formats: Vec<AudioFormat>,
}

impl ServerState {
    pub async fn new() -> Result<Self, VoirsError> {
        println!("Initializing VoiRS server...");
        
        // Create optimized pipeline for server use
        let pipeline = VoirsPipelineBuilder::new()
            .with_cache_size(2048)  // 2GB cache for server
            .with_concurrent_limit(8)  // Allow 8 concurrent syntheses
            .with_quality(0.8)      // Good quality for web use
            .build()
            .await?;
        
        let stats = Arc::new(RwLock::new(ServerStats {
            requests_processed: 0,
            total_latency_ms: 0.0,
            errors: 0,
            concurrent_requests: 0,
        }));
        
        let config = ServerConfig {
            max_text_length: 10000,  // 10k character limit
            max_concurrent_requests: 10,
            cache_directory: "/tmp/voirs_cache".to_string(),
            allowed_formats: vec![AudioFormat::Wav, AudioFormat::Mp3],
        };
        
        println!("VoiRS server initialized successfully!");
        
        Ok(Self {
            pipeline: Arc::new(pipeline),
            stats,
            config,
        })
    }
    
    pub async fn handle_synthesis(&self, request: SynthesisRequest) -> Response {
        let start_time = std::time::Instant::now();
        
        // Validate request
        if let Err(error_msg) = self.validate_request(&request) {
            return self.error_response(400, &error_msg);
        }
        
        // Update concurrent request count
        {
            let mut stats = self.stats.write().await;
            stats.concurrent_requests += 1;
        }
        
        let result = self.process_synthesis_request(request).await;
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.concurrent_requests -= 1;
            stats.requests_processed += 1;
            stats.total_latency_ms += start_time.elapsed().as_millis() as f64;
            
            if result.0 != 200 {
                stats.errors += 1;
            }
        }
        
        result
    }
    
    async fn process_synthesis_request(&self, request: SynthesisRequest) -> Response {
        // Configure pipeline for this request
        let mut pipeline = self.pipeline.as_ref().clone();
        
        if let Some(voice) = &request.voice {
            if let Err(e) = pipeline.set_voice(voice) {
                return self.error_response(400, &format!("Invalid voice: {}", e));
            }
        }
        
        if let Some(quality) = request.quality {
            if let Err(e) = pipeline.set_quality(quality) {
                return self.error_response(400, &format!("Invalid quality: {}", e));
            }
        }
        
        if let Some(speed) = request.speed {
            if let Err(e) = pipeline.set_speed(speed) {
                return self.error_response(400, &format!("Invalid speed: {}", e));
            }
        }
        
        // Perform synthesis
        match pipeline.synthesize(&request.text) {
            Ok(audio) => {
                match self.save_and_respond(audio, request.format.unwrap_or(AudioFormat::Wav)).await {
                    Ok(response) => response,
                    Err(e) => self.error_response(500, &format!("Failed to save audio: {}", e)),
                }
            }
            Err(e) => self.error_response(500, &format!("Synthesis failed: {}", e)),
        }
    }
    
    async fn save_and_respond(&self, audio: AudioBuffer, format: AudioFormat) -> Result<Response, VoirsError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let filename = match format {
            AudioFormat::Wav => format!("synthesis_{}.wav", timestamp),
            AudioFormat::Mp3 => format!("synthesis_{}.mp3", timestamp),
            AudioFormat::Ogg => format!("synthesis_{}.ogg", timestamp),
        };
        
        // Save audio file
        let file_path = format!("/tmp/{}", filename);
        match format {
            AudioFormat::Wav => audio.save_wav(&file_path)?,
            _ => {
                // For demo purposes, save as WAV even for other formats
                // In a real implementation, you'd convert to the requested format
                audio.save_wav(&file_path)?;
            }
        }
        
        let response = SynthesisResponse {
            success: true,
            message: "Synthesis completed successfully".to_string(),
            audio_url: Some(format!("/audio/{}", filename)),
            duration: Some(audio.duration()),
            size_bytes: Some(audio.len() * 4), // 4 bytes per f32 sample
        };
        
        let json_response = serde_json::to_string(&response).unwrap();
        Ok((200, json_response.into_bytes(), "application/json".to_string()))
    }
    
    pub async fn handle_voices_list(&self) -> Response {
        match self.pipeline.list_voices() {
            Ok(voices) => {
                let voice_infos: Vec<VoiceInfo> = voices.iter().map(|v| VoiceInfo {
                    name: v.name.clone(),
                    language: v.language.clone(),
                    gender: Some(format!("{:?}", v.gender.unwrap_or(voirs_sdk::voice::Gender::Unknown))),
                    quality_score: v.quality_score,
                    description: v.description.clone().unwrap_or_else(|| "No description".to_string()),
                }).collect();
                
                let json_response = serde_json::to_string(&voice_infos).unwrap();
                (200, json_response.into_bytes(), "application/json".to_string())
            }
            Err(e) => self.error_response(500, &format!("Failed to list voices: {}", e)),
        }
    }
    
    pub async fn handle_status(&self) -> Response {
        let stats = self.stats.read().await;
        let voices = self.pipeline.list_voices().unwrap_or_default();
        
        let average_latency = if stats.requests_processed > 0 {
            stats.total_latency_ms / stats.requests_processed as f64
        } else {
            0.0
        };
        
        let status = StatusResponse {
            status: "healthy".to_string(),
            version: "1.0.0".to_string(),
            available_voices: voices.len(),
            cache_usage: CacheStats {
                models_cached: 5, // Mock data
                cache_size_mb: 512.0,
                hit_rate: 0.85,
            },
            performance: PerformanceStats {
                requests_processed: stats.requests_processed,
                average_latency_ms: average_latency,
                current_load: stats.concurrent_requests as f32 / self.config.max_concurrent_requests as f32,
            },
        };
        
        let json_response = serde_json::to_string(&status).unwrap();
        (200, json_response.into_bytes(), "application/json".to_string())
    }
    
    fn validate_request(&self, request: &SynthesisRequest) -> Result<(), String> {
        if request.text.is_empty() {
            return Err("Text cannot be empty".to_string());
        }
        
        if request.text.len() > self.config.max_text_length {
            return Err(format!("Text too long (max {} characters)", self.config.max_text_length));
        }
        
        if let Some(quality) = request.quality {
            if quality < 0.0 || quality > 1.0 {
                return Err("Quality must be between 0.0 and 1.0".to_string());
            }
        }
        
        if let Some(speed) = request.speed {
            if speed < 0.1 || speed > 3.0 {
                return Err("Speed must be between 0.1 and 3.0".to_string());
            }
        }
        
        Ok(())
    }
    
    fn error_response(&self, status: StatusCode, message: &str) -> Response {
        let error_response = SynthesisResponse {
            success: false,
            message: message.to_string(),
            audio_url: None,
            duration: None,
            size_bytes: None,
        };
        
        let json_response = serde_json::to_string(&error_response).unwrap();
        (status, json_response.into_bytes(), "application/json".to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    println!("=== VoiRS Web Server Integration Demo ===\n");
    
    // Initialize the server
    let server = ServerState::new().await?;
    
    // Simulate various API requests
    demo_synthesis_requests(&server).await;
    demo_voice_management(&server).await;
    demo_status_monitoring(&server).await;
    demo_error_handling(&server).await;
    demo_concurrent_requests(&server).await;
    
    println!("Web server integration demo completed!");
    Ok(())
}

async fn demo_synthesis_requests(server: &ServerState) {
    println!("=== Synthesis API Demo ===");
    
    let requests = vec![
        SynthesisRequest {
            text: "Hello, welcome to our text-to-speech API!".to_string(),
            voice: None,
            quality: Some(0.8),
            speed: Some(1.0),
            format: Some(AudioFormat::Wav),
        },
        SynthesisRequest {
            text: "This is a high-quality synthesis example.".to_string(),
            voice: Some("premium_voice".to_string()),
            quality: Some(0.95),
            speed: Some(0.9),
            format: Some(AudioFormat::Mp3),
        },
        SynthesisRequest {
            text: "Quick synthesis for testing purposes.".to_string(),
            voice: None,
            quality: Some(0.6),
            speed: Some(1.5),
            format: Some(AudioFormat::Wav),
        },
    ];
    
    for (i, request) in requests.iter().enumerate() {
        println!("Request {}: '{}'", i + 1, request.text);
        let (status, body, content_type) = server.handle_synthesis(request.clone()).await;
        println!("  Status: {}, Content-Type: {}", status, content_type);
        
        if status == 200 {
            let response: SynthesisResponse = serde_json::from_slice(&body).unwrap();
            println!("  Duration: {:.2}s", response.duration.unwrap_or(0.0));
            println!("  Size: {} bytes", response.size_bytes.unwrap_or(0));
        }
        println!();
    }
}

async fn demo_voice_management(server: &ServerState) {
    println!("=== Voice Management API Demo ===");
    
    let (status, body, _) = server.handle_voices_list().await;
    println!("Voices list status: {}", status);
    
    if status == 200 {
        let voices: Vec<VoiceInfo> = serde_json::from_slice(&body).unwrap();
        println!("Available voices: {}", voices.len());
        
        for voice in voices.iter().take(3) {
            println!("  {}: {} (Quality: {:.1})", 
                     voice.name, voice.language, voice.quality_score);
        }
    }
    println!();
}

async fn demo_status_monitoring(server: &ServerState) {
    println!("=== Status Monitoring API Demo ===");
    
    let (status, body, _) = server.handle_status().await;
    println!("Status check: {}", status);
    
    if status == 200 {
        let status_info: StatusResponse = serde_json::from_slice(&body).unwrap();
        println!("Server status: {}", status_info.status);
        println!("Requests processed: {}", status_info.performance.requests_processed);
        println!("Average latency: {:.2}ms", status_info.performance.average_latency_ms);
        println!("Current load: {:.1}%", status_info.performance.current_load * 100.0);
        println!("Cache hit rate: {:.1}%", status_info.cache_usage.hit_rate * 100.0);
    }
    println!();
}

async fn demo_error_handling(server: &ServerState) {
    println!("=== Error Handling Demo ===");
    
    let error_requests = vec![
        SynthesisRequest {
            text: "".to_string(), // Empty text
            voice: None,
            quality: None,
            speed: None,
            format: None,
        },
        SynthesisRequest {
            text: "x".repeat(20000), // Too long
            voice: None,
            quality: None,
            speed: None,
            format: None,
        },
        SynthesisRequest {
            text: "Test".to_string(),
            voice: None,
            quality: Some(1.5), // Invalid quality
            speed: None,
            format: None,
        },
    ];
    
    for (i, request) in error_requests.iter().enumerate() {
        println!("Error test {}", i + 1);
        let (status, body, _) = server.handle_synthesis(request.clone()).await;
        
        if status != 200 {
            let error_response: SynthesisResponse = serde_json::from_slice(&body).unwrap();
            println!("  Expected error: {} - {}", status, error_response.message);
        }
    }
    println!();
}

async fn demo_concurrent_requests(server: &ServerState) {
    println!("=== Concurrent Requests Demo ===");
    
    let concurrent_requests = vec![
        "First concurrent request processing",
        "Second concurrent request processing", 
        "Third concurrent request processing",
        "Fourth concurrent request processing",
    ];
    
    let futures: Vec<_> = concurrent_requests.into_iter().enumerate().map(|(i, text)| {
        let request = SynthesisRequest {
            text: text.to_string(),
            voice: None,
            quality: Some(0.7),
            speed: Some(1.0),
            format: Some(AudioFormat::Wav),
        };
        
        async move {
            let start = std::time::Instant::now();
            let (status, _, _) = server.handle_synthesis(request).await;
            let duration = start.elapsed();
            (i + 1, status, duration)
        }
    }).collect();
    
    let results = futures::future::join_all(futures).await;
    
    println!("Concurrent request results:");
    for (id, status, duration) in results {
        println!("  Request {}: Status {} in {:?}", id, status, duration);
    }
    
    // Check final status
    let (_, body, _) = server.handle_status().await;
    let status_info: StatusResponse = serde_json::from_slice(&body).unwrap();
    println!("Total requests processed: {}", status_info.performance.requests_processed);
    println!();
}

/// Example of streaming API endpoint
async fn demo_streaming_api(server: &ServerState) -> Result<(), VoirsError> {
    println!("=== Streaming API Demo ===");
    
    // In a real web server, this would be an SSE (Server-Sent Events) endpoint
    let text = "This would be streamed as chunks in a real implementation.";
    let mut stream = server.pipeline.synthesize_streaming(text)?;
    
    println!("Streaming response for: '{}'", text);
    
    let mut chunk_id = 0;
    while let Some(chunk_result) = futures::StreamExt::next(&mut stream).await {
        let chunk = chunk_result?;
        chunk_id += 1;
        
        // In a real implementation, you'd send this as an SSE event
        println!("event: audio_chunk");
        println!("data: {{\"chunk_id\": {}, \"samples\": {}}}", chunk_id, chunk.len());
        println!();
    }
    
    println!("event: synthesis_complete");
    println!("data: {{\"total_chunks\": {}}}]", chunk_id);
    println!();
    
    Ok(())
}