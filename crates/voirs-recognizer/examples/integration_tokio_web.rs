//! Integration Example: Tokio Web Server with VoiRS Recognizer
//!
//! This example demonstrates how to integrate VoiRS Recognizer into a
//! web application using Tokio and common web frameworks.
//!
//! Features Demonstrated:
//! - Web API endpoints for speech recognition
//! - Async file upload handling
//! - Real-time streaming recognition via WebSockets
//! - Error handling and status reporting
//! - Performance monitoring and metrics
//! - Concurrent request handling
//!
//! Prerequisites: Complete Tutorial series
//!
//! Usage:
//! ```bash
//! cargo run --example integration_tokio_web --features="whisper-pure"
//! ```

use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;
use voirs_recognizer::asr::{ASRBackend, WhisperModelSize};
use voirs_recognizer::audio_utilities::*;
use voirs_recognizer::prelude::*;

// Mock web framework types (in real use, these would be from axum, warp, etc.)
type WebRequest = HashMap<String, String>;
type WebResponse = HashMap<String, String>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🌐 Integration Example: Tokio Web Server with VoiRS Recognizer");
    println!("===============================================================\n");

    // Step 1: Initialize the web service
    println!("🚀 Step 1: Initializing VoiRS Web Service");
    let web_service = VoirsWebService::new().await?;

    // Step 2: Demonstrate different endpoint types
    println!(
        "
📡 Step 2: Web API Endpoints"
    );
    demonstrate_web_endpoints(&web_service).await?;

    // Step 3: Demonstrate streaming recognition
    println!(
        "
🌊 Step 3: WebSocket Streaming"
    );
    demonstrate_websocket_streaming(&web_service).await?;

    // Step 4: Demonstrate concurrent handling
    println!(
        "
⚡ Step 4: Concurrent Request Handling"
    );
    demonstrate_concurrent_requests(&web_service).await?;

    // Step 5: Demonstrate monitoring and metrics
    println!(
        "
📊 Step 5: Monitoring and Metrics"
    );
    demonstrate_monitoring(&web_service).await?;

    // Step 6: Conclusion
    println!(
        "
🎉 Integration Example Complete!"
    );
    println!(
        "
📖 What you learned:"
    );
    println!("   • How to integrate VoiRS into web applications");
    println!("   • Handling file uploads and processing");
    println!("   • Implementing real-time streaming recognition");
    println!("   • Managing concurrent requests efficiently");
    println!("   • Monitoring performance and collecting metrics");

    println!(
        "
🔧 Web Framework Integration:"
    );
    println!("   • Axum: Use VoiRS in handlers with State<Arc<VoirsWebService>>");
    println!("   • Warp: Create filters that inject VoiRS service");
    println!("   • Actix: Use VoiRS as application data");
    println!("   • Tide: Store VoiRS service in application state");

    Ok(())
}

// Web service that wraps VoiRS functionality
struct VoirsWebService {
    recognizer: Arc<RwLock<Option<MockRecognizer>>>,
    analyzer: Arc<RwLock<Option<MockAnalyzer>>>,
    metrics: Arc<RwLock<ServiceMetrics>>,
}

impl VoirsWebService {
    async fn new() -> Result<Self, Box<dyn Error>> {
        println!("   🔧 Initializing VoiRS components...");

        // Initialize ASR system
        let config = ASRConfig {
            preferred_models: vec!["whisper".to_string()],
            whisper_model_size: Some("base".to_string()),
            language: Some(LanguageCode::EnUs),
            enable_voice_activity_detection: true,
            ..Default::default()
        };

        let recognizer = MockRecognizer::new(config).await?;

        // Initialize audio analyzer
        let analyzer_config = AudioAnalysisConfig::default();
        let analyzer = MockAnalyzer::new(analyzer_config).await?;

        println!("   ✅ VoiRS components initialized");

        Ok(Self {
            recognizer: Arc::new(RwLock::new(Some(recognizer))),
            analyzer: Arc::new(RwLock::new(Some(analyzer))),
            metrics: Arc::new(RwLock::new(ServiceMetrics::new())),
        })
    }

    // REST API endpoint for file upload and recognition
    async fn handle_file_upload(
        &self,
        request: WebRequest,
    ) -> Result<WebResponse, Box<dyn Error + Send + Sync>> {
        let start_time = Instant::now();

        // Simulate file upload handling
        let file_path = request
            .get("file_path")
            .map(|s| s.as_str())
            .unwrap_or("default.wav");
        println!("   📁 Processing uploaded file: {}", file_path);

        // Load and preprocess audio
        let audio = self.load_audio_mock(file_path).await?;

        // Perform recognition
        let recognizer = self.recognizer.read().await;
        let result = if let Some(ref recognizer) = *recognizer {
            recognizer
                .recognize(&audio)
                .await
                .map_err(|e| -> Box<dyn Error + Send + Sync> {
                    format!("Recognition error: {}", e).into()
                })?
        } else {
            return Err("Recognizer not initialized".into());
        };

        // Update metrics
        let elapsed = start_time.elapsed();
        self.update_metrics("file_upload", elapsed, true).await;

        // Return response
        let mut response = HashMap::new();
        response.insert("status".to_string(), "success".to_string());
        response.insert("transcript".to_string(), result.text);
        response.insert(
            "confidence".to_string(),
            format!("{:.2}", result.confidence),
        );
        response.insert(
            "processing_time_ms".to_string(),
            format!("{}", elapsed.as_millis()),
        );

        Ok(response)
    }

    // WebSocket endpoint for real-time streaming
    async fn handle_websocket_stream(
        &self,
        _audio_chunk: Vec<f32>,
    ) -> Result<WebResponse, Box<dyn Error>> {
        let start_time = Instant::now();

        // Simulate processing audio chunk
        println!(
            "   🌊 Processing audio chunk ({} samples)",
            _audio_chunk.len()
        );

        // Simulate streaming recognition
        sleep(Duration::from_millis(50)).await; // Low latency processing

        let result = MockRecognitionResult {
            text: "Partial recognition result...".to_string(),
            confidence: 0.85,
            is_final: false,
        };

        // Update metrics
        let elapsed = start_time.elapsed();
        self.update_metrics("websocket_stream", elapsed, true).await;

        // Return streaming response
        let mut response = HashMap::new();
        response.insert("type".to_string(), "partial_result".to_string());
        response.insert("transcript".to_string(), result.text);
        response.insert(
            "confidence".to_string(),
            format!("{:.2}", result.confidence),
        );
        response.insert("is_final".to_string(), format!("{}", result.is_final));
        response.insert("latency_ms".to_string(), format!("{}", elapsed.as_millis()));

        Ok(response)
    }

    // Health check endpoint
    async fn health_check(&self) -> Result<WebResponse, Box<dyn Error>> {
        let mut response = HashMap::new();

        // Check component health
        let recognizer_healthy = self.recognizer.read().await.is_some();
        let analyzer_healthy = self.analyzer.read().await.is_some();

        response.insert("status".to_string(), "healthy".to_string());
        response.insert("recognizer".to_string(), format!("{}", recognizer_healthy));
        response.insert("analyzer".to_string(), format!("{}", analyzer_healthy));
        response.insert("timestamp".to_string(), format!("{:?}", Instant::now()));

        Ok(response)
    }

    // Metrics endpoint
    async fn get_metrics(&self) -> Result<WebResponse, Box<dyn Error>> {
        let metrics = self.metrics.read().await;
        let mut response = HashMap::new();

        response.insert(
            "total_requests".to_string(),
            format!("{}", metrics.total_requests),
        );
        response.insert(
            "successful_requests".to_string(),
            format!("{}", metrics.successful_requests),
        );
        response.insert(
            "failed_requests".to_string(),
            format!("{}", metrics.failed_requests),
        );
        response.insert(
            "average_latency_ms".to_string(),
            format!("{:.2}", metrics.average_latency_ms),
        );
        response.insert(
            "uptime_seconds".to_string(),
            format!("{}", metrics.uptime.as_secs()),
        );

        Ok(response)
    }

    // Helper methods
    async fn load_audio_mock(
        &self,
        _file_path: &str,
    ) -> Result<AudioBuffer, Box<dyn Error + Send + Sync>> {
        // Simulate audio loading
        sleep(Duration::from_millis(100)).await;

        let samples = vec![0.0; 16000]; // 1 second of silence
        Ok(AudioBuffer::mono(samples, 16000))
    }

    async fn update_metrics(&self, endpoint: &str, latency: Duration, success: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;

        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }

        metrics.total_latency += latency;
        metrics.average_latency_ms =
            metrics.total_latency.as_millis() as f64 / metrics.total_requests as f64;

        println!(
            "   📊 Updated metrics for {}: {}ms latency",
            endpoint,
            latency.as_millis()
        );
    }
}

async fn demonstrate_web_endpoints(service: &VoirsWebService) -> Result<(), Box<dyn Error>> {
    println!("   Web API endpoints demonstration:");

    // 1. File upload endpoint
    println!(
        "   
   📤 1. File Upload Endpoint:"
    );
    let mut upload_request = HashMap::new();
    upload_request.insert("file_path".to_string(), "sample_audio.wav".to_string());

    let response = service
        .handle_file_upload(upload_request)
        .await
        .map_err(|e| -> Box<dyn Error> { format!("File upload error: {}", e).into() })?;
    println!("   Request: POST /api/recognize");
    println!("   Response: {:#?}", response);

    // 2. Health check endpoint
    println!(
        "   
   🏥 2. Health Check Endpoint:"
    );
    let health_response = service.health_check().await?;
    println!("   Request: GET /api/health");
    println!("   Response: {:#?}", health_response);

    // 3. Metrics endpoint
    println!(
        "   
   📊 3. Metrics Endpoint:"
    );
    let metrics_response = service.get_metrics().await?;
    println!("   Request: GET /api/metrics");
    println!("   Response: {:#?}", metrics_response);

    println!(
        "   
   🔧 Real Web Framework Integration:"
    );
    println!("   ```rust");
    println!("   // Axum example");
    println!("   use axum::{{routing::post, Router, State}};");
    println!("   ");
    println!("   async fn recognize_handler(");
    println!("       State(service): State<Arc<VoirsWebService>>,");
    println!("       // multipart file upload");
    println!("   ) -> Result<Json<RecognitionResponse>, StatusCode> {{");
    println!("       let response = service.handle_file_upload(request).await?;");
    println!("       Ok(Json(response))");
    println!("   }}");
    println!("   ");
    println!("   let app = Router::new()");
    println!("       .route(\"/api/recognize\", post(recognize_handler))");
    println!("       .with_state(Arc::new(service));");
    println!("   ```");

    Ok(())
}

async fn demonstrate_websocket_streaming(service: &VoirsWebService) -> Result<(), Box<dyn Error>> {
    println!("   WebSocket streaming demonstration:");

    println!(
        "   
   🌊 Simulating Real-time Audio Stream:"
    );

    // Simulate audio chunks arriving
    let audio_chunks = vec![
        vec![0.1; 1600], // 100ms chunk
        vec![0.2; 1600], // 100ms chunk
        vec![0.3; 1600], // 100ms chunk
        vec![0.4; 1600], // 100ms chunk
    ];

    for (i, chunk) in audio_chunks.iter().enumerate() {
        println!("   Chunk {}: {} samples", i + 1, chunk.len());

        let response = service.handle_websocket_stream(chunk.clone()).await?;
        println!("   Response: {:#?}", response);

        // Simulate real-time arrival
        sleep(Duration::from_millis(100)).await;
    }

    println!(
        "   
   🔧 WebSocket Implementation:"
    );
    println!("   ```rust");
    println!("   use tokio_tungstenite::tungstenite::Message;");
    println!("   ");
    println!("   async fn handle_websocket(");
    println!("       ws: WebSocket,");
    println!("       service: Arc<VoirsWebService>,");
    println!("   ) {{");
    println!("       while let Some(msg) = ws.next().await {{");
    println!("           if let Ok(Message::Binary(audio_data)) = msg {{");
    println!("               let response = service");
    println!("                   .handle_websocket_stream(audio_data)");
    println!("                   .await?;");
    println!("               ");
    println!("               let response_msg = serde_json::to_string(&response)?;");
    println!("               ws.send(Message::Text(response_msg)).await?;");
    println!("           }}");
    println!("       }}");
    println!("   }}");
    println!("   ```");

    Ok(())
}

async fn demonstrate_concurrent_requests(service: &VoirsWebService) -> Result<(), Box<dyn Error>> {
    println!("   Concurrent request handling demonstration:");

    println!(
        "   
   ⚡ Simulating Multiple Concurrent Requests:"
    );

    // Create multiple concurrent requests
    let mut handles = Vec::new();

    for i in 0..5 {
        let service = service.clone();
        let handle = tokio::spawn(async move {
            let mut request = HashMap::new();
            request.insert("file_path".to_string(), format!("audio_{}.wav", i));

            let start_time = Instant::now();
            let result = service.handle_file_upload(request).await;
            let elapsed = start_time.elapsed();

            (i, result, elapsed)
        });

        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        let (request_id, result, elapsed) = handle.await?;
        match result {
            Ok(response) => {
                println!(
                    "   Request {}: ✅ Success ({}ms)",
                    request_id,
                    elapsed.as_millis()
                );
                println!(
                    "   • Transcript: {}",
                    response.get("transcript").unwrap_or(&"N/A".to_string())
                );
            }
            Err(e) => {
                println!(
                    "   Request {}: ❌ Failed ({}ms): {}",
                    request_id,
                    elapsed.as_millis(),
                    e
                );
            }
        }
    }

    println!(
        "   
   🔧 Concurrent Handling Best Practices:"
    );
    println!("   • Use Arc<RwLock<>> for shared state");
    println!("   • Implement connection pooling for database access");
    println!("   • Use semaphores to limit concurrent model usage");
    println!("   • Implement request queuing for resource management");
    println!("   • Monitor resource usage and implement backpressure");

    Ok(())
}

async fn demonstrate_monitoring(service: &VoirsWebService) -> Result<(), Box<dyn Error>> {
    println!("   Monitoring and metrics demonstration:");

    // Get current metrics
    let metrics_response = service.get_metrics().await?;
    println!(
        "   
   📊 Current Service Metrics:"
    );
    for (key, value) in &metrics_response {
        println!("   • {}: {}", key, value);
    }

    println!(
        "   
   🔍 Monitoring Integration:"
    );
    println!("   ```rust");
    println!("   // Prometheus metrics");
    println!("   use prometheus::{{Counter, Histogram, Gauge}};");
    println!("   ");
    println!("   lazy_static! {{");
    println!("       static ref REQUESTS_TOTAL: Counter = Counter::new(");
    println!("           \"voirs_requests_total\", \"Total requests processed\"");
    println!("       ).unwrap();");
    println!("       ");
    println!("       static ref REQUEST_DURATION: Histogram = Histogram::new(");
    println!("           \"voirs_request_duration_seconds\", \"Request duration\"");
    println!("       ).unwrap();");
    println!("       ");
    println!("       static ref ACTIVE_CONNECTIONS: Gauge = Gauge::new(");
    println!("           \"voirs_active_connections\", \"Active connections\"");
    println!("       ).unwrap();");
    println!("   }}");
    println!("   ");
    println!("   // In your handlers");
    println!("   let timer = REQUEST_DURATION.start_timer();");
    println!("   let result = service.handle_file_upload(request).await;");
    println!("   timer.observe_duration();");
    println!("   REQUESTS_TOTAL.inc();");
    println!("   ```");

    println!(
        "   
   🚨 Alerting and Health Checks:"
    );
    println!("   • Monitor request latency percentiles");
    println!("   • Track error rates and types");
    println!("   • Monitor memory usage and model performance");
    println!("   • Set up alerts for high latency or error rates");
    println!("   • Implement circuit breakers for failing dependencies");

    Ok(())
}

// Mock implementations for demonstration
struct MockRecognizer {
    config: ASRConfig,
}

impl MockRecognizer {
    async fn new(config: ASRConfig) -> Result<Self, Box<dyn Error>> {
        Ok(Self { config })
    }

    async fn recognize(
        &self,
        audio: &AudioBuffer,
    ) -> Result<MockRecognitionResult, Box<dyn Error>> {
        // Simulate processing time
        sleep(Duration::from_millis(200)).await;

        Ok(MockRecognitionResult {
            text: format!("Recognized {} seconds of audio", audio.duration()),
            confidence: 0.9,
            is_final: true,
        })
    }
}

impl Clone for MockRecognizer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
        }
    }
}

struct MockAnalyzer {
    config: AudioAnalysisConfig,
}

impl MockAnalyzer {
    async fn new(config: AudioAnalysisConfig) -> Result<Self, Box<dyn Error>> {
        Ok(Self { config })
    }
}

impl Clone for MockAnalyzer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
        }
    }
}

#[derive(Debug)]
struct MockRecognitionResult {
    text: String,
    confidence: f32,
    is_final: bool,
}

#[derive(Debug)]
struct ServiceMetrics {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    total_latency: Duration,
    average_latency_ms: f64,
    uptime: Duration,
    start_time: Instant,
}

impl ServiceMetrics {
    fn new() -> Self {
        let start_time = Instant::now();
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_latency: Duration::from_millis(0),
            average_latency_ms: 0.0,
            uptime: Duration::from_millis(0),
            start_time,
        }
    }
}

impl Clone for VoirsWebService {
    fn clone(&self) -> Self {
        Self {
            recognizer: self.recognizer.clone(),
            analyzer: self.analyzer.clone(),
            metrics: self.metrics.clone(),
        }
    }
}
