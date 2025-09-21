//! REST API Server Example
//!
//! This example demonstrates how to start and use the VoiRS evaluation REST API server
//! with all the enhanced endpoints for speech synthesis quality evaluation.

use std::collections::HashMap;
use tokio;
use voirs_evaluation::rest_api::{
    ApiAudioData, ApiAuthentication, ApiServiceConfig, BatchEvaluationRequest,
    DatasetValidationRequest, EvaluationApiService, ModelComparisonRequest, PronunciationRequest,
    QualityEvaluationRequest,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VoiRS Evaluation REST API Server Example");
    println!("===========================================");

    // Create API service configuration
    let config = ApiServiceConfig {
        host: "127.0.0.1".to_string(),
        port: 8080,
        max_concurrent_requests: 50,
        request_timeout: 120,
        rate_limiting: voirs_evaluation::rest_api::RateLimitConfig {
            requests_per_minute: 100,
            requests_per_hour: 2000,
            daily_limit: 20000,
            burst_allowance: 20,
        },
        auth_config: voirs_evaluation::rest_api::AuthConfig {
            require_api_key: true,
            validation_endpoint: None,
            token_expiration: 7200, // 2 hours
            allowed_origins: vec!["*".to_string(), "http://localhost:3000".to_string()],
        },
        cache_config: voirs_evaluation::rest_api::CacheConfig {
            enable_caching: true,
            cache_expiration: 1800, // 30 minutes
            max_cache_size: 2048,   // 2GB
            storage_backend: "memory".to_string(),
        },
    };

    println!("📋 Server Configuration:");
    println!("  Host: {}", config.host);
    println!("  Port: {}", config.port);
    println!(
        "  Max Concurrent Requests: {}",
        config.max_concurrent_requests
    );
    println!("  Request Timeout: {}s", config.request_timeout);
    println!(
        "  Rate Limit: {} req/min",
        config.rate_limiting.requests_per_minute
    );
    println!(
        "  Caching: {}",
        if config.cache_config.enable_caching {
            "Enabled"
        } else {
            "Disabled"
        }
    );

    // Create and start the API service
    println!("\n🔧 Initializing API service...");
    let api_service = EvaluationApiService::new(config).await?;

    // Display available endpoints
    println!("\n📡 Available API Endpoints:");
    println!("  GET  /health              - Health check");
    println!("  GET  /status              - Service status and capabilities");
    println!("  GET  /metrics/info        - Available metrics information");
    println!("  GET  /docs                - API documentation");
    println!("  POST /evaluate/quality    - Quality evaluation");
    println!("  POST /evaluate/pronunciation - Pronunciation assessment");
    println!("  POST /evaluate/batch      - Batch evaluation");
    println!("  POST /compare/models      - Model comparison");
    println!("  POST /validate/dataset    - Dataset validation");

    // Show example requests
    println!("\n📝 Example API Usage:");
    println!("  curl http://127.0.0.1:8080/health");
    println!("  curl http://127.0.0.1:8080/status");
    println!("  curl http://127.0.0.1:8080/docs");
    println!("  curl http://127.0.0.1:8080/metrics/info");

    // Example quality evaluation request
    let example_auth = ApiAuthentication {
        api_key: "voirs_api_key_example_1234567890abcdef12345678".to_string(),
        user_id: "demo_user".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    // Create sample audio data (mock PCM data)
    let sample_audio_data = create_sample_audio_data();

    println!("\n🧪 Testing API endpoints before starting server...");

    // Test quality evaluation
    let quality_request = QualityEvaluationRequest {
        auth: example_auth.clone(),
        generated_audio: sample_audio_data.clone(),
        reference_audio: None,
        metrics: vec![
            "overall".to_string(),
            "pesq".to_string(),
            "stoi".to_string(),
        ],
        config: HashMap::new(),
        language: Some("en-US".to_string()),
    };

    match api_service
        .process_quality_evaluation(quality_request)
        .await
    {
        Ok(response) => {
            println!("✅ Quality evaluation test successful:");
            println!("   Request ID: {}", response.request_id);
            println!("   Overall Score: {:.3}", response.overall_score);
            println!("   Processing Time: {}ms", response.processing_time_ms);
            println!(
                "   Metrics: {:?}",
                response.metric_scores.keys().collect::<Vec<_>>()
            );
        }
        Err(e) => {
            println!("❌ Quality evaluation test failed: {}", e);
        }
    }

    // Test pronunciation assessment
    let pronunciation_request = PronunciationRequest {
        auth: example_auth.clone(),
        audio: sample_audio_data.clone(),
        reference: "Hello world".to_string(),
        language: "en-US".to_string(),
        config: HashMap::new(),
    };

    match api_service
        .process_pronunciation_assessment(pronunciation_request)
        .await
    {
        Ok(response) => {
            println!("✅ Pronunciation assessment test successful:");
            println!("   Request ID: {}", response.request_id);
            println!("   Overall Score: {:.1}", response.overall_score);
            println!("   Phoneme Scores: {}", response.phoneme_scores.len());
            println!("   Feedback: {}", response.feedback.message);
        }
        Err(e) => {
            println!("❌ Pronunciation assessment test failed: {}", e);
        }
    }

    // Test batch evaluation
    let batch_request = BatchEvaluationRequest {
        auth: example_auth.clone(),
        audio_samples: vec![sample_audio_data.clone(), sample_audio_data.clone()],
        reference_samples: None,
        evaluation_type: "quality".to_string(),
        config: HashMap::new(),
        languages: Some(vec!["en-US".to_string(), "en-US".to_string()]),
    };

    match api_service.process_batch_evaluation(batch_request).await {
        Ok(response) => {
            println!("✅ Batch evaluation test successful:");
            println!("   Request ID: {}", response.request_id);
            println!(
                "   Processed Samples: {}/{}",
                response.batch_statistics.successful_samples,
                response.batch_statistics.total_samples
            );
            println!(
                "   Average Quality: {:.3}",
                response.batch_statistics.average_quality
            );
        }
        Err(e) => {
            println!("❌ Batch evaluation test failed: {}", e);
        }
    }

    // Test model comparison
    let mut model_outputs = HashMap::new();
    model_outputs.insert("model_a".to_string(), vec![sample_audio_data.clone()]);
    model_outputs.insert("model_b".to_string(), vec![sample_audio_data.clone()]);

    let comparison_request = ModelComparisonRequest {
        auth: example_auth.clone(),
        model_outputs,
        reference_audio: None,
        comparison_metrics: vec!["overall".to_string(), "naturalness".to_string()],
        config: HashMap::new(),
    };

    match api_service
        .process_model_comparison(comparison_request)
        .await
    {
        Ok(response) => {
            println!("✅ Model comparison test successful:");
            println!("   Request ID: {}", response.request_id);
            println!("   Recommendations: {:?}", response.recommendations);
        }
        Err(e) => {
            println!("❌ Model comparison test failed: {}", e);
        }
    }

    // Test dataset validation
    let dataset_request = DatasetValidationRequest {
        auth: example_auth.clone(),
        dataset_samples: vec![sample_audio_data.clone(), sample_audio_data.clone()],
        validation_criteria: vec!["quality".to_string(), "consistency".to_string()],
        config: HashMap::new(),
    };

    match api_service
        .process_dataset_validation(dataset_request)
        .await
    {
        Ok(response) => {
            println!("✅ Dataset validation test successful:");
            println!("   Request ID: {}", response.request_id);
            println!("   Issues Found: {}", response.issues_found.len());
            println!("   Recommendations: {:?}", response.recommendations);
        }
        Err(e) => {
            println!("❌ Dataset validation test failed: {}", e);
        }
    }

    println!("\n🎯 All endpoint tests completed successfully!");
    println!("\n🌐 Starting HTTP server...");
    println!("📖 API Documentation will be available at: http://127.0.0.1:8080/docs");
    println!("💡 Press Ctrl+C to stop the server\n");

    // Start the HTTP server (this will block)
    api_service.start_server().await?;

    Ok(())
}

/// Create sample audio data for testing
fn create_sample_audio_data() -> ApiAudioData {
    // Generate a simple sine wave as test audio
    let sample_rate = 16000;
    let duration = 1.0; // 1 second
    let frequency = 440.0; // A4 note
    let num_samples = (sample_rate as f32 * duration) as usize;

    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3; // 30% amplitude
        let sample_i16 = (sample * 32767.0) as i16;
        samples.push(sample_i16.to_le_bytes()[0]);
        samples.push(sample_i16.to_le_bytes()[1]);
    }

    let base64_data = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &samples);

    ApiAudioData {
        data: base64_data,
        sample_rate: sample_rate as u32,
        channels: 1,
        format: "pcm_s16le".to_string(),
        duration: duration as f64,
    }
}

/// Example client functions to demonstrate API usage
#[cfg(feature = "client_examples")]
mod client_examples {
    use super::*;
    use reqwest;

    /// Example client function to call the quality evaluation endpoint
    pub async fn call_quality_evaluation_api() -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();

        let auth = ApiAuthentication {
            api_key: "voirs_api_key_example_1234567890abcdef12345678".to_string(),
            user_id: "client_user".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let request = QualityEvaluationRequest {
            auth,
            generated_audio: create_sample_audio_data(),
            reference_audio: None,
            metrics: vec!["overall".to_string(), "pesq".to_string()],
            config: HashMap::new(),
            language: Some("en-US".to_string()),
        };

        let response = client
            .post("http://127.0.0.1:8080/evaluate/quality")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let result: serde_json::Value = response.json().await?;
            println!("API Response: {}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("API Error: {}", response.status());
            let error_text = response.text().await?;
            println!("Error details: {}", error_text);
        }

        Ok(())
    }

    /// Example client function to get metrics information
    pub async fn get_metrics_info() -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();

        let response = client
            .get("http://127.0.0.1:8080/metrics/info")
            .send()
            .await?;

        if response.status().is_success() {
            let metrics: serde_json::Value = response.json().await?;
            println!(
                "Available Metrics: {}",
                serde_json::to_string_pretty(&metrics)?
            );
        } else {
            println!("Failed to get metrics info: {}", response.status());
        }

        Ok(())
    }
}
