// Cloud API integration for VoiRS external service integration
use anyhow::Result;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use voirs_sdk::types::SynthesisConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudApiConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
    pub rate_limit_requests_per_minute: u32,
    pub enabled_services: Vec<CloudService>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CloudService {
    Translation,
    ContentManagement,
    Analytics,
    QualityAssurance,
    VoiceTraining,
    AudioProcessing,
    SpeechRecognition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationRequest {
    pub text: String,
    pub source_language: String,
    pub target_language: String,
    pub preserve_ssml: bool,
    pub quality_level: TranslationQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranslationQuality {
    Fast,
    Balanced,
    HighQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResponse {
    pub translated_text: String,
    pub detected_language: Option<String>,
    pub confidence_score: f32,
    pub processing_time_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentAnalysisRequest {
    pub content: String,
    pub analysis_types: Vec<AnalysisType>,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisType {
    Sentiment,
    Entities,
    Keywords,
    Readability,
    Appropriateness,
    Complexity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentAnalysisResponse {
    pub sentiment: Option<SentimentAnalysis>,
    pub entities: Vec<EntityExtraction>,
    pub keywords: Vec<KeywordExtraction>,
    pub readability_score: Option<f32>,
    pub appropriateness_rating: Option<AppropriattenessRating>,
    pub complexity_level: Option<ComplexityLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    pub sentiment: String, // positive, negative, neutral
    pub confidence: f32,
    pub emotional_tone: Vec<EmotionScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionScore {
    pub emotion: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtraction {
    pub text: String,
    pub entity_type: String,
    pub confidence: f32,
    pub start_offset: usize,
    pub end_offset: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordExtraction {
    pub keyword: String,
    pub relevance: f32,
    pub frequency: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AppropriattenessRating {
    Appropriate,
    Questionable,
    Inappropriate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    Advanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEvent {
    pub event_type: String,
    pub timestamp: u64,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessmentRequest {
    pub audio_data: Vec<u8>,
    pub text: String,
    pub synthesis_config: SynthesisConfig,
    pub assessment_types: Vec<QualityMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityMetric {
    Naturalness,
    Intelligibility,
    Prosody,
    Pronunciation,
    OverallQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessmentResponse {
    pub overall_score: f32,
    pub metric_scores: HashMap<String, f32>,
    pub detailed_feedback: Vec<QualityFeedback>,
    pub improvement_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFeedback {
    pub metric: String,
    pub score: f32,
    pub description: String,
    pub timestamp_range: Option<(f32, f32)>, // Start and end times in seconds
}

pub struct CloudApiClient {
    client: Client,
    config: CloudApiConfig,
    rate_limiter: RateLimiter,
}

struct RateLimiter {
    requests_per_minute: u32,
    request_times: Vec<std::time::Instant>,
}

impl CloudApiClient {
    pub fn new(config: CloudApiConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()?;

        let rate_limiter = RateLimiter::new(config.rate_limit_requests_per_minute);

        Ok(Self {
            client,
            config,
            rate_limiter,
        })
    }

    /// Translate text using cloud translation service
    pub async fn translate_text(
        &mut self,
        request: TranslationRequest,
    ) -> Result<TranslationResponse> {
        self.rate_limiter.wait_if_needed().await;

        if !self
            .config
            .enabled_services
            .contains(&CloudService::Translation)
        {
            return Err(anyhow::anyhow!("Translation service not enabled"));
        }

        let url = format!("{}/v1/translate", self.config.base_url);
        let response = self.make_request("POST", &url, Some(&request)).await?;

        let translation_response: TranslationResponse = response.json().await?;
        Ok(translation_response)
    }

    /// Analyze content using cloud AI services
    pub async fn analyze_content(
        &mut self,
        request: ContentAnalysisRequest,
    ) -> Result<ContentAnalysisResponse> {
        self.rate_limiter.wait_if_needed().await;

        if !self
            .config
            .enabled_services
            .contains(&CloudService::ContentManagement)
        {
            return Err(anyhow::anyhow!("Content analysis service not enabled"));
        }

        let url = format!("{}/v1/analyze", self.config.base_url);
        let response = self.make_request("POST", &url, Some(&request)).await?;

        let analysis_response: ContentAnalysisResponse = response.json().await?;
        Ok(analysis_response)
    }

    /// Send analytics events to cloud analytics service
    pub async fn send_analytics_event(&mut self, event: AnalyticsEvent) -> Result<()> {
        self.rate_limiter.wait_if_needed().await;

        if !self
            .config
            .enabled_services
            .contains(&CloudService::Analytics)
        {
            return Err(anyhow::anyhow!("Analytics service not enabled"));
        }

        let url = format!("{}/v1/analytics/events", self.config.base_url);
        let _response = self.make_request("POST", &url, Some(&event)).await?;

        Ok(())
    }

    /// Assess audio quality using cloud QA service
    pub async fn assess_quality(
        &mut self,
        request: QualityAssessmentRequest,
    ) -> Result<QualityAssessmentResponse> {
        self.rate_limiter.wait_if_needed().await;

        if !self
            .config
            .enabled_services
            .contains(&CloudService::QualityAssurance)
        {
            return Err(anyhow::anyhow!("Quality assessment service not enabled"));
        }

        let url = format!("{}/v1/quality/assess", self.config.base_url);

        // For quality assessment, we'll use JSON format instead of multipart for simplicity
        let payload = serde_json::json!({
            "audio_data_base64": base64::encode(&request.audio_data),
            "text": request.text,
            "config": request.synthesis_config,
            "metrics": request.assessment_types
        });

        let response = self.client.post(&url).json(&payload).send().await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Quality assessment request failed: {}",
                response.status()
            ));
        }

        let quality_response: QualityAssessmentResponse = response.json().await?;
        Ok(quality_response)
    }

    /// Get service health status
    pub async fn get_service_health(&mut self) -> Result<ServiceHealth> {
        let url = format!("{}/v1/health", self.config.base_url);
        let response = self.make_request("GET", &url, None::<&()>).await?;

        let health: ServiceHealth = response.json().await?;
        Ok(health)
    }

    async fn make_request<T: Serialize>(
        &self,
        method: &str,
        url: &str,
        body: Option<&T>,
    ) -> Result<Response> {
        let mut request_builder = match method {
            "GET" => self.client.get(url),
            "POST" => self.client.post(url),
            "PUT" => self.client.put(url),
            "DELETE" => self.client.delete(url),
            _ => return Err(anyhow::anyhow!("Unsupported HTTP method: {}", method)),
        };

        // Add authentication if available
        if let Some(api_key) = &self.config.api_key {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        // Add body if provided
        if let Some(body) = body {
            request_builder = request_builder.json(body);
        }

        // Execute request with retries
        let mut last_error = None;
        for attempt in 0..=self.config.retry_attempts {
            match request_builder.try_clone() {
                Some(request) => match request.send().await {
                    Ok(response) => {
                        if response.status().is_success() {
                            return Ok(response);
                        } else {
                            last_error = Some(anyhow::anyhow!("HTTP error: {}", response.status()));
                        }
                    }
                    Err(e) => {
                        last_error = Some(anyhow::anyhow!("Request error: {}", e));
                    }
                },
                None => {
                    last_error = Some(anyhow::anyhow!("Failed to clone request"));
                    break;
                }
            }

            if attempt < self.config.retry_attempts {
                tokio::time::sleep(Duration::from_millis(1000 * (2_u64.pow(attempt)))).await;
                // Exponential backoff
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Request failed after all retries")))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealth {
    pub status: String,
    pub services: HashMap<String, ServiceStatus>,
    pub response_time_ms: u32,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub healthy: bool,
    pub response_time_ms: Option<u32>,
    pub error_message: Option<String>,
}

impl RateLimiter {
    fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            request_times: Vec::new(),
        }
    }

    async fn wait_if_needed(&mut self) {
        let now = std::time::Instant::now();
        let one_minute_ago = now - Duration::from_secs(60);

        // Remove old request times
        self.request_times.retain(|&time| time > one_minute_ago);

        // Check if we need to wait
        if self.request_times.len() >= self.requests_per_minute as usize {
            if let Some(&oldest) = self.request_times.first() {
                let wait_until = oldest + Duration::from_secs(60);
                if now < wait_until {
                    let wait_duration = wait_until - now;
                    tokio::time::sleep(wait_duration).await;
                }
            }
        }

        // Record current request time
        self.request_times.push(now);
    }
}

impl Default for CloudApiConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.voirs.cloud".to_string(),
            api_key: None,
            timeout_seconds: 30,
            retry_attempts: 3,
            rate_limit_requests_per_minute: 60,
            enabled_services: vec![
                CloudService::Translation,
                CloudService::ContentManagement,
                CloudService::Analytics,
                CloudService::QualityAssurance,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_api_config_default() {
        let config = CloudApiConfig::default();
        assert_eq!(config.timeout_seconds, 30);
        assert_eq!(config.retry_attempts, 3);
        assert!(config.enabled_services.len() > 0);
    }

    #[test]
    fn test_translation_request_serialization() {
        let request = TranslationRequest {
            text: "Hello world".to_string(),
            source_language: "en".to_string(),
            target_language: "es".to_string(),
            preserve_ssml: true,
            quality_level: TranslationQuality::HighQuality,
        };

        let serialized = serde_json::to_string(&request);
        assert!(serialized.is_ok());

        let deserialized: Result<TranslationRequest, _> =
            serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_analytics_event_creation() {
        let mut properties = HashMap::new();
        properties.insert(
            "voice_id".to_string(),
            serde_json::Value::String("en-US-1".to_string()),
        );
        properties.insert(
            "duration_ms".to_string(),
            serde_json::Value::Number(serde_json::Number::from(5000)),
        );

        let event = AnalyticsEvent {
            event_type: "synthesis_completed".to_string(),
            timestamp: 1620000000,
            user_id: Some("user123".to_string()),
            session_id: Some("session456".to_string()),
            properties,
        };

        assert_eq!(event.event_type, "synthesis_completed");
        assert_eq!(event.properties.len(), 2);
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(60); // 60 requests per minute

        // This should not block
        let start = std::time::Instant::now();
        limiter.wait_if_needed().await;
        let elapsed = start.elapsed();

        // Should be nearly instantaneous for first request
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_quality_feedback_serialization() {
        let feedback = QualityFeedback {
            metric: "naturalness".to_string(),
            score: 0.85,
            description: "Speech sounds natural with minor robotic artifacts".to_string(),
            timestamp_range: Some((1.5, 3.2)),
        };

        let serialized = serde_json::to_string(&feedback);
        assert!(serialized.is_ok());

        let deserialized: Result<QualityFeedback, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }
}
