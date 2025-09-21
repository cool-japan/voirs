//! API framework for VoiRS feedback integration

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{AuthConfig, IntegrationError, IntegrationResult, RateLimitConfig};
use crate::traits::{FeedbackResponse, SessionState, UserPreferences, UserProgress};

/// API request model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequest<T> {
    /// Request ID
    pub request_id: String,
    /// API version
    pub version: String,
    /// Request timestamp
    pub timestamp: DateTime<Utc>,
    /// Request data
    pub data: T,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// API response model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Response ID
    pub response_id: String,
    /// Request ID this responds to
    pub request_id: String,
    /// Success flag
    pub success: bool,
    /// Response data
    pub data: Option<T>,
    /// Error information
    pub error: Option<ApiError>,
    /// Response timestamp
    pub timestamp: DateTime<Utc>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// API error details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error details
    pub details: Option<HashMap<String, String>>,
}

/// API endpoint definitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ApiEndpoint {
    /// Create feedback session
    CreateSession,
    /// Get session status
    GetSession,
    /// Update session
    UpdateSession,
    /// End session
    EndSession,
    /// Get user progress
    GetProgress,
    /// Update user progress
    UpdateProgress,
    /// Get user preferences
    GetPreferences,
    /// Update user preferences
    UpdatePreferences,
    /// Generate feedback
    GenerateFeedback,
    /// Get feedback history
    GetFeedbackHistory,
    /// Health check
    HealthCheck,
    /// Get statistics
    GetStatistics,
}

/// API manager trait
#[async_trait]
pub trait ApiManager: Send + Sync {
    /// Initialize the API manager
    async fn initialize(&mut self) -> IntegrationResult<()>;

    /// Handle API request
    async fn handle_request<T, R>(
        &self,
        endpoint: ApiEndpoint,
        request: ApiRequest<T>,
    ) -> IntegrationResult<ApiResponse<R>>
    where
        T: Send + Sync + Serialize + for<'de> Deserialize<'de>,
        R: Send + Sync + Serialize + for<'de> Deserialize<'de>;

    /// Validate authentication
    async fn validate_auth(&self, auth_header: &str) -> IntegrationResult<bool>;

    /// Check rate limits
    async fn check_rate_limit(&self, client_id: &str) -> IntegrationResult<bool>;

    /// Get API statistics
    async fn get_api_statistics(&self) -> IntegrationResult<ApiStatistics>;
}

/// API statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiStatistics {
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Rate limited requests
    pub rate_limited_requests: u64,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Requests by endpoint
    pub requests_by_endpoint: HashMap<String, u64>,
    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,
}

/// Feedback API implementation
pub struct FeedbackApiManager {
    auth_config: AuthConfig,
    rate_limit_config: RateLimitConfig,
    statistics: ApiStatistics,
    rate_limit_store: HashMap<String, RateLimitEntry>,
}

/// Rate limit tracking entry
#[derive(Debug, Clone)]
struct RateLimitEntry {
    requests: u32,
    window_start: DateTime<Utc>,
    burst_used: u32,
}

impl FeedbackApiManager {
    /// Create a new feedback API manager
    pub fn new(auth_config: AuthConfig, rate_limit_config: RateLimitConfig) -> Self {
        Self {
            auth_config,
            rate_limit_config,
            statistics: ApiStatistics {
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                rate_limited_requests: 0,
                avg_response_time_ms: 0.0,
                requests_by_endpoint: HashMap::new(),
                errors_by_type: HashMap::new(),
            },
            rate_limit_store: HashMap::new(),
        }
    }

    /// Update statistics
    fn update_statistics(&mut self, endpoint: &ApiEndpoint, success: bool, response_time_ms: u64) {
        self.statistics.total_requests += 1;

        if success {
            self.statistics.successful_requests += 1;
        } else {
            self.statistics.failed_requests += 1;
        }

        // Update average response time
        let total_time =
            self.statistics.avg_response_time_ms * (self.statistics.total_requests - 1) as f64;
        self.statistics.avg_response_time_ms =
            (total_time + response_time_ms as f64) / self.statistics.total_requests as f64;

        // Update endpoint statistics
        let endpoint_name = format!("{:?}", endpoint);
        *self
            .statistics
            .requests_by_endpoint
            .entry(endpoint_name)
            .or_insert(0) += 1;
    }

    /// Check rate limit for client
    fn check_client_rate_limit(&mut self, client_id: &str) -> bool {
        if !self.rate_limit_config.enabled {
            return true;
        }

        let now = Utc::now();
        let entry = self
            .rate_limit_store
            .entry(client_id.to_string())
            .or_insert_with(|| RateLimitEntry {
                requests: 0,
                window_start: now,
                burst_used: 0,
            });

        // Check if window has expired
        let window_elapsed = now.signed_duration_since(entry.window_start).num_seconds() as u32;
        if window_elapsed >= self.rate_limit_config.window_seconds {
            // Reset window
            entry.requests = 0;
            entry.window_start = now;
            entry.burst_used = 0;
        }

        // Check burst allowance
        if entry.burst_used < self.rate_limit_config.burst_allowance {
            entry.burst_used += 1;
            entry.requests += 1;
            return true;
        }

        // Check regular rate limit
        if entry.requests < self.rate_limit_config.max_requests {
            entry.requests += 1;
            true
        } else {
            self.statistics.rate_limited_requests += 1;
            false
        }
    }
}

#[async_trait]
impl ApiManager for FeedbackApiManager {
    async fn initialize(&mut self) -> IntegrationResult<()> {
        log::info!("Initializing Feedback API Manager");
        // Initialize any necessary resources
        Ok(())
    }

    async fn handle_request<T, R>(
        &self,
        endpoint: ApiEndpoint,
        request: ApiRequest<T>,
    ) -> IntegrationResult<ApiResponse<R>>
    where
        T: Send + Sync + Serialize + for<'de> Deserialize<'de>,
        R: Send + Sync + Serialize + for<'de> Deserialize<'de>,
    {
        let start_time = std::time::Instant::now();
        let response_id = uuid::Uuid::new_v4().to_string();

        // Simulate endpoint handling (in real implementation, this would route to actual handlers)
        let result: Result<(), IntegrationError> = match endpoint {
            ApiEndpoint::HealthCheck => {
                // Health check always succeeds
                log::info!(
                    "Health check successful for request: {}",
                    request.request_id
                );
                Ok(())
            }
            ApiEndpoint::CreateSession => {
                // Validate session creation request
                log::info!("Creating new session for request: {}", request.request_id);
                Ok(())
            }
            ApiEndpoint::GetSession => {
                // Validate session ID and return session data
                log::info!("Getting session for request: {}", request.request_id);
                Ok(())
            }
            ApiEndpoint::UpdateSession => {
                // Update existing session
                log::info!("Updating session for request: {}", request.request_id);
                Ok(())
            }
            ApiEndpoint::EndSession => {
                // End session and cleanup resources
                log::info!("Ending session for request: {}", request.request_id);
                Ok(())
            }
            ApiEndpoint::GetProgress => {
                // Get user progress data
                log::info!("Getting user progress for request: {}", request.request_id);
                Ok(())
            }
            ApiEndpoint::UpdateProgress => {
                // Update user progress
                log::info!("Updating user progress for request: {}", request.request_id);
                Ok(())
            }
            ApiEndpoint::GetPreferences => {
                // Get user preferences
                log::info!(
                    "Getting user preferences for request: {}",
                    request.request_id
                );
                Ok(())
            }
            ApiEndpoint::UpdatePreferences => {
                // Update user preferences
                log::info!(
                    "Updating user preferences for request: {}",
                    request.request_id
                );
                Ok(())
            }
            ApiEndpoint::GenerateFeedback => {
                // Process feedback generation request
                log::info!("Generating feedback for request: {}", request.request_id);
                Ok(())
            }
            ApiEndpoint::GetFeedbackHistory => {
                // Get feedback history for user/session
                log::info!(
                    "Getting feedback history for request: {}",
                    request.request_id
                );
                Ok(())
            }
            ApiEndpoint::GetStatistics => {
                // Get system statistics
                log::info!(
                    "Getting system statistics for request: {}",
                    request.request_id
                );
                Ok(())
            }
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(_) => Ok(ApiResponse {
                response_id,
                request_id: request.request_id,
                success: true,
                data: None, // Would contain actual response data
                error: None,
                timestamp: Utc::now(),
                processing_time_ms: processing_time,
            }),
            Err(error) => Ok(ApiResponse {
                response_id,
                request_id: request.request_id,
                success: false,
                data: None,
                error: Some(ApiError {
                    code: "PROCESSING_ERROR".to_string(),
                    message: error.to_string(),
                    details: None,
                }),
                timestamp: Utc::now(),
                processing_time_ms: processing_time,
            }),
        }
    }

    async fn validate_auth(&self, auth_header: &str) -> IntegrationResult<bool> {
        match &self.auth_config.auth_type {
            crate::integration::AuthType::None => Ok(true),
            crate::integration::AuthType::ApiKey => {
                if let Some(expected_key) = &self.auth_config.api_key {
                    Ok(auth_header.trim_start_matches("Bearer ") == expected_key)
                } else {
                    Ok(false)
                }
            }
            crate::integration::AuthType::Basic => {
                // Basic auth validation would be implemented here
                Ok(auth_header.starts_with("Basic "))
            }
            _ => {
                log::warn!(
                    "Auth type {:?} not yet implemented",
                    self.auth_config.auth_type
                );
                Err(IntegrationError::AuthenticationError {
                    message: "Authentication method not implemented".to_string(),
                })
            }
        }
    }

    async fn check_rate_limit(&self, client_id: &str) -> IntegrationResult<bool> {
        // Note: This would need to be implemented with proper thread safety in production
        Ok(true) // Simplified implementation
    }

    async fn get_api_statistics(&self) -> IntegrationResult<ApiStatistics> {
        Ok(self.statistics.clone())
    }
}

/// API request builders for common operations
pub mod builders {
    use super::*;

    /// Session creation request data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CreateSessionRequest {
        /// User ID
        pub user_id: String,
        /// Session configuration
        pub config: Option<SessionConfig>,
    }

    /// Session configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SessionConfig {
        /// Enable real-time feedback
        pub enable_realtime: bool,
        /// Session timeout in seconds
        pub timeout_seconds: u64,
        /// Custom settings
        pub custom_settings: HashMap<String, String>,
    }

    /// Feedback generation request data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GenerateFeedbackRequest {
        /// User ID
        pub user_id: String,
        /// Session ID
        pub session_id: String,
        /// Audio data (base64 encoded)
        pub audio_data: String,
        /// Expected text
        pub expected_text: String,
        /// Feedback options
        pub options: FeedbackOptions,
    }

    /// Feedback generation options
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FeedbackOptions {
        /// Include pronunciation analysis
        pub include_pronunciation: bool,
        /// Include quality assessment
        pub include_quality: bool,
        /// Include suggestions
        pub include_suggestions: bool,
        /// Detail level (0.0 to 1.0)
        pub detail_level: f32,
    }

    /// Update session request data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UpdateSessionRequest {
        /// Session ID
        pub session_id: String,
        /// Updated configuration
        pub config: Option<SessionConfig>,
        /// Status update
        pub status: Option<String>,
    }

    /// End session request data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EndSessionRequest {
        /// Session ID
        pub session_id: String,
        /// End reason
        pub reason: Option<String>,
        /// Final statistics
        pub final_stats: Option<HashMap<String, String>>,
    }

    /// Update progress request data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UpdateProgressRequest {
        /// User ID
        pub user_id: String,
        /// Progress data
        pub progress_data: HashMap<String, f64>,
        /// Completion status
        pub completion_status: Option<String>,
    }

    /// Update preferences request data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UpdatePreferencesRequest {
        /// User ID
        pub user_id: String,
        /// Preference updates
        pub preferences: HashMap<String, String>,
        /// Notification settings
        pub notifications: Option<bool>,
    }

    /// Get feedback history request data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GetFeedbackHistoryRequest {
        /// User ID
        pub user_id: String,
        /// Session ID (optional)
        pub session_id: Option<String>,
        /// Limit number of results
        pub limit: Option<u32>,
        /// Offset for pagination
        pub offset: Option<u32>,
    }

    /// Create API request builder
    pub fn create_api_request<T>(data: T) -> ApiRequest<T> {
        ApiRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            version: "v1".to_string(),
            timestamp: Utc::now(),
            data,
            metadata: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integration::{AuthConfig, AuthType};

    #[tokio::test]
    async fn test_api_manager_creation() {
        let auth_config = AuthConfig {
            auth_type: AuthType::ApiKey,
            api_key: Some("test_key".to_string()),
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };

        let rate_limit_config = RateLimitConfig::default();
        let mut manager = FeedbackApiManager::new(auth_config, rate_limit_config);

        manager.initialize().await.unwrap();
        let stats = manager.get_api_statistics().await.unwrap();
        assert_eq!(stats.total_requests, 0);
    }

    #[tokio::test]
    async fn test_auth_validation() {
        let auth_config = AuthConfig {
            auth_type: AuthType::ApiKey,
            api_key: Some("test_key".to_string()),
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };

        let rate_limit_config = RateLimitConfig::default();
        let manager = FeedbackApiManager::new(auth_config, rate_limit_config);

        // Valid API key
        assert!(manager.validate_auth("Bearer test_key").await.unwrap());

        // Invalid API key
        assert!(!manager.validate_auth("Bearer wrong_key").await.unwrap());
    }

    #[test]
    fn test_api_request_builder() {
        let request_data = builders::CreateSessionRequest {
            user_id: "test_user".to_string(),
            config: Some(builders::SessionConfig {
                enable_realtime: true,
                timeout_seconds: 300,
                custom_settings: HashMap::new(),
            }),
        };

        let request = builders::create_api_request(request_data);
        assert_eq!(request.version, "v1");
        assert!(!request.request_id.is_empty());
    }

    #[tokio::test]
    async fn test_api_request_handling() {
        let auth_config = AuthConfig {
            auth_type: AuthType::None,
            api_key: None,
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };

        let rate_limit_config = RateLimitConfig::default();
        let manager = FeedbackApiManager::new(auth_config, rate_limit_config);

        // Test health check endpoint
        let health_request = builders::create_api_request(());
        let response = manager
            .handle_request::<(), ()>(ApiEndpoint::HealthCheck, health_request)
            .await
            .unwrap();

        assert!(response.success);
        assert!(response.processing_time_ms < 1000); // Should be fast
    }

    #[tokio::test]
    async fn test_different_auth_types() {
        // Test None authentication
        let auth_config = AuthConfig {
            auth_type: AuthType::None,
            api_key: None,
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };
        let manager = FeedbackApiManager::new(auth_config, RateLimitConfig::default());
        assert!(manager.validate_auth("any_header").await.unwrap());

        // Test Basic authentication
        let auth_config = AuthConfig {
            auth_type: AuthType::Basic,
            api_key: None,
            oauth_token: None,
            jwt_token: None,
            basic_auth: Some(crate::integration::BasicAuth {
                username: "user".to_string(),
                password: "pass".to_string(),
            }),
            custom_headers: HashMap::new(),
        };
        let manager = FeedbackApiManager::new(auth_config, RateLimitConfig::default());
        assert!(manager.validate_auth("Basic dXNlcjpwYXNz").await.unwrap());
        assert!(!manager.validate_auth("Bearer token").await.unwrap());
    }

    #[tokio::test]
    async fn test_api_statistics_tracking() {
        let auth_config = AuthConfig {
            auth_type: AuthType::None,
            api_key: None,
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };

        let rate_limit_config = RateLimitConfig::default();
        let mut manager = FeedbackApiManager::new(auth_config, rate_limit_config);

        // Initially no statistics
        let initial_stats = manager.get_api_statistics().await.unwrap();
        assert_eq!(initial_stats.total_requests, 0);

        // Make a request to update statistics
        manager.update_statistics(&ApiEndpoint::HealthCheck, true, 50);
        assert_eq!(manager.statistics.total_requests, 1);
        assert_eq!(manager.statistics.successful_requests, 1);
        assert_eq!(manager.statistics.failed_requests, 0);

        // Make a failed request
        manager.update_statistics(&ApiEndpoint::CreateSession, false, 100);
        assert_eq!(manager.statistics.total_requests, 2);
        assert_eq!(manager.statistics.successful_requests, 1);
        assert_eq!(manager.statistics.failed_requests, 1);
        assert_eq!(manager.statistics.avg_response_time_ms, 75.0); // (50 + 100) / 2
    }

    #[test]
    fn test_rate_limiting() {
        let auth_config = AuthConfig {
            auth_type: AuthType::None,
            api_key: None,
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };

        let rate_limit_config = RateLimitConfig {
            enabled: true,
            max_requests: 5,
            window_seconds: 60,
            burst_allowance: 2,
        };

        let mut manager = FeedbackApiManager::new(auth_config, rate_limit_config);

        let client_id = "test_client";

        // Should allow burst requests
        assert!(manager.check_client_rate_limit(client_id));
        assert!(manager.check_client_rate_limit(client_id));

        // Should still allow within rate limit
        assert!(manager.check_client_rate_limit(client_id));
        assert!(manager.check_client_rate_limit(client_id));
        assert!(manager.check_client_rate_limit(client_id));

        // Should hit rate limit
        assert!(!manager.check_client_rate_limit(client_id));
    }

    #[test]
    fn test_rate_limiting_disabled() {
        let auth_config = AuthConfig {
            auth_type: AuthType::None,
            api_key: None,
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };

        let rate_limit_config = RateLimitConfig {
            enabled: false,
            max_requests: 1,
            window_seconds: 60,
            burst_allowance: 0,
        };

        let mut manager = FeedbackApiManager::new(auth_config, rate_limit_config);

        let client_id = "test_client";

        // Should always allow when disabled
        for _ in 0..10 {
            assert!(manager.check_client_rate_limit(client_id));
        }
    }

    #[tokio::test]
    async fn test_all_api_endpoints() {
        let auth_config = AuthConfig {
            auth_type: AuthType::None,
            api_key: None,
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };

        let rate_limit_config = RateLimitConfig::default();
        let manager = FeedbackApiManager::new(auth_config, rate_limit_config);

        // Test all implemented endpoints
        let endpoints = vec![
            ApiEndpoint::HealthCheck,
            ApiEndpoint::CreateSession,
            ApiEndpoint::GetSession,
            ApiEndpoint::UpdateSession,
            ApiEndpoint::EndSession,
            ApiEndpoint::GetProgress,
            ApiEndpoint::UpdateProgress,
            ApiEndpoint::GetPreferences,
            ApiEndpoint::UpdatePreferences,
            ApiEndpoint::GenerateFeedback,
            ApiEndpoint::GetFeedbackHistory,
            ApiEndpoint::GetStatistics,
        ];

        for endpoint in endpoints {
            let endpoint_clone = endpoint.clone();
            let request = builders::create_api_request(());
            let response = manager
                .handle_request::<(), ()>(endpoint, request)
                .await
                .unwrap();

            assert!(
                response.success,
                "Endpoint {:?} should succeed",
                endpoint_clone
            );
            assert!(response.error.is_none());
            assert!(response.processing_time_ms < 1000); // Should be fast
        }
    }

    #[test]
    fn test_api_response_structure() {
        let response = ApiResponse::<String> {
            response_id: "resp_123".to_string(),
            request_id: "req_123".to_string(),
            success: true,
            data: Some("test_data".to_string()),
            error: None,
            timestamp: chrono::Utc::now(),
            processing_time_ms: 50,
        };

        assert_eq!(response.response_id, "resp_123");
        assert_eq!(response.request_id, "req_123");
        assert!(response.success);
        assert_eq!(response.data, Some("test_data".to_string()));
        assert!(response.error.is_none());
        assert_eq!(response.processing_time_ms, 50);
    }

    #[test]
    fn test_api_error_structure() {
        let mut details = HashMap::new();
        details.insert("field".to_string(), "invalid".to_string());

        let error = ApiError {
            code: "VALIDATION_ERROR".to_string(),
            message: "Invalid input data".to_string(),
            details: Some(details),
        };

        assert_eq!(error.code, "VALIDATION_ERROR");
        assert_eq!(error.message, "Invalid input data");
        assert!(error.details.is_some());
        assert_eq!(
            error.details.unwrap().get("field"),
            Some(&"invalid".to_string())
        );
    }

    #[test]
    fn test_builders_structures() {
        // Test CreateSessionRequest
        let session_request = builders::CreateSessionRequest {
            user_id: "user123".to_string(),
            config: Some(builders::SessionConfig {
                enable_realtime: true,
                timeout_seconds: 300,
                custom_settings: HashMap::new(),
            }),
        };
        assert_eq!(session_request.user_id, "user123");
        assert!(session_request.config.is_some());

        // Test GenerateFeedbackRequest
        let feedback_request = builders::GenerateFeedbackRequest {
            user_id: "user123".to_string(),
            session_id: "session456".to_string(),
            audio_data: "base64_audio_data".to_string(),
            expected_text: "Hello world".to_string(),
            options: builders::FeedbackOptions {
                include_pronunciation: true,
                include_quality: true,
                include_suggestions: false,
                detail_level: 0.8,
            },
        };
        assert_eq!(feedback_request.user_id, "user123");
        assert_eq!(feedback_request.session_id, "session456");
        assert_eq!(feedback_request.options.detail_level, 0.8);

        // Test UpdateSessionRequest
        let update_session_request = builders::UpdateSessionRequest {
            session_id: "session456".to_string(),
            config: None,
            status: Some("active".to_string()),
        };
        assert_eq!(update_session_request.session_id, "session456");
        assert_eq!(update_session_request.status, Some("active".to_string()));

        // Test EndSessionRequest
        let end_session_request = builders::EndSessionRequest {
            session_id: "session456".to_string(),
            reason: Some("completed".to_string()),
            final_stats: Some({
                let mut stats = HashMap::new();
                stats.insert("duration".to_string(), "300".to_string());
                stats
            }),
        };
        assert_eq!(end_session_request.session_id, "session456");
        assert!(end_session_request.final_stats.is_some());

        // Test UpdateProgressRequest
        let update_progress_request = builders::UpdateProgressRequest {
            user_id: "user123".to_string(),
            progress_data: {
                let mut data = HashMap::new();
                data.insert("pronunciation".to_string(), 0.85);
                data.insert("fluency".to_string(), 0.75);
                data
            },
            completion_status: Some("in_progress".to_string()),
        };
        assert_eq!(update_progress_request.user_id, "user123");
        assert_eq!(update_progress_request.progress_data.len(), 2);

        // Test UpdatePreferencesRequest
        let update_preferences_request = builders::UpdatePreferencesRequest {
            user_id: "user123".to_string(),
            preferences: {
                let mut prefs = HashMap::new();
                prefs.insert("language".to_string(), "en".to_string());
                prefs.insert("difficulty".to_string(), "intermediate".to_string());
                prefs
            },
            notifications: Some(true),
        };
        assert_eq!(update_preferences_request.user_id, "user123");
        assert_eq!(update_preferences_request.notifications, Some(true));

        // Test GetFeedbackHistoryRequest
        let get_history_request = builders::GetFeedbackHistoryRequest {
            user_id: "user123".to_string(),
            session_id: Some("session456".to_string()),
            limit: Some(10),
            offset: Some(0),
        };
        assert_eq!(get_history_request.user_id, "user123");
        assert_eq!(get_history_request.limit, Some(10));
    }

    #[tokio::test]
    async fn test_auth_validation_edge_cases() {
        // Test JWT authentication (not implemented)
        let auth_config = AuthConfig {
            auth_type: AuthType::Jwt,
            api_key: None,
            oauth_token: None,
            jwt_token: Some("jwt_token".to_string()),
            basic_auth: None,
            custom_headers: HashMap::new(),
        };
        let manager = FeedbackApiManager::new(auth_config, RateLimitConfig::default());

        let result = manager.validate_auth("Bearer jwt_token").await;
        assert!(result.is_err()); // Should return error for unimplemented auth type

        // Test OAuth authentication (not implemented)
        let auth_config = AuthConfig {
            auth_type: AuthType::OAuth,
            api_key: None,
            oauth_token: Some("oauth_token".to_string()),
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };
        let manager = FeedbackApiManager::new(auth_config, RateLimitConfig::default());

        let result = manager.validate_auth("Bearer oauth_token").await;
        assert!(result.is_err()); // Should return error for unimplemented auth type
    }

    #[tokio::test]
    async fn test_api_key_validation_edge_cases() {
        // Test API key auth without key configured
        let auth_config = AuthConfig {
            auth_type: AuthType::ApiKey,
            api_key: None, // No key configured
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };
        let manager = FeedbackApiManager::new(auth_config, RateLimitConfig::default());

        assert!(!manager.validate_auth("Bearer any_key").await.unwrap());

        // Test empty header
        let auth_config = AuthConfig {
            auth_type: AuthType::ApiKey,
            api_key: Some("test_key".to_string()),
            oauth_token: None,
            jwt_token: None,
            basic_auth: None,
            custom_headers: HashMap::new(),
        };
        let manager = FeedbackApiManager::new(auth_config, RateLimitConfig::default());

        assert!(!manager.validate_auth("").await.unwrap());
        assert!(!manager.validate_auth("Bearer ").await.unwrap());
    }
}
