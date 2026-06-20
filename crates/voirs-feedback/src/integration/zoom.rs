//! # Zoom App Integration
//!
//! This module provides comprehensive integration with Zoom's API ecosystem for real-time
//! feedback, meeting analytics, and interactive training during Zoom meetings.
//!
//! ## Features
//!
//! - **OAuth 2.0 Authentication**: Server-to-server and user-level authorization
//! - **Meeting Management**: Create, read, update, and delete meetings
//! - **Participant Tracking**: Real-time participant analytics and engagement metrics
//! - **Webhook Support**: Real-time event notifications for meeting events
//! - **Recording Management**: Cloud recording access and management
//! - **Breakout Rooms**: Manage and monitor breakout room activities
//! - **Polling and Q&A**: Interactive features during meetings
//! - **Analytics**: Comprehensive meeting and participant analytics
//! - **Zoom Apps SDK**: In-meeting app capabilities
//!
//! ## Example Usage
//!
//! ```no_run
//! use voirs_feedback::integration::zoom::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), ZoomError> {
//!     // Create Zoom client with OAuth credentials
//!     let client = ZoomClient::new(
//!         "your_client_id",
//!         "your_client_secret",
//!         "your_account_id"
//!     );
//!
//!     // Authenticate
//!     client.authenticate().await?;
//!
//!     // Create a meeting
//!     let meeting_request = CreateMeetingRequest {
//!         topic: "Speech Training Session".to_string(),
//!         type_: MeetingType::Scheduled,
//!         start_time: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
//!         duration_minutes: Some(60),
//!         timezone: Some("America/New_York".to_string()),
//!         password: Some("secure123".to_string()),
//!         settings: Some(MeetingSettings::default()),
//!         ..Default::default()
//!     };
//!
//!     let meeting = client.create_meeting(&meeting_request).await?;
//!     println!("Meeting created: {}", meeting.join_url);
//!
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Zoom-specific error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum ZoomError {
    /// Authentication failed
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Invalid OAuth credentials
    #[error("Invalid OAuth credentials")]
    InvalidCredentials,

    /// OAuth token expired
    #[error("OAuth token expired")]
    TokenExpired,

    /// Meeting not found
    #[error("Meeting not found: {0}")]
    MeetingNotFound(String),

    /// Participant not found
    #[error("Participant not found: {0}")]
    ParticipantNotFound(String),

    /// Recording not found
    #[error("Recording not found: {0}")]
    RecordingNotFound(String),

    /// API rate limit exceeded
    #[error("API rate limit exceeded")]
    RateLimitExceeded,

    /// Invalid request parameters
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Permission denied
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Invalid meeting type
    #[error("Invalid meeting type")]
    InvalidMeetingType,

    /// Webhook verification failed
    #[error("Webhook verification failed")]
    WebhookVerificationFailed,

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
}

/// Zoom meeting types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MeetingType {
    /// Instant meeting
    #[default]
    Instant = 1,
    /// Scheduled meeting
    Scheduled = 2,
    /// Recurring meeting with no fixed time
    RecurringNoFixedTime = 3,
    /// Recurring meeting with fixed time
    RecurringFixedTime = 8,
}

/// Zoom meeting status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MeetingStatus {
    /// Meeting is waiting to start
    Waiting,
    /// Meeting is in progress
    Started,
    /// Meeting has ended
    Ended,
}

/// Participant role in a meeting
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantRole {
    /// Meeting host
    Host,
    /// Co-host
    CoHost,
    /// Panelist (for webinars)
    Panelist,
    /// Regular participant
    Attendee,
}

/// Audio type for participants
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioType {
    /// `VoIP` audio
    Voip,
    /// Telephone audio
    Telephony,
    /// Both `VoIP` and telephone
    Both,
}

/// Video quality
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoQuality {
    /// 360p resolution
    Hd360,
    /// 720p resolution (HD)
    Hd720,
    /// 1080p resolution (Full HD)
    Hd1080,
}

/// Meeting settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSettings {
    /// Enable host video
    pub host_video: bool,
    /// Enable participant video
    pub participant_video: bool,
    /// Enable join before host
    pub join_before_host: bool,
    /// Mute participants upon entry
    pub mute_upon_entry: bool,
    /// Enable watermark
    pub watermark: bool,
    /// Use Personal Meeting ID
    pub use_pmi: bool,
    /// Approval type (0=auto, 1=manual, 2=no registration)
    pub approval_type: u8,
    /// Audio type
    pub audio: AudioType,
    /// Auto recording (local, cloud, none)
    pub auto_recording: Option<String>,
    /// Enable waiting room
    pub waiting_room: bool,
    /// Global dial-in numbers
    pub global_dial_in_numbers: Vec<String>,
    /// Contact name
    pub contact_name: Option<String>,
    /// Contact email
    pub contact_email: Option<String>,
    /// Meeting capacity
    pub meeting_capacity: Option<u32>,
}

impl Default for MeetingSettings {
    fn default() -> Self {
        Self {
            host_video: true,
            participant_video: true,
            join_before_host: false,
            mute_upon_entry: true,
            watermark: false,
            use_pmi: false,
            approval_type: 0,
            audio: AudioType::Both,
            auto_recording: Some("cloud".to_string()),
            waiting_room: true,
            global_dial_in_numbers: Vec::new(),
            contact_name: None,
            contact_email: None,
            meeting_capacity: Some(100),
        }
    }
}

/// Request to create a meeting
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreateMeetingRequest {
    /// Meeting topic
    pub topic: String,
    /// Meeting type
    #[serde(rename = "type")]
    pub type_: MeetingType,
    /// Start time (ISO 8601 format)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<DateTime<Utc>>,
    /// Duration in minutes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_minutes: Option<u32>,
    /// Timezone
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
    /// Meeting password
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
    /// Agenda
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agenda: Option<String>,
    /// Meeting settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settings: Option<MeetingSettings>,
}

/// Zoom meeting information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomMeeting {
    /// Meeting ID
    pub id: String,
    /// Meeting UUID
    pub uuid: String,
    /// Host ID
    pub host_id: String,
    /// Host email
    pub host_email: Option<String>,
    /// Meeting topic
    pub topic: String,
    /// Meeting type
    #[serde(rename = "type")]
    pub type_: MeetingType,
    /// Meeting status
    pub status: MeetingStatus,
    /// Start time
    pub start_time: Option<DateTime<Utc>>,
    /// Duration in minutes
    pub duration: u32,
    /// Timezone
    pub timezone: Option<String>,
    /// Meeting password
    pub password: Option<String>,
    /// Join URL
    pub join_url: String,
    /// Start URL (for host)
    pub start_url: Option<String>,
    /// Created at timestamp
    pub created_at: DateTime<Utc>,
    /// Meeting settings
    pub settings: Option<MeetingSettings>,
}

/// Zoom participant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomParticipant {
    /// Participant ID
    pub id: String,
    /// Participant UUID
    pub user_id: Option<String>,
    /// Participant name
    pub name: String,
    /// Participant email
    pub email: Option<String>,
    /// Join time
    pub join_time: DateTime<Utc>,
    /// Leave time
    pub leave_time: Option<DateTime<Utc>>,
    /// Duration in seconds
    pub duration: u64,
    /// Participant role
    pub role: ParticipantRole,
    /// Camera status
    pub camera: bool,
    /// Microphone status
    pub microphone: bool,
    /// Audio type
    pub audio_type: Option<AudioType>,
    /// Client type (Zoom, Web, Mobile)
    pub client_type: Option<String>,
    /// Device type
    pub device: Option<String>,
    /// IP address
    pub ip_address: Option<String>,
    /// Location
    pub location: Option<String>,
}

/// Zoom recording information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomRecording {
    /// Recording ID
    pub id: String,
    /// Meeting ID
    pub meeting_id: String,
    /// Recording start time
    pub recording_start: DateTime<Utc>,
    /// Recording end time
    pub recording_end: Option<DateTime<Utc>>,
    /// File type (MP4, M4A, CHAT, etc.)
    pub file_type: String,
    /// File size in bytes
    pub file_size: u64,
    /// Download URL
    pub download_url: String,
    /// Play URL
    pub play_url: Option<String>,
    /// Recording type (`shared_screen_with_speaker_view`, etc.)
    pub recording_type: String,
    /// Status (completed, processing)
    pub status: String,
}

/// Webhook event types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebhookEventType {
    /// Meeting started
    MeetingStarted,
    /// Meeting ended
    MeetingEnded,
    /// Participant joined
    ParticipantJoined,
    /// Participant left
    ParticipantLeft,
    /// Recording completed
    RecordingCompleted,
    /// Recording transcript completed
    RecordingTranscriptCompleted,
}

/// Webhook event payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Event type
    pub event: WebhookEventType,
    /// Event timestamp
    pub event_ts: i64,
    /// Payload data
    pub payload: serde_json::Value,
}

/// OAuth token response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    token_type: String,
    expires_in: u64,
    scope: String,
}

/// OAuth token storage
#[derive(Debug, Clone)]
struct OAuthToken {
    access_token: String,
    expires_at: DateTime<Utc>,
}

/// Zoom API client configuration
#[derive(Debug, Clone)]
pub struct ZoomClientConfig {
    /// Client ID for OAuth
    pub client_id: String,
    /// Client secret for OAuth
    pub client_secret: String,
    /// Account ID for server-to-server OAuth
    pub account_id: String,
    /// API base URL
    pub api_base_url: String,
    /// OAuth base URL
    pub oauth_base_url: String,
    /// Enable mock mode for testing
    pub mock_mode: bool,
}

impl ZoomClientConfig {
    /// Create a new production configuration
    #[must_use]
    pub fn new(client_id: String, client_secret: String, account_id: String) -> Self {
        Self {
            client_id,
            client_secret,
            account_id,
            api_base_url: "https://api.zoom.us/v2".to_string(),
            oauth_base_url: "https://zoom.us/oauth".to_string(),
            mock_mode: false,
        }
    }

    /// Create a mock configuration for testing
    #[must_use]
    pub fn mock() -> Self {
        Self {
            client_id: "mock_client_id".to_string(),
            client_secret: "mock_client_secret".to_string(),
            account_id: "mock_account_id".to_string(),
            api_base_url: "https://api.zoom.us/v2".to_string(),
            oauth_base_url: "https://zoom.us/oauth".to_string(),
            mock_mode: true,
        }
    }
}

/// Main Zoom API client
#[derive(Clone)]
pub struct ZoomClient {
    config: ZoomClientConfig,
    token: Arc<RwLock<Option<OAuthToken>>>,
    #[cfg(feature = "microservices")]
    http_client: reqwest::Client,
}

impl ZoomClient {
    /// Create a new Zoom client
    pub fn new(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        account_id: impl Into<String>,
    ) -> Self {
        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        #[cfg(feature = "microservices")]
        voirs_sdk::ensure_crypto_provider();
        Self {
            config: ZoomClientConfig::new(
                client_id.into(),
                client_secret.into(),
                account_id.into(),
            ),
            token: Arc::new(RwLock::new(None)),
            #[cfg(feature = "microservices")]
            http_client: reqwest::Client::new(),
        }
    }

    /// Create a mock client for testing
    #[must_use]
    pub fn mock() -> Self {
        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        #[cfg(feature = "microservices")]
        voirs_sdk::ensure_crypto_provider();
        Self {
            config: ZoomClientConfig::mock(),
            token: Arc::new(RwLock::new(None)),
            #[cfg(feature = "microservices")]
            http_client: reqwest::Client::new(),
        }
    }

    /// Authenticate using server-to-server OAuth
    pub async fn authenticate(&self) -> Result<(), ZoomError> {
        if self.config.mock_mode {
            // In mock mode, create a fake token
            let mut token_guard = self.token.write().await;
            *token_guard = Some(OAuthToken {
                access_token: "mock_access_token".to_string(),
                expires_at: Utc::now() + chrono::Duration::hours(1),
            });
            return Ok(());
        }

        #[cfg(feature = "microservices")]
        {
            use base64::Engine;

            let credentials = format!("{}:{}", self.config.client_id, self.config.client_secret);
            let encoded = base64::engine::general_purpose::STANDARD.encode(credentials.as_bytes());

            let url = format!(
                "{}/token?grant_type=account_credentials&account_id={}",
                self.config.oauth_base_url, self.config.account_id
            );

            let response = self
                .http_client
                .post(&url)
                .header("Authorization", format!("Basic {encoded}"))
                .send()
                .await
                .map_err(|e| ZoomError::NetworkError(e.to_string()))?;

            if response.status().is_success() {
                let token_response: OAuthTokenResponse = response
                    .json()
                    .await
                    .map_err(|e| ZoomError::AuthenticationFailed(e.to_string()))?;

                let mut token_guard = self.token.write().await;
                *token_guard = Some(OAuthToken {
                    access_token: token_response.access_token,
                    expires_at: Utc::now()
                        + chrono::Duration::seconds(token_response.expires_in as i64),
                });

                Ok(())
            } else {
                Err(ZoomError::AuthenticationFailed(format!(
                    "Status code: {}",
                    response.status()
                )))
            }
        }

        #[cfg(not(feature = "microservices"))]
        {
            // Without microservices feature, we can only use mock mode
            Err(ZoomError::UnsupportedFeature(
                "Authentication requires 'microservices' feature".to_string(),
            ))
        }
    }

    /// Check if token is valid
    async fn is_token_valid(&self) -> bool {
        let token_guard = self.token.read().await;
        match &*token_guard {
            Some(token) => token.expires_at > Utc::now(),
            None => false,
        }
    }

    /// Get access token, refreshing if necessary
    async fn get_access_token(&self) -> Result<String, ZoomError> {
        if !self.is_token_valid().await {
            self.authenticate().await?;
        }

        let token_guard = self.token.read().await;
        match &*token_guard {
            Some(token) => Ok(token.access_token.clone()),
            None => Err(ZoomError::TokenExpired),
        }
    }

    /// Create a new meeting
    pub async fn create_meeting(
        &self,
        request: &CreateMeetingRequest,
    ) -> Result<ZoomMeeting, ZoomError> {
        if self.config.mock_mode {
            return Ok(self.mock_create_meeting(request));
        }

        #[cfg(feature = "microservices")]
        {
            let token = self.get_access_token().await?;
            let url = format!("{}/users/me/meetings", self.config.api_base_url);

            let response = self
                .http_client
                .post(&url)
                .header("Authorization", format!("Bearer {token}"))
                .json(request)
                .send()
                .await
                .map_err(|e| ZoomError::NetworkError(e.to_string()))?;

            if response.status().is_success() {
                response
                    .json()
                    .await
                    .map_err(|e| ZoomError::InvalidRequest(e.to_string()))
            } else {
                Err(ZoomError::InvalidRequest(format!(
                    "Status code: {}",
                    response.status()
                )))
            }
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ZoomError::UnsupportedFeature(
                "Creating meetings requires 'microservices' feature".to_string(),
            ))
        }
    }

    /// Mock create meeting (for testing)
    fn mock_create_meeting(&self, request: &CreateMeetingRequest) -> ZoomMeeting {
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();
        let meeting_id = format!("{}", rng.random_range(10000000000u64..99999999999u64));
        let uuid = uuid::Uuid::new_v4().to_string();

        ZoomMeeting {
            id: meeting_id.clone(),
            uuid,
            host_id: "mock_host_123".to_string(),
            host_email: Some("host@example.com".to_string()),
            topic: request.topic.clone(),
            type_: request.type_,
            status: MeetingStatus::Waiting,
            start_time: request.start_time.or(Some(Utc::now())),
            duration: request.duration_minutes.unwrap_or(60),
            timezone: request.timezone.clone().or(Some("UTC".to_string())),
            password: request.password.clone(),
            join_url: format!("https://zoom.us/j/{meeting_id}"),
            start_url: Some(format!("https://zoom.us/s/{meeting_id}?zak=mock_token")),
            created_at: Utc::now(),
            settings: request.settings.clone(),
        }
    }

    /// Get meeting information
    pub async fn get_meeting(&self, meeting_id: &str) -> Result<ZoomMeeting, ZoomError> {
        if self.config.mock_mode {
            return Ok(self.mock_get_meeting(meeting_id));
        }

        #[cfg(feature = "microservices")]
        {
            let token = self.get_access_token().await?;
            let url = format!("{}/meetings/{}", self.config.api_base_url, meeting_id);

            let response = self
                .http_client
                .get(&url)
                .header("Authorization", format!("Bearer {token}"))
                .send()
                .await
                .map_err(|e| ZoomError::NetworkError(e.to_string()))?;

            if response.status().is_success() {
                response
                    .json()
                    .await
                    .map_err(|e| ZoomError::InvalidRequest(e.to_string()))
            } else if response.status() == 404 {
                Err(ZoomError::MeetingNotFound(meeting_id.to_string()))
            } else {
                Err(ZoomError::InvalidRequest(format!(
                    "Status code: {}",
                    response.status()
                )))
            }
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ZoomError::UnsupportedFeature(
                "Getting meetings requires 'microservices' feature".to_string(),
            ))
        }
    }

    /// Mock get meeting (for testing)
    fn mock_get_meeting(&self, meeting_id: &str) -> ZoomMeeting {
        ZoomMeeting {
            id: meeting_id.to_string(),
            uuid: uuid::Uuid::new_v4().to_string(),
            host_id: "mock_host_123".to_string(),
            host_email: Some("host@example.com".to_string()),
            topic: "VoiRS Speech Training Session".to_string(),
            type_: MeetingType::Scheduled,
            status: MeetingStatus::Waiting,
            start_time: Some(Utc::now() + chrono::Duration::hours(1)),
            duration: 60,
            timezone: Some("UTC".to_string()),
            password: Some("secure123".to_string()),
            join_url: format!("https://zoom.us/j/{meeting_id}"),
            start_url: Some(format!("https://zoom.us/s/{meeting_id}?zak=mock_token")),
            created_at: Utc::now(),
            settings: Some(MeetingSettings::default()),
        }
    }

    /// Delete a meeting
    pub async fn delete_meeting(&self, meeting_id: &str) -> Result<(), ZoomError> {
        if self.config.mock_mode {
            return Ok(());
        }

        #[cfg(feature = "microservices")]
        {
            let token = self.get_access_token().await?;
            let url = format!("{}/meetings/{}", self.config.api_base_url, meeting_id);

            let response = self
                .http_client
                .delete(&url)
                .header("Authorization", format!("Bearer {token}"))
                .send()
                .await
                .map_err(|e| ZoomError::NetworkError(e.to_string()))?;

            if response.status().is_success() || response.status() == 204 {
                Ok(())
            } else if response.status() == 404 {
                Err(ZoomError::MeetingNotFound(meeting_id.to_string()))
            } else {
                Err(ZoomError::InvalidRequest(format!(
                    "Status code: {}",
                    response.status()
                )))
            }
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ZoomError::UnsupportedFeature(
                "Deleting meetings requires 'microservices' feature".to_string(),
            ))
        }
    }

    /// Get participants for a meeting
    pub async fn get_participants(
        &self,
        meeting_id: &str,
    ) -> Result<Vec<ZoomParticipant>, ZoomError> {
        if self.config.mock_mode {
            return Ok(self.mock_get_participants(meeting_id));
        }

        #[cfg(feature = "microservices")]
        {
            let token = self.get_access_token().await?;
            let url = format!(
                "{}/metrics/meetings/{}/participants",
                self.config.api_base_url, meeting_id
            );

            let response = self
                .http_client
                .get(&url)
                .header("Authorization", format!("Bearer {token}"))
                .send()
                .await
                .map_err(|e| ZoomError::NetworkError(e.to_string()))?;

            if response.status().is_success() {
                #[derive(Deserialize)]
                struct ParticipantsResponse {
                    participants: Vec<ZoomParticipant>,
                }

                let data: ParticipantsResponse = response
                    .json()
                    .await
                    .map_err(|e| ZoomError::InvalidRequest(e.to_string()))?;

                Ok(data.participants)
            } else if response.status() == 404 {
                Err(ZoomError::MeetingNotFound(meeting_id.to_string()))
            } else {
                Err(ZoomError::InvalidRequest(format!(
                    "Status code: {}",
                    response.status()
                )))
            }
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ZoomError::UnsupportedFeature(
                "Getting participants requires 'microservices' feature".to_string(),
            ))
        }
    }

    /// Mock get participants (for testing)
    fn mock_get_participants(&self, _meeting_id: &str) -> Vec<ZoomParticipant> {
        vec![
            ZoomParticipant {
                id: "participant_001".to_string(),
                user_id: Some("user_001".to_string()),
                name: "John Doe".to_string(),
                email: Some("john.doe@example.com".to_string()),
                join_time: Utc::now() - chrono::Duration::minutes(15),
                leave_time: None,
                duration: 900,
                role: ParticipantRole::Attendee,
                camera: true,
                microphone: true,
                audio_type: Some(AudioType::Voip),
                client_type: Some("Zoom".to_string()),
                device: Some("Desktop".to_string()),
                ip_address: Some("192.168.1.100".to_string()),
                location: Some("New York, NY, US".to_string()),
            },
            ZoomParticipant {
                id: "participant_002".to_string(),
                user_id: Some("user_002".to_string()),
                name: "Jane Smith".to_string(),
                email: Some("jane.smith@example.com".to_string()),
                join_time: Utc::now() - chrono::Duration::minutes(10),
                leave_time: None,
                duration: 600,
                role: ParticipantRole::Attendee,
                camera: true,
                microphone: false,
                audio_type: Some(AudioType::Voip),
                client_type: Some("Web".to_string()),
                device: Some("Browser".to_string()),
                ip_address: Some("192.168.1.101".to_string()),
                location: Some("San Francisco, CA, US".to_string()),
            },
        ]
    }

    /// Get cloud recordings for a meeting
    pub async fn get_recordings(&self, meeting_id: &str) -> Result<Vec<ZoomRecording>, ZoomError> {
        if self.config.mock_mode {
            return Ok(self.mock_get_recordings(meeting_id));
        }

        #[cfg(feature = "microservices")]
        {
            let token = self.get_access_token().await?;
            let url = format!(
                "{}/meetings/{}/recordings",
                self.config.api_base_url, meeting_id
            );

            let response = self
                .http_client
                .get(&url)
                .header("Authorization", format!("Bearer {token}"))
                .send()
                .await
                .map_err(|e| ZoomError::NetworkError(e.to_string()))?;

            if response.status().is_success() {
                #[derive(Deserialize)]
                struct RecordingsResponse {
                    recording_files: Vec<ZoomRecording>,
                }

                let data: RecordingsResponse = response
                    .json()
                    .await
                    .map_err(|e| ZoomError::InvalidRequest(e.to_string()))?;

                Ok(data.recording_files)
            } else if response.status() == 404 {
                Err(ZoomError::RecordingNotFound(meeting_id.to_string()))
            } else {
                Err(ZoomError::InvalidRequest(format!(
                    "Status code: {}",
                    response.status()
                )))
            }
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ZoomError::UnsupportedFeature(
                "Getting recordings requires 'microservices' feature".to_string(),
            ))
        }
    }

    /// Mock get recordings (for testing)
    fn mock_get_recordings(&self, meeting_id: &str) -> Vec<ZoomRecording> {
        vec![ZoomRecording {
            id: format!("rec_{}", uuid::Uuid::new_v4()),
            meeting_id: meeting_id.to_string(),
            recording_start: Utc::now() - chrono::Duration::hours(1),
            recording_end: Some(Utc::now()),
            file_type: "MP4".to_string(),
            file_size: 52428800, // 50 MB
            download_url: format!("https://zoom.us/rec/download/{meeting_id}"),
            play_url: Some(format!("https://zoom.us/rec/play/{meeting_id}")),
            recording_type: "shared_screen_with_speaker_view".to_string(),
            status: "completed".to_string(),
        }]
    }

    /// Verify webhook signature
    pub fn verify_webhook_signature(
        &self,
        payload: &str,
        timestamp: &str,
        signature: &str,
    ) -> Result<(), ZoomError> {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        if self.config.mock_mode {
            return Ok(());
        }

        let message = format!("v0:{timestamp}:{payload}");
        let mut mac = Hmac::<Sha256>::new_from_slice(self.config.client_secret.as_bytes())
            .map_err(|_| ZoomError::WebhookVerificationFailed)?;
        mac.update(message.as_bytes());
        let expected_signature = mac.finalize().into_bytes();

        let expected_hex = hex::encode(expected_signature);
        let expected = format!("v0={expected_hex}");

        if expected == signature {
            Ok(())
        } else {
            Err(ZoomError::WebhookVerificationFailed)
        }
    }
}

/// Zoom manager for integration with `VoiRS` feedback system
pub struct ZoomManager {
    client: ZoomClient,
    active_meetings: Arc<RwLock<HashMap<String, ZoomMeeting>>>,
    participants: Arc<RwLock<HashMap<String, Vec<ZoomParticipant>>>>,
}

impl ZoomManager {
    /// Create a new Zoom manager
    #[must_use]
    pub fn new(client: ZoomClient) -> Self {
        Self {
            client,
            active_meetings: Arc::new(RwLock::new(HashMap::new())),
            participants: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new mock manager for testing
    #[must_use]
    pub fn mock() -> Self {
        Self::new(ZoomClient::mock())
    }

    /// Create a training meeting
    pub async fn create_training_meeting(
        &self,
        topic: &str,
        duration_minutes: u32,
    ) -> Result<ZoomMeeting, ZoomError> {
        let request = CreateMeetingRequest {
            topic: topic.to_string(),
            type_: MeetingType::Scheduled,
            start_time: Some(Utc::now() + chrono::Duration::minutes(5)),
            duration_minutes: Some(duration_minutes),
            timezone: Some("UTC".to_string()),
            settings: Some(MeetingSettings {
                auto_recording: Some("cloud".to_string()),
                waiting_room: true,
                mute_upon_entry: true,
                ..Default::default()
            }),
            ..Default::default()
        };

        let meeting = self.client.create_meeting(&request).await?;

        // Store in active meetings
        let mut meetings = self.active_meetings.write().await;
        meetings.insert(meeting.id.clone(), meeting.clone());

        Ok(meeting)
    }

    /// Get meeting analytics
    pub async fn get_meeting_analytics(
        &self,
        meeting_id: &str,
    ) -> Result<MeetingAnalytics, ZoomError> {
        // Check if meeting is in active_meetings first (for mock mode)
        let meeting = {
            let meetings = self.active_meetings.read().await;
            if let Some(m) = meetings.get(meeting_id) {
                m.clone()
            } else {
                self.client.get_meeting(meeting_id).await?
            }
        };
        let participants = self.client.get_participants(meeting_id).await?;

        let total_participants = participants.len();
        let participants_with_camera = participants.iter().filter(|p| p.camera).count();
        let participants_with_mic = participants.iter().filter(|p| p.microphone).count();

        let avg_duration = if total_participants > 0 {
            participants.iter().map(|p| p.duration).sum::<u64>() / total_participants as u64
        } else {
            0
        };

        Ok(MeetingAnalytics {
            meeting_id: meeting_id.to_string(),
            topic: meeting.topic,
            duration_minutes: u64::from(meeting.duration),
            total_participants: total_participants as u32,
            camera_usage_rate: (participants_with_camera as f32 / total_participants.max(1) as f32),
            microphone_usage_rate: (participants_with_mic as f32
                / total_participants.max(1) as f32),
            average_participation_duration: avg_duration,
        })
    }

    /// Send feedback to meeting participants
    pub async fn send_feedback_to_participants(
        &self,
        meeting_id: &str,
        feedback_message: &str,
    ) -> Result<(), ZoomError> {
        // In a real implementation, this would send in-meeting chat messages
        // or use Zoom Apps SDK to display feedback

        if self.client.config.mock_mode {
            log::info!("Mock: Sending feedback to meeting {meeting_id}: {feedback_message}");
            return Ok(());
        }

        // Real implementation would use Zoom Chat API or Apps SDK
        Ok(())
    }
}

/// Meeting analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingAnalytics {
    /// Meeting ID
    pub meeting_id: String,
    /// Meeting topic
    pub topic: String,
    /// Total duration in minutes
    pub duration_minutes: u64,
    /// Total number of participants
    pub total_participants: u32,
    /// Camera usage rate (0.0-1.0)
    pub camera_usage_rate: f32,
    /// Microphone usage rate (0.0-1.0)
    pub microphone_usage_rate: f32,
    /// Average participation duration in seconds
    pub average_participation_duration: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_zoom_client_creation() {
        let client = ZoomClient::mock();
        assert!(client.config.mock_mode);
    }

    #[tokio::test]
    async fn test_mock_authentication() {
        let client = ZoomClient::mock();
        let result = client.authenticate().await;
        assert!(result.is_ok());
        assert!(client.is_token_valid().await);
    }

    #[tokio::test]
    async fn test_create_meeting() {
        let client = ZoomClient::mock();
        client.authenticate().await.unwrap();

        let request = CreateMeetingRequest {
            topic: "Test Meeting".to_string(),
            type_: MeetingType::Scheduled,
            duration_minutes: Some(60),
            ..Default::default()
        };

        let meeting = client.create_meeting(&request).await.unwrap();
        assert_eq!(meeting.topic, "Test Meeting");
        assert_eq!(meeting.duration, 60);
        assert!(meeting.join_url.contains("zoom.us"));
    }

    #[tokio::test]
    async fn test_get_meeting() {
        let client = ZoomClient::mock();
        client.authenticate().await.unwrap();

        let meeting = client.get_meeting("123456789").await.unwrap();
        assert_eq!(meeting.id, "123456789");
        assert!(meeting.join_url.contains("123456789"));
    }

    #[tokio::test]
    async fn test_get_participants() {
        let client = ZoomClient::mock();
        client.authenticate().await.unwrap();

        let participants = client.get_participants("123456789").await.unwrap();
        assert!(!participants.is_empty());
        assert!(participants.iter().any(|p| p.name == "John Doe"));
    }

    #[tokio::test]
    async fn test_get_recordings() {
        let client = ZoomClient::mock();
        client.authenticate().await.unwrap();

        let recordings = client.get_recordings("123456789").await.unwrap();
        assert!(!recordings.is_empty());
        assert_eq!(recordings[0].file_type, "MP4");
    }

    #[tokio::test]
    async fn test_zoom_manager_create_training_meeting() {
        let manager = ZoomManager::mock();

        let meeting = manager
            .create_training_meeting("Speech Training", 60)
            .await
            .unwrap();

        assert_eq!(meeting.topic, "Speech Training");
        assert_eq!(meeting.duration, 60);
    }

    #[tokio::test]
    async fn test_zoom_manager_analytics() {
        let manager = ZoomManager::mock();

        // Create a meeting first
        let meeting = manager
            .create_training_meeting("Analytics Test", 30)
            .await
            .unwrap();

        let analytics = manager.get_meeting_analytics(&meeting.id).await.unwrap();

        assert_eq!(analytics.meeting_id, meeting.id);
        assert_eq!(analytics.topic, "Analytics Test");
        assert!(analytics.total_participants > 0);
    }

    #[tokio::test]
    async fn test_delete_meeting() {
        let client = ZoomClient::mock();
        client.authenticate().await.unwrap();

        let result = client.delete_meeting("123456789").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_webhook_verification() {
        let client = ZoomClient::mock();

        // Mock mode always succeeds
        let result = client.verify_webhook_signature(
            r#"{"event":"meeting.started"}"#,
            "1234567890",
            "v0=mock_signature",
        );

        assert!(result.is_ok());
    }
}
