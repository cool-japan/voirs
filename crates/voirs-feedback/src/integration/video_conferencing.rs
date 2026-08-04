//! # Video Conferencing Integration
//!
//! This module provides seamless integration with major video conferencing platforms
//! including Zoom, Microsoft Teams, Google Meet, and others. It supports real-time
//! meeting feedback, presentation coaching, meeting analytics, and remote training facilitation.

use crate::realtime::types::RealtimeConfig;
use crate::traits::{FeedbackSession, UserProgress};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::time::{Duration, SystemTime};

#[cfg(feature = "microservices")]
use reqwest::Client;

/// Video conferencing integration error types
#[derive(Debug, Clone)]
pub enum VideoConferencingError {
    /// Authentication failed with message
    AuthenticationFailed(String),
    /// Connection timeout occurred
    ConnectionTimeout,
    /// Invalid API key provided
    InvalidApiKey,
    /// Meeting not found with ID
    MeetingNotFound(String),
    /// Participant not found with ID
    ParticipantNotFound(String),
    /// Permission denied with reason
    PermissionDenied(String),
    /// Network error with message
    NetworkError(String),
    /// Configuration error with message
    ConfigurationError(String),
    /// API rate limit exceeded
    RateLimitExceeded,
    /// Unauthorized access attempt
    UnauthorizedAccess,
    /// Recording failed with message
    RecordingFailed(String),
    /// Plugin installation failed with message
    PluginInstallationFailed(String),
    /// Webhook error with message
    WebhookError(String),
}

impl fmt::Display for VideoConferencingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoConferencingError::AuthenticationFailed(msg) => {
                write!(f, "Authentication failed: {msg}")
            }
            VideoConferencingError::ConnectionTimeout => write!(f, "Connection timeout"),
            VideoConferencingError::InvalidApiKey => write!(f, "Invalid API key"),
            VideoConferencingError::MeetingNotFound(id) => write!(f, "Meeting not found: {id}"),
            VideoConferencingError::ParticipantNotFound(id) => {
                write!(f, "Participant not found: {id}")
            }
            VideoConferencingError::PermissionDenied(msg) => {
                write!(f, "Permission denied: {msg}")
            }
            VideoConferencingError::NetworkError(msg) => write!(f, "Network error: {msg}"),
            VideoConferencingError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {msg}")
            }
            VideoConferencingError::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            VideoConferencingError::UnauthorizedAccess => write!(f, "Unauthorized access"),
            VideoConferencingError::RecordingFailed(msg) => write!(f, "Recording failed: {msg}"),
            VideoConferencingError::PluginInstallationFailed(msg) => {
                write!(f, "Plugin installation failed: {msg}")
            }
            VideoConferencingError::WebhookError(msg) => write!(f, "Webhook error: {msg}"),
        }
    }
}

impl Error for VideoConferencingError {}

/// Supported video conferencing platforms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoConferencingPlatform {
    /// Zoom platform
    Zoom,
    /// Microsoft Teams platform
    MicrosoftTeams,
    /// Google Meet platform
    GoogleMeet,
    /// Cisco `WebEx` platform
    WebEx,
    /// `GoToMeeting` platform
    GoToMeeting,
    /// `BlueJeans` platform
    BlueJeans,
    /// Jitsi Meet platform
    Jitsi,
    /// Skype platform
    Skype,
    /// Custom platform with name
    Custom(String),
}

impl fmt::Display for VideoConferencingPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoConferencingPlatform::Zoom => write!(f, "Zoom"),
            VideoConferencingPlatform::MicrosoftTeams => write!(f, "Microsoft Teams"),
            VideoConferencingPlatform::GoogleMeet => write!(f, "Google Meet"),
            VideoConferencingPlatform::WebEx => write!(f, "Cisco WebEx"),
            VideoConferencingPlatform::GoToMeeting => write!(f, "GoToMeeting"),
            VideoConferencingPlatform::BlueJeans => write!(f, "BlueJeans"),
            VideoConferencingPlatform::Jitsi => write!(f, "Jitsi Meet"),
            VideoConferencingPlatform::Skype => write!(f, "Skype"),
            VideoConferencingPlatform::Custom(name) => write!(f, "Custom: {name}"),
        }
    }
}

/// Video conferencing authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConferencingAuthConfig {
    /// Target platform
    pub platform: VideoConferencingPlatform,
    /// API key for authentication
    pub api_key: String,
    /// API secret for authentication
    pub api_secret: Option<String>,
    /// Base URL for API requests
    pub base_url: String,
    /// OAuth client ID
    pub oauth_client_id: Option<String>,
    /// OAuth client secret
    pub oauth_client_secret: Option<String>,
    /// Zoom Server-to-Server OAuth account ID (required for the
    /// `account_credentials` grant type).
    pub account_id: Option<String>,
    /// Microsoft Entra ID (Azure AD) tenant ID, required for the Microsoft
    /// identity platform's client-credentials token endpoint.
    pub tenant_id: Option<String>,
    /// Webhook callback URL
    pub webhook_url: Option<String>,
    /// Webhook secret for verification
    pub webhook_secret: Option<String>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Enable meeting recording
    pub enable_recording: bool,
    /// Enable real-time feedback
    pub enable_real_time_feedback: bool,
}

impl Default for VideoConferencingAuthConfig {
    fn default() -> Self {
        Self {
            platform: VideoConferencingPlatform::Zoom,
            api_key: String::new(),
            api_secret: None,
            base_url: String::new(),
            oauth_client_id: None,
            oauth_client_secret: None,
            account_id: None,
            tenant_id: None,
            webhook_url: None,
            webhook_secret: None,
            timeout_seconds: 30,
            enable_recording: false,
            enable_real_time_feedback: true,
        }
    }
}

/// Meeting information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingInfo {
    /// Meeting identifier
    pub meeting_id: String,
    /// Meeting UUID
    pub meeting_uuid: Option<String>,
    /// Meeting topic
    pub topic: String,
    /// Meeting start time
    pub start_time: SystemTime,
    /// Meeting duration in minutes
    pub duration_minutes: u32,
    /// Host user ID
    pub host_id: String,
    /// Host display name
    pub host_name: String,
    /// List of participants
    pub participants: Vec<MeetingParticipant>,
    /// Current meeting status
    pub status: MeetingStatus,
    /// Meeting join URL
    pub meeting_url: String,
    /// Whether recording is enabled
    pub recording_enabled: bool,
}

/// Meeting participant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingParticipant {
    /// Participant identifier
    pub participant_id: String,
    /// User identifier
    pub user_id: Option<String>,
    /// Participant display name
    pub name: String,
    /// Participant email address
    pub email: Option<String>,
    /// Time participant joined
    pub join_time: SystemTime,
    /// Time participant left
    pub leave_time: Option<SystemTime>,
    /// Duration in meeting in seconds
    pub duration_seconds: u32,
    /// Whether camera is on
    pub camera_on: bool,
    /// Whether microphone is on
    pub microphone_on: bool,
    /// Participant role
    pub role: ParticipantRole,
    /// Speech analytics data
    pub speech_analytics: Option<SpeechAnalytics>,
}

/// Meeting status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeetingStatus {
    /// Meeting is scheduled
    Scheduled,
    /// Meeting is in progress
    InProgress,
    /// Meeting has ended
    Ended,
    /// Meeting was cancelled
    Cancelled,
}

/// Participant role in the meeting
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParticipantRole {
    /// Meeting host
    Host,
    /// Co-host
    CoHost,
    /// Regular participant
    Participant,
    /// Panelist
    Panelist,
    /// Attendee
    Attendee,
}

/// Speech analytics for meeting participants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechAnalytics {
    /// Total speaking time in seconds
    pub total_speaking_time_seconds: u32,
    /// Average volume level
    pub average_volume_level: f32,
    /// Speech clarity score
    pub speech_clarity_score: f32,
    /// Speech pace score
    pub pace_score: f32,
    /// Confidence score
    pub confidence_score: f32,
    /// Number of filler words
    pub filler_words_count: u32,
    /// Number of interruptions
    pub interruptions_count: u32,
    /// Engagement score
    pub engagement_score: f32,
    /// Sentiment score
    pub sentiment_score: f32,
}

/// Real-time feedback for video conferencing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeMeetingFeedback {
    /// Participant receiving feedback
    pub participant_id: String,
    /// Feedback timestamp
    pub timestamp: SystemTime,
    /// Type of feedback
    pub feedback_type: FeedbackType,
    /// Feedback message
    pub message: String,
    /// Feedback score
    pub score: f32,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
    /// Feedback urgency level
    pub urgency: FeedbackUrgency,
}

/// Types of real-time feedback
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Volume level feedback
    VolumeLevel,
    /// Speech clarity feedback
    SpeechClarity,
    /// Speech pace feedback
    SpeechPace,
    /// Filler words feedback
    FillerWords,
    /// Background noise feedback
    BackgroundNoise,
    /// Camera position feedback
    CameraPosition,
    /// Engagement feedback
    Engagement,
    /// Interruption feedback
    Interruption,
    /// Turn-taking feedback
    TurnTaking,
}

/// Urgency level of feedback
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeedbackUrgency {
    /// Low urgency
    Low,
    /// Medium urgency
    Medium,
    /// High urgency
    High,
    /// Critical urgency
    Critical,
}

/// Meeting analytics and insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingAnalytics {
    /// Meeting identifier
    pub meeting_id: String,
    /// Total meeting duration in minutes
    pub total_duration_minutes: u32,
    /// Total number of participants
    pub total_participants: u32,
    /// Average participation time
    pub average_participation_time: f32,
    /// Speaker distribution by participant ID
    pub speaker_distribution: HashMap<String, f32>,
    /// Engagement metrics
    pub engagement_metrics: EngagementMetrics,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Interaction patterns
    pub interaction_patterns: InteractionPatterns,
    /// Analytics generation timestamp
    pub generated_at: SystemTime,
}

/// Engagement metrics for the meeting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementMetrics {
    /// Overall engagement score
    pub overall_engagement_score: f32,
    /// Percentage with camera on
    pub camera_on_percentage: f32,
    /// Percentage using microphone
    pub microphone_usage_percentage: f32,
    /// Percentage of active speakers
    pub active_speakers_percentage: f32,
    /// Number of questions asked
    pub question_count: u32,
    /// Number of interactions
    pub interaction_count: u32,
}

/// Quality metrics for the meeting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Average audio quality
    pub average_audio_quality: f32,
    /// Average video quality
    pub average_video_quality: f32,
    /// Connection stability score
    pub connection_stability_score: f32,
    /// Number of technical issues
    pub technical_issues_count: u32,
    /// Background noise level
    pub background_noise_level: f32,
}

/// Interaction patterns in the meeting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPatterns {
    /// Turn-taking efficiency score
    pub turn_taking_efficiency: f32,
    /// Interruption rate
    pub interruption_rate: f32,
    /// Percentage of simultaneous speech
    pub simultaneous_speech_percentage: f32,
    /// Number of silence periods
    pub silence_periods_count: u32,
    /// Average response time in seconds
    pub average_response_time_seconds: f32,
}

/// Plugin configuration for video conferencing integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin name
    pub plugin_name: String,
    /// Plugin version
    pub version: String,
    /// Whether to auto-install
    pub auto_install: bool,
    /// Required permissions
    pub permissions: Vec<PluginPermission>,
    /// Plugin settings
    pub settings: HashMap<String, String>,
}

/// Plugin permissions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginPermission {
    /// Audio access permission
    AudioAccess,
    /// Video access permission
    VideoAccess,
    /// Screen sharing permission
    ScreenShare,
    /// Chat access permission
    ChatAccess,
    /// Participant data permission
    ParticipantData,
    /// Recording permission
    Recording,
    /// Notifications permission
    Notifications,
}

/// Video conferencing integration manager
pub struct VideoConferencingIntegrationManager {
    /// Authentication configuration
    config: VideoConferencingAuthConfig,
    /// Real-time configuration
    realtime_config: RealtimeConfig,
    /// API rate limiter
    rate_limiter: VideoConferencingRateLimiter,
    /// Active meeting sessions
    active_sessions: HashMap<String, ActiveMeetingSession>,
    /// HTTP client used for real platform API requests
    #[cfg(feature = "microservices")]
    http_client: Client,
}

/// Active meeting session with real-time feedback
struct ActiveMeetingSession {
    /// Meeting identifier
    meeting_id: String,
    /// Participants in the meeting
    participants: HashMap<String, MeetingParticipant>,
    /// Session start time
    start_time: SystemTime,
    /// Feedback buffer
    feedback_buffer: Vec<RealtimeMeetingFeedback>,
    /// Meeting analytics
    analytics: MeetingAnalytics,
}

impl VideoConferencingIntegrationManager {
    /// Create a new video conferencing integration manager
    #[must_use]
    pub fn new(config: VideoConferencingAuthConfig, realtime_config: RealtimeConfig) -> Self {
        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        #[cfg(feature = "microservices")]
        voirs_sdk::ensure_crypto_provider();

        Self {
            config,
            realtime_config,
            rate_limiter: VideoConferencingRateLimiter::new(100, Duration::from_secs(60)),
            active_sessions: HashMap::new(),
            #[cfg(feature = "microservices")]
            http_client: Client::new(),
        }
    }

    /// Authenticate with the video conferencing platform
    pub async fn authenticate(&mut self) -> Result<(), VideoConferencingError> {
        self.rate_limiter.check_rate_limit()?;

        match self.config.platform {
            VideoConferencingPlatform::Zoom => self.authenticate_zoom().await,
            VideoConferencingPlatform::MicrosoftTeams => self.authenticate_teams().await,
            VideoConferencingPlatform::GoogleMeet => self.authenticate_meet().await,
            VideoConferencingPlatform::WebEx => self.authenticate_webex().await,
            VideoConferencingPlatform::GoToMeeting => self.authenticate_gotomeeting().await,
            VideoConferencingPlatform::BlueJeans => self.authenticate_bluejeans().await,
            VideoConferencingPlatform::Jitsi => self.authenticate_jitsi().await,
            VideoConferencingPlatform::Skype => self.authenticate_skype().await,
            VideoConferencingPlatform::Custom(_) => self.authenticate_custom().await,
        }
    }

    /// Install plugin for the video conferencing platform
    pub async fn install_plugin(
        &mut self,
        plugin_config: &PluginConfig,
    ) -> Result<(), VideoConferencingError> {
        self.rate_limiter.check_rate_limit()?;

        match self.config.platform {
            VideoConferencingPlatform::Zoom => self.install_zoom_plugin(plugin_config).await,
            VideoConferencingPlatform::MicrosoftTeams => {
                self.install_teams_plugin(plugin_config).await
            }
            VideoConferencingPlatform::GoogleMeet => self.install_meet_plugin(plugin_config).await,
            _ => Err(VideoConferencingError::ConfigurationError(
                "Plugin not supported for this platform".to_string(),
            )),
        }
    }

    /// Get meeting information
    pub async fn get_meeting(
        &mut self,
        meeting_id: &str,
    ) -> Result<MeetingInfo, VideoConferencingError> {
        self.rate_limiter.check_rate_limit()?;

        match self.config.platform {
            VideoConferencingPlatform::Zoom => self.get_zoom_meeting(meeting_id).await,
            VideoConferencingPlatform::MicrosoftTeams => self.get_teams_meeting(meeting_id).await,
            VideoConferencingPlatform::GoogleMeet => self.get_meet_meeting(meeting_id).await,
            _ => Err(VideoConferencingError::ConfigurationError(
                "Platform not supported yet".to_string(),
            )),
        }
    }

    /// Start real-time feedback session for a meeting
    pub async fn start_realtime_feedback(
        &mut self,
        meeting_id: &str,
    ) -> Result<(), VideoConferencingError> {
        let meeting_info = self.get_meeting(meeting_id).await?;

        let session = ActiveMeetingSession {
            meeting_id: meeting_id.to_string(),
            participants: meeting_info
                .participants
                .into_iter()
                .map(|p| (p.participant_id.clone(), p))
                .collect(),
            start_time: SystemTime::now(),
            feedback_buffer: Vec::new(),
            analytics: MeetingAnalytics {
                meeting_id: meeting_id.to_string(),
                total_duration_minutes: 0,
                total_participants: 0,
                average_participation_time: 0.0,
                speaker_distribution: HashMap::new(),
                engagement_metrics: EngagementMetrics {
                    overall_engagement_score: 0.0,
                    camera_on_percentage: 0.0,
                    microphone_usage_percentage: 0.0,
                    active_speakers_percentage: 0.0,
                    question_count: 0,
                    interaction_count: 0,
                },
                quality_metrics: QualityMetrics {
                    average_audio_quality: 0.0,
                    average_video_quality: 0.0,
                    connection_stability_score: 0.0,
                    technical_issues_count: 0,
                    background_noise_level: 0.0,
                },
                interaction_patterns: InteractionPatterns {
                    turn_taking_efficiency: 0.0,
                    interruption_rate: 0.0,
                    simultaneous_speech_percentage: 0.0,
                    silence_periods_count: 0,
                    average_response_time_seconds: 0.0,
                },
                generated_at: SystemTime::now(),
            },
        };

        self.active_sessions.insert(meeting_id.to_string(), session);
        Ok(())
    }

    /// Process real-time audio for feedback generation
    pub async fn process_realtime_audio(
        &mut self,
        meeting_id: &str,
        participant_id: &str,
        audio_data: &[f32],
    ) -> Result<Option<RealtimeMeetingFeedback>, VideoConferencingError> {
        // Check if session exists first
        if !self.active_sessions.contains_key(meeting_id) {
            return Err(VideoConferencingError::MeetingNotFound(
                meeting_id.to_string(),
            ));
        }

        // Analyze audio for real-time feedback (this doesn't need mutable self)
        let feedback = Self::analyze_audio_for_feedback_static(participant_id, audio_data).await?;

        // Determine if we need to send realtime feedback
        let should_send_feedback = if let Some(ref feedback_item) = feedback {
            feedback_item.urgency == FeedbackUrgency::High
                || feedback_item.urgency == FeedbackUrgency::Critical
        } else {
            false
        };

        // Update the session
        if let Some(session) = self.active_sessions.get_mut(meeting_id) {
            if let Some(ref feedback_item) = feedback {
                session.feedback_buffer.push(feedback_item.clone());
            }
        } // Borrow of self.active_sessions ends here

        // Send feedback if urgency is high or critical (after releasing the borrow)
        if should_send_feedback {
            if let Some(ref feedback_item) = feedback {
                let meeting_id_owned = meeting_id.to_string();
                self.send_realtime_feedback(&meeting_id_owned, feedback_item)
                    .await?;
            }
        }

        Ok(feedback)
    }

    /// End real-time feedback session and generate analytics
    pub async fn end_realtime_feedback(
        &mut self,
        meeting_id: &str,
    ) -> Result<MeetingAnalytics, VideoConferencingError> {
        if let Some(mut session) = self.active_sessions.remove(meeting_id) {
            // Finalize analytics
            session.analytics.total_duration_minutes =
                session.start_time.elapsed().unwrap_or_default().as_secs() as u32 / 60;
            session.analytics.total_participants = session.participants.len() as u32;

            // Calculate engagement metrics
            self.calculate_engagement_metrics(&mut session.analytics, &session.participants);

            // Generate meeting report
            self.generate_meeting_report(&session.analytics).await?;

            Ok(session.analytics)
        } else {
            Err(VideoConferencingError::MeetingNotFound(
                meeting_id.to_string(),
            ))
        }
    }

    /// Get meeting analytics
    pub async fn get_meeting_analytics(
        &self,
        meeting_id: &str,
    ) -> Result<MeetingAnalytics, VideoConferencingError> {
        if let Some(session) = self.active_sessions.get(meeting_id) {
            Ok(session.analytics.clone())
        } else {
            // Fetch historical analytics from platform
            match self.config.platform {
                VideoConferencingPlatform::Zoom => self.get_zoom_analytics(meeting_id).await,
                VideoConferencingPlatform::MicrosoftTeams => {
                    self.get_teams_analytics(meeting_id).await
                }
                VideoConferencingPlatform::GoogleMeet => self.get_meet_analytics(meeting_id).await,
                _ => Err(VideoConferencingError::ConfigurationError(
                    "Analytics not supported for this platform".to_string(),
                )),
            }
        }
    }

    // Platform-specific authentication methods
    #[cfg(feature = "microservices")]
    async fn authenticate_zoom(&self) -> Result<(), VideoConferencingError> {
        self.get_zoom_access_token().await.map(|_token| ())
    }

    #[cfg(not(feature = "microservices"))]
    async fn authenticate_zoom(&self) -> Result<(), VideoConferencingError> {
        Err(vc_feature_disabled_error())
    }

    /// Exchange the configured Server-to-Server OAuth credentials for a
    /// real Zoom access token via the `account_credentials` grant.
    ///
    /// <https://developers.zoom.us/docs/internal-apps/s2s-oauth/>
    #[cfg(feature = "microservices")]
    async fn get_zoom_access_token(&self) -> Result<String, VideoConferencingError> {
        let account_id = self
            .config
            .account_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                VideoConferencingError::ConfigurationError(
                    "Zoom account_id is not configured".to_string(),
                )
            })?;
        let client_id = self
            .config
            .oauth_client_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                VideoConferencingError::ConfigurationError(
                    "Zoom oauth_client_id is not configured".to_string(),
                )
            })?;
        let client_secret = self
            .config
            .oauth_client_secret
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                VideoConferencingError::ConfigurationError(
                    "Zoom oauth_client_secret is not configured".to_string(),
                )
            })?;

        let url = format!(
            "https://zoom.us/oauth/token?grant_type=account_credentials&account_id={}",
            urlencoding::encode(account_id)
        );

        let response = self
            .http_client
            .post(&url)
            .basic_auth(client_id, Some(client_secret))
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_vc_reqwest_err)?;

        let response = ensure_vc_success(response).await?;
        let data: serde_json::Value = response.json().await.map_err(|e| {
            VideoConferencingError::AuthenticationFailed(format!(
                "failed to parse Zoom token response: {e}"
            ))
        })?;

        data["access_token"]
            .as_str()
            .map(std::string::ToString::to_string)
            .ok_or_else(|| {
                VideoConferencingError::AuthenticationFailed(
                    "Zoom token response is missing 'access_token'".to_string(),
                )
            })
    }

    #[cfg(feature = "microservices")]
    async fn authenticate_teams(&self) -> Result<(), VideoConferencingError> {
        // Microsoft identity platform: app-only client-credentials grant.
        // NOTE: the client-credentials grant requires a tenant-specific (or
        // "organizations") endpoint; "common" is only valid for
        // user-delegated flows and would always be rejected here.
        let tenant_id = self
            .config
            .tenant_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                VideoConferencingError::ConfigurationError(
                    "Microsoft tenant_id is not configured".to_string(),
                )
            })?;
        let client_id = self
            .config
            .oauth_client_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                VideoConferencingError::ConfigurationError(
                    "Microsoft oauth_client_id is not configured".to_string(),
                )
            })?;
        let client_secret = self
            .config
            .oauth_client_secret
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                VideoConferencingError::ConfigurationError(
                    "Microsoft oauth_client_secret is not configured".to_string(),
                )
            })?;

        let url = format!("https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token");
        let form = [
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("scope", "https://graph.microsoft.com/.default"),
            ("grant_type", "client_credentials"),
        ];

        let response = self
            .http_client
            .post(&url)
            .form(&form)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_vc_reqwest_err)?;

        let response = ensure_vc_success(response).await?;
        let data: serde_json::Value = response.json().await.map_err(|e| {
            VideoConferencingError::AuthenticationFailed(format!(
                "failed to parse Microsoft token response: {e}"
            ))
        })?;

        if data["access_token"].as_str().is_none() {
            return Err(VideoConferencingError::AuthenticationFailed(
                "Microsoft token response is missing 'access_token'".to_string(),
            ));
        }
        Ok(())
    }

    #[cfg(not(feature = "microservices"))]
    async fn authenticate_teams(&self) -> Result<(), VideoConferencingError> {
        Err(vc_feature_disabled_error())
    }

    async fn authenticate_meet(&self) -> Result<(), VideoConferencingError> {
        // Google does not support the plain OAuth2 `client_credentials`
        // grant used by the other platforms; server-to-server access to the
        // Meet REST API requires a signed service-account JWT assertion
        // (RFC 7523), which is not implemented here.
        Err(VideoConferencingError::ConfigurationError(
            "Google Meet server-to-server authentication is not yet implemented (requires \
             service-account JWT assertion, not a client_credentials grant)"
                .to_string(),
        ))
    }

    async fn authenticate_webex(&self) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "Cisco WebEx integration is not yet implemented".to_string(),
        ))
    }

    async fn authenticate_gotomeeting(&self) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "GoToMeeting integration is not yet implemented".to_string(),
        ))
    }

    async fn authenticate_bluejeans(&self) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "BlueJeans integration is not yet implemented".to_string(),
        ))
    }

    async fn authenticate_jitsi(&self) -> Result<(), VideoConferencingError> {
        // Public Jitsi Meet instances (e.g. meet.jit.si) genuinely accept
        // anonymous participants with no authentication step, so there is
        // no request to make here. Self-hosted instances configured to
        // require JWT authentication are not supported by this client.
        Ok(())
    }

    async fn authenticate_skype(&self) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "Skype integration is not yet implemented".to_string(),
        ))
    }

    async fn authenticate_custom(&self) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "custom video conferencing platforms have no built-in client".to_string(),
        ))
    }

    // Plugin installation methods
    //
    // Installing an app/add-on into Zoom Marketplace, the Microsoft Teams
    // App Store, or the Google Workspace Marketplace is a manual submission
    // + review workflow on each platform, not a REST call this client can
    // make on a caller's behalf, so these fail closed rather than claiming
    // an installation that never happened.
    async fn install_zoom_plugin(
        &self,
        _config: &PluginConfig,
    ) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "Zoom app installation requires manual Marketplace submission and is not automatable"
                .to_string(),
        ))
    }

    async fn install_teams_plugin(
        &self,
        _config: &PluginConfig,
    ) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "Teams app installation requires manual App Store submission and is not automatable"
                .to_string(),
        ))
    }

    async fn install_meet_plugin(
        &self,
        _config: &PluginConfig,
    ) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "Google Meet add-on installation requires manual Workspace Marketplace submission \
             and is not automatable"
                .to_string(),
        ))
    }

    // Meeting retrieval methods
    #[cfg(feature = "microservices")]
    async fn get_zoom_meeting(
        &self,
        meeting_id: &str,
    ) -> Result<MeetingInfo, VideoConferencingError> {
        let token = self.get_zoom_access_token().await?;
        let url = format!(
            "https://api.zoom.us/v2/meetings/{}",
            urlencoding::encode(meeting_id)
        );

        let response = self
            .http_client
            .get(&url)
            .bearer_auth(&token)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_vc_reqwest_err)?;

        if response.status().as_u16() == 404 {
            return Err(VideoConferencingError::MeetingNotFound(
                meeting_id.to_string(),
            ));
        }
        let response = ensure_vc_success(response).await?;
        let data: serde_json::Value = response.json().await.map_err(|e| {
            VideoConferencingError::NetworkError(format!(
                "failed to parse Zoom meeting response: {e}"
            ))
        })?;

        parse_zoom_meeting(&data, meeting_id, self.config.enable_recording)
    }

    #[cfg(not(feature = "microservices"))]
    async fn get_zoom_meeting(
        &self,
        _meeting_id: &str,
    ) -> Result<MeetingInfo, VideoConferencingError> {
        Err(vc_feature_disabled_error())
    }

    async fn get_teams_meeting(
        &self,
        _meeting_id: &str,
    ) -> Result<MeetingInfo, VideoConferencingError> {
        // Microsoft Graph's app-only `onlineMeetings` API is addressed as
        // `/users/{userId}/onlineMeetings/{meetingId}` — it requires the
        // organizer's Graph user ID, which this method's signature (a bare
        // meeting_id) does not provide, so a correct request cannot be
        // formed here.
        Err(VideoConferencingError::ConfigurationError(
            "Teams meeting retrieval is not yet implemented: Graph's onlineMeetings API \
             requires the organizer's user ID in addition to the meeting ID"
                .to_string(),
        ))
    }

    async fn get_meet_meeting(
        &self,
        _meeting_id: &str,
    ) -> Result<MeetingInfo, VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "Google Meet meeting retrieval is not yet implemented (blocked on Meet \
             authentication, see authenticate_meet)"
                .to_string(),
        ))
    }

    // Analytics methods
    //
    // VoiRS's `MeetingAnalytics` schema (engagement/quality/interaction
    // scores) is a bespoke model that none of these platforms' REST APIs
    // expose directly; deriving it honestly requires running VoiRS's own
    // audio analysis over captured meeting audio (see
    // `process_realtime_audio`/`end_realtime_feedback` for the live-session
    // path), not a single GET request. Fail closed rather than fabricate
    // plausible-looking numbers.
    async fn get_zoom_analytics(
        &self,
        _meeting_id: &str,
    ) -> Result<MeetingAnalytics, VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "VoiRS has no analytics backend for Zoom; use an active real-time feedback session \
             (start_realtime_feedback/end_realtime_feedback) to derive real analytics instead"
                .to_string(),
        ))
    }

    async fn get_teams_analytics(
        &self,
        _meeting_id: &str,
    ) -> Result<MeetingAnalytics, VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "VoiRS has no analytics backend for Microsoft Teams; use an active real-time \
             feedback session (start_realtime_feedback/end_realtime_feedback) to derive real \
             analytics instead"
                .to_string(),
        ))
    }

    async fn get_meet_analytics(
        &self,
        _meeting_id: &str,
    ) -> Result<MeetingAnalytics, VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "VoiRS has no analytics backend for Google Meet; use an active real-time feedback \
             session (start_realtime_feedback/end_realtime_feedback) to derive real analytics \
             instead"
                .to_string(),
        ))
    }

    // Real-time feedback methods
    async fn analyze_audio_for_feedback_static(
        participant_id: &str,
        audio_data: &[f32],
    ) -> Result<Option<RealtimeMeetingFeedback>, VideoConferencingError> {
        // Simple audio analysis for demonstration
        let volume_level =
            audio_data.iter().map(|&x| x.abs()).sum::<f32>() / audio_data.len() as f32;

        if volume_level < 0.1 {
            Ok(Some(RealtimeMeetingFeedback {
                participant_id: participant_id.to_string(),
                timestamp: SystemTime::now(),
                feedback_type: FeedbackType::VolumeLevel,
                message: "Your microphone volume is too low. Please speak louder or check your microphone settings.".to_string(),
                score: volume_level * 100.0,
                suggestions: vec![
                    "Move closer to your microphone".to_string(),
                    "Check microphone gain settings".to_string(),
                    "Ensure your microphone is not muted".to_string(),
                ],
                urgency: FeedbackUrgency::Medium,
            }))
        } else if volume_level > 0.9 {
            Ok(Some(RealtimeMeetingFeedback {
                participant_id: participant_id.to_string(),
                timestamp: SystemTime::now(),
                feedback_type: FeedbackType::VolumeLevel,
                message: "Your microphone volume is too high. Please speak softer or adjust your microphone settings.".to_string(),
                score: volume_level * 100.0,
                suggestions: vec![
                    "Move further from your microphone".to_string(),
                    "Lower microphone gain settings".to_string(),
                    "Speak more softly".to_string(),
                ],
                urgency: FeedbackUrgency::Medium,
            }))
        } else {
            Ok(None) // No feedback needed
        }
    }

    async fn send_realtime_feedback(
        &self,
        meeting_id: &str,
        feedback: &RealtimeMeetingFeedback,
    ) -> Result<(), VideoConferencingError> {
        // Send feedback through the video conferencing platform
        match self.config.platform {
            VideoConferencingPlatform::Zoom => self.send_zoom_feedback(meeting_id, feedback).await,
            VideoConferencingPlatform::MicrosoftTeams => {
                self.send_teams_feedback(meeting_id, feedback).await
            }
            VideoConferencingPlatform::GoogleMeet => {
                self.send_meet_feedback(meeting_id, feedback).await
            }
            _ => Ok(()), // Not all platforms support real-time feedback
        }
    }

    async fn send_zoom_feedback(
        &self,
        _meeting_id: &str,
        _feedback: &RealtimeMeetingFeedback,
    ) -> Result<(), VideoConferencingError> {
        // Delivering an in-meeting chat message for real requires the Zoom
        // Team Chat API (`chat_message:write` scope) or the in-meeting SDK;
        // neither is wired up, so this fails closed instead of silently
        // dropping the feedback while reporting success.
        Err(VideoConferencingError::ConfigurationError(
            "Zoom in-meeting feedback delivery is not yet implemented".to_string(),
        ))
    }

    async fn send_teams_feedback(
        &self,
        _meeting_id: &str,
        _feedback: &RealtimeMeetingFeedback,
    ) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "Teams in-meeting feedback delivery is not yet implemented".to_string(),
        ))
    }

    async fn send_meet_feedback(
        &self,
        _meeting_id: &str,
        _feedback: &RealtimeMeetingFeedback,
    ) -> Result<(), VideoConferencingError> {
        Err(VideoConferencingError::ConfigurationError(
            "Google Meet in-meeting feedback delivery is not yet implemented".to_string(),
        ))
    }

    // Utility methods
    fn calculate_engagement_metrics(
        &self,
        analytics: &mut MeetingAnalytics,
        participants: &HashMap<String, MeetingParticipant>,
    ) {
        if participants.is_empty() {
            return;
        }

        let total_participants = participants.len() as f32;
        let camera_on_count = participants.values().filter(|p| p.camera_on).count() as f32;
        let microphone_on_count = participants.values().filter(|p| p.microphone_on).count() as f32;

        analytics.engagement_metrics.camera_on_percentage =
            (camera_on_count / total_participants) * 100.0;
        analytics.engagement_metrics.microphone_usage_percentage =
            (microphone_on_count / total_participants) * 100.0;

        // Calculate overall engagement score
        analytics.engagement_metrics.overall_engagement_score = f32::midpoint(
            analytics.engagement_metrics.camera_on_percentage,
            analytics.engagement_metrics.microphone_usage_percentage,
        );
    }

    async fn generate_meeting_report(
        &self,
        analytics: &MeetingAnalytics,
    ) -> Result<(), VideoConferencingError> {
        // No database/email/webhook destination is configured in this
        // module, so there is nothing external to persist the report to
        // yet; still perform the one real piece of work available today
        // (serialize and emit the real, just-computed analytics) instead of
        // a bare no-op that claims to have "generated" a report.
        match serde_json::to_string(analytics) {
            Ok(json) => {
                log::info!("Meeting report for {}: {}", analytics.meeting_id, json);
                Ok(())
            }
            Err(e) => Err(VideoConferencingError::WebhookError(format!(
                "failed to serialize meeting report: {e}"
            ))),
        }
    }
}

/// Build the error returned by every real-platform method when the crate is
/// compiled without the `microservices` feature (no HTTP client available).
#[cfg(not(feature = "microservices"))]
fn vc_feature_disabled_error() -> VideoConferencingError {
    VideoConferencingError::ConfigurationError(
        "the `microservices` feature (reqwest HTTP client) is not enabled".to_string(),
    )
}

/// Map a [`reqwest::Error`] to the appropriate [`VideoConferencingError`],
/// distinguishing timeouts from other transport failures.
#[cfg(feature = "microservices")]
fn map_vc_reqwest_err(e: reqwest::Error) -> VideoConferencingError {
    if e.is_timeout() {
        VideoConferencingError::ConnectionTimeout
    } else {
        VideoConferencingError::NetworkError(e.to_string())
    }
}

/// Turn a non-2xx response into a typed [`VideoConferencingError`], carrying
/// the real response body instead of discarding it.
#[cfg(feature = "microservices")]
async fn ensure_vc_success(
    response: reqwest::Response,
) -> Result<reqwest::Response, VideoConferencingError> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(match status.as_u16() {
        401 | 403 => VideoConferencingError::AuthenticationFailed(format!("HTTP {status}: {body}")),
        429 => VideoConferencingError::RateLimitExceeded,
        _ => VideoConferencingError::NetworkError(format!("HTTP {status}: {body}")),
    })
}

/// Convert a JSON id field to a `String`, accepting either a JSON string or
/// a bare number (Zoom's numeric meeting IDs are sometimes serialized as
/// JSON numbers rather than strings).
#[cfg(feature = "microservices")]
fn json_id_to_string(value: &serde_json::Value) -> Option<String> {
    value
        .as_str()
        .map(std::string::ToString::to_string)
        .or_else(|| value.as_u64().map(|n| n.to_string()))
        .or_else(|| value.as_i64().map(|n| n.to_string()))
}

/// Parse Zoom's `GET /v2/meetings/{meetingId}` response.
///
/// <https://developers.zoom.us/docs/api/meetings/#tag/meetings/GET/meetings/{meetingId}>
#[cfg(feature = "microservices")]
fn parse_zoom_meeting(
    value: &serde_json::Value,
    fallback_id: &str,
    recording_enabled: bool,
) -> Result<MeetingInfo, VideoConferencingError> {
    let meeting_id = json_id_to_string(&value["id"]).unwrap_or_else(|| fallback_id.to_string());
    let topic = value["topic"]
        .as_str()
        .unwrap_or("Untitled Meeting")
        .to_string();
    let host_id = value["host_id"].as_str().unwrap_or_default().to_string();
    // The basic meeting-get response does not include the host's display
    // name, only their Zoom user ID and (usually) their email; prefer the
    // email since it is at least a genuine identifier, not a fabricated name.
    let host_name = value["host_email"]
        .as_str()
        .map(String::from)
        .unwrap_or_else(|| host_id.clone());

    let start_time = value["start_time"]
        .as_str()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| SystemTime::UNIX_EPOCH + Duration::from_secs(dt.timestamp().max(0) as u64))
        // Instant meetings have no scheduled start time in the response.
        .unwrap_or_else(SystemTime::now);

    let duration_minutes = value["duration"].as_u64().unwrap_or(0) as u32;

    let status = match value["status"].as_str() {
        Some("started") => MeetingStatus::InProgress,
        Some("finished" | "ended") => MeetingStatus::Ended,
        _ => MeetingStatus::Scheduled,
    };

    let meeting_url = value["join_url"]
        .as_str()
        .map(String::from)
        .unwrap_or_else(|| format!("https://zoom.us/j/{meeting_id}"));

    Ok(MeetingInfo {
        meeting_id,
        meeting_uuid: value["uuid"].as_str().map(String::from),
        topic,
        start_time,
        duration_minutes,
        host_id,
        host_name,
        // Zoom's basic meeting-get endpoint does not return a participant
        // roster; that requires a separate Dashboard/Report API call with
        // additional scopes not assumed to be granted here. An honestly
        // empty list, not a fabricated participant.
        participants: vec![],
        status,
        meeting_url,
        recording_enabled,
    })
}

/// Rate limiter for video conferencing API requests
struct VideoConferencingRateLimiter {
    /// Maximum requests per window
    max_requests: u32,
    /// Time window duration
    window_duration: Duration,
    /// Request timestamps
    requests: Vec<SystemTime>,
}

impl VideoConferencingRateLimiter {
    fn new(max_requests: u32, window_duration: Duration) -> Self {
        Self {
            max_requests,
            window_duration,
            requests: Vec::new(),
        }
    }

    fn check_rate_limit(&mut self) -> Result<(), VideoConferencingError> {
        let now = SystemTime::now();
        let window_start = now - self.window_duration;

        // Remove old requests
        self.requests.retain(|&time| time > window_start);

        if self.requests.len() >= self.max_requests as usize {
            return Err(VideoConferencingError::RateLimitExceeded);
        }

        self.requests.push(now);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_conferencing_auth_config_default() {
        let config = VideoConferencingAuthConfig::default();
        assert_eq!(config.platform, VideoConferencingPlatform::Zoom);
        assert_eq!(config.timeout_seconds, 30);
        assert!(config.enable_real_time_feedback);
    }

    #[test]
    fn test_video_conferencing_platform_display() {
        assert_eq!(VideoConferencingPlatform::Zoom.to_string(), "Zoom");
        assert_eq!(
            VideoConferencingPlatform::MicrosoftTeams.to_string(),
            "Microsoft Teams"
        );
        assert_eq!(
            VideoConferencingPlatform::Custom("MyPlatform".to_string()).to_string(),
            "Custom: MyPlatform"
        );
    }

    #[tokio::test]
    async fn test_video_conferencing_manager_creation() {
        let config = VideoConferencingAuthConfig::default();
        let realtime_config = RealtimeConfig::default();
        let manager = VideoConferencingIntegrationManager::new(config, realtime_config);
        // Manager should be created successfully
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = VideoConferencingRateLimiter::new(2, Duration::from_secs(1));

        // First two requests should succeed
        assert!(limiter.check_rate_limit().is_ok());
        assert!(limiter.check_rate_limit().is_ok());

        // Third request should fail
        assert!(matches!(
            limiter.check_rate_limit(),
            Err(VideoConferencingError::RateLimitExceeded)
        ));
    }

    #[test]
    fn test_meeting_status_enum() {
        assert_eq!(MeetingStatus::InProgress, MeetingStatus::InProgress);
        assert_ne!(MeetingStatus::InProgress, MeetingStatus::Ended);
    }

    #[test]
    fn test_feedback_urgency_enum() {
        assert_eq!(FeedbackUrgency::High, FeedbackUrgency::High);
        assert_ne!(FeedbackUrgency::Low, FeedbackUrgency::Critical);
    }

    #[tokio::test]
    async fn test_audio_analysis_low_volume() {
        // Test low volume audio
        let low_volume_audio = vec![0.05; 1000]; // Very low volume
        let feedback = VideoConferencingIntegrationManager::analyze_audio_for_feedback_static(
            "participant_1",
            &low_volume_audio,
        )
        .await
        .unwrap();

        assert!(feedback.is_some());
        let feedback = feedback.unwrap();
        assert_eq!(feedback.feedback_type, FeedbackType::VolumeLevel);
        assert_eq!(feedback.urgency, FeedbackUrgency::Medium);
    }

    #[tokio::test]
    async fn test_audio_analysis_high_volume() {
        // Test high volume audio
        let high_volume_audio = vec![0.95; 1000]; // Very high volume
        let feedback = VideoConferencingIntegrationManager::analyze_audio_for_feedback_static(
            "participant_1",
            &high_volume_audio,
        )
        .await
        .unwrap();

        assert!(feedback.is_some());
        let feedback = feedback.unwrap();
        assert_eq!(feedback.feedback_type, FeedbackType::VolumeLevel);
        assert_eq!(feedback.urgency, FeedbackUrgency::Medium);
    }

    #[tokio::test]
    async fn test_audio_analysis_normal_volume() {
        // Test normal volume audio
        let normal_volume_audio = vec![0.5; 1000]; // Normal volume
        let feedback = VideoConferencingIntegrationManager::analyze_audio_for_feedback_static(
            "participant_1",
            &normal_volume_audio,
        )
        .await
        .unwrap();

        assert!(feedback.is_none()); // No feedback needed for normal volume
    }

    // --- Fail-closed behavior -------------------------------------------

    #[tokio::test]
    async fn test_unconfigured_zoom_fails_closed_without_network() {
        // Default config has no account_id/oauth credentials: authenticate
        // must reject locally, never attempt a request nor fabricate a token.
        let config = VideoConferencingAuthConfig::default();
        let mut manager =
            VideoConferencingIntegrationManager::new(config, RealtimeConfig::default());

        let result = manager.authenticate().await;
        assert!(matches!(
            result,
            Err(VideoConferencingError::ConfigurationError(_))
        ));
    }

    #[tokio::test]
    async fn test_unimplemented_platforms_fail_closed_not_ok() {
        for platform in [
            VideoConferencingPlatform::WebEx,
            VideoConferencingPlatform::GoToMeeting,
            VideoConferencingPlatform::BlueJeans,
            VideoConferencingPlatform::Skype,
            VideoConferencingPlatform::Custom("Acme Meet".to_string()),
        ] {
            let config = VideoConferencingAuthConfig {
                platform,
                ..VideoConferencingAuthConfig::default()
            };
            let mut manager =
                VideoConferencingIntegrationManager::new(config, RealtimeConfig::default());
            let result = manager.authenticate().await;
            assert!(
                result.is_err(),
                "unimplemented platform must never report Ok"
            );
        }
    }

    #[tokio::test]
    async fn test_meeting_analytics_without_session_fails_closed() {
        // No fabricated decimal analytics may be returned for a meeting
        // that was never tracked through a real-time feedback session.
        let config = VideoConferencingAuthConfig::default();
        let manager = VideoConferencingIntegrationManager::new(config, RealtimeConfig::default());

        let result = manager.get_meeting_analytics("meeting-not-tracked").await;
        assert!(matches!(
            result,
            Err(VideoConferencingError::ConfigurationError(_))
        ));
    }

    // --- Real response parsing (offline, JSON fixtures) ------------------

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_zoom_meeting_from_real_response_shape() {
        let value = serde_json::json!({
            "uuid": "abc123==",
            "id": 987654321_u64,
            "host_id": "host-abc",
            "host_email": "trainer@example.com",
            "topic": "VoiRS Speech Training Session",
            "status": "started",
            "start_time": "2024-05-01T15:00:00Z",
            "duration": 45,
            "join_url": "https://zoom.us/j/987654321"
        });

        let meeting = parse_zoom_meeting(&value, "unused", true).unwrap();
        assert_eq!(meeting.meeting_id, "987654321");
        assert_eq!(meeting.meeting_uuid.as_deref(), Some("abc123=="));
        assert_eq!(meeting.topic, "VoiRS Speech Training Session");
        assert_eq!(meeting.host_name, "trainer@example.com");
        assert_eq!(meeting.duration_minutes, 45);
        assert_eq!(meeting.status, MeetingStatus::InProgress);
        assert_eq!(meeting.meeting_url, "https://zoom.us/j/987654321");
        assert!(meeting.recording_enabled);
        // Honestly empty: the basic meeting-get response never includes a
        // participant roster.
        assert!(meeting.participants.is_empty());

        // A different response must parse into genuinely different data.
        let other = serde_json::json!({
            "id": 111,
            "topic": "Other Meeting",
            "status": "waiting",
            "duration": 15
        });
        let other_meeting = parse_zoom_meeting(&other, "unused", false).unwrap();
        assert_ne!(meeting.topic, other_meeting.topic);
        assert_eq!(other_meeting.status, MeetingStatus::Scheduled);
        assert!(!other_meeting.recording_enabled);
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_json_id_to_string_accepts_string_and_number() {
        assert_eq!(
            json_id_to_string(&serde_json::json!(42)),
            Some("42".to_string())
        );
        assert_eq!(
            json_id_to_string(&serde_json::json!("abc")),
            Some("abc".to_string())
        );
        assert_eq!(json_id_to_string(&serde_json::json!(null)), None);
    }

    /// Live end-to-end test against the real Zoom API. Gated behind
    /// environment variables so the default offline test run never makes a
    /// network call.
    #[cfg(feature = "microservices")]
    #[tokio::test]
    async fn test_zoom_get_meeting_live() {
        let (Ok(account_id), Ok(client_id), Ok(client_secret), Ok(meeting_id)) = (
            std::env::var("VOIRS_TEST_ZOOM_ACCOUNT_ID"),
            std::env::var("VOIRS_TEST_ZOOM_CLIENT_ID"),
            std::env::var("VOIRS_TEST_ZOOM_CLIENT_SECRET"),
            std::env::var("VOIRS_TEST_ZOOM_MEETING_ID"),
        ) else {
            eprintln!(
                "skipping test_zoom_get_meeting_live: set VOIRS_TEST_ZOOM_ACCOUNT_ID, \
                 VOIRS_TEST_ZOOM_CLIENT_ID, VOIRS_TEST_ZOOM_CLIENT_SECRET and \
                 VOIRS_TEST_ZOOM_MEETING_ID to run this against a real Zoom account"
            );
            return;
        };

        let config = VideoConferencingAuthConfig {
            platform: VideoConferencingPlatform::Zoom,
            account_id: Some(account_id),
            oauth_client_id: Some(client_id),
            oauth_client_secret: Some(client_secret),
            ..VideoConferencingAuthConfig::default()
        };
        let mut manager =
            VideoConferencingIntegrationManager::new(config, RealtimeConfig::default());

        let meeting = manager
            .get_meeting(&meeting_id)
            .await
            .expect("live Zoom API call should succeed with valid credentials");
        assert_eq!(meeting.meeting_id, meeting_id);
    }
}
