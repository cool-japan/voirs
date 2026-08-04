//! # Learning Management System (LMS) Integration
//!
//! This module provides comprehensive integration with major LMS platforms including
//! Canvas, Blackboard, Moodle, and others. It supports grade passback, assignment
//! integration, progress reporting, and Single Sign-On (SSO) authentication.

use crate::traits::{FeedbackSession, FocusArea, SessionScores, UserProgress};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "microservices")]
use reqwest::Client;

/// LMS integration error types
#[derive(Debug, Clone)]
pub enum LMSError {
    /// Authentication failed with message
    AuthenticationFailed(String),
    /// Connection timeout occurred
    ConnectionTimeout,
    /// Invalid API key provided
    InvalidApiKey,
    /// Grade passback failed with message
    GradePassbackFailed(String),
    /// Assignment not found with ID
    AssignmentNotFound(String),
    /// Student not found with ID
    StudentNotFound(String),
    /// Course not found with ID
    CourseNotFound(String),
    /// Network error occurred with message
    NetworkError(String),
    /// Configuration error with message
    ConfigurationError(String),
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Unauthorized access attempted
    UnauthorizedAccess,
    /// Data validation error with message
    DataValidationError(String),
}

impl fmt::Display for LMSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LMSError::AuthenticationFailed(msg) => write!(f, "Authentication failed: {msg}"),
            LMSError::ConnectionTimeout => write!(f, "Connection timeout"),
            LMSError::InvalidApiKey => write!(f, "Invalid API key"),
            LMSError::GradePassbackFailed(msg) => write!(f, "Grade passback failed: {msg}"),
            LMSError::AssignmentNotFound(id) => write!(f, "Assignment not found: {id}"),
            LMSError::StudentNotFound(id) => write!(f, "Student not found: {id}"),
            LMSError::CourseNotFound(id) => write!(f, "Course not found: {id}"),
            LMSError::NetworkError(msg) => write!(f, "Network error: {msg}"),
            LMSError::ConfigurationError(msg) => write!(f, "Configuration error: {msg}"),
            LMSError::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            LMSError::UnauthorizedAccess => write!(f, "Unauthorized access"),
            LMSError::DataValidationError(msg) => write!(f, "Data validation error: {msg}"),
        }
    }
}

impl Error for LMSError {}

/// Supported LMS platforms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LMSPlatform {
    /// Canvas LMS
    Canvas,
    /// Blackboard Learn
    Blackboard,
    /// Moodle LMS
    Moodle,
    /// Desire2Learn/Brightspace
    D2L,
    /// Schoology platform
    Schoology,
    /// Sakai platform
    Sakai,
    /// Custom LMS with name
    Custom(String),
}

impl fmt::Display for LMSPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LMSPlatform::Canvas => write!(f, "Canvas"),
            LMSPlatform::Blackboard => write!(f, "Blackboard"),
            LMSPlatform::Moodle => write!(f, "Moodle"),
            LMSPlatform::D2L => write!(f, "D2L/Brightspace"),
            LMSPlatform::Schoology => write!(f, "Schoology"),
            LMSPlatform::Sakai => write!(f, "Sakai"),
            LMSPlatform::Custom(name) => write!(f, "Custom: {name}"),
        }
    }
}

/// LMS authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSAuthConfig {
    /// LMS platform type
    pub platform: LMSPlatform,
    /// API key for authentication
    pub api_key: String,
    /// Optional API secret
    pub api_secret: Option<String>,
    /// Base URL of LMS instance
    pub base_url: String,
    /// OAuth client ID
    pub oauth_client_id: Option<String>,
    /// OAuth client secret
    pub oauth_client_secret: Option<String>,
    /// LTI consumer key
    pub consumer_key: Option<String>,
    /// LTI shared secret
    pub shared_secret: Option<String>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
}

impl Default for LMSAuthConfig {
    fn default() -> Self {
        Self {
            platform: LMSPlatform::Canvas,
            api_key: String::new(),
            api_secret: None,
            base_url: String::new(),
            oauth_client_id: None,
            oauth_client_secret: None,
            consumer_key: None,
            shared_secret: None,
            timeout_seconds: 30,
        }
    }
}

/// LMS assignment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSAssignment {
    /// Assignment identifier
    pub id: String,
    /// Assignment name
    pub name: String,
    /// Assignment description
    pub description: String,
    /// Associated course ID
    pub course_id: String,
    /// Maximum points for assignment
    pub max_points: f64,
    /// Assignment due date
    pub due_date: Option<SystemTime>,
    /// Whether assignment is published
    pub published: bool,
    /// Allowed submission types
    pub submission_types: Vec<String>,
    /// Grading criteria for assignment
    pub grading_criteria: Vec<GradingCriterion>,
}

/// Grading criteria for assignments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradingCriterion {
    /// Criterion name
    pub name: String,
    /// Criterion description
    pub description: String,
    /// Points allocated to criterion
    pub points: f64,
    /// Related focus area
    pub focus_area: Option<FocusArea>,
}

/// Student information from LMS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSStudent {
    /// Student identifier
    pub id: String,
    /// External student ID
    pub external_id: Option<String>,
    /// Student full name
    pub name: String,
    /// Student email address
    pub email: String,
    /// Associated course ID
    pub course_id: String,
    /// Enrollment status
    pub enrollment_status: String,
    /// Student role
    pub role: String,
}

/// Course information from LMS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSCourse {
    /// Course identifier
    pub id: String,
    /// Course name
    pub name: String,
    /// Course code
    pub course_code: String,
    /// Academic term
    pub term: String,
    /// Course start date
    pub start_date: Option<SystemTime>,
    /// Course end date
    pub end_date: Option<SystemTime>,
    /// Enrollment term ID
    pub enrollment_term_id: Option<String>,
    /// Whether course is published
    pub published: bool,
}

/// Grade submission data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradeSubmission {
    /// Student identifier
    pub student_id: String,
    /// Course identifier (required to address the real per-platform grade
    /// passback endpoint, e.g. Canvas's
    /// `/courses/{course_id}/assignments/{assignment_id}/submissions/{student_id}`).
    pub course_id: String,
    /// Assignment identifier
    pub assignment_id: String,
    /// Earned score
    pub score: f64,
    /// Maximum possible score
    pub max_score: f64,
    /// Optional comment
    pub comment: Option<String>,
    /// Submission timestamp
    pub submission_time: SystemTime,
    /// Detailed feedback by criterion
    pub detailed_feedback: Vec<DetailedFeedback>,
}

/// Detailed feedback for specific skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedFeedback {
    /// Grading criterion name
    pub criterion_name: String,
    /// Score for this criterion
    pub score: f64,
    /// Maximum score for criterion
    pub max_score: f64,
    /// Feedback text
    pub feedback: String,
    /// Related focus area
    pub focus_area: Option<FocusArea>,
}

/// Session data for LMS integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSSession {
    /// Session timestamp
    pub timestamp: SystemTime,
    /// Session duration
    pub duration: Duration,
    /// Session scores
    pub score: Option<SessionScores>,
    /// Session feedback items
    pub feedback: Vec<LMSFeedbackItem>,
}

/// Feedback item for LMS session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSFeedbackItem {
    /// Feedback message
    pub message: String,
    /// Feedback priority
    pub priority: f64,
    /// Feedback category
    pub category: String,
}

/// Progress report for LMS integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSProgressReport {
    /// Student identifier
    pub student_id: String,
    /// Course identifier
    pub course_id: String,
    /// Overall progress score (0.0-1.0)
    pub overall_progress: f64,
    /// Completion percentage
    pub completion_percentage: f64,
    /// Number of completed sessions
    pub sessions_completed: u32,
    /// Total time spent in minutes
    pub time_spent_minutes: u32,
    /// Skill scores by focus area
    pub skill_breakdown: HashMap<FocusArea, f64>,
    /// Earned achievements
    pub achievements: Vec<String>,
    /// Report generation timestamp
    pub generated_at: SystemTime,
}

/// LMS integration manager
pub struct LMSIntegrationManager {
    /// Authentication configuration
    config: LMSAuthConfig,
    /// Rate limiter for API requests
    rate_limiter: RateLimiter,
    /// Data cache
    cache: LMSCache,
    /// HTTP client used for real platform API requests
    #[cfg(feature = "microservices")]
    http_client: Client,
}

impl LMSIntegrationManager {
    /// Create a new LMS integration manager
    #[must_use]
    pub fn new(config: LMSAuthConfig) -> Self {
        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        #[cfg(feature = "microservices")]
        voirs_sdk::ensure_crypto_provider();

        Self {
            config,
            rate_limiter: RateLimiter::new(100, Duration::from_secs(60)), // 100 requests per minute
            cache: LMSCache::new(),
            #[cfg(feature = "microservices")]
            http_client: Client::new(),
        }
    }

    /// Verify the manager has the minimum configuration required to reach a
    /// real LMS endpoint, failing closed instead of attempting a request
    /// that could never succeed (or, worse, silently no-op).
    fn ensure_configured(&self) -> Result<(), LMSError> {
        if self.config.base_url.trim().is_empty() {
            return Err(LMSError::ConfigurationError(
                "LMS base_url is not configured".to_string(),
            ));
        }
        if self.config.api_key.trim().is_empty() {
            return Err(LMSError::ConfigurationError(
                "LMS api_key is not configured".to_string(),
            ));
        }
        Ok(())
    }

    /// Authenticate with the LMS platform
    pub async fn authenticate(&mut self) -> Result<(), LMSError> {
        self.ensure_configured()?;
        self.rate_limiter.check_rate_limit()?;

        match self.config.platform {
            LMSPlatform::Canvas => self.authenticate_canvas().await,
            LMSPlatform::Blackboard => self.authenticate_blackboard().await,
            LMSPlatform::Moodle => self.authenticate_moodle().await,
            LMSPlatform::D2L => self.authenticate_d2l().await,
            LMSPlatform::Schoology => self.authenticate_schoology().await,
            LMSPlatform::Sakai => self.authenticate_sakai().await,
            LMSPlatform::Custom(_) => self.authenticate_custom().await,
        }
    }

    /// Get course information
    pub async fn get_course(&mut self, course_id: &str) -> Result<LMSCourse, LMSError> {
        if let Some(cached_course) = self.cache.get_course(course_id) {
            return Ok(cached_course.clone());
        }

        self.ensure_configured()?;
        self.rate_limiter.check_rate_limit()?;

        let course = match self.config.platform {
            LMSPlatform::Canvas => self.get_canvas_course(course_id).await?,
            LMSPlatform::Blackboard => self.get_blackboard_course(course_id).await?,
            LMSPlatform::Moodle => self.get_moodle_course(course_id).await?,
            _ => {
                return Err(LMSError::ConfigurationError(
                    "Platform not supported yet".to_string(),
                ))
            }
        };

        self.cache.cache_course(course.clone());
        Ok(course)
    }

    /// Get students in a course
    pub async fn get_course_students(
        &mut self,
        course_id: &str,
    ) -> Result<Vec<LMSStudent>, LMSError> {
        self.ensure_configured()?;
        self.rate_limiter.check_rate_limit()?;

        match self.config.platform {
            LMSPlatform::Canvas => self.get_canvas_students(course_id).await,
            LMSPlatform::Blackboard => self.get_blackboard_students(course_id).await,
            LMSPlatform::Moodle => self.get_moodle_students(course_id).await,
            _ => Err(LMSError::ConfigurationError(
                "Platform not supported yet".to_string(),
            )),
        }
    }

    /// Get assignments for a course
    pub async fn get_course_assignments(
        &mut self,
        course_id: &str,
    ) -> Result<Vec<LMSAssignment>, LMSError> {
        self.ensure_configured()?;
        self.rate_limiter.check_rate_limit()?;

        match self.config.platform {
            LMSPlatform::Canvas => self.get_canvas_assignments(course_id).await,
            LMSPlatform::Blackboard => self.get_blackboard_assignments(course_id).await,
            LMSPlatform::Moodle => self.get_moodle_assignments(course_id).await,
            _ => Err(LMSError::ConfigurationError(
                "Platform not supported yet".to_string(),
            )),
        }
    }

    /// Submit grade for a student
    pub async fn submit_grade(&mut self, submission: &GradeSubmission) -> Result<(), LMSError> {
        self.validate_grade_submission(submission)?;
        self.ensure_configured()?;
        self.rate_limiter.check_rate_limit()?;

        match self.config.platform {
            LMSPlatform::Canvas => self.submit_canvas_grade(submission).await,
            LMSPlatform::Blackboard => self.submit_blackboard_grade(submission).await,
            LMSPlatform::Moodle => self.submit_moodle_grade(submission).await,
            _ => Err(LMSError::ConfigurationError(
                "Platform not supported yet".to_string(),
            )),
        }
    }

    /// Generate and submit progress report
    pub async fn submit_progress_report(
        &mut self,
        student_id: &str,
        user_progress: &UserProgress,
        sessions: &[LMSSession],
    ) -> Result<LMSProgressReport, LMSError> {
        let report = self.generate_progress_report(student_id, user_progress, sessions)?;

        // Submit to LMS (implementation depends on platform capabilities)
        match self.config.platform {
            LMSPlatform::Canvas => self.submit_canvas_progress(&report).await?,
            LMSPlatform::Blackboard => self.submit_blackboard_progress(&report).await?,
            LMSPlatform::Moodle => self.submit_moodle_progress(&report).await?,
            _ => {} // Some platforms may not support progress reports
        }

        Ok(report)
    }

    /// Convert `VoiRS` session to LMS grade submission
    pub fn session_to_grade_submission(
        &self,
        student_id: &str,
        assignment_id: &str,
        session: &LMSSession,
        assignment: &LMSAssignment,
    ) -> Result<GradeSubmission, LMSError> {
        let overall_score = self.calculate_overall_score(session);
        let grade_score = overall_score * assignment.max_points;

        let detailed_feedback = self.generate_detailed_feedback(session, assignment)?;
        let comment = self.generate_session_comment(session);

        Ok(GradeSubmission {
            student_id: student_id.to_string(),
            course_id: assignment.course_id.clone(),
            assignment_id: assignment_id.to_string(),
            score: grade_score,
            max_score: assignment.max_points,
            comment: Some(comment),
            submission_time: SystemTime::now(),
            detailed_feedback,
        })
    }

    // Platform-specific authentication methods
    #[cfg(feature = "microservices")]
    async fn authenticate_canvas(&self) -> Result<(), LMSError> {
        // Canvas API: a plain bearer-token GET against any authenticated
        // endpoint proves the token is valid. `/users/self` is the smallest
        // one available.
        let url = format!("{}/api/v1/users/self", self.config.base_url);

        let response = self
            .http_client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_reqwest_err)?;

        ensure_lms_success(response).await?;
        Ok(())
    }

    #[cfg(not(feature = "microservices"))]
    async fn authenticate_canvas(&self) -> Result<(), LMSError> {
        Err(feature_disabled_error())
    }

    #[cfg(feature = "microservices")]
    async fn authenticate_blackboard(&self) -> Result<(), LMSError> {
        // Blackboard Learn REST: OAuth2 client-credentials grant, application
        // key/secret sent as HTTP Basic auth per Blackboard's documented flow.
        let url = format!("{}/learn/api/public/v1/oauth2/token", self.config.base_url);

        let response = self
            .http_client
            .post(&url)
            .basic_auth(&self.config.api_key, self.config.api_secret.as_deref())
            .form(&[("grant_type", "client_credentials")])
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_reqwest_err)?;

        let response = ensure_lms_success(response).await?;
        let token: serde_json::Value = response.json().await.map_err(|e| {
            LMSError::AuthenticationFailed(format!(
                "failed to parse Blackboard token response: {e}"
            ))
        })?;

        if token["access_token"].as_str().is_none() {
            return Err(LMSError::AuthenticationFailed(
                "Blackboard token response is missing 'access_token'".to_string(),
            ));
        }
        Ok(())
    }

    #[cfg(not(feature = "microservices"))]
    async fn authenticate_blackboard(&self) -> Result<(), LMSError> {
        Err(feature_disabled_error())
    }

    #[cfg(feature = "microservices")]
    async fn authenticate_moodle(&self) -> Result<(), LMSError> {
        // Moodle web services: the token and requested function are query
        // parameters on a GET, not a header; errors come back as HTTP 200
        // with an embedded `exception` object rather than a non-2xx status.
        let url = format!(
            "{}/webservice/rest/server.php?wstoken={}&wsfunction=core_webservice_get_site_info&moodlewsrestformat=json",
            self.config.base_url,
            urlencoding::encode(&self.config.api_key),
        );

        let response = self
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_reqwest_err)?;

        let response = ensure_lms_success(response).await?;
        let data: serde_json::Value = response.json().await.map_err(|e| {
            LMSError::AuthenticationFailed(format!("failed to parse Moodle response: {e}"))
        })?;
        check_moodle_exception(&data)
    }

    #[cfg(not(feature = "microservices"))]
    async fn authenticate_moodle(&self) -> Result<(), LMSError> {
        Err(feature_disabled_error())
    }

    async fn authenticate_d2l(&self) -> Result<(), LMSError> {
        Err(LMSError::ConfigurationError(
            "D2L/Brightspace (Valence API) integration is not yet implemented".to_string(),
        ))
    }

    async fn authenticate_schoology(&self) -> Result<(), LMSError> {
        Err(LMSError::ConfigurationError(
            "Schoology integration is not yet implemented".to_string(),
        ))
    }

    async fn authenticate_sakai(&self) -> Result<(), LMSError> {
        Err(LMSError::ConfigurationError(
            "Sakai integration is not yet implemented".to_string(),
        ))
    }

    async fn authenticate_custom(&self) -> Result<(), LMSError> {
        Err(LMSError::ConfigurationError(
            "custom LMS platforms have no built-in client; supply one via a platform-specific integration".to_string(),
        ))
    }

    // Platform-specific course retrieval methods
    #[cfg(feature = "microservices")]
    async fn get_canvas_course(&self, course_id: &str) -> Result<LMSCourse, LMSError> {
        let url = format!("{}/api/v1/courses/{}", self.config.base_url, course_id);

        let response = self
            .http_client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_reqwest_err)?;

        if response.status().as_u16() == 404 {
            return Err(LMSError::CourseNotFound(course_id.to_string()));
        }
        let response = ensure_lms_success(response).await?;
        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LMSError::NetworkError(format!("failed to parse Canvas course: {e}")))?;
        parse_canvas_course(&data, course_id)
    }

    #[cfg(not(feature = "microservices"))]
    async fn get_canvas_course(&self, _course_id: &str) -> Result<LMSCourse, LMSError> {
        Err(feature_disabled_error())
    }

    #[cfg(feature = "microservices")]
    async fn get_blackboard_course(&self, _course_id: &str) -> Result<LMSCourse, LMSError> {
        Err(LMSError::ConfigurationError(
            "Blackboard course retrieval is not yet implemented (only OAuth2 authentication is)"
                .to_string(),
        ))
    }

    #[cfg(not(feature = "microservices"))]
    async fn get_blackboard_course(&self, _course_id: &str) -> Result<LMSCourse, LMSError> {
        Err(feature_disabled_error())
    }

    #[cfg(feature = "microservices")]
    async fn get_moodle_course(&self, course_id: &str) -> Result<LMSCourse, LMSError> {
        let url = format!(
            "{}/webservice/rest/server.php?wstoken={}&wsfunction=core_course_get_courses&moodlewsrestformat=json&options[ids][0]={}",
            self.config.base_url,
            urlencoding::encode(&self.config.api_key),
            urlencoding::encode(course_id),
        );

        let response = self
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_reqwest_err)?;

        let response = ensure_lms_success(response).await?;
        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LMSError::NetworkError(format!("failed to parse Moodle course: {e}")))?;
        check_moodle_exception(&data)?;

        let course_value = data
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or_else(|| LMSError::CourseNotFound(course_id.to_string()))?;
        parse_moodle_course(course_value, course_id)
    }

    #[cfg(not(feature = "microservices"))]
    async fn get_moodle_course(&self, _course_id: &str) -> Result<LMSCourse, LMSError> {
        Err(feature_disabled_error())
    }

    // Platform-specific student retrieval methods
    #[cfg(feature = "microservices")]
    async fn get_canvas_students(&self, course_id: &str) -> Result<Vec<LMSStudent>, LMSError> {
        let url = format!(
            "{}/api/v1/courses/{}/students?include[]=email&per_page=100",
            self.config.base_url, course_id
        );

        let response = self
            .http_client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_reqwest_err)?;

        if response.status().as_u16() == 404 {
            return Err(LMSError::CourseNotFound(course_id.to_string()));
        }
        let response = ensure_lms_success(response).await?;
        let data: Vec<serde_json::Value> = response.json().await.map_err(|e| {
            LMSError::NetworkError(format!("failed to parse Canvas students response: {e}"))
        })?;

        data.iter()
            .map(|value| parse_canvas_student(value, course_id))
            .collect()
    }

    #[cfg(not(feature = "microservices"))]
    async fn get_canvas_students(&self, _course_id: &str) -> Result<Vec<LMSStudent>, LMSError> {
        Err(feature_disabled_error())
    }

    async fn get_blackboard_students(&self, _course_id: &str) -> Result<Vec<LMSStudent>, LMSError> {
        Err(LMSError::ConfigurationError(
            "Blackboard student roster retrieval is not yet implemented".to_string(),
        ))
    }

    async fn get_moodle_students(&self, _course_id: &str) -> Result<Vec<LMSStudent>, LMSError> {
        Err(LMSError::ConfigurationError(
            "Moodle student roster retrieval is not yet implemented".to_string(),
        ))
    }

    // Platform-specific assignment retrieval methods
    #[cfg(feature = "microservices")]
    async fn get_canvas_assignments(
        &self,
        course_id: &str,
    ) -> Result<Vec<LMSAssignment>, LMSError> {
        let url = format!(
            "{}/api/v1/courses/{}/assignments?include[]=rubric&per_page=100",
            self.config.base_url, course_id
        );

        let response = self
            .http_client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_reqwest_err)?;

        if response.status().as_u16() == 404 {
            return Err(LMSError::CourseNotFound(course_id.to_string()));
        }
        let response = ensure_lms_success(response).await?;
        let data: Vec<serde_json::Value> = response.json().await.map_err(|e| {
            LMSError::NetworkError(format!("failed to parse Canvas assignments response: {e}"))
        })?;

        data.iter()
            .map(|value| parse_canvas_assignment(value, course_id))
            .collect()
    }

    #[cfg(not(feature = "microservices"))]
    async fn get_canvas_assignments(
        &self,
        _course_id: &str,
    ) -> Result<Vec<LMSAssignment>, LMSError> {
        Err(feature_disabled_error())
    }

    async fn get_blackboard_assignments(
        &self,
        _course_id: &str,
    ) -> Result<Vec<LMSAssignment>, LMSError> {
        Err(LMSError::ConfigurationError(
            "Blackboard assignment retrieval is not yet implemented".to_string(),
        ))
    }

    async fn get_moodle_assignments(
        &self,
        _course_id: &str,
    ) -> Result<Vec<LMSAssignment>, LMSError> {
        Err(LMSError::ConfigurationError(
            "Moodle assignment retrieval is not yet implemented".to_string(),
        ))
    }

    // Platform-specific grade submission methods
    #[cfg(feature = "microservices")]
    async fn submit_canvas_grade(&self, submission: &GradeSubmission) -> Result<(), LMSError> {
        // Canvas grade passback: PUT .../submissions/:user_id with the grade
        // and (optional) comment as form-encoded nested params.
        let url = format!(
            "{}/api/v1/courses/{}/assignments/{}/submissions/{}",
            self.config.base_url,
            submission.course_id,
            submission.assignment_id,
            submission.student_id
        );

        let mut form: Vec<(&str, String)> =
            vec![("submission[posted_grade]", submission.score.to_string())];
        if let Some(comment) = &submission.comment {
            form.push(("comment[text_comment]", comment.clone()));
        }

        let response = self
            .http_client
            .put(&url)
            .bearer_auth(&self.config.api_key)
            .form(&form)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_reqwest_err)?;

        ensure_lms_success(response).await?;
        Ok(())
    }

    #[cfg(not(feature = "microservices"))]
    async fn submit_canvas_grade(&self, _submission: &GradeSubmission) -> Result<(), LMSError> {
        Err(feature_disabled_error())
    }

    async fn submit_blackboard_grade(&self, _submission: &GradeSubmission) -> Result<(), LMSError> {
        Err(LMSError::ConfigurationError(
            "Blackboard grade passback is not yet implemented".to_string(),
        ))
    }

    async fn submit_moodle_grade(&self, _submission: &GradeSubmission) -> Result<(), LMSError> {
        Err(LMSError::ConfigurationError(
            "Moodle grade passback is not yet implemented".to_string(),
        ))
    }

    // Progress report submission methods
    #[cfg(feature = "microservices")]
    async fn submit_canvas_progress(&self, report: &LMSProgressReport) -> Result<(), LMSError> {
        // Canvas has no first-class "progress report" resource; deliver it
        // as a real Conversations message to the student, which is a
        // documented, genuine Canvas API action.
        let url = format!("{}/api/v1/conversations", self.config.base_url);
        let body = format_progress_report_message(report);

        let form: Vec<(&str, String)> = vec![
            ("recipients[]", report.student_id.clone()),
            ("subject", "VoiRS Progress Report".to_string()),
            ("body", body),
        ];

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .form(&form)
            .timeout(Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await
            .map_err(map_reqwest_err)?;

        ensure_lms_success(response).await?;
        Ok(())
    }

    #[cfg(not(feature = "microservices"))]
    async fn submit_canvas_progress(&self, _report: &LMSProgressReport) -> Result<(), LMSError> {
        Err(feature_disabled_error())
    }

    async fn submit_blackboard_progress(
        &self,
        _report: &LMSProgressReport,
    ) -> Result<(), LMSError> {
        Err(LMSError::ConfigurationError(
            "Blackboard progress reporting is not yet implemented".to_string(),
        ))
    }

    async fn submit_moodle_progress(&self, _report: &LMSProgressReport) -> Result<(), LMSError> {
        Err(LMSError::ConfigurationError(
            "Moodle progress reporting is not yet implemented".to_string(),
        ))
    }

    // Utility methods
    fn calculate_overall_score(&self, session: &LMSSession) -> f64 {
        match &session.score {
            Some(score) => f64::from(score.overall_score),
            None => 0.0,
        }
    }

    fn generate_detailed_feedback(
        &self,
        session: &LMSSession,
        assignment: &LMSAssignment,
    ) -> Result<Vec<DetailedFeedback>, LMSError> {
        let mut feedback = Vec::new();

        if let Some(score) = &session.score {
            for criterion in &assignment.grading_criteria {
                let score_value = match criterion.focus_area {
                    Some(FocusArea::Pronunciation) => f64::from(score.average_pronunciation),
                    Some(FocusArea::Fluency) => f64::from(score.average_fluency),
                    Some(FocusArea::Intonation) => f64::from(score.average_quality), // Use quality as proxy for intonation
                    _ => f64::from(score.overall_score),
                };

                feedback.push(DetailedFeedback {
                    criterion_name: criterion.name.clone(),
                    score: score_value * criterion.points,
                    max_score: criterion.points,
                    feedback: self.generate_criterion_feedback(&criterion.name, score_value),
                    focus_area: criterion.focus_area.clone(),
                });
            }
        }

        Ok(feedback)
    }

    fn generate_criterion_feedback(&self, criterion: &str, score: f64) -> String {
        let performance = if score >= 0.9 {
            "Excellent"
        } else if score >= 0.8 {
            "Good"
        } else if score >= 0.7 {
            "Satisfactory"
        } else if score >= 0.6 {
            "Needs Improvement"
        } else {
            "Requires Significant Work"
        };

        format!(
            "{}: {} (Score: {:.1}%)",
            criterion,
            performance,
            score * 100.0
        )
    }

    fn generate_session_comment(&self, session: &LMSSession) -> String {
        let mut comment = format!(
            "VoiRS Session completed on {}. ",
            session
                .timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        if let Some(score) = &session.score {
            comment.push_str(&format!(
                "Overall performance: Pronunciation {:.1}%, Fluency {:.1}%, Quality {:.1}%. ",
                score.average_pronunciation * 100.0,
                score.average_fluency * 100.0,
                score.average_quality * 100.0
            ));
        }

        if !session.feedback.is_empty() {
            comment.push_str("Key feedback: ");
            let feedback_messages: Vec<String> = session
                .feedback
                .iter()
                .take(3)
                .map(|f| f.message.clone())
                .collect();
            comment.push_str(&feedback_messages.join("; "));
        }

        comment
    }

    fn generate_progress_report(
        &self,
        student_id: &str,
        user_progress: &UserProgress,
        sessions: &[LMSSession],
    ) -> Result<LMSProgressReport, LMSError> {
        let total_sessions = sessions.len() as u32;
        let time_spent = sessions
            .iter()
            .map(|s| s.duration.as_secs() as u32 / 60)
            .sum();

        let mut skill_breakdown = HashMap::new();

        // Calculate average scores by focus area
        if !sessions.is_empty() {
            let mut pronunciation_sum = 0.0;
            let mut fluency_sum = 0.0;
            let mut intonation_sum = 0.0;
            let mut count = 0;

            for session in sessions {
                if let Some(score) = &session.score {
                    pronunciation_sum += f64::from(score.average_pronunciation);
                    fluency_sum += f64::from(score.average_fluency);
                    intonation_sum += f64::from(score.average_quality); // Use quality as proxy for intonation
                    count += 1;
                }
            }

            if count > 0 {
                skill_breakdown.insert(
                    FocusArea::Pronunciation,
                    pronunciation_sum / f64::from(count),
                );
                skill_breakdown.insert(FocusArea::Fluency, fluency_sum / f64::from(count));
                skill_breakdown.insert(FocusArea::Intonation, intonation_sum / f64::from(count));
            }
        }

        Ok(LMSProgressReport {
            student_id: student_id.to_string(),
            course_id: "unknown".to_string(), // Would need to be provided
            overall_progress: f64::from(user_progress.overall_skill_level), // Use actual skill level
            completion_percentage: if total_sessions >= 10 {
                100.0
            } else {
                f64::from(total_sessions) * 10.0
            },
            sessions_completed: total_sessions,
            time_spent_minutes: time_spent,
            skill_breakdown,
            achievements: vec![
                format!("Completed {} sessions", total_sessions),
                format!(
                    "Overall skill level: {:.1}%",
                    user_progress.overall_skill_level * 100.0
                ),
            ],
            generated_at: SystemTime::now(),
        })
    }

    fn validate_grade_submission(&self, submission: &GradeSubmission) -> Result<(), LMSError> {
        if submission.student_id.is_empty() {
            return Err(LMSError::DataValidationError(
                "Student ID cannot be empty".to_string(),
            ));
        }

        if submission.assignment_id.is_empty() {
            return Err(LMSError::DataValidationError(
                "Assignment ID cannot be empty".to_string(),
            ));
        }

        if submission.score < 0.0 || submission.score > submission.max_score {
            return Err(LMSError::DataValidationError(format!(
                "Score {} is out of range [0, {}]",
                submission.score, submission.max_score
            )));
        }

        Ok(())
    }
}

/// Build the error returned by every real-platform method when the crate is
/// compiled without the `microservices` feature (no HTTP client available).
#[cfg(feature = "microservices")]
fn feature_disabled_error() -> LMSError {
    LMSError::ConfigurationError(
        "the `microservices` feature (reqwest HTTP client) is not enabled".to_string(),
    )
}

/// Map a [`reqwest::Error`] to the appropriate [`LMSError`], distinguishing
/// timeouts from other transport failures.
#[cfg(feature = "microservices")]
fn map_reqwest_err(e: reqwest::Error) -> LMSError {
    if e.is_timeout() {
        LMSError::ConnectionTimeout
    } else {
        LMSError::NetworkError(e.to_string())
    }
}

/// Turn a non-2xx response into a typed [`LMSError`], carrying the real
/// response body instead of discarding it.
#[cfg(feature = "microservices")]
async fn ensure_lms_success(response: reqwest::Response) -> Result<reqwest::Response, LMSError> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(match status.as_u16() {
        401 | 403 => LMSError::AuthenticationFailed(format!("HTTP {status}: {body}")),
        429 => LMSError::RateLimitExceeded,
        _ => LMSError::NetworkError(format!("HTTP {status}: {body}")),
    })
}

/// Moodle web services report failures as HTTP 200 responses whose JSON
/// body contains an `exception` object rather than using HTTP status codes.
#[cfg(feature = "microservices")]
fn check_moodle_exception(value: &serde_json::Value) -> Result<(), LMSError> {
    if value.get("exception").is_some() {
        let message = value["message"]
            .as_str()
            .unwrap_or("Moodle web service returned an exception")
            .to_string();
        return Err(LMSError::AuthenticationFailed(message));
    }
    Ok(())
}

/// Convert a JSON id field to a `String`, accepting either a JSON string
/// (Moodle, most REST APIs) or a bare number (Canvas returns numeric IDs).
#[cfg(feature = "microservices")]
fn json_number_or_string(value: &serde_json::Value) -> Option<String> {
    value
        .as_str()
        .map(std::string::ToString::to_string)
        .or_else(|| value.as_u64().map(|n| n.to_string()))
        .or_else(|| value.as_i64().map(|n| n.to_string()))
}

/// Parse an ISO 8601 / RFC 3339 timestamp (Canvas's wire format for
/// `*_at` fields) into a [`SystemTime`], if present and well-formed.
#[cfg(feature = "microservices")]
fn parse_iso8601(value: &serde_json::Value) -> Option<SystemTime> {
    let raw = value.as_str()?;
    let dt = chrono::DateTime::parse_from_rfc3339(raw).ok()?;
    let secs = dt.timestamp();
    if secs < 0 {
        return None;
    }
    Some(SystemTime::UNIX_EPOCH + Duration::from_secs(secs as u64))
}

/// Parse a Canvas `Course` JSON object.
///
/// <https://canvas.instructure.com/doc/api/courses.html>
#[cfg(feature = "microservices")]
fn parse_canvas_course(
    value: &serde_json::Value,
    fallback_id: &str,
) -> Result<LMSCourse, LMSError> {
    let id = json_number_or_string(&value["id"]).unwrap_or_else(|| fallback_id.to_string());
    let name = value["name"]
        .as_str()
        .ok_or_else(|| {
            LMSError::DataValidationError(format!("Canvas course {id} response is missing 'name'"))
        })?
        .to_string();

    Ok(LMSCourse {
        id,
        name,
        course_code: value["course_code"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        term: value["term"]["name"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        start_date: parse_iso8601(&value["start_at"]),
        end_date: parse_iso8601(&value["end_at"]),
        enrollment_term_id: json_number_or_string(&value["enrollment_term_id"]),
        published: value["workflow_state"].as_str() == Some("available"),
    })
}

/// Parse a Canvas `User` JSON object as returned from the
/// `/courses/:id/students` endpoint.
#[cfg(feature = "microservices")]
fn parse_canvas_student(
    value: &serde_json::Value,
    course_id: &str,
) -> Result<LMSStudent, LMSError> {
    let id = json_number_or_string(&value["id"]).ok_or_else(|| {
        LMSError::DataValidationError("Canvas student response is missing 'id'".to_string())
    })?;
    let name = value["name"]
        .as_str()
        .ok_or_else(|| {
            LMSError::DataValidationError(format!("Canvas student {id} is missing 'name'"))
        })?
        .to_string();

    Ok(LMSStudent {
        id,
        external_id: value["sis_user_id"].as_str().map(String::from),
        name,
        // Only present when the request includes `?include[]=email` and the
        // caller has permission to view it.
        email: value["email"].as_str().unwrap_or_default().to_string(),
        course_id: course_id.to_string(),
        // This endpoint only ever returns actively-enrolled students, and by
        // definition every entry it returns is a student.
        enrollment_status: "active".to_string(),
        role: "student".to_string(),
    })
}

/// Parse a Canvas `Assignment` JSON object.
///
/// <https://canvas.instructure.com/doc/api/assignments.html>
#[cfg(feature = "microservices")]
fn parse_canvas_assignment(
    value: &serde_json::Value,
    course_id: &str,
) -> Result<LMSAssignment, LMSError> {
    let id = json_number_or_string(&value["id"]).ok_or_else(|| {
        LMSError::DataValidationError("Canvas assignment response is missing 'id'".to_string())
    })?;
    let name = value["name"]
        .as_str()
        .ok_or_else(|| {
            LMSError::DataValidationError(format!("Canvas assignment {id} is missing 'name'"))
        })?
        .to_string();

    let submission_types = value["submission_types"]
        .as_array()
        .map(|types| {
            types
                .iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // Only present when the request includes `?include[]=rubric`; genuinely
    // absent otherwise, so an empty list (not a fabricated default rubric)
    // is the honest result.
    let grading_criteria = value["rubric"]
        .as_array()
        .map(|criteria| {
            criteria
                .iter()
                .filter_map(|c| {
                    Some(GradingCriterion {
                        name: c["description"].as_str()?.to_string(),
                        description: c["long_description"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        points: c["points"].as_f64().unwrap_or(0.0),
                        // Canvas has no concept of VoiRS focus areas.
                        focus_area: None,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(LMSAssignment {
        id,
        name,
        description: value["description"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        course_id: course_id.to_string(),
        max_points: value["points_possible"].as_f64().unwrap_or(0.0),
        due_date: parse_iso8601(&value["due_at"]),
        published: value["published"].as_bool().unwrap_or(false),
        submission_types,
        grading_criteria,
    })
}

/// Parse a Moodle `core_course_get_courses` course entry.
///
/// <https://docs.moodle.org/dev/Web_service_API_functions#core_course_get_courses>
#[cfg(feature = "microservices")]
fn parse_moodle_course(
    value: &serde_json::Value,
    fallback_id: &str,
) -> Result<LMSCourse, LMSError> {
    let id = json_number_or_string(&value["id"]).unwrap_or_else(|| fallback_id.to_string());
    let name = value["fullname"]
        .as_str()
        .ok_or_else(|| {
            LMSError::DataValidationError(format!(
                "Moodle course {id} response is missing 'fullname'"
            ))
        })?
        .to_string();

    let unix_time = |field: &serde_json::Value| {
        field
            .as_u64()
            .filter(|&secs| secs > 0)
            .map(|secs| SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
    };

    Ok(LMSCourse {
        id,
        name,
        course_code: value["shortname"].as_str().unwrap_or_default().to_string(),
        // Moodle has no separate "term" concept on this resource.
        term: String::new(),
        start_date: unix_time(&value["startdate"]),
        end_date: unix_time(&value["enddate"]),
        enrollment_term_id: None,
        published: value["visible"].as_u64().is_none_or(|v| v == 1),
    })
}

/// Render a [`LMSProgressReport`] as a human-readable message body for
/// delivery through an LMS messaging endpoint (e.g. Canvas Conversations).
#[cfg(feature = "microservices")]
fn format_progress_report_message(report: &LMSProgressReport) -> String {
    let mut lines = vec![
        "VoiRS Progress Report".to_string(),
        format!(
            "Overall progress: {:.1}% ({} sessions, {} minutes)",
            report.overall_progress * 100.0,
            report.sessions_completed,
            report.time_spent_minutes
        ),
        format!("Completion: {:.1}%", report.completion_percentage),
    ];

    if !report.skill_breakdown.is_empty() {
        lines.push("Skill breakdown:".to_string());
        for (area, score) in &report.skill_breakdown {
            lines.push(format!("  - {area:?}: {:.1}%", score * 100.0));
        }
    }

    if !report.achievements.is_empty() {
        lines.push("Achievements:".to_string());
        for achievement in &report.achievements {
            lines.push(format!("  - {achievement}"));
        }
    }

    lines.join("\n")
}

/// Rate limiter for API requests
struct RateLimiter {
    /// Maximum number of requests allowed
    max_requests: u32,
    /// Time window for rate limiting
    window_duration: Duration,
    /// Request timestamps
    requests: Vec<SystemTime>,
}

impl RateLimiter {
    fn new(max_requests: u32, window_duration: Duration) -> Self {
        Self {
            max_requests,
            window_duration,
            requests: Vec::new(),
        }
    }

    fn check_rate_limit(&mut self) -> Result<(), LMSError> {
        let now = SystemTime::now();
        let window_start = now - self.window_duration;

        // Remove old requests
        self.requests.retain(|&time| time > window_start);

        if self.requests.len() >= self.max_requests as usize {
            return Err(LMSError::RateLimitExceeded);
        }

        self.requests.push(now);
        Ok(())
    }
}

/// Cache for LMS data
struct LMSCache {
    /// Cached courses with timestamps
    courses: HashMap<String, (LMSCourse, SystemTime)>,
    /// Cache entry duration
    cache_duration: Duration,
}

impl LMSCache {
    fn new() -> Self {
        Self {
            courses: HashMap::new(),
            cache_duration: Duration::from_secs(300), // 5 minutes
        }
    }

    fn get_course(&self, course_id: &str) -> Option<&LMSCourse> {
        if let Some((course, timestamp)) = self.courses.get(course_id) {
            if timestamp.elapsed().unwrap_or(Duration::MAX) < self.cache_duration {
                return Some(course);
            }
        }
        None
    }

    fn cache_course(&mut self, course: LMSCourse) {
        self.courses
            .insert(course.id.clone(), (course, SystemTime::now()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lms_auth_config_default() {
        let config = LMSAuthConfig::default();
        assert_eq!(config.platform, LMSPlatform::Canvas);
        assert_eq!(config.timeout_seconds, 30);
    }

    #[test]
    fn test_lms_platform_display() {
        assert_eq!(LMSPlatform::Canvas.to_string(), "Canvas");
        assert_eq!(LMSPlatform::Blackboard.to_string(), "Blackboard");
        assert_eq!(
            LMSPlatform::Custom("MyLMS".to_string()).to_string(),
            "Custom: MyLMS"
        );
    }

    #[tokio::test]
    async fn test_lms_integration_manager_creation() {
        let config = LMSAuthConfig::default();
        let manager = LMSIntegrationManager::new(config);
        // Manager should be created successfully
    }

    #[tokio::test]
    async fn test_grade_submission_validation() {
        let config = LMSAuthConfig::default();
        let manager = LMSIntegrationManager::new(config);

        let valid_submission = GradeSubmission {
            student_id: "123".to_string(),
            course_id: "789".to_string(),
            assignment_id: "456".to_string(),
            score: 85.0,
            max_score: 100.0,
            comment: Some("Good work".to_string()),
            submission_time: SystemTime::now(),
            detailed_feedback: vec![],
        };

        assert!(manager.validate_grade_submission(&valid_submission).is_ok());

        let invalid_submission = GradeSubmission {
            student_id: "".to_string(),
            course_id: "789".to_string(),
            assignment_id: "456".to_string(),
            score: 85.0,
            max_score: 100.0,
            comment: None,
            submission_time: SystemTime::now(),
            detailed_feedback: vec![],
        };

        assert!(manager
            .validate_grade_submission(&invalid_submission)
            .is_err());
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(2, Duration::from_secs(1));

        // First two requests should succeed
        assert!(limiter.check_rate_limit().is_ok());
        assert!(limiter.check_rate_limit().is_ok());

        // Third request should fail
        assert!(matches!(
            limiter.check_rate_limit(),
            Err(LMSError::RateLimitExceeded)
        ));
    }

    #[test]
    fn test_lms_cache() {
        let mut cache = LMSCache::new();

        let course = LMSCourse {
            id: "123".to_string(),
            name: "Test Course".to_string(),
            course_code: "TEST101".to_string(),
            term: "Fall 2024".to_string(),
            start_date: None,
            end_date: None,
            enrollment_term_id: None,
            published: true,
        };

        cache.cache_course(course.clone());

        let cached_course = cache.get_course("123");
        assert!(cached_course.is_some());
        assert_eq!(cached_course.unwrap().name, "Test Course");
    }

    // --- Fail-closed behavior -------------------------------------------

    #[tokio::test]
    async fn test_unconfigured_manager_fails_closed_without_network() {
        // Default config has an empty base_url/api_key: every real entry
        // point must reject the call locally, never attempt a request nor
        // fabricate a success.
        let mut manager = LMSIntegrationManager::new(LMSAuthConfig::default());

        assert!(matches!(
            manager.authenticate().await,
            Err(LMSError::ConfigurationError(_))
        ));
        assert!(matches!(
            manager.get_course("123").await,
            Err(LMSError::ConfigurationError(_))
        ));
        assert!(matches!(
            manager.get_course_students("123").await,
            Err(LMSError::ConfigurationError(_))
        ));
        assert!(matches!(
            manager.get_course_assignments("123").await,
            Err(LMSError::ConfigurationError(_))
        ));
    }

    #[tokio::test]
    async fn test_unimplemented_platforms_fail_closed_not_ok() {
        for platform in [
            LMSPlatform::D2L,
            LMSPlatform::Schoology,
            LMSPlatform::Sakai,
            LMSPlatform::Custom("Acme LMS".to_string()),
        ] {
            let config = LMSAuthConfig {
                platform,
                api_key: "key".to_string(),
                base_url: "https://lms.example.com".to_string(),
                ..LMSAuthConfig::default()
            };
            let mut manager = LMSIntegrationManager::new(config);
            let result = manager.authenticate().await;
            assert!(
                result.is_err(),
                "unimplemented platform must never report Ok"
            );
        }
    }

    #[test]
    fn test_session_to_grade_submission_uses_assignment_course_id() {
        let config = LMSAuthConfig::default();
        let manager = LMSIntegrationManager::new(config);

        let assignment = LMSAssignment {
            id: "assign-1".to_string(),
            name: "Test".to_string(),
            description: String::new(),
            course_id: "course-77".to_string(),
            max_points: 100.0,
            due_date: None,
            published: true,
            submission_types: vec![],
            grading_criteria: vec![],
        };
        let session = LMSSession {
            timestamp: SystemTime::now(),
            duration: Duration::from_secs(120),
            score: None,
            feedback: vec![],
        };

        let submission = manager
            .session_to_grade_submission("student-1", "assign-1", &session, &assignment)
            .unwrap();

        assert_eq!(submission.course_id, "course-77");
        assert_eq!(submission.assignment_id, "assign-1");
    }

    // --- Real response parsing (offline, JSON fixtures) ------------------

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_canvas_course_from_real_response_shape() {
        let value = serde_json::json!({
            "id": 12345,
            "name": "Speech Communication 101",
            "course_code": "COMM101",
            "workflow_state": "available",
            "term": { "name": "Fall 2024" },
            "start_at": "2024-08-26T00:00:00Z",
            "end_at": null,
            "enrollment_term_id": 42
        });

        let course = parse_canvas_course(&value, "unused").unwrap();
        assert_eq!(course.id, "12345");
        assert_eq!(course.name, "Speech Communication 101");
        assert_eq!(course.course_code, "COMM101");
        assert_eq!(course.term, "Fall 2024");
        assert!(course.published);
        assert!(course.start_date.is_some());
        assert_eq!(course.enrollment_term_id.as_deref(), Some("42"));

        // A different response must produce genuinely different data.
        let other = serde_json::json!({
            "id": 999,
            "name": "Advanced Diction",
            "workflow_state": "unpublished",
            "course_code": "DICT200"
        });
        let other_course = parse_canvas_course(&other, "unused").unwrap();
        assert_ne!(course.name, other_course.name);
        assert!(!other_course.published);
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_canvas_student_from_real_response_shape() {
        let value = serde_json::json!({
            "id": 1001,
            "name": "Grace Hopper",
            "email": "grace@example.edu",
            "sis_user_id": "sis-1001"
        });

        let student = parse_canvas_student(&value, "course-1").unwrap();
        assert_eq!(student.id, "1001");
        assert_eq!(student.name, "Grace Hopper");
        assert_eq!(student.email, "grace@example.edu");
        assert_eq!(student.course_id, "course-1");
        assert_eq!(student.role, "student");
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_canvas_assignment_with_rubric() {
        let value = serde_json::json!({
            "id": 55,
            "name": "Pronunciation Assessment",
            "description": "Read the passage aloud",
            "points_possible": 100.0,
            "published": true,
            "submission_types": ["online_upload", "media_recording"],
            "due_at": "2024-09-01T23:59:00Z",
            "rubric": [
                { "description": "Clarity", "long_description": "How clear", "points": 40.0 },
                { "description": "Fluency", "points": 60.0 }
            ]
        });

        let assignment = parse_canvas_assignment(&value, "course-9").unwrap();
        assert_eq!(assignment.id, "55");
        assert_eq!(assignment.course_id, "course-9");
        assert_eq!(assignment.max_points, 100.0);
        assert_eq!(assignment.submission_types.len(), 2);
        assert_eq!(assignment.grading_criteria.len(), 2);
        assert_eq!(assignment.grading_criteria[0].name, "Clarity");
        assert!(assignment.due_date.is_some());

        // No rubric included: honest empty list, not a fabricated default.
        let no_rubric = serde_json::json!({
            "id": 56,
            "name": "Untitled",
            "points_possible": 10.0
        });
        let assignment2 = parse_canvas_assignment(&no_rubric, "course-9").unwrap();
        assert!(assignment2.grading_criteria.is_empty());
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_canvas_course_missing_name_fails_closed() {
        let value = serde_json::json!({ "id": 1, "workflow_state": "available" });
        assert!(matches!(
            parse_canvas_course(&value, "1"),
            Err(LMSError::DataValidationError(_))
        ));
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_moodle_course_from_real_response_shape() {
        let value = serde_json::json!({
            "id": 7,
            "fullname": "Pronunciation Practice",
            "shortname": "PRON101",
            "startdate": 1_704_067_200_u64,
            "enddate": 0,
            "visible": 1
        });

        let course = parse_moodle_course(&value, "unused").unwrap();
        assert_eq!(course.id, "7");
        assert_eq!(course.name, "Pronunciation Practice");
        assert_eq!(course.course_code, "PRON101");
        assert!(course.published);
        assert!(course.start_date.is_some());
        // enddate of 0 means "no end date" per Moodle's convention.
        assert!(course.end_date.is_none());
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_check_moodle_exception_detects_embedded_errors() {
        let error_body = serde_json::json!({
            "exception": "moodle_exception",
            "errorcode": "invalidtoken",
            "message": "Invalid token - token not found"
        });
        let result = check_moodle_exception(&error_body);
        assert!(
            matches!(result, Err(LMSError::AuthenticationFailed(msg)) if msg.contains("Invalid token"))
        );

        let ok_body = serde_json::json!({ "sitename": "Test Site" });
        assert!(check_moodle_exception(&ok_body).is_ok());
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_json_number_or_string() {
        assert_eq!(
            json_number_or_string(&serde_json::json!(42)),
            Some("42".to_string())
        );
        assert_eq!(
            json_number_or_string(&serde_json::json!("abc")),
            Some("abc".to_string())
        );
        assert_eq!(json_number_or_string(&serde_json::json!(null)), None);
    }

    /// Live end-to-end tests are opt-in: set `VOIRS_TEST_CANVAS_BASE_URL`
    /// and `VOIRS_TEST_CANVAS_TOKEN` (and optionally
    /// `VOIRS_TEST_CANVAS_COURSE_ID`) to exercise this against a real Canvas
    /// instance. The default offline test run never touches the network.
    #[cfg(feature = "microservices")]
    #[tokio::test]
    async fn test_canvas_get_course_live() {
        let (Ok(base_url), Ok(api_key)) = (
            std::env::var("VOIRS_TEST_CANVAS_BASE_URL"),
            std::env::var("VOIRS_TEST_CANVAS_TOKEN"),
        ) else {
            eprintln!(
                "skipping test_canvas_get_course_live: set VOIRS_TEST_CANVAS_BASE_URL and \
                 VOIRS_TEST_CANVAS_TOKEN to run this against a real Canvas instance"
            );
            return;
        };
        let course_id =
            std::env::var("VOIRS_TEST_CANVAS_COURSE_ID").unwrap_or_else(|_| "1".to_string());

        let config = LMSAuthConfig {
            platform: LMSPlatform::Canvas,
            api_key,
            base_url,
            ..LMSAuthConfig::default()
        };
        let mut manager = LMSIntegrationManager::new(config);
        let course = manager
            .get_course(&course_id)
            .await
            .expect("live Canvas API call should succeed with a valid token");
        assert_eq!(course.id, course_id);
    }
}
