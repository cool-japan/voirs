//! Google Classroom Integration
//!
//! This module provides comprehensive integration with Google Classroom API,
//! enabling assignment management, grade passback, student roster synchronization,
//! and real-time class updates for pronunciation training exercises.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[cfg(feature = "microservices")]
use reqwest::Client;

#[cfg(feature = "microservices")]
use chrono::DateTime;

/// Google Classroom integration errors
#[derive(Error, Debug, Clone)]
pub enum ClassroomError {
    /// Authentication failed
    #[error("Google Classroom authentication failed: {message}")]
    AuthFailed {
        /// Error message
        message: String,
    },

    /// API error
    #[error("Google Classroom API error: {message}")]
    ApiError {
        /// Error message
        message: String,
    },

    /// Course not found
    #[error("Course not found: {course_id}")]
    CourseNotFound {
        /// Course identifier
        course_id: String,
    },

    /// Assignment not found
    #[error("Assignment not found: {assignment_id}")]
    AssignmentNotFound {
        /// Assignment identifier
        assignment_id: String,
    },

    /// Student not found
    #[error("Student not found: {student_id}")]
    StudentNotFound {
        /// Student identifier
        student_id: String,
    },

    /// Permission denied
    #[error("Permission denied: {action}")]
    PermissionDenied {
        /// Action that was denied
        action: String,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {message}")]
    InvalidConfig {
        /// Error message
        message: String,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {seconds} seconds")]
    RateLimitExceeded {
        /// Seconds to wait before retry
        seconds: u64,
    },
}

/// Result type for Google Classroom operations
pub type ClassroomResult<T> = Result<T, ClassroomError>;

/// Google Classroom course state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CourseState {
    /// Active course
    Active,
    /// Archived course
    Archived,
    /// Provisioned but not active
    Provisioned,
    /// Declined invitation
    Declined,
    /// Suspended
    Suspended,
}

/// Google Classroom user role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserRole {
    /// Teacher/instructor
    Teacher,
    /// Student
    Student,
    /// Course owner
    Owner,
}

/// Google Classroom course information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Course {
    /// Course ID
    pub id: String,
    /// Course name
    pub name: String,
    /// Course section
    pub section: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Course state
    pub state: CourseState,
    /// Room location
    pub room: Option<String>,
    /// Owner ID
    pub owner_id: String,
    /// Creation time (Unix timestamp)
    pub creation_time: u64,
    /// Update time (Unix timestamp)
    pub update_time: u64,
    /// Enrollment code
    pub enrollment_code: Option<String>,
    /// Calendar ID
    pub calendar_id: Option<String>,
}

/// Google Classroom assignment/coursework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseWork {
    /// Assignment ID
    pub id: String,
    /// Course ID
    pub course_id: String,
    /// Title
    pub title: String,
    /// Description
    pub description: Option<String>,
    /// Materials (URLs, drive files, etc.)
    pub materials: Vec<Material>,
    /// State (published, draft, deleted)
    pub state: String,
    /// Maximum points
    pub max_points: Option<f64>,
    /// Due date (ISO 8601)
    pub due_date: Option<String>,
    /// Creation time (Unix timestamp)
    pub creation_time: u64,
    /// Update time (Unix timestamp)
    pub update_time: u64,
    /// Associated Grade category
    pub grade_category: Option<String>,
}

/// Assignment material
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    /// Material type
    pub material_type: MaterialType,
    /// Title
    pub title: String,
    /// URL or resource identifier
    pub resource: String,
}

/// Material type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialType {
    /// Link to external resource
    Link,
    /// Google Drive file
    DriveFile,
    /// `YouTube` video
    YouTubeVideo,
    /// Google Form
    Form,
}

/// Student submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Submission {
    /// Submission ID
    pub id: String,
    /// Course ID
    pub course_id: String,
    /// Course work ID
    pub course_work_id: String,
    /// User ID
    pub user_id: String,
    /// State (new, created, `turned_in`, returned, reclaimed)
    pub state: String,
    /// Assigned grade
    pub assigned_grade: Option<f64>,
    /// Draft grade
    pub draft_grade: Option<f64>,
    /// Creation time (Unix timestamp)
    pub creation_time: u64,
    /// Update time (Unix timestamp)
    pub update_time: u64,
    /// Late indicator
    pub late: bool,
}

/// Grade submission request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradeSubmission {
    /// Course ID
    pub course_id: String,
    /// Course work ID
    pub course_work_id: String,
    /// Student ID
    pub student_id: String,
    /// Grade (0.0 - `max_points`)
    pub grade: f64,
    /// Feedback comment
    pub comment: Option<String>,
}

/// Student roster entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Student {
    /// Course ID
    pub course_id: String,
    /// User ID
    pub user_id: String,
    /// Profile information
    pub profile: UserProfile,
}

/// User profile information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// User ID
    pub id: String,
    /// Full name
    pub name: String,
    /// Email address
    pub email: String,
    /// Photo URL
    pub photo_url: Option<String>,
}

/// Google Classroom configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassroomConfig {
    /// OAuth 2.0 client ID
    pub client_id: String,
    /// OAuth 2.0 client secret
    pub client_secret: String,
    /// Redirect URI for OAuth flow
    pub redirect_uri: String,
    /// Access token (obtained via OAuth)
    pub access_token: Option<String>,
    /// Refresh token
    pub refresh_token: Option<String>,
    /// Token expiry time (Unix timestamp)
    pub token_expiry: Option<u64>,
}

/// Google Classroom client
pub struct ClassroomClient {
    config: ClassroomConfig,
    #[cfg(feature = "microservices")]
    http_client: Client,
}

impl ClassroomClient {
    /// Create a new Google Classroom client
    #[must_use]
    pub fn new(config: ClassroomConfig) -> Self {
        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        #[cfg(feature = "microservices")]
        voirs_sdk::ensure_crypto_provider();
        Self {
            config,
            #[cfg(feature = "microservices")]
            http_client: Client::new(),
        }
    }

    /// Check if client is authenticated
    #[must_use]
    pub fn is_authenticated(&self) -> bool {
        self.config.access_token.is_some()
    }

    /// Get authorization URL for OAuth flow
    #[must_use]
    pub fn get_auth_url(&self) -> String {
        let scopes = [
            "https://www.googleapis.com/auth/classroom.courses.readonly",
            "https://www.googleapis.com/auth/classroom.rosters.readonly",
            "https://www.googleapis.com/auth/classroom.coursework.students",
            "https://www.googleapis.com/auth/classroom.student-submissions.students.readonly",
        ]
        .join("%20");

        format!(
            "https://accounts.google.com/o/oauth2/v2/auth?client_id={}&redirect_uri={}&response_type=code&scope={}&access_type=offline",
            self.config.client_id,
            urlencoding::encode(&self.config.redirect_uri),
            scopes
        )
    }

    /// Exchange authorization code for access token
    pub async fn exchange_code(&mut self, code: &str) -> ClassroomResult<()> {
        #[cfg(feature = "microservices")]
        {
            let params = [
                ("code", code),
                ("client_id", &self.config.client_id),
                ("client_secret", &self.config.client_secret),
                ("redirect_uri", &self.config.redirect_uri),
                ("grant_type", "authorization_code"),
            ];

            let response = self
                .http_client
                .post("https://oauth2.googleapis.com/token")
                .form(&params)
                .send()
                .await
                .map_err(|e| ClassroomError::ApiError {
                    message: e.to_string(),
                })?;

            let token_response: serde_json::Value =
                response
                    .json()
                    .await
                    .map_err(|e| ClassroomError::ApiError {
                        message: e.to_string(),
                    })?;

            self.config.access_token = token_response["access_token"]
                .as_str()
                .map(std::string::ToString::to_string);

            self.config.refresh_token = token_response["refresh_token"]
                .as_str()
                .map(std::string::ToString::to_string);

            if let Some(expires_in) = token_response["expires_in"].as_u64() {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("value should be present")
                    .as_secs();
                self.config.token_expiry = Some(now + expires_in);
            }
        }

        Ok(())
    }

    /// List courses for the authenticated user
    pub async fn list_courses(&self, teacher_only: bool) -> ClassroomResult<Vec<Course>> {
        if !self.is_authenticated() {
            return Err(ClassroomError::AuthFailed {
                message: "Not authenticated".to_string(),
            });
        }

        #[cfg(feature = "microservices")]
        {
            let mut url = "https://classroom.googleapis.com/v1/courses".to_string();
            if teacher_only {
                url.push_str("?teacherId=me");
            }

            let response = self
                .http_client
                .get(&url)
                .bearer_auth(
                    self.config
                        .access_token
                        .as_ref()
                        .expect("value should be present"),
                )
                .send()
                .await
                .map_err(|e| ClassroomError::ApiError {
                    message: e.to_string(),
                })?;

            let data = parse_json_body(ensure_success(response).await?).await?;

            data["courses"]
                .as_array()
                .map(std::vec::Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .map(parse_course)
                .collect()
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ClassroomError::InvalidConfig {
                message: "the `microservices` feature (reqwest HTTP client) is not enabled"
                    .to_string(),
            })
        }
    }

    /// Get course by ID
    pub async fn get_course(&self, course_id: &str) -> ClassroomResult<Course> {
        if !self.is_authenticated() {
            return Err(ClassroomError::AuthFailed {
                message: "Not authenticated".to_string(),
            });
        }

        #[cfg(feature = "microservices")]
        {
            let url = format!("https://classroom.googleapis.com/v1/courses/{course_id}");

            let response = self
                .http_client
                .get(&url)
                .bearer_auth(
                    self.config
                        .access_token
                        .as_ref()
                        .expect("value should be present"),
                )
                .send()
                .await
                .map_err(|e| ClassroomError::ApiError {
                    message: e.to_string(),
                })?;

            if response.status().as_u16() == 404 {
                return Err(ClassroomError::CourseNotFound {
                    course_id: course_id.to_string(),
                });
            }

            let data = parse_json_body(ensure_success(response).await?).await?;
            parse_course(&data)
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ClassroomError::InvalidConfig {
                message: "the `microservices` feature (reqwest HTTP client) is not enabled"
                    .to_string(),
            })
        }
    }

    /// List students in a course
    pub async fn list_students(&self, course_id: &str) -> ClassroomResult<Vec<Student>> {
        if !self.is_authenticated() {
            return Err(ClassroomError::AuthFailed {
                message: "Not authenticated".to_string(),
            });
        }

        #[cfg(feature = "microservices")]
        {
            let url = format!("https://classroom.googleapis.com/v1/courses/{course_id}/students");

            let response = self
                .http_client
                .get(&url)
                .bearer_auth(
                    self.config
                        .access_token
                        .as_ref()
                        .expect("value should be present"),
                )
                .send()
                .await
                .map_err(|e| ClassroomError::ApiError {
                    message: e.to_string(),
                })?;

            if response.status().as_u16() == 404 {
                return Err(ClassroomError::CourseNotFound {
                    course_id: course_id.to_string(),
                });
            }

            let data = parse_json_body(ensure_success(response).await?).await?;

            data["students"]
                .as_array()
                .map(std::vec::Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .map(parse_student)
                .collect()
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ClassroomError::InvalidConfig {
                message: "the `microservices` feature (reqwest HTTP client) is not enabled"
                    .to_string(),
            })
        }
    }

    /// Create course work (assignment)
    pub async fn create_course_work(&self, course_work: &CourseWork) -> ClassroomResult<String> {
        if !self.is_authenticated() {
            return Err(ClassroomError::AuthFailed {
                message: "Not authenticated".to_string(),
            });
        }

        #[cfg(feature = "microservices")]
        {
            let url = format!(
                "https://classroom.googleapis.com/v1/courses/{}/courseWork",
                course_work.course_id
            );

            let response = self
                .http_client
                .post(&url)
                .bearer_auth(
                    self.config
                        .access_token
                        .as_ref()
                        .expect("value should be present"),
                )
                .json(&course_work)
                .send()
                .await
                .map_err(|e| ClassroomError::ApiError {
                    message: e.to_string(),
                })?;

            let data = parse_json_body(ensure_success(response).await?).await?;

            json_id_to_string(&data["id"]).ok_or_else(|| ClassroomError::ApiError {
                message: "created courseWork response is missing an 'id' field".to_string(),
            })
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ClassroomError::InvalidConfig {
                message: "the `microservices` feature (reqwest HTTP client) is not enabled"
                    .to_string(),
            })
        }
    }

    /// Submit grade for student
    pub async fn submit_grade(&self, grade: &GradeSubmission) -> ClassroomResult<()> {
        if !self.is_authenticated() {
            return Err(ClassroomError::AuthFailed {
                message: "Not authenticated".to_string(),
            });
        }

        #[cfg(feature = "microservices")]
        {
            // Draft the grade first (assignedGrade can only be set by the app that
            // created the courseWork; draftGrade is always settable and is what the
            // Classroom UI surfaces to the teacher for review before posting).
            let url = format!(
                "https://classroom.googleapis.com/v1/courses/{}/courseWork/{}/studentSubmissions/{}?updateMask=draftGrade",
                grade.course_id, grade.course_work_id, grade.student_id
            );

            let response = self
                .http_client
                .patch(&url)
                .bearer_auth(
                    self.config
                        .access_token
                        .as_ref()
                        .expect("value should be present"),
                )
                .json(&serde_json::json!({
                    "draftGrade": grade.grade,
                }))
                .send()
                .await
                .map_err(|e| ClassroomError::ApiError {
                    message: e.to_string(),
                })?;

            ensure_success(response).await?;

            if let Some(comment) = &grade.comment {
                // Comments are delivered as a private (student <-> teacher) comment
                // on the submission rather than a field on the grade itself.
                let comment_url = format!(
                    "https://classroom.googleapis.com/v1/courses/{}/courseWork/{}/studentSubmissions/{}/addComment",
                    grade.course_id, grade.course_work_id, grade.student_id
                );

                let response = self
                    .http_client
                    .post(&comment_url)
                    .bearer_auth(
                        self.config
                            .access_token
                            .as_ref()
                            .expect("value should be present"),
                    )
                    .json(&serde_json::json!({ "text": comment }))
                    .send()
                    .await
                    .map_err(|e| ClassroomError::ApiError {
                        message: e.to_string(),
                    })?;

                ensure_success(response).await?;
            }

            Ok(())
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ClassroomError::InvalidConfig {
                message: "the `microservices` feature (reqwest HTTP client) is not enabled"
                    .to_string(),
            })
        }
    }

    /// Get student submissions for course work
    pub async fn get_submissions(
        &self,
        course_id: &str,
        course_work_id: &str,
    ) -> ClassroomResult<Vec<Submission>> {
        if !self.is_authenticated() {
            return Err(ClassroomError::AuthFailed {
                message: "Not authenticated".to_string(),
            });
        }

        #[cfg(feature = "microservices")]
        {
            let url = format!(
                "https://classroom.googleapis.com/v1/courses/{course_id}/courseWork/{course_work_id}/studentSubmissions"
            );

            let response = self
                .http_client
                .get(&url)
                .bearer_auth(
                    self.config
                        .access_token
                        .as_ref()
                        .expect("value should be present"),
                )
                .send()
                .await
                .map_err(|e| ClassroomError::ApiError {
                    message: e.to_string(),
                })?;

            if response.status().as_u16() == 404 {
                return Err(ClassroomError::AssignmentNotFound {
                    assignment_id: course_work_id.to_string(),
                });
            }

            let data = parse_json_body(ensure_success(response).await?).await?;

            data["studentSubmissions"]
                .as_array()
                .map(std::vec::Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .map(parse_submission)
                .collect()
        }

        #[cfg(not(feature = "microservices"))]
        {
            Err(ClassroomError::InvalidConfig {
                message: "the `microservices` feature (reqwest HTTP client) is not enabled"
                    .to_string(),
            })
        }
    }
}

/// Convert a JSON `id` value to a `String`, accepting either the string IDs
/// Google Classroom normally returns or a bare number (defensive: some
/// proxies/mocks re-encode numeric-looking IDs as JSON numbers).
#[cfg(feature = "microservices")]
fn json_id_to_string(value: &serde_json::Value) -> Option<String> {
    value
        .as_str()
        .map(std::string::ToString::to_string)
        .or_else(|| value.as_u64().map(|n| n.to_string()))
        .or_else(|| value.as_i64().map(|n| n.to_string()))
}

/// Extract the human-readable message from a Google API JSON error body,
/// e.g. `{"error": {"code": 404, "message": "...", "status": "NOT_FOUND"}}`.
#[cfg(feature = "microservices")]
fn extract_google_error_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value["error"]["message"]
        .as_str()
        .map(std::string::ToString::to_string)
}

/// Turn a non-2xx `reqwest::Response` into a typed [`ClassroomError`],
/// pulling out Google's structured error message when present instead of
/// discarding the body.
#[cfg(feature = "microservices")]
async fn ensure_success(response: reqwest::Response) -> ClassroomResult<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let message = extract_google_error_message(&body).unwrap_or(body);

    Err(match status.as_u16() {
        401 | 403 => ClassroomError::PermissionDenied { action: message },
        429 => ClassroomError::RateLimitExceeded { seconds: 60 },
        _ => ClassroomError::ApiError {
            message: format!("HTTP {status}: {message}"),
        },
    })
}

/// Parse a successful response body as JSON, mapping decode failures to a
/// typed error instead of letting a malformed body panic the caller.
#[cfg(feature = "microservices")]
async fn parse_json_body(response: reqwest::Response) -> ClassroomResult<serde_json::Value> {
    response.json().await.map_err(|e| ClassroomError::ApiError {
        message: format!("failed to parse response body as JSON: {e}"),
    })
}

/// Parse an RFC 3339 timestamp (Google Classroom's wire format for all
/// `*Time` fields) into Unix seconds.
#[cfg(feature = "microservices")]
fn parse_required_timestamp(value: &serde_json::Value, field: &str) -> ClassroomResult<u64> {
    let raw = value.as_str().ok_or_else(|| ClassroomError::ApiError {
        message: format!("response is missing required timestamp field '{field}'"),
    })?;
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.timestamp().max(0) as u64)
        .map_err(|e| ClassroomError::ApiError {
            message: format!("field '{field}' has an unparseable timestamp '{raw}': {e}"),
        })
}

/// Parse a `Course` resource from a Google Classroom API JSON response.
///
/// <https://developers.google.com/classroom/reference/rest/v1/courses#Course>
#[cfg(feature = "microservices")]
fn parse_course(value: &serde_json::Value) -> ClassroomResult<Course> {
    let id = json_id_to_string(&value["id"]).ok_or_else(|| ClassroomError::ApiError {
        message: "course response is missing required field 'id'".to_string(),
    })?;
    let name = value["name"]
        .as_str()
        .ok_or_else(|| ClassroomError::ApiError {
            message: format!("course {id} response is missing required field 'name'"),
        })?
        .to_string();
    let owner_id = value["ownerId"]
        .as_str()
        .ok_or_else(|| ClassroomError::ApiError {
            message: format!("course {id} response is missing required field 'ownerId'"),
        })?
        .to_string();
    let state = match value["courseState"].as_str() {
        Some("ACTIVE") => CourseState::Active,
        Some("ARCHIVED") => CourseState::Archived,
        Some("PROVISIONED") => CourseState::Provisioned,
        Some("DECLINED") => CourseState::Declined,
        Some("SUSPENDED") => CourseState::Suspended,
        other => {
            return Err(ClassroomError::ApiError {
                message: format!("course {id} has unrecognized or missing courseState: {other:?}"),
            })
        }
    };

    Ok(Course {
        id: id.clone(),
        name,
        section: value["section"].as_str().map(String::from),
        description: value["description"].as_str().map(String::from),
        state,
        room: value["room"].as_str().map(String::from),
        owner_id,
        creation_time: parse_required_timestamp(&value["creationTime"], "creationTime")?,
        update_time: parse_required_timestamp(&value["updateTime"], "updateTime")?,
        enrollment_code: value["enrollmentCode"].as_str().map(String::from),
        calendar_id: value["calendarId"].as_str().map(String::from),
    })
}

/// Parse a `Student` resource from a Google Classroom API JSON response.
///
/// <https://developers.google.com/classroom/reference/rest/v1/courses.students#Student>
#[cfg(feature = "microservices")]
fn parse_student(value: &serde_json::Value) -> ClassroomResult<Student> {
    let course_id =
        json_id_to_string(&value["courseId"]).ok_or_else(|| ClassroomError::ApiError {
            message: "student response is missing required field 'courseId'".to_string(),
        })?;
    let user_id = json_id_to_string(&value["userId"]).ok_or_else(|| ClassroomError::ApiError {
        message: "student response is missing required field 'userId'".to_string(),
    })?;
    let profile = &value["profile"];
    let id = json_id_to_string(&profile["id"]).unwrap_or_else(|| user_id.clone());
    let name = profile["name"]["fullName"]
        .as_str()
        .ok_or_else(|| ClassroomError::ApiError {
            message: format!("student {user_id} profile is missing 'name.fullName'"),
        })?
        .to_string();
    let email = profile["emailAddress"]
        .as_str()
        .ok_or_else(|| ClassroomError::ApiError {
            message: format!("student {user_id} profile is missing 'emailAddress'"),
        })?
        .to_string();

    Ok(Student {
        course_id,
        user_id,
        profile: UserProfile {
            id,
            name,
            email,
            photo_url: profile["photoUrl"].as_str().map(String::from),
        },
    })
}

/// Parse a `StudentSubmission` resource from a Google Classroom API JSON
/// response.
///
/// <https://developers.google.com/classroom/reference/rest/v1/courses.courseWork.studentSubmissions#StudentSubmission>
#[cfg(feature = "microservices")]
fn parse_submission(value: &serde_json::Value) -> ClassroomResult<Submission> {
    let id = json_id_to_string(&value["id"]).ok_or_else(|| ClassroomError::ApiError {
        message: "submission response is missing required field 'id'".to_string(),
    })?;
    let course_id =
        json_id_to_string(&value["courseId"]).ok_or_else(|| ClassroomError::ApiError {
            message: format!("submission {id} is missing required field 'courseId'"),
        })?;
    let course_work_id =
        json_id_to_string(&value["courseWorkId"]).ok_or_else(|| ClassroomError::ApiError {
            message: format!("submission {id} is missing required field 'courseWorkId'"),
        })?;
    let user_id = json_id_to_string(&value["userId"]).ok_or_else(|| ClassroomError::ApiError {
        message: format!("submission {id} is missing required field 'userId'"),
    })?;
    let state = value["state"]
        .as_str()
        .ok_or_else(|| ClassroomError::ApiError {
            message: format!("submission {id} is missing required field 'state'"),
        })?
        .to_string();

    Ok(Submission {
        id: id.clone(),
        course_id,
        course_work_id,
        user_id,
        state,
        assigned_grade: value["assignedGrade"].as_f64(),
        draft_grade: value["draftGrade"].as_f64(),
        creation_time: parse_required_timestamp(&value["creationTime"], "creationTime")?,
        update_time: parse_required_timestamp(&value["updateTime"], "updateTime")?,
        late: value["late"].as_bool().unwrap_or(false),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ClassroomConfig {
        ClassroomConfig {
            client_id: "test-client-id".to_string(),
            client_secret: "test-client-secret".to_string(),
            redirect_uri: "http://localhost:8080/callback".to_string(),
            access_token: Some("test-access-token".to_string()),
            refresh_token: Some("test-refresh-token".to_string()),
            token_expiry: Some(9999999999),
        }
    }

    #[test]
    fn test_client_creation() {
        let config = create_test_config();
        let client = ClassroomClient::new(config);
        assert!(client.is_authenticated());
    }

    #[test]
    fn test_auth_url_generation() {
        let config = ClassroomConfig {
            client_id: "test-client-id".to_string(),
            client_secret: "test-client-secret".to_string(),
            redirect_uri: "http://localhost:8080/callback".to_string(),
            access_token: None,
            refresh_token: None,
            token_expiry: None,
        };

        let client = ClassroomClient::new(config);
        let auth_url = client.get_auth_url();

        assert!(auth_url.contains("accounts.google.com/o/oauth2"));
        assert!(auth_url.contains("test-client-id"));
        assert!(auth_url.contains("classroom.courses.readonly"));
    }

    #[test]
    fn test_authentication_check() {
        let mut config = create_test_config();
        let client = ClassroomClient::new(config.clone());
        assert!(client.is_authenticated());

        config.access_token = None;
        let client2 = ClassroomClient::new(config);
        assert!(!client2.is_authenticated());
    }

    #[test]
    fn test_course_state_serialization() {
        let state = CourseState::Active;
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("Active"));

        let deserialized: CourseState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, CourseState::Active);
    }

    #[test]
    fn test_material_type() {
        let material = Material {
            material_type: MaterialType::Link,
            title: "Test Resource".to_string(),
            resource: "https://example.com".to_string(),
        };

        assert_eq!(material.material_type, MaterialType::Link);
        assert_eq!(material.title, "Test Resource");
    }

    #[test]
    fn test_grade_submission() {
        let grade = GradeSubmission {
            course_id: "course-123".to_string(),
            course_work_id: "work-456".to_string(),
            student_id: "student-789".to_string(),
            grade: 95.0,
            comment: Some("Excellent work!".to_string()),
        };

        assert_eq!(grade.grade, 95.0);
        assert!(grade.comment.is_some());
    }

    /// Live end-to-end test against the real Google Classroom API. Gated
    /// behind an environment variable so the default offline test run never
    /// makes a network call: set `VOIRS_TEST_GOOGLE_ACCESS_TOKEN` and
    /// `VOIRS_TEST_GOOGLE_COURSE_ID` to exercise this against a real course.
    #[cfg(feature = "microservices")]
    #[tokio::test]
    async fn test_get_course_live() {
        let (Ok(token), Ok(course_id)) = (
            std::env::var("VOIRS_TEST_GOOGLE_ACCESS_TOKEN"),
            std::env::var("VOIRS_TEST_GOOGLE_COURSE_ID"),
        ) else {
            eprintln!(
                "skipping test_get_course_live: set VOIRS_TEST_GOOGLE_ACCESS_TOKEN and \
                 VOIRS_TEST_GOOGLE_COURSE_ID to run this against a real account"
            );
            return;
        };

        let mut config = create_test_config();
        config.access_token = Some(token);
        let client = ClassroomClient::new(config);

        let course = client
            .get_course(&course_id)
            .await
            .expect("live Google Classroom API call should succeed with a valid token");
        assert_eq!(course.id, course_id);
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_course_from_real_response_shape() {
        let value = serde_json::json!({
            "id": "123456789",
            "name": "Speech Communication 101",
            "section": "Section A",
            "descriptionHeading": "Welcome",
            "description": "Practice pronunciation with VoiRS",
            "room": "201",
            "ownerId": "teacher-42",
            "creationTime": "2024-01-15T09:30:00.000Z",
            "updateTime": "2024-02-01T12:00:00.000Z",
            "enrollmentCode": "abc123",
            "courseState": "ACTIVE",
            "alternateLink": "https://classroom.google.com/c/123456789",
            "calendarId": "cal-1"
        });

        let course = parse_course(&value).unwrap();
        assert_eq!(course.id, "123456789");
        assert_eq!(course.name, "Speech Communication 101");
        assert_eq!(course.section.as_deref(), Some("Section A"));
        assert_eq!(course.state, CourseState::Active);
        assert_eq!(course.owner_id, "teacher-42");
        assert_eq!(course.enrollment_code.as_deref(), Some("abc123"));
        // 2024-01-15T09:30:00Z
        assert_eq!(course.creation_time, 1_705_310_400);

        // A different response must parse into genuinely different data,
        // proving this is real parsing rather than a fixed return value.
        let other = serde_json::json!({
            "id": "999",
            "name": "Advanced Pronunciation",
            "ownerId": "teacher-7",
            "creationTime": "2025-06-01T00:00:00.000Z",
            "updateTime": "2025-06-02T00:00:00.000Z",
            "courseState": "ARCHIVED"
        });
        let other_course = parse_course(&other).unwrap();
        assert_ne!(course.id, other_course.id);
        assert_ne!(course.name, other_course.name);
        assert_eq!(other_course.state, CourseState::Archived);
        assert_eq!(other_course.section, None);
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_course_missing_required_field_fails_closed() {
        // No 'id' field at all: must error, never silently substitute a
        // placeholder ID.
        let value = serde_json::json!({
            "name": "No ID Course",
            "ownerId": "teacher-1",
            "creationTime": "2024-01-01T00:00:00.000Z",
            "updateTime": "2024-01-01T00:00:00.000Z",
            "courseState": "ACTIVE"
        });
        let result = parse_course(&value);
        assert!(matches!(result, Err(ClassroomError::ApiError { .. })));

        // Unrecognized courseState must also fail closed rather than
        // guessing an arbitrary CourseState variant.
        let value = serde_json::json!({
            "id": "1",
            "name": "Weird State",
            "ownerId": "teacher-1",
            "creationTime": "2024-01-01T00:00:00.000Z",
            "updateTime": "2024-01-01T00:00:00.000Z",
            "courseState": "COURSE_STATE_UNSPECIFIED"
        });
        let result = parse_course(&value);
        assert!(matches!(result, Err(ClassroomError::ApiError { .. })));
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_student_from_real_response_shape() {
        let value = serde_json::json!({
            "courseId": "123",
            "userId": "student-99",
            "profile": {
                "id": "student-99",
                "name": { "givenName": "Ada", "familyName": "Lovelace", "fullName": "Ada Lovelace" },
                "emailAddress": "ada@example.edu",
                "photoUrl": "//example.com/photo.jpg"
            }
        });

        let student = parse_student(&value).unwrap();
        assert_eq!(student.course_id, "123");
        assert_eq!(student.user_id, "student-99");
        assert_eq!(student.profile.name, "Ada Lovelace");
        assert_eq!(student.profile.email, "ada@example.edu");
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_parse_submission_from_real_response_shape() {
        let value = serde_json::json!({
            "id": "sub-1",
            "courseId": "123",
            "courseWorkId": "work-1",
            "userId": "student-99",
            "state": "TURNED_IN",
            "assignedGrade": 95.0,
            "draftGrade": 90.0,
            "creationTime": "2024-03-01T00:00:00.000Z",
            "updateTime": "2024-03-02T00:00:00.000Z",
            "late": true
        });

        let submission = parse_submission(&value).unwrap();
        assert_eq!(submission.id, "sub-1");
        assert_eq!(submission.state, "TURNED_IN");
        assert_eq!(submission.assigned_grade, Some(95.0));
        assert_eq!(submission.draft_grade, Some(90.0));
        assert!(submission.late);
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_json_id_to_string_accepts_string_and_number() {
        assert_eq!(
            json_id_to_string(&serde_json::json!("abc")),
            Some("abc".to_string())
        );
        assert_eq!(
            json_id_to_string(&serde_json::json!(42)),
            Some("42".to_string())
        );
        assert_eq!(json_id_to_string(&serde_json::json!(null)), None);
    }

    #[cfg(feature = "microservices")]
    #[test]
    fn test_extract_google_error_message() {
        let body = r#"{"error": {"code": 404, "message": "Requested entity was not found.", "status": "NOT_FOUND"}}"#;
        assert_eq!(
            extract_google_error_message(body).as_deref(),
            Some("Requested entity was not found.")
        );
        assert_eq!(extract_google_error_message("not json"), None);
    }

    #[tokio::test]
    async fn test_list_courses_requires_auth() {
        let mut config = create_test_config();
        config.access_token = None;

        let client = ClassroomClient::new(config);
        let result = client.list_courses(false).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ClassroomError::AuthFailed { .. }
        ));
    }

    #[tokio::test]
    async fn test_list_students_requires_auth() {
        let mut config = create_test_config();
        config.access_token = None;

        let client = ClassroomClient::new(config);
        let result = client.list_students("course-123").await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ClassroomError::AuthFailed { .. }
        ));
    }

    #[tokio::test]
    async fn test_submit_grade_requires_auth() {
        let mut config = create_test_config();
        config.access_token = None;

        let client = ClassroomClient::new(config);
        let grade = GradeSubmission {
            course_id: "course-123".to_string(),
            course_work_id: "work-456".to_string(),
            student_id: "student-789".to_string(),
            grade: 95.0,
            comment: None,
        };

        let result = client.submit_grade(&grade).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ClassroomError::AuthFailed { .. }
        ));
    }
}
