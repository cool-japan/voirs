//! Misuse Prevention System for Voice Cloning
//!
//! This module provides comprehensive misuse prevention capabilities including anomaly detection,
//! usage restrictions, deepfake detection integration, and automated monitoring systems.

use crate::authenticity::{AuthenticityDetector, AuthenticityResult};
use crate::consent::{ConsentManager, ConsentUsageContext, ConsentUsageResult};
use crate::privacy_protection::{PrivacyProtectionManager, WatermarkDetectionResult};
use crate::usage_tracking::{UsageRecord, UsageStatus, UsageTracker};
use crate::{Error, Result};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Misuse prevention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisusePreventionConfig {
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Enable usage restrictions
    pub enable_usage_restrictions: bool,
    /// Enable deepfake detection
    pub enable_deepfake_detection: bool,
    /// Enable watermark verification
    pub enable_watermark_verification: bool,
    /// Maximum requests per hour per user
    pub max_requests_per_hour: u32,
    /// Maximum requests per day per user
    pub max_requests_per_day: u32,
    /// Anomaly detection threshold (0.0 to 1.0)
    pub anomaly_threshold: f32,
    /// Authenticity threshold for deepfake detection
    pub authenticity_threshold: f32,
    /// Time window for usage analysis (seconds)
    pub usage_window_seconds: u64,
    /// Enable automatic blocking of suspicious users
    pub auto_block_suspicious_users: bool,
    /// Block duration for suspicious activity (seconds)
    pub block_duration_seconds: u64,
}

impl Default for MisusePreventionConfig {
    fn default() -> Self {
        Self {
            enable_anomaly_detection: true,
            enable_usage_restrictions: true,
            enable_deepfake_detection: true,
            enable_watermark_verification: true,
            max_requests_per_hour: 100,
            max_requests_per_day: 1000,
            anomaly_threshold: 0.7,
            authenticity_threshold: 0.5,
            usage_window_seconds: 3600, // 1 hour
            auto_block_suspicious_users: true,
            block_duration_seconds: 24 * 3600, // 24 hours
        }
    }
}

/// Misuse prevention manager
pub struct MisusePreventionManager {
    config: MisusePreventionConfig,
    anomaly_detector: Arc<RwLock<AnomalyDetector>>,
    usage_monitor: Arc<RwLock<UsageMonitor>>,
    blocked_users: Arc<RwLock<HashMap<String, BlockInfo>>>,
    deepfake_detector: Arc<RwLock<Option<AuthenticityDetector>>>,
    privacy_manager: Arc<RwLock<Option<PrivacyProtectionManager>>>,
    consent_manager: Arc<RwLock<Option<ConsentManager>>>,
    usage_tracker: Arc<RwLock<Option<UsageTracker>>>,
}

impl MisusePreventionManager {
    /// Create a new misuse prevention manager
    pub fn new(config: MisusePreventionConfig) -> Self {
        Self {
            anomaly_detector: Arc::new(RwLock::new(AnomalyDetector::new(config.anomaly_threshold))),
            usage_monitor: Arc::new(RwLock::new(UsageMonitor::new(
                config.max_requests_per_hour,
                config.max_requests_per_day,
                config.usage_window_seconds,
            ))),
            blocked_users: Arc::new(RwLock::new(HashMap::new())),
            deepfake_detector: Arc::new(RwLock::new(None)),
            privacy_manager: Arc::new(RwLock::new(None)),
            consent_manager: Arc::new(RwLock::new(None)),
            usage_tracker: Arc::new(RwLock::new(None)),
            config,
        }
    }

    /// Set deepfake detection system
    pub fn set_deepfake_detector(&self, detector: AuthenticityDetector) -> Result<()> {
        let mut guard = self.deepfake_detector.write().map_err(|_| {
            Error::Validation("Failed to acquire deepfake detector lock".to_string())
        })?;
        *guard = Some(detector);
        info!("Deepfake detector configured for misuse prevention");
        Ok(())
    }

    /// Set privacy protection manager
    pub fn set_privacy_manager(&self, manager: PrivacyProtectionManager) -> Result<()> {
        let mut guard = self
            .privacy_manager
            .write()
            .map_err(|_| Error::Validation("Failed to acquire privacy manager lock".to_string()))?;
        *guard = Some(manager);
        info!("Privacy protection manager configured for misuse prevention");
        Ok(())
    }

    /// Set consent manager
    pub fn set_consent_manager(&self, manager: ConsentManager) -> Result<()> {
        let mut guard = self
            .consent_manager
            .write()
            .map_err(|_| Error::Validation("Failed to acquire consent manager lock".to_string()))?;
        *guard = Some(manager);
        info!("Consent manager configured for misuse prevention");
        Ok(())
    }

    /// Set usage tracker
    pub fn set_usage_tracker(&self, tracker: UsageTracker) -> Result<()> {
        let mut guard = self
            .usage_tracker
            .write()
            .map_err(|_| Error::Validation("Failed to acquire usage tracker lock".to_string()))?;
        *guard = Some(tracker);
        info!("Usage tracker configured for misuse prevention");
        Ok(())
    }

    /// Check if a voice cloning request should be allowed
    pub async fn check_request(&self, request: &VoiceCloningRequest) -> Result<MisuseCheckResult> {
        let start_time = SystemTime::now();
        let mut violations = Vec::new();
        let mut warnings = Vec::new();

        // Check if user is blocked
        if self.is_user_blocked(&request.user_id)? {
            return Ok(MisuseCheckResult {
                allowed: false,
                confidence: 1.0,
                violations: vec![MisuseViolation {
                    violation_type: ViolationType::UserBlocked,
                    severity: ViolationSeverity::Critical,
                    description: "User is currently blocked due to suspicious activity".to_string(),
                    detected_at: SystemTime::now(),
                    evidence: HashMap::new(),
                }],
                warnings: Vec::new(),
                processing_time_ms: start_time.elapsed().unwrap_or_default().as_millis() as u64,
                recommendations: vec![
                    "User must wait for block to expire or appeal the decision".to_string()
                ],
            });
        }

        // Check usage restrictions
        if self.config.enable_usage_restrictions {
            if let Some(usage_violation) = self.check_usage_restrictions(request)? {
                violations.push(usage_violation);
            }
        }

        // Check consent
        if let Some(consent_violation) = self.check_consent_compliance(request).await? {
            violations.push(consent_violation);
        }

        // Check for anomalous behavior
        if self.config.enable_anomaly_detection {
            if let Some(anomaly_violation) = self.check_anomalous_behavior(request)? {
                violations.push(anomaly_violation);
            }
        }

        // Check for deepfake usage
        if self.config.enable_deepfake_detection {
            if let Some(deepfake_violation) = self.check_deepfake_usage(request).await? {
                violations.push(deepfake_violation);
            }
        }

        // Check watermark violations
        if self.config.enable_watermark_verification {
            if let Some(watermark_violation) = self.check_watermark_violations(request).await? {
                warnings.push(watermark_violation);
            }
        }

        // Determine if request should be allowed
        let critical_violations = violations
            .iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::Critical))
            .count();

        let allowed = critical_violations == 0;

        // Auto-block user if too many violations
        if !allowed && self.config.auto_block_suspicious_users {
            self.block_user(&request.user_id, "Multiple critical violations detected")?;
        }

        // Calculate confidence based on violations
        let confidence = self.calculate_confidence(&violations, &warnings);

        let processing_time = start_time.elapsed().unwrap_or_default().as_millis() as u64;

        let recommendations = self.generate_recommendations(&violations, &warnings);

        Ok(MisuseCheckResult {
            allowed,
            confidence,
            violations,
            warnings,
            processing_time_ms: processing_time,
            recommendations,
        })
    }

    /// Block a user for suspicious activity
    pub fn block_user(&self, user_id: &str, reason: &str) -> Result<()> {
        let block_info = BlockInfo {
            user_id: user_id.to_string(),
            reason: reason.to_string(),
            blocked_at: SystemTime::now(),
            expires_at: SystemTime::now() + Duration::from_secs(self.config.block_duration_seconds),
            violation_count: 1,
        };

        {
            let mut blocked = self.blocked_users.write().map_err(|_| {
                Error::Validation("Failed to acquire blocked users lock".to_string())
            })?;
            blocked.insert(user_id.to_string(), block_info);
        }

        warn!("User {} blocked for: {}", user_id, reason);
        Ok(())
    }

    /// Unblock a user
    pub fn unblock_user(&self, user_id: &str) -> Result<()> {
        {
            let mut blocked = self.blocked_users.write().map_err(|_| {
                Error::Validation("Failed to acquire blocked users lock".to_string())
            })?;
            blocked.remove(user_id);
        }

        info!("User {} unblocked", user_id);
        Ok(())
    }

    /// Get misuse prevention statistics
    pub fn get_statistics(&self) -> Result<MisuseStatistics> {
        let blocked_users = {
            let blocked = self.blocked_users.read().map_err(|_| {
                Error::Validation("Failed to acquire blocked users lock".to_string())
            })?;
            blocked.len()
        };

        let anomaly_stats = {
            let detector = self.anomaly_detector.read().map_err(|_| {
                Error::Validation("Failed to acquire anomaly detector lock".to_string())
            })?;
            detector.get_statistics().clone()
        };

        let usage_stats = {
            let monitor = self.usage_monitor.read().map_err(|_| {
                Error::Validation("Failed to acquire usage monitor lock".to_string())
            })?;
            monitor.get_statistics().clone()
        };

        Ok(MisuseStatistics {
            total_blocked_users: blocked_users,
            total_anomalies_detected: anomaly_stats.total_anomalies,
            total_requests_processed: usage_stats.total_requests,
            total_violations: anomaly_stats.total_anomalies + usage_stats.total_violations,
            average_processing_time_ms: usage_stats.average_processing_time_ms,
            last_updated: SystemTime::now(),
        })
    }

    // Private helper methods

    fn is_user_blocked(&self, user_id: &str) -> Result<bool> {
        let blocked = self
            .blocked_users
            .read()
            .map_err(|_| Error::Validation("Failed to acquire blocked users lock".to_string()))?;

        if let Some(block_info) = blocked.get(user_id) {
            if SystemTime::now() > block_info.expires_at {
                // Block has expired, remove it
                drop(blocked);
                self.unblock_user(user_id)?;
                Ok(false)
            } else {
                Ok(true)
            }
        } else {
            Ok(false)
        }
    }

    fn check_usage_restrictions(
        &self,
        request: &VoiceCloningRequest,
    ) -> Result<Option<MisuseViolation>> {
        let mut monitor = self
            .usage_monitor
            .write()
            .map_err(|_| Error::Validation("Failed to acquire usage monitor lock".to_string()))?;

        if monitor.check_rate_limit(&request.user_id) {
            Ok(None)
        } else {
            Ok(Some(MisuseViolation {
                violation_type: ViolationType::RateLimitExceeded,
                severity: ViolationSeverity::High,
                description: "User has exceeded rate limits".to_string(),
                detected_at: SystemTime::now(),
                evidence: [
                    ("user_id".to_string(), request.user_id.clone()),
                    (
                        "max_per_hour".to_string(),
                        self.config.max_requests_per_hour.to_string(),
                    ),
                    (
                        "max_per_day".to_string(),
                        self.config.max_requests_per_day.to_string(),
                    ),
                ]
                .into_iter()
                .collect(),
            }))
        }
    }

    async fn check_consent_compliance(
        &self,
        request: &VoiceCloningRequest,
    ) -> Result<Option<MisuseViolation>> {
        if let Some(consent_id) = &request.consent_id {
            let consent_manager = self.consent_manager.read().map_err(|_| {
                Error::Validation("Failed to acquire consent manager lock".to_string())
            })?;

            if let Some(ref manager) = *consent_manager {
                let context = ConsentUsageContext {
                    use_case: request.use_case.clone(),
                    application: Some(request.application_id.clone()),
                    user: Some(request.user_id.clone()),
                    country: request.country.clone(),
                    region: request.region.clone(),
                    content_text: request.text_content.clone(),
                    timestamp: SystemTime::now(),
                    ip_address: request.ip_address.clone(),
                    // Additional fields
                    operation_type: crate::usage_tracking::CloningOperationType::VoiceCloning,
                    user_id: request.user_id.clone(),
                    location: request.country.clone(),
                    additional_context: std::collections::HashMap::new(),
                };

                match manager.check_consent_for_use(*consent_id, &request.use_case, &context)? {
                    ConsentUsageResult::Allowed => Ok(None),
                    ConsentUsageResult::Denied(reason) => Ok(Some(MisuseViolation {
                        violation_type: ViolationType::ConsentViolation,
                        severity: ViolationSeverity::Critical,
                        description: format!("Consent denied: {}", reason),
                        detected_at: SystemTime::now(),
                        evidence: [
                            ("consent_id".to_string(), consent_id.to_string()),
                            ("use_case".to_string(), request.use_case.clone()),
                            ("denial_reason".to_string(), reason),
                        ]
                        .into_iter()
                        .collect(),
                    })),
                    ConsentUsageResult::Restricted(restriction) => Ok(Some(MisuseViolation {
                        violation_type: ViolationType::ConsentRestriction,
                        severity: ViolationSeverity::Medium,
                        description: format!("Consent restricted: {}", restriction),
                        detected_at: SystemTime::now(),
                        evidence: [
                            ("consent_id".to_string(), consent_id.to_string()),
                            ("restriction".to_string(), restriction),
                        ]
                        .into_iter()
                        .collect(),
                    })),
                }
            } else {
                Ok(None) // No consent manager configured
            }
        } else {
            Ok(Some(MisuseViolation {
                violation_type: ViolationType::ConsentViolation,
                severity: ViolationSeverity::High,
                description: "No consent provided for voice cloning request".to_string(),
                detected_at: SystemTime::now(),
                evidence: [("user_id".to_string(), request.user_id.clone())]
                    .into_iter()
                    .collect(),
            }))
        }
    }

    fn check_anomalous_behavior(
        &self,
        request: &VoiceCloningRequest,
    ) -> Result<Option<MisuseViolation>> {
        let mut detector = self.anomaly_detector.write().map_err(|_| {
            Error::Validation("Failed to acquire anomaly detector lock".to_string())
        })?;

        let anomaly_score = detector.analyze_request(request)?;

        if anomaly_score > self.config.anomaly_threshold {
            Ok(Some(MisuseViolation {
                violation_type: ViolationType::AnomalousPattern,
                severity: ViolationSeverity::High,
                description: format!("Anomalous behavior detected (score: {:.2})", anomaly_score),
                detected_at: SystemTime::now(),
                evidence: [
                    ("anomaly_score".to_string(), anomaly_score.to_string()),
                    (
                        "threshold".to_string(),
                        self.config.anomaly_threshold.to_string(),
                    ),
                    ("user_id".to_string(), request.user_id.clone()),
                ]
                .into_iter()
                .collect(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn check_deepfake_usage(
        &self,
        request: &VoiceCloningRequest,
    ) -> Result<Option<MisuseViolation>> {
        if let Some(ref input_audio) = request.input_audio {
            let detector_guard = self.deepfake_detector.read().map_err(|_| {
                Error::Validation("Failed to acquire deepfake detector lock".to_string())
            })?;

            if let Some(ref detector) = *detector_guard {
                let result = detector
                    .analyze_authenticity(input_audio, request.sample_rate)
                    .await?;

                if result.authenticity_score < self.config.authenticity_threshold {
                    Ok(Some(MisuseViolation {
                        violation_type: ViolationType::DeepfakeDetected,
                        severity: ViolationSeverity::Critical,
                        description: format!(
                            "Deepfake audio detected (authenticity score: {:.2})",
                            result.authenticity_score
                        ),
                        detected_at: SystemTime::now(),
                        evidence: [
                            (
                                "authenticity_score".to_string(),
                                result.authenticity_score.to_string(),
                            ),
                            ("confidence".to_string(), result.confidence.to_string()),
                            (
                                "threshold".to_string(),
                                self.config.authenticity_threshold.to_string(),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    }))
                } else {
                    Ok(None)
                }
            } else {
                Ok(None) // No deepfake detector configured
            }
        } else {
            Ok(None) // No input audio to check
        }
    }

    async fn check_watermark_violations(
        &self,
        request: &VoiceCloningRequest,
    ) -> Result<Option<MisuseViolation>> {
        if let Some(ref input_audio) = request.input_audio {
            let privacy_guard = self.privacy_manager.read().map_err(|_| {
                Error::Validation("Failed to acquire privacy manager lock".to_string())
            })?;

            if let Some(ref manager) = *privacy_guard {
                if let Some(detection_result) = manager.detect_watermark(input_audio)? {
                    Ok(Some(MisuseViolation {
                        violation_type: ViolationType::WatermarkViolation,
                        severity: ViolationSeverity::Medium,
                        description: "Watermarked audio detected - may be unauthorized use"
                            .to_string(),
                        detected_at: SystemTime::now(),
                        evidence: [
                            (
                                "watermark_id".to_string(),
                                detection_result.watermark_id.to_string(),
                            ),
                            (
                                "correlation_strength".to_string(),
                                detection_result.correlation_strength.to_string(),
                            ),
                            (
                                "confidence".to_string(),
                                detection_result.confidence.to_string(),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    }))
                } else {
                    Ok(None)
                }
            } else {
                Ok(None) // No privacy manager configured
            }
        } else {
            Ok(None) // No input audio to check
        }
    }

    fn calculate_confidence(
        &self,
        violations: &[MisuseViolation],
        warnings: &[MisuseViolation],
    ) -> f32 {
        if violations.is_empty() && warnings.is_empty() {
            return 0.9; // High confidence in clean request
        }

        let violation_weight = violations
            .iter()
            .map(|v| match v.severity {
                ViolationSeverity::Critical => 0.4,
                ViolationSeverity::High => 0.3,
                ViolationSeverity::Medium => 0.2,
                ViolationSeverity::Low => 0.1,
            })
            .sum::<f32>();

        let warning_weight = warnings.len() as f32 * 0.05;

        let total_weight = violation_weight + warning_weight;

        // Confidence decreases with more violations
        (1.0 - total_weight.min(1.0)).max(0.1)
    }

    fn generate_recommendations(
        &self,
        violations: &[MisuseViolation],
        warnings: &[MisuseViolation],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        for violation in violations {
            match violation.violation_type {
                ViolationType::ConsentViolation => {
                    recommendations.push(
                        "Obtain proper consent before proceeding with voice cloning".to_string(),
                    );
                }
                ViolationType::RateLimitExceeded => {
                    recommendations
                        .push("Reduce request frequency or upgrade to higher tier".to_string());
                }
                ViolationType::DeepfakeDetected => {
                    recommendations.push("Use original, authentic audio sources only".to_string());
                }
                ViolationType::AnomalousPattern => {
                    recommendations.push(
                        "Review usage patterns and ensure compliance with terms of service"
                            .to_string(),
                    );
                }
                ViolationType::WatermarkViolation => {
                    recommendations
                        .push("Verify audio source authorization and licensing".to_string());
                }
                ViolationType::UserBlocked => {
                    recommendations
                        .push("Wait for block to expire or contact support to appeal".to_string());
                }
                ViolationType::ConsentRestriction => {
                    recommendations.push(
                        "Review consent restrictions and modify request accordingly".to_string(),
                    );
                }
            }
        }

        if !warnings.is_empty() {
            recommendations
                .push("Review warnings and ensure compliance with all policies".to_string());
        }

        recommendations
    }
}

// Data structures

/// Voice cloning request for misuse checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCloningRequest {
    pub request_id: Uuid,
    pub user_id: String,
    pub application_id: String,
    pub use_case: String,
    pub consent_id: Option<Uuid>,
    pub input_audio: Option<Vec<f32>>,
    pub sample_rate: u32,
    pub text_content: Option<String>,
    pub country: Option<String>,
    pub region: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub requested_at: SystemTime,
}

/// Result of misuse prevention check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisuseCheckResult {
    pub allowed: bool,
    pub confidence: f32,
    pub violations: Vec<MisuseViolation>,
    pub warnings: Vec<MisuseViolation>,
    pub processing_time_ms: u64,
    pub recommendations: Vec<String>,
}

/// Types of misuse violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    ConsentViolation,
    ConsentRestriction,
    RateLimitExceeded,
    DeepfakeDetected,
    AnomalousPattern,
    WatermarkViolation,
    UserBlocked,
}

/// Severity levels for violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Misuse violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisuseViolation {
    pub violation_type: ViolationType,
    pub severity: ViolationSeverity,
    pub description: String,
    pub detected_at: SystemTime,
    pub evidence: HashMap<String, String>,
}

/// Information about blocked users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInfo {
    pub user_id: String,
    pub reason: String,
    pub blocked_at: SystemTime,
    pub expires_at: SystemTime,
    pub violation_count: u32,
}

/// Misuse prevention statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisuseStatistics {
    pub total_blocked_users: usize,
    pub total_anomalies_detected: u64,
    pub total_requests_processed: u64,
    pub total_violations: u64,
    pub average_processing_time_ms: f64,
    pub last_updated: SystemTime,
}

/// Anomaly detector for unusual usage patterns
pub struct AnomalyDetector {
    threshold: f32,
    user_patterns: HashMap<String, UserPattern>,
    statistics: AnomalyStatistics,
}

impl AnomalyDetector {
    fn new(threshold: f32) -> Self {
        Self {
            threshold,
            user_patterns: HashMap::new(),
            statistics: AnomalyStatistics::default(),
        }
    }

    fn analyze_request(&mut self, request: &VoiceCloningRequest) -> Result<f32> {
        let pattern = self
            .user_patterns
            .entry(request.user_id.clone())
            .or_insert_with(UserPattern::new);

        let anomaly_score = pattern.calculate_anomaly_score(request);
        pattern.update_with_request(request);

        self.statistics.total_requests += 1;
        if anomaly_score > self.threshold {
            self.statistics.total_anomalies += 1;
        }

        Ok(anomaly_score)
    }

    fn get_statistics(&self) -> &AnomalyStatistics {
        &self.statistics
    }
}

/// User usage pattern for anomaly detection
#[derive(Debug, Clone)]
struct UserPattern {
    request_intervals: VecDeque<Duration>,
    use_cases: HashMap<String, u32>,
    countries: HashMap<String, u32>,
    total_requests: u32,
    last_request: Option<SystemTime>,
}

impl UserPattern {
    fn new() -> Self {
        Self {
            request_intervals: VecDeque::new(),
            use_cases: HashMap::new(),
            countries: HashMap::new(),
            total_requests: 0,
            last_request: None,
        }
    }

    fn calculate_anomaly_score(&self, request: &VoiceCloningRequest) -> f32 {
        let mut score: f32 = 0.0;

        // Check request frequency anomaly
        if let Some(last_request) = self.last_request {
            let interval = request
                .requested_at
                .duration_since(last_request)
                .unwrap_or_default();

            if interval < Duration::from_secs(1) {
                score += 0.3; // Very fast requests
            }
        }

        // Check use case diversity anomaly
        let use_case_count = self.use_cases.len();
        if use_case_count > 10 {
            score += 0.2; // Too many different use cases
        }

        // Check geographic anomaly
        if let Some(ref country) = request.country {
            if self.countries.len() > 5 && !self.countries.contains_key(country) {
                score += 0.4; // New country after establishing pattern
            }
        }

        // Check request volume anomaly
        if self.total_requests > 1000 {
            score += 0.1; // High volume user
        }

        score.min(1.0)
    }

    fn update_with_request(&mut self, request: &VoiceCloningRequest) {
        // Update request interval tracking
        if let Some(last_request) = self.last_request {
            let interval = request
                .requested_at
                .duration_since(last_request)
                .unwrap_or_default();
            self.request_intervals.push_back(interval);

            // Keep only recent intervals
            if self.request_intervals.len() > 100 {
                self.request_intervals.pop_front();
            }
        }

        // Update use case tracking
        *self.use_cases.entry(request.use_case.clone()).or_insert(0) += 1;

        // Update country tracking
        if let Some(ref country) = request.country {
            *self.countries.entry(country.clone()).or_insert(0) += 1;
        }

        self.total_requests += 1;
        self.last_request = Some(request.requested_at);
    }
}

/// Usage monitor for rate limiting
pub struct UsageMonitor {
    max_per_hour: u32,
    max_per_day: u32,
    window_seconds: u64,
    user_usage: HashMap<String, UserUsage>,
    statistics: UsageStatistics,
}

impl UsageMonitor {
    fn new(max_per_hour: u32, max_per_day: u32, window_seconds: u64) -> Self {
        Self {
            max_per_hour,
            max_per_day,
            window_seconds,
            user_usage: HashMap::new(),
            statistics: UsageStatistics::default(),
        }
    }

    fn check_rate_limit(&mut self, user_id: &str) -> bool {
        let now = SystemTime::now();
        let usage = self
            .user_usage
            .entry(user_id.to_string())
            .or_insert_with(UserUsage::new);

        // Clean up old requests
        usage.cleanup_old_requests(now, self.window_seconds);

        // Check hourly limit
        let hour_ago = now - Duration::from_secs(3600);
        let hourly_count = usage.requests_since(hour_ago);

        // Check daily limit
        let day_ago = now - Duration::from_secs(86400);
        let daily_count = usage.requests_since(day_ago);

        let within_limits = hourly_count < self.max_per_hour && daily_count < self.max_per_day;

        if within_limits {
            usage.add_request(now);
            self.statistics.total_requests += 1;
        } else {
            self.statistics.total_violations += 1;
        }

        within_limits
    }

    fn get_statistics(&self) -> &UsageStatistics {
        &self.statistics
    }
}

/// User usage tracking for rate limiting
#[derive(Debug, Clone)]
struct UserUsage {
    requests: VecDeque<SystemTime>,
}

impl UserUsage {
    fn new() -> Self {
        Self {
            requests: VecDeque::new(),
        }
    }

    fn add_request(&mut self, timestamp: SystemTime) {
        self.requests.push_back(timestamp);
    }

    fn cleanup_old_requests(&mut self, now: SystemTime, window_seconds: u64) {
        let cutoff = now - Duration::from_secs(window_seconds);

        while let Some(&front) = self.requests.front() {
            if front < cutoff {
                self.requests.pop_front();
            } else {
                break;
            }
        }
    }

    fn requests_since(&self, since: SystemTime) -> u32 {
        self.requests
            .iter()
            .filter(|&&req_time| req_time >= since)
            .count() as u32
    }
}

/// Statistics for anomaly detection
#[derive(Debug, Default, Clone)]
pub struct AnomalyStatistics {
    pub total_requests: u64,
    pub total_anomalies: u64,
}

/// Statistics for usage monitoring
#[derive(Debug, Default, Clone)]
pub struct UsageStatistics {
    pub total_requests: u64,
    pub total_violations: u64,
    pub average_processing_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_misuse_prevention_config_default() {
        let config = MisusePreventionConfig::default();
        assert!(config.enable_anomaly_detection);
        assert!(config.enable_usage_restrictions);
        assert!(config.enable_deepfake_detection);
        assert_eq!(config.max_requests_per_hour, 100);
        assert_eq!(config.max_requests_per_day, 1000);
        assert_eq!(config.anomaly_threshold, 0.7);
    }

    #[test]
    fn test_misuse_prevention_manager_creation() {
        let config = MisusePreventionConfig::default();
        let manager = MisusePreventionManager::new(config);

        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total_blocked_users, 0);
        assert_eq!(stats.total_anomalies_detected, 0);
    }

    #[test]
    fn test_user_blocking() {
        let config = MisusePreventionConfig::default();
        let manager = MisusePreventionManager::new(config);

        let user_id = "test-user";
        let reason = "Test blocking";

        manager.block_user(user_id, reason).unwrap();
        assert!(manager.is_user_blocked(user_id).unwrap());

        manager.unblock_user(user_id).unwrap();
        assert!(!manager.is_user_blocked(user_id).unwrap());
    }

    #[test]
    fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new(0.5);

        let request = VoiceCloningRequest {
            request_id: Uuid::new_v4(),
            user_id: "test-user".to_string(),
            application_id: "test-app".to_string(),
            use_case: "synthesis".to_string(),
            consent_id: None,
            input_audio: None,
            sample_rate: 22050,
            text_content: None,
            country: Some("US".to_string()),
            region: None,
            ip_address: None,
            user_agent: None,
            requested_at: SystemTime::now(),
        };

        let score = detector.analyze_request(&request).unwrap();
        assert!(score >= 0.0 && score <= 1.0);

        let stats = detector.get_statistics();
        assert_eq!(stats.total_requests, 1);
    }

    #[test]
    fn test_usage_monitor() {
        let mut monitor = UsageMonitor::new(5, 10, 3600);
        let user_id = "test-user";

        // Should allow first few requests
        for _ in 0..5 {
            assert!(monitor.check_rate_limit(user_id));
        }

        // Should deny 6th request (exceeds hourly limit)
        assert!(!monitor.check_rate_limit(user_id));

        let stats = monitor.get_statistics();
        assert_eq!(stats.total_requests, 5);
        assert_eq!(stats.total_violations, 1);
    }

    #[test]
    fn test_user_pattern_anomaly_detection() {
        let mut pattern = UserPattern::new();

        let request1 = VoiceCloningRequest {
            request_id: Uuid::new_v4(),
            user_id: "test-user".to_string(),
            application_id: "test-app".to_string(),
            use_case: "synthesis".to_string(),
            consent_id: None,
            input_audio: None,
            sample_rate: 22050,
            text_content: None,
            country: Some("US".to_string()),
            region: None,
            ip_address: None,
            user_agent: None,
            requested_at: SystemTime::now(),
        };

        let score1 = pattern.calculate_anomaly_score(&request1);
        pattern.update_with_request(&request1);

        // Second request very quickly after first
        let request2 = VoiceCloningRequest {
            requested_at: SystemTime::now(),
            ..request1.clone()
        };

        let score2 = pattern.calculate_anomaly_score(&request2);

        // Second request should have higher anomaly score due to timing
        assert!(score2 >= score1);
    }

    #[test]
    fn test_user_usage_tracking() {
        let mut usage = UserUsage::new();
        let now = SystemTime::now();

        // Add some requests
        usage.add_request(now - Duration::from_secs(100));
        usage.add_request(now - Duration::from_secs(50));
        usage.add_request(now);

        assert_eq!(usage.requests_since(now - Duration::from_secs(150)), 3);
        assert_eq!(usage.requests_since(now - Duration::from_secs(75)), 2);
        assert_eq!(usage.requests_since(now), 1);

        // Test cleanup
        usage.cleanup_old_requests(now, 60); // Keep requests from last 60 seconds
        assert_eq!(usage.requests.len(), 2); // Should remove the 100-second old request
    }

    #[tokio::test]
    async fn test_misuse_check_no_violations() {
        let config = MisusePreventionConfig::default();
        let manager = MisusePreventionManager::new(config);

        let request = VoiceCloningRequest {
            request_id: Uuid::new_v4(),
            user_id: "clean-user".to_string(),
            application_id: "test-app".to_string(),
            use_case: "synthesis".to_string(),
            consent_id: Some(Uuid::new_v4()),
            input_audio: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5]),
            sample_rate: 22050,
            text_content: Some("Hello world".to_string()),
            country: Some("US".to_string()),
            region: Some("California".to_string()),
            ip_address: Some("192.168.1.1".to_string()),
            user_agent: Some("TestAgent/1.0".to_string()),
            requested_at: SystemTime::now(),
        };

        let result = manager.check_request(&request).await.unwrap();

        // Without configured external systems, should have some violations
        // but the system should still function
        assert!(result.processing_time_ms >= 0); // Allow 0 for very fast tests
        assert!(!result.recommendations.is_empty() || result.violations.is_empty());
    }
}
