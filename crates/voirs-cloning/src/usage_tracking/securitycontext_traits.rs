//! # SecurityContext - Trait Implementations
//!
//! This module contains trait implementations for `SecurityContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for SecurityContext {
    fn default() -> Self {
        SecurityContext {
            security_checks: Vec::new(),
            anomaly_detection: AnomalyDetectionResult {
                anomaly_score: 0.0,
                anomaly_threshold: 0.5,
                is_anomalous: false,
                anomaly_type: None,
                anomaly_details: String::new(),
            },
            rate_limiting: RateLimitingInfo {
                rate_limit_applied: false,
                current_usage: 0,
                rate_limit_threshold: 100,
                reset_time: None,
                quota_remaining: None,
            },
            threat_assessment: ThreatAssessment {
                threat_level: ThreatLevel::None,
                threat_indicators: Vec::new(),
                mitigation_actions: Vec::new(),
            },
            access_control: AccessControlInfo {
                permissions_granted: Vec::new(),
                permissions_denied: Vec::new(),
                access_level: AccessLevel::Basic,
                authentication_strength: AuthenticationStrength::Moderate,
            },
        }
    }
}
