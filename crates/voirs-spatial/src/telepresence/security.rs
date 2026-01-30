//! Privacy, security, encryption, and authentication settings

use super::types::*;
use serde::{Deserialize, Serialize};

/// Privacy settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    /// Data collection preferences
    pub data_collection: DataCollectionSettings,

    /// Recording settings
    pub recording_settings: RecordingSettings,

    /// Anonymization settings
    pub anonymization: AnonymizationSettings,

    /// Consent management
    pub consent_management: ConsentManagementSettings,
}

/// Data collection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCollectionSettings {
    /// Allow telemetry data
    pub telemetry: bool,

    /// Allow analytics data
    pub analytics: bool,

    /// Allow performance data
    pub performance_data: bool,

    /// Allow usage statistics
    pub usage_statistics: bool,

    /// Data retention period (days)
    pub retention_period: u32,
}

/// Recording settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingSettings {
    /// Allow session recording
    pub allow_recording: bool,

    /// Require explicit consent
    pub explicit_consent: bool,

    /// Recording notification
    pub notification_required: bool,

    /// Local recording only
    pub local_only: bool,
}

/// Anonymization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizationSettings {
    /// Anonymize voice data
    pub voice_anonymization: bool,

    /// Anonymize position data
    pub position_anonymization: bool,

    /// Anonymization method
    pub method: AnonymizationMethod,

    /// Anonymization strength
    pub strength: AnonymizationStrength,
}

/// Consent management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentManagementSettings {
    /// Require consent for data processing
    pub data_processing_consent: bool,

    /// Require consent for recording
    pub recording_consent: bool,

    /// Require consent for analytics
    pub analytics_consent: bool,

    /// Consent withdrawal mechanism
    pub withdrawal_mechanism: ConsentWithdrawalMechanism,
}

/// Security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySettings {
    /// Encryption requirements
    pub encryption: EncryptionSettings,

    /// Authentication settings
    pub authentication: AuthenticationSettings,

    /// Rate limiting
    pub rate_limiting: RateLimitingSettings,

    /// Access control
    pub access_control: AccessControlSettings,
}

/// Encryption settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionSettings {
    /// Require encryption
    pub required: bool,

    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,

    /// Key exchange method
    pub key_exchange: KeyExchangeMethod,

    /// Key rotation interval (minutes)
    pub key_rotation_interval: u32,
}

/// Authentication settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationSettings {
    /// Authentication required
    pub required: bool,

    /// Authentication method
    pub method: AuthenticationMethod,

    /// Token expiry (minutes)
    pub token_expiry: u32,

    /// Multi-factor authentication
    pub mfa_required: bool,
}

/// Rate limiting settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingSettings {
    /// Enable rate limiting
    pub enabled: bool,

    /// Requests per minute
    pub requests_per_minute: u32,

    /// Bandwidth limit per user (kbps)
    pub bandwidth_limit: u32,

    /// Connection limit per IP
    pub connections_per_ip: u32,
}

/// Access control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlSettings {
    /// Whitelist/blacklist mode
    pub mode: AccessControlMode,

    /// Allowed IP ranges
    pub allowed_ips: Vec<String>,

    /// Blocked IP ranges
    pub blocked_ips: Vec<String>,

    /// Geographic restrictions
    pub geo_restrictions: Vec<String>,
}

// Default implementations

impl Default for PrivacySettings {
    fn default() -> Self {
        Self {
            data_collection: DataCollectionSettings::default(),
            recording_settings: RecordingSettings::default(),
            anonymization: AnonymizationSettings::default(),
            consent_management: ConsentManagementSettings::default(),
        }
    }
}

impl Default for DataCollectionSettings {
    fn default() -> Self {
        Self {
            telemetry: false,
            analytics: false,
            performance_data: true,
            usage_statistics: false,
            retention_period: 30, // 30 days
        }
    }
}

impl Default for RecordingSettings {
    fn default() -> Self {
        Self {
            allow_recording: false,
            explicit_consent: true,
            notification_required: true,
            local_only: true,
        }
    }
}

impl Default for AnonymizationSettings {
    fn default() -> Self {
        Self {
            voice_anonymization: false,
            position_anonymization: false,
            method: AnonymizationMethod::VoiceConversion,
            strength: AnonymizationStrength::Medium,
        }
    }
}

impl Default for ConsentManagementSettings {
    fn default() -> Self {
        Self {
            data_processing_consent: true,
            recording_consent: true,
            analytics_consent: false,
            withdrawal_mechanism: ConsentWithdrawalMechanism::Immediate,
        }
    }
}

impl Default for SecuritySettings {
    fn default() -> Self {
        Self {
            encryption: EncryptionSettings::default(),
            authentication: AuthenticationSettings::default(),
            rate_limiting: RateLimitingSettings::default(),
            access_control: AccessControlSettings::default(),
        }
    }
}

impl Default for EncryptionSettings {
    fn default() -> Self {
        Self {
            required: true,
            algorithm: EncryptionAlgorithm::DTLSSRTP,
            key_exchange: KeyExchangeMethod::ECDH,
            key_rotation_interval: 60, // 1 hour
        }
    }
}

impl Default for AuthenticationSettings {
    fn default() -> Self {
        Self {
            required: false,
            method: AuthenticationMethod::Token,
            token_expiry: 60, // 1 hour
            mfa_required: false,
        }
    }
}

impl Default for RateLimitingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_minute: 60,
            bandwidth_limit: 320, // 320 kbps
            connections_per_ip: 10,
        }
    }
}

impl Default for AccessControlSettings {
    fn default() -> Self {
        Self {
            mode: AccessControlMode::AllowAll,
            allowed_ips: vec![],
            blocked_ips: vec![],
            geo_restrictions: vec![],
        }
    }
}
