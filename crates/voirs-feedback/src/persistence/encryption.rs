//! Encryption and data anonymization for privacy protection

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::persistence::{PersistenceError, PersistenceResult};

/// Privacy level for data handling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PrivacyLevel {
    /// Full data with all personally identifiable information
    Full,
    /// Anonymized data with PII removed or hashed
    Anonymized,
    /// Public data safe for sharing
    Public,
    /// Minimal data with only essential information
    Minimal,
}

/// Data anonymization service
pub struct DataAnonymizer;

impl DataAnonymizer {
    /// Create a new data anonymizer
    pub fn new() -> Self {
        Self
    }

    /// Anonymize user ID
    pub fn anonymize_user_id(&self, user_id: &str) -> String {
        // If already anonymized, return as-is (idempotent)
        if user_id.starts_with("anon_") {
            return user_id.to_string();
        }

        // Simple hash-based anonymization (in production, use proper hashing)
        format!("anon_{}", self.simple_hash(user_id))
    }

    /// Anonymize session data
    pub fn anonymize_session(
        &self,
        session: &mut crate::traits::SessionState,
        level: PrivacyLevel,
    ) {
        match level {
            PrivacyLevel::Full => {
                // No anonymization needed
            }
            PrivacyLevel::Anonymized => {
                // Only anonymize if not already anonymized
                if !session.user_id.starts_with("anon_") {
                    session.user_id = self.anonymize_user_id(&session.user_id);
                }
                // Clear any sensitive metadata
                session.stats = crate::traits::SessionStats::default();
            }
            PrivacyLevel::Public => {
                session.user_id = "public_user".to_string();
                session.stats = crate::traits::SessionStats::default();
                session.preferences = crate::traits::UserPreferences::default();
            }
            PrivacyLevel::Minimal => {
                session.user_id = "user".to_string();
                session.stats = crate::traits::SessionStats::default();
                session.preferences = crate::traits::UserPreferences::default();
                session.adaptive_state = crate::traits::AdaptiveState::default();
            }
        }
    }

    /// Anonymize feedback data
    pub fn anonymize_feedback(
        &self,
        feedback: &mut crate::traits::FeedbackResponse,
        level: PrivacyLevel,
    ) {
        match level {
            PrivacyLevel::Full => {
                // No anonymization needed
            }
            PrivacyLevel::Anonymized | PrivacyLevel::Public | PrivacyLevel::Minimal => {
                // Remove or anonymize any metadata that might contain PII
                for item in &mut feedback.feedback_items {
                    item.metadata.clear();
                }
            }
        }
    }

    /// Anonymize user progress data
    pub fn anonymize_progress(
        &self,
        progress: &mut crate::traits::UserProgress,
        level: PrivacyLevel,
    ) {
        match level {
            PrivacyLevel::Full => {
                // No anonymization needed
            }
            PrivacyLevel::Anonymized => {
                // Keep structure but remove identifying details
                // Only anonymize if not already anonymized
                if !progress.user_id.starts_with("anon_") {
                    progress.user_id = self.anonymize_user_id(&progress.user_id);
                }
            }
            PrivacyLevel::Public | PrivacyLevel::Minimal => {
                // More aggressive anonymization
                progress.user_id = "anonymous".to_string();
                // Could also round/bucket numerical values for privacy
            }
        }
    }

    /// Simple hash function for demonstration (use proper crypto hash in production)
    fn simple_hash(&self, input: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for DataAnonymizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Encryption service for sensitive data
#[cfg(feature = "privacy")]
pub struct EncryptionService {
    key: [u8; 32], // AES-256 key
}

#[cfg(feature = "privacy")]
impl EncryptionService {
    /// Create a new encryption service with a generated key
    pub fn new() -> PersistenceResult<Self> {
        use ring::rand::{SecureRandom, SystemRandom};

        let rng = SystemRandom::new();
        let mut key = [0u8; 32];
        rng.fill(&mut key)
            .map_err(|e| PersistenceError::EncryptionError {
                message: format!("Failed to generate encryption key: {:?}", e),
            })?;

        Ok(Self { key })
    }

    /// Create encryption service with existing key
    pub fn with_key(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Encrypt data
    pub fn encrypt(&self, data: &[u8]) -> PersistenceResult<Vec<u8>> {
        use aes_gcm::aead::{Aead, AeadCore, OsRng};
        use aes_gcm::{Aes256Gcm, KeyInit, Nonce};

        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| {
            PersistenceError::EncryptionError {
                message: format!("Failed to create cipher: {}", e),
            }
        })?;

        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext =
            cipher
                .encrypt(&nonce, data)
                .map_err(|e| PersistenceError::EncryptionError {
                    message: format!("Encryption failed: {}", e),
                })?;

        // Prepend nonce to ciphertext
        let mut result = nonce.to_vec();
        result.extend_from_slice(&ciphertext);
        Ok(result)
    }

    /// Decrypt data
    pub fn decrypt(&self, encrypted_data: &[u8]) -> PersistenceResult<Vec<u8>> {
        use aes_gcm::aead::Aead;
        use aes_gcm::{Aes256Gcm, KeyInit, Nonce};

        if encrypted_data.len() < 12 {
            return Err(PersistenceError::EncryptionError {
                message: "Encrypted data too short".to_string(),
            });
        }

        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| {
            PersistenceError::EncryptionError {
                message: format!("Failed to create cipher: {}", e),
            }
        })?;

        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| PersistenceError::EncryptionError {
                message: format!("Decryption failed: {}", e),
            })
    }

    /// Encrypt JSON data
    pub fn encrypt_json<T: Serialize>(&self, data: &T) -> PersistenceResult<Vec<u8>> {
        let json_data =
            serde_json::to_vec(data).map_err(|e| PersistenceError::SerializationError {
                message: format!("JSON serialization failed: {}", e),
            })?;

        self.encrypt(&json_data)
    }

    /// Decrypt JSON data
    pub fn decrypt_json<T: for<'de> Deserialize<'de>>(
        &self,
        encrypted_data: &[u8],
    ) -> PersistenceResult<T> {
        let decrypted_data = self.decrypt(encrypted_data)?;

        serde_json::from_slice(&decrypted_data).map_err(|e| PersistenceError::SerializationError {
            message: format!("JSON deserialization failed: {}", e),
        })
    }

    /// Get key for backup/restore (be careful with this!)
    pub fn get_key(&self) -> &[u8; 32] {
        &self.key
    }
}

#[cfg(not(feature = "privacy"))]
/// Stub encryption service when privacy feature is disabled
pub struct EncryptionService;

#[cfg(not(feature = "privacy"))]
impl EncryptionService {
    /// Create a new encryption service (returns error without privacy feature)
    pub fn new() -> PersistenceResult<Self> {
        Err(PersistenceError::EncryptionError {
            message: "Encryption requires 'privacy' feature".to_string(),
        })
    }
}

/// Password hashing service for user authentication
#[cfg(feature = "privacy")]
pub struct PasswordHasher;

#[cfg(feature = "privacy")]
impl PasswordHasher {
    /// Hash a password using Argon2
    pub fn hash_password(password: &str) -> PersistenceResult<String> {
        use argon2::password_hash::{rand_core::OsRng, SaltString};
        use argon2::{Argon2, PasswordHasher as ArgonHasher};

        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();

        argon2
            .hash_password(password.as_bytes(), &salt)
            .map(|hash| hash.to_string())
            .map_err(|e| PersistenceError::EncryptionError {
                message: format!("Password hashing failed: {}", e),
            })
    }

    /// Verify a password against a hash
    pub fn verify_password(password: &str, hash: &str) -> PersistenceResult<bool> {
        use argon2::password_hash::PasswordHash;
        use argon2::{Argon2, PasswordVerifier};

        let parsed_hash =
            PasswordHash::new(hash).map_err(|e| PersistenceError::EncryptionError {
                message: format!("Invalid password hash: {}", e),
            })?;

        let argon2 = Argon2::default();
        Ok(argon2
            .verify_password(password.as_bytes(), &parsed_hash)
            .is_ok())
    }
}

/// Privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Default privacy level
    pub default_privacy_level: PrivacyLevel,
    /// Enable encryption at rest
    pub encrypt_at_rest: bool,
    /// Enable data anonymization
    pub enable_anonymization: bool,
    /// Data retention period in days
    pub data_retention_days: u32,
    /// Auto-delete expired data
    pub auto_delete_expired: bool,
    /// Allowed data export formats
    pub allowed_export_formats: Vec<ExportFormat>,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            default_privacy_level: PrivacyLevel::Anonymized,
            encrypt_at_rest: true,
            enable_anonymization: true,
            data_retention_days: 365,
            auto_delete_expired: false, // Manual approval required
            allowed_export_formats: vec![ExportFormat::Json, ExportFormat::Csv],
        }
    }
}

/// Supported export formats for user data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// XML format
    Xml,
    /// PDF format (for reports)
    Pdf,
}

/// Privacy-aware data export service
pub struct PrivacyExportService {
    anonymizer: DataAnonymizer,
    config: PrivacyConfig,
}

impl PrivacyExportService {
    /// Create a new privacy export service
    pub fn new(config: PrivacyConfig) -> Self {
        Self {
            anonymizer: DataAnonymizer::new(),
            config,
        }
    }

    /// Export user data with privacy controls
    pub fn export_user_data(
        &self,
        mut export_data: crate::persistence::UserDataExport,
        privacy_level: PrivacyLevel,
        format: ExportFormat,
    ) -> PersistenceResult<Vec<u8>> {
        // Check if format is allowed
        if !self.config.allowed_export_formats.contains(&format) {
            return Err(PersistenceError::ConfigError {
                message: format!("Export format {:?} not allowed", format),
            });
        }

        // Apply privacy level
        match privacy_level {
            PrivacyLevel::Full => {
                // No anonymization needed for the top-level user_id
            }
            PrivacyLevel::Anonymized => {
                // Only anonymize if not already anonymized
                if !export_data.user_id.starts_with("anon_") {
                    export_data.user_id = self.anonymizer.anonymize_user_id(&export_data.user_id);
                }
            }
            PrivacyLevel::Public => {
                export_data.user_id = "public_user".to_string();
            }
            PrivacyLevel::Minimal => {
                export_data.user_id = "user".to_string();
            }
        }

        self.anonymizer
            .anonymize_progress(&mut export_data.progress, privacy_level.clone());

        for session in &mut export_data.sessions {
            self.anonymizer
                .anonymize_session(session, privacy_level.clone());
        }

        for feedback in &mut export_data.feedback_history {
            self.anonymizer
                .anonymize_feedback(feedback, privacy_level.clone());
        }

        // Export in requested format
        match format {
            ExportFormat::Json => serde_json::to_vec_pretty(&export_data).map_err(|e| {
                PersistenceError::SerializationError {
                    message: format!("JSON export failed: {}", e),
                }
            }),
            ExportFormat::Csv => {
                // Simplified CSV export (would need proper CSV library)
                let csv_data = format!(
                    "user_id,export_timestamp,total_sessions,total_feedback\n{},{},{},{}",
                    export_data.user_id,
                    export_data.export_timestamp,
                    export_data.sessions.len(),
                    export_data.feedback_history.len()
                );
                Ok(csv_data.into_bytes())
            }
            ExportFormat::Xml | ExportFormat::Pdf => Err(PersistenceError::ConfigError {
                message: format!("Export format {:?} not yet implemented", format),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_anonymizer() {
        let anonymizer = DataAnonymizer::new();

        let user_id = "user123";
        let anonymized = anonymizer.anonymize_user_id(user_id);

        assert!(anonymized.starts_with("anon_"));
        assert_ne!(anonymized, user_id);

        // Same input should produce same output
        let anonymized2 = anonymizer.anonymize_user_id(user_id);
        assert_eq!(anonymized, anonymized2);
    }

    #[cfg(feature = "privacy")]
    #[test]
    fn test_encryption_service() {
        let service = EncryptionService::new().unwrap();
        let data = b"Hello, World!";

        let encrypted = service.encrypt(data).unwrap();
        assert_ne!(encrypted, data);

        let decrypted = service.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[cfg(feature = "privacy")]
    #[test]
    fn test_password_hashing() {
        let password = "secure_password_123";
        let hash = PasswordHasher::hash_password(password).unwrap();

        assert!(PasswordHasher::verify_password(password, &hash).unwrap());
        assert!(!PasswordHasher::verify_password("wrong_password", &hash).unwrap());
    }

    #[test]
    fn test_privacy_config_default() {
        let config = PrivacyConfig::default();
        assert_eq!(config.default_privacy_level, PrivacyLevel::Anonymized);
        assert!(config.encrypt_at_rest);
        assert!(config.enable_anonymization);
    }
}
