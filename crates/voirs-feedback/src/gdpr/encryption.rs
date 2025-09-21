//! GDPR Encryption and Privacy Utilities
//!
//! This module provides encryption, anonymization, and differential privacy
//! functionality for GDPR compliance.

use aes_gcm::{aead::Aead, Aes256Gcm, KeyInit};
use rand::{thread_rng, Rng};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::types::{GdprError, GdprResult};

/// GDPR-compliant encryption manager
#[derive(Debug)]
pub struct GdprEncryption {
    /// Master encryption key
    master_key: [u8; 32],
    /// Salt for key derivation
    salt: [u8; 16],
}

/// Differential privacy noise generator
#[derive(Debug)]
pub struct DifferentialPrivacy {
    /// Privacy budget epsilon
    epsilon: f64,
    /// Sensitivity of queries
    sensitivity: f64,
}

/// Privacy-preserving analytics manager
#[derive(Debug)]
pub struct PrivacyPreservingAnalytics {
    /// Privacy budget
    epsilon: f64,
    /// Encryption key for sensitive metrics
    encryption_key: Vec<u8>,
    /// Store for anonymized metrics
    anonymized_metrics: Arc<RwLock<HashMap<String, f64>>>,
}

impl GdprEncryption {
    /// Create new GDPR encryption manager
    pub fn new() -> Self {
        let mut master_key = [0u8; 32];
        let mut salt = [0u8; 16];
        thread_rng().fill(&mut master_key[..]);
        thread_rng().fill(&mut salt[..]);

        Self { master_key, salt }
    }

    /// Encrypt sensitive GDPR data end-to-end
    pub fn encrypt_sensitive_data(&self, data: &str) -> GdprResult<Vec<u8>> {
        let cipher = Aes256Gcm::new(&self.master_key.into());

        let mut nonce_bytes = [0u8; 12];
        thread_rng().fill(&mut nonce_bytes[..]);
        let nonce = &nonce_bytes.into();

        let ciphertext =
            cipher
                .encrypt(nonce, data.as_bytes())
                .map_err(|e| GdprError::AnonymizationFailed {
                    message: format!("Encryption failed: {}", e),
                })?;

        // Prepend nonce to ciphertext for decryption
        let mut result = nonce_bytes.to_vec();
        result.extend(ciphertext);

        Ok(result)
    }

    /// Decrypt sensitive GDPR data
    pub fn decrypt_sensitive_data(&self, encrypted_data: &[u8]) -> GdprResult<String> {
        if encrypted_data.len() < 12 {
            return Err(GdprError::AnonymizationFailed {
                message: String::from("Invalid encrypted data format"),
            });
        }

        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let cipher = Aes256Gcm::new(&self.master_key.into());
        let nonce = nonce_bytes.into();

        let plaintext =
            cipher
                .decrypt(nonce, ciphertext)
                .map_err(|e| GdprError::AnonymizationFailed {
                    message: format!("Decryption failed: {}", e),
                })?;

        String::from_utf8(plaintext).map_err(|e| GdprError::AnonymizationFailed {
            message: format!("Invalid UTF-8 in decrypted data: {}", e),
        })
    }

    /// Generate pseudonymized identifier for analytics
    pub fn pseudonymize_identifier(&self, original_id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.salt);
        hasher.update(original_id.as_bytes());
        let hash = hasher.finalize();
        format!("pseudo_{}", self.encode_hex(&hash[..8]))
    }

    /// Create secure hash of sensitive data for deduplication
    pub fn secure_hash(&self, data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.master_key);
        hasher.update(data.as_bytes());
        self.encode_hex(&hasher.finalize())
    }

    /// Helper function to encode bytes as hex string
    fn encode_hex(&self, bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join("")
    }
}

impl DifferentialPrivacy {
    /// Create new differential privacy manager
    pub fn new(epsilon: f64, sensitivity: f64) -> Self {
        Self {
            epsilon,
            sensitivity,
        }
    }

    /// Add Laplace noise for differential privacy
    pub fn add_laplace_noise(&self, true_value: f64) -> f64 {
        let scale = self.sensitivity / self.epsilon;
        let noise = self.sample_laplace(scale);
        true_value + noise
    }

    /// Sample from Laplace distribution
    fn sample_laplace(&self, scale: f64) -> f64 {
        let u: f64 = thread_rng().gen_range(-0.5..0.5);
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    /// Check if privacy budget allows for query
    pub fn check_privacy_budget(&self, requested_epsilon: f64) -> bool {
        requested_epsilon <= self.epsilon
    }
}

impl PrivacyPreservingAnalytics {
    /// Create new privacy-preserving analytics manager
    pub fn new(epsilon: f64) -> Self {
        let mut encryption_key = vec![0u8; 32];
        thread_rng().fill(&mut encryption_key[..]);

        Self {
            epsilon,
            encryption_key,
            anonymized_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add differentially private noise to analytical queries
    pub fn add_differential_privacy_noise(&self, true_value: f64) -> f64 {
        let dp = DifferentialPrivacy::new(self.epsilon, 1.0);
        dp.add_laplace_noise(true_value)
    }

    /// Aggregate metrics with privacy preservation
    pub async fn aggregate_privacy_safe_metrics(
        &self,
        metrics: Vec<(&str, f64)>,
    ) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        let mut agg_metrics = self.anonymized_metrics.write().await;

        for (metric_name, value) in metrics {
            let noisy_value = self.add_differential_privacy_noise(value);
            agg_metrics.insert(metric_name.to_string(), noisy_value);
            result.insert(metric_name.to_string(), noisy_value);
        }

        result
    }

    /// Generate anonymized user behavior insights
    pub async fn generate_anonymized_insights(
        &self,
        user_count: usize,
    ) -> GdprResult<HashMap<String, f64>> {
        if user_count < 10 {
            return Err(GdprError::PrivacyPolicyViolation {
                violation: String::from("Insufficient user count for privacy-safe analytics"),
            });
        }

        let mut insights = HashMap::new();
        let metrics = self.anonymized_metrics.read().await;

        for (key, value) in metrics.iter() {
            // Add additional noise for small populations
            let privacy_safe_value = if user_count < 100 {
                self.add_differential_privacy_noise(*value)
            } else {
                *value
            };
            insights.insert(key.clone(), privacy_safe_value);
        }

        Ok(insights)
    }
}

impl Default for GdprEncryption {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PrivacyPreservingAnalytics {
    fn default() -> Self {
        Self::new(1.0) // Default epsilon of 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_privacy_preserving_analytics() {
        let analytics = PrivacyPreservingAnalytics::new(1.0);

        // Test basic differential privacy noise addition
        let original_value = 100.0;
        let noisy_value = analytics.add_differential_privacy_noise(original_value);

        // Noise should be added, so values should be different
        assert_ne!(original_value, noisy_value);

        // Test metrics aggregation
        let metrics = vec![("sessions", 50.0), ("completions", 45.0)];
        let result = analytics.aggregate_privacy_safe_metrics(metrics).await;

        assert!(result.contains_key("sessions"));
        assert!(result.contains_key("completions"));
    }

    #[tokio::test]
    async fn test_privacy_preserving_analytics_insufficient_users() {
        let analytics = PrivacyPreservingAnalytics::new(1.0);

        // Test with insufficient user count
        let result = analytics.generate_anonymized_insights(5).await;
        assert!(result.is_err());

        match result {
            Err(GdprError::PrivacyPolicyViolation { violation }) => {
                assert!(violation.contains("Insufficient user count"));
            }
            _ => panic!("Expected privacy policy violation"),
        }
    }

    #[tokio::test]
    async fn test_gdpr_encryption() {
        let encryption = GdprEncryption::new();

        // Test encryption and decryption
        let sensitive_data = "user_personal_information@example.com";
        let encrypted = encryption.encrypt_sensitive_data(sensitive_data).unwrap();
        let decrypted = encryption.decrypt_sensitive_data(&encrypted).unwrap();

        assert_eq!(sensitive_data, decrypted);

        // Test pseudonymization
        let user_id = "user123";
        let pseudo_id = encryption.pseudonymize_identifier(user_id);
        assert!(pseudo_id.starts_with("pseudo_"));
        assert_ne!(user_id, pseudo_id);

        // Same input should produce same pseudonym
        let pseudo_id2 = encryption.pseudonymize_identifier(user_id);
        assert_eq!(pseudo_id, pseudo_id2);
    }

    #[tokio::test]
    async fn test_differential_privacy() {
        let dp = DifferentialPrivacy::new(1.0, 1.0);

        // Test privacy budget checking
        assert!(dp.check_privacy_budget(0.5));
        assert!(dp.check_privacy_budget(1.0));
        assert!(!dp.check_privacy_budget(1.5));

        // Test Laplace noise addition
        let original_value = 100.0;
        let noisy_value = dp.add_laplace_noise(original_value);

        // Should add noise
        assert_ne!(original_value, noisy_value);
    }

    #[tokio::test]
    async fn test_gdpr_encryption_error_handling() {
        let encryption = GdprEncryption::new();

        // Test decryption with invalid data
        let invalid_data = vec![1, 2, 3];
        let result = encryption.decrypt_sensitive_data(&invalid_data);
        assert!(result.is_err());

        match result {
            Err(GdprError::AnonymizationFailed { message }) => {
                assert!(message.contains("Invalid encrypted data format"));
            }
            _ => panic!("Expected anonymization failed error"),
        }
    }

    #[tokio::test]
    async fn test_secure_hash_consistency() {
        let encryption = GdprEncryption::new();

        let data = "sensitive_information";
        let hash1 = encryption.secure_hash(data);
        let hash2 = encryption.secure_hash(data);

        // Same data should produce same hash
        assert_eq!(hash1, hash2);

        // Different data should produce different hash
        let hash3 = encryption.secure_hash("different_data");
        assert_ne!(hash1, hash3);
    }
}
