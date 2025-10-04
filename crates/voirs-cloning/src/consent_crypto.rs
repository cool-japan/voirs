//! Cryptographic Consent Verification System
//!
//! This module provides concrete implementations of cryptographic consent verification,
//! digital signatures, and audit logging for the voice cloning consent management system.

use crate::consent::{
    ConsentAccessLog, ConsentAuditAction, ConsentAuditLogger, ConsentRecord,
    ConsentVerificationProvider, ConsentVerificationStatus, ConsentViolationLog, DigitalSignature,
    DigitalSigningService, SignatureType,
};
use crate::{Error, Result};

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use base64::{engine::general_purpose, Engine as _};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use ring::{digest, hmac, rand::SystemRandom};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Cryptographic configuration for consent verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoConfig {
    /// Key derivation iterations for password-based keys
    pub pbkdf2_iterations: u32,
    /// Salt length for key derivation
    pub salt_length: usize,
    /// Signature algorithm to use
    pub signature_algorithm: SignatureAlgorithm,
    /// Encryption algorithm for sensitive data
    pub encryption_algorithm: EncryptionAlgorithm,
    /// HMAC key for integrity verification
    pub hmac_key: Option<Vec<u8>>,
}

/// Supported signature algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    Ed25519,
    Rsa2048,
    EcdsaP256,
}

/// Supported encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    Aes256Gcm,
    ChaCha20Poly1305,
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            pbkdf2_iterations: 100_000,
            salt_length: 32,
            signature_algorithm: SignatureAlgorithm::Ed25519,
            encryption_algorithm: EncryptionAlgorithm::Aes256Gcm,
            hmac_key: None,
        }
    }
}

/// Cryptographic consent verification provider
pub struct CryptoConsentVerifier {
    config: CryptoConfig,
    signing_keys: Arc<RwLock<HashMap<String, SigningKey>>>,
    verification_keys: Arc<RwLock<HashMap<String, VerifyingKey>>>,
    rng: SystemRandom,
}

impl CryptoConsentVerifier {
    /// Create a new cryptographic consent verifier
    pub fn new(config: CryptoConfig) -> Self {
        Self {
            config,
            signing_keys: Arc::new(RwLock::new(HashMap::new())),
            verification_keys: Arc::new(RwLock::new(HashMap::new())),
            rng: SystemRandom::new(),
        }
    }

    /// Register a signing key for a subject
    pub fn register_signing_key(&self, subject_id: &str, signing_key: SigningKey) -> Result<()> {
        let verification_key = signing_key.verifying_key();

        {
            let mut keys = self.signing_keys.write().map_err(|_| {
                Error::Verification("Failed to acquire signing keys lock".to_string())
            })?;
            keys.insert(subject_id.to_string(), signing_key);
        }

        {
            let mut keys = self.verification_keys.write().map_err(|_| {
                Error::Verification("Failed to acquire verification keys lock".to_string())
            })?;
            keys.insert(subject_id.to_string(), verification_key);
        }

        info!("Registered cryptographic keys for subject: {}", subject_id);
        Ok(())
    }

    /// Generate a new signing key pair for a subject
    pub fn generate_key_pair(&self, subject_id: &str) -> Result<(SigningKey, VerifyingKey)> {
        let signing_key = SigningKey::from_bytes(&scirs2_core::random::random::<[u8; 32]>());
        let verification_key = signing_key.verifying_key();

        self.register_signing_key(subject_id, signing_key.clone())?;

        Ok((signing_key, verification_key))
    }

    /// Create cryptographic proof of consent
    fn create_consent_proof(&self, consent: &ConsentRecord) -> Result<String> {
        // Create a structured consent summary for signing
        let consent_data = ConsentProofData {
            consent_id: consent.consent_id,
            subject_id: consent.subject_identity.subject_id.clone(),
            consent_type: format!("{:?}", consent.consent_type),
            permissions: format!("{:?}", consent.permissions),
            timestamp: consent.timestamps.created_at,
            legal_basis: format!("{:?}", consent.legal_info.legal_basis),
        };

        let serialized =
            serde_json::to_string(&consent_data).map_err(|e| Error::Serialization(e))?;

        // Create HMAC for integrity
        let hmac_key = self
            .config
            .hmac_key
            .as_ref()
            .ok_or_else(|| Error::Verification("HMAC key not configured".to_string()))?;

        let key = hmac::Key::new(hmac::HMAC_SHA256, hmac_key);
        let signature = hmac::sign(&key, serialized.as_bytes());

        // Encode as base64
        let proof = general_purpose::STANDARD.encode(signature.as_ref());

        debug!(
            "Created cryptographic proof for consent: {}",
            consent.consent_id
        );
        Ok(proof)
    }

    /// Verify cryptographic proof of consent
    fn verify_consent_proof(&self, consent: &ConsentRecord, proof: &str) -> Result<bool> {
        let consent_data = ConsentProofData {
            consent_id: consent.consent_id,
            subject_id: consent.subject_identity.subject_id.clone(),
            consent_type: format!("{:?}", consent.consent_type),
            permissions: format!("{:?}", consent.permissions),
            timestamp: consent.timestamps.created_at,
            legal_basis: format!("{:?}", consent.legal_info.legal_basis),
        };

        let serialized =
            serde_json::to_string(&consent_data).map_err(|e| Error::Serialization(e))?;

        // Decode proof
        let signature_bytes = general_purpose::STANDARD
            .decode(proof)
            .map_err(|e| Error::Verification(format!("Invalid proof encoding: {}", e)))?;

        // Verify HMAC
        let hmac_key = self
            .config
            .hmac_key
            .as_ref()
            .ok_or_else(|| Error::Verification("HMAC key not configured".to_string()))?;

        let key = hmac::Key::new(hmac::HMAC_SHA256, hmac_key);

        match hmac::verify(&key, serialized.as_bytes(), &signature_bytes) {
            Ok(()) => {
                debug!(
                    "Cryptographic proof verified for consent: {}",
                    consent.consent_id
                );
                Ok(true)
            }
            Err(_) => {
                warn!(
                    "Cryptographic proof verification failed for consent: {}",
                    consent.consent_id
                );
                Ok(false)
            }
        }
    }
}

impl ConsentVerificationProvider for CryptoConsentVerifier {
    fn verify_consent(&self, consent: &ConsentRecord) -> Result<ConsentVerificationStatus> {
        debug!(
            "Starting cryptographic verification for consent: {}",
            consent.consent_id
        );

        // Check if we have a cryptographic proof
        if let Some(ref proof) = consent.verification.cryptographic_proof {
            match self.verify_consent_proof(consent, proof) {
                Ok(true) => {
                    info!("Consent cryptographically verified: {}", consent.consent_id);
                    Ok(ConsentVerificationStatus::Verified)
                }
                Ok(false) => {
                    warn!(
                        "Consent cryptographic verification failed: {}",
                        consent.consent_id
                    );
                    Ok(ConsentVerificationStatus::Failed)
                }
                Err(e) => {
                    error!("Error during consent verification: {}", e);
                    Ok(ConsentVerificationStatus::Failed)
                }
            }
        } else {
            // No cryptographic proof present
            warn!(
                "No cryptographic proof found for consent: {}",
                consent.consent_id
            );
            Ok(ConsentVerificationStatus::Pending)
        }
    }

    fn get_provider_name(&self) -> &str {
        "CryptographicVerifier"
    }

    fn supports_method(&self, method: &crate::consent::ConsentVerificationMethod) -> bool {
        matches!(
            method,
            crate::consent::ConsentVerificationMethod::DigitalSignature
        )
    }
}

/// Data structure for consent proof
#[derive(Debug, Serialize, Deserialize)]
struct ConsentProofData {
    consent_id: Uuid,
    subject_id: String,
    consent_type: String,
    permissions: String,
    timestamp: SystemTime,
    legal_basis: String,
}

/// Digital signing service implementation
pub struct Ed25519SigningService {
    config: CryptoConfig,
    signing_keys: Arc<RwLock<HashMap<String, SigningKey>>>,
    verification_keys: Arc<RwLock<HashMap<String, VerifyingKey>>>,
}

impl Ed25519SigningService {
    /// Create a new Ed25519 signing service
    pub fn new(config: CryptoConfig) -> Self {
        Self {
            config,
            signing_keys: Arc::new(RwLock::new(HashMap::new())),
            verification_keys: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a signing key
    pub fn register_key(&self, signer_id: &str, signing_key: SigningKey) -> Result<()> {
        let verification_key = signing_key.verifying_key();

        {
            let mut keys = self.signing_keys.write().map_err(|_| {
                Error::Verification("Failed to acquire signing keys lock".to_string())
            })?;
            keys.insert(signer_id.to_string(), signing_key);
        }

        {
            let mut keys = self.verification_keys.write().map_err(|_| {
                Error::Verification("Failed to acquire verification keys lock".to_string())
            })?;
            keys.insert(signer_id.to_string(), verification_key);
        }

        info!("Registered signing key for: {}", signer_id);
        Ok(())
    }

    /// Generate a new key pair
    pub fn generate_key_pair(&self, signer_id: &str) -> Result<(SigningKey, VerifyingKey)> {
        let signing_key = SigningKey::from_bytes(&scirs2_core::random::random::<[u8; 32]>());
        let verification_key = signing_key.verifying_key();

        self.register_key(signer_id, signing_key.clone())?;

        Ok((signing_key, verification_key))
    }
}

impl DigitalSigningService for Ed25519SigningService {
    fn sign_consent(&self, consent: &ConsentRecord) -> Result<DigitalSignature> {
        let subject_id = &consent.subject_identity.subject_id;

        // Get signing key
        let signing_key = {
            let keys = self.signing_keys.read().map_err(|_| {
                Error::Verification("Failed to acquire signing keys lock".to_string())
            })?;
            keys.get(subject_id)
                .ok_or_else(|| {
                    Error::Verification(format!("No signing key found for subject: {}", subject_id))
                })?
                .clone()
        };

        // Create consent data to sign
        let consent_data = ConsentSigningData {
            consent_id: consent.consent_id,
            subject_id: subject_id.clone(),
            consent_type: format!("{:?}", consent.consent_type),
            permissions_hash: self.hash_permissions(&consent.permissions)?,
            timestamps: consent.timestamps.clone(),
        };

        let data_to_sign =
            serde_json::to_vec(&consent_data).map_err(|e| Error::Serialization(e))?;

        // Sign the data
        let signature = signing_key.sign(&data_to_sign);

        let digital_signature = DigitalSignature {
            signature_id: Uuid::new_v4(),
            signer_identity: subject_id.clone(),
            signature_algorithm: "Ed25519".to_string(),
            signature_value: signature.to_bytes().to_vec(),
            certificate: None, // Could add X.509 certificate support later
            timestamp: SystemTime::now(),
            signature_type: SignatureType::SubjectSignature,
        };

        info!(
            "Created digital signature for consent: {}",
            consent.consent_id
        );
        Ok(digital_signature)
    }

    fn verify_signature(&self, signature: &DigitalSignature, data: &[u8]) -> Result<bool> {
        // Get verification key
        let verification_key = {
            let keys = self.verification_keys.read().map_err(|_| {
                Error::Verification("Failed to acquire verification keys lock".to_string())
            })?;
            keys.get(&signature.signer_identity)
                .ok_or_else(|| {
                    Error::Verification(format!(
                        "No verification key found for signer: {}",
                        signature.signer_identity
                    ))
                })?
                .clone()
        };

        // Convert signature bytes
        let sig_bytes: [u8; 64] = signature
            .signature_value
            .as_slice()
            .try_into()
            .map_err(|_| Error::Verification("Invalid signature length".to_string()))?;
        let ed_signature = Signature::from_bytes(&sig_bytes);

        // Verify signature
        match verification_key.verify(data, &ed_signature) {
            Ok(()) => {
                debug!(
                    "Signature verified for signer: {}",
                    signature.signer_identity
                );
                Ok(true)
            }
            Err(_) => {
                warn!(
                    "Signature verification failed for signer: {}",
                    signature.signer_identity
                );
                Ok(false)
            }
        }
    }

    fn get_certificate(&self, _signature_id: &Uuid) -> Result<Option<Vec<u8>>> {
        // X.509 certificate support could be added here
        Ok(None)
    }
}

impl Ed25519SigningService {
    fn hash_permissions(&self, permissions: &crate::consent::ConsentPermissions) -> Result<String> {
        let serialized = serde_json::to_string(permissions).map_err(|e| Error::Serialization(e))?;

        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        let hash = hasher.finalize();

        Ok(general_purpose::STANDARD.encode(hash))
    }
}

/// Data structure for signing consent
#[derive(Debug, Serialize, Deserialize)]
struct ConsentSigningData {
    consent_id: Uuid,
    subject_id: String,
    consent_type: String,
    permissions_hash: String,
    timestamps: crate::consent::ConsentTimestamps,
}

/// Secure audit logger implementation
pub struct SecureAuditLogger {
    config: CryptoConfig,
    log_encryption_key: Arc<RwLock<Option<Key<Aes256Gcm>>>>,
    audit_log: Arc<RwLock<Vec<EncryptedAuditEntry>>>,
}

impl SecureAuditLogger {
    /// Create a new secure audit logger
    pub fn new(config: CryptoConfig) -> Self {
        Self {
            config,
            log_encryption_key: Arc::new(RwLock::new(None)),
            audit_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize with encryption key
    pub fn initialize_with_key(&self, key: &[u8]) -> Result<()> {
        if key.len() != 32 {
            return Err(Error::Validation(
                "Encryption key must be 32 bytes".to_string(),
            ));
        }

        let aes_key = Key::<Aes256Gcm>::from_slice(key);
        let mut key_guard = self
            .log_encryption_key
            .write()
            .map_err(|_| Error::Verification("Failed to acquire key lock".to_string()))?;
        *key_guard = Some(*aes_key);

        info!("Initialized secure audit logger with encryption");
        Ok(())
    }

    /// Generate a new encryption key
    pub fn generate_key(&self) -> Result<Vec<u8>> {
        let key = Aes256Gcm::generate_key(&mut OsRng);
        self.initialize_with_key(&key)?;
        Ok(key.to_vec())
    }

    fn encrypt_log_entry(&self, entry: &AuditLogEntry) -> Result<EncryptedAuditEntry> {
        let key_guard = self
            .log_encryption_key
            .read()
            .map_err(|_| Error::Verification("Failed to acquire key lock".to_string()))?;
        let key = key_guard
            .as_ref()
            .ok_or_else(|| Error::Verification("Encryption key not initialized".to_string()))?;

        let cipher = Aes256Gcm::new(key);
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

        let plaintext = serde_json::to_vec(entry).map_err(|e| Error::Serialization(e))?;

        let ciphertext = cipher
            .encrypt(&nonce, plaintext.as_ref())
            .map_err(|_| Error::Verification("Failed to encrypt audit entry".to_string()))?;

        let integrity_hash = self.compute_integrity_hash(&ciphertext)?;

        Ok(EncryptedAuditEntry {
            id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            nonce: nonce.to_vec(),
            ciphertext,
            integrity_hash,
        })
    }

    fn compute_integrity_hash(&self, data: &[u8]) -> Result<String> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        Ok(general_purpose::STANDARD.encode(hash))
    }
}

impl ConsentAuditLogger for SecureAuditLogger {
    fn log_consent_action(&self, action: ConsentAuditAction) -> Result<()> {
        let entry = AuditLogEntry::ConsentAction(action);
        let encrypted_entry = self.encrypt_log_entry(&entry)?;

        {
            let mut log = self
                .audit_log
                .write()
                .map_err(|_| Error::Verification("Failed to acquire audit log lock".to_string()))?;
            log.push(encrypted_entry);
        }

        debug!("Logged consent action to secure audit log");
        Ok(())
    }

    fn log_access(&self, access: ConsentAccessLog) -> Result<()> {
        let entry = AuditLogEntry::Access(access);
        let encrypted_entry = self.encrypt_log_entry(&entry)?;

        {
            let mut log = self
                .audit_log
                .write()
                .map_err(|_| Error::Verification("Failed to acquire audit log lock".to_string()))?;
            log.push(encrypted_entry);
        }

        debug!("Logged consent access to secure audit log");
        Ok(())
    }

    fn log_violation(&self, violation: ConsentViolationLog) -> Result<()> {
        let entry = AuditLogEntry::Violation(violation);
        let encrypted_entry = self.encrypt_log_entry(&entry)?;

        {
            let mut log = self
                .audit_log
                .write()
                .map_err(|_| Error::Verification("Failed to acquire audit log lock".to_string()))?;
            log.push(encrypted_entry);
        }

        warn!("Logged consent violation to secure audit log");
        Ok(())
    }
}

/// Internal audit log entry types
#[derive(Debug, Serialize, Deserialize)]
enum AuditLogEntry {
    ConsentAction(ConsentAuditAction),
    Access(ConsentAccessLog),
    Violation(ConsentViolationLog),
}

/// Encrypted audit log entry
#[derive(Debug, Clone)]
struct EncryptedAuditEntry {
    id: Uuid,
    timestamp: SystemTime,
    nonce: Vec<u8>,
    ciphertext: Vec<u8>,
    integrity_hash: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::*;

    #[test]
    fn test_crypto_config_default() {
        let config = CryptoConfig::default();
        assert_eq!(config.pbkdf2_iterations, 100_000);
        assert_eq!(config.salt_length, 32);
        assert!(matches!(
            config.signature_algorithm,
            SignatureAlgorithm::Ed25519
        ));
        assert!(matches!(
            config.encryption_algorithm,
            EncryptionAlgorithm::Aes256Gcm
        ));
    }

    #[test]
    fn test_crypto_consent_verifier_creation() {
        let config = CryptoConfig::default();
        let verifier = CryptoConsentVerifier::new(config);
        assert_eq!(verifier.get_provider_name(), "CryptographicVerifier");
    }

    #[test]
    fn test_ed25519_signing_service_creation() {
        let config = CryptoConfig::default();
        let signing_service = Ed25519SigningService::new(config);

        // Generate a key pair
        let result = signing_service.generate_key_pair("test-subject");
        assert!(result.is_ok());
    }

    #[test]
    fn test_secure_audit_logger_creation() {
        let config = CryptoConfig::default();
        let logger = SecureAuditLogger::new(config);

        // Generate and initialize with key
        let key_result = logger.generate_key();
        assert!(key_result.is_ok());
        assert_eq!(key_result.unwrap().len(), 32);
    }

    #[test]
    fn test_key_pair_generation() {
        let config = CryptoConfig::default();
        let verifier = CryptoConsentVerifier::new(config);

        let result = verifier.generate_key_pair("test-subject");
        assert!(result.is_ok());

        let (signing_key, verification_key) = result.unwrap();

        // Test signing and verification
        let message = b"test message";
        let signature = signing_key.sign(message);
        assert!(verification_key.verify(message, &signature).is_ok());
    }

    #[test]
    fn test_consent_verification_support() {
        let config = CryptoConfig::default();
        let verifier = CryptoConsentVerifier::new(config);

        assert!(verifier.supports_method(&ConsentVerificationMethod::DigitalSignature));
        assert!(!verifier.supports_method(&ConsentVerificationMethod::BiometricAuth));
    }

    #[test]
    fn test_signature_verification_invalid_key() {
        let config = CryptoConfig::default();
        let signing_service = Ed25519SigningService::new(config);

        let signature = DigitalSignature {
            signature_id: Uuid::new_v4(),
            signer_identity: "nonexistent-signer".to_string(),
            signature_algorithm: "Ed25519".to_string(),
            signature_value: vec![0; 64],
            certificate: None,
            timestamp: SystemTime::now(),
            signature_type: SignatureType::SubjectSignature,
        };

        let result = signing_service.verify_signature(&signature, b"test data");
        assert!(result.is_err());
    }

    #[test]
    fn test_audit_logger_without_key() {
        let config = CryptoConfig::default();
        let logger = SecureAuditLogger::new(config);

        let action = ConsentAuditAction {
            action_id: Uuid::new_v4(),
            consent_id: Uuid::new_v4(),
            action_type: ConsentActionType::ConsentCreated,
            actor: "test-actor".to_string(),
            timestamp: SystemTime::now(),
            details: HashMap::new(),
            ip_address: None,
            user_agent: None,
        };

        // Should fail without encryption key
        let result = logger.log_consent_action(action);
        assert!(result.is_err());
    }
}
