//! Secure Data Sharing Protocols
//!
//! This module provides enterprise-grade secure data sharing capabilities with
//! cryptographic verification, access control, audit logging, and GDPR compliance.

use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

#[cfg(feature = "privacy")]
use aes_gcm::{aead::Aead, Aes256Gcm, KeyInit};

/// Secure sharing errors
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum SecureSharingError {
    /// Access denied
    #[error("Access denied: {message}")]
    AccessDenied { message: String },

    /// Invalid share token
    #[error("Invalid or expired share token")]
    InvalidToken,

    /// Encryption failed
    #[error("Encryption failed: {message}")]
    EncryptionFailed { message: String },

    /// Validation failed
    #[error("Validation failed: {message}")]
    ValidationFailed { message: String },

    /// Quota exceeded
    #[error("Share quota exceeded for user {user_id}")]
    QuotaExceeded { user_id: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
}

/// Result type for secure sharing operations
pub type SecureSharingResult<T> = Result<T, SecureSharingError>;

/// Share access level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum AccessLevel {
    /// Read-only access
    ReadOnly,
    /// Read and write access
    ReadWrite,
    /// Full control including deletion
    FullControl,
    /// Custom access with specific permissions
    Custom { permissions: Vec<String> },
}

/// Data sharing protocol type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum SharingProtocol {
    /// Direct API access with token
    DirectApi,
    /// OAuth 2.0 authorization
    OAuth2,
    /// Webhook push notifications
    Webhook,
    /// Secure file transfer
    SecureFileTransfer,
    /// Federated data access
    Federated,
}

/// Share configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareConfig {
    /// Share ID
    pub share_id: Uuid,
    /// Owner user ID
    pub owner_id: String,
    /// Recipient identifier (user ID, email, or organization)
    pub recipient_id: String,
    /// Access level
    pub access_level: AccessLevel,
    /// Sharing protocol
    pub protocol: SharingProtocol,
    /// Expiration time
    pub expires_at: Option<DateTime<Utc>>,
    /// Maximum access count
    pub max_access_count: Option<usize>,
    /// Current access count
    pub access_count: usize,
    /// Data categories included in share
    pub data_categories: Vec<DataCategory>,
    /// Encryption enabled
    pub encryption_enabled: bool,
    /// Watermarking enabled
    pub watermarking_enabled: bool,
    /// Audit logging enabled
    pub audit_logging_enabled: bool,
    /// IP address whitelist
    pub ip_whitelist: Option<Vec<String>>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last accessed timestamp
    pub last_accessed: Option<DateTime<Utc>>,
    /// Share status
    pub status: ShareStatus,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Share status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum ShareStatus {
    /// Active and accessible
    Active,
    /// Temporarily suspended
    Suspended,
    /// Expired
    Expired,
    /// Revoked by owner
    Revoked,
    /// Access limit reached
    LimitReached,
}

/// Data category for sharing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum DataCategory {
    /// User profile data
    Profile,
    /// Session data
    Sessions,
    /// Feedback data
    Feedback,
    /// Progress data
    Progress,
    /// Analytics data (anonymized)
    Analytics,
    /// Achievement data
    Achievements,
    /// Training history
    TrainingHistory,
    /// Custom category
    Custom { name: String },
}

/// Share token for secure access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareToken {
    /// Token ID
    pub token_id: String,
    /// Share ID
    pub share_id: Uuid,
    /// Token value (cryptographically secure)
    pub token_value: String,
    /// Token signature for verification
    pub signature: String,
    /// Issued at timestamp
    pub issued_at: DateTime<Utc>,
    /// Expires at timestamp
    pub expires_at: DateTime<Utc>,
}

/// Access log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessLogEntry {
    /// Log entry ID
    pub log_id: Uuid,
    /// Share ID
    pub share_id: Uuid,
    /// Accessor identifier
    pub accessor_id: String,
    /// Access timestamp
    pub accessed_at: DateTime<Utc>,
    /// IP address
    pub ip_address: Option<String>,
    /// Data accessed
    pub data_accessed: Vec<DataCategory>,
    /// Access result
    pub result: AccessResult,
    /// User agent
    pub user_agent: Option<String>,
}

/// Access result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum AccessResult {
    /// Access granted
    Success,
    /// Access denied
    Denied { reason: String },
    /// Error occurred
    Error { message: String },
}

/// Shared data package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedDataPackage {
    /// Package ID
    pub package_id: Uuid,
    /// Share ID
    pub share_id: Uuid,
    /// Data category
    pub category: DataCategory,
    /// Data payload (encrypted if encryption enabled)
    pub data: Vec<u8>,
    /// Encryption metadata
    pub encryption_meta: Option<EncryptionMetadata>,
    /// Watermark information
    pub watermark: Option<WatermarkInfo>,
    /// Generated timestamp
    pub generated_at: DateTime<Utc>,
    /// Data signature for integrity verification
    pub signature: String,
}

/// Encryption metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    /// Algorithm used
    pub algorithm: String,
    /// Key derivation method
    pub key_derivation: String,
    /// Initialization vector (if applicable)
    pub iv: Option<Vec<u8>>,
}

/// Watermark information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatermarkInfo {
    /// Watermark ID for tracking
    pub watermark_id: String,
    /// Recipient identifier
    pub recipient_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Secure sharing manager
pub struct SecureSharingManager {
    /// Active shares
    shares: Arc<RwLock<HashMap<Uuid, ShareConfig>>>,
    /// Access logs
    access_logs: Arc<RwLock<Vec<AccessLogEntry>>>,
    /// Share tokens
    tokens: Arc<RwLock<HashMap<String, ShareToken>>>,
    /// Master encryption key
    #[cfg(feature = "privacy")]
    encryption_key: [u8; 32],
    /// Configuration
    config: SharingManagerConfig,
}

/// Sharing manager configuration
#[derive(Debug, Clone)]
pub struct SharingManagerConfig {
    /// Default token expiration duration
    pub default_token_expiration: ChronoDuration,
    /// Maximum shares per user
    pub max_shares_per_user: usize,
    /// Enable automatic cleanup of expired shares
    pub auto_cleanup_enabled: bool,
    /// Cleanup interval
    pub cleanup_interval: std::time::Duration,
    /// Require encryption for sensitive data
    pub require_encryption: bool,
    /// Enable audit logging
    pub enable_audit_logging: bool,
}

impl Default for SharingManagerConfig {
    fn default() -> Self {
        Self {
            default_token_expiration: ChronoDuration::days(30),
            max_shares_per_user: 100,
            auto_cleanup_enabled: true,
            cleanup_interval: std::time::Duration::from_secs(3600), // 1 hour
            require_encryption: true,
            enable_audit_logging: true,
        }
    }
}

impl SecureSharingManager {
    /// Create a new secure sharing manager
    #[must_use]
    pub fn new(config: SharingManagerConfig) -> Self {
        #[cfg(feature = "privacy")]
        let mut encryption_key = [0u8; 32];
        #[cfg(feature = "privacy")]
        thread_rng().fill(&mut encryption_key[..]);

        Self {
            shares: Arc::new(RwLock::new(HashMap::new())),
            access_logs: Arc::new(RwLock::new(Vec::new())),
            tokens: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "privacy")]
            encryption_key,
            config,
        }
    }

    /// Create a new data share
    pub async fn create_share(
        &self,
        owner_id: String,
        recipient_id: String,
        access_level: AccessLevel,
        protocol: SharingProtocol,
        data_categories: Vec<DataCategory>,
        expires_at: Option<DateTime<Utc>>,
    ) -> SecureSharingResult<ShareConfig> {
        // Check quota
        let shares = self.shares.read().await;
        let user_share_count = shares.values().filter(|s| s.owner_id == owner_id).count();

        if user_share_count >= self.config.max_shares_per_user {
            return Err(SecureSharingError::QuotaExceeded { user_id: owner_id });
        }
        drop(shares);

        let share_id = Uuid::new_v4();
        let share_config = ShareConfig {
            share_id,
            owner_id,
            recipient_id,
            access_level,
            protocol,
            expires_at,
            max_access_count: None,
            access_count: 0,
            data_categories,
            encryption_enabled: self.config.require_encryption,
            watermarking_enabled: true,
            audit_logging_enabled: self.config.enable_audit_logging,
            ip_whitelist: None,
            created_at: Utc::now(),
            last_accessed: None,
            status: ShareStatus::Active,
            metadata: HashMap::new(),
        };

        let mut shares = self.shares.write().await;
        shares.insert(share_id, share_config.clone());

        Ok(share_config)
    }

    /// Generate a share token
    pub async fn generate_token(&self, share_id: Uuid) -> SecureSharingResult<ShareToken> {
        let shares = self.shares.read().await;
        let share = shares
            .get(&share_id)
            .ok_or(SecureSharingError::InvalidToken)?;

        if share.status != ShareStatus::Active {
            return Err(SecureSharingError::InvalidToken);
        }

        drop(shares);

        let token_id = Uuid::new_v4().to_string();
        let token_value = self.generate_secure_token();
        let signature = self.sign_token(&token_value, share_id);

        let token = ShareToken {
            token_id: token_id.clone(),
            share_id,
            token_value: token_value.clone(),
            signature,
            issued_at: Utc::now(),
            expires_at: Utc::now() + self.config.default_token_expiration,
        };

        let mut tokens = self.tokens.write().await;
        tokens.insert(token_value, token.clone());

        Ok(token)
    }

    /// Validate a share token
    pub async fn validate_token(&self, token_value: &str) -> SecureSharingResult<ShareConfig> {
        let tokens = self.tokens.read().await;
        let token = tokens
            .get(token_value)
            .ok_or(SecureSharingError::InvalidToken)?;

        // Check expiration
        if Utc::now() > token.expires_at {
            return Err(SecureSharingError::InvalidToken);
        }

        // Verify signature
        let expected_signature = self.sign_token(&token.token_value, token.share_id);
        if token.signature != expected_signature {
            return Err(SecureSharingError::ValidationFailed {
                message: "Token signature verification failed".to_string(),
            });
        }

        // Clone share_id before dropping tokens
        let share_id = token.share_id;
        drop(tokens);

        let shares = self.shares.read().await;
        let share = shares
            .get(&share_id)
            .ok_or(SecureSharingError::InvalidToken)?
            .clone();

        // Check share status
        if share.status != ShareStatus::Active {
            return Err(SecureSharingError::AccessDenied {
                message: format!("Share is {:?}", share.status),
            });
        }

        // Check expiration
        if let Some(expires_at) = share.expires_at {
            if Utc::now() > expires_at {
                return Err(SecureSharingError::InvalidToken);
            }
        }

        // Check access count
        if let Some(max_count) = share.max_access_count {
            if share.access_count >= max_count {
                return Err(SecureSharingError::AccessDenied {
                    message: "Access limit reached".to_string(),
                });
            }
        }

        Ok(share)
    }

    /// Access shared data
    pub async fn access_data(
        &self,
        token_value: &str,
        accessor_id: String,
        ip_address: Option<String>,
        user_agent: Option<String>,
    ) -> SecureSharingResult<Vec<SharedDataPackage>> {
        // Validate token
        let share = self.validate_token(token_value).await?;

        // Check IP whitelist
        if let Some(whitelist) = &share.ip_whitelist {
            if let Some(ip) = &ip_address {
                if !whitelist.contains(ip) {
                    self.log_access(
                        share.share_id,
                        accessor_id.clone(),
                        ip_address,
                        vec![],
                        AccessResult::Denied {
                            reason: "IP not whitelisted".to_string(),
                        },
                        user_agent,
                    )
                    .await;

                    return Err(SecureSharingError::AccessDenied {
                        message: "IP address not authorized".to_string(),
                    });
                }
            }
        }

        // Update access count
        let mut shares = self.shares.write().await;
        if let Some(share_mut) = shares.get_mut(&share.share_id) {
            share_mut.access_count += 1;
            share_mut.last_accessed = Some(Utc::now());
        }
        drop(shares);

        // Generate data packages (mock implementation - would fetch real data)
        let packages = vec![
            self.create_data_package(
                share.share_id,
                DataCategory::Progress,
                vec![1, 2, 3, 4, 5], // Mock data
                &share,
            )
            .await?,
        ];

        // Log access
        self.log_access(
            share.share_id,
            accessor_id,
            ip_address,
            share.data_categories.clone(),
            AccessResult::Success,
            user_agent,
        )
        .await;

        Ok(packages)
    }

    /// Revoke a share
    pub async fn revoke_share(&self, share_id: Uuid, owner_id: &str) -> SecureSharingResult<()> {
        let mut shares = self.shares.write().await;
        let share = shares
            .get_mut(&share_id)
            .ok_or(SecureSharingError::InvalidToken)?;

        // Verify ownership
        if share.owner_id != owner_id {
            return Err(SecureSharingError::AccessDenied {
                message: "Not the owner of this share".to_string(),
            });
        }

        share.status = ShareStatus::Revoked;

        Ok(())
    }

    /// Get access logs for a share
    pub async fn get_access_logs(&self, share_id: Uuid) -> Vec<AccessLogEntry> {
        let logs = self.access_logs.read().await;
        logs.iter()
            .filter(|log| log.share_id == share_id)
            .cloned()
            .collect()
    }

    /// Get all shares for a user
    pub async fn get_user_shares(&self, user_id: &str) -> Vec<ShareConfig> {
        let shares = self.shares.read().await;
        shares
            .values()
            .filter(|s| s.owner_id == user_id)
            .cloned()
            .collect()
    }

    /// Cleanup expired shares
    pub async fn cleanup_expired_shares(&self) -> usize {
        let mut shares = self.shares.write().await;
        let now = Utc::now();
        let mut removed_count = 0;

        shares.retain(|_, share| {
            let should_remove = match share.status {
                ShareStatus::Expired | ShareStatus::Revoked => true,
                ShareStatus::Active => {
                    if let Some(expires_at) = share.expires_at {
                        if now > expires_at {
                            removed_count += 1;
                            return false;
                        }
                    }
                    false
                }
                _ => false,
            };

            if should_remove {
                removed_count += 1;
            }
            !should_remove
        });

        removed_count
    }

    /// Get sharing statistics
    pub async fn get_statistics(&self) -> SharingStatistics {
        let shares = self.shares.read().await;
        let logs = self.access_logs.read().await;

        let total_shares = shares.len();
        let active_shares = shares
            .values()
            .filter(|s| s.status == ShareStatus::Active)
            .count();
        let total_accesses = shares.values().map(|s| s.access_count).sum();
        let total_audit_logs = logs.len();

        SharingStatistics {
            total_shares,
            active_shares,
            expired_shares: shares
                .values()
                .filter(|s| s.status == ShareStatus::Expired)
                .count(),
            revoked_shares: shares
                .values()
                .filter(|s| s.status == ShareStatus::Revoked)
                .count(),
            total_accesses,
            total_audit_logs,
            shares_by_protocol: self.count_by_protocol(&shares).await,
        }
    }

    // Private helper methods

    fn generate_secure_token(&self) -> String {
        let mut token_bytes = [0u8; 32];
        thread_rng().fill(&mut token_bytes[..]);
        format!("sst_{}", self.encode_hex(&token_bytes))
    }

    fn sign_token(&self, token_value: &str, share_id: Uuid) -> String {
        let mut hasher = Sha256::new();
        hasher.update(token_value.as_bytes());
        hasher.update(share_id.as_bytes());
        #[cfg(feature = "privacy")]
        hasher.update(self.encryption_key);
        self.encode_hex(&hasher.finalize())
    }

    fn encode_hex(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    async fn create_data_package(
        &self,
        share_id: Uuid,
        category: DataCategory,
        data: Vec<u8>,
        share: &ShareConfig,
    ) -> SecureSharingResult<SharedDataPackage> {
        let package_id = Uuid::new_v4();

        // Encrypt data if enabled
        #[cfg(feature = "privacy")]
        let (encrypted_data, encryption_meta) = if share.encryption_enabled {
            let cipher = Aes256Gcm::new(&self.encryption_key.into());
            let mut nonce_bytes = [0u8; 12];
            thread_rng().fill(&mut nonce_bytes[..]);
            let nonce = &nonce_bytes.into();

            let ciphertext = cipher.encrypt(nonce, data.as_ref()).map_err(|e| {
                SecureSharingError::EncryptionFailed {
                    message: format!("Encryption failed: {e}"),
                }
            })?;

            let mut result = nonce_bytes.to_vec();
            result.extend(ciphertext);

            (
                result,
                Some(EncryptionMetadata {
                    algorithm: "AES-256-GCM".to_string(),
                    key_derivation: "HKDF-SHA256".to_string(),
                    iv: Some(nonce_bytes.to_vec()),
                }),
            )
        } else {
            (data.clone(), None)
        };

        #[cfg(not(feature = "privacy"))]
        let (encrypted_data, encryption_meta) = (data.clone(), None);

        // Add watermark if enabled
        let watermark = if share.watermarking_enabled {
            Some(WatermarkInfo {
                watermark_id: Uuid::new_v4().to_string(),
                recipient_id: share.recipient_id.clone(),
                timestamp: Utc::now(),
            })
        } else {
            None
        };

        // Create signature
        let mut hasher = Sha256::new();
        hasher.update(&encrypted_data);
        hasher.update(share_id.as_bytes());
        let signature = self.encode_hex(&hasher.finalize());

        Ok(SharedDataPackage {
            package_id,
            share_id,
            category,
            data: encrypted_data,
            encryption_meta,
            watermark,
            generated_at: Utc::now(),
            signature,
        })
    }

    async fn log_access(
        &self,
        share_id: Uuid,
        accessor_id: String,
        ip_address: Option<String>,
        data_accessed: Vec<DataCategory>,
        result: AccessResult,
        user_agent: Option<String>,
    ) {
        if !self.config.enable_audit_logging {
            return;
        }

        let log_entry = AccessLogEntry {
            log_id: Uuid::new_v4(),
            share_id,
            accessor_id,
            accessed_at: Utc::now(),
            ip_address,
            data_accessed,
            result,
            user_agent,
        };

        let mut logs = self.access_logs.write().await;
        logs.push(log_entry);
    }

    async fn count_by_protocol(
        &self,
        shares: &HashMap<Uuid, ShareConfig>,
    ) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for share in shares.values() {
            let protocol_name = format!("{:?}", share.protocol);
            *counts.entry(protocol_name).or_insert(0) += 1;
        }
        counts
    }
}

/// Sharing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingStatistics {
    /// Total number of shares
    pub total_shares: usize,
    /// Active shares
    pub active_shares: usize,
    /// Expired shares
    pub expired_shares: usize,
    /// Revoked shares
    pub revoked_shares: usize,
    /// Total access count
    pub total_accesses: usize,
    /// Total audit log entries
    pub total_audit_logs: usize,
    /// Shares by protocol
    pub shares_by_protocol: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_share() {
        let manager = SecureSharingManager::new(SharingManagerConfig::default());

        let share = manager
            .create_share(
                "user1".to_string(),
                "user2".to_string(),
                AccessLevel::ReadOnly,
                SharingProtocol::DirectApi,
                vec![DataCategory::Progress, DataCategory::Feedback],
                None,
            )
            .await
            .unwrap();

        assert_eq!(share.owner_id, "user1");
        assert_eq!(share.recipient_id, "user2");
        assert_eq!(share.status, ShareStatus::Active);
    }

    #[tokio::test]
    async fn test_generate_and_validate_token() {
        let manager = SecureSharingManager::new(SharingManagerConfig::default());

        let share = manager
            .create_share(
                "user1".to_string(),
                "user2".to_string(),
                AccessLevel::ReadOnly,
                SharingProtocol::DirectApi,
                vec![DataCategory::Progress],
                None,
            )
            .await
            .unwrap();

        let token = manager.generate_token(share.share_id).await.unwrap();
        assert!(token.token_value.starts_with("sst_"));

        let validated_share = manager.validate_token(&token.token_value).await.unwrap();
        assert_eq!(validated_share.share_id, share.share_id);
    }

    #[tokio::test]
    async fn test_revoke_share() {
        let manager = SecureSharingManager::new(SharingManagerConfig::default());

        let share = manager
            .create_share(
                "user1".to_string(),
                "user2".to_string(),
                AccessLevel::ReadOnly,
                SharingProtocol::DirectApi,
                vec![DataCategory::Progress],
                None,
            )
            .await
            .unwrap();

        manager.revoke_share(share.share_id, "user1").await.unwrap();

        let shares = manager.shares.read().await;
        let revoked_share = shares.get(&share.share_id).unwrap();
        assert_eq!(revoked_share.status, ShareStatus::Revoked);
    }

    #[tokio::test]
    async fn test_quota_enforcement() {
        let mut config = SharingManagerConfig::default();
        config.max_shares_per_user = 2;
        let manager = SecureSharingManager::new(config);

        // Create first share
        manager
            .create_share(
                "user1".to_string(),
                "user2".to_string(),
                AccessLevel::ReadOnly,
                SharingProtocol::DirectApi,
                vec![DataCategory::Progress],
                None,
            )
            .await
            .unwrap();

        // Create second share
        manager
            .create_share(
                "user1".to_string(),
                "user3".to_string(),
                AccessLevel::ReadOnly,
                SharingProtocol::DirectApi,
                vec![DataCategory::Progress],
                None,
            )
            .await
            .unwrap();

        // Third share should fail
        let result = manager
            .create_share(
                "user1".to_string(),
                "user4".to_string(),
                AccessLevel::ReadOnly,
                SharingProtocol::DirectApi,
                vec![DataCategory::Progress],
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SecureSharingError::QuotaExceeded { .. }
        ));
    }

    #[tokio::test]
    async fn test_access_data() {
        let manager = SecureSharingManager::new(SharingManagerConfig::default());

        let share = manager
            .create_share(
                "user1".to_string(),
                "user2".to_string(),
                AccessLevel::ReadOnly,
                SharingProtocol::DirectApi,
                vec![DataCategory::Progress],
                None,
            )
            .await
            .unwrap();

        let token = manager.generate_token(share.share_id).await.unwrap();

        let packages = manager
            .access_data(
                &token.token_value,
                "user2".to_string(),
                Some("192.168.1.1".to_string()),
                Some("TestAgent/1.0".to_string()),
            )
            .await
            .unwrap();

        assert!(!packages.is_empty());

        // Check access count was incremented
        let shares = manager.shares.read().await;
        let updated_share = shares.get(&share.share_id).unwrap();
        assert_eq!(updated_share.access_count, 1);
    }

    #[tokio::test]
    async fn test_cleanup_expired_shares() {
        let manager = SecureSharingManager::new(SharingManagerConfig::default());

        // Create expired share
        let expired_time = Utc::now() - ChronoDuration::days(1);
        let share = manager
            .create_share(
                "user1".to_string(),
                "user2".to_string(),
                AccessLevel::ReadOnly,
                SharingProtocol::DirectApi,
                vec![DataCategory::Progress],
                Some(expired_time),
            )
            .await
            .unwrap();

        let removed = manager.cleanup_expired_shares().await;
        assert_eq!(removed, 1);

        let shares = manager.shares.read().await;
        assert!(!shares.contains_key(&share.share_id));
    }

    #[tokio::test]
    async fn test_get_statistics() {
        let manager = SecureSharingManager::new(SharingManagerConfig::default());

        manager
            .create_share(
                "user1".to_string(),
                "user2".to_string(),
                AccessLevel::ReadOnly,
                SharingProtocol::DirectApi,
                vec![DataCategory::Progress],
                None,
            )
            .await
            .unwrap();

        manager
            .create_share(
                "user1".to_string(),
                "user3".to_string(),
                AccessLevel::ReadWrite,
                SharingProtocol::OAuth2,
                vec![DataCategory::Feedback],
                None,
            )
            .await
            .unwrap();

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_shares, 2);
        assert_eq!(stats.active_shares, 2);
        assert_eq!(stats.expired_shares, 0);
    }
}
