//! Cache encryption functionality for secure data storage.

use serde::{Deserialize, Serialize};

// Only compile encryption functionality when cloud feature is enabled
#[cfg(feature = "cloud")]
mod encryption_impl {
    use super::*;
    use crate::error::{Result, VoirsError};
    use aes_gcm::aead::generic_array::GenericArray;
    use aes_gcm::{
        aead::{Aead, AeadCore, KeyInit, OsRng},
        Aes256Gcm, Key,
    };
    use std::convert::TryInto;

    /// Cache encryption manager
    #[derive(Clone)]
    pub struct CacheEncryption {
        cipher: Aes256Gcm,
    }

    impl CacheEncryption {
        /// Create new encryption manager with random key
        pub fn new() -> Self {
            let key = Aes256Gcm::generate_key(&mut OsRng);
            let cipher = Aes256Gcm::new(&key);

            Self { cipher }
        }

        /// Create encryption manager with provided key
        pub fn with_key(key: &[u8]) -> Result<Self> {
            if key.len() != 32 {
                return Err(VoirsError::config_error(
                    "Encryption key must be exactly 32 bytes",
                ));
            }

            let key: &Key<Aes256Gcm> = key
                .try_into()
                .map_err(|_| VoirsError::config_error("Invalid encryption key format"))?;
            let cipher = Aes256Gcm::new(key);

            Ok(Self { cipher })
        }

        /// Encrypt data
        pub fn encrypt(&self, data: &[u8]) -> Result<EncryptedData> {
            let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

            let encrypted_data = self
                .cipher
                .encrypt(&nonce, data)
                .map_err(|e| VoirsError::cache_error(format!("Encryption failed: {e}")))?;

            // Calculate checksum for integrity verification
            let checksum = self.calculate_checksum(data);

            let metadata = EncryptionMetadata {
                algorithm: "AES-256-GCM".to_string(),
                key_derivation: "Random".to_string(),
                timestamp: std::time::SystemTime::now(),
                checksum,
            };

            Ok(EncryptedData {
                data: encrypted_data,
                nonce: nonce.to_vec(),
                metadata,
            })
        }

        /// Decrypt data
        pub fn decrypt(&self, encrypted_data: &EncryptedData) -> Result<Vec<u8>> {
            let nonce = GenericArray::from_slice(&encrypted_data.nonce[..12]); // AES-GCM typically uses 12-byte nonces

            let decrypted_data = self
                .cipher
                .decrypt(&nonce, encrypted_data.data.as_slice())
                .map_err(|e| VoirsError::cache_error(format!("Decryption failed: {e}")))?;

            // Verify data integrity
            let expected_checksum = self.calculate_checksum(&decrypted_data);
            if expected_checksum != encrypted_data.metadata.checksum {
                return Err(VoirsError::cache_error(
                    "Data integrity verification failed",
                ));
            }

            Ok(decrypted_data)
        }

        /// Calculate data checksum for integrity verification
        fn calculate_checksum(&self, data: &[u8]) -> String {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(data);
            format!("{:x}", hasher.finalize())
        }

        /// Encrypt and serialize data
        pub fn encrypt_serialize<T: Serialize>(&self, data: &T) -> Result<EncryptedData> {
            let serialized = bincode::serde::encode_to_vec(data, bincode::config::standard())
                .map_err(|e| VoirsError::cache_error(format!("Serialization failed: {e}")))?;
            self.encrypt(&serialized)
        }

        /// Decrypt and deserialize data
        pub fn decrypt_deserialize<T: for<'de> Deserialize<'de>>(
            &self,
            encrypted_data: &EncryptedData,
        ) -> Result<T> {
            let decrypted = self.decrypt(encrypted_data)?;
            bincode::serde::decode_from_slice(&decrypted, bincode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| VoirsError::cache_error(format!("Deserialization failed: {e}")))
        }
    }

    impl Default for CacheEncryption {
        fn default() -> Self {
            Self::new()
        }
    }
}

// Re-export types when cloud feature is enabled
#[cfg(feature = "cloud")]
pub use encryption_impl::*;

// Provide no-op/stub implementations when cloud feature is disabled
#[cfg(not(feature = "cloud"))]
mod encryption_stub {
    use super::*;
    use crate::error::{Result, VoirsError};

    /// Stub encryption manager when cloud features are disabled
    #[derive(Debug, Clone)]
    pub struct CacheEncryption;

    impl CacheEncryption {
        /// Create new encryption manager (stub)
        pub fn new() -> Self {
            Self
        }

        /// Create encryption manager with provided key (stub)
        pub fn with_key(_key: &[u8]) -> Result<Self> {
            Err(VoirsError::config_error(
                "Encryption requires 'cloud' feature to be enabled",
            ))
        }

        /// Encrypt data (stub)
        pub fn encrypt(&self, _data: &[u8]) -> Result<EncryptedData> {
            Err(VoirsError::config_error(
                "Encryption requires 'cloud' feature to be enabled",
            ))
        }

        /// Decrypt data (stub)
        pub fn decrypt(&self, _encrypted_data: &EncryptedData) -> Result<Vec<u8>> {
            Err(VoirsError::config_error(
                "Decryption requires 'cloud' feature to be enabled",
            ))
        }

        /// Encrypt and serialize data (stub)
        pub fn encrypt_serialize<T: Serialize>(&self, _data: &T) -> Result<EncryptedData> {
            Err(VoirsError::config_error(
                "Encryption requires 'cloud' feature to be enabled",
            ))
        }

        /// Decrypt and deserialize data (stub)
        pub fn decrypt_deserialize<T: for<'de> Deserialize<'de>>(
            &self,
            _encrypted_data: &EncryptedData,
        ) -> Result<T> {
            Err(VoirsError::config_error(
                "Decryption requires 'cloud' feature to be enabled",
            ))
        }
    }

    impl Default for CacheEncryption {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(not(feature = "cloud"))]
pub use encryption_stub::*;

/// Encrypted data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    /// Encrypted content
    pub data: Vec<u8>,
    /// Nonce used for encryption
    pub nonce: Vec<u8>,
    /// Metadata about encryption
    pub metadata: EncryptionMetadata,
}

/// Encryption metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    /// Algorithm used
    pub algorithm: String,
    /// Key derivation method
    pub key_derivation: String,
    /// Encryption timestamp
    pub timestamp: std::time::SystemTime,
    /// Data integrity checksum
    pub checksum: String,
}

/// Cache encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Enable encryption
    pub enabled: bool,
    /// Algorithm to use
    pub algorithm: EncryptionAlgorithm,
    /// Key derivation method
    pub key_derivation: KeyDerivationMethod,
    /// Encryption key file path (optional)
    pub key_file: Option<std::path::PathBuf>,
}

/// Supported encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    Aes256Gcm,
    ChaCha20Poly1305,
}

/// Key derivation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivationMethod {
    Random,
    Pbkdf2,
    Scrypt,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: cfg!(feature = "cloud"),
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            key_derivation: KeyDerivationMethod::Random,
            key_file: None,
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cloud")]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_roundtrip() {
        let encryption = CacheEncryption::new();
        let original_data = b"Hello, World! This is test data for encryption.";

        let encrypted = encryption.encrypt(original_data).unwrap();
        let decrypted = encryption.decrypt(&encrypted).unwrap();

        assert_eq!(original_data.to_vec(), decrypted);
    }

    #[test]
    fn test_encrypt_serialize_roundtrip() {
        let encryption = CacheEncryption::new();
        let original_data = vec!["test", "data", "for", "serialization"];

        let encrypted = encryption.encrypt_serialize(&original_data).unwrap();
        let decrypted: Vec<String> = encryption.decrypt_deserialize(&encrypted).unwrap();

        assert_eq!(original_data, decrypted);
    }

    #[test]
    fn test_encryption_with_custom_key() {
        let key = [0u8; 32]; // Simple test key
        let encryption = CacheEncryption::with_key(&key).unwrap();
        let data = b"Test data with custom key";

        let encrypted = encryption.encrypt(data).unwrap();
        let decrypted = encryption.decrypt(&encrypted).unwrap();

        assert_eq!(data.to_vec(), decrypted);
    }

    #[test]
    fn test_invalid_key_length() {
        let short_key = [0u8; 16]; // Too short
        let result = CacheEncryption::with_key(&short_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_integrity_verification() {
        let encryption = CacheEncryption::new();
        let data = b"Test data for integrity check";

        let mut encrypted = encryption.encrypt(data).unwrap();

        // Tamper with the checksum
        encrypted.metadata.checksum = "invalid_checksum".to_string();

        let result = encryption.decrypt(&encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_encryption_metadata() {
        let encryption = CacheEncryption::new();
        let data = b"Test metadata generation";

        let encrypted = encryption.encrypt(data).unwrap();

        assert_eq!(encrypted.metadata.algorithm, "AES-256-GCM");
        assert_eq!(encrypted.metadata.key_derivation, "Random");
        assert!(!encrypted.metadata.checksum.is_empty());
    }
}
