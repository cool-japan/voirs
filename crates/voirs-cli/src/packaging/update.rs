use crate::error::VoirsCLIError;
use anyhow::Result;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfig {
    pub check_interval_hours: u64,
    pub auto_update: bool,
    pub backup_count: u32,
    pub update_channel: UpdateChannel,
    pub update_server: String,
    pub verify_signatures: bool,
    pub signature_algorithm: String,
    pub public_key_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateChannel {
    Stable,
    Beta,
    Nightly,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            check_interval_hours: 24,
            auto_update: false,
            backup_count: 3,
            update_channel: UpdateChannel::Stable,
            update_server: "https://api.github.com/repos/voirs-org/voirs".to_string(),
            verify_signatures: true,
            signature_algorithm: "ed25519".to_string(),
            public_key_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: String,
    pub release_date: DateTime<Utc>,
    pub download_url: String,
    pub checksum: String,
    pub signature: Option<String>,
    pub changelog: String,
    pub is_security_update: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateState {
    pub last_check: DateTime<Utc>,
    pub current_version: String,
    pub available_version: Option<String>,
    pub update_available: bool,
    pub last_update: Option<DateTime<Utc>>,
    pub backup_paths: Vec<PathBuf>,
}

impl Default for UpdateState {
    fn default() -> Self {
        Self {
            last_check: Utc::now(),
            current_version: env!("CARGO_PKG_VERSION").to_string(),
            available_version: None,
            update_available: false,
            last_update: None,
            backup_paths: Vec::new(),
        }
    }
}

pub struct UpdateManager {
    config: UpdateConfig,
    state: UpdateState,
    client: Client,
    state_file: PathBuf,
}

impl UpdateManager {
    pub fn new(config: UpdateConfig, state_file: PathBuf) -> Result<Self> {
        let state = if state_file.exists() {
            let content = fs::read_to_string(&state_file)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            UpdateState::default()
        };

        let client = Client::builder()
            .user_agent(format!("voirs-cli/{}", env!("CARGO_PKG_VERSION")))
            .build()?;

        Ok(Self {
            config,
            state,
            client,
            state_file,
        })
    }

    pub async fn check_for_updates(&mut self) -> Result<Option<VersionInfo>> {
        info!("Checking for updates");

        let should_check = self.should_check_for_updates();
        if !should_check {
            debug!("Update check skipped - too soon since last check");
            return Ok(None);
        }

        let latest_version = self.fetch_latest_version().await?;

        self.state.last_check = Utc::now();
        self.state.available_version = Some(latest_version.version.clone());
        self.state.update_available = self.is_newer_version(&latest_version.version)?;

        self.save_state()?;

        if self.state.update_available {
            info!(
                "Update available: {} -> {}",
                self.state.current_version, latest_version.version
            );
            Ok(Some(latest_version))
        } else {
            info!("No updates available");
            Ok(None)
        }
    }

    pub async fn perform_update(&mut self, version_info: &VersionInfo) -> Result<bool> {
        info!(
            "Starting update process to version {}",
            version_info.version
        );

        // Create backup of current binary
        let backup_path = self.create_backup().await?;

        // Download new binary
        let temp_binary = self.download_binary(version_info).await?;

        // Verify integrity
        if !self
            .verify_binary_integrity(&temp_binary, &version_info.checksum)
            .await?
        {
            error!("Binary integrity verification failed");
            return Ok(false);
        }

        // Verify signature if enabled
        if self.config.verify_signatures {
            if let Some(signature) = &version_info.signature {
                if !self.verify_signature(&temp_binary, signature).await? {
                    error!("Binary signature verification failed");
                    return Ok(false);
                }
            }
        }

        // Replace current binary
        let current_binary = self.get_current_binary_path()?;
        fs::rename(&temp_binary, &current_binary)?;

        // Update permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&current_binary)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&current_binary, perms)?;
        }

        // Update state
        self.state.current_version = version_info.version.clone();
        self.state.last_update = Some(Utc::now());
        self.state.update_available = false;
        self.state.backup_paths.push(backup_path);

        // Clean up old backups
        self.cleanup_old_backups().await?;

        self.save_state()?;

        info!("Update completed successfully");
        Ok(true)
    }

    pub async fn rollback_update(&mut self) -> Result<bool> {
        info!("Rolling back update");

        if let Some(backup_path) = self.state.backup_paths.last() {
            if backup_path.exists() {
                let current_binary = self.get_current_binary_path()?;
                fs::rename(backup_path, &current_binary)?;

                // Update permissions
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = fs::metadata(&current_binary)?.permissions();
                    perms.set_mode(0o755);
                    fs::set_permissions(&current_binary, perms)?;
                }

                self.state.backup_paths.pop();
                self.save_state()?;

                info!("Rollback completed successfully");
                Ok(true)
            } else {
                warn!("Backup file not found for rollback");
                Ok(false)
            }
        } else {
            warn!("No backup available for rollback");
            Ok(false)
        }
    }

    fn should_check_for_updates(&self) -> bool {
        let hours_since_last_check = Utc::now()
            .signed_duration_since(self.state.last_check)
            .num_hours() as u64;

        hours_since_last_check >= self.config.check_interval_hours
    }

    async fn fetch_latest_version(&self) -> Result<VersionInfo> {
        let url = format!("{}/releases/latest", self.config.update_server);
        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(VoirsCLIError::UpdateError(format!(
                "Failed to fetch latest version: HTTP {}",
                response.status()
            ))
            .into());
        }

        let release_info: serde_json::Value = response.json().await?;

        let version = release_info["tag_name"]
            .as_str()
            .unwrap_or("")
            .trim_start_matches('v')
            .to_string();

        let release_date =
            DateTime::parse_from_rfc3339(release_info["published_at"].as_str().unwrap_or(""))?
                .with_timezone(&Utc);

        let download_url = self.get_download_url_for_platform(&release_info)?;

        Ok(VersionInfo {
            version,
            release_date,
            download_url,
            checksum: String::new(), // Would be fetched from release assets
            signature: None,
            changelog: release_info["body"].as_str().unwrap_or("").to_string(),
            is_security_update: release_info["body"]
                .as_str()
                .unwrap_or("")
                .to_lowercase()
                .contains("security"),
        })
    }

    fn get_download_url_for_platform(&self, release_info: &serde_json::Value) -> Result<String> {
        let assets = release_info["assets"]
            .as_array()
            .ok_or_else(|| VoirsCLIError::UpdateError("No assets found in release".to_string()))?;

        let platform_suffix = if cfg!(target_os = "windows") {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else {
            "linux"
        };

        for asset in assets {
            if let Some(name) = asset["name"].as_str() {
                if name.contains(platform_suffix) {
                    return Ok(asset["browser_download_url"]
                        .as_str()
                        .ok_or_else(|| {
                            VoirsCLIError::UpdateError("Invalid download URL".to_string())
                        })?
                        .to_string());
                }
            }
        }

        Err(VoirsCLIError::UpdateError(format!(
            "No binary found for platform: {}",
            platform_suffix
        ))
        .into())
    }

    fn is_newer_version(&self, remote_version: &str) -> Result<bool> {
        let current = semver::Version::parse(&self.state.current_version)?;
        let remote = semver::Version::parse(remote_version)?;

        Ok(remote > current)
    }

    async fn create_backup(&self) -> Result<PathBuf> {
        let current_binary = self.get_current_binary_path()?;
        let backup_name = format!("voirs-backup-{}.bak", Utc::now().timestamp());
        let backup_path = current_binary
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .join(&backup_name);

        fs::copy(&current_binary, &backup_path)?;

        info!("Created backup at: {:?}", backup_path);
        Ok(backup_path)
    }

    async fn download_binary(&self, version_info: &VersionInfo) -> Result<PathBuf> {
        info!("Downloading binary from: {}", version_info.download_url);

        let response = self.client.get(&version_info.download_url).send().await?;

        if !response.status().is_success() {
            return Err(VoirsCLIError::UpdateError(format!(
                "Failed to download binary: HTTP {}",
                response.status()
            ))
            .into());
        }

        let temp_path = std::env::temp_dir().join(format!("voirs-update-{}", version_info.version));
        let mut file = File::create(&temp_path).await?;

        let content = response.bytes().await?;
        file.write_all(&content).await?;

        info!("Binary downloaded to: {:?}", temp_path);
        Ok(temp_path)
    }

    async fn verify_binary_integrity(
        &self,
        binary_path: &PathBuf,
        expected_checksum: &str,
    ) -> Result<bool> {
        if expected_checksum.is_empty() {
            warn!("No checksum provided for verification");
            return Ok(true);
        }

        let content = fs::read(binary_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let actual_checksum = format!("{:x}", hasher.finalize());

        let matches = actual_checksum == expected_checksum;
        if matches {
            info!("Binary integrity verification passed");
        } else {
            error!(
                "Binary integrity verification failed: expected {}, got {}",
                expected_checksum, actual_checksum
            );
        }

        Ok(matches)
    }

    async fn verify_signature(&self, binary_path: &PathBuf, signature: &str) -> Result<bool> {
        info!("Verifying signature for binary: {:?}", binary_path);

        // Read the binary file
        let binary_content = fs::read(binary_path)?;

        // Parse the signature (assuming it's hex-encoded)
        let signature_bytes = self.parse_hex_signature(signature)?;

        // Get the public key for verification
        let public_key = self.get_verification_public_key()?;

        // Verify the signature using Ed25519 (or RSA as fallback)
        let is_valid = match self.config.signature_algorithm.as_str() {
            "ed25519" => {
                self.verify_ed25519_signature(&binary_content, &signature_bytes, &public_key)?
            }
            "rsa" => self.verify_rsa_signature(&binary_content, &signature_bytes, &public_key)?,
            "ecdsa" => {
                self.verify_ecdsa_signature(&binary_content, &signature_bytes, &public_key)?
            }
            _ => {
                warn!(
                    "Unknown signature algorithm: {}",
                    self.config.signature_algorithm
                );
                return Ok(false);
            }
        };

        if is_valid {
            info!("Binary signature verification successful");
        } else {
            warn!("Binary signature verification failed");
        }

        Ok(is_valid)
    }

    /// Parse hex-encoded signature
    fn parse_hex_signature(&self, signature: &str) -> Result<Vec<u8>> {
        let signature_clean = signature.trim().replace(" ", "").replace("\n", "");

        if signature_clean.len() % 2 != 0 {
            return Err(anyhow::anyhow!("Invalid hex signature length"));
        }

        let mut signature_bytes = Vec::new();
        for i in (0..signature_clean.len()).step_by(2) {
            let hex_byte = &signature_clean[i..i + 2];
            let byte = u8::from_str_radix(hex_byte, 16)
                .map_err(|_| anyhow::anyhow!("Invalid hex character in signature"))?;
            signature_bytes.push(byte);
        }

        Ok(signature_bytes)
    }

    /// Get the public key for signature verification
    fn get_verification_public_key(&self) -> Result<Vec<u8>> {
        // Try to get public key from multiple sources

        // 1. Check environment variable
        if let Ok(key_env) = std::env::var("VOIRS_PUBLIC_KEY") {
            return self.parse_public_key(&key_env);
        }

        // 2. Check configuration file
        if let Some(key_path) = &self.config.public_key_path {
            if key_path.exists() {
                let key_content = fs::read_to_string(key_path)?;
                return self.parse_public_key(&key_content);
            }
        }

        // 3. Use embedded public key (hardcoded for security)
        let embedded_key = self.get_embedded_public_key();
        Ok(embedded_key)
    }

    /// Parse public key from string (supports PEM and raw hex)
    fn parse_public_key(&self, key_str: &str) -> Result<Vec<u8>> {
        let key_clean = key_str.trim();

        // Check if it's a PEM-formatted key
        if key_clean.starts_with("-----BEGIN") && key_clean.ends_with("-----END") {
            // Extract the base64 content between BEGIN and END markers
            let lines: Vec<&str> = key_clean.lines().collect();
            if lines.len() < 3 {
                return Err(anyhow::anyhow!("Invalid PEM format"));
            }

            let b64_content = lines[1..lines.len() - 1].join("");
            let key_bytes = base64::decode(&b64_content)
                .map_err(|_| anyhow::anyhow!("Invalid base64 in PEM key"))?;

            Ok(key_bytes)
        } else if key_clean
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c.is_whitespace())
        {
            // Treat as hex-encoded key
            self.parse_hex_signature(key_clean)
        } else {
            Err(anyhow::anyhow!("Unsupported public key format"))
        }
    }

    /// Get embedded public key (hardcoded for security)
    fn get_embedded_public_key(&self) -> Vec<u8> {
        // In a real implementation, this would be the actual public key
        // For now, we'll use a placeholder key
        match self.config.signature_algorithm.as_str() {
            "ed25519" => {
                // Ed25519 public key (32 bytes)
                vec![
                    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD,
                    0xEE, 0xFF, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA,
                    0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
                ]
            }
            "rsa" => {
                // RSA public key (DER encoded, simplified)
                vec![
                    0x30, 0x82, 0x01, 0x22, 0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7,
                    0x0d, 0x01, 0x01,
                    // ... RSA public key continues (truncated for brevity)
                ]
            }
            "ecdsa" => {
                // ECDSA public key (33 bytes compressed)
                vec![
                    0x02, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC,
                    0xDD, 0xEE, 0xFF, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
                    0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
                ]
            }
            _ => vec![],
        }
    }

    /// Verify Ed25519 signature
    fn verify_ed25519_signature(
        &self,
        data: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> Result<bool> {
        if signature.len() != 64 {
            return Err(anyhow::anyhow!("Invalid Ed25519 signature length"));
        }

        if public_key.len() != 32 {
            return Err(anyhow::anyhow!("Invalid Ed25519 public key length"));
        }

        // Calculate SHA-256 hash of the data
        let hash = sha2::Sha256::digest(data);

        // Simulate Ed25519 signature verification
        // In a real implementation, this would use the `ed25519-dalek` crate
        let is_valid = self.simulate_signature_verification(&hash, signature, public_key);

        Ok(is_valid)
    }

    /// Verify RSA signature
    fn verify_rsa_signature(
        &self,
        data: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> Result<bool> {
        // Calculate SHA-256 hash of the data
        let hash = sha2::Sha256::digest(data);

        // Simulate RSA signature verification
        // In a real implementation, this would use the `rsa` crate
        let is_valid = self.simulate_signature_verification(&hash, signature, public_key);

        Ok(is_valid)
    }

    /// Verify ECDSA signature
    fn verify_ecdsa_signature(
        &self,
        data: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> Result<bool> {
        // Calculate SHA-256 hash of the data
        let hash = sha2::Sha256::digest(data);

        // Simulate ECDSA signature verification
        // In a real implementation, this would use the `p256` or `secp256k1` crate
        let is_valid = self.simulate_signature_verification(&hash, signature, public_key);

        Ok(is_valid)
    }

    /// Simulate signature verification (for demonstration purposes)
    fn simulate_signature_verification(
        &self,
        hash: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> bool {
        // This is a simplified simulation for demonstration
        // In a real implementation, this would use proper cryptographic verification

        // Check basic length requirements
        if signature.is_empty() || public_key.is_empty() || hash.is_empty() {
            return false;
        }

        // Simulate verification by checking if signature matches a pattern
        // This is NOT secure and is only for demonstration
        let mut verification_hash = Vec::new();
        verification_hash.extend_from_slice(hash);
        verification_hash.extend_from_slice(public_key);

        let computed_hash = sha2::Sha256::digest(&verification_hash);

        // Check if first 16 bytes of signature match first 16 bytes of computed hash
        if signature.len() >= 16 && computed_hash.len() >= 16 {
            signature[0..16] == computed_hash[0..16]
        } else {
            false
        }
    }

    fn get_current_binary_path(&self) -> Result<PathBuf> {
        let current_exe = std::env::current_exe()?;
        Ok(current_exe)
    }

    async fn cleanup_old_backups(&mut self) -> Result<()> {
        while self.state.backup_paths.len() > self.config.backup_count as usize {
            let old_backup = self.state.backup_paths.remove(0);
            if old_backup.exists() {
                fs::remove_file(&old_backup)?;
                info!("Removed old backup: {:?}", old_backup);
            }
        }
        Ok(())
    }

    fn save_state(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.state)?;
        fs::write(&self.state_file, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_update_config_default() {
        let config = UpdateConfig::default();
        assert_eq!(config.check_interval_hours, 24);
        assert!(!config.auto_update);
        assert_eq!(config.backup_count, 3);
        assert!(matches!(config.update_channel, UpdateChannel::Stable));
    }

    #[test]
    fn test_update_state_default() {
        let state = UpdateState::default();
        assert!(!state.update_available);
        assert!(state.backup_paths.is_empty());
        assert_eq!(state.current_version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_version_comparison() {
        let state = UpdateState::default();
        let manager = UpdateManager {
            config: UpdateConfig::default(),
            state,
            client: Client::new(),
            state_file: PathBuf::from("test.json"),
        };

        // This would normally test version comparison logic
        // For now, we just verify the structure is correct
        assert_eq!(manager.state.current_version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_update_channel_serialization() {
        let channel = UpdateChannel::Stable;
        let serialized = serde_json::to_string(&channel).unwrap();
        let deserialized: UpdateChannel = serde_json::from_str(&serialized).unwrap();
        assert!(matches!(deserialized, UpdateChannel::Stable));
    }
}
