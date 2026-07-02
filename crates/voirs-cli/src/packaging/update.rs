use super::signature;
use crate::error::VoirsCLIError;
use anyhow::Result;
use chrono::{DateTime, Utc};
use hex;
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

        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        voirs_acoustic::hub::ensure_crypto_provider();

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

        // Verify signature if enabled. `require_signature_for_update` is the
        // fail-closed gate: it returns `Err` when verification is required
        // but no signature was supplied, instead of silently skipping
        // straight through to `fs::rename` over the live binary.
        if let Some(signature) = self.require_signature_for_update(version_info)? {
            if !self.verify_signature(&temp_binary, signature).await? {
                error!("Binary signature verification failed");
                return Ok(false);
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

        // KNOWN GAP (out of scope for the signature-verification fix in
        // `packaging::signature`): this does not yet fetch the real
        // checksum/signature files published alongside a GitHub release
        // asset. Wiring that up requires deciding on a concrete naming
        // convention for the checksum/signature assets and is left for a
        // follow-up change rather than being fabricated here. A direct,
        // intentional consequence: with `UpdateConfig::verify_signatures =
        // true` (the default), `UpdateManager::perform_update` will now
        // correctly refuse (fail closed) to apply any update fetched via
        // this method, because `signature` below is always `None`. That is
        // safe-by-default behavior, not a bug — self-update is not
        // end-to-end functional until both this fetch and a real embedded
        // public key (see `packaging::signature::embedded_public_key`) are
        // provisioned.
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
        let actual_checksum = hex::encode(hasher.finalize());

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

    /// Decide what, if anything, must be checked against `version_info`'s
    /// signature before an update is allowed to proceed.
    ///
    /// - Returns `Ok(None)` when `UpdateConfig::verify_signatures` is
    ///   disabled (nothing to check).
    /// - Returns `Ok(Some(signature))` when verification is enabled and a
    ///   signature was supplied — the caller must then actually verify it.
    /// - Returns `Err` when verification is enabled but no signature was
    ///   supplied at all.
    ///
    /// SECURITY: that last case is the fail-closed fix for a real bypass —
    /// the original code only ever invoked signature verification inside an
    /// `if let Some(signature) = &version_info.signature`, so a
    /// server-controlled `VersionInfo` with `signature: None` (which is
    /// exactly what `fetch_latest_version` produces today — see the KNOWN
    /// GAP comment there) silently skipped verification entirely and fell
    /// through to `fs::rename` over the live binary. Extracted into its own
    /// method so this decision can be unit-tested directly without needing
    /// to drive the network- and filesystem-heavy rest of the update
    /// pipeline.
    fn require_signature_for_update<'a>(
        &self,
        version_info: &'a VersionInfo,
    ) -> Result<Option<&'a str>> {
        if !self.config.verify_signatures {
            return Ok(None);
        }

        match &version_info.signature {
            Some(signature) => Ok(Some(signature.as_str())),
            None => {
                error!(
                    "verify_signatures is enabled but no signature was provided for this \
                     update; refusing to apply it"
                );
                Err(anyhow::anyhow!(
                    "update rejected: signature verification is required \
                     (UpdateConfig::verify_signatures = true) but VersionInfo contained no \
                     signature"
                ))
            }
        }
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

        if !signature_clean.len().is_multiple_of(2) {
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

    /// Get the public key for signature verification.
    ///
    /// Tries, in order: the `VOIRS_PUBLIC_KEY` environment variable, the
    /// configured `public_key_path`, and finally a compiled-in embedded key
    /// (see [`signature::embedded_public_key`]).
    ///
    /// SECURITY: if none of these sources yields a key, this returns `Err`
    /// rather than falling back to any default. A missing key must never be
    /// treated as "verification passed" — see the module-level doc comment
    /// on `packaging::signature` for the full rationale and for what a
    /// maintainer needs to do to provision a real key.
    fn get_verification_public_key(&self) -> Result<Vec<u8>> {
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

        // 3. Use embedded public key, if one has been provisioned.
        if let Some(embedded_key) = signature::embedded_public_key(&self.config.signature_algorithm)
        {
            return Ok(embedded_key.to_vec());
        }

        // 4. Fail closed: no real key configured anywhere.
        Err(anyhow::anyhow!(
            "no update-signing public key configured for algorithm '{}': set the \
             VOIRS_PUBLIC_KEY environment variable, configure UpdateConfig::public_key_path, \
             or embed a real key in packaging::signature::embedded_public_key. Refusing to \
             treat an unverifiable update as trusted.",
            self.config.signature_algorithm
        ))
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

    /// Verify a real Ed25519 signature.
    ///
    /// Delegates to [`signature::verify_ed25519`], which performs genuine
    /// asymmetric cryptographic verification via `ed25519-dalek`. See that
    /// module's doc comment for the message convention (the signature
    /// covers the raw update-package bytes, not a pre-hashed digest).
    fn verify_ed25519_signature(
        &self,
        data: &[u8],
        signature_bytes: &[u8],
        public_key: &[u8],
    ) -> Result<bool> {
        signature::verify_ed25519(data, signature_bytes, public_key)
    }

    /// Verify a real RSASSA-PKCS1-v1_5 (SHA-256) signature.
    ///
    /// Delegates to [`signature::verify_rsa_pkcs1v15_sha256`], which
    /// performs genuine asymmetric cryptographic verification via the `rsa`
    /// crate.
    fn verify_rsa_signature(
        &self,
        data: &[u8],
        signature_bytes: &[u8],
        public_key: &[u8],
    ) -> Result<bool> {
        signature::verify_rsa_pkcs1v15_sha256(data, signature_bytes, public_key)
    }

    /// Verify a real ECDSA P-256 (SHA-256) signature.
    ///
    /// Delegates to [`signature::verify_ecdsa_p256_sha256`], which performs
    /// genuine asymmetric cryptographic verification via the `p256` crate.
    fn verify_ecdsa_signature(
        &self,
        data: &[u8],
        signature_bytes: &[u8],
        public_key: &[u8],
    ) -> Result<bool> {
        signature::verify_ecdsa_p256_sha256(data, signature_bytes, public_key)
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
        // `Client::new()` panics when no rustls CryptoProvider is installed
        // (reqwest is built with `rustls-no-provider`). `UpdateManager::new`
        // installs it, but this test constructs the struct directly, so install
        // it here too. Once-guarded; safe to repeat.
        voirs_acoustic::hub::ensure_crypto_provider();

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

    /// Builds an `UpdateManager` directly (bypassing `UpdateManager::new`'s
    /// state-file I/O), mirroring the existing `test_version_comparison`
    /// pattern above.
    fn test_manager(config: UpdateConfig) -> UpdateManager {
        voirs_acoustic::hub::ensure_crypto_provider();
        UpdateManager {
            config,
            state: UpdateState::default(),
            client: Client::new(),
            state_file: PathBuf::from("test.json"),
        }
    }

    #[test]
    fn test_get_verification_public_key_fails_closed_with_no_key_source() {
        // Guard against leftover state from another test in this same
        // process (nextest runs each test in its own process by default,
        // but this keeps the test correct under `cargo test` too).
        std::env::remove_var("VOIRS_PUBLIC_KEY");

        let manager = test_manager(UpdateConfig::default());

        let result = manager.get_verification_public_key();
        assert!(
            result.is_err(),
            "with no VOIRS_PUBLIC_KEY env var, no public_key_path, and no embedded key, \
             the lookup must fail closed instead of returning a fabricated default key"
        );
    }

    #[test]
    fn test_get_verification_public_key_uses_env_var_when_present() {
        std::env::remove_var("VOIRS_PUBLIC_KEY");
        std::env::set_var("VOIRS_PUBLIC_KEY", "aabbccdd");

        let manager = test_manager(UpdateConfig::default());
        let key = manager
            .get_verification_public_key()
            .expect("a hex-encoded VOIRS_PUBLIC_KEY must be usable as a key source");
        assert_eq!(key, vec![0xAA, 0xBB, 0xCC, 0xDD]);

        std::env::remove_var("VOIRS_PUBLIC_KEY");
    }

    #[tokio::test]
    async fn test_verify_signature_end_to_end_accepts_genuine_ed25519_signature() {
        use ed25519_dalek::{Signer, SigningKey};
        use rand_core::{OsRng, RngCore};

        std::env::remove_var("VOIRS_PUBLIC_KEY");

        let mut seed = [0_u8; 32];
        OsRng.fill_bytes(&mut seed);
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();

        // Stand-in for a downloaded update package.
        let temp_dir = TempDir::new().expect("failed to create temp dir for test");
        let binary_path = temp_dir.path().join("fake-update-binary");
        let binary_content = b"pretend-this-is-a-real-voirs-cli-binary".to_vec();
        fs::write(&binary_path, &binary_content).expect("failed to write fake binary");

        let signature_hex = hex::encode(signing_key.sign(&binary_content).to_bytes());
        std::env::set_var("VOIRS_PUBLIC_KEY", hex::encode(verifying_key.as_bytes()));

        let manager = test_manager(UpdateConfig::default()); // signature_algorithm = "ed25519"

        let is_valid = manager
            .verify_signature(&binary_path, &signature_hex)
            .await
            .expect("verification with a well-formed real key/signature must not error");
        assert!(
            is_valid,
            "a genuine end-to-end Ed25519 signature must verify through UpdateManager::verify_signature"
        );

        std::env::remove_var("VOIRS_PUBLIC_KEY");
    }

    #[tokio::test]
    async fn test_verify_signature_end_to_end_rejects_tampered_signature() {
        use ed25519_dalek::{Signer, SigningKey};
        use rand_core::{OsRng, RngCore};

        std::env::remove_var("VOIRS_PUBLIC_KEY");

        let mut seed = [0_u8; 32];
        OsRng.fill_bytes(&mut seed);
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();

        let temp_dir = TempDir::new().expect("failed to create temp dir for test");
        let binary_path = temp_dir.path().join("fake-update-binary");
        let binary_content = b"pretend-this-is-a-real-voirs-cli-binary".to_vec();
        fs::write(&binary_path, &binary_content).expect("failed to write fake binary");

        let mut forged_signature_bytes = signing_key.sign(&binary_content).to_bytes().to_vec();
        let last = forged_signature_bytes.len() - 1;
        forged_signature_bytes[last] ^= 0xFF;
        let forged_signature_hex = hex::encode(forged_signature_bytes);

        std::env::set_var("VOIRS_PUBLIC_KEY", hex::encode(verifying_key.as_bytes()));

        let manager = test_manager(UpdateConfig::default());

        let is_valid = manager
            .verify_signature(&binary_path, &forged_signature_hex)
            .await
            .expect("verification with a well-formed but wrong signature must not error");
        assert!(
            !is_valid,
            "a tampered signature must be rejected end-to-end through UpdateManager::verify_signature"
        );

        std::env::remove_var("VOIRS_PUBLIC_KEY");
    }

    fn version_info_with_signature(signature: Option<String>) -> VersionInfo {
        VersionInfo {
            version: "9.9.9".to_string(),
            release_date: Utc::now(),
            download_url: "https://example.invalid/voirs-update".to_string(),
            checksum: String::new(),
            signature,
            changelog: String::new(),
            is_security_update: false,
        }
    }

    #[test]
    fn test_require_signature_for_update_fails_closed_when_missing() {
        // This is the exact security-critical branch that `perform_update`
        // relies on: `verify_signatures = true` (the default) combined with
        // a `VersionInfo` that carries no signature at all — precisely what
        // `fetch_latest_version` produces today — must be a hard error, not
        // a silent "nothing to verify, proceed".
        let manager = test_manager(UpdateConfig::default());
        assert!(
            manager.config.verify_signatures,
            "test assumes the default has verification enabled"
        );

        let version_info = version_info_with_signature(None);
        let result = manager.require_signature_for_update(&version_info);

        assert!(
            result.is_err(),
            "verify_signatures=true with no signature present must fail closed"
        );
    }

    #[test]
    fn test_require_signature_for_update_passes_through_when_present() {
        let manager = test_manager(UpdateConfig::default());
        let version_info = version_info_with_signature(Some("deadbeef".to_string()));

        let result = manager
            .require_signature_for_update(&version_info)
            .expect("a present signature must not itself be treated as an error");

        assert_eq!(
            result,
            Some("deadbeef"),
            "the supplied signature must be passed through unchanged for actual verification"
        );
    }

    #[test]
    fn test_require_signature_for_update_skips_when_verification_disabled() {
        let manager = test_manager(UpdateConfig {
            verify_signatures: false,
            ..UpdateConfig::default()
        });
        let version_info = version_info_with_signature(None);

        let result = manager
            .require_signature_for_update(&version_info)
            .expect("disabling verification must never itself be an error");

        assert_eq!(
            result, None,
            "with verify_signatures=false there is nothing to check, regardless of signature presence"
        );
    }
}
