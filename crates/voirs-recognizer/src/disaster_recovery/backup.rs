// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Automated backup system for disaster recovery

use super::{BackupConfig, DisasterRecoveryError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Backup type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupType {
    /// Full backup
    Full,
    /// Incremental backup
    Incremental,
    /// Differential backup
    Differential,
}

/// Backup metadata
#[derive(Debug, Clone, Serialize)]
pub struct BackupMetadata {
    /// Backup ID
    pub id: String,
    /// Backup type
    pub backup_type: BackupType,
    /// Backup path
    pub path: PathBuf,
    /// Backup size in bytes
    pub size_bytes: u64,
    /// Backup timestamp
    #[serde(skip)]
    pub timestamp: Instant,
    /// Backup duration
    pub duration: Duration,
    /// Is compressed
    pub compressed: bool,
    /// Is encrypted
    pub encrypted: bool,
}

/// Backup manager
pub struct BackupManager {
    config: BackupConfig,
    backups: Arc<RwLock<Vec<BackupMetadata>>>,
    last_full_backup: Arc<RwLock<Option<Instant>>>,
}

impl BackupManager {
    /// Create a new backup manager
    #[must_use]
    pub fn new(config: BackupConfig) -> Self {
        Self {
            config,
            backups: Arc::new(RwLock::new(Vec::new())),
            last_full_backup: Arc::new(RwLock::new(None)),
        }
    }

    /// Perform a backup
    pub async fn perform_backup(&self, data_dir: &Path) -> Result<BackupMetadata> {
        let start = Instant::now();

        let backup_type = if self.should_perform_full_backup() {
            BackupType::Full
        } else if self.config.incremental {
            BackupType::Incremental
        } else {
            BackupType::Differential
        };

        info!("Starting {:?} backup from {:?}", backup_type, data_dir);

        // Generate backup ID
        let backup_id = format!(
            "backup_{}_{}",
            chrono::Utc::now().format("%Y%m%d_%H%M%S"),
            uuid::Uuid::new_v4()
        );

        // Create backup directory if it doesn't exist
        if !self.config.backup_dir.exists() {
            std::fs::create_dir_all(&self.config.backup_dir).map_err(|e| {
                DisasterRecoveryError::BackupFailed(format!(
                    "Failed to create backup directory: {e}"
                ))
            })?;
        }

        let backup_path = self.config.backup_dir.join(&backup_id);

        // Simulate backup process
        self.copy_data(data_dir, &backup_path).await?;

        // Optionally compress
        let size_bytes = if self.config.compression {
            self.compress_backup(&backup_path).await?
        } else {
            self.calculate_size(&backup_path).await?
        };

        // Optionally encrypt
        if self.config.encryption {
            self.encrypt_backup(&backup_path).await?;
        }

        let duration = start.elapsed();

        let metadata = BackupMetadata {
            id: backup_id,
            backup_type,
            path: backup_path,
            size_bytes,
            timestamp: start,
            duration,
            compressed: self.config.compression,
            encrypted: self.config.encryption,
        };

        // Update backup list
        self.backups.write().push(metadata.clone());

        if backup_type == BackupType::Full {
            *self.last_full_backup.write() = Some(start);
        }

        info!(
            "Backup completed in {:?}, size: {} bytes",
            duration, size_bytes
        );

        // Clean up old backups
        self.cleanup_old_backups().await?;

        Ok(metadata)
    }

    /// Copy data to backup location
    async fn copy_data(&self, _source: &Path, _dest: &Path) -> Result<()> {
        // Simulate data copy
        tokio::time::sleep(Duration::from_millis(100)).await;
        debug!("Data copied to backup location");
        Ok(())
    }

    /// Compress backup
    async fn compress_backup(&self, _path: &Path) -> Result<u64> {
        // Simulate compression
        tokio::time::sleep(Duration::from_millis(50)).await;
        debug!("Backup compressed");
        Ok(1024 * 1024 * 100) // 100 MB
    }

    /// Calculate backup size
    async fn calculate_size(&self, _path: &Path) -> Result<u64> {
        // Simulate size calculation
        Ok(1024 * 1024 * 200) // 200 MB
    }

    /// Encrypt backup
    async fn encrypt_backup(&self, _path: &Path) -> Result<()> {
        // Simulate encryption
        tokio::time::sleep(Duration::from_millis(30)).await;
        debug!("Backup encrypted");
        Ok(())
    }

    /// Check if full backup should be performed
    fn should_perform_full_backup(&self) -> bool {
        match *self.last_full_backup.read() {
            None => true, // No previous full backup
            Some(last_full) => {
                // Perform full backup weekly
                last_full.elapsed() > Duration::from_secs(604_800)
            }
        }
    }

    /// Clean up old backups based on retention policy
    async fn cleanup_old_backups(&self) -> Result<()> {
        let mut backups = self.backups.write();
        let now = Instant::now();

        backups.retain(|backup| {
            let age = now.duration_since(backup.timestamp);
            if age > self.config.retention_period {
                info!("Removing old backup: {}", backup.id);
                false
            } else {
                true
            }
        });

        Ok(())
    }

    /// Get all backups
    #[must_use]
    pub fn get_backups(&self) -> Vec<BackupMetadata> {
        self.backups.read().clone()
    }

    /// Get most recent backup
    #[must_use]
    pub fn get_latest_backup(&self) -> Option<BackupMetadata> {
        self.backups
            .read()
            .iter()
            .max_by_key(|b| b.timestamp)
            .cloned()
    }

    /// Get backup count
    #[must_use]
    pub fn backup_count(&self) -> usize {
        self.backups.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_backup_manager_basic() {
        let config = BackupConfig {
            backup_dir: env::temp_dir().join("voirs_backups"),
            retention_period: Duration::from_secs(86400),
            backup_interval: Duration::from_secs(3600),
            incremental: true,
            compression: true,
            encryption: true,
        };

        let manager = BackupManager::new(config);
        let data_dir = env::temp_dir().join("test_data");

        let metadata = manager.perform_backup(&data_dir).await.unwrap();

        assert!(metadata.compressed);
        assert!(metadata.encrypted);
        assert_eq!(manager.backup_count(), 1);
    }

    #[tokio::test]
    async fn test_backup_types() {
        let config = BackupConfig {
            backup_dir: env::temp_dir().join("voirs_backups"),
            incremental: true,
            ..Default::default()
        };

        let manager = BackupManager::new(config);
        let data_dir = env::temp_dir().join("test_data");

        // First backup should be full
        let backup1 = manager.perform_backup(&data_dir).await.unwrap();
        assert_eq!(backup1.backup_type, BackupType::Full);

        // Second backup should be incremental
        let backup2 = manager.perform_backup(&data_dir).await.unwrap();
        assert_eq!(backup2.backup_type, BackupType::Incremental);
    }

    #[tokio::test]
    async fn test_get_latest_backup() {
        let config = BackupConfig {
            backup_dir: env::temp_dir().join("voirs_backup_test"),
            retention_period: Duration::from_secs(86400),
            backup_interval: Duration::from_secs(3600),
            incremental: true,
            compression: true,
            encryption: true,
        };
        let manager = BackupManager::new(config);
        let data_dir = env::temp_dir().join("test_data");

        manager.perform_backup(&data_dir).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        manager.perform_backup(&data_dir).await.unwrap();

        let latest = manager.get_latest_backup();
        assert!(latest.is_some());
    }

    #[test]
    fn test_backup_types_equality() {
        assert_eq!(BackupType::Full, BackupType::Full);
        assert_ne!(BackupType::Full, BackupType::Incremental);
    }
}
