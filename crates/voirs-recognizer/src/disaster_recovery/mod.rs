// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Disaster recovery and business continuity planning
//!
//! This module provides comprehensive disaster recovery capabilities including
//! automated backups, point-in-time recovery, disaster recovery testing, and
//! compliance with recovery time objectives (RTO) and recovery point objectives (RPO).

pub mod backup;
pub mod recovery;
pub mod replication;
pub mod testing;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

/// Disaster recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisasterRecoveryConfig {
    /// Enable disaster recovery features
    pub enabled: bool,
    /// Backup configuration
    pub backup: BackupConfig,
    /// Recovery configuration
    pub recovery: RecoveryConfig,
    /// Replication strategy
    pub replication_strategy: ReplicationStrategy,
    /// Enable automated testing
    pub automated_testing: bool,
    /// Testing interval
    pub testing_interval: Duration,
}

impl Default for DisasterRecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backup: BackupConfig::default(),
            recovery: RecoveryConfig::default(),
            replication_strategy: ReplicationStrategy::Synchronous,
            automated_testing: true,
            testing_interval: Duration::from_secs(86400), // Daily
        }
    }
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Backup directory
    pub backup_dir: PathBuf,
    /// Backup retention period
    pub retention_period: Duration,
    /// Backup interval
    pub backup_interval: Duration,
    /// Enable incremental backups
    pub incremental: bool,
    /// Enable compression
    pub compression: bool,
    /// Enable encryption
    pub encryption: bool,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            backup_dir: PathBuf::from("/var/lib/voirs/backups"),
            retention_period: Duration::from_secs(2_592_000), // 30 days
            backup_interval: Duration::from_secs(3600),       // Hourly
            incremental: true,
            compression: true,
            encryption: true,
        }
    }
}

/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Recovery Time Objective (RTO) - maximum acceptable downtime
    pub rto: Duration,
    /// Recovery Point Objective (RPO) - maximum acceptable data loss
    pub rpo: Duration,
    /// Enable point-in-time recovery
    pub point_in_time_recovery: bool,
    /// Enable automated recovery
    pub automated_recovery: bool,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            rto: Duration::from_secs(300), // 5 minutes
            rpo: Duration::from_secs(60),  // 1 minute
            point_in_time_recovery: true,
            automated_recovery: true,
        }
    }
}

/// Replication strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    /// Synchronous replication
    Synchronous,
    /// Asynchronous replication
    Asynchronous,
    /// Multi-region replication
    MultiRegion,
    /// Cross-cloud replication
    CrossCloud,
}

/// Disaster recovery error
#[derive(Debug, Error)]
pub enum DisasterRecoveryError {
    /// Backup failed
    #[error("Backup failed: {0}")]
    BackupFailed(String),
    /// Recovery failed
    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),
    /// Replication error
    #[error("Replication error: {0}")]
    ReplicationError(String),
    /// RTO exceeded
    #[error("RTO exceeded: took {actual:?}, expected {expected:?}")]
    RtoExceeded {
        /// Expected recovery time objective
        expected: Duration,
        /// Actual recovery time
        actual: Duration,
    },
    /// RPO exceeded
    #[error("RPO exceeded: data loss {data_loss:?}, expected {expected:?}")]
    RpoExceeded {
        /// Expected recovery point objective
        expected: Duration,
        /// Actual data loss duration
        data_loss: Duration,
    },
    /// Testing failed
    #[error("DR testing failed: {0}")]
    TestingFailed(String),
}

/// Disaster recovery result type
pub type Result<T> = std::result::Result<T, DisasterRecoveryError>;

/// Disaster recovery metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisasterRecoveryMetrics {
    /// Total backups performed
    pub total_backups: u64,
    /// Successful backups
    pub successful_backups: u64,
    /// Failed backups
    pub failed_backups: u64,
    /// Total recovery attempts
    pub total_recoveries: u64,
    /// Successful recoveries
    pub successful_recoveries: u64,
    /// Failed recoveries
    pub failed_recoveries: u64,
    /// Average recovery time
    pub avg_recovery_time: Duration,
    /// Last backup time
    #[serde(skip, default)]
    pub last_backup_time: Option<std::time::Instant>,
    /// Last recovery time
    #[serde(skip, default)]
    pub last_recovery_time: Option<std::time::Instant>,
    /// Current RTO compliance
    pub rto_compliance: f32,
    /// Current RPO compliance
    pub rpo_compliance: f32,
}

impl Default for DisasterRecoveryMetrics {
    fn default() -> Self {
        Self {
            total_backups: 0,
            successful_backups: 0,
            failed_backups: 0,
            total_recoveries: 0,
            successful_recoveries: 0,
            failed_recoveries: 0,
            avg_recovery_time: Duration::from_secs(0),
            last_backup_time: None,
            last_recovery_time: None,
            rto_compliance: 100.0,
            rpo_compliance: 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dr_config_default() {
        let config = DisasterRecoveryConfig::default();
        assert!(config.enabled);
        assert!(config.automated_testing);
        assert_eq!(
            config.replication_strategy,
            ReplicationStrategy::Synchronous
        );
    }

    #[test]
    fn test_backup_config() {
        let config = BackupConfig::default();
        assert!(config.incremental);
        assert!(config.compression);
        assert!(config.encryption);
    }

    #[test]
    fn test_recovery_config() {
        let config = RecoveryConfig::default();
        assert!(config.point_in_time_recovery);
        assert!(config.automated_recovery);
        assert_eq!(config.rto, Duration::from_secs(300));
    }

    #[test]
    fn test_replication_strategies() {
        assert_eq!(
            ReplicationStrategy::Synchronous,
            ReplicationStrategy::Synchronous
        );
        assert_ne!(
            ReplicationStrategy::Synchronous,
            ReplicationStrategy::Asynchronous
        );
    }

    #[test]
    fn test_dr_metrics_default() {
        let metrics = DisasterRecoveryMetrics::default();
        assert_eq!(metrics.total_backups, 0);
        assert_eq!(metrics.total_recoveries, 0);
        assert!((metrics.rto_compliance - 100.0).abs() < f32::EPSILON);
    }
}
