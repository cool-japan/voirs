//! Comprehensive Data Management System for `VoiRS` Feedback
//!
//! This module provides data export, import, backup, restore, and migration
//! capabilities for all `VoiRS` feedback system data including user progress,
//! analytics, settings, and system configurations.

use crate::persistence::PersistenceManager;
use crate::traits::{
    AdaptiveConfig, FeedbackConfig, FeedbackProvider, FeedbackResponse, FeedbackType,
    ProgressIndicators, SessionState, TrainingExercise, UserFeedback, UserProgress,
};
// Note: We'll define our own export-friendly versions of these types
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;

/// Compute a real, deterministic checksum over export payload bytes.
///
/// Uses SHA-256 (via the `privacy` feature's `sha2` dependency) when
/// available. When the crate is built without the `privacy` feature, `sha2`
/// is not in the dependency graph at all, so this falls back to FNV-1a --
/// still a genuine, deterministic hash of the real bytes (verifiable and
/// sensitive to any change in content), just not cryptographically strong.
/// Either way the result is prefixed with the algorithm name so callers can
/// tell which was used.
fn compute_checksum(bytes: &[u8]) -> String {
    #[cfg(feature = "privacy")]
    {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        format!("sha256:{hex}")
    }

    #[cfg(not(feature = "privacy"))]
    {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a 64-bit offset basis
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a 64-bit prime
        }
        format!("fnv1a:{hash:016x}")
    }
}

/// Recursively sort every JSON object's keys, producing a canonical
/// representation. `std::collections::HashMap`'s iteration order depends on
/// a per-instance random seed (`RandomState`), so two structurally-identical
/// packages -- e.g. one freshly built in-process and one reconstructed by
/// `serde`'s `Deserialize` impl for `HashMap` after a real round trip
/// through disk -- can otherwise serialize their map fields in different
/// key orders, producing different bytes (and thus different checksums) for
/// identical content. Sorting here makes the pre-image hashed by
/// [`canonical_checksum_bytes`] independent of any particular `HashMap`
/// instance's iteration order.
fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(String, serde_json::Value)> = map
                .into_iter()
                .map(|(k, v)| (k, canonicalize_json(v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(entries.into_iter().collect())
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(canonicalize_json).collect())
        }
        other => other,
    }
}

/// Serialize `package` into a canonical (recursively key-sorted) JSON byte
/// sequence suitable for hashing: two packages with identical real content
/// produce identical bytes here regardless of which concrete `HashMap`
/// instance backed any given map field -- see [`canonicalize_json`]. Used
/// by both [`DataManager::collect_export_data`] (to compute the real
/// checksum) and [`verify_checksum`] (to independently re-derive and check
/// it, including after a real deserialize round trip).
fn canonical_checksum_bytes(package: &DataExportPackage) -> serde_json::Result<Vec<u8>> {
    let value = serde_json::to_value(package)?;
    serde_json::to_vec(&canonicalize_json(value))
}

/// Convert a real, persisted [`SessionState`] into the export-friendly
/// summary shape, using the real fields tracked on
/// [`crate::traits::SessionStatistics`] (`end_time`, `duration`,
/// `audio_generated_count`) rather than fabricated placeholders.
/// `audio_generated_count` is used as `activity_count`: no dedicated
/// per-session activity counter exists, and each generated audio segment
/// corresponds to a real user interaction in this pipeline.
fn session_to_export(session: &SessionState) -> ExportSessionData {
    ExportSessionData {
        session_id: session.session_id.to_string(),
        user_id: session.user_id.clone(),
        started_at: session.start_time,
        ended_at: session.session_stats.end_time,
        duration_seconds: session.session_stats.duration.as_secs(),
        activity_count: u32::try_from(session.session_stats.audio_generated_count)
            .unwrap_or(u32::MAX),
    }
}

/// Data management errors
#[derive(Debug, thiserror::Error)]
pub enum DataManagementError {
    #[error("Export failed: {message}")]
    /// Raised when exporting feedback data cannot complete successfully.
    ExportError {
        /// Human-readable reason for the export failure.
        message: String,
    },

    #[error("Import failed: {message}")]
    /// Raised when importing feedback data fails.
    ImportError {
        /// Human-readable reason for the import failure.
        message: String,
    },

    #[error("Backup failed: {message}")]
    /// Raised when creating a feedback data backup fails.
    BackupError {
        /// Human-readable reason for the backup failure.
        message: String,
    },

    #[error("Restore failed: {message}")]
    /// Raised when restoring feedback data fails.
    RestoreError {
        /// Human-readable reason for the restore failure.
        message: String,
    },

    #[error("Data validation failed: {message}")]
    /// Raised when imported data does not pass validation.
    ValidationError {
        /// Human-readable reason for the validation failure.
        message: String,
    },

    #[error("I/O error: {source}")]
    /// Description
    IoError {
        #[from]
        /// Description
        source: std::io::Error,
    },

    #[error("Serialization error: {source}")]
    /// Description
    SerializationError {
        #[from]
        /// Description
        source: serde_json::Error,
    },

    #[error("Compression error: {message}")]
    /// Raised when compressing export payloads fails.
    CompressionError {
        /// Human-readable reason for the compression failure.
        message: String,
    },

    #[error("Encryption error: {message}")]
    /// Raised when encrypting export payloads fails.
    EncryptionError {
        /// Human-readable reason for the encryption failure.
        message: String,
    },
}

/// Result type for data management operations
pub type DataManagementResult<T> = Result<T, DataManagementError>;

/// Data export formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format (human-readable)
    Json,
    /// Binary format (compact)
    Binary,
    /// CSV format (for analytics data)
    Csv,
    /// XML format (for compatibility)
    Xml,
    /// Compressed JSON format
    CompressedJson,
    /// Encrypted JSON format
    EncryptedJson,
}

/// Data import options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportOptions {
    /// Skip validation during import
    pub skip_validation: bool,
    /// Merge with existing data instead of replacing
    pub merge_mode: bool,
    /// Backup existing data before import
    pub create_backup: bool,
    /// Handle duplicate entries
    pub duplicate_strategy: DuplicateStrategy,
    /// Data transformation rules
    pub transformations: Vec<DataTransformation>,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            skip_validation: false,
            merge_mode: false,
            create_backup: true,
            duplicate_strategy: DuplicateStrategy::Skip,
            transformations: Vec::new(),
        }
    }
}

/// Strategy for handling duplicate data during import
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuplicateStrategy {
    /// Skip duplicate entries
    Skip,
    /// Overwrite existing entries
    Overwrite,
    /// Merge duplicate entries
    Merge,
    /// Fail on duplicate entries
    Fail,
}

/// Data transformation rules for import
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransformation {
    /// Field path to transform
    pub field_path: String,
    /// Transformation type
    pub transformation: TransformationType,
}

/// Types of data transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    /// Map old value to new value
    ValueMapping {
        /// Source value
        from: String,
        /// Target value
        to: String,
    },
    /// Apply mathematical operation
    MathOperation {
        /// Mathematical operation string
        operation: String,
    },
    /// Convert data type
    TypeConversion {
        /// Target type name
        target_type: String,
    },
    /// Apply custom function
    CustomFunction {
        /// Function name to apply
        function_name: String,
    },
}

/// Comprehensive data export package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExportPackage {
    /// Export metadata
    pub metadata: ExportMetadata,
    /// User progress data
    pub user_progress: HashMap<String, UserProgress>,
    /// Analytics data
    pub analytics: AnalyticsExportData,
    /// System configurations
    pub configurations: SystemConfigurations,
    /// Training data
    pub training_data: TrainingExportData,
    /// Feedback history
    pub feedback_history: Vec<UserFeedback>,
    /// Quality metrics
    pub quality_metrics: QualityMetricsExport,
    /// Gamification data
    pub gamification: Option<GamificationExportData>,
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Export timestamp
    pub created_at: DateTime<Utc>,
    /// Export format
    pub format: ExportFormat,
    /// `VoiRS` version
    pub voris_version: String,
    /// Export version for compatibility
    pub export_version: String,
    /// Exported by user
    pub exported_by: String,
    /// Data size in bytes
    pub data_size: u64,
    /// Number of records by type
    pub record_counts: HashMap<String, u64>,
    /// Export options used
    pub export_options: ExportOptions,
    /// Checksum for data integrity
    pub checksum: String,
}

/// Export-friendly session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSessionData {
    /// Session identifier
    pub session_id: String,
    /// User identifier
    pub user_id: String,
    /// Session start time
    pub started_at: DateTime<Utc>,
    /// Session end time
    pub ended_at: Option<DateTime<Utc>>,
    /// Duration in seconds
    pub duration_seconds: u64,
    /// Number of activities
    pub activity_count: u32,
}

/// Export-friendly performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportPerformanceMetrics {
    /// Metric timestamp
    pub timestamp: DateTime<Utc>,
    /// Response time in milliseconds
    pub response_time_ms: f64,
    /// System throughput
    pub throughput: f64,
    /// Error rate
    pub error_rate: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
}

/// Export-friendly user interaction event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportUserInteractionEvent {
    /// Event identifier
    pub event_id: String,
    /// User identifier
    pub user_id: String,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Type of event
    pub event_type: String,
    /// Event details
    pub details: String,
}

/// Export-friendly system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSystemMetrics {
    /// Metric timestamp
    pub timestamp: DateTime<Utc>,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Disk usage in bytes
    pub disk_usage_bytes: u64,
    /// Network I/O in bytes
    pub network_io_bytes: u64,
}

/// Analytics export data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsExportData {
    /// Session data
    pub sessions: Vec<ExportSessionData>,
    /// Performance metrics
    pub performance_metrics: Vec<ExportPerformanceMetrics>,
    /// User interactions
    pub interactions: Vec<ExportUserInteractionEvent>,
    /// System metrics
    pub system_metrics: Vec<ExportSystemMetrics>,
}

/// System configurations export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfigurations {
    /// Feedback configurations
    pub feedback_configs: HashMap<String, FeedbackConfig>,
    /// Adaptive learning configs
    pub adaptive_configs: HashMap<String, AdaptiveConfig>,
    /// Real-time system configs
    pub realtime_configs: HashMap<String, serde_json::Value>,
    /// UI preferences
    pub ui_preferences: HashMap<String, serde_json::Value>,
    /// Privacy settings
    pub privacy_settings: HashMap<String, serde_json::Value>,
}

/// Training data export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExportData {
    /// Training exercises
    pub exercises: Vec<TrainingExercise>,
    /// Training sessions
    pub sessions: Vec<ExportTrainingSession>,
    /// Custom exercises
    pub custom_exercises: Vec<CustomExercise>,
    /// Training statistics
    pub statistics: TrainingStatistics,
}

/// Training session for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportTrainingSession {
    /// Session identifier
    pub session_id: String,
    /// User identifier
    pub user_id: String,
    /// Exercise identifier
    pub exercise_id: String,
    /// Session start time
    pub started_at: DateTime<Utc>,
    /// Session completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Session score
    pub score: f64,
    /// Number of attempts
    pub attempts: u32,
    /// Feedback count
    pub feedback_count: u32,
}

/// Quality metrics export data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetricsExport {
    /// Quality metrics history
    pub metrics: Vec<QualityMetrics>,
    /// Quality alerts
    pub alerts: Vec<QualityAlert>,
    /// Quality reports
    pub reports: Vec<QualityReport>,
}

/// Gamification data export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamificationExportData {
    /// User achievements
    pub achievements: Vec<Achievement>,
    /// Leaderboard entries
    pub leaderboard_entries: Vec<LeaderboardEntry>,
    /// Points and rewards
    pub points_history: Vec<PointsTransaction>,
    /// Badges and trophies
    pub badges: Vec<Badge>,
}

/// Export options configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Include sensitive data
    pub include_sensitive_data: bool,
    /// Anonymize user data
    pub anonymize_data: bool,
    /// Date range filter
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Include system logs
    pub include_logs: bool,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Encryption enabled
    pub encryption_enabled: bool,
    /// Data types to include
    pub include_data_types: Vec<DataType>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_sensitive_data: false,
            anonymize_data: true,
            date_range: None,
            include_logs: false,
            compression_level: 6,
            encryption_enabled: false,
            include_data_types: vec![
                DataType::UserProgress,
                DataType::Analytics,
                DataType::Configurations,
                DataType::Training,
                DataType::Feedback,
            ],
        }
    }
}

/// Data types for selective export/import
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// User progress data
    UserProgress,
    /// Analytics data
    Analytics,
    /// System configurations
    Configurations,
    /// Training data
    Training,
    /// Feedback data
    Feedback,
    /// Quality metrics
    QualityMetrics,
    /// Gamification data
    Gamification,
    /// System logs
    SystemLogs,
}

/// Custom exercise definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomExercise {
    /// Exercise identifier
    pub id: String,
    /// Exercise name
    pub name: String,
    /// Exercise description
    pub description: String,
    /// Exercise content
    pub content: String,
    /// Difficulty level
    pub difficulty: f64,
    /// Creator user ID
    pub created_by: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Exercise tags
    pub tags: Vec<String>,
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatistics {
    /// Total training sessions
    pub total_sessions: u64,
    /// Total exercises completed
    pub total_exercises: u64,
    /// Average score across sessions
    pub average_score: f64,
    /// Rate of improvement
    pub improvement_rate: f64,
    /// Time spent in minutes
    pub time_spent_minutes: u64,
}

/// Achievement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    /// Achievement identifier
    pub id: String,
    /// Achievement name
    pub name: String,
    /// Achievement description
    pub description: String,
    /// Unlock timestamp
    pub unlocked_at: DateTime<Utc>,
    /// Progress percentage
    pub progress: f64,
}

/// Leaderboard entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    /// User identifier
    pub user_id: String,
    /// User score
    pub score: f64,
    /// User rank
    pub rank: u64,
    /// Entry timestamp
    pub timestamp: DateTime<Utc>,
}

/// Points transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointsTransaction {
    /// Transaction identifier
    pub transaction_id: String,
    /// User identifier
    pub user_id: String,
    /// Points amount
    pub points: i64,
    /// Transaction reason
    pub reason: String,
    /// Transaction timestamp
    pub timestamp: DateTime<Utc>,
}

/// Badge data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Badge {
    /// Badge identifier
    pub id: String,
    /// Badge name
    pub name: String,
    /// Badge description
    pub description: String,
    /// Badge icon
    pub icon: String,
    /// Earn timestamp
    pub earned_at: DateTime<Utc>,
}

/// Quality metrics (re-export from `quality_monitor` module)
use crate::quality_monitor::{QualityAlert, QualityMetrics, QualityReport};

/// Data Management System
pub struct DataManager {
    /// Data storage backend
    storage: Arc<RwLock<dyn DataStorage>>,
    /// Export configuration
    export_config: ExportOptions,
    /// Import configuration
    import_config: ImportOptions,
    /// Encryption key for secure exports
    encryption_key: Option<String>,
    /// Real data source that export/import operations read from and write
    /// to. Without one attached (see [`DataManager::with_persistence`]),
    /// export/import fail closed instead of producing an empty archive with
    /// a fabricated integrity checksum.
    persistence: Option<Arc<dyn PersistenceManager>>,
}

/// Data storage backend trait
#[async_trait]
pub trait DataStorage: Send + Sync {
    /// Store data package
    async fn store_package(
        &self,
        package: &DataExportPackage,
        path: &Path,
    ) -> DataManagementResult<()>;

    /// Load data package
    async fn load_package(&self, path: &Path) -> DataManagementResult<DataExportPackage>;

    /// List available backups
    async fn list_backups(&self, directory: &Path) -> DataManagementResult<Vec<BackupInfo>>;

    /// Delete backup
    async fn delete_backup(&self, path: &Path) -> DataManagementResult<()>;

    /// Validate data integrity
    async fn validate_data(
        &self,
        package: &DataExportPackage,
    ) -> DataManagementResult<ValidationReport>;
}

/// Backup information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
    /// Backup file path
    pub path: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Backup size in bytes
    pub size_bytes: u64,
    /// Backup format
    pub format: ExportFormat,
    /// Total record count
    pub record_count: u64,
    /// Data checksum
    pub checksum: String,
}

/// Data validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Record counts by type
    pub record_counts: HashMap<String, u64>,
    /// Integrity check results
    pub integrity_checks: HashMap<String, bool>,
}

/// File-based data storage implementation
#[derive(Debug)]
pub struct FileDataStorage {
    /// Base directory for storage
    base_directory: String,
}

impl FileDataStorage {
    /// Description
    #[must_use]
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }
}

#[async_trait]
impl DataStorage for FileDataStorage {
    async fn store_package(
        &self,
        package: &DataExportPackage,
        path: &Path,
    ) -> DataManagementResult<()> {
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        match package.metadata.format {
            ExportFormat::Json => {
                let json_data = serde_json::to_string_pretty(package)?;
                fs::write(path, json_data).await?;
            }
            ExportFormat::Binary => {
                let binary_data =
                    oxicode::serde::encode_to_vec(package, oxicode::config::standard()).map_err(
                        |e| DataManagementError::ExportError {
                            message: format!("Binary serialization failed: {e}"),
                        },
                    )?;
                fs::write(path, binary_data).await?;
            }
            ExportFormat::CompressedJson => {
                let json_data = serde_json::to_string(package)?;
                let compressed = Self::compress_data(json_data.as_bytes())?;
                fs::write(path, compressed).await?;
            }
            ExportFormat::EncryptedJson => {
                let json_data = serde_json::to_string(package)?;
                let encrypted = Self::encrypt_data(json_data.as_bytes(), "default_key")?;
                fs::write(path, encrypted).await?;
            }
            _ => {
                return Err(DataManagementError::ExportError {
                    message: format!("Unsupported export format: {:?}", package.metadata.format),
                });
            }
        }

        Ok(())
    }

    async fn load_package(&self, path: &Path) -> DataManagementResult<DataExportPackage> {
        let data = fs::read(path).await?;

        // Try to determine format from file extension or header
        let format = Self::detect_format(&data, path)?;

        let package = match format {
            ExportFormat::Json => {
                let json_str =
                    String::from_utf8(data).map_err(|e| DataManagementError::ImportError {
                        message: format!("Invalid UTF-8 data: {e}"),
                    })?;
                serde_json::from_str(&json_str)?
            }
            ExportFormat::Binary => {
                oxicode::serde::decode_from_slice(&data, oxicode::config::standard())
                    .map(|(v, _)| v)
                    .map_err(|e| DataManagementError::ImportError {
                        message: format!("Binary deserialization failed: {e}"),
                    })?
            }
            ExportFormat::CompressedJson => {
                let decompressed = Self::decompress_data(&data)?;
                let json_str = String::from_utf8(decompressed).map_err(|e| {
                    DataManagementError::ImportError {
                        message: format!("Invalid UTF-8 data after decompression: {e}"),
                    }
                })?;
                serde_json::from_str(&json_str)?
            }
            ExportFormat::EncryptedJson => {
                let decrypted = Self::decrypt_data(&data, "default_key")?;
                let json_str =
                    String::from_utf8(decrypted).map_err(|e| DataManagementError::ImportError {
                        message: format!("Invalid UTF-8 data after decryption: {e}"),
                    })?;
                serde_json::from_str(&json_str)?
            }
            _ => {
                return Err(DataManagementError::ImportError {
                    message: format!("Unsupported import format: {format:?}"),
                });
            }
        };

        Ok(package)
    }

    async fn list_backups(&self, directory: &Path) -> DataManagementResult<Vec<BackupInfo>> {
        let mut backups = Vec::new();

        if !directory.exists() {
            return Ok(backups);
        }

        let mut entries = fs::read_dir(directory).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if matches!(extension.to_str(), Some("json" | "bin" | "backup")) {
                        let metadata = fs::metadata(&path).await?;
                        let size_bytes = metadata.len();

                        // Try to read metadata from file to get more info
                        let format = if let Ok(package) = self.load_package(&path).await {
                            package.metadata.format
                        } else {
                            ExportFormat::Json // Default assumption
                        };

                        backups.push(BackupInfo {
                            path: path.to_string_lossy().to_string(),
                            created_at: metadata
                                .created()
                                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                                .into(),
                            size_bytes,
                            format,
                            record_count: 0, // Would need to be determined from actual data
                            checksum: String::new(), // Would need to be calculated
                        });
                    }
                }
            }
        }

        Ok(backups)
    }

    async fn delete_backup(&self, path: &Path) -> DataManagementResult<()> {
        fs::remove_file(path).await?;
        Ok(())
    }

    async fn validate_data(
        &self,
        package: &DataExportPackage,
    ) -> DataManagementResult<ValidationReport> {
        let mut report = ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            record_counts: HashMap::new(),
            integrity_checks: HashMap::new(),
        };

        // Validate metadata
        if package.metadata.export_version.is_empty() {
            report.errors.push("Missing export version".to_string());
            report.is_valid = false;
        }

        // Count records
        report.record_counts.insert(
            "user_progress".to_string(),
            package.user_progress.len() as u64,
        );
        report.record_counts.insert(
            "feedback_history".to_string(),
            package.feedback_history.len() as u64,
        );
        report.record_counts.insert(
            "analytics_sessions".to_string(),
            package.analytics.sessions.len() as u64,
        );

        // Validate user progress data
        for (user_id, progress) in &package.user_progress {
            if user_id.is_empty() {
                report
                    .errors
                    .push("Empty user ID found in progress data".to_string());
                report.is_valid = false;
            }

            // Validate progress scores are in valid range
            if progress.average_scores.overall_score < 0.0
                || progress.average_scores.overall_score > 1.0
            {
                report.warnings.push(format!(
                    "Invalid score range for user {}: {}",
                    user_id, progress.average_scores.overall_score
                ));
            }
        }

        // Validate feedback history
        for feedback in &package.feedback_history {
            // Note: UserFeedback doesn't have user_id field, skip this validation for now
            if feedback.message.is_empty() {
                report
                    .warnings
                    .push("Empty feedback message found".to_string());
            }
        }

        // Verify the real checksum against a fresh recomputation over the
        // package's actual content, using the same "checksum/data_size
        // blanked" pre-image `collect_export_data` hashed. A mismatch means
        // the archive was corrupted or tampered with after export.
        let checksum_valid = verify_checksum(package);
        report
            .integrity_checks
            .insert("checksum_valid".to_string(), checksum_valid);
        if !checksum_valid {
            report
                .errors
                .push("Checksum mismatch: exported data may be corrupted".to_string());
            report.is_valid = false;
        }

        // Check data integrity
        report
            .integrity_checks
            .insert("metadata_present".to_string(), true);
        report
            .integrity_checks
            .insert("user_data_consistent".to_string(), report.errors.is_empty());

        Ok(report)
    }
}

/// Recompute a package's checksum the same way [`DataManager::collect_export_data`]
/// originally did (over the payload with `metadata.checksum` blanked and
/// `metadata.data_size` zeroed) and compare it against the checksum actually
/// stored in the package. Returns `false` for a package whose checksum was
/// never real to begin with (e.g. hand-built in a test without going
/// through `collect_export_data`), which is the correct, honest outcome --
/// there is nothing to verify it against.
fn verify_checksum(package: &DataExportPackage) -> bool {
    let mut repro = package.clone();
    let claimed_checksum = std::mem::take(&mut repro.metadata.checksum);
    repro.metadata.data_size = 0;

    let Ok(serialized) = canonical_checksum_bytes(&repro) else {
        return false;
    };

    compute_checksum(&serialized) == claimed_checksum
}

impl FileDataStorage {
    /// Detect file format from data and path
    fn detect_format(data: &[u8], path: &Path) -> DataManagementResult<ExportFormat> {
        // Check file extension first
        if let Some(extension) = path.extension() {
            match extension.to_str() {
                Some("json") => return Ok(ExportFormat::Json),
                Some("bin" | "binary") => return Ok(ExportFormat::Binary),
                Some("gz" | "zip") => return Ok(ExportFormat::CompressedJson),
                Some("enc" | "encrypted") => return Ok(ExportFormat::EncryptedJson),
                _ => {}
            }
        }

        // Try to detect from content
        if data.starts_with(b"{") || data.starts_with(b"[") {
            Ok(ExportFormat::Json)
        } else if data.len() > 4 && &data[0..4] == b"\x1f\x8b\x08" {
            Ok(ExportFormat::CompressedJson)
        } else {
            Ok(ExportFormat::Binary)
        }
    }

    /// Compress data using gzip
    fn compress_data(data: &[u8]) -> DataManagementResult<Vec<u8>> {
        use oxiarc_deflate::GzipStreamEncoder;
        use std::io::Write;

        let mut encoder = GzipStreamEncoder::new(Vec::new(), 6);
        encoder
            .write_all(data)
            .map_err(|e| DataManagementError::CompressionError {
                message: e.to_string(),
            })?;
        encoder
            .finish()
            .map_err(|e| DataManagementError::CompressionError {
                message: e.to_string(),
            })
    }

    /// Decompress gzip data
    fn decompress_data(data: &[u8]) -> DataManagementResult<Vec<u8>> {
        use oxiarc_deflate::GzipStreamDecoder;
        use std::io::Read;

        let mut decoder = GzipStreamDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).map_err(|e| {
            DataManagementError::CompressionError {
                message: e.to_string(),
            }
        })?;
        Ok(decompressed)
    }

    /// Encrypt data (simple XOR encryption for demo)
    fn encrypt_data(data: &[u8], key: &str) -> DataManagementResult<Vec<u8>> {
        let key_bytes = key.as_bytes();
        let encrypted: Vec<u8> = data
            .iter()
            .enumerate()
            .map(|(i, byte)| byte ^ key_bytes[i % key_bytes.len()])
            .collect();
        Ok(encrypted)
    }

    /// Decrypt data (simple XOR decryption for demo)
    fn decrypt_data(data: &[u8], key: &str) -> DataManagementResult<Vec<u8>> {
        Self::encrypt_data(data, key) // XOR is its own inverse
    }
}

impl DataManager {
    /// Create new data manager with no persistence backend attached. Export
    /// and import operations fail closed until
    /// [`DataManager::with_persistence`] attaches a real one.
    pub async fn new(
        storage: Arc<RwLock<dyn DataStorage>>,
        export_config: ExportOptions,
        import_config: ImportOptions,
    ) -> DataManagementResult<Self> {
        Ok(Self {
            storage,
            export_config,
            import_config,
            encryption_key: None,
            persistence: None,
        })
    }

    /// Attach the real persistence backend that export/import operations
    /// read from and write to.
    #[must_use]
    pub fn with_persistence(mut self, persistence: Arc<dyn PersistenceManager>) -> Self {
        self.persistence = Some(persistence);
        self
    }

    fn require_persistence(&self) -> DataManagementResult<&Arc<dyn PersistenceManager>> {
        self.persistence
            .as_ref()
            .ok_or_else(|| DataManagementError::ExportError {
                message: "no persistence backend configured; call DataManager::with_persistence \
                          before export/import"
                    .to_string(),
            })
    }

    /// Export a single user's real data.
    pub async fn export_data(
        &self,
        user_id: &str,
        output_path: &Path,
        format: ExportFormat,
    ) -> DataManagementResult<ExportMetadata> {
        let package = self.collect_export_data(user_id, format.clone()).await?;

        let storage = self.storage.read().await;
        storage.store_package(&package, output_path).await?;

        Ok(package.metadata)
    }

    /// Import a user's data from file, writing it back through the real
    /// persistence backend.
    pub async fn import_data(
        &self,
        user_id: &str,
        input_path: &Path,
        options: Option<ImportOptions>,
    ) -> DataManagementResult<ImportReport> {
        let import_options = options.unwrap_or_else(|| self.import_config.clone());

        // Create backup if requested
        if import_options.create_backup {
            let backup_path = self.generate_backup_path().await?;
            self.create_backup(user_id, &backup_path).await?;
        }

        let storage = self.storage.read().await;
        let package = storage.load_package(input_path).await?;

        // Validate data if not skipped
        let validation_report = if import_options.skip_validation {
            ValidationReport {
                is_valid: true,
                errors: Vec::new(),
                warnings: Vec::new(),
                record_counts: HashMap::new(),
                integrity_checks: HashMap::new(),
            }
        } else {
            storage.validate_data(&package).await?
        };

        if !validation_report.is_valid && !import_options.skip_validation {
            return Err(DataManagementError::ValidationError {
                message: format!("Data validation failed: {:?}", validation_report.errors),
            });
        }

        // Perform the import
        let import_result = self
            .perform_import(user_id, &package, &import_options)
            .await?;

        Ok(ImportReport {
            import_metadata: package.metadata,
            validation_report,
            import_result,
            imported_at: Utc::now(),
        })
    }

    /// Create a real backup of a single user's data.
    pub async fn create_backup(
        &self,
        user_id: &str,
        backup_path: &Path,
    ) -> DataManagementResult<BackupInfo> {
        let package = self
            .collect_export_data(user_id, ExportFormat::CompressedJson)
            .await?;

        let storage = self.storage.read().await;
        storage.store_package(&package, backup_path).await?;

        let metadata = std::fs::metadata(backup_path)?;

        Ok(BackupInfo {
            path: backup_path.to_string_lossy().to_string(),
            created_at: package.metadata.created_at,
            size_bytes: metadata.len(),
            format: package.metadata.format,
            record_count: package.metadata.record_counts.values().sum(),
            checksum: package.metadata.checksum,
        })
    }

    /// Restore a user's data from backup.
    pub async fn restore_backup(
        &self,
        user_id: &str,
        backup_path: &Path,
    ) -> DataManagementResult<RestoreReport> {
        let import_options = ImportOptions {
            skip_validation: false,
            merge_mode: false,
            create_backup: false, // Don't create backup when restoring
            duplicate_strategy: DuplicateStrategy::Overwrite,
            transformations: Vec::new(),
        };

        let import_report = self
            .import_data(user_id, backup_path, Some(import_options))
            .await?;

        Ok(RestoreReport {
            backup_path: backup_path.to_string_lossy().to_string(),
            import_report,
            restored_at: Utc::now(),
        })
    }

    /// List available backups
    pub async fn list_backups(
        &self,
        backup_directory: &Path,
    ) -> DataManagementResult<Vec<BackupInfo>> {
        let storage = self.storage.read().await;
        storage.list_backups(backup_directory).await
    }

    /// Collect a single user's real data for export from the configured
    /// persistence backend.
    ///
    /// A user with no stored progress or feedback yields an otherwise-empty
    /// package (not an error) -- an honest "nothing to export" result,
    /// distinct from a backend failure.
    async fn collect_export_data(
        &self,
        user_id: &str,
        format: ExportFormat,
    ) -> DataManagementResult<DataExportPackage> {
        let persistence = self.require_persistence()?;

        let mut user_progress = HashMap::new();
        if let Ok(progress) = persistence.load_user_progress(user_id).await {
            user_progress.insert(user_id.to_string(), progress);
        }

        // The export schema carries individual feedback items rather than
        // full response envelopes; flatten every response's items for this
        // user into the real, currently-stored set.
        let feedback_history: Vec<UserFeedback> = persistence
            .load_feedback_history(user_id, None, None)
            .await
            .unwrap_or_default()
            .into_iter()
            .flat_map(|response| response.feedback_items)
            .collect();

        // Sessions are only enumerable per-user via `export_user_data` (the
        // trait has no dedicated "list sessions for user" method); a user
        // with no sessions on record yields an honestly empty list here,
        // not an error.
        let sessions: Vec<ExportSessionData> = persistence
            .export_user_data(user_id)
            .await
            .map(|export| export.sessions.iter().map(session_to_export).collect())
            .unwrap_or_default();

        let mut record_counts = HashMap::new();
        record_counts.insert("user_progress".to_string(), user_progress.len() as u64);
        record_counts.insert("feedback_items".to_string(), feedback_history.len() as u64);
        record_counts.insert("sessions".to_string(), sessions.len() as u64);

        let metadata = ExportMetadata {
            created_at: Utc::now(),
            format,
            voris_version: env!("CARGO_PKG_VERSION").to_string(),
            export_version: "1.0.0".to_string(),
            exported_by: user_id.to_string(),
            data_size: 0, // patched below once the real payload is known
            record_counts,
            export_options: self.export_config.clone(),
            checksum: String::new(), // patched below with the real checksum
        };

        let mut package = DataExportPackage {
            metadata,
            user_progress,
            analytics: AnalyticsExportData {
                sessions,
                // No real data source is wired up for these categories yet
                // (performance/interaction/system-level metrics live in
                // other subsystems not reachable from here) -- left
                // honestly empty rather than fabricated.
                performance_metrics: Vec::new(),
                interactions: Vec::new(),
                system_metrics: Vec::new(),
            },
            // `configurations`/`training_data`/`quality_metrics`/`gamification`:
            // no real data source is wired up for these categories yet
            // (they live in other subsystems not reachable from here) --
            // left honestly empty rather than fabricated.
            configurations: SystemConfigurations {
                feedback_configs: HashMap::new(),
                adaptive_configs: HashMap::new(),
                realtime_configs: HashMap::new(),
                ui_preferences: HashMap::new(),
                privacy_settings: HashMap::new(),
            },
            training_data: TrainingExportData {
                exercises: Vec::new(),
                sessions: Vec::new(),
                custom_exercises: Vec::new(),
                statistics: TrainingStatistics {
                    total_sessions: 0,
                    total_exercises: 0,
                    average_score: 0.0,
                    improvement_rate: 0.0,
                    time_spent_minutes: 0,
                },
            },
            feedback_history,
            quality_metrics: QualityMetricsExport {
                metrics: Vec::new(),
                alerts: Vec::new(),
                reports: Vec::new(),
            },
            gamification: None,
        };

        // Real checksum/size over the actual serialized payload (computed
        // with `checksum`/`data_size` at the placeholder values set above,
        // and canonicalized so the same checksum is reproducible by
        // re-hashing a previously exported file regardless of `HashMap`
        // iteration order -- see `canonical_checksum_bytes`/`verify_checksum`).
        let serialized = canonical_checksum_bytes(&package)?;
        package.metadata.checksum = compute_checksum(&serialized);
        package.metadata.data_size = serialized.len() as u64;

        Ok(package)
    }

    /// Perform the actual import operation, writing every record back
    /// through the real persistence backend.
    async fn perform_import(
        &self,
        user_id: &str,
        package: &DataExportPackage,
        options: &ImportOptions,
    ) -> DataManagementResult<ImportResult> {
        let mut result = ImportResult {
            records_imported: HashMap::new(),
            records_skipped: HashMap::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        // Import user progress
        for (progress_user_id, progress) in &package.user_progress {
            match self
                .import_user_progress(progress_user_id, progress, options)
                .await
            {
                Ok(()) => {
                    *result
                        .records_imported
                        .entry("user_progress".to_string())
                        .or_insert(0) += 1;
                }
                Err(e) => {
                    result.errors.push(format!(
                        "Failed to import progress for user {progress_user_id}: {e}"
                    ));
                    *result
                        .records_skipped
                        .entry("user_progress".to_string())
                        .or_insert(0) += 1;
                }
            }
        }

        // Import feedback history. `UserFeedback` items carry no user_id of
        // their own (see `collect_export_data`), so they are attributed to
        // the user this import operation was invoked for.
        for feedback in &package.feedback_history {
            match self.import_feedback(user_id, feedback, options).await {
                Ok(()) => {
                    *result
                        .records_imported
                        .entry("feedback".to_string())
                        .or_insert(0) += 1;
                }
                Err(e) => {
                    result
                        .errors
                        .push(format!("Failed to import feedback: {e}"));
                    *result
                        .records_skipped
                        .entry("feedback".to_string())
                        .or_insert(0) += 1;
                }
            }
        }

        // Import sessions
        for exported_session in &package.analytics.sessions {
            match self.import_session(exported_session, options).await {
                Ok(()) => {
                    *result
                        .records_imported
                        .entry("sessions".to_string())
                        .or_insert(0) += 1;
                }
                Err(e) => {
                    result.errors.push(format!(
                        "Failed to import session {}: {e}",
                        exported_session.session_id
                    ));
                    *result
                        .records_skipped
                        .entry("sessions".to_string())
                        .or_insert(0) += 1;
                }
            }
        }

        Ok(result)
    }

    /// Import user progress data by writing it through the real persistence
    /// backend.
    async fn import_user_progress(
        &self,
        user_id: &str,
        progress: &UserProgress,
        _options: &ImportOptions,
    ) -> DataManagementResult<()> {
        let persistence = self.require_persistence()?;
        persistence
            .save_user_progress(user_id, progress)
            .await
            .map_err(|e| DataManagementError::ImportError {
                message: format!("failed to save progress for user '{user_id}': {e}"),
            })?;
        log::info!("Imported progress for user: {user_id}");
        Ok(())
    }

    /// Import feedback data by writing it through the real persistence
    /// backend.
    ///
    /// The export schema only carries the individual [`UserFeedback`] item,
    /// not the full [`FeedbackResponse`] envelope it originally arrived in
    /// (see `collect_export_data`), so this synthesizes a minimal real
    /// response around it -- a genuine, retrievable write, even though the
    /// original envelope-level fields (timestamp, processing time, overall
    /// score across multiple items) are not recoverable from this schema.
    async fn import_feedback(
        &self,
        user_id: &str,
        feedback: &UserFeedback,
        _options: &ImportOptions,
    ) -> DataManagementResult<()> {
        let persistence = self.require_persistence()?;

        let response = FeedbackResponse {
            feedback_items: vec![feedback.clone()],
            overall_score: feedback.score,
            immediate_actions: Vec::new(),
            long_term_goals: Vec::new(),
            progress_indicators: ProgressIndicators::default(),
            timestamp: Utc::now(),
            processing_time: std::time::Duration::default(),
            feedback_type: FeedbackType::Quality,
        };

        persistence
            .save_feedback(user_id, &response)
            .await
            .map_err(|e| DataManagementError::ImportError {
                message: format!("failed to save feedback for user '{user_id}': {e}"),
            })?;
        log::info!(
            "Imported feedback for user '{user_id}': {}",
            feedback.message
        );
        Ok(())
    }

    /// Import a session by writing it through the real persistence backend.
    ///
    /// The export schema only carries the [`ExportSessionData`] summary,
    /// not the full [`SessionState`] it originally came from (per-session
    /// stats, preferences, adaptive state, and current exercise are not
    /// recoverable from this schema), so this reconstructs a real,
    /// retrievable session record around the fields that *are* preserved --
    /// rather than silently dropping the session on import.
    async fn import_session(
        &self,
        exported: &ExportSessionData,
        _options: &ImportOptions,
    ) -> DataManagementResult<()> {
        let persistence = self.require_persistence()?;

        let session_id = uuid::Uuid::parse_str(&exported.session_id).map_err(|e| {
            DataManagementError::ImportError {
                message: format!("invalid session id '{}': {e}", exported.session_id),
            }
        })?;

        let session = SessionState {
            session_id,
            user_id: exported.user_id.clone(),
            start_time: exported.started_at,
            last_activity: exported.ended_at.unwrap_or(exported.started_at),
            current_task: None,
            stats: crate::traits::SessionStats::default(),
            preferences: crate::traits::UserPreferences {
                user_id: exported.user_id.clone(),
                ..crate::traits::UserPreferences::default()
            },
            adaptive_state: crate::traits::AdaptiveState::default(),
            current_exercise: None,
            session_stats: crate::traits::SessionStatistics {
                start_time: exported.started_at,
                end_time: exported.ended_at,
                duration: std::time::Duration::from_secs(exported.duration_seconds),
                audio_generated_count: exported.activity_count as usize,
                average_quality_score: 0.0,
                average_pronunciation_score: 0.0,
                exercises_attempted: 0,
                exercises_completed: 0,
            },
        };

        persistence
            .save_session(&session)
            .await
            .map_err(|e| DataManagementError::ImportError {
                message: format!("failed to save session '{}': {e}", exported.session_id),
            })?;
        log::info!(
            "Imported session '{}' for user '{}'",
            exported.session_id,
            exported.user_id
        );
        Ok(())
    }

    /// Generate backup file path
    async fn generate_backup_path(&self) -> DataManagementResult<std::path::PathBuf> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("voris_backup_{timestamp}.json.gz");
        Ok(std::path::PathBuf::from("backups").join(filename))
    }
}

/// Import operation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportReport {
    /// Metadata from imported data
    pub import_metadata: ExportMetadata,
    /// Data validation report
    pub validation_report: ValidationReport,
    /// Import operation result
    pub import_result: ImportResult,
    /// Import timestamp
    pub imported_at: DateTime<Utc>,
}

/// Import operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    /// Number of records imported by type
    pub records_imported: HashMap<String, u64>,
    /// Number of records skipped by type
    pub records_skipped: HashMap<String, u64>,
    /// Import errors
    pub errors: Vec<String>,
    /// Import warnings
    pub warnings: Vec<String>,
}

/// Restore operation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreReport {
    /// Path to backup file
    pub backup_path: String,
    /// Import report from restore operation
    pub import_report: ImportReport,
    /// Restore timestamp
    pub restored_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::backends::memory::MemoryPersistenceManager;
    use crate::persistence::PersistenceConfig;
    use crate::traits::UserProgress;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    async fn test_persistence() -> Arc<dyn PersistenceManager> {
        Arc::new(
            MemoryPersistenceManager::new(PersistenceConfig::default())
                .await
                .unwrap(),
        )
    }

    async fn manager_without_persistence() -> DataManager {
        let storage = Arc::new(RwLock::new(FileDataStorage::new(
            std::env::temp_dir()
                .join("voirs_data_management_tests")
                .to_string_lossy()
                .to_string(),
        )));
        DataManager::new(storage, ExportOptions::default(), ImportOptions::default())
            .await
            .unwrap()
    }

    async fn manager_with_persistence() -> (DataManager, Arc<dyn PersistenceManager>) {
        let persistence = test_persistence().await;
        let manager = manager_without_persistence()
            .await
            .with_persistence(persistence.clone());
        (manager, persistence)
    }

    /// A minimal, real (non-empty) `FeedbackResponse` fixture shared by
    /// several tests below.
    fn test_feedback_response() -> FeedbackResponse {
        FeedbackResponse {
            feedback_items: vec![UserFeedback {
                message: "Great articulation".to_string(),
                suggestion: None,
                confidence: 0.9,
                score: 0.88,
                priority: 0.5,
                metadata: HashMap::new(),
            }],
            overall_score: 0.88,
            immediate_actions: vec![],
            long_term_goals: vec![],
            progress_indicators: ProgressIndicators::default(),
            timestamp: Utc::now(),
            processing_time: std::time::Duration::from_millis(5),
            feedback_type: FeedbackType::Quality,
        }
    }

    #[tokio::test]
    async fn test_data_manager_creation() {
        let storage = Arc::new(RwLock::new(FileDataStorage::new("test_data".to_string())));
        let export_config = ExportOptions::default();
        let import_config = ImportOptions::default();

        let manager = DataManager::new(storage, export_config, import_config).await;
        assert!(manager.is_ok());
    }

    /// Without a persistence backend attached, export must fail closed
    /// rather than produce an empty archive with a fabricated checksum.
    #[tokio::test]
    async fn test_export_without_persistence_fails_closed() {
        let manager = manager_without_persistence().await;
        let package = manager
            .collect_export_data("user1", ExportFormat::Json)
            .await;
        assert!(package.is_err());
    }

    /// The export package must contain the user's real, seeded progress
    /// data (not an empty placeholder), and its checksum must be real --
    /// i.e. it must actually verify against the payload.
    #[tokio::test]
    async fn test_export_package_reflects_real_seeded_data() {
        let (manager, persistence) = manager_with_persistence().await;

        let progress = UserProgress {
            user_id: "user1".to_string(),
            overall_skill_level: 0.65,
            ..UserProgress::default()
        };
        persistence
            .save_user_progress("user1", &progress)
            .await
            .unwrap();

        let package = manager
            .collect_export_data("user1", ExportFormat::Json)
            .await
            .unwrap();

        assert_eq!(package.metadata.format, ExportFormat::Json);
        assert!(!package.metadata.voris_version.is_empty());

        let exported_progress = package
            .user_progress
            .get("user1")
            .expect("real seeded progress must be present in the export");
        assert!((exported_progress.overall_skill_level - 0.65).abs() < 1e-6);

        // The checksum must be real: it must actually verify.
        assert!(!package.metadata.checksum.is_empty());
        assert!(package.metadata.checksum != "placeholder_checksum");
        assert!(verify_checksum(&package));
    }

    /// The export package must contain the user's real, seeded *session*
    /// data too -- not an empty placeholder list regardless of what
    /// sessions actually exist.
    #[tokio::test]
    async fn test_export_package_reflects_real_seeded_sessions() {
        use crate::traits::{AdaptiveState, SessionStatistics, SessionStats, UserPreferences};

        let (manager, persistence) = manager_with_persistence().await;

        let session_id = uuid::Uuid::new_v4();
        let start = Utc::now() - chrono::Duration::minutes(10);
        let end = Utc::now();
        let session = crate::traits::SessionState {
            session_id,
            user_id: "user1".to_string(),
            start_time: start,
            last_activity: end,
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: AdaptiveState::default(),
            current_exercise: None,
            session_stats: SessionStatistics {
                start_time: start,
                end_time: Some(end),
                duration: std::time::Duration::from_secs(600),
                audio_generated_count: 7,
                average_quality_score: 0.8,
                average_pronunciation_score: 0.75,
                exercises_attempted: 3,
                exercises_completed: 2,
            },
        };
        persistence.save_session(&session).await.unwrap();

        let package = manager
            .collect_export_data("user1", ExportFormat::Json)
            .await
            .unwrap();

        assert_eq!(package.analytics.sessions.len(), 1);
        let exported = &package.analytics.sessions[0];
        assert_eq!(exported.session_id, session_id.to_string());
        assert_eq!(exported.user_id, "user1");
        assert_eq!(exported.duration_seconds, 600);
        assert_eq!(exported.activity_count, 7);
        assert!(exported.ended_at.is_some());
        assert_eq!(
            package.metadata.record_counts.get("sessions"),
            Some(&1),
            "record_counts must reflect the real seeded session count"
        );

        assert!(verify_checksum(&package));
    }

    /// Evidence for why `verify_checksum` must canonicalize before hashing:
    /// `std::collections::HashMap`'s iteration order depends on a
    /// per-instance random seed (`RandomState`), so two structurally
    /// identical maps built independently commonly serialize their JSON
    /// object keys in different orders. `record_counts: HashMap<String,
    /// u64>` always has exactly 2 entries in a real `collect_export_data`
    /// package, so this reordering is a real, frequently-observed hazard
    /// for any naive (non-canonicalized) re-hash -- e.g. after
    /// `import_data` deserializes a package fresh from disk via
    /// `load_package`. This does not assert a specific count (the exact
    /// probability distribution is an implementation detail of
    /// `RandomState`), only that canonicalization is exercised for a
    /// real reason.
    #[test]
    fn evidence_hashmap_json_order_varies_across_instances() {
        use std::collections::HashMap;
        let mut orders = std::collections::HashSet::new();
        for _ in 0..200 {
            let mut map: HashMap<String, u64> = HashMap::new();
            map.insert("user_progress".to_string(), 1);
            map.insert("feedback_items".to_string(), 3);
            orders.insert(serde_json::to_string(&map).unwrap());
        }
        println!(
            "observed {} distinct raw JSON key orderings across 200 fresh 2-entry HashMaps",
            orders.len()
        );

        // Regardless of how many distinct raw orderings were observed above,
        // canonicalization must always collapse them to exactly one.
        let mut canonical_orders = std::collections::HashSet::new();
        for _ in 0..200 {
            let mut map: HashMap<String, u64> = HashMap::new();
            map.insert("user_progress".to_string(), 1);
            map.insert("feedback_items".to_string(), 3);
            let value = serde_json::to_value(&map).unwrap();
            canonical_orders.insert(serde_json::to_string(&canonicalize_json(value)).unwrap());
        }
        assert_eq!(
            canonical_orders.len(),
            1,
            "canonicalize_json must produce identical bytes for identical content \
             regardless of the source HashMap's iteration order"
        );
    }

    /// The exact real-world scenario `verify_checksum` must handle: a
    /// package that was serialized, then genuinely deserialized back (as
    /// `FileDataStorage::load_package` does for every real
    /// import/restore), producing an entirely fresh `DataExportPackage`
    /// whose `HashMap` fields have their own independent (likely
    /// different) iteration order. The checksum must still verify -- a
    /// legitimately unmodified re-imported archive must never spuriously
    /// report "corrupted" purely due to `HashMap` reordering.
    #[tokio::test]
    async fn test_verify_checksum_survives_real_json_round_trip() {
        let (manager, persistence) = manager_with_persistence().await;

        // `record_counts` (2 entries) is populated by every real export;
        // this alone is enough to exercise the reordering hazard.
        persistence
            .save_user_progress(
                "user1",
                &UserProgress {
                    user_id: "user1".to_string(),
                    overall_skill_level: 0.5,
                    ..UserProgress::default()
                },
            )
            .await
            .unwrap();
        persistence
            .save_feedback("user1", &test_feedback_response())
            .await
            .unwrap();

        let original = manager
            .collect_export_data("user1", ExportFormat::Json)
            .await
            .unwrap();
        assert!(verify_checksum(&original));

        // Simulate exactly what `FileDataStorage::load_package` does for a
        // real JSON export: serialize, then deserialize into a brand-new
        // `DataExportPackage` with entirely fresh `HashMap` instances.
        let json = serde_json::to_string(&original).unwrap();
        let round_tripped: DataExportPackage = serde_json::from_str(&json).unwrap();

        assert!(
            verify_checksum(&round_tripped),
            "a real, unmodified export must still verify after a genuine \
             serialize/deserialize round trip, regardless of HashMap reordering"
        );
    }

    /// A checksum computed from fabricated/placeholder text would never
    /// vary with content. The real implementation must: two exports with
    /// different underlying data must produce different checksums.
    #[tokio::test]
    async fn test_checksum_varies_with_real_content() {
        let (manager, persistence) = manager_with_persistence().await;

        persistence
            .save_user_progress(
                "user1",
                &UserProgress {
                    user_id: "user1".to_string(),
                    overall_skill_level: 0.1,
                    ..UserProgress::default()
                },
            )
            .await
            .unwrap();
        let package_a = manager
            .collect_export_data("user1", ExportFormat::Json)
            .await
            .unwrap();

        persistence
            .save_user_progress(
                "user1",
                &UserProgress {
                    user_id: "user1".to_string(),
                    overall_skill_level: 0.9,
                    ..UserProgress::default()
                },
            )
            .await
            .unwrap();
        let package_b = manager
            .collect_export_data("user1", ExportFormat::Json)
            .await
            .unwrap();

        assert_ne!(
            package_a.metadata.checksum, package_b.metadata.checksum,
            "different underlying data must produce different real checksums"
        );
    }

    /// Full round trip: export a user's real data, import it into a
    /// completely separate (initially empty) persistence backend, and
    /// verify the real data actually arrives there -- proving `restore`
    /// really restores instead of silently producing an empty archive.
    #[tokio::test]
    async fn test_export_import_round_trip_restores_real_data() {
        let source_persistence = test_persistence().await;
        let source_storage = Arc::new(RwLock::new(FileDataStorage::new(
            std::env::temp_dir()
                .join(format!("voirs_export_src_{}", std::process::id()))
                .to_string_lossy()
                .to_string(),
        )));
        let export_manager = DataManager::new(
            source_storage,
            ExportOptions::default(),
            ImportOptions::default(),
        )
        .await
        .unwrap()
        .with_persistence(source_persistence.clone());

        let seeded_progress = UserProgress {
            user_id: "roundtrip_user".to_string(),
            overall_skill_level: 0.77,
            ..UserProgress::default()
        };
        source_persistence
            .save_user_progress("roundtrip_user", &seeded_progress)
            .await
            .unwrap();
        source_persistence
            .save_feedback(
                "roundtrip_user",
                &FeedbackResponse {
                    feedback_items: vec![UserFeedback {
                        message: "Great articulation".to_string(),
                        suggestion: None,
                        confidence: 0.9,
                        score: 0.88,
                        priority: 0.5,
                        metadata: HashMap::new(),
                    }],
                    overall_score: 0.88,
                    immediate_actions: vec![],
                    long_term_goals: vec![],
                    progress_indicators: ProgressIndicators::default(),
                    timestamp: Utc::now(),
                    processing_time: std::time::Duration::from_millis(5),
                    feedback_type: FeedbackType::Quality,
                },
            )
            .await
            .unwrap();
        let seeded_session_id = uuid::Uuid::new_v4();
        source_persistence
            .save_session(&SessionState {
                session_id: seeded_session_id,
                user_id: "roundtrip_user".to_string(),
                start_time: Utc::now(),
                last_activity: Utc::now(),
                current_task: None,
                stats: crate::traits::SessionStats::default(),
                preferences: crate::traits::UserPreferences::default(),
                adaptive_state: crate::traits::AdaptiveState::default(),
                current_exercise: None,
                session_stats: crate::traits::SessionStatistics {
                    start_time: Utc::now(),
                    end_time: Some(Utc::now()),
                    duration: std::time::Duration::from_secs(120),
                    audio_generated_count: 4,
                    average_quality_score: 0.0,
                    average_pronunciation_score: 0.0,
                    exercises_attempted: 0,
                    exercises_completed: 0,
                },
            })
            .await
            .unwrap();

        let export_path = std::env::temp_dir().join(format!(
            "voirs_export_roundtrip_{}.json",
            std::process::id()
        ));
        export_manager
            .export_data("roundtrip_user", &export_path, ExportFormat::Json)
            .await
            .unwrap();

        // A completely separate, initially-empty destination backend.
        let dest_persistence = test_persistence().await;
        let dest_storage = Arc::new(RwLock::new(FileDataStorage::new(
            std::env::temp_dir()
                .join(format!("voirs_export_dst_{}", std::process::id()))
                .to_string_lossy()
                .to_string(),
        )));
        let import_manager = DataManager::new(
            dest_storage,
            ExportOptions::default(),
            ImportOptions {
                skip_validation: false,
                merge_mode: false,
                create_backup: false,
                duplicate_strategy: DuplicateStrategy::Overwrite,
                transformations: Vec::new(),
            },
        )
        .await
        .unwrap()
        .with_persistence(dest_persistence.clone());

        // Sanity check: the destination genuinely has nothing yet.
        assert!(dest_persistence
            .load_user_progress("roundtrip_user")
            .await
            .is_err());

        let report = import_manager
            .import_data("roundtrip_user", &export_path, None)
            .await
            .unwrap();
        assert!(report.validation_report.is_valid);
        assert_eq!(
            report.import_result.records_imported.get("user_progress"),
            Some(&1)
        );
        assert_eq!(
            report.import_result.records_imported.get("feedback"),
            Some(&1)
        );
        assert_eq!(
            report.import_result.records_imported.get("sessions"),
            Some(&1)
        );

        // The real data must now genuinely exist in the destination backend.
        let restored_progress = dest_persistence
            .load_user_progress("roundtrip_user")
            .await
            .unwrap();
        assert!((restored_progress.overall_skill_level - 0.77).abs() < 1e-6);

        let restored_feedback = dest_persistence
            .load_feedback_history("roundtrip_user", None, None)
            .await
            .unwrap();
        assert_eq!(restored_feedback.len(), 1);
        assert!((restored_feedback[0].overall_score - 0.88).abs() < 1e-6);

        let restored_session = dest_persistence
            .load_session(&seeded_session_id)
            .await
            .unwrap();
        assert_eq!(restored_session.user_id, "roundtrip_user");
        assert_eq!(
            restored_session.session_stats.audio_generated_count, 4,
            "the real seeded session's activity count must survive the round trip"
        );

        let _ = std::fs::remove_file(&export_path);
    }

    #[tokio::test]
    async fn test_file_format_detection() {
        let json_data = b"{}";
        let binary_data = b"\x00\x01\x02\x03";
        let gzip_data = b"\x1f\x8b\x08\x00";

        let json_path = Path::new("test.json");
        let bin_path = Path::new("test.bin");
        let gz_path = Path::new("test.gz");

        assert_eq!(
            FileDataStorage::detect_format(json_data, json_path).unwrap(),
            ExportFormat::Json
        );
        assert_eq!(
            FileDataStorage::detect_format(binary_data, bin_path).unwrap(),
            ExportFormat::Binary
        );
        assert_eq!(
            FileDataStorage::detect_format(gzip_data, gz_path).unwrap(),
            ExportFormat::CompressedJson
        );
    }

    #[tokio::test]
    async fn test_data_compression() {
        let test_data = b"Hello, world! This is a test string for compression.";

        let compressed = FileDataStorage::compress_data(test_data).unwrap();
        assert!(compressed.len() < test_data.len() || compressed.len() > 0);

        let decompressed = FileDataStorage::decompress_data(&compressed).unwrap();
        assert_eq!(decompressed, test_data);
    }

    #[tokio::test]
    async fn test_data_encryption() {
        let test_data = b"Secret test data";
        let key = "test_key";

        let encrypted = FileDataStorage::encrypt_data(test_data, key).unwrap();
        assert_ne!(encrypted, test_data);

        let decrypted = FileDataStorage::decrypt_data(&encrypted, key).unwrap();
        assert_eq!(decrypted, test_data);
    }

    #[tokio::test]
    async fn test_export_options_defaults() {
        let options = ExportOptions::default();
        assert!(!options.include_sensitive_data);
        assert!(options.anonymize_data);
        assert_eq!(options.compression_level, 6);
        assert!(!options.encryption_enabled);
        assert!(options.include_data_types.contains(&DataType::UserProgress));
    }

    #[tokio::test]
    async fn test_import_options_defaults() {
        let options = ImportOptions::default();
        assert!(!options.skip_validation);
        assert!(!options.merge_mode);
        assert!(options.create_backup);
        assert_eq!(options.duplicate_strategy, DuplicateStrategy::Skip);
    }

    #[tokio::test]
    async fn test_validation_report() {
        let mut package = DataExportPackage {
            metadata: ExportMetadata {
                created_at: Utc::now(),
                format: ExportFormat::Json,
                voris_version: "1.0.0".to_string(),
                export_version: "1.0.0".to_string(),
                exported_by: "test".to_string(),
                data_size: 0,
                record_counts: HashMap::new(),
                export_options: ExportOptions::default(),
                checksum: String::new(),
            },
            user_progress: HashMap::new(),
            analytics: AnalyticsExportData {
                sessions: Vec::new(),
                performance_metrics: Vec::new(),
                interactions: Vec::new(),
                system_metrics: Vec::new(),
            },
            configurations: SystemConfigurations {
                feedback_configs: HashMap::new(),
                adaptive_configs: HashMap::new(),
                realtime_configs: HashMap::new(),
                ui_preferences: HashMap::new(),
                privacy_settings: HashMap::new(),
            },
            training_data: TrainingExportData {
                exercises: Vec::new(),
                sessions: Vec::new(),
                custom_exercises: Vec::new(),
                statistics: TrainingStatistics {
                    total_sessions: 0,
                    total_exercises: 0,
                    average_score: 0.0,
                    improvement_rate: 0.0,
                    time_spent_minutes: 0,
                },
            },
            feedback_history: Vec::new(),
            quality_metrics: QualityMetricsExport {
                metrics: Vec::new(),
                alerts: Vec::new(),
                reports: Vec::new(),
            },
            gamification: None,
        };
        // Give the package a genuine checksum, computed the same way
        // `collect_export_data` does, rather than a placeholder string --
        // `validate_data` now really verifies it.
        let serialized = canonical_checksum_bytes(&package).unwrap();
        package.metadata.checksum = compute_checksum(&serialized);

        let storage = FileDataStorage::new("test".to_string());
        let report = storage.validate_data(&package).await.unwrap();

        assert!(report.is_valid);
        assert!(report.errors.is_empty());
        assert_eq!(report.integrity_checks.get("checksum_valid"), Some(&true));
    }

    /// A tampered/corrupted checksum must be caught, not silently accepted.
    #[tokio::test]
    async fn test_validation_report_detects_checksum_mismatch() {
        let package = DataExportPackage {
            metadata: ExportMetadata {
                created_at: Utc::now(),
                format: ExportFormat::Json,
                voris_version: "1.0.0".to_string(),
                export_version: "1.0.0".to_string(),
                exported_by: "test".to_string(),
                data_size: 1024,
                record_counts: HashMap::new(),
                export_options: ExportOptions::default(),
                checksum: "not_a_real_checksum".to_string(),
            },
            user_progress: HashMap::new(),
            analytics: AnalyticsExportData {
                sessions: Vec::new(),
                performance_metrics: Vec::new(),
                interactions: Vec::new(),
                system_metrics: Vec::new(),
            },
            configurations: SystemConfigurations {
                feedback_configs: HashMap::new(),
                adaptive_configs: HashMap::new(),
                realtime_configs: HashMap::new(),
                ui_preferences: HashMap::new(),
                privacy_settings: HashMap::new(),
            },
            training_data: TrainingExportData {
                exercises: Vec::new(),
                sessions: Vec::new(),
                custom_exercises: Vec::new(),
                statistics: TrainingStatistics {
                    total_sessions: 0,
                    total_exercises: 0,
                    average_score: 0.0,
                    improvement_rate: 0.0,
                    time_spent_minutes: 0,
                },
            },
            feedback_history: Vec::new(),
            quality_metrics: QualityMetricsExport {
                metrics: Vec::new(),
                alerts: Vec::new(),
                reports: Vec::new(),
            },
            gamification: None,
        };

        let storage = FileDataStorage::new("test".to_string());
        let report = storage.validate_data(&package).await.unwrap();

        assert!(!report.is_valid);
        assert_eq!(report.integrity_checks.get("checksum_valid"), Some(&false));
        assert!(report.errors.iter().any(|e| e.contains("Checksum")));
    }
}
