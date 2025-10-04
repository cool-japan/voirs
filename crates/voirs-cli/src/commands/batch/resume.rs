//! Resume functionality for batch processing.

use super::files::BatchInput;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use voirs_sdk::Result;

/// State file for tracking batch processing progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchState {
    /// Version of the state format
    pub version: String,
    /// Timestamp when processing started
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Input file path
    pub input_path: PathBuf,
    /// Output directory
    pub output_dir: PathBuf,
    /// Total number of items
    pub total_items: usize,
    /// Items that have been completed successfully
    pub completed_items: HashMap<String, CompletedItem>,
    /// Items that failed (with retry count)
    pub failed_items: HashMap<String, FailedItem>,
    /// Processing configuration hash (to detect config changes)
    pub config_hash: String,
}

/// Information about a completed item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedItem {
    /// When it was completed
    pub completed_at: chrono::DateTime<chrono::Utc>,
    /// Output file path
    pub output_path: PathBuf,
    /// Processing duration in milliseconds
    pub duration_ms: u64,
    /// Generated audio duration in seconds
    pub audio_duration: f32,
}

/// Information about a failed item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedItem {
    /// Number of retry attempts
    pub retry_count: u32,
    /// Last error message
    pub last_error: String,
    /// When it last failed
    pub failed_at: chrono::DateTime<chrono::Utc>,
}

impl BatchState {
    /// Create a new batch state
    pub fn new(
        input_path: PathBuf,
        output_dir: PathBuf,
        total_items: usize,
        config_hash: String,
    ) -> Self {
        let now = chrono::Utc::now();
        Self {
            version: "1.0".to_string(),
            started_at: now,
            updated_at: now,
            input_path,
            output_dir,
            total_items,
            completed_items: HashMap::new(),
            failed_items: HashMap::new(),
            config_hash,
        }
    }

    /// Get the state file path for a given input
    pub fn get_state_file_path(input_path: &PathBuf) -> PathBuf {
        let mut state_path = input_path.clone();
        if let Some(stem) = input_path.file_stem() {
            let mut filename = stem.to_string_lossy().to_string();
            filename.push_str(".voirs_state.json");
            state_path.set_file_name(filename);
        } else {
            state_path.push(".voirs_state.json");
        }
        state_path
    }

    /// Load state from file
    pub fn load_from_file(path: &PathBuf) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(path)?;
        let state: BatchState = serde_json::from_str(&content)
            .map_err(|e| voirs_sdk::VoirsError::config_error(&e.to_string()))?;

        Ok(Some(state))
    }

    /// Save state to file
    pub fn save_to_file(&self, path: &PathBuf) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| voirs_sdk::VoirsError::config_error(&e.to_string()))?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Mark an item as completed
    pub fn mark_completed(
        &mut self,
        item_id: &str,
        output_path: PathBuf,
        duration_ms: u64,
        audio_duration: f32,
    ) {
        let completed_item = CompletedItem {
            completed_at: chrono::Utc::now(),
            output_path,
            duration_ms,
            audio_duration,
        };

        self.completed_items
            .insert(item_id.to_string(), completed_item);
        self.failed_items.remove(item_id); // Remove from failed if it was there
        self.updated_at = chrono::Utc::now();
    }

    /// Mark an item as failed
    pub fn mark_failed(&mut self, item_id: &str, error: &str) {
        let current_count = self
            .failed_items
            .get(item_id)
            .map(|f| f.retry_count)
            .unwrap_or(0);

        let failed_item = FailedItem {
            retry_count: current_count + 1,
            last_error: error.to_string(),
            failed_at: chrono::Utc::now(),
        };

        self.failed_items.insert(item_id.to_string(), failed_item);
        self.updated_at = chrono::Utc::now();
    }

    /// Check if an item is already completed
    pub fn is_completed(&self, item_id: &str) -> bool {
        self.completed_items.contains_key(item_id)
    }

    /// Check if an item should be retried
    pub fn should_retry(&self, item_id: &str, max_retries: u32) -> bool {
        if let Some(failed_item) = self.failed_items.get(item_id) {
            failed_item.retry_count < max_retries
        } else {
            true // First attempt
        }
    }

    /// Get items that need processing (not completed and under retry limit)
    pub fn get_pending_items<'a>(
        &self,
        all_items: &'a [BatchInput],
        max_retries: u32,
    ) -> Vec<&'a BatchInput> {
        all_items
            .iter()
            .filter(|item| !self.is_completed(&item.id) && self.should_retry(&item.id, max_retries))
            .collect()
    }

    /// Get progress information
    pub fn get_progress(&self) -> BatchProgress {
        let completed = self.completed_items.len();
        let failed_permanently = self
            .failed_items
            .values()
            .filter(|f| f.retry_count >= 3) // Assuming max retries is 3
            .count();
        let pending = self.total_items - completed - failed_permanently;

        BatchProgress {
            total: self.total_items,
            completed,
            failed: failed_permanently,
            pending,
            completion_percentage: if self.total_items > 0 {
                (completed as f32 / self.total_items as f32) * 100.0
            } else {
                0.0
            },
        }
    }

    /// Check if configuration has changed
    pub fn config_changed(&self, new_config_hash: &str) -> bool {
        self.config_hash != new_config_hash
    }
}

/// Progress information for batch processing
#[derive(Debug, Clone)]
pub struct BatchProgress {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub pending: usize,
    pub completion_percentage: f32,
}

/// Calculate configuration hash for detecting changes
pub fn calculate_config_hash(
    quality: &voirs_sdk::types::QualityLevel,
    rate: f32,
    pitch: f32,
    volume: f32,
) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    format!("{:?}", quality).hash(&mut hasher);
    rate.to_bits().hash(&mut hasher);
    pitch.to_bits().hash(&mut hasher);
    volume.to_bits().hash(&mut hasher);

    format!("{:x}", hasher.finish())
}

/// Resume batch processing from saved state
pub async fn resume_batch_processing(
    input_path: &PathBuf,
    all_items: &[BatchInput],
    batch_config: &super::BatchConfig,
) -> Result<Option<BatchState>> {
    let state_file = BatchState::get_state_file_path(input_path);

    if let Some(mut state) = BatchState::load_from_file(&state_file)? {
        // Check if configuration has changed
        let current_hash = calculate_config_hash(
            &batch_config.quality,
            batch_config.speaking_rate,
            batch_config.pitch,
            batch_config.volume,
        );

        if state.config_changed(&current_hash) {
            tracing::warn!("Configuration has changed since last run. Cannot resume.");
            return Ok(None);
        }

        // Check if input file has changed
        if state.total_items != all_items.len() {
            tracing::warn!("Input file has changed since last run. Cannot resume.");
            return Ok(None);
        }

        Ok(Some(state))
    } else {
        Ok(None)
    }
}

/// Clean up old state files
pub fn cleanup_old_state_files(directory: &PathBuf, max_age_days: u32) -> Result<()> {
    let cutoff_time = chrono::Utc::now() - chrono::Duration::days(max_age_days as i64);

    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();

        if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.ends_with("voirs_state.json"))
            .unwrap_or(false)
        {
            if let Ok(metadata) = entry.metadata() {
                if let Ok(modified) = metadata.modified() {
                    let modified_datetime = chrono::DateTime::<chrono::Utc>::from(modified);
                    if modified_datetime < cutoff_time {
                        if let Err(e) = std::fs::remove_file(&path) {
                            tracing::warn!(
                                "Failed to remove old state file {}: {}",
                                path.display(),
                                e
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn test_batch_state_creation() {
        let state = BatchState::new(
            PathBuf::from("/input.txt"),
            PathBuf::from("/output"),
            100,
            "hash123".to_string(),
        );

        assert_eq!(state.total_items, 100);
        assert_eq!(state.config_hash, "hash123");
        assert!(state.completed_items.is_empty());
        assert!(state.failed_items.is_empty());
    }

    #[test]
    fn test_mark_completed() {
        let mut state = BatchState::new(
            PathBuf::from("/input.txt"),
            PathBuf::from("/output"),
            100,
            "hash123".to_string(),
        );

        state.mark_completed("item1", PathBuf::from("/output/item1.wav"), 1000, 2.5);

        assert!(state.is_completed("item1"));
        assert!(!state.is_completed("item2"));
    }

    #[test]
    fn test_mark_failed_and_retry() {
        let mut state = BatchState::new(
            PathBuf::from("/input.txt"),
            PathBuf::from("/output"),
            100,
            "hash123".to_string(),
        );

        state.mark_failed("item1", "Test error");
        assert!(state.should_retry("item1", 3));

        state.mark_failed("item1", "Test error 2");
        state.mark_failed("item1", "Test error 3");
        assert!(!state.should_retry("item1", 3));
    }

    #[test]
    fn test_calculate_config_hash() {
        let hash1 = calculate_config_hash(&voirs_sdk::QualityLevel::High, 1.0, 0.0, 0.0);

        let hash2 = calculate_config_hash(&voirs_sdk::QualityLevel::Medium, 1.0, 0.0, 0.0);

        assert_ne!(hash1, hash2);

        let hash3 = calculate_config_hash(&voirs_sdk::QualityLevel::High, 1.0, 0.0, 0.0);

        assert_eq!(hash1, hash3);
    }
}
