//! Telemetry storage system

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use super::{
    events::{EventType, TelemetryEvent},
    TelemetryError, TelemetryStatistics,
};

/// Telemetry storage
pub struct TelemetryStorage {
    storage_path: PathBuf,
    index_path: PathBuf,
}

impl TelemetryStorage {
    /// Create a new telemetry storage
    pub fn new(storage_path: &Path) -> Result<Self, TelemetryError> {
        let index_path = storage_path.join("index.json");

        Ok(Self {
            storage_path: storage_path.to_path_buf(),
            index_path,
        })
    }

    /// Initialize storage directory
    pub async fn initialize(&self) -> Result<(), TelemetryError> {
        fs::create_dir_all(&self.storage_path).await?;
        Ok(())
    }

    /// Store a telemetry event
    pub async fn store_event(&mut self, event: &TelemetryEvent) -> Result<(), TelemetryError> {
        self.initialize().await?;

        // Create daily event file
        let date = event.timestamp.format("%Y-%m-%d").to_string();
        let event_file = self.storage_path.join(format!("events-{}.jsonl", date));

        // Append event to file
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&event_file)
            .await?;

        let json = serde_json::to_string(event)?;
        file.write_all(json.as_bytes()).await?;
        file.write_all(b"\n").await?;
        file.flush().await?;

        Ok(())
    }

    /// Get all events
    pub async fn get_all_events(&self) -> Result<Vec<TelemetryEvent>, TelemetryError> {
        let mut events = Vec::new();

        if !self.storage_path.exists() {
            return Ok(events);
        }

        let mut entries = fs::read_dir(&self.storage_path).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                let file_events = self.read_events_from_file(&path).await?;
                events.extend(file_events);
            }
        }

        Ok(events)
    }

    /// Read events from a JSONL file
    async fn read_events_from_file(
        &self,
        path: &Path,
    ) -> Result<Vec<TelemetryEvent>, TelemetryError> {
        let content = fs::read_to_string(path).await?;
        let mut events = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<TelemetryEvent>(line) {
                Ok(event) => events.push(event),
                Err(e) => {
                    eprintln!("Failed to parse event: {}", e);
                    continue;
                }
            }
        }

        Ok(events)
    }

    /// Get telemetry statistics
    pub async fn get_statistics(&self) -> Result<TelemetryStatistics, TelemetryError> {
        let events = self.get_all_events().await?;

        if events.is_empty() {
            return Ok(TelemetryStatistics::default());
        }

        let total_events = events.len() as u64;
        let mut events_by_type: HashMap<String, u64> = HashMap::new();
        let mut synthesis_count = 0u64;
        let mut synthesis_durations = Vec::new();
        let mut error_count = 0u64;
        let mut command_counts: HashMap<String, u64> = HashMap::new();
        let mut voice_counts: HashMap<String, u64> = HashMap::new();

        let mut start_time = events[0].timestamp;
        let mut end_time = events[0].timestamp;

        for event in &events {
            // Count events by type
            *events_by_type
                .entry(event.event_type.to_string())
                .or_insert(0) += 1;

            // Track time range
            if event.timestamp < start_time {
                start_time = event.timestamp;
            }
            if event.timestamp > end_time {
                end_time = event.timestamp;
            }

            // Synthesis statistics
            if event.event_type == EventType::SynthesisRequest {
                synthesis_count += 1;
                if let Some(duration) = event.metadata.get("duration_ms") {
                    if let Ok(dur) = duration.parse::<u64>() {
                        synthesis_durations.push(dur);
                    }
                }
                if let Some(voice) = event.metadata.get("voice") {
                    *voice_counts.entry(voice.clone()).or_insert(0) += 1;
                }
            }

            // Error statistics
            if event.event_type == EventType::Error {
                error_count += 1;
            }

            // Command statistics
            if event.event_type == EventType::CommandExecuted {
                if let Some(command) = event.metadata.get("command") {
                    *command_counts.entry(command.clone()).or_insert(0) += 1;
                }
            }
        }

        // Calculate average synthesis duration
        let avg_synthesis_duration = if !synthesis_durations.is_empty() {
            synthesis_durations.iter().sum::<u64>() as f64 / synthesis_durations.len() as f64
        } else {
            0.0
        };

        // Get most used commands
        let mut most_used_commands: Vec<(String, u64)> = command_counts.into_iter().collect();
        most_used_commands.sort_by(|a, b| b.1.cmp(&a.1));
        most_used_commands.truncate(10);

        // Get most used voices
        let mut most_used_voices: Vec<(String, u64)> = voice_counts.into_iter().collect();
        most_used_voices.sort_by(|a, b| b.1.cmp(&a.1));
        most_used_voices.truncate(10);

        // Calculate storage size
        let storage_size_bytes = self.calculate_storage_size().await?;

        Ok(TelemetryStatistics {
            total_events,
            events_by_type,
            synthesis_requests: synthesis_count,
            avg_synthesis_duration,
            total_errors: error_count,
            most_used_commands,
            most_used_voices,
            start_time: Some(start_time),
            end_time: Some(end_time),
            storage_size_bytes,
        })
    }

    /// Calculate total storage size
    async fn calculate_storage_size(&self) -> Result<u64, TelemetryError> {
        let mut total_size = 0u64;

        if !self.storage_path.exists() {
            return Ok(0);
        }

        let mut entries = fs::read_dir(&self.storage_path).await?;

        while let Some(entry) = entries.next_entry().await? {
            let metadata = entry.metadata().await?;
            total_size += metadata.len();
        }

        Ok(total_size)
    }

    /// Clear all telemetry data
    pub async fn clear(&mut self) -> Result<(), TelemetryError> {
        if self.storage_path.exists() {
            fs::remove_dir_all(&self.storage_path).await?;
        }
        Ok(())
    }

    /// Get events within a time range
    pub async fn get_events_in_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<TelemetryEvent>, TelemetryError> {
        let all_events = self.get_all_events().await?;

        Ok(all_events
            .into_iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect())
    }

    /// Get events by type
    pub async fn get_events_by_type(
        &self,
        event_type: EventType,
    ) -> Result<Vec<TelemetryEvent>, TelemetryError> {
        let all_events = self.get_all_events().await?;

        Ok(all_events
            .into_iter()
            .filter(|e| e.event_type == event_type)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::events::TelemetryEvent;

    #[tokio::test]
    async fn test_storage_creation() {
        let temp_dir = std::env::temp_dir().join("voirs_storage_test");
        let storage = TelemetryStorage::new(&temp_dir);
        assert!(storage.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn test_initialize() {
        let temp_dir = std::env::temp_dir().join("voirs_storage_test_init");
        let storage =
            TelemetryStorage::new(&temp_dir).expect("Failed to create telemetry storage for test");

        let result = storage.initialize().await;
        assert!(result.is_ok());
        assert!(temp_dir.exists());

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn test_store_and_retrieve_event() {
        let temp_dir = std::env::temp_dir().join("voirs_storage_test_store");
        let mut storage =
            TelemetryStorage::new(&temp_dir).expect("Failed to create telemetry storage for test");

        let event = TelemetryEvent::command_executed("test".to_string(), 100);
        assert!(storage.store_event(&event).await.is_ok());

        let events = storage
            .get_all_events()
            .await
            .expect("Failed to retrieve telemetry events");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::CommandExecuted);

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn test_get_statistics() {
        let temp_dir = std::env::temp_dir().join("voirs_storage_test_stats");
        let mut storage =
            TelemetryStorage::new(&temp_dir).expect("Failed to create telemetry storage for test");

        // Store multiple events
        let event1 = TelemetryEvent::command_executed("synthesize".to_string(), 100);
        let event2 = TelemetryEvent::synthesis_request("voice1".to_string(), 50, 200, true);
        let event3 = TelemetryEvent::synthesis_request("voice2".to_string(), 60, 250, true);

        storage
            .store_event(&event1)
            .await
            .expect("Failed to store event1");
        storage
            .store_event(&event2)
            .await
            .expect("Failed to store event2");
        storage
            .store_event(&event3)
            .await
            .expect("Failed to store event3");

        let stats = storage
            .get_statistics()
            .await
            .expect("Failed to get statistics");
        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.synthesis_requests, 2);
        assert!(stats.avg_synthesis_duration > 0.0);

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn test_clear_data() {
        let temp_dir = std::env::temp_dir().join("voirs_storage_test_clear");
        let mut storage =
            TelemetryStorage::new(&temp_dir).expect("Failed to create telemetry storage for test");

        let event = TelemetryEvent::command_executed("test".to_string(), 100);
        storage
            .store_event(&event)
            .await
            .expect("Failed to store event");

        assert!(storage.clear().await.is_ok());
        assert!(!temp_dir.exists());
    }

    #[tokio::test]
    async fn test_get_events_by_type() {
        let temp_dir = std::env::temp_dir().join("voirs_storage_test_by_type");
        let mut storage =
            TelemetryStorage::new(&temp_dir).expect("Failed to create telemetry storage for test");

        let event1 = TelemetryEvent::command_executed("test1".to_string(), 100);
        let event2 = TelemetryEvent::synthesis_request("voice".to_string(), 50, 200, true);
        let event3 = TelemetryEvent::command_executed("test2".to_string(), 150);

        storage
            .store_event(&event1)
            .await
            .expect("Failed to store event1");
        storage
            .store_event(&event2)
            .await
            .expect("Failed to store event2");
        storage
            .store_event(&event3)
            .await
            .expect("Failed to store event3");

        let command_events = storage
            .get_events_by_type(EventType::CommandExecuted)
            .await
            .expect("Failed to get events by type CommandExecuted");
        assert_eq!(command_events.len(), 2);

        let synthesis_events = storage
            .get_events_by_type(EventType::SynthesisRequest)
            .await
            .expect("Failed to get events by type SynthesisRequest");
        assert_eq!(synthesis_events.len(), 1);

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
