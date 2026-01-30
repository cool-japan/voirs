//! Privacy controls for telemetry system

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::events::{EventMetadata, TelemetryEvent};

/// Anonymization level for privacy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnonymizationLevel {
    /// No anonymization - full data collection (use with caution)
    None,

    /// Low anonymization - hash user IDs, keep most data
    Low,

    /// Medium anonymization - hash IDs, remove paths, generalize data
    Medium,

    /// High anonymization - maximum privacy, minimal data collection
    High,
}

impl AnonymizationLevel {
    /// Check if user IDs should be hashed
    pub fn hash_user_ids(&self) -> bool {
        !matches!(self, AnonymizationLevel::None)
    }

    /// Check if file paths should be sanitized
    pub fn sanitize_paths(&self) -> bool {
        matches!(self, AnonymizationLevel::Medium | AnonymizationLevel::High)
    }

    /// Check if metadata should be filtered
    pub fn filter_metadata(&self) -> bool {
        matches!(self, AnonymizationLevel::High)
    }

    /// Check if text content should be removed
    pub fn remove_text_content(&self) -> bool {
        matches!(self, AnonymizationLevel::Medium | AnonymizationLevel::High)
    }
}

impl std::fmt::Display for AnonymizationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnonymizationLevel::None => write!(f, "none"),
            AnonymizationLevel::Low => write!(f, "low"),
            AnonymizationLevel::Medium => write!(f, "medium"),
            AnonymizationLevel::High => write!(f, "high"),
        }
    }
}

/// Privacy control for telemetry
pub struct PrivacyControl {
    level: AnonymizationLevel,
    salt: String,
}

impl PrivacyControl {
    /// Create a new privacy control with the specified level
    pub fn new(level: AnonymizationLevel) -> Self {
        Self {
            level,
            salt: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Apply privacy controls to an event
    pub fn anonymize_event(&self, mut event: TelemetryEvent) -> TelemetryEvent {
        // Hash user ID if required
        if self.level.hash_user_ids() {
            if let Some(user_id) = event.user_id {
                event.user_id = Some(self.hash_string(&user_id));
            }
        }

        // Sanitize paths in metadata
        if self.level.sanitize_paths() {
            self.sanitize_metadata_paths(&mut event.metadata);
        }

        // Remove text content if required
        if self.level.remove_text_content() {
            self.remove_text_from_metadata(&mut event.metadata);
        }

        // Filter metadata if required
        if self.level.filter_metadata() {
            self.filter_sensitive_metadata(&mut event.metadata);
        }

        event
    }

    /// Hash a string using SHA-256 with salt
    fn hash_string(&self, input: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        hasher.update(self.salt.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Sanitize file paths in metadata
    fn sanitize_metadata_paths(&self, metadata: &mut EventMetadata) {
        let sensitive_keys = ["path", "file", "directory", "output", "input"];

        for key in sensitive_keys {
            if let Some(value) = metadata.get(key) {
                let sanitized = self.sanitize_path(value);
                metadata.set(key, sanitized);
            }
        }
    }

    /// Sanitize a file path by removing user-specific information
    fn sanitize_path(&self, path: &str) -> String {
        // Replace user home directory with placeholder
        let mut sanitized = path.to_string();

        if let Ok(home) = std::env::var("HOME") {
            if !home.is_empty() {
                sanitized = sanitized.replace(&home, "$HOME");
            }
        }

        if let Ok(userprofile) = std::env::var("USERPROFILE") {
            if !userprofile.is_empty() {
                sanitized = sanitized.replace(&userprofile, "$HOME");
            }
        }

        // Get just the filename if it's a full path
        if let Some(filename) = std::path::Path::new(&sanitized)
            .file_name()
            .and_then(|s| s.to_str())
        {
            filename.to_string()
        } else {
            sanitized
        }
    }

    /// Remove text content from metadata
    fn remove_text_from_metadata(&self, metadata: &mut EventMetadata) {
        let text_keys = ["text", "message", "content", "input_text"];

        for key in text_keys {
            if let Some(value) = metadata.get(key) {
                // Replace with length information only
                metadata.set(key, format!("<redacted {} chars>", value.len()));
            }
        }
    }

    /// Filter sensitive metadata
    fn filter_sensitive_metadata(&self, metadata: &mut EventMetadata) {
        let allowed_keys = [
            "command",
            "voice",
            "duration_ms",
            "success",
            "error_type",
            "severity",
            "metric_name",
            "value",
            "unit",
            "event_type",
        ];

        // Keep only allowed keys
        let current_keys: Vec<String> = metadata.keys().cloned().collect();
        for key in current_keys {
            if !allowed_keys.contains(&key.as_str()) {
                metadata.remove(&key);
            }
        }
    }

    /// Get current anonymization level
    pub fn level(&self) -> AnonymizationLevel {
        self.level
    }

    /// Check if data collection is allowed for a specific type
    pub fn allows_data_type(&self, data_type: &str) -> bool {
        match self.level {
            AnonymizationLevel::None | AnonymizationLevel::Low => true,
            AnonymizationLevel::Medium => {
                // Allow most data types except personal info
                !matches!(data_type, "text_content" | "file_path" | "user_name")
            }
            AnonymizationLevel::High => {
                // Allow only essential metrics
                matches!(
                    data_type,
                    "command" | "duration" | "error_type" | "performance"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::events::EventType;

    #[test]
    fn test_anonymization_level_flags() {
        assert!(!AnonymizationLevel::None.hash_user_ids());
        assert!(AnonymizationLevel::Low.hash_user_ids());
        assert!(AnonymizationLevel::Medium.hash_user_ids());
        assert!(AnonymizationLevel::High.hash_user_ids());

        assert!(!AnonymizationLevel::None.sanitize_paths());
        assert!(!AnonymizationLevel::Low.sanitize_paths());
        assert!(AnonymizationLevel::Medium.sanitize_paths());
        assert!(AnonymizationLevel::High.sanitize_paths());
    }

    #[test]
    fn test_hash_string() {
        let control = PrivacyControl::new(AnonymizationLevel::Medium);
        let hash1 = control.hash_string("test");
        let hash2 = control.hash_string("test");

        assert_eq!(hash1, hash2); // Same input, same hash
        assert_ne!(hash1, "test"); // Hash is different from input
        assert_eq!(hash1.len(), 64); // SHA-256 produces 64 hex characters
    }

    #[test]
    fn test_user_id_hashing() {
        let control = PrivacyControl::new(AnonymizationLevel::Low);
        let mut event =
            TelemetryEvent::new(EventType::CommandExecuted).with_user_id("user123".to_string());

        let anonymized = control.anonymize_event(event.clone());
        assert_ne!(anonymized.user_id.as_ref().unwrap(), "user123");
        assert_eq!(anonymized.user_id.unwrap().len(), 64); // SHA-256 hash
    }

    #[test]
    fn test_path_sanitization() {
        let control = PrivacyControl::new(AnonymizationLevel::Medium);

        // Test with a path containing the actual HOME directory
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/user".to_string());
        let path = format!("{}/documents/file.txt", home);
        let sanitized = control.sanitize_path(&path);

        // Should extract just the filename
        assert_eq!(sanitized, "file.txt");
    }

    #[test]
    fn test_text_removal() {
        let control = PrivacyControl::new(AnonymizationLevel::Medium);
        let mut event = TelemetryEvent::new(EventType::SynthesisRequest);
        event.metadata.set("text", "Hello, this is sensitive text");

        let anonymized = control.anonymize_event(event);
        let text_value = anonymized.metadata.get("text").unwrap();

        assert!(text_value.contains("redacted"));
        assert!(!text_value.contains("Hello"));
    }

    #[test]
    fn test_metadata_filtering() {
        let control = PrivacyControl::new(AnonymizationLevel::High);
        let mut event = TelemetryEvent::new(EventType::CommandExecuted);
        event.metadata.set("command", "synthesize");
        event.metadata.set("user_name", "john_doe");
        event.metadata.set("duration_ms", "1500");

        let anonymized = control.anonymize_event(event);

        assert!(anonymized.metadata.contains("command"));
        assert!(anonymized.metadata.contains("duration_ms"));
        assert!(!anonymized.metadata.contains("user_name"));
    }

    #[test]
    fn test_allows_data_type() {
        let none_control = PrivacyControl::new(AnonymizationLevel::None);
        assert!(none_control.allows_data_type("text_content"));
        assert!(none_control.allows_data_type("file_path"));

        let high_control = PrivacyControl::new(AnonymizationLevel::High);
        assert!(!high_control.allows_data_type("text_content"));
        assert!(high_control.allows_data_type("command"));
        assert!(high_control.allows_data_type("performance"));
    }

    #[test]
    fn test_anonymization_level_display() {
        assert_eq!(AnonymizationLevel::None.to_string(), "none");
        assert_eq!(AnonymizationLevel::Low.to_string(), "low");
        assert_eq!(AnonymizationLevel::Medium.to_string(), "medium");
        assert_eq!(AnonymizationLevel::High.to_string(), "high");
    }

    #[test]
    fn test_privacy_control_level() {
        let control = PrivacyControl::new(AnonymizationLevel::Medium);
        assert_eq!(control.level(), AnonymizationLevel::Medium);
    }
}
