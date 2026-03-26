//! Command history tracking and suggestions.
//!
//! This module tracks command usage and provides intelligent suggestions
//! based on historical patterns.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use voirs_sdk::Result;

/// History entry for a command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Command name
    pub command: String,
    /// Command arguments (sanitized - no sensitive data)
    pub args: Vec<String>,
    /// Timestamp of execution
    pub timestamp: DateTime<Utc>,
    /// Execution status (success/failure)
    pub status: ExecutionStatus,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
}

/// Execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Command succeeded
    Success,
    /// Command failed
    Failed,
    /// Command was cancelled
    Cancelled,
}

/// Command history manager
pub struct HistoryManager {
    /// Path to history file
    history_path: PathBuf,
    /// Maximum entries to keep
    max_entries: usize,
}

impl HistoryManager {
    /// Create a new history manager
    pub fn new() -> Result<Self> {
        let history_path = Self::get_history_path()?;
        Ok(Self {
            history_path,
            max_entries: 1000,
        })
    }

    /// Get the path to the history file
    fn get_history_path() -> Result<PathBuf> {
        let data_dir = dirs::data_dir()
            .ok_or_else(|| voirs_sdk::VoirsError::config_error("Could not find data directory"))?;

        let voirs_dir = data_dir.join("voirs");
        fs::create_dir_all(&voirs_dir).map_err(|e| voirs_sdk::VoirsError::IoError {
            path: voirs_dir.clone(),
            operation: voirs_sdk::error::IoOperation::Write,
            source: e,
        })?;

        Ok(voirs_dir.join("history.json"))
    }

    /// Load history from file
    pub fn load(&self) -> Result<Vec<HistoryEntry>> {
        if !self.history_path.exists() {
            return Ok(Vec::new());
        }

        let content =
            fs::read_to_string(&self.history_path).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: self.history_path.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;

        let entries: Vec<HistoryEntry> = serde_json::from_str(&content).unwrap_or_default();
        Ok(entries)
    }

    /// Save history to file
    fn save(&self, entries: &[HistoryEntry]) -> Result<()> {
        let content = serde_json::to_string_pretty(entries).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to serialize history: {}", e))
        })?;

        fs::write(&self.history_path, content).map_err(|e| voirs_sdk::VoirsError::IoError {
            path: self.history_path.clone(),
            operation: voirs_sdk::error::IoOperation::Write,
            source: e,
        })?;

        Ok(())
    }

    /// Add a command to history
    pub fn add_entry(&self, entry: HistoryEntry) -> Result<()> {
        let mut entries = self.load()?;
        entries.push(entry);

        // Trim to max entries
        if entries.len() > self.max_entries {
            entries.drain(0..(entries.len() - self.max_entries));
        }

        self.save(&entries)
    }

    /// Get recent command history
    pub fn get_recent(&self, limit: usize) -> Result<Vec<HistoryEntry>> {
        let entries = self.load()?;
        let start = if entries.len() > limit {
            entries.len() - limit
        } else {
            0
        };
        Ok(entries[start..].to_vec())
    }

    /// Get command usage statistics
    pub fn get_statistics(&self) -> Result<CommandStatistics> {
        let entries = self.load()?;
        let mut stats = CommandStatistics::default();

        let mut command_counts: HashMap<String, usize> = HashMap::new();
        let mut total_success = 0;
        let mut total_failed = 0;

        for entry in &entries {
            *command_counts.entry(entry.command.clone()).or_insert(0) += 1;

            match entry.status {
                ExecutionStatus::Success => total_success += 1,
                ExecutionStatus::Failed => total_failed += 1,
                ExecutionStatus::Cancelled => {}
            }
        }

        stats.total_commands = entries.len();
        stats.total_success = total_success;
        stats.total_failed = total_failed;
        stats.command_counts = command_counts;

        // Find most used command
        if let Some((cmd, count)) = stats.command_counts.iter().max_by_key(|(_, count)| *count) {
            stats.most_used_command = Some(cmd.clone());
            stats.most_used_count = *count;
        }

        Ok(stats)
    }

    /// Get intelligent suggestions based on history
    pub fn get_suggestions(&self, limit: usize) -> Result<Vec<String>> {
        let entries = self.load()?;
        let mut suggestions = Vec::new();

        // Get recent successful commands
        let recent_successful: Vec<_> = entries
            .iter()
            .rev()
            .filter(|e| e.status == ExecutionStatus::Success)
            .take(limit * 2)
            .collect();

        // Build frequency map
        let mut freq_map: HashMap<String, usize> = HashMap::new();
        for entry in recent_successful {
            let cmd_line = format!("{} {}", entry.command, entry.args.join(" "));
            *freq_map.entry(cmd_line).or_insert(0) += 1;
        }

        // Sort by frequency and take top suggestions
        let mut freq_vec: Vec<_> = freq_map.into_iter().collect();
        freq_vec.sort_by_key(|b| std::cmp::Reverse(b.1));

        for (cmd, _) in freq_vec.iter().take(limit) {
            suggestions.push(cmd.clone());
        }

        Ok(suggestions)
    }

    /// Clear all history
    pub fn clear(&self) -> Result<()> {
        if self.history_path.exists() {
            fs::remove_file(&self.history_path).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: self.history_path.clone(),
                operation: voirs_sdk::error::IoOperation::Delete,
                source: e,
            })?;
        }
        Ok(())
    }
}

impl Default for HistoryManager {
    fn default() -> Self {
        Self::new().expect("Failed to create history manager")
    }
}

/// Command usage statistics
#[derive(Debug, Default)]
pub struct CommandStatistics {
    /// Total number of commands executed
    pub total_commands: usize,
    /// Number of successful executions
    pub total_success: usize,
    /// Number of failed executions
    pub total_failed: usize,
    /// Command usage counts
    pub command_counts: HashMap<String, usize>,
    /// Most used command
    pub most_used_command: Option<String>,
    /// Count of most used command
    pub most_used_count: usize,
}

/// Run history command
pub async fn run_history(limit: usize, show_stats: bool, suggest: bool, clear: bool) -> Result<()> {
    let manager = HistoryManager::new()?;

    if clear {
        manager.clear()?;
        println!("✅ Command history cleared");
        return Ok(());
    }

    if show_stats {
        let stats = manager.get_statistics()?;
        println!("📊 Command Usage Statistics");
        println!("============================");
        println!("Total commands: {}", stats.total_commands);
        println!(
            "Successful: {} ({:.1}%)",
            stats.total_success,
            if stats.total_commands > 0 {
                (stats.total_success as f64 / stats.total_commands as f64) * 100.0
            } else {
                0.0
            }
        );
        println!(
            "Failed: {} ({:.1}%)",
            stats.total_failed,
            if stats.total_commands > 0 {
                (stats.total_failed as f64 / stats.total_commands as f64) * 100.0
            } else {
                0.0
            }
        );
        println!();

        if let Some(most_used) = stats.most_used_command {
            println!(
                "Most used command: {} ({} times)",
                most_used, stats.most_used_count
            );
        }
        println!();

        println!("Command breakdown:");
        let mut cmd_vec: Vec<_> = stats.command_counts.iter().collect();
        cmd_vec.sort_by(|a, b| b.1.cmp(a.1));
        for (cmd, count) in cmd_vec.iter().take(10) {
            println!("  {}: {} times", cmd, count);
        }
        return Ok(());
    }

    if suggest {
        let suggestions = manager.get_suggestions(limit)?;
        if suggestions.is_empty() {
            println!("No suggestions available yet. Use VoiRS more to build history!");
        } else {
            println!("💡 Suggested commands based on your usage:");
            for (i, suggestion) in suggestions.iter().enumerate() {
                println!("  {}. voirs {}", i + 1, suggestion);
            }
        }
        return Ok(());
    }

    // Show recent history
    let entries = manager.get_recent(limit)?;
    if entries.is_empty() {
        println!("No command history yet.");
    } else {
        println!("📜 Recent command history (last {}):", entries.len());
        println!();
        for (i, entry) in entries.iter().rev().enumerate() {
            let status_icon = match entry.status {
                ExecutionStatus::Success => "✅",
                ExecutionStatus::Failed => "❌",
                ExecutionStatus::Cancelled => "🚫",
            };
            println!(
                "{} {} [{}] voirs {} {}",
                i + 1,
                status_icon,
                entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
                entry.command,
                entry.args.join(" ")
            );
            if let Some(duration) = entry.duration_ms {
                println!("      Duration: {}ms", duration);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_entry_serialization() {
        let entry = HistoryEntry {
            command: "synthesize".to_string(),
            args: vec!["test".to_string()],
            timestamp: Utc::now(),
            status: ExecutionStatus::Success,
            duration_ms: Some(1000),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: HistoryEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.command, "synthesize");
        assert_eq!(deserialized.args, vec!["test"]);
    }

    #[test]
    fn test_command_statistics() {
        let entries = vec![
            HistoryEntry {
                command: "synthesize".to_string(),
                args: vec![],
                timestamp: Utc::now(),
                status: ExecutionStatus::Success,
                duration_ms: None,
            },
            HistoryEntry {
                command: "synthesize".to_string(),
                args: vec![],
                timestamp: Utc::now(),
                status: ExecutionStatus::Success,
                duration_ms: None,
            },
            HistoryEntry {
                command: "list-voices".to_string(),
                args: vec![],
                timestamp: Utc::now(),
                status: ExecutionStatus::Failed,
                duration_ms: None,
            },
        ];

        let mut command_counts = HashMap::new();
        for entry in &entries {
            *command_counts.entry(entry.command.clone()).or_insert(0) += 1;
        }

        assert_eq!(command_counts.get("synthesize"), Some(&2));
        assert_eq!(command_counts.get("list-voices"), Some(&1));
    }
}
