//! Session management for interactive mode
//!
//! Handles:
//! - Synthesis history tracking
//! - Session state persistence
//! - Export capabilities
//! - Auto-save functionality

use crate::error::{Result, VoirsCliError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// Maximum number of synthesis entries to keep in history
const MAX_HISTORY_SIZE: usize = 1000;

/// Session data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    /// Session metadata
    pub metadata: SessionMetadata,

    /// Current voice settings
    pub voice_settings: VoiceSettings,

    /// Synthesis history
    pub history: Vec<SynthesisEntry>,

    /// Session statistics
    pub stats: SessionStats,
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session creation time
    pub created_at: DateTime<Utc>,

    /// Last modified time
    pub modified_at: DateTime<Utc>,

    /// Session name
    pub name: Option<String>,

    /// Session description
    pub description: Option<String>,

    /// Session version for compatibility
    pub version: String,
}

/// Voice and synthesis settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSettings {
    /// Current voice
    pub current_voice: Option<String>,

    /// Speed setting
    pub speed: f32,

    /// Pitch setting
    pub pitch: f32,

    /// Volume setting
    pub volume: f32,
}

/// Individual synthesis entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisEntry {
    /// Timestamp of synthesis
    pub timestamp: DateTime<Utc>,

    /// Text that was synthesized
    pub text: String,

    /// Voice used for synthesis
    pub voice: Option<String>,

    /// Synthesis parameters at time of synthesis
    pub parameters: VoiceSettings,

    /// Duration of synthesis (if available)
    pub duration_ms: Option<u64>,

    /// Audio file path (if saved)
    pub audio_file: Option<PathBuf>,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    /// Total number of synthesis operations
    pub total_syntheses: usize,

    /// Total characters synthesized
    pub total_characters: usize,

    /// Total synthesis time (if tracked)
    pub total_time_ms: u64,

    /// Voices used in this session
    pub voices_used: Vec<String>,

    /// Session duration
    pub session_duration_ms: u64,
}

/// Session manager
pub struct SessionManager {
    /// Current session data
    session_data: SessionData,

    /// Auto-save enabled
    auto_save: bool,

    /// Current session file path
    session_file: Option<PathBuf>,

    /// History buffer for efficient operations
    history_buffer: VecDeque<SynthesisEntry>,

    /// Session start time
    session_start: DateTime<Utc>,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(auto_save: bool) -> Self {
        let now = Utc::now();

        let session_data = SessionData {
            metadata: SessionMetadata {
                created_at: now,
                modified_at: now,
                name: None,
                description: None,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            voice_settings: VoiceSettings {
                current_voice: None,
                speed: 1.0,
                pitch: 0.0,
                volume: 1.0,
            },
            history: Vec::new(),
            stats: SessionStats {
                total_syntheses: 0,
                total_characters: 0,
                total_time_ms: 0,
                voices_used: Vec::new(),
                session_duration_ms: 0,
            },
        };

        Self {
            session_data,
            auto_save,
            session_file: None,
            history_buffer: VecDeque::new(),
            session_start: now,
        }
    }

    /// Load session from file
    pub async fn load_session<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        let content = tokio::fs::read_to_string(path).await.map_err(|e| {
            VoirsCliError::IoError(format!(
                "Failed to read session file '{}': {}",
                path.display(),
                e
            ))
        })?;

        self.session_data = serde_json::from_str(&content).map_err(|e| {
            VoirsCliError::SerializationError(format!("Failed to parse session file: {}", e))
        })?;

        // Rebuild history buffer
        self.history_buffer.clear();
        for entry in &self.session_data.history {
            self.history_buffer.push_back(entry.clone());
        }

        self.session_file = Some(path.to_path_buf());

        println!("✓ Session loaded from: {}", path.display());
        if let Some(ref name) = self.session_data.metadata.name {
            println!("  Session name: {}", name);
        }
        println!(
            "  Created: {}",
            self.session_data
                .metadata
                .created_at
                .format("%Y-%m-%d %H:%M:%S UTC")
        );
        println!("  History entries: {}", self.session_data.history.len());

        Ok(())
    }

    /// Save session to file
    pub async fn save_session<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Update session data before saving
        self.sync_session_data();

        let content = serde_json::to_string_pretty(&self.session_data).map_err(|e| {
            VoirsCliError::SerializationError(format!("Failed to serialize session: {}", e))
        })?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                VoirsCliError::IoError(format!(
                    "Failed to create directory '{}': {}",
                    parent.display(),
                    e
                ))
            })?;
        }

        tokio::fs::write(path, content).await.map_err(|e| {
            VoirsCliError::IoError(format!(
                "Failed to write session file '{}': {}",
                path.display(),
                e
            ))
        })?;

        self.session_file = Some(path.to_path_buf());

        println!("✓ Session saved to: {}", path.display());

        Ok(())
    }

    /// Auto-save session if enabled
    pub async fn auto_save(&mut self) -> Result<()> {
        if self.auto_save {
            if let Some(ref session_file) = self.session_file.clone() {
                self.save_session(session_file).await?;
            }
        }
        Ok(())
    }

    /// Add a synthesis entry to the session
    pub fn add_synthesis(&mut self, text: &str, voice: &Option<String>) {
        let entry = SynthesisEntry {
            timestamp: Utc::now(),
            text: text.to_string(),
            voice: voice.clone(),
            parameters: self.session_data.voice_settings.clone(),
            duration_ms: None,
            audio_file: None,
        };

        // Add to buffer
        self.history_buffer.push_back(entry.clone());

        // Limit buffer size
        if self.history_buffer.len() > MAX_HISTORY_SIZE {
            self.history_buffer.pop_front();
        }

        // Update statistics
        self.session_data.stats.total_syntheses += 1;
        self.session_data.stats.total_characters += text.len();

        if let Some(ref voice) = voice {
            if !self.session_data.stats.voices_used.contains(voice) {
                self.session_data.stats.voices_used.push(voice.clone());
            }
        }

        self.session_data.metadata.modified_at = Utc::now();

        // Auto-save if enabled
        if self.auto_save {
            // Note: In a real implementation, we'd want to do this asynchronously
            // without blocking the current operation
            tokio::spawn(async move {
                // Auto-save logic would go here
            });
        }
    }

    /// Get current voice
    pub fn get_current_voice(&self) -> Option<&String> {
        self.session_data.voice_settings.current_voice.as_ref()
    }

    /// Set current voice
    pub fn set_current_voice(&mut self, voice: String) {
        self.session_data.voice_settings.current_voice = Some(voice.clone());

        // Add to used voices if not already present
        if !self.session_data.stats.voices_used.contains(&voice) {
            self.session_data.stats.voices_used.push(voice);
        }

        self.session_data.metadata.modified_at = Utc::now();
    }

    /// Update voice settings
    pub fn update_voice_settings(
        &mut self,
        speed: Option<f32>,
        pitch: Option<f32>,
        volume: Option<f32>,
    ) {
        if let Some(s) = speed {
            self.session_data.voice_settings.speed = s;
        }
        if let Some(p) = pitch {
            self.session_data.voice_settings.pitch = p;
        }
        if let Some(v) = volume {
            self.session_data.voice_settings.volume = v;
        }

        self.session_data.metadata.modified_at = Utc::now();
    }

    /// Get synthesis history
    pub fn get_history(&self) -> Vec<&SynthesisEntry> {
        self.history_buffer.iter().collect()
    }

    /// Get recent history (last N entries)
    pub fn get_recent_history(&self, count: usize) -> Vec<&SynthesisEntry> {
        self.history_buffer.iter().rev().take(count).collect()
    }

    /// Clear session history
    pub fn clear_history(&mut self) {
        self.history_buffer.clear();
        self.session_data.history.clear();
        self.session_data.stats.total_syntheses = 0;
        self.session_data.stats.total_characters = 0;
        self.session_data.stats.total_time_ms = 0;
        self.session_data.metadata.modified_at = Utc::now();
    }

    /// Get session statistics
    pub fn get_stats(&self) -> &SessionStats {
        &self.session_data.stats
    }

    /// Set session metadata
    pub fn set_metadata(&mut self, name: Option<String>, description: Option<String>) {
        self.session_data.metadata.name = name;
        self.session_data.metadata.description = description;
        self.session_data.metadata.modified_at = Utc::now();
    }

    /// Export session history to various formats
    pub async fn export_history<P: AsRef<Path>>(
        &self,
        path: P,
        format: ExportFormat,
    ) -> Result<()> {
        let path = path.as_ref();

        match format {
            ExportFormat::Json => {
                let content = serde_json::to_string_pretty(&self.session_data).map_err(|e| {
                    VoirsCliError::SerializationError(format!("Failed to serialize session: {}", e))
                })?;

                tokio::fs::write(path, content).await.map_err(|e| {
                    VoirsCliError::IoError(format!("Failed to write export file: {}", e))
                })?;
            }
            ExportFormat::Csv => {
                let mut csv_content = String::from("timestamp,text,voice,speed,pitch,volume\\n");

                for entry in &self.history_buffer {
                    csv_content.push_str(&format!(
                        "{},{},{},{},{},{}\\n",
                        entry.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
                        entry.text.replace(',', ";").replace("\\n", " "),
                        entry.voice.as_deref().unwrap_or(""),
                        entry.parameters.speed,
                        entry.parameters.pitch,
                        entry.parameters.volume
                    ));
                }

                tokio::fs::write(path, csv_content).await.map_err(|e| {
                    VoirsCliError::IoError(format!("Failed to write CSV file: {}", e))
                })?;
            }
            ExportFormat::Text => {
                let mut text_content = String::new();

                for entry in &self.history_buffer {
                    text_content.push_str(&format!(
                        "[{}] {}: {}\\n",
                        entry.timestamp.format("%H:%M:%S"),
                        entry.voice.as_deref().unwrap_or("unknown"),
                        entry.text
                    ));
                }

                tokio::fs::write(path, text_content).await.map_err(|e| {
                    VoirsCliError::IoError(format!("Failed to write text file: {}", e))
                })?;
            }
        }

        println!("✓ History exported to: {}", path.display());
        Ok(())
    }

    /// Sync session data with current state
    fn sync_session_data(&mut self) {
        // Update history from buffer
        self.session_data.history = self.history_buffer.iter().cloned().collect();

        // Update session duration
        let session_duration = Utc::now() - self.session_start;
        self.session_data.stats.session_duration_ms = session_duration.num_milliseconds() as u64;

        self.session_data.metadata.modified_at = Utc::now();
    }
}

/// Export format options
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Text,
}
