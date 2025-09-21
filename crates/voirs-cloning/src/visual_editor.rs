//! Visual Voice Editor for VoiRS Voice Cloning
//!
//! This module provides a comprehensive GUI interface for voice characteristic tuning,
//! allowing users to interactively adjust voice parameters, preview changes in real-time,
//! and save/load voice profiles with custom settings.

use crate::core::VoiceCloner;
use crate::quality::{CloningQualityAssessor, QualityMetrics};
use crate::similarity::SimilarityMeasurer;
use crate::types::{
    CloningMethod, SpeakerProfile, VoiceCloneRequest, VoiceCloneResult, VoiceSample,
};
use crate::usage_tracking::SimilarityMetrics;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tokio::sync::Mutex;

/// Voice parameter control types for the GUI
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterControl {
    /// Slider control for continuous values
    Slider {
        min: f32,
        max: f32,
        step: f32,
        default: f32,
        value: f32,
    },
    /// Dropdown selection for discrete choices
    Dropdown {
        options: Vec<String>,
        default_index: usize,
        selected_index: usize,
    },
    /// Checkbox for boolean values
    Checkbox { default: bool, checked: bool },
    /// Text input for string values
    TextInput {
        default: String,
        value: String,
        max_length: Option<usize>,
    },
    /// Color picker for visual parameters
    ColorPicker {
        default: (u8, u8, u8),
        color: (u8, u8, u8),
    },
}

/// Voice characteristic categories for organization
#[derive(Debug, Clone, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub enum CharacteristicCategory {
    /// Fundamental frequency and pitch characteristics
    Pitch,
    /// Formant frequencies and vocal tract shape
    FormantFrequencies,
    /// Speaking rate and rhythm patterns
    Rhythm,
    /// Voice quality and breathiness
    VoiceQuality,
    /// Emotional characteristics
    Emotion,
    /// Prosodic features
    Prosody,
    /// Articulation and pronunciation
    Articulation,
    /// Age and gender characteristics
    Demographics,
    /// Advanced neural model parameters
    Advanced,
}

/// Voice parameter definition for the editor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceParameter {
    /// Parameter unique identifier
    pub id: String,
    /// Human-readable parameter name
    pub name: String,
    /// Parameter description and usage hints
    pub description: String,
    /// Parameter category for organization
    pub category: CharacteristicCategory,
    /// GUI control type and configuration
    pub control: ParameterControl,
    /// Whether this parameter affects audio output
    pub affects_audio: bool,
    /// Parameter importance level (0.0 to 1.0)
    pub importance: f32,
    /// Whether parameter requires model recomputation
    pub requires_recomputation: bool,
}

/// Visual editor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualEditorConfig {
    /// Enable real-time preview during parameter changes
    pub enable_realtime_preview: bool,
    /// Preview update delay in milliseconds
    pub preview_delay_ms: u64,
    /// Maximum preview audio duration in seconds
    pub max_preview_duration: f32,
    /// Enable automatic quality assessment
    pub enable_quality_assessment: bool,
    /// Enable similarity comparison with original
    pub enable_similarity_comparison: bool,
    /// Number of recent parameter changes to keep in history
    pub parameter_history_size: usize,
    /// Enable undo/redo functionality
    pub enable_undo_redo: bool,
    /// Auto-save interval in seconds
    pub auto_save_interval_secs: u64,
    /// GUI theme configuration
    pub theme: EditorTheme,
}

/// Editor theme and visual configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorTheme {
    /// Primary color for UI elements
    pub primary_color: (u8, u8, u8),
    /// Secondary color for accents
    pub secondary_color: (u8, u8, u8),
    /// Background color
    pub background_color: (u8, u8, u8),
    /// Text color
    pub text_color: (u8, u8, u8),
    /// Font size for UI elements
    pub font_size: u16,
    /// Whether to use dark theme
    pub dark_theme: bool,
}

/// Parameter change history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterChange {
    /// Parameter that was changed
    pub parameter_id: String,
    /// Previous value
    pub old_value: String,
    /// New value
    pub new_value: String,
    /// Timestamp of change
    pub timestamp: SystemTime,
    /// Optional user comment
    pub comment: Option<String>,
}

/// Visual editor session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorSession {
    /// Session unique identifier
    pub session_id: String,
    /// Speaker profile being edited
    pub speaker_profile: SpeakerProfile,
    /// Current parameter values
    pub current_parameters: HashMap<String, String>,
    /// Parameter change history
    pub change_history: Vec<ParameterChange>,
    /// Current history position for undo/redo
    pub history_position: usize,
    /// Session creation time
    pub created_at: SystemTime,
    /// Last modification time
    pub modified_at: SystemTime,
    /// Whether session has unsaved changes
    pub has_unsaved_changes: bool,
}

/// Real-time preview result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewResult {
    /// Generated preview audio
    pub audio: Vec<f32>,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Quality metrics for the preview
    pub quality_metrics: QualityMetrics,
    /// Similarity to original speaker
    pub similarity_metrics: SimilarityMetrics,
    /// Parameters used for generation
    pub parameters_used: HashMap<String, String>,
    /// Generation time
    pub generation_time: Duration,
    /// Preview timestamp
    pub timestamp: SystemTime,
}

/// Visual Voice Editor main interface
pub struct VisualVoiceEditor {
    /// Editor configuration
    config: VisualEditorConfig,
    /// Available voice parameters
    parameters: HashMap<String, VoiceParameter>,
    /// Active editor sessions
    sessions: Arc<RwLock<HashMap<String, EditorSession>>>,
    /// Voice cloner for preview generation
    cloner: Arc<VoiceCloner>,
    /// Quality assessor for real-time evaluation
    quality_assessor: Arc<CloningQualityAssessor>,
    /// Similarity measurer for comparison
    similarity_measurer: Arc<SimilarityMeasurer>,
    /// Preview generation mutex to prevent conflicts
    preview_mutex: Arc<Mutex<()>>,
    /// Recent preview results cache
    preview_cache: Arc<RwLock<HashMap<String, PreviewResult>>>,
}

impl Default for VisualEditorConfig {
    fn default() -> Self {
        Self {
            enable_realtime_preview: true,
            preview_delay_ms: 500,
            max_preview_duration: 5.0,
            enable_quality_assessment: true,
            enable_similarity_comparison: true,
            parameter_history_size: 100,
            enable_undo_redo: true,
            auto_save_interval_secs: 30,
            theme: EditorTheme::default(),
        }
    }
}

impl Default for EditorTheme {
    fn default() -> Self {
        Self {
            primary_color: (70, 130, 180),
            secondary_color: (255, 165, 0),
            background_color: (248, 248, 255),
            text_color: (25, 25, 25),
            font_size: 12,
            dark_theme: false,
        }
    }
}

impl VisualVoiceEditor {
    /// Create new visual voice editor
    pub async fn new(config: VisualEditorConfig) -> Result<Self> {
        let cloner = Arc::new(VoiceCloner::new()?);
        let quality_assessor = Arc::new(CloningQualityAssessor::new()?);
        let similarity_config = crate::similarity::SimilarityConfig::default();
        let similarity_measurer = Arc::new(SimilarityMeasurer::new(similarity_config));

        let parameters = Self::initialize_default_parameters();

        Ok(Self {
            config,
            parameters,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            cloner,
            quality_assessor,
            similarity_measurer,
            preview_mutex: Arc::new(Mutex::new(())),
            preview_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Initialize default voice parameters for the editor
    fn initialize_default_parameters() -> HashMap<String, VoiceParameter> {
        let mut parameters = HashMap::new();

        // Pitch parameters
        parameters.insert(
            "f0_mean".to_string(),
            VoiceParameter {
                id: "f0_mean".to_string(),
                name: "Average Pitch".to_string(),
                description: "Overall fundamental frequency (pitch) of the voice".to_string(),
                category: CharacteristicCategory::Pitch,
                control: ParameterControl::Slider {
                    min: 80.0,
                    max: 500.0,
                    step: 1.0,
                    default: 150.0,
                    value: 150.0,
                },
                affects_audio: true,
                importance: 0.9,
                requires_recomputation: true,
            },
        );

        parameters.insert(
            "f0_variation".to_string(),
            VoiceParameter {
                id: "f0_variation".to_string(),
                name: "Pitch Variation".to_string(),
                description: "Amount of pitch variation and intonation".to_string(),
                category: CharacteristicCategory::Pitch,
                control: ParameterControl::Slider {
                    min: 0.1,
                    max: 2.0,
                    step: 0.1,
                    default: 1.0,
                    value: 1.0,
                },
                affects_audio: true,
                importance: 0.8,
                requires_recomputation: true,
            },
        );

        // Formant parameters
        parameters.insert(
            "formant_f1".to_string(),
            VoiceParameter {
                id: "formant_f1".to_string(),
                name: "First Formant (F1)".to_string(),
                description: "Controls vowel height and mouth opening".to_string(),
                category: CharacteristicCategory::FormantFrequencies,
                control: ParameterControl::Slider {
                    min: 200.0,
                    max: 1200.0,
                    step: 10.0,
                    default: 500.0,
                    value: 500.0,
                },
                affects_audio: true,
                importance: 0.9,
                requires_recomputation: true,
            },
        );

        parameters.insert(
            "formant_f2".to_string(),
            VoiceParameter {
                id: "formant_f2".to_string(),
                name: "Second Formant (F2)".to_string(),
                description: "Controls vowel backness and tongue position".to_string(),
                category: CharacteristicCategory::FormantFrequencies,
                control: ParameterControl::Slider {
                    min: 800.0,
                    max: 3000.0,
                    step: 10.0,
                    default: 1500.0,
                    value: 1500.0,
                },
                affects_audio: true,
                importance: 0.9,
                requires_recomputation: true,
            },
        );

        // Speaking rate parameters
        parameters.insert(
            "speaking_rate".to_string(),
            VoiceParameter {
                id: "speaking_rate".to_string(),
                name: "Speaking Rate".to_string(),
                description: "Speed of speech (words per minute)".to_string(),
                category: CharacteristicCategory::Rhythm,
                control: ParameterControl::Slider {
                    min: 80.0,
                    max: 250.0,
                    step: 5.0,
                    default: 150.0,
                    value: 150.0,
                },
                affects_audio: true,
                importance: 0.7,
                requires_recomputation: true,
            },
        );

        // Voice quality parameters
        parameters.insert(
            "breathiness".to_string(),
            VoiceParameter {
                id: "breathiness".to_string(),
                name: "Breathiness".to_string(),
                description: "Amount of breath in the voice".to_string(),
                category: CharacteristicCategory::VoiceQuality,
                control: ParameterControl::Slider {
                    min: 0.0,
                    max: 1.0,
                    step: 0.05,
                    default: 0.2,
                    value: 0.2,
                },
                affects_audio: true,
                importance: 0.6,
                requires_recomputation: true,
            },
        );

        // Emotional parameters
        parameters.insert(
            "emotion_valence".to_string(),
            VoiceParameter {
                id: "emotion_valence".to_string(),
                name: "Emotional Valence".to_string(),
                description: "Positive/negative emotional tone (-1.0 to 1.0)".to_string(),
                category: CharacteristicCategory::Emotion,
                control: ParameterControl::Slider {
                    min: -1.0,
                    max: 1.0,
                    step: 0.1,
                    default: 0.0,
                    value: 0.0,
                },
                affects_audio: true,
                importance: 0.5,
                requires_recomputation: true,
            },
        );

        parameters.insert(
            "emotion_arousal".to_string(),
            VoiceParameter {
                id: "emotion_arousal".to_string(),
                name: "Emotional Arousal".to_string(),
                description: "Energy level and activation (0.0 to 1.0)".to_string(),
                category: CharacteristicCategory::Emotion,
                control: ParameterControl::Slider {
                    min: 0.0,
                    max: 1.0,
                    step: 0.1,
                    default: 0.5,
                    value: 0.5,
                },
                affects_audio: true,
                importance: 0.5,
                requires_recomputation: true,
            },
        );

        // Advanced parameters
        parameters.insert(
            "neural_temperature".to_string(),
            VoiceParameter {
                id: "neural_temperature".to_string(),
                name: "Neural Temperature".to_string(),
                description: "Randomness in neural model generation".to_string(),
                category: CharacteristicCategory::Advanced,
                control: ParameterControl::Slider {
                    min: 0.1,
                    max: 2.0,
                    step: 0.1,
                    default: 1.0,
                    value: 1.0,
                },
                affects_audio: true,
                importance: 0.3,
                requires_recomputation: false,
            },
        );

        parameters
    }

    /// Create new editing session
    pub async fn create_session(&self, speaker_profile: SpeakerProfile) -> Result<String> {
        let session_id = format!(
            "session_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        // Initialize parameters with default values
        let current_parameters: HashMap<String, String> = self
            .parameters
            .iter()
            .map(|(id, param)| {
                let default_value = match &param.control {
                    ParameterControl::Slider { default, .. } => default.to_string(),
                    ParameterControl::Dropdown {
                        options,
                        default_index,
                        ..
                    } => options
                        .get(*default_index)
                        .unwrap_or(&"".to_string())
                        .clone(),
                    ParameterControl::Checkbox { default, .. } => default.to_string(),
                    ParameterControl::TextInput { default, .. } => default.clone(),
                    ParameterControl::ColorPicker { default, .. } => {
                        format!("rgb({},{},{})", default.0, default.1, default.2)
                    }
                };
                (id.clone(), default_value)
            })
            .collect();

        let session = EditorSession {
            session_id: session_id.clone(),
            speaker_profile,
            current_parameters,
            change_history: Vec::new(),
            history_position: 0,
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            has_unsaved_changes: false,
        };

        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session_id.clone(), session);

        Ok(session_id)
    }

    /// Update parameter value in session
    pub async fn update_parameter(
        &self,
        session_id: &str,
        parameter_id: &str,
        value: String,
    ) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        // Validate parameter exists
        let parameter = self
            .parameters
            .get(parameter_id)
            .ok_or_else(|| Error::Validation(format!("Parameter not found: {parameter_id}")))?;

        // Validate parameter value
        self.validate_parameter_value(parameter, &value)?;

        // Record change in history
        if let Some(old_value) = session.current_parameters.get(parameter_id) {
            let change = ParameterChange {
                parameter_id: parameter_id.to_string(),
                old_value: old_value.clone(),
                new_value: value.clone(),
                timestamp: SystemTime::now(),
                comment: None,
            };

            // Trim history if at size limit
            if session.change_history.len() >= self.config.parameter_history_size {
                session.change_history.remove(0);
            }

            session.change_history.push(change);
            session.history_position = session.change_history.len();
        }

        // Update parameter value
        session
            .current_parameters
            .insert(parameter_id.to_string(), value);
        session.modified_at = SystemTime::now();
        session.has_unsaved_changes = true;

        // Generate real-time preview if enabled
        if self.config.enable_realtime_preview && parameter.affects_audio {
            self.generate_preview(session_id).await?;
        }

        Ok(())
    }

    /// Validate parameter value against control constraints
    fn validate_parameter_value(&self, parameter: &VoiceParameter, value: &str) -> Result<()> {
        match &parameter.control {
            ParameterControl::Slider { min, max, .. } => {
                let parsed_value: f32 = value
                    .parse()
                    .map_err(|_| Error::Validation(format!("Invalid number: {value}")))?;
                if parsed_value < *min || parsed_value > *max {
                    return Err(Error::Validation(format!(
                        "Value {} out of range [{}, {}]",
                        parsed_value, min, max
                    )));
                }
            }
            ParameterControl::Dropdown { options, .. } => {
                if !options.contains(&value.to_string()) {
                    return Err(Error::Validation(format!(
                        "Invalid option: {}. Valid options: {:?}",
                        value, options
                    )));
                }
            }
            ParameterControl::Checkbox { .. } => {
                let _ = value
                    .parse::<bool>()
                    .map_err(|_| Error::Validation(format!("Invalid boolean: {value}")))?;
            }
            ParameterControl::TextInput { max_length, .. } => {
                if let Some(max_len) = max_length {
                    if value.len() > *max_len {
                        return Err(Error::Validation(format!(
                            "Text too long: {} characters (max {})",
                            value.len(),
                            max_len
                        )));
                    }
                }
            }
            ParameterControl::ColorPicker { .. } => {
                // Validate RGB format
                if !value.starts_with("rgb(") || !value.ends_with(')') {
                    return Err(Error::Validation(format!(
                        "Invalid color format: {}",
                        value
                    )));
                }
            }
        }
        Ok(())
    }

    /// Generate real-time preview for current session parameters
    async fn generate_preview(&self, session_id: &str) -> Result<()> {
        let _preview_lock = self.preview_mutex.lock().await;

        let sessions = self.sessions.read().unwrap();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        // Create preview request with current parameters
        let preview_text = "This is a voice cloning preview with current parameters.";
        let request = VoiceCloneRequest::new(
            format!("preview_{session_id}"),
            crate::types::SpeakerData::new(session.speaker_profile.clone()),
            CloningMethod::FewShot,
            preview_text.to_string(),
        );

        let start_time = std::time::Instant::now();

        // Generate preview (simplified - in real implementation would apply parameters)
        let result = self.cloner.clone_voice(request).await?;

        let generation_time = start_time.elapsed();

        // Assess quality if enabled
        let quality_metrics = if self.config.enable_quality_assessment {
            // Create voice samples for quality assessment
            let original_sample = session.speaker_profile.samples.first().ok_or_else(|| {
                Error::InsufficientData("No reference samples available".to_string())
            })?;
            let cloned_sample = VoiceSample::new(
                "cloned_result".to_string(),
                result.audio.clone(),
                result.sample_rate,
            );
            let mut quality_assessor = CloningQualityAssessor::new()?;
            quality_assessor
                .assess_quality(original_sample, &cloned_sample)
                .await?
        } else {
            QualityMetrics::default()
        };

        // Assess similarity if enabled
        let similarity_metrics = if self.config.enable_similarity_comparison {
            // Create reference sample for comparison
            let reference_sample = session.speaker_profile.samples.first().ok_or_else(|| {
                Error::InsufficientData("No reference samples available".to_string())
            })?;

            // Create voice samples for similarity measurement
            let result_sample = VoiceSample::new(
                "preview_result".to_string(),
                result.audio.clone(),
                result.sample_rate,
            );

            let similarity_score = self
                .similarity_measurer
                .measure_sample_similarity(&result_sample, reference_sample)
                .await?;

            // Convert to SimilarityMetrics format
            SimilarityMetrics {
                speaker_similarity: similarity_score.overall_score as f64,
                prosody_similarity: similarity_score.perceptual_similarities.erb_similarity as f64,
                acoustic_similarity: similarity_score
                    .spectral_similarities
                    .spectral_centroid_similarity as f64,
                perceptual_similarity: similarity_score
                    .perceptual_similarities
                    .psychoacoustic_similarity as f64,
            }
        } else {
            SimilarityMetrics {
                speaker_similarity: 0.0,
                prosody_similarity: 0.0,
                acoustic_similarity: 0.0,
                perceptual_similarity: 0.0,
            }
        };

        let preview_result = PreviewResult {
            audio: result.audio,
            sample_rate: result.sample_rate,
            quality_metrics,
            similarity_metrics,
            parameters_used: session.current_parameters.clone(),
            generation_time,
            timestamp: SystemTime::now(),
        };

        // Cache preview result
        let mut cache = self.preview_cache.write().unwrap();
        cache.insert(session_id.to_string(), preview_result);

        // Limit cache size
        if cache.len() > 10 {
            let oldest_key = cache.keys().next().unwrap().clone();
            cache.remove(&oldest_key);
        }

        Ok(())
    }

    /// Get current preview result for session
    pub async fn get_preview(&self, session_id: &str) -> Result<Option<PreviewResult>> {
        let cache = self.preview_cache.read().unwrap();
        Ok(cache.get(session_id).cloned())
    }

    /// Undo last parameter change
    pub async fn undo(&self, session_id: &str) -> Result<()> {
        if !self.config.enable_undo_redo {
            return Err(Error::Config("Undo/redo not enabled".to_string()));
        }

        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        if session.history_position == 0 {
            return Err(Error::Validation("Nothing to undo".to_string()));
        }

        session.history_position -= 1;
        let change = &session.change_history[session.history_position];

        // Restore old value
        session
            .current_parameters
            .insert(change.parameter_id.clone(), change.old_value.clone());
        session.modified_at = SystemTime::now();
        session.has_unsaved_changes = true;

        Ok(())
    }

    /// Redo last undone parameter change
    pub async fn redo(&self, session_id: &str) -> Result<()> {
        if !self.config.enable_undo_redo {
            return Err(Error::Config("Undo/redo not enabled".to_string()));
        }

        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        if session.history_position >= session.change_history.len() {
            return Err(Error::Validation("Nothing to redo".to_string()));
        }

        let change = &session.change_history[session.history_position];

        // Apply new value
        session
            .current_parameters
            .insert(change.parameter_id.clone(), change.new_value.clone());
        session.modified_at = SystemTime::now();
        session.has_unsaved_changes = true;

        session.history_position += 1;

        Ok(())
    }

    /// Get available parameters grouped by category
    pub fn get_parameters_by_category(
        &self,
    ) -> HashMap<CharacteristicCategory, Vec<&VoiceParameter>> {
        let mut categorized = HashMap::new();

        for parameter in self.parameters.values() {
            categorized
                .entry(parameter.category.clone())
                .or_insert_with(Vec::new)
                .push(parameter);
        }

        // Sort parameters by importance within each category
        for parameters in categorized.values_mut() {
            parameters.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        }

        categorized
    }

    /// Save session to file
    pub async fn save_session(&self, session_id: &str, file_path: &str) -> Result<()> {
        let sessions = self.sessions.read().unwrap();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        let json = serde_json::to_string_pretty(session).map_err(|e| Error::Serialization(e))?;

        tokio::fs::write(file_path, json)
            .await
            .map_err(|e| Error::Io(e))?;

        Ok(())
    }

    /// Load session from file
    pub async fn load_session(&self, file_path: &str) -> Result<String> {
        let json = tokio::fs::read_to_string(file_path)
            .await
            .map_err(|e| Error::Io(e))?;

        let session: EditorSession =
            serde_json::from_str(&json).map_err(|e| Error::Serialization(e))?;

        let session_id = session.session_id.clone();

        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session_id.clone(), session);

        Ok(session_id)
    }

    /// Get session information
    pub async fn get_session_info(&self, session_id: &str) -> Result<EditorSession> {
        let sessions = self.sessions.read().unwrap();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        Ok(session.clone())
    }

    /// List all active sessions
    pub async fn list_sessions(&self) -> Vec<String> {
        let sessions = self.sessions.read().unwrap();
        sessions.keys().cloned().collect()
    }

    /// Close and cleanup session
    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(session_id);

        let mut cache = self.preview_cache.write().unwrap();
        cache.remove(session_id);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceSample;

    #[tokio::test]
    async fn test_editor_creation() {
        let config = VisualEditorConfig::default();
        let editor = VisualVoiceEditor::new(config).await.unwrap();

        // Verify parameters are initialized
        assert!(!editor.parameters.is_empty());
        assert!(editor.parameters.contains_key("f0_mean"));
        assert!(editor.parameters.contains_key("speaking_rate"));
    }

    #[tokio::test]
    async fn test_session_management() {
        let config = VisualEditorConfig::default();
        let editor = VisualVoiceEditor::new(config).await.unwrap();

        // Create test speaker profile
        let speaker_profile = SpeakerProfile {
            id: "test_speaker".to_string(),
            name: "Test Speaker".to_string(),
            samples: vec![VoiceSample::new(
                "test_sample".to_string(),
                vec![0.0; 1000],
                16000,
            )],
            ..Default::default()
        };

        // Create session
        let session_id = editor.create_session(speaker_profile).await.unwrap();
        assert!(!session_id.is_empty());

        // Verify session exists
        let sessions = editor.list_sessions().await;
        assert!(sessions.contains(&session_id));

        // Close session
        editor.close_session(&session_id).await.unwrap();
        let sessions = editor.list_sessions().await;
        assert!(!sessions.contains(&session_id));
    }

    #[tokio::test]
    async fn test_parameter_update() {
        let config = VisualEditorConfig::default();
        let editor = VisualVoiceEditor::new(config).await.unwrap();

        let speaker_profile = SpeakerProfile::default();
        let session_id = editor.create_session(speaker_profile).await.unwrap();

        // Update parameter
        editor
            .update_parameter(&session_id, "f0_mean", "200.0".to_string())
            .await
            .unwrap();

        // Verify parameter was updated
        let session_info = editor.get_session_info(&session_id).await.unwrap();
        assert_eq!(
            session_info.current_parameters.get("f0_mean"),
            Some(&"200.0".to_string())
        );
        assert!(session_info.has_unsaved_changes);
        assert!(!session_info.change_history.is_empty());
    }

    #[tokio::test]
    async fn test_parameter_validation() {
        let config = VisualEditorConfig::default();
        let editor = VisualVoiceEditor::new(config).await.unwrap();

        let speaker_profile = SpeakerProfile::default();
        let session_id = editor.create_session(speaker_profile).await.unwrap();

        // Test invalid value (out of range)
        let result = editor
            .update_parameter(&session_id, "f0_mean", "1000.0".to_string())
            .await;
        assert!(result.is_err());

        // Test invalid parameter ID
        let result = editor
            .update_parameter(&session_id, "nonexistent", "100.0".to_string())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_undo_redo() {
        let mut config = VisualEditorConfig::default();
        config.enable_undo_redo = true;
        let editor = VisualVoiceEditor::new(config).await.unwrap();

        let speaker_profile = SpeakerProfile::default();
        let session_id = editor.create_session(speaker_profile).await.unwrap();

        // Make changes
        editor
            .update_parameter(&session_id, "f0_mean", "200.0".to_string())
            .await
            .unwrap();
        editor
            .update_parameter(&session_id, "f0_mean", "250.0".to_string())
            .await
            .unwrap();

        // Undo
        editor.undo(&session_id).await.unwrap();
        let session_info = editor.get_session_info(&session_id).await.unwrap();
        assert_eq!(
            session_info.current_parameters.get("f0_mean"),
            Some(&"200.0".to_string())
        );

        // Redo
        editor.redo(&session_id).await.unwrap();
        let session_info = editor.get_session_info(&session_id).await.unwrap();
        assert_eq!(
            session_info.current_parameters.get("f0_mean"),
            Some(&"250.0".to_string())
        );
    }

    #[tokio::test]
    async fn test_parameters_by_category() {
        let config = VisualEditorConfig::default();
        let editor = VisualVoiceEditor::new(config).await.unwrap();

        let categorized = editor.get_parameters_by_category();

        // Verify categories exist
        assert!(categorized.contains_key(&CharacteristicCategory::Pitch));
        assert!(categorized.contains_key(&CharacteristicCategory::FormantFrequencies));
        assert!(categorized.contains_key(&CharacteristicCategory::Emotion));

        // Verify parameters are sorted by importance
        if let Some(pitch_params) = categorized.get(&CharacteristicCategory::Pitch) {
            if pitch_params.len() > 1 {
                assert!(pitch_params[0].importance >= pitch_params[1].importance);
            }
        }
    }
}
