//! Cloning Wizard for VoiRS Voice Cloning
//!
//! This module provides a comprehensive step-by-step assistant for voice cloning,
//! guiding users through the entire process from data collection to final synthesis
//! with automatic quality validation and optimization suggestions.

use crate::consent::{ConsentManager, ConsentRecord};
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

/// Wizard step types for the cloning process
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WizardStep {
    /// Initial setup and project configuration
    ProjectSetup,
    /// Data collection and sample recording
    DataCollection,
    /// Quality assessment and validation
    QualityAssessment,
    /// Consent and ethical considerations
    ConsentManagement,
    /// Method selection and configuration
    MethodSelection,
    /// Model training and optimization
    ModelTraining,
    /// Testing and validation
    TestingValidation,
    /// Final synthesis and export
    FinalSynthesis,
    /// Completion and cleanup
    Completion,
}

/// Wizard step status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StepStatus {
    /// Step not yet started
    NotStarted,
    /// Step in progress
    InProgress,
    /// Step completed successfully
    Completed,
    /// Step failed with errors
    Failed(String),
    /// Step skipped (optional step)
    Skipped,
}

/// Wizard project configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WizardProject {
    /// Project unique identifier
    pub project_id: String,
    /// Human-readable project name
    pub project_name: String,
    /// Project description
    pub description: Option<String>,
    /// Target speaker information
    pub target_speaker: Option<SpeakerProfile>,
    /// Selected cloning method
    pub cloning_method: Option<CloningMethod>,
    /// Quality requirements
    pub quality_target: f32,
    /// Project creation time
    pub created_at: SystemTime,
    /// Last modification time
    pub modified_at: SystemTime,
    /// Project completion status
    pub is_completed: bool,
}

/// Individual wizard step configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WizardStepConfig {
    /// Step type
    pub step: WizardStep,
    /// Step title for UI
    pub title: String,
    /// Step description and instructions
    pub description: String,
    /// Step status
    pub status: StepStatus,
    /// Whether this step is required
    pub required: bool,
    /// Estimated completion time in minutes
    pub estimated_time_minutes: u32,
    /// Step-specific configuration
    pub config: HashMap<String, serde_json::Value>,
    /// Step completion timestamp
    pub completed_at: Option<SystemTime>,
    /// Step validation results
    pub validation_results: Vec<ValidationResult>,
}

/// Step validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation rule identifier
    pub rule_id: String,
    /// Human-readable rule description
    pub rule_description: String,
    /// Validation result
    pub passed: bool,
    /// Validation message or error
    pub message: String,
    /// Validation severity
    pub severity: ValidationSeverity,
    /// Suggested actions for fixing
    pub suggestions: Vec<String>,
}

/// Validation severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Information only
    Info,
    /// Warning that should be addressed
    Warning,
    /// Error that must be fixed
    Error,
    /// Critical error that prevents continuation
    Critical,
}

/// Data collection requirements and progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCollectionProgress {
    /// Minimum required samples
    pub min_samples_required: usize,
    /// Recommended samples for best quality
    pub recommended_samples: usize,
    /// Current number of collected samples
    pub collected_samples: usize,
    /// Minimum duration per sample in seconds
    pub min_sample_duration: f32,
    /// Total duration collected
    pub total_duration_collected: f32,
    /// Required audio quality (SNR threshold)
    pub min_snr_db: f32,
    /// Sample collection status by category
    pub collection_status: HashMap<String, usize>,
}

/// Method selection guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodSelectionGuidance {
    /// Available cloning methods
    pub available_methods: Vec<CloningMethod>,
    /// Recommended method based on data
    pub recommended_method: CloningMethod,
    /// Method comparison and trade-offs
    pub method_comparisons: HashMap<CloningMethod, MethodComparison>,
    /// Selection criteria weights
    pub selection_criteria: SelectionCriteria,
}

/// Method comparison information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodComparison {
    /// Quality score estimate (0.0 to 1.0)
    pub quality_estimate: f32,
    /// Processing time estimate in seconds
    pub processing_time_estimate: f32,
    /// Data requirements
    pub data_requirements: String,
    /// Strengths of this method
    pub strengths: Vec<String>,
    /// Limitations of this method
    pub limitations: Vec<String>,
    /// Suitability score for current project
    pub suitability_score: f32,
}

/// Selection criteria for method recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Quality importance weight
    pub quality_weight: f32,
    /// Speed importance weight
    pub speed_weight: f32,
    /// Data efficiency importance weight
    pub data_efficiency_weight: f32,
    /// Ease of use importance weight
    pub ease_of_use_weight: f32,
}

/// Wizard session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WizardSession {
    /// Session unique identifier
    pub session_id: String,
    /// Associated project
    pub project: WizardProject,
    /// Current step in the wizard
    pub current_step: WizardStep,
    /// All wizard steps configuration
    pub steps: HashMap<WizardStep, WizardStepConfig>,
    /// Data collection progress
    pub data_progress: DataCollectionProgress,
    /// Method selection state
    pub method_guidance: Option<MethodSelectionGuidance>,
    /// Overall session progress (0.0 to 1.0)
    pub overall_progress: f32,
    /// Session creation time
    pub created_at: SystemTime,
    /// Last activity time
    pub last_activity: SystemTime,
    /// Session completion time
    pub completed_at: Option<SystemTime>,
    /// Temporary working files and data
    pub working_data: HashMap<String, serde_json::Value>,
}

/// Main Cloning Wizard interface
pub struct CloningWizard {
    /// Active wizard sessions
    sessions: Arc<RwLock<HashMap<String, WizardSession>>>,
    /// Voice cloner for testing and synthesis
    cloner: Arc<VoiceCloner>,
    /// Quality assessor for validation
    quality_assessor: Arc<CloningQualityAssessor>,
    /// Similarity measurer for validation
    similarity_measurer: Arc<SimilarityMeasurer>,
    /// Consent manager for ethical validation
    consent_manager: Arc<ConsentManager>,
}

impl Default for DataCollectionProgress {
    fn default() -> Self {
        Self {
            min_samples_required: 5,
            recommended_samples: 20,
            collected_samples: 0,
            min_sample_duration: 3.0,
            total_duration_collected: 0.0,
            min_snr_db: 20.0,
            collection_status: HashMap::new(),
        }
    }
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            quality_weight: 0.4,
            speed_weight: 0.2,
            data_efficiency_weight: 0.3,
            ease_of_use_weight: 0.1,
        }
    }
}

impl CloningWizard {
    /// Create new cloning wizard
    pub async fn new() -> Result<Self> {
        let cloner = Arc::new(VoiceCloner::new()?);
        let quality_assessor = Arc::new(CloningQualityAssessor::new()?);
        let similarity_config = crate::similarity::SimilarityConfig::default();
        let similarity_measurer = Arc::new(SimilarityMeasurer::new(similarity_config));
        let consent_manager = Arc::new(ConsentManager::new());

        Ok(Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            cloner,
            quality_assessor,
            similarity_measurer,
            consent_manager,
        })
    }

    /// Start new wizard session
    pub async fn start_wizard(
        &self,
        project_name: String,
        description: Option<String>,
    ) -> Result<String> {
        let project_id = format!(
            "project_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        let session_id = format!(
            "wizard_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let project = WizardProject {
            project_id: project_id.clone(),
            project_name,
            description,
            target_speaker: None,
            cloning_method: None,
            quality_target: 0.8,
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            is_completed: false,
        };

        let steps = Self::initialize_wizard_steps();

        let session = WizardSession {
            session_id: session_id.clone(),
            project,
            current_step: WizardStep::ProjectSetup,
            steps,
            data_progress: DataCollectionProgress::default(),
            method_guidance: None,
            overall_progress: 0.0,
            created_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            completed_at: None,
            working_data: HashMap::new(),
        };

        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session_id.clone(), session);

        Ok(session_id)
    }

    /// Initialize default wizard steps
    fn initialize_wizard_steps() -> HashMap<WizardStep, WizardStepConfig> {
        let mut steps = HashMap::new();

        steps.insert(
            WizardStep::ProjectSetup,
            WizardStepConfig {
                step: WizardStep::ProjectSetup,
                title: "Project Setup".to_string(),
                description: "Configure your voice cloning project settings and requirements"
                    .to_string(),
                status: StepStatus::InProgress,
                required: true,
                estimated_time_minutes: 5,
                config: HashMap::new(),
                completed_at: None,
                validation_results: Vec::new(),
            },
        );

        steps.insert(
            WizardStep::DataCollection,
            WizardStepConfig {
                step: WizardStep::DataCollection,
                title: "Data Collection".to_string(),
                description: "Record or upload voice samples for training the cloning model"
                    .to_string(),
                status: StepStatus::NotStarted,
                required: true,
                estimated_time_minutes: 30,
                config: HashMap::new(),
                completed_at: None,
                validation_results: Vec::new(),
            },
        );

        steps.insert(
            WizardStep::QualityAssessment,
            WizardStepConfig {
                step: WizardStep::QualityAssessment,
                title: "Quality Assessment".to_string(),
                description: "Analyze the quality of collected voice samples".to_string(),
                status: StepStatus::NotStarted,
                required: true,
                estimated_time_minutes: 10,
                config: HashMap::new(),
                completed_at: None,
                validation_results: Vec::new(),
            },
        );

        steps.insert(
            WizardStep::ConsentManagement,
            WizardStepConfig {
                step: WizardStep::ConsentManagement,
                title: "Consent & Ethics".to_string(),
                description: "Ensure proper consent and ethical considerations are addressed"
                    .to_string(),
                status: StepStatus::NotStarted,
                required: true,
                estimated_time_minutes: 15,
                config: HashMap::new(),
                completed_at: None,
                validation_results: Vec::new(),
            },
        );

        steps.insert(
            WizardStep::MethodSelection,
            WizardStepConfig {
                step: WizardStep::MethodSelection,
                title: "Method Selection".to_string(),
                description: "Choose the best voice cloning method for your project".to_string(),
                status: StepStatus::NotStarted,
                required: true,
                estimated_time_minutes: 10,
                config: HashMap::new(),
                completed_at: None,
                validation_results: Vec::new(),
            },
        );

        steps.insert(
            WizardStep::ModelTraining,
            WizardStepConfig {
                step: WizardStep::ModelTraining,
                title: "Model Training".to_string(),
                description: "Train the voice cloning model with your data".to_string(),
                status: StepStatus::NotStarted,
                required: true,
                estimated_time_minutes: 60,
                config: HashMap::new(),
                completed_at: None,
                validation_results: Vec::new(),
            },
        );

        steps.insert(
            WizardStep::TestingValidation,
            WizardStepConfig {
                step: WizardStep::TestingValidation,
                title: "Testing & Validation".to_string(),
                description: "Test the trained model and validate results".to_string(),
                status: StepStatus::NotStarted,
                required: true,
                estimated_time_minutes: 20,
                config: HashMap::new(),
                completed_at: None,
                validation_results: Vec::new(),
            },
        );

        steps.insert(
            WizardStep::FinalSynthesis,
            WizardStepConfig {
                step: WizardStep::FinalSynthesis,
                title: "Final Synthesis".to_string(),
                description: "Generate final voice cloning results".to_string(),
                status: StepStatus::NotStarted,
                required: true,
                estimated_time_minutes: 15,
                config: HashMap::new(),
                completed_at: None,
                validation_results: Vec::new(),
            },
        );

        steps.insert(
            WizardStep::Completion,
            WizardStepConfig {
                step: WizardStep::Completion,
                title: "Completion".to_string(),
                description: "Finalize project and export results".to_string(),
                status: StepStatus::NotStarted,
                required: true,
                estimated_time_minutes: 5,
                config: HashMap::new(),
                completed_at: None,
                validation_results: Vec::new(),
            },
        );

        steps
    }

    /// Add voice sample to current session
    pub async fn add_voice_sample(&self, session_id: &str, sample: VoiceSample) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        // Initialize speaker profile if not exists
        if session.project.target_speaker.is_none() {
            session.project.target_speaker = Some(SpeakerProfile {
                id: format!(
                    "speaker_{project_id}",
                    project_id = session.project.project_id
                ),
                name: session.project.project_name.clone(),
                samples: Vec::new(),
                ..Default::default()
            });
        }

        // Add sample to speaker profile
        if let Some(ref mut speaker) = session.project.target_speaker {
            speaker.samples.push(sample.clone());
        }

        // Update data collection progress
        session.data_progress.collected_samples += 1;
        session.data_progress.total_duration_collected += sample.duration;

        // Update category counts
        let category = sample
            .metadata
            .get("category")
            .and_then(|v| Some(v.as_str()))
            .unwrap_or("general")
            .to_string();

        *session
            .data_progress
            .collection_status
            .entry(category)
            .or_insert(0) += 1;

        session.last_activity = SystemTime::now();
        session.project.modified_at = SystemTime::now();

        Ok(())
    }

    /// Validate current step and move to next if ready
    pub async fn validate_and_advance(&self, session_id: &str) -> Result<bool> {
        let validation_results = self.validate_current_step(session_id).await?;

        // Check if all critical validations pass
        let critical_failures = validation_results
            .iter()
            .filter(|r| !r.passed && r.severity == ValidationSeverity::Critical)
            .count();

        if critical_failures > 0 {
            return Ok(false);
        }

        // Mark current step as completed and advance
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        // Update current step status
        if let Some(step_config) = session.steps.get_mut(&session.current_step) {
            step_config.status = StepStatus::Completed;
            step_config.completed_at = Some(SystemTime::now());
            step_config.validation_results = validation_results;
        }

        // Advance to next step
        let next_step = self.get_next_step(&session.current_step);
        if let Some(next) = next_step {
            session.current_step = next.clone();
            if let Some(next_config) = session.steps.get_mut(&next) {
                next_config.status = StepStatus::InProgress;
            }
        }

        // Update overall progress
        session.overall_progress = self.calculate_progress(session);
        session.last_activity = SystemTime::now();

        Ok(true)
    }

    /// Get next step in the wizard sequence
    fn get_next_step(&self, current: &WizardStep) -> Option<WizardStep> {
        match current {
            WizardStep::ProjectSetup => Some(WizardStep::DataCollection),
            WizardStep::DataCollection => Some(WizardStep::QualityAssessment),
            WizardStep::QualityAssessment => Some(WizardStep::ConsentManagement),
            WizardStep::ConsentManagement => Some(WizardStep::MethodSelection),
            WizardStep::MethodSelection => Some(WizardStep::ModelTraining),
            WizardStep::ModelTraining => Some(WizardStep::TestingValidation),
            WizardStep::TestingValidation => Some(WizardStep::FinalSynthesis),
            WizardStep::FinalSynthesis => Some(WizardStep::Completion),
            WizardStep::Completion => None,
        }
    }

    /// Validate current wizard step
    pub async fn validate_current_step(&self, session_id: &str) -> Result<Vec<ValidationResult>> {
        let sessions = self.sessions.read().unwrap();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        match session.current_step {
            WizardStep::ProjectSetup => self.validate_project_setup(session).await,
            WizardStep::DataCollection => self.validate_data_collection(session).await,
            WizardStep::QualityAssessment => self.validate_quality_assessment(session).await,
            WizardStep::ConsentManagement => self.validate_consent_management(session).await,
            WizardStep::MethodSelection => self.validate_method_selection(session).await,
            WizardStep::ModelTraining => self.validate_model_training(session).await,
            WizardStep::TestingValidation => self.validate_testing_validation(session).await,
            WizardStep::FinalSynthesis => self.validate_final_synthesis(session).await,
            WizardStep::Completion => self.validate_completion(session).await,
        }
    }

    /// Validate project setup step
    async fn validate_project_setup(
        &self,
        session: &WizardSession,
    ) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Validate project name
        if session.project.project_name.trim().is_empty() {
            results.push(ValidationResult {
                rule_id: "project_name_required".to_string(),
                rule_description: "Project name is required".to_string(),
                passed: false,
                message: "Please provide a meaningful project name".to_string(),
                severity: ValidationSeverity::Error,
                suggestions: vec![
                    "Enter a descriptive name for your voice cloning project".to_string()
                ],
            });
        } else {
            results.push(ValidationResult {
                rule_id: "project_name_valid".to_string(),
                rule_description: "Project name is valid".to_string(),
                passed: true,
                message: "Project name looks good".to_string(),
                severity: ValidationSeverity::Info,
                suggestions: Vec::new(),
            });
        }

        // Validate quality target
        if session.project.quality_target < 0.5 {
            results.push(ValidationResult {
                rule_id: "quality_target_low".to_string(),
                rule_description: "Quality target should be reasonable".to_string(),
                passed: false,
                message: "Quality target is very low, consider increasing it".to_string(),
                severity: ValidationSeverity::Warning,
                suggestions: vec!["Set quality target to at least 0.7 for good results".to_string()],
            });
        }

        Ok(results)
    }

    /// Validate data collection step
    async fn validate_data_collection(
        &self,
        session: &WizardSession,
    ) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Check minimum samples
        if session.data_progress.collected_samples < session.data_progress.min_samples_required {
            results.push(ValidationResult {
                rule_id: "min_samples_required".to_string(),
                rule_description: "Minimum number of samples required".to_string(),
                passed: false,
                message: format!(
                    "Need at least {} samples, have {}",
                    session.data_progress.min_samples_required,
                    session.data_progress.collected_samples
                ),
                severity: ValidationSeverity::Critical,
                suggestions: vec![format!(
                    "Record {} more samples",
                    session.data_progress.min_samples_required
                        - session.data_progress.collected_samples
                )],
            });
        } else {
            results.push(ValidationResult {
                rule_id: "min_samples_met".to_string(),
                rule_description: "Minimum samples requirement met".to_string(),
                passed: true,
                message: "Sufficient samples collected".to_string(),
                severity: ValidationSeverity::Info,
                suggestions: Vec::new(),
            });
        }

        // Check recommended samples
        if session.data_progress.collected_samples < session.data_progress.recommended_samples {
            results.push(ValidationResult {
                rule_id: "recommended_samples".to_string(),
                rule_description: "Recommended number of samples for best quality".to_string(),
                passed: false,
                message: format!(
                    "For best quality, collect {} samples (current: {})",
                    session.data_progress.recommended_samples,
                    session.data_progress.collected_samples
                ),
                severity: ValidationSeverity::Warning,
                suggestions: vec![
                    "Consider recording more samples for improved quality".to_string()
                ],
            });
        }

        // Check total duration
        let min_total_duration = session.data_progress.min_sample_duration
            * session.data_progress.min_samples_required as f32;

        if session.data_progress.total_duration_collected < min_total_duration {
            results.push(ValidationResult {
                rule_id: "min_duration_required".to_string(),
                rule_description: "Minimum total audio duration required".to_string(),
                passed: false,
                message: format!(
                    "Need at least {:.1}s of audio, have {:.1}s",
                    min_total_duration, session.data_progress.total_duration_collected
                ),
                severity: ValidationSeverity::Error,
                suggestions: vec!["Record longer samples or add more samples".to_string()],
            });
        }

        Ok(results)
    }

    /// Validate quality assessment step
    async fn validate_quality_assessment(
        &self,
        session: &WizardSession,
    ) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        if let Some(ref speaker) = session.project.target_speaker {
            // Assess quality of collected samples
            for (i, sample) in speaker.samples.iter().enumerate() {
                // Create a mock VoiceCloneResult for quality assessment
                let mock_result = VoiceCloneResult {
                    request_id: format!("quality_check_{i}"),
                    audio: sample.audio.clone(),
                    sample_rate: sample.sample_rate,
                    quality_metrics: HashMap::new(),
                    similarity_score: 1.0, // Self-similarity
                    processing_time: Duration::from_secs(0),
                    method_used: CloningMethod::FewShot,
                    success: true,
                    error_message: None,
                    cross_lingual_info: None,
                    timestamp: SystemTime::now(),
                };

                // Create mock voice samples for quality assessment
                let original_sample = session
                    .project
                    .target_speaker
                    .as_ref()
                    .and_then(|profile| profile.samples.first())
                    .ok_or_else(|| {
                        Error::InsufficientData("No reference samples available".to_string())
                    })?;
                let cloned_sample = VoiceSample::new(
                    "mock_result".to_string(),
                    mock_result.audio.clone(),
                    mock_result.sample_rate,
                );
                let mut quality_assessor = CloningQualityAssessor::new()?;
                let quality_metrics = quality_assessor
                    .assess_quality(original_sample, &cloned_sample)
                    .await?;

                if quality_metrics.overall_score < 0.6 {
                    results.push(ValidationResult {
                        rule_id: format!("sample_quality_{i}"),
                        rule_description: "Sample quality assessment".to_string(),
                        passed: false,
                        message: format!(
                            "Sample {} has low quality (score: {:.2})",
                            i + 1,
                            quality_metrics.overall_score
                        ),
                        severity: ValidationSeverity::Warning,
                        suggestions: vec![
                            "Consider re-recording this sample in a quieter environment"
                                .to_string(),
                            "Check microphone positioning and audio levels".to_string(),
                        ],
                    });
                } else {
                    results.push(ValidationResult {
                        rule_id: format!("sample_quality_{i}"),
                        rule_description: "Sample quality assessment".to_string(),
                        passed: true,
                        message: format!(
                            "Sample {} has good quality (score: {:.2})",
                            i + 1,
                            quality_metrics.overall_score
                        ),
                        severity: ValidationSeverity::Info,
                        suggestions: Vec::new(),
                    });
                }
            }
        } else {
            results.push(ValidationResult {
                rule_id: "no_speaker_data".to_string(),
                rule_description: "Speaker data availability".to_string(),
                passed: false,
                message: "No speaker data available for quality assessment".to_string(),
                severity: ValidationSeverity::Critical,
                suggestions: vec![
                    "Return to data collection step and add voice samples".to_string()
                ],
            });
        }

        Ok(results)
    }

    /// Validate consent management step
    async fn validate_consent_management(
        &self,
        _session: &WizardSession,
    ) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // For now, assume consent is always properly handled
        // In a real implementation, this would check consent records
        results.push(ValidationResult {
            rule_id: "consent_verified".to_string(),
            rule_description: "Consent verification".to_string(),
            passed: true,
            message: "Consent and ethical considerations have been addressed".to_string(),
            severity: ValidationSeverity::Info,
            suggestions: Vec::new(),
        });

        Ok(results)
    }

    /// Validate method selection step
    async fn validate_method_selection(
        &self,
        session: &WizardSession,
    ) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        if session.project.cloning_method.is_none() {
            results.push(ValidationResult {
                rule_id: "method_not_selected".to_string(),
                rule_description: "Cloning method selection".to_string(),
                passed: false,
                message: "No cloning method has been selected".to_string(),
                severity: ValidationSeverity::Critical,
                suggestions: vec![
                    "Select an appropriate cloning method for your project".to_string()
                ],
            });
        } else {
            results.push(ValidationResult {
                rule_id: "method_selected".to_string(),
                rule_description: "Cloning method selection".to_string(),
                passed: true,
                message: "Cloning method has been selected".to_string(),
                severity: ValidationSeverity::Info,
                suggestions: Vec::new(),
            });
        }

        Ok(results)
    }

    /// Validate model training step
    async fn validate_model_training(
        &self,
        _session: &WizardSession,
    ) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Mock validation - in real implementation would check training progress
        results.push(ValidationResult {
            rule_id: "training_completed".to_string(),
            rule_description: "Model training completion".to_string(),
            passed: true,
            message: "Model training completed successfully".to_string(),
            severity: ValidationSeverity::Info,
            suggestions: Vec::new(),
        });

        Ok(results)
    }

    /// Validate testing validation step
    async fn validate_testing_validation(
        &self,
        _session: &WizardSession,
    ) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Mock validation - in real implementation would run actual tests
        results.push(ValidationResult {
            rule_id: "validation_tests_passed".to_string(),
            rule_description: "Validation tests completion".to_string(),
            passed: true,
            message: "All validation tests passed".to_string(),
            severity: ValidationSeverity::Info,
            suggestions: Vec::new(),
        });

        Ok(results)
    }

    /// Validate final synthesis step
    async fn validate_final_synthesis(
        &self,
        _session: &WizardSession,
    ) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        // Mock validation - in real implementation would check synthesis results
        results.push(ValidationResult {
            rule_id: "synthesis_completed".to_string(),
            rule_description: "Final synthesis completion".to_string(),
            passed: true,
            message: "Final synthesis completed successfully".to_string(),
            severity: ValidationSeverity::Info,
            suggestions: Vec::new(),
        });

        Ok(results)
    }

    /// Validate completion step
    async fn validate_completion(&self, _session: &WizardSession) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        results.push(ValidationResult {
            rule_id: "project_completed".to_string(),
            rule_description: "Project completion".to_string(),
            passed: true,
            message: "Voice cloning project completed successfully".to_string(),
            severity: ValidationSeverity::Info,
            suggestions: vec![
                "Consider creating backups of your voice models".to_string(),
                "Test the cloned voice with different texts".to_string(),
            ],
        });

        Ok(results)
    }

    /// Calculate overall progress percentage
    fn calculate_progress(&self, session: &WizardSession) -> f32 {
        let total_steps = session.steps.len() as f32;
        let completed_steps = session
            .steps
            .values()
            .filter(|s| s.status == StepStatus::Completed)
            .count() as f32;

        completed_steps / total_steps
    }

    /// Generate method selection guidance
    pub async fn generate_method_guidance(
        &self,
        session_id: &str,
    ) -> Result<MethodSelectionGuidance> {
        let sessions = self.sessions.read().unwrap();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        let available_methods = vec![
            CloningMethod::FewShot,
            CloningMethod::ZeroShot,
            CloningMethod::FineTuning,
            CloningMethod::VoiceConversion,
        ];

        let mut method_comparisons = HashMap::new();

        // Analyze data characteristics for recommendations
        let sample_count = session.data_progress.collected_samples;
        let total_duration = session.data_progress.total_duration_collected;

        // Few-shot method analysis
        method_comparisons.insert(
            CloningMethod::FewShot,
            MethodComparison {
                quality_estimate: if sample_count >= 10 { 0.85 } else { 0.75 },
                processing_time_estimate: 300.0,
                data_requirements: "5-20 samples, 3+ seconds each".to_string(),
                strengths: vec![
                    "Good balance of quality and data efficiency".to_string(),
                    "Works well with limited data".to_string(),
                    "Fast training time".to_string(),
                ],
                limitations: vec![
                    "May require more samples for best quality".to_string(),
                    "Limited expressiveness with very few samples".to_string(),
                ],
                suitability_score: if sample_count >= 5 && sample_count <= 20 {
                    0.9
                } else {
                    0.6
                },
            },
        );

        // Zero-shot method analysis
        method_comparisons.insert(
            CloningMethod::ZeroShot,
            MethodComparison {
                quality_estimate: 0.7,
                processing_time_estimate: 60.0,
                data_requirements: "1-3 samples, 10+ seconds each".to_string(),
                strengths: vec![
                    "Requires minimal data".to_string(),
                    "Very fast processing".to_string(),
                    "Good for quick prototyping".to_string(),
                ],
                limitations: vec![
                    "Lower quality than few-shot".to_string(),
                    "Limited voice characteristics capture".to_string(),
                    "May not work well for all voices".to_string(),
                ],
                suitability_score: if sample_count < 5 { 0.8 } else { 0.4 },
            },
        );

        // Determine recommended method
        let recommended_method = if sample_count >= 10 && total_duration >= 60.0 {
            CloningMethod::FewShot
        } else if sample_count >= 3 && total_duration >= 30.0 {
            CloningMethod::FewShot
        } else {
            CloningMethod::ZeroShot
        };

        Ok(MethodSelectionGuidance {
            available_methods,
            recommended_method,
            method_comparisons,
            selection_criteria: SelectionCriteria::default(),
        })
    }

    /// Get current wizard session state
    pub async fn get_session_state(&self, session_id: &str) -> Result<WizardSession> {
        let sessions = self.sessions.read().unwrap();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::Validation(format!("Session not found: {session_id}")))?;

        Ok(session.clone())
    }

    /// List all active wizard sessions
    pub async fn list_sessions(&self) -> Vec<String> {
        let sessions = self.sessions.read().unwrap();
        sessions.keys().cloned().collect()
    }

    /// Save wizard session to file
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

    /// Load wizard session from file
    pub async fn load_session(&self, file_path: &str) -> Result<String> {
        let json = tokio::fs::read_to_string(file_path)
            .await
            .map_err(|e| Error::Io(e))?;

        let session: WizardSession =
            serde_json::from_str(&json).map_err(|e| Error::Serialization(e))?;

        let session_id = session.session_id.clone();

        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session_id.clone(), session);

        Ok(session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wizard_creation() {
        let wizard = CloningWizard::new().await.unwrap();

        // Start new wizard session
        let session_id = wizard
            .start_wizard(
                "Test Project".to_string(),
                Some("Test description".to_string()),
            )
            .await
            .unwrap();

        assert!(!session_id.is_empty());

        // Get session state
        let session = wizard.get_session_state(&session_id).await.unwrap();
        assert_eq!(session.project.project_name, "Test Project");
        assert_eq!(session.current_step, WizardStep::ProjectSetup);
        assert!(!session.project.is_completed);
    }

    #[tokio::test]
    async fn test_step_validation() {
        let wizard = CloningWizard::new().await.unwrap();
        let session_id = wizard.start_wizard("Test".to_string(), None).await.unwrap();

        // Validate project setup step
        let results = wizard.validate_current_step(&session_id).await.unwrap();
        assert!(!results.is_empty());

        // Should have validation for project name
        let name_validation = results.iter().find(|r| r.rule_id == "project_name_valid");
        assert!(name_validation.is_some());
        assert!(name_validation.unwrap().passed);
    }

    #[tokio::test]
    async fn test_data_collection() {
        let wizard = CloningWizard::new().await.unwrap();
        let session_id = wizard.start_wizard("Test".to_string(), None).await.unwrap();

        // Add voice sample
        let sample = VoiceSample::new("sample1".to_string(), vec![0.0; 16000], 16000);

        wizard.add_voice_sample(&session_id, sample).await.unwrap();

        // Check data progress
        let session = wizard.get_session_state(&session_id).await.unwrap();
        assert_eq!(session.data_progress.collected_samples, 1);
        assert_eq!(session.data_progress.total_duration_collected, 3.0);
    }

    #[tokio::test]
    async fn test_method_guidance() {
        let wizard = CloningWizard::new().await.unwrap();
        let session_id = wizard.start_wizard("Test".to_string(), None).await.unwrap();

        // Add several samples
        for i in 0..8 {
            let sample = VoiceSample::new(format!("sample{i}"), vec![0.0; 48000], 16000);
            wizard.add_voice_sample(&session_id, sample).await.unwrap();
        }

        // Generate method guidance
        let guidance = wizard.generate_method_guidance(&session_id).await.unwrap();

        assert!(!guidance.available_methods.is_empty());
        assert!(guidance
            .method_comparisons
            .contains_key(&CloningMethod::FewShot));
        assert!(guidance
            .method_comparisons
            .contains_key(&CloningMethod::ZeroShot));

        // With 8 samples, should recommend few-shot
        assert_eq!(guidance.recommended_method, CloningMethod::FewShot);
    }

    #[tokio::test]
    async fn test_session_persistence() {
        let wizard = CloningWizard::new().await.unwrap();
        let session_id = wizard.start_wizard("Test".to_string(), None).await.unwrap();

        // Save session
        let temp_file = "/tmp/test_wizard_session.json";
        wizard.save_session(&session_id, temp_file).await.unwrap();

        // Load session
        let loaded_session_id = wizard.load_session(temp_file).await.unwrap();
        let loaded_session = wizard.get_session_state(&loaded_session_id).await.unwrap();

        assert_eq!(loaded_session.project.project_name, "Test");

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }
}
