//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::consent::{ConsentManager, ConsentUsageContext, ConsentUsageResult};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use super::types::*;
/// Trait for processing usage events
pub trait UsageEventProcessor: Send + Sync {
    fn process_usage_event(&self, usage: &UsageRecord) -> Result<()>;
    fn get_processor_name(&self) -> &str;
}
/// Trait for usage storage backends
pub trait UsageStorageBackend: Send + Sync {
    fn store_usage_record(&self, usage: &UsageRecord) -> Result<()>;
    fn store_batch(&self, usages: &[UsageRecord]) -> Result<()>;
    fn retrieve_usage_records(&self, filters: &UsageQueryFilters) -> Result<Vec<UsageRecord>>;
    fn get_backend_name(&self) -> &str;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_usage_tracking_creation() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);
        let user_context = UserContext {
            user_id: Some("test-user-001".to_string()),
            application_id: "test-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::API,
            session_id: Some("session-001".to_string()),
            request_id: Some("req-001".to_string()),
            auth_method: Some(AuthenticationMethod::APIKey),
            user_agent: None,
        };
        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceSynthesis,
            speaker_id: Some("test_speaker".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: format!("req-{}", uuid::Uuid::new_v4()),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_tracker_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
            input_data: InputDataInfo {
                data_type: InputDataType::TextPrompt,
                data_size_bytes: 1024,
                audio_duration_seconds: None,
                text_length: Some(100),
                language: Some("en".to_string()),
                content_hash: None,
                input_quality_score: None,
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::Standard,
                processing_mode: ProcessingMode::Balanced,
                model_config: ModelConfiguration {
                    model_name: "test-model".to_string(),
                    model_version: "1.0".to_string(),
                    model_type: ModelType::Acoustic,
                    model_size_mb: Some(100.0),
                    training_data_info: None,
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::SynthesizedAudio,
                data_size_bytes: 0,
                audio_duration_seconds: None,
                quality_score: None,
                similarity_score: None,
                format: Some("wav".to_string()),
                sample_rate: Some(22050),
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "pipeline-001".to_string(),
                pipeline_version: "1.0".to_string(),
                components_used: vec!["acoustic".to_string(), "vocoder".to_string()],
                processing_stages: Vec::new(),
            },
        };
        let usage_id = tracker.start_tracking(user_context, operation).unwrap();
        assert!(!usage_id.is_nil());
        let stats = tracker.get_statistics();
        assert_eq!(stats.total_operations, 0);
    }
    #[test]
    fn test_consent_checking() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);
        let user_context = UserContext {
            user_id: Some("test-user-consent".to_string()),
            application_id: "test-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::API,
            session_id: None,
            request_id: None,
            auth_method: None,
            user_agent: None,
        };
        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceSynthesis,
            speaker_id: Some("test_speaker".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: format!("req-{}", uuid::Uuid::new_v4()),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_tracker_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
            input_data: InputDataInfo {
                data_type: InputDataType::TextPrompt,
                data_size_bytes: 512,
                audio_duration_seconds: None,
                text_length: Some(50),
                language: Some("en".to_string()),
                content_hash: None,
                input_quality_score: None,
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::Standard,
                processing_mode: ProcessingMode::Balanced,
                model_config: ModelConfiguration {
                    model_name: "test-model".to_string(),
                    model_version: "1.0".to_string(),
                    model_type: ModelType::Acoustic,
                    model_size_mb: Some(100.0),
                    training_data_info: None,
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::SynthesizedAudio,
                data_size_bytes: 0,
                audio_duration_seconds: None,
                quality_score: None,
                similarity_score: None,
                format: Some("wav".to_string()),
                sample_rate: Some(16000),
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "test-pipeline".to_string(),
                pipeline_version: "1.0".to_string(),
                components_used: vec!["vocoder".to_string()],
                processing_stages: Vec::new(),
            },
        };
        let usage_id = tracker.start_tracking(user_context, operation).unwrap();
        let result = tracker.check_consent(usage_id, None, "voice_synthesis");
        assert!(result.is_ok());
        match result.unwrap() {
            crate::consent::ConsentUsageResult::Allowed => {}
            _ => panic!("Should allow by default when no consent manager"),
        }
    }
    #[test]
    fn test_usage_record_queries() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);
        for i in 0..5 {
            let user_context = UserContext {
                user_id: Some(format!("user-{i}")),
                application_id: format!("app-{}", i % 2),
                application_version: "1.0.0".to_string(),
                client_type: if i % 2 == 0 {
                    ClientType::API
                } else {
                    ClientType::WebBrowser
                },
                session_id: Some(format!("session-{i}")),
                request_id: Some(format!("req-{i}")),
                auth_method: Some(AuthenticationMethod::APIKey),
                user_agent: None,
            };
            let operation = CloningOperation {
                operation_type: if i % 2 == 0 {
                    CloningOperationType::VoiceSynthesis
                } else {
                    CloningOperationType::VoiceTraining
                },
                speaker_id: Some(format!("speaker_{i}")),
                target_speaker_id: None,
                request_metadata: OperationRequestMetadata {
                    request_id: format!("req-stress-{i}"),
                    timestamp: SystemTime::now(),
                    priority: Priority::Normal,
                    source_application: "stress_test".to_string(),
                    user_preferences: UserPreferences::default(),
                },
                input_data: InputDataInfo {
                    data_type: InputDataType::TextPrompt,
                    data_size_bytes: 1024 * (i + 1),
                    audio_duration_seconds: None,
                    text_length: Some(100),
                    language: Some("en".to_string()),
                    content_hash: None,
                    input_quality_score: None,
                },
                processing_params: ProcessingParameters {
                    quality_level: QualityLevel::Standard,
                    processing_mode: ProcessingMode::Balanced,
                    model_config: ModelConfiguration {
                        model_name: format!("model-{i}"),
                        model_version: "1.0".to_string(),
                        model_type: ModelType::Acoustic,
                        model_size_mb: Some(100.0),
                        training_data_info: None,
                    },
                    advanced_params: HashMap::new(),
                },
                output_data: OutputDataInfo {
                    output_type: OutputDataType::SynthesizedAudio,
                    data_size_bytes: 0,
                    audio_duration_seconds: None,
                    quality_score: None,
                    similarity_score: None,
                    format: Some("wav".to_string()),
                    sample_rate: Some(22050),
                },
                pipeline_info: PipelineInfo {
                    pipeline_id: format!("pipeline-{i}"),
                    pipeline_version: "1.0".to_string(),
                    components_used: vec!["vocoder".to_string()],
                    processing_stages: Vec::new(),
                },
            };
            let _usage_id = tracker.start_tracking(user_context, operation).unwrap();
        }
        let filters = UsageQueryFilters {
            user_id: None,
            application_id: None,
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: None,
        };
        let records = tracker.query_usage_records(&filters).unwrap();
        assert_eq!(records.len(), 5);
        let filters = UsageQueryFilters {
            user_id: Some("user-0".to_string()),
            application_id: None,
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: None,
        };
        let records = tracker.query_usage_records(&filters).unwrap();
        assert_eq!(records.len(), 1);
        let filters = UsageQueryFilters {
            user_id: None,
            application_id: Some("app-0".to_string()),
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: None,
        };
        let records = tracker.query_usage_records(&filters).unwrap();
        assert_eq!(records.len(), 3);
        let filters = UsageQueryFilters {
            user_id: None,
            application_id: None,
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: Some(3),
        };
        let records = tracker.query_usage_records(&filters).unwrap();
        assert_eq!(records.len(), 3);
    }
    #[test]
    fn test_usage_report_generation() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);
        for i in 0..3 {
            let user_context = UserContext {
                user_id: Some(format!("report-user-{i}")),
                application_id: "report-app".to_string(),
                application_version: "1.0.0".to_string(),
                client_type: ClientType::API,
                session_id: None,
                request_id: None,
                auth_method: None,
                user_agent: None,
            };
            let operation = CloningOperation {
                operation_type: CloningOperationType::VoiceSynthesis,
                speaker_id: Some(format!("report_speaker_{i}")),
                target_speaker_id: None,
                request_metadata: OperationRequestMetadata {
                    request_id: format!("report_req_{i}"),
                    timestamp: SystemTime::now(),
                    priority: Priority::Normal,
                    source_application: "usage_report".to_string(),
                    user_preferences: UserPreferences::default(),
                },
                input_data: InputDataInfo {
                    data_type: InputDataType::TextPrompt,
                    data_size_bytes: 1000,
                    audio_duration_seconds: None,
                    text_length: Some(75),
                    language: Some("en".to_string()),
                    content_hash: None,
                    input_quality_score: None,
                },
                processing_params: ProcessingParameters {
                    quality_level: QualityLevel::Standard,
                    processing_mode: ProcessingMode::Balanced,
                    model_config: ModelConfiguration {
                        model_name: "report-model".to_string(),
                        model_version: "1.0".to_string(),
                        model_type: ModelType::Acoustic,
                        model_size_mb: Some(100.0),
                        training_data_info: None,
                    },
                    advanced_params: HashMap::new(),
                },
                output_data: OutputDataInfo {
                    output_type: OutputDataType::SynthesizedAudio,
                    data_size_bytes: 0,
                    audio_duration_seconds: None,
                    quality_score: None,
                    similarity_score: None,
                    format: Some("wav".to_string()),
                    sample_rate: Some(22050),
                },
                pipeline_info: PipelineInfo {
                    pipeline_id: "report-pipeline".to_string(),
                    pipeline_version: "1.0".to_string(),
                    components_used: vec!["vocoder".to_string()],
                    processing_stages: Vec::new(),
                },
            };
            let _usage_id = tracker.start_tracking(user_context, operation).unwrap();
        }
        let filters = UsageQueryFilters {
            user_id: None,
            application_id: Some("report-app".to_string()),
            operation_type: None,
            start_time: None,
            end_time: None,
            status: None,
            limit: None,
        };
        let report = tracker.generate_usage_report(&filters).unwrap();
        assert_eq!(report.total_records, 3);
        assert_eq!(report.records.len(), 3);
        assert!(report.generated_at > UNIX_EPOCH);
    }
    #[test]
    fn test_error_handling() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);
        let fake_usage_id = Uuid::new_v4();
        let outcome = UsageOutcome {
            status: UsageStatus::Failed,
            error: Some(UsageError {
                error_type: UsageErrorType::ProcessingError,
                error_code: "TEST_ERR_001".to_string(),
                error_message: "Test error".to_string(),
                retry_possible: false,
                retry_after: None,
            }),
            compliance_status: ComplianceStatus {
                is_compliant: false,
                compliance_checks: Vec::new(),
                violations: Vec::new(),
                risk_level: RiskLevel::High,
            },
            consent_result: None,
            restrictions_applied: Vec::new(),
            warnings: Vec::new(),
        };
        let resources = ResourceUsage::default();
        let result = tracker.complete_tracking(fake_usage_id, outcome, resources, None);
        assert!(result.is_err());
        if let Err(Error::Validation(msg)) = result {
            assert!(msg.contains("Usage record not found"));
        } else {
            panic!("Expected validation error");
        }
        let consent_manager = Arc::new(Mutex::new(crate::consent::ConsentManager::new()));
        let tracker_with_consent = {
            let mut t = UsageTracker::new(UsageTrackingConfig::default());
            t.set_consent_manager(consent_manager);
            t
        };
        let fake_consent_id = Uuid::new_v4();
        let result = tracker_with_consent.check_consent(
            fake_usage_id,
            Some(fake_consent_id),
            "test_use_case",
        );
        assert!(result.is_err());
    }
    #[test]
    fn test_session_management() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);
        let session_id = "test-session-123";
        for i in 0..3 {
            let user_context = UserContext {
                user_id: Some("session-user".to_string()),
                application_id: "session-app".to_string(),
                application_version: "1.0.0".to_string(),
                client_type: ClientType::WebBrowser,
                session_id: Some(session_id.to_string()),
                request_id: Some(format!("req-{i}")),
                auth_method: Some(AuthenticationMethod::OAuth2),
                user_agent: Some("Test User Agent".to_string()),
            };
            let operation = CloningOperation {
                operation_type: CloningOperationType::VoiceSynthesis,
                speaker_id: Some(format!("report_speaker_{i}")),
                target_speaker_id: None,
                request_metadata: OperationRequestMetadata {
                    request_id: format!("report_req_{i}"),
                    timestamp: SystemTime::now(),
                    priority: Priority::Normal,
                    source_application: "usage_report".to_string(),
                    user_preferences: UserPreferences::default(),
                },
                input_data: InputDataInfo {
                    data_type: InputDataType::TextPrompt,
                    data_size_bytes: 500 + (i * 100),
                    audio_duration_seconds: None,
                    text_length: Some((50 + (i * 10)) as usize),
                    language: Some("en".to_string()),
                    content_hash: None,
                    input_quality_score: None,
                },
                processing_params: ProcessingParameters {
                    quality_level: QualityLevel::Standard,
                    processing_mode: ProcessingMode::Balanced,
                    model_config: ModelConfiguration {
                        model_name: "session-model".to_string(),
                        model_version: "1.0".to_string(),
                        model_type: ModelType::Acoustic,
                        model_size_mb: Some(100.0),
                        training_data_info: None,
                    },
                    advanced_params: HashMap::new(),
                },
                output_data: OutputDataInfo {
                    output_type: OutputDataType::SynthesizedAudio,
                    data_size_bytes: 0,
                    audio_duration_seconds: None,
                    quality_score: None,
                    similarity_score: None,
                    format: Some("wav".to_string()),
                    sample_rate: Some(22050),
                },
                pipeline_info: PipelineInfo {
                    pipeline_id: "session-pipeline".to_string(),
                    pipeline_version: "1.0".to_string(),
                    components_used: vec!["vocoder".to_string()],
                    processing_stages: Vec::new(),
                },
            };
            let _usage_id = tracker.start_tracking(user_context, operation).unwrap();
        }
        let sessions = tracker
            .active_sessions
            .read()
            .expect("lock should not be poisoned");
        assert!(sessions.contains_key(session_id));
        let session = sessions.get(session_id).unwrap();
        assert_eq!(session.request_count, 3);
        assert_eq!(session.current_operations.len(), 3);
        assert_eq!(session.user_id, Some("session-user".to_string()));
    }
    #[test]
    fn test_statistics_tracking() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);
        let initial_stats = tracker.get_statistics();
        assert_eq!(initial_stats.total_operations, 0);
        assert_eq!(initial_stats.successful_operations, 0);
        assert_eq!(initial_stats.failed_operations, 0);
        let user_context = UserContext {
            user_id: Some("stats-user".to_string()),
            application_id: "stats-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::API,
            session_id: None,
            request_id: None,
            auth_method: None,
            user_agent: None,
        };
        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceSynthesis,
            speaker_id: Some("test_speaker".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: format!("req-{}", uuid::Uuid::new_v4()),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_tracker_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
            input_data: InputDataInfo {
                data_type: InputDataType::TextPrompt,
                data_size_bytes: 1000,
                audio_duration_seconds: None,
                text_length: Some(100),
                language: Some("en".to_string()),
                content_hash: None,
                input_quality_score: None,
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::High,
                processing_mode: ProcessingMode::HighQuality,
                model_config: ModelConfiguration {
                    model_name: "stats-model".to_string(),
                    model_version: "1.0".to_string(),
                    model_type: ModelType::Acoustic,
                    model_size_mb: Some(200.0),
                    training_data_info: None,
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::SynthesizedAudio,
                data_size_bytes: 0,
                audio_duration_seconds: None,
                quality_score: None,
                similarity_score: None,
                format: Some("wav".to_string()),
                sample_rate: Some(44100),
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "stats-pipeline".to_string(),
                pipeline_version: "1.0".to_string(),
                components_used: vec!["preprocessing".to_string(), "vocoder".to_string()],
                processing_stages: Vec::new(),
            },
        };
        let usage_id = tracker.start_tracking(user_context, operation).unwrap();
        let outcome = UsageOutcome {
            status: UsageStatus::Success,
            error: None,
            compliance_status: ComplianceStatus {
                is_compliant: true,
                compliance_checks: vec![ComplianceCheck {
                    check_name: "consent".to_string(),
                    check_result: ComplianceCheckResult::Pass,
                    check_details: "Consent verified successfully".to_string(),
                }],
                violations: Vec::new(),
                risk_level: RiskLevel::Low,
            },
            consent_result: None,
            restrictions_applied: Vec::new(),
            warnings: Vec::new(),
        };
        let resources = ResourceUsage {
            total_processing_time_ms: 5000,
            cpu_usage: CpuUsage {
                peak_cpu_percent: 70.0,
                average_cpu_percent: 50.0,
                cpu_time_seconds: 4.5,
                cpu_cores_used: 2,
            },
            memory_usage: MemoryUsage {
                peak_memory_mb: 1024.0,
                average_memory_mb: 800.0,
                memory_allocated_mb: 1024.0,
                memory_freed_mb: 1024.0,
            },
            gpu_usage: None,
            network_usage: NetworkUsage::default(),
            storage_usage: StorageUsage::default(),
            cost_estimate: Some(CostBreakdown {
                compute_cost: 0.25,
                storage_cost: 0.05,
                network_cost: 0.02,
                total_cost: 0.32,
                currency: "USD".to_string(),
            }),
        };
        tracker
            .complete_tracking(usage_id, outcome, resources, None)
            .unwrap();
        let updated_stats = tracker.get_statistics();
        assert_eq!(updated_stats.total_operations, 1);
        assert_eq!(updated_stats.successful_operations, 1);
        assert_eq!(updated_stats.failed_operations, 0);
        assert_eq!(updated_stats.total_resource_cost, 0.32);
    }
    #[test]
    fn test_disabled_tracking() {
        let config = UsageTrackingConfig {
            enable_tracking: false,
            ..Default::default()
        };
        let tracker = UsageTracker::new(config);
        let user_context = UserContext {
            user_id: Some("disabled-user".to_string()),
            application_id: "disabled-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::API,
            session_id: None,
            request_id: None,
            auth_method: None,
            user_agent: None,
        };
        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceSynthesis,
            speaker_id: Some("test_speaker".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: format!("req-{}", uuid::Uuid::new_v4()),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_tracker_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
            input_data: InputDataInfo {
                data_type: InputDataType::TextPrompt,
                data_size_bytes: 500,
                audio_duration_seconds: None,
                text_length: Some(50),
                language: Some("en".to_string()),
                content_hash: None,
                input_quality_score: None,
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::Standard,
                processing_mode: ProcessingMode::Balanced,
                model_config: ModelConfiguration {
                    model_name: "disabled-model".to_string(),
                    model_version: "1.0".to_string(),
                    model_type: ModelType::Acoustic,
                    model_size_mb: Some(100.0),
                    training_data_info: None,
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::SynthesizedAudio,
                data_size_bytes: 0,
                audio_duration_seconds: None,
                quality_score: None,
                similarity_score: None,
                format: Some("wav".to_string()),
                sample_rate: Some(22050),
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "disabled-pipeline".to_string(),
                pipeline_version: "1.0".to_string(),
                components_used: vec!["vocoder".to_string()],
                processing_stages: Vec::new(),
            },
        };
        let usage_id = tracker.start_tracking(user_context, operation).unwrap();
        assert!(!usage_id.is_nil());
        let stats = tracker.get_statistics();
        assert_eq!(stats.total_operations, 0);
    }
    #[test]
    fn test_usage_completion() {
        let config = UsageTrackingConfig::default();
        let tracker = UsageTracker::new(config);
        let user_context = UserContext {
            user_id: Some("test-user-002".to_string()),
            application_id: "test-app".to_string(),
            application_version: "1.0.0".to_string(),
            client_type: ClientType::DesktopApp,
            session_id: None,
            request_id: None,
            auth_method: None,
            user_agent: None,
        };
        let operation = CloningOperation {
            operation_type: CloningOperationType::VoiceTraining,
            input_data: InputDataInfo {
                data_type: InputDataType::AudioFile,
                data_size_bytes: 1024000,
                audio_duration_seconds: Some(30.0),
                text_length: None,
                language: Some("en".to_string()),
                content_hash: Some("hash123".to_string()),
                input_quality_score: Some(0.85),
            },
            processing_params: ProcessingParameters {
                quality_level: QualityLevel::High,
                processing_mode: ProcessingMode::HighQuality,
                model_config: ModelConfiguration {
                    model_name: "advanced-model".to_string(),
                    model_version: "2.0".to_string(),
                    model_type: ModelType::Hybrid,
                    model_size_mb: Some(500.0),
                    training_data_info: Some("high-quality dataset".to_string()),
                },
                advanced_params: HashMap::new(),
            },
            output_data: OutputDataInfo {
                output_type: OutputDataType::VoiceModel,
                data_size_bytes: 50000000,
                audio_duration_seconds: None,
                quality_score: Some(0.92),
                similarity_score: Some(0.88),
                format: Some("model".to_string()),
                sample_rate: None,
            },
            pipeline_info: PipelineInfo {
                pipeline_id: "advanced-pipeline".to_string(),
                pipeline_version: "2.0".to_string(),
                components_used: vec!["preprocessing".to_string(), "training".to_string()],
                processing_stages: Vec::new(),
            },
            speaker_id: Some("test_speaker_advanced".to_string()),
            target_speaker_id: None,
            request_metadata: OperationRequestMetadata {
                request_id: "advanced_test_req".to_string(),
                timestamp: SystemTime::now(),
                priority: Priority::Normal,
                source_application: "usage_test".to_string(),
                user_preferences: UserPreferences::default(),
            },
        };
        let usage_id = tracker.start_tracking(user_context, operation).unwrap();
        let outcome = UsageOutcome {
            status: UsageStatus::Success,
            error: None,
            compliance_status: ComplianceStatus {
                is_compliant: true,
                compliance_checks: Vec::new(),
                violations: Vec::new(),
                risk_level: RiskLevel::Low,
            },
            consent_result: None,
            restrictions_applied: Vec::new(),
            warnings: Vec::new(),
        };
        let resources = ResourceUsage {
            total_processing_time_ms: 30000,
            cpu_usage: CpuUsage {
                peak_cpu_percent: 85.0,
                average_cpu_percent: 60.0,
                cpu_time_seconds: 25.0,
                cpu_cores_used: 4,
            },
            memory_usage: MemoryUsage {
                peak_memory_mb: 2048.0,
                average_memory_mb: 1500.0,
                memory_allocated_mb: 2048.0,
                memory_freed_mb: 2048.0,
            },
            gpu_usage: Some(GpuUsage {
                gpu_time_seconds: 20.0,
                peak_gpu_memory_mb: 4096.0,
                gpu_utilization_percent: 75.0,
                gpu_device_name: "RTX 4090".to_string(),
            }),
            network_usage: NetworkUsage::default(),
            storage_usage: StorageUsage::default(),
            cost_estimate: Some(CostBreakdown {
                compute_cost: 0.50,
                storage_cost: 0.10,
                network_cost: 0.05,
                total_cost: 0.65,
                currency: "USD".to_string(),
            }),
        };
        tracker
            .complete_tracking(usage_id, outcome, resources, None)
            .unwrap();
        let stats = tracker.get_statistics();
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.successful_operations, 1);
    }
}
