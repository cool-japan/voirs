//! Security tests for VoiRS feedback system
//!
//! This module provides comprehensive security testing for user data protection,
//! encryption, authentication, and privacy features.

use std::collections::HashMap;
use voirs_feedback::persistence::encryption::{
    DataAnonymizer, ExportFormat, PrivacyConfig, PrivacyExportService, PrivacyLevel,
};
use voirs_feedback::persistence::{PersistenceError, UserDataExport};
use voirs_feedback::traits::{
    AdaptiveState, FeedbackResponse, FeedbackType, ProgressIndicators, SessionScores, SessionState,
    SessionStatistics, SessionStats, TrainingStatistics, UserFeedback, UserPreferences,
    UserProgress,
};

#[cfg(feature = "privacy")]
use voirs_feedback::persistence::encryption::{EncryptionService, PasswordHasher};

/// Test suite for data anonymization security
#[cfg(test)]
mod anonymization_tests {
    use super::*;

    #[test]
    fn test_user_id_anonymization_consistency() {
        let anonymizer = DataAnonymizer::new();
        let user_id = "user123@example.com";

        let anonymized1 = anonymizer.anonymize_user_id(user_id);
        let anonymized2 = anonymizer.anonymize_user_id(user_id);

        // Should be consistent
        assert_eq!(anonymized1, anonymized2);
        // Should be different from original
        assert_ne!(anonymized1, user_id);
        // Should start with anon_ prefix
        assert!(anonymized1.starts_with("anon_"));
    }

    #[test]
    fn test_session_anonymization_levels() {
        let anonymizer = DataAnonymizer::new();
        let mut session = create_test_session();

        // Test Full level - no changes
        let mut session_full = session.clone();
        anonymizer.anonymize_session(&mut session_full, PrivacyLevel::Full);
        assert_eq!(session_full.user_id, session.user_id);

        // Test Anonymized level
        let mut session_anon = session.clone();
        anonymizer.anonymize_session(&mut session_anon, PrivacyLevel::Anonymized);
        assert_ne!(session_anon.user_id, session.user_id);
        assert!(session_anon.user_id.starts_with("anon_"));

        // Test Public level
        let mut session_public = session.clone();
        anonymizer.anonymize_session(&mut session_public, PrivacyLevel::Public);
        assert_eq!(session_public.user_id, "public_user");

        // Test Minimal level
        let mut session_minimal = session.clone();
        anonymizer.anonymize_session(&mut session_minimal, PrivacyLevel::Minimal);
        assert_eq!(session_minimal.user_id, "user");
    }

    #[test]
    fn test_feedback_anonymization() {
        let anonymizer = DataAnonymizer::new();
        let mut feedback = create_test_feedback();

        // Add some metadata
        feedback.feedback_items[0]
            .metadata
            .insert("pii_data".to_string(), "sensitive_info".to_string());

        anonymizer.anonymize_feedback(&mut feedback, PrivacyLevel::Anonymized);

        // Metadata should be cleared
        assert!(feedback.feedback_items[0].metadata.is_empty());
    }

    #[test]
    fn test_progress_anonymization() {
        let anonymizer = DataAnonymizer::new();
        let mut progress = create_test_progress();

        // Test different privacy levels
        let mut progress_anon = progress.clone();
        anonymizer.anonymize_progress(&mut progress_anon, PrivacyLevel::Anonymized);
        assert_ne!(progress_anon.user_id, progress.user_id);
        assert!(progress_anon.user_id.starts_with("anon_"));

        let mut progress_public = progress.clone();
        anonymizer.anonymize_progress(&mut progress_public, PrivacyLevel::Public);
        assert_eq!(progress_public.user_id, "anonymous");
    }

    fn create_test_session() -> SessionState {
        SessionState {
            user_id: "test_user_123".to_string(),
            session_id: uuid::Uuid::new_v4(),
            start_time: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: AdaptiveState::default(),
            current_exercise: None,
            session_stats: SessionStatistics::default(),
        }
    }

    fn create_test_feedback() -> FeedbackResponse {
        FeedbackResponse {
            feedback_items: vec![UserFeedback {
                message: "Test feedback".to_string(),
                suggestion: Some("Continue practicing".to_string()),
                confidence: 0.85,
                score: 0.85,
                priority: 0.7,
                metadata: HashMap::new(),
            }],
            overall_score: 0.85,
            immediate_actions: vec!["Keep practicing".to_string()],
            long_term_goals: vec!["Improve pronunciation".to_string()],
            progress_indicators: ProgressIndicators::default(),
            timestamp: chrono::Utc::now(),
            processing_time: std::time::Duration::from_millis(100),
            feedback_type: FeedbackType::Quality,
        }
    }

    fn create_test_progress() -> UserProgress {
        UserProgress {
            user_id: "test_user_123".to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: vec![],
            achievements: vec![],
            training_stats: TrainingStatistics::default(),
            goals: vec![],
            last_updated: chrono::Utc::now(),
            average_scores: SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: vec![],
            personal_bests: HashMap::new(),
            session_count: 10,
            total_practice_time: std::time::Duration::from_secs(3600),
        }
    }
}

/// Test suite for encryption security
#[cfg(all(test, feature = "privacy"))]
mod encryption_tests {
    use super::*;

    #[test]
    fn test_encryption_service_basic() {
        let service = EncryptionService::new().unwrap();
        let data = b"Hello, World!";

        let encrypted = service.encrypt(data).unwrap();
        assert_ne!(encrypted, data);
        assert!(encrypted.len() > data.len()); // Should be longer due to nonce

        let decrypted = service.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_encryption_service_json() {
        let service = EncryptionService::new().unwrap();
        let data = serde_json::json!({
            "user_id": "test_user",
            "sensitive_data": "secret_value",
            "score": 95.5
        });

        let encrypted = service.encrypt_json(&data).unwrap();
        let decrypted: serde_json::Value = service.decrypt_json(&encrypted).unwrap();

        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_encryption_service_tampering_detection() {
        let service = EncryptionService::new().unwrap();
        let data = b"Important data";

        let mut encrypted = service.encrypt(data).unwrap();

        // Tamper with the encrypted data
        encrypted[5] ^= 0x01;

        // Decryption should fail
        let result = service.decrypt(&encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_encryption_service_short_data() {
        let service = EncryptionService::new().unwrap();
        let short_data = b"short";

        let encrypted = service.encrypt(short_data).unwrap();
        let decrypted = service.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, short_data);
    }

    #[test]
    fn test_encryption_service_empty_data() {
        let service = EncryptionService::new().unwrap();
        let empty_data = b"";

        let encrypted = service.encrypt(empty_data).unwrap();
        let decrypted = service.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, empty_data);
    }

    #[test]
    fn test_encryption_service_invalid_data() {
        let service = EncryptionService::new().unwrap();
        let invalid_data = b"too_short";

        let result = service.decrypt(invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_encryption_service_key_isolation() {
        let service1 = EncryptionService::new().unwrap();
        let service2 = EncryptionService::new().unwrap();
        let data = b"test data";

        let encrypted1 = service1.encrypt(data).unwrap();

        // Different service should not be able to decrypt
        let result = service2.decrypt(&encrypted1);
        assert!(result.is_err());
    }

    #[test]
    fn test_encryption_service_key_reuse() {
        let service1 = EncryptionService::new().unwrap();
        let key = *service1.get_key();
        let service2 = EncryptionService::with_key(key);
        let data = b"test data";

        let encrypted1 = service1.encrypt(data).unwrap();
        let decrypted2 = service2.decrypt(&encrypted1).unwrap();

        assert_eq!(decrypted2, data);
    }
}

/// Test suite for password security
#[cfg(all(test, feature = "privacy"))]
mod password_tests {
    use super::*;

    #[test]
    fn test_password_hashing_basic() {
        let password = "secure_password_123";
        let hash = PasswordHasher::hash_password(password).unwrap();

        assert_ne!(hash, password);
        assert!(hash.len() > password.len());
        assert!(PasswordHasher::verify_password(password, &hash).unwrap());
    }

    #[test]
    fn test_password_hashing_wrong_password() {
        let password = "correct_password";
        let hash = PasswordHasher::hash_password(password).unwrap();

        assert!(!PasswordHasher::verify_password("wrong_password", &hash).unwrap());
    }

    #[test]
    fn test_password_hashing_uniqueness() {
        let password = "same_password";
        let hash1 = PasswordHasher::hash_password(password).unwrap();
        let hash2 = PasswordHasher::hash_password(password).unwrap();

        // Hashes should be different due to salt
        assert_ne!(hash1, hash2);

        // But both should verify successfully
        assert!(PasswordHasher::verify_password(password, &hash1).unwrap());
        assert!(PasswordHasher::verify_password(password, &hash2).unwrap());
    }

    #[test]
    fn test_password_hashing_empty_password() {
        let password = "";
        let hash = PasswordHasher::hash_password(password).unwrap();

        assert!(PasswordHasher::verify_password(password, &hash).unwrap());
        assert!(!PasswordHasher::verify_password("not_empty", &hash).unwrap());
    }

    #[test]
    fn test_password_hashing_special_characters() {
        let password = "p@ssw0rd!@#$%^&*()";
        let hash = PasswordHasher::hash_password(password).unwrap();

        assert!(PasswordHasher::verify_password(password, &hash).unwrap());
    }

    #[test]
    fn test_password_hashing_unicode() {
        let password = "パスワード123";
        let hash = PasswordHasher::hash_password(password).unwrap();

        assert!(PasswordHasher::verify_password(password, &hash).unwrap());
    }

    #[test]
    fn test_password_hashing_long_password() {
        let password = "very_long_password_that_exceeds_normal_length_and_should_still_work_correctly_with_the_hashing_algorithm";
        let hash = PasswordHasher::hash_password(password).unwrap();

        assert!(PasswordHasher::verify_password(password, &hash).unwrap());
    }
}

/// Test suite for privacy configuration and export
#[cfg(test)]
mod privacy_tests {
    use super::*;

    #[test]
    fn test_privacy_config_default() {
        let config = PrivacyConfig::default();

        assert_eq!(config.default_privacy_level, PrivacyLevel::Anonymized);
        assert!(config.encrypt_at_rest);
        assert!(config.enable_anonymization);
        assert_eq!(config.data_retention_days, 365);
        assert!(!config.auto_delete_expired);
        assert!(config.allowed_export_formats.contains(&ExportFormat::Json));
        assert!(config.allowed_export_formats.contains(&ExportFormat::Csv));
    }

    #[test]
    fn test_privacy_export_service_json() {
        let config = PrivacyConfig::default();
        let service = PrivacyExportService::new(config);
        let export_data = create_test_export_data();

        let result =
            service.export_user_data(export_data, PrivacyLevel::Anonymized, ExportFormat::Json);

        assert!(result.is_ok());
        let exported = result.unwrap();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_slice(&exported).unwrap();
        assert!(parsed.is_object());
    }

    #[test]
    fn test_privacy_export_service_csv() {
        let config = PrivacyConfig::default();
        let service = PrivacyExportService::new(config);
        let export_data = create_test_export_data();

        let result = service.export_user_data(export_data, PrivacyLevel::Public, ExportFormat::Csv);

        assert!(result.is_ok());
        let exported = result.unwrap();
        let csv_content = String::from_utf8(exported).unwrap();

        // Should contain CSV header
        assert!(csv_content.contains("user_id,export_timestamp"));
    }

    #[test]
    fn test_privacy_export_service_forbidden_format() {
        let mut config = PrivacyConfig::default();
        config.allowed_export_formats = vec![ExportFormat::Json]; // Only JSON allowed

        let service = PrivacyExportService::new(config);
        let export_data = create_test_export_data();

        let result = service.export_user_data(
            export_data,
            PrivacyLevel::Full,
            ExportFormat::Csv, // Not allowed
        );

        assert!(result.is_err());
        match result {
            Err(PersistenceError::ConfigError { message }) => {
                assert!(message.contains("not allowed"));
            }
            Err(other_error) => {
                assert!(false, "Expected ConfigError but got: {:?}", other_error);
            }
            Ok(_) => {
                assert!(false, "Expected ConfigError but operation succeeded");
            }
        }
    }

    #[test]
    fn test_privacy_export_service_anonymization() {
        let config = PrivacyConfig::default();
        let service = PrivacyExportService::new(config);
        let export_data = create_test_export_data();

        let result =
            service.export_user_data(export_data, PrivacyLevel::Anonymized, ExportFormat::Json);

        assert!(result.is_ok());
        let exported = result.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&exported).unwrap();

        // User ID should be anonymized
        let user_id = parsed["user_id"].as_str().unwrap();
        assert!(user_id.starts_with("anon_"));
    }

    #[test]
    fn test_privacy_levels_hierarchy() {
        let levels = vec![
            PrivacyLevel::Full,
            PrivacyLevel::Anonymized,
            PrivacyLevel::Public,
            PrivacyLevel::Minimal,
        ];

        for level in levels {
            let config = PrivacyConfig::default();
            let service = PrivacyExportService::new(config);
            let export_data = create_test_export_data();

            let result = service.export_user_data(export_data, level, ExportFormat::Json);

            assert!(
                result.is_ok(),
                "Export should succeed for all privacy levels"
            );
        }
    }

    fn create_test_export_data() -> UserDataExport {
        UserDataExport {
            user_id: "test_user_123".to_string(),
            export_timestamp: chrono::Utc::now(),
            progress: UserProgress {
                user_id: "test_user_123".to_string(),
                overall_skill_level: 0.8,
                skill_breakdown: HashMap::new(),
                progress_history: vec![],
                achievements: vec![],
                training_stats: TrainingStatistics::default(),
                goals: vec![],
                last_updated: chrono::Utc::now(),
                average_scores: SessionScores::default(),
                skill_levels: HashMap::new(),
                recent_sessions: vec![],
                personal_bests: HashMap::new(),
                session_count: 5,
                total_practice_time: std::time::Duration::from_secs(1800),
            },
            sessions: vec![SessionState {
                user_id: "test_user_123".to_string(),
                session_id: uuid::Uuid::new_v4(),
                start_time: chrono::Utc::now(),
                last_activity: chrono::Utc::now(),
                current_task: None,
                stats: SessionStats::default(),
                preferences: UserPreferences::default(),
                adaptive_state: AdaptiveState::default(),
                current_exercise: None,
                session_stats: SessionStatistics::default(),
            }],
            feedback_history: vec![FeedbackResponse {
                feedback_items: vec![UserFeedback {
                    message: "Test feedback".to_string(),
                    suggestion: Some("Continue practicing".to_string()),
                    confidence: 0.8,
                    score: 0.8,
                    priority: 0.7,
                    metadata: HashMap::new(),
                }],
                overall_score: 0.8,
                immediate_actions: vec!["Keep practicing".to_string()],
                long_term_goals: vec!["Improve pronunciation".to_string()],
                progress_indicators: ProgressIndicators::default(),
                timestamp: chrono::Utc::now(),
                processing_time: std::time::Duration::from_millis(100),
                feedback_type: FeedbackType::Quality,
            }],
            metadata: HashMap::new(),
            preferences: UserPreferences::default(),
        }
    }
}

/// Test suite for data validation and security constraints
#[cfg(test)]
mod data_validation_tests {
    use super::*;

    #[test]
    fn test_privacy_level_serialization() {
        let levels = vec![
            PrivacyLevel::Full,
            PrivacyLevel::Anonymized,
            PrivacyLevel::Public,
            PrivacyLevel::Minimal,
        ];

        for level in levels {
            let serialized = serde_json::to_string(&level).unwrap();
            let deserialized: PrivacyLevel = serde_json::from_str(&serialized).unwrap();
            assert_eq!(level, deserialized);
        }
    }

    #[test]
    fn test_export_format_serialization() {
        let formats = vec![
            ExportFormat::Json,
            ExportFormat::Csv,
            ExportFormat::Xml,
            ExportFormat::Pdf,
        ];

        for format in formats {
            let serialized = serde_json::to_string(&format).unwrap();
            let deserialized: ExportFormat = serde_json::from_str(&serialized).unwrap();
            assert_eq!(format, deserialized);
        }
    }

    #[test]
    fn test_privacy_config_serialization() {
        let config = PrivacyConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: PrivacyConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            config.default_privacy_level,
            deserialized.default_privacy_level
        );
        assert_eq!(config.encrypt_at_rest, deserialized.encrypt_at_rest);
        assert_eq!(
            config.enable_anonymization,
            deserialized.enable_anonymization
        );
    }

    #[test]
    fn test_data_retention_validation() {
        let config = PrivacyConfig {
            data_retention_days: 0,
            ..Default::default()
        };

        // Should handle zero retention days
        assert_eq!(config.data_retention_days, 0);

        let config = PrivacyConfig {
            data_retention_days: u32::MAX,
            ..Default::default()
        };

        // Should handle maximum retention days
        assert_eq!(config.data_retention_days, u32::MAX);
    }

    #[test]
    fn test_user_id_validation() {
        let anonymizer = DataAnonymizer::new();

        // Test edge cases
        let empty_id = "";
        let long_id = "a".repeat(1000);
        let special_chars = "user!@#$%^&*()_+{}|:<>?[]\\;',./`~";

        // Should handle all cases without panic
        let _ = anonymizer.anonymize_user_id(empty_id);
        let _ = anonymizer.anonymize_user_id(&long_id);
        let _ = anonymizer.anonymize_user_id(special_chars);
    }
}

/// Test suite for security edge cases and error handling
#[cfg(test)]
mod security_edge_cases {
    use super::*;

    #[test]
    fn test_anonymization_consistency_across_instances() {
        let anonymizer1 = DataAnonymizer::new();
        let anonymizer2 = DataAnonymizer::new();
        let user_id = "test_user";

        let anon1 = anonymizer1.anonymize_user_id(user_id);
        let anon2 = anonymizer2.anonymize_user_id(user_id);

        // Should be consistent across instances
        assert_eq!(anon1, anon2);
    }

    #[test]
    fn test_multiple_anonymization_operations() {
        let anonymizer = DataAnonymizer::new();
        let mut progress = UserProgress {
            user_id: "original_user".to_string(),
            overall_skill_level: 0.8,
            skill_breakdown: HashMap::new(),
            progress_history: vec![],
            achievements: vec![],
            training_stats: TrainingStatistics::default(),
            goals: vec![],
            last_updated: chrono::Utc::now(),
            average_scores: SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: vec![],
            personal_bests: HashMap::new(),
            session_count: 10,
            total_practice_time: std::time::Duration::from_secs(3600),
        };

        // Apply multiple anonymization operations
        anonymizer.anonymize_progress(&mut progress, PrivacyLevel::Anonymized);
        let first_anon = progress.user_id.clone();

        anonymizer.anonymize_progress(&mut progress, PrivacyLevel::Anonymized);
        let second_anon = progress.user_id.clone();

        // Should be consistent
        assert_eq!(first_anon, second_anon);
    }

    #[test]
    fn test_privacy_level_transitions() {
        let anonymizer = DataAnonymizer::new();
        let mut session = SessionState {
            user_id: "test_user".to_string(),
            session_id: uuid::Uuid::new_v4(),
            start_time: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: AdaptiveState::default(),
            current_exercise: None,
            session_stats: SessionStatistics::default(),
        };

        // Test transitions between privacy levels
        anonymizer.anonymize_session(&mut session, PrivacyLevel::Full);
        assert_eq!(session.user_id, "test_user");

        anonymizer.anonymize_session(&mut session, PrivacyLevel::Anonymized);
        assert!(session.user_id.starts_with("anon_"));

        anonymizer.anonymize_session(&mut session, PrivacyLevel::Public);
        assert_eq!(session.user_id, "public_user");

        anonymizer.anonymize_session(&mut session, PrivacyLevel::Minimal);
        assert_eq!(session.user_id, "user");
    }

    #[cfg(feature = "privacy")]
    #[test]
    fn test_encryption_with_large_data() {
        let service = EncryptionService::new().unwrap();
        let large_data = vec![0u8; 1024 * 1024]; // 1MB of data

        let encrypted = service.encrypt(&large_data).unwrap();
        let decrypted = service.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, large_data);
    }

    #[test]
    fn test_privacy_export_with_empty_data() {
        let config = PrivacyConfig::default();
        let service = PrivacyExportService::new(config);
        let export_data = UserDataExport {
            user_id: "test_user".to_string(),
            export_timestamp: chrono::Utc::now(),
            progress: UserProgress {
                user_id: "test_user".to_string(),
                overall_skill_level: 0.0,
                skill_breakdown: HashMap::new(),
                progress_history: vec![],
                achievements: vec![],
                training_stats: TrainingStatistics::default(),
                goals: vec![],
                last_updated: chrono::Utc::now(),
                average_scores: SessionScores::default(),
                skill_levels: HashMap::new(),
                recent_sessions: vec![],
                personal_bests: HashMap::new(),
                session_count: 0,
                total_practice_time: std::time::Duration::from_secs(0),
            },
            sessions: vec![],
            feedback_history: vec![],
            metadata: HashMap::new(),
            preferences: UserPreferences::default(),
        };

        let result = service.export_user_data(export_data, PrivacyLevel::Full, ExportFormat::Json);

        assert!(result.is_ok());
    }
}

/// Test suite for concurrent access and thread safety
#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_anonymizer_thread_safety() {
        let anonymizer = Arc::new(DataAnonymizer::new());
        let user_id = "test_user_123";

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let anonymizer = Arc::clone(&anonymizer);
                let user_id = user_id.to_string();
                thread::spawn(move || anonymizer.anonymize_user_id(&user_id))
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All results should be the same
        let first_result = &results[0];
        for result in &results[1..] {
            assert_eq!(result, first_result);
        }
    }

    #[cfg(feature = "privacy")]
    #[test]
    fn test_encryption_thread_safety() {
        let service = Arc::new(EncryptionService::new().unwrap());
        let data = b"test data for encryption";

        let handles: Vec<_> = (0..5)
            .map(|_| {
                let service = Arc::clone(&service);
                thread::spawn(move || {
                    let encrypted = service.encrypt(data).unwrap();
                    let decrypted = service.decrypt(&encrypted).unwrap();
                    assert_eq!(decrypted, data);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_privacy_export_thread_safety() {
        let config = PrivacyConfig::default();
        let service = Arc::new(PrivacyExportService::new(config));

        let handles: Vec<_> = (0..3)
            .map(|i| {
                let service = Arc::clone(&service);
                thread::spawn(move || {
                    let export_data = UserDataExport {
                        user_id: format!("user_{}", i),
                        export_timestamp: chrono::Utc::now(),
                        progress: UserProgress {
                            user_id: format!("user_{}", i),
                            overall_skill_level: 0.8,
                            skill_breakdown: HashMap::new(),
                            progress_history: vec![],
                            achievements: vec![],
                            training_stats: TrainingStatistics::default(),
                            goals: vec![],
                            last_updated: chrono::Utc::now(),
                            average_scores: SessionScores::default(),
                            skill_levels: HashMap::new(),
                            recent_sessions: vec![],
                            personal_bests: HashMap::new(),
                            session_count: i,
                            total_practice_time: std::time::Duration::from_secs(i as u64 * 100),
                        },
                        sessions: vec![],
                        feedback_history: vec![],
                        metadata: HashMap::new(),
                        preferences: UserPreferences::default(),
                    };

                    service
                        .export_user_data(export_data, PrivacyLevel::Anonymized, ExportFormat::Json)
                        .unwrap();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
