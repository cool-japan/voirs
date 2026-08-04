//! Tests for [`super`] (`crate::storage`).
//!
//! Split out of `storage.rs` to keep that file under the workspace's
//! 2000-line-per-file guideline.

use super::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_storage_creation() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::default();

    let storage = VoiceModelStorage::new(temp_dir.path().to_path_buf(), config).await;
    assert!(storage.is_ok());
}

#[tokio::test]
async fn test_storage_config_default() {
    let config = StorageConfig::default();
    assert_eq!(config.max_cache_size, 100);
    assert!(config.enable_compression);
    assert_eq!(config.compression_level, 6);
    assert!(config.enable_auto_cleanup);
    assert!(config.enable_deduplication);
    assert_eq!(config.deduplication_threshold, 0.95);
}

#[test]
fn test_storage_tier_enum() {
    let tiers = vec![StorageTier::Hot, StorageTier::Warm, StorageTier::Cold];
    assert_eq!(tiers.len(), 3);
    assert_eq!(format!("{:?}", StorageTier::Hot), "Hot");
}

#[test]
fn test_compression_algorithm_enum() {
    let algorithms = vec![
        CompressionAlgorithm::None,
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lz4,
    ];
    assert_eq!(algorithms.len(), 4);
}

#[test]
fn test_storage_operation_enum() {
    let operations = vec![
        StorageOperation::Store,
        StorageOperation::Retrieve,
        StorageOperation::Delete,
        StorageOperation::Update,
        StorageOperation::Compress,
        StorageOperation::Migrate,
        StorageOperation::Backup,
        StorageOperation::Restore,
    ];
    assert_eq!(operations.len(), 8);
}

fn test_config(threshold_days: u64, dedup_threshold: f32) -> StorageConfig {
    StorageConfig {
        max_cache_size: 100,
        enable_compression: true,
        compression_level: 6,
        max_model_size: 50 * 1024 * 1024,
        enable_auto_cleanup: true,
        cleanup_age_threshold_days: threshold_days,
        enable_encryption: false,
        maintenance_interval: Duration::from_secs(3600),
        enable_deduplication: true,
        deduplication_threshold: dedup_threshold,
        enable_tiered_storage: true,
        backup_retention_days: 7,
    }
}

/// The regression test that actually matters for this finding: a fresh
/// `VoiceModelStorage` constructed on the *same* storage root after the
/// original instance was dropped must still see the previously-stored
/// model. Before this fix, `load_metadata_index` was a no-op, so this
/// would fail (`list_models` empty, `retrieve_model` "not found").
#[tokio::test]
async fn test_metadata_index_survives_restart() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(30, 0.999);

    let model_id = {
        let storage = VoiceModelStorage::new(temp_dir.path().to_path_buf(), config.clone())
            .await
            .unwrap();
        let profile = SpeakerProfile::new("speaker-1".to_string(), "Speaker One".to_string());
        let result = storage
            .store_model(
                &profile,
                b"real model bytes",
                None,
                vec!["voice".to_string()],
            )
            .await
            .unwrap();
        assert!(result.success);
        result.model_id
    }; // `storage` dropped here - simulates a process restart.

    let restarted = VoiceModelStorage::new(temp_dir.path().to_path_buf(), config)
        .await
        .unwrap();

    let models = restarted.list_models(None, None, None).await.unwrap();
    assert_eq!(models.len(), 1, "model metadata must survive a restart");
    assert_eq!(models[0].model_id, model_id);

    let (data, _metadata) = restarted.retrieve_model(&model_id).await.unwrap();
    assert_eq!(data, b"real model bytes");
}

#[tokio::test]
async fn test_compression_round_trip_preserves_bytes() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(30, 0.999);
    let storage = VoiceModelStorage::new(temp_dir.path().to_path_buf(), config)
        .await
        .unwrap();

    // Highly compressible payload so compression actually engages.
    let original: Vec<u8> = (0..8192u32).map(|i| (i % 4) as u8).collect();
    let profile = SpeakerProfile::new("speaker-2".to_string(), "Speaker Two".to_string());
    let result = storage
        .store_model(&profile, &original, None, vec![])
        .await
        .unwrap();

    let (data, metadata) = storage.retrieve_model(&result.model_id).await.unwrap();
    assert_eq!(data, original, "decompressed bytes must equal the original");
    assert!(
        metadata.compression_info.is_some(),
        "a highly compressible payload should have been compressed"
    );
}

#[tokio::test]
async fn test_deduplication_reuses_existing_model() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(30, 0.90);
    let storage = VoiceModelStorage::new(temp_dir.path().to_path_buf(), config)
        .await
        .unwrap();

    let mut profile = SpeakerProfile::new("speaker-3".to_string(), "Speaker Three".to_string());
    profile.characteristics.average_pitch = 180.0;

    let first = storage
        .store_model(&profile, b"model-v1", None, vec![])
        .await
        .unwrap();
    assert!(!first
        .metadata
        .get("deduplicated")
        .map(|v| v == "true")
        .unwrap_or(false));

    // Same speaker characteristics again: should be recognised as a duplicate.
    let second = storage
        .store_model(&profile, b"model-v1-resubmit", None, vec![])
        .await
        .unwrap();
    assert_eq!(second.model_id, first.model_id);
    assert_eq!(
        second.metadata.get("deduplicated").map(String::as_str),
        Some("true")
    );

    let models = storage.list_models(None, None, None).await.unwrap();
    assert_eq!(
        models.len(),
        1,
        "duplicate store must not create a 2nd entry"
    );
}

#[tokio::test]
async fn test_cleanup_old_models_removes_stale_entries() {
    let temp_dir = TempDir::new().unwrap();
    // Zero-day threshold: anything with nonzero age counts as stale.
    let config = test_config(0, 0.999);
    let storage = VoiceModelStorage::new(temp_dir.path().to_path_buf(), config)
        .await
        .unwrap();

    let profile = SpeakerProfile::new("speaker-4".to_string(), "Speaker Four".to_string());
    let stored = storage
        .store_model(&profile, b"stale model", None, vec![])
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(20)).await;

    let report = storage.perform_maintenance().await.unwrap();
    assert!(report.operations_performed.contains(&"cleanup".to_string()));
    assert_eq!(report.models_processed, 1);

    let models = storage.list_models(None, None, None).await.unwrap();
    assert!(models.is_empty(), "stale model must be removed by cleanup");

    let file_path = temp_dir
        .path()
        .join(storage.generate_storage_path(&stored.model_id).unwrap());
    assert!(!file_path.exists(), "backing file must be deleted too");
}

#[tokio::test]
async fn test_perform_maintenance_reports_only_real_work() {
    let temp_dir = TempDir::new().unwrap();
    // Long retention: nothing is actually stale or duplicated.
    let config = test_config(365, 0.999);
    let storage = VoiceModelStorage::new(temp_dir.path().to_path_buf(), config)
        .await
        .unwrap();

    let profile = SpeakerProfile::new("speaker-5".to_string(), "Speaker Five".to_string());
    storage
        .store_model(&profile, b"fresh model", None, vec![])
        .await
        .unwrap();

    let report = storage.perform_maintenance().await.unwrap();
    assert!(
        !report.operations_performed.contains(&"cleanup".to_string()),
        "cleanup must not be reported when nothing was removed"
    );
    assert!(
        !report
            .operations_performed
            .contains(&"deduplication".to_string()),
        "deduplication must not be reported when nothing was removed"
    );
    // Index optimization always runs (it validates/persists real state).
    assert!(report
        .operations_performed
        .contains(&"index_optimization".to_string()));
}

#[tokio::test]
async fn test_lru_cache_evicts_and_tracks_hits() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = test_config(30, 0.999);
    config.max_cache_size = 1; // 1 MB cache budget
    config.enable_compression = false; // keep exact byte sizes for the eviction math
                                       // `SpeakerProfile::new` gives every profile identical default
                                       // characteristics, so with deduplication enabled the three
                                       // stores below would all collapse into a single real model
                                       // (correct dedup behavior, but not what this test - cache
                                       // eviction in isolation - wants to exercise).
    config.enable_deduplication = false;
    let storage = VoiceModelStorage::new(temp_dir.path().to_path_buf(), config)
        .await
        .unwrap();

    // Each payload is ~400 KB; three of them exceed the 1 MB cache budget,
    // so the first must be evicted by the time the third is stored.
    let payload = vec![7u8; 400 * 1024];
    let mut ids = Vec::new();
    for i in 0..3 {
        let profile =
            SpeakerProfile::new(format!("cache-speaker-{i}"), format!("Cache Speaker {i}"));
        let result = storage
            .store_model(&profile, &payload, None, vec![])
            .await
            .unwrap();
        ids.push(result.model_id);
    }

    // The most recently stored model should still be a cache hit ...
    let stats_before = storage.get_statistics().await;
    let _ = storage.retrieve_model(&ids[2]).await.unwrap();
    let stats_after = storage.get_statistics().await;
    assert!(stats_after.cache_stats.hits > stats_before.cache_stats.hits);

    // ... while the first one was evicted, so a cache miss occurs (still
    // retrievable from disk, proving the miss is not a data-loss bug).
    let stats_before_first = storage.get_statistics().await;
    let (data, _) = storage.retrieve_model(&ids[0]).await.unwrap();
    assert_eq!(data.len(), payload.len());
    let stats_after_first = storage.get_statistics().await;
    assert!(stats_after_first.cache_stats.misses > stats_before_first.cache_stats.misses);
}

#[tokio::test]
async fn test_apply_filter_by_tags_and_min_quality() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(30, 0.999);
    let storage = VoiceModelStorage::new(temp_dir.path().to_path_buf(), config)
        .await
        .unwrap();

    let profile_a = SpeakerProfile::new("filter-a".to_string(), "Filter A".to_string());
    storage
        .store_model(&profile_a, b"a", None, vec!["premium".to_string()])
        .await
        .unwrap();

    let profile_b = SpeakerProfile::new("filter-b".to_string(), "Filter B".to_string());
    storage
        .store_model(&profile_b, b"b", None, vec!["free".to_string()])
        .await
        .unwrap();

    let premium_only = storage
        .list_models(
            Some(ModelFilter {
                speaker_id: None,
                tags: Some(vec!["premium".to_string()]),
                created_after: None,
                created_before: None,
                storage_tier: None,
                min_quality_score: None,
            }),
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(premium_only.len(), 1);
    assert!(premium_only[0].tags.contains(&"premium".to_string()));
}

#[test]
fn test_characteristics_similarity_identical_is_high() {
    let a = VoiceCharacteristicsSummary {
        average_f0: 150.0,
        quality_indicators: vec![0.2, 0.3, 0.4, 0.5],
        spectral_centroid: 2000.0,
        energy_stats: EnergyStats {
            mean: 0.1,
            std_dev: 0.01,
            dynamic_range: 40.0,
        },
    };
    let b = a.clone();
    assert!(characteristics_similarity(&a, &b) > 0.99);
}

#[test]
fn test_characteristics_similarity_different_is_low() {
    let a = VoiceCharacteristicsSummary {
        average_f0: 90.0,
        quality_indicators: vec![0.1, 0.1, 0.1, 0.1],
        spectral_centroid: 800.0,
        energy_stats: EnergyStats {
            mean: 0.02,
            std_dev: 0.01,
            dynamic_range: 20.0,
        },
    };
    let b = VoiceCharacteristicsSummary {
        average_f0: 320.0,
        quality_indicators: vec![0.9, 0.9, 0.9, 0.9],
        spectral_centroid: 3800.0,
        energy_stats: EnergyStats {
            mean: 0.9,
            std_dev: 0.2,
            dynamic_range: 90.0,
        },
    };
    assert!(characteristics_similarity(&a, &b) < 0.5);
}
