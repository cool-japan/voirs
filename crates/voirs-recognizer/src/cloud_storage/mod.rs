//! # Cloud Storage Integration
//!
//! Provides unified cloud storage integration for model management across
//! AWS S3, Google Cloud Storage, and Azure Blob Storage.
//!
//! Features:
//! - Multi-cloud model storage and retrieval
//! - Automatic caching and version management
//! - Parallel download optimization
//! - Checksum verification
//! - Retry logic with exponential backoff

/// Public types for cloud storage (errors, configs, metadata, statistics).
pub mod types;
pub use types::*;

mod manager;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    #[test]
    fn test_cloud_storage_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudStorageConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_local_filesystem_download() {
        let temp_cache = TempDir::new().unwrap();
        let temp_bucket = TempDir::new().unwrap();

        // Create a test model file
        let model_path = temp_bucket.path().join("test_model.bin");
        std::fs::write(&model_path, b"test model data").unwrap();

        let config = CloudStorageConfig {
            provider: CloudProvider::LocalFilesystem,
            bucket_name: temp_bucket.path().to_string_lossy().to_string(),
            cache_dir: temp_cache.path().to_path_buf(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        let result = manager.download_model("test_model.bin").await;
        assert!(result.is_ok());

        let downloaded_path = result.unwrap();
        assert!(downloaded_path.exists());
        assert_eq!(
            std::fs::read_to_string(downloaded_path).unwrap(),
            "test model data"
        );
    }

    #[tokio::test]
    async fn test_local_filesystem_upload() {
        let temp_cache = TempDir::new().unwrap();
        let temp_bucket = TempDir::new().unwrap();

        // Create source model file
        let source_model = temp_cache.path().join("source_model.bin");
        std::fs::write(&source_model, b"upload test data").unwrap();

        let config = CloudStorageConfig {
            provider: CloudProvider::LocalFilesystem,
            bucket_name: temp_bucket.path().to_string_lossy().to_string(),
            cache_dir: temp_cache.path().to_path_buf(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        let metadata = ModelMetadata {
            name: "test_upload.bin".to_string(),
            version: "1.0.0".to_string(),
            size_bytes: 16,
            checksum: "test_checksum".to_string(),
            last_modified: chrono::Utc::now(),
            model_type: "test".to_string(),
            storage_path: "test_upload.bin".to_string(),
            tags: std::collections::HashMap::new(),
        };

        let result = manager
            .upload_model(&source_model, "test_upload.bin", metadata)
            .await;
        assert!(result.is_ok());

        // Verify uploaded file
        let uploaded_path = temp_bucket.path().join("test_upload.bin");
        assert!(uploaded_path.exists());
    }

    #[tokio::test]
    async fn test_list_local_models() {
        let temp_bucket = TempDir::new().unwrap();

        // Create test model files
        std::fs::write(temp_bucket.path().join("model1.bin"), b"model1").unwrap();
        std::fs::write(temp_bucket.path().join("model2.bin"), b"model2").unwrap();

        let config = CloudStorageConfig {
            provider: CloudProvider::LocalFilesystem,
            bucket_name: temp_bucket.path().to_string_lossy().to_string(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        let models = manager.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_download_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloudStorageConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        manager.update_download_stats(true, Duration::from_secs(10));
        manager.update_download_stats(true, Duration::from_secs(5));
        manager.update_download_stats(false, Duration::from_secs(1));

        let stats = manager.get_download_stats();
        assert_eq!(stats.total_downloads, 3);
        assert_eq!(stats.successful_downloads, 2);
        assert_eq!(stats.failed_downloads, 1);
    }

    #[test]
    fn test_cache_cleanup() {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        let file1 = temp_dir.path().join("file1.bin");
        let file2 = temp_dir.path().join("file2.bin");

        std::fs::write(&file1, vec![0u8; 1024 * 1024]).unwrap(); // 1MB
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(&file2, vec![0u8; 1024 * 1024]).unwrap(); // 1MB

        let config = CloudStorageConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            max_cache_size_mb: 1, // 1MB limit
            ..CloudStorageConfig::default()
        };

        let manager = CloudStorageManager::new(config).unwrap();

        // Cleanup should remove the oldest file
        manager.cleanup_cache().unwrap();

        // file1 should be removed, file2 should exist
        assert!(!file1.exists());
        assert!(file2.exists());
    }
}

#[cfg(all(test, feature = "cloud"))]
mod cloud_auth_tests {
    use super::*;

    #[test]
    fn test_sigv4_canonical_request() {
        let result = crate::cloud_auth::sign_s3_request(
            "GET",
            "https://examplebucket.s3.amazonaws.com/?list-type=2",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "us-east-1",
            "s3",
            "20130524T000000Z",
        );
        assert!(
            result.is_ok(),
            "SigV4 signing should succeed: {:?}",
            result.err()
        );
        let headers = result.unwrap();
        assert!(
            headers.contains_key("Authorization"),
            "Should have Authorization header"
        );
        let auth = &headers["Authorization"];
        assert!(
            auth.starts_with("AWS4-HMAC-SHA256"),
            "Should use AWS4-HMAC-SHA256"
        );
        assert!(
            auth.contains("Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request"),
            "Credential scope must match: {}",
            auth
        );
        assert!(
            auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"),
            "Signed headers must be canonical: {}",
            auth
        );
    }

    #[test]
    fn test_azure_shared_key_signing() {
        let result = crate::cloud_auth::sign_azure_request(
            "GET",
            "devstoreaccount1",
            "models",
            "",
            0,
            "",
            "Thu, 19 Jun 2025 12:00:00 GMT",
            "2020-10-02",
            "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==",
        );
        assert!(
            result.is_ok(),
            "Azure signing should succeed: {:?}",
            result.err()
        );
        let auth = result.unwrap();
        assert!(
            auth.starts_with("SharedKey devstoreaccount1:"),
            "Should use SharedKey scheme: {}",
            auth
        );
    }
}

#[cfg(test)]
mod checksum_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_verify_checksum_correct() {
        use sha2::{Digest, Sha256};
        let temp_dir = std::env::temp_dir().join("voirs_checksum_test");
        std::fs::create_dir_all(&temp_dir).ok();
        let file_path = temp_dir.join("test_model.bin");
        let test_data = b"Hello, VoiRS checksum test!";
        std::fs::write(&file_path, test_data).expect("write test file");

        let mut hasher = Sha256::new();
        hasher.update(test_data);
        let expected = hex::encode(hasher.finalize());

        let config = CloudStorageConfig {
            cache_dir: temp_dir.clone(),
            ..CloudStorageConfig::default()
        };
        let manager = CloudStorageManager::new(config).expect("create manager");
        manager.metadata_cache.write().insert(
            "test_model.bin".to_string(),
            ModelMetadata {
                name: "test_model.bin".to_string(),
                version: "1.0.0".to_string(),
                size_bytes: test_data.len() as u64,
                checksum: expected.clone(),
                last_modified: chrono::Utc::now(),
                model_type: "test".to_string(),
                storage_path: file_path.to_string_lossy().to_string(),
                tags: HashMap::new(),
            },
        );

        let result = manager.verify_checksum(&file_path, "test_model.bin");
        assert!(
            result.is_ok(),
            "Correct checksum should pass: {:?}",
            result.err()
        );
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        let temp_dir = std::env::temp_dir().join("voirs_checksum_mismatch_test");
        std::fs::create_dir_all(&temp_dir).ok();
        let file_path = temp_dir.join("bad_model.bin");
        std::fs::write(&file_path, b"tampered data").expect("write file");

        let config = CloudStorageConfig {
            cache_dir: temp_dir.clone(),
            ..CloudStorageConfig::default()
        };
        let manager = CloudStorageManager::new(config).expect("create manager");
        manager.metadata_cache.write().insert(
            "bad_model.bin".to_string(),
            ModelMetadata {
                name: "bad_model.bin".to_string(),
                version: "1.0.0".to_string(),
                size_bytes: 13,
                checksum: "0000000000000000000000000000000000000000000000000000000000000000"
                    .to_string(),
                last_modified: chrono::Utc::now(),
                model_type: "test".to_string(),
                storage_path: file_path.to_string_lossy().to_string(),
                tags: HashMap::new(),
            },
        );

        let result = manager.verify_checksum(&file_path, "bad_model.bin");
        assert!(
            matches!(result, Err(CloudStorageError::ChecksumMismatch { .. })),
            "Mismatched checksum should return ChecksumMismatch error, got: {:?}",
            result
        );
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    #[ignore = "Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, TEST_BUCKET, TEST_AWS_REGION env vars"]
    async fn test_s3_roundtrip() {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID must be set");
        let secret_key =
            std::env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY must be set");
        let bucket = std::env::var("TEST_BUCKET").expect("TEST_BUCKET must be set");
        let region = std::env::var("TEST_AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

        let temp_dir = std::env::temp_dir().join("voirs_s3_roundtrip");
        std::fs::create_dir_all(&temp_dir).ok();
        let test_model = temp_dir.join("s3_test_model.bin");
        std::fs::write(&test_model, b"VoiRS S3 roundtrip test data").expect("write test model");

        let config = CloudStorageConfig {
            provider: CloudProvider::AwsS3,
            bucket_name: bucket,
            region: Some(region),
            access_key_id: Some(access_key),
            secret_access_key: Some(secret_key),
            cache_dir: temp_dir.clone(),
            ..CloudStorageConfig::default()
        };
        let manager = CloudStorageManager::new(config).expect("create manager");

        let meta = ModelMetadata {
            name: "s3_test_model.bin".to_string(),
            version: "1.0.0".to_string(),
            size_bytes: 28,
            checksum: String::new(),
            last_modified: chrono::Utc::now(),
            model_type: "test".to_string(),
            storage_path: "s3_test_model.bin".to_string(),
            tags: HashMap::new(),
        };

        manager
            .upload_model(&test_model, "s3_test_model.bin", meta)
            .await
            .expect("upload should succeed");

        let models = manager.list_models().await.expect("list should succeed");
        assert!(
            models.iter().any(|m| m.name.contains("s3_test_model")),
            "Uploaded model should appear in listing"
        );

        std::fs::remove_file(&test_model).ok();
        let downloaded = manager
            .download_model("s3_test_model.bin")
            .await
            .expect("download should succeed");
        assert!(downloaded.exists(), "Downloaded file should exist");
        let content = std::fs::read(&downloaded).expect("read downloaded file");
        assert_eq!(content, b"VoiRS S3 roundtrip test data");

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
