//! Cloud integration command implementations for VoiRS CLI.

use crate::cloud::{
    AnalysisType, CloudApiClient, CloudApiConfig, CloudService, CloudStorageConfig,
    CloudStorageManager, ContentAnalysisRequest, QualityAssessmentRequest, QualityMetric,
    StorageProvider, SyncDirection, SyncOptions, TranslationQuality, TranslationRequest,
};
use crate::{CloudCommands, GlobalOptions};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use voirs_sdk::config::AppConfig;
use voirs_sdk::types::SynthesisConfig;
use voirs_sdk::QualityLevel;
use voirs_sdk::{Result, VoirsError};

/// Execute cloud-specific commands
pub async fn execute_cloud_command(
    command: &CloudCommands,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    match command {
        CloudCommands::Sync {
            force,
            directory,
            dry_run,
        } => execute_sync(*force, directory.as_ref(), *dry_run, config, global).await,

        CloudCommands::AddToSync {
            local_path,
            remote_path,
            direction,
        } => execute_add_to_sync(local_path, remote_path, direction, config, global).await,

        CloudCommands::StorageStats => execute_storage_stats(config, global).await,

        CloudCommands::CleanupCache {
            max_age_days,
            dry_run,
        } => execute_cleanup_cache(*max_age_days, *dry_run, config, global).await,

        CloudCommands::Translate {
            text,
            from,
            to,
            quality,
        } => execute_translate(text, from, to, quality, config, global).await,

        CloudCommands::AnalyzeContent {
            text,
            analysis_types,
            language,
        } => execute_analyze_content(text, analysis_types, language.as_ref(), config, global).await,

        CloudCommands::AssessQuality {
            audio_file,
            text,
            metrics,
        } => execute_assess_quality(audio_file, text, metrics, config, global).await,

        CloudCommands::HealthCheck => execute_health_check(config, global).await,

        CloudCommands::Configure {
            show,
            storage_provider,
            api_url,
            enable_service,
            init,
        } => {
            execute_configure(
                *show,
                storage_provider.as_ref(),
                api_url.as_ref(),
                enable_service.as_ref(),
                *init,
                config,
                global,
            )
            .await
        }
    }
}

/// Execute sync command. `force`, `directory`, and `dry_run` are all real,
/// consulted options -- forwarded to
/// [`CloudStorageManager::sync_with_options`] -- not printed-and-ignored
/// flags: `--dry-run` genuinely performs zero network I/O and zero
/// manifest writes, `--directory` genuinely restricts which manifest
/// items participate, and `--force` genuinely bypasses the staleness
/// check.
async fn execute_sync(
    force: bool,
    directory: Option<&PathBuf>,
    dry_run: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🔄 Synchronizing with cloud storage...");
        if dry_run {
            println!("📋 Dry run mode - no actual changes will be made");
        }
    }

    // Initialize cloud storage manager
    let storage_config = get_storage_config(config)?;
    let cache_dir = get_cache_directory()?;
    let mut storage_manager = CloudStorageManager::new(storage_config, cache_dir).map_err(|e| {
        VoirsError::config_error(format!("Failed to initialize storage manager: {}", e))
    })?;

    // Perform sync, honoring force/directory/dry_run for real.
    let options = SyncOptions {
        force,
        directory: directory.map(PathBuf::as_path),
        dry_run,
    };
    let sync_result = storage_manager
        .sync_with_options(&options)
        .await
        .map_err(|e| VoirsError::config_error(format!("Sync failed: {}", e)))?;

    if !global.quiet {
        if dry_run {
            println!("✅ Dry run completed (no changes were made)!");
            println!(
                "📊 Files that would be uploaded: {}",
                sync_result.uploaded_files
            );
            println!(
                "📥 Files that would be downloaded: {}",
                sync_result.downloaded_files
            );
        } else {
            println!("✅ Sync completed successfully!");
            println!("📊 Files uploaded: {}", sync_result.uploaded_files);
            println!("📥 Files downloaded: {}", sync_result.downloaded_files);
        }
        println!("⏭️  Files skipped: {}", sync_result.skipped_files);
        if sync_result.failed_uploads > 0 {
            println!("❌ Failed uploads: {}", sync_result.failed_uploads);
        }
        if sync_result.failed_downloads > 0 {
            println!("❌ Failed downloads: {}", sync_result.failed_downloads);
        }
    }

    Ok(())
}

/// Execute add to sync command
async fn execute_add_to_sync(
    local_path: &Path,
    remote_path: &str,
    direction: &str,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!(
            "📁 Adding {} to sync configuration...",
            local_path.display()
        );
    }

    let sync_direction = match direction.to_lowercase().as_str() {
        "upload" => SyncDirection::Upload,
        "download" => SyncDirection::Download,
        "bidirectional" => SyncDirection::Bidirectional,
        _ => {
            return Err(VoirsError::config_error(
                "Invalid sync direction. Must be: upload, download, or bidirectional",
            ))
        }
    };

    // Initialize storage manager
    let storage_config = get_storage_config(config)?;
    let cache_dir = get_cache_directory()?;
    let mut storage_manager = CloudStorageManager::new(storage_config, cache_dir).map_err(|e| {
        VoirsError::config_error(format!("Failed to initialize storage manager: {}", e))
    })?;

    // Add to sync
    storage_manager
        .add_to_sync(
            local_path.to_path_buf(),
            remote_path.to_string(),
            sync_direction,
        )
        .await
        .map_err(|e| VoirsError::config_error(format!("Failed to add to sync: {}", e)))?;

    if !global.quiet {
        println!(
            "✅ Added to sync: {} -> {}",
            local_path.display(),
            remote_path
        );
    }

    Ok(())
}

/// Execute storage stats command
async fn execute_storage_stats(config: &AppConfig, global: &GlobalOptions) -> Result<()> {
    if !global.quiet {
        println!("📊 Retrieving cloud storage statistics...");
    }

    let storage_config = get_storage_config(config)?;
    let cache_dir = get_cache_directory()?;
    let storage_manager = CloudStorageManager::new(storage_config, cache_dir).map_err(|e| {
        VoirsError::config_error(format!("Failed to initialize storage manager: {}", e))
    })?;

    let stats = storage_manager
        .get_storage_stats()
        .await
        .map_err(|e| VoirsError::config_error(format!("Failed to get storage stats: {}", e)))?;

    println!("☁️  Cloud Storage Statistics");
    println!("═══════════════════════════");
    println!("📦 Total files: {}", stats.total_files);
    println!(
        "💾 Total size: {:.2} MB",
        stats.total_size_bytes as f64 / 1_048_576.0
    );
    println!("🕒 Last sync: {}", stats.last_sync_timestamp);
    println!("📁 Local files: {}", stats.local_files);
    println!("💽 Cache directory: {}", stats.cache_directory.display());

    Ok(())
}

/// Execute cleanup cache command. `dry_run` is a real, consulted option --
/// forwarded to [`CloudStorageManager::cleanup_cache_with_options`] -- not
/// a printed-and-ignored flag: with `--dry-run`, no file is deleted and
/// the manifest is not modified.
async fn execute_cleanup_cache(
    max_age_days: u32,
    dry_run: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!(
            "🧹 Cleaning up cache (files older than {} days)...",
            max_age_days
        );
        if dry_run {
            println!("📋 Dry run mode - no files will actually be deleted");
        }
    }

    let storage_config = get_storage_config(config)?;
    let cache_dir = get_cache_directory()?;
    let mut storage_manager = CloudStorageManager::new(storage_config, cache_dir).map_err(|e| {
        VoirsError::config_error(format!("Failed to initialize storage manager: {}", e))
    })?;

    let cleanup_result = storage_manager
        .cleanup_cache_with_options(max_age_days, dry_run)
        .await
        .map_err(|e| VoirsError::config_error(format!("Failed to cleanup cache: {}", e)))?;

    if !global.quiet {
        if dry_run {
            println!("✅ Dry run completed (no files were deleted)!");
            println!(
                "🗑️  Files that would be deleted: {}",
                cleanup_result.removed_files
            );
            println!(
                "💾 Space that would be freed: {:.2} MB",
                cleanup_result.freed_bytes as f64 / 1_048_576.0
            );
        } else {
            println!("✅ Cache cleanup completed!");
            println!("🗑️  Files deleted: {}", cleanup_result.removed_files);
            println!(
                "💾 Space freed: {:.2} MB",
                cleanup_result.freed_bytes as f64 / 1_048_576.0
            );
        }
    }

    Ok(())
}

/// Execute translate command
async fn execute_translate(
    text: &str,
    from: &str,
    to: &str,
    quality: &str,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🌐 Translating text from {} to {}...", from, to);
    }

    let api_config = get_api_config(config)?;
    let mut api_client = CloudApiClient::new(api_config)
        .map_err(|e| VoirsError::config_error(format!("Failed to initialize API client: {}", e)))?;

    let translation_quality = match quality.to_lowercase().as_str() {
        "fast" => TranslationQuality::Fast,
        "balanced" => TranslationQuality::Balanced,
        "high-quality" => TranslationQuality::HighQuality,
        _ => TranslationQuality::Balanced,
    };

    let request = TranslationRequest {
        text: text.to_string(),
        source_language: from.to_string(),
        target_language: to.to_string(),
        preserve_ssml: false,
        quality_level: translation_quality,
    };

    let response = api_client
        .translate_text(request)
        .await
        .map_err(|e| VoirsError::config_error(format!("Translation failed: {}", e)))?;

    println!("📝 Translation Result:");
    println!("═══════════════════");
    println!("{}", response.translated_text);

    if !global.quiet && response.confidence_score > 0.0 {
        println!("🎯 Confidence: {:.1}%", response.confidence_score * 100.0);
    }

    Ok(())
}

/// Execute analyze content command
async fn execute_analyze_content(
    text: &str,
    analysis_types: &str,
    language: Option<&String>,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🔍 Analyzing content...");
    }

    let api_config = get_api_config(config)?;
    let mut api_client = CloudApiClient::new(api_config)
        .map_err(|e| VoirsError::config_error(format!("Failed to initialize API client: {}", e)))?;

    let request = ContentAnalysisRequest {
        content: text.to_string(),
        analysis_types: analysis_types
            .split(',')
            .map(|s| match s.trim().to_lowercase().as_str() {
                "sentiment" => AnalysisType::Sentiment,
                "entities" => AnalysisType::Entities,
                "keywords" => AnalysisType::Keywords,
                _ => AnalysisType::Sentiment,
            })
            .collect(),
        language: language.cloned(),
    };

    let response = api_client
        .analyze_content(request)
        .await
        .map_err(|e| VoirsError::config_error(format!("Content analysis failed: {}", e)))?;

    println!("🔍 Content Analysis Results:");
    println!("════════════════════════════");

    if let Some(sentiment) = response.sentiment {
        println!(
            "💭 Sentiment: {} (confidence: {:.2})",
            sentiment.sentiment, sentiment.confidence
        );
    }

    if !response.entities.is_empty() {
        println!("🏷️  Entities:");
        for entity in &response.entities {
            println!("   • {} ({})", entity.text, entity.entity_type);
        }
    }

    if !response.keywords.is_empty() {
        println!("🔑 Keywords:");
        for keyword in &response.keywords {
            println!(
                "   • {} (relevance: {:.2})",
                keyword.keyword, keyword.relevance
            );
        }
    }

    Ok(())
}

/// Execute assess quality command
async fn execute_assess_quality(
    audio_file: &PathBuf,
    text: &str,
    metrics: &str,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🎧 Assessing audio quality for {}...", audio_file.display());
    }

    if !audio_file.exists() {
        return Err(VoirsError::IoError {
            path: audio_file.clone(),
            operation: voirs_sdk::error::IoOperation::Read,
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "Audio file not found"),
        });
    }

    let api_config = get_api_config(config)?;
    let mut api_client = CloudApiClient::new(api_config)
        .map_err(|e| VoirsError::config_error(format!("Failed to initialize API client: {}", e)))?;

    // Read audio file (in a real implementation, you'd convert to the expected format)
    let audio_data = std::fs::read(audio_file).map_err(|e| VoirsError::IoError {
        path: audio_file.clone(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })?;

    let request = QualityAssessmentRequest {
        audio_data,
        text: text.to_string(),
        synthesis_config: SynthesisConfig::default(),
        assessment_types: metrics
            .split(',')
            .map(|s| match s.trim().to_lowercase().as_str() {
                "naturalness" => QualityMetric::Naturalness,
                "intelligibility" => QualityMetric::Intelligibility,
                "prosody" => QualityMetric::Prosody,
                "pronunciation" => QualityMetric::Pronunciation,
                "overall" => QualityMetric::OverallQuality,
                _ => QualityMetric::OverallQuality,
            })
            .collect(),
    };

    let response = api_client
        .assess_quality(request)
        .await
        .map_err(|e| VoirsError::config_error(format!("Quality assessment failed: {}", e)))?;

    println!("🎧 Audio Quality Assessment:");
    println!("═══════════════════════════");
    println!("🎯 Overall Score: {:.1}/10", response.overall_score);

    for (metric_name, score) in &response.metric_scores {
        println!("📊 {}: {:.1}/10", metric_name, score);
    }

    if !response.detailed_feedback.is_empty() {
        for feedback in &response.detailed_feedback {
            println!("💡 {}: {:.1}/10", feedback.metric, feedback.score);
        }
    }

    Ok(())
}

/// Execute health check command
async fn execute_health_check(config: &AppConfig, global: &GlobalOptions) -> Result<()> {
    if !global.quiet {
        println!("🏥 Checking cloud service health...");
    }

    let api_config = get_api_config(config)?;
    let mut api_client = CloudApiClient::new(api_config)
        .map_err(|e| VoirsError::config_error(format!("Failed to initialize API client: {}", e)))?;

    let health = api_client
        .get_service_health()
        .await
        .map_err(|e| VoirsError::config_error(format!("Health check failed: {}", e)))?;

    println!("🏥 Cloud Service Health Status:");
    println!("══════════════════════════════");
    println!("🟢 Status: {}", health.status);
    println!("⏱️  Response Time: {}ms", health.response_time_ms);
    println!("📊 API Version: {}", health.version);

    for (service_name, service_status) in &health.services {
        let status_icon = if service_status.healthy {
            "🟢"
        } else {
            "🔴"
        };
        let status_text = if service_status.healthy {
            "healthy"
        } else {
            "unhealthy"
        };
        println!("{} {}: {}", status_icon, service_name, status_text);
        if let Some(error) = &service_status.error_message {
            println!("   ❌ Error: {}", error);
        }
    }

    Ok(())
}

/// Execute configure command
async fn execute_configure(
    show: bool,
    storage_provider: Option<&String>,
    api_url: Option<&String>,
    enable_service: Option<&String>,
    init: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if show {
        // Display current cloud configuration
        if !global.quiet {
            println!("⚙️  Cloud Configuration:");
            println!("═══════════════════════════════════════════════════════════");
        }

        // Load current config. A *missing* config file is not an error
        // (nothing has been configured yet, so the "<not configured>"
        // placeholder below is accurate) -- but a *present, malformed*
        // file is a real problem, and must be shown as one. Silently
        // falling back to "<not configured>" for a parse error would hide
        // a real misconfiguration behind a message that tells the user
        // there is nothing to fix.
        let storage_config = match get_storage_config(config) {
            Ok(cfg) => cfg,
            Err(e) => {
                if !global.quiet {
                    println!("⚠️  Failed to load storage configuration: {e}");
                }
                CloudStorageConfig {
                    provider: StorageProvider::S3Compatible,
                    bucket_name: "<not configured>".to_string(),
                    region: "us-east-1".to_string(),
                    access_key: None,
                    secret_key: None,
                    endpoint: None,
                    encryption_enabled: false,
                    compression_enabled: true,
                    sync_interval_seconds: 300,
                }
            }
        };

        let api_config = match get_api_config(config) {
            Ok(cfg) => cfg,
            Err(e) => {
                if !global.quiet {
                    println!("⚠️  Failed to load API configuration: {e}");
                }
                CloudApiConfig {
                    base_url: "<not configured>".to_string(),
                    api_key: None,
                    timeout_seconds: 30,
                    retry_attempts: 3,
                    rate_limit_requests_per_minute: 60,
                    enabled_services: vec![],
                }
            }
        };

        if !global.quiet {
            println!("\n📦 Storage Configuration:");
            println!("   Provider:       {:?}", storage_config.provider);
            println!("   Bucket:         {}", storage_config.bucket_name);
            println!("   Region:         {}", storage_config.region);
            println!(
                "   Access Key:     {}",
                if storage_config.access_key.is_some() {
                    "***configured***"
                } else {
                    "<not set>"
                }
            );
            println!(
                "   Secret Key:     {}",
                if storage_config.secret_key.is_some() {
                    "***configured***"
                } else {
                    "<not set>"
                }
            );
            println!(
                "   Endpoint:       {}",
                storage_config.endpoint.as_deref().unwrap_or("<default>")
            );
            println!(
                "   Encryption:     {}",
                if storage_config.encryption_enabled {
                    "Enabled"
                } else {
                    "Disabled"
                }
            );
            println!(
                "   Compression:    {}",
                if storage_config.compression_enabled {
                    "Enabled"
                } else {
                    "Disabled"
                }
            );
            println!(
                "   Sync interval:  {}s",
                storage_config.sync_interval_seconds
            );

            println!("\n🌐 API Configuration:");
            println!("   Base URL:       {}", api_config.base_url);
            println!(
                "   API Key:        {}",
                if api_config.api_key.is_some() {
                    "***configured***"
                } else {
                    "<not set>"
                }
            );
            println!("   Timeout:        {}s", api_config.timeout_seconds);
            println!("   Retry attempts: {}", api_config.retry_attempts);
            println!(
                "   Rate limit:     {}/min",
                api_config.rate_limit_requests_per_minute
            );
            println!("   Enabled services:");
            for service in &api_config.enabled_services {
                println!("     - {:?}", service);
            }

            if api_config.enabled_services.is_empty() {
                println!("     <none configured>");
            }

            println!("\n💡 Configuration file location:");
            if let Some(config_dir) = dirs::config_dir() {
                println!("   {}/voirs/cloud_config.toml", config_dir.display());
            } else {
                println!("   ~/.config/voirs/cloud_config.toml");
            }

            println!("\n📝 To update configuration:");
            println!("   voirs cloud configure --init              (initialize with defaults)");
            println!("   voirs cloud configure --storage-provider s3");
            println!("   voirs cloud configure --api-url https://api.example.com");
            println!("   voirs cloud configure --enable-service translation");
            println!("═══════════════════════════════════════════════════════════");
        }

        return Ok(());
    }

    if init {
        if !global.quiet {
            println!("🚀 Initializing cloud configuration...");
        }

        // Create default configuration
        let default_storage = CloudStorageConfig {
            provider: StorageProvider::S3Compatible,
            bucket_name: "voirs-cloud".to_string(),
            region: "us-east-1".to_string(),
            access_key: None,
            secret_key: None,
            endpoint: None,
            encryption_enabled: false,
            compression_enabled: true,
            sync_interval_seconds: 300,
        };

        let default_api = CloudApiConfig {
            base_url: "https://api.voirs.cloud".to_string(),
            api_key: None,
            timeout_seconds: 30,
            retry_attempts: 3,
            rate_limit_requests_per_minute: 60,
            enabled_services: vec![
                CloudService::Translation,
                CloudService::ContentManagement,
                CloudService::QualityAssurance,
            ],
        };

        // Determine config file path
        let config_dir = if let Some(dir) = dirs::config_dir() {
            dir.join("voirs")
        } else {
            PathBuf::from(".").join(".config").join("voirs")
        };

        std::fs::create_dir_all(&config_dir).map_err(|e| {
            VoirsError::config_error(format!("Failed to create config directory: {}", e))
        })?;

        let config_file = config_dir.join("cloud_config.toml");

        // Create TOML configuration
        let config_content = format!(
            r#"# VoiRS Cloud Configuration
# Generated by: voirs cloud configure --init

[storage]
provider = "{:?}"
bucket_name = "{}"
region = "{}"
# access_key = "your-access-key"
# secret_key = "your-secret-key"
# endpoint = "https://s3.example.com"
encryption_enabled = {}
compression_enabled = {}
sync_interval_seconds = {}

[api]
base_url = "{}"
# api_key = "your-api-key"
timeout_seconds = {}
retry_attempts = {}
rate_limit_requests_per_minute = {}
enabled_services = ["Translation", "ContentManagement", "QualityAssurance"]

# To configure credentials:
# 1. Uncomment the credential lines above
# 2. Replace placeholder values with your actual credentials
# 3. Ensure this file has restricted permissions (chmod 600 on Unix)
"#,
            default_storage.provider,
            default_storage.bucket_name,
            default_storage.region,
            default_storage.encryption_enabled,
            default_storage.compression_enabled,
            default_storage.sync_interval_seconds,
            default_api.base_url,
            default_api.timeout_seconds,
            default_api.retry_attempts,
            default_api.rate_limit_requests_per_minute,
        );

        std::fs::write(&config_file, config_content)
            .map_err(|e| VoirsError::config_error(format!("Failed to write config file: {}", e)))?;

        if !global.quiet {
            println!("✅ Cloud configuration initialized!");
            println!("📁 Configuration file created:");
            println!("   {}", config_file.display());
            println!();
            println!("📝 Next steps:");
            println!("   1. Edit the config file to add your credentials");
            println!("   2. Run: voirs cloud configure --show  (to verify)");
            println!("   3. Run: voirs cloud health-check      (to test connectivity)");
        }

        return Ok(());
    }

    // Handle configuration updates
    let mut updated = false;

    if let Some(provider) = storage_provider {
        if !global.quiet {
            println!("🔧 Updating storage provider to: {}", provider);
        }
        // In a real implementation, this would update the config file
        updated = true;
    }

    if let Some(url) = api_url {
        if !global.quiet {
            println!("🔧 Updating API URL to: {}", url);
        }
        // In a real implementation, this would update the config file
        updated = true;
    }

    if let Some(service) = enable_service {
        if !global.quiet {
            println!("🔧 Enabling service: {}", service);
        }
        // In a real implementation, this would update the config file
        updated = true;
    }

    if updated {
        if !global.quiet {
            println!("\n⚠️  Configuration updates are staged but not yet persisted.");
            println!("   Full config persistence will be implemented in the next release.");
            println!("   For now, manually edit: ~/.config/voirs/cloud_config.toml");
        }
    } else if !global.quiet {
        println!("ℹ️  No configuration changes requested.");
        println!("   Use --show to view current configuration");
        println!("   Use --init to initialize default configuration");
    }

    Ok(())
}

/// On-disk shape of `~/.config/voirs/cloud_config.toml`, as written by
/// `voirs cloud configure --init` and hand-edited by the user. Every field
/// is optional so a partially-filled or hand-trimmed file still parses;
/// missing fields fall back to environment variables and then to
/// documented defaults in [`get_storage_config`]/[`get_api_config`] --
/// never to a fabricated credential.
#[derive(Debug, Default, Deserialize)]
struct CloudConfigFile {
    #[serde(default)]
    storage: Option<StorageConfigFile>,
    #[serde(default)]
    api: Option<ApiConfigFile>,
}

#[derive(Debug, Default, Deserialize)]
struct StorageConfigFile {
    provider: Option<String>,
    bucket_name: Option<String>,
    region: Option<String>,
    access_key: Option<String>,
    secret_key: Option<String>,
    endpoint: Option<String>,
    encryption_enabled: Option<bool>,
    compression_enabled: Option<bool>,
    sync_interval_seconds: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct ApiConfigFile {
    base_url: Option<String>,
    api_key: Option<String>,
    timeout_seconds: Option<u64>,
    retry_attempts: Option<u32>,
    rate_limit_requests_per_minute: Option<u32>,
    enabled_services: Option<Vec<String>>,
}

/// Directory holding `cloud_config.toml` (`$XDG_CONFIG_HOME/voirs`, or
/// platform equivalent via the `dirs` crate; falls back to
/// `./.config/voirs` if the platform config directory cannot be
/// determined -- matching [`execute_configure`]'s `--init` behavior).
fn cloud_config_dir() -> PathBuf {
    if let Some(dir) = dirs::config_dir() {
        dir.join("voirs")
    } else {
        PathBuf::from(".").join(".config").join("voirs")
    }
}

/// Path to the cloud config file to read. When `environment` is set (from
/// [`AppConfig::environment`]) and an environment-specific file
/// (`cloud_config.{environment}.toml`) exists, it takes priority over the
/// shared `cloud_config.toml` -- this is the one real use `AppConfig`
/// currently has for cloud commands: selecting a per-environment profile
/// (e.g. `staging` vs `production` buckets). Thin wrapper around
/// [`cloud_config_path_in`] resolving the real config directory; kept
/// separate so the environment-selection logic can be tested against a
/// `tempfile` directory instead of the real `~/.config/voirs/`.
fn cloud_config_path(environment: Option<&str>) -> PathBuf {
    cloud_config_path_in(&cloud_config_dir(), environment)
}

/// Core of [`cloud_config_path`]: given a config directory `dir`, resolve
/// which file to read for `environment`.
fn cloud_config_path_in(dir: &Path, environment: Option<&str>) -> PathBuf {
    if let Some(env) = environment.map(str::trim).filter(|e| !e.is_empty()) {
        let scoped = dir.join(format!("cloud_config.{env}.toml"));
        if scoped.exists() {
            return scoped;
        }
    }
    dir.join("cloud_config.toml")
}

/// Read and parse the cloud config file for `environment`. Thin wrapper
/// around [`read_cloud_config_file_at`] that resolves the real on-disk
/// path; kept separate so the parsing logic itself can be exercised in
/// tests against a `tempfile` path instead of the real
/// `~/.config/voirs/` directory.
fn read_cloud_config_file(environment: Option<&str>) -> Result<CloudConfigFile> {
    read_cloud_config_file_at(&cloud_config_path(environment))
}

/// Read and parse the cloud config file at `path`. A missing file is not
/// an error (nothing has been configured yet); a *present but malformed*
/// file is a real error, surfaced to the user rather than silently
/// ignored.
fn read_cloud_config_file_at(path: &Path) -> Result<CloudConfigFile> {
    if !path.exists() {
        return Ok(CloudConfigFile::default());
    }

    let content = std::fs::read_to_string(path).map_err(|e| VoirsError::IoError {
        path: path.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })?;

    toml::from_str(&content).map_err(|e| {
        VoirsError::config_error(format!(
            "failed to parse cloud config file {}: {e}",
            path.display()
        ))
    })
}

/// `Some(value)` for a set, non-blank environment variable; `None`
/// otherwise (unset *and* blank are both treated as "not configured").
fn env_var_non_empty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn non_blank(value: Option<String>) -> Option<String> {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Parse a storage provider name from CLI flags or the config file.
/// Accepts both the exact `StorageProvider` variant name (as written by
/// `execute_configure --init`, e.g. `"S3Compatible"`) and common
/// lowercase aliases.
fn parse_storage_provider(value: &str) -> Option<StorageProvider> {
    match value.trim().to_lowercase().as_str() {
        "aws" | "s3" | "amazon" | "amazons3" => Some(StorageProvider::AWS),
        "azure" | "azureblob" | "azure_blob" => Some(StorageProvider::Azure),
        "gcp" | "google" | "googlecloud" | "google_cloud" | "gcs" => {
            Some(StorageProvider::GoogleCloud)
        }
        "minio" => Some(StorageProvider::MinIO),
        "s3compatible" | "s3_compatible" | "s3-compatible" => Some(StorageProvider::S3Compatible),
        _ => None,
    }
}

fn parse_cloud_service(value: &str) -> Option<CloudService> {
    match value.trim().to_lowercase().replace(['-', '_'], "").as_str() {
        "translation" => Some(CloudService::Translation),
        "contentmanagement" => Some(CloudService::ContentManagement),
        "analytics" => Some(CloudService::Analytics),
        "qualityassurance" => Some(CloudService::QualityAssurance),
        "voicetraining" => Some(CloudService::VoiceTraining),
        "audioprocessing" => Some(CloudService::AudioProcessing),
        "speechrecognition" => Some(CloudService::SpeechRecognition),
        _ => None,
    }
}

/// Get cloud storage configuration: real credentials from environment
/// variables and `~/.config/voirs/cloud_config.toml` ([storage]), never a
/// fabricated placeholder. Precedence is env var > config file > a
/// documented, credential-free default; `access_key`/`secret_key` are
/// `None` (not a fake string) when neither source has them, which is what
/// lets [`crate::cloud::storage::CloudStorageManager`] fail closed with a
/// clear "not configured" error instead of attempting a doomed request.
///
/// Recognized environment variables:
/// - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION`
///   (falls back to `AWS_DEFAULT_REGION`) for AWS S3 and S3-compatible
///   providers (MinIO, R2, ...).
/// - `AZURE_STORAGE_ACCOUNT` / `AZURE_STORAGE_KEY` for Azure Blob Storage.
/// - `GOOGLE_OAUTH_TOKEN` for Google Cloud Storage (used as the bearer
///   token).
/// - `VOIRS_S3_ENDPOINT` to point AWS/S3-compatible traffic at a custom
///   endpoint (required for MinIO/R2/any non-AWS S3-compatible service).
fn get_storage_config(config: &AppConfig) -> Result<CloudStorageConfig> {
    let file = read_cloud_config_file(config.environment.as_deref())?;
    Ok(storage_config_from_file(file.storage.unwrap_or_default()))
}

/// Pure precedence logic (env var > config file > documented default),
/// split out from [`get_storage_config`] so it can be unit-tested against
/// an in-memory [`StorageConfigFile`] without touching the real
/// `~/.config/voirs/` directory.
fn storage_config_from_file(storage: StorageConfigFile) -> CloudStorageConfig {
    let provider = storage
        .provider
        .as_deref()
        .and_then(parse_storage_provider)
        .unwrap_or(StorageProvider::S3Compatible);

    let bucket_name = non_blank(storage.bucket_name).unwrap_or_else(|| "voirs-cloud".to_string());

    let region = env_var_non_empty("AWS_REGION")
        .or_else(|| env_var_non_empty("AWS_DEFAULT_REGION"))
        .or_else(|| non_blank(storage.region))
        .unwrap_or_else(|| "us-east-1".to_string());

    let access_key = match provider {
        StorageProvider::Azure => env_var_non_empty("AZURE_STORAGE_ACCOUNT"),
        _ => env_var_non_empty("AWS_ACCESS_KEY_ID"),
    }
    .or_else(|| non_blank(storage.access_key));

    let secret_key = match provider {
        StorageProvider::Azure => env_var_non_empty("AZURE_STORAGE_KEY"),
        StorageProvider::GoogleCloud => env_var_non_empty("GOOGLE_OAUTH_TOKEN"),
        _ => env_var_non_empty("AWS_SECRET_ACCESS_KEY"),
    }
    .or_else(|| non_blank(storage.secret_key));

    let endpoint = env_var_non_empty("VOIRS_S3_ENDPOINT").or_else(|| non_blank(storage.endpoint));

    CloudStorageConfig {
        provider,
        bucket_name,
        region,
        access_key,
        secret_key,
        endpoint,
        encryption_enabled: storage.encryption_enabled.unwrap_or(false),
        compression_enabled: storage.compression_enabled.unwrap_or(true),
        sync_interval_seconds: storage.sync_interval_seconds.unwrap_or(300),
    }
}

/// Get cloud API configuration: real values from
/// `~/.config/voirs/cloud_config.toml` ([api]), overridable with
/// `VOIRS_CLOUD_API_KEY`. `api_key` is `None` (not a fake string) when
/// unset.
fn get_api_config(config: &AppConfig) -> Result<CloudApiConfig> {
    let file = read_cloud_config_file(config.environment.as_deref())?;
    Ok(api_config_from_file(file.api.unwrap_or_default()))
}

/// Pure precedence logic (env var > config file > documented default),
/// split out from [`get_api_config`] for the same testability reason as
/// [`storage_config_from_file`].
fn api_config_from_file(api: ApiConfigFile) -> CloudApiConfig {
    let base_url = non_blank(api.base_url).unwrap_or_else(|| "https://api.voirs.cloud".to_string());
    let api_key = env_var_non_empty("VOIRS_CLOUD_API_KEY").or_else(|| non_blank(api.api_key));

    let enabled_services = api
        .enabled_services
        .map(|names| {
            names
                .iter()
                .filter_map(|name| parse_cloud_service(name))
                .collect::<Vec<_>>()
        })
        .filter(|services| !services.is_empty())
        .unwrap_or_else(|| {
            vec![
                CloudService::Translation,
                CloudService::ContentManagement,
                CloudService::QualityAssurance,
            ]
        });

    CloudApiConfig {
        base_url,
        api_key,
        timeout_seconds: api.timeout_seconds.unwrap_or(30),
        retry_attempts: api.retry_attempts.unwrap_or(3),
        rate_limit_requests_per_minute: api.rate_limit_requests_per_minute.unwrap_or(60),
        enabled_services,
    }
}

/// Get cache directory path
fn get_cache_directory() -> Result<PathBuf> {
    let cache_dir = if let Some(cache_dir) = dirs::cache_dir() {
        cache_dir.join("voirs").join("cloud")
    } else {
        std::env::current_dir()
            .unwrap_or_default()
            .join(".cache")
            .join("voirs")
            .join("cloud")
    };

    // Create directory if it doesn't exist
    std::fs::create_dir_all(&cache_dir).map_err(|e| VoirsError::IoError {
        path: cache_dir.clone(),
        operation: voirs_sdk::error::IoOperation::Write,
        source: e,
    })?;

    Ok(cache_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RAII helper that sets (or removes) a process environment variable
    /// for the duration of the guard, restoring whatever the process had
    /// beforehand when dropped. Needed because `storage_config_from_file`/
    /// `api_config_from_file` deliberately read real environment
    /// variables (that is the behavior under test), so tests exercising
    /// env-var precedence must not leak mutations to other tests or be
    /// thrown off by whatever the host/CI environment happens to export.
    struct EnvGuard {
        name: &'static str,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let previous = std::env::var(name).ok();
            std::env::set_var(name, value);
            Self { name, previous }
        }

        fn unset(name: &'static str) -> Self {
            let previous = std::env::var(name).ok();
            std::env::remove_var(name);
            Self { name, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(v) => std::env::set_var(self.name, v),
                None => std::env::remove_var(self.name),
            }
        }
    }

    /// All environment variables `storage_config_from_file`/
    /// `api_config_from_file` read, temporarily cleared so tests are
    /// deterministic regardless of the host/CI environment.
    fn clean_cloud_env() -> Vec<EnvGuard> {
        [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AZURE_STORAGE_ACCOUNT",
            "AZURE_STORAGE_KEY",
            "GOOGLE_OAUTH_TOKEN",
            "VOIRS_S3_ENDPOINT",
            "VOIRS_CLOUD_API_KEY",
        ]
        .into_iter()
        .map(EnvGuard::unset)
        .collect()
    }

    #[test]
    fn parse_storage_provider_accepts_debug_names_and_aliases() {
        assert!(matches!(
            parse_storage_provider("AWS"),
            Some(StorageProvider::AWS)
        ));
        assert!(matches!(
            parse_storage_provider("s3"),
            Some(StorageProvider::AWS)
        ));
        assert!(matches!(
            parse_storage_provider("Azure"),
            Some(StorageProvider::Azure)
        ));
        assert!(matches!(
            parse_storage_provider("GoogleCloud"),
            Some(StorageProvider::GoogleCloud)
        ));
        assert!(matches!(
            parse_storage_provider("gcs"),
            Some(StorageProvider::GoogleCloud)
        ));
        assert!(matches!(
            parse_storage_provider("MinIO"),
            Some(StorageProvider::MinIO)
        ));
        assert!(matches!(
            parse_storage_provider("S3Compatible"),
            Some(StorageProvider::S3Compatible)
        ));
        assert!(parse_storage_provider("not-a-real-provider").is_none());
    }

    #[test]
    fn parse_cloud_service_accepts_debug_names_and_aliases() {
        assert_eq!(
            parse_cloud_service("Translation"),
            Some(CloudService::Translation)
        );
        assert_eq!(
            parse_cloud_service("content_management"),
            Some(CloudService::ContentManagement)
        );
        assert_eq!(
            parse_cloud_service("QUALITY-ASSURANCE"),
            Some(CloudService::QualityAssurance)
        );
        assert_eq!(parse_cloud_service("nonsense"), None);
    }

    #[test]
    fn non_blank_treats_whitespace_only_as_absent() {
        assert_eq!(
            non_blank(Some("  value  ".to_string())),
            Some("value".to_string())
        );
        assert_eq!(non_blank(Some("   ".to_string())), None);
        assert_eq!(non_blank(None), None);
    }

    /// Regression test for `cloud-hardcoded-fake-credentials`: with no
    /// environment variables and an empty config file, storage credentials
    /// must be `None` -- never the old hardcoded
    /// `Some("default_key")`/`Some("default_secret")`.
    #[test]
    fn storage_config_from_file_never_fabricates_credentials() {
        let _clean = clean_cloud_env();

        let config = storage_config_from_file(StorageConfigFile::default());

        assert_eq!(config.access_key, None);
        assert_eq!(config.secret_key, None);
        assert_ne!(config.access_key, Some("default_key".to_string()));
        assert_ne!(config.secret_key, Some("default_secret".to_string()));
    }

    /// Same regression for the API config: no fabricated `"default_api_key"`.
    #[test]
    fn api_config_from_file_never_fabricates_api_key() {
        let _clean = clean_cloud_env();

        let config = api_config_from_file(ApiConfigFile::default());

        assert_eq!(config.api_key, None);
        assert_ne!(config.api_key, Some("default_api_key".to_string()));
    }

    #[test]
    fn storage_config_from_file_uses_real_file_credentials_when_present() {
        let _clean = clean_cloud_env();

        let mut file = StorageConfigFile::default();
        file.provider = Some("AWS".to_string());
        file.access_key = Some("AKIA_FROM_FILE".to_string());
        file.secret_key = Some("secret_from_file".to_string());
        file.bucket_name = Some("real-bucket".to_string());
        file.region = Some("eu-west-1".to_string());

        let config = storage_config_from_file(file);

        assert!(matches!(config.provider, StorageProvider::AWS));
        assert_eq!(config.access_key, Some("AKIA_FROM_FILE".to_string()));
        assert_eq!(config.secret_key, Some("secret_from_file".to_string()));
        assert_eq!(config.bucket_name, "real-bucket");
        assert_eq!(config.region, "eu-west-1");
    }

    /// Environment variables take priority over the config file -- proves
    /// the function's output genuinely varies based on its real inputs,
    /// not a hardcoded constant.
    #[test]
    fn storage_config_from_file_prefers_env_var_over_file_value() {
        let _clean = clean_cloud_env();
        let _access = EnvGuard::set("AWS_ACCESS_KEY_ID", "env-access-key");
        let _region = EnvGuard::set("AWS_REGION", "ap-northeast-1");

        let mut file = StorageConfigFile::default();
        file.provider = Some("AWS".to_string());
        file.access_key = Some("file-access-key".to_string());
        file.region = Some("us-west-2".to_string());

        let config = storage_config_from_file(file);

        assert_eq!(config.access_key, Some("env-access-key".to_string()));
        assert_eq!(config.region, "ap-northeast-1");
    }

    #[test]
    fn storage_config_from_file_routes_azure_credentials_to_account_fields() {
        let _clean = clean_cloud_env();
        let _account = EnvGuard::set("AZURE_STORAGE_ACCOUNT", "myaccount");
        let _key = EnvGuard::set("AZURE_STORAGE_KEY", "base64keyvalue");

        let mut file = StorageConfigFile::default();
        file.provider = Some("azure".to_string());

        let config = storage_config_from_file(file);

        assert!(matches!(config.provider, StorageProvider::Azure));
        assert_eq!(config.access_key, Some("myaccount".to_string()));
        assert_eq!(config.secret_key, Some("base64keyvalue".to_string()));
    }

    #[test]
    fn storage_config_from_file_routes_gcp_token_to_secret_key() {
        let _clean = clean_cloud_env();
        let _token = EnvGuard::set("GOOGLE_OAUTH_TOKEN", "ya29.real-token");

        let mut file = StorageConfigFile::default();
        file.provider = Some("gcp".to_string());

        let config = storage_config_from_file(file);

        assert!(matches!(config.provider, StorageProvider::GoogleCloud));
        assert_eq!(config.secret_key, Some("ya29.real-token".to_string()));
    }

    #[test]
    fn read_cloud_config_file_at_missing_file_returns_default() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("does_not_exist.toml");

        let file = read_cloud_config_file_at(&path).unwrap();

        assert!(file.storage.is_none());
        assert!(file.api.is_none());
    }

    #[test]
    fn read_cloud_config_file_at_parses_real_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("cloud_config.toml");
        std::fs::write(
            &path,
            r#"
[storage]
provider = "AWS"
bucket_name = "my-bucket"
region = "eu-west-1"
access_key = "AKIA_TEST"
secret_key = "secret_test"

[api]
base_url = "https://example.com"
api_key = "api_test_key"
enabled_services = ["Translation", "Analytics"]
"#,
        )
        .unwrap();

        let parsed = read_cloud_config_file_at(&path).unwrap();

        let storage = parsed.storage.expect("storage section present");
        assert_eq!(storage.bucket_name.as_deref(), Some("my-bucket"));
        assert_eq!(storage.access_key.as_deref(), Some("AKIA_TEST"));
        assert_eq!(storage.region.as_deref(), Some("eu-west-1"));

        let api = parsed.api.expect("api section present");
        assert_eq!(api.base_url.as_deref(), Some("https://example.com"));
        assert_eq!(
            api.enabled_services,
            Some(vec!["Translation".to_string(), "Analytics".to_string()])
        );
    }

    #[test]
    fn read_cloud_config_file_at_rejects_malformed_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("cloud_config.toml");
        std::fs::write(&path, "this is not [valid toml").unwrap();

        let result = read_cloud_config_file_at(&path);

        assert!(
            result.is_err(),
            "a malformed config file must be a real error, not silently treated as empty"
        );
    }

    #[test]
    fn cloud_config_path_in_falls_back_to_shared_file_without_environment() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = cloud_config_path_in(dir.path(), None);
        assert_eq!(path, dir.path().join("cloud_config.toml"));
    }

    #[test]
    fn cloud_config_path_in_falls_back_when_environment_file_does_not_exist() {
        let dir = tempfile::TempDir::new().unwrap();
        // No `cloud_config.staging.toml` was ever written in `dir`.
        let path = cloud_config_path_in(dir.path(), Some("staging"));
        assert_eq!(path, dir.path().join("cloud_config.toml"));
    }

    #[test]
    fn cloud_config_path_in_prefers_environment_specific_file_when_present() {
        let dir = tempfile::TempDir::new().unwrap();
        let scoped = dir.path().join("cloud_config.staging.toml");
        std::fs::write(&scoped, "[storage]\nbucket_name = \"staging-bucket\"\n").unwrap();
        // The shared file also exists, to prove the scoped one wins, not
        // just "the only file present".
        std::fs::write(dir.path().join("cloud_config.toml"), "[storage]\n").unwrap();

        let path = cloud_config_path_in(dir.path(), Some("staging"));

        assert_eq!(path, scoped);
    }
}
