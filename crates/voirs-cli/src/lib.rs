//! # VoiRS CLI
//!
//! Command-line interface for VoiRS speech synthesis framework.
//! Provides easy-to-use commands for synthesis, voice management, and more.

use crate::cli_types::{CliAudioFormat, CliQualityLevel};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use voirs::{AudioFormat, QualityLevel, Result, VoirsPipeline};
use voirs_sdk::config::{AppConfig, PipelineConfig};

pub mod audio;
pub mod cli_types;
pub mod cloud;
pub mod commands;
pub mod completion;
pub mod config;
pub mod error;
pub mod help;
pub mod model_types;
pub mod output;
pub mod packaging;
pub mod performance;
pub mod platform;
pub mod plugins;
pub mod progress;
pub mod ssml;
pub mod synthesis;

// Re-export important types are already imported above

/// VoiRS CLI application
#[derive(Parser)]
#[command(name = "voirs")]
#[command(about = "A pure Rust text-to-speech synthesis framework")]
#[command(version = env!("CARGO_PKG_VERSION"))]
pub struct CliApp {
    /// Global options
    #[command(flatten)]
    pub global: GlobalOptions,

    /// Subcommands
    #[command(subcommand)]
    pub command: Commands,
}

/// Global CLI options
#[derive(Parser)]
pub struct GlobalOptions {
    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Quiet mode (suppress most output)
    #[arg(short, long)]
    pub quiet: bool,

    /// Output format (overrides config)
    #[arg(long)]
    pub format: Option<CliAudioFormat>,

    /// Voice to use (overrides config)
    #[arg(long)]
    pub voice: Option<String>,

    /// Enable GPU acceleration
    #[arg(long)]
    pub gpu: bool,

    /// Number of threads to use
    #[arg(long)]
    pub threads: Option<usize>,
}

/// Cloud-specific commands
#[derive(Subcommand)]
pub enum CloudCommands {
    /// Synchronize files with cloud storage
    Sync {
        /// Force full synchronization
        #[arg(long)]
        force: bool,

        /// Specific directory to sync
        #[arg(long)]
        directory: Option<PathBuf>,

        /// Dry run (show what would be synced)
        #[arg(long)]
        dry_run: bool,
    },

    /// Add file or directory to cloud sync
    AddToSync {
        /// Local path to sync
        local_path: PathBuf,

        /// Remote path in cloud storage
        remote_path: String,

        /// Sync direction (upload, download, bidirectional)
        #[arg(long, default_value = "bidirectional")]
        direction: String,
    },

    /// Show cloud storage statistics
    StorageStats,

    /// Clean up old cached files
    CleanupCache {
        /// Maximum age in days
        #[arg(long, default_value = "30")]
        max_age_days: u32,

        /// Dry run (show what would be cleaned)
        #[arg(long)]
        dry_run: bool,
    },

    /// Translate text using cloud services
    Translate {
        /// Text to translate
        text: String,

        /// Source language code
        #[arg(long)]
        from: String,

        /// Target language code  
        #[arg(long)]
        to: String,

        /// Translation quality (fast, balanced, high-quality)
        #[arg(long, default_value = "balanced")]
        quality: String,
    },

    /// Analyze content using cloud AI
    AnalyzeContent {
        /// Text content to analyze
        text: String,

        /// Analysis types (comma-separated: sentiment, entities, keywords, etc.)
        #[arg(long, default_value = "sentiment,entities")]
        analysis_types: String,

        /// Language code (optional)
        #[arg(long)]
        language: Option<String>,
    },

    /// Assess audio quality using cloud services
    AssessQuality {
        /// Audio file to assess
        audio_file: PathBuf,

        /// Text that was synthesized
        text: String,

        /// Assessment metrics (comma-separated)
        #[arg(long, default_value = "naturalness,intelligibility,overall")]
        metrics: String,
    },

    /// Check cloud service health
    HealthCheck,

    /// Configure cloud integration
    Configure {
        /// Show current configuration
        #[arg(long)]
        show: bool,

        /// Set cloud storage provider
        #[arg(long)]
        storage_provider: Option<String>,

        /// Set API base URL
        #[arg(long)]
        api_url: Option<String>,

        /// Enable/disable specific services
        #[arg(long)]
        enable_service: Option<String>,

        /// Initialize cloud configuration
        #[arg(long)]
        init: bool,
    },
}

/// Dataset-specific commands
#[derive(Subcommand)]
pub enum DatasetCommands {
    /// Validate dataset structure and quality
    Validate {
        /// Dataset directory path
        path: PathBuf,

        /// Dataset type (auto-detect if not specified)
        #[arg(long)]
        dataset_type: Option<String>,

        /// Perform detailed quality analysis
        #[arg(long)]
        detailed: bool,
    },

    /// Convert between dataset formats
    Convert {
        /// Input dataset path
        input: PathBuf,

        /// Output dataset path
        output: PathBuf,

        /// Source dataset format
        #[arg(long)]
        from: String,

        /// Target dataset format
        #[arg(long)]
        to: String,
    },

    /// Split dataset into train/validation/test sets
    Split {
        /// Dataset directory path
        path: PathBuf,

        /// Training set ratio (0.0-1.0)
        #[arg(long, default_value = "0.8")]
        train_ratio: f32,

        /// Validation set ratio (0.0-1.0)
        #[arg(long, default_value = "0.1")]
        val_ratio: f32,

        /// Test set ratio (auto-calculated if not specified)
        #[arg(long)]
        test_ratio: Option<f32>,

        /// Seed for reproducible splits
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Preprocess dataset for training
    Preprocess {
        /// Input dataset path
        input: PathBuf,

        /// Output directory for preprocessed data
        output: PathBuf,

        /// Target sample rate
        #[arg(long, default_value = "22050")]
        sample_rate: u32,

        /// Normalize audio levels
        #[arg(long)]
        normalize: bool,

        /// Apply audio filters
        #[arg(long)]
        filter: bool,
    },

    /// Generate dataset statistics and analysis
    Analyze {
        /// Dataset directory path
        path: PathBuf,

        /// Output file for analysis report
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Include detailed per-file statistics
        #[arg(long)]
        detailed: bool,
    },
}

/// CLI commands
#[derive(Subcommand)]
pub enum Commands {
    /// Synthesize text to speech
    Synthesize {
        /// Text to synthesize
        text: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Speaking rate (0.5 - 2.0)
        #[arg(long, default_value = "1.0")]
        rate: f32,

        /// Pitch shift in semitones (-12.0 - 12.0)
        #[arg(long, default_value = "0.0")]
        pitch: f32,

        /// Volume gain in dB (-20.0 - 20.0)
        #[arg(long, default_value = "0.0")]
        volume: f32,

        /// Quality level
        #[arg(long, default_value = "high")]
        quality: CliQualityLevel,

        /// Enable audio enhancement
        #[arg(long)]
        enhance: bool,
    },

    /// Synthesize from file
    SynthesizeFile {
        /// Input text file
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Speaking rate
        #[arg(long, default_value = "1.0")]
        rate: f32,

        /// Quality level
        #[arg(long, default_value = "high")]
        quality: CliQualityLevel,
    },

    /// List available voices
    ListVoices {
        /// Filter by language
        #[arg(long)]
        language: Option<String>,

        /// Show detailed information
        #[arg(long)]
        detailed: bool,
    },

    /// Get voice information
    VoiceInfo {
        /// Voice ID
        voice_id: String,
    },

    /// Download voice
    DownloadVoice {
        /// Voice ID to download
        voice_id: String,

        /// Force download even if voice exists
        #[arg(long)]
        force: bool,
    },

    /// Compare multiple voices side by side
    CompareVoices {
        /// Voice IDs to compare
        voice_ids: Vec<String>,
    },

    /// Test synthesis pipeline
    Test {
        /// Test text
        #[arg(default_value = "Hello, this is a test of VoiRS speech synthesis.")]
        text: String,

        /// Play audio instead of saving
        #[arg(long)]
        play: bool,
    },

    /// Run cross-language consistency tests for FFI bindings
    CrossLangTest {
        /// Output format for test report (json, yaml)
        #[arg(long, default_value = "json")]
        format: String,

        /// Save detailed test report to file
        #[arg(long)]
        save_report: bool,
    },

    /// Show configuration
    Config {
        /// Show configuration and exit
        #[arg(long)]
        show: bool,

        /// Initialize default configuration
        #[arg(long)]
        init: bool,

        /// Configuration file path for init
        #[arg(long)]
        path: Option<PathBuf>,
    },

    /// List available models
    ListModels {
        /// Filter by backend
        #[arg(long)]
        backend: Option<String>,

        /// Show detailed information
        #[arg(long)]
        detailed: bool,
    },

    /// Download a model
    DownloadModel {
        /// Model ID to download
        model_id: String,

        /// Force download even if model exists
        #[arg(long)]
        force: bool,
    },

    /// Benchmark models
    BenchmarkModels {
        /// Model IDs to benchmark
        model_ids: Vec<String>,

        /// Number of iterations
        #[arg(short, long, default_value = "3")]
        iterations: u32,

        /// Include accuracy testing against CMU test set (>95% phoneme accuracy target)
        #[arg(long)]
        accuracy: bool,
    },

    /// Optimize model for current hardware
    OptimizeModel {
        /// Model ID to optimize
        model_id: String,

        /// Output path for optimized model
        #[arg(short, long)]
        output: Option<String>,

        /// Optimization strategy (speed, quality, memory, balanced)
        #[arg(long, default_value = "balanced")]
        strategy: String,
    },

    /// Batch process multiple texts
    Batch {
        /// Input file or directory
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Number of parallel workers
        #[arg(short, long)]
        workers: Option<usize>,

        /// Speaking rate
        #[arg(long, default_value = "1.0")]
        rate: f32,

        /// Pitch shift in semitones
        #[arg(long, default_value = "0.0")]
        pitch: f32,

        /// Volume gain in dB
        #[arg(long, default_value = "0.0")]
        volume: f32,

        /// Quality level
        #[arg(long, default_value = "high")]
        quality: CliQualityLevel,

        /// Enable resume functionality
        #[arg(long)]
        resume: bool,
    },

    /// Server mode (future feature)
    Server {
        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },

    /// Interactive mode for real-time synthesis
    Interactive {
        /// Initial voice to use
        #[arg(short, long)]
        voice: Option<String>,

        /// Disable audio playback (synthesis only)
        #[arg(long)]
        no_audio: bool,

        /// Enable debug output
        #[arg(long)]
        debug: bool,

        /// Load session from file
        #[arg(long)]
        load_session: Option<PathBuf>,

        /// Auto-save session changes
        #[arg(long)]
        auto_save: bool,
    },

    /// Show detailed help and guides
    Guide {
        /// Command to get help for
        command: Option<String>,

        /// Show getting started guide
        #[arg(long)]
        getting_started: bool,

        /// Show examples for all commands
        #[arg(long)]
        examples: bool,
    },

    /// Generate shell completion scripts
    GenerateCompletion {
        /// Shell to generate completion for
        #[arg(value_enum)]
        shell: clap_complete::Shell,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,

        /// Show installation instructions
        #[arg(long)]
        install_help: bool,

        /// Generate installation script
        #[arg(long)]
        install_script: bool,

        /// Show completion status for all shells
        #[arg(long)]
        status: bool,
    },

    /// Dataset management and validation commands
    Dataset {
        /// Dataset subcommand to execute
        #[command(subcommand)]
        command: DatasetCommands,
    },

    /// Cloud integration commands
    Cloud {
        /// Cloud subcommand to execute
        #[command(subcommand)]
        command: CloudCommands,
    },

    /// Accuracy benchmarking commands
    Accuracy {
        /// Accuracy command configuration
        #[command(flatten)]
        command: commands::accuracy::AccuracyCommand,
    },

    /// Performance targets testing and monitoring
    Performance {
        /// Performance command configuration
        #[command(flatten)]
        command: commands::performance::PerformanceCommand,
    },

    /// Emotion control commands
    #[cfg(feature = "emotion")]
    Emotion {
        /// Emotion subcommand to execute
        #[command(subcommand)]
        command: commands::emotion::EmotionCommand,
    },

    /// Voice cloning commands
    #[cfg(feature = "cloning")]
    Clone {
        /// Cloning subcommand to execute
        #[command(subcommand)]
        command: commands::cloning::CloningCommand,
    },

    /// Voice conversion commands
    #[cfg(feature = "conversion")]
    Convert {
        /// Conversion subcommand to execute
        #[command(subcommand)]
        command: commands::conversion::ConversionCommand,
    },

    /// Singing voice synthesis commands
    #[cfg(feature = "singing")]
    Sing {
        /// Singing subcommand to execute
        #[command(subcommand)]
        command: commands::singing::SingingCommand,
    },

    /// 3D spatial audio commands
    #[cfg(feature = "spatial")]
    Spatial {
        /// Spatial subcommand to execute
        #[command(subcommand)]
        command: commands::spatial::SpatialCommand,
    },

    /// Feature detection and capability reporting
    Capabilities {
        /// Capabilities subcommand to execute
        #[command(subcommand)]
        command: commands::capabilities::CapabilitiesCommand,
    },

    /// Advanced monitoring and debugging commands
    Monitor {
        /// Monitoring subcommand to execute
        #[command(subcommand)]
        command: commands::monitoring::MonitoringCommand,
    },
}

/// CLI application implementation
impl CliApp {
    /// Run the CLI application
    pub async fn run() -> Result<()> {
        let app = Self::parse();

        // Initialize logging
        app.init_logging()?;

        // Load configuration
        let config = app.load_config().await?;

        // Execute command
        app.execute_command(config).await
    }

    /// Initialize logging based on verbosity
    fn init_logging(&self) -> Result<()> {
        let level = if self.global.quiet {
            tracing::Level::ERROR
        } else {
            match self.global.verbose {
                0 => tracing::Level::INFO,
                1 => tracing::Level::DEBUG,
                _ => tracing::Level::TRACE,
            }
        };

        tracing_subscriber::fmt()
            .with_max_level(level)
            .with_target(false)
            .init();

        Ok(())
    }

    /// Load configuration from file or use defaults
    async fn load_config(&self) -> Result<AppConfig> {
        let mut config = if let Some(config_path) = &self.global.config {
            tracing::info!("Loading configuration from {:?}", config_path);
            self.load_config_from_file(config_path).await?
        } else {
            // Try to load from default locations
            self.load_config_from_default_locations().await?
        };

        // Apply CLI overrides
        self.apply_cli_overrides(&mut config);

        Ok(config)
    }

    /// Load configuration from a specific file
    async fn load_config_from_file(&self, config_path: &std::path::Path) -> Result<AppConfig> {
        if !config_path.exists() {
            tracing::warn!(
                "Configuration file not found: {}, using defaults",
                config_path.display()
            );
            return Ok(AppConfig::default());
        }

        let content =
            std::fs::read_to_string(config_path).map_err(|e| voirs::VoirsError::IoError {
                path: config_path.to_path_buf(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;

        // Optimized format detection - use content analysis for better performance
        let config = match config_path.extension().and_then(|ext| ext.to_str()) {
            Some("toml") => {
                // For TOML files, try TOML first but allow fallback for compatibility
                toml::from_str(&content).or_else(|_| {
                    // Fallback to auto-detection for compatibility with tests
                    self.parse_config_auto_detect(&content)
                })?
            }
            Some("json") => {
                // For JSON files, try JSON first but allow fallback for compatibility
                serde_json::from_str(&content).or_else(|_| {
                    // Fallback to auto-detection for compatibility
                    self.parse_config_auto_detect(&content)
                })?
            }
            Some("yaml") | Some("yml") => {
                // For YAML files, try YAML first but allow fallback for compatibility
                serde_yaml::from_str(&content).or_else(|_| {
                    // Fallback to auto-detection for compatibility
                    self.parse_config_auto_detect(&content)
                })?
            }
            _ => {
                // Auto-detect format with optimized content analysis
                self.parse_config_auto_detect(&content)?
            }
        };

        tracing::info!(
            "Successfully loaded configuration from {}",
            config_path.display()
        );
        Ok(config)
    }

    /// Load configuration from default locations
    async fn load_config_from_default_locations(&self) -> Result<AppConfig> {
        let possible_paths = get_default_config_paths();

        for path in possible_paths {
            if path.exists() {
                tracing::info!("Found configuration file at: {}", path.display());
                return self.load_config_from_file(&path).await;
            }
        }

        // No config file found, use defaults
        tracing::info!("No configuration file found, using defaults");
        Ok(AppConfig::default())
    }

    /// Parse configuration with optimized auto-detection
    fn parse_config_auto_detect(&self, content: &str) -> Result<AppConfig> {
        // Optimized format detection using content analysis
        // Check for format indicators without parsing the entire content
        let trimmed = content.trim_start();

        if trimmed.starts_with('{') {
            // Likely JSON format
            serde_json::from_str(content).map_err(|e| {
                voirs::VoirsError::config_error(format!(
                    "Failed to parse JSON configuration: {}",
                    e
                ))
            })
        } else if trimmed.contains("---") || content.contains(": ") {
            // Likely YAML format (contains YAML indicators)
            serde_yaml::from_str(content).or_else(|yaml_err| {
                // Try TOML as fallback
                toml::from_str(content).map_err(|toml_err| {
                    voirs::VoirsError::config_error(format!(
                        "Failed to parse configuration. YAML error: {}, TOML error: {}",
                        yaml_err, toml_err
                    ))
                })
            })
        } else {
            // Try TOML first, then JSON, then YAML
            toml::from_str(content)
                .or_else(|_| serde_json::from_str(content))
                .or_else(|_| serde_yaml::from_str(content))
                .map_err(|e| {
                    voirs::VoirsError::config_error(format!(
                        "Unable to parse configuration file. Supported formats: TOML, JSON, YAML. Last error: {}", e
                    ))
                })
        }
    }

    /// Apply CLI overrides to configuration
    fn apply_cli_overrides(&self, config: &mut AppConfig) {
        if self.global.gpu {
            config.pipeline.use_gpu = true;
        }

        if let Some(threads) = self.global.threads {
            config.pipeline.num_threads = Some(threads);
        }

        if let Some(ref voice) = self.global.voice {
            config.cli.default_voice = Some(voice.clone());
        }

        if let Some(ref format) = self.global.format {
            config.cli.default_format = (*format).into();
            // Also update the synthesis config format
            config.pipeline.default_synthesis.output_format = (*format).into();
        }
    }

    /// Execute the specified command
    async fn execute_command(&self, config: AppConfig) -> Result<()> {
        match &self.command {
            Commands::Synthesize {
                text,
                output,
                rate,
                pitch,
                volume,
                quality,
                enhance,
            } => {
                commands::synthesize::run_synthesize(
                    text,
                    output.as_deref(),
                    *rate,
                    *pitch,
                    *volume,
                    (*quality).into(),
                    *enhance,
                    &config,
                    &self.global,
                )
                .await
            }

            Commands::SynthesizeFile {
                input,
                output_dir,
                rate,
                quality,
            } => {
                commands::synthesize::run_synthesize_file(
                    input,
                    output_dir.as_deref(),
                    *rate,
                    (*quality).into(),
                    &config,
                    &self.global,
                )
                .await
            }

            Commands::ListVoices { language, detailed } => {
                commands::voices::run_list_voices(language.as_deref(), *detailed, &config).await
            }

            Commands::VoiceInfo { voice_id } => {
                commands::voices::run_voice_info(voice_id, &config).await
            }

            Commands::DownloadVoice { voice_id, force } => {
                commands::voices::run_download_voice(voice_id, *force, &config).await
            }

            Commands::CompareVoices { voice_ids } => {
                commands::voices::run_compare_voices(voice_ids.clone(), &config).await
            }

            Commands::Test { text, play } => {
                commands::test::run_test(text, *play, &config, &self.global).await
            }

            Commands::CrossLangTest {
                format,
                save_report,
            } => {
                commands::cross_lang_test::run_cross_lang_tests(
                    format,
                    *save_report,
                    &config,
                    &self.global,
                )
                .await
            }

            Commands::Config { show, init, path } => {
                commands::config::run_config(*show, *init, path.as_deref(), &config).await
            }

            Commands::ListModels { backend, detailed } => {
                commands::models::run_list_models(
                    backend.as_deref(),
                    *detailed,
                    &config,
                    &self.global,
                )
                .await
            }

            Commands::DownloadModel { model_id, force } => {
                commands::models::run_download_model(model_id, *force, &config, &self.global).await
            }

            Commands::BenchmarkModels {
                model_ids,
                iterations,
                accuracy,
            } => {
                commands::models::run_benchmark_models(
                    model_ids,
                    *iterations,
                    *accuracy,
                    &config,
                    &self.global,
                )
                .await
            }

            Commands::OptimizeModel {
                model_id,
                output,
                strategy,
            } => {
                commands::models::run_optimize_model(
                    model_id,
                    output.as_deref(),
                    Some(strategy),
                    &config,
                    &self.global,
                )
                .await
            }

            Commands::Batch {
                input,
                output_dir,
                workers,
                rate,
                pitch,
                volume,
                quality,
                resume,
            } => {
                commands::batch::run_batch_process(
                    input,
                    output_dir.as_ref(),
                    *workers,
                    (*quality).into(),
                    *rate,
                    *pitch,
                    *volume,
                    *resume,
                    &config,
                    &self.global,
                )
                .await
            }

            Commands::Server { port, host } => {
                commands::server::run_server(host, *port, &config).await
            }

            Commands::Interactive {
                voice,
                no_audio,
                debug,
                load_session,
                auto_save,
            } => {
                let options = commands::interactive::InteractiveOptions {
                    voice: voice.clone(),
                    no_audio: *no_audio,
                    debug: *debug,
                    load_session: load_session.clone(),
                    auto_save: *auto_save,
                };

                commands::interactive::run_interactive(options)
                    .await
                    .map_err(Into::into)
            }

            Commands::Guide {
                command,
                getting_started,
                examples,
            } => {
                let help_system = help::HelpSystem::new();

                if *getting_started {
                    println!("{}", help::display_getting_started());
                } else if *examples {
                    println!("{}", help_system.display_command_overview());
                } else if let Some(cmd) = command {
                    println!("{}", help_system.display_command_help(cmd));
                } else {
                    println!("{}", help_system.display_command_overview());
                }

                Ok(())
            }

            Commands::GenerateCompletion {
                shell,
                output,
                install_help,
                install_script,
                status,
            } => {
                if *status {
                    println!("{}", completion::display_completion_status());
                } else if *install_script {
                    println!("{}", completion::generate_install_script());
                } else if *install_help {
                    println!("{}", completion::get_installation_instructions(*shell));
                } else if let Some(output_path) = output {
                    completion::generate_completion_to_file(*shell, output_path).map_err(|e| {
                        voirs::VoirsError::IoError {
                            path: output_path.clone(),
                            operation: voirs_sdk::error::IoOperation::Write,
                            source: e,
                        }
                    })?;
                    println!("Completion script generated: {}", output_path.display());
                } else {
                    completion::generate_completion_to_stdout(*shell).map_err(|e| {
                        voirs::VoirsError::IoError {
                            path: std::env::current_dir().unwrap_or_default(),
                            operation: voirs_sdk::error::IoOperation::Write,
                            source: e,
                        }
                    })?;
                }

                Ok(())
            }

            Commands::Dataset { command } => {
                commands::dataset::execute_dataset_command(command, &config, &self.global).await
            }

            Commands::Cloud { command } => {
                commands::cloud::execute_cloud_command(command, &config, &self.global).await
            }

            Commands::Accuracy { command } => {
                commands::accuracy::execute_accuracy_command(command.clone())
                    .await
                    .map_err(|e| {
                        voirs::VoirsError::config_error(format!("Accuracy command failed: {}", e))
                    })
            }

            Commands::Performance { command } => {
                commands::performance::execute_performance_command(command.clone())
                    .await
                    .map_err(|e| {
                        voirs::VoirsError::config_error(format!(
                            "Performance command failed: {}",
                            e
                        ))
                    })
            }

            #[cfg(feature = "emotion")]
            Commands::Emotion { command } => {
                use crate::output::OutputFormatter;
                let output_formatter = OutputFormatter::new(!self.global.quiet, false);
                commands::emotion::execute_emotion_command(command.clone(), &output_formatter)
                    .await
                    .map_err(|e| {
                        voirs::VoirsError::config_error(format!("Emotion command failed: {}", e))
                    })
            }

            #[cfg(feature = "cloning")]
            Commands::Clone { command } => {
                use crate::output::OutputFormatter;
                let output_formatter = OutputFormatter::new(!self.global.quiet, false);
                commands::cloning::execute_cloning_command(command.clone(), &output_formatter)
                    .await
                    .map_err(|e| {
                        voirs::VoirsError::config_error(format!("Cloning command failed: {}", e))
                    })
            }

            #[cfg(feature = "conversion")]
            Commands::Convert { command } => {
                use crate::output::OutputFormatter;
                let output_formatter = OutputFormatter::new(!self.global.quiet, false);
                commands::conversion::execute_conversion_command(command.clone(), &output_formatter)
                    .await
                    .map_err(|e| {
                        voirs::VoirsError::config_error(format!("Conversion command failed: {}", e))
                    })
            }

            #[cfg(feature = "singing")]
            Commands::Sing { command } => {
                use crate::output::OutputFormatter;
                let output_formatter = OutputFormatter::new(!self.global.quiet, false);
                commands::singing::execute_singing_command(command.clone(), &output_formatter)
                    .await
                    .map_err(|e| {
                        voirs::VoirsError::config_error(format!("Singing command failed: {}", e))
                    })
            }

            #[cfg(feature = "spatial")]
            Commands::Spatial { command } => {
                use crate::output::OutputFormatter;
                let output_formatter = OutputFormatter::new(!self.global.quiet, false);
                commands::spatial::execute_spatial_command(command.clone(), &output_formatter)
                    .await
                    .map_err(|e| {
                        voirs::VoirsError::config_error(format!("Spatial command failed: {}", e))
                    })
            }

            Commands::Capabilities { command } => {
                use crate::output::OutputFormatter;
                let output_formatter = OutputFormatter::new(!self.global.quiet, false);
                commands::capabilities::execute_capabilities_command(
                    command.clone(),
                    &output_formatter,
                    &config,
                )
                .await
                .map_err(|e| {
                    voirs::VoirsError::config_error(format!("Capabilities command failed: {}", e))
                })
            }

            Commands::Monitor { command } => {
                use crate::output::OutputFormatter;
                let output_formatter = OutputFormatter::new(!self.global.quiet, false);
                commands::monitoring::execute_monitoring_command(
                    command.clone(),
                    &output_formatter,
                    &config,
                )
                .await
                .map_err(|e| {
                    voirs::VoirsError::config_error(format!("Monitoring command failed: {}", e))
                })
            }
        }
    }
}

/// Utility functions for CLI
pub mod utils {
    use crate::cli_types::CliAudioFormat;
    use std::path::Path;
    use voirs::AudioFormat;

    /// Determine output format from file extension
    pub fn format_from_extension(path: &Path) -> Option<AudioFormat> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "wav" => Some(AudioFormat::Wav),
                "flac" => Some(AudioFormat::Flac),
                "mp3" => Some(AudioFormat::Mp3),
                "opus" => Some(AudioFormat::Opus),
                "ogg" => Some(AudioFormat::Ogg),
                _ => None,
            })
    }

    /// Generate output filename for text
    pub fn generate_output_filename(text: &str, format: AudioFormat) -> String {
        let safe_text = text
            .chars()
            .take(30)
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .replace(' ', "_")
            .to_lowercase();

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        format!("voirs_{}_{}.{}", safe_text, timestamp, format.extension())
    }
}

/// Get default configuration file paths in order of preference
fn get_default_config_paths() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();

    // 1. Current directory
    paths.push(
        std::env::current_dir()
            .unwrap_or_default()
            .join("voirs.toml"),
    );
    paths.push(
        std::env::current_dir()
            .unwrap_or_default()
            .join("voirs.json"),
    );
    paths.push(
        std::env::current_dir()
            .unwrap_or_default()
            .join("voirs.yaml"),
    );

    // 2. User config directory
    if let Some(config_dir) = dirs::config_dir() {
        let voirs_config_dir = config_dir.join("voirs");
        paths.push(voirs_config_dir.join("config.toml"));
        paths.push(voirs_config_dir.join("config.json"));
        paths.push(voirs_config_dir.join("config.yaml"));
        paths.push(voirs_config_dir.join("voirs.toml"));
        paths.push(voirs_config_dir.join("voirs.json"));
        paths.push(voirs_config_dir.join("voirs.yaml"));
    }

    // 3. Home directory
    if let Some(home_dir) = dirs::home_dir() {
        paths.push(home_dir.join(".voirs.toml"));
        paths.push(home_dir.join(".voirs.json"));
        paths.push(home_dir.join(".voirs.yaml"));
        paths.push(home_dir.join(".voirsrc"));
        paths.push(home_dir.join(".config").join("voirs").join("config.toml"));
    }

    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_from_extension() {
        use std::path::Path;

        assert_eq!(
            utils::format_from_extension(Path::new("test.wav")),
            Some(AudioFormat::Wav)
        );
        assert_eq!(
            utils::format_from_extension(Path::new("test.flac")),
            Some(AudioFormat::Flac)
        );
        assert_eq!(
            utils::format_from_extension(Path::new("test.unknown")),
            None
        );
    }

    #[test]
    fn test_generate_output_filename() {
        let filename = utils::generate_output_filename("Hello World", AudioFormat::Wav);
        assert!(filename.starts_with("voirs_hello_world_"));
        assert!(filename.ends_with(".wav"));
    }
}
