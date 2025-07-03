//! # VoiRS CLI
//! 
//! Command-line interface for VoiRS speech synthesis framework.
//! Provides easy-to-use commands for synthesis, voice management, and more.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use voirs::{
    config::{AppConfig, PipelineConfig},
    error::Result,
    types::{AudioFormat, QualityLevel},
    VoirsPipeline,
};
use crate::cli_types::{CliAudioFormat, CliQualityLevel};

pub mod audio;
pub mod cli_types;
pub mod commands;
pub mod config;
pub mod error;
pub mod output;
pub mod progress;
pub mod ssml;
pub mod model_types;

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
    
    /// Test synthesis pipeline
    Test {
        /// Test text
        #[arg(default_value = "Hello, this is a test of VoiRS speech synthesis.")]
        text: String,
        
        /// Play audio instead of saving
        #[arg(long)]
        play: bool,
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
    },
    
    /// Optimize model for current hardware
    OptimizeModel {
        /// Model ID to optimize
        model_id: String,
        
        /// Output path for optimized model
        #[arg(short, long)]
        output: Option<String>,
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
        if let Some(config_path) = &self.global.config {
            tracing::info!("Loading configuration from {:?}", config_path);
            // TODO: Load configuration from file
            Ok(AppConfig::default())
        } else {
            // Use default configuration with CLI overrides
            let mut config = AppConfig::default();
            
            // Apply CLI overrides
            if self.global.gpu {
                config.pipeline.use_gpu = true;
            }
            
            if let Some(threads) = self.global.threads {
                config.pipeline.num_threads = Some(threads);
            }
            
            Ok(config)
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
                ).await
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
                ).await
            }
            
            Commands::ListVoices { language, detailed } => {
                commands::voices::run_list_voices(
                    language.as_deref(),
                    *detailed,
                    &config,
                ).await
            }
            
            Commands::VoiceInfo { voice_id } => {
                commands::voices::run_voice_info(voice_id, &config).await
            }
            
            Commands::DownloadVoice { voice_id, force } => {
                commands::voices::run_download_voice(voice_id, *force, &config).await
            }
            
            Commands::Test { text, play } => {
                commands::test::run_test(text, *play, &config, &self.global).await
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
                ).await
            }
            
            Commands::DownloadModel { model_id, force } => {
                commands::models::run_download_model(
                    model_id,
                    *force,
                    &config,
                    &self.global,
                ).await
            }
            
            Commands::BenchmarkModels { model_ids, iterations } => {
                commands::models::run_benchmark_models(
                    model_ids,
                    *iterations,
                    &config,
                    &self.global,
                ).await
            }
            
            Commands::OptimizeModel { model_id, output } => {
                commands::models::run_optimize_model(
                    model_id,
                    output.as_deref(),
                    &config,
                    &self.global,
                ).await
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
                ).await
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
                
                commands::interactive::run_interactive(options).await
            }
        }
    }
}

/// Utility functions for CLI
pub mod utils {
    use std::path::Path;
    use voirs::types::AudioFormat;
    use crate::cli_types::CliAudioFormat;
    
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