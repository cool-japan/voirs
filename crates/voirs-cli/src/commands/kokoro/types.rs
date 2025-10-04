//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)


/// Voice metadata
#[derive(Debug, Clone)]
struct VoiceInfo {
    index: usize,
    name: &'static str,
    language: &'static str,
    language_short: &'static str,
    gender: &'static str,
}
/// Kokoro-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroConfig {
    /// Default language code
    pub default_lang: Option<String>,
    /// Default voice name
    pub default_voice: Option<String>,
    /// Default speaking speed
    pub default_speed: Option<f32>,
    /// Path to Kokoro model directory
    pub model_dir: Option<PathBuf>,
    /// Path to eSpeak NG binary
    pub espeak_path: Option<PathBuf>,
}
impl KokoroConfig {
    /// Load Kokoro config from standard locations
    pub fn load() -> Self {
        let config_paths = get_kokoro_config_paths();
        for path in config_paths {
            if path.exists() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(config) = toml::from_str::<KokoroConfig>(&content) {
                        tracing::debug!("Loaded Kokoro config from {}", path.display());
                        return config;
                    }
                    if let Ok(config) = serde_json::from_str::<KokoroConfig>(&content) {
                        tracing::debug!("Loaded Kokoro config from {}", path.display());
                        return config;
                    }
                    if let Ok(config) = serde_yaml::from_str::<KokoroConfig>(&content) {
                        tracing::debug!("Loaded Kokoro config from {}", path.display());
                        return config;
                    }
                }
            }
        }
        tracing::debug!("No Kokoro config found, using defaults");
        Self::default()
    }
    /// Save config to specified path
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(
                    format!("Failed to serialize config: {}", e),
                )
            })?;
        std::fs::write(path, content)
            .map_err(|e| voirs_sdk::VoirsError::IoError {
                path: path.clone(),
                operation: voirs_sdk::error::IoOperation::Write,
                source: e,
            })?;
        Ok(())
    }
}
/// Kokoro-specific commands
#[derive(Subcommand)]
pub enum KokoroCommands {
    /// Synthesize text using Kokoro multilingual TTS
    Synth {
        /// Text to synthesize
        text: String,
        /// Output file path
        output: PathBuf,
        /// Language code (en-us, en-gb, es, fr, hi, it, pt-br, ja, zh)
        #[arg(short, long)]
        lang: String,
        /// Voice index (0-53, or use voice name)
        #[arg(short = 'i', long)]
        voice_index: Option<usize>,
        /// Voice name (e.g., af_jessica, bf_alice, ef_dora)
        #[arg(short = 'n', long)]
        voice_name: Option<String>,
        /// Speaking speed (0.5 - 2.0)
        #[arg(long, default_value = "1.0")]
        speed: f32,
        /// Manual IPA phonemes (disables automatic IPA generation)
        #[arg(long)]
        ipa: Option<String>,
        /// Play audio after synthesis
        #[arg(short, long)]
        play: bool,
        /// Kokoro model directory
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
    /// List available Kokoro voices
    Voices {
        /// Language filter (optional)
        #[arg(short, long)]
        lang: Option<String>,
        /// Show detailed voice information
        #[arg(long)]
        detailed: bool,
        /// Output format (table, json, csv)
        #[arg(long, default_value = "table")]
        format: String,
    },
    /// List supported languages
    Languages {
        /// Show IPA generation method for each language
        #[arg(long)]
        show_ipa_method: bool,
    },
    /// Test Kokoro model installation
    Test {
        /// Kokoro model directory
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Language to test (optional, tests all if not specified)
        #[arg(short, long)]
        lang: Option<String>,
    },
    /// Download Kokoro model files
    Download {
        /// Output directory for model files
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Force re-download if files exist
        #[arg(long)]
        force: bool,
    },
    /// Convert text to IPA phonemes
    TextToIpa {
        /// Text to convert
        text: String,
        /// Language code
        #[arg(short, long)]
        lang: String,
        /// Method (espeak, misaki, manual)
        #[arg(long, default_value = "auto")]
        method: String,
        /// Output file (optional, prints to stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Batch synthesis from CSV
    Batch {
        /// Input CSV file with columns: text, language, voice, output_file
        input: PathBuf,
        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Number of parallel jobs
        #[arg(short, long, default_value = "4")]
        jobs: usize,
        /// Treat text column as manual IPA (disables automatic IPA generation)
        #[arg(long)]
        manual_ipa: bool,
    },
    /// Manage Kokoro configuration
    Config {
        /// Show current configuration
        #[arg(long)]
        show: bool,
        /// Initialize default configuration file
        #[arg(long)]
        init: bool,
        /// Path for config file (defaults to ~/.config/voirs/kokoro.toml)
        #[arg(long)]
        path: Option<PathBuf>,
    },
}
