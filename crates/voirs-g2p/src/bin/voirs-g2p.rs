//! VoiRS G2P Command Line Interface

use std::path::PathBuf;
use std::fs;
use std::io::{self, Write};
use std::collections::HashMap;

use clap::{Parser, Subcommand};
use serde_json;
use tokio;

use voirs_g2p::{
    G2p, LanguageCode, Result, G2pError,
    DummyG2p, G2pConverter,
    rules::EnglishRuleG2p,
    performance::{CachedG2p, BatchProcessor},
    config::ConfigManager,
};

/// VoiRS G2P - Grapheme-to-Phoneme conversion tool
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert text to phonemes
    Convert {
        /// Input text to convert
        text: String,
        
        /// Language code (e.g., en-US, ja, de)
        #[arg(short, long)]
        lang: Option<String>,
        
        /// Output format (plain, json, csv, ssml)
        #[arg(short, long, default_value = "plain")]
        format: String,
        
        /// Output file path (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// G2P backend to use
        #[arg(short, long, default_value = "rule-based")]
        backend: String,
        
        /// Show confidence scores
        #[arg(long)]
        show_confidence: bool,
        
        /// Show stress information
        #[arg(long)]
        show_stress: bool,
        
        /// Show syllable positions
        #[arg(long)]
        show_syllables: bool,
    },
    
    /// Process file or multiple files
    File {
        /// Input file path
        input: PathBuf,
        
        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Language code
        #[arg(short, long)]
        lang: Option<String>,
        
        /// Output format (plain, json, csv)
        #[arg(short, long, default_value = "json")]
        format: String,
        
        /// G2P backend to use
        #[arg(short, long, default_value = "rule-based")]
        backend: String,
    },
    
    /// Batch process multiple texts
    Batch {
        /// Input CSV file with text and optional language columns
        input: PathBuf,
        
        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Maximum concurrent processing
        #[arg(short, long, default_value = "4")]
        concurrency: usize,
        
        /// G2P backend to use
        #[arg(short, long, default_value = "rule-based")]
        backend: String,
    },
    
    /// Configuration management
    Config {
        #[command(subcommand)]
        config_cmd: ConfigCommands,
    },
    
    /// List available backends and models
    List {
        /// List type (backends, languages, models)
        #[arg(default_value = "backends")]
        list_type: String,
    },
    
    /// Benchmark performance
    Benchmark {
        /// Test text to benchmark
        #[arg(default_value = "The quick brown fox jumps over the lazy dog")]
        text: String,
        
        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,
        
        /// G2P backend to use
        #[arg(short, long, default_value = "rule-based")]
        backend: String,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Show current configuration
    Show,
    
    /// Set configuration value
    Set {
        /// Configuration key
        key: String,
        /// Configuration value
        value: String,
    },
    
    /// Get configuration value
    Get {
        /// Configuration key
        key: String,
    },
    
    /// Generate default configuration file
    Generate {
        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

/// Output format for phonemes
#[derive(Debug, Clone)]
enum OutputFormat {
    Plain,
    Json,
    Csv,
    Ssml,
}

impl From<&str> for OutputFormat {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => OutputFormat::Json,
            "csv" => OutputFormat::Csv,
            "ssml" => OutputFormat::Ssml,
            _ => OutputFormat::Plain,
        }
    }
}

/// Format phonemes for output
fn format_phonemes(
    phonemes: &[voirs_g2p::Phoneme],
    format: &OutputFormat,
    show_confidence: bool,
    show_stress: bool,
    show_syllables: bool,
) -> Result<String> {
    match format {
        OutputFormat::Plain => {
            let mut result = Vec::new();
            for phoneme in phonemes {
                let mut parts = vec![phoneme.symbol.clone()];
                
                if show_stress && phoneme.stress > 0 {
                    parts.push(format!("(stress:{})", phoneme.stress));
                }
                
                if show_syllables {
                    parts.push(format!("(pos:{:?})", phoneme.syllable_position));
                }
                
                if show_confidence {
                    parts.push(format!("(conf:{:.2})", phoneme.confidence));
                }
                
                result.push(parts.join(" "));
            }
            Ok(result.join(" "))
        }
        
        OutputFormat::Json => {
            let json_value = serde_json::to_string_pretty(phonemes)
                .map_err(|e| G2pError::ConfigError(format!("JSON serialization failed: {}", e)))?;
            Ok(json_value)
        }
        
        OutputFormat::Csv => {
            let mut result = vec!["symbol,stress,syllable_position,confidence,duration_ms".to_string()];
            
            for phoneme in phonemes {
                let row = format!("{},{},{:?},{},{}",
                    phoneme.symbol,
                    phoneme.stress,
                    phoneme.syllable_position,
                    phoneme.confidence,
                    phoneme.duration_ms.map_or("".to_string(), |d| d.to_string())
                );
                result.push(row);
            }
            
            Ok(result.join("\n"))
        }
        
        OutputFormat::Ssml => {
            let phoneme_str = phonemes.iter()
                .map(|p| p.symbol.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            
            Ok(format!("<phoneme alphabet=\"ipa\" ph=\"{}\">{}</phoneme>",
                phoneme_str, phoneme_str))
        }
    }
}

/// Parse language code from string
fn parse_language_code(lang: &str) -> Result<LanguageCode> {
    match lang.to_lowercase().as_str() {
        "en-us" | "en_us" | "en" => Ok(LanguageCode::EnUs),
        "en-gb" | "en_gb" => Ok(LanguageCode::EnGb),
        "ja" | "jp" => Ok(LanguageCode::Ja),
        "zh-cn" | "zh_cn" | "zh" => Ok(LanguageCode::ZhCn),
        "ko" | "kr" => Ok(LanguageCode::Ko),
        "de" => Ok(LanguageCode::De),
        "fr" => Ok(LanguageCode::Fr),
        "es" => Ok(LanguageCode::Es),
        _ => Err(G2pError::ConfigError(format!("Unsupported language: {}", lang))),
    }
}

/// Create G2P backend from name
async fn create_backend(name: &str) -> Result<Box<dyn G2p>> {
    match name.to_lowercase().as_str() {
        "rule-based" | "rules" => {
            let backend = EnglishRuleG2p::new()?;
            Ok(Box::new(backend))
        }
        "dummy" | "test" => {
            let backend = DummyG2p::new();
            Ok(Box::new(backend))
        }
        "cached-rule-based" | "cached-rules" => {
            let backend = EnglishRuleG2p::new()?;
            let cached_backend = CachedG2p::new(backend, 1000);
            Ok(Box::new(cached_backend))
        }
        _ => Err(G2pError::ConfigError(format!("Unknown backend: {}", name))),
    }
}

/// Process single text conversion
async fn process_convert(
    text: &str,
    lang: Option<String>,
    format: &OutputFormat,
    backend_name: &str,
    show_confidence: bool,
    show_stress: bool,
    show_syllables: bool,
) -> Result<String> {
    let backend = create_backend(backend_name).await?;
    
    let language_code = if let Some(lang) = lang {
        Some(parse_language_code(&lang)?)
    } else {
        None
    };
    
    let phonemes = backend.to_phonemes(text, language_code).await?;
    
    format_phonemes(&phonemes, format, show_confidence, show_stress, show_syllables)
}

/// Process file conversion
async fn process_file(
    input_path: &PathBuf,
    output_path: Option<&PathBuf>,
    lang: Option<String>,
    format: &OutputFormat,
    backend_name: &str,
) -> Result<()> {
    let content = fs::read_to_string(input_path)
        .map_err(|e| G2pError::ConfigError(format!("Failed to read file: {}", e)))?;
    
    let backend = create_backend(backend_name).await?;
    
    let language_code = if let Some(lang) = lang {
        Some(parse_language_code(&lang)?)
    } else {
        None
    };
    
    let phonemes = backend.to_phonemes(&content, language_code).await?;
    
    let output = format_phonemes(&phonemes, format, false, false, false)?;
    
    if let Some(output_path) = output_path {
        fs::write(output_path, output)
            .map_err(|e| G2pError::ConfigError(format!("Failed to write file: {}", e)))?;
    } else {
        println!("{}", output);
    }
    
    Ok(())
}

/// Process batch conversion
async fn process_batch(
    input_path: &PathBuf,
    output_path: Option<&PathBuf>,
    concurrency: usize,
    backend_name: &str,
) -> Result<()> {
    let content = fs::read_to_string(input_path)
        .map_err(|e| G2pError::ConfigError(format!("Failed to read file: {}", e)))?;
    
    let backend = std::sync::Arc::new(create_backend(backend_name).await?);
    
    // Parse CSV-like input
    let mut texts = Vec::new();
    for line in content.lines() {
        if !line.trim().is_empty() {
            texts.push(line.trim().to_string());
        }
    }
    
    if texts.is_empty() {
        return Err(G2pError::ConfigError("No texts found in input file".to_string()));
    }
    
    let results = BatchProcessor::process_batch(
        backend,
        texts,
        Some(LanguageCode::EnUs), // Default language
        concurrency,
    ).await?;
    
    let mut output_results = Vec::new();
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(phonemes) => {
                let formatted = format_phonemes(phonemes, &OutputFormat::Json, false, false, false)?;
                output_results.push(format!("{{\"index\": {}, \"phonemes\": {}}}", i, formatted));
            }
            Err(e) => {
                output_results.push(format!("{{\"index\": {}, \"error\": \"{}\"}}", i, e));
            }
        }
    }
    
    let output = format!("[{}]", output_results.join(", "));
    
    if let Some(output_path) = output_path {
        fs::write(output_path, output)
            .map_err(|e| G2pError::ConfigError(format!("Failed to write file: {}", e)))?;
    } else {
        println!("{}", output);
    }
    
    Ok(())
}

/// List available backends, languages, or models
async fn list_items(list_type: &str) -> Result<()> {
    match list_type.to_lowercase().as_str() {
        "backends" => {
            println!("Available G2P backends:");
            println!("  rule-based        - Rule-based G2P with phonological rules");
            println!("  cached-rule-based - Cached rule-based G2P");
            println!("  dummy             - Dummy G2P for testing");
        }
        "languages" => {
            println!("Supported languages:");
            println!("  en-US  - English (US)");
            println!("  en-GB  - English (UK)");
            println!("  ja     - Japanese");
            println!("  zh-CN  - Chinese (Simplified)");
            println!("  ko     - Korean");
            println!("  de     - German");
            println!("  fr     - French");
            println!("  es     - Spanish");
        }
        "models" => {
            println!("Available models:");
            println!("  English rule-based model (built-in)");
            println!("  Dummy model for testing (built-in)");
        }
        _ => {
            return Err(G2pError::ConfigError(format!("Unknown list type: {}", list_type)));
        }
    }
    Ok(())
}

/// Run performance benchmark
async fn benchmark(text: &str, iterations: usize, backend_name: &str) -> Result<()> {
    let backend = create_backend(backend_name).await?;
    
    println!("Running benchmark...");
    println!("Text: {}", text);
    println!("Iterations: {}", iterations);
    println!("Backend: {}", backend_name);
    
    let start = std::time::Instant::now();
    
    for _ in 0..iterations {
        let _ = backend.to_phonemes(text, Some(LanguageCode::EnUs)).await?;
    }
    
    let duration = start.elapsed();
    let avg_duration = duration / iterations as u32;
    
    println!("\nResults:");
    println!("  Total time: {:?}", duration);
    println!("  Average time per conversion: {:?}", avg_duration);
    println!("  Throughput: {:.2} conversions/second", iterations as f64 / duration.as_secs_f64());
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    if cli.verbose {
        std::env::set_var("RUST_LOG", "debug");
    }
    
    tracing_subscriber::fmt::init();
    
    match &cli.command {
        Commands::Convert { 
            text, 
            lang, 
            format, 
            output, 
            backend, 
            show_confidence, 
            show_stress, 
            show_syllables,
        } => {
            let output_format = OutputFormat::from(format.as_str());
            let result = process_convert(
                text,
                lang.clone(),
                &output_format,
                backend,
                *show_confidence,
                *show_stress,
                *show_syllables,
            ).await?;
            
            if let Some(output_path) = output {
                fs::write(output_path, result)
                    .map_err(|e| G2pError::ConfigError(format!("Failed to write file: {}", e)))?;
            } else {
                println!("{}", result);
            }
        }
        
        Commands::File { input, output, lang, format, backend } => {
            let output_format = OutputFormat::from(format.as_str());
            process_file(input, output.as_ref(), lang.clone(), &output_format, backend).await?;
        }
        
        Commands::Batch { input, output, concurrency, backend } => {
            process_batch(input, output.as_ref(), *concurrency, backend).await?;
        }
        
        Commands::Config { config_cmd } => {
            match config_cmd {
                ConfigCommands::Show => {
                    println!("Current configuration:");
                    println!("  Default backend: rule-based");
                    println!("  Default language: en-US");
                    println!("  Cache size: 1000");
                }
                ConfigCommands::Generate { output } => {
                    let default_config = r#"
# VoiRS G2P Configuration File

[general]
default_backend = "rule-based"
default_language = "en-US"
verbose = false

[cache]
enabled = true
size = 1000
ttl_minutes = 60

[backends]
rule_based_enabled = true
neural_enabled = false
hybrid_enabled = false

[performance]
max_concurrent = 4
batch_size = 100
"#;
                    
                    if let Some(output_path) = output {
                        fs::write(output_path, default_config)
                            .map_err(|e| G2pError::ConfigError(format!("Failed to write config: {}", e)))?;
                        println!("Configuration written to: {}", output_path.display());
                    } else {
                        println!("{}", default_config);
                    }
                }
                _ => {
                    println!("Configuration management not fully implemented yet");
                }
            }
        }
        
        Commands::List { list_type } => {
            list_items(list_type).await?;
        }
        
        Commands::Benchmark { text, iterations, backend } => {
            benchmark(text, *iterations, backend).await?;
        }
    }
    
    Ok(())
}