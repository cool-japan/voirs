//! Enhanced help system for VoiRS CLI
//!
//! Provides context-sensitive help, examples, and tips for better user experience.

use console::{style, Style};
use std::collections::HashMap;

/// Help content for commands and concepts
pub struct HelpSystem {
    /// Command-specific help content
    pub command_help: HashMap<String, CommandHelp>,
    /// General tips and concepts
    general_tips: Vec<HelpTip>,
}

/// Help content for a specific command
#[derive(Debug, Clone)]
pub struct CommandHelp {
    /// Brief description
    pub description: String,
    /// Detailed explanation
    pub detailed: String,
    /// Usage examples
    pub examples: Vec<HelpExample>,
    /// Related commands
    pub related: Vec<String>,
    /// Common issues and solutions
    pub troubleshooting: Vec<HelpTip>,
}

/// A help example with description
#[derive(Debug, Clone)]
pub struct HelpExample {
    /// Example command
    pub command: String,
    /// Description of what it does
    pub description: String,
    /// Expected output or result
    pub expected: Option<String>,
}

/// A help tip or troubleshooting item
#[derive(Debug, Clone)]
pub struct HelpTip {
    /// Title of the tip
    pub title: String,
    /// Detailed explanation
    pub content: String,
    /// Severity level (info, warning, error)
    pub level: TipLevel,
}

/// Severity level for tips
#[derive(Debug, Clone, PartialEq)]
pub enum TipLevel {
    Info,
    Warning,
    Error,
}

impl HelpSystem {
    /// Create a new help system with all content loaded
    pub fn new() -> Self {
        let mut help_system = Self {
            command_help: HashMap::new(),
            general_tips: Vec::new(),
        };

        help_system.load_command_help();
        help_system.load_general_tips();
        help_system
    }

    /// Get help for a specific command
    pub fn get_command_help(&self, command: &str) -> Option<&CommandHelp> {
        self.command_help.get(command)
    }

    /// Display formatted help for a command
    pub fn display_command_help(&self, command: &str) -> String {
        if let Some(help) = self.get_command_help(command) {
            self.format_command_help(command, help)
        } else {
            self.format_unknown_command(command)
        }
    }

    /// Get context-sensitive help based on error or situation
    pub fn get_contextual_help(&self, context: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        match context {
            "voice_not_found" => {
                suggestions.push("List available voices with: voirs list-voices".to_string());
                suggestions
                    .push("Download a voice with: voirs download-voice <voice-id>".to_string());
            }
            "model_not_found" => {
                suggestions.push("List available models with: voirs list-models".to_string());
                suggestions
                    .push("Download a model with: voirs download-model <model-id>".to_string());
            }
            "file_not_found" => {
                suggestions.push("Check if the file path is correct".to_string());
                suggestions.push("Use absolute paths for clarity".to_string());
            }
            "permission_denied" => {
                suggestions.push("Check file/directory permissions".to_string());
                suggestions.push("Try running with appropriate permissions".to_string());
            }
            "gpu_error" => {
                suggestions.push("Check GPU availability with: voirs test --gpu".to_string());
                suggestions.push("Fallback to CPU mode by removing --gpu flag".to_string());
            }
            _ => {
                suggestions.push("Use 'voirs guide' for general help".to_string());
                suggestions
                    .push("Use 'voirs guide <command>' for command-specific help".to_string());
            }
        }

        suggestions
    }

    /// Display all available commands with brief descriptions
    pub fn display_command_overview(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "{}\n\n",
            style("VoiRS CLI Commands").bold().cyan()
        ));

        let categories = vec![
            (
                "Synthesis",
                vec!["synthesize", "synthesize-file", "batch", "interactive"],
            ),
            (
                "Advanced Voice Features",
                vec!["emotion", "clone", "convert", "sing", "spatial"],
            ),
            (
                "Voice Management",
                vec!["list-voices", "voice-info", "download-voice"],
            ),
            (
                "Model Management",
                vec![
                    "list-models",
                    "download-model",
                    "benchmark-models",
                    "optimize-model",
                ],
            ),
            ("System", vec!["config", "test", "server", "capabilities"]),
        ];

        for (category, commands) in categories {
            output.push_str(&format!("{}\n", style(category).bold().yellow()));

            for cmd in commands {
                if let Some(help) = self.get_command_help(cmd) {
                    output.push_str(&format!(
                        "  {} - {}\n",
                        style(cmd).green(),
                        help.description
                    ));
                }
            }
            output.push('\n');
        }

        output.push_str(&format!(
            "\n{}\n",
            style("Use 'voirs guide <command>' for detailed information about a specific command.")
                .dim()
        ));

        output
    }

    /// Format command help output
    fn format_command_help(&self, command: &str, help: &CommandHelp) -> String {
        let mut output = String::new();

        // Command title
        output.push_str(&format!(
            "{}\n",
            style(format!("voirs {}", command)).bold().cyan()
        ));
        output.push_str(&format!("{}\n\n", help.description));

        // Detailed description
        if !help.detailed.is_empty() {
            output.push_str(&format!("{}\n", style("Description:").bold().yellow()));
            output.push_str(&format!("{}\n\n", help.detailed));
        }

        // Examples
        if !help.examples.is_empty() {
            output.push_str(&format!("{}\n", style("Examples:").bold().yellow()));
            for example in &help.examples {
                output.push_str(&format!("  {}\n", style(&example.command).green()));
                output.push_str(&format!("    {}\n", example.description));

                if let Some(expected) = &example.expected {
                    output.push_str(&format!("    Expected: {}\n", style(expected).dim()));
                }
                output.push('\n');
            }
        }

        // Related commands
        if !help.related.is_empty() {
            output.push_str(&format!("{}\n", style("Related Commands:").bold().yellow()));
            for related in &help.related {
                output.push_str(&format!("  voirs {}\n", style(related).green()));
            }
            output.push('\n');
        }

        // Troubleshooting
        if !help.troubleshooting.is_empty() {
            output.push_str(&format!("{}\n", style("Troubleshooting:").bold().yellow()));
            for tip in &help.troubleshooting {
                let tip_style = match tip.level {
                    TipLevel::Info => Style::new().blue(),
                    TipLevel::Warning => Style::new().yellow(),
                    TipLevel::Error => Style::new().red(),
                };

                output.push_str(&format!(
                    "  {} {}\n",
                    tip_style.apply_to("•"),
                    style(&tip.title).bold()
                ));
                output.push_str(&format!("    {}\n", tip.content));
            }
        }

        output
    }

    /// Format response for unknown command
    fn format_unknown_command(&self, command: &str) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "{}\n\n",
            style(format!("Unknown command: '{}'", command))
                .red()
                .bold()
        ));

        // Suggest similar commands
        let similar = self.find_similar_commands(command);
        if !similar.is_empty() {
            output.push_str(&format!("{}\n", style("Did you mean:").yellow()));
            for suggestion in similar {
                output.push_str(&format!("  voirs {}\n", style(suggestion).green()));
            }
            output.push('\n');
        }

        output.push_str(&format!(
            "{}\n",
            style("Use 'voirs guide' to see all available commands.").dim()
        ));

        output
    }

    /// Find commands similar to the given input
    fn find_similar_commands(&self, input: &str) -> Vec<String> {
        let mut similar = Vec::new();

        for command in self.command_help.keys() {
            if command.contains(input) || input.contains(command) {
                similar.push(command.clone());
            }
        }

        // Limit to top 3 suggestions
        similar.truncate(3);
        similar
    }

    /// Load all command help content
    fn load_command_help(&mut self) {
        // Synthesis commands
        self.add_synthesize_help();
        self.add_synthesize_file_help();
        self.add_batch_help();
        self.add_interactive_help();

        // Advanced voice features
        self.add_emotion_help();
        self.add_clone_help();
        self.add_convert_help();
        self.add_sing_help();
        self.add_spatial_help();

        // Voice management
        self.add_list_voices_help();
        self.add_voice_info_help();
        self.add_download_voice_help();

        // Model management
        self.add_list_models_help();
        self.add_download_model_help();
        self.add_benchmark_models_help();
        self.add_optimize_model_help();

        // System commands
        self.add_config_help();
        self.add_test_help();
        self.add_server_help();
        self.add_guide_help();
        self.add_capabilities_help();
    }

    /// Load general tips and concepts
    fn load_general_tips(&mut self) {
        self.general_tips.extend(vec![
            HelpTip {
                title: "Getting Started".to_string(),
                content: "Run 'voirs test' to verify your installation and download a basic voice.".to_string(),
                level: TipLevel::Info,
            },
            HelpTip {
                title: "Configuration".to_string(),
                content: "Use 'voirs config --init' to create a configuration file with your preferences.".to_string(),
                level: TipLevel::Info,
            },
            HelpTip {
                title: "Performance".to_string(),
                content: "Enable GPU acceleration with --gpu flag for faster synthesis on supported hardware.".to_string(),
                level: TipLevel::Info,
            },
        ]);
    }

    // Helper methods to add specific command help
    fn add_synthesize_help(&mut self) {
        self.command_help.insert("synthesize".to_string(), CommandHelp {
            description: "Convert text to speech using neural synthesis".to_string(),
            detailed: "The synthesize command processes input text and generates high-quality speech audio. \
                      It supports various audio formats, speaking rates, pitch adjustments, and quality levels.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs synthesize \"Hello, world!\"".to_string(),
                    description: "Basic synthesis with default settings".to_string(),
                    expected: Some("Creates audio file with generated filename".to_string()),
                },
                HelpExample {
                    command: "voirs synthesize \"Hello, world!\" --output hello.wav".to_string(),
                    description: "Synthesize to specific output file".to_string(),
                    expected: Some("Creates hello.wav".to_string()),
                },
                HelpExample {
                    command: "voirs synthesize \"Fast speech\" --rate 1.5 --quality ultra".to_string(),
                    description: "Faster speech with highest quality".to_string(),
                    expected: None,
                },
            ],
            related: vec!["synthesize-file".to_string(), "batch".to_string(), "interactive".to_string()],
            troubleshooting: vec![
                HelpTip {
                    title: "Slow synthesis".to_string(),
                    content: "Try using --gpu flag for hardware acceleration or lower quality settings".to_string(),
                    level: TipLevel::Info,
                },
                HelpTip {
                    title: "Quality issues".to_string(),
                    content: "Use --quality ultra for best results, or check voice compatibility".to_string(),
                    level: TipLevel::Warning,
                },
            ],
        });
    }

    fn add_synthesize_file_help(&mut self) {
        self.command_help.insert("synthesize-file".to_string(), CommandHelp {
            description: "Synthesize text from input file".to_string(),
            detailed: "Process a text file and generate speech audio. The input file is read line by line, \
                      with each line treated as a separate synthesis request.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs synthesize-file input.txt".to_string(),
                    description: "Synthesize all lines from input.txt".to_string(),
                    expected: Some("Creates multiple audio files".to_string()),
                },
                HelpExample {
                    command: "voirs synthesize-file input.txt --output-dir ./audio/".to_string(),
                    description: "Output to specific directory".to_string(),
                    expected: None,
                },
            ],
            related: vec!["synthesize".to_string(), "batch".to_string()],
            troubleshooting: vec![],
        });
    }

    fn add_batch_help(&mut self) {
        self.command_help.insert("batch".to_string(), CommandHelp {
            description: "Process multiple texts with parallel synthesis".to_string(),
            detailed: "Batch processing supports various input formats (TXT, CSV, JSON, JSONL) and \
                      provides efficient parallel processing with progress tracking and resume capability.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs batch texts.csv --workers 4".to_string(),
                    description: "Process CSV file with 4 parallel workers".to_string(),
                    expected: None,
                },
                HelpExample {
                    command: "voirs batch data.jsonl --resume".to_string(),
                    description: "Resume interrupted batch processing".to_string(),
                    expected: None,
                },
            ],
            related: vec!["synthesize".to_string(), "synthesize-file".to_string()],
            troubleshooting: vec![
                HelpTip {
                    title: "Memory usage".to_string(),
                    content: "Reduce --workers count if experiencing high memory usage".to_string(),
                    level: TipLevel::Warning,
                },
            ],
        });
    }

    fn add_interactive_help(&mut self) {
        self.command_help.insert(
            "interactive".to_string(),
            CommandHelp {
                description: "Real-time interactive synthesis shell".to_string(),
                detailed:
                    "Interactive mode provides a shell-like interface for real-time text-to-speech \
                      synthesis with session management and live audio playback."
                        .to_string(),
                examples: vec![
                    HelpExample {
                        command: "voirs interactive".to_string(),
                        description: "Start interactive mode with default voice".to_string(),
                        expected: Some("Opens interactive shell".to_string()),
                    },
                    HelpExample {
                        command: "voirs interactive --voice en-us-female".to_string(),
                        description: "Start with specific voice".to_string(),
                        expected: None,
                    },
                ],
                related: vec!["synthesize".to_string()],
                troubleshooting: vec![HelpTip {
                    title: "Audio playback issues".to_string(),
                    content:
                        "Use --no-audio flag to disable playback if experiencing audio problems"
                            .to_string(),
                    level: TipLevel::Warning,
                }],
            },
        );
    }

    fn add_list_voices_help(&mut self) {
        self.command_help.insert(
            "list-voices".to_string(),
            CommandHelp {
                description: "Display available voices".to_string(),
                detailed: "List all installed and available voices with their characteristics, \
                      languages, and installation status."
                    .to_string(),
                examples: vec![
                    HelpExample {
                        command: "voirs list-voices".to_string(),
                        description: "Show all available voices".to_string(),
                        expected: None,
                    },
                    HelpExample {
                        command: "voirs list-voices --language en".to_string(),
                        description: "Filter by English voices only".to_string(),
                        expected: None,
                    },
                ],
                related: vec!["voice-info".to_string(), "download-voice".to_string()],
                troubleshooting: vec![],
            },
        );
    }

    fn add_voice_info_help(&mut self) {
        self.command_help.insert(
            "voice-info".to_string(),
            CommandHelp {
                description: "Get detailed information about a specific voice".to_string(),
                detailed: "Display comprehensive information about a voice including metadata, \
                      quality metrics, and usage examples."
                    .to_string(),
                examples: vec![HelpExample {
                    command: "voirs voice-info en-us-female".to_string(),
                    description: "Show details for English US female voice".to_string(),
                    expected: None,
                }],
                related: vec!["list-voices".to_string(), "download-voice".to_string()],
                troubleshooting: vec![],
            },
        );
    }

    fn add_download_voice_help(&mut self) {
        self.command_help.insert(
            "download-voice".to_string(),
            CommandHelp {
                description: "Download and install a voice model".to_string(),
                detailed: "Download voice models from the repository with progress tracking \
                      and checksum verification."
                    .to_string(),
                examples: vec![HelpExample {
                    command: "voirs download-voice en-us-female".to_string(),
                    description: "Download English US female voice".to_string(),
                    expected: Some("Downloads and installs voice model".to_string()),
                }],
                related: vec!["list-voices".to_string(), "voice-info".to_string()],
                troubleshooting: vec![HelpTip {
                    title: "Download fails".to_string(),
                    content: "Check internet connection and available disk space".to_string(),
                    level: TipLevel::Error,
                }],
            },
        );
    }

    fn add_list_models_help(&mut self) {
        self.command_help.insert(
            "list-models".to_string(),
            CommandHelp {
                description: "Display available synthesis models".to_string(),
                detailed: "List acoustic models, vocoders, and other synthesis components \
                      with compatibility and performance information."
                    .to_string(),
                examples: vec![HelpExample {
                    command: "voirs list-models".to_string(),
                    description: "Show all available models".to_string(),
                    expected: None,
                }],
                related: vec!["download-model".to_string(), "benchmark-models".to_string()],
                troubleshooting: vec![],
            },
        );
    }

    fn add_download_model_help(&mut self) {
        self.command_help.insert(
            "download-model".to_string(),
            CommandHelp {
                description: "Download and install synthesis models".to_string(),
                detailed: "Download acoustic models, vocoders, and other synthesis components \
                      required for text-to-speech synthesis."
                    .to_string(),
                examples: vec![HelpExample {
                    command: "voirs download-model tacotron2-en".to_string(),
                    description: "Download Tacotron2 English model".to_string(),
                    expected: None,
                }],
                related: vec!["list-models".to_string(), "optimize-model".to_string()],
                troubleshooting: vec![],
            },
        );
    }

    fn add_benchmark_models_help(&mut self) {
        self.command_help.insert(
            "benchmark-models".to_string(),
            CommandHelp {
                description: "Benchmark model performance".to_string(),
                detailed: "Run performance tests on synthesis models to measure speed, \
                      quality, and resource usage."
                    .to_string(),
                examples: vec![HelpExample {
                    command: "voirs benchmark-models tacotron2-en hifigan-en".to_string(),
                    description: "Benchmark multiple models".to_string(),
                    expected: None,
                }],
                related: vec!["list-models".to_string(), "optimize-model".to_string()],
                troubleshooting: vec![],
            },
        );
    }

    fn add_optimize_model_help(&mut self) {
        self.command_help.insert(
            "optimize-model".to_string(),
            CommandHelp {
                description: "Optimize models for current hardware".to_string(),
                detailed: "Apply hardware-specific optimizations to improve synthesis \
                      performance and reduce memory usage."
                    .to_string(),
                examples: vec![HelpExample {
                    command: "voirs optimize-model tacotron2-en".to_string(),
                    description: "Optimize model for current hardware".to_string(),
                    expected: None,
                }],
                related: vec!["benchmark-models".to_string(), "list-models".to_string()],
                troubleshooting: vec![],
            },
        );
    }

    fn add_config_help(&mut self) {
        self.command_help.insert(
            "config".to_string(),
            CommandHelp {
                description: "Manage configuration settings".to_string(),
                detailed: "View, modify, and manage VoiRS configuration including \
                      default voices, quality settings, and preferences."
                    .to_string(),
                examples: vec![
                    HelpExample {
                        command: "voirs config --show".to_string(),
                        description: "Display current configuration".to_string(),
                        expected: None,
                    },
                    HelpExample {
                        command: "voirs config --init".to_string(),
                        description: "Create default configuration file".to_string(),
                        expected: Some("Creates config.toml".to_string()),
                    },
                ],
                related: vec![],
                troubleshooting: vec![],
            },
        );
    }

    fn add_test_help(&mut self) {
        self.command_help.insert(
            "test".to_string(),
            CommandHelp {
                description: "Test synthesis pipeline and installation".to_string(),
                detailed: "Verify that VoiRS is properly installed and configured \
                      by running a test synthesis."
                    .to_string(),
                examples: vec![
                    HelpExample {
                        command: "voirs test".to_string(),
                        description: "Run basic synthesis test".to_string(),
                        expected: Some("Creates test audio file".to_string()),
                    },
                    HelpExample {
                        command: "voirs test --play".to_string(),
                        description: "Test with audio playback".to_string(),
                        expected: Some("Plays synthesized audio".to_string()),
                    },
                ],
                related: vec!["config".to_string()],
                troubleshooting: vec![HelpTip {
                    title: "Test fails".to_string(),
                    content: "Check that voices and models are installed with 'voirs list-voices'"
                        .to_string(),
                    level: TipLevel::Error,
                }],
            },
        );
    }

    fn add_server_help(&mut self) {
        self.command_help.insert(
            "server".to_string(),
            CommandHelp {
                description: "Run VoiRS as HTTP API server".to_string(),
                detailed: "Start a production-ready HTTP server providing REST API \
                      endpoints for synthesis, voice management, and system status."
                    .to_string(),
                examples: vec![
                    HelpExample {
                        command: "voirs server".to_string(),
                        description: "Start server on default port 8080".to_string(),
                        expected: Some("Server runs at http://127.0.0.1:8080".to_string()),
                    },
                    HelpExample {
                        command: "voirs server --port 3000 --host 0.0.0.0".to_string(),
                        description: "Start server on custom address".to_string(),
                        expected: None,
                    },
                ],
                related: vec![],
                troubleshooting: vec![HelpTip {
                    title: "Port already in use".to_string(),
                    content: "Use --port flag to specify a different port number".to_string(),
                    level: TipLevel::Error,
                }],
            },
        );
    }

    fn add_guide_help(&mut self) {
        self.command_help.insert("guide".to_string(), CommandHelp {
            description: "Show detailed help and guides".to_string(),
            detailed: "Display comprehensive help information, getting started guides, \
                      and examples for VoiRS commands. Use this to learn about specific \
                      commands or get an overview of available functionality.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs guide".to_string(),
                    description: "Show overview of all commands".to_string(),
                    expected: Some("Lists all command categories".to_string()),
                },
                HelpExample {
                    command: "voirs guide synthesize".to_string(),
                    description: "Get detailed help for synthesize command".to_string(),
                    expected: Some("Shows usage, examples, and troubleshooting".to_string()),
                },
                HelpExample {
                    command: "voirs guide --getting-started".to_string(),
                    description: "Show getting started guide".to_string(),
                    expected: Some("Step-by-step setup instructions".to_string()),
                },
                HelpExample {
                    command: "voirs guide --examples".to_string(),
                    description: "Show examples for all commands".to_string(),
                    expected: Some("Command examples with descriptions".to_string()),
                },
            ],
            related: vec!["test".to_string(), "config".to_string()],
            troubleshooting: vec![
                HelpTip {
                    title: "Need more help".to_string(),
                    content: "Visit the documentation or check the project repository for additional resources".to_string(),
                    level: TipLevel::Info,
                },
            ],
        });
    }

    // Advanced voice features help
    fn add_emotion_help(&mut self) {
        self.command_help.insert("emotion".to_string(), CommandHelp {
            description: "Control emotional expression in synthesized speech".to_string(),
            detailed: "The emotion command allows you to modify the emotional tone and expression \
                      of synthesized speech. You can use predefined emotion presets, blend multiple \
                      emotions, or create custom emotion configurations.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs emotion list".to_string(),
                    description: "List all available emotion presets".to_string(),
                    expected: Some("Shows emotions like happy, sad, angry, calm, etc.".to_string()),
                },
                HelpExample {
                    command: "voirs emotion synth --emotion happy --intensity 0.7 \"Hello world!\" output.wav".to_string(),
                    description: "Synthesize with happy emotion at 70% intensity".to_string(),
                    expected: Some("Creates emotionally expressive audio".to_string()),
                },
                HelpExample {
                    command: "voirs emotion blend --emotions happy,calm --weights 0.6,0.4 \"Text\" output.wav".to_string(),
                    description: "Blend multiple emotions with specified weights".to_string(),
                    expected: None,
                },
                HelpExample {
                    command: "voirs emotion create-preset --name custom --config emotion.json".to_string(),
                    description: "Create a custom emotion preset".to_string(),
                    expected: Some("Saves custom emotion configuration".to_string()),
                },
            ],
            related: vec!["synthesize".to_string(), "convert".to_string()],
            troubleshooting: vec![
                HelpTip {
                    title: "Emotion not working".to_string(),
                    content: "Check if emotion models are downloaded and the voice supports emotional expression".to_string(),
                    level: TipLevel::Warning,
                },
                HelpTip {
                    title: "Intensity too high".to_string(),
                    content: "Reduce intensity value (0.0-1.0) if emotion sounds unnatural".to_string(),
                    level: TipLevel::Info,
                },
            ],
        });
    }

    fn add_clone_help(&mut self) {
        self.command_help.insert("clone".to_string(), CommandHelp {
            description: "Clone voices from reference audio samples".to_string(),
            detailed: "Voice cloning allows you to create new voices from reference audio samples. \
                      You can use multiple reference files for better quality, and the system \
                      supports both quick cloning and detailed speaker adaptation.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs clone clone --reference-files voice_samples/*.wav --text \"Hello world\" output.wav".to_string(),
                    description: "Clone voice from multiple reference files".to_string(),
                    expected: Some("Creates audio with cloned voice".to_string()),
                },
                HelpExample {
                    command: "voirs clone quick --reference-files sample.wav --text \"Hello world\" output.wav".to_string(),
                    description: "Quick cloning from single reference file".to_string(),
                    expected: Some("Faster but potentially lower quality cloning".to_string()),
                },
                HelpExample {
                    command: "voirs clone list-profiles".to_string(),
                    description: "List cached speaker profiles".to_string(),
                    expected: Some("Shows previously cloned speaker profiles".to_string()),
                },
                HelpExample {
                    command: "voirs clone validate --reference-files samples/*.wav".to_string(),
                    description: "Validate reference audio quality for cloning".to_string(),
                    expected: Some("Reports audio quality and suitability".to_string()),
                },
            ],
            related: vec!["convert".to_string(), "synthesize".to_string()],
            troubleshooting: vec![
                HelpTip {
                    title: "Poor cloning quality".to_string(),
                    content: "Use high-quality reference audio (16kHz+) with clear speech and minimal background noise".to_string(),
                    level: TipLevel::Warning,
                },
                HelpTip {
                    title: "Cloning fails".to_string(),
                    content: "Ensure reference files are at least 30 seconds of clear speech from the target speaker".to_string(),
                    level: TipLevel::Error,
                },
            ],
        });
    }

    fn add_convert_help(&mut self) {
        self.command_help.insert("convert".to_string(), CommandHelp {
            description: "Convert and transform voice characteristics".to_string(),
            detailed: "Voice conversion allows you to transform existing audio by changing \
                      speaker characteristics, age, gender, or other vocal properties. \
                      Supports both file-based and real-time conversion.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs convert speaker --input source.wav --target-speaker target_voice --output converted.wav".to_string(),
                    description: "Convert to a different speaker".to_string(),
                    expected: Some("Creates audio with target speaker's voice".to_string()),
                },
                HelpExample {
                    command: "voirs convert age --input voice.wav --target-age 25 --output aged.wav".to_string(),
                    description: "Change apparent age of the speaker".to_string(),
                    expected: Some("Creates audio with modified age characteristics".to_string()),
                },
                HelpExample {
                    command: "voirs convert gender --input voice.wav --target male --output converted.wav".to_string(),
                    description: "Convert gender characteristics".to_string(),
                    expected: Some("Creates audio with modified gender characteristics".to_string()),
                },
                HelpExample {
                    command: "voirs convert morph --voice1 voice1.model --voice2 voice2.model --ratio 0.5".to_string(),
                    description: "Morph between two voice models".to_string(),
                    expected: Some("Creates blended voice characteristics".to_string()),
                },
            ],
            related: vec!["clone".to_string(), "emotion".to_string()],
            troubleshooting: vec![
                HelpTip {
                    title: "Conversion artifacts".to_string(),
                    content: "Reduce conversion intensity or use higher quality settings".to_string(),
                    level: TipLevel::Warning,
                },
                HelpTip {
                    title: "Real-time conversion slow".to_string(),
                    content: "Use GPU acceleration or reduce audio quality for real-time processing".to_string(),
                    level: TipLevel::Info,
                },
            ],
        });
    }

    fn add_sing_help(&mut self) {
        self.command_help.insert("sing".to_string(), CommandHelp {
            description: "Synthesize singing voice from musical scores".to_string(),
            detailed: "Singing voice synthesis converts musical scores and lyrics into \
                      realistic singing. Supports various input formats including MusicXML, \
                      MIDI, and custom score formats with advanced vocal technique controls.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs sing score --score score.musicxml --voice singer.model --output song.wav".to_string(),
                    description: "Synthesize from MusicXML score".to_string(),
                    expected: Some("Creates singing audio from musical score".to_string()),
                },
                HelpExample {
                    command: "voirs sing midi --midi input.mid --lyrics lyrics.txt --voice singer.model --output song.wav".to_string(),
                    description: "Synthesize from MIDI file with lyrics".to_string(),
                    expected: Some("Creates singing audio from MIDI and lyrics".to_string()),
                },
                HelpExample {
                    command: "voirs sing create-voice --samples singing_samples/ --output singer.model".to_string(),
                    description: "Create singing voice model from samples".to_string(),
                    expected: Some("Trains custom singing voice model".to_string()),
                },
                HelpExample {
                    command: "voirs sing effects --input song.wav --vibrato 1.2 --expression happy --output processed.wav".to_string(),
                    description: "Apply singing effects and expression".to_string(),
                    expected: Some("Adds vocal effects to singing audio".to_string()),
                },
            ],
            related: vec!["emotion".to_string(), "synthesize".to_string()],
            troubleshooting: vec![
                HelpTip {
                    title: "Pitch issues".to_string(),
                    content: "Check score format and ensure correct pitch notation".to_string(),
                    level: TipLevel::Warning,
                },
                HelpTip {
                    title: "Timing problems".to_string(),
                    content: "Verify tempo and rhythm marks in the musical score".to_string(),
                    level: TipLevel::Warning,
                },
            ],
        });
    }

    fn add_spatial_help(&mut self) {
        self.command_help.insert("spatial".to_string(), CommandHelp {
            description: "Create 3D spatial audio with positioning effects".to_string(),
            detailed: "Spatial audio synthesis creates immersive 3D soundscapes by applying \
                      Head-Related Transfer Functions (HRTF) and room acoustics. Supports \
                      binaural rendering for headphones and multi-channel output for speakers.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs spatial synth --text \"hello\" --position 1,0,0 --output 3d_audio.wav".to_string(),
                    description: "Synthesize with 3D positioning".to_string(),
                    expected: Some("Creates 3D positioned audio".to_string()),
                },
                HelpExample {
                    command: "voirs spatial hrtf --input mono.wav --position x,y,z --output binaural.wav".to_string(),
                    description: "Apply HRTF processing to existing audio".to_string(),
                    expected: Some("Creates binaural audio for headphones".to_string()),
                },
                HelpExample {
                    command: "voirs spatial room --input voice.wav --room-config room.json --output spatial.wav".to_string(),
                    description: "Apply room acoustics simulation".to_string(),
                    expected: Some("Creates audio with room acoustics".to_string()),
                },
                HelpExample {
                    command: "voirs spatial movement --input voice.wav --path movement.json --output dynamic.wav".to_string(),
                    description: "Apply dynamic movement path".to_string(),
                    expected: Some("Creates audio with moving sound source".to_string()),
                },
            ],
            related: vec!["synthesize".to_string()],
            troubleshooting: vec![
                HelpTip {
                    title: "HRTF not working".to_string(),
                    content: "Check if HRTF dataset is installed and compatible with your setup".to_string(),
                    level: TipLevel::Warning,
                },
                HelpTip {
                    title: "Poor spatial effect".to_string(),
                    content: "Use high-quality headphones and ensure proper audio driver configuration".to_string(),
                    level: TipLevel::Info,
                },
            ],
        });
    }

    fn add_capabilities_help(&mut self) {
        self.command_help.insert("capabilities".to_string(), CommandHelp {
            description: "Show system capabilities and feature availability".to_string(),
            detailed: "The capabilities command provides comprehensive information about \
                      available features, system requirements, configuration status, and \
                      hardware capabilities. Use this to troubleshoot issues and verify \
                      your VoiRS installation.".to_string(),
            examples: vec![
                HelpExample {
                    command: "voirs capabilities list".to_string(),
                    description: "Show all available features and their status".to_string(),
                    expected: Some("Lists features with availability status".to_string()),
                },
                HelpExample {
                    command: "voirs capabilities check emotion".to_string(),
                    description: "Check if emotion control feature is available".to_string(),
                    expected: Some("Shows emotion feature status and requirements".to_string()),
                },
                HelpExample {
                    command: "voirs capabilities requirements".to_string(),
                    description: "Show system requirements for all features".to_string(),
                    expected: Some("Lists hardware and software requirements".to_string()),
                },
                HelpExample {
                    command: "voirs capabilities test cloning".to_string(),
                    description: "Test voice cloning functionality".to_string(),
                    expected: Some("Runs functional tests for voice cloning".to_string()),
                },
            ],
            related: vec!["test".to_string(), "config".to_string()],
            troubleshooting: vec![
                HelpTip {
                    title: "Feature unavailable".to_string(),
                    content: "Check if required models are installed and feature is enabled in configuration".to_string(),
                    level: TipLevel::Warning,
                },
                HelpTip {
                    title: "System requirements".to_string(),
                    content: "Use 'voirs capabilities requirements' to check specific hardware needs".to_string(),
                    level: TipLevel::Info,
                },
            ],
        });
    }
}

impl Default for HelpSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Display help for getting started with VoiRS
pub fn display_getting_started() -> String {
    let mut output = String::new();

    output.push_str(&format!(
        "{}\n\n",
        style("Getting Started with VoiRS").bold().cyan()
    ));

    let steps = vec![
        (
            "1. Test Installation",
            "voirs test",
            "Verify VoiRS is working correctly",
        ),
        (
            "2. List Voices",
            "voirs list-voices",
            "See available voices",
        ),
        (
            "3. Download a Voice",
            "voirs download-voice en-us-female",
            "Get a voice for synthesis",
        ),
        (
            "4. Synthesize Text",
            "voirs synthesize \"Hello, world!\"",
            "Create your first audio",
        ),
        (
            "5. Interactive Mode",
            "voirs interactive",
            "Try real-time synthesis",
        ),
    ];

    for (step, command, description) in steps {
        output.push_str(&format!("{}\n", style(step).bold().yellow()));
        output.push_str(&format!("  {}\n", style(command).green()));
        output.push_str(&format!("  {}\n\n", description));
    }

    output.push_str(&format!(
        "{}\n",
        style("Use 'voirs help <command>' for detailed information about specific commands.").dim()
    ));

    output
}
