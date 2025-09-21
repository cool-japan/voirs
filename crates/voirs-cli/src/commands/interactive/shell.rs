//! Interactive shell implementation for VoiRS CLI
//!
//! Provides a command-line interface with:
//! - History support
//! - Tab completion for commands and voices
//! - Context-aware suggestions
//! - Session state management

use super::{
    commands::CommandProcessor, session::SessionManager, synthesis::SynthesisEngine,
    InteractiveOptions,
};
use crate::config::Config;
use crate::error::{Result, VoirsCliError};
use console::{style, Term};
use dialoguer::{theme::ColorfulTheme, Input};
use std::collections::HashMap;
use std::io::Write;

/// Interactive shell state
pub struct InteractiveShell {
    /// Synthesis engine for real-time TTS
    synthesis_engine: SynthesisEngine,

    /// Session manager for state persistence
    session_manager: SessionManager,

    /// Command processor for interactive commands
    command_processor: CommandProcessor,

    /// Command history
    history: Vec<String>,

    /// Current session state
    current_voice: Option<String>,
    current_speed: f32,
    current_pitch: f32,
    current_volume: f32,

    /// Shell options
    options: InteractiveOptions,

    /// Terminal interface
    term: Term,

    /// Available voices cache
    available_voices: Vec<String>,

    /// Running state
    running: bool,
}

impl InteractiveShell {
    /// Create a new interactive shell
    pub async fn new(options: InteractiveOptions) -> Result<Self> {
        let term = Term::stdout();

        // Initialize synthesis engine
        let synthesis_engine = SynthesisEngine::new().await?;

        // Initialize session manager
        let mut session_manager = SessionManager::new(options.auto_save);

        // Load session if specified
        if let Some(session_path) = &options.load_session {
            session_manager.load_session(session_path).await?;
        }

        // Get available voices
        let available_voices = synthesis_engine.list_voices().await?;

        // Initialize command processor
        let command_processor = CommandProcessor::new(available_voices.clone());

        // Set initial voice
        let current_voice = options
            .voice
            .clone()
            .or_else(|| session_manager.get_current_voice().cloned())
            .or_else(|| available_voices.first().cloned());

        let mut shell = Self {
            synthesis_engine,
            session_manager,
            command_processor,
            history: Vec::new(),
            current_voice: current_voice.clone(),
            current_speed: 1.0,
            current_pitch: 0.0,
            current_volume: 1.0,
            options,
            term,
            available_voices,
            running: true,
        };

        // Set initial voice in synthesis engine
        if let Some(voice) = current_voice {
            shell.synthesis_engine.set_voice(&voice).await?;
        }

        Ok(shell)
    }

    /// Run the main interactive loop
    pub async fn run(&mut self) -> Result<()> {
        self.print_welcome();
        self.print_help_hint();

        while self.running {
            match self.read_command().await {
                Ok(command) => {
                    if let Err(e) = self.process_command(&command).await {
                        self.print_error(&e);
                    }
                }
                Err(e) => {
                    if self.should_exit_on_error(&e) {
                        break;
                    }
                    self.print_error(&e);
                }
            }
        }

        self.print_goodbye();
        Ok(())
    }

    /// Read a command from user input
    async fn read_command(&mut self) -> Result<String> {
        let prompt = self.create_prompt();

        // Check if we're in a non-terminal environment (like tests)
        if !console::Term::stdout().is_term() {
            // In non-terminal mode, read from stdin using standard input
            use std::io::{self, BufRead};
            let stdin = io::stdin();
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => {
                    // EOF reached, exit gracefully
                    self.running = false;
                    return Ok("quit".to_string());
                }
                Ok(_) => {
                    let input = line.trim().to_string();
                    if input == "quit" || input == "exit" {
                        self.running = false;
                    }
                    return Ok(input);
                }
                Err(e) => {
                    return Err(VoirsCliError::IoError(format!("Input error: {}", e)));
                }
            }
        }

        let input: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt(&prompt)
            .interact_text()
            .map_err(|e| VoirsCliError::IoError(format!("Input error: {}", e)))?;

        // Add to history if not empty and not a duplicate
        if !input.trim().is_empty() && self.history.last() != Some(&input) {
            self.history.push(input.clone());

            // Limit history size
            if self.history.len() > 1000 {
                self.history.remove(0);
            }
        }

        Ok(input)
    }

    /// Process a user command
    async fn process_command(&mut self, command: &str) -> Result<()> {
        let command = command.trim();

        if command.is_empty() {
            return Ok(());
        }

        // Check for interactive commands (starting with :)
        if command.starts_with(':') {
            return self.handle_shell_command(command).await;
        }

        // Regular text for synthesis
        self.synthesize_text(command).await?;

        // Add to session history
        self.session_manager
            .add_synthesis(command, &self.current_voice);

        Ok(())
    }

    /// Handle shell commands (starting with :)
    async fn handle_shell_command(&mut self, command: &str) -> Result<()> {
        // We need to extract the command processor to avoid borrowing self multiple ways
        let available_voices = self.command_processor.available_voices().clone();
        let temp_processor = CommandProcessor::new(available_voices);
        temp_processor.process_command(self, command).await
    }

    /// Synthesize text and optionally play audio
    async fn synthesize_text(&mut self, text: &str) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Show synthesis indicator
        self.print_status(&format!("Synthesizing: \"{}\"", text));

        // Perform synthesis
        let audio_data = self.synthesis_engine.synthesize(text).await?;

        let synthesis_time = start_time.elapsed();

        // Play audio if not disabled
        if !self.options.no_audio {
            self.synthesis_engine.play_audio(&audio_data).await?;
        }

        // Show completion status
        self.print_status(&format!(
            "✓ Synthesis completed in {:.2}s ({} samples)",
            synthesis_time.as_secs_f64(),
            audio_data.len()
        ));

        Ok(())
    }

    /// Create the command prompt
    fn create_prompt(&self) -> String {
        let voice_part = if let Some(ref voice) = self.current_voice {
            format!("{}@{}", style("voirs").cyan(), style(voice).green())
        } else {
            style("voirs").cyan().to_string()
        };

        let params =
            if self.current_speed != 1.0 || self.current_pitch != 0.0 || self.current_volume != 1.0
            {
                format!(
                    " [s:{:.1} p:{:.1} v:{:.1}]",
                    self.current_speed, self.current_pitch, self.current_volume
                )
            } else {
                String::new()
            };

        format!("{}{}> ", voice_part, style(&params).dim())
    }

    /// Provide tab completion suggestions
    fn complete_input(&self, input: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if input.starts_with(':') {
            // Command completion
            let commands = [
                ":help", ":voice", ":voices", ":speed", ":pitch", ":volume", ":save", ":load",
                ":history", ":clear", ":status", ":quit", ":exit",
            ];

            for cmd in &commands {
                if cmd.starts_with(input) {
                    suggestions.push(cmd.to_string());
                }
            }
        } else if input.contains(" ") && input.starts_with(":voice ") {
            // Voice name completion
            let voice_prefix = input.strip_prefix(":voice ").unwrap_or("");
            for voice in &self.available_voices {
                if voice.starts_with(voice_prefix) {
                    suggestions.push(format!(":voice {}", voice));
                }
            }
        }

        suggestions
    }

    /// Print welcome message
    fn print_welcome(&self) {
        println!(
            "{}",
            style("Welcome to VoiRS Interactive Mode").bold().cyan()
        );
        println!("Type text to synthesize, or use :help for commands");

        if let Some(ref voice) = self.current_voice {
            println!("Current voice: {}", style(voice).green());
        }

        println!();
    }

    /// Print help hint
    fn print_help_hint(&self) {
        println!(
            "{}",
            style("Hint: Type ':help' for available commands").dim()
        );
        println!();
    }

    /// Print goodbye message
    fn print_goodbye(&self) {
        println!("\\n{}", style("Goodbye! 👋").cyan());
    }

    /// Print status message
    fn print_status(&self, message: &str) {
        println!("{} {}", style("ℹ").blue(), message);
    }

    /// Print error message
    fn print_error(&self, error: &VoirsCliError) {
        eprintln!("{} {}", style("✗").red(), style(error).red());
    }

    /// Check if we should exit on this error
    fn should_exit_on_error(&self, _error: &VoirsCliError) -> bool {
        // For now, don't exit on any errors - keep the shell running
        false
    }

    /// Get current voice
    pub fn current_voice(&self) -> Option<&str> {
        self.current_voice.as_deref()
    }

    /// Set current voice
    pub async fn set_voice(&mut self, voice: String) -> Result<()> {
        self.synthesis_engine.set_voice(&voice).await?;
        self.current_voice = Some(voice.clone());
        self.session_manager.set_current_voice(voice);
        Ok(())
    }

    /// Get available voices
    pub fn available_voices(&self) -> &[String] {
        &self.available_voices
    }

    /// Get current synthesis parameters
    pub fn current_params(&self) -> (f32, f32, f32) {
        (self.current_speed, self.current_pitch, self.current_volume)
    }

    /// Set synthesis parameters
    pub async fn set_params(
        &mut self,
        speed: Option<f32>,
        pitch: Option<f32>,
        volume: Option<f32>,
    ) -> Result<()> {
        if let Some(s) = speed {
            self.current_speed = s.clamp(0.1, 3.0);
            self.synthesis_engine.set_speed(self.current_speed).await?;
        }

        if let Some(p) = pitch {
            self.current_pitch = p.clamp(-12.0, 12.0);
            self.synthesis_engine.set_pitch(self.current_pitch).await?;
        }

        if let Some(v) = volume {
            self.current_volume = v.clamp(0.0, 2.0);
            self.synthesis_engine
                .set_volume(self.current_volume)
                .await?;
        }

        Ok(())
    }

    /// Get session manager
    pub fn session_manager(&mut self) -> &mut SessionManager {
        &mut self.session_manager
    }

    /// Exit the shell
    pub fn exit(&mut self) {
        self.running = false;
    }
}
