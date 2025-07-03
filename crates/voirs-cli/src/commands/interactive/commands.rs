//! Interactive command processor
//!
//! Handles commands that start with ':' in interactive mode:
//! - Voice changing (:voice)
//! - Parameter adjustment (:speed, :pitch, :volume)
//! - Session operations (:save, :load, :history)
//! - Utility commands (:help, :status, :quit)

use crate::error::{Result, VoirsCliError};
use super::{shell::InteractiveShell, session::ExportFormat};
use console::{style, Term};
use std::path::PathBuf;

/// Command processor for interactive shell commands
pub struct CommandProcessor {
    /// Available voices for validation
    available_voices: Vec<String>,
}

impl CommandProcessor {
    /// Create a new command processor
    pub fn new(available_voices: Vec<String>) -> Self {
        Self {
            available_voices,
        }
    }
    
    /// Process an interactive command
    pub async fn process_command(&self, shell: &mut InteractiveShell, command: &str) -> Result<()> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }
        
        let cmd = parts[0];
        let args = &parts[1..];
        
        match cmd {
            ":help" | ":h" => self.show_help(),
            ":voice" | ":v" => self.handle_voice_command(shell, args).await,
            ":voices" => self.list_voices(),
            ":speed" | ":s" => self.handle_speed_command(shell, args).await,
            ":pitch" | ":p" => self.handle_pitch_command(shell, args).await,
            ":volume" | ":vol" => self.handle_volume_command(shell, args).await,
            ":save" => self.handle_save_command(shell, args).await,
            ":load" => self.handle_load_command(shell, args).await,
            ":history" | ":hist" => self.show_history(shell, args),
            ":clear" => self.clear_screen(),
            ":status" | ":stat" => self.show_status(shell),
            ":export" => self.handle_export_command(shell, args).await,
            ":quit" | ":exit" | ":q" => self.handle_quit(shell),
            _ => self.handle_unknown_command(cmd),
        }
    }
    
    /// Show help information
    fn show_help(&self) -> Result<()> {
        println!("\\n{}", style("VoiRS Interactive Mode - Commands").bold().cyan());
        println!("{}",   style("=====================================").cyan());
        println!();
        
        println!("{}", style("Voice Commands:").bold());
        println!("  :voice <name>     Set current voice");
        println!("  :voices           List available voices");
        println!();
        
        println!("{}", style("Parameter Commands:").bold());
        println!("  :speed <value>    Set synthesis speed (0.1-3.0)");
        println!("  :pitch <value>    Set pitch adjustment (-12.0 to 12.0 semitones)");
        println!("  :volume <value>   Set volume (0.0-2.0)");
        println!();
        
        println!("{}", style("Session Commands:").bold());
        println!("  :save [file]      Save current session");
        println!("  :load <file>      Load session from file");
        println!("  :history [N]      Show synthesis history (last N entries)");
        println!("  :export <format> <file>  Export history (json/csv/text)");
        println!();
        
        println!("{}", style("Utility Commands:").bold());
        println!("  :status           Show current status and settings");
        println!("  :clear            Clear screen");
        println!("  :help             Show this help");
        println!("  :quit             Exit interactive mode");
        println!();
        
        println!("{}", style("Usage Examples:").dim());
        println!("  Hello world                    # Synthesize text");
        println!("  :voice en-us-female-01         # Change voice");
        println!("  :speed 1.2                     # Speak 20% faster");
        println!("  :pitch 2.0                     # Raise pitch by 2 semitones");
        println!("  :save my_session.json          # Save session");
        println!("  :history 10                    # Show last 10 syntheses");
        println!();
        
        Ok(())
    }
    
    /// Handle voice command
    async fn handle_voice_command(&self, shell: &mut InteractiveShell, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            if let Some(current) = shell.current_voice() {
                println!("Current voice: {}", style(current).green());
            } else {
                println!("No voice currently selected");
            }
            return Ok(());
        }
        
        let voice = args[0];
        
        // Validate voice
        if !self.available_voices.contains(&voice.to_string()) {
            return Err(VoirsCliError::VoiceError(format!(
                "Voice '{}' not found. Use ':voices' to see available voices.",
                voice
            )));
        }
        
        shell.set_voice(voice.to_string()).await?;
        
        Ok(())
    }
    
    /// List available voices
    fn list_voices(&self) -> Result<()> {
        println!("\\n{}", style("Available Voices:").bold());
        for (i, voice) in self.available_voices.iter().enumerate() {
            println!("  {}. {}", i + 1, style(voice).green());
        }
        println!();
        
        Ok(())
    }
    
    /// Handle speed command
    async fn handle_speed_command(&self, shell: &mut InteractiveShell, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            let (speed, _, _) = shell.current_params();
            println!("Current speed: {:.1}x", speed);
            return Ok(());
        }
        
        let speed: f32 = args[0].parse().map_err(|_| {
            VoirsCliError::InvalidArgument(format!(
                "Invalid speed value '{}'. Expected a number between 0.1 and 3.0",
                args[0]
            ))
        })?;
        
        if !(0.1..=3.0).contains(&speed) {
            return Err(VoirsCliError::InvalidArgument(
                "Speed must be between 0.1 and 3.0".to_string()
            ));
        }
        
        shell.set_params(Some(speed), None, None).await?;
        
        Ok(())
    }
    
    /// Handle pitch command
    async fn handle_pitch_command(&self, shell: &mut InteractiveShell, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            let (_, pitch, _) = shell.current_params();
            println!("Current pitch: {:.1} semitones", pitch);
            return Ok(());
        }
        
        let pitch: f32 = args[0].parse().map_err(|_| {
            VoirsCliError::InvalidArgument(format!(
                "Invalid pitch value '{}'. Expected a number between -12.0 and 12.0",
                args[0]
            ))
        })?;
        
        if !(-12.0..=12.0).contains(&pitch) {
            return Err(VoirsCliError::InvalidArgument(
                "Pitch must be between -12.0 and 12.0 semitones".to_string()
            ));
        }
        
        shell.set_params(None, Some(pitch), None).await?;
        
        Ok(())
    }
    
    /// Handle volume command
    async fn handle_volume_command(&self, shell: &mut InteractiveShell, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            let (_, _, volume) = shell.current_params();
            println!("Current volume: {:.1}", volume);
            return Ok(());
        }
        
        let volume: f32 = args[0].parse().map_err(|_| {
            VoirsCliError::InvalidArgument(format!(
                "Invalid volume value '{}'. Expected a number between 0.0 and 2.0",
                args[0]
            ))
        })?;
        
        if !(0.0..=2.0).contains(&volume) {
            return Err(VoirsCliError::InvalidArgument(
                "Volume must be between 0.0 and 2.0".to_string()
            ));
        }
        
        shell.set_params(None, None, Some(volume)).await?;
        
        Ok(())
    }
    
    /// Handle save command
    async fn handle_save_command(&self, shell: &mut InteractiveShell, args: &[&str]) -> Result<()> {
        let filename = if args.is_empty() {
            // Generate default filename
            let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            format!("voirs_session_{}.json", timestamp)
        } else {
            args[0].to_string()
        };
        
        let path = PathBuf::from(filename);
        shell.session_manager().save_session(&path).await?;
        
        Ok(())
    }
    
    /// Handle load command
    async fn handle_load_command(&self, shell: &mut InteractiveShell, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(VoirsCliError::InvalidArgument(
                "Usage: :load <filename>".to_string()
            ));
        }
        
        let path = PathBuf::from(args[0]);
        shell.session_manager().load_session(&path).await?;
        
        Ok(())
    }
    
    /// Show synthesis history
    fn show_history(&self, shell: &mut InteractiveShell, args: &[&str]) -> Result<()> {
        let count = if args.is_empty() {
            10 // Default to last 10 entries
        } else {
            args[0].parse().map_err(|_| {
                VoirsCliError::InvalidArgument(format!(
                    "Invalid count '{}'. Expected a positive number",
                    args[0]
                ))
            })?
        };
        
        let history = shell.session_manager().get_recent_history(count);
        
        if history.is_empty() {
            println!("No synthesis history available");
            return Ok(());
        }
        
        println!("\\n{}", style("Synthesis History:").bold());
        println!("{}",   style("==================").cyan());
        
        for (i, entry) in history.iter().enumerate() {
            let time = entry.timestamp.format("%H:%M:%S");
            let voice = entry.voice.as_deref().unwrap_or("unknown");
            let text = if entry.text.len() > 50 {
                format!("{}...", &entry.text[..47])
            } else {
                entry.text.clone()
            };
            
            println!(
                "{}. {} [{}] {}",
                style(count - i).dim(),
                style(time).cyan(),
                style(voice).green(),
                text
            );
        }
        
        println!();
        
        Ok(())
    }
    
    /// Handle export command
    async fn handle_export_command(&self, shell: &mut InteractiveShell, args: &[&str]) -> Result<()> {
        if args.len() < 2 {
            return Err(VoirsCliError::InvalidArgument(
                "Usage: :export <format> <filename>\\nFormats: json, csv, text".to_string()
            ));
        }
        
        let format = match args[0].to_lowercase().as_str() {
            "json" => ExportFormat::Json,
            "csv" => ExportFormat::Csv,
            "text" | "txt" => ExportFormat::Text,
            _ => {
                return Err(VoirsCliError::InvalidArgument(
                    "Invalid format. Supported formats: json, csv, text".to_string()
                ));
            }
        };
        
        let path = PathBuf::from(args[1]);
        shell.session_manager().export_history(&path, format).await?;
        
        Ok(())
    }
    
    /// Clear the screen
    fn clear_screen(&self) -> Result<()> {
        let term = Term::stdout();
        term.clear_screen().map_err(|e| {
            VoirsCliError::IoError(format!("Failed to clear screen: {}", e))
        })?;
        
        Ok(())
    }
    
    /// Show current status
    fn show_status(&self, shell: &InteractiveShell) -> Result<()> {
        println!("\\n{}", style("VoiRS Interactive Status").bold().cyan());
        println!("{}",   style("========================").cyan());
        
        // Voice information
        if let Some(voice) = shell.current_voice() {
            println!("Voice: {}", style(voice).green());
        } else {
            println!("Voice: {}", style("Not set").red());
        }
        
        // Parameters
        let (speed, pitch, volume) = shell.current_params();
        println!("Speed: {}x", style(format!("{:.1}", speed)).yellow());
        println!("Pitch: {} semitones", style(format!("{:.1}", pitch)).yellow());
        println!("Volume: {}", style(format!("{:.1}", volume)).yellow());
        
        // Session statistics
        let stats = shell.session_manager().get_stats();
        println!();
        println!("{}", style("Session Statistics:").bold());
        println!("  Syntheses: {}", stats.total_syntheses);
        println!("  Characters: {}", stats.total_characters);
        println!("  Voices used: {}", stats.voices_used.len());
        
        if !stats.voices_used.is_empty() {
            println!("  Used voices: {}", stats.voices_used.join(", "));
        }
        
        println!();
        
        Ok(())
    }
    
    /// Handle quit command
    fn handle_quit(&self, shell: &mut InteractiveShell) -> Result<()> {
        shell.exit();
        Ok(())
    }
    
    /// Handle unknown command
    fn handle_unknown_command(&self, command: &str) -> Result<()> {
        println!(
            "{} Unknown command: {}. Type ':help' for available commands.",
            style("!").yellow(),
            style(command).red()
        );
        
        // Suggest similar commands
        let suggestions = self.suggest_similar_commands(command);
        if !suggestions.is_empty() {
            println!("Did you mean: {}", suggestions.join(", "));
        }
        
        Ok(())
    }
    
    /// Suggest similar commands
    fn suggest_similar_commands(&self, command: &str) -> Vec<String> {
        let commands = [
            ":help", ":voice", ":voices", ":speed", ":pitch", ":volume",
            ":save", ":load", ":history", ":clear", ":status", ":export", ":quit"
        ];
        
        let mut suggestions = Vec::new();
        
        for cmd in &commands {
            if cmd.starts_with(command) || 
               self.levenshtein_distance(command, cmd) <= 2 {
                suggestions.push(cmd.to_string());
            }
        }
        
        suggestions
    }
    
    /// Calculate Levenshtein distance for command suggestions
    fn levenshtein_distance(&self, a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let a_len = a_chars.len();
        let b_len = b_chars.len();
        
        if a_len == 0 { return b_len; }
        if b_len == 0 { return a_len; }
        
        let mut matrix = vec![vec![0; b_len + 1]; a_len + 1];
        
        for i in 0..=a_len { matrix[i][0] = i; }
        for j in 0..=b_len { matrix[0][j] = j; }
        
        for i in 1..=a_len {
            for j in 1..=b_len {
                let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i-1][j] + 1)
                    .min(matrix[i][j-1] + 1)
                    .min(matrix[i-1][j-1] + cost);
            }
        }
        
        matrix[a_len][b_len]
    }
}