//! Interactive mode for VoiRS CLI
//!
//! This module provides an interactive shell interface for real-time text-to-speech synthesis.
//! Features include:
//! - Command-line interface with history and tab completion
//! - Real-time synthesis with immediate audio playback
//! - Voice switching and parameter adjustments during session
//! - Session state management and export capabilities

pub mod commands;
pub mod session;
pub mod shell;
pub mod synthesis;

use crate::error::Result;
use clap::Args;

/// Interactive mode arguments
#[derive(Args, Debug)]
pub struct InteractiveOptions {
    /// Initial voice to use
    #[arg(short, long)]
    pub voice: Option<String>,

    /// Disable audio playback (synthesis only)
    #[arg(long)]
    pub no_audio: bool,

    /// Enable debug output
    #[arg(long)]
    pub debug: bool,

    /// Load session from file
    #[arg(long)]
    pub load_session: Option<std::path::PathBuf>,

    /// Auto-save session changes
    #[arg(long)]
    pub auto_save: bool,
}

/// Run the interactive mode
pub async fn run_interactive(options: InteractiveOptions) -> Result<()> {
    // Initialize the interactive shell
    let mut shell = shell::InteractiveShell::new(options).await?;

    // Start the main interactive loop
    shell.run().await
}
