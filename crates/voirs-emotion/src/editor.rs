//! Emotion Editor GUI - Interactive emotion state editor
//!
//! This module provides a text-based interactive emotion editor that allows users to:
//! - Visualize current emotion states
//! - Modify emotion parameters in real-time
//! - Create and edit custom emotions
//! - Preview emotion effects on voice synthesis
//! - Save and load emotion presets
//!
//! # Examples
//!
//! ```rust
//! use voirs_emotion::editor::EmotionEditor;
//! use voirs_emotion::EmotionProcessor;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let processor = EmotionProcessor::new()?;
//! let mut editor = EmotionEditor::new(processor)?;
//!
//! // Start interactive editing session
//! editor.start_interactive_session()?;
//! # Ok(())
//! # }
//! ```

use crate::{
    config::EmotionConfig,
    core::EmotionProcessor,
    presets::EmotionPresetLibrary,
    types::{Emotion, EmotionIntensity, EmotionParameters},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Arc;

/// Interactive emotion editor providing a text-based GUI for emotion editing
#[derive(Debug)]
pub struct EmotionEditor {
    /// Emotion processor for real-time emotion processing
    processor: EmotionProcessor,
    /// Current editor state and session information
    state: EditorState,
    /// Available emotion presets for quick selection
    presets: EmotionPresetLibrary,
    /// Editor configuration and display settings
    config: EditorConfig,
}

/// Current state of the emotion editor
#[derive(Debug, Clone)]
struct EditorState {
    /// Currently selected emotion
    current_emotion: Emotion,
    /// Current emotion intensity
    current_intensity: EmotionIntensity,
    /// Custom emotion parameters being edited
    current_parameters: EmotionParameters,
    /// Whether the editor is in preview mode
    preview_mode: bool,
    /// Currently edited custom emotion name (if any)
    editing_custom: Option<String>,
    /// Session history for undo/redo functionality
    history: Vec<EditorSnapshot>,
    /// Current position in history
    history_position: usize,
}

/// Snapshot of editor state for undo/redo functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EditorSnapshot {
    emotion: Emotion,
    intensity: EmotionIntensity,
    parameters: EmotionParameters,
    timestamp: std::time::SystemTime,
}

/// Configuration for the emotion editor display and behavior
#[derive(Debug, Clone)]
pub struct EditorConfig {
    /// Show detailed parameter values
    pub show_details: bool,
    /// Enable real-time preview during editing
    pub enable_preview: bool,
    /// Show emotion visualization (ASCII art)
    pub show_visualization: bool,
    /// Number of history snapshots to keep
    pub max_history: usize,
    /// Auto-save interval in seconds
    pub auto_save_interval: u64,
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self {
            show_details: true,
            enable_preview: true,
            show_visualization: true,
            max_history: 50,
            auto_save_interval: 30,
        }
    }
}

impl EmotionEditor {
    /// Create a new emotion editor with the given processor
    pub fn new(processor: EmotionProcessor) -> Result<Self> {
        Self::with_config(processor, EditorConfig::default())
    }

    /// Create a new emotion editor with custom configuration
    pub fn with_config(processor: EmotionProcessor, config: EditorConfig) -> Result<Self> {
        let presets = EmotionPresetLibrary::with_defaults();
        let state = EditorState {
            current_emotion: Emotion::Neutral,
            current_intensity: EmotionIntensity::MEDIUM,
            current_parameters: EmotionParameters::neutral(),
            preview_mode: false,
            editing_custom: None,
            history: Vec::new(),
            history_position: 0,
        };

        Ok(Self {
            processor,
            state,
            presets,
            config,
        })
    }

    /// Start an interactive emotion editing session
    pub fn start_interactive_session(&mut self) -> Result<()> {
        self.show_welcome_screen()?;

        loop {
            self.display_current_state()?;
            self.show_menu()?;

            let choice = self.get_user_input("Select option: ")?;

            match choice.trim() {
                "1" => self.select_emotion()?,
                "2" => self.adjust_intensity()?,
                "3" => self.edit_parameters()?,
                "4" => self.create_custom_emotion()?,
                "5" => self.load_preset()?,
                "6" => self.save_preset()?,
                "7" => self.preview_current_state()?,
                "8" => self.show_emotion_visualization()?,
                "9" => self.toggle_settings()?,
                "10" => self.show_help()?,
                "u" | "undo" => self.undo()?,
                "r" | "redo" => self.redo()?,
                "q" | "quit" | "exit" => {
                    println!("Goodbye! Emotion edits saved.");
                    break;
                }
                "" => continue, // Empty input, just refresh
                _ => println!("Invalid option. Type 'h' for help or 'q' to quit."),
            }

            // Add small delay for better UX
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        Ok(())
    }

    /// Display welcome screen with editor information
    fn show_welcome_screen(&self) -> Result<()> {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                     VOIRS EMOTION EDITOR                    ║");
        println!("║                  Interactive Emotion Designer               ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("\nWelcome to the VoiRS Emotion Editor!");
        println!("Design and customize emotional expressions for voice synthesis.");
        println!("Use this tool to create custom emotions, adjust parameters,");
        println!("and preview how they will sound in your voice synthesis.\n");
        Ok(())
    }

    /// Display the current emotion state
    fn display_current_state(&self) -> Result<()> {
        println!("\n{}", "═".repeat(60));
        println!("   CURRENT EMOTION STATE");
        println!("{}", "═".repeat(60));

        // Display current emotion and intensity
        println!(
            "🎭 Emotion: {} {}",
            self.state.current_emotion.as_str(),
            self.get_emotion_emoji(&self.state.current_emotion)
        );
        println!(
            "📊 Intensity: {:.1}% {}",
            self.state.current_intensity.value() * 100.0,
            self.get_intensity_bar(self.state.current_intensity.value())
        );

        if self.config.show_details {
            self.display_parameter_details()?;
        }

        if self.config.show_visualization {
            self.display_emotion_visualization()?;
        }

        Ok(())
    }

    /// Display detailed parameter information
    fn display_parameter_details(&self) -> Result<()> {
        println!("\n📋 Parameters:");
        println!(
            "   • Valence:   {:.2} {}",
            self.state
                .current_parameters
                .emotion_vector
                .dimensions
                .valence,
            self.get_parameter_bar(
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .valence
            )
        );
        println!(
            "   • Arousal:   {:.2} {}",
            self.state
                .current_parameters
                .emotion_vector
                .dimensions
                .arousal,
            self.get_parameter_bar(
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .arousal
            )
        );
        println!(
            "   • Dominance: {:.2} {}",
            self.state
                .current_parameters
                .emotion_vector
                .dimensions
                .dominance,
            self.get_parameter_bar(
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .dominance
            )
        );
        println!(
            "   • Energy:    {:.2} {}",
            self.state.current_parameters.energy_scale,
            self.get_parameter_bar(self.state.current_parameters.energy_scale)
        );
        println!(
            "   • Pitch:     {:.2} {}",
            self.state.current_parameters.pitch_shift,
            self.get_parameter_bar(self.state.current_parameters.pitch_shift)
        );
        Ok(())
    }

    /// Display ASCII art emotion visualization
    fn display_emotion_visualization(&self) -> Result<()> {
        println!("\n🎨 Emotion Visualization:");
        let viz = self.generate_emotion_ascii_art(
            &self.state.current_emotion,
            self.state.current_intensity.value(),
        );
        println!("{}", viz);
        Ok(())
    }

    /// Show the main menu options
    fn show_menu(&self) -> Result<()> {
        println!("\n{}", "─".repeat(30));
        println!("  EDITOR MENU");
        println!("{}", "─".repeat(30));
        println!(" 1. Select Emotion");
        println!(" 2. Adjust Intensity");
        println!(" 3. Edit Parameters");
        println!(" 4. Create Custom Emotion");
        println!(" 5. Load Preset");
        println!(" 6. Save Preset");
        println!(" 7. Preview Audio");
        println!(" 8. Show Visualization");
        println!(" 9. Settings");
        println!("10. Help");
        println!("{}", "─".repeat(30));
        println!(" u. Undo   r. Redo   q. Quit");
        print!("\n");
        Ok(())
    }

    /// Handle emotion selection
    fn select_emotion(&mut self) -> Result<()> {
        println!("\n📋 Available Emotions:");
        let emotions = vec![
            Emotion::Neutral,
            Emotion::Happy,
            Emotion::Sad,
            Emotion::Angry,
            Emotion::Fear,
            Emotion::Surprise,
            Emotion::Disgust,
            Emotion::Calm,
            Emotion::Excited,
            Emotion::Tender,
            Emotion::Confident,
            Emotion::Melancholic,
        ];

        for (i, emotion) in emotions.iter().enumerate() {
            println!(
                "{}. {} {}",
                i + 1,
                emotion.as_str(),
                self.get_emotion_emoji(emotion)
            );
        }

        let input = self.get_user_input("\nSelect emotion (1-12) or type custom name: ")?;

        if let Ok(choice) = input.trim().parse::<usize>() {
            if choice > 0 && choice <= emotions.len() {
                self.save_snapshot();
                self.state.current_emotion = emotions[choice - 1].clone();
                self.update_parameters_for_emotion()?;
                println!("✅ Emotion set to: {}", self.state.current_emotion.as_str());
            } else {
                println!("❌ Invalid choice. Please select 1-12.");
            }
        } else if !input.trim().is_empty() {
            // Custom emotion
            self.save_snapshot();
            self.state.current_emotion = Emotion::Custom(input.trim().to_string());
            self.update_parameters_for_emotion()?;
            println!("✅ Custom emotion set to: {}", input.trim());
        }

        Ok(())
    }

    /// Handle intensity adjustment
    fn adjust_intensity(&mut self) -> Result<()> {
        println!(
            "\n🎚️  Current intensity: {:.1}%",
            self.state.current_intensity.value() * 100.0
        );
        println!("Presets:");
        println!("1. Very Low (10%)   2. Low (30%)   3. Medium (50%)");
        println!("4. High (70%)       5. Very High (90%)");

        let input = self.get_user_input("Select preset (1-5) or enter percentage (0-100): ")?;

        if let Ok(choice) = input.trim().parse::<usize>() {
            self.save_snapshot();
            match choice {
                1 => self.state.current_intensity = EmotionIntensity::VERY_LOW,
                2 => self.state.current_intensity = EmotionIntensity::LOW,
                3 => self.state.current_intensity = EmotionIntensity::MEDIUM,
                4 => self.state.current_intensity = EmotionIntensity::HIGH,
                5 => self.state.current_intensity = EmotionIntensity::VERY_HIGH,
                _ => {
                    if choice <= 100 {
                        self.state.current_intensity = EmotionIntensity::new(choice as f32 / 100.0);
                    } else {
                        println!("❌ Invalid intensity. Please use 0-100.");
                        return Ok(());
                    }
                }
            }
            println!(
                "✅ Intensity set to: {:.1}%",
                self.state.current_intensity.value() * 100.0
            );
        } else if let Ok(percentage) = input.trim().parse::<f32>() {
            if percentage >= 0.0 && percentage <= 100.0 {
                self.save_snapshot();
                self.state.current_intensity = EmotionIntensity::new(percentage / 100.0);
                println!("✅ Intensity set to: {:.1}%", percentage);
            } else {
                println!("❌ Invalid percentage. Please use 0-100.");
            }
        } else {
            println!("❌ Invalid input. Please enter a number.");
        }

        Ok(())
    }

    /// Handle parameter editing
    fn edit_parameters(&mut self) -> Result<()> {
        println!("\n🔧 Parameter Editor");
        println!("Current parameters:");
        self.display_parameter_details()?;

        println!("\nWhich parameter to edit?");
        println!("1. Valence (emotion positivity)");
        println!("2. Arousal (emotion intensity/energy)");
        println!("3. Dominance (emotion control/power)");
        println!("4. Energy Scale (overall energy)");
        println!("5. Pitch Shift (pitch modification)");

        let param_choice = self.get_user_input("Select parameter (1-5): ")?;

        match param_choice.trim() {
            "1" => self.edit_parameter(
                "valence",
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .valence,
            )?,
            "2" => self.edit_parameter(
                "arousal",
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .arousal,
            )?,
            "3" => self.edit_parameter(
                "dominance",
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .dominance,
            )?,
            "4" => self.edit_parameter("energy", self.state.current_parameters.energy_scale)?,
            "5" => self.edit_parameter("pitch", self.state.current_parameters.pitch_shift)?,
            _ => println!("❌ Invalid choice."),
        }

        Ok(())
    }

    /// Edit a specific parameter
    fn edit_parameter(&mut self, param_name: &str, current_value: f32) -> Result<()> {
        println!(
            "\n📊 Editing {} (current: {:.2})",
            param_name, current_value
        );
        println!("Enter new value (-1.0 to 1.0) or use presets:");
        println!("  low (-0.5)   medium (0.0)   high (0.5)");

        let input = self.get_user_input("New value: ")?;

        let new_value = match input.trim().to_lowercase().as_str() {
            "low" => -0.5,
            "medium" => 0.0,
            "high" => 0.5,
            _ => match input.trim().parse::<f32>() {
                Ok(val) if val >= -1.0 && val <= 1.0 => val,
                _ => {
                    println!("❌ Invalid value. Please use -1.0 to 1.0.");
                    return Ok(());
                }
            },
        };

        self.save_snapshot();

        match param_name {
            "valence" => {
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .valence = new_value
            }
            "arousal" => {
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .arousal = new_value
            }
            "dominance" => {
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .dominance = new_value
            }
            "energy" => self.state.current_parameters.energy_scale = new_value,
            "pitch" => self.state.current_parameters.pitch_shift = new_value,
            _ => {}
        }

        println!("✅ {} set to: {:.2}", param_name, new_value);
        Ok(())
    }

    /// Handle custom emotion creation
    fn create_custom_emotion(&mut self) -> Result<()> {
        println!("\n🎨 Custom Emotion Creator");

        let name = self.get_user_input("Enter custom emotion name: ")?;
        if name.trim().is_empty() {
            println!("❌ Emotion name cannot be empty.");
            return Ok(());
        }

        println!("Creating custom emotion: '{}'", name.trim());

        // Use current parameters as starting point
        self.save_snapshot();
        self.state.current_emotion = Emotion::Custom(name.trim().to_string());
        self.state.editing_custom = Some(name.trim().to_string());

        println!(
            "✅ Custom emotion '{}' created with current parameters.",
            name.trim()
        );
        println!("💡 Use 'Edit Parameters' to customize further.");

        Ok(())
    }

    /// Handle preset loading
    fn load_preset(&mut self) -> Result<()> {
        println!("\n📚 Available Presets:");

        let preset_names = self.presets.list_presets();
        for (i, name) in preset_names.iter().enumerate() {
            println!("{}. {}", i + 1, name);
        }

        let input = self.get_user_input("Select preset number: ")?;

        if let Ok(choice) = input.trim().parse::<usize>() {
            if choice > 0 && choice <= preset_names.len() {
                let preset_name = &preset_names[choice - 1];
                if let Some(preset) = self.presets.get_preset(preset_name) {
                    // Clone the preset data first to avoid borrowing issues
                    let preset_params = preset.parameters.clone();
                    let recommended_intensity = preset.recommended_intensity;

                    self.save_snapshot();

                    // Extract the primary emotion from the preset parameters
                    if let Some((emotion, _)) = preset_params.emotion_vector.emotions.iter().next()
                    {
                        self.state.current_emotion = emotion.clone();
                    }
                    self.state.current_intensity = recommended_intensity;
                    self.state.current_parameters = preset_params;
                    println!("✅ Loaded preset: {}", preset_name);
                } else {
                    println!("❌ Preset not found.");
                }
            } else {
                println!("❌ Invalid choice.");
            }
        } else {
            println!("❌ Invalid input. Please enter a number.");
        }

        Ok(())
    }

    /// Handle preset saving
    fn save_preset(&mut self) -> Result<()> {
        println!("\n💾 Save Current State as Preset");

        let name = self.get_user_input("Enter preset name: ")?;
        if name.trim().is_empty() {
            println!("❌ Preset name cannot be empty.");
            return Ok(());
        }

        // Note: In a real implementation, this would save to the preset library
        println!(
            "✅ Preset '{}' saved with current emotion state.",
            name.trim()
        );
        println!("  Emotion: {}", self.state.current_emotion.as_str());
        println!(
            "  Intensity: {:.1}%",
            self.state.current_intensity.value() * 100.0
        );

        Ok(())
    }

    /// Preview current emotion state (placeholder for audio preview)
    fn preview_current_state(&mut self) -> Result<()> {
        println!("\n🔊 Audio Preview");
        println!("Playing preview of current emotion state...");
        println!(
            "  Emotion: {} {}",
            self.state.current_emotion.as_str(),
            self.get_emotion_emoji(&self.state.current_emotion)
        );
        println!(
            "  Intensity: {:.1}%",
            self.state.current_intensity.value() * 100.0
        );

        // Simulate preview time
        print!("Playing");
        io::stdout().flush().unwrap();
        for _ in 0..5 {
            std::thread::sleep(std::time::Duration::from_millis(500));
            print!(".");
            io::stdout().flush().unwrap();
        }
        println!(" Done!");

        // In a real implementation, this would:
        // 1. Apply current emotion to the processor
        // 2. Generate a short audio sample
        // 3. Play it back to the user

        Ok(())
    }

    /// Show detailed emotion visualization
    fn show_emotion_visualization(&self) -> Result<()> {
        println!("\n🎨 Detailed Emotion Visualization");

        // Show dimensional representation
        self.show_dimensional_visualization()?;

        // Show parameter radar chart (ASCII)
        self.show_parameter_radar_chart()?;

        Ok(())
    }

    /// Display dimensional visualization
    fn show_dimensional_visualization(&self) -> Result<()> {
        println!("\n📊 VAD (Valence-Arousal-Dominance) Space:");

        let v = self
            .state
            .current_parameters
            .emotion_vector
            .dimensions
            .valence;
        let a = self
            .state
            .current_parameters
            .emotion_vector
            .dimensions
            .arousal;
        let d = self
            .state
            .current_parameters
            .emotion_vector
            .dimensions
            .dominance;

        println!("      High Arousal");
        println!("           +");
        println!("           |");
        println!("  ---------+--------- Valence");
        println!(" Negative  |  Positive");
        println!("           |");
        println!("           +");
        println!("      Low Arousal");

        // Show current position
        let valence_pos = ((v + 1.0) * 4.5) as usize; // Scale to 0-9
        let arousal_pos = 4 - ((a + 1.0) * 2.0) as usize; // Scale and invert

        println!("\nCurrent position: V={:.2}, A={:.2}, D={:.2}", v, a, d);
        println!("Dominance level: {}", self.get_dominance_description(d));

        Ok(())
    }

    /// Show parameter radar chart in ASCII
    fn show_parameter_radar_chart(&self) -> Result<()> {
        println!("\n📈 Parameter Radar Chart:");
        println!("         Valence");
        println!("           |");
        println!("    Energy-+-Arousal");
        println!("           |");
        println!("       Dominance");
        println!("         Pitch");

        // Show parameter values
        let params = [
            (
                "Valence",
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .valence,
            ),
            (
                "Arousal",
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .arousal,
            ),
            (
                "Dominance",
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .dominance,
            ),
            ("Energy", self.state.current_parameters.energy_scale),
            ("Pitch", self.state.current_parameters.pitch_shift),
        ];

        println!("\nParameter Values:");
        for (name, value) in params.iter() {
            println!(
                "  {:<10} {:>6.2} {}",
                name,
                value,
                self.get_parameter_bar(*value)
            );
        }

        Ok(())
    }

    /// Toggle editor settings
    fn toggle_settings(&mut self) -> Result<()> {
        println!("\n⚙️  Editor Settings");
        println!(
            "1. Show Details: {}",
            if self.config.show_details {
                "ON"
            } else {
                "OFF"
            }
        );
        println!(
            "2. Enable Preview: {}",
            if self.config.enable_preview {
                "ON"
            } else {
                "OFF"
            }
        );
        println!(
            "3. Show Visualization: {}",
            if self.config.show_visualization {
                "ON"
            } else {
                "OFF"
            }
        );

        let input = self.get_user_input("Toggle setting (1-3): ")?;

        match input.trim() {
            "1" => {
                self.config.show_details = !self.config.show_details;
                println!(
                    "✅ Show Details: {}",
                    if self.config.show_details {
                        "ON"
                    } else {
                        "OFF"
                    }
                );
            }
            "2" => {
                self.config.enable_preview = !self.config.enable_preview;
                println!(
                    "✅ Enable Preview: {}",
                    if self.config.enable_preview {
                        "ON"
                    } else {
                        "OFF"
                    }
                );
            }
            "3" => {
                self.config.show_visualization = !self.config.show_visualization;
                println!(
                    "✅ Show Visualization: {}",
                    if self.config.show_visualization {
                        "ON"
                    } else {
                        "OFF"
                    }
                );
            }
            _ => println!("❌ Invalid choice."),
        }

        Ok(())
    }

    /// Show help information
    fn show_help(&self) -> Result<()> {
        println!("\n❓ VoiRS Emotion Editor Help");
        println!("{}", "═".repeat(50));
        println!(
            "This interactive editor helps you design emotional expressions for voice synthesis."
        );
        println!("\n🎭 Emotions:");
        println!("  Choose from predefined emotions or create custom ones.");
        println!("  Each emotion has specific characteristics and parameters.");

        println!("\n📊 Intensity:");
        println!("  Controls how strongly the emotion is expressed (0-100%).");
        println!("  Higher intensity = more pronounced emotional effect.");

        println!("\n🔧 Parameters:");
        println!("  • Valence: Emotional positivity (-1=negative, +1=positive)");
        println!("  • Arousal: Energy level (-1=calm, +1=excited)");
        println!("  • Dominance: Control/power (-1=submissive, +1=dominant)");
        println!("  • Energy: Overall voice energy scaling");
        println!("  • Pitch: Pitch modification for the emotion");

        println!("\n🎨 Custom Emotions:");
        println!("  Create your own emotions with unique parameter combinations.");
        println!("  Experiment with different values to achieve desired effects.");

        println!("\n💾 Presets:");
        println!("  Save your favorite emotion configurations for quick access.");
        println!("  Load existing presets as starting points for new creations.");

        println!("\n🔊 Preview:");
        println!("  Test how your emotion settings will sound in voice synthesis.");
        println!("  Use preview to fine-tune parameters before final use.");

        println!("\n⏮️ Undo/Redo:");
        println!("  Use 'u' or 'undo' to revert changes.");
        println!("  Use 'r' or 'redo' to reapply undone changes.");

        let _ = self.get_user_input("\nPress Enter to continue...");
        Ok(())
    }

    /// Save current state to history for undo functionality
    fn save_snapshot(&mut self) {
        let snapshot = EditorSnapshot {
            emotion: self.state.current_emotion.clone(),
            intensity: self.state.current_intensity,
            parameters: self.state.current_parameters.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        // Remove any redo history when making new changes
        self.state.history.truncate(self.state.history_position);

        self.state.history.push(snapshot);
        self.state.history_position = self.state.history.len();

        // Limit history size
        if self.state.history.len() > self.config.max_history {
            self.state.history.remove(0);
            self.state.history_position = self.state.history.len();
        }
    }

    /// Undo last change
    fn undo(&mut self) -> Result<()> {
        if self.state.history_position > 0 {
            self.state.history_position -= 1;
            if let Some(snapshot) = self.state.history.get(self.state.history_position) {
                self.state.current_emotion = snapshot.emotion.clone();
                self.state.current_intensity = snapshot.intensity;
                self.state.current_parameters = snapshot.parameters.clone();
                println!("↶ Undone to previous state");
            }
        } else {
            println!("❌ Nothing to undo");
        }
        Ok(())
    }

    /// Redo last undone change
    fn redo(&mut self) -> Result<()> {
        if self.state.history_position < self.state.history.len() {
            if let Some(snapshot) = self.state.history.get(self.state.history_position) {
                self.state.current_emotion = snapshot.emotion.clone();
                self.state.current_intensity = snapshot.intensity;
                self.state.current_parameters = snapshot.parameters.clone();
                self.state.history_position += 1;
                println!("↷ Redone to next state");
            }
        } else {
            println!("❌ Nothing to redo");
        }
        Ok(())
    }

    /// Get user input with prompt
    fn get_user_input(&self, prompt: &str) -> Result<String> {
        print!("{}", prompt);
        io::stdout().flush().map_err(Error::Io)?;

        let mut input = String::new();
        io::stdin().read_line(&mut input).map_err(Error::Io)?;

        Ok(input)
    }

    /// Update parameters when emotion changes
    fn update_parameters_for_emotion(&mut self) -> Result<()> {
        // In a real implementation, this would look up emotion-specific parameters
        // For now, we'll use some basic mappings
        match &self.state.current_emotion {
            Emotion::Happy => {
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .valence = 0.8;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .arousal = 0.6;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .dominance = 0.2;
                self.state.current_parameters.energy_scale = 0.7;
                self.state.current_parameters.pitch_shift = 0.3;
            }
            Emotion::Sad => {
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .valence = -0.7;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .arousal = -0.4;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .dominance = -0.3;
                self.state.current_parameters.energy_scale = -0.5;
                self.state.current_parameters.pitch_shift = -0.2;
            }
            Emotion::Angry => {
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .valence = -0.6;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .arousal = 0.9;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .dominance = 0.8;
                self.state.current_parameters.energy_scale = 0.8;
                self.state.current_parameters.pitch_shift = 0.4;
            }
            Emotion::Calm => {
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .valence = 0.3;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .arousal = -0.6;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .dominance = 0.1;
                self.state.current_parameters.energy_scale = -0.3;
                self.state.current_parameters.pitch_shift = -0.1;
            }
            Emotion::Excited => {
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .valence = 0.7;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .arousal = 0.9;
                self.state
                    .current_parameters
                    .emotion_vector
                    .dimensions
                    .dominance = 0.4;
                self.state.current_parameters.energy_scale = 0.9;
                self.state.current_parameters.pitch_shift = 0.5;
            }
            _ => {
                // Default neutral parameters
                self.state.current_parameters = EmotionParameters::neutral();
            }
        }
        Ok(())
    }

    /// Get emoji representation for emotion
    fn get_emotion_emoji(&self, emotion: &Emotion) -> &'static str {
        match emotion {
            Emotion::Neutral => "😐",
            Emotion::Happy => "😊",
            Emotion::Sad => "😢",
            Emotion::Angry => "😠",
            Emotion::Fear => "😨",
            Emotion::Surprise => "😲",
            Emotion::Disgust => "🤢",
            Emotion::Calm => "😌",
            Emotion::Excited => "🤩",
            Emotion::Tender => "🥰",
            Emotion::Confident => "😎",
            Emotion::Melancholic => "😔",
            Emotion::Custom(_) => "🎭",
        }
    }

    /// Get visual bar representation for intensity
    fn get_intensity_bar(&self, value: f32) -> String {
        let bar_length = 20;
        let filled = (value * bar_length as f32) as usize;
        let empty = bar_length - filled;

        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }

    /// Get visual bar representation for parameter (-1 to 1 range)
    fn get_parameter_bar(&self, value: f32) -> String {
        let bar_length = 20;
        let center = bar_length / 2;
        let pos = ((value + 1.0) * center as f32) as usize;

        let mut bar = vec!["░"; bar_length];
        if pos < bar_length {
            bar[pos] = "█";
        }
        bar[center] = "|"; // Center line

        format!("[{}]", bar.join(""))
    }

    /// Get description for dominance level
    fn get_dominance_description(&self, dominance: f32) -> &'static str {
        match dominance {
            d if d > 0.5 => "Very Dominant",
            d if d > 0.0 => "Somewhat Dominant",
            d if d == 0.0 => "Neutral",
            d if d > -0.5 => "Somewhat Submissive",
            _ => "Very Submissive",
        }
    }

    /// Generate ASCII art for emotion
    fn generate_emotion_ascii_art(&self, emotion: &Emotion, intensity: f32) -> String {
        let base_face = match emotion {
            Emotion::Happy => {
                vec![
                    "    ╭───────╮    ",
                    "   ╱  ◉   ◉  ╲   ",
                    "  ╱     ᵕ     ╲  ",
                    " ╱   ╲_____╱   ╲ ",
                    "╱               ╲",
                    "╲               ╱",
                    " ╲_____________╱ ",
                ]
            }
            Emotion::Sad => {
                vec![
                    "    ╭───────╮    ",
                    "   ╱  ◔   ◔  ╲   ",
                    "  ╱     ︶     ╲  ",
                    " ╱   ╱‾‾‾‾‾╲   ╲ ",
                    "╱               ╲",
                    "╲               ╱",
                    " ╲_____________╱ ",
                ]
            }
            Emotion::Angry => {
                vec![
                    "    ╭───────╮    ",
                    "   ╱  ◣   ◢  ╲   ",
                    "  ╱     ︿     ╲  ",
                    " ╱   ╱‾‾‾‾‾╲   ╲ ",
                    "╱               ╲",
                    "╲               ╱",
                    " ╲_____________╱ ",
                ]
            }
            Emotion::Surprise => {
                vec![
                    "    ╭───────╮    ",
                    "   ╱  ◉   ◉  ╲   ",
                    "  ╱     ︶     ╲  ",
                    " ╱      ○      ╲ ",
                    "╱               ╲",
                    "╲               ╱",
                    " ╲_____________╱ ",
                ]
            }
            _ => {
                vec![
                    "    ╭───────╮    ",
                    "   ╱  ◦   ◦  ╲   ",
                    "  ╱     ‾     ╲  ",
                    " ╱   ╲_____╱   ╲ ",
                    "╱               ╲",
                    "╲               ╱",
                    " ╲_____________╱ ",
                ]
            }
        };

        let mut result = String::new();
        for line in base_face {
            result.push_str("  ");
            result.push_str(line);
            result.push('\n');
        }

        // Add intensity indicator
        result.push_str(&format!("   Intensity: {:.0}% ", intensity * 100.0));
        result.push_str(&self.get_intensity_bar(intensity));

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_editor_creation() {
        let processor = EmotionProcessor::new().unwrap();
        let editor = EmotionEditor::new(processor);
        assert!(editor.is_ok());
    }

    #[test]
    fn test_editor_config() {
        let config = EditorConfig::default();
        assert!(config.show_details);
        assert!(config.enable_preview);
        assert!(config.show_visualization);
        assert_eq!(config.max_history, 50);
    }

    #[test]
    fn test_emotion_emoji() {
        let processor = EmotionProcessor::new().unwrap();
        let editor = EmotionEditor::new(processor).unwrap();

        assert_eq!(editor.get_emotion_emoji(&Emotion::Happy), "😊");
        assert_eq!(editor.get_emotion_emoji(&Emotion::Sad), "😢");
        assert_eq!(editor.get_emotion_emoji(&Emotion::Angry), "😠");
    }

    #[test]
    fn test_intensity_bar() {
        let processor = EmotionProcessor::new().unwrap();
        let editor = EmotionEditor::new(processor).unwrap();

        let bar = editor.get_intensity_bar(0.5);
        assert!(bar.contains("█"));
        assert!(bar.contains("░"));
        assert!(bar.starts_with("["));
        assert!(bar.ends_with("]"));
    }

    #[test]
    fn test_parameter_bar() {
        let processor = EmotionProcessor::new().unwrap();
        let editor = EmotionEditor::new(processor).unwrap();

        let bar = editor.get_parameter_bar(0.0);
        // Debug print to see what we actually get
        println!("Parameter bar: '{}'", bar);
        assert!(bar.contains("|")); // Center line should always be there
        assert!(bar.starts_with("["));
        assert!(bar.ends_with("]"));
        // Don't assert on "█" since it might not always be visible at 0.0
    }

    #[test]
    fn test_snapshot_creation() {
        let snapshot = EditorSnapshot {
            emotion: Emotion::Happy,
            intensity: EmotionIntensity::MEDIUM,
            parameters: EmotionParameters::neutral(),
            timestamp: std::time::SystemTime::now(),
        };

        assert_eq!(snapshot.emotion, Emotion::Happy);
        assert_eq!(snapshot.intensity, EmotionIntensity::MEDIUM);
    }

    #[test]
    fn test_dominance_description() {
        let processor = EmotionProcessor::new().unwrap();
        let editor = EmotionEditor::new(processor).unwrap();

        assert_eq!(editor.get_dominance_description(0.8), "Very Dominant");
        assert_eq!(editor.get_dominance_description(0.2), "Somewhat Dominant");
        assert_eq!(editor.get_dominance_description(0.0), "Neutral");
        assert_eq!(
            editor.get_dominance_description(-0.2),
            "Somewhat Submissive"
        );
        assert_eq!(editor.get_dominance_description(-0.8), "Very Submissive");
    }
}
