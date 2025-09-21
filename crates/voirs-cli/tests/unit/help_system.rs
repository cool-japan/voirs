//! Unit tests for the help system

use voirs_cli::help::{display_getting_started, HelpSystem};

#[test]
fn test_help_system_creation() {
    let help_system = HelpSystem::new();

    // Test that basic commands have help
    assert!(help_system.get_command_help("synthesize").is_some());
    assert!(help_system.get_command_help("guide").is_some());
    assert!(help_system.get_command_help("server").is_some());
    assert!(help_system.get_command_help("interactive").is_some());
}

#[test]
fn test_synthesize_command_help() {
    let help_system = HelpSystem::new();
    let help = help_system.get_command_help("synthesize").unwrap();

    assert!(help.description.contains("text to speech"));
    assert!(!help.examples.is_empty());
    assert!(help
        .examples
        .iter()
        .any(|ex| ex.command.contains("synthesize")));
}

#[test]
fn test_unknown_command_help() {
    let help_system = HelpSystem::new();
    assert!(help_system.get_command_help("nonexistent").is_none());
}

#[test]
fn test_help_display() {
    let help_system = HelpSystem::new();
    let help_text = help_system.display_command_help("synthesize");

    assert!(help_text.contains("synthesize"));
    assert!(help_text.contains("Examples:"));
}

#[test]
fn test_unknown_command_display() {
    let help_system = HelpSystem::new();
    let help_text = help_system.display_command_help("synth");

    assert!(help_text.contains("Unknown command"));
    assert!(help_text.contains("Did you mean"));
}

#[test]
fn test_command_overview() {
    let help_system = HelpSystem::new();
    let overview = help_system.display_command_overview();

    assert!(overview.contains("VoiRS CLI Commands"));
    assert!(overview.contains("Synthesis"));
    assert!(overview.contains("Voice Management"));
    assert!(overview.contains("synthesize"));
    assert!(overview.contains("list-voices"));
}

#[test]
fn test_contextual_help() {
    let help_system = HelpSystem::new();

    let voice_help = help_system.get_contextual_help("voice_not_found");
    assert!(voice_help.iter().any(|h| h.contains("list-voices")));

    let gpu_help = help_system.get_contextual_help("gpu_error");
    assert!(gpu_help.iter().any(|h| h.contains("--gpu")));

    let generic_help = help_system.get_contextual_help("unknown_error");
    assert!(generic_help.iter().any(|h| h.contains("voirs guide")));
}

#[test]
fn test_getting_started_guide() {
    let guide = display_getting_started();

    assert!(guide.contains("Getting Started with VoiRS"));
    assert!(guide.contains("voirs test"));
    assert!(guide.contains("voirs list-voices"));
    assert!(guide.contains("voirs synthesize"));
    assert!(guide.contains("voirs interactive"));
}

#[test]
fn test_help_examples_are_valid() {
    let help_system = HelpSystem::new();

    for (command, help) in &help_system.command_help {
        for example in &help.examples {
            // Check that examples contain the command name
            assert!(
                example.command.contains("voirs"),
                "Example for {} should contain 'voirs': {}",
                command,
                example.command
            );

            // Check that examples have descriptions
            assert!(
                !example.description.is_empty(),
                "Example for {command} should have description"
            );
        }
    }
}

#[test]
fn test_help_troubleshooting() {
    let help_system = HelpSystem::new();
    let synthesize_help = help_system.get_command_help("synthesize").unwrap();

    // Should have troubleshooting tips
    assert!(!synthesize_help.troubleshooting.is_empty());

    // Tips should have titles and content
    for tip in &synthesize_help.troubleshooting {
        assert!(!tip.title.is_empty());
        assert!(!tip.content.is_empty());
    }
}
