//! Unit tests for error message handling

use voirs_cli::help::HelpSystem;

#[test]
fn test_contextual_error_messages() {
    let help_system = HelpSystem::new();

    // Test voice not found error context
    let suggestions = help_system.get_contextual_help("voice_not_found");
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| s.contains("list-voices")));
    assert!(suggestions.iter().any(|s| s.contains("download-voice")));
}

#[test]
fn test_model_error_context() {
    let help_system = HelpSystem::new();

    let suggestions = help_system.get_contextual_help("model_not_found");
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| s.contains("list-models")));
    assert!(suggestions.iter().any(|s| s.contains("download-model")));
}

#[test]
fn test_file_error_context() {
    let help_system = HelpSystem::new();

    let suggestions = help_system.get_contextual_help("file_not_found");
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| s.contains("file path")));
    assert!(suggestions.iter().any(|s| s.contains("absolute paths")));
}

#[test]
fn test_permission_error_context() {
    let help_system = HelpSystem::new();

    let suggestions = help_system.get_contextual_help("permission_denied");
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| s.contains("permissions")));
}

#[test]
fn test_gpu_error_context() {
    let help_system = HelpSystem::new();

    let suggestions = help_system.get_contextual_help("gpu_error");
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| s.contains("GPU")));
    assert!(suggestions.iter().any(|s| s.contains("CPU mode")));
}

#[test]
fn test_unknown_error_context() {
    let help_system = HelpSystem::new();

    let suggestions = help_system.get_contextual_help("unknown_error");
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| s.contains("voirs guide")));
}

#[test]
fn test_error_suggestions_are_actionable() {
    let help_system = HelpSystem::new();

    let contexts = vec![
        "voice_not_found",
        "model_not_found",
        "file_not_found",
        "permission_denied",
        "gpu_error",
    ];

    for context in contexts {
        let suggestions = help_system.get_contextual_help(context);
        assert!(
            !suggestions.is_empty(),
            "No suggestions for context: {context}"
        );

        // All suggestions should be non-empty and actionable
        for suggestion in suggestions {
            assert!(!suggestion.is_empty());
            assert!(suggestion.len() > 10); // Should be meaningful text
        }
    }
}
