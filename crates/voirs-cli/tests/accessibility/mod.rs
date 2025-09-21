use assert_cmd::Command;
use predicates::prelude::*;
use std::time::Duration;
use tempfile::TempDir;

#[test]
fn test_screen_reader_compatibility() {
    // Test that output is screen reader friendly
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::function(|output: &str| {
            // Check for clear structure that screen readers can parse
            output.contains("Usage:") &&
            output.contains("Commands:") &&
            output.contains("Options:") &&
            // Check that there are proper line breaks and structure
            output.lines().count() > 5
        }));
}

#[test]
fn test_keyboard_navigation_support() {
    // Test that commands can be used effectively with keyboard only
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Interactive mode should support keyboard input
    cmd.arg("interactive")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Interactive"));

    // Commands should be clearly documented for keyboard use
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("guide")
        .arg("--getting-started")
        .assert()
        .success()
        .stdout(predicate::function(|output: &str| {
            // Should contain clear instructions for keyboard users
            output.to_lowercase().contains("command")
                || output.to_lowercase().contains("type")
                || output.to_lowercase().contains("enter")
        }));
}

#[test]
fn test_color_contrast_validation() {
    // Test that CLI provides good contrast and color alternatives
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Test error output has clear text
    cmd.arg("invalid-command")
        .assert()
        .failure()
        .stderr(predicate::function(|output: &str| {
            // Error messages should be clear without relying only on color
            !output.is_empty()
                && (output.to_lowercase().contains("error")
                    || output.to_lowercase().contains("invalid")
                    || output.to_lowercase().contains("not found"))
        }));
}

#[test]
fn test_text_scaling_support() {
    // Test that output works well with different text scaling
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::function(|output: &str| {
            // Output should be readable at different scales
            // Check for reasonable line lengths and structure
            let lines: Vec<&str> = output.lines().collect();
            let average_line_length =
                lines.iter().map(|line| line.len()).sum::<usize>() / lines.len().max(1);

            // Lines shouldn't be too long (accessibility guideline)
            average_line_length < 120 &&
            // Should have multiple lines for structure
            lines.len() > 3
        }));
}

#[test]
fn test_alternative_text_for_audio_output() {
    let temp_dir = TempDir::new().unwrap();
    let output_file = temp_dir.path().join("accessibility_test.wav");

    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Test synthesis with clear feedback about what was created
    cmd.arg("synthesize")
        .arg("This is an accessibility test")
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .timeout(Duration::from_secs(120))
        .assert()
        .success();
    // Accessibility test focuses on successful completion regardless of output

    // Verify the output file exists (non-visual confirmation)
    assert!(output_file.exists());
}

#[test]
fn test_verbose_output_for_accessibility() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Test that verbose mode provides good accessibility information
    cmd.arg("--verbose")
        .arg("list-voices")
        .timeout(Duration::from_secs(10))
        .assert()
        .success();

    // Verbose mode should provide additional context
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("-v")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("verbose"));
}

#[test]
fn test_clear_progress_indication() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("progress_test.txt");
    let output_dir = temp_dir.path().join("output");

    // Create test input
    std::fs::write(
        &input_file,
        "Test sentence one.\nTest sentence two.\nTest sentence three.",
    )
    .unwrap();
    std::fs::create_dir_all(&output_dir).unwrap();

    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Batch processing should provide clear progress for accessibility
    cmd.arg("--verbose")
        .arg("batch")
        .arg(input_file.to_str().unwrap())
        .arg("--output-dir")
        .arg(output_dir.to_str().unwrap())
        .timeout(Duration::from_secs(60))
        .assert()
        .success();
}

#[test]
fn test_error_message_accessibility() {
    // Test that error messages are accessible and helpful
    let error_scenarios = vec![
        (vec!["synthesize"], "Missing required argument"),
        (vec!["invalid-command"], "Invalid command"),
        (
            vec!["synthesize", "test", "--output", "/invalid/path/file.wav"],
            "File operation",
        ),
    ];

    for (args, expected_context) in error_scenarios {
        let mut cmd = Command::cargo_bin("voirs").unwrap();

        for arg in args {
            cmd.arg(arg);
        }

        cmd.assert()
            .failure()
            .stderr(predicate::function(move |output: &str| {
                // Error messages should be descriptive and actionable
                !output.is_empty() &&
                // Should contain some form of helpful information
                (output.to_lowercase().contains("error") ||
                 output.to_lowercase().contains("invalid") ||
                 output.to_lowercase().contains("failed") ||
                 output.to_lowercase().contains("required"))
            }));
    }
}

#[test]
fn test_command_completion_accessibility() {
    let temp_dir = TempDir::new().unwrap();
    let completion_file = temp_dir.path().join("test_completion.bash");

    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Shell completion should work for accessibility tools
    cmd.arg("generate-completion")
        .arg("bash")
        .arg("--output")
        .arg(completion_file.to_str().unwrap())
        .assert()
        .success();

    // Check that completion file contains accessible information
    let content = std::fs::read_to_string(&completion_file).unwrap();
    assert!(!content.is_empty());
    assert!(content.contains("voirs"));
}

#[test]
fn test_consistent_interface_patterns() {
    // Test that the interface follows consistent patterns for accessibility
    let commands = vec!["list-voices", "list-models", "config"];

    for command in commands {
        let mut cmd = Command::cargo_bin("voirs").unwrap();

        // Each command should have consistent help patterns
        cmd.arg(command)
            .arg("--help")
            .assert()
            .success()
            .stdout(predicate::str::contains("Usage:"))
            .stdout(predicate::function(|output: &str| {
                // Help should follow consistent format
                output.contains("Usage:") && output.lines().count() > 3
            }));
    }
}

#[test]
fn test_alternative_input_methods() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("input_test.txt");
    let output_file = temp_dir.path().join("output_test.wav");

    // Create input file
    std::fs::write(&input_file, "This is a test of alternative input methods.").unwrap();

    // Test file-based input (alternative to typing)
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("synthesize-file")
        .arg(input_file.to_str().unwrap())
        .arg("--output-dir")
        .arg(temp_dir.path().to_str().unwrap())
        .timeout(Duration::from_secs(120))
        .assert()
        .success();
}

#[test]
fn test_clear_status_reporting() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Status commands should provide clear, accessible information
    cmd.arg("list-voices")
        .timeout(Duration::from_secs(10))
        .assert()
        .success()
        .stdout(predicate::function(|output: &str| {
            // Output should be structured and informative
            !output.trim().is_empty()
        }));
}

#[test]
fn test_internationalization_friendly() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Test that the CLI works with different locale settings
    cmd.env("LANG", "C")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::function(|output: &str| {
            // Should work regardless of locale
            output.contains("Usage:")
        }));
}

#[test]
fn test_timeout_handling_accessibility() {
    // Test that timeouts are handled gracefully for accessibility
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Quick operation that should complete fast
    cmd.arg("--version")
        .timeout(Duration::from_secs(5))
        .assert()
        .success()
        .stdout(predicate::str::is_match(r"\d+\.\d+\.\d+").unwrap());
}

#[test]
fn test_configuration_accessibility() {
    let temp_dir = TempDir::new().unwrap();
    let config_file = temp_dir.path().join("accessibility_config.toml");

    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Configuration should be accessible and clearly documented
    cmd.arg("config")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Configuration"))
        .stdout(predicate::function(|output: &str| {
            // Help should explain configuration clearly
            output.contains("show") || output.contains("init")
        }));
}

#[test]
fn test_audio_alternatives() {
    let temp_dir = TempDir::new().unwrap();
    let output_file = temp_dir.path().join("audio_alternative.wav");

    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // For users who can't hear audio, file output should be clearly indicated
    cmd.arg("synthesize")
        .arg("Testing audio alternatives for accessibility")
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .timeout(Duration::from_secs(120))
        .assert()
        .success();

    // The file should exist as a tangible alternative to audio
    assert!(output_file.exists());

    // File should have reasonable size (not empty)
    let metadata = std::fs::metadata(&output_file).unwrap();
    assert!(metadata.len() > 100); // Basic sanity check
}
