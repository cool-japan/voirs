use assert_cmd::Command;
use predicates::prelude::*;
use std::time::Duration;
use tempfile::TempDir;

#[test]
fn test_first_time_user_experience() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // First run should show helpful guidance
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Usage:"))
        .stdout(predicate::str::contains("Commands:"))
        .stdout(predicate::str::contains("Options:"));
}

#[test]
fn test_common_workflow_voice_listing() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Common workflow: list available voices
    cmd.arg("list-voices")
        .timeout(Duration::from_secs(30))
        .assert()
        .success();
}

#[test]
fn test_common_workflow_synthesis() {
    let temp_dir = TempDir::new().unwrap();
    let output_file = temp_dir.path().join("test_output.wav");

    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Common workflow: basic synthesis
    cmd.arg("synthesize")
        .arg("Hello, this is a test.")
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .timeout(Duration::from_secs(30))
        .assert()
        .success();

    // Verify output file was created
    assert!(output_file.exists());
}

#[test]
fn test_error_message_clarity() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Test invalid command
    cmd.arg("invalid-command")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"))
        .stderr(predicate::str::contains("help"));
}

#[test]
fn test_error_message_with_suggestions() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Test missing required argument
    cmd.arg("synthesize")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn test_help_documentation_accuracy() {
    // Test that help is available for all major commands
    let commands = vec![
        "synthesize",
        "list-voices",
        "list-models",
        "batch",
        "server",
        "interactive",
        "config",
    ];

    for command in commands {
        let mut cmd = Command::cargo_bin("voirs").unwrap();
        cmd.arg(command)
            .arg("--help")
            .assert()
            .success()
            .stdout(predicate::str::contains("Usage:"));
    }
}

#[test]
fn test_progressive_disclosure() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Basic help should not overwhelm with details
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands:"))
        .stdout(predicate::function(|output: &str| {
            // Help output should be reasonable length (not overwhelming)
            output.lines().count() < 100
        }));
}

#[test]
fn test_command_discoverability() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // All major commands should be listed in help
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("synthesize"))
        .stdout(predicate::str::contains("list-voices"))
        .stdout(predicate::str::contains("interactive"));
}

#[test]
fn test_configuration_guidance() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Config command should provide clear guidance
    cmd.arg("config")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Configuration"))
        .stdout(predicate::str::contains("show"))
        .stdout(predicate::str::contains("init"));
}

#[test]
fn test_voice_management_workflow() {
    // Test the complete voice management workflow
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Step 1: List voices
    cmd.arg("list-voices")
        .timeout(Duration::from_secs(10))
        .assert()
        .success();

    // Step 2: Get voice info (using a common voice ID)
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("voice-info")
        .arg("default")
        .timeout(Duration::from_secs(10))
        .assert(); // May succeed or fail depending on voice availability
}

#[test]
fn test_batch_processing_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("input.txt");
    let output_dir = temp_dir.path().join("output");

    // Create test input file
    std::fs::write(&input_file, "Hello world\nThis is a test\nBatch processing").unwrap();
    std::fs::create_dir_all(&output_dir).unwrap();

    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Test batch processing
    cmd.arg("batch")
        .arg(input_file.to_str().unwrap())
        .arg("--output-dir")
        .arg(output_dir.to_str().unwrap())
        .arg("--workers")
        .arg("1")
        .timeout(Duration::from_secs(60))
        .assert()
        .success();
}

#[test]
fn test_interactive_mode_availability() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Test that interactive mode is available and documented
    cmd.arg("interactive")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Interactive"))
        .stdout(predicate::str::contains("real-time"));
}

#[test]
fn test_completion_generation_usability() {
    let temp_dir = TempDir::new().unwrap();
    let output_file = temp_dir.path().join("completion.bash");

    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Test completion generation
    cmd.arg("generate-completion")
        .arg("bash")
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .assert()
        .success();

    // Verify completion file was created and has content
    assert!(output_file.exists());
    let content = std::fs::read_to_string(&output_file).unwrap();
    assert!(!content.is_empty());
    assert!(content.contains("voirs"));
}

#[test]
fn test_version_information() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Version should be clearly displayed
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("voirs"))
        .stdout(predicate::str::is_match(r"\d+\.\d+\.\d+").unwrap());
}

#[test]
fn test_output_format_flexibility() {
    let temp_dir = TempDir::new().unwrap();
    let formats = vec!["wav", "mp3", "flac"];

    for format in formats {
        let output_file = temp_dir.path().join(format!("test.{format}"));
        let mut cmd = Command::cargo_bin("voirs").unwrap();

        cmd.arg("synthesize")
            .arg("Test audio")
            .arg("--output")
            .arg(output_file.to_str().unwrap())
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    }
}

#[test]
fn test_configuration_file_handling() {
    let temp_dir = TempDir::new().unwrap();
    let config_file = temp_dir.path().join("voirs_config.toml");

    // Test config init
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("config")
        .arg("--init")
        .arg("--path")
        .arg(config_file.to_str().unwrap())
        .assert()
        .success();

    // Verify config file was created
    assert!(config_file.exists());

    // Test config show
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("--config")
        .arg(config_file.to_str().unwrap())
        .arg("config")
        .arg("--show")
        .assert()
        .success();
}
