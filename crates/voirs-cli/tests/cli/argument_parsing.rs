//! Tests for command-line argument parsing

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_guide_command() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("guide")
        .assert()
        .success()
        .stdout(predicate::str::contains("VoiRS CLI Commands"));
}

#[test]
fn test_guide_with_command() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("guide")
        .arg("synthesize")
        .assert()
        .success()
        .stdout(predicate::str::contains("synthesize"));
}

#[test]
fn test_guide_getting_started() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("guide")
        .arg("--getting-started")
        .assert()
        .success()
        .stdout(predicate::str::contains("Getting Started with VoiRS"));
}

#[test]
fn test_completion_generation() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("generate-completion")
        .arg("bash")
        .assert()
        .success()
        .stdout(predicate::str::contains("voirs"));
}

#[test]
fn test_completion_status() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("generate-completion")
        .arg("bash")
        .arg("--status")
        .assert()
        .success()
        .stdout(predicate::str::contains("Shell Completion Status"));
}

#[test]
fn test_completion_install_help() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("generate-completion")
        .arg("zsh")
        .arg("--install-help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Zsh completion"));
}

#[test]
fn test_invalid_command() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("invalid-command")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[test]
fn test_missing_required_argument() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("synthesize")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn test_version_flag() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn test_verbose_flag() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("--verbose").arg("guide").assert().success();
}

#[test]
fn test_config_flag() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("--config")
        .arg("test.toml")
        .arg("guide")
        .assert()
        .success();
}

#[test]
fn test_global_flags_with_commands() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("--verbose")
        .arg("--gpu")
        .arg("--threads")
        .arg("4")
        .arg("guide")
        .assert()
        .success();
}

#[test]
fn test_synthesize_command_structure() {
    // This would normally fail due to missing voice/models, but should parse correctly
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("synthesize")
        .arg("Hello world")
        .arg("--output")
        .arg("test.wav")
        .arg("--rate")
        .arg("1.2")
        .arg("--quality")
        .arg("high")
        .assert()
        .failure(); // Expected to fail due to missing dependencies, but args should parse
}

#[test]
fn test_server_command_structure() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("server")
        .arg("--port")
        .arg("3000")
        .arg("--host")
        .arg("0.0.0.0")
        .timeout(std::time::Duration::from_secs(1)) // Don't actually start server
        .assert()
        .failure(); // Expected to timeout/fail, but args should parse
}

#[test]
fn test_batch_command_structure() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("batch")
        .arg("nonexistent.txt")
        .arg("--workers")
        .arg("2")
        .arg("--rate")
        .arg("1.0")
        .assert()
        .failure(); // Expected to fail due to missing file, but args should parse
}
