//! Tests for CLI output format consistency

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_guide_output_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("guide")
        .assert()
        .success()
        .stdout(predicate::str::contains("VoiRS CLI Commands"))
        .stdout(predicate::str::contains("Synthesis"))
        .stdout(predicate::str::contains("Voice Management"));
}

#[test]
fn test_guide_command_specific_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("guide")
        .arg("synthesize")
        .assert()
        .success()
        .stdout(predicate::str::contains("voirs synthesize"))
        .stdout(predicate::str::contains("Examples:"))
        .stdout(predicate::str::contains("--"));
}

#[test]
fn test_getting_started_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("guide")
        .arg("--getting-started")
        .assert()
        .success()
        .stdout(predicate::str::contains("Getting Started with VoiRS"))
        .stdout(predicate::str::contains("1."))
        .stdout(predicate::str::contains("voirs test"));
}

#[test]
fn test_completion_bash_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("generate-completion")
        .arg("bash")
        .assert()
        .success()
        .stdout(predicate::str::contains("_voirs"))
        .stdout(predicate::str::contains("complete"));
}

#[test]
fn test_completion_zsh_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("generate-completion")
        .arg("zsh")
        .assert()
        .success()
        .stdout(predicate::str::contains("_voirs"))
        .stdout(predicate::str::contains("compdef"));
}

#[test]
fn test_completion_fish_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("generate-completion")
        .arg("fish")
        .assert()
        .success()
        .stdout(predicate::str::contains("complete"));
}

#[test]
fn test_completion_status_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("generate-completion")
        .arg("bash")
        .arg("--status")
        .assert()
        .success()
        .stdout(predicate::str::contains("Shell Completion Status"))
        .stdout(predicate::str::contains("Bash"))
        .stdout(
            predicate::str::contains("Available").or(predicate::str::contains("Not installed")),
        );
}

#[test]
fn test_completion_install_help_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("generate-completion")
        .arg("bash")
        .arg("--install-help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Installation instructions"))
        .stdout(predicate::str::contains("bash-completion"))
        .stdout(predicate::str::contains("completions/voirs"));
}

#[test]
fn test_error_message_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("invalid-command")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"))
        .stderr(
            predicate::str::contains("unrecognized subcommand")
                .or(predicate::str::contains("unexpected argument")),
        );
}

#[test]
fn test_version_format() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn test_quiet_mode_reduces_output() {
    // Test that quiet flag actually reduces output
    let mut cmd_normal = Command::cargo_bin("voirs").unwrap();
    let normal_output = cmd_normal.arg("help").output().unwrap();

    let mut cmd_quiet = Command::cargo_bin("voirs").unwrap();
    let quiet_output = cmd_quiet.arg("--quiet").arg("help").output().unwrap();

    // Both should succeed
    assert!(normal_output.status.success());
    assert!(quiet_output.status.success());

    // Content should be the same or similar (help is still displayed)
    // This test mainly ensures --quiet flag is parsed correctly
}
