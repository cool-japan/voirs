//! Tests for CLI exit codes

use assert_cmd::Command;
use std::process::ExitStatus;

fn get_exit_code(status: ExitStatus) -> Option<i32> {
    #[cfg(unix)]
    {
        status.code()
    }
    #[cfg(windows)]
    {
        status.code()
    }
}

#[test]
fn test_guide_success_exit_code() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    let output = cmd.arg("guide").output().unwrap();

    assert!(output.status.success());
    assert_eq!(get_exit_code(output.status), Some(0));
}

#[test]
fn test_version_success_exit_code() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    let output = cmd.arg("--version").output().unwrap();

    assert!(output.status.success());
    assert_eq!(get_exit_code(output.status), Some(0));
}

#[test]
fn test_invalid_command_error_exit_code() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    let output = cmd.arg("invalid-command").output().unwrap();

    assert!(!output.status.success());
    // Should return non-zero exit code
    assert_ne!(get_exit_code(output.status), Some(0));
}

#[test]
fn test_missing_argument_error_exit_code() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    let output = cmd.arg("synthesize").output().unwrap();

    assert!(!output.status.success());
    assert_ne!(get_exit_code(output.status), Some(0));
}

#[test]
fn test_completion_generation_success_exit_code() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    let output = cmd.arg("generate-completion").arg("bash").output().unwrap();

    assert!(output.status.success());
    assert_eq!(get_exit_code(output.status), Some(0));
}

#[test]
fn test_guide_with_command_success_exit_code() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    let output = cmd.arg("guide").arg("synthesize").output().unwrap();

    assert!(output.status.success());
    assert_eq!(get_exit_code(output.status), Some(0));
}

#[test]
fn test_invalid_shell_error_exit_code() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    let output = cmd
        .arg("generate-completion")
        .arg("invalid-shell")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_ne!(get_exit_code(output.status), Some(0));
}

#[test]
fn test_invalid_argument_value_error_exit_code() {
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    let output = cmd
        .arg("synthesize")
        .arg("test")
        .arg("--rate")
        .arg("invalid-rate")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_ne!(get_exit_code(output.status), Some(0));
}
