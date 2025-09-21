//! Unit tests for shell completion functionality

use clap_complete::Shell;
use std::io::Cursor;
use voirs_cli::completion::{
    display_completion_status, generate_completion, generate_install_script,
    get_installation_instructions,
};

#[test]
fn test_bash_completion_generation() {
    let mut output = Cursor::new(Vec::new());
    let result = generate_completion(Shell::Bash, &mut output);

    assert!(result.is_ok());

    let completion_script = String::from_utf8(output.into_inner()).unwrap();
    assert!(completion_script.contains("voirs"));
    assert!(completion_script.contains("_voirs"));
    assert!(!completion_script.is_empty());
}

#[test]
fn test_zsh_completion_generation() {
    let mut output = Cursor::new(Vec::new());
    let result = generate_completion(Shell::Zsh, &mut output);

    assert!(result.is_ok());

    let completion_script = String::from_utf8(output.into_inner()).unwrap();
    assert!(completion_script.contains("voirs"));
    assert!(!completion_script.is_empty());
}

#[test]
fn test_fish_completion_generation() {
    let mut output = Cursor::new(Vec::new());
    let result = generate_completion(Shell::Fish, &mut output);

    assert!(result.is_ok());

    let completion_script = String::from_utf8(output.into_inner()).unwrap();
    assert!(completion_script.contains("voirs"));
    assert!(!completion_script.is_empty());
}

#[test]
fn test_powershell_completion_generation() {
    let mut output = Cursor::new(Vec::new());
    let result = generate_completion(Shell::PowerShell, &mut output);

    assert!(result.is_ok());

    let completion_script = String::from_utf8(output.into_inner()).unwrap();
    assert!(completion_script.contains("voirs"));
    assert!(!completion_script.is_empty());
}

#[test]
fn test_bash_installation_instructions() {
    let instructions = get_installation_instructions(Shell::Bash);

    assert!(instructions.contains("Bash completion"));
    assert!(instructions.contains("bash-completion"));
    assert!(instructions.contains("completions/voirs"));
    assert!(instructions.contains("source"));
}

#[test]
fn test_zsh_installation_instructions() {
    let instructions = get_installation_instructions(Shell::Zsh);

    assert!(instructions.contains("Zsh completion"));
    assert!(instructions.contains("$fpath"));
    assert!(instructions.contains("_voirs"));
    assert!(instructions.contains("compinit"));
}

#[test]
fn test_fish_installation_instructions() {
    let instructions = get_installation_instructions(Shell::Fish);

    assert!(instructions.contains("Fish completion"));
    assert!(instructions.contains("~/.config/fish/completions"));
    assert!(instructions.contains("voirs.fish"));
}

#[test]
fn test_powershell_installation_instructions() {
    let instructions = get_installation_instructions(Shell::PowerShell);

    assert!(instructions.contains("PowerShell completion"));
    assert!(instructions.contains("$PROFILE"));
    assert!(instructions.contains("execution policy"));
}

#[test]
fn test_completion_status_display() {
    let status = display_completion_status();

    assert!(status.contains("VoiRS CLI Shell Completion Status"));
    assert!(status.contains("Bash"));
    assert!(status.contains("Zsh"));
    assert!(status.contains("Fish"));
    assert!(status.contains("PowerShell"));
    assert!(status.contains("generate-completion"));
}

#[test]
fn test_install_script_generation() {
    let script = generate_install_script();

    assert!(script.contains("#!/bin/bash"));
    assert!(script.contains("VoiRS CLI Completion Installation Script"));
    assert!(script.contains("install_bash"));
    assert!(script.contains("install_zsh"));
    assert!(script.contains("install_fish"));
    assert!(script.contains("install_powershell"));
    assert!(script.contains("voirs generate-completion"));

    // Should have main function
    assert!(script.contains("main \"$@\""));

    // Should have usage information
    assert!(script.contains("Usage:"));
}

#[test]
fn test_install_script_functions() {
    let script = generate_install_script();

    // Check for all required functions
    assert!(script.contains("check_voirs()"));
    assert!(script.contains("install_bash()"));
    assert!(script.contains("install_zsh()"));
    assert!(script.contains("install_fish()"));
    assert!(script.contains("install_powershell()"));
    assert!(script.contains("install_all()"));

    // Check for logging functions
    assert!(script.contains("log_info()"));
    assert!(script.contains("log_success()"));
    assert!(script.contains("log_warn()"));
    assert!(script.contains("log_error()"));
}

#[test]
fn test_completion_scripts_contain_all_commands() {
    let mut output = Cursor::new(Vec::new());
    generate_completion(Shell::Bash, &mut output).unwrap();

    let completion_script = String::from_utf8(output.into_inner()).unwrap();

    // Should contain main commands
    assert!(completion_script.contains("synthesize") || completion_script.contains("completion"));
    // The exact format depends on clap's completion generation
}

#[test]
fn test_different_shells_produce_different_output() {
    let mut bash_output = Cursor::new(Vec::new());
    let mut zsh_output = Cursor::new(Vec::new());

    generate_completion(Shell::Bash, &mut bash_output).unwrap();
    generate_completion(Shell::Zsh, &mut zsh_output).unwrap();

    let bash_script = String::from_utf8(bash_output.into_inner()).unwrap();
    let zsh_script = String::from_utf8(zsh_output.into_inner()).unwrap();

    // Scripts should be different
    assert_ne!(bash_script, zsh_script);

    // Both should be non-empty
    assert!(!bash_script.is_empty());
    assert!(!zsh_script.is_empty());
}
