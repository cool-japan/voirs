//! Integration tests for help system

use clap::Parser;
use clap_complete::Shell;
use voirs_cli::{CliApp, Commands};

#[test]
fn test_guide_command_integration() {
    let args = vec!["voirs", "guide"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Guide {
            command,
            getting_started,
            examples,
        } => {
            assert!(command.is_none());
            assert!(!getting_started);
            assert!(!examples);
        }
        _ => panic!("Expected Guide command"),
    }
}

#[test]
fn test_guide_with_command_integration() {
    let commands = vec![
        "synthesize",
        "list-voices",
        "server",
        "interactive",
        "batch",
    ];

    for cmd in commands {
        let args = vec!["voirs", "guide", cmd];
        let app = CliApp::try_parse_from(args).unwrap();

        match app.command {
            Commands::Guide { command, .. } => {
                assert_eq!(command.unwrap(), cmd);
            }
            _ => panic!("Expected Guide command for {cmd}"),
        }
    }
}

#[test]
fn test_guide_getting_started_flag() {
    let args = vec!["voirs", "guide", "--getting-started"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Guide {
            getting_started, ..
        } => {
            assert!(getting_started);
        }
        _ => panic!("Expected Guide command"),
    }
}

#[test]
fn test_guide_examples_flag() {
    let args = vec!["voirs", "guide", "--examples"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Guide { examples, .. } => {
            assert!(examples);
        }
        _ => panic!("Expected Guide command"),
    }
}

#[test]
fn test_completion_generation_integration() {
    let shells = vec!["bash", "zsh", "fish", "powershell"];

    for shell in shells {
        let args = vec!["voirs", "generate-completion", shell];
        let app = CliApp::try_parse_from(args);
        assert!(
            app.is_ok(),
            "Failed to parse completion command for {shell}"
        );
    }
}

#[test]
fn test_completion_with_output_file() {
    let args = vec![
        "voirs",
        "generate-completion",
        "bash",
        "--output",
        "completion.bash",
    ];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::GenerateCompletion { shell, output, .. } => {
            assert_eq!(shell, Shell::Bash);
            assert_eq!(output.unwrap().to_str().unwrap(), "completion.bash");
        }
        _ => panic!("Expected GenerateCompletion command"),
    }
}

#[test]
fn test_completion_install_help() {
    let args = vec!["voirs", "generate-completion", "zsh", "--install-help"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::GenerateCompletion {
            shell,
            install_help,
            ..
        } => {
            assert_eq!(shell, Shell::Zsh);
            assert!(install_help);
        }
        _ => panic!("Expected GenerateCompletion command"),
    }
}

#[test]
fn test_completion_status() {
    let args = vec!["voirs", "generate-completion", "bash", "--status"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::GenerateCompletion { status, .. } => {
            assert!(status);
        }
        _ => panic!("Expected GenerateCompletion command"),
    }
}
