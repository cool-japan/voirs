//! Unit tests for CLI argument parsing

use clap::Parser;
use clap_complete::Shell;
use voirs_cli::{CliApp, Commands};

#[test]
fn test_basic_synthesize_command() {
    let args = vec!["voirs", "synthesize", "Hello, world!"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Synthesize { text, .. } => {
            assert_eq!(text, "Hello, world!");
        }
        _ => panic!("Expected Synthesize command"),
    }
}

#[test]
fn test_synthesize_with_options() {
    let args = vec![
        "voirs",
        "synthesize",
        "Test text",
        "--output",
        "test.wav",
        "--rate",
        "1.5",
        "--pitch",
        "2.0",
        "--volume",
        "5.0",
        "--quality",
        "ultra",
        "--enhance",
    ];

    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Synthesize {
            text,
            output,
            rate,
            pitch,
            volume,
            quality,
            enhance,
        } => {
            assert_eq!(text, "Test text");
            assert_eq!(output.unwrap().to_str().unwrap(), "test.wav");
            assert_eq!(rate, 1.5);
            assert_eq!(pitch, 2.0);
            assert_eq!(volume, 5.0);
            assert!(enhance);
        }
        _ => panic!("Expected Synthesize command"),
    }
}

#[test]
fn test_guide_command() {
    let args = vec!["voirs", "guide"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Guide { command, .. } => {
            assert!(command.is_none());
        }
        _ => panic!("Expected Guide command"),
    }
}

#[test]
fn test_guide_with_specific_command() {
    let args = vec!["voirs", "guide", "synthesize"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Guide { command, .. } => {
            assert_eq!(command.unwrap(), "synthesize");
        }
        _ => panic!("Expected Guide command"),
    }
}

#[test]
fn test_completion_command() {
    let args = vec!["voirs", "generate-completion", "bash"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::GenerateCompletion { shell, .. } => {
            assert_eq!(shell, Shell::Bash);
        }
        _ => panic!("Expected GenerateCompletion command"),
    }
}

#[test]
fn test_server_command() {
    let args = vec!["voirs", "server", "--port", "3000", "--host", "0.0.0.0"];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Server { port, host } => {
            assert_eq!(port, 3000);
            assert_eq!(host, "0.0.0.0");
        }
        _ => panic!("Expected Server command"),
    }
}

#[test]
fn test_interactive_command() {
    let args = vec![
        "voirs",
        "interactive",
        "--voice",
        "en-us-female",
        "--no-audio",
    ];
    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Interactive {
            voice, no_audio, ..
        } => {
            assert_eq!(voice.unwrap(), "en-us-female");
            assert!(no_audio);
        }
        _ => panic!("Expected Interactive command"),
    }
}

#[test]
fn test_batch_command() {
    let args = vec![
        "voirs",
        "batch",
        "input.txt",
        "--workers",
        "4",
        "--rate",
        "0.8",
        "--resume",
    ];

    let app = CliApp::try_parse_from(args).unwrap();

    match app.command {
        Commands::Batch {
            workers,
            rate,
            resume,
            ..
        } => {
            assert_eq!(workers.unwrap(), 4);
            assert_eq!(rate, 0.8);
            assert!(resume);
        }
        _ => panic!("Expected Batch command"),
    }
}

#[test]
fn test_global_options() {
    let args = vec![
        "voirs",
        "--verbose",
        "--config",
        "custom.toml",
        "--gpu",
        "--threads",
        "8",
        "synthesize",
        "test",
    ];

    let app = CliApp::try_parse_from(args).unwrap();

    assert_eq!(app.global.verbose, 1);
    assert_eq!(app.global.config.unwrap().to_str().unwrap(), "custom.toml");
    assert!(app.global.gpu);
    assert_eq!(app.global.threads.unwrap(), 8);
}

#[test]
fn test_invalid_command_fails() {
    let args = vec!["voirs", "invalid-command"];
    let result = CliApp::try_parse_from(args);
    assert!(result.is_err());
}

#[test]
fn test_missing_required_arg_fails() {
    let args = vec!["voirs", "synthesize"];
    let result = CliApp::try_parse_from(args);
    assert!(result.is_err());
}
