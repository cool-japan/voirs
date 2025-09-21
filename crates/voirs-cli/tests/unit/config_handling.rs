//! Unit tests for configuration handling

use voirs_cli::GlobalOptions;

#[test]
fn test_global_options_defaults() {
    let options = GlobalOptions {
        config: None,
        verbose: 0,
        quiet: false,
        format: None,
        voice: None,
        gpu: false,
        threads: None,
    };

    assert!(options.config.is_none());
    assert_eq!(options.verbose, 0);
    assert!(!options.quiet);
    assert!(options.format.is_none());
    assert!(options.voice.is_none());
    assert!(!options.gpu);
    assert!(options.threads.is_none());
}

#[test]
fn test_global_options_with_values() {
    use std::path::PathBuf;
    use voirs_cli::cli_types::CliAudioFormat;

    let options = GlobalOptions {
        config: Some(PathBuf::from("config.toml")),
        verbose: 2,
        quiet: false,
        format: Some(CliAudioFormat::Flac),
        voice: Some("en-us-female".to_string()),
        gpu: true,
        threads: Some(4),
    };

    assert_eq!(options.config.unwrap().to_str().unwrap(), "config.toml");
    assert_eq!(options.verbose, 2);
    assert_eq!(options.format.unwrap(), CliAudioFormat::Flac);
    assert_eq!(options.voice.unwrap(), "en-us-female");
    assert!(options.gpu);
    assert_eq!(options.threads.unwrap(), 4);
}
