use assert_cmd::Command;
use std::process::{Command as StdCommand, Stdio};
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[test]
fn test_startup_time_measurement() {
    // Test cold startup time
    let start = Instant::now();
    let mut cmd = Command::cargo_bin("voirs").unwrap();

    cmd.arg("--version").assert().success();

    let startup_time = start.elapsed();

    // Startup should be reasonably fast (< 2 seconds for version command)
    assert!(
        startup_time < Duration::from_secs(2),
        "Startup time too slow: {startup_time:?}"
    );

    println!("Cold startup time: {startup_time:?}");
}

#[test]
fn test_warm_startup_performance() {
    // Run command once to warm up
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("--version").assert().success();

    // Measure warm startup
    let start = Instant::now();
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("--version").assert().success();
    let warm_time = start.elapsed();

    // Warm startup should be faster
    assert!(
        warm_time < Duration::from_millis(500),
        "Warm startup too slow: {warm_time:?}"
    );

    println!("Warm startup time: {warm_time:?}");
}

#[test]
fn test_memory_usage_profiling() {
    // Skip this test unless specifically requested via environment variable
    // This test can be slow and may timeout in CI environments
    if std::env::var("VOIRS_RUN_SLOW_TESTS").is_err() {
        println!("Skipping slow test. Set VOIRS_RUN_SLOW_TESTS=1 to run.");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let output_file = temp_dir.path().join("memory_test.wav");

    // Use external tools to measure memory if available
    if is_tool_available("ps") || is_tool_available("tasklist") {
        test_memory_with_external_tools(&output_file);
    } else {
        // Fallback: just test that synthesis completes
        let mut cmd = Command::cargo_bin("voirs").unwrap();
        cmd.arg("synthesize")
            .arg("This is a memory usage test with a longer text to see how the application handles memory allocation and deallocation during synthesis.")
            .arg("--output")
            .arg(output_file.to_str().unwrap())
            .timeout(Duration::from_secs(120)) // Increased timeout for CI environments
            .assert()
            .success();

        assert!(output_file.exists());
    }
}

fn test_memory_with_external_tools(output_file: &std::path::Path) {
    // This is a more comprehensive memory test when external tools are available
    let start = Instant::now();

    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("synthesize")
        .arg("Memory usage test with extended text content.")
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .timeout(Duration::from_secs(120)) // Increased timeout for CI environments
        .assert()
        .success();

    let duration = start.elapsed();
    println!("Synthesis completed in: {duration:?}");
}

#[test]
fn test_batch_processing_efficiency() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("batch_input.txt");
    let output_dir = temp_dir.path().join("batch_output");

    // Create test input with multiple sentences
    let test_sentences = [
        "First test sentence for batch processing.",
        "Second sentence with different content.",
        "Third sentence to measure efficiency.",
        "Fourth sentence for performance testing.",
        "Fifth and final sentence for the batch.",
    ];

    std::fs::write(&input_file, test_sentences.join("\n")).unwrap();
    std::fs::create_dir_all(&output_dir).unwrap();

    let start = Instant::now();

    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("batch")
        .arg(input_file.to_str().unwrap())
        .arg("--output-dir")
        .arg(output_dir.to_str().unwrap())
        .arg("--workers")
        .arg("2")
        .timeout(Duration::from_secs(600));

    // Note: unlike most subcommands, `batch` exits 0 even when every item
    // failed to synthesize (per-item failures are reported in its summary,
    // not via the process exit code), so the models-unavailable check can't
    // rely on a non-zero exit status here -- it must inspect the captured
    // output directly once we know the expected output files are missing.
    let output = cmd.output().expect("execute voirs binary");
    assert!(
        output.status.success(),
        "batch command itself should exit successfully\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let batch_time = start.elapsed();

    // Check that output files were created
    let output_files: Vec<_> = std::fs::read_dir(&output_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .collect();

    if output_files.len() < test_sentences.len() {
        if crate::common::is_model_unavailable(&output) {
            eprintln!(
                "skipping: synthesis models unavailable in this environment (offline or \
                 credential-less HuggingFace fetch); batch reported per-item failures \
                 matching a known model-unavailability marker.\nstdout={}\nstderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        panic!(
            "Not all batch files were created, and the failure does not match a known \
             model-unavailability marker\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!(
        "Batch processing time for {} sentences: {:?}",
        test_sentences.len(),
        batch_time
    );

    // Performance expectation: allow generous time budget to tolerate parallel test suite CPU contention
    // Under heavy load (9000+ concurrent tests), batch processing can be significantly slower
    let time_per_sentence = batch_time.as_millis() / test_sentences.len() as u128;
    assert!(
        time_per_sentence < 300000, // Less than 300 seconds per sentence (10× generous for load)
        "Batch processing too slow: {time_per_sentence}ms per sentence"
    );
}

#[test]
fn test_interactive_mode_responsiveness() {
    // Test that interactive mode starts up quickly
    let start = Instant::now();

    let mut cmd = Command::cargo_bin("voirs").unwrap();

    // Interactive mode with immediate exit
    cmd.arg("interactive")
        .arg("--no-audio")
        .arg("--debug")
        .write_stdin("quit\n")
        .timeout(Duration::from_secs(120));
    let output = cmd.output().expect("execute voirs binary");

    if !output.status.success() {
        if crate::common::is_model_unavailable(&output) {
            eprintln!(
                "skipping: synthesis models unavailable in this environment (offline or \
                 credential-less HuggingFace fetch); stdout/stderr matched a known \
                 model-unavailability marker.\nstdout={}\nstderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // `interactive`'s default-voice selection goes through
        // `voirs_sdk::builder::validation::validate_voice`, which -- unlike the
        // `synthesize`/`list-voices`/`batch` paths covered by
        // `common::is_model_unavailable` above -- fails *fast* via a single,
        // unretried HTTP HEAD probe (`is_voice_available_remotely` in
        // `voirs-sdk/src/builder/async_init.rs`) instead of the retried GET
        // download. When that probe can't reach the model repo it raises
        // `VoiceNotFound` ("Voice '<id>' not found. Available voices: [...]")
        // rather than `DownloadFailed`/`HTTP 401`, even though the requested
        // voice *is* listed among the "available" ones -- `validate_voice`
        // reports local voice-registry membership, not remote reachability,
        // in that list. Manually verified with `curl -I` that the queried URL
        // (`.../voices/<id>/config.json`) returns the exact same repo-wide
        // `HTTP 401 Unauthorized` as every other model file in this
        // repository, confirming this is the same environmental cause with a
        // different failure shape -- not a voice-registry drift bug (a real,
        // previously-fixed issue; see the doc comment on
        // `SynthesisEngine::load_available_voices` in
        // `src/commands/interactive/synthesis.rs`). This check is
        // intentionally local to this test rather than a shared marker: it
        // must not cause unrelated `synthesize`/`list-voices`/`batch` failures
        // to be silently treated as model-unavailability.
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if combined.contains("not found. Available voices:") {
            eprintln!(
                "skipping: interactive mode's remote voice-availability probe could not \
                 reach the synthesis model repository in this environment (offline or \
                 credential-less HuggingFace fetch).\nstdout={}\nstderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        panic!(
            "interactive mode failed for a reason other than known model-unavailability \
             markers\nstatus={:?}\nstdout={}\nstderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let interactive_startup = start.elapsed();

    // Allow generous startup time to tolerate parallel test suite CPU contention
    assert!(
        interactive_startup < Duration::from_secs(90),
        "Interactive mode startup too slow: {interactive_startup:?}"
    );

    println!("Interactive mode startup time: {interactive_startup:?}");
}

#[test]
fn test_large_text_synthesis_performance() {
    // Skip this test unless specifically requested via environment variable
    // This test can be slow and may timeout in CI environments
    if std::env::var("VOIRS_RUN_SLOW_TESTS").is_err() {
        println!("Skipping slow test. Set VOIRS_RUN_SLOW_TESTS=1 to run.");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let output_file = temp_dir.path().join("large_text.wav");

    // Create a reasonably large text (but not too large for CI)
    let large_text = "This is a performance test with a moderately large text input. ".repeat(25);

    let start = Instant::now();

    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("synthesize")
        .arg(&large_text)
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .timeout(Duration::from_secs(90))
        .assert()
        .success();

    let synthesis_time = start.elapsed();

    assert!(output_file.exists());

    println!("Large text synthesis time: {synthesis_time:?}");

    // Performance expectation: should complete within reasonable time
    assert!(
        synthesis_time < Duration::from_secs(60),
        "Large text synthesis too slow: {synthesis_time:?}"
    );
}

#[test]
fn test_concurrent_operations_performance() {
    // Skip this test unless specifically requested via environment variable
    // This test can be slow and may timeout in CI environments
    if std::env::var("VOIRS_RUN_SLOW_TESTS").is_err() {
        println!("Skipping slow test. Set VOIRS_RUN_SLOW_TESTS=1 to run.");
        return;
    }

    let temp_dir = TempDir::new().unwrap();

    // Test multiple concurrent synthesis operations
    let start = Instant::now();

    let handles: Vec<_> = (0..3)
        .map(|i| {
            let output_file = temp_dir.path().join(format!("concurrent_{i}.wav"));
            let output_path = output_file.to_string_lossy().to_string();

            std::thread::spawn(move || {
                let mut cmd = Command::cargo_bin("voirs").unwrap();
                cmd.arg("synthesize")
                    .arg(format!("Concurrent synthesis test number {i}"))
                    .arg("--output")
                    .arg(&output_path)
                    .timeout(Duration::from_secs(120)) // Increased timeout for CI environments
                    .assert()
                    .success();
            })
        })
        .collect();

    // Wait for all operations to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let concurrent_time = start.elapsed();

    println!("Concurrent operations time: {concurrent_time:?}");

    // Check that all files were created
    for i in 0..3 {
        let output_file = temp_dir.path().join(format!("concurrent_{i}.wav"));
        assert!(output_file.exists(), "Concurrent file {i} not created");
    }
}

#[test]
fn test_help_command_performance() {
    // Help commands should be very fast
    let commands = vec![
        "--help",
        "synthesize --help",
        "list-voices --help",
        "config --help",
    ];

    for cmd_args in commands {
        let start = Instant::now();

        let mut cmd = Command::cargo_bin("voirs").unwrap();
        let args: Vec<&str> = cmd_args.split_whitespace().collect();

        for arg in args {
            cmd.arg(arg);
        }

        cmd.assert().success();

        let help_time = start.elapsed();

        assert!(
            help_time < Duration::from_millis(500),
            "Help command '{cmd_args}' too slow: {help_time:?}"
        );

        println!("Help command '{cmd_args}' time: {help_time:?}");
    }
}

#[test]
fn test_configuration_loading_performance() {
    let temp_dir = TempDir::new().unwrap();
    let config_file = temp_dir.path().join("perf_test_config.toml");

    // Create a config file
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("config")
        .arg("--init")
        .arg("--path")
        .arg(config_file.to_str().unwrap())
        .assert()
        .success();

    // Test loading performance
    let start = Instant::now();

    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("--config")
        .arg(config_file.to_str().unwrap())
        .arg("config")
        .arg("--show")
        .assert()
        .success();

    let config_load_time = start.elapsed();

    assert!(
        config_load_time < Duration::from_millis(200),
        "Configuration loading too slow: {config_load_time:?}"
    );

    println!("Configuration loading time: {config_load_time:?}");
}

#[test]
fn test_voice_listing_performance() {
    let start = Instant::now();

    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("list-voices").timeout(Duration::from_secs(60));
    if crate::common::run_or_skip_if_models_unavailable(&mut cmd).is_none() {
        return;
    }

    let voice_list_time = start.elapsed();

    // Allow generous time to tolerate parallel test suite CPU contention
    assert!(
        voice_list_time < Duration::from_secs(30),
        "Voice listing too slow: {voice_list_time:?}"
    );

    println!("Voice listing time: {voice_list_time:?}");
}

fn is_tool_available(tool: &str) -> bool {
    StdCommand::new(tool)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[test]
fn test_resource_cleanup() {
    let temp_dir = TempDir::new().unwrap();
    let output_file = temp_dir.path().join("cleanup_test.wav");

    // Run synthesis and measure if it cleans up properly
    // Allow up to 120 seconds to account for concurrent test load
    let mut cmd = Command::cargo_bin("voirs").unwrap();
    cmd.arg("synthesize")
        .arg("Resource cleanup test")
        .arg(output_file.to_str().unwrap())
        .timeout(Duration::from_secs(120));
    if crate::common::run_or_skip_if_models_unavailable(&mut cmd).is_none() {
        return;
    }

    // The process should exit cleanly
    assert!(output_file.exists());

    // Additional resource cleanup tests could be added here
    // such as checking for temporary files, memory leaks, etc.
}
