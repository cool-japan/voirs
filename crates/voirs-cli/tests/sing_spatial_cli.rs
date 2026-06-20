//! Integration tests for `sing` and `spatial` CLI subcommands.
//!
//! All tests use `assert_cmd::Command::cargo_bin("voirs")` to drive the actual binary.
//! File I/O is isolated to temp directories obtained via `tempfile::tempdir()`.

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write as IoWrite;

// ---------------------------------------------------------------------------
// Group 1 — Alias resolution (--help, exit 0)
// ---------------------------------------------------------------------------

#[cfg(feature = "singing")]
#[test]
fn test_sing_from_score_alias_resolves() {
    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args(["sing", "from-score", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("score"));
}

#[cfg(feature = "singing")]
#[test]
fn test_sing_from_midi_alias_resolves() {
    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args(["sing", "from-midi", "--help"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("midi").or(predicate::str::contains("MIDI")),
        );
}

#[cfg(feature = "singing")]
#[test]
fn test_sing_apply_effects_alias_resolves() {
    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args(["sing", "apply-effects", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[cfg(feature = "singing")]
#[test]
fn test_sing_create_model_alias_resolves() {
    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args(["sing", "create-model", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[cfg(feature = "spatial")]
#[test]
fn test_spatial_synthesize_alias_resolves() {
    // The help output contains "<TEXT>" (uppercase) and "Text to synthesize".
    // Use "synthesize" which appears verbatim in the description.
    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args(["spatial", "synthesize", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("synthesize"));
}

#[cfg(feature = "spatial")]
#[test]
fn test_spatial_apply_hrtf_alias_resolves() {
    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args(["spatial", "apply-hrtf", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[cfg(feature = "spatial")]
#[test]
fn test_spatial_apply_room_alias_resolves() {
    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args(["spatial", "apply-room", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[cfg(feature = "spatial")]
#[test]
fn test_spatial_animate_alias_resolves() {
    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args(["spatial", "animate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

// ---------------------------------------------------------------------------
// Group 2 — `sing create-voice` (no WAV)
// ---------------------------------------------------------------------------

#[cfg(feature = "singing")]
#[test]
fn test_sing_create_voice_writes_json() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let samples_dir = dir.path().join("samples");
    std::fs::create_dir_all(&samples_dir).expect("create samples dir");

    let output = dir.path().join("test-voice.json");

    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args([
            "sing",
            "create-voice",
            samples_dir.to_str().expect("samples dir is valid UTF-8"),
            "--output",
            output.to_str().expect("output path is valid UTF-8"),
            "--name",
            "test-voice",
            "--voice-type",
            "soprano",
        ])
        .assert()
        .success();

    assert!(output.exists(), "output JSON should exist");

    let content = std::fs::read_to_string(&output).expect("read output JSON");
    let val: serde_json::Value =
        serde_json::from_str(&content).expect("output should be valid JSON");
    assert!(
        val.get("voice_type").is_some(),
        "JSON should contain 'voice_type' field"
    );
}

// ---------------------------------------------------------------------------
// Group 3 — `sing from-score` (minimal MusicXML)
// ---------------------------------------------------------------------------

#[cfg(feature = "singing")]
#[test]
fn test_sing_from_score_writes_wav() {
    let dir = tempfile::tempdir().expect("create tempdir");

    let musicxml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise>
  <work><work-title>Test Song</work-title></work>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths><mode>major</mode></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <lyric><syllabic>single</syllabic><text>la</text></lyric>
      </note>
    </measure>
  </part>
</score-partwise>
"#;

    let score_file = dir.path().join("test.musicxml");
    std::fs::write(&score_file, musicxml).expect("write MusicXML file");

    let output = dir.path().join("out.wav");

    let cmd_output = Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args([
            "sing",
            "from-score",
            "--score",
            score_file.to_str().expect("score path is valid UTF-8"),
            "--voice",
            "soprano",
            output.to_str().expect("output path is valid UTF-8"),
        ])
        .output()
        .expect("execute voirs binary");

    if cmd_output.status.success() {
        // Full synthesis pipeline available: verify WAV output.
        assert!(output.exists(), "output WAV should exist on success");
        let reader = hound::WavReader::open(&output).expect("open output WAV");
        assert!(reader.len() > 0, "output WAV should contain samples");
    } else {
        // In CI/test environments without downloaded synthesis models the engine
        // reports "No synthesis models available".  Treat that as a known-skip
        // condition rather than a test failure.
        let stderr = String::from_utf8_lossy(&cmd_output.stderr);
        let stdout = String::from_utf8_lossy(&cmd_output.stdout);
        let combined = format!("{}{}", stderr, stdout);
        assert!(
            combined.contains("No synthesis models available")
                || combined.contains("synthesis models"),
            "unexpected failure: stderr={stderr} stdout={stdout}"
        );
    }
}

// ---------------------------------------------------------------------------
// Group 4 — `sing effects` (positional input/output)
// ---------------------------------------------------------------------------

/// Write a minimal mono 440 Hz sine WAV to `path` at 44100 Hz (4410 samples).
fn write_mono_sine_wav(path: &std::path::Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create WAV writer");
    let sample_count: u32 = 4410;
    let freq = 440.0_f32;
    let sr = 44100.0_f32;
    for i in 0..sample_count {
        let t = i as f32 / sr;
        let sample = (2.0 * std::f32::consts::PI * freq * t).sin();
        let sample_i16 = (sample * 32767.0) as i16;
        writer.write_sample(sample_i16).expect("write WAV sample");
    }
    writer.finalize().expect("finalize WAV writer");
}

#[cfg(feature = "singing")]
#[test]
fn test_sing_effects_writes_wav() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let input = dir.path().join("input.wav");
    write_mono_sine_wav(&input);

    let output = dir.path().join("output.wav");

    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args([
            "sing",
            "effects",
            input.to_str().expect("input path is valid UTF-8"),
            output.to_str().expect("output path is valid UTF-8"),
            "--effects",
            "reverb",
        ])
        .assert()
        .success();

    assert!(output.exists(), "output WAV should exist");

    let reader = hound::WavReader::open(&output).expect("open output WAV");
    assert!(reader.len() > 0, "output WAV should contain samples");
}

// ---------------------------------------------------------------------------
// Group 5 — `spatial synth`
// ---------------------------------------------------------------------------

#[cfg(feature = "spatial")]
#[test]
fn test_spatial_synth_writes_stereo_wav() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let output = dir.path().join("out.wav");

    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args([
            "spatial",
            "synth",
            "hello",
            output.to_str().expect("output path is valid UTF-8"),
            "--position",
            "1.0,0.0,-1.0",
        ])
        .assert()
        .success();

    assert!(output.exists(), "output WAV should exist");

    let reader = hound::WavReader::open(&output).expect("open output WAV");
    assert_eq!(reader.spec().channels, 2, "output WAV should be stereo");
    assert!(reader.len() > 0, "output WAV should contain samples");
}

// ---------------------------------------------------------------------------
// Group 6 — `spatial hrtf`
// ---------------------------------------------------------------------------

#[cfg(feature = "spatial")]
#[test]
fn test_spatial_hrtf_writes_stereo_wav() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let input = dir.path().join("input.wav");
    write_mono_sine_wav(&input);

    let output = dir.path().join("output.wav");

    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args([
            "spatial",
            "hrtf",
            input.to_str().expect("input path is valid UTF-8"),
            output.to_str().expect("output path is valid UTF-8"),
            "--position",
            "1.0,0.0,-1.0",
        ])
        .assert()
        .success();

    assert!(output.exists(), "output WAV should exist");

    let reader = hound::WavReader::open(&output).expect("open output WAV");
    assert_eq!(reader.spec().channels, 2, "output WAV should be stereo");
    assert!(reader.len() > 0, "output WAV should contain samples");
}

// ---------------------------------------------------------------------------
// Group 7 — `spatial room`
// ---------------------------------------------------------------------------

/// Build a JSON string for a single `Material` object.
fn material_json(name: &str) -> String {
    format!(
        r#"{{
            "name": "{}",
            "absorption_coefficients": [{{"frequency": 1000.0, "coefficient": 0.3}}],
            "scattering_coefficient": 0.1,
            "transmission_coefficient": 0.01
        }}"#,
        name
    )
}

/// Build the complete RoomConfig JSON.
///
/// Note: `serde` serialises Rust `(f32, f32, f32)` tuples as JSON arrays,
/// e.g. `[5.0, 3.0, 4.0]`.
fn room_config_json() -> String {
    let mat_floor = material_json("concrete");
    let mat_ceiling = material_json("plaster");
    let mat_left = material_json("brick");
    let mat_right = material_json("brick");
    let mat_front = material_json("glass");
    let mat_back = material_json("wood");

    format!(
        r#"{{
            "dimensions": [5.0, 3.0, 4.0],
            "wall_materials": {{
                "floor": {floor},
                "ceiling": {ceiling},
                "left_wall": {left},
                "right_wall": {right},
                "front_wall": {front},
                "back_wall": {back}
            }},
            "reverb_time": 0.5,
            "volume": 60.0,
            "surface_area": 94.0,
            "temperature": 20.0,
            "humidity": 50.0,
            "enable_air_absorption": false
        }}"#,
        floor = mat_floor,
        ceiling = mat_ceiling,
        left = mat_left,
        right = mat_right,
        front = mat_front,
        back = mat_back,
    )
}

#[cfg(feature = "spatial")]
#[test]
fn test_spatial_room_writes_stereo_wav() {
    let dir = tempfile::tempdir().expect("create tempdir");

    let input = dir.path().join("input.wav");
    write_mono_sine_wav(&input);

    let room_file = dir.path().join("room.json");
    std::fs::write(&room_file, room_config_json()).expect("write room config JSON");

    let output = dir.path().join("output.wav");

    Command::cargo_bin("voirs")
        .expect("binary should exist")
        .args([
            "spatial",
            "room",
            input.to_str().expect("input path is valid UTF-8"),
            output.to_str().expect("output path is valid UTF-8"),
            "--room-config",
            room_file.to_str().expect("room config path is valid UTF-8"),
            "--source-position",
            "1.0,0.0,-1.0",
            "--listener-position",
            "0.0,0.0,0.0",
        ])
        .assert()
        .success();

    assert!(output.exists(), "output WAV should exist");

    let reader = hound::WavReader::open(&output).expect("open output WAV");
    assert_eq!(reader.spec().channels, 2, "output WAV should be stereo");
    assert!(reader.len() > 0, "output WAV should contain samples");
}
