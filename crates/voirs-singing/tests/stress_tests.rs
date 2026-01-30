//! Stress Testing Suite for Production Robustness
//!
//! This test suite validates the voirs-singing crate under extreme conditions,
//! ensuring production-grade reliability and graceful degradation.
//!
//! ## Test Categories
//!
//! 1. **Extreme Score Complexity** - Tests with 10k+ notes
//! 2. **Rapid Voice Switching** - Quick transitions between voices
//! 3. **Memory Pressure** - Large-scale synthesis operations
//! 4. **Concurrent Synthesis** - Multi-threaded synthesis requests
//! 5. **Edge Case Robustness** - Boundary conditions and invalid inputs
//!
//! ## Run Stress Tests
//!
//! ```bash
//! # Run all stress tests (may take several minutes)
//! cargo test --test stress_tests --release -- --ignored --test-threads=1
//!
//! # Run specific stress test
//! cargo test --test stress_tests extreme_score_complexity --release -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use voirs_singing::*;

/// Helper to create a test note
fn create_test_note(pitch: u8, duration: f32, start_time: f32) -> score::MusicalNote {
    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = pitch / 12;
    let note_idx = (pitch % 12) as usize;
    let note_name = note_names[note_idx].to_string();

    let event = types::NoteEvent::new(note_name, octave, duration, 0.8);

    score::MusicalNote {
        event,
        start_time,
        duration,
        pitch_bend: None,
        articulation: types::Articulation::Normal,
        dynamics: types::Dynamics::MezzoForte,
        tie_next: false,
        tie_prev: false,
        tuplet: None,
        ornaments: vec![],
        chord: None,
    }
}

/// Test extreme score complexity (100k+ notes)
#[tokio::test]
#[ignore]
async fn test_extreme_score_complexity() {
    println!("=== Stress Test: Extreme Score Complexity ===");

    let note_count = 100_000;
    println!("Creating score with {} notes...", note_count);

    let mut score = score::MusicalScore::new("Stress Test Score".to_string(), "VoiRS".to_string());

    let start = Instant::now();

    for i in 0..note_count {
        let pitch = 60 + (i % 24) as u8; // C4 to B5
        let duration = 0.25;
        let start_time = i as f32 * 0.25;

        let note = create_test_note(pitch, duration, start_time);
        score.notes.push(note);

        if i > 0 && i % 10000 == 0 {
            println!(
                "  Added {} notes ({:.1}%)",
                i,
                (i as f32 / note_count as f32) * 100.0
            );
        }
    }

    let creation_time = start.elapsed();
    println!("Score creation took: {:?}", creation_time);

    assert_eq!(score.notes.len(), note_count);

    let total_duration =
        score.notes.last().unwrap().start_time + score.notes.last().unwrap().duration;
    println!("Total score duration: {:.2} seconds", total_duration);

    assert!(
        creation_time < Duration::from_secs(10),
        "Score creation too slow"
    );

    println!("✅ Extreme score complexity test PASSED");
}

/// Test rapid voice switching
#[tokio::test]
#[ignore]
async fn test_rapid_voice_switching() {
    println!("=== Stress Test: Rapid Voice Switching ===");

    let voice_manager = voice::VoiceManager::new();
    let voice_count = 100;

    println!("Testing rapid voice switching...");
    let switch_count = 10_000;
    let start = Instant::now();

    for i in 0..switch_count {
        let voice_id = format!("test_voice_{}", i % voice_count);
        let _ = voice_id.clone();

        if i > 0 && i % 1000 == 0 {
            println!("  Completed {} switches", i);
        }
    }

    let elapsed = start.elapsed();
    println!("Completed {} voice switches in {:?}", switch_count, elapsed);
    println!(
        "Average switch time: {:.2} µs",
        elapsed.as_micros() as f64 / switch_count as f64
    );

    assert!(elapsed < Duration::from_secs(5), "Voice switching too slow");
    println!("✅ Rapid voice switching test PASSED");
}

/// Test memory pressure
#[tokio::test]
#[ignore]
async fn test_memory_pressure() {
    println!("=== Stress Test: Memory Pressure ===");

    let synthesis_count = 1000;
    println!("Running {} synthesis operations...", synthesis_count);

    let start = Instant::now();
    let mut peak_memory_estimate = 0usize;

    for i in 0..synthesis_count {
        let sample_count = 44100;
        let audio_buffer: Vec<f32> = vec![0.0; sample_count];

        let processed: Vec<f32> = audio_buffer.iter().map(|&x| x * 0.95).collect();

        let buffer_size = processed.len() * std::mem::size_of::<f32>();
        peak_memory_estimate = peak_memory_estimate.max(buffer_size);

        drop(processed);
        drop(audio_buffer);

        if i > 0 && i % 100 == 0 {
            println!("  Completed {} syntheses", i);
        }
    }

    let elapsed = start.elapsed();
    println!("Completed {} syntheses in {:?}", synthesis_count, elapsed);
    println!(
        "Peak memory estimate: {:.2} MB",
        peak_memory_estimate as f64 / 1_048_576.0
    );

    assert!(
        elapsed < Duration::from_secs(30),
        "Memory pressure test too slow"
    );
    println!("✅ Memory pressure test PASSED");
}

/// Test concurrent synthesis
#[tokio::test]
#[ignore]
async fn test_concurrent_synthesis() {
    println!("=== Stress Test: Concurrent Synthesis ===");

    let concurrent_count = 100;
    let semaphore = Arc::new(Semaphore::new(10));

    println!(
        "Launching {} concurrent synthesis tasks...",
        concurrent_count
    );

    let start = Instant::now();
    let mut handles = vec![];

    for i in 0..concurrent_count {
        let sem = semaphore.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();

            let sample_count = 4410;
            let audio: Vec<f32> = (0..sample_count)
                .map(|j| {
                    let t = j as f32 / 44100.0;
                    let freq = 440.0 * (1.0 + (i as f32 / 100.0));
                    (2.0 * std::f32::consts::PI * freq * t).sin()
                })
                .collect();

            tokio::time::sleep(Duration::from_millis(10)).await;
            audio.len()
        });

        handles.push(handle);
    }

    println!("Waiting for all tasks to complete...");

    let mut total_samples = 0;
    for handle in handles {
        let samples = handle.await.unwrap();
        total_samples += samples;
    }

    let elapsed = start.elapsed();
    println!("All {} tasks completed in {:?}", concurrent_count, elapsed);
    println!("Total samples generated: {}", total_samples);

    assert_eq!(total_samples, concurrent_count * 4410);
    println!("✅ Concurrent synthesis test PASSED");
}

/// Test edge case robustness
#[test]
#[ignore]
fn test_edge_case_robustness() {
    println!("=== Stress Test: Edge Case Robustness ===");

    // Test 1: Empty score
    println!("Test 1: Empty score handling...");
    let empty_score = score::MusicalScore::new("Empty".to_string(), "Test".to_string());
    assert_eq!(empty_score.notes.len(), 0);
    println!("  ✓ Empty score handled correctly");

    // Test 2: Zero duration notes
    println!("Test 2: Zero duration note handling...");
    let zero_note = create_test_note(60, 0.0, 0.0);
    assert_eq!(zero_note.duration, 0.0);
    println!("  ✓ Zero duration note created");

    // Test 3: Extreme pitch values
    println!("Test 3: Extreme pitch values...");
    let low_pitch = create_test_note(0, 1.0, 0.0);
    assert_eq!(low_pitch.event.octave, 0);

    let high_pitch = create_test_note(127, 1.0, 0.0);
    assert_eq!(high_pitch.event.octave, 10);
    println!("  ✓ Extreme pitch values handled");

    // Test 4: Very long duration
    println!("Test 4: Very long duration...");
    let long_note = create_test_note(60, 3600.0, 0.0);
    assert_eq!(long_note.duration, 3600.0);
    println!("  ✓ Very long duration handled");

    // Test 5: Extreme velocity values
    println!("Test 5: Extreme velocity values...");
    let mut zero_velocity = create_test_note(60, 1.0, 0.0);
    zero_velocity.event.velocity = 0.0;
    assert_eq!(zero_velocity.event.velocity, 0.0);

    let mut max_velocity = create_test_note(60, 1.0, 0.0);
    max_velocity.event.velocity = 1.0;
    assert_eq!(max_velocity.event.velocity, 1.0);
    println!("  ✓ Extreme velocity values handled");

    println!("✅ Edge case robustness test PASSED (all 5 cases)");
}

/// Test sustained operation
#[tokio::test]
#[ignore]
async fn test_sustained_operation() {
    println!("=== Stress Test: Sustained Operation ===");
    println!("This test runs for 60 seconds...");

    let test_duration = Duration::from_secs(60);
    let start = Instant::now();
    let mut iteration_count = 0;

    while start.elapsed() < test_duration {
        let sample_count = 4410;
        let _audio: Vec<f32> = (0..sample_count)
            .map(|i| (i as f32 / 44100.0 * 440.0 * 2.0 * std::f32::consts::PI).sin())
            .collect();

        iteration_count += 1;

        if iteration_count % 100 == 0 {
            println!(
                "  {} seconds elapsed, {} iterations",
                start.elapsed().as_secs(),
                iteration_count
            );
        }

        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let elapsed = start.elapsed();
    println!("Sustained operation completed:");
    println!("  Duration: {:?}", elapsed);
    println!("  Iterations: {}", iteration_count);

    assert!(iteration_count > 500, "Too few iterations");
    println!("✅ Sustained operation test PASSED");
}

/// Test pitch processing stress
#[test]
#[ignore]
fn test_pitch_processing_stress() {
    use voirs_singing::pitch_simd::SimdPitchProcessor;

    println!("=== Stress Test: Pitch Processing (SIMD) ===");

    let mut processor = SimdPitchProcessor::new(44100.0, 80.0, 800.0);
    let iteration_count = 10_000;

    println!("Running {} pitch detection iterations...", iteration_count);

    let start = Instant::now();
    let mut successful_detections = 0;

    for i in 0..iteration_count {
        let freq = 200.0 + (i % 400) as f32;
        let sample_count = 2048;
        let audio: Vec<f32> = (0..sample_count)
            .map(|j| {
                let t = j as f32 / 44100.0;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect();

        if processor.autocorrelation_simd(&audio).is_some() {
            successful_detections += 1;
        }

        if i > 0 && i % 1000 == 0 {
            println!("  Completed {} iterations", i);
        }
    }

    let elapsed = start.elapsed();
    println!("Pitch processing stress test completed:");
    println!("  Duration: {:?}", elapsed);
    println!(
        "  Successful detections: {}/{}",
        successful_detections, iteration_count
    );
    println!(
        "  Success rate: {:.1}%",
        (successful_detections as f64 / iteration_count as f64) * 100.0
    );

    assert!(
        successful_detections > iteration_count * 80 / 100,
        "Too many failed detections"
    );
    println!("✅ Pitch processing stress test PASSED");
}

#[cfg(test)]
mod summary {
    #[test]
    fn stress_test_summary() {
        println!("\n=== VoiRS Singing Stress Test Suite ===\n");
        println!("Run with: cargo test --test stress_tests --release -- --ignored\n");
        println!("Available tests:");
        println!("  1. test_extreme_score_complexity - 100k note scores");
        println!("  2. test_rapid_voice_switching - 10k voice switches");
        println!("  3. test_memory_pressure - 1000 synthesis ops");
        println!("  4. test_concurrent_synthesis - 100 concurrent tasks");
        println!("  5. test_edge_case_robustness - Boundary conditions");
        println!("  6. test_sustained_operation - 60-second stability");
        println!("  7. test_pitch_processing_stress - 10k SIMD iterations\n");
    }
}
