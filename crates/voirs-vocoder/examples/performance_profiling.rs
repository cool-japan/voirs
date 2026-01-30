//! Performance Profiling Example
//!
//! Demonstrates how to use the advanced profiling system to monitor
//! vocoder performance in production environments.

use std::time::Duration;
use voirs_vocoder::{
    profiling::{AdvancedProfiler, ProcessingStage, ProfilerConfig},
    AudioBuffer, DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder,
};

/// Generate a realistic test mel spectrogram
fn generate_test_mel(n_mels: usize, n_frames: usize, sample_rate: u32) -> MelSpectrogram {
    let mut data = Vec::with_capacity(n_mels);
    for mel_idx in 0..n_mels {
        let mut frame = Vec::with_capacity(n_frames);
        for frame_idx in 0..n_frames {
            // Generate realistic mel values with harmonic content
            let base_freq = (mel_idx as f32 / n_mels as f32) * 4000.0 + 80.0;
            let time = frame_idx as f32 / (sample_rate as f32 / 256.0);

            // Add fundamental and harmonics
            let fundamental = (2.0 * std::f32::consts::PI * base_freq * time / 8000.0).sin();
            let harmonic2 =
                0.5 * (2.0 * std::f32::consts::PI * base_freq * 2.0 * time / 8000.0).sin();
            let harmonic3 =
                0.25 * (2.0 * std::f32::consts::PI * base_freq * 3.0 * time / 8000.0).sin();

            let magnitude = -20.0 + 15.0 * (fundamental + harmonic2 + harmonic3).abs();
            frame.push(magnitude);
        }
        data.push(frame);
    }

    MelSpectrogram::new(data, sample_rate, 256)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 VoiRS Vocoder - Performance Profiling Example");
    println!("==================================================\n");

    // Create a profiler with custom configuration
    let profiler_config = ProfilerConfig {
        detailed_latency: true,
        track_memory: true,
        track_throughput: true,
        detect_regressions: true,
        max_history_size: 100,
        sampling_rate: 1, // Profile every operation
    };

    let profiler = AdvancedProfiler::with_config(profiler_config);

    // Create vocoder
    let vocoder = DummyVocoder::new();
    let synthesis_config = SynthesisConfig::default();

    println!("1️⃣  Running baseline performance measurements...");
    println!("   Processing 10 samples to establish baseline\n");

    // Establish baseline performance
    for i in 0..10 {
        let mel = generate_test_mel(80, 86, 22050); // ~1 second of audio

        let mut session = profiler.start_operation();

        // Simulate preprocessing stage
        session.start_stage(ProcessingStage::Preprocessing);
        std::thread::sleep(Duration::from_micros(100));
        session.end_stage(ProcessingStage::Preprocessing);

        // Simulate inference stage
        session.start_stage(ProcessingStage::Inference);
        let audio = vocoder.vocode(&mel, Some(&synthesis_config)).await?;
        session.end_stage(ProcessingStage::Inference);

        // Simulate postprocessing stage
        session.start_stage(ProcessingStage::Postprocessing);
        std::thread::sleep(Duration::from_micros(50));
        session.end_stage(ProcessingStage::Postprocessing);

        session.complete(&mel, &audio);

        print!("   Sample {}/10 completed\r", i + 1);
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!("\n   ✅ Baseline established\n");

    // Set baseline for regression detection
    profiler.set_baseline();

    println!("2️⃣  Running normal operation with profiling...");
    println!("   Processing additional 20 samples\n");

    // Run normal operations
    for i in 0..20 {
        let mel = generate_test_mel(80, 86, 22050);

        let mut session = profiler.start_operation();

        session.start_stage(ProcessingStage::Preprocessing);
        std::thread::sleep(Duration::from_micros(100));
        session.end_stage(ProcessingStage::Preprocessing);

        session.start_stage(ProcessingStage::Inference);
        let audio = vocoder.vocode(&mel, Some(&synthesis_config)).await?;
        session.end_stage(ProcessingStage::Inference);

        session.start_stage(ProcessingStage::Postprocessing);
        std::thread::sleep(Duration::from_micros(50));
        session.end_stage(ProcessingStage::Postprocessing);

        session.complete(&mel, &audio);

        print!("   Sample {}/20 completed\r", i + 1);
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!("\n   ✅ Normal operation completed\n");

    println!("3️⃣  Checking for performance regressions...");

    // Check for regressions (10% threshold)
    match profiler.detect_regression(10.0) {
        Some(regression_report) => {
            println!("   ⚠️  Performance regression detected!");
            println!("   Threshold: {:.1}%", regression_report.threshold_percent);
            for regression in &regression_report.regressions {
                match regression {
                    voirs_vocoder::profiling::RegressionType::Latency {
                        baseline,
                        current,
                        increase_percent,
                    } => {
                        println!("   - Latency increased by {:.1}%", increase_percent);
                        println!("     Baseline: {:.2} ms", baseline.as_secs_f32() * 1000.0);
                        println!("     Current:  {:.2} ms", current.as_secs_f32() * 1000.0);
                    }
                    voirs_vocoder::profiling::RegressionType::Rtf {
                        baseline,
                        current,
                        increase_percent,
                    } => {
                        println!("   - RTF increased by {:.1}%", increase_percent);
                        println!("     Baseline: {:.3}x", baseline);
                        println!("     Current:  {:.3}x", current);
                    }
                    voirs_vocoder::profiling::RegressionType::Throughput {
                        baseline,
                        current,
                        decrease_percent,
                    } => {
                        println!("   - Throughput decreased by {:.1}%", decrease_percent);
                        println!("     Baseline: {:.1} frames/s", baseline);
                        println!("     Current:  {:.1} frames/s", current);
                    }
                }
            }
        }
        None => {
            println!("   ✅ No performance regressions detected");
        }
    }
    println!();

    println!("4️⃣  Generating comprehensive performance report...\n");

    let report = profiler.generate_report();
    println!("{}", report);

    println!("\n5️⃣  Performance Analysis:");

    let metrics = profiler.get_metrics();

    // Analyze latency
    let latency_variability =
        if let (Some(min), Some(max)) = (metrics.min_latency, metrics.max_latency) {
            (max.as_secs_f32() - min.as_secs_f32()) / min.as_secs_f32() * 100.0
        } else {
            0.0
        };

    println!("   Latency Variability: {:.1}%", latency_variability);

    if latency_variability < 20.0 {
        println!("   ✅ Excellent latency consistency");
    } else if latency_variability < 50.0 {
        println!("   ⚡ Good latency consistency");
    } else {
        println!("   ⚠️  High latency variability - consider investigation");
    }

    // Analyze RTF
    println!("\n   Real-Time Factor Analysis:");
    println!("   - Average RTF: {:.3}x", metrics.rtf_stats.avg_rtf);
    println!("   - Best RTF:    {:.3}x", metrics.rtf_stats.min_rtf);
    println!("   - Worst RTF:   {:.3}x", metrics.rtf_stats.max_rtf);
    println!(
        "   - Real-time %: {:.1}%",
        metrics.rtf_stats.realtime_percentage
    );

    if metrics.rtf_stats.avg_rtf < 0.5 {
        println!("   ✅ Excellent - Much faster than real-time");
    } else if metrics.rtf_stats.avg_rtf < 1.0 {
        println!("   ✅ Good - Faster than real-time");
    } else {
        println!("   ⚠️  Warning - Slower than real-time");
    }

    // Analyze stage breakdown
    println!("\n   Processing Stage Analysis:");
    let total_stage_time = metrics.latency_breakdown.preprocessing.as_secs_f32()
        + metrics.latency_breakdown.inference.as_secs_f32()
        + metrics.latency_breakdown.postprocessing.as_secs_f32();

    if total_stage_time > 0.0 {
        let prep_pct =
            metrics.latency_breakdown.preprocessing.as_secs_f32() / total_stage_time * 100.0;
        let infer_pct =
            metrics.latency_breakdown.inference.as_secs_f32() / total_stage_time * 100.0;
        let post_pct =
            metrics.latency_breakdown.postprocessing.as_secs_f32() / total_stage_time * 100.0;

        println!("   - Preprocessing:  {:.1}%", prep_pct);
        println!("   - Inference:      {:.1}%", infer_pct);
        println!("   - Postprocessing: {:.1}%", post_pct);

        if infer_pct > 80.0 {
            println!("   💡 Most time in inference - consider model optimization");
        } else if prep_pct > 30.0 {
            println!("   💡 High preprocessing time - consider input optimization");
        }
    }

    println!("\n6️⃣  Recommendations:");

    if metrics.rtf_stats.avg_rtf < 0.3 {
        println!("   ✅ Performance is excellent - no action needed");
    } else if metrics.rtf_stats.avg_rtf < 0.7 {
        println!("   ⚡ Performance is good - monitor for regressions");
    } else if metrics.rtf_stats.avg_rtf < 1.0 {
        println!("   ⚠️  Performance is acceptable but close to real-time limit");
        println!("      Consider optimization for safety margin");
    } else {
        println!("   🔴 Performance is below real-time - optimization required!");
    }

    if latency_variability > 50.0 {
        println!("   💡 High latency variability detected");
        println!("      - Check for resource contention");
        println!("      - Consider CPU pinning for critical threads");
        println!("      - Review memory allocation patterns");
    }

    println!("\n✅ Performance profiling example completed!");
    println!("\n📊 Summary:");
    println!("   Total operations: {}", metrics.total_operations);
    println!(
        "   Average latency:  {:.2} ms",
        metrics.avg_latency.as_secs_f32() * 1000.0
    );
    println!("   Average RTF:      {:.3}x", metrics.rtf_stats.avg_rtf);
    println!(
        "   Throughput:       {:.1} frames/s",
        metrics.throughput_stats.avg_frames_per_second
    );

    Ok(())
}
