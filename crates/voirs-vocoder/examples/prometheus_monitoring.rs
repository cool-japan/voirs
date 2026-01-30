//! Prometheus Monitoring Integration Example
//!
//! Demonstrates how to integrate vocoder performance profiling with Prometheus
//! for production monitoring and observability.

use std::time::Duration;
use voirs_vocoder::{
    profiling::{
        prometheus_metrics_text, AdvancedProfiler, ProcessingStage, ProfilerConfig,
        PrometheusExporter,
    },
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

/// Simulate vocoder service with Prometheus metrics
async fn run_vocoder_service() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 VoiRS Vocoder - Prometheus Monitoring Example");
    println!("===================================================\n");

    // Configure profiler for production monitoring
    let profiler_config = ProfilerConfig {
        detailed_latency: true,
        track_memory: true,
        track_throughput: true,
        detect_regressions: true,
        max_history_size: 1000,
        sampling_rate: 1, // Profile all operations in this demo
    };

    let profiler = AdvancedProfiler::with_config(profiler_config);
    let vocoder = DummyVocoder::new();
    let synthesis_config = SynthesisConfig::default();

    println!("1️⃣  Starting vocoder service simulation...");
    println!("   Processing requests and collecting metrics\n");

    // Simulate processing several requests
    for i in 1..=20 {
        let mel = generate_test_mel(80, 86, 22050);

        let mut session = profiler.start_operation();

        // Preprocessing stage
        session.start_stage(ProcessingStage::Preprocessing);
        std::thread::sleep(Duration::from_micros(100));
        session.end_stage(ProcessingStage::Preprocessing);

        // Inference stage
        session.start_stage(ProcessingStage::Inference);
        let audio = vocoder.vocode(&mel, Some(&synthesis_config)).await?;
        session.end_stage(ProcessingStage::Inference);

        // Postprocessing stage
        session.start_stage(ProcessingStage::Postprocessing);
        std::thread::sleep(Duration::from_micros(50));
        session.end_stage(ProcessingStage::Postprocessing);

        session.complete(&mel, &audio);

        if i % 5 == 0 {
            println!("   ✓ Processed {} requests", i);
        }
    }

    println!("\n2️⃣  Exporting metrics in Prometheus format...\n");

    // Create Prometheus exporter
    let exporter = PrometheusExporter::new("voirs_vocoder", &profiler);

    // Export basic metrics
    let metrics = exporter.export_metrics();
    println!("📊 Prometheus Metrics (Basic Export):");
    println!("{}", "=".repeat(60));
    println!("{}", metrics);

    println!("\n3️⃣  Exporting metrics with custom labels...\n");

    // Export metrics with labels for multi-model deployment
    let labels = vec![
        ("model", "dummy_vocoder"),
        ("backend", "native"),
        ("version", "0.1.0"),
        ("environment", "demo"),
    ];

    let labeled_metrics = exporter.export_metrics_with_labels(&labels);
    println!("📊 Prometheus Metrics (With Labels):");
    println!("{}", "=".repeat(60));
    println!("{}", labeled_metrics);

    println!("\n4️⃣  Using helper function for quick export...\n");

    // Helper function for quick export
    let quick_metrics = prometheus_metrics_text(&profiler);
    println!("📊 Quick Export (First 10 lines):");
    println!("{}", "=".repeat(60));
    for (i, line) in quick_metrics.lines().take(10).enumerate() {
        println!("{:2}. {}", i + 1, line);
    }
    println!("   ... (truncated for brevity)");

    println!("\n5️⃣  Multi-instance deployment example...\n");

    // Simulate multi-instance scenario with subsystems
    let exporter_hifigan = PrometheusExporter::with_subsystem("voirs", "hifigan", &profiler);
    let exporter_diffwave = PrometheusExporter::with_subsystem("voirs", "diffwave", &profiler);

    println!("📊 HiFi-GAN Instance Metrics (Sample):");
    println!("{}", "=".repeat(60));
    let hifigan_metrics = exporter_hifigan.export_metrics();
    for line in hifigan_metrics.lines().take(6) {
        println!("{}", line);
    }

    println!("\n📊 DiffWave Instance Metrics (Sample):");
    println!("{}", "=".repeat(60));
    let diffwave_metrics = exporter_diffwave.export_metrics();
    for line in diffwave_metrics.lines().take(6) {
        println!("{}", line);
    }

    println!("\n6️⃣  Production deployment guide:\n");
    println!("   To integrate with Prometheus in production:");
    println!();
    println!("   A. HTTP Server Setup:");
    println!("      ```rust");
    println!("      // Serve metrics via HTTP endpoint");
    println!("      async fn metrics_handler(profiler: Arc<AdvancedProfiler>) -> String {{");
    println!("          prometheus_metrics_text(&profiler)");
    println!("      }}");
    println!("      ```");
    println!();
    println!("   B. Prometheus Configuration:");
    println!("      ```yaml");
    println!("      scrape_configs:");
    println!("        - job_name: 'voirs_vocoder'");
    println!("          static_configs:");
    println!("            - targets: ['localhost:9090']");
    println!("          scrape_interval: 15s");
    println!("      ```");
    println!();
    println!("   C. Grafana Dashboard:");
    println!("      - Import metrics into Grafana");
    println!("      - Create dashboards for RTF, latency, throughput");
    println!("      - Set up alerts based on thresholds");
    println!();
    println!("   D. Alert Rules Example:");
    println!("      ```yaml");
    println!("      groups:");
    println!("        - name: vocoder_alerts");
    println!("          rules:");
    println!("            - alert: HighRTF");
    println!("              expr: voirs_vocoder_rtf_avg > 0.9");
    println!("              for: 5m");
    println!("              labels:");
    println!("                severity: warning");
    println!("              annotations:");
    println!("                summary: 'Vocoder RTF is high'");
    println!("            - alert: LowRealtimePercentage");
    println!("              expr: voirs_vocoder_realtime_percentage < 95");
    println!("              for: 10m");
    println!("              labels:");
    println!("                severity: critical");
    println!("      ```");

    println!("\n7️⃣  Key Metrics for Production Monitoring:\n");

    let metrics_info = profiler.get_metrics();
    println!(
        "   ✅ Operations: {} total requests processed",
        metrics_info.total_operations
    );
    println!(
        "   ✅ Average RTF: {:.3}x (target: <1.0x for real-time)",
        metrics_info.rtf_stats.avg_rtf
    );
    println!(
        "   ✅ P95 Latency: {:.2}ms (SLA compliance)",
        metrics_info.p95_latency.as_secs_f64() * 1000.0
    );
    println!(
        "   ✅ P99 Latency: {:.2}ms (tail latency)",
        metrics_info.p99_latency.as_secs_f64() * 1000.0
    );
    println!(
        "   ✅ Real-time %: {:.1}% (target: >95%)",
        metrics_info.rtf_stats.realtime_percentage
    );
    println!(
        "   ✅ Throughput: {:.1} frames/s",
        metrics_info.throughput_stats.avg_frames_per_second
    );

    println!("\n8️⃣  Recommended Prometheus Queries:\n");
    println!("   Rate of operations:");
    println!("   └─ rate(voirs_vocoder_operations_total[5m])");
    println!();
    println!("   Real-time factor trend:");
    println!("   └─ voirs_vocoder_rtf_avg");
    println!();
    println!("   Latency by quantile:");
    println!("   └─ voirs_vocoder_latency_ms{{quantile=\"0.95\"}}");
    println!();
    println!("   Processing stage breakdown:");
    println!("   └─ voirs_vocoder_stage_inference_ms / (");
    println!("      voirs_vocoder_stage_preprocessing_ms +");
    println!("      voirs_vocoder_stage_inference_ms +");
    println!("      voirs_vocoder_stage_postprocessing_ms) * 100");
    println!();
    println!("   Throughput (audio seconds per wall-clock second):");
    println!("   └─ rate(voirs_vocoder_audio_seconds_total[5m])");

    println!("\n9️⃣  Example PromQL Alert Expressions:\n");
    println!("   High latency (P95 > 100ms):");
    println!("   └─ voirs_vocoder_latency_ms{{quantile=\"0.95\"}} > 100");
    println!();
    println!("   RTF approaching real-time limit:");
    println!("   └─ voirs_vocoder_rtf_avg > 0.9");
    println!();
    println!("   Throughput degradation (20% drop over 15 minutes):");
    println!("   └─ (rate(voirs_vocoder_operations_total[15m])");
    println!("      / rate(voirs_vocoder_operations_total[15m] offset 15m) - 1) < -0.2");
    println!();
    println!("   Real-time percentage below SLA:");
    println!("   └─ voirs_vocoder_realtime_percentage < 95.0");

    println!("\n🔟 Integration with existing monitoring stacks:\n");
    println!("   • Kubernetes: Deploy as sidecar or use ServiceMonitor");
    println!("   • AWS: Export to CloudWatch via CloudWatch Agent");
    println!("   • GCP: Use GCP Monitoring with Prometheus integration");
    println!("   • Azure: Azure Monitor Prometheus integration");
    println!("   • Datadog: Prometheus metrics collection");
    println!("   • New Relic: Prometheus remote write");

    println!("\n✅ Prometheus monitoring example completed!");
    println!("\n📚 Next Steps:");
    println!("   1. Set up an HTTP server to expose /metrics endpoint");
    println!("   2. Configure Prometheus to scrape your service");
    println!("   3. Create Grafana dashboards for visualization");
    println!("   4. Configure alerting rules based on your SLAs");
    println!("   5. Integrate with your incident management system");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_vocoder_service().await
}
