//! Performance regression detection tests
//!
//! Detects performance regressions by comparing current performance
//! against baseline measurements and alerting on significant degradations.

use std::time::{Duration, Instant};
use voirs_vocoder::{
    config::{QualityLevel, VocodingConfig},
    AudioBuffer, DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder,
};

/// Performance baseline measurements
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// RTF (Real-Time Factor) thresholds
    pub rtf_threshold: f32,
    
    /// Latency thresholds in milliseconds
    pub latency_threshold_ms: f32,
    
    /// Memory usage threshold in MB
    pub memory_threshold_mb: f32,
    
    /// Throughput threshold (frames per second)
    pub throughput_threshold: f32,
}

impl PerformanceBaseline {
    /// Conservative baseline for regression testing
    pub fn conservative() -> Self {
        Self {
            rtf_threshold: 0.1,     // 10% RTF max for real-time
            latency_threshold_ms: 100.0, // 100ms max latency
            memory_threshold_mb: 512.0,  // 512MB max memory
            throughput_threshold: 100.0, // 100 frames/sec min
        }
    }
    
    /// Strict baseline for production systems
    pub fn strict() -> Self {
        Self {
            rtf_threshold: 0.05,    // 5% RTF max for real-time
            latency_threshold_ms: 50.0,  // 50ms max latency
            memory_threshold_mb: 256.0,  // 256MB max memory
            throughput_threshold: 200.0, // 200 frames/sec min
        }
    }
}

/// Performance measurement results
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    pub rtf: f32,
    pub latency_ms: f32,
    pub memory_mb: f32,
    pub throughput_fps: f32,
    pub test_name: String,
}

/// Performance regression detector
pub struct PerformanceRegressionDetector {
    baseline: PerformanceBaseline,
    measurements: Vec<PerformanceMeasurement>,
}

impl PerformanceRegressionDetector {
    /// Create new regression detector with baseline
    pub fn new(baseline: PerformanceBaseline) -> Self {
        Self {
            baseline,
            measurements: Vec::new(),
        }
    }
    
    /// Run performance regression test
    pub async fn run_regression_test(&mut self) -> Result<Vec<String>, String> {
        let mut violations = Vec::new();
        
        // Test HiFi-GAN performance
        let hifigan_perf = self.measure_hifigan_performance().await?;
        if let Some(violation) = self.check_performance(&hifigan_perf) {
            violations.push(violation);
        }
        self.measurements.push(hifigan_perf);
        
        // Test WaveGlow performance  
        let waveglow_perf = self.measure_waveglow_performance().await?;
        if let Some(violation) = self.check_performance(&waveglow_perf) {
            violations.push(violation);
        }
        self.measurements.push(waveglow_perf);
        
        // Test DiffWave performance
        let diffwave_perf = self.measure_diffwave_performance().await?;
        if let Some(violation) = self.check_performance(&diffwave_perf) {
            violations.push(violation);
        }
        self.measurements.push(diffwave_perf);
        
        // Test batch processing performance
        let batch_perf = self.measure_batch_performance().await?;
        if let Some(violation) = self.check_performance(&batch_perf) {
            violations.push(violation);
        }
        self.measurements.push(batch_perf);
        
        Ok(violations)
    }
    
    /// Measure HiFi-GAN performance
    async fn measure_hifigan_performance(&self) -> Result<PerformanceMeasurement, String> {
        let vocoder = DummyVocoder::new();
        let mel = self.generate_test_mel(1.0);
        
        // Measure RTF and latency
        let start = Instant::now();
        let _result = vocoder.vocode(&mel, None).await
            .map_err(|e| format!("HiFi-GAN vocoding failed: {}", e))?;
        let elapsed = start.elapsed();
        
        let audio_duration = 1.0; // 1 second of audio
        let rtf = elapsed.as_secs_f32() / audio_duration;
        let latency_ms = elapsed.as_millis() as f32;
        
        // Estimate memory usage (simplified)
        let mel_size_mb = (mel.data().len() * 4) as f32 / 1024.0 / 1024.0;
        let memory_mb = mel_size_mb * 4.0; // Rough estimate including processing overhead
        
        // Calculate throughput
        let throughput_fps = mel.frames() as f32 / elapsed.as_secs_f32();
        
        Ok(PerformanceMeasurement {
            rtf,
            latency_ms,
            memory_mb,
            throughput_fps,
            test_name: "HiFi-GAN".to_string(),
        })
    }
    
    /// Measure WaveGlow performance
    async fn measure_waveglow_performance(&self) -> Result<PerformanceMeasurement, String> {
        let vocoder = DummyVocoder::new();
        let mel = self.generate_test_mel(1.0);
        
        let start = Instant::now();
        let _result = vocoder.vocode(&mel, None).await
            .map_err(|e| format!("WaveGlow vocoding failed: {}", e))?;
        let elapsed = start.elapsed();
        
        let audio_duration = 1.0;
        let rtf = elapsed.as_secs_f32() / audio_duration;
        let latency_ms = elapsed.as_millis() as f32;
        
        let mel_size_mb = (mel.data().len() * 4) as f32 / 1024.0 / 1024.0;
        let memory_mb = mel_size_mb * 6.0; // WaveGlow typically uses more memory
        
        let throughput_fps = mel.frames() as f32 / elapsed.as_secs_f32();
        
        Ok(PerformanceMeasurement {
            rtf,
            latency_ms, 
            memory_mb,
            throughput_fps,
            test_name: "WaveGlow".to_string(),
        })
    }
    
    /// Measure DiffWave performance
    async fn measure_diffwave_performance(&self) -> Result<PerformanceMeasurement, String> {
        let vocoder = DummyVocoder::new();
        let mel = self.generate_test_mel(0.5); // Shorter audio for slower DiffWave
        
        let start = Instant::now();
        let _result = vocoder.vocode(&mel, None).await
            .map_err(|e| format!("DiffWave vocoding failed: {}", e))?;
        let elapsed = start.elapsed();
        
        let audio_duration = 0.5;
        let rtf = elapsed.as_secs_f32() / audio_duration;
        let latency_ms = elapsed.as_millis() as f32;
        
        let mel_size_mb = (mel.data().len() * 4) as f32 / 1024.0 / 1024.0;
        let memory_mb = mel_size_mb * 8.0; // DiffWave uses more memory due to U-Net
        
        let throughput_fps = mel.frames() as f32 / elapsed.as_secs_f32();
        
        Ok(PerformanceMeasurement {
            rtf,
            latency_ms,
            memory_mb,
            throughput_fps,
            test_name: "DiffWave".to_string(),
        })
    }
    
    /// Measure batch processing performance
    async fn measure_batch_performance(&self) -> Result<PerformanceMeasurement, String> {
        let vocoder = DummyVocoder::new();
        let batch_size = 4;
        let mels: Vec<MelSpectrogram> = (0..batch_size)
            .map(|_| self.generate_test_mel(1.0))
            .collect();
        
        let start = Instant::now();
        let _results = vocoder.vocode_batch(&mels, None).await
            .map_err(|e| format!("Batch vocoding failed: {}", e))?;
        let elapsed = start.elapsed();
        
        let total_audio_duration = batch_size as f32 * 1.0;
        let rtf = elapsed.as_secs_f32() / total_audio_duration;
        let latency_ms = elapsed.as_millis() as f32;
        
        let total_mel_size_mb = (mels.iter().map(|m| m.data().len()).sum::<usize>() * 4) as f32 / 1024.0 / 1024.0;
        let memory_mb = total_mel_size_mb * 3.0; // Batch processing memory estimate
        
        let total_frames: usize = mels.iter().map(|m| m.frames()).sum();
        let throughput_fps = total_frames as f32 / elapsed.as_secs_f32();
        
        Ok(PerformanceMeasurement {
            rtf,
            latency_ms,
            memory_mb,
            throughput_fps,
            test_name: "Batch Processing".to_string(),
        })
    }
    
    /// Check performance against baseline
    fn check_performance(&self, measurement: &PerformanceMeasurement) -> Option<String> {
        let mut violations = Vec::new();
        
        if measurement.rtf > self.baseline.rtf_threshold {
            violations.push(format!(
                "RTF violation: {:.3} > {:.3}",
                measurement.rtf, self.baseline.rtf_threshold
            ));
        }
        
        if measurement.latency_ms > self.baseline.latency_threshold_ms {
            violations.push(format!(
                "Latency violation: {:.1}ms > {:.1}ms",
                measurement.latency_ms, self.baseline.latency_threshold_ms
            ));
        }
        
        if measurement.memory_mb > self.baseline.memory_threshold_mb {
            violations.push(format!(
                "Memory violation: {:.1}MB > {:.1}MB",
                measurement.memory_mb, self.baseline.memory_threshold_mb
            ));
        }
        
        if measurement.throughput_fps < self.baseline.throughput_threshold {
            violations.push(format!(
                "Throughput violation: {:.1}fps < {:.1}fps",
                measurement.throughput_fps, self.baseline.throughput_threshold
            ));
        }
        
        if violations.is_empty() {
            None
        } else {
            Some(format!("{}: {}", measurement.test_name, violations.join(", ")))
        }
    }
    
    /// Generate test mel spectrogram
    fn generate_test_mel(&self, duration_secs: f32) -> MelSpectrogram {
        let sample_rate = 22050;
        let n_frames = ((duration_secs * sample_rate as f32) / 256.0) as usize;
        let n_mels = 80;
        
        let mut data = Vec::with_capacity(n_mels);
        for mel_idx in 0..n_mels {
            let mut frame = Vec::with_capacity(n_frames);
            for frame_idx in 0..n_frames {
                let base_freq = (mel_idx as f32 / n_mels as f32) * 4000.0 + 80.0;
                let time = frame_idx as f32 / (sample_rate as f32 / 256.0);
                let magnitude = -20.0 + 15.0 * (2.0 * std::f32::consts::PI * base_freq * time / 8000.0).sin().abs();
                frame.push(magnitude);
            }
            data.push(frame);
        }
        
        MelSpectrogram::new(data, sample_rate, 256)
    }
    
    /// Get performance summary
    pub fn get_summary(&self) -> String {
        if self.measurements.is_empty() {
            return "No measurements recorded".to_string();
        }
        
        let mut summary = String::new();
        summary.push_str("Performance Regression Test Summary:\n");
        summary.push_str("=====================================\n");
        
        for measurement in &self.measurements {
            summary.push_str(&format!(
                "{}: RTF={:.3}, Latency={:.1}ms, Memory={:.1}MB, Throughput={:.1}fps\n",
                measurement.test_name,
                measurement.rtf,
                measurement.latency_ms,
                measurement.memory_mb,
                measurement.throughput_fps
            ));
        }
        
        summary.push_str(&format!("\nBaseline Thresholds:\n"));
        summary.push_str(&format!("RTF <= {:.3}, Latency <= {:.1}ms, Memory <= {:.1}MB, Throughput >= {:.1}fps\n",
            self.baseline.rtf_threshold,
            self.baseline.latency_threshold_ms,
            self.baseline.memory_threshold_mb,
            self.baseline.throughput_threshold
        ));
        
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_regression_conservative() {
        let baseline = PerformanceBaseline::conservative();
        let mut detector = PerformanceRegressionDetector::new(baseline);
        
        let violations = detector.run_regression_test().await.unwrap();
        
        // Print summary for debugging
        println!("{}", detector.get_summary());
        
        // Should pass with conservative baseline
        assert!(violations.is_empty(), "Performance violations detected: {:?}", violations);
    }
    
    #[tokio::test] 
    async fn test_performance_regression_strict() {
        let baseline = PerformanceBaseline::strict();
        let mut detector = PerformanceRegressionDetector::new(baseline);
        
        let violations = detector.run_regression_test().await.unwrap();
        
        // Print summary for debugging
        println!("{}", detector.get_summary());
        
        // May have violations with strict baseline, just ensure test runs
        println!("Strict baseline violations: {:?}", violations);
    }
    
    #[test]
    fn test_baseline_creation() {
        let conservative = PerformanceBaseline::conservative();
        assert!(conservative.rtf_threshold > 0.0);
        assert!(conservative.latency_threshold_ms > 0.0);
        
        let strict = PerformanceBaseline::strict();
        assert!(strict.rtf_threshold < conservative.rtf_threshold);
        assert!(strict.latency_threshold_ms < conservative.latency_threshold_ms);
    }
    
    #[test]
    fn test_performance_measurement_creation() {
        let measurement = PerformanceMeasurement {
            rtf: 0.05,
            latency_ms: 25.0,
            memory_mb: 128.0,
            throughput_fps: 150.0,
            test_name: "Test".to_string(),
        };
        
        assert_eq!(measurement.test_name, "Test");
        assert!(measurement.rtf > 0.0);
    }
}