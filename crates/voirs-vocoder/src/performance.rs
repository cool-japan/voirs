//! Real-time performance monitoring and quality metrics tracking
//!
//! Provides comprehensive monitoring of vocoder performance including:
//! - Quality metrics over time
//! - Processing latency tracking
//! - Memory usage monitoring
//! - Adaptive quality control
//! - Performance alerting

use crate::metrics::{QualityCalculator, QualityConfig, QualityMetrics};
use crate::{AudioBuffer, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Performance metrics for vocoder operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Processing latency in milliseconds
    pub latency_ms: f32,

    /// CPU usage percentage (0.0-100.0)
    pub cpu_usage: f32,

    /// Memory usage in MB
    pub memory_usage_mb: f32,

    /// Real-time factor (processing_time / audio_duration)
    pub real_time_factor: f32,

    /// Quality metrics
    pub quality: QualityMetrics,

    /// Timestamp of measurement
    pub timestamp: Instant,

    /// Buffer underruns count
    pub buffer_underruns: u32,

    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f32,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new(
        latency_ms: f32,
        cpu_usage: f32,
        memory_usage_mb: f32,
        real_time_factor: f32,
        quality: QualityMetrics,
        buffer_underruns: u32,
        cache_hit_rate: f32,
    ) -> Self {
        Self {
            latency_ms,
            cpu_usage,
            memory_usage_mb,
            real_time_factor,
            quality,
            timestamp: Instant::now(),
            buffer_underruns,
            cache_hit_rate,
        }
    }

    /// Check if performance is within acceptable bounds
    pub fn is_acceptable(&self, thresholds: &PerformanceThresholds) -> bool {
        self.latency_ms <= thresholds.max_latency_ms
            && self.cpu_usage <= thresholds.max_cpu_usage
            && self.memory_usage_mb <= thresholds.max_memory_mb
            && self.real_time_factor <= thresholds.max_rtf
            && self.quality.mos_estimate >= thresholds.min_quality_score
    }

    /// Calculate performance score (0.0-1.0, higher is better)
    pub fn performance_score(&self, thresholds: &PerformanceThresholds) -> f32 {
        let latency_score = (thresholds.max_latency_ms
            - self.latency_ms.min(thresholds.max_latency_ms))
            / thresholds.max_latency_ms;
        let cpu_score = (thresholds.max_cpu_usage - self.cpu_usage.min(thresholds.max_cpu_usage))
            / thresholds.max_cpu_usage;
        let memory_score = (thresholds.max_memory_mb
            - self.memory_usage_mb.min(thresholds.max_memory_mb))
            / thresholds.max_memory_mb;
        let rtf_score = (thresholds.max_rtf - self.real_time_factor.min(thresholds.max_rtf))
            / thresholds.max_rtf;
        let quality_score = (self.quality.mos_estimate - 1.0) / 4.0; // MOS 1-5 to 0-1

        (latency_score * 0.25
            + cpu_score * 0.2
            + memory_score * 0.15
            + rtf_score * 0.2
            + quality_score * 0.2)
            .clamp(0.0, 1.0)
    }
}

/// Performance thresholds for quality control
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f32,

    /// Maximum CPU usage percentage
    pub max_cpu_usage: f32,

    /// Maximum memory usage in MB
    pub max_memory_mb: f32,

    /// Maximum real-time factor
    pub max_rtf: f32,

    /// Minimum quality score (MOS)
    pub min_quality_score: f32,

    /// Maximum buffer underruns per second
    pub max_underruns_per_sec: u32,

    /// Minimum cache hit rate
    pub min_cache_hit_rate: f32,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 100.0,    // 100ms max latency
            max_cpu_usage: 80.0,      // 80% max CPU
            max_memory_mb: 1024.0,    // 1GB max memory
            max_rtf: 0.5,             // 0.5x real-time factor
            min_quality_score: 3.5,   // Minimum MOS 3.5
            max_underruns_per_sec: 5, // Max 5 underruns/sec
            min_cache_hit_rate: 0.8,  // 80% minimum cache hit rate
        }
    }
}

/// Performance alert types
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceAlert {
    /// High latency detected
    HighLatency { current: u32, threshold: u32 },

    /// High CPU usage
    HighCpuUsage { current: u32, threshold: u32 },

    /// High memory usage
    HighMemoryUsage { current: u32, threshold: u32 },

    /// Real-time factor too high
    HighRealTimeFactor { current_rtf: f32, threshold: f32 },

    /// Quality degradation
    QualityDegradation { current_mos: f32, threshold: f32 },

    /// Buffer underruns
    BufferUnderruns { count: u32, threshold: u32 },

    /// Low cache hit rate
    LowCacheHitRate { current: f32, threshold: f32 },
}

/// Real-time performance monitor
pub struct PerformanceMonitor {
    /// Quality calculator for metrics
    quality_calculator: QualityCalculator,

    /// Recent performance history
    history: Arc<Mutex<VecDeque<PerformanceMetrics>>>,

    /// Performance thresholds
    thresholds: PerformanceThresholds,

    /// Maximum history size
    max_history_size: usize,

    /// Alert callback function
    alert_callback: Option<Arc<dyn Fn(PerformanceAlert) + Send + Sync>>,

    /// Last reference audio for quality comparison
    last_reference: Option<AudioBuffer>,

    /// Performance statistics
    total_samples_processed: u64,
    total_processing_time: Duration,
    buffer_underrun_count: u32,
    cache_hits: u64,
    cache_accesses: u64,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(quality_config: QualityConfig, thresholds: PerformanceThresholds) -> Self {
        Self {
            quality_calculator: QualityCalculator::new(quality_config),
            history: Arc::new(Mutex::new(VecDeque::new())),
            thresholds,
            max_history_size: 1000, // Keep last 1000 measurements
            alert_callback: None,
            last_reference: None,
            total_samples_processed: 0,
            total_processing_time: Duration::new(0, 0),
            buffer_underrun_count: 0,
            cache_hits: 0,
            cache_accesses: 0,
        }
    }

    /// Set alert callback function
    pub fn set_alert_callback<F>(&mut self, callback: F)
    where
        F: Fn(PerformanceAlert) + Send + Sync + 'static,
    {
        self.alert_callback = Some(Arc::new(callback));
    }

    /// Record processing performance metrics
    pub fn record_processing(
        &mut self,
        reference: Option<&AudioBuffer>,
        output: &AudioBuffer,
        processing_time: Duration,
    ) -> Result<()> {
        let _start_time = Instant::now();

        // Calculate processing metrics
        let audio_duration = Duration::from_secs_f32(
            output.samples().len() as f32
                / (output.sample_rate() as f32 * output.channels() as f32),
        );
        let real_time_factor = processing_time.as_secs_f32() / audio_duration.as_secs_f32();
        let latency_ms = processing_time.as_secs_f32() * 1000.0;

        // Estimate system metrics (simplified)
        let cpu_usage = self.estimate_cpu_usage(real_time_factor);
        let memory_usage_mb = self.estimate_memory_usage();
        let cache_hit_rate = if self.cache_accesses > 0 {
            self.cache_hits as f32 / self.cache_accesses as f32
        } else {
            1.0
        };

        // Calculate quality metrics
        let quality = if let Some(ref_audio) = reference {
            self.quality_calculator
                .calculate_metrics(ref_audio, output)?
        } else if let Some(ref last_ref) = &self.last_reference {
            self.quality_calculator
                .calculate_metrics(last_ref, output)?
        } else {
            // Create basic quality estimate without reference
            self.estimate_quality_without_reference(output)
        };

        // Create performance metrics
        let metrics = PerformanceMetrics::new(
            latency_ms,
            cpu_usage,
            memory_usage_mb,
            real_time_factor,
            quality,
            self.buffer_underrun_count,
            cache_hit_rate,
        );

        // Check for alerts
        self.check_alerts(&metrics);

        // Add to history
        {
            let mut history = self.history.lock().unwrap();
            history.push_back(metrics.clone());
            if history.len() > self.max_history_size {
                history.pop_front();
            }
        }

        // Update statistics
        self.total_samples_processed += output.samples().len() as u64;
        self.total_processing_time += processing_time;

        // Store reference for next comparison
        if let Some(ref_audio) = reference {
            self.last_reference = Some(ref_audio.clone());
        }

        Ok(())
    }

    /// Record buffer underrun event
    pub fn record_buffer_underrun(&mut self) {
        self.buffer_underrun_count += 1;
    }

    /// Record cache access
    pub fn record_cache_access(&mut self, hit: bool) {
        self.cache_accesses += 1;
        if hit {
            self.cache_hits += 1;
        }
    }

    /// Get recent performance history
    pub fn get_history(&self, max_samples: Option<usize>) -> Vec<PerformanceMetrics> {
        let history = self.history.lock().unwrap();
        let samples = max_samples.unwrap_or(history.len());
        history.iter().rev().take(samples).cloned().collect()
    }

    /// Get current average performance
    pub fn get_average_performance(&self, duration: Duration) -> Option<PerformanceMetrics> {
        let history = self.history.lock().unwrap();
        let cutoff_time = Instant::now() - duration;

        let recent_metrics: Vec<_> = history
            .iter()
            .filter(|m| m.timestamp > cutoff_time)
            .collect();

        if recent_metrics.is_empty() {
            return None;
        }

        // Calculate averages
        let avg_latency =
            recent_metrics.iter().map(|m| m.latency_ms).sum::<f32>() / recent_metrics.len() as f32;
        let avg_cpu =
            recent_metrics.iter().map(|m| m.cpu_usage).sum::<f32>() / recent_metrics.len() as f32;
        let avg_memory = recent_metrics
            .iter()
            .map(|m| m.memory_usage_mb)
            .sum::<f32>()
            / recent_metrics.len() as f32;
        let avg_rtf = recent_metrics
            .iter()
            .map(|m| m.real_time_factor)
            .sum::<f32>()
            / recent_metrics.len() as f32;
        let avg_mos = recent_metrics
            .iter()
            .map(|m| m.quality.mos_estimate)
            .sum::<f32>()
            / recent_metrics.len() as f32;
        let avg_cache_hit = recent_metrics.iter().map(|m| m.cache_hit_rate).sum::<f32>()
            / recent_metrics.len() as f32;

        // Calculate averages for quality metrics from recent history
        let avg_snr =
            recent_metrics.iter().map(|m| m.quality.snr).sum::<f32>() / recent_metrics.len() as f32;
        let avg_thd_n = recent_metrics.iter().map(|m| m.quality.thd_n).sum::<f32>()
            / recent_metrics.len() as f32;
        let avg_lsd =
            recent_metrics.iter().map(|m| m.quality.lsd).sum::<f32>() / recent_metrics.len() as f32;
        let avg_psnr = recent_metrics.iter().map(|m| m.quality.psnr).sum::<f32>()
            / recent_metrics.len() as f32;
        let avg_spectral_convergence = recent_metrics
            .iter()
            .map(|m| m.quality.spectral_convergence)
            .sum::<f32>()
            / recent_metrics.len() as f32;

        // Create synthetic quality metrics with calculated averages
        let avg_quality = QualityMetrics {
            pesq: None,
            stoi: None,
            si_sdr: None,
            mos_prediction: None,
            snr: avg_snr,
            thd_n: avg_thd_n,
            lsd: avg_lsd,
            mcd: None,
            psnr: avg_psnr,
            spectral_convergence: avg_spectral_convergence,
            mos_estimate: avg_mos,
        };

        Some(PerformanceMetrics::new(
            avg_latency,
            avg_cpu,
            avg_memory,
            avg_rtf,
            avg_quality,
            self.buffer_underrun_count,
            avg_cache_hit,
        ))
    }

    /// Check for performance alerts
    fn check_alerts(&self, metrics: &PerformanceMetrics) {
        if let Some(ref callback) = self.alert_callback {
            // Check latency
            if metrics.latency_ms > self.thresholds.max_latency_ms {
                callback(PerformanceAlert::HighLatency {
                    current: metrics.latency_ms as u32,
                    threshold: self.thresholds.max_latency_ms as u32,
                });
            }

            // Check CPU usage
            if metrics.cpu_usage > self.thresholds.max_cpu_usage {
                callback(PerformanceAlert::HighCpuUsage {
                    current: metrics.cpu_usage as u32,
                    threshold: self.thresholds.max_cpu_usage as u32,
                });
            }

            // Check memory usage
            if metrics.memory_usage_mb > self.thresholds.max_memory_mb {
                callback(PerformanceAlert::HighMemoryUsage {
                    current: metrics.memory_usage_mb as u32,
                    threshold: self.thresholds.max_memory_mb as u32,
                });
            }

            // Check real-time factor
            if metrics.real_time_factor > self.thresholds.max_rtf {
                callback(PerformanceAlert::HighRealTimeFactor {
                    current_rtf: metrics.real_time_factor,
                    threshold: self.thresholds.max_rtf,
                });
            }

            // Check quality
            if metrics.quality.mos_estimate < self.thresholds.min_quality_score {
                callback(PerformanceAlert::QualityDegradation {
                    current_mos: metrics.quality.mos_estimate,
                    threshold: self.thresholds.min_quality_score,
                });
            }

            // Check cache hit rate
            if metrics.cache_hit_rate < self.thresholds.min_cache_hit_rate {
                callback(PerformanceAlert::LowCacheHitRate {
                    current: metrics.cache_hit_rate,
                    threshold: self.thresholds.min_cache_hit_rate,
                });
            }
        }
    }

    /// Estimate CPU usage based on real-time factor
    fn estimate_cpu_usage(&self, rtf: f32) -> f32 {
        // Simple heuristic: higher RTF generally means higher CPU usage
        (rtf * 100.0).clamp(0.0, 100.0)
    }

    /// Estimate current memory usage (simplified)
    fn estimate_memory_usage(&self) -> f32 {
        // In a real implementation, this would query actual memory usage
        // For now, provide a reasonable estimate based on processing activity
        let base_usage = 100.0; // Base 100MB
        let processing_overhead = (self.total_samples_processed as f32 / 1_000_000.0) * 10.0;
        (base_usage + processing_overhead).min(2048.0) // Cap at 2GB
    }

    /// Estimate quality without reference audio
    fn estimate_quality_without_reference(&self, output: &AudioBuffer) -> QualityMetrics {
        let samples = output.samples();

        // Calculate basic signal statistics
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let peak = samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let snr = if rms > 0.0 {
            20.0 * (peak / rms).log10()
        } else {
            0.0
        };

        // Estimate MOS based on signal characteristics
        let estimated_mos = if rms < 0.001 {
            1.0 // Very quiet signal
        } else if rms > 0.3 {
            2.0 // Likely clipped
        } else if snr < 10.0 {
            2.5 // Low SNR
        } else if snr > 40.0 {
            4.5 // Very high quality
        } else {
            2.0 + (snr - 10.0) / 30.0 * 2.5 // Linear interpolation
        };

        // Calculate THD+N estimate based on signal characteristics
        let thd_n = if snr > 50.0 {
            0.001 // Very low distortion for high SNR
        } else if snr > 30.0 {
            0.01 + (50.0 - snr) / 20.0 * 0.09 // 0.01% to 0.1%
        } else {
            0.1 + (30.0 - snr.max(0.0)) / 30.0 * 4.9 // 0.1% to 5.0%
        };

        // Estimate LSD based on signal smoothness
        let signal_variance = samples
            .iter()
            .zip(samples.iter().skip(1))
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / (samples.len() - 1) as f32;
        let lsd = (signal_variance * 100.0).clamp(0.1, 2.0);

        // Estimate spectral convergence based on signal stability
        let spectral_convergence = (if snr > 40.0 {
            0.01 + signal_variance * 10.0 // Very good convergence for high SNR
        } else {
            0.05 + (40.0 - snr.max(0.0)) / 40.0 * 0.45 // 0.05 to 0.5
        })
        .clamp(0.01, 0.5);

        QualityMetrics {
            pesq: None,
            stoi: None,
            si_sdr: None,
            mos_prediction: Some(estimated_mos),
            snr,
            thd_n,
            lsd,
            mcd: None,
            psnr: snr + 10.0, // Rough estimate
            spectral_convergence,
            mos_estimate: estimated_mos,
        }
    }

    /// Get overall performance statistics
    pub fn get_statistics(&self) -> PerformanceStatistics {
        let avg_processing_time = if self.total_samples_processed > 0 {
            self.total_processing_time.as_secs_f32() / self.total_samples_processed as f32
        } else {
            0.0
        };

        let cache_hit_rate = if self.cache_accesses > 0 {
            self.cache_hits as f32 / self.cache_accesses as f32
        } else {
            0.0
        };

        PerformanceStatistics {
            total_samples_processed: self.total_samples_processed,
            total_processing_time: self.total_processing_time,
            average_processing_time_per_sample: avg_processing_time,
            buffer_underrun_count: self.buffer_underrun_count,
            cache_hit_rate,
            history_size: {
                let history = self.history.lock().unwrap();
                history.len()
            },
        }
    }
}

/// Overall performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    /// Total audio samples processed
    pub total_samples_processed: u64,

    /// Total time spent processing
    pub total_processing_time: Duration,

    /// Average processing time per sample
    pub average_processing_time_per_sample: f32,

    /// Total buffer underruns
    pub buffer_underrun_count: u32,

    /// Overall cache hit rate
    pub cache_hit_rate: f32,

    /// Performance history size
    pub history_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioBuffer;
    use std::time::Duration;

    #[test]
    fn test_performance_monitor_creation() {
        let quality_config = QualityConfig::default();
        let thresholds = PerformanceThresholds::default();
        let _monitor = PerformanceMonitor::new(quality_config, thresholds);
    }

    #[test]
    fn test_performance_metrics_scoring() {
        let quality = QualityMetrics {
            pesq: None,
            stoi: None,
            si_sdr: None,
            mos_prediction: None,
            snr: 20.0,
            thd_n: 1.0,
            lsd: 0.5,
            mcd: None,
            psnr: 30.0,
            spectral_convergence: 0.1,
            mos_estimate: 4.0,
        };

        let metrics = PerformanceMetrics::new(
            50.0,  // 50ms latency
            40.0,  // 40% CPU
            512.0, // 512MB memory
            0.3,   // 0.3x RTF
            quality, 0,   // No underruns
            0.9, // 90% cache hit rate
        );

        let thresholds = PerformanceThresholds::default();
        assert!(metrics.is_acceptable(&thresholds));

        let score = metrics.performance_score(&thresholds);
        assert!(score > 0.5); // Should be good performance
    }

    #[test]
    fn test_buffer_underrun_recording() {
        let quality_config = QualityConfig::default();
        let thresholds = PerformanceThresholds::default();
        let mut monitor = PerformanceMonitor::new(quality_config, thresholds);

        assert_eq!(monitor.buffer_underrun_count, 0);
        monitor.record_buffer_underrun();
        assert_eq!(monitor.buffer_underrun_count, 1);
    }

    #[test]
    fn test_cache_access_recording() {
        let quality_config = QualityConfig::default();
        let thresholds = PerformanceThresholds::default();
        let mut monitor = PerformanceMonitor::new(quality_config, thresholds);

        monitor.record_cache_access(true);
        monitor.record_cache_access(false);
        monitor.record_cache_access(true);

        let stats = monitor.get_statistics();
        assert_eq!(stats.cache_hit_rate, 2.0 / 3.0); // 2 hits out of 3 accesses
    }

    #[test]
    fn test_performance_recording() {
        let quality_config = QualityConfig {
            include_expensive: false, // Disable expensive metrics for test
            ..QualityConfig::default()
        };
        let thresholds = PerformanceThresholds::default();
        let mut monitor = PerformanceMonitor::new(quality_config, thresholds);

        // Create test audio
        let samples = vec![0.1; 1024];
        let audio = AudioBuffer::new(samples, 22050, 1);

        // Record processing
        let processing_time = Duration::from_millis(10);
        let result = monitor.record_processing(None, &audio, processing_time);
        assert!(result.is_ok());

        // Check history
        let history = monitor.get_history(Some(1));
        assert_eq!(history.len(), 1);
        assert!(history[0].latency_ms > 0.0);
    }
}
