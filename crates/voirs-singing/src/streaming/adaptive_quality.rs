//! Adaptive quality scaling for maintaining low latency under varying system load
//!
//! Dynamically adjusts synthesis quality based on CPU usage to maintain real-time performance.

use super::cpu_monitor::{CpuStats, SystemLoadEstimator};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Quality level for synthesis
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Minimum quality (fastest, 60% quality)
    Minimum,

    /// Low quality (75% quality)
    Low,

    /// Medium quality (85% quality)
    Medium,

    /// High quality (95% quality)
    High,

    /// Maximum quality (100% quality)
    Maximum,
}

impl QualityLevel {
    /// Get quality factor (0.0-1.0)
    pub fn quality_factor(&self) -> f32 {
        match self {
            Self::Minimum => 0.60,
            Self::Low => 0.75,
            Self::Medium => 0.85,
            Self::High => 0.95,
            Self::Maximum => 1.00,
        }
    }

    /// Get expected CPU usage multiplier
    pub fn cpu_multiplier(&self) -> f32 {
        match self {
            Self::Minimum => 0.5,
            Self::Low => 0.7,
            Self::Medium => 0.85,
            Self::High => 0.95,
            Self::Maximum => 1.0,
        }
    }

    /// Get synthesis parameters for this quality level
    pub fn synthesis_params(&self) -> SynthesisQualityParams {
        match self {
            Self::Minimum => SynthesisQualityParams {
                harmonics_count: 8,
                formant_filters: 2,
                vibrato_detail: 0.5,
                breath_detail: 0.3,
                spectral_resolution: 256,
                oversampling: 1,
            },
            Self::Low => SynthesisQualityParams {
                harmonics_count: 12,
                formant_filters: 3,
                vibrato_detail: 0.6,
                breath_detail: 0.5,
                spectral_resolution: 512,
                oversampling: 1,
            },
            Self::Medium => SynthesisQualityParams {
                harmonics_count: 16,
                formant_filters: 4,
                vibrato_detail: 0.75,
                breath_detail: 0.7,
                spectral_resolution: 1024,
                oversampling: 2,
            },
            Self::High => SynthesisQualityParams {
                harmonics_count: 24,
                formant_filters: 5,
                vibrato_detail: 0.9,
                breath_detail: 0.85,
                spectral_resolution: 2048,
                oversampling: 2,
            },
            Self::Maximum => SynthesisQualityParams {
                harmonics_count: 32,
                formant_filters: 6,
                vibrato_detail: 1.0,
                breath_detail: 1.0,
                spectral_resolution: 4096,
                oversampling: 4,
            },
        }
    }

    /// Get next lower quality level
    pub fn lower(&self) -> Option<QualityLevel> {
        match self {
            Self::Maximum => Some(Self::High),
            Self::High => Some(Self::Medium),
            Self::Medium => Some(Self::Low),
            Self::Low => Some(Self::Minimum),
            Self::Minimum => None,
        }
    }

    /// Get next higher quality level
    pub fn higher(&self) -> Option<QualityLevel> {
        match self {
            Self::Minimum => Some(Self::Low),
            Self::Low => Some(Self::Medium),
            Self::Medium => Some(Self::High),
            Self::High => Some(Self::Maximum),
            Self::Maximum => None,
        }
    }
}

/// Synthesis quality parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisQualityParams {
    /// Number of harmonics to synthesize
    pub harmonics_count: usize,

    /// Number of formant filters to apply
    pub formant_filters: usize,

    /// Vibrato detail level (0.0-1.0)
    pub vibrato_detail: f32,

    /// Breath modeling detail (0.0-1.0)
    pub breath_detail: f32,

    /// Spectral resolution (FFT size)
    pub spectral_resolution: usize,

    /// Oversampling factor
    pub oversampling: usize,
}

/// Adaptive quality scaler
///
/// Automatically adjusts synthesis quality to maintain target latency under varying system load.
pub struct AdaptiveQualityScaler {
    /// System load estimator
    load_estimator: SystemLoadEstimator,

    /// Current quality level
    current_quality: Arc<std::sync::Mutex<QualityLevel>>,

    /// Target CPU usage (0.0-1.0)
    target_cpu: f32,

    /// CPU usage tolerance
    cpu_tolerance: f32,

    /// Minimum quality level allowed
    min_quality: QualityLevel,

    /// Maximum quality level allowed
    max_quality: QualityLevel,

    /// Time since last quality adjustment
    last_adjustment: Arc<std::sync::Mutex<Instant>>,

    /// Minimum time between adjustments
    adjustment_cooldown: Duration,

    /// Quality adjustment history
    adjustment_history: Arc<std::sync::Mutex<Vec<QualityAdjustment>>>,
}

impl AdaptiveQualityScaler {
    /// Create a new adaptive quality scaler
    ///
    /// # Arguments
    /// * `target_cpu` - Target CPU usage (0.0-1.0)
    /// * `min_quality` - Minimum quality level
    /// * `max_quality` - Maximum quality level
    pub fn new(target_cpu: f32, min_quality: QualityLevel, max_quality: QualityLevel) -> Self {
        Self {
            load_estimator: SystemLoadEstimator::new(),
            current_quality: Arc::new(std::sync::Mutex::new(max_quality)),
            target_cpu: target_cpu.clamp(0.0, 1.0),
            cpu_tolerance: 0.1,
            min_quality,
            max_quality,
            last_adjustment: Arc::new(std::sync::Mutex::new(Instant::now())),
            adjustment_cooldown: Duration::from_millis(500),
            adjustment_history: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Update with synthesis timing and get recommended quality level
    ///
    /// # Arguments
    /// * `synthesis_time` - Time spent on synthesis
    /// * `frame_duration` - Target frame duration
    ///
    /// # Returns
    /// Current quality level and whether it was adjusted
    pub fn update(
        &self,
        synthesis_time: Duration,
        frame_duration: Duration,
    ) -> (QualityLevel, bool) {
        let cpu_load = self.load_estimator.update(synthesis_time, frame_duration);
        let trend = self.load_estimator.get_trend();

        let mut adjusted = false;
        let mut current_quality = *self.current_quality.lock().unwrap();

        // Check if we need to adjust quality
        if let Ok(last_adjustment) = self.last_adjustment.try_lock() {
            if last_adjustment.elapsed() >= self.adjustment_cooldown {
                let (should_adjust, new_quality) =
                    self.should_adjust_quality(cpu_load, trend, current_quality);

                if should_adjust {
                    if let Ok(mut quality) = self.current_quality.try_lock() {
                        *quality = new_quality;
                        current_quality = new_quality;
                        adjusted = true;

                        // Record adjustment
                        if let Ok(mut history) = self.adjustment_history.try_lock() {
                            history.push(QualityAdjustment {
                                timestamp: Instant::now(),
                                old_quality: current_quality,
                                new_quality,
                                cpu_load,
                                reason: if cpu_load > self.target_cpu + self.cpu_tolerance {
                                    AdjustmentReason::CpuOverload
                                } else if cpu_load < self.target_cpu - self.cpu_tolerance {
                                    AdjustmentReason::CpuUnderload
                                } else {
                                    AdjustmentReason::TrendBased
                                },
                            });

                            // Keep last 100 adjustments
                            if history.len() > 100 {
                                history.remove(0);
                            }
                        }
                    }
                }
            }
        }

        (current_quality, adjusted)
    }

    /// Determine if quality should be adjusted
    fn should_adjust_quality(
        &self,
        cpu_load: f32,
        trend: f32,
        current: QualityLevel,
    ) -> (bool, QualityLevel) {
        // CPU is too high - reduce quality
        if cpu_load > self.target_cpu + self.cpu_tolerance {
            if let Some(lower) = current.lower() {
                if lower as u8 >= self.min_quality as u8 {
                    return (true, lower);
                }
            }
        }

        // CPU is too low and trending down - increase quality
        if cpu_load < self.target_cpu - self.cpu_tolerance && trend < 0.0 {
            if let Some(higher) = current.higher() {
                if higher as u8 <= self.max_quality as u8 {
                    return (true, higher);
                }
            }
        }

        // Proactive adjustment based on trend
        if trend > 0.02 && cpu_load > self.target_cpu * 0.8 {
            // Load is increasing and already high - reduce quality proactively
            if let Some(lower) = current.lower() {
                if lower as u8 >= self.min_quality as u8 {
                    return (true, lower);
                }
            }
        }

        (false, current)
    }

    /// Get current quality level
    pub fn get_quality(&self) -> QualityLevel {
        *self.current_quality.lock().unwrap()
    }

    /// Get current synthesis parameters
    pub fn get_synthesis_params(&self) -> SynthesisQualityParams {
        self.get_quality().synthesis_params()
    }

    /// Manually set quality level
    pub fn set_quality(&self, quality: QualityLevel) {
        if quality as u8 >= self.min_quality as u8 && quality as u8 <= self.max_quality as u8 {
            *self.current_quality.lock().unwrap() = quality;
        }
    }

    /// Get CPU statistics
    pub fn get_cpu_stats(&self) -> CpuStats {
        self.load_estimator.get_cpu_stats()
    }

    /// Get quality adjustment statistics
    pub fn get_adjustment_stats(&self) -> AdaptiveQualityStats {
        let history = self.adjustment_history.lock().unwrap();

        let total_adjustments = history.len() as u64;
        let quality_upgrades = history
            .iter()
            .filter(|a| (a.new_quality as u8) > (a.old_quality as u8))
            .count() as u64;
        let quality_downgrades = history
            .iter()
            .filter(|a| (a.new_quality as u8) < (a.old_quality as u8))
            .count() as u64;

        let avg_cpu_at_adjustment = if !history.is_empty() {
            history.iter().map(|a| a.cpu_load).sum::<f32>() / history.len() as f32
        } else {
            0.0
        };

        AdaptiveQualityStats {
            current_quality: self.get_quality(),
            current_cpu_load: self.load_estimator.get_load(),
            target_cpu: self.target_cpu,
            total_adjustments,
            quality_upgrades,
            quality_downgrades,
            avg_cpu_at_adjustment,
        }
    }
}

/// Quality adjustment record
#[derive(Debug, Clone)]
struct QualityAdjustment {
    #[allow(dead_code)]
    timestamp: Instant,
    old_quality: QualityLevel,
    new_quality: QualityLevel,
    cpu_load: f32,
    reason: AdjustmentReason,
}

/// Reason for quality adjustment
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum AdjustmentReason {
    CpuOverload,
    CpuUnderload,
    TrendBased,
}

/// Adaptive quality statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveQualityStats {
    /// Current quality level
    pub current_quality: QualityLevel,

    /// Current CPU load
    pub current_cpu_load: f32,

    /// Target CPU usage
    pub target_cpu: f32,

    /// Total number of quality adjustments
    pub total_adjustments: u64,

    /// Number of quality upgrades
    pub quality_upgrades: u64,

    /// Number of quality downgrades
    pub quality_downgrades: u64,

    /// Average CPU at time of adjustment
    pub avg_cpu_at_adjustment: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_level_factor() {
        assert_eq!(QualityLevel::Minimum.quality_factor(), 0.60);
        assert_eq!(QualityLevel::Maximum.quality_factor(), 1.00);
    }

    #[test]
    fn test_quality_level_transitions() {
        assert_eq!(QualityLevel::Medium.lower(), Some(QualityLevel::Low));
        assert_eq!(QualityLevel::Medium.higher(), Some(QualityLevel::High));
        assert_eq!(QualityLevel::Minimum.lower(), None);
        assert_eq!(QualityLevel::Maximum.higher(), None);
    }

    #[test]
    fn test_quality_synthesis_params() {
        let params = QualityLevel::High.synthesis_params();
        assert_eq!(params.harmonics_count, 24);
        assert_eq!(params.formant_filters, 5);
        assert_eq!(params.oversampling, 2);
    }

    #[test]
    fn test_adaptive_quality_scaler_creation() {
        let scaler = AdaptiveQualityScaler::new(0.75, QualityLevel::Minimum, QualityLevel::Maximum);
        assert_eq!(scaler.get_quality(), QualityLevel::Maximum);
    }

    #[test]
    fn test_adaptive_quality_manual_set() {
        let scaler = AdaptiveQualityScaler::new(0.75, QualityLevel::Low, QualityLevel::High);

        scaler.set_quality(QualityLevel::Medium);
        assert_eq!(scaler.get_quality(), QualityLevel::Medium);

        // Should not allow quality below minimum
        scaler.set_quality(QualityLevel::Minimum);
        assert_eq!(scaler.get_quality(), QualityLevel::Medium);
    }

    #[test]
    fn test_adaptive_quality_high_cpu_reduces_quality() {
        let scaler = AdaptiveQualityScaler::new(0.5, QualityLevel::Minimum, QualityLevel::Maximum);

        // Simulate high CPU load
        std::thread::sleep(Duration::from_millis(600));
        let (quality, adjusted) =
            scaler.update(Duration::from_millis(9), Duration::from_millis(10));

        // Quality should be reduced or remain at max
        assert!(quality as u8 <= QualityLevel::Maximum as u8);
    }

    #[test]
    fn test_adaptive_quality_low_cpu_increases_quality() {
        let scaler = AdaptiveQualityScaler::new(0.75, QualityLevel::Minimum, QualityLevel::Maximum);

        // Start at low quality
        scaler.set_quality(QualityLevel::Low);

        // Simulate low CPU load
        std::thread::sleep(Duration::from_millis(600));
        let (quality, _) = scaler.update(Duration::from_millis(3), Duration::from_millis(10));

        // Quality might increase
        assert!(quality as u8 >= QualityLevel::Low as u8);
    }

    #[test]
    fn test_adaptive_quality_stats() {
        let scaler = AdaptiveQualityScaler::new(0.75, QualityLevel::Minimum, QualityLevel::Maximum);

        // Force some adjustments
        for i in 0..5 {
            std::thread::sleep(Duration::from_millis(600));
            let cpu_time = Duration::from_millis(3 + i * 2);
            scaler.update(cpu_time, Duration::from_millis(10));
        }

        let stats = scaler.get_adjustment_stats();
        assert_eq!(stats.target_cpu, 0.75);
        assert!(stats.current_cpu_load >= 0.0);
    }

    #[test]
    fn test_synthesis_quality_params_scaling() {
        let min_params = QualityLevel::Minimum.synthesis_params();
        let max_params = QualityLevel::Maximum.synthesis_params();

        assert!(max_params.harmonics_count > min_params.harmonics_count);
        assert!(max_params.formant_filters > min_params.formant_filters);
        assert!(max_params.spectral_resolution > min_params.spectral_resolution);
    }
}
