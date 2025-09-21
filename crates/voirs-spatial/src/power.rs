//! Power Consumption Optimization for VoiRS Spatial Audio
//!
//! This module provides power management and optimization strategies for battery-powered devices,
//! including mobile phones, VR headsets, and other portable spatial audio devices.

use crate::config::SpatialConfig;
use crate::mobile::{MobileConfig, PowerState, QualityPreset};
use crate::types::Position3D;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Power management strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerStrategy {
    /// Maximum performance, highest power consumption
    Performance,
    /// Balanced performance and power consumption
    Balanced,
    /// Prioritize battery life over performance
    PowerSaver,
    /// Minimum power consumption, basic functionality only
    UltraLowPower,
    /// Adaptive strategy based on usage patterns
    Adaptive,
}

/// Device type for power optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Mobile phone or tablet
    Mobile,
    /// VR headset
    VrHeadset,
    /// AR glasses
    ArGlasses,
    /// Gaming handheld
    GamingHandheld,
    /// Smart earbuds
    Earbuds,
    /// Other portable device
    Other,
}

/// Power optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerConfig {
    /// Power management strategy
    pub strategy: PowerStrategy,
    /// Target device type
    pub device_type: DeviceType,
    /// Battery capacity (mAh)
    pub battery_capacity: u32,
    /// Target battery life (hours)
    pub target_battery_life: f32,
    /// Current battery level (0.0 - 1.0)
    pub current_battery_level: f32,
    /// Thermal threshold (°C)
    pub thermal_threshold: f32,
    /// Enable aggressive power saving
    pub aggressive_power_saving: bool,
    /// Minimum quality level to maintain
    pub min_quality_level: f32,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f32,
    /// Enable display-off optimizations
    pub display_off_optimizations: bool,
    /// Enable background processing optimizations
    pub background_optimizations: bool,
}

impl Default for PowerConfig {
    fn default() -> Self {
        Self {
            strategy: PowerStrategy::Balanced,
            device_type: DeviceType::Mobile,
            battery_capacity: 3000,
            target_battery_life: 8.0,
            current_battery_level: 1.0,
            thermal_threshold: 40.0,
            aggressive_power_saving: false,
            min_quality_level: 0.2,
            max_cpu_usage: 25.0,
            display_off_optimizations: true,
            background_optimizations: true,
        }
    }
}

/// Power consumption profile for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerProfile {
    /// Base power consumption (mW)
    pub base_power: f32,
    /// CPU processing power per source (mW/source)
    pub cpu_power_per_source: f32,
    /// GPU processing power (mW) when enabled
    pub gpu_power: f32,
    /// Memory access power (mW/MB)
    pub memory_power: f32,
    /// Audio output power (mW)
    pub audio_output_power: f32,
    /// Sensor power (IMU, tracking) (mW)
    pub sensor_power: f32,
    /// Display power (mW) - for VR/AR
    pub display_power: f32,
}

impl PowerProfile {
    /// Get power profile for device type
    pub fn for_device_type(device_type: DeviceType) -> Self {
        match device_type {
            DeviceType::Mobile => Self {
                base_power: 200.0,
                cpu_power_per_source: 15.0,
                gpu_power: 300.0,
                memory_power: 0.5,
                audio_output_power: 50.0,
                sensor_power: 20.0,
                display_power: 800.0,
            },
            DeviceType::VrHeadset => Self {
                base_power: 500.0,
                cpu_power_per_source: 25.0,
                gpu_power: 2000.0,
                memory_power: 1.0,
                audio_output_power: 100.0,
                sensor_power: 150.0,
                display_power: 3000.0,
            },
            DeviceType::ArGlasses => Self {
                base_power: 300.0,
                cpu_power_per_source: 20.0,
                gpu_power: 800.0,
                memory_power: 0.7,
                audio_output_power: 75.0,
                sensor_power: 100.0,
                display_power: 500.0,
            },
            DeviceType::GamingHandheld => Self {
                base_power: 400.0,
                cpu_power_per_source: 20.0,
                gpu_power: 1500.0,
                memory_power: 0.8,
                audio_output_power: 100.0,
                sensor_power: 50.0,
                display_power: 1200.0,
            },
            DeviceType::Earbuds => Self {
                base_power: 20.0,
                cpu_power_per_source: 5.0,
                gpu_power: 0.0,
                memory_power: 0.1,
                audio_output_power: 30.0,
                sensor_power: 10.0,
                display_power: 0.0,
            },
            DeviceType::Other => Self {
                base_power: 250.0,
                cpu_power_per_source: 18.0,
                gpu_power: 500.0,
                memory_power: 0.6,
                audio_output_power: 75.0,
                sensor_power: 75.0,
                display_power: 600.0,
            },
        }
    }
}

/// Power usage metrics
#[derive(Debug, Clone, Default)]
pub struct PowerMetrics {
    /// Current power consumption (mW)
    pub current_power: f32,
    /// Average power consumption (mW)
    pub average_power: f32,
    /// Peak power consumption (mW)
    pub peak_power: f32,
    /// Estimated battery life remaining (hours)
    pub estimated_battery_life: f32,
    /// Power efficiency (operations per watt)
    pub efficiency: f32,
    /// Thermal state (0.0 = cool, 1.0 = hot)
    pub thermal_state: f32,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// GPU usage percentage (if applicable)
    pub gpu_usage: Option<f32>,
    /// Memory usage (MB)
    pub memory_usage: f32,
}

/// Power optimization history entry
#[derive(Debug, Clone)]
struct PowerHistoryEntry {
    timestamp: Instant,
    power_consumption: f32,
    cpu_usage: f32,
    thermal_state: f32,
    quality_level: f32,
    source_count: u32,
}

/// Power optimization manager
pub struct PowerOptimizer {
    config: PowerConfig,
    profile: PowerProfile,
    metrics: PowerMetrics,
    history: VecDeque<PowerHistoryEntry>,
    adaptive_params: AdaptiveParams,
    last_optimization: Instant,
    optimization_interval: Duration,
}

/// Adaptive power management parameters
#[derive(Debug, Clone)]
struct AdaptiveParams {
    /// Learning rate for adaptive optimization
    learning_rate: f32,
    /// Usage pattern weights
    usage_weights: [f32; 5], // Different usage scenarios
    /// Quality adjustment factor
    quality_factor: f32,
    /// Thermal response factor
    thermal_factor: f32,
    /// Battery level response factor
    battery_factor: f32,
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            usage_weights: [0.2, 0.2, 0.2, 0.2, 0.2],
            quality_factor: 1.0,
            thermal_factor: 1.0,
            battery_factor: 1.0,
        }
    }
}

impl PowerOptimizer {
    /// Create a new power optimizer
    pub fn new(config: PowerConfig) -> Self {
        let profile = PowerProfile::for_device_type(config.device_type);
        let optimization_interval = match config.strategy {
            PowerStrategy::Performance => Duration::from_secs(10),
            PowerStrategy::Balanced => Duration::from_secs(5),
            PowerStrategy::PowerSaver => Duration::from_secs(2),
            PowerStrategy::UltraLowPower => Duration::from_secs(1),
            PowerStrategy::Adaptive => Duration::from_secs(3),
        };

        Self {
            config,
            profile,
            metrics: PowerMetrics::default(),
            history: VecDeque::with_capacity(3600), // 1 hour of history at 1s intervals
            adaptive_params: AdaptiveParams::default(),
            last_optimization: Instant::now(),
            optimization_interval,
        }
    }

    /// Update system state and optimize power consumption
    #[allow(clippy::too_many_arguments)]
    pub fn update_state(
        &mut self,
        battery_level: f32,
        thermal_temp: f32,
        cpu_usage: f32,
        gpu_usage: Option<f32>,
        memory_usage: f32,
        source_count: u32,
        quality_level: f32,
    ) {
        self.config.current_battery_level = battery_level.clamp(0.0, 1.0);
        self.metrics.thermal_state = (thermal_temp - 20.0) / (self.config.thermal_threshold - 20.0);
        self.metrics.thermal_state = self.metrics.thermal_state.clamp(0.0, 1.0);
        self.metrics.cpu_usage = cpu_usage;
        self.metrics.gpu_usage = gpu_usage;
        self.metrics.memory_usage = memory_usage;

        // Calculate current power consumption
        self.metrics.current_power = self.calculate_power_consumption(
            source_count,
            quality_level,
            cpu_usage,
            gpu_usage.unwrap_or(0.0),
            memory_usage,
        );

        // Update history
        let history_entry = PowerHistoryEntry {
            timestamp: Instant::now(),
            power_consumption: self.metrics.current_power,
            cpu_usage,
            thermal_state: self.metrics.thermal_state,
            quality_level,
            source_count,
        };

        self.history.push_back(history_entry);
        if self.history.len() > 3600 {
            self.history.pop_front();
        }

        // Update average and peak power
        self.update_power_statistics();

        // Calculate estimated battery life
        self.update_battery_estimation();

        // Adaptive learning
        if self.config.strategy == PowerStrategy::Adaptive {
            self.update_adaptive_params();
        }
    }

    /// Calculate power consumption based on current state
    fn calculate_power_consumption(
        &self,
        source_count: u32,
        quality_level: f32,
        cpu_usage: f32,
        gpu_usage: f32,
        memory_usage: f32,
    ) -> f32 {
        let mut total_power = self.profile.base_power;

        // CPU power scales with source count and quality
        total_power += self.profile.cpu_power_per_source * source_count as f32 * quality_level;

        // GPU power when enabled
        if gpu_usage > 0.0 {
            total_power += self.profile.gpu_power * (gpu_usage / 100.0);
        }

        // Memory access power
        total_power += self.profile.memory_power * memory_usage;

        // Audio output power
        total_power += self.profile.audio_output_power;

        // Sensor power
        total_power += self.profile.sensor_power;

        // Display power (if applicable)
        if matches!(
            self.config.device_type,
            DeviceType::VrHeadset | DeviceType::ArGlasses | DeviceType::GamingHandheld
        ) {
            total_power += self.profile.display_power;
        }

        // Thermal scaling
        if self.metrics.thermal_state > 0.8 {
            total_power *= 1.2; // Thermal throttling increases power
        }

        total_power
    }

    /// Update power statistics
    fn update_power_statistics(&mut self) {
        if self.history.is_empty() {
            return;
        }

        let recent_power: Vec<f32> = self
            .history
            .iter()
            .rev()
            .take(60) // Last minute
            .map(|entry| entry.power_consumption)
            .collect();

        self.metrics.average_power = recent_power.iter().sum::<f32>() / recent_power.len() as f32;
        self.metrics.peak_power = recent_power.iter().cloned().fold(0.0, f32::max);
    }

    /// Update battery life estimation
    fn update_battery_estimation(&mut self) {
        if self.metrics.average_power > 0.0 {
            let remaining_capacity_mah =
                self.config.battery_capacity as f32 * self.config.current_battery_level;
            let remaining_capacity_mwh = remaining_capacity_mah * 3.7; // Convert mAh to mWh (assuming 3.7V)
            let hours_remaining = remaining_capacity_mwh / self.metrics.average_power; // mWh / mW = hours
            self.metrics.estimated_battery_life = hours_remaining;
        }
    }

    /// Update adaptive parameters based on usage patterns
    fn update_adaptive_params(&mut self) {
        if self.history.len() < 60 {
            return; // Need at least 1 minute of data
        }

        // Analyze recent usage patterns
        let recent_entries: Vec<&PowerHistoryEntry> = self.history.iter().rev().take(300).collect(); // Last 5 minutes

        // Calculate correlation between quality and power consumption
        let quality_power_correlation = self.calculate_correlation(
            &recent_entries
                .iter()
                .map(|e| e.quality_level)
                .collect::<Vec<f32>>(),
            &recent_entries
                .iter()
                .map(|e| e.power_consumption)
                .collect::<Vec<f32>>(),
        );

        // Adjust adaptive parameters
        if quality_power_correlation > 0.7 {
            self.adaptive_params.quality_factor =
                (self.adaptive_params.quality_factor * 0.95).max(0.5);
        } else if quality_power_correlation < 0.3 {
            self.adaptive_params.quality_factor =
                (self.adaptive_params.quality_factor * 1.05).min(1.5);
        }

        // Thermal adaptation
        let avg_thermal = recent_entries.iter().map(|e| e.thermal_state).sum::<f32>()
            / recent_entries.len() as f32;
        if avg_thermal > 0.7 {
            self.adaptive_params.thermal_factor *= 0.9;
        } else if avg_thermal < 0.3 {
            self.adaptive_params.thermal_factor *= 1.1;
        }

        // Battery adaptation
        if self.config.current_battery_level < 0.2 {
            self.adaptive_params.battery_factor *= 0.8;
        } else if self.config.current_battery_level > 0.8 {
            self.adaptive_params.battery_factor *= 1.1;
        }

        // Clamp factors
        self.adaptive_params.quality_factor = self.adaptive_params.quality_factor.clamp(0.3, 2.0);
        self.adaptive_params.thermal_factor = self.adaptive_params.thermal_factor.clamp(0.5, 1.5);
        self.adaptive_params.battery_factor = self.adaptive_params.battery_factor.clamp(0.3, 1.5);
    }

    /// Calculate correlation between two data series
    fn calculate_correlation(&self, x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f32;
        let mean_x = x.iter().sum::<f32>() / n;
        let mean_y = y.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Get optimized spatial configuration for current power state
    pub fn get_optimized_config(&self) -> SpatialConfig {
        let mut config = SpatialConfig::default();

        // Base optimization based on strategy
        match self.config.strategy {
            PowerStrategy::Performance => {
                config.quality_level = 1.0;
                config.max_sources = 32;
                config.use_gpu = true;
                config.buffer_size = 1024;
            }
            PowerStrategy::Balanced => {
                config.quality_level = 0.7 * self.adaptive_params.quality_factor;
                config.max_sources = 16;
                config.use_gpu = self.metrics.thermal_state < 0.6;
                config.buffer_size = 2048;
            }
            PowerStrategy::PowerSaver => {
                config.quality_level = 0.4 * self.adaptive_params.quality_factor;
                config.max_sources = 8;
                config.use_gpu = false;
                config.buffer_size = 4096;
            }
            PowerStrategy::UltraLowPower => {
                config.quality_level = self.config.min_quality_level;
                config.max_sources = 4;
                config.use_gpu = false;
                config.buffer_size = 8192;
                config.sample_rate = 22050; // Lower sample rate
            }
            PowerStrategy::Adaptive => {
                // Use adaptive parameters
                config.quality_level = (0.6 * self.adaptive_params.quality_factor)
                    .clamp(self.config.min_quality_level, 1.0);
                config.max_sources = if self.config.current_battery_level > 0.5 {
                    16
                } else {
                    8
                };
                config.use_gpu =
                    self.metrics.thermal_state < 0.5 && self.config.current_battery_level > 0.3;
                config.buffer_size = if self.config.current_battery_level > 0.5 {
                    2048
                } else {
                    4096
                };
            }
        }

        // Additional optimizations based on device state
        if self.config.current_battery_level < 0.1 {
            // Critical battery
            config.quality_level = self.config.min_quality_level;
            config.max_sources = 2;
            config.use_gpu = false;
            config.buffer_size = 8192;
            config.sample_rate = 16000;
        } else if self.metrics.thermal_state > 0.8 {
            // Thermal throttling
            config.quality_level *= 0.5;
            config.max_sources = (config.max_sources / 2).max(2);
            config.use_gpu = false;
        }

        // Device-specific optimizations
        match self.config.device_type {
            DeviceType::Earbuds => {
                config.max_sources = config.max_sources.min(4);
                config.buffer_size = config.buffer_size.max(2048);
            }
            DeviceType::VrHeadset => {
                // VR needs lower latency but can use more power
                config.buffer_size = config.buffer_size.min(2048);
            }
            DeviceType::ArGlasses => {
                // AR needs to balance processing with transparency
                config.quality_level *= 0.9;
            }
            _ => {} // No specific optimizations
        }

        config
    }

    /// Get current power metrics
    pub fn get_metrics(&self) -> PowerMetrics {
        self.metrics.clone()
    }

    /// Force a specific power strategy
    pub fn set_power_strategy(&mut self, strategy: PowerStrategy) {
        self.config.strategy = strategy;
    }

    /// Enable/disable aggressive power saving
    pub fn set_aggressive_power_saving(&mut self, enabled: bool) {
        self.config.aggressive_power_saving = enabled;
    }

    /// Check if optimization update is needed
    pub fn should_optimize(&self) -> bool {
        self.last_optimization.elapsed() >= self.optimization_interval
    }

    /// Perform optimization cycle
    pub fn optimize(&mut self) -> Result<()> {
        self.last_optimization = Instant::now();

        // Adaptive strategy learns from usage patterns
        if self.config.strategy == PowerStrategy::Adaptive {
            self.update_adaptive_params();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_config_creation() {
        let config = PowerConfig::default();
        assert_eq!(config.strategy, PowerStrategy::Balanced);
        assert_eq!(config.device_type, DeviceType::Mobile);
    }

    #[test]
    fn test_power_profile_device_specific() {
        let mobile_profile = PowerProfile::for_device_type(DeviceType::Mobile);
        let vr_profile = PowerProfile::for_device_type(DeviceType::VrHeadset);
        let earbuds_profile = PowerProfile::for_device_type(DeviceType::Earbuds);

        // VR should consume more power than mobile
        assert!(vr_profile.base_power > mobile_profile.base_power);
        assert!(vr_profile.gpu_power > mobile_profile.gpu_power);

        // Earbuds should consume least power
        assert!(earbuds_profile.base_power < mobile_profile.base_power);
        assert_eq!(earbuds_profile.gpu_power, 0.0);
    }

    #[test]
    fn test_power_optimizer_creation() {
        let config = PowerConfig::default();
        let optimizer = PowerOptimizer::new(config);

        assert_eq!(optimizer.config.strategy, PowerStrategy::Balanced);
        assert_eq!(optimizer.history.len(), 0);
    }

    #[test]
    fn test_power_consumption_calculation() {
        let config = PowerConfig::default();
        let optimizer = PowerOptimizer::new(config);

        let power = optimizer.calculate_power_consumption(8, 0.8, 50.0, 0.0, 100.0);
        assert!(power > 0.0);

        // More sources should use more power
        let power_more_sources = optimizer.calculate_power_consumption(16, 0.8, 50.0, 0.0, 100.0);
        assert!(power_more_sources > power);
    }

    #[test]
    fn test_power_state_updates() {
        let config = PowerConfig::default();
        let mut optimizer = PowerOptimizer::new(config);

        optimizer.update_state(0.5, 35.0, 40.0, Some(60.0), 150.0, 12, 0.7);

        assert_eq!(optimizer.config.current_battery_level, 0.5);
        assert!(optimizer.metrics.current_power > 0.0);
        assert_eq!(optimizer.history.len(), 1);
    }

    #[test]
    fn test_optimized_config_generation() {
        let config = PowerConfig {
            strategy: PowerStrategy::PowerSaver,
            device_type: DeviceType::Mobile,
            ..Default::default()
        };
        let optimizer = PowerOptimizer::new(config);

        let spatial_config = optimizer.get_optimized_config();
        assert!(spatial_config.quality_level <= 0.5);
        assert!(!spatial_config.use_gpu);
        assert!(spatial_config.buffer_size >= 4096);
    }

    #[test]
    fn test_critical_battery_optimization() {
        let config = PowerConfig {
            current_battery_level: 0.05, // Critical battery
            ..Default::default()
        };
        let optimizer = PowerOptimizer::new(config);

        let spatial_config = optimizer.get_optimized_config();
        assert_eq!(spatial_config.max_sources, 2);
        assert!(!spatial_config.use_gpu);
        assert_eq!(spatial_config.sample_rate, 16000);
    }

    #[test]
    fn test_thermal_throttling() {
        let config = PowerConfig::default();
        let mut optimizer = PowerOptimizer::new(config);

        // Simulate high thermal state
        optimizer.metrics.thermal_state = 0.9;

        let spatial_config = optimizer.get_optimized_config();
        assert!(!spatial_config.use_gpu);
        assert!(spatial_config.quality_level < 0.5);
    }

    #[test]
    fn test_device_specific_optimization() {
        let earbuds_config = PowerConfig {
            device_type: DeviceType::Earbuds,
            ..Default::default()
        };
        let earbuds_optimizer = PowerOptimizer::new(earbuds_config);

        let spatial_config = earbuds_optimizer.get_optimized_config();
        assert!(spatial_config.max_sources <= 4);

        let vr_config = PowerConfig {
            device_type: DeviceType::VrHeadset,
            ..Default::default()
        };
        let vr_optimizer = PowerOptimizer::new(vr_config);

        let vr_spatial_config = vr_optimizer.get_optimized_config();
        assert!(vr_spatial_config.buffer_size <= 2048); // VR needs lower latency
    }

    #[test]
    fn test_adaptive_strategy() {
        let config = PowerConfig {
            strategy: PowerStrategy::Adaptive,
            ..Default::default()
        };
        let mut optimizer = PowerOptimizer::new(config);

        // Simulate usage pattern
        for i in 0..100 {
            optimizer.update_state(0.8, 30.0, 20.0, None, 100.0, 8, 0.8);
            if i % 10 == 0 {
                optimizer.optimize().unwrap();
            }
        }

        assert!(optimizer.adaptive_params.quality_factor > 0.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let config = PowerConfig::default();
        let optimizer = PowerOptimizer::new(config);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let correlation = optimizer.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 0.01); // Perfect positive correlation
    }

    #[test]
    fn test_battery_estimation() {
        let config = PowerConfig {
            battery_capacity: 3000,
            current_battery_level: 0.5,
            ..Default::default()
        };
        let mut optimizer = PowerOptimizer::new(config);

        optimizer.metrics.average_power = 1000.0; // 1W
        optimizer.update_battery_estimation();

        assert!(optimizer.metrics.estimated_battery_life > 0.0);
        assert!(optimizer.metrics.estimated_battery_life < 100.0); // Reasonable range (less than 100 hours)
    }
}
