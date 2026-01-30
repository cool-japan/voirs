//! Battery management for mobile devices
//!
//! This module handles battery-aware processing, including:
//! - Battery level monitoring
//! - Charging status detection
//! - Power mode management
//! - Task scheduling based on battery state

use super::PowerMode;
use crate::RecognitionError;

/// Battery manager for power-aware processing
pub struct BatteryManager {
    /// Minimum battery percentage for heavy tasks
    min_battery_percent: f32,
    /// Current battery level (0.0 to 1.0)
    battery_level: f32,
    /// Is device currently charging
    is_charging: bool,
    /// Current power mode
    power_mode: PowerMode,
    /// Battery status last updated
    last_update: std::time::Instant,
}

impl BatteryManager {
    /// Create a new battery manager
    ///
    /// # Errors
    ///
    /// Returns an error if battery status cannot be initialized
    pub fn new(min_battery_percent: f32) -> Result<Self, RecognitionError> {
        let mut manager = Self {
            min_battery_percent,
            battery_level: 1.0, // Assume full battery initially
            is_charging: false,
            power_mode: PowerMode::Balanced,
            last_update: std::time::Instant::now(),
        };

        // Try to get initial battery status
        let _ = manager.read_battery_status();

        Ok(manager)
    }

    /// Update battery status from system
    ///
    /// # Errors
    ///
    /// Returns an error if battery status cannot be read
    pub async fn update_battery_status(&mut self) -> Result<(), RecognitionError> {
        self.read_battery_status()?;
        self.last_update = std::time::Instant::now();
        Ok(())
    }

    /// Read battery status from platform
    fn read_battery_status(&mut self) -> Result<(), RecognitionError> {
        // Platform-specific battery reading
        #[cfg(target_os = "linux")]
        {
            self.read_battery_linux()?;
        }
        #[cfg(target_os = "macos")]
        {
            self.read_battery_macos()?;
        }
        #[cfg(target_os = "ios")]
        {
            self.read_battery_ios()?;
        }
        #[cfg(target_os = "android")]
        {
            self.read_battery_android()?;
        }
        #[cfg(not(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android"
        )))]
        {
            // Default to full battery for unsupported platforms
            self.battery_level = 1.0;
            self.is_charging = false;
        }

        Ok(())
    }

    /// Read battery status on Linux
    #[cfg(target_os = "linux")]
    fn read_battery_linux(&mut self) -> Result<(), RecognitionError> {
        use std::fs;

        // Try to read from /sys/class/power_supply/BAT0/
        if let Ok(capacity) = fs::read_to_string("/sys/class/power_supply/BAT0/capacity") {
            if let Ok(percent) = capacity.trim().parse::<f32>() {
                self.battery_level = percent / 100.0;
            }
        }

        if let Ok(status) = fs::read_to_string("/sys/class/power_supply/BAT0/status") {
            self.is_charging = status.trim() == "Charging" || status.trim() == "Full";
        }

        Ok(())
    }

    /// Read battery status on macOS
    #[cfg(target_os = "macos")]
    fn read_battery_macos(&mut self) -> Result<(), RecognitionError> {
        use std::process::Command;

        // Use pmset to get battery info
        if let Ok(output) = Command::new("pmset").args(["-g", "batt"]).output() {
            let output_str = String::from_utf8_lossy(&output.stdout);

            // Parse battery percentage
            if let Some(percent_str) = output_str.lines().find(|line| line.contains("%")) {
                if let Some(start) = percent_str.find(char::is_numeric) {
                    if let Some(end) = percent_str[start..].find('%') {
                        if let Ok(percent) = percent_str[start..start + end].parse::<f32>() {
                            self.battery_level = percent / 100.0;
                        }
                    }
                }
            }

            // Check charging status
            self.is_charging = output_str.contains("AC Power") || output_str.contains("charging");
        }

        Ok(())
    }

    /// Read battery status on iOS
    #[cfg(target_os = "ios")]
    fn read_battery_ios(&mut self) -> Result<(), RecognitionError> {
        // iOS battery reading would require Objective-C bindings to UIDevice
        // For now, use conservative defaults
        self.battery_level = 0.8; // Assume 80% battery
        self.is_charging = false;
        Ok(())
    }

    /// Read battery status on Android
    #[cfg(target_os = "android")]
    fn read_battery_android(&mut self) -> Result<(), RecognitionError> {
        // Android battery reading would require JNI calls
        // For now, use conservative defaults
        self.battery_level = 0.8; // Assume 80% battery
        self.is_charging = false;
        Ok(())
    }

    /// Set power mode
    pub fn set_power_mode(&mut self, mode: PowerMode) {
        self.power_mode = mode;
    }

    /// Check if heavy tasks can run based on battery level
    pub fn can_run_heavy_tasks(&self) -> bool {
        // Allow heavy tasks if charging or battery above threshold
        self.is_charging || (self.battery_level * 100.0 >= self.min_battery_percent)
    }

    /// Get current battery level
    pub fn battery_level(&self) -> f32 {
        self.battery_level
    }

    /// Check if device is charging
    pub fn is_charging(&self) -> bool {
        self.is_charging
    }

    /// Get recommended processing intensity based on battery state
    pub fn recommended_intensity(&self) -> ProcessingIntensity {
        if self.is_charging {
            return ProcessingIntensity::Full;
        }

        match self.power_mode {
            PowerMode::Performance => ProcessingIntensity::Full,
            PowerMode::Balanced => {
                if self.battery_level > 0.5 {
                    ProcessingIntensity::High
                } else if self.battery_level > 0.3 {
                    ProcessingIntensity::Medium
                } else {
                    ProcessingIntensity::Low
                }
            }
            PowerMode::LowPower => ProcessingIntensity::Low,
            PowerMode::UltraLowPower => ProcessingIntensity::Minimal,
        }
    }

    /// Get time since last battery status update
    pub fn time_since_update(&self) -> std::time::Duration {
        self.last_update.elapsed()
    }
}

/// Processing intensity recommendation based on battery state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingIntensity {
    /// Full processing power
    Full,
    /// High processing (80-100%)
    High,
    /// Medium processing (50-80%)
    Medium,
    /// Low processing (30-50%)
    Low,
    /// Minimal processing (<30%)
    Minimal,
}

impl ProcessingIntensity {
    /// Get CPU usage target for this intensity level
    pub fn cpu_usage_target(&self) -> f32 {
        match self {
            ProcessingIntensity::Full => 1.0,
            ProcessingIntensity::High => 0.8,
            ProcessingIntensity::Medium => 0.5,
            ProcessingIntensity::Low => 0.3,
            ProcessingIntensity::Minimal => 0.15,
        }
    }

    /// Get batch size multiplier for this intensity level
    pub fn batch_size_multiplier(&self) -> f32 {
        match self {
            ProcessingIntensity::Full => 1.0,
            ProcessingIntensity::High => 0.75,
            ProcessingIntensity::Medium => 0.5,
            ProcessingIntensity::Low => 0.25,
            ProcessingIntensity::Minimal => 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_battery_manager_creation() {
        let manager = BatteryManager::new(20.0);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_can_run_heavy_tasks() {
        let mut manager = BatteryManager::new(20.0).unwrap();

        // High battery, not charging
        manager.battery_level = 0.8;
        manager.is_charging = false;
        assert!(manager.can_run_heavy_tasks());

        // Low battery, not charging
        manager.battery_level = 0.1;
        manager.is_charging = false;
        assert!(!manager.can_run_heavy_tasks());

        // Low battery, but charging
        manager.battery_level = 0.1;
        manager.is_charging = true;
        assert!(manager.can_run_heavy_tasks());
    }

    #[test]
    fn test_processing_intensity() {
        let mut manager = BatteryManager::new(20.0).unwrap();

        // Test Performance mode
        manager.set_power_mode(PowerMode::Performance);
        manager.battery_level = 0.3;
        manager.is_charging = false;
        assert_eq!(manager.recommended_intensity(), ProcessingIntensity::Full);

        // Test Balanced mode with high battery
        manager.set_power_mode(PowerMode::Balanced);
        manager.battery_level = 0.6;
        assert_eq!(manager.recommended_intensity(), ProcessingIntensity::High);

        // Test Balanced mode with medium battery
        manager.battery_level = 0.4;
        assert_eq!(manager.recommended_intensity(), ProcessingIntensity::Medium);

        // Test Balanced mode with low battery
        manager.battery_level = 0.2;
        assert_eq!(manager.recommended_intensity(), ProcessingIntensity::Low);

        // Test Low Power mode
        manager.set_power_mode(PowerMode::LowPower);
        assert_eq!(manager.recommended_intensity(), ProcessingIntensity::Low);

        // Test Ultra Low Power mode
        manager.set_power_mode(PowerMode::UltraLowPower);
        assert_eq!(
            manager.recommended_intensity(),
            ProcessingIntensity::Minimal
        );
    }

    #[test]
    fn test_intensity_metrics() {
        assert_eq!(ProcessingIntensity::Full.cpu_usage_target(), 1.0);
        assert_eq!(ProcessingIntensity::High.cpu_usage_target(), 0.8);
        assert_eq!(ProcessingIntensity::Medium.cpu_usage_target(), 0.5);
        assert_eq!(ProcessingIntensity::Low.cpu_usage_target(), 0.3);
        assert_eq!(ProcessingIntensity::Minimal.cpu_usage_target(), 0.15);
    }
}
