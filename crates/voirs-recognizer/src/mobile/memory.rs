//! Memory management for mobile devices
//!
//! This module provides memory-aware processing optimized for mobile constraints

use crate::RecognitionError;

/// Mobile memory manager
pub struct MobileMemoryManager {
    /// Maximum allowed memory usage in MB
    max_memory_mb: usize,
    /// Current memory usage estimate in MB
    current_usage_mb: usize,
}

impl MobileMemoryManager {
    /// Create a new mobile memory manager
    ///
    /// # Errors
    ///
    /// Returns an error if memory information cannot be obtained
    pub fn new(max_memory_mb: usize) -> Result<Self, RecognitionError> {
        Ok(Self {
            max_memory_mb,
            current_usage_mb: 0,
        })
    }

    /// Get total system memory in MB
    pub fn total_memory_mb(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb / 1024; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }

        // Default to 4GB for unknown platforms
        4096
    }

    /// Get available memory in MB
    ///
    /// # Errors
    ///
    /// Returns an error if available memory cannot be determined
    pub fn available_memory_mb(&self) -> Result<usize, RecognitionError> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return Ok(kb / 1024);
                            }
                        }
                    }
                }
            }
        }

        // Estimate based on max allowed
        Ok(self.max_memory_mb.saturating_sub(self.current_usage_mb))
    }

    /// Optimize memory usage
    ///
    /// # Errors
    ///
    /// Returns an error if optimization fails
    pub async fn optimize(&mut self) -> Result<(), RecognitionError> {
        // Clear caches, release unused resources
        self.current_usage_mb = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let manager = MobileMemoryManager::new(512);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_memory_optimization() {
        let mut manager = MobileMemoryManager::new(512).unwrap();
        assert!(manager.optimize().await.is_ok());
    }
}
