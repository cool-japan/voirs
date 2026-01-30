//! Network awareness for mobile platforms
//!
//! This module handles network-aware processing

use super::NetworkType;
use crate::RecognitionError;

/// Network manager for connectivity-aware processing
pub struct NetworkManager {
    /// Current network type
    network_type: NetworkType,
}

impl NetworkManager {
    /// Create a new network manager
    ///
    /// # Errors
    ///
    /// Returns an error if network status cannot be determined
    pub fn new() -> Result<Self, RecognitionError> {
        let network_type = Self::detect_network_type();
        Ok(Self { network_type })
    }

    /// Detect current network type
    fn detect_network_type() -> NetworkType {
        // Platform-specific network detection
        // For now, assume WiFi for development
        NetworkType::Wifi
    }

    /// Get current network type
    pub fn network_type(&self) -> NetworkType {
        self.network_type
    }

    /// Check if network is suitable for large downloads
    pub fn is_suitable_for_downloads(&self) -> bool {
        matches!(
            self.network_type,
            NetworkType::Wifi | NetworkType::Cellular4G | NetworkType::Cellular5G
        )
    }

    /// Check if network is suitable for streaming
    pub fn is_suitable_for_streaming(&self) -> bool {
        matches!(
            self.network_type,
            NetworkType::Wifi | NetworkType::Cellular4G | NetworkType::Cellular5G
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_manager_creation() {
        let manager = NetworkManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_download_suitability() {
        let manager = NetworkManager::new().unwrap();
        // Should be suitable with default WiFi
        assert!(manager.is_suitable_for_downloads());
    }
}
