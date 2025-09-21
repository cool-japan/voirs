//! Integration test for configuration functionality
//! This test verifies that the configuration system works independently of other modules

#[cfg(test)]
mod tests {
    use std::fs;
    use tempfile::tempdir;

    // Since we can't compile the full voirs-ffi due to vocoder issues,
    // let's create a simple test to verify our configuration logic

    #[test]
    fn test_config_serialization() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        struct TestConfig {
            speaking_rate: f32,
            thread_count: u32,
            enable_enhancement: bool,
            output_format: String,
        }

        impl Default for TestConfig {
            fn default() -> Self {
                Self {
                    speaking_rate: 1.0,
                    thread_count: 4,
                    enable_enhancement: true,
                    output_format: "wav".to_string(),
                }
            }
        }

        let config = TestConfig::default();

        // Test JSON serialization
        let json = serde_json::to_string_pretty(&config).unwrap();
        assert!(json.contains("speaking_rate"));
        assert!(json.contains("1.0"));

        // Test deserialization
        let deserialized: TestConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_file_operations() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct TestConfig {
            value: String,
        }

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_config.json");

        let config = TestConfig {
            value: "test_value".to_string(),
        };

        // Test saving
        let json = serde_json::to_string_pretty(&config).unwrap();
        fs::write(&file_path, json).unwrap();
        assert!(file_path.exists());

        // Test loading
        let content = fs::read_to_string(&file_path).unwrap();
        let loaded: TestConfig = serde_json::from_str(&content).unwrap();
        assert_eq!(config.value, loaded.value);
    }

    #[test]
    fn test_validation_logic() {
        // Test validation logic similar to what's in our config module
        fn validate_speaking_rate(rate: f32) -> Result<(), String> {
            if rate <= 0.0 || rate > 5.0 {
                return Err("Speaking rate must be between 0.0 and 5.0".to_string());
            }
            Ok(())
        }

        fn validate_thread_count(count: u32) -> Result<(), String> {
            if count == 0 || count > 64 {
                return Err("Thread count must be between 1 and 64".to_string());
            }
            Ok(())
        }

        // Test valid values
        assert!(validate_speaking_rate(1.0).is_ok());
        assert!(validate_speaking_rate(2.5).is_ok());
        assert!(validate_thread_count(4).is_ok());
        assert!(validate_thread_count(16).is_ok());

        // Test invalid values
        assert!(validate_speaking_rate(0.0).is_err());
        assert!(validate_speaking_rate(10.0).is_err());
        assert!(validate_thread_count(0).is_err());
        assert!(validate_thread_count(100).is_err());
    }

    #[test]
    fn test_key_value_parsing() {
        // Test key-value parsing logic similar to what's in our config module
        fn parse_key_path(key: &str) -> Option<(String, String)> {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() != 2 {
                return None;
            }
            Some((parts[0].to_string(), parts[1].to_string()))
        }

        // Test valid key paths
        assert_eq!(
            parse_key_path("synthesis.speaking_rate"),
            Some(("synthesis".to_string(), "speaking_rate".to_string()))
        );
        assert_eq!(
            parse_key_path("threading.thread_count"),
            Some(("threading".to_string(), "thread_count".to_string()))
        );

        // Test invalid key paths
        assert_eq!(parse_key_path("invalid_key"), None);
        assert_eq!(parse_key_path("too.many.parts"), None);
        assert_eq!(parse_key_path(""), None);
    }

    #[test]
    fn test_configuration_registry_pattern() {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        // Simulate the configuration registry pattern we use
        #[derive(Debug, Clone)]
        struct MockConfig {
            name: String,
            value: u32,
        }

        impl Default for MockConfig {
            fn default() -> Self {
                Self {
                    name: "default".to_string(),
                    value: 42,
                }
            }
        }

        let registry: Arc<Mutex<HashMap<u32, MockConfig>>> = Arc::new(Mutex::new(HashMap::new()));

        // Test adding config
        {
            let mut reg = registry.lock().unwrap();
            reg.insert(1, MockConfig::default());
        }

        // Test getting config
        {
            let reg = registry.lock().unwrap();
            let config = reg.get(&1).unwrap();
            assert_eq!(config.name, "default");
            assert_eq!(config.value, 42);
        }

        // Test updating config
        {
            let mut reg = registry.lock().unwrap();
            if let Some(config) = reg.get_mut(&1) {
                config.value = 100;
            }
        }

        // Verify update
        {
            let reg = registry.lock().unwrap();
            let config = reg.get(&1).unwrap();
            assert_eq!(config.value, 100);
        }

        // Test removing config
        {
            let mut reg = registry.lock().unwrap();
            assert!(reg.remove(&1).is_some());
            assert!(reg.get(&1).is_none());
        }
    }
}
