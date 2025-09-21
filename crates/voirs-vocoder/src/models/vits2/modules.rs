//! Common modules for VITS2

use serde::{Deserialize, Serialize};

/// Module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleConfig {
    /// Module type
    pub module_type: String,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            module_type: "default".to_string(),
        }
    }
}
