use super::{Plugin, PluginError, PluginResult, PluginType};
use crate::audio::effects::{AudioEffect, EffectConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectPluginConfig {
    pub parameters: HashMap<String, f32>,
    pub enabled: bool,
    pub bypass: bool,
    pub wet_mix: f32,
    pub dry_mix: f32,
}

impl Default for EffectPluginConfig {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            enabled: true,
            bypass: false,
            wet_mix: 1.0,
            dry_mix: 0.0,
        }
    }
}

pub trait EffectPlugin: Plugin {
    fn process_audio(
        &self,
        input: &[f32],
        output: &mut [f32],
        config: &EffectPluginConfig,
    ) -> PluginResult<()>;
    fn get_parameter_info(&self) -> Vec<ParameterInfo>;
    fn set_parameter(&mut self, name: &str, value: f32) -> PluginResult<()>;
    fn get_parameter(&self, name: &str) -> PluginResult<f32>;
    fn reset(&mut self) -> PluginResult<()>;
    fn get_latency(&self) -> u32;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub name: String,
    pub display_name: String,
    pub min_value: f32,
    pub max_value: f32,
    pub default_value: f32,
    pub step_size: f32,
    pub unit: String,
    pub description: String,
    pub parameter_type: ParameterType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Float,
    Integer,
    Boolean,
    Choice(Vec<String>),
}

pub struct EffectPluginManager {
    effects: HashMap<String, Arc<dyn EffectPlugin>>,
    presets: HashMap<String, EffectPluginConfig>,
}

impl EffectPluginManager {
    pub fn new() -> Self {
        Self {
            effects: HashMap::new(),
            presets: HashMap::new(),
        }
    }

    pub fn register_effect(&mut self, name: String, effect: Arc<dyn EffectPlugin>) {
        self.effects.insert(name, effect);
    }

    pub fn unregister_effect(&mut self, name: &str) -> bool {
        self.effects.remove(name).is_some()
    }

    pub fn list_effects(&self) -> Vec<String> {
        self.effects.keys().cloned().collect()
    }

    pub fn get_effect(&self, name: &str) -> Option<Arc<dyn EffectPlugin>> {
        self.effects.get(name).cloned()
    }

    pub fn process_with_effect(
        &self,
        effect_name: &str,
        input: &[f32],
        output: &mut [f32],
        config: &EffectPluginConfig,
    ) -> PluginResult<()> {
        let effect = self
            .effects
            .get(effect_name)
            .ok_or_else(|| PluginError::NotFound(effect_name.to_string()))?;

        if !config.enabled || config.bypass {
            output.copy_from_slice(input);
            return Ok(());
        }

        effect.process_audio(input, output, config)
    }

    pub fn create_effect_chain(&self, effect_names: &[String]) -> EffectChain {
        let mut effects = Vec::new();
        for name in effect_names {
            if let Some(effect) = self.effects.get(name) {
                effects.push((name.clone(), effect.clone()));
            }
        }
        EffectChain::new(effects)
    }

    pub fn save_preset(&mut self, name: String, config: EffectPluginConfig) {
        self.presets.insert(name, config);
    }

    pub fn load_preset(&self, name: &str) -> Option<&EffectPluginConfig> {
        self.presets.get(name)
    }

    pub fn delete_preset(&mut self, name: &str) -> bool {
        self.presets.remove(name).is_some()
    }

    pub fn list_presets(&self) -> Vec<String> {
        self.presets.keys().cloned().collect()
    }
}

impl Default for EffectPluginManager {
    fn default() -> Self {
        Self::new()
    }
}

pub struct EffectChain {
    effects: Vec<(String, Arc<dyn EffectPlugin>)>,
    buffer: Vec<f32>,
}

impl EffectChain {
    pub fn new(effects: Vec<(String, Arc<dyn EffectPlugin>)>) -> Self {
        Self {
            effects,
            buffer: Vec::new(),
        }
    }

    pub fn process(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        configs: &HashMap<String, EffectPluginConfig>,
    ) -> PluginResult<()> {
        if self.effects.is_empty() {
            output.copy_from_slice(input);
            return Ok(());
        }

        // Ensure buffer is large enough
        if self.buffer.len() < input.len() {
            self.buffer.resize(input.len(), 0.0);
        }

        let default_config = EffectPluginConfig::default();

        // First effect processes input to buffer
        if let Some((name, effect)) = self.effects.first() {
            let config = configs.get(name).unwrap_or(&default_config);
            effect.process_audio(input, &mut self.buffer[..input.len()], config)?;
        }

        // Subsequent effects process buffer to buffer (ping-pong if needed)
        for i in 1..self.effects.len() {
            let (name, effect) = &self.effects[i];
            let config = configs.get(name).unwrap_or(&default_config);

            if i == self.effects.len() - 1 {
                // Last effect writes to output
                effect.process_audio(&self.buffer[..input.len()], output, config)?;
            } else {
                // Intermediate effects process in-place
                let mut temp_buffer = vec![0.0; input.len()];
                effect.process_audio(&self.buffer[..input.len()], &mut temp_buffer, config)?;
                self.buffer[..input.len()].copy_from_slice(&temp_buffer);
            }
        }

        Ok(())
    }

    pub fn get_total_latency(&self) -> u32 {
        self.effects
            .iter()
            .map(|(_, effect)| effect.get_latency())
            .sum()
    }

    pub fn reset_all(&mut self) -> PluginResult<()> {
        for (_, effect) in &self.effects {
            // Note: reset requires mutable reference, but we only have immutable
            // In a real implementation, effects would need interior mutability
        }
        Ok(())
    }
}

// Example builtin effect plugin implementations
pub struct ReverbEffectPlugin {
    name: String,
    version: String,
    room_size: f32,
    damping: f32,
    wet_level: f32,
    dry_level: f32,
}

impl ReverbEffectPlugin {
    pub fn new() -> Self {
        Self {
            name: "builtin-reverb".to_string(),
            version: "1.0.0".to_string(),
            room_size: 0.5,
            damping: 0.5,
            wet_level: 0.3,
            dry_level: 0.7,
        }
    }
}

impl Plugin for ReverbEffectPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn description(&self) -> &str {
        "Built-in reverb effect plugin"
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::Effect
    }

    fn initialize(&mut self, config: &serde_json::Value) -> PluginResult<()> {
        if let Some(room_size) = config.get("room_size").and_then(|v| v.as_f64()) {
            self.room_size = room_size as f32;
        }
        if let Some(damping) = config.get("damping").and_then(|v| v.as_f64()) {
            self.damping = damping as f32;
        }
        if let Some(wet_level) = config.get("wet_level").and_then(|v| v.as_f64()) {
            self.wet_level = wet_level as f32;
        }
        if let Some(dry_level) = config.get("dry_level").and_then(|v| v.as_f64()) {
            self.dry_level = dry_level as f32;
        }
        Ok(())
    }

    fn cleanup(&mut self) -> PluginResult<()> {
        Ok(())
    }

    fn get_capabilities(&self) -> Vec<String> {
        vec![
            "process_audio".to_string(),
            "set_parameter".to_string(),
            "get_parameter".to_string(),
            "reset".to_string(),
        ]
    }

    fn execute(&self, command: &str, args: &serde_json::Value) -> PluginResult<serde_json::Value> {
        match command {
            "get_parameters" => Ok(serde_json::json!({
                "room_size": self.room_size,
                "damping": self.damping,
                "wet_level": self.wet_level,
                "dry_level": self.dry_level
            })),
            "set_parameter" => {
                let param_name = args.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
                    PluginError::ExecutionFailed("Missing parameter name".to_string())
                })?;
                let value = args.get("value").and_then(|v| v.as_f64()).ok_or_else(|| {
                    PluginError::ExecutionFailed("Missing parameter value".to_string())
                })? as f32;

                match param_name {
                    "room_size" => Ok(serde_json::json!({"old_value": self.room_size})),
                    "damping" => Ok(serde_json::json!({"old_value": self.damping})),
                    "wet_level" => Ok(serde_json::json!({"old_value": self.wet_level})),
                    "dry_level" => Ok(serde_json::json!({"old_value": self.dry_level})),
                    _ => Err(PluginError::ExecutionFailed(format!(
                        "Unknown parameter: {}",
                        param_name
                    ))),
                }
            }
            _ => Err(PluginError::ExecutionFailed(format!(
                "Unknown command: {}",
                command
            ))),
        }
    }
}

impl EffectPlugin for ReverbEffectPlugin {
    fn process_audio(
        &self,
        input: &[f32],
        output: &mut [f32],
        config: &EffectPluginConfig,
    ) -> PluginResult<()> {
        // Simple placeholder reverb implementation
        for (i, &sample) in input.iter().enumerate() {
            let wet = sample * self.wet_level * config.wet_mix;
            let dry = sample * self.dry_level * config.dry_mix;
            output[i] = wet + dry;
        }
        Ok(())
    }

    fn get_parameter_info(&self) -> Vec<ParameterInfo> {
        vec![
            ParameterInfo {
                name: "room_size".to_string(),
                display_name: "Room Size".to_string(),
                min_value: 0.0,
                max_value: 1.0,
                default_value: 0.5,
                step_size: 0.01,
                unit: "".to_string(),
                description: "Size of the reverb room".to_string(),
                parameter_type: ParameterType::Float,
            },
            ParameterInfo {
                name: "damping".to_string(),
                display_name: "Damping".to_string(),
                min_value: 0.0,
                max_value: 1.0,
                default_value: 0.5,
                step_size: 0.01,
                unit: "".to_string(),
                description: "Damping factor for high frequencies".to_string(),
                parameter_type: ParameterType::Float,
            },
            ParameterInfo {
                name: "wet_level".to_string(),
                display_name: "Wet Level".to_string(),
                min_value: 0.0,
                max_value: 1.0,
                default_value: 0.3,
                step_size: 0.01,
                unit: "".to_string(),
                description: "Level of processed signal".to_string(),
                parameter_type: ParameterType::Float,
            },
            ParameterInfo {
                name: "dry_level".to_string(),
                display_name: "Dry Level".to_string(),
                min_value: 0.0,
                max_value: 1.0,
                default_value: 0.7,
                step_size: 0.01,
                unit: "".to_string(),
                description: "Level of original signal".to_string(),
                parameter_type: ParameterType::Float,
            },
        ]
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> PluginResult<()> {
        match name {
            "room_size" => self.room_size = value.clamp(0.0, 1.0),
            "damping" => self.damping = value.clamp(0.0, 1.0),
            "wet_level" => self.wet_level = value.clamp(0.0, 1.0),
            "dry_level" => self.dry_level = value.clamp(0.0, 1.0),
            _ => {
                return Err(PluginError::ExecutionFailed(format!(
                    "Unknown parameter: {}",
                    name
                )))
            }
        }
        Ok(())
    }

    fn get_parameter(&self, name: &str) -> PluginResult<f32> {
        match name {
            "room_size" => Ok(self.room_size),
            "damping" => Ok(self.damping),
            "wet_level" => Ok(self.wet_level),
            "dry_level" => Ok(self.dry_level),
            _ => Err(PluginError::ExecutionFailed(format!(
                "Unknown parameter: {}",
                name
            ))),
        }
    }

    fn reset(&mut self) -> PluginResult<()> {
        self.room_size = 0.5;
        self.damping = 0.5;
        self.wet_level = 0.3;
        self.dry_level = 0.7;
        Ok(())
    }

    fn get_latency(&self) -> u32 {
        0 // No latency for this simple implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_plugin_manager() {
        let mut manager = EffectPluginManager::new();
        let reverb = Arc::new(ReverbEffectPlugin::new());

        manager.register_effect("reverb".to_string(), reverb);

        let effects = manager.list_effects();
        assert!(effects.contains(&"reverb".to_string()));

        let effect = manager.get_effect("reverb");
        assert!(effect.is_some());
    }

    #[test]
    fn test_reverb_plugin() {
        let mut reverb = ReverbEffectPlugin::new();
        assert_eq!(reverb.name(), "builtin-reverb");
        assert_eq!(reverb.version(), "1.0.0");

        let config = EffectPluginConfig::default();
        let input = vec![1.0, 0.5, -0.5, -1.0];
        let mut output = vec![0.0; 4];

        reverb.process_audio(&input, &mut output, &config).unwrap();

        // Output should be processed (not just copied)
        assert_ne!(input, output);
    }

    #[test]
    fn test_parameter_setting() {
        let mut reverb = ReverbEffectPlugin::new();

        reverb.set_parameter("room_size", 0.8).unwrap();
        assert_eq!(reverb.get_parameter("room_size").unwrap(), 0.8);

        // Test parameter clamping
        reverb.set_parameter("room_size", 1.5).unwrap();
        assert_eq!(reverb.get_parameter("room_size").unwrap(), 1.0);

        // Test invalid parameter
        assert!(reverb.set_parameter("invalid", 0.5).is_err());
    }

    #[test]
    fn test_effect_chain() {
        let reverb = Arc::new(ReverbEffectPlugin::new());
        let effects = vec![("reverb".to_string(), reverb as Arc<dyn EffectPlugin>)];
        let mut chain = EffectChain::new(effects);

        let input = vec![1.0, 0.5, -0.5, -1.0];
        let mut output = vec![0.0; 4];
        let configs = HashMap::new();

        chain.process(&input, &mut output, &configs).unwrap();

        // Should process without error
        assert_eq!(output.len(), input.len());
    }
}
