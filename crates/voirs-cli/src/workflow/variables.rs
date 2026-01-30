//! Variable Resolution and Substitution
//!
//! Handles variable resolution, substitution, and scoping for workflows.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Variable scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableScope {
    /// Global workflow variables
    Global,
    /// Step-level variables
    Step(String),
    /// Loop iteration variables
    Loop { step: String, iteration: usize },
}

/// Variable resolver
pub struct VariableResolver {
    /// Global variables
    global: HashMap<String, serde_json::Value>,
    /// Step-scoped variables
    step_scoped: HashMap<String, HashMap<String, serde_json::Value>>,
}

impl VariableResolver {
    /// Create new resolver
    pub fn new(global: HashMap<String, serde_json::Value>) -> Self {
        Self {
            global,
            step_scoped: HashMap::new(),
        }
    }

    /// Set a global variable
    pub fn set_global(&mut self, name: String, value: serde_json::Value) {
        self.global.insert(name, value);
    }

    /// Set a step-scoped variable
    pub fn set_step_variable(&mut self, step: String, name: String, value: serde_json::Value) {
        self.step_scoped
            .entry(step)
            .or_default()
            .insert(name, value);
    }

    /// Get a variable (checks step scope first, then global)
    pub fn get(&self, name: &str, current_step: Option<&str>) -> Option<serde_json::Value> {
        // Check step scope first
        if let Some(step_name) = current_step {
            if let Some(step_vars) = self.step_scoped.get(step_name) {
                if let Some(value) = step_vars.get(name) {
                    return Some(value.clone());
                }
            }
        }

        // Check global scope
        self.global.get(name).cloned()
    }

    /// Resolve a string with variable substitution
    pub fn resolve_string(&self, input: &str, current_step: Option<&str>) -> String {
        let mut result = input.to_string();

        // Find all ${var} patterns
        while let Some(start) = result.find("${") {
            if let Some(end) = result[start..].find('}') {
                let var_name = &result[start + 2..start + end];
                let replacement = self
                    .get(var_name, current_step)
                    .and_then(|v| match v {
                        serde_json::Value::String(s) => Some(s),
                        serde_json::Value::Number(n) => Some(n.to_string()),
                        serde_json::Value::Bool(b) => Some(b.to_string()),
                        _ => None,
                    })
                    .unwrap_or_else(|| format!("${{{}}}", var_name));

                result.replace_range(start..start + end + 1, &replacement);
            } else {
                break;
            }
        }

        result
    }

    /// Resolve all variables in a parameters map
    pub fn resolve_parameters(
        &self,
        params: &HashMap<String, serde_json::Value>,
        current_step: Option<&str>,
    ) -> HashMap<String, serde_json::Value> {
        let mut resolved = HashMap::new();

        for (key, value) in params {
            let resolved_value = self.resolve_value(value, current_step);
            resolved.insert(key.clone(), resolved_value);
        }

        resolved
    }

    /// Resolve a single value
    fn resolve_value(
        &self,
        value: &serde_json::Value,
        current_step: Option<&str>,
    ) -> serde_json::Value {
        match value {
            serde_json::Value::String(s) => {
                // Check if it's a variable reference
                if let Some(var_name) = s.strip_prefix("${").and_then(|s| s.strip_suffix('}')) {
                    self.get(var_name, current_step)
                        .unwrap_or(serde_json::Value::Null)
                } else {
                    // Resolve inline substitutions
                    serde_json::Value::String(self.resolve_string(s, current_step))
                }
            }
            serde_json::Value::Array(arr) => serde_json::Value::Array(
                arr.iter()
                    .map(|v| self.resolve_value(v, current_step))
                    .collect(),
            ),
            serde_json::Value::Object(obj) => serde_json::Value::Object(
                obj.iter()
                    .map(|(k, v)| (k.clone(), self.resolve_value(v, current_step)))
                    .collect(),
            ),
            _ => value.clone(),
        }
    }

    /// Get all variables in current scope
    pub fn get_all(&self, current_step: Option<&str>) -> HashMap<String, serde_json::Value> {
        let mut all_vars = self.global.clone();

        // Merge step-scoped variables
        if let Some(step_name) = current_step {
            if let Some(step_vars) = self.step_scoped.get(step_name) {
                all_vars.extend(step_vars.clone());
            }
        }

        all_vars
    }

    /// Clear step-scoped variables for a step
    pub fn clear_step_scope(&mut self, step: &str) {
        self.step_scoped.remove(step);
    }

    /// Check if a variable exists
    pub fn has_variable(&self, name: &str, current_step: Option<&str>) -> bool {
        self.get(name, current_step).is_some()
    }

    /// List all variable names
    pub fn list_variable_names(&self, current_step: Option<&str>) -> Vec<String> {
        let mut names: Vec<String> = self.global.keys().cloned().collect();

        if let Some(step_name) = current_step {
            if let Some(step_vars) = self.step_scoped.get(step_name) {
                names.extend(step_vars.keys().cloned());
            }
        }

        names.sort();
        names.dedup();
        names
    }
}

impl Default for VariableResolver {
    fn default() -> Self {
        Self::new(HashMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_resolver_creation() {
        let resolver = VariableResolver::new(HashMap::new());
        assert_eq!(resolver.global.len(), 0);
    }

    #[test]
    fn test_set_and_get_global() {
        let mut resolver = VariableResolver::new(HashMap::new());
        resolver.set_global("key1".to_string(), serde_json::json!("value1"));

        let value = resolver.get("key1", None);
        assert!(value.is_some());
        assert_eq!(value.unwrap().as_str().unwrap(), "value1");
    }

    #[test]
    fn test_set_and_get_step_variable() {
        let mut resolver = VariableResolver::new(HashMap::new());
        resolver.set_step_variable(
            "step1".to_string(),
            "key1".to_string(),
            serde_json::json!("value1"),
        );

        let value = resolver.get("key1", Some("step1"));
        assert!(value.is_some());
        assert_eq!(value.unwrap().as_str().unwrap(), "value1");

        let value2 = resolver.get("key1", Some("step2"));
        assert!(value2.is_none());
    }

    #[test]
    fn test_step_scope_overrides_global() {
        let mut global = HashMap::new();
        global.insert("key1".to_string(), serde_json::json!("global_value"));

        let mut resolver = VariableResolver::new(global);
        resolver.set_step_variable(
            "step1".to_string(),
            "key1".to_string(),
            serde_json::json!("step_value"),
        );

        // Step scope should override global
        let value = resolver.get("key1", Some("step1"));
        assert_eq!(value.unwrap().as_str().unwrap(), "step_value");

        // Without step context, should get global
        let value2 = resolver.get("key1", None);
        assert_eq!(value2.unwrap().as_str().unwrap(), "global_value");
    }

    #[test]
    fn test_resolve_string_simple() {
        let mut resolver = VariableResolver::new(HashMap::new());
        resolver.set_global("name".to_string(), serde_json::json!("World"));

        let result = resolver.resolve_string("Hello, ${name}!", None);
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_resolve_string_multiple_variables() {
        let mut resolver = VariableResolver::new(HashMap::new());
        resolver.set_global("first".to_string(), serde_json::json!("John"));
        resolver.set_global("last".to_string(), serde_json::json!("Doe"));

        let result = resolver.resolve_string("Name: ${first} ${last}", None);
        assert_eq!(result, "Name: John Doe");
    }

    #[test]
    fn test_resolve_string_missing_variable() {
        let resolver = VariableResolver::new(HashMap::new());
        let result = resolver.resolve_string("Hello, ${missing}!", None);
        assert_eq!(result, "Hello, ${missing}!");
    }

    #[test]
    fn test_resolve_parameters() {
        let mut resolver = VariableResolver::new(HashMap::new());
        resolver.set_global("voice".to_string(), serde_json::json!("en-US-neural"));

        let mut params = HashMap::new();
        params.insert("voice".to_string(), serde_json::json!("${voice}"));
        params.insert("text".to_string(), serde_json::json!("Hello"));

        let resolved = resolver.resolve_parameters(&params, None);

        assert_eq!(
            resolved.get("voice").unwrap().as_str().unwrap(),
            "en-US-neural"
        );
        assert_eq!(resolved.get("text").unwrap().as_str().unwrap(), "Hello");
    }

    #[test]
    fn test_has_variable() {
        let mut resolver = VariableResolver::new(HashMap::new());
        resolver.set_global("exists".to_string(), serde_json::json!("yes"));

        assert!(resolver.has_variable("exists", None));
        assert!(!resolver.has_variable("missing", None));
    }

    #[test]
    fn test_list_variable_names() {
        let mut resolver = VariableResolver::new(HashMap::new());
        resolver.set_global("var1".to_string(), serde_json::json!("value1"));
        resolver.set_global("var2".to_string(), serde_json::json!("value2"));
        resolver.set_step_variable(
            "step1".to_string(),
            "var3".to_string(),
            serde_json::json!("value3"),
        );

        let names = resolver.list_variable_names(Some("step1"));
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"var1".to_string()));
        assert!(names.contains(&"var2".to_string()));
        assert!(names.contains(&"var3".to_string()));
    }

    #[test]
    fn test_clear_step_scope() {
        let mut resolver = VariableResolver::new(HashMap::new());
        resolver.set_step_variable(
            "step1".to_string(),
            "key1".to_string(),
            serde_json::json!("value1"),
        );

        assert!(resolver.get("key1", Some("step1")).is_some());

        resolver.clear_step_scope("step1");

        assert!(resolver.get("key1", Some("step1")).is_none());
    }
}
