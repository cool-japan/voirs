use crate::types::VoirsResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigValidationRules {
    pub parameter_rules: HashMap<String, ParameterRule>,
    pub compatibility_rules: Vec<CompatibilityRule>,
    pub resource_requirements: Vec<ResourceRequirement>,
    pub constraints: Vec<Constraint>,
    pub validation_mode: ValidationMode,
}

impl Default for ConfigValidationRules {
    fn default() -> Self {
        Self {
            parameter_rules: HashMap::new(),
            compatibility_rules: Vec::new(),
            resource_requirements: Vec::new(),
            constraints: Vec::new(),
            validation_mode: ValidationMode::Strict,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMode {
    Strict,      // All validations must pass
    Lenient,     // Warnings are allowed
    WarningOnly, // Only emit warnings, never fail
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRule {
    pub name: String,
    pub parameter_type: ParameterType,
    pub required: bool,
    pub default_value: Option<ConfigValue>,
    pub validation: ParameterValidation,
    pub description: String,
    pub deprecated: bool,
    pub deprecation_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
    Path,
    Enum(Vec<String>),
    Duration,
    MemorySize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
    Null,
}

impl ConfigValue {
    pub fn as_string(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        if let Self::Integer(i) = self {
            Some(*i)
        } else {
            None
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        if let Self::Float(f) = self {
            Some(*f)
        } else {
            None
        }
    }

    pub fn as_boolean(&self) -> Option<bool> {
        if let Self::Boolean(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    pub fn as_array(&self) -> Option<&Vec<ConfigValue>> {
        if let Self::Array(arr) = self {
            Some(arr)
        } else {
            None
        }
    }

    pub fn as_object(&self) -> Option<&HashMap<String, ConfigValue>> {
        if let Self::Object(obj) = self {
            Some(obj)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParameterValidation {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub allowed_values: Option<Vec<ConfigValue>>,
    pub custom_validator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityRule {
    pub name: String,
    pub condition: CompatibilityCondition,
    pub action: CompatibilityAction,
    pub severity: ValidationSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityCondition {
    ParameterEquals {
        parameter: String,
        value: ConfigValue,
    },
    ParameterExists {
        parameter: String,
    },
    ParameterRange {
        parameter: String,
        min: f64,
        max: f64,
    },
    And(Vec<CompatibilityCondition>),
    Or(Vec<CompatibilityCondition>),
    Not(Box<CompatibilityCondition>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityAction {
    Require {
        parameters: Vec<String>,
    },
    Forbid {
        parameters: Vec<String>,
    },
    SetDefault {
        parameter: String,
        value: ConfigValue,
    },
    Recommend {
        parameter: String,
        value: ConfigValue,
    },
    Warning {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub name: String,
    pub resource_type: ResourceType,
    pub minimum: f64,
    pub recommended: f64,
    pub unit: String,
    pub check_availability: bool,
    pub severity: ValidationSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    Memory,
    DiskSpace,
    CpuCores,
    GpuMemory,
    NetworkBandwidth,
    FileHandles,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub severity: ValidationSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    MutuallyExclusive {
        parameters: Vec<String>,
    },
    RequireAny {
        parameters: Vec<String>,
    },
    RequireAll {
        parameters: Vec<String>,
    },
    DependsOn {
        parameter: String,
        depends_on: Vec<String>,
    },
    Custom {
        expression: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ConfigValidationResult {
    pub is_valid: bool,
    pub validated_config: HashMap<String, ConfigValue>,
    pub errors: Vec<ConfigValidationError>,
    pub warnings: Vec<ConfigValidationWarning>,
    pub resource_checks: Vec<ResourceCheckResult>,
    pub applied_defaults: Vec<DefaultApplication>,
    pub compatibility_issues: Vec<CompatibilityIssue>,
}

#[derive(Debug, Clone)]
pub struct ConfigValidationError {
    pub parameter: String,
    pub error_type: ConfigErrorType,
    pub message: String,
    pub severity: ValidationSeverity,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ConfigErrorType {
    Missing,
    InvalidType,
    OutOfRange,
    InvalidValue,
    PatternMismatch,
    Deprecated,
    Incompatible,
    ResourceUnavailable,
    ConstraintViolation,
}

#[derive(Debug, Clone)]
pub struct ConfigValidationWarning {
    pub parameter: String,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResourceCheckResult {
    pub resource: String,
    pub required: f64,
    pub available: f64,
    pub unit: String,
    pub status: ResourceStatus,
}

#[derive(Debug, Clone)]
pub enum ResourceStatus {
    Sufficient,
    Insufficient,
    BelowRecommended,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct DefaultApplication {
    pub parameter: String,
    pub value: ConfigValue,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct CompatibilityIssue {
    pub rule_name: String,
    pub message: String,
    pub severity: ValidationSeverity,
    pub affected_parameters: Vec<String>,
}

/// Type alias for custom validator functions
type CustomValidator = Box<dyn Fn(&ConfigValue) -> bool + Send + Sync>;

pub struct ConfigValidator {
    rules: ConfigValidationRules,
    custom_validators: HashMap<String, CustomValidator>,
}

impl ConfigValidator {
    pub fn new(rules: ConfigValidationRules) -> Self {
        Self {
            rules,
            custom_validators: HashMap::new(),
        }
    }

    pub fn validate_config(
        &self,
        config: &HashMap<String, ConfigValue>,
    ) -> VoirsResult<ConfigValidationResult> {
        let mut result = ConfigValidationResult {
            is_valid: true,
            validated_config: config.clone(),
            errors: Vec::new(),
            warnings: Vec::new(),
            resource_checks: Vec::new(),
            applied_defaults: Vec::new(),
            compatibility_issues: Vec::new(),
        };

        // Apply defaults first
        self.apply_defaults(&mut result);

        // Validate individual parameters
        self.validate_parameters(&mut result);

        // Check resource requirements
        self.check_resource_requirements(&mut result);

        // Validate compatibility rules
        self.validate_compatibility(&mut result);

        // Check constraints
        self.validate_constraints(&mut result);

        // Determine overall validity
        result.is_valid = self.determine_validity(&result);

        Ok(result)
    }

    fn apply_defaults(&self, result: &mut ConfigValidationResult) {
        for (param_name, rule) in &self.rules.parameter_rules {
            if !result.validated_config.contains_key(param_name) {
                if let Some(ref default_value) = rule.default_value {
                    result
                        .validated_config
                        .insert(param_name.clone(), default_value.clone());
                    result.applied_defaults.push(DefaultApplication {
                        parameter: param_name.clone(),
                        value: default_value.clone(),
                        reason: "Missing parameter, applied default".to_string(),
                    });
                } else if rule.required {
                    result.errors.push(ConfigValidationError {
                        parameter: param_name.clone(),
                        error_type: ConfigErrorType::Missing,
                        message: format!("Required parameter '{param_name}' is missing"),
                        severity: ValidationSeverity::Error,
                        suggestion: Some(
                            "Add the required parameter to your configuration".to_string(),
                        ),
                    });
                }
            }
        }
    }

    fn validate_parameters(&self, result: &mut ConfigValidationResult) {
        for (param_name, value) in &result.validated_config.clone() {
            if let Some(rule) = self.rules.parameter_rules.get(param_name) {
                self.validate_parameter(param_name, value, rule, result);
            } else {
                result.warnings.push(ConfigValidationWarning {
                    parameter: param_name.clone(),
                    message: format!("Unknown parameter '{param_name}' in configuration"),
                    suggestion: Some("Remove unknown parameter or check spelling".to_string()),
                });
            }
        }
    }

    fn validate_parameter(
        &self,
        name: &str,
        value: &ConfigValue,
        rule: &ParameterRule,
        result: &mut ConfigValidationResult,
    ) {
        // Check deprecation
        if rule.deprecated {
            let message = rule
                .deprecation_message
                .clone()
                .unwrap_or_else(|| format!("Parameter '{name}' is deprecated"));

            result.warnings.push(ConfigValidationWarning {
                parameter: name.to_string(),
                message,
                suggestion: Some("Consider migrating to the new parameter".to_string()),
            });
        }

        // Type validation
        if !self.validate_type(value, &rule.parameter_type) {
            result.errors.push(ConfigValidationError {
                parameter: name.to_string(),
                error_type: ConfigErrorType::InvalidType,
                message: format!("Parameter '{name}' has invalid type"),
                severity: ValidationSeverity::Error,
                suggestion: Some(format!("Expected type: {:?}", rule.parameter_type)),
            });
            return;
        }

        // Value validation
        self.validate_parameter_value(name, value, &rule.validation, result);

        // Custom validation
        if let Some(ref validator_name) = rule.validation.custom_validator {
            if let Some(validator) = self.custom_validators.get(validator_name) {
                if !validator(value) {
                    result.errors.push(ConfigValidationError {
                        parameter: name.to_string(),
                        error_type: ConfigErrorType::InvalidValue,
                        message: format!("Parameter '{name}' failed custom validation"),
                        severity: ValidationSeverity::Error,
                        suggestion: None,
                    });
                }
            }
        }
    }

    fn validate_type(&self, value: &ConfigValue, expected_type: &ParameterType) -> bool {
        match (value, expected_type) {
            (ConfigValue::String(_), ParameterType::String) => true,
            (ConfigValue::Integer(_), ParameterType::Integer) => true,
            (ConfigValue::Float(_), ParameterType::Float) => true,
            (ConfigValue::Boolean(_), ParameterType::Boolean) => true,
            (ConfigValue::Array(_), ParameterType::Array) => true,
            (ConfigValue::Object(_), ParameterType::Object) => true,
            (ConfigValue::String(s), ParameterType::Path) => Path::new(s).exists() || !s.is_empty(),
            (ConfigValue::String(s), ParameterType::Enum(values)) => values.contains(s),
            (ConfigValue::String(s), ParameterType::Duration) => self.validate_duration_format(s),
            (ConfigValue::String(s), ParameterType::MemorySize) => {
                self.validate_memory_size_format(s)
            }
            _ => false,
        }
    }

    fn validate_parameter_value(
        &self,
        name: &str,
        value: &ConfigValue,
        validation: &ParameterValidation,
        result: &mut ConfigValidationResult,
    ) {
        // Range validation
        if let Some(min) = validation.min_value {
            let val = match value {
                ConfigValue::Integer(i) => Some(*i as f64),
                ConfigValue::Float(f) => Some(*f),
                _ => None,
            };

            if let Some(v) = val {
                if v < min {
                    result.errors.push(ConfigValidationError {
                        parameter: name.to_string(),
                        error_type: ConfigErrorType::OutOfRange,
                        message: format!("Parameter '{name}' value {v} is below minimum {min}"),
                        severity: ValidationSeverity::Error,
                        suggestion: Some(format!("Use a value >= {min}")),
                    });
                }
            }
        }

        if let Some(max) = validation.max_value {
            let val = match value {
                ConfigValue::Integer(i) => Some(*i as f64),
                ConfigValue::Float(f) => Some(*f),
                _ => None,
            };

            if let Some(v) = val {
                if v > max {
                    result.errors.push(ConfigValidationError {
                        parameter: name.to_string(),
                        error_type: ConfigErrorType::OutOfRange,
                        message: format!("Parameter '{name}' value {v} is above maximum {max}"),
                        severity: ValidationSeverity::Error,
                        suggestion: Some(format!("Use a value <= {max}")),
                    });
                }
            }
        }

        // Length validation
        if let Some(min_len) = validation.min_length {
            let length = match value {
                ConfigValue::String(s) => Some(s.len()),
                ConfigValue::Array(arr) => Some(arr.len()),
                _ => None,
            };

            if let Some(len) = length {
                if len < min_len {
                    result.errors.push(ConfigValidationError {
                        parameter: name.to_string(),
                        error_type: ConfigErrorType::OutOfRange,
                        message: format!(
                            "Parameter '{name}' length {len} is below minimum {min_len}"
                        ),
                        severity: ValidationSeverity::Error,
                        suggestion: Some(format!("Use a length >= {min_len}")),
                    });
                }
            }
        }

        if let Some(max_len) = validation.max_length {
            let length = match value {
                ConfigValue::String(s) => Some(s.len()),
                ConfigValue::Array(arr) => Some(arr.len()),
                _ => None,
            };

            if let Some(len) = length {
                if len > max_len {
                    result.errors.push(ConfigValidationError {
                        parameter: name.to_string(),
                        error_type: ConfigErrorType::OutOfRange,
                        message: format!(
                            "Parameter '{name}' length {len} is above maximum {max_len}"
                        ),
                        severity: ValidationSeverity::Error,
                        suggestion: Some(format!("Use a length <= {max_len}")),
                    });
                }
            }
        }

        // Pattern validation
        if let Some(ref pattern) = validation.pattern {
            if let ConfigValue::String(s) = value {
                if let Ok(regex) = regex::Regex::new(pattern) {
                    if !regex.is_match(s) {
                        result.errors.push(ConfigValidationError {
                            parameter: name.to_string(),
                            error_type: ConfigErrorType::PatternMismatch,
                            message: format!(
                                "Parameter '{name}' value '{s}' does not match pattern '{pattern}'"
                            ),
                            severity: ValidationSeverity::Error,
                            suggestion: Some(format!("Use a value matching pattern: {pattern}")),
                        });
                    }
                }
            }
        }

        // Allowed values validation
        if let Some(ref allowed) = validation.allowed_values {
            if !allowed.contains(value) {
                result.errors.push(ConfigValidationError {
                    parameter: name.to_string(),
                    error_type: ConfigErrorType::InvalidValue,
                    message: format!("Parameter '{name}' has invalid value"),
                    severity: ValidationSeverity::Error,
                    suggestion: Some(format!("Allowed values: {allowed:?}")),
                });
            }
        }
    }

    fn check_resource_requirements(&self, result: &mut ConfigValidationResult) {
        for requirement in &self.rules.resource_requirements {
            if requirement.check_availability {
                let available = self.get_available_resource(&requirement.resource_type);

                let status = if available >= requirement.recommended {
                    ResourceStatus::Sufficient
                } else if available >= requirement.minimum {
                    ResourceStatus::BelowRecommended
                } else {
                    ResourceStatus::Insufficient
                };

                result.resource_checks.push(ResourceCheckResult {
                    resource: requirement.name.clone(),
                    required: requirement.minimum,
                    available,
                    unit: requirement.unit.clone(),
                    status: status.clone(),
                });

                match status {
                    ResourceStatus::Insufficient => {
                        result.errors.push(ConfigValidationError {
                            parameter: requirement.name.clone(),
                            error_type: ConfigErrorType::ResourceUnavailable,
                            message: format!(
                                "Insufficient {}: {} {} available, {} {} required",
                                requirement.name,
                                available,
                                requirement.unit,
                                requirement.minimum,
                                requirement.unit
                            ),
                            severity: requirement.severity,
                            suggestion: Some(
                                "Increase available resources or reduce requirements".to_string(),
                            ),
                        });
                    }
                    ResourceStatus::BelowRecommended => {
                        result.warnings.push(ConfigValidationWarning {
                            parameter: requirement.name.clone(),
                            message: format!(
                                "{} below recommended: {} {} available, {} {} recommended",
                                requirement.name,
                                available,
                                requirement.unit,
                                requirement.recommended,
                                requirement.unit
                            ),
                            suggestion: Some(
                                "Consider increasing available resources for optimal performance"
                                    .to_string(),
                            ),
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    fn validate_compatibility(&self, result: &mut ConfigValidationResult) {
        for rule in &self.rules.compatibility_rules {
            if Self::evaluate_compatibility_condition(&rule.condition, &result.validated_config) {
                self.apply_compatibility_action(&rule.action, rule, result);
            }
        }
    }

    fn validate_constraints(&self, result: &mut ConfigValidationResult) {
        for constraint in &self.rules.constraints {
            let violation = match &constraint.constraint_type {
                ConstraintType::MutuallyExclusive { parameters } => {
                    let present_params: Vec<_> = parameters
                        .iter()
                        .filter(|p| result.validated_config.contains_key(*p))
                        .collect();
                    present_params.len() > 1
                }
                ConstraintType::RequireAny { parameters } => !parameters
                    .iter()
                    .any(|p| result.validated_config.contains_key(p)),
                ConstraintType::RequireAll { parameters } => !parameters
                    .iter()
                    .all(|p| result.validated_config.contains_key(p)),
                ConstraintType::DependsOn {
                    parameter,
                    depends_on,
                } => {
                    result.validated_config.contains_key(parameter)
                        && !depends_on
                            .iter()
                            .all(|p| result.validated_config.contains_key(p))
                }
                ConstraintType::Custom { expression } => {
                    self.evaluate_custom_expression(expression, &result.validated_config)
                }
            };

            if violation {
                match constraint.severity {
                    ValidationSeverity::Error | ValidationSeverity::Critical => {
                        result.errors.push(ConfigValidationError {
                            parameter: constraint.name.clone(),
                            error_type: ConfigErrorType::ConstraintViolation,
                            message: constraint.message.clone(),
                            severity: constraint.severity,
                            suggestion: None,
                        });
                    }
                    _ => {
                        result.warnings.push(ConfigValidationWarning {
                            parameter: constraint.name.clone(),
                            message: constraint.message.clone(),
                            suggestion: None,
                        });
                    }
                }
            }
        }
    }

    fn evaluate_compatibility_condition(
        condition: &CompatibilityCondition,
        config: &HashMap<String, ConfigValue>,
    ) -> bool {
        match condition {
            CompatibilityCondition::ParameterEquals { parameter, value } => {
                config.get(parameter) == Some(value)
            }
            CompatibilityCondition::ParameterExists { parameter } => config.contains_key(parameter),
            CompatibilityCondition::ParameterRange {
                parameter,
                min,
                max,
            } => {
                if let Some(param_value) = config.get(parameter) {
                    let value = match param_value {
                        ConfigValue::Integer(i) => Some(*i as f64),
                        ConfigValue::Float(f) => Some(*f),
                        _ => None,
                    };
                    value.is_some_and(|v| v >= *min && v <= *max)
                } else {
                    false
                }
            }
            CompatibilityCondition::And(conditions) => conditions
                .iter()
                .all(|c| Self::evaluate_compatibility_condition(c, config)),
            CompatibilityCondition::Or(conditions) => conditions
                .iter()
                .any(|c| Self::evaluate_compatibility_condition(c, config)),
            CompatibilityCondition::Not(condition) => {
                !Self::evaluate_compatibility_condition(condition, config)
            }
        }
    }

    fn apply_compatibility_action(
        &self,
        action: &CompatibilityAction,
        rule: &CompatibilityRule,
        result: &mut ConfigValidationResult,
    ) {
        match action {
            CompatibilityAction::Require { parameters } => {
                for param in parameters {
                    if !result.validated_config.contains_key(param) {
                        result.errors.push(ConfigValidationError {
                            parameter: param.clone(),
                            error_type: ConfigErrorType::Incompatible,
                            message: format!(
                                "Parameter '{}' is required by compatibility rule '{}'",
                                param, rule.name
                            ),
                            severity: rule.severity,
                            suggestion: Some(format!(
                                "Add parameter '{param}' to satisfy compatibility requirements"
                            )),
                        });
                    }
                }
            }
            CompatibilityAction::Forbid { parameters } => {
                for param in parameters {
                    if result.validated_config.contains_key(param) {
                        result.errors.push(ConfigValidationError {
                            parameter: param.clone(),
                            error_type: ConfigErrorType::Incompatible,
                            message: format!(
                                "Parameter '{}' is forbidden by compatibility rule '{}'",
                                param, rule.name
                            ),
                            severity: rule.severity,
                            suggestion: Some(format!(
                                "Remove parameter '{param}' to satisfy compatibility requirements"
                            )),
                        });
                    }
                }
            }
            CompatibilityAction::SetDefault { parameter, value } => {
                if !result.validated_config.contains_key(parameter) {
                    result
                        .validated_config
                        .insert(parameter.clone(), value.clone());
                    result.applied_defaults.push(DefaultApplication {
                        parameter: parameter.clone(),
                        value: value.clone(),
                        reason: format!("Applied by compatibility rule '{}'", rule.name),
                    });
                }
            }
            CompatibilityAction::Recommend { parameter, value } => {
                if !result.validated_config.contains_key(parameter) {
                    result.warnings.push(ConfigValidationWarning {
                        parameter: parameter.clone(),
                        message: format!(
                            "Recommended to set '{parameter}' to '{value:?}' for compatibility"
                        ),
                        suggestion: Some(format!(
                            "Add '{parameter}' with value '{value:?}' for optimal compatibility"
                        )),
                    });
                }
            }
            CompatibilityAction::Warning { message } => {
                result.warnings.push(ConfigValidationWarning {
                    parameter: rule.name.clone(),
                    message: message.clone(),
                    suggestion: None,
                });
            }
        }
    }

    fn get_available_resource(&self, resource_type: &ResourceType) -> f64 {
        // Simplified resource checking - in production this would query actual system resources
        match resource_type {
            ResourceType::Memory => {
                // Return available memory in MB
                8192.0 // 8GB default
            }
            ResourceType::DiskSpace => {
                // Return available disk space in MB
                102400.0 // 100GB default
            }
            ResourceType::CpuCores => {
                // Return number of CPU cores
                8.0
            }
            ResourceType::GpuMemory => {
                // Return GPU memory in MB
                4096.0 // 4GB default
            }
            ResourceType::NetworkBandwidth => {
                // Return network bandwidth in Mbps
                1000.0 // 1Gbps default
            }
            ResourceType::FileHandles => {
                // Return available file handles
                1024.0
            }
            ResourceType::Custom(_) => {
                // For custom resources, return a default value
                1.0
            }
        }
    }

    fn determine_validity(&self, result: &ConfigValidationResult) -> bool {
        match self.rules.validation_mode {
            ValidationMode::Strict => result.errors.is_empty(),
            ValidationMode::Lenient => !result.errors.iter().any(|e| {
                matches!(
                    e.severity,
                    ValidationSeverity::Critical | ValidationSeverity::Error
                )
            }),
            ValidationMode::WarningOnly => true,
        }
    }

    pub fn add_custom_validator<F>(&mut self, name: String, validator: F)
    where
        F: Fn(&ConfigValue) -> bool + Send + Sync + 'static,
    {
        self.custom_validators.insert(name, Box::new(validator));
    }

    /// Validate duration format (e.g., "1s", "500ms", "1m", "1h")
    fn validate_duration_format(&self, duration_str: &str) -> bool {
        if duration_str.is_empty() {
            return false;
        }

        let duration_str = duration_str.trim();

        // Check for valid duration patterns
        let valid_patterns = [
            // Nanoseconds
            r"^\d+ns$",
            // Microseconds
            r"^\d+us$",
            // Milliseconds
            r"^\d+ms$",
            // Seconds
            r"^\d+s$",
            // Minutes
            r"^\d+m$",
            // Hours
            r"^\d+h$",
            // Days
            r"^\d+d$",
            // Composite patterns (e.g., "1h30m", "2m30s")
            r"^\d+h\d+m$",
            r"^\d+h\d+s$",
            r"^\d+m\d+s$",
            r"^\d+h\d+m\d+s$",
        ];

        for pattern in &valid_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if regex.is_match(duration_str) {
                    return true;
                }
            }
        }

        // Try to parse as a simple number (assuming seconds)
        duration_str.parse::<f64>().is_ok()
    }

    /// Validate memory size format (e.g., "1KB", "512MB", "2GB", "1TB")
    fn validate_memory_size_format(&self, memory_str: &str) -> bool {
        if memory_str.is_empty() {
            return false;
        }

        let memory_str = memory_str.trim().to_uppercase();

        // Check for valid memory size patterns
        let valid_patterns = [
            // Bytes
            r"^\d+B$",
            // Kilobytes
            r"^\d+KB$",
            // Megabytes
            r"^\d+MB$",
            // Gigabytes
            r"^\d+GB$",
            // Terabytes
            r"^\d+TB$",
            // Petabytes
            r"^\d+PB$",
            // Binary units (powers of 1024)
            r"^\d+KIB$",
            r"^\d+MIB$",
            r"^\d+GIB$",
            r"^\d+TIB$",
            r"^\d+PIB$",
            // Decimal numbers with units
            r"^\d+\.\d+KB$",
            r"^\d+\.\d+MB$",
            r"^\d+\.\d+GB$",
            r"^\d+\.\d+TB$",
            r"^\d+\.\d+PB$",
            r"^\d+\.\d+KIB$",
            r"^\d+\.\d+MIB$",
            r"^\d+\.\d+GIB$",
            r"^\d+\.\d+TIB$",
            r"^\d+\.\d+PIB$",
        ];

        for pattern in &valid_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if regex.is_match(&memory_str) {
                    return true;
                }
            }
        }

        // Try to parse as a simple number (assuming bytes)
        memory_str.parse::<u64>().is_ok()
    }

    /// Evaluate custom expression for constraint validation
    /// This is a simplified expression evaluator that supports basic operations
    fn evaluate_custom_expression(
        &self,
        expression: &str,
        config: &HashMap<String, ConfigValue>,
    ) -> bool {
        let expression = expression.trim();

        // Handle simple comparison expressions
        if let Some(result) = self.evaluate_comparison_expression(expression, config) {
            return result;
        }

        // Handle logical expressions
        if let Some(result) = self.evaluate_logical_expression(expression, config) {
            return result;
        }

        // Handle existence checks
        if let Some(result) = self.evaluate_existence_expression(expression, config) {
            return result;
        }

        // If we can't parse the expression, return false for safety
        false
    }

    /// Evaluate comparison expressions like "param > 10", "param == 'value'"
    fn evaluate_comparison_expression(
        &self,
        expression: &str,
        config: &HashMap<String, ConfigValue>,
    ) -> Option<bool> {
        let operators = [">=", "<=", "!=", "==", ">", "<"];

        for op in &operators {
            if let Some(pos) = expression.find(op) {
                let left_part = expression[..pos].trim();
                let right_part = expression[pos + op.len()..].trim();

                // Get the value from config
                let left_value = config.get(left_part)?;

                // Parse the right side
                let right_value = self.parse_value_from_string(right_part)?;

                return Some(self.compare_values(left_value, &right_value, op));
            }
        }

        None
    }

    /// Evaluate logical expressions like "param1 && param2", "param1 || param2"
    fn evaluate_logical_expression(
        &self,
        expression: &str,
        config: &HashMap<String, ConfigValue>,
    ) -> Option<bool> {
        // Handle AND expressions
        if expression.contains("&&") {
            let parts: Vec<&str> = expression.split("&&").collect();
            for part in parts {
                let part = part.trim();
                if !self.evaluate_single_expression(part, config) {
                    return Some(false);
                }
            }
            return Some(true);
        }

        // Handle OR expressions
        if expression.contains("||") {
            let parts: Vec<&str> = expression.split("||").collect();
            for part in parts {
                let part = part.trim();
                if self.evaluate_single_expression(part, config) {
                    return Some(true);
                }
            }
            return Some(false);
        }

        None
    }

    /// Evaluate existence expressions like "exists(param)", "!exists(param)"
    fn evaluate_existence_expression(
        &self,
        expression: &str,
        config: &HashMap<String, ConfigValue>,
    ) -> Option<bool> {
        if expression.starts_with("exists(") && expression.ends_with(')') {
            let param_name = &expression[7..expression.len() - 1];
            return Some(config.contains_key(param_name));
        }

        if expression.starts_with("!exists(") && expression.ends_with(')') {
            let param_name = &expression[8..expression.len() - 1];
            return Some(!config.contains_key(param_name));
        }

        None
    }

    /// Evaluate a single expression (used for logical operations)
    fn evaluate_single_expression(
        &self,
        expression: &str,
        config: &HashMap<String, ConfigValue>,
    ) -> bool {
        // Try comparison first
        if let Some(result) = self.evaluate_comparison_expression(expression, config) {
            return result;
        }

        // Try existence check
        if let Some(result) = self.evaluate_existence_expression(expression, config) {
            return result;
        }

        // Check if it's just a parameter name (boolean check)
        if let Some(value) = config.get(expression) {
            return match value {
                ConfigValue::Boolean(b) => *b,
                ConfigValue::Null => false,
                _ => true, // Non-null values are truthy
            };
        }

        false
    }

    /// Parse a string value into a ConfigValue
    fn parse_value_from_string(&self, value_str: &str) -> Option<ConfigValue> {
        let value_str = value_str.trim();

        // Remove quotes if present
        let value_str = if (value_str.starts_with('"') && value_str.ends_with('"'))
            || (value_str.starts_with('\'') && value_str.ends_with('\''))
        {
            &value_str[1..value_str.len() - 1]
        } else {
            value_str
        };

        // Try to parse as different types
        if let Ok(b) = value_str.parse::<bool>() {
            Some(ConfigValue::Boolean(b))
        } else if let Ok(i) = value_str.parse::<i64>() {
            Some(ConfigValue::Integer(i))
        } else if let Ok(f) = value_str.parse::<f64>() {
            Some(ConfigValue::Float(f))
        } else if value_str == "null" {
            Some(ConfigValue::Null)
        } else {
            Some(ConfigValue::String(value_str.to_string()))
        }
    }

    /// Compare two ConfigValues using the given operator
    fn compare_values(&self, left: &ConfigValue, right: &ConfigValue, operator: &str) -> bool {
        match (left, right) {
            (ConfigValue::Integer(l), ConfigValue::Integer(r)) => match operator {
                "==" => l == r,
                "!=" => l != r,
                ">" => l > r,
                "<" => l < r,
                ">=" => l >= r,
                "<=" => l <= r,
                _ => false,
            },
            (ConfigValue::Float(l), ConfigValue::Float(r)) => match operator {
                "==" => (l - r).abs() < f64::EPSILON,
                "!=" => (l - r).abs() >= f64::EPSILON,
                ">" => l > r,
                "<" => l < r,
                ">=" => l >= r,
                "<=" => l <= r,
                _ => false,
            },
            (ConfigValue::String(l), ConfigValue::String(r)) => match operator {
                "==" => l == r,
                "!=" => l != r,
                ">" => l > r,
                "<" => l < r,
                ">=" => l >= r,
                "<=" => l <= r,
                _ => false,
            },
            (ConfigValue::Boolean(l), ConfigValue::Boolean(r)) => match operator {
                "==" => l == r,
                "!=" => l != r,
                _ => false,
            },
            // Type conversion for mixed numeric types
            (ConfigValue::Integer(l), ConfigValue::Float(r)) => {
                let l_f = *l as f64;
                match operator {
                    "==" => (l_f - r).abs() < f64::EPSILON,
                    "!=" => (l_f - r).abs() >= f64::EPSILON,
                    ">" => l_f > *r,
                    "<" => l_f < *r,
                    ">=" => l_f >= *r,
                    "<=" => l_f <= *r,
                    _ => false,
                }
            }
            (ConfigValue::Float(l), ConfigValue::Integer(r)) => {
                let r_f = *r as f64;
                match operator {
                    "==" => (l - r_f).abs() < f64::EPSILON,
                    "!=" => (l - r_f).abs() >= f64::EPSILON,
                    ">" => *l > r_f,
                    "<" => *l < r_f,
                    ">=" => *l >= r_f,
                    "<=" => *l <= r_f,
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

pub fn validate_config_basic(config: &HashMap<String, ConfigValue>) -> VoirsResult<bool> {
    let validator = ConfigValidator::new(ConfigValidationRules::default());
    let result = validator.validate_config(config)?;
    Ok(result.is_valid)
}

pub fn create_default_validation_rules() -> ConfigValidationRules {
    let mut rules = ConfigValidationRules::default();

    // Add common parameter rules
    rules.parameter_rules.insert(
        "sample_rate".to_string(),
        ParameterRule {
            name: "sample_rate".to_string(),
            parameter_type: ParameterType::Integer,
            required: false,
            default_value: Some(ConfigValue::Integer(22050)),
            validation: ParameterValidation {
                min_value: Some(8000.0),
                max_value: Some(48000.0),
                allowed_values: Some(vec![
                    ConfigValue::Integer(8000),
                    ConfigValue::Integer(16000),
                    ConfigValue::Integer(22050),
                    ConfigValue::Integer(44100),
                    ConfigValue::Integer(48000),
                ]),
                ..Default::default()
            },
            description: "Audio sample rate in Hz".to_string(),
            deprecated: false,
            deprecation_message: None,
        },
    );

    rules.parameter_rules.insert(
        "device".to_string(),
        ParameterRule {
            name: "device".to_string(),
            parameter_type: ParameterType::Enum(vec![
                "cpu".to_string(),
                "cuda".to_string(),
                "auto".to_string(),
            ]),
            required: false,
            default_value: Some(ConfigValue::String("auto".to_string())),
            validation: ParameterValidation::default(),
            description: "Device to use for computation".to_string(),
            deprecated: false,
            deprecation_message: None,
        },
    );

    // Add resource requirements
    rules.resource_requirements.push(ResourceRequirement {
        name: "memory".to_string(),
        resource_type: ResourceType::Memory,
        minimum: 512.0,
        recommended: 2048.0,
        unit: "MB".to_string(),
        check_availability: true,
        severity: ValidationSeverity::Error,
    });

    rules
}
