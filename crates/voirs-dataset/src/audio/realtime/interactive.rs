//! Interactive processing configuration for real-time audio

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Interactive processing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InteractiveConfig {
    /// Control interface configuration
    pub control_interface: ControlInterface,
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
    /// Access control configuration
    pub access_control: AccessControlConfig,
    /// Feedback configuration
    pub feedback_config: FeedbackConfig,
    /// Interactive visualization configuration
    pub interactive_visualization: InteractiveVisualizationConfig,
}

/// Control interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlInterface {
    /// Interface type
    pub interface_type: InterfaceType,
    /// Port number (for network interfaces)
    pub port: Option<u16>,
    /// Host address (for network interfaces)
    pub host: Option<String>,
    /// Update configuration
    pub update_config: UpdateConfig,
}

/// Interface types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterfaceType {
    /// Web interface
    Web,
    /// REST API
    RestAPI,
    /// WebSocket
    WebSocket,
    /// Command line interface
    CLI,
    /// Custom interface
    Custom(String),
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication method
    pub method: AuthenticationMethod,
    /// Session timeout
    pub session_timeout: Duration,
    /// Enable multi-factor authentication
    pub multi_factor_auth: bool,
    /// Token configuration
    pub token_config: TokenConfig,
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    /// No authentication
    None,
    /// API key authentication
    ApiKey,
    /// JWT token authentication
    JWT,
    /// OAuth authentication
    OAuth,
    /// Custom authentication
    Custom(String),
}

/// Token configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    /// Token expiry time
    pub expiry_time: Duration,
    /// Token refresh interval
    pub refresh_interval: Duration,
    /// Token secret key
    pub secret_key: Option<String>,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessControlConfig {
    /// Enable role-based access control
    pub role_based_access: bool,
    /// User roles and permissions
    pub roles: HashMap<String, Vec<Permission>>,
    /// IP whitelist
    pub ip_whitelist: Vec<String>,
    /// Rate limiting
    pub rate_limiting: RateLimitingConfig,
}

/// Permissions for access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
    /// Read access
    Read,
    /// Write access
    Write,
    /// Execute access
    Execute,
    /// Admin access
    Admin,
    /// Custom permission
    Custom(String),
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Maximum requests per minute
    pub max_requests_per_minute: usize,
    /// Burst limit
    pub burst_limit: usize,
    /// Rate limiting window
    pub window: Duration,
}

/// Feedback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    /// Feedback delivery configuration
    pub delivery: FeedbackDelivery,
    /// Feedback aggregation configuration
    pub aggregation: FeedbackAggregation,
    /// Enable real-time feedback
    pub real_time_feedback: bool,
    /// Feedback history retention
    pub history_retention: Duration,
}

/// Feedback delivery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackDelivery {
    /// Delivery method
    pub method: FeedbackDeliveryMethod,
    /// Delivery interval
    pub interval: Duration,
    /// Batch size for batched delivery
    pub batch_size: usize,
    /// Priority levels
    pub priority_levels: Vec<FeedbackPriority>,
}

/// Feedback delivery methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackDeliveryMethod {
    /// Real-time delivery
    RealTime,
    /// Batched delivery
    Batched,
    /// On-demand delivery
    OnDemand,
    /// Custom delivery method
    Custom(String),
}

/// Feedback priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Feedback aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackAggregation {
    /// Aggregation method
    pub method: FeedbackAggregationMethod,
    /// Aggregation window
    pub window: Duration,
    /// Minimum feedback count
    pub min_count: usize,
    /// Statistical measures to include
    pub statistical_measures: Vec<StatisticalMeasure>,
}

/// Feedback aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackAggregationMethod {
    /// Average aggregation
    Average,
    /// Weighted average
    WeightedAverage,
    /// Median aggregation
    Median,
    /// Custom aggregation
    Custom(String),
}

/// Statistical measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalMeasure {
    /// Mean
    Mean,
    /// Median
    Median,
    /// Standard deviation
    StandardDeviation,
    /// Percentiles
    Percentiles(Vec<f32>),
}

/// Interactive visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveVisualizationConfig {
    /// Enable interactive controls
    pub interactive_controls: bool,
    /// Control types
    pub control_types: Vec<ControlType>,
    /// Update configuration
    pub update_config: UpdateConfig,
    /// Layout configuration
    pub layout_config: LayoutConfig,
}

/// Control types for interactive visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlType {
    /// Slider controls
    Slider,
    /// Button controls
    Button,
    /// Toggle controls
    Toggle,
    /// Text input controls
    TextInput,
    /// Custom controls
    Custom(String),
}

/// Update configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfig {
    /// Update interval
    pub interval: Duration,
    /// Enable automatic updates
    pub automatic_updates: bool,
    /// Update triggers
    pub triggers: Vec<UpdateTrigger>,
}

/// Update triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateTrigger {
    /// Time-based trigger
    TimeBased(Duration),
    /// Event-based trigger
    EventBased(String),
    /// Threshold-based trigger
    ThresholdBased(f32),
    /// Custom trigger
    Custom(String),
}

/// Layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Layout type
    pub layout_type: LayoutType,
    /// Grid configuration
    pub grid_config: GridConfig,
    /// Responsive design
    pub responsive_design: bool,
}

/// Layout types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutType {
    /// Grid layout
    Grid,
    /// Flex layout
    Flex,
    /// Fixed layout
    Fixed,
    /// Custom layout
    Custom(String),
}

/// Grid configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Number of columns
    pub columns: usize,
    /// Number of rows
    pub rows: usize,
    /// Cell spacing
    pub spacing: f32,
    /// Cell padding
    pub padding: f32,
}

impl Default for ControlInterface {
    fn default() -> Self {
        Self {
            interface_type: InterfaceType::Web,
            port: Some(8080),
            host: Some("localhost".to_string()),
            update_config: UpdateConfig::default(),
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            method: AuthenticationMethod::None,
            session_timeout: Duration::from_secs(3600),
            multi_factor_auth: false,
            token_config: TokenConfig::default(),
        }
    }
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            expiry_time: Duration::from_secs(3600),
            refresh_interval: Duration::from_secs(300),
            secret_key: None,
        }
    }
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_requests_per_minute: 60,
            burst_limit: 10,
            window: Duration::from_secs(60),
        }
    }
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            delivery: FeedbackDelivery::default(),
            aggregation: FeedbackAggregation::default(),
            real_time_feedback: true,
            history_retention: Duration::from_secs(86400),
        }
    }
}

impl Default for FeedbackDelivery {
    fn default() -> Self {
        Self {
            method: FeedbackDeliveryMethod::RealTime,
            interval: Duration::from_millis(100),
            batch_size: 10,
            priority_levels: vec![FeedbackPriority::Medium, FeedbackPriority::High],
        }
    }
}

impl Default for FeedbackAggregation {
    fn default() -> Self {
        Self {
            method: FeedbackAggregationMethod::Average,
            window: Duration::from_secs(60),
            min_count: 3,
            statistical_measures: vec![
                StatisticalMeasure::Mean,
                StatisticalMeasure::StandardDeviation,
            ],
        }
    }
}

impl Default for InteractiveVisualizationConfig {
    fn default() -> Self {
        Self {
            interactive_controls: true,
            control_types: vec![ControlType::Slider, ControlType::Toggle],
            update_config: UpdateConfig::default(),
            layout_config: LayoutConfig::default(),
        }
    }
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(100),
            automatic_updates: true,
            triggers: vec![UpdateTrigger::TimeBased(Duration::from_millis(100))],
        }
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            layout_type: LayoutType::Grid,
            grid_config: GridConfig::default(),
            responsive_design: true,
        }
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            columns: 3,
            rows: 3,
            spacing: 10.0,
            padding: 5.0,
        }
    }
}
