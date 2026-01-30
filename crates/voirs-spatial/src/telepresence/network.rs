//! Network settings, connection preferences, and QoS configuration

use super::types::*;
use serde::{Deserialize, Serialize};

/// Network settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSettings {
    /// Connection preferences
    pub connection_preferences: ConnectionPreferences,

    /// QoS settings
    pub qos_settings: QosSettings,

    /// Firewall and NAT
    pub firewall_settings: FirewallSettings,

    /// Redundancy settings
    pub redundancy_settings: RedundancySettings,
}

/// Connection preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPreferences {
    /// Preferred connection type
    pub preferred_type: ConnectionType,

    /// Allowed connection types
    pub allowed_types: Vec<ConnectionType>,

    /// Connection timeout (ms)
    pub timeout: u32,

    /// Retry attempts
    pub retry_attempts: u8,

    /// IPv6 preference
    pub ipv6_preferred: bool,
}

/// Quality of Service settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosSettings {
    /// DSCP marking
    pub dscp_marking: DscpMarking,

    /// Traffic shaping
    pub traffic_shaping: TrafficShapingSettings,

    /// Congestion control
    pub congestion_control: CongestionControlSettings,

    /// Jitter buffer settings
    pub jitter_buffer: JitterBufferSettings,
}

/// Traffic shaping settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficShapingSettings {
    /// Enable traffic shaping
    pub enabled: bool,

    /// Maximum burst size (bytes)
    pub max_burst: u32,

    /// Sustained rate (bps)
    pub sustained_rate: u32,

    /// Peak rate (bps)
    pub peak_rate: u32,
}

/// Congestion control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionControlSettings {
    /// Congestion control algorithm
    pub algorithm: CongestionControlAlgorithm,

    /// Initial bandwidth estimate (bps)
    pub initial_bandwidth: u32,

    /// Bandwidth probe interval (ms)
    pub probe_interval: u32,

    /// Congestion window size
    pub window_size: u32,
}

/// Jitter buffer settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterBufferSettings {
    /// Minimum buffer size (ms)
    pub min_buffer: u32,

    /// Maximum buffer size (ms)
    pub max_buffer: u32,

    /// Target buffer size (ms)
    pub target_buffer: u32,

    /// Adaptive buffer sizing
    pub adaptive: bool,

    /// Fast adaptation
    pub fast_adaptation: bool,
}

/// Firewall and NAT settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallSettings {
    /// STUN server configuration
    pub stun_servers: Vec<StunServerConfig>,

    /// TURN server configuration
    pub turn_servers: Vec<TurnServerConfig>,

    /// ICE settings
    pub ice_settings: IceSettings,

    /// Port range for media
    pub port_range: Option<(u16, u16)>,
}

/// STUN server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StunServerConfig {
    /// Server URL
    pub url: String,

    /// Port
    pub port: u16,

    /// Protocol
    pub protocol: StunProtocol,
}

/// TURN server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnServerConfig {
    /// Server URL
    pub url: String,

    /// Port
    pub port: u16,

    /// Username
    pub username: String,

    /// Credential
    pub credential: String,

    /// Protocol
    pub protocol: TurnProtocol,
}

/// ICE (Interactive Connectivity Establishment) settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceSettings {
    /// ICE gathering policy
    pub gathering_policy: IceGatheringPolicy,

    /// ICE transport policy
    pub transport_policy: IceTransportPolicy,

    /// Candidate timeout (ms)
    pub candidate_timeout: u32,

    /// Connection check timeout (ms)
    pub connection_timeout: u32,
}

/// Redundancy settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancySettings {
    /// Enable redundancy
    pub enabled: bool,

    /// Redundancy type
    pub redundancy_type: RedundancyType,

    /// Backup connections
    pub backup_connections: u8,

    /// Failover timeout (ms)
    pub failover_timeout: u32,
}

/// Network conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    /// Bandwidth (kbps)
    pub bandwidth: u32,

    /// Round-trip time (ms)
    pub rtt: f32,

    /// Packet loss percentage
    pub packet_loss: f32,

    /// Jitter (ms)
    pub jitter: f32,

    /// Connection quality score
    pub quality_score: f32,
}

/// Network condition thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkThresholds {
    /// Bandwidth thresholds (kbps)
    pub bandwidth_thresholds: Vec<u32>,

    /// Latency thresholds (ms)
    pub latency_thresholds: Vec<u32>,

    /// Packet loss thresholds (percentage)
    pub packet_loss_thresholds: Vec<f32>,

    /// Jitter thresholds (ms)
    pub jitter_thresholds: Vec<f32>,
}

// Default implementations

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            connection_preferences: ConnectionPreferences::default(),
            qos_settings: QosSettings::default(),
            firewall_settings: FirewallSettings::default(),
            redundancy_settings: RedundancySettings::default(),
        }
    }
}

impl Default for ConnectionPreferences {
    fn default() -> Self {
        Self {
            preferred_type: ConnectionType::Auto,
            allowed_types: vec![
                ConnectionType::P2P,
                ConnectionType::ServerMediated,
                ConnectionType::TurnRelay,
            ],
            timeout: 30000, // 30 seconds
            retry_attempts: 3,
            ipv6_preferred: false,
        }
    }
}

impl Default for QosSettings {
    fn default() -> Self {
        Self {
            dscp_marking: DscpMarking::Voice,
            traffic_shaping: TrafficShapingSettings::default(),
            congestion_control: CongestionControlSettings::default(),
            jitter_buffer: JitterBufferSettings::default(),
        }
    }
}

impl Default for TrafficShapingSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            max_burst: 16384,       // 16 KB
            sustained_rate: 320000, // 320 kbps
            peak_rate: 512000,      // 512 kbps
        }
    }
}

impl Default for CongestionControlSettings {
    fn default() -> Self {
        Self {
            algorithm: CongestionControlAlgorithm::WebRTC,
            initial_bandwidth: 128000, // 128 kbps
            probe_interval: 1000,      // 1 second
            window_size: 4096,
        }
    }
}

impl Default for JitterBufferSettings {
    fn default() -> Self {
        Self {
            min_buffer: 20,    // 20ms
            max_buffer: 200,   // 200ms
            target_buffer: 60, // 60ms
            adaptive: true,
            fast_adaptation: true,
        }
    }
}

impl Default for FirewallSettings {
    fn default() -> Self {
        Self {
            stun_servers: vec![StunServerConfig {
                url: "stun:stun.l.google.com".to_string(),
                port: 19302,
                protocol: StunProtocol::UDP,
            }],
            turn_servers: vec![],
            ice_settings: IceSettings::default(),
            port_range: Some((49152, 65535)),
        }
    }
}

impl Default for IceSettings {
    fn default() -> Self {
        Self {
            gathering_policy: IceGatheringPolicy::All,
            transport_policy: IceTransportPolicy::All,
            candidate_timeout: 10000,  // 10 seconds
            connection_timeout: 30000, // 30 seconds
        }
    }
}

impl Default for RedundancySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            redundancy_type: RedundancyType::ActivePassive,
            backup_connections: 1,
            failover_timeout: 5000, // 5 seconds
        }
    }
}

impl Default for NetworkThresholds {
    fn default() -> Self {
        Self {
            bandwidth_thresholds: vec![32, 64, 128, 256, 320],
            latency_thresholds: vec![50, 100, 200, 500],
            packet_loss_thresholds: vec![0.5, 1.0, 2.0, 5.0],
            jitter_thresholds: vec![10.0, 20.0, 50.0, 100.0],
        }
    }
}
