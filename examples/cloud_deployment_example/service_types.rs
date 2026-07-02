use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime};

use crate::config::CloudDeploymentConfig;
use crate::infrastructure_support::{AutoScaler, LoadBalancer, StorageManager};
use crate::monitoring::CloudMonitor;

pub struct CloudVoiRSService {
    // Fields are `pub(crate)` (rather than private) because the inherent
    // `impl CloudVoiRSService` block is split across sibling modules
    // (`service_core` and `iac_templates`); this grants those modules the
    // same access the original single-file version had implicitly.
    pub(crate) config: CloudDeploymentConfig,
    pub(crate) request_queue: Arc<Mutex<VecDeque<SynthesisRequest>>>,
    pub(crate) instances: Arc<RwLock<HashMap<String, ServiceInstance>>>,
    pub(crate) load_balancer: LoadBalancer,
    pub(crate) storage_manager: StorageManager,
    pub(crate) monitor: CloudMonitor,
    pub(crate) scaler: AutoScaler,
    pub(crate) next_request_id: Arc<Mutex<u64>>,
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub id: u64,
    pub text: String,
    pub voice_id: String,
    pub priority: RequestPriority,
    pub region: String,
    pub client_id: String,
    pub callback_url: Option<String>,
    pub timestamp: SystemTime,
    pub timeout_seconds: u32,
    pub quality: AudioQuality,
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub enum RequestPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

#[derive(Debug, Clone, Copy)]
pub enum AudioQuality {
    Compressed,
    Standard,
    HighQuality,
    Lossless,
}

#[derive(Debug, Clone)]
pub struct ServiceInstance {
    pub id: String,
    pub region: String,
    pub status: InstanceStatus,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub request_count: u64,
    pub last_health_check: Instant,
    pub created_at: Instant,
}

#[derive(Debug, Clone, Copy)]
pub enum InstanceStatus {
    Starting,
    Healthy,
    Unhealthy,
    Terminating,
}
