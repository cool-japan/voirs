#[derive(Debug, Clone)]
pub enum CloudError {
    InitializationFailed(String),
    DeploymentFailed(String),
    ScalingError(String),
    StorageError(String),
    NetworkError(String),
    MonitoringError(String),
    ThreadLockError,
    NoHealthyInstances,
    RequestTimeout(u64),
    InfrastructureError(String),
}

impl std::fmt::Display for CloudError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            CloudError::DeploymentFailed(msg) => write!(f, "Deployment failed: {}", msg),
            CloudError::ScalingError(msg) => write!(f, "Scaling error: {}", msg),
            CloudError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            CloudError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            CloudError::MonitoringError(msg) => write!(f, "Monitoring error: {}", msg),
            CloudError::ThreadLockError => write!(f, "Thread lock error"),
            CloudError::NoHealthyInstances => write!(f, "No healthy instances available"),
            CloudError::RequestTimeout(id) => write!(f, "Request {} timed out", id),
            CloudError::InfrastructureError(msg) => write!(f, "Infrastructure error: {}", msg),
        }
    }
}

impl std::error::Error for CloudError {}
