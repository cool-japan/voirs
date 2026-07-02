use crate::config::CloudProvider;

#[derive(Debug)]
pub struct DeploymentResult {
    pub instances_deployed: usize,
    pub regions_active: usize,
    pub endpoint_url: String,
    pub deployment_time_seconds: u32,
}

#[derive(Debug)]
pub struct ProcessingStats {
    pub requests_processed: u32,
    pub requests_failed: u32,
    pub processing_time_ms: u32,
    pub active_instances: u32,
    pub queue_length: usize,
}

#[derive(Debug)]
pub struct SynthesisResult {
    pub request_id: u64,
    pub audio_url: String,
    pub duration_ms: u32,
    pub file_size_bytes: u64,
    pub processed_by: String,
    pub processing_time_ms: u64,
}

#[derive(Debug)]
pub struct DeploymentStatus {
    pub provider: CloudProvider,
    pub healthy_instances: u32,
    pub total_instances: u32,
    pub total_requests_processed: u64,
    pub average_cpu_usage: f32,
    pub average_memory_usage: f32,
    pub queue_length: usize,
    pub uptime_hours: f32,
}

#[derive(Debug)]
pub struct InfrastructureCode {
    pub aws_cloudformation: String,
    pub azure_arm: String,
    pub gcp_deployment_manager: String,
    pub kubernetes_yaml: String,
    pub terraform: String,
    pub docker_compose: String,
}
