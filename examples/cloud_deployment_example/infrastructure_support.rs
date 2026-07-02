//! Supporting runtime infrastructure components for `CloudVoiRSService`:
//! load balancing, cloud storage access, and auto-scaling policy state.

use std::thread;
use std::time::Duration;

use crate::config::{
    LoadBalancerConfig, LoadBalancingAlgorithm, ScalingConfig, StorageConfig, StorageType,
};
use crate::error::CloudError;
use crate::service_types::{InstanceStatus, ServiceInstance};

pub struct LoadBalancer {
    algorithm: LoadBalancingAlgorithm,
    targets: Vec<String>,
}

impl LoadBalancer {
    pub fn new(config: &LoadBalancerConfig) -> Result<Self, CloudError> {
        Ok(Self {
            algorithm: config.algorithm,
            targets: Vec::new(),
        })
    }

    pub fn update_targets(&mut self, instances: Vec<&ServiceInstance>) -> Result<(), CloudError> {
        self.targets = instances
            .iter()
            .filter(|i| matches!(i.status, InstanceStatus::Healthy))
            .map(|i| i.id.clone())
            .collect();
        Ok(())
    }

    pub fn select_instance(&self, _region: &str) -> Result<String, CloudError> {
        if self.targets.is_empty() {
            return Err(CloudError::NoHealthyInstances);
        }

        match self.algorithm {
            LoadBalancingAlgorithm::RoundRobin => {
                let index = (self.targets[0].len() * 17) % self.targets.len();
                Ok(self.targets[index].clone())
            }
            _ => Ok(self.targets[0].clone()),
        }
    }
}

pub struct StorageManager {
    storage_type: StorageType,
}

impl StorageManager {
    pub fn new(config: &StorageConfig) -> Result<Self, CloudError> {
        Ok(Self {
            storage_type: config.audio_storage.storage_type,
        })
    }

    pub fn store_audio(&self, key: &str, data: Vec<u8>) -> Result<(), CloudError> {
        println!(
            "💾 Storing audio: {} ({} bytes) in {:?}",
            key,
            data.len(),
            self.storage_type
        );
        // Simulate storage operation
        thread::sleep(Duration::from_millis(50));
        Ok(())
    }
}

pub struct AutoScaler {
    // Pre-existing in the original single-file example (predates this module
    // split): stored for potential future scaling-policy introspection but
    // never read back today. `set_scaling_policies` takes its own `&ScalingConfig`
    // parameter instead of using this field, matching the original behavior.
    #[allow(dead_code)]
    scaling_config: ScalingConfig,
}

impl AutoScaler {
    pub fn new(config: &ScalingConfig) -> Result<Self, CloudError> {
        Ok(Self {
            scaling_config: config.clone(),
        })
    }

    pub fn set_scaling_policies(&self, _config: &ScalingConfig) -> Result<(), CloudError> {
        println!("📈 Setting auto-scaling policies...");
        Ok(())
    }
}
