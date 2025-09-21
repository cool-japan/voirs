use std::sync::Arc;
use crate::VocoderError;

pub mod pool;
pub mod streaming;
pub mod resources;

pub use pool::*;
pub use streaming::*;
pub use resources::*;

pub type Result<T> = std::result::Result<T, VocoderError>;

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub pool_size: usize,
    pub max_buffers: usize,
    pub enable_recycling: bool,
    pub enable_prefetch: bool,
    pub alignment: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size: 1024 * 1024 * 64, // 64MB
            max_buffers: 128,
            enable_recycling: true,
            enable_prefetch: true,
            alignment: 64, // Cache line alignment
        }
    }
}

pub struct MemoryManager {
    config: MemoryConfig,
    pool: Arc<MemoryPool>,
    resources: Arc<ResourceManager>,
}

impl MemoryManager {
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let pool = Arc::new(MemoryPool::new(config.clone())?);
        let resources = Arc::new(ResourceManager::new(config.clone())?);
        
        Ok(Self {
            config,
            pool,
            resources,
        })
    }

    pub fn pool(&self) -> &Arc<MemoryPool> {
        &self.pool
    }

    pub fn resources(&self) -> &Arc<ResourceManager> {
        &self.resources
    }

    pub fn allocate_audio_buffer(&self, size: usize) -> Result<AllocatedBuffer> {
        self.pool.allocate_audio_buffer(size)
    }

    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            pool_stats: self.pool.stats(),
            resource_stats: self.resources.stats(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub pool_stats: PoolStats,
    pub resource_stats: ResourceStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config).unwrap();
        
        let stats = manager.stats();
        assert!(stats.pool_stats.total_capacity > 0);
    }

    #[test]
    fn test_memory_manager_allocation() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config).unwrap();
        
        let buffer = manager.allocate_audio_buffer(1024).unwrap();
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_memory_config_defaults() {
        let config = MemoryConfig::default();
        assert_eq!(config.pool_size, 1024 * 1024 * 64);
        assert_eq!(config.max_buffers, 128);
        assert!(config.enable_recycling);
        assert!(config.enable_prefetch);
        assert_eq!(config.alignment, 64);
    }
}