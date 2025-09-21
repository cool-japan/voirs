//! Database sharding strategies for horizontal scaling
//!
//! This module provides database sharding capabilities to distribute data across
//! multiple database instances for improved performance and scalability.

use async_trait::async_trait;
use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

use super::{PersistenceError, PersistenceManager, PersistenceResult};
use crate::traits::{FeedbackResponse, SessionState, UserPreferences, UserProgress};

/// Sharding strategy errors
#[derive(Error, Debug)]
pub enum ShardingError {
    /// Shard configuration error
    #[error("Shard configuration error: {message}")]
    ConfigError {
        /// Error message
        message: String,
    },

    /// Shard not available
    #[error("Shard {shard_id} is not available")]
    ShardUnavailable {
        /// Shard ID
        shard_id: String,
    },

    /// Rebalancing in progress
    #[error("Shard rebalancing is in progress")]
    RebalancingInProgress,

    /// Consistency error
    #[error("Data consistency error: {message}")]
    ConsistencyError {
        /// Error message
        message: String,
    },
}

/// Sharding strategy types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Hash-based sharding using user ID
    HashBased,
    /// Range-based sharding
    RangeBased,
    /// Geographic sharding
    Geographic,
    /// Time-based sharding
    TimeBased,
    /// Consistent hashing
    ConsistentHashing,
}

/// Shard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Shard identifier
    pub shard_id: String,
    /// Database connection string
    pub connection_string: String,
    /// Shard weight for load balancing
    pub weight: f32,
    /// Geographic region
    pub region: Option<String>,
    /// Read replicas
    pub read_replicas: Vec<String>,
    /// Shard status
    pub status: ShardStatus,
    /// Capacity limits
    pub capacity: ShardCapacity,
}

/// Shard status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShardStatus {
    /// Shard is active and available
    Active,
    /// Shard is in maintenance mode
    Maintenance,
    /// Shard is being migrated
    Migrating,
    /// Shard is offline
    Offline,
}

/// Shard capacity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCapacity {
    /// Maximum number of users
    pub max_users: usize,
    /// Maximum storage size in bytes
    pub max_storage_bytes: u64,
    /// Current user count
    pub current_users: usize,
    /// Current storage usage in bytes
    pub current_storage_bytes: u64,
}

/// Shard routing information
#[derive(Debug, Clone)]
pub struct ShardRoute {
    /// Primary shard for writes
    pub primary_shard: String,
    /// Read replicas
    pub read_replicas: Vec<String>,
    /// Routing hash
    pub routing_hash: u64,
}

/// Sharding manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    /// Sharding strategy
    pub strategy: ShardingStrategy,
    /// Number of virtual nodes for consistent hashing
    pub virtual_nodes: usize,
    /// Replication factor
    pub replication_factor: usize,
    /// Auto-rebalancing enabled
    pub auto_rebalancing: bool,
    /// Shard configurations
    pub shards: Vec<ShardConfig>,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
}

/// Consistency levels for distributed operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Eventually consistent
    Eventual,
    /// Strong consistency
    Strong,
    /// Causal consistency
    Causal,
    /// Session consistency
    Session,
}

/// Sharding manager for distributing data across multiple database instances
pub struct ShardingManager {
    /// Sharding configuration
    config: ShardingConfig,
    /// Shard managers
    shard_managers: HashMap<String, Arc<dyn PersistenceManager>>,
    /// Consistent hash ring for consistent hashing strategy
    hash_ring: ConsistentHashRing,
    /// Migration state
    migration_state: Option<MigrationState>,
}

/// Migration state for data rebalancing
#[derive(Debug, Clone)]
pub struct MigrationState {
    /// Source shard
    pub source_shard: String,
    /// Target shard
    pub target_shard: String,
    /// Migration progress (0.0 to 1.0)
    pub progress: f32,
    /// Migration start time
    pub start_time: DateTime<Utc>,
    /// Estimated completion time
    pub estimated_completion: DateTime<Utc>,
}

/// Consistent hash ring for consistent hashing strategy
pub struct ConsistentHashRing {
    /// Ring nodes
    nodes: Vec<HashNode>,
    /// Virtual nodes per physical node
    virtual_nodes: usize,
}

/// Hash node in the consistent hash ring
#[derive(Debug, Clone)]
pub struct HashNode {
    /// Node identifier
    pub node_id: String,
    /// Hash value
    pub hash: u64,
    /// Physical shard ID
    pub shard_id: String,
}

impl ShardingManager {
    /// Create a new sharding manager
    pub async fn new(
        config: ShardingConfig,
        shard_managers: HashMap<String, Arc<dyn PersistenceManager>>,
    ) -> PersistenceResult<Self> {
        let hash_ring = ConsistentHashRing::new(&config.shards, config.virtual_nodes);

        Ok(Self {
            config,
            shard_managers,
            hash_ring,
            migration_state: None,
        })
    }

    /// Route a user to the appropriate shard
    pub fn route_user(&self, user_id: &str) -> PersistenceResult<ShardRoute> {
        match self.config.strategy {
            ShardingStrategy::HashBased => self.route_hash_based(user_id),
            ShardingStrategy::ConsistentHashing => self.route_consistent_hash(user_id),
            ShardingStrategy::RangeBased => self.route_range_based(user_id),
            ShardingStrategy::Geographic => self.route_geographic(user_id),
            ShardingStrategy::TimeBased => self.route_time_based(),
        }
    }

    /// Hash-based routing
    fn route_hash_based(&self, user_id: &str) -> PersistenceResult<ShardRoute> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        user_id.hash(&mut hasher);
        let hash = hasher.finish();

        let shard_index = (hash % self.config.shards.len() as u64) as usize;
        let primary_shard = &self.config.shards[shard_index];

        if primary_shard.status != ShardStatus::Active {
            return Err(PersistenceError::ConnectionError {
                message: format!("Primary shard {} is not active", primary_shard.shard_id),
            });
        }

        Ok(ShardRoute {
            primary_shard: primary_shard.shard_id.clone(),
            read_replicas: primary_shard.read_replicas.clone(),
            routing_hash: hash,
        })
    }

    /// Consistent hashing routing
    fn route_consistent_hash(&self, user_id: &str) -> PersistenceResult<ShardRoute> {
        let node = self.hash_ring.find_node(user_id)?;
        let shard_config = self
            .config
            .shards
            .iter()
            .find(|s| s.shard_id == node.shard_id)
            .ok_or_else(|| PersistenceError::ConfigError {
                message: format!("Shard {} not found in configuration", node.shard_id),
            })?;

        if shard_config.status != ShardStatus::Active {
            return Err(PersistenceError::ConnectionError {
                message: format!("Primary shard {} is not active", shard_config.shard_id),
            });
        }

        Ok(ShardRoute {
            primary_shard: shard_config.shard_id.clone(),
            read_replicas: shard_config.read_replicas.clone(),
            routing_hash: node.hash,
        })
    }

    /// Range-based routing (simplified implementation)
    fn route_range_based(&self, user_id: &str) -> PersistenceResult<ShardRoute> {
        // Simple alphabetical range partitioning
        let first_char = user_id.chars().next().unwrap_or('a').to_ascii_lowercase();
        let partition =
            ((first_char as u8 - b'a') as f32 / 26.0 * self.config.shards.len() as f32) as usize;
        let shard_index = partition.min(self.config.shards.len() - 1);

        let primary_shard = &self.config.shards[shard_index];

        Ok(ShardRoute {
            primary_shard: primary_shard.shard_id.clone(),
            read_replicas: primary_shard.read_replicas.clone(),
            routing_hash: first_char as u64,
        })
    }

    /// Geographic routing (simplified implementation)
    fn route_geographic(&self, _user_id: &str) -> PersistenceResult<ShardRoute> {
        // For now, route to the first available shard
        // In practice, this would use user's geographic location
        let primary_shard = self
            .config
            .shards
            .iter()
            .find(|s| s.status == ShardStatus::Active)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: "No active shards available".to_string(),
            })?;

        Ok(ShardRoute {
            primary_shard: primary_shard.shard_id.clone(),
            read_replicas: primary_shard.read_replicas.clone(),
            routing_hash: 0,
        })
    }

    /// Time-based routing
    fn route_time_based(&self) -> PersistenceResult<ShardRoute> {
        let now = Utc::now();
        let hour = now.hour();
        let shard_index = (hour % self.config.shards.len() as u32) as usize;

        let primary_shard = &self.config.shards[shard_index];

        Ok(ShardRoute {
            primary_shard: primary_shard.shard_id.clone(),
            read_replicas: primary_shard.read_replicas.clone(),
            routing_hash: hour as u64,
        })
    }

    /// Get shard manager for a specific shard
    pub fn get_shard_manager(&self, shard_id: &str) -> Option<Arc<dyn PersistenceManager>> {
        self.shard_managers.get(shard_id).cloned()
    }

    /// Start data migration between shards
    pub async fn start_migration(
        &mut self,
        source_shard: String,
        target_shard: String,
    ) -> PersistenceResult<()> {
        if self.migration_state.is_some() {
            return Err(PersistenceError::ConfigError {
                message: "Migration already in progress".to_string(),
            });
        }

        self.migration_state = Some(MigrationState {
            source_shard,
            target_shard,
            progress: 0.0,
            start_time: Utc::now(),
            estimated_completion: Utc::now() + chrono::Duration::hours(2), // Estimated 2 hours
        });

        Ok(())
    }

    /// Get migration status
    pub fn get_migration_status(&self) -> Option<&MigrationState> {
        self.migration_state.as_ref()
    }

    /// Add a new shard to the cluster
    pub async fn add_shard(&mut self, shard_config: ShardConfig) -> PersistenceResult<()> {
        // Validate shard configuration
        if self
            .config
            .shards
            .iter()
            .any(|s| s.shard_id == shard_config.shard_id)
        {
            return Err(PersistenceError::ConfigError {
                message: format!("Shard {} already exists", shard_config.shard_id),
            });
        }

        // Add to configuration
        self.config.shards.push(shard_config.clone());

        // Rebuild hash ring if using consistent hashing
        if self.config.strategy == ShardingStrategy::ConsistentHashing {
            self.hash_ring =
                ConsistentHashRing::new(&self.config.shards, self.config.virtual_nodes);
        }

        Ok(())
    }

    /// Remove a shard from the cluster
    pub async fn remove_shard(&mut self, shard_id: &str) -> PersistenceResult<()> {
        let shard_index = self
            .config
            .shards
            .iter()
            .position(|s| s.shard_id == shard_id)
            .ok_or_else(|| PersistenceError::ConfigError {
                message: format!("Shard {} not found", shard_id),
            })?;

        // Mark shard as offline first
        self.config.shards[shard_index].status = ShardStatus::Offline;

        // Remove shard manager
        self.shard_managers.remove(shard_id);

        // Remove from configuration
        self.config.shards.remove(shard_index);

        // Rebuild hash ring if using consistent hashing
        if self.config.strategy == ShardingStrategy::ConsistentHashing {
            self.hash_ring =
                ConsistentHashRing::new(&self.config.shards, self.config.virtual_nodes);
        }

        Ok(())
    }

    /// Get cluster health status
    pub fn get_cluster_health(&self) -> ClusterHealth {
        let total_shards = self.config.shards.len();
        let active_shards = self
            .config
            .shards
            .iter()
            .filter(|s| s.status == ShardStatus::Active)
            .count();

        let total_capacity = self
            .config
            .shards
            .iter()
            .map(|s| s.capacity.max_users)
            .sum::<usize>();

        let used_capacity = self
            .config
            .shards
            .iter()
            .map(|s| s.capacity.current_users)
            .sum::<usize>();

        ClusterHealth {
            total_shards,
            active_shards,
            capacity_utilization: if total_capacity > 0 {
                used_capacity as f32 / total_capacity as f32
            } else {
                0.0
            },
            migration_in_progress: self.migration_state.is_some(),
            shard_health: self
                .config
                .shards
                .iter()
                .map(|s| ShardHealth {
                    shard_id: s.shard_id.clone(),
                    status: s.status.clone(),
                    capacity_utilization: if s.capacity.max_users > 0 {
                        s.capacity.current_users as f32 / s.capacity.max_users as f32
                    } else {
                        0.0
                    },
                    storage_utilization: if s.capacity.max_storage_bytes > 0 {
                        s.capacity.current_storage_bytes as f32
                            / s.capacity.max_storage_bytes as f32
                    } else {
                        0.0
                    },
                })
                .collect(),
        }
    }
}

/// Cluster health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHealth {
    /// Total number of shards
    pub total_shards: usize,
    /// Number of active shards
    pub active_shards: usize,
    /// Overall capacity utilization (0.0 to 1.0)
    pub capacity_utilization: f32,
    /// Whether migration is in progress
    pub migration_in_progress: bool,
    /// Individual shard health
    pub shard_health: Vec<ShardHealth>,
}

/// Individual shard health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardHealth {
    /// Shard identifier
    pub shard_id: String,
    /// Shard status
    pub status: ShardStatus,
    /// Capacity utilization (0.0 to 1.0)
    pub capacity_utilization: f32,
    /// Storage utilization (0.0 to 1.0)
    pub storage_utilization: f32,
}

impl ConsistentHashRing {
    /// Create a new consistent hash ring
    pub fn new(shards: &[ShardConfig], virtual_nodes: usize) -> Self {
        let mut nodes = Vec::new();

        for shard in shards {
            if shard.status == ShardStatus::Active {
                for i in 0..virtual_nodes {
                    let virtual_node_id = format!("{}:{}", shard.shard_id, i);
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    virtual_node_id.hash(&mut hasher);
                    let hash = hasher.finish();

                    nodes.push(HashNode {
                        node_id: virtual_node_id,
                        hash,
                        shard_id: shard.shard_id.clone(),
                    });
                }
            }
        }

        // Sort nodes by hash value
        nodes.sort_by(|a, b| a.hash.cmp(&b.hash));

        Self {
            nodes,
            virtual_nodes,
        }
    }

    /// Find the appropriate node for a given key
    pub fn find_node(&self, key: &str) -> PersistenceResult<&HashNode> {
        if self.nodes.is_empty() {
            return Err(PersistenceError::ConfigError {
                message: "No active nodes in hash ring".to_string(),
            });
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        // Find the first node with hash >= key hash
        for node in &self.nodes {
            if node.hash >= hash {
                return Ok(node);
            }
        }

        // Wrap around to the first node
        Ok(&self.nodes[0])
    }
}

/// Sharded persistence manager that implements the PersistenceManager trait
pub struct ShardedPersistenceManager {
    /// Sharding manager
    sharding_manager: Arc<ShardingManager>,
}

impl ShardedPersistenceManager {
    /// Create a new sharded persistence manager
    pub async fn new(
        config: ShardingConfig,
        shard_managers: HashMap<String, Arc<dyn PersistenceManager>>,
    ) -> PersistenceResult<Self> {
        let sharding_manager = Arc::new(ShardingManager::new(config, shard_managers).await?);

        Ok(Self { sharding_manager })
    }

    /// Get the sharding manager
    pub fn get_sharding_manager(&self) -> Arc<ShardingManager> {
        self.sharding_manager.clone()
    }
}

#[async_trait]
impl PersistenceManager for ShardedPersistenceManager {
    async fn initialize(&mut self) -> PersistenceResult<()> {
        // Initialize all shard managers
        for manager in self.sharding_manager.shard_managers.values() {
            // Note: This would require making shard managers mutable
            // In practice, initialization would be handled differently
        }
        Ok(())
    }

    async fn save_session(&self, session: &SessionState) -> PersistenceResult<()> {
        let route = self.sharding_manager.route_user(&session.user_id)?;
        let manager = self
            .sharding_manager
            .get_shard_manager(&route.primary_shard)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: format!("Shard manager {} not found", route.primary_shard),
            })?;

        manager.save_session(session).await
    }

    async fn load_session(&self, session_id: &Uuid) -> PersistenceResult<SessionState> {
        // For session loading, we need to search across shards or maintain an index
        // This is a simplified implementation
        for manager in self.sharding_manager.shard_managers.values() {
            if let Ok(session) = manager.load_session(session_id).await {
                return Ok(session);
            }
        }

        Err(PersistenceError::NotFound {
            entity_type: "session".to_string(),
            id: session_id.to_string(),
        })
    }

    async fn save_user_progress(
        &self,
        user_id: &str,
        progress: &UserProgress,
    ) -> PersistenceResult<()> {
        let route = self.sharding_manager.route_user(user_id)?;
        let manager = self
            .sharding_manager
            .get_shard_manager(&route.primary_shard)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: format!("Shard manager {} not found", route.primary_shard),
            })?;

        manager.save_user_progress(user_id, progress).await
    }

    async fn load_user_progress(&self, user_id: &str) -> PersistenceResult<UserProgress> {
        let route = self.sharding_manager.route_user(user_id)?;
        let manager = self
            .sharding_manager
            .get_shard_manager(&route.primary_shard)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: format!("Shard manager {} not found", route.primary_shard),
            })?;

        manager.load_user_progress(user_id).await
    }

    async fn save_feedback(
        &self,
        user_id: &str,
        feedback: &FeedbackResponse,
    ) -> PersistenceResult<()> {
        let route = self.sharding_manager.route_user(user_id)?;
        let manager = self
            .sharding_manager
            .get_shard_manager(&route.primary_shard)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: format!("Shard manager {} not found", route.primary_shard),
            })?;

        manager.save_feedback(user_id, feedback).await
    }

    async fn load_feedback_history(
        &self,
        user_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PersistenceResult<Vec<FeedbackResponse>> {
        let route = self.sharding_manager.route_user(user_id)?;
        let manager = self
            .sharding_manager
            .get_shard_manager(&route.primary_shard)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: format!("Shard manager {} not found", route.primary_shard),
            })?;

        manager.load_feedback_history(user_id, limit, offset).await
    }

    async fn save_preferences(
        &self,
        user_id: &str,
        preferences: &UserPreferences,
    ) -> PersistenceResult<()> {
        let route = self.sharding_manager.route_user(user_id)?;
        let manager = self
            .sharding_manager
            .get_shard_manager(&route.primary_shard)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: format!("Shard manager {} not found", route.primary_shard),
            })?;

        manager.save_preferences(user_id, preferences).await
    }

    async fn load_preferences(&self, user_id: &str) -> PersistenceResult<UserPreferences> {
        let route = self.sharding_manager.route_user(user_id)?;
        let manager = self
            .sharding_manager
            .get_shard_manager(&route.primary_shard)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: format!("Shard manager {} not found", route.primary_shard),
            })?;

        manager.load_preferences(user_id).await
    }

    async fn delete_user_data(&self, user_id: &str) -> PersistenceResult<()> {
        let route = self.sharding_manager.route_user(user_id)?;
        let manager = self
            .sharding_manager
            .get_shard_manager(&route.primary_shard)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: format!("Shard manager {} not found", route.primary_shard),
            })?;

        manager.delete_user_data(user_id).await
    }

    async fn export_user_data(&self, user_id: &str) -> PersistenceResult<super::UserDataExport> {
        let route = self.sharding_manager.route_user(user_id)?;
        let manager = self
            .sharding_manager
            .get_shard_manager(&route.primary_shard)
            .ok_or_else(|| PersistenceError::ConnectionError {
                message: format!("Shard manager {} not found", route.primary_shard),
            })?;

        manager.export_user_data(user_id).await
    }

    async fn get_storage_stats(&self) -> PersistenceResult<super::StorageStats> {
        // Aggregate stats from all shards
        let mut total_users = 0;
        let mut total_sessions = 0;
        let mut total_feedback_records = 0;
        let mut total_storage_bytes = 0;
        let mut last_cleanup: Option<DateTime<Utc>> = None;

        for manager in self.sharding_manager.shard_managers.values() {
            let stats = manager.get_storage_stats().await?;
            total_users += stats.total_users;
            total_sessions += stats.total_sessions;
            total_feedback_records += stats.total_feedback_records;
            total_storage_bytes += stats.storage_size_bytes;

            if let Some(cleanup_time) = stats.last_cleanup {
                last_cleanup = Some(match last_cleanup {
                    Some(existing) => existing.max(cleanup_time),
                    None => cleanup_time,
                });
            }
        }

        Ok(super::StorageStats {
            total_users,
            total_sessions,
            total_feedback_records,
            storage_size_bytes: total_storage_bytes,
            last_cleanup,
            db_version: "sharded-v1.0".to_string(),
        })
    }

    async fn cleanup(&self, older_than: DateTime<Utc>) -> PersistenceResult<super::CleanupResult> {
        let mut total_sessions_cleaned = 0;
        let mut total_feedback_records_cleaned = 0;
        let mut total_bytes_reclaimed = 0;
        let start_time = std::time::Instant::now();

        // Cleanup all shards in parallel
        let mut futures = Vec::new();
        for manager in self.sharding_manager.shard_managers.values() {
            let manager = manager.clone();
            let older_than = older_than;
            futures.push(async move { manager.cleanup(older_than).await });
        }

        // Wait for all cleanup operations to complete
        let results = futures::future::join_all(futures).await;

        for result in results {
            match result {
                Ok(cleanup_result) => {
                    total_sessions_cleaned += cleanup_result.sessions_cleaned;
                    total_feedback_records_cleaned += cleanup_result.feedback_records_cleaned;
                    total_bytes_reclaimed += cleanup_result.bytes_reclaimed;
                }
                Err(e) => {
                    log::warn!("Shard cleanup failed: {}", e);
                }
            }
        }

        Ok(super::CleanupResult {
            sessions_cleaned: total_sessions_cleaned,
            feedback_records_cleaned: total_feedback_records_cleaned,
            bytes_reclaimed: total_bytes_reclaimed,
            cleanup_duration: start_time.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_shard_config(id: &str) -> ShardConfig {
        ShardConfig {
            shard_id: id.to_string(),
            connection_string: format!("test://{}", id),
            weight: 1.0,
            region: Some("us-east-1".to_string()),
            read_replicas: vec![],
            status: ShardStatus::Active,
            capacity: ShardCapacity {
                max_users: 1000,
                max_storage_bytes: 1_000_000_000,
                current_users: 0,
                current_storage_bytes: 0,
            },
        }
    }

    #[test]
    fn test_consistent_hash_ring() {
        let shards = vec![
            create_test_shard_config("shard1"),
            create_test_shard_config("shard2"),
            create_test_shard_config("shard3"),
        ];

        let ring = ConsistentHashRing::new(&shards, 3);
        assert_eq!(ring.nodes.len(), 9); // 3 shards * 3 virtual nodes

        // Test consistent routing
        let node1 = ring.find_node("user123").unwrap();
        let node2 = ring.find_node("user123").unwrap();
        assert_eq!(node1.shard_id, node2.shard_id);
    }

    #[tokio::test]
    async fn test_sharding_manager_creation() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::HashBased,
            virtual_nodes: 3,
            replication_factor: 2,
            auto_rebalancing: true,
            shards: vec![
                create_test_shard_config("shard1"),
                create_test_shard_config("shard2"),
            ],
            consistency_level: ConsistencyLevel::Eventual,
        };

        let shard_managers = HashMap::new();
        let manager = ShardingManager::new(config, shard_managers).await.unwrap();

        assert_eq!(manager.config.shards.len(), 2);
    }

    #[test]
    fn test_hash_based_routing() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::HashBased,
            virtual_nodes: 3,
            replication_factor: 2,
            auto_rebalancing: true,
            shards: vec![
                create_test_shard_config("shard1"),
                create_test_shard_config("shard2"),
            ],
            consistency_level: ConsistencyLevel::Eventual,
        };

        let shard_managers = HashMap::new();
        let manager = ShardingManager {
            config,
            shard_managers,
            hash_ring: ConsistentHashRing::new(&[], 0),
            migration_state: None,
        };

        let route = manager.route_hash_based("user123").unwrap();
        assert!(route.primary_shard == "shard1" || route.primary_shard == "shard2");

        // Same user should always route to the same shard
        let route2 = manager.route_hash_based("user123").unwrap();
        assert_eq!(route.primary_shard, route2.primary_shard);
    }

    #[test]
    fn test_cluster_health() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::HashBased,
            virtual_nodes: 3,
            replication_factor: 2,
            auto_rebalancing: true,
            shards: vec![
                create_test_shard_config("shard1"),
                create_test_shard_config("shard2"),
            ],
            consistency_level: ConsistencyLevel::Eventual,
        };

        let shard_managers = HashMap::new();
        let manager = ShardingManager {
            config,
            shard_managers,
            hash_ring: ConsistentHashRing::new(&[], 0),
            migration_state: None,
        };

        let health = manager.get_cluster_health();
        assert_eq!(health.total_shards, 2);
        assert_eq!(health.active_shards, 2);
        assert_eq!(health.capacity_utilization, 0.0);
        assert!(!health.migration_in_progress);
    }
}
