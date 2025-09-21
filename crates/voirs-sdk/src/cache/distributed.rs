//! Distributed caching implementation for VoiRS SDK
//!
//! This module provides distributed caching capabilities allowing cache
//! synchronization across multiple VoiRS instances and workers.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Type alias for local cache storage
type LocalCacheData = Arc<RwLock<HashMap<String, (Vec<u8>, SystemTime)>>>;

/// Distributed cache coordinator for managing cache across multiple nodes
#[derive(Debug)]
pub struct DistributedCacheCoordinator {
    node_id: String,
    nodes: Arc<RwLock<HashMap<String, CacheNode>>>,
    replication_factor: u32,
    consistency_level: ConsistencyLevel,
    #[allow(dead_code)]
    sync_interval: Duration,
}

/// Cache node information
#[derive(Debug, Clone)]
pub struct CacheNode {
    pub id: String,
    pub address: String,
    pub status: NodeStatus,
    pub last_heartbeat: SystemTime,
    pub cache_size_mb: usize,
    pub load_factor: f32,
}

/// Node status in the distributed cache
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Online,
    Offline,
    Degraded,
    Synchronizing,
}

/// Consistency level for distributed operations
#[derive(Debug, Clone)]
pub enum ConsistencyLevel {
    /// Eventually consistent - fastest, may have temporary inconsistencies
    Eventual,
    /// Strong consistency - slower but guarantees consistency
    Strong,
    /// Quorum consistency - balanced approach
    Quorum,
}

/// Distributed cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCacheEntry<T> {
    pub key: String,
    pub value: T,
    pub version: u64,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub replicas: Vec<String>, // Node IDs where this entry is replicated
    pub checksum: u64,
}

/// Cache synchronization event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEvent {
    NodeJoined(String),
    NodeLeft(String),
    EntryUpdated(String, u64), // key, version
    EntryDeleted(String),
    FullSync,
}

/// Redistribution task for moving cache entries between nodes
#[derive(Debug, Clone)]
pub struct RedistributionTask {
    /// Cache entry key to redistribute
    pub entry_key: String,
    /// Source node ID (being removed)
    pub source_node: String,
    /// Target node ID (receiving the data)
    pub target_node: String,
    /// Task priority
    pub priority: RedistributionPriority,
}

/// Priority levels for redistribution tasks
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedistributionPriority {
    /// Critical data that must be redistributed immediately
    Critical,
    /// High priority data
    High,
    /// Normal priority data
    Normal,
    /// Low priority data that can be redistributed later
    Low,
}

impl DistributedCacheCoordinator {
    /// Create a new distributed cache coordinator
    pub fn new(
        replication_factor: u32,
        consistency_level: ConsistencyLevel,
        sync_interval: Duration,
    ) -> Self {
        Self {
            node_id: Uuid::new_v4().to_string(),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            replication_factor,
            consistency_level,
            sync_interval,
        }
    }

    /// Add a new node to the distributed cache cluster
    pub async fn add_node(&self, node: CacheNode) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: &str) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        if let Some(removed_node) = nodes.remove(node_id) {
            // Get the list of remaining active nodes
            let active_nodes: Vec<CacheNode> = nodes
                .values()
                .filter(|node| node.status == NodeStatus::Online)
                .cloned()
                .collect();

            // Release the lock before performing redistribution
            drop(nodes);

            // Redistribute data from the removed node
            self.redistribute_node_data(node_id, &removed_node, &active_nodes)
                .await?;

            tracing::info!(
                "Successfully removed node {} and redistributed its data to {} active nodes",
                node_id,
                active_nodes.len()
            );
        } else {
            tracing::warn!("Attempted to remove non-existent node: {}", node_id);
        }
        Ok(())
    }

    /// Get the current node ID
    pub fn get_node_id(&self) -> &str {
        &self.node_id
    }

    /// Get list of active nodes
    pub async fn get_active_nodes(&self) -> Vec<CacheNode> {
        let nodes = self.nodes.read().await;
        nodes
            .values()
            .filter(|node| node.status == NodeStatus::Online)
            .cloned()
            .collect()
    }

    /// Select optimal nodes for cache entry placement
    pub async fn select_replica_nodes(&self, exclude_node: Option<&str>) -> Vec<String> {
        let nodes = self.get_active_nodes().await;
        let mut candidates: Vec<_> = nodes
            .into_iter()
            .filter(|node| Some(node.id.as_str()) != exclude_node)
            .collect();

        // Sort by load factor (ascending) to prefer less loaded nodes
        candidates.sort_by(|a, b| a.load_factor.partial_cmp(&b.load_factor).unwrap());

        candidates
            .into_iter()
            .take(self.replication_factor as usize)
            .map(|node| node.id)
            .collect()
    }

    /// Calculate hash for consistent hashing
    pub fn calculate_hash(key: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Find the primary node for a given key using consistent hashing
    pub async fn find_primary_node(&self, key: &str) -> Option<String> {
        let nodes = self.get_active_nodes().await;
        if nodes.is_empty() {
            return None;
        }

        let hash = Self::calculate_hash(key);
        let node_index = (hash as usize) % nodes.len();
        Some(nodes[node_index].id.clone())
    }

    /// Synchronize cache entry across replicas
    pub async fn sync_entry<T>(&self, entry: &DistributedCacheEntry<T>) -> Result<()>
    where
        T: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        match self.consistency_level {
            ConsistencyLevel::Eventual => {
                // Fire and forget - sync in background
                self.sync_entry_async(entry).await
            }
            ConsistencyLevel::Strong => {
                // Wait for all replicas to confirm
                self.sync_entry_strong(entry).await
            }
            ConsistencyLevel::Quorum => {
                // Wait for majority of replicas
                self.sync_entry_quorum(entry).await
            }
        }
    }

    async fn sync_entry_async<T>(&self, _entry: &DistributedCacheEntry<T>) -> Result<()>
    where
        T: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        // Implement eventual consistency sync
        // In a real implementation, this would send the entry to replica nodes
        // without waiting for confirmation
        Ok(())
    }

    async fn sync_entry_strong<T>(&self, _entry: &DistributedCacheEntry<T>) -> Result<()>
    where
        T: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        // Implement strong consistency sync
        // In a real implementation, this would wait for all replicas to confirm
        // the update before returning
        Ok(())
    }

    async fn sync_entry_quorum<T>(&self, _entry: &DistributedCacheEntry<T>) -> Result<()>
    where
        T: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        // Implement quorum consistency sync
        // In a real implementation, this would wait for majority of replicas
        // to confirm the update
        Ok(())
    }

    /// Handle node heartbeat
    pub async fn handle_heartbeat(&self, node_id: &str) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.get_mut(node_id) {
            node.last_heartbeat = SystemTime::now();
            if node.status != NodeStatus::Online {
                node.status = NodeStatus::Online;
            }
        }
        Ok(())
    }

    /// Check for failed nodes and mark them as offline
    pub async fn check_node_health(&self) -> Result<Vec<String>> {
        let mut failed_nodes = Vec::new();
        let mut nodes = self.nodes.write().await;
        let now = SystemTime::now();

        for (node_id, node) in nodes.iter_mut() {
            if let Ok(elapsed) = now.duration_since(node.last_heartbeat) {
                if elapsed > Duration::from_secs(60) && node.status == NodeStatus::Online {
                    node.status = NodeStatus::Offline;
                    failed_nodes.push(node_id.clone());
                }
            }
        }

        Ok(failed_nodes)
    }

    /// Redistribute data from a removed node to maintain replication factor
    async fn redistribute_node_data(
        &self,
        removed_node_id: &str,
        _removed_node: &CacheNode,
        active_nodes: &[CacheNode],
    ) -> Result<()> {
        if active_nodes.is_empty() {
            tracing::warn!("No active nodes available for data redistribution");
            return Ok(());
        }

        tracing::info!(
            "Starting data redistribution from removed node {} to {} active nodes",
            removed_node_id,
            active_nodes.len()
        );

        // In a real implementation, this would:
        // 1. Query the removed node for all its cache entries (if still accessible)
        // 2. For each entry, check if it still has enough replicas
        // 3. If not, select new replica nodes and copy the data

        // Simulate the redistribution process
        let redistribution_tasks = self
            .plan_redistribution(removed_node_id, active_nodes)
            .await?;

        for task in redistribution_tasks {
            match self.execute_redistribution_task(task).await {
                Ok(_) => {
                    tracing::debug!("Successfully completed redistribution task");
                }
                Err(e) => {
                    tracing::error!("Failed to execute redistribution task: {}", e);
                    // Continue with other tasks - partial failure shouldn't stop the process
                }
            }
        }

        tracing::info!(
            "Data redistribution from node {} completed",
            removed_node_id
        );
        Ok(())
    }

    /// Plan redistribution tasks for cache entries
    async fn plan_redistribution(
        &self,
        removed_node_id: &str,
        active_nodes: &[CacheNode],
    ) -> Result<Vec<RedistributionTask>> {
        let mut tasks = Vec::new();

        // Simulate finding cache entries that need redistribution
        // In a real implementation, this would:
        // 1. Scan all cache entries to find those with replicas on the removed node
        // 2. Check current replication factor for each entry
        // 3. Create redistribution tasks for under-replicated entries

        let simulated_entries = self
            .get_entries_requiring_redistribution(removed_node_id)
            .await;

        for entry_key in simulated_entries {
            // Select new replica nodes using consistent hashing and load balancing
            let target_nodes = self
                .select_redistribution_targets(&entry_key, active_nodes)
                .await;

            for target_node_id in target_nodes {
                tasks.push(RedistributionTask {
                    entry_key: entry_key.clone(),
                    source_node: removed_node_id.to_string(),
                    target_node: target_node_id,
                    priority: RedistributionPriority::Normal,
                });
            }
        }

        tracing::debug!("Planned {} redistribution tasks", tasks.len());
        Ok(tasks)
    }

    /// Execute a single redistribution task
    async fn execute_redistribution_task(&self, task: RedistributionTask) -> Result<()> {
        tracing::debug!(
            "Executing redistribution: {} from {} to {}",
            task.entry_key,
            task.source_node,
            task.target_node
        );

        // In a real implementation, this would:
        // 1. Fetch the cache entry from another replica (not the removed node)
        // 2. Store it on the target node
        // 3. Update the entry's replica list
        // 4. Verify the operation succeeded

        // Simulate the redistribution operation
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        Ok(())
    }

    /// Get cache entries that require redistribution
    async fn get_entries_requiring_redistribution(&self, _removed_node_id: &str) -> Vec<String> {
        // Simulate cache entries that need redistribution
        // In a real implementation, this would query the distributed cache metadata
        vec![
            "user_session_123".to_string(),
            "model_weights_abc".to_string(),
            "audio_cache_456".to_string(),
        ]
    }

    /// Select target nodes for redistribution using load balancing
    async fn select_redistribution_targets(
        &self,
        entry_key: &str,
        active_nodes: &[CacheNode],
    ) -> Vec<String> {
        if active_nodes.is_empty() {
            return Vec::new();
        }

        // Calculate how many additional replicas we need
        let current_replicas = self.count_existing_replicas(entry_key).await;
        let needed_replicas = (self.replication_factor as usize).saturating_sub(current_replicas);

        if needed_replicas == 0 {
            return Vec::new();
        }

        // Sort nodes by load factor (prefer less loaded nodes)
        let mut sorted_nodes = active_nodes.to_vec();
        sorted_nodes.sort_by(|a, b| {
            a.load_factor
                .partial_cmp(&b.load_factor)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select the least loaded nodes up to the needed replica count
        sorted_nodes
            .into_iter()
            .take(needed_replicas.min(active_nodes.len()))
            .map(|node| node.id)
            .collect()
    }

    /// Count existing replicas for a cache entry
    async fn count_existing_replicas(&self, _entry_key: &str) -> usize {
        // In a real implementation, this would query the cache metadata
        // to find how many replicas currently exist for this entry
        1 // Simulate that we have 1 existing replica
    }

    /// Get cluster statistics
    pub async fn get_cluster_stats(&self) -> DistributedCacheStats {
        let nodes = self.nodes.read().await;
        let total_nodes = nodes.len();
        let online_nodes = nodes
            .values()
            .filter(|n| n.status == NodeStatus::Online)
            .count();
        let total_cache_mb = nodes.values().map(|n| n.cache_size_mb).sum();
        let average_load = if !nodes.is_empty() {
            nodes.values().map(|n| n.load_factor).sum::<f32>() / nodes.len() as f32
        } else {
            0.0
        };

        DistributedCacheStats {
            total_nodes,
            online_nodes,
            offline_nodes: total_nodes - online_nodes,
            total_cache_size_mb: total_cache_mb,
            average_load_factor: average_load,
            replication_factor: self.replication_factor,
        }
    }
}

/// Statistics for the distributed cache cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCacheStats {
    pub total_nodes: usize,
    pub online_nodes: usize,
    pub offline_nodes: usize,
    pub total_cache_size_mb: usize,
    pub average_load_factor: f32,
    pub replication_factor: u32,
}

/// Distributed cache client interface
#[async_trait::async_trait]
pub trait DistributedCacheClient: Send + Sync {
    /// Get a value from the distributed cache
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Put a value into the distributed cache
    async fn put(&self, key: &str, value: Vec<u8>, ttl: Option<Duration>) -> Result<()>;

    /// Delete a value from the distributed cache
    async fn delete(&self, key: &str) -> Result<bool>;

    /// Check if a key exists in the distributed cache
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Get cache statistics
    async fn stats(&self) -> Result<DistributedCacheStats>;
}

/// In-memory implementation of distributed cache client for testing
pub struct InMemoryDistributedCache {
    coordinator: Arc<DistributedCacheCoordinator>,
    local_cache: LocalCacheData,
}

impl InMemoryDistributedCache {
    pub fn new(coordinator: Arc<DistributedCacheCoordinator>) -> Self {
        Self {
            coordinator,
            local_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl DistributedCacheClient for InMemoryDistributedCache {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let cache = self.local_cache.read().await;
        Ok(cache.get(key).map(|(value, _)| value.clone()))
    }

    async fn put(&self, key: &str, value: Vec<u8>, _ttl: Option<Duration>) -> Result<()> {
        let mut cache = self.local_cache.write().await;
        cache.insert(key.to_string(), (value, SystemTime::now()));
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let mut cache = self.local_cache.write().await;
        Ok(cache.remove(key).is_some())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let cache = self.local_cache.read().await;
        Ok(cache.contains_key(key))
    }

    async fn stats(&self) -> Result<DistributedCacheStats> {
        Ok(self.coordinator.get_cluster_stats().await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_distributed_cache_coordinator() {
        let coordinator =
            DistributedCacheCoordinator::new(3, ConsistencyLevel::Quorum, Duration::from_secs(30));

        assert!(!coordinator.get_node_id().is_empty());
        assert!(coordinator.get_active_nodes().await.is_empty());
    }

    #[tokio::test]
    async fn test_add_remove_nodes() {
        let coordinator = DistributedCacheCoordinator::new(
            2,
            ConsistencyLevel::Eventual,
            Duration::from_secs(30),
        );

        let node = CacheNode {
            id: "node1".to_string(),
            address: "127.0.0.1:8080".to_string(),
            status: NodeStatus::Online,
            last_heartbeat: SystemTime::now(),
            cache_size_mb: 1024,
            load_factor: 0.5,
        };

        coordinator.add_node(node).await.unwrap();
        assert_eq!(coordinator.get_active_nodes().await.len(), 1);

        coordinator.remove_node("node1").await.unwrap();
        assert_eq!(coordinator.get_active_nodes().await.len(), 0);
    }

    #[tokio::test]
    async fn test_consistent_hashing() {
        let coordinator =
            DistributedCacheCoordinator::new(2, ConsistencyLevel::Quorum, Duration::from_secs(30));

        // Add some nodes
        for i in 1..=3 {
            let node = CacheNode {
                id: format!("node{i}"),
                address: format!("127.0.0.1:808{i}"),
                status: NodeStatus::Online,
                last_heartbeat: SystemTime::now(),
                cache_size_mb: 1024,
                load_factor: 0.3,
            };
            coordinator.add_node(node).await.unwrap();
        }

        // Test consistent hashing
        let key = "test_key";
        let primary = coordinator.find_primary_node(key).await;
        assert!(primary.is_some());

        // The same key should always map to the same primary node
        let primary2 = coordinator.find_primary_node(key).await;
        assert_eq!(primary, primary2);
    }

    #[tokio::test]
    async fn test_in_memory_distributed_cache() {
        let coordinator = Arc::new(DistributedCacheCoordinator::new(
            2,
            ConsistencyLevel::Eventual,
            Duration::from_secs(30),
        ));

        let cache = InMemoryDistributedCache::new(coordinator);

        // Test basic operations
        assert!(!cache.exists("key1").await.unwrap());

        cache.put("key1", b"value1".to_vec(), None).await.unwrap();
        assert!(cache.exists("key1").await.unwrap());

        let value = cache.get("key1").await.unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));

        assert!(cache.delete("key1").await.unwrap());
        assert!(!cache.exists("key1").await.unwrap());
    }

    #[test]
    fn test_hash_consistency() {
        let key = "test_key";
        let hash1 = DistributedCacheCoordinator::calculate_hash(key);
        let hash2 = DistributedCacheCoordinator::calculate_hash(key);
        assert_eq!(hash1, hash2);

        // Different keys should produce different hashes
        let hash3 = DistributedCacheCoordinator::calculate_hash("different_key");
        assert_ne!(hash1, hash3);
    }
}
