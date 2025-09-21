//! NUMA-aware processing for optimized memory access patterns
//!
//! Provides utilities for detecting NUMA topology and optimizing
//! thread placement and memory allocation for better performance.

use crate::{Result, VocoderError};
use std::collections::HashMap;

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: usize,
    pub cpu_cores: Vec<usize>,
    pub memory_size_mb: u64,
    pub is_local: bool,
}

impl NumaNode {
    pub fn new(node_id: usize, cpu_cores: Vec<usize>, memory_size_mb: u64) -> Self {
        Self {
            node_id,
            cpu_cores,
            memory_size_mb,
            is_local: true,
        }
    }

    /// Get the number of CPU cores in this NUMA node
    pub fn core_count(&self) -> usize {
        self.cpu_cores.len()
    }

    /// Check if a CPU core belongs to this NUMA node
    pub fn contains_core(&self, core_id: usize) -> bool {
        self.cpu_cores.contains(&core_id)
    }
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
    pub total_cores: usize,
    pub total_memory_mb: u64,
    pub numa_enabled: bool,
}

impl NumaTopology {
    /// Detect NUMA topology on the current system
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self::detect_fallback()
        }
    }

    /// Linux-specific NUMA detection
    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        use std::fs;
        use std::path::Path;

        let numa_path = Path::new("/sys/devices/system/node");

        if !numa_path.exists() {
            return Self::detect_fallback();
        }

        let mut nodes = Vec::new();
        let mut total_cores = 0;
        let mut total_memory = 0;

        // Read NUMA node directories
        if let Ok(entries) = fs::read_dir(numa_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("node") {
                        if let Ok(node_id) = name[4..].parse::<usize>() {
                            let cpu_cores = Self::read_node_cpus(&path);
                            let memory_mb = Self::read_node_memory(&path);

                            total_cores += cpu_cores.len();
                            total_memory += memory_mb;

                            nodes.push(NumaNode::new(node_id, cpu_cores, memory_mb));
                        }
                    }
                }
            }
        }

        // If no NUMA nodes found, fall back
        if nodes.is_empty() {
            return Self::detect_fallback();
        }

        Self {
            nodes,
            total_cores,
            total_memory_mb: total_memory,
            numa_enabled: true,
        }
    }

    #[cfg(target_os = "linux")]
    fn read_node_cpus(node_path: &std::path::Path) -> Vec<usize> {
        use std::fs;

        let cpulist_path = node_path.join("cpulist");

        if let Ok(content) = fs::read_to_string(cpulist_path) {
            Self::parse_cpu_list(&content.trim())
        } else {
            Vec::new()
        }
    }

    #[cfg(target_os = "linux")]
    fn read_node_memory(node_path: &std::path::Path) -> u64 {
        use std::fs;

        let meminfo_path = node_path.join("meminfo");

        if let Ok(content) = fs::read_to_string(meminfo_path) {
            for line in content.lines() {
                if line.starts_with("Node") && line.contains("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(2) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb / 1024; // Convert KB to MB
                        }
                    }
                }
            }
        }

        0
    }

    fn parse_cpu_list(cpu_list: &str) -> Vec<usize> {
        let mut cpus = Vec::new();

        for range in cpu_list.split(',') {
            let range = range.trim();
            if range.contains('-') {
                let parts: Vec<&str> = range.split('-').collect();
                if parts.len() == 2 {
                    if let (Ok(start), Ok(end)) =
                        (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                    {
                        for cpu in start..=end {
                            cpus.push(cpu);
                        }
                    }
                }
            } else if let Ok(cpu) = range.parse::<usize>() {
                cpus.push(cpu);
            }
        }

        cpus
    }

    /// Fallback detection for non-Linux systems
    fn detect_fallback() -> Self {
        let num_cpus = num_cpus::get();
        let cpu_cores: Vec<usize> = (0..num_cpus).collect();

        // Create a single NUMA node containing all cores
        let node = NumaNode::new(0, cpu_cores, 0);

        Self {
            nodes: vec![node],
            total_cores: num_cpus,
            total_memory_mb: 0,
            numa_enabled: false,
        }
    }

    /// Get NUMA node for a specific CPU core
    pub fn node_for_core(&self, core_id: usize) -> Option<&NumaNode> {
        self.nodes.iter().find(|node| node.contains_core(core_id))
    }

    /// Get the best NUMA node for a given number of threads
    pub fn best_node_for_threads(&self, num_threads: usize) -> Option<&NumaNode> {
        self.nodes
            .iter()
            .filter(|node| node.core_count() >= num_threads)
            .min_by_key(|node| node.core_count())
    }

    /// Get CPU affinity recommendation for a thread pool
    pub fn thread_affinity_recommendation(&self, num_threads: usize) -> Vec<usize> {
        if !self.numa_enabled || self.nodes.is_empty() {
            // No NUMA, just return sequential cores
            return (0..num_threads.min(self.total_cores)).collect();
        }

        let mut recommended_cores = Vec::new();
        let mut remaining_threads = num_threads;

        // Try to place threads within NUMA nodes
        for node in &self.nodes {
            if remaining_threads == 0 {
                break;
            }

            let cores_to_use = remaining_threads.min(node.core_count());
            recommended_cores.extend(node.cpu_cores.iter().take(cores_to_use).cloned());
            remaining_threads -= cores_to_use;
        }

        // If we still need more threads, wrap around
        while recommended_cores.len() < num_threads && recommended_cores.len() < self.total_cores {
            let next_core = recommended_cores.len() % self.total_cores;
            if !recommended_cores.contains(&next_core) {
                recommended_cores.push(next_core);
            } else {
                break;
            }
        }

        recommended_cores
    }

    /// Check if NUMA is available and beneficial
    pub fn is_numa_beneficial(&self) -> bool {
        self.numa_enabled && self.nodes.len() > 1
    }
}

/// NUMA-aware thread manager
pub struct NumaThreadManager {
    topology: NumaTopology,
    thread_assignments: HashMap<std::thread::ThreadId, usize>, // Thread ID -> NUMA node
}

impl NumaThreadManager {
    /// Create a new NUMA-aware thread manager
    pub fn new() -> Self {
        let topology = NumaTopology::detect();

        Self {
            topology,
            thread_assignments: HashMap::new(),
        }
    }

    /// Get the NUMA topology
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }

    /// Set CPU affinity for the current thread
    pub fn set_thread_affinity(&self, cores: &[usize]) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.set_affinity_linux(cores)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = cores; // Suppress unused parameter warning
                           // Affinity setting not supported on this platform
            Ok(())
        }
    }

    /// Set CPU affinity for the current thread from a CPU list string
    /// Format: "0-3,8,12-15" (ranges and individual CPUs separated by commas)
    /// This method uses the internal parse_cpu_list function
    pub fn set_thread_affinity_from_string(&self, cpu_list: &str) -> Result<()> {
        let cores = NumaTopology::parse_cpu_list(cpu_list);
        self.set_thread_affinity(&cores)
    }

    #[cfg(target_os = "linux")]
    fn set_affinity_linux(&self, cores: &[usize]) -> Result<()> {
        use std::mem;

        // This is a simplified version - in a real implementation,
        // you'd use libc bindings to call sched_setaffinity

        // For now, just validate the cores exist
        for &core in cores {
            if core >= self.topology.total_cores {
                return Err(VocoderError::ProcessingError(format!(
                    "Invalid core ID: {}",
                    core
                )));
            }
        }

        Ok(())
    }

    /// Bind thread to a specific NUMA node
    pub fn bind_to_numa_node(&mut self, node_id: usize) -> Result<()> {
        let node = self
            .topology
            .nodes
            .iter()
            .find(|n| n.node_id == node_id)
            .ok_or_else(|| {
                VocoderError::ProcessingError(format!("NUMA node {node_id} not found"))
            })?;

        let thread_id = std::thread::current().id();
        self.thread_assignments.insert(thread_id, node_id);

        // Set CPU affinity to this node's cores
        self.set_thread_affinity(&node.cpu_cores)
    }

    /// Get optimal thread distribution across NUMA nodes
    pub fn optimal_thread_distribution(&self, total_threads: usize) -> Vec<NumaThreadDistribution> {
        if !self.topology.is_numa_beneficial() {
            // Single node or no NUMA, put all threads on node 0
            return vec![NumaThreadDistribution {
                node_id: 0,
                thread_count: total_threads,
                core_assignments: self.topology.thread_affinity_recommendation(total_threads),
            }];
        }

        let mut distributions = Vec::new();
        let threads_per_node = total_threads / self.topology.nodes.len();
        let extra_threads = total_threads % self.topology.nodes.len();

        for (i, node) in self.topology.nodes.iter().enumerate() {
            let mut thread_count = threads_per_node;
            if i < extra_threads {
                thread_count += 1; // Distribute remainder evenly
            }

            thread_count = thread_count.min(node.core_count());

            if thread_count > 0 {
                let core_assignments = node.cpu_cores.iter().take(thread_count).cloned().collect();

                distributions.push(NumaThreadDistribution {
                    node_id: node.node_id,
                    thread_count,
                    core_assignments,
                });
            }
        }

        distributions
    }

    /// Allocate memory with NUMA awareness
    pub fn allocate_numa_memory(&self, size: usize, _node_id: Option<usize>) -> Result<Vec<u8>> {
        // In a real implementation, this would use numa_alloc_onnode() or similar
        // For now, just allocate regular memory
        let memory = vec![0; size];
        Ok(memory)
    }

    /// Get current thread's NUMA node assignment
    pub fn current_thread_numa_node(&self) -> Option<usize> {
        let thread_id = std::thread::current().id();
        self.thread_assignments.get(&thread_id).copied()
    }
}

impl Default for NumaThreadManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread distribution across NUMA nodes
#[derive(Debug, Clone)]
pub struct NumaThreadDistribution {
    pub node_id: usize,
    pub thread_count: usize,
    pub core_assignments: Vec<usize>,
}

impl NumaThreadDistribution {
    /// Get the CPU cores for this distribution
    pub fn cores(&self) -> &[usize] {
        &self.core_assignments
    }

    /// Get the number of threads for this node
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_node_creation() {
        let node = NumaNode::new(0, vec![0, 1, 2, 3], 8192);
        assert_eq!(node.node_id, 0);
        assert_eq!(node.core_count(), 4);
        assert!(node.contains_core(2));
        assert!(!node.contains_core(5));
    }

    #[test]
    fn test_cpu_list_parsing() {
        assert_eq!(NumaTopology::parse_cpu_list("0,1,2,3"), vec![0, 1, 2, 3]);
        assert_eq!(NumaTopology::parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(NumaTopology::parse_cpu_list("0,2-4,7"), vec![0, 2, 3, 4, 7]);
        assert_eq!(NumaTopology::parse_cpu_list(""), Vec::<usize>::new());
    }

    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect();
        assert!(!topology.nodes.is_empty());
        assert!(topology.total_cores > 0);
    }

    #[test]
    fn test_thread_affinity_recommendation() {
        let topology = NumaTopology::detect();
        let recommendations = topology.thread_affinity_recommendation(4);

        assert!(!recommendations.is_empty());
        assert!(recommendations.len() <= 4);
        assert!(recommendations.len() <= topology.total_cores);
    }

    #[test]
    fn test_numa_thread_manager_creation() {
        let manager = NumaThreadManager::new();
        assert!(!manager.topology().nodes.is_empty());
    }

    #[test]
    fn test_optimal_thread_distribution() {
        let manager = NumaThreadManager::new();
        let distributions = manager.optimal_thread_distribution(8);

        assert!(!distributions.is_empty());

        let total_assigned: usize = distributions.iter().map(|d| d.thread_count).sum();
        assert!(total_assigned <= 8);
    }

    #[test]
    fn test_numa_memory_allocation() {
        let manager = NumaThreadManager::new();
        let memory = manager.allocate_numa_memory(1024, None);

        assert!(memory.is_ok());
        let mem = memory.unwrap();
        assert_eq!(mem.len(), 1024);
    }

    #[test]
    fn test_node_for_core() {
        let topology = NumaTopology::detect();

        if let Some(first_node) = topology.nodes.first() {
            if let Some(first_core) = first_node.cpu_cores.first() {
                let found_node = topology.node_for_core(*first_core);
                assert!(found_node.is_some());
                assert_eq!(found_node.unwrap().node_id, first_node.node_id);
            }
        }
    }

    #[test]
    fn test_best_node_for_threads() {
        let topology = NumaTopology::detect();

        if topology.nodes.len() > 1 {
            let best_node = topology.best_node_for_threads(2);
            assert!(best_node.is_some());
            assert!(best_node.unwrap().core_count() >= 2);
        }
    }

    #[test]
    fn test_numa_beneficial() {
        let topology = NumaTopology::detect();

        // Should be consistent with detection
        if topology.nodes.len() > 1 && topology.numa_enabled {
            assert!(topology.is_numa_beneficial());
        } else {
            assert!(!topology.is_numa_beneficial());
        }
    }
}
