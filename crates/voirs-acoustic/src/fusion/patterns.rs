//! Fusion Pattern Matching for Kernel Optimization
//! 
//! This module defines patterns for identifying fusible operation sequences
//! in computation graphs and provides matching algorithms.

use std::collections::HashMap;
use crate::error::AcousticError;
use super::graph::{OpGraph, OpNode, FusionGraph, FusionMatch};

/// Defines a pattern of operations that can be fused together
#[derive(Debug, Clone)]
pub struct FusionPattern {
    name: String,
    operations: Vec<String>,
    rule: FusionRule,
    priority: i32,
    constraints: Vec<FusionConstraint>,
}

impl FusionPattern {
    /// Create a new fusion pattern
    pub fn new(name: &str, operations: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            operations: operations.into_iter().map(|s| s.to_string()).collect(),
            rule: FusionRule::Default,
            priority: 0,
            constraints: Vec::new(),
        }
    }

    /// Set the fusion rule
    pub fn with_rule(mut self, rule: FusionRule) -> Self {
        self.rule = rule;
        self
    }

    /// Set the priority (higher = more important)
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: FusionConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Get pattern name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get operations in this pattern
    pub fn operations(&self) -> &[String] {
        &self.operations
    }

    /// Get fusion rule
    pub fn rule(&self) -> &FusionRule {
        &self.rule
    }

    /// Get priority
    pub fn priority(&self) -> i32 {
        self.priority
    }

    /// Check if this pattern matches a sequence of operations
    pub fn matches(&self, ops: &[&str]) -> bool {
        if ops.len() != self.operations.len() {
            return false;
        }

        ops.iter().zip(self.operations.iter())
            .all(|(op, pattern_op)| op == pattern_op || pattern_op == "*")
    }

    /// Estimate speedup for applying this pattern
    pub fn estimate_speedup(&self, nodes: &[&OpNode]) -> f32 {
        match self.rule {
            FusionRule::ElementWise => self.estimate_elementwise_speedup(nodes),
            FusionRule::Pointwise => self.estimate_pointwise_speedup(nodes),
            FusionRule::ConvolutionBased => self.estimate_conv_speedup(nodes),
            FusionRule::LinearAlgebra => self.estimate_linear_algebra_speedup(nodes),
            FusionRule::Normalization => self.estimate_normalization_speedup(nodes),
            FusionRule::Default => 1.2, // Conservative default
        }
    }

    /// Check if constraints are satisfied
    pub fn check_constraints(&self, nodes: &[&OpNode]) -> bool {
        self.constraints.iter().all(|constraint| constraint.check(nodes))
    }

    /// Estimate speedup for element-wise operations
    fn estimate_elementwise_speedup(&self, nodes: &[&OpNode]) -> f32 {
        // Element-wise operations benefit from memory bandwidth reduction
        let memory_ops = nodes.len() as f32;
        let compute_ops = nodes.iter().map(|n| n.compute_cost()).sum::<f32>();
        
        // Memory-bound operations see higher speedup
        let memory_bandwidth_factor = 1.5;
        let fusion_overhead = 0.9; // Small overhead for fusion
        
        (memory_ops * memory_bandwidth_factor * fusion_overhead).min(3.0)
    }

    /// Estimate speedup for pointwise operations
    fn estimate_pointwise_speedup(&self, nodes: &[&OpNode]) -> f32 {
        // Pointwise operations (like activations) benefit from register reuse
        let base_speedup = 1.3;
        let complexity_factor = nodes.len() as f32 * 0.1;
        
        (base_speedup + complexity_factor).min(2.5)
    }

    /// Estimate speedup for convolution-based operations
    fn estimate_conv_speedup(&self, nodes: &[&OpNode]) -> f32 {
        // Convolution fusion can be very beneficial
        let has_conv = nodes.iter().any(|n| n.op_type().contains("conv"));
        let has_bias = nodes.iter().any(|n| n.op_type() == "add");
        let has_activation = nodes.iter().any(|n| 
            matches!(n.op_type(), "relu" | "tanh" | "sigmoid" | "gelu"));

        let mut speedup = 1.0;
        if has_conv {
            speedup *= 1.4;
        }
        if has_bias {
            speedup *= 1.2;
        }
        if has_activation {
            speedup *= 1.3;
        }

        speedup.min(3.5)
    }

    /// Estimate speedup for linear algebra operations
    fn estimate_linear_algebra_speedup(&self, nodes: &[&OpNode]) -> f32 {
        // Matrix operations can benefit from better cache utilization
        let matmul_count = nodes.iter().filter(|n| n.op_type() == "matmul").count();
        
        if matmul_count >= 2 {
            2.0 + (matmul_count as f32 - 2.0) * 0.3
        } else {
            1.1
        }
    }

    /// Estimate speedup for normalization operations
    fn estimate_normalization_speedup(&self, nodes: &[&OpNode]) -> f32 {
        // Normalization patterns can be quite beneficial
        let reduction_ops = nodes.iter().filter(|n| n.is_reduction()).count();
        let elementwise_ops = nodes.iter().filter(|n| n.is_elementwise()).count();
        
        1.5 + (reduction_ops as f32 * 0.3) + (elementwise_ops as f32 * 0.1)
    }
}

/// Types of fusion rules that determine how operations can be combined
#[derive(Debug, Clone, PartialEq)]
pub enum FusionRule {
    /// Default fusion rule
    Default,
    /// Element-wise operations that work on individual elements
    ElementWise,
    /// Point-wise operations like activations
    Pointwise,
    /// Convolution-based operations
    ConvolutionBased,
    /// Linear algebra operations
    LinearAlgebra,
    /// Normalization operations
    Normalization,
}

/// Constraints that must be satisfied for a fusion to be valid
#[derive(Debug, Clone)]
pub enum FusionConstraint {
    /// Maximum number of operations in fusion
    MaxOperations(usize),
    /// All operations must have same output shape
    SameOutputShape,
    /// Operations must be element-wise compatible
    ElementWiseCompatible,
    /// Maximum memory overhead allowed
    MaxMemoryOverhead(f32),
    /// Minimum expected speedup
    MinSpeedup(f32),
    /// Custom constraint with validation function
    Custom(String),
}

impl FusionConstraint {
    /// Check if this constraint is satisfied by the given nodes
    pub fn check(&self, nodes: &[&OpNode]) -> bool {
        match self {
            FusionConstraint::MaxOperations(max) => nodes.len() <= *max,
            
            FusionConstraint::SameOutputShape => {
                if nodes.is_empty() {
                    return true;
                }
                let first_shape = nodes[0].output_shape();
                nodes.iter().all(|node| node.output_shape() == first_shape)
            },
            
            FusionConstraint::ElementWiseCompatible => {
                nodes.iter().all(|node| node.is_elementwise())
            },
            
            FusionConstraint::MaxMemoryOverhead(max_overhead) => {
                // Simplified memory overhead estimation
                let total_memory = nodes.iter()
                    .map(|node| node.output_shape().elem_count() as f32 * 4.0) // assume f32
                    .sum::<f32>();
                let overhead = total_memory * 0.1; // Estimated 10% overhead
                overhead / total_memory <= *max_overhead
            },
            
            FusionConstraint::MinSpeedup(min_speedup) => {
                // This would need actual speedup estimation
                *min_speedup <= 2.0 // Placeholder
            },
            
            FusionConstraint::Custom(_) => {
                // Custom constraints would need specialized handling
                true
            },
        }
    }
}

/// Pattern matcher for finding fusible operation sequences
#[derive(Debug)]
pub struct PatternMatcher<'a> {
    patterns: &'a [FusionPattern],
}

impl<'a> PatternMatcher<'a> {
    /// Create a new pattern matcher
    pub fn new(patterns: &'a [FusionPattern]) -> Self {
        Self { patterns }
    }

    /// Find all matching patterns in the fusion graph
    pub fn find_matches(&self, graph: &FusionGraph) -> Result<Option<Vec<FusionMatch>>, AcousticError> {
        let mut matches = Vec::new();
        let op_graph = graph.op_graph();
        
        // Get topologically sorted nodes
        let sorted_nodes = op_graph.topological_sort()?;
        
        // Try to match patterns starting from each node
        for &start_node in &sorted_nodes {
            for pattern in self.patterns {
                if let Some(fusion_match) = self.try_match_pattern(pattern, start_node, op_graph)? {
                    matches.push(fusion_match);
                }
            }
        }

        // Sort matches by priority and estimated speedup
        matches.sort_by(|a, b| {
            b.estimated_speedup.partial_cmp(&a.estimated_speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if matches.is_empty() {
            Ok(None)
        } else {
            Ok(Some(matches))
        }
    }

    /// Try to match a specific pattern starting from a node
    fn try_match_pattern(
        &self,
        pattern: &FusionPattern,
        start_node: usize,
        graph: &OpGraph,
    ) -> Result<Option<FusionMatch>, AcousticError> {
        let mut current_nodes = vec![start_node];
        let mut matched_ops = Vec::new();

        // Try to find a sequence matching the pattern
        for target_op in pattern.operations() {
            if let Some(node) = graph.get_node(start_node + matched_ops.len()) {
                if node.op_type() == target_op || target_op == "*" {
                    matched_ops.push(node.op_type().to_string());
                    current_nodes.push(node.id());
                } else {
                    return Ok(None); // Pattern doesn't match
                }
            } else {
                return Ok(None); // Not enough nodes
            }
        }

        // Check if pattern matches
        let op_refs: Vec<&str> = matched_ops.iter().map(|s| s.as_str()).collect();
        if !pattern.matches(&op_refs) {
            return Ok(None);
        }

        // Get node references for constraint checking
        let node_refs: Vec<&OpNode> = current_nodes.iter()
            .filter_map(|&id| graph.get_node(id))
            .collect();

        // Check constraints
        if !pattern.check_constraints(&node_refs) {
            return Ok(None);
        }

        // Estimate speedup
        let speedup = pattern.estimate_speedup(&node_refs);

        Ok(Some(FusionMatch::new(
            current_nodes,
            pattern.name().to_string(),
            speedup,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Shape};

    #[test]
    fn test_fusion_pattern_creation() {
        let pattern = FusionPattern::new("add_mul", vec!["add", "mul"])
            .with_rule(FusionRule::ElementWise)
            .with_priority(10);

        assert_eq!(pattern.name(), "add_mul");
        assert_eq!(pattern.operations(), &["add", "mul"]);
        assert_eq!(pattern.rule(), &FusionRule::ElementWise);
        assert_eq!(pattern.priority(), 10);
    }

    #[test]
    fn test_pattern_matching() {
        let pattern = FusionPattern::new("add_mul", vec!["add", "mul"]);
        
        assert!(pattern.matches(&["add", "mul"]));
        assert!(!pattern.matches(&["mul", "add"]));
        assert!(!pattern.matches(&["add"]));
        assert!(!pattern.matches(&["add", "mul", "sub"]));
    }

    #[test]
    fn test_wildcard_pattern() {
        let pattern = FusionPattern::new("any_relu", vec!["*", "relu"]);
        
        assert!(pattern.matches(&["add", "relu"]));
        assert!(pattern.matches(&["mul", "relu"]));
        assert!(!pattern.matches(&["add", "tanh"]));
    }

    #[test]
    fn test_fusion_constraints() {
        let shape = Shape::from_dims(&[10, 20]);
        let node1 = OpNode::new(0, "add".to_string(), shape.clone(), DType::F32);
        let node2 = OpNode::new(1, "mul".to_string(), shape, DType::F32);
        let nodes = vec![&node1, &node2];

        let max_ops = FusionConstraint::MaxOperations(2);
        assert!(max_ops.check(&nodes));

        let max_ops_fail = FusionConstraint::MaxOperations(1);
        assert!(!max_ops_fail.check(&nodes));

        let same_shape = FusionConstraint::SameOutputShape;
        assert!(same_shape.check(&nodes));

        let elementwise = FusionConstraint::ElementWiseCompatible;
        assert!(elementwise.check(&nodes));
    }

    #[test]
    fn test_speedup_estimation() {
        let pattern = FusionPattern::new("add_mul", vec!["add", "mul"])
            .with_rule(FusionRule::ElementWise);

        let shape = Shape::from_dims(&[1000]);
        let node1 = OpNode::new(0, "add".to_string(), shape.clone(), DType::F32);
        let node2 = OpNode::new(1, "mul".to_string(), shape, DType::F32);
        let nodes = vec![&node1, &node2];

        let speedup = pattern.estimate_speedup(&nodes);
        assert!(speedup > 1.0);
        assert!(speedup <= 3.0);
    }

    #[test]
    fn test_pattern_with_constraints() {
        let pattern = FusionPattern::new("elementwise_pair", vec!["add", "mul"])
            .with_rule(FusionRule::ElementWise)
            .with_constraint(FusionConstraint::MaxOperations(2))
            .with_constraint(FusionConstraint::ElementWiseCompatible);

        let shape = Shape::from_dims(&[10]);
        let node1 = OpNode::new(0, "add".to_string(), shape.clone(), DType::F32);
        let node2 = OpNode::new(1, "mul".to_string(), shape, DType::F32);
        let nodes = vec![&node1, &node2];

        assert!(pattern.check_constraints(&nodes));
    }
}