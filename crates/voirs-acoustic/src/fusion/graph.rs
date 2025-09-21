//! Operation Graph Representation for Kernel Fusion
//! 
//! This module provides data structures and algorithms for representing
//! and manipulating computational graphs for kernel fusion optimization.

use std::collections::{HashMap, HashSet, VecDeque};
use candle_core::{DType, Shape};
use crate::error::AcousticError;

/// Represents a single operation node in the computation graph
#[derive(Debug, Clone, PartialEq)]
pub struct OpNode {
    id: usize,
    op_type: String,
    input_shapes: Vec<Shape>,
    output_shape: Shape,
    dtype: DType,
    parameters: HashMap<String, OpParameter>,
    dependencies: Vec<usize>,
}

impl OpNode {
    /// Create a new operation node
    pub fn new(id: usize, op_type: String, output_shape: Shape, dtype: DType) -> Self {
        Self {
            id,
            op_type,
            input_shapes: Vec::new(),
            output_shape,
            dtype,
            parameters: HashMap::new(),
            dependencies: Vec::new(),
        }
    }

    /// Set input shapes for this operation
    pub fn with_input_shapes(mut self, shapes: Vec<Shape>) -> Self {
        self.input_shapes = shapes;
        self
    }

    /// Add a parameter to this operation
    pub fn with_parameter(mut self, name: String, value: OpParameter) -> Self {
        self.parameters.insert(name, value);
        self
    }

    /// Add dependencies (input node IDs)
    pub fn with_dependencies(mut self, deps: Vec<usize>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Get operation type
    pub fn op_type(&self) -> &str {
        &self.op_type
    }

    /// Get input shapes
    pub fn input_shapes(&self) -> &[Shape] {
        &self.input_shapes
    }

    /// Get output shape
    pub fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get node ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get dependencies
    pub fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    /// Get parameter value
    pub fn parameter(&self, name: &str) -> Option<&OpParameter> {
        self.parameters.get(name)
    }

    /// Check if this operation is element-wise
    pub fn is_elementwise(&self) -> bool {
        matches!(self.op_type.as_str(), "add" | "mul" | "sub" | "div" | "relu" | "tanh" | "sigmoid" | "gelu")
    }

    /// Check if this operation is a reduction
    pub fn is_reduction(&self) -> bool {
        matches!(self.op_type.as_str(), "sum" | "mean" | "max" | "min" | "argmax" | "argmin")
    }

    /// Estimate computational cost
    pub fn compute_cost(&self) -> f32 {
        let output_size = self.output_shape.elem_count() as f32;
        
        match self.op_type.as_str() {
            "add" | "sub" | "mul" | "div" => output_size,
            "relu" | "tanh" | "sigmoid" => output_size * 2.0,
            "gelu" => output_size * 4.0,
            "matmul" => {
                if self.input_shapes.len() >= 2 {
                    let a_shape = &self.input_shapes[0];
                    let b_shape = &self.input_shapes[1];
                    if a_shape.dims().len() >= 2 && b_shape.dims().len() >= 2 {
                        let m = a_shape.dims()[a_shape.dims().len()-2] as f32;
                        let k = a_shape.dims()[a_shape.dims().len()-1] as f32;
                        let n = b_shape.dims()[b_shape.dims().len()-1] as f32;
                        2.0 * m * k * n
                    } else {
                        output_size * 2.0
                    }
                } else {
                    output_size * 2.0
                }
            },
            "conv1d" => {
                // Simplified cost estimation for 1D convolution
                output_size * 10.0
            },
            _ => output_size * 1.5,
        }
    }
}

/// Parameter types for operations
#[derive(Debug, Clone, PartialEq)]
pub enum OpParameter {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
}

/// Represents a computational graph of operations
#[derive(Debug, Clone)]
pub struct OpGraph {
    nodes: HashMap<usize, OpNode>,
    edges: HashMap<usize, Vec<usize>>, // node_id -> list of dependent node_ids
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    next_id: usize,
}

impl OpGraph {
    /// Create a new empty operation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, mut node: OpNode) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        node.id = id;
        self.nodes.insert(id, node);
        self.edges.insert(id, Vec::new());
        
        id
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), AcousticError> {
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) {
            return Err(AcousticError::GraphError {
                message: "Node not found in graph".to_string(),
            });
        }

        if let Some(edges) = self.edges.get_mut(&from) {
            if !edges.contains(&to) {
                edges.push(to);
            }
        }

        // Add to dependencies of target node
        if let Some(node) = self.nodes.get_mut(&to) {
            if !node.dependencies.contains(&from) {
                node.dependencies.push(from);
            }
        }

        Ok(())
    }

    /// Set input nodes
    pub fn set_inputs(&mut self, inputs: Vec<usize>) {
        self.inputs = inputs;
    }

    /// Set output nodes
    pub fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.outputs = outputs;
    }

    /// Get a node by ID
    pub fn get_node(&self, id: usize) -> Option<&OpNode> {
        self.nodes.get(&id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<usize, OpNode> {
        &self.nodes
    }

    /// Get successors of a node
    pub fn successors(&self, id: usize) -> Option<&Vec<usize>> {
        self.edges.get(&id)
    }

    /// Get input nodes
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }

    /// Get output nodes
    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }

    /// Perform topological sort of the graph
    pub fn topological_sort(&self) -> Result<Vec<usize>, AcousticError> {
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Calculate in-degrees
        for &node_id in self.nodes.keys() {
            in_degree.insert(node_id, 0);
        }

        for edges in self.edges.values() {
            for &to in edges {
                *in_degree.get_mut(&to).unwrap() += 1;
            }
        }

        // Find nodes with in-degree 0
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        // Process nodes
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            if let Some(successors) = self.edges.get(&node_id) {
                for &successor in successors {
                    let degree = in_degree.get_mut(&successor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(successor);
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(AcousticError::GraphError {
                message: "Graph contains cycles".to_string(),
            });
        }

        Ok(result)
    }

    /// Find strongly connected components
    pub fn find_cycles(&self) -> Vec<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut cycles = Vec::new();

        for &node_id in self.nodes.keys() {
            if !visited.contains(&node_id) {
                self.dfs_cycles(node_id, &mut visited, &mut rec_stack, &mut cycles, &mut Vec::new());
            }
        }

        cycles
    }

    /// DFS helper for cycle detection
    fn dfs_cycles(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        rec_stack: &mut HashSet<usize>,
        cycles: &mut Vec<Vec<usize>>,
        current_path: &mut Vec<usize>,
    ) {
        visited.insert(node);
        rec_stack.insert(node);
        current_path.push(node);

        if let Some(successors) = self.edges.get(&node) {
            for &successor in successors {
                if !visited.contains(&successor) {
                    self.dfs_cycles(successor, visited, rec_stack, cycles, current_path);
                } else if rec_stack.contains(&successor) {
                    // Found a cycle
                    if let Some(start_idx) = current_path.iter().position(|&x| x == successor) {
                        cycles.push(current_path[start_idx..].to_vec());
                    }
                }
            }
        }

        current_path.pop();
        rec_stack.remove(&node);
    }
}

impl Default for OpGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Fusion graph with grouped operations for kernel fusion
#[derive(Debug, Clone)]
pub struct FusionGraph {
    op_graph: OpGraph,
    fusion_groups: Vec<FusionGroup>,
}

impl FusionGraph {
    /// Create fusion graph from operation graph
    pub fn from_op_graph(op_graph: OpGraph) -> Self {
        Self {
            op_graph,
            fusion_groups: Vec::new(),
        }
    }

    /// Apply a fusion transformation
    pub fn apply_fusion(&mut self, fusion_match: FusionMatch) -> Result<(), AcousticError> {
        let group = FusionGroup {
            id: self.fusion_groups.len(),
            nodes: fusion_match.nodes,
            pattern: fusion_match.pattern,
            estimated_speedup: fusion_match.estimated_speedup,
        };

        self.fusion_groups.push(group);
        Ok(())
    }

    /// Get the underlying operation graph
    pub fn op_graph(&self) -> &OpGraph {
        &self.op_graph
    }

    /// Get fusion groups
    pub fn fusion_groups(&self) -> &[FusionGroup] {
        &self.fusion_groups
    }

    /// Calculate total estimated speedup
    pub fn total_speedup(&self) -> f32 {
        self.fusion_groups.iter()
            .map(|group| group.estimated_speedup)
            .sum::<f32>()
            .max(1.0)
    }
}

/// Represents a group of operations that can be fused together
#[derive(Debug, Clone)]
pub struct FusionGroup {
    pub id: usize,
    pub nodes: Vec<usize>,
    pub pattern: String,
    pub estimated_speedup: f32,
}

/// Represents a matched fusion pattern in the graph
#[derive(Debug, Clone)]
pub struct FusionMatch {
    pub nodes: Vec<usize>,
    pub pattern: String,
    pub estimated_speedup: f32,
}

impl FusionMatch {
    /// Create a new fusion match
    pub fn new(nodes: Vec<usize>, pattern: String, estimated_speedup: f32) -> Self {
        Self {
            nodes,
            pattern,
            estimated_speedup,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Shape};

    #[test]
    fn test_op_node_creation() {
        let shape = Shape::from_dims(&[10, 20]);
        let node = OpNode::new(0, "add".to_string(), shape.clone(), DType::F32);
        
        assert_eq!(node.id(), 0);
        assert_eq!(node.op_type(), "add");
        assert_eq!(node.output_shape(), &shape);
        assert_eq!(node.dtype(), DType::F32);
        assert!(node.is_elementwise());
        assert!(!node.is_reduction());
    }

    #[test]
    fn test_op_node_with_parameters() {
        let shape = Shape::from_dims(&[10]);
        let node = OpNode::new(0, "conv1d".to_string(), shape, DType::F32)
            .with_parameter("kernel_size".to_string(), OpParameter::Int(3))
            .with_parameter("stride".to_string(), OpParameter::Int(1));
        
        assert_eq!(node.parameter("kernel_size"), Some(&OpParameter::Int(3)));
        assert_eq!(node.parameter("stride"), Some(&OpParameter::Int(1)));
        assert_eq!(node.parameter("nonexistent"), None);
    }

    #[test]
    fn test_op_graph_creation() {
        let mut graph = OpGraph::new();
        
        let shape = Shape::from_dims(&[10, 20]);
        let node1 = OpNode::new(0, "input".to_string(), shape.clone(), DType::F32);
        let node2 = OpNode::new(1, "relu".to_string(), shape, DType::F32);
        
        let id1 = graph.add_node(node1);
        let id2 = graph.add_node(node2);
        
        assert!(graph.add_edge(id1, id2).is_ok());
        assert_eq!(graph.nodes().len(), 2);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = OpGraph::new();
        
        let shape = Shape::from_dims(&[10]);
        let node1 = OpNode::new(0, "input".to_string(), shape.clone(), DType::F32);
        let node2 = OpNode::new(1, "relu".to_string(), shape.clone(), DType::F32);
        let node3 = OpNode::new(2, "output".to_string(), shape, DType::F32);
        
        let id1 = graph.add_node(node1);
        let id2 = graph.add_node(node2);
        let id3 = graph.add_node(node3);
        
        graph.add_edge(id1, id2).unwrap();
        graph.add_edge(id2, id3).unwrap();
        
        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted.len(), 3);
        
        // Check that dependencies come before their dependents
        let pos1 = sorted.iter().position(|&x| x == id1).unwrap();
        let pos2 = sorted.iter().position(|&x| x == id2).unwrap();
        let pos3 = sorted.iter().position(|&x| x == id3).unwrap();
        
        assert!(pos1 < pos2);
        assert!(pos2 < pos3);
    }

    #[test]
    fn test_fusion_graph() {
        let op_graph = OpGraph::new();
        let fusion_graph = FusionGraph::from_op_graph(op_graph);
        
        assert_eq!(fusion_graph.fusion_groups().len(), 0);
        assert_eq!(fusion_graph.total_speedup(), 1.0);
    }

    #[test]
    fn test_compute_cost() {
        let shape = Shape::from_dims(&[1000]);
        
        let add_node = OpNode::new(0, "add".to_string(), shape.clone(), DType::F32);
        let matmul_shapes = vec![Shape::from_dims(&[100, 200]), Shape::from_dims(&[200, 300])];
        let matmul_node = OpNode::new(1, "matmul".to_string(), Shape::from_dims(&[100, 300]), DType::F32)
            .with_input_shapes(matmul_shapes);
        
        assert_eq!(add_node.compute_cost(), 1000.0);
        assert_eq!(matmul_node.compute_cost(), 2.0 * 100.0 * 200.0 * 300.0);
    }
}