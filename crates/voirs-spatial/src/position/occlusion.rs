//! Occlusion detection and obstruction processing

use super::types::Box3D;
use crate::types::Position3D;
use serde::{Deserialize, Serialize};

/// Occlusion calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OcclusionMethod {
    /// Simple line-of-sight check
    LineOfSight,
    /// Ray casting with multiple rays
    RayCasting,
    /// Fresnel zone checking
    FresnelZone,
    /// Wave diffraction modeling
    Diffraction,
}

/// Material properties for occlusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcclusionMaterial {
    /// Material name
    pub name: String,
    /// Transmission coefficient (0.0 = full block, 1.0 = no block)
    pub transmission: f32,
    /// High-frequency absorption
    pub high_freq_absorption: f32,
    /// Low-frequency absorption
    pub low_freq_absorption: f32,
    /// Scattering coefficient
    pub scattering: f32,
}

/// Occlusion result
#[derive(Debug, Clone)]
pub struct OcclusionResult {
    /// Is source occluded
    pub is_occluded: bool,
    /// Transmission factor (0.0 = fully blocked, 1.0 = no blocking)
    pub transmission_factor: f32,
    /// High-frequency attenuation
    pub high_freq_attenuation: f32,
    /// Low-frequency attenuation
    pub low_freq_attenuation: f32,
    /// Diffraction paths (if any)
    pub diffraction_paths: Vec<DiffractionPath>,
}

/// Sound diffraction path around obstacle
#[derive(Debug, Clone)]
pub struct DiffractionPath {
    /// Path around obstacle
    pub path: Vec<Position3D>,
    /// Path length
    pub length: f32,
    /// Attenuation factor
    pub attenuation: f32,
    /// Delay in samples
    pub delay_samples: usize,
}

/// Occlusion and obstruction detector
pub struct OcclusionDetector {
    /// Occlusion geometry (simple box obstacles for now)
    obstacles: Vec<Box3D>,
    /// Occlusion calculation method
    method: OcclusionMethod,
    /// Material properties for obstacles
    materials: std::collections::HashMap<String, OcclusionMaterial>,
}

impl OcclusionDetector {
    /// Create new occlusion detector
    pub fn new() -> Self {
        Self {
            obstacles: Vec::new(),
            method: OcclusionMethod::LineOfSight,
            materials: std::collections::HashMap::new(),
        }
    }

    /// Add obstacle
    pub fn add_obstacle(&mut self, obstacle: Box3D) {
        self.obstacles.push(obstacle);
    }

    /// Add material
    pub fn add_material(&mut self, material: OcclusionMaterial) {
        self.materials.insert(material.name.clone(), material);
    }

    /// Check occlusion between source and listener
    pub fn check_occlusion(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        match self.method {
            OcclusionMethod::LineOfSight => self.line_of_sight_check(source, listener),
            OcclusionMethod::RayCasting => self.ray_casting_check(source, listener),
            OcclusionMethod::FresnelZone => self.fresnel_zone_check(source, listener),
            OcclusionMethod::Diffraction => self.diffraction_check(source, listener),
        }
    }

    /// Simple line-of-sight occlusion check
    fn line_of_sight_check(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        for obstacle in &self.obstacles {
            if self.line_intersects_box(source, listener, obstacle) {
                let material = self
                    .materials
                    .get(&obstacle.material_id)
                    .cloned()
                    .unwrap_or_else(OcclusionMaterial::default);

                return OcclusionResult {
                    is_occluded: true,
                    transmission_factor: material.transmission,
                    high_freq_attenuation: material.high_freq_absorption,
                    low_freq_attenuation: material.low_freq_absorption,
                    diffraction_paths: Vec::new(),
                };
            }
        }

        OcclusionResult {
            is_occluded: false,
            transmission_factor: 1.0,
            high_freq_attenuation: 1.0,
            low_freq_attenuation: 1.0,
            diffraction_paths: Vec::new(),
        }
    }

    /// Ray casting occlusion check with multiple rays
    fn ray_casting_check(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        // For now, delegate to line-of-sight
        // In a full implementation, this would cast multiple rays
        self.line_of_sight_check(source, listener)
    }

    /// Fresnel zone occlusion check
    fn fresnel_zone_check(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        // Simplified implementation - delegate to line-of-sight
        // A full implementation would check Fresnel zone clearance
        self.line_of_sight_check(source, listener)
    }

    /// Diffraction-based occlusion check
    fn diffraction_check(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        let mut result = self.line_of_sight_check(source, listener);

        // If occluded, try to find diffraction paths
        if result.is_occluded {
            result.diffraction_paths = self.find_diffraction_paths(source, listener);

            // If diffraction paths exist, adjust transmission
            if !result.diffraction_paths.is_empty() {
                result.transmission_factor = result.transmission_factor.max(0.1);
                // Some sound gets through
            }
        }

        result
    }

    /// Check if line intersects with 3D box
    pub fn line_intersects_box(&self, start: Position3D, end: Position3D, box3d: &Box3D) -> bool {
        // Simplified box-line intersection test
        let dir = Position3D::new(end.x - start.x, end.y - start.y, end.z - start.z);
        let length = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();

        if length == 0.0 {
            return false;
        }

        let dir_norm = Position3D::new(dir.x / length, dir.y / length, dir.z / length);

        // Ray-box intersection using slab method
        let inv_dir = Position3D::new(
            if dir_norm.x != 0.0 {
                1.0 / dir_norm.x
            } else {
                f32::INFINITY
            },
            if dir_norm.y != 0.0 {
                1.0 / dir_norm.y
            } else {
                f32::INFINITY
            },
            if dir_norm.z != 0.0 {
                1.0 / dir_norm.z
            } else {
                f32::INFINITY
            },
        );

        let t1 = (box3d.min.x - start.x) * inv_dir.x;
        let t2 = (box3d.max.x - start.x) * inv_dir.x;
        let t3 = (box3d.min.y - start.y) * inv_dir.y;
        let t4 = (box3d.max.y - start.y) * inv_dir.y;
        let t5 = (box3d.min.z - start.z) * inv_dir.z;
        let t6 = (box3d.max.z - start.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        tmax >= 0.0 && tmin <= tmax && tmin <= length
    }

    /// Find diffraction paths around obstacles
    fn find_diffraction_paths(
        &self,
        _source: Position3D,
        _listener: Position3D,
    ) -> Vec<DiffractionPath> {
        // Simplified implementation - returns empty for now
        // A full implementation would find paths around obstacle edges
        Vec::new()
    }
}

impl Default for OcclusionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for OcclusionMaterial {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            transmission: 0.1,
            high_freq_absorption: 0.8,
            low_freq_absorption: 0.3,
            scattering: 0.2,
        }
    }
}
