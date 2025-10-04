//! Room acoustics simulation and reverberation processing

pub mod adaptive_acoustics;
pub mod simulation;

use crate::types::Position3D;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
pub use simulation::{
    AcousticResponse, AdvancedRoomSimulator, DiffractionProcessor, DynamicEnvironmentManager,
    FrequencyDependentProperty, Material as SimMaterial, MaterialDatabase, RayTracingEngine,
    RoomGeometry, RoomSimulationConfig,
};
use std::collections::{HashMap, VecDeque};

/// Room acoustics simulator
#[derive(Debug, Clone)]
pub struct RoomSimulator {
    /// Room configuration
    config: RoomConfig,
    /// Reverberation processor
    reverb_processor: ReverbProcessor,
    /// Early reflection processor
    early_reflections: EarlyReflectionProcessor,
    /// Late reverberation processor
    late_reverb: LateReverbProcessor,
    /// Room impulse response
    room_ir: Option<RoomImpulseResponse>,
}

/// Room acoustics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomConfig {
    /// Room dimensions (width, height, depth) in meters
    pub dimensions: (f32, f32, f32),
    /// Wall materials and absorption coefficients
    pub wall_materials: WallMaterials,
    /// Reverberation time (RT60) in seconds
    pub reverb_time: f32,
    /// Room volume in cubic meters
    pub volume: f32,
    /// Surface area in square meters
    pub surface_area: f32,
    /// Temperature in Celsius
    pub temperature: f32,
    /// Humidity percentage
    pub humidity: f32,
    /// Air absorption enabled
    pub enable_air_absorption: bool,
}

/// Wall materials and their acoustic properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WallMaterials {
    /// Floor material
    pub floor: Material,
    /// Ceiling material
    pub ceiling: Material,
    /// Left wall material
    pub left_wall: Material,
    /// Right wall material
    pub right_wall: Material,
    /// Front wall material
    pub front_wall: Material,
    /// Back wall material
    pub back_wall: Material,
}

/// Acoustic material properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    /// Material name
    pub name: String,
    /// Absorption coefficients by frequency band
    pub absorption_coefficients: Vec<FrequencyBandAbsorption>,
    /// Scattering coefficient (0.0 = pure reflection, 1.0 = pure diffusion)
    pub scattering_coefficient: f32,
    /// Transmission coefficient
    pub transmission_coefficient: f32,
}

/// Frequency band absorption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyBandAbsorption {
    /// Center frequency in Hz
    pub frequency: f32,
    /// Absorption coefficient (0.0 = no absorption, 1.0 = full absorption)
    pub coefficient: f32,
}

/// Room impulse response
#[derive(Debug, Clone)]
pub struct RoomImpulseResponse {
    /// Early reflections
    pub early_reflections: Array1<f32>,
    /// Late reverberation
    pub late_reverb: Array1<f32>,
    /// Combined impulse response
    pub combined_ir: Array1<f32>,
    /// Sample rate
    pub sample_rate: u32,
}

/// Early reflection processor
#[derive(Debug, Clone)]
pub struct EarlyReflectionProcessor {
    /// Reflection paths
    #[allow(dead_code)]
    reflection_paths: Vec<ReflectionPath>,
    /// Maximum reflection order
    #[allow(dead_code)]
    max_order: usize,
    /// Speed of sound
    #[allow(dead_code)]
    speed_of_sound: f32,
    /// Sample rate
    sample_rate: f32,
}

/// Late reverberation processor
#[derive(Debug, Clone)]
pub struct LateReverbProcessor {
    /// Feedback delay networks
    #[allow(dead_code)]
    feedback_networks: Vec<FeedbackDelayNetwork>,
    /// Diffusion all-pass filters
    #[allow(dead_code)]
    diffusion_filters: Vec<AllPassFilter>,
    /// Reverb time
    reverb_time: f32,
    /// Diffusion amount
    #[allow(dead_code)]
    diffusion: f32,
}

/// Combined reverberation processor
#[derive(Debug, Clone)]
pub struct ReverbProcessor {
    /// Early reflections
    #[allow(dead_code)]
    early_processor: EarlyReflectionProcessor,
    /// Late reverb
    #[allow(dead_code)]
    late_processor: LateReverbProcessor,
    /// Crossover frequency between early and late
    #[allow(dead_code)]
    crossover_frequency: f32,
    /// Mix levels
    #[allow(dead_code)]
    dry_level: f32,
    #[allow(dead_code)]
    early_level: f32,
    #[allow(dead_code)]
    late_level: f32,
}

/// Reflection path in the room
#[derive(Debug, Clone)]
pub struct ReflectionPath {
    /// Path from source to reflection point to listener
    pub path: Vec<Position3D>,
    /// Total path length
    pub length: f32,
    /// Delay in samples
    pub delay_samples: usize,
    /// Attenuation factor
    pub attenuation: f32,
    /// Reflection surfaces
    pub surfaces: Vec<SurfaceReflection>,
}

/// Surface reflection information
#[derive(Debug, Clone)]
pub struct SurfaceReflection {
    /// Surface position
    pub position: Position3D,
    /// Surface normal
    pub normal: Position3D,
    /// Material properties
    pub material: Material,
    /// Incident angle
    pub incident_angle: f32,
}

/// Feedback delay network for late reverberation
#[derive(Debug, Clone)]
pub struct FeedbackDelayNetwork {
    /// Delay lines
    #[allow(dead_code)]
    delay_lines: Vec<DelayLine>,
    /// Feedback matrix
    #[allow(dead_code)]
    feedback_matrix: Array2<f32>,
    /// Input gains
    #[allow(dead_code)]
    input_gains: Array1<f32>,
    /// Output gains
    #[allow(dead_code)]
    output_gains: Array1<f32>,
}

/// All-pass filter for diffusion
#[derive(Debug, Clone)]
pub struct AllPassFilter {
    /// Delay line
    #[allow(dead_code)]
    delay_line: DelayLine,
    /// Feedback coefficient
    #[allow(dead_code)]
    feedback: f32,
    /// Feed-forward coefficient
    #[allow(dead_code)]
    feedforward: f32,
}

/// Delay line with interpolation
#[derive(Debug, Clone)]
pub struct DelayLine {
    /// Buffer
    buffer: VecDeque<f32>,
    /// Delay in samples
    delay_samples: f32,
    /// Maximum delay
    max_delay: usize,
}

/// Wall structure for ray tracing
#[derive(Debug, Clone)]
pub struct Wall {
    /// Wall normal vector
    pub normal: Position3D,
    /// Point on the wall surface
    pub point: Position3D,
    /// Wall material properties
    pub material: Material,
}

impl RoomSimulator {
    /// Create new room simulator
    pub fn new(dimensions: (f32, f32, f32), reverb_time: f32) -> crate::Result<Self> {
        let config = RoomConfig::new(dimensions, reverb_time);
        let reverb_processor = ReverbProcessor::new(&config)?;
        let early_reflections = EarlyReflectionProcessor::new(&config)?;
        let late_reverb = LateReverbProcessor::new(&config)?;

        Ok(Self {
            config,
            reverb_processor,
            early_reflections,
            late_reverb,
            room_ir: None,
        })
    }

    /// Create room simulator with custom configuration
    pub fn with_config(config: RoomConfig) -> crate::Result<Self> {
        let reverb_processor = ReverbProcessor::new(&config)?;
        let early_reflections = EarlyReflectionProcessor::new(&config)?;
        let late_reverb = LateReverbProcessor::new(&config)?;

        Ok(Self {
            config,
            reverb_processor,
            early_reflections,
            late_reverb,
            room_ir: None,
        })
    }

    /// Process audio with room reverb
    pub async fn process_reverb(
        &self,
        left_channel: &mut Array1<f32>,
        right_channel: &mut Array1<f32>,
        source_position: &Position3D,
    ) -> crate::Result<()> {
        // Apply early reflections
        self.early_reflections
            .process(left_channel, right_channel, source_position)
            .await?;

        // Apply late reverberation
        self.late_reverb
            .process(left_channel, right_channel)
            .await?;

        Ok(())
    }

    /// Calculate room impulse response for a specific source-listener pair
    pub fn calculate_room_ir(
        &mut self,
        source_position: Position3D,
        listener_position: Position3D,
        sample_rate: u32,
    ) -> crate::Result<RoomImpulseResponse> {
        let ir_length = (self.config.reverb_time * sample_rate as f32) as usize;
        let mut early_reflections = Array1::zeros(ir_length);
        let mut late_reverb = Array1::zeros(ir_length);

        // Calculate early reflections using image source method
        let reflection_paths =
            self.calculate_reflection_paths(source_position, listener_position, 3)?;

        for path in &reflection_paths {
            if path.delay_samples < early_reflections.len() {
                early_reflections[path.delay_samples] += path.attenuation;
            }
        }

        // Generate late reverberation using statistical model
        self.generate_late_reverb(&mut late_reverb, sample_rate)?;

        // Combine early and late
        let mut combined_ir = Array1::zeros(ir_length);
        for i in 0..ir_length {
            combined_ir[i] = early_reflections[i] + late_reverb[i];
        }

        let room_ir = RoomImpulseResponse {
            early_reflections,
            late_reverb,
            combined_ir,
            sample_rate,
        };

        self.room_ir = Some(room_ir.clone());
        Ok(room_ir)
    }

    /// Calculate reflection paths using enhanced ray tracing method
    fn calculate_reflection_paths(
        &self,
        source: Position3D,
        listener: Position3D,
        max_order: usize,
    ) -> crate::Result<Vec<ReflectionPath>> {
        let mut paths = Vec::new();
        let (width, height, depth) = self.config.dimensions;

        // Direct path
        let direct_path = ReflectionPath {
            path: vec![source, listener],
            length: source.distance_to(&listener),
            delay_samples: (source.distance_to(&listener) / 343.0 * 44100.0) as usize,
            attenuation: 1.0 / (1.0 + source.distance_to(&listener)),
            surfaces: Vec::new(),
        };
        paths.push(direct_path);

        // Enhanced ray tracing with multiple orders
        self.calculate_ray_traced_paths(&mut paths, source, listener, max_order)?;

        Ok(paths)
    }

    /// Enhanced ray tracing algorithm for multiple reflection orders
    fn calculate_ray_traced_paths(
        &self,
        paths: &mut Vec<ReflectionPath>,
        source: Position3D,
        listener: Position3D,
        max_order: usize,
    ) -> crate::Result<()> {
        let (width, height, depth) = self.config.dimensions;

        // First, add deterministic first-order reflections for compatibility
        if max_order >= 1 {
            self.add_wall_reflections(paths, source, listener, width, height, depth);
        }

        // Then add enhanced ray tracing for higher order reflections
        if max_order >= 2 {
            self.add_ray_traced_reflections(paths, source, listener, max_order)?;
        }

        Ok(())
    }

    /// Add enhanced ray-traced reflections for higher order paths
    fn add_ray_traced_reflections(
        &self,
        paths: &mut Vec<ReflectionPath>,
        source: Position3D,
        listener: Position3D,
        max_order: usize,
    ) -> crate::Result<()> {
        let walls = self.get_room_walls(
            self.config.dimensions.0,
            self.config.dimensions.1,
            self.config.dimensions.2,
        );

        // Use deterministic ray directions based on listener position
        let to_listener = Position3D::new(
            listener.x - source.x,
            listener.y - source.y,
            listener.z - source.z,
        )
        .normalized();

        // Generate rays in directions likely to reach listener after reflections
        let base_directions = vec![
            to_listener,
            Position3D::new(to_listener.x, -to_listener.y, to_listener.z).normalized(),
            Position3D::new(-to_listener.x, to_listener.y, to_listener.z).normalized(),
            Position3D::new(to_listener.x, to_listener.y, -to_listener.z).normalized(),
        ];

        for base_direction in base_directions {
            let mut current_pos = source;
            let mut ray_direction = base_direction;
            let mut current_path = vec![source];
            let mut total_attenuation = 1.0;
            let mut total_length = 0.0;
            let mut surface_reflections = Vec::new();

            for order in 1..=max_order {
                if let Some((intersection_point, wall_normal, material)) =
                    self.find_nearest_wall_intersection(current_pos, ray_direction, &walls)?
                {
                    let distance = current_pos.distance_to(&intersection_point);
                    total_length += distance;
                    current_path.push(intersection_point);

                    // Apply material-dependent attenuation
                    let frequency_attenuation = self.calculate_frequency_attenuation(&material);
                    total_attenuation *= frequency_attenuation;

                    // Record surface reflection
                    surface_reflections.push(SurfaceReflection {
                        position: intersection_point,
                        normal: wall_normal,
                        material: material.clone(),
                        incident_angle: self.calculate_incident_angle(ray_direction, wall_normal),
                    });

                    // Calculate reflected ray direction (perfect specular reflection)
                    let dot = ray_direction.dot(&wall_normal);
                    ray_direction = Position3D::new(
                        ray_direction.x - 2.0 * dot * wall_normal.x,
                        ray_direction.y - 2.0 * dot * wall_normal.y,
                        ray_direction.z - 2.0 * dot * wall_normal.z,
                    )
                    .normalized();

                    current_pos = intersection_point;

                    // For higher order reflections, check if we can reach listener
                    if order >= 2 {
                        let listener_distance = intersection_point.distance_to(&listener);
                        if listener_distance < 2.0 && total_attenuation > 0.01 {
                            current_path.push(listener);
                            total_length += listener_distance;

                            let reflection_path = ReflectionPath {
                                path: current_path.clone(),
                                length: total_length,
                                delay_samples: (total_length / 343.0 * 44100.0) as usize,
                                attenuation: total_attenuation / (1.0 + total_length),
                                surfaces: surface_reflections.clone(),
                            };

                            paths.push(reflection_path);
                            break;
                        }
                    }
                } else {
                    break; // No intersection found
                }
            }
        }

        Ok(())
    }

    /// Generate random ray direction in 3D space
    fn random_ray_direction(
        &self,
        rng: &mut scirs2_core::random::prelude::ThreadRng,
    ) -> Position3D {
        use scirs2_core::random::Rng;

        let theta = rng.gen_range(0.0..std::f32::consts::PI * 2.0);
        let phi = rng.gen_range(0.0..std::f32::consts::PI);

        Position3D::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos())
    }

    /// Get room walls as geometric surfaces
    fn get_room_walls(&self, width: f32, height: f32, depth: f32) -> Vec<Wall> {
        vec![
            Wall {
                normal: Position3D::new(-1.0, 0.0, 0.0),
                point: Position3D::new(0.0, 0.0, 0.0),
                material: self.config.wall_materials.left_wall.clone(),
            },
            Wall {
                normal: Position3D::new(1.0, 0.0, 0.0),
                point: Position3D::new(width, 0.0, 0.0),
                material: self.config.wall_materials.right_wall.clone(),
            },
            Wall {
                normal: Position3D::new(0.0, -1.0, 0.0),
                point: Position3D::new(0.0, 0.0, 0.0),
                material: self.config.wall_materials.floor.clone(),
            },
            Wall {
                normal: Position3D::new(0.0, 1.0, 0.0),
                point: Position3D::new(0.0, height, 0.0),
                material: self.config.wall_materials.ceiling.clone(),
            },
            Wall {
                normal: Position3D::new(0.0, 0.0, -1.0),
                point: Position3D::new(0.0, 0.0, 0.0),
                material: self.config.wall_materials.front_wall.clone(),
            },
            Wall {
                normal: Position3D::new(0.0, 0.0, 1.0),
                point: Position3D::new(0.0, 0.0, depth),
                material: self.config.wall_materials.back_wall.clone(),
            },
        ]
    }

    /// Find nearest wall intersection with ray
    fn find_nearest_wall_intersection(
        &self,
        ray_origin: Position3D,
        ray_direction: Position3D,
        walls: &[Wall],
    ) -> crate::Result<Option<(Position3D, Position3D, Material)>> {
        let mut nearest_intersection = None;
        let mut nearest_distance = f32::INFINITY;

        for wall in walls {
            if let Some((intersection_point, distance)) =
                self.ray_plane_intersection(ray_origin, ray_direction, &wall.point, &wall.normal)?
            {
                if distance > 0.001 && distance < nearest_distance {
                    // Check if intersection is within room bounds
                    if self.is_within_room_bounds(intersection_point) {
                        nearest_distance = distance;
                        nearest_intersection =
                            Some((intersection_point, wall.normal, wall.material.clone()));
                    }
                }
            }
        }

        Ok(nearest_intersection)
    }

    /// Calculate ray-plane intersection
    fn ray_plane_intersection(
        &self,
        ray_origin: Position3D,
        ray_direction: Position3D,
        plane_point: &Position3D,
        plane_normal: &Position3D,
    ) -> crate::Result<Option<(Position3D, f32)>> {
        let denominator = ray_direction.dot(plane_normal);

        if denominator.abs() < 1e-6 {
            return Ok(None); // Ray parallel to plane
        }

        let diff = Position3D::new(
            plane_point.x - ray_origin.x,
            plane_point.y - ray_origin.y,
            plane_point.z - ray_origin.z,
        );

        let t = diff.dot(plane_normal) / denominator;

        if t >= 0.0 {
            let intersection = Position3D::new(
                ray_origin.x + t * ray_direction.x,
                ray_origin.y + t * ray_direction.y,
                ray_origin.z + t * ray_direction.z,
            );
            Ok(Some((intersection, t)))
        } else {
            Ok(None)
        }
    }

    /// Check if point is within room bounds
    fn is_within_room_bounds(&self, point: Position3D) -> bool {
        let (width, height, depth) = self.config.dimensions;
        point.x >= 0.0
            && point.x <= width
            && point.y >= 0.0
            && point.y <= height
            && point.z >= 0.0
            && point.z <= depth
    }

    /// Calculate frequency-dependent attenuation based on material
    fn calculate_frequency_attenuation(&self, material: &Material) -> f32 {
        // Simplified frequency attenuation - average across all bands
        1.0 - material.average_absorption()
    }

    /// Calculate incident angle between ray and surface normal
    fn calculate_incident_angle(&self, ray_direction: Position3D, normal: Position3D) -> f32 {
        let dot_product = ray_direction.dot(&normal).abs();
        dot_product.acos()
    }

    /// Calculate reflection direction with specular and diffuse components
    fn calculate_reflection_direction(
        &self,
        incident: Position3D,
        normal: Position3D,
        scattering_coeff: f32,
        rng: &mut scirs2_core::random::prelude::ThreadRng,
    ) -> Position3D {
        use scirs2_core::random::Rng;

        // Specular reflection component
        let dot = incident.dot(&normal);
        let specular = Position3D::new(
            incident.x - 2.0 * dot * normal.x,
            incident.y - 2.0 * dot * normal.y,
            incident.z - 2.0 * dot * normal.z,
        );

        // Diffuse reflection component (Lambert's cosine law)
        let diffuse = self.random_ray_direction(rng);

        // Mix specular and diffuse based on scattering coefficient
        let mix_factor = 1.0 - scattering_coeff;
        Position3D::new(
            mix_factor * specular.x + scattering_coeff * diffuse.x,
            mix_factor * specular.y + scattering_coeff * diffuse.y,
            mix_factor * specular.z + scattering_coeff * diffuse.z,
        )
        .normalized()
    }

    /// Add first-order wall reflections
    fn add_wall_reflections(
        &self,
        paths: &mut Vec<ReflectionPath>,
        source: Position3D,
        listener: Position3D,
        width: f32,
        height: f32,
        depth: f32,
    ) {
        let walls = [
            // Left wall (x = 0)
            (
                Position3D::new(0.0, source.y, source.z),
                Position3D::new(-1.0, 0.0, 0.0),
            ),
            // Right wall (x = width)
            (
                Position3D::new(width, source.y, source.z),
                Position3D::new(1.0, 0.0, 0.0),
            ),
            // Floor (y = 0)
            (
                Position3D::new(source.x, 0.0, source.z),
                Position3D::new(0.0, -1.0, 0.0),
            ),
            // Ceiling (y = height)
            (
                Position3D::new(source.x, height, source.z),
                Position3D::new(0.0, 1.0, 0.0),
            ),
            // Front wall (z = 0)
            (
                Position3D::new(source.x, source.y, 0.0),
                Position3D::new(0.0, 0.0, -1.0),
            ),
            // Back wall (z = depth)
            (
                Position3D::new(source.x, source.y, depth),
                Position3D::new(0.0, 0.0, 1.0),
            ),
        ];

        for (reflection_point, normal) in walls {
            // Calculate reflected path
            let source_to_reflection = reflection_point.distance_to(&source);
            let reflection_to_listener = reflection_point.distance_to(&listener);
            let total_length = source_to_reflection + reflection_to_listener;

            let path = ReflectionPath {
                path: vec![source, reflection_point, listener],
                length: total_length,
                delay_samples: (total_length / 343.0 * 44100.0) as usize,
                attenuation: 0.7 / (1.0 + total_length), // Simplified attenuation
                surfaces: vec![SurfaceReflection {
                    position: reflection_point,
                    normal,
                    material: Material::default(),
                    incident_angle: 0.0, // Simplified
                }],
            };

            paths.push(path);
        }
    }

    /// Generate late reverberation using statistical model
    fn generate_late_reverb(
        &self,
        buffer: &mut Array1<f32>,
        sample_rate: u32,
    ) -> crate::Result<()> {
        let decay_rate = -60.0 / (self.config.reverb_time * sample_rate as f32); // dB per sample

        for (i, sample) in buffer.iter_mut().enumerate() {
            let time = i as f32 / sample_rate as f32;
            let amplitude = (decay_rate * time / 20.0).exp(); // Convert dB to linear
            *sample = amplitude * (scirs2_core::random::random::<f32>() - 0.5) * 2.0;
            // Noise with decay
        }

        Ok(())
    }

    /// Get room configuration
    pub fn config(&self) -> &RoomConfig {
        &self.config
    }

    /// Set room configuration
    pub fn set_config(&mut self, config: RoomConfig) -> crate::Result<()> {
        self.config = config;
        self.reverb_processor = ReverbProcessor::new(&self.config)?;
        self.early_reflections = EarlyReflectionProcessor::new(&self.config)?;
        self.late_reverb = LateReverbProcessor::new(&self.config)?;
        Ok(())
    }
}

/// Trait for room acoustics processing
pub trait RoomAcoustics {
    /// Process audio with room acoustics
    fn process_acoustics(
        &mut self,
        input: &Array1<f32>,
        output: &mut Array1<f32>,
        source_position: Position3D,
        listener_position: Position3D,
    ) -> crate::Result<()>;

    /// Calculate reverberation time
    fn calculate_reverb_time(&self) -> f32;

    /// Get room properties
    fn room_properties(&self) -> RoomProperties;
}

/// Room acoustic properties
#[derive(Debug, Clone)]
pub struct RoomProperties {
    /// Volume in cubic meters
    pub volume: f32,
    /// Surface area in square meters
    pub surface_area: f32,
    /// Average absorption coefficient
    pub average_absorption: f32,
    /// Critical distance
    pub critical_distance: f32,
    /// Reverberation time
    pub reverb_time: f32,
}

impl RoomConfig {
    /// Create new room configuration
    pub fn new(dimensions: (f32, f32, f32), reverb_time: f32) -> Self {
        let (width, height, depth) = dimensions;
        let volume = width * height * depth;
        let surface_area = 2.0 * (width * height + width * depth + height * depth);

        Self {
            dimensions,
            wall_materials: WallMaterials::default(),
            reverb_time,
            volume,
            surface_area,
            temperature: 20.0,
            humidity: 50.0,
            enable_air_absorption: true,
        }
    }

    /// Calculate average absorption coefficient
    pub fn average_absorption(&self) -> f32 {
        // Simplified calculation - would need frequency-dependent calculation in practice
        let materials = [
            &self.wall_materials.floor,
            &self.wall_materials.ceiling,
            &self.wall_materials.left_wall,
            &self.wall_materials.right_wall,
            &self.wall_materials.front_wall,
            &self.wall_materials.back_wall,
        ];

        let total_absorption: f32 = materials
            .iter()
            .map(|material| material.average_absorption())
            .sum();

        total_absorption / materials.len() as f32
    }
}

impl Material {
    /// Get average absorption coefficient
    pub fn average_absorption(&self) -> f32 {
        if self.absorption_coefficients.is_empty() {
            return 0.1; // Default
        }

        let sum: f32 = self
            .absorption_coefficients
            .iter()
            .map(|band| band.coefficient)
            .sum();

        sum / self.absorption_coefficients.len() as f32
    }

    /// Create concrete material
    pub fn concrete() -> Self {
        Self {
            name: "Concrete".to_string(),
            absorption_coefficients: vec![
                FrequencyBandAbsorption {
                    frequency: 125.0,
                    coefficient: 0.01,
                },
                FrequencyBandAbsorption {
                    frequency: 250.0,
                    coefficient: 0.01,
                },
                FrequencyBandAbsorption {
                    frequency: 500.0,
                    coefficient: 0.02,
                },
                FrequencyBandAbsorption {
                    frequency: 1000.0,
                    coefficient: 0.02,
                },
                FrequencyBandAbsorption {
                    frequency: 2000.0,
                    coefficient: 0.02,
                },
                FrequencyBandAbsorption {
                    frequency: 4000.0,
                    coefficient: 0.02,
                },
            ],
            scattering_coefficient: 0.1,
            transmission_coefficient: 0.01,
        }
    }

    /// Create carpet material
    pub fn carpet() -> Self {
        Self {
            name: "Carpet".to_string(),
            absorption_coefficients: vec![
                FrequencyBandAbsorption {
                    frequency: 125.0,
                    coefficient: 0.08,
                },
                FrequencyBandAbsorption {
                    frequency: 250.0,
                    coefficient: 0.24,
                },
                FrequencyBandAbsorption {
                    frequency: 500.0,
                    coefficient: 0.57,
                },
                FrequencyBandAbsorption {
                    frequency: 1000.0,
                    coefficient: 0.69,
                },
                FrequencyBandAbsorption {
                    frequency: 2000.0,
                    coefficient: 0.71,
                },
                FrequencyBandAbsorption {
                    frequency: 4000.0,
                    coefficient: 0.73,
                },
            ],
            scattering_coefficient: 0.3,
            transmission_coefficient: 0.05,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::concrete()
    }
}

impl Default for WallMaterials {
    fn default() -> Self {
        Self {
            floor: Material::carpet(),
            ceiling: Material::concrete(),
            left_wall: Material::concrete(),
            right_wall: Material::concrete(),
            front_wall: Material::concrete(),
            back_wall: Material::concrete(),
        }
    }
}

impl EarlyReflectionProcessor {
    /// Create new early reflection processor
    pub fn new(_config: &RoomConfig) -> crate::Result<Self> {
        Ok(Self {
            reflection_paths: Vec::new(),
            max_order: 3,
            speed_of_sound: 343.0,
            sample_rate: 44100.0,
        })
    }

    /// Process early reflections
    pub async fn process(
        &self,
        left_channel: &mut Array1<f32>,
        right_channel: &mut Array1<f32>,
        _source_position: &Position3D,
    ) -> crate::Result<()> {
        // Simplified early reflection processing
        // Apply a simple delay and attenuation
        let delay_samples = (0.02 * self.sample_rate) as usize; // 20ms delay
        let attenuation = 0.3;

        if delay_samples < left_channel.len() {
            for i in delay_samples..left_channel.len() {
                left_channel[i] += left_channel[i - delay_samples] * attenuation;
                right_channel[i] += right_channel[i - delay_samples] * attenuation;
            }
        }

        Ok(())
    }
}

impl LateReverbProcessor {
    /// Create new late reverb processor
    pub fn new(config: &RoomConfig) -> crate::Result<Self> {
        let feedback_networks = vec![FeedbackDelayNetwork::new(&[0.03, 0.032, 0.034, 0.036])?];

        let diffusion_filters = vec![
            AllPassFilter::new(0.005, 0.7)?,
            AllPassFilter::new(0.012, 0.5)?,
        ];

        Ok(Self {
            feedback_networks,
            diffusion_filters,
            reverb_time: config.reverb_time,
            diffusion: 0.7,
        })
    }

    /// Process late reverberation
    pub async fn process(
        &self,
        left_channel: &mut Array1<f32>,
        right_channel: &mut Array1<f32>,
    ) -> crate::Result<()> {
        // Simplified late reverb processing
        // Apply exponential decay
        let decay_rate = (-60.0 / (self.reverb_time * 44100.0)).exp();

        for i in 1..left_channel.len() {
            left_channel[i] += left_channel[i - 1] * decay_rate * 0.1;
            right_channel[i] += right_channel[i - 1] * decay_rate * 0.1;
        }

        Ok(())
    }
}

impl ReverbProcessor {
    /// Create new reverb processor
    pub fn new(config: &RoomConfig) -> crate::Result<Self> {
        Ok(Self {
            early_processor: EarlyReflectionProcessor::new(config)?,
            late_processor: LateReverbProcessor::new(config)?,
            crossover_frequency: 500.0,
            dry_level: 0.7,
            early_level: 0.3,
            late_level: 0.4,
        })
    }
}

impl FeedbackDelayNetwork {
    /// Create new feedback delay network
    pub fn new(delays: &[f32]) -> crate::Result<Self> {
        let mut delay_lines = Vec::new();
        for &delay in delays {
            delay_lines.push(DelayLine::new(delay, 44100.0)?);
        }

        let size = delays.len();
        let feedback_matrix = Array2::eye(size) * 0.7; // Simplified feedback matrix
        let input_gains = Array1::ones(size);
        let output_gains = Array1::ones(size);

        Ok(Self {
            delay_lines,
            feedback_matrix,
            input_gains,
            output_gains,
        })
    }
}

impl AllPassFilter {
    /// Create new all-pass filter
    pub fn new(delay: f32, feedback: f32) -> crate::Result<Self> {
        Ok(Self {
            delay_line: DelayLine::new(delay, 44100.0)?,
            feedback,
            feedforward: -feedback,
        })
    }
}

impl DelayLine {
    /// Create new delay line
    pub fn new(delay_time: f32, sample_rate: f32) -> crate::Result<Self> {
        let delay_samples = delay_time * sample_rate;
        let max_delay = delay_samples.ceil() as usize + 1;
        let buffer = VecDeque::with_capacity(max_delay);

        Ok(Self {
            buffer,
            delay_samples,
            max_delay,
        })
    }

    /// Process sample through delay line
    pub fn process(&mut self, input: f32) -> f32 {
        // Add input to buffer
        self.buffer.push_back(input);

        // Remove old samples if buffer is too large
        while self.buffer.len() > self.max_delay {
            self.buffer.pop_front();
        }

        // Get delayed output (simplified - no interpolation)
        let delay_index = self.delay_samples as usize;
        if self.buffer.len() > delay_index {
            self.buffer[self.buffer.len() - 1 - delay_index]
        } else {
            0.0
        }
    }
}

/// Multi-room environment system
pub struct MultiRoomEnvironment {
    /// Individual rooms in the environment
    pub rooms: HashMap<String, Room>,
    /// Connections between rooms (doors, openings, etc.)
    pub connections: Vec<RoomConnection>,
    /// Global acoustic properties
    pub global_config: GlobalAcousticConfig,
    /// Inter-room sound propagation cache
    propagation_cache: HashMap<(String, String), PropagationPath>,
}

/// Individual room in a multi-room environment
#[derive(Debug, Clone)]
pub struct Room {
    /// Room identifier
    pub id: String,
    /// Room simulator
    pub simulator: RoomSimulator,
    /// Room position in global coordinate system
    pub position: Position3D,
    /// Room orientation (yaw, pitch, roll)
    pub orientation: (f32, f32, f32),
    /// Room volume level adjustment
    pub volume_adjustment: f32,
}

/// Connection between rooms (doors, openings, vents)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomConnection {
    /// Connection ID
    pub id: String,
    /// Source room ID
    pub from_room: String,
    /// Target room ID
    pub to_room: String,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Position of connection in from_room
    pub from_position: Position3D,
    /// Position of connection in to_room
    pub to_position: Position3D,
    /// Opening dimensions (width, height)
    pub dimensions: (f32, f32),
    /// Acoustic properties
    pub acoustic_properties: ConnectionAcousticProperties,
    /// Current state (open, closed, partially open)
    pub state: ConnectionState,
}

/// Type of connection between rooms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Standard door
    Door,
    /// Open doorway/archway
    Doorway,
    /// Window
    Window,
    /// Air vent
    Vent,
    /// Large opening
    Opening,
    /// Sound isolation barrier
    SoundBarrier,
}

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConnectionState {
    /// Fully open
    Open,
    /// Fully closed
    Closed,
    /// Partially open with specified percentage (0.0-1.0)
    PartiallyOpen(f32),
}

/// Acoustic properties of room connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionAcousticProperties {
    /// Sound transmission coefficient (0.0-1.0)
    pub transmission_coefficient: f32,
    /// Frequency-dependent transmission
    pub frequency_transmission: Vec<FrequencyBandAbsorption>,
    /// Attenuation through the connection (dB)
    pub attenuation_db: f32,
    /// Reflection coefficient at the opening
    pub reflection_coefficient: f32,
    /// Diffraction around edges
    pub diffraction_enabled: bool,
}

/// Global acoustic configuration for multi-room environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalAcousticConfig {
    /// Speed of sound (m/s)
    pub speed_of_sound: f32,
    /// Global temperature (°C)
    pub temperature: f32,
    /// Global humidity (%)
    pub humidity: f32,
    /// Air absorption enabled globally
    pub enable_air_absorption: bool,
    /// Maximum propagation distance between rooms
    pub max_propagation_distance: f32,
    /// Inter-room delay processing
    pub enable_inter_room_delays: bool,
}

/// Sound propagation path between rooms
#[derive(Debug, Clone)]
pub struct PropagationPath {
    /// Sequence of rooms the sound passes through
    pub room_sequence: Vec<String>,
    /// Connection IDs used in the path
    pub connections: Vec<String>,
    /// Total attenuation along the path
    pub total_attenuation: f32,
    /// Total delay in samples
    pub total_delay_samples: usize,
    /// Frequency response of the path
    pub frequency_response: Vec<f32>,
}

impl Default for MultiRoomEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiRoomEnvironment {
    /// Create new multi-room environment
    pub fn new() -> Self {
        Self {
            rooms: HashMap::new(),
            connections: Vec::new(),
            global_config: GlobalAcousticConfig::default(),
            propagation_cache: HashMap::new(),
        }
    }

    /// Add room to the environment
    pub fn add_room(&mut self, room: Room) -> crate::Result<()> {
        if self.rooms.contains_key(&room.id) {
            return Err(crate::Error::LegacyRoom(format!(
                "Room with ID '{}' already exists",
                room.id
            )));
        }
        self.rooms.insert(room.id.clone(), room);
        Ok(())
    }

    /// Add connection between rooms
    pub fn add_connection(&mut self, connection: RoomConnection) -> crate::Result<()> {
        // Validate that both rooms exist
        if !self.rooms.contains_key(&connection.from_room) {
            return Err(crate::Error::LegacyRoom(format!(
                "Source room '{}' does not exist",
                connection.from_room
            )));
        }
        if !self.rooms.contains_key(&connection.to_room) {
            return Err(crate::Error::LegacyRoom(format!(
                "Target room '{}' does not exist",
                connection.to_room
            )));
        }

        self.connections.push(connection);
        self.invalidate_propagation_cache();
        Ok(())
    }

    /// Calculate multi-room acoustic propagation
    pub async fn process_multi_room_audio(
        &mut self,
        source_room_id: &str,
        source_position: Position3D,
        listener_room_id: &str,
        listener_position: Position3D,
        input_audio: &Array1<f32>,
        sample_rate: u32,
    ) -> crate::Result<(Array1<f32>, Array1<f32>)> {
        let mut left_output = Array1::zeros(input_audio.len());
        let mut right_output = Array1::zeros(input_audio.len());

        if source_room_id == listener_room_id {
            // Same room - use standard room acoustics
            let room = self.rooms.get(source_room_id).ok_or_else(|| {
                crate::Error::LegacyRoom(format!("Room '{source_room_id}' not found"))
            })?;

            // Process with room acoustics
            self.apply_room_acoustics(
                &room.simulator,
                input_audio,
                &mut left_output,
                &mut right_output,
                source_position,
                listener_position,
            )
            .await?;
        } else {
            // Different rooms - calculate inter-room propagation
            let propagation_paths = self
                .find_propagation_paths(source_room_id, listener_room_id)
                .await?;

            for path in &propagation_paths {
                let mut path_left = input_audio.clone();
                let mut path_right = input_audio.clone();

                // Apply path-specific processing
                self.apply_propagation_path(path, &mut path_left, &mut path_right, sample_rate)
                    .await?;

                // Add to output
                for i in 0..left_output.len() {
                    left_output[i] += path_left[i];
                    right_output[i] += path_right[i];
                }
            }
        }

        Ok((left_output, right_output))
    }

    /// Apply room acoustics to audio
    async fn apply_room_acoustics(
        &self,
        room_simulator: &RoomSimulator,
        input: &Array1<f32>,
        left_output: &mut Array1<f32>,
        right_output: &mut Array1<f32>,
        source_position: Position3D,
        _listener_position: Position3D,
    ) -> crate::Result<()> {
        // Copy input to outputs
        for i in 0..input.len() {
            left_output[i] = input[i];
            right_output[i] = input[i];
        }

        // Apply room reverb
        room_simulator
            .process_reverb(left_output, right_output, &source_position)
            .await?;

        Ok(())
    }

    /// Find all possible propagation paths between rooms
    async fn find_propagation_paths(
        &mut self,
        source_room: &str,
        target_room: &str,
    ) -> crate::Result<Vec<PropagationPath>> {
        // Check cache first
        let cache_key = (source_room.to_string(), target_room.to_string());
        if let Some(cached_path) = self.propagation_cache.get(&cache_key) {
            return Ok(vec![cached_path.clone()]);
        }

        // Use breadth-first search to find shortest paths
        let mut paths = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        // Initialize with source room
        queue.push_back(PropagationPath {
            room_sequence: vec![source_room.to_string()],
            connections: Vec::new(),
            total_attenuation: 1.0,
            total_delay_samples: 0,
            frequency_response: vec![1.0; 10], // Simplified frequency bands
        });

        while let Some(current_path) = queue.pop_front() {
            let current_room = current_path.room_sequence.last().unwrap();

            if current_room == target_room {
                paths.push(current_path.clone());
                continue;
            }

            if visited.contains(current_room) || current_path.room_sequence.len() > 5 {
                continue; // Avoid cycles and limit depth
            }

            visited.insert(current_room.clone());

            // Find connections from current room
            for connection in &self.connections {
                if connection.from_room == *current_room
                    && connection.state != ConnectionState::Closed
                {
                    let mut new_path = current_path.clone();
                    new_path.room_sequence.push(connection.to_room.clone());
                    new_path.connections.push(connection.id.clone());

                    // Calculate additional attenuation and delay
                    let connection_attenuation = self.calculate_connection_attenuation(connection);
                    new_path.total_attenuation *= connection_attenuation;

                    // Estimate delay based on distance (simplified)
                    let distance = connection
                        .from_position
                        .distance_to(&connection.to_position);
                    let delay_samples =
                        (distance / self.global_config.speed_of_sound * 44100.0) as usize;
                    new_path.total_delay_samples += delay_samples;

                    queue.push_back(new_path);
                }
            }
        }

        // Cache the result
        if !paths.is_empty() {
            self.propagation_cache.insert(cache_key, paths[0].clone());
        }

        Ok(paths)
    }

    /// Calculate attenuation through a connection
    fn calculate_connection_attenuation(&self, connection: &RoomConnection) -> f32 {
        let base_attenuation = match connection.state {
            ConnectionState::Open => connection.acoustic_properties.transmission_coefficient,
            ConnectionState::Closed => 0.01, // Very little transmission when closed
            ConnectionState::PartiallyOpen(ratio) => {
                connection.acoustic_properties.transmission_coefficient * ratio
            }
        };

        // Apply frequency-independent attenuation for simplification
        base_attenuation * 10_f32.powf(-connection.acoustic_properties.attenuation_db / 20.0)
    }

    /// Apply propagation path effects to audio
    async fn apply_propagation_path(
        &self,
        path: &PropagationPath,
        left_audio: &mut Array1<f32>,
        right_audio: &mut Array1<f32>,
        _sample_rate: u32,
    ) -> crate::Result<()> {
        // Apply total attenuation
        for sample in left_audio.iter_mut() {
            *sample *= path.total_attenuation;
        }
        for sample in right_audio.iter_mut() {
            *sample *= path.total_attenuation;
        }

        // Apply delay (simplified - would need proper delay line in production)
        if path.total_delay_samples > 0 && path.total_delay_samples < left_audio.len() {
            // Shift samples to apply delay
            for i in (path.total_delay_samples..left_audio.len()).rev() {
                left_audio[i] = left_audio[i - path.total_delay_samples];
                right_audio[i] = right_audio[i - path.total_delay_samples];
            }
            for i in 0..path.total_delay_samples {
                left_audio[i] = 0.0;
                right_audio[i] = 0.0;
            }
        }

        Ok(())
    }

    /// Invalidate propagation cache when rooms or connections change
    fn invalidate_propagation_cache(&mut self) {
        self.propagation_cache.clear();
    }

    /// Get room by ID
    pub fn get_room(&self, room_id: &str) -> Option<&Room> {
        self.rooms.get(room_id)
    }

    /// Get room by ID (mutable)
    pub fn get_room_mut(&mut self, room_id: &str) -> Option<&mut Room> {
        self.rooms.get_mut(room_id)
    }

    /// Update connection state
    pub fn set_connection_state(
        &mut self,
        connection_id: &str,
        state: ConnectionState,
    ) -> crate::Result<()> {
        if let Some(connection) = self.connections.iter_mut().find(|c| c.id == connection_id) {
            connection.state = state;
            self.invalidate_propagation_cache();
            Ok(())
        } else {
            Err(crate::Error::LegacyRoom(format!(
                "Connection '{connection_id}' not found"
            )))
        }
    }
}

impl Room {
    /// Create new room
    pub fn new(
        id: String,
        dimensions: (f32, f32, f32),
        reverb_time: f32,
        position: Position3D,
    ) -> crate::Result<Self> {
        Ok(Self {
            id,
            simulator: RoomSimulator::new(dimensions, reverb_time)?,
            position,
            orientation: (0.0, 0.0, 0.0),
            volume_adjustment: 1.0,
        })
    }

    /// Create room with custom configuration
    pub fn with_config(
        id: String,
        config: RoomConfig,
        position: Position3D,
    ) -> crate::Result<Self> {
        Ok(Self {
            id,
            simulator: RoomSimulator::with_config(config)?,
            position,
            orientation: (0.0, 0.0, 0.0),
            volume_adjustment: 1.0,
        })
    }
}

impl Default for GlobalAcousticConfig {
    fn default() -> Self {
        Self {
            speed_of_sound: 343.0,
            temperature: 20.0,
            humidity: 50.0,
            enable_air_absorption: true,
            max_propagation_distance: 100.0,
            enable_inter_room_delays: true,
        }
    }
}

impl Default for ConnectionAcousticProperties {
    fn default() -> Self {
        Self {
            transmission_coefficient: 0.3,
            frequency_transmission: vec![
                FrequencyBandAbsorption {
                    frequency: 125.0,
                    coefficient: 0.2,
                },
                FrequencyBandAbsorption {
                    frequency: 500.0,
                    coefficient: 0.3,
                },
                FrequencyBandAbsorption {
                    frequency: 2000.0,
                    coefficient: 0.4,
                },
                FrequencyBandAbsorption {
                    frequency: 8000.0,
                    coefficient: 0.2,
                },
            ],
            attenuation_db: 3.0,
            reflection_coefficient: 0.1,
            diffraction_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_room_simulator_creation() {
        let simulator = RoomSimulator::new((10.0, 8.0, 3.0), 1.2);
        assert!(simulator.is_ok());
    }

    #[test]
    fn test_room_config() {
        let config = RoomConfig::new((5.0, 4.0, 3.0), 1.0);
        assert_eq!(config.dimensions, (5.0, 4.0, 3.0));
        assert_eq!(config.volume, 60.0);
        assert!(config.average_absorption() > 0.0);
    }

    #[test]
    fn test_material_properties() {
        let concrete = Material::concrete();
        let carpet = Material::carpet();

        assert!(carpet.average_absorption() > concrete.average_absorption());
    }

    #[test]
    fn test_delay_line() {
        let mut delay_line = DelayLine::new(0.001, 44100.0).unwrap(); // 1ms delay

        // Feed impulse
        let output1 = delay_line.process(1.0);
        assert_eq!(output1, 0.0); // Should be silent initially

        // Process enough samples for delay
        for _ in 0..50 {
            delay_line.process(0.0);
        }

        let _output2 = delay_line.process(0.0);
        // Should now output the delayed impulse (simplified test)
    }

    #[test]
    fn test_reflection_path_calculation() {
        let simulator = RoomSimulator::new((10.0, 8.0, 3.0), 1.2).unwrap();
        let source = Position3D::new(2.0, 1.0, 1.0);
        let listener = Position3D::new(8.0, 1.0, 2.0);

        let paths = simulator
            .calculate_reflection_paths(source, listener, 1)
            .unwrap();
        assert!(!paths.is_empty());

        // Should have direct path plus wall reflections
        assert!(paths.len() >= 7); // Direct + 6 walls
    }

    #[test]
    fn test_multi_room_environment_creation() {
        let mut env = MultiRoomEnvironment::new();
        assert_eq!(env.rooms.len(), 0);
        assert_eq!(env.connections.len(), 0);
    }

    #[test]
    fn test_room_creation() {
        let room = Room::new(
            "living_room".to_string(),
            (5.0, 4.0, 3.0),
            1.2,
            Position3D::new(0.0, 0.0, 0.0),
        )
        .unwrap();

        assert_eq!(room.id, "living_room");
        assert_eq!(room.position, Position3D::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_multi_room_environment_add_room() {
        let mut env = MultiRoomEnvironment::new();

        let room = Room::new(
            "kitchen".to_string(),
            (4.0, 3.0, 2.5),
            0.8,
            Position3D::new(5.0, 0.0, 0.0),
        )
        .unwrap();

        env.add_room(room).unwrap();
        assert_eq!(env.rooms.len(), 1);
        assert!(env.get_room("kitchen").is_some());
    }

    #[test]
    fn test_room_connection() {
        let mut env = MultiRoomEnvironment::new();

        // Add two rooms
        let living_room = Room::new(
            "living_room".to_string(),
            (5.0, 4.0, 3.0),
            1.2,
            Position3D::new(0.0, 0.0, 0.0),
        )
        .unwrap();

        let kitchen = Room::new(
            "kitchen".to_string(),
            (4.0, 3.0, 2.5),
            0.8,
            Position3D::new(5.0, 0.0, 0.0),
        )
        .unwrap();

        env.add_room(living_room).unwrap();
        env.add_room(kitchen).unwrap();

        // Create connection between rooms
        let connection = RoomConnection {
            id: "door_1".to_string(),
            from_room: "living_room".to_string(),
            to_room: "kitchen".to_string(),
            connection_type: ConnectionType::Door,
            from_position: Position3D::new(5.0, 2.0, 1.0),
            to_position: Position3D::new(0.0, 2.0, 1.0),
            dimensions: (0.8, 2.0),
            acoustic_properties: ConnectionAcousticProperties::default(),
            state: ConnectionState::Open,
        };

        env.add_connection(connection).unwrap();
        assert_eq!(env.connections.len(), 1);
    }

    #[test]
    fn test_connection_state_changes() {
        let mut env = MultiRoomEnvironment::new();

        // Add rooms and connection
        let living_room = Room::new(
            "living_room".to_string(),
            (5.0, 4.0, 3.0),
            1.2,
            Position3D::new(0.0, 0.0, 0.0),
        )
        .unwrap();
        let kitchen = Room::new(
            "kitchen".to_string(),
            (4.0, 3.0, 2.5),
            0.8,
            Position3D::new(5.0, 0.0, 0.0),
        )
        .unwrap();

        env.add_room(living_room).unwrap();
        env.add_room(kitchen).unwrap();

        let connection = RoomConnection {
            id: "door_1".to_string(),
            from_room: "living_room".to_string(),
            to_room: "kitchen".to_string(),
            connection_type: ConnectionType::Door,
            from_position: Position3D::new(5.0, 2.0, 1.0),
            to_position: Position3D::new(0.0, 2.0, 1.0),
            dimensions: (0.8, 2.0),
            acoustic_properties: ConnectionAcousticProperties::default(),
            state: ConnectionState::Open,
        };

        env.add_connection(connection).unwrap();

        // Test state changes
        env.set_connection_state("door_1", ConnectionState::Closed)
            .unwrap();
        assert_eq!(env.connections[0].state, ConnectionState::Closed);

        env.set_connection_state("door_1", ConnectionState::PartiallyOpen(0.5))
            .unwrap();
        assert_eq!(
            env.connections[0].state,
            ConnectionState::PartiallyOpen(0.5)
        );
    }

    #[tokio::test]
    async fn test_multi_room_audio_processing() {
        let mut env = MultiRoomEnvironment::new();

        // Add rooms
        let living_room = Room::new(
            "living_room".to_string(),
            (5.0, 4.0, 3.0),
            1.2,
            Position3D::new(0.0, 0.0, 0.0),
        )
        .unwrap();
        let kitchen = Room::new(
            "kitchen".to_string(),
            (4.0, 3.0, 2.5),
            0.8,
            Position3D::new(5.0, 0.0, 0.0),
        )
        .unwrap();

        env.add_room(living_room).unwrap();
        env.add_room(kitchen).unwrap();

        // Add connection
        let connection = RoomConnection {
            id: "door_1".to_string(),
            from_room: "living_room".to_string(),
            to_room: "kitchen".to_string(),
            connection_type: ConnectionType::Door,
            from_position: Position3D::new(5.0, 2.0, 1.0),
            to_position: Position3D::new(0.0, 2.0, 1.0),
            dimensions: (0.8, 2.0),
            acoustic_properties: ConnectionAcousticProperties::default(),
            state: ConnectionState::Open,
        };

        env.add_connection(connection).unwrap();

        // Test audio processing
        let input_audio = Array1::from_vec(vec![0.5; 1000]);
        let source_pos = Position3D::new(2.0, 2.0, 1.5);
        let listener_pos = Position3D::new(2.0, 1.5, 1.5);

        // Same room processing
        let result = env
            .process_multi_room_audio(
                "living_room",
                source_pos,
                "living_room",
                listener_pos,
                &input_audio,
                44100,
            )
            .await;

        assert!(result.is_ok());
        let (left, right) = result.unwrap();
        assert_eq!(left.len(), input_audio.len());
        assert_eq!(right.len(), input_audio.len());
    }
}
