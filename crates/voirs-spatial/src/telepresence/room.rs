//! Room simulation, virtual room parameters, materials, and layout

use super::types::*;
use crate::Position3D;
use serde::{Deserialize, Serialize};

/// Room simulation for telepresence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomSimulationSettings {
    /// Enable room simulation
    pub enabled: bool,

    /// Virtual room parameters
    pub virtual_room: VirtualRoomParameters,

    /// Acoustic matching
    pub acoustic_matching: AcousticMatchingSettings,

    /// Cross-room interaction
    pub cross_room_interaction: CrossRoomSettings,
}

/// Virtual room parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualRoomParameters {
    /// Room dimensions (width, height, depth in meters)
    pub dimensions: (f32, f32, f32),

    /// Room materials
    pub materials: RoomMaterials,

    /// Room layout
    pub layout: RoomLayout,

    /// Acoustic properties
    pub acoustic_properties: AcousticProperties,
}

/// Room materials configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomMaterials {
    /// Wall materials
    pub walls: Vec<MaterialProperties>,

    /// Floor material
    pub floor: MaterialProperties,

    /// Ceiling material
    pub ceiling: MaterialProperties,

    /// Furniture and objects
    pub objects: Vec<ObjectMaterial>,
}

/// Material acoustic properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    /// Material name
    pub name: String,

    /// Absorption coefficients by frequency
    pub absorption: Vec<(f32, f32)>,

    /// Scattering coefficients by frequency
    pub scattering: Vec<(f32, f32)>,

    /// Transmission coefficients
    pub transmission: Vec<(f32, f32)>,
}

impl MaterialProperties {
    /// Create default wall material properties (Drywall)
    pub fn default_wall() -> Self {
        Self {
            name: "Drywall".to_string(),
            absorption: vec![
                (125.0, 0.1),
                (250.0, 0.05),
                (500.0, 0.04),
                (1000.0, 0.06),
                (2000.0, 0.07),
                (4000.0, 0.09),
            ],
            scattering: vec![
                (125.0, 0.1),
                (250.0, 0.1),
                (500.0, 0.1),
                (1000.0, 0.1),
                (2000.0, 0.1),
                (4000.0, 0.1),
            ],
            transmission: vec![
                (125.0, 0.01),
                (250.0, 0.005),
                (500.0, 0.002),
                (1000.0, 0.001),
                (2000.0, 0.0005),
                (4000.0, 0.0002),
            ],
        }
    }

    /// Create default floor material properties (Carpet)
    pub fn default_floor() -> Self {
        Self {
            name: "Carpet".to_string(),
            absorption: vec![
                (125.0, 0.05),
                (250.0, 0.1),
                (500.0, 0.25),
                (1000.0, 0.45),
                (2000.0, 0.65),
                (4000.0, 0.8),
            ],
            scattering: vec![
                (125.0, 0.2),
                (250.0, 0.2),
                (500.0, 0.2),
                (1000.0, 0.2),
                (2000.0, 0.2),
                (4000.0, 0.2),
            ],
            transmission: vec![
                (125.0, 0.01),
                (250.0, 0.01),
                (500.0, 0.01),
                (1000.0, 0.01),
                (2000.0, 0.01),
                (4000.0, 0.01),
            ],
        }
    }

    /// Create default ceiling material properties (Acoustic Tile)
    pub fn default_ceiling() -> Self {
        Self {
            name: "Acoustic Tile".to_string(),
            absorption: vec![
                (125.0, 0.2),
                (250.0, 0.3),
                (500.0, 0.5),
                (1000.0, 0.7),
                (2000.0, 0.8),
                (4000.0, 0.85),
            ],
            scattering: vec![
                (125.0, 0.15),
                (250.0, 0.15),
                (500.0, 0.15),
                (1000.0, 0.15),
                (2000.0, 0.15),
                (4000.0, 0.15),
            ],
            transmission: vec![
                (125.0, 0.05),
                (250.0, 0.03),
                (500.0, 0.02),
                (1000.0, 0.01),
                (2000.0, 0.005),
                (4000.0, 0.002),
            ],
        }
    }
}

/// Object material configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMaterial {
    /// Object identifier
    pub object_id: String,

    /// Object position
    pub position: Position3D,

    /// Object dimensions
    pub dimensions: (f32, f32, f32),

    /// Material properties
    pub material: MaterialProperties,
}

/// Room layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomLayout {
    /// Room shape
    pub shape: RoomShape,

    /// Doorways and openings
    pub openings: Vec<Opening>,

    /// Furniture placement
    pub furniture: Vec<FurnitureItem>,

    /// User positions
    pub user_positions: Vec<super::spatial::UserPosition>,
}

/// Opening configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Opening {
    /// Opening identifier
    pub id: String,

    /// Opening type
    pub opening_type: OpeningType,

    /// Position and dimensions
    pub geometry: OpeningGeometry,

    /// Acoustic properties
    pub acoustic_properties: OpeningAcoustics,
}

/// Opening geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpeningGeometry {
    /// Position
    pub position: Position3D,

    /// Width (m)
    pub width: f32,

    /// Height (m)
    pub height: f32,

    /// Depth (m)
    pub depth: f32,

    /// Orientation (degrees)
    pub orientation: f32,
}

/// Opening acoustic properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpeningAcoustics {
    /// Open state (0.0 = closed, 1.0 = fully open)
    pub open_state: f32,

    /// Sound transmission coefficient
    pub transmission_coefficient: f32,

    /// Diffraction coefficient
    pub diffraction_coefficient: f32,
}

/// Furniture item configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FurnitureItem {
    /// Item identifier
    pub id: String,

    /// Furniture type
    pub furniture_type: FurnitureType,

    /// Position and size
    pub geometry: FurnitureGeometry,

    /// Acoustic impact
    pub acoustic_impact: FurnitureAcoustics,
}

/// Furniture geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FurnitureGeometry {
    /// Position
    pub position: Position3D,

    /// Dimensions (width, height, depth)
    pub dimensions: (f32, f32, f32),

    /// Rotation (degrees)
    pub rotation: f32,
}

/// Furniture acoustic properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FurnitureAcoustics {
    /// Absorption coefficient
    pub absorption: f32,

    /// Scattering coefficient
    pub scattering: f32,

    /// Occlusion factor
    pub occlusion_factor: f32,
}

/// Acoustic properties for rooms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticProperties {
    /// Reverberation time (seconds)
    pub reverb_time: f32,

    /// Early decay time (seconds)
    pub early_decay_time: f32,

    /// Clarity index
    pub clarity: f32,

    /// Definition
    pub definition: f32,

    /// Intimacy time (ms)
    pub intimacy_time: f32,

    /// Background noise level (dB)
    pub background_noise: f32,
}

/// Acoustic matching settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticMatchingSettings {
    /// Enable acoustic matching
    pub enabled: bool,

    /// Matching algorithm
    pub algorithm: AcousticMatchingAlgorithm,

    /// Matching strength (0.0-1.0)
    pub strength: f32,

    /// Real-time adaptation
    pub real_time_adaptation: bool,
}

/// Cross-room interaction settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossRoomSettings {
    /// Enable cross-room audio
    pub enabled: bool,

    /// Attenuation between rooms
    pub inter_room_attenuation: f32,

    /// Room isolation level
    pub isolation_level: f32,

    /// Shared spaces
    pub shared_spaces: Vec<SharedSpace>,
}

/// Shared space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedSpace {
    /// Space identifier
    pub id: String,

    /// Space type
    pub space_type: SharedSpaceType,

    /// Connected rooms
    pub connected_rooms: Vec<String>,

    /// Acoustic properties
    pub acoustic_properties: AcousticProperties,
}

/// Distance modeling settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceModelingSettings {
    /// Enable distance modeling
    pub enabled: bool,

    /// Attenuation model
    pub attenuation_model: AttenuationModel,

    /// Air absorption
    pub air_absorption: AirAbsorptionSettings,

    /// Maximum audible distance
    pub max_distance: f32,

    /// Near field compensation
    pub near_field_compensation: bool,
}

/// Air absorption settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirAbsorptionSettings {
    /// Enable air absorption
    pub enabled: bool,

    /// Temperature (Celsius)
    pub temperature: f32,

    /// Humidity (percentage)
    pub humidity: f32,

    /// Atmospheric pressure (Pa)
    pub pressure: f32,
}

/// Doppler effects settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DopplerEffectsSettings {
    /// Enable Doppler effects
    pub enabled: bool,

    /// Doppler factor scaling
    pub factor_scaling: f32,

    /// Maximum Doppler shift (Hz)
    pub max_shift: f32,

    /// Smoothing factor
    pub smoothing: f32,
}

// Default implementations

impl Default for RoomSimulationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            virtual_room: VirtualRoomParameters::default(),
            acoustic_matching: AcousticMatchingSettings::default(),
            cross_room_interaction: CrossRoomSettings::default(),
        }
    }
}

impl Default for VirtualRoomParameters {
    fn default() -> Self {
        Self {
            dimensions: (8.0, 3.0, 6.0), // 8m x 3m x 6m room
            materials: RoomMaterials::default(),
            layout: RoomLayout::default(),
            acoustic_properties: AcousticProperties::default(),
        }
    }
}

impl Default for RoomMaterials {
    fn default() -> Self {
        Self {
            walls: vec![MaterialProperties::default_wall()],
            floor: MaterialProperties::default_floor(),
            ceiling: MaterialProperties::default_ceiling(),
            objects: vec![],
        }
    }
}

impl Default for RoomLayout {
    fn default() -> Self {
        Self {
            shape: RoomShape::Rectangular,
            openings: vec![],
            furniture: vec![],
            user_positions: vec![],
        }
    }
}

impl Default for AcousticProperties {
    fn default() -> Self {
        Self {
            reverb_time: 0.6, // 600ms reverb time
            early_decay_time: 0.15,
            clarity: 5.0,
            definition: 0.7,
            intimacy_time: 20.0,
            background_noise: -45.0, // -45 dB background noise
        }
    }
}

impl Default for AcousticMatchingSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: AcousticMatchingAlgorithm::Direct,
            strength: 0.7,
            real_time_adaptation: false,
        }
    }
}

impl Default for CrossRoomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            inter_room_attenuation: 20.0, // 20 dB attenuation between rooms
            isolation_level: 0.8,
            shared_spaces: vec![],
        }
    }
}

impl Default for DistanceModelingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            attenuation_model: AttenuationModel::InverseSquare,
            air_absorption: AirAbsorptionSettings::default(),
            max_distance: 50.0, // 50 meters
            near_field_compensation: true,
        }
    }
}

impl Default for AirAbsorptionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            temperature: 20.0,  // 20°C
            humidity: 50.0,     // 50% relative humidity
            pressure: 101325.0, // Standard atmospheric pressure
        }
    }
}

impl Default for DopplerEffectsSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            factor_scaling: 1.0,
            max_shift: 500.0, // 500 Hz maximum shift
            smoothing: 0.8,
        }
    }
}
