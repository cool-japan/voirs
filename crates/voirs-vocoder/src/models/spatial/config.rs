//! Configuration for 3D spatial audio vocoder.

use serde::{Deserialize, Serialize};

/// Configuration for 3D spatial audio vocoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialVocoderConfig {
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Hop size for frame processing
    pub hop_size: usize,
    /// HRTF processing configuration
    pub hrtf: HrtfConfig,
    /// Binaural rendering configuration
    pub binaural: BinauralConfig,
    /// 3D positioning configuration
    pub positioning: PositioningConfig,
    /// Room acoustics configuration
    pub acoustics: AcousticsConfig,
    /// Quality metrics configuration
    pub quality_metrics: SpatialQualityMetricsConfig,
}

/// Configuration for HRTF processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrtfConfig {
    /// Enable HRTF processing
    pub enable_hrtf: bool,
    /// HRTF database to use
    pub hrtf_database: HrtfDatabase,
    /// Interpolation method for HRTF
    pub interpolation_method: HrtfInterpolation,
    /// Head circumference for personalization (cm)
    pub head_circumference: f32,
    /// Enable personalization
    pub enable_personalization: bool,
    /// Distance model for HRTF
    pub distance_model: DistanceModel,
}

/// Configuration for binaural rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinauralConfig {
    /// Enable binaural rendering
    pub enable_binaural: bool,
    /// Crossfeed amount (0.0-1.0)
    pub crossfeed_amount: f32,
    /// Enable head tracking
    pub enable_head_tracking: bool,
    /// Head tracking update rate (Hz)
    pub head_tracking_rate: f32,
    /// Enable dynamic range compression
    pub enable_compression: bool,
    /// Compression threshold (dB)
    pub compression_threshold: f32,
    /// Compression ratio
    pub compression_ratio: f32,
}

/// Configuration for 3D positioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositioningConfig {
    /// Enable 3D positioning
    pub enable_positioning: bool,
    /// Coordinate system to use
    pub coordinate_system: CoordinateSystem,
    /// Maximum distance for audio sources
    pub max_distance: f32,
    /// Minimum distance for audio sources
    pub min_distance: f32,
    /// Distance attenuation model
    pub attenuation_model: AttenuationModel,
    /// Doppler effect configuration
    pub doppler_config: DopplerConfig,
}

/// Configuration for room acoustics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticsConfig {
    /// Enable room acoustics simulation
    pub enable_acoustics: bool,
    /// Room size (width, height, depth in meters)
    pub room_size: (f32, f32, f32),
    /// Room material properties
    pub room_materials: RoomMaterials,
    /// Reverb configuration
    pub reverb_config: ReverbConfig,
    /// Early reflections configuration
    pub early_reflections_config: EarlyReflectionsConfig,
    /// Air absorption configuration
    pub air_absorption_config: AirAbsorptionConfig,
}

/// Configuration for spatial quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQualityMetricsConfig {
    /// Enable quality metrics calculation
    pub enable_metrics: bool,
    /// Calculate localization accuracy
    pub calculate_localization_accuracy: bool,
    /// Calculate spatial impression
    pub calculate_spatial_impression: bool,
    /// Calculate immersion level
    pub calculate_immersion_level: bool,
    /// Calculate binaural quality
    pub calculate_binaural_quality: bool,
}

/// HRTF database options
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HrtfDatabase {
    /// MIT KEMAR database
    MitKemar,
    /// CIPIC database
    Cipic,
    /// ARI database
    Ari,
    /// Generic database
    Generic,
}

/// HRTF interpolation methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HrtfInterpolation {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
    /// Spherical interpolation
    Spherical,
}

/// Distance models for HRTF
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DistanceModel {
    /// No distance effects
    None,
    /// Linear distance model
    Linear,
    /// Inverse distance model
    Inverse,
    /// Inverse square distance model
    InverseSquare,
    /// Exponential distance model
    Exponential,
}

/// Coordinate systems
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CoordinateSystem {
    /// Cartesian coordinates (x, y, z)
    Cartesian,
    /// Spherical coordinates (azimuth, elevation, distance)
    Spherical,
    /// Cylindrical coordinates (rho, phi, z)
    Cylindrical,
}

/// Attenuation models
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AttenuationModel {
    /// No attenuation
    None,
    /// Linear attenuation
    Linear,
    /// Inverse distance attenuation
    InverseDistance,
    /// Inverse square attenuation
    InverseSquare,
    /// Exponential attenuation
    Exponential,
}

/// Doppler effect configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DopplerConfig {
    /// Enable Doppler effect
    pub enable_doppler: bool,
    /// Speed of sound (m/s)
    pub speed_of_sound: f32,
    /// Doppler factor (0.0-1.0)
    pub doppler_factor: f32,
    /// Maximum frequency shift
    pub max_frequency_shift: f32,
}

/// Room material properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomMaterials {
    /// Wall absorption coefficient (0.0-1.0)
    pub wall_absorption: f32,
    /// Floor absorption coefficient (0.0-1.0)
    pub floor_absorption: f32,
    /// Ceiling absorption coefficient (0.0-1.0)
    pub ceiling_absorption: f32,
    /// Diffusion coefficient (0.0-1.0)
    pub diffusion: f32,
    /// Scattering coefficient (0.0-1.0)
    pub scattering: f32,
}

/// Reverb configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverbConfig {
    /// Enable reverb
    pub enable_reverb: bool,
    /// Reverb time (RT60) in seconds
    pub reverb_time: f32,
    /// Pre-delay in milliseconds
    pub pre_delay: f32,
    /// Reverb level (0.0-1.0)
    pub reverb_level: f32,
    /// High frequency damping
    pub hf_damping: f32,
    /// Low frequency damping
    pub lf_damping: f32,
}

/// Early reflections configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyReflectionsConfig {
    /// Enable early reflections
    pub enable_early_reflections: bool,
    /// Number of early reflections
    pub reflection_count: u32,
    /// Early reflection level (0.0-1.0)
    pub reflection_level: f32,
    /// Maximum reflection delay (ms)
    pub max_reflection_delay: f32,
    /// Reflection density
    pub reflection_density: f32,
}

/// Air absorption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirAbsorptionConfig {
    /// Enable air absorption
    pub enable_air_absorption: bool,
    /// Temperature (Celsius)
    pub temperature: f32,
    /// Humidity (percentage)
    pub humidity: f32,
    /// Pressure (Pa)
    pub pressure: f32,
    /// High frequency rolloff
    pub hf_rolloff: f32,
}

impl Default for SpatialVocoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            hop_size: 1024,
            hrtf: HrtfConfig::default(),
            binaural: BinauralConfig::default(),
            positioning: PositioningConfig::default(),
            acoustics: AcousticsConfig::default(),
            quality_metrics: SpatialQualityMetricsConfig::default(),
        }
    }
}

impl Default for HrtfConfig {
    fn default() -> Self {
        Self {
            enable_hrtf: true,
            hrtf_database: HrtfDatabase::MitKemar,
            interpolation_method: HrtfInterpolation::Linear,
            head_circumference: 57.0,
            enable_personalization: false,
            distance_model: DistanceModel::InverseSquare,
        }
    }
}

impl Default for BinauralConfig {
    fn default() -> Self {
        Self {
            enable_binaural: true,
            crossfeed_amount: 0.3,
            enable_head_tracking: false,
            head_tracking_rate: 60.0,
            enable_compression: false,
            compression_threshold: -20.0,
            compression_ratio: 4.0,
        }
    }
}

impl Default for PositioningConfig {
    fn default() -> Self {
        Self {
            enable_positioning: true,
            coordinate_system: CoordinateSystem::Spherical,
            max_distance: 100.0,
            min_distance: 0.1,
            attenuation_model: AttenuationModel::InverseSquare,
            doppler_config: DopplerConfig::default(),
        }
    }
}

impl Default for AcousticsConfig {
    fn default() -> Self {
        Self {
            enable_acoustics: true,
            room_size: (10.0, 3.0, 8.0), // Typical living room size
            room_materials: RoomMaterials::default(),
            reverb_config: ReverbConfig::default(),
            early_reflections_config: EarlyReflectionsConfig::default(),
            air_absorption_config: AirAbsorptionConfig::default(),
        }
    }
}

impl Default for SpatialQualityMetricsConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            calculate_localization_accuracy: true,
            calculate_spatial_impression: true,
            calculate_immersion_level: true,
            calculate_binaural_quality: true,
        }
    }
}

impl Default for DopplerConfig {
    fn default() -> Self {
        Self {
            enable_doppler: false,
            speed_of_sound: 343.0,
            doppler_factor: 1.0,
            max_frequency_shift: 1000.0,
        }
    }
}

impl Default for RoomMaterials {
    fn default() -> Self {
        Self {
            wall_absorption: 0.1,
            floor_absorption: 0.3,
            ceiling_absorption: 0.2,
            diffusion: 0.5,
            scattering: 0.3,
        }
    }
}

impl Default for ReverbConfig {
    fn default() -> Self {
        Self {
            enable_reverb: true,
            reverb_time: 0.8,
            pre_delay: 20.0,
            reverb_level: 0.2,
            hf_damping: 0.7,
            lf_damping: 0.3,
        }
    }
}

impl Default for EarlyReflectionsConfig {
    fn default() -> Self {
        Self {
            enable_early_reflections: true,
            reflection_count: 10,
            reflection_level: 0.3,
            max_reflection_delay: 80.0,
            reflection_density: 0.8,
        }
    }
}

impl Default for AirAbsorptionConfig {
    fn default() -> Self {
        Self {
            enable_air_absorption: true,
            temperature: 20.0,
            humidity: 50.0,
            pressure: 101325.0,
            hf_rolloff: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_vocoder_config_default() {
        let config = SpatialVocoderConfig::default();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.hop_size, 1024);
        assert!(config.hrtf.enable_hrtf);
        assert!(config.binaural.enable_binaural);
        assert!(config.positioning.enable_positioning);
        assert!(config.acoustics.enable_acoustics);
        assert!(config.quality_metrics.enable_metrics);
    }

    #[test]
    fn test_hrtf_config_default() {
        let config = HrtfConfig::default();
        assert!(config.enable_hrtf);
        assert_eq!(config.hrtf_database, HrtfDatabase::MitKemar);
        assert_eq!(config.interpolation_method, HrtfInterpolation::Linear);
        assert_eq!(config.head_circumference, 57.0);
        assert!(!config.enable_personalization);
        assert_eq!(config.distance_model, DistanceModel::InverseSquare);
    }

    #[test]
    fn test_binaural_config_default() {
        let config = BinauralConfig::default();
        assert!(config.enable_binaural);
        assert_eq!(config.crossfeed_amount, 0.3);
        assert!(!config.enable_head_tracking);
        assert_eq!(config.head_tracking_rate, 60.0);
        assert!(!config.enable_compression);
        assert_eq!(config.compression_threshold, -20.0);
        assert_eq!(config.compression_ratio, 4.0);
    }

    #[test]
    fn test_positioning_config_default() {
        let config = PositioningConfig::default();
        assert!(config.enable_positioning);
        assert_eq!(config.coordinate_system, CoordinateSystem::Spherical);
        assert_eq!(config.max_distance, 100.0);
        assert_eq!(config.min_distance, 0.1);
        assert_eq!(config.attenuation_model, AttenuationModel::InverseSquare);
        assert!(!config.doppler_config.enable_doppler);
    }

    #[test]
    fn test_acoustics_config_default() {
        let config = AcousticsConfig::default();
        assert!(config.enable_acoustics);
        assert_eq!(config.room_size, (10.0, 3.0, 8.0));
        assert!(config.reverb_config.enable_reverb);
        assert!(config.early_reflections_config.enable_early_reflections);
        assert!(config.air_absorption_config.enable_air_absorption);
    }

    #[test]
    fn test_room_materials_default() {
        let materials = RoomMaterials::default();
        assert_eq!(materials.wall_absorption, 0.1);
        assert_eq!(materials.floor_absorption, 0.3);
        assert_eq!(materials.ceiling_absorption, 0.2);
        assert_eq!(materials.diffusion, 0.5);
        assert_eq!(materials.scattering, 0.3);
    }

    #[test]
    fn test_reverb_config_default() {
        let config = ReverbConfig::default();
        assert!(config.enable_reverb);
        assert_eq!(config.reverb_time, 0.8);
        assert_eq!(config.pre_delay, 20.0);
        assert_eq!(config.reverb_level, 0.2);
        assert_eq!(config.hf_damping, 0.7);
        assert_eq!(config.lf_damping, 0.3);
    }

    #[test]
    fn test_doppler_config_default() {
        let config = DopplerConfig::default();
        assert!(!config.enable_doppler);
        assert_eq!(config.speed_of_sound, 343.0);
        assert_eq!(config.doppler_factor, 1.0);
        assert_eq!(config.max_frequency_shift, 1000.0);
    }

    #[test]
    fn test_enums() {
        assert_eq!(HrtfDatabase::MitKemar, HrtfDatabase::MitKemar);
        assert_eq!(HrtfInterpolation::Linear, HrtfInterpolation::Linear);
        assert_eq!(DistanceModel::InverseSquare, DistanceModel::InverseSquare);
        assert_eq!(CoordinateSystem::Spherical, CoordinateSystem::Spherical);
        assert_eq!(
            AttenuationModel::InverseSquare,
            AttenuationModel::InverseSquare
        );
    }

    #[test]
    fn test_serialization() {
        let config = SpatialVocoderConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SpatialVocoderConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.sample_rate, deserialized.sample_rate);
        assert_eq!(config.hop_size, deserialized.hop_size);
    }
}
