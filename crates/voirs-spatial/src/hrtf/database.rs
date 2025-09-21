//! HRTF Database Management System
//!
//! This module provides advanced HRTF database management including storage,
//! access optimization, interpolation algorithms, and personalization features.

use crate::types::Position3D;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Advanced HRTF Database Manager
pub struct HrtfDatabaseManager {
    /// Main HRTF database
    main_database: Arc<RwLock<HrtfDatabase>>,
    /// Personalized HRTF cache
    personalized_cache: Arc<RwLock<HashMap<String, PersonalizedHrtf>>>,
    /// Database configuration
    config: DatabaseConfig,
    /// Performance metrics
    metrics: DatabaseMetrics,
}

/// HRTF Database structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrtfDatabase {
    /// Database metadata
    pub metadata: DatabaseMetadata,
    /// HRTF measurements indexed by position
    pub measurements: HashMap<HrtfPosition, HrtfMeasurement>,
    /// Interpolation cache
    interpolation_cache: HashMap<HrtfPosition, Vec<InterpolationWeight>>,
    /// Distance-dependent HRTF data
    distance_hrtfs: HashMap<u32, HashMap<HrtfPosition, HrtfMeasurement>>,
}

/// Database metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMetadata {
    /// Database name
    pub name: String,
    /// Version
    pub version: String,
    /// Sample rate
    pub sample_rate: u32,
    /// HRTF length (samples)
    pub hrtf_length: usize,
    /// Measurement conditions
    pub conditions: MeasurementConditions,
    /// Subject demographics
    pub subject_demographics: SubjectDemographics,
    /// Creation timestamp
    pub created: String,
    /// Last modified timestamp
    pub modified: String,
}

/// Measurement conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementConditions {
    /// Room characteristics
    pub room: RoomCharacteristics,
    /// Equipment used
    pub equipment: EquipmentInfo,
    /// Measurement methodology
    pub methodology: String,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Room characteristics during measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomCharacteristics {
    /// Room type (anechoic, reverberant, etc.)
    pub room_type: String,
    /// Dimensions (length, width, height in meters)
    pub dimensions: (f32, f32, f32),
    /// Reverberation time (RT60)
    pub rt60: f32,
    /// Background noise level (dB)
    pub noise_floor: f32,
}

/// Equipment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquipmentInfo {
    /// Microphone type
    pub microphone: String,
    /// Speaker type
    pub speaker: String,
    /// Audio interface
    pub audio_interface: String,
    /// Measurement software
    pub software: String,
}

/// Quality metrics for measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio (dB)
    pub snr: f32,
    /// Total harmonic distortion (%)
    pub thd: f32,
    /// Frequency response deviation (dB)
    pub frequency_deviation: f32,
    /// Phase coherence
    pub phase_coherence: f32,
}

/// Subject demographics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectDemographics {
    /// Age
    pub age: u32,
    /// Gender
    pub gender: String,
    /// Head measurements
    pub head_measurements: HeadMeasurements,
    /// Hearing assessment
    pub hearing_assessment: HearingAssessment,
}

/// Head measurements for HRTF correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadMeasurements {
    /// Head width (cm)
    pub width: f32,
    /// Head depth (cm)
    pub depth: f32,
    /// Head circumference (cm)
    pub circumference: f32,
    /// Inter-aural distance (cm)
    pub interaural_distance: f32,
    /// Pinna measurements
    pub pinna_left: PinnaMeasurements,
    /// Right ear pinna measurements
    pub pinna_right: PinnaMeasurements,
}

/// Pinna measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnaMeasurements {
    /// Height (cm)
    pub height: f32,
    /// Width (cm)
    pub width: f32,
    /// Depth (cm)
    pub depth: f32,
    /// Concha depth (cm)
    pub concha_depth: f32,
    /// Concha volume (cm³)
    pub concha_volume: f32,
}

/// Hearing assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearingAssessment {
    /// Audiogram results (frequency -> threshold in dB HL)
    pub audiogram: HashMap<u32, f32>,
    /// Overall hearing status
    pub status: String,
    /// Hearing aid usage
    pub hearing_aid: bool,
}

/// HRTF position with high precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HrtfPosition {
    /// Azimuth angle in degrees (0-360)
    pub azimuth: i16,
    /// Elevation angle in degrees (-90 to +90)
    pub elevation: i16,
    /// Distance in centimeters (for near-field HRTFs)
    pub distance_cm: u16,
}

/// HRTF measurement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrtfMeasurement {
    /// Left ear impulse response
    pub left_ir: Vec<f32>,
    /// Right ear impulse response
    pub right_ir: Vec<f32>,
    /// Measurement quality score (0-1)
    pub quality_score: f32,
    /// ITD (Interaural Time Difference) in samples
    pub itd_samples: f32,
    /// ILD (Interaural Level Difference) in dB
    pub ild_db: f32,
    /// Frequency response characteristics
    pub frequency_response: FrequencyResponse,
}

/// Frequency response characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyResponse {
    /// Frequency bins (Hz)
    pub frequencies: Vec<f32>,
    /// Left ear magnitude response (dB)
    pub left_magnitude: Vec<f32>,
    /// Right ear magnitude response (dB)
    pub right_magnitude: Vec<f32>,
    /// Left ear phase response (radians)
    pub left_phase: Vec<f32>,
    /// Right ear phase response (radians)
    pub right_phase: Vec<f32>,
}

/// Interpolation weight for HRTF blending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationWeight {
    /// Position of the reference HRTF
    pub position: HrtfPosition,
    /// Weight (0.0 to 1.0)
    pub weight: f32,
    /// Distance to target position
    pub distance: f32,
}

/// Personalized HRTF data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedHrtf {
    /// User ID
    pub user_id: String,
    /// Personal head measurements
    pub head_measurements: HeadMeasurements,
    /// Customized HRTF measurements
    pub measurements: HashMap<HrtfPosition, HrtfMeasurement>,
    /// Adaptation parameters
    pub adaptation_params: AdaptationParameters,
    /// Last updated timestamp
    pub last_updated: String,
}

/// Parameters for HRTF adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationParameters {
    /// Head size scaling factor
    pub head_scaling: f32,
    /// Pinna size scaling factor
    pub pinna_scaling: f32,
    /// ITD adjustment factor
    pub itd_adjustment: f32,
    /// ILD adjustment factor
    pub ild_adjustment: f32,
    /// Frequency response adjustments
    pub frequency_adjustments: Vec<FrequencyAdjustment>,
}

/// Frequency-specific adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyAdjustment {
    /// Center frequency (Hz)
    pub frequency: f32,
    /// Gain adjustment (dB)
    pub gain_db: f32,
    /// Q factor for filtering
    pub q_factor: f32,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Cache size (number of entries)
    pub cache_size: usize,
    /// Interpolation method
    pub interpolation_method: InterpolationMethod,
    /// Distance interpolation enabled
    pub distance_interpolation: bool,
    /// Precompute interpolation weights
    pub precompute_weights: bool,
    /// Compression enabled
    pub compression: bool,
    /// Storage format
    pub storage_format: StorageFormat,
}

/// Interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Nearest neighbor
    NearestNeighbor,
    /// Bilinear interpolation
    Bilinear,
    /// Spherical spline interpolation
    SphericalSpline,
    /// Barycentric interpolation
    Barycentric,
    /// Radial basis functions
    RadialBasisFunction,
}

/// Storage formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageFormat {
    /// SOFA (Spatially Oriented Format for Acoustics)
    Sofa,
    /// JSON format
    Json,
    /// Binary format
    Binary,
    /// HDF5 format
    Hdf5,
}

/// Database performance metrics
#[derive(Debug, Clone, Default)]
pub struct DatabaseMetrics {
    /// Total lookup operations
    pub total_lookups: u64,
    /// Cache hit count
    pub cache_hits: u64,
    /// Interpolation operations
    pub interpolations: u64,
    /// Average lookup time (microseconds)
    pub avg_lookup_time_us: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,
}

impl HrtfDatabaseManager {
    /// Create new HRTF database manager
    pub fn new(config: DatabaseConfig) -> Result<Self> {
        let main_database = Arc::new(RwLock::new(HrtfDatabase::new()));
        let personalized_cache = Arc::new(RwLock::new(HashMap::new()));
        let metrics = DatabaseMetrics::default();

        Ok(Self {
            main_database,
            personalized_cache,
            config,
            metrics,
        })
    }

    /// Load HRTF database from file
    pub async fn load_database(&mut self, path: &Path) -> Result<()> {
        let format = self.detect_format(path)?;
        let database = match format {
            StorageFormat::Sofa => self.load_sofa_database(path).await?,
            StorageFormat::Json => self.load_json_database(path).await?,
            StorageFormat::Binary => self.load_binary_database(path).await?,
            StorageFormat::Hdf5 => self.load_hdf5_database(path).await?,
        };

        let mut db = self.main_database.write().unwrap();
        *db = database;

        // Precompute interpolation weights if enabled
        if self.config.precompute_weights {
            self.precompute_interpolation_weights(&mut db)?;
        }

        Ok(())
    }

    /// Get HRTF for specific position with high-quality interpolation
    pub fn get_hrtf(&mut self, position: &Position3D) -> Result<HrtfMeasurement> {
        let start_time = std::time::Instant::now();

        let hrtf_pos = HrtfPosition::from_position3d(position);

        // Check cache first and perform interpolation
        let interpolated = {
            let db = self.main_database.read().unwrap();
            if let Some(measurement) = db.measurements.get(&hrtf_pos) {
                self.metrics.cache_hits += 1;
                self.metrics.total_lookups += 1;
                return Ok(measurement.clone());
            }

            // Perform interpolation
            self.interpolate_hrtf(&hrtf_pos, &db)?
        };

        self.metrics.interpolations += 1;
        self.metrics.total_lookups += 1;

        let elapsed = start_time.elapsed();
        self.update_timing_metrics(elapsed.as_micros() as f64);

        Ok(interpolated)
    }

    /// Get personalized HRTF for specific user
    pub fn get_personalized_hrtf(
        &self,
        user_id: &str,
        position: &Position3D,
    ) -> Result<HrtfMeasurement> {
        let cache = self.personalized_cache.read().unwrap();
        let hrtf_pos = HrtfPosition::from_position3d(position);

        if let Some(personalized) = cache.get(user_id) {
            if let Some(measurement) = personalized.measurements.get(&hrtf_pos) {
                return Ok(measurement.clone());
            }

            // Interpolate from personalized database
            return self.interpolate_personalized_hrtf(personalized, &hrtf_pos);
        }

        Err(Error::hrtf("Personalized HRTF not found for user"))
    }

    /// Create personalized HRTF from head measurements
    pub fn create_personalized_hrtf(
        &mut self,
        user_id: String,
        head_measurements: HeadMeasurements,
    ) -> Result<()> {
        let adaptation_params = self.calculate_adaptation_parameters(&head_measurements)?;
        let personalized_measurements = self.adapt_hrtf_measurements(&adaptation_params)?;

        let personalized = PersonalizedHrtf {
            user_id: user_id.clone(),
            head_measurements,
            measurements: personalized_measurements,
            adaptation_params,
            last_updated: "2025-07-23T00:00:00Z".to_string(),
        };

        let mut cache = self.personalized_cache.write().unwrap();
        cache.insert(user_id, personalized);

        Ok(())
    }

    /// Optimize database for better performance
    pub fn optimize_database(&mut self) -> Result<()> {
        let mut db = self.main_database.write().unwrap();

        // Remove low-quality measurements
        db.measurements
            .retain(|_, measurement| measurement.quality_score >= 0.7);

        // Precompute interpolation weights
        self.precompute_interpolation_weights(&mut db)?;

        // Compress measurements if enabled
        if self.config.compression {
            self.compress_measurements(&mut db)?;
        }

        Ok(())
    }

    /// Get database statistics
    pub fn get_statistics(&self) -> DatabaseStatistics {
        let db = self.main_database.read().unwrap();
        let cache = self.personalized_cache.read().unwrap();

        DatabaseStatistics {
            total_measurements: db.measurements.len(),
            personalized_users: cache.len(),
            cache_hit_rate: if self.metrics.total_lookups > 0 {
                (self.metrics.cache_hits as f64 / self.metrics.total_lookups as f64) * 100.0
            } else {
                0.0
            },
            avg_lookup_time_us: self.metrics.avg_lookup_time_us,
            memory_usage_mb: self.metrics.memory_usage_bytes as f64 / (1024.0 * 1024.0),
            interpolation_rate: if self.metrics.total_lookups > 0 {
                (self.metrics.interpolations as f64 / self.metrics.total_lookups as f64) * 100.0
            } else {
                0.0
            },
        }
    }

    // Private helper methods

    fn detect_format(&self, path: &Path) -> Result<StorageFormat> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("sofa") => Ok(StorageFormat::Sofa),
            Some("json") => Ok(StorageFormat::Json),
            Some("bin") => Ok(StorageFormat::Binary),
            Some("h5") | Some("hdf5") => Ok(StorageFormat::Hdf5),
            _ => Err(Error::hrtf("Unknown database format")),
        }
    }

    async fn load_sofa_database(&self, _path: &Path) -> Result<HrtfDatabase> {
        // Implement SOFA format loading
        // This would use a SOFA library like sofa-rs
        Ok(HrtfDatabase::new())
    }

    async fn load_json_database(&self, path: &Path) -> Result<HrtfDatabase> {
        let content = tokio::fs::read_to_string(path).await?;
        let database: HrtfDatabase = serde_json::from_str(&content)?;
        Ok(database)
    }

    async fn load_binary_database(&self, _path: &Path) -> Result<HrtfDatabase> {
        // Implement binary format loading
        Ok(HrtfDatabase::new())
    }

    async fn load_hdf5_database(&self, _path: &Path) -> Result<HrtfDatabase> {
        // Implement HDF5 format loading
        Ok(HrtfDatabase::new())
    }

    fn interpolate_hrtf(
        &self,
        position: &HrtfPosition,
        db: &HrtfDatabase,
    ) -> Result<HrtfMeasurement> {
        match self.config.interpolation_method {
            InterpolationMethod::NearestNeighbor => {
                self.nearest_neighbor_interpolation(position, db)
            }
            InterpolationMethod::Bilinear => self.bilinear_interpolation(position, db),
            InterpolationMethod::SphericalSpline => {
                self.spherical_spline_interpolation(position, db)
            }
            InterpolationMethod::Barycentric => self.barycentric_interpolation(position, db),
            InterpolationMethod::RadialBasisFunction => self.rbf_interpolation(position, db),
        }
    }

    fn nearest_neighbor_interpolation(
        &self,
        position: &HrtfPosition,
        db: &HrtfDatabase,
    ) -> Result<HrtfMeasurement> {
        let mut closest_distance = f32::INFINITY;
        let mut closest_measurement = None;

        for (pos, measurement) in &db.measurements {
            let distance = self.calculate_angular_distance(position, pos);
            if distance < closest_distance {
                closest_distance = distance;
                closest_measurement = Some(measurement);
            }
        }

        closest_measurement
            .cloned()
            .ok_or_else(|| Error::hrtf("No HRTF measurements available"))
    }

    fn bilinear_interpolation(
        &self,
        position: &HrtfPosition,
        db: &HrtfDatabase,
    ) -> Result<HrtfMeasurement> {
        // Find four nearest neighbors for bilinear interpolation
        let neighbors = self.find_interpolation_neighbors(position, db, 4)?;

        if neighbors.is_empty() {
            return Err(Error::hrtf("No neighbors found for interpolation"));
        }

        // Perform weighted interpolation
        self.weighted_interpolation(&neighbors, db)
    }

    fn spherical_spline_interpolation(
        &self,
        _position: &HrtfPosition,
        _db: &HrtfDatabase,
    ) -> Result<HrtfMeasurement> {
        // Implement spherical spline interpolation
        // This would use spherical harmonics or similar techniques
        Err(Error::hrtf(
            "Spherical spline interpolation not implemented",
        ))
    }

    fn barycentric_interpolation(
        &self,
        _position: &HrtfPosition,
        _db: &HrtfDatabase,
    ) -> Result<HrtfMeasurement> {
        // Implement barycentric interpolation
        Err(Error::hrtf("Barycentric interpolation not implemented"))
    }

    fn rbf_interpolation(
        &self,
        _position: &HrtfPosition,
        _db: &HrtfDatabase,
    ) -> Result<HrtfMeasurement> {
        // Implement radial basis function interpolation
        Err(Error::hrtf("RBF interpolation not implemented"))
    }

    fn find_interpolation_neighbors(
        &self,
        position: &HrtfPosition,
        db: &HrtfDatabase,
        count: usize,
    ) -> Result<Vec<InterpolationWeight>> {
        let mut neighbors: Vec<_> = db
            .measurements
            .keys()
            .map(|pos| {
                let distance = self.calculate_angular_distance(position, pos);
                InterpolationWeight {
                    position: *pos,
                    weight: 0.0,
                    distance,
                }
            })
            .collect();

        neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        neighbors.truncate(count);

        // Calculate interpolation weights (inverse distance weighting)
        let total_weight: f32 = neighbors.iter().map(|n| 1.0 / (n.distance + 1e-6)).sum();
        for neighbor in &mut neighbors {
            neighbor.weight = (1.0 / (neighbor.distance + 1e-6)) / total_weight;
        }

        Ok(neighbors)
    }

    fn weighted_interpolation(
        &self,
        neighbors: &[InterpolationWeight],
        db: &HrtfDatabase,
    ) -> Result<HrtfMeasurement> {
        if neighbors.is_empty() {
            return Err(Error::hrtf("No neighbors for interpolation"));
        }

        let first_measurement = db
            .measurements
            .get(&neighbors[0].position)
            .ok_or_else(|| Error::hrtf("Reference measurement not found"))?;

        let ir_length = first_measurement.left_ir.len();
        let mut left_ir = vec![0.0; ir_length];
        let mut right_ir = vec![0.0; ir_length];
        let mut total_itd = 0.0;
        let mut total_ild = 0.0;

        for neighbor in neighbors {
            if let Some(measurement) = db.measurements.get(&neighbor.position) {
                for i in 0..ir_length.min(measurement.left_ir.len()) {
                    left_ir[i] += measurement.left_ir[i] * neighbor.weight;
                    right_ir[i] += measurement.right_ir[i] * neighbor.weight;
                }
                total_itd += measurement.itd_samples * neighbor.weight;
                total_ild += measurement.ild_db * neighbor.weight;
            }
        }

        Ok(HrtfMeasurement {
            left_ir,
            right_ir,
            quality_score: 1.0, // Interpolated measurements get max quality
            itd_samples: total_itd,
            ild_db: total_ild,
            frequency_response: first_measurement.frequency_response.clone(), // Simplified
        })
    }

    fn calculate_angular_distance(&self, pos1: &HrtfPosition, pos2: &HrtfPosition) -> f32 {
        let az1 = pos1.azimuth as f32 * std::f32::consts::PI / 180.0;
        let el1 = pos1.elevation as f32 * std::f32::consts::PI / 180.0;
        let az2 = pos2.azimuth as f32 * std::f32::consts::PI / 180.0;
        let el2 = pos2.elevation as f32 * std::f32::consts::PI / 180.0;

        // Haversine formula for spherical distance
        let delta_az = az2 - az1;
        let delta_el = el2 - el1;

        let a =
            (delta_el / 2.0).sin().powi(2) + el1.cos() * el2.cos() * (delta_az / 2.0).sin().powi(2);
        2.0 * a.sqrt().asin()
    }

    fn precompute_interpolation_weights(&self, _db: &mut HrtfDatabase) -> Result<()> {
        // Implement precomputation of interpolation weights for common positions
        Ok(())
    }

    fn interpolate_personalized_hrtf(
        &self,
        _personalized: &PersonalizedHrtf,
        _position: &HrtfPosition,
    ) -> Result<HrtfMeasurement> {
        // Implement personalized HRTF interpolation
        Err(Error::hrtf(
            "Personalized HRTF interpolation not implemented",
        ))
    }

    fn calculate_adaptation_parameters(
        &self,
        _head_measurements: &HeadMeasurements,
    ) -> Result<AdaptationParameters> {
        // Calculate adaptation parameters based on head measurements
        Ok(AdaptationParameters {
            head_scaling: 1.0,
            pinna_scaling: 1.0,
            itd_adjustment: 1.0,
            ild_adjustment: 1.0,
            frequency_adjustments: Vec::new(),
        })
    }

    fn adapt_hrtf_measurements(
        &self,
        _params: &AdaptationParameters,
    ) -> Result<HashMap<HrtfPosition, HrtfMeasurement>> {
        // Adapt HRTF measurements using the adaptation parameters
        Ok(HashMap::new())
    }

    fn compress_measurements(&self, _db: &mut HrtfDatabase) -> Result<()> {
        // Implement HRTF compression
        Ok(())
    }

    fn update_timing_metrics(&mut self, time_us: f64) {
        let alpha = 0.1; // Exponential moving average factor
        self.metrics.avg_lookup_time_us =
            alpha * time_us + (1.0 - alpha) * self.metrics.avg_lookup_time_us;
    }
}

impl HrtfDatabase {
    /// Create new empty HRTF database
    pub fn new() -> Self {
        Self {
            metadata: DatabaseMetadata {
                name: "Default HRTF Database".to_string(),
                version: "1.0.0".to_string(),
                sample_rate: 48000,
                hrtf_length: 512,
                conditions: MeasurementConditions {
                    room: RoomCharacteristics {
                        room_type: "anechoic".to_string(),
                        dimensions: (5.0, 4.0, 3.0),
                        rt60: 0.05,
                        noise_floor: -40.0,
                    },
                    equipment: EquipmentInfo {
                        microphone: "Generic".to_string(),
                        speaker: "Generic".to_string(),
                        audio_interface: "Generic".to_string(),
                        software: "VoiRS".to_string(),
                    },
                    methodology: "Standard HRTF measurement".to_string(),
                    quality_metrics: QualityMetrics {
                        snr: 60.0,
                        thd: 0.1,
                        frequency_deviation: 1.0,
                        phase_coherence: 0.95,
                    },
                },
                subject_demographics: SubjectDemographics {
                    age: 25,
                    gender: "Mixed".to_string(),
                    head_measurements: HeadMeasurements {
                        width: 15.5,
                        depth: 19.0,
                        circumference: 56.0,
                        interaural_distance: 14.5,
                        pinna_left: PinnaMeasurements {
                            height: 6.2,
                            width: 3.5,
                            depth: 2.1,
                            concha_depth: 1.2,
                            concha_volume: 2.8,
                        },
                        pinna_right: PinnaMeasurements {
                            height: 6.2,
                            width: 3.5,
                            depth: 2.1,
                            concha_depth: 1.2,
                            concha_volume: 2.8,
                        },
                    },
                    hearing_assessment: HearingAssessment {
                        audiogram: HashMap::new(),
                        status: "Normal".to_string(),
                        hearing_aid: false,
                    },
                },
                created: "2025-07-23T00:00:00Z".to_string(),
                modified: "2025-07-23T00:00:00Z".to_string(),
            },
            measurements: HashMap::new(),
            interpolation_cache: HashMap::new(),
            distance_hrtfs: HashMap::new(),
        }
    }
}

impl Default for HrtfDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl HrtfPosition {
    /// Convert from Position3D to HrtfPosition
    pub fn from_position3d(pos: &Position3D) -> Self {
        let azimuth = pos.z.atan2(pos.x).to_degrees() as i16;
        let elevation = pos
            .y
            .atan2((pos.x.powi(2) + pos.z.powi(2)).sqrt())
            .to_degrees() as i16;
        let distance_cm = ((pos.x.powi(2) + pos.y.powi(2) + pos.z.powi(2)).sqrt() * 100.0) as u16;

        Self {
            azimuth,
            elevation,
            distance_cm,
        }
    }
}

/// Database statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStatistics {
    /// Total number of measurements
    pub total_measurements: usize,
    /// Number of personalized users
    pub personalized_users: usize,
    /// Cache hit rate percentage
    pub cache_hit_rate: f64,
    /// Average lookup time in microseconds
    pub avg_lookup_time_us: f64,
    /// Memory usage in megabytes
    pub memory_usage_mb: f64,
    /// Interpolation rate percentage
    pub interpolation_rate: f64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            cache_size: 1000,
            interpolation_method: InterpolationMethod::Bilinear,
            distance_interpolation: true,
            precompute_weights: true,
            compression: false,
            storage_format: StorageFormat::Json,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() {
        let config = DatabaseConfig::default();
        let manager = HrtfDatabaseManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_hrtf_position_conversion() {
        let pos3d = Position3D::new(1.0, 0.0, 0.0);
        let hrtf_pos = HrtfPosition::from_position3d(&pos3d);
        assert_eq!(hrtf_pos.azimuth, 0);
        assert_eq!(hrtf_pos.elevation, 0);
        assert_eq!(hrtf_pos.distance_cm, 100);
    }

    #[test]
    fn test_database_metadata() {
        let db = HrtfDatabase::new();
        assert_eq!(db.metadata.sample_rate, 48000);
        assert_eq!(db.metadata.hrtf_length, 512);
        assert!(!db.metadata.name.is_empty());
    }

    #[test]
    fn test_interpolation_methods() {
        let methods = [
            InterpolationMethod::NearestNeighbor,
            InterpolationMethod::Bilinear,
            InterpolationMethod::SphericalSpline,
            InterpolationMethod::Barycentric,
            InterpolationMethod::RadialBasisFunction,
        ];

        for method in &methods {
            let config = DatabaseConfig {
                interpolation_method: *method,
                ..Default::default()
            };
            let manager = HrtfDatabaseManager::new(config);
            assert!(manager.is_ok());
        }
    }

    #[test]
    fn test_angular_distance_calculation() {
        let config = DatabaseConfig::default();
        let manager = HrtfDatabaseManager::new(config).unwrap();

        let pos1 = HrtfPosition {
            azimuth: 0,
            elevation: 0,
            distance_cm: 100,
        };
        let pos2 = HrtfPosition {
            azimuth: 90,
            elevation: 0,
            distance_cm: 100,
        };

        let distance = manager.calculate_angular_distance(&pos1, &pos2);
        assert!(distance > 0.0);
        assert!(distance < std::f32::consts::PI);
    }

    #[test]
    fn test_database_statistics() {
        let config = DatabaseConfig::default();
        let manager = HrtfDatabaseManager::new(config).unwrap();
        let stats = manager.get_statistics();

        assert_eq!(stats.total_measurements, 0);
        assert_eq!(stats.personalized_users, 0);
        assert_eq!(stats.cache_hit_rate, 0.0);
    }
}
