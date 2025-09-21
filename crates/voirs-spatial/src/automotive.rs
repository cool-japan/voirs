//! Automotive Spatial Audio System
//!
//! This module provides spatial audio processing specifically designed for in-vehicle
//! environments, including passenger positioning, engine noise compensation, and
//! vehicle-specific acoustic modeling.

use crate::{
    room::{Room, RoomAcoustics},
    types::{Position3D, SpatialResult},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Vehicle types for acoustic modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VehicleType {
    /// Sedan car
    Sedan,
    /// SUV/truck
    SUV,
    /// Sports car
    Sports,
    /// Van/minivan
    Van,
    /// Truck/pickup
    Truck,
    /// Bus
    Bus,
    /// Motorcycle
    Motorcycle,
    /// Electric vehicle (quieter)
    Electric,
    /// Custom vehicle configuration
    Custom {
        /// Vehicle dimensions (length, width, height in meters)
        dimensions: (f32, f32, f32),
        /// Seating capacity
        seating_capacity: u8,
        /// Engine type for noise characteristics
        engine_type: EngineType,
    },
}

/// Engine types for noise modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineType {
    /// Gasoline engine
    Gasoline {
        /// Number of cylinders
        cylinders: u8,
        /// Engine displacement in liters
        displacement_l: f32,
    },
    /// Diesel engine
    Diesel {
        /// Number of cylinders
        cylinders: u8,
        /// Engine displacement in liters
        displacement_l: f32,
    },
    /// Electric motor
    Electric {
        /// Number of electric motors
        motor_count: u8,
    },
    /// Hybrid (electric + combustion)
    Hybrid {
        /// Number of electric motors
        electric_motors: u8,
        /// Number of internal combustion engine cylinders
        ice_cylinders: u8,
    },
    /// Hydrogen fuel cell
    Hydrogen,
}

/// Passenger seat positions in vehicle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeatPosition {
    /// Driver seat
    Driver,
    /// Front passenger
    FrontPassenger,
    /// Rear left passenger
    RearLeft,
    /// Rear center passenger
    RearCenter,
    /// Rear right passenger
    RearRight,
    /// Third row left
    ThirdRowLeft,
    /// Third row center
    ThirdRowCenter,
    /// Third row right
    ThirdRowRight,
    /// Custom position with coordinates
    Custom(Position3D),
}

/// Vehicle audio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleAudioConfig {
    /// Vehicle type and characteristics
    pub vehicle_type: VehicleType,
    /// Speaker configuration
    pub speaker_config: VehicleSpeakerConfig,
    /// Acoustic environment settings
    pub acoustic_config: VehicleAcousticConfig,
    /// Noise compensation settings
    pub noise_compensation: NoiseCompensationConfig,
    /// Passenger settings
    pub passenger_config: PassengerConfig,
    /// Safety settings
    pub safety_config: SafetyConfig,
}

/// Vehicle speaker system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleSpeakerConfig {
    /// Front speakers (dashboard/doors)
    pub front_speakers: Vec<VehicleSpeaker>,
    /// Rear speakers
    pub rear_speakers: Vec<VehicleSpeaker>,
    /// Subwoofers
    pub subwoofers: Vec<VehicleSpeaker>,
    /// Tweeters/high-frequency speakers
    pub tweeters: Vec<VehicleSpeaker>,
    /// Headrest speakers
    pub headrest_speakers: Vec<VehicleSpeaker>,
    /// Ceiling speakers (premium systems)
    pub ceiling_speakers: Vec<VehicleSpeaker>,
}

/// Individual vehicle speaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleSpeaker {
    /// Speaker identifier
    pub id: String,
    /// Physical position in vehicle
    pub position: Position3D,
    /// Speaker characteristics
    pub characteristics: SpeakerCharacteristics,
    /// Mounting location
    pub mounting: SpeakerMounting,
    /// Associated seat (if any)
    pub associated_seat: Option<SeatPosition>,
}

/// Speaker characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerCharacteristics {
    /// Frequency response (Hz)
    pub frequency_range: (f32, f32),
    /// Power rating (watts RMS)
    pub power_watts: f32,
    /// Impedance (ohms)
    pub impedance_ohms: f32,
    /// Driver size (inches)
    pub driver_size_inches: f32,
    /// Speaker type
    pub speaker_type: SpeakerType,
}

/// Speaker types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpeakerType {
    /// Full-range driver
    FullRange,
    /// Woofer (low frequencies)
    Woofer,
    /// Mid-range driver
    Midrange,
    /// Tweeter (high frequencies)
    Tweeter,
    /// Subwoofer (very low frequencies)
    Subwoofer,
}

/// Speaker mounting locations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpeakerMounting {
    /// Door panel mount
    Door {
        /// Which side of the vehicle
        side: VehicleSide,
    },
    /// Dashboard mount
    Dashboard {
        /// Which side of the dashboard
        side: DashboardSide,
    },
    /// A-pillar mount
    APillar {
        /// Which side of the vehicle
        side: VehicleSide,
    },
    /// Headrest mount
    Headrest {
        /// Which seat position
        seat: SeatPosition,
    },
    /// Ceiling mount
    Ceiling,
    /// Floor/trunk mount
    Floor,
    /// Custom mounting location
    Custom(String),
}

/// Vehicle sides
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VehicleSide {
    /// Left side (driver side in right-hand traffic)
    Left,
    /// Right side (passenger side in right-hand traffic)
    Right,
}

/// Dashboard sides
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardSide {
    /// Left dashboard
    Left,
    /// Center dashboard
    Center,
    /// Right dashboard
    Right,
}

/// Vehicle acoustic environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleAcousticConfig {
    /// Interior materials and surfaces
    pub interior_materials: InteriorMaterials,
    /// Window configuration
    pub windows: WindowConfig,
    /// Acoustic treatment
    pub acoustic_treatment: AcousticTreatment,
    /// Air conditioning/HVAC impact
    pub hvac_config: HvacConfig,
}

/// Interior materials affecting acoustics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteriorMaterials {
    /// Seat materials
    pub seats: MaterialType,
    /// Dashboard materials
    pub dashboard: MaterialType,
    /// Door panel materials
    pub door_panels: MaterialType,
    /// Ceiling materials
    pub ceiling: MaterialType,
    /// Floor materials
    pub floor: MaterialType,
    /// Carpet presence and type
    pub carpet: Option<MaterialType>,
}

/// Material types with acoustic properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialType {
    /// Leather (absorptive)
    Leather,
    /// Cloth/fabric (highly absorptive)
    Cloth,
    /// Plastic (reflective)
    Plastic,
    /// Metal (highly reflective)
    Metal,
    /// Glass (reflective)
    Glass,
    /// Acoustic foam (highly absorptive)
    AcousticFoam,
    /// Wood (moderately absorptive)
    Wood,
    /// Custom material with absorption coefficient
    Custom {
        /// Material absorption coefficient (0.0-1.0)
        absorption_coefficient: f32,
    },
}

/// Window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Window tinting (affects acoustics)
    pub tinting: WindowTinting,
    /// Window thickness (mm)
    pub thickness_mm: f32,
    /// Current window state
    pub window_state: WindowState,
    /// Acoustic glass features
    pub acoustic_glass: bool,
}

/// Window tinting levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowTinting {
    /// No tinting
    None,
    /// Light tinting (70%+ light transmission)
    Light,
    /// Medium tinting (50-70% light transmission)
    Medium,
    /// Dark tinting (20-50% light transmission)
    Dark,
    /// Very dark tinting (<20% light transmission)
    VeryDark,
}

/// Window state affecting acoustics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowState {
    /// All windows closed
    Closed,
    /// Partially open windows
    PartiallyOpen {
        /// Window opening percentage (0.0-100.0)
        opening_percentage: f32,
    },
    /// Fully open windows
    Open,
    /// Sunroof open
    SunroofOpen,
}

/// Acoustic treatment in vehicle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticTreatment {
    /// Sound deadening materials
    pub sound_deadening: bool,
    /// Acoustic insulation level (0.0-1.0)
    pub insulation_level: f32,
    /// Active noise control system
    pub active_noise_control: bool,
    /// Vibration damping
    pub vibration_damping: bool,
}

/// HVAC system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HvacConfig {
    /// Current fan speed (0-10)
    pub fan_speed: u8,
    /// Air recirculation mode
    pub recirculation: bool,
    /// Vent positions open
    pub open_vents: Vec<VentPosition>,
    /// Noise level (dB) at different speeds
    pub noise_levels: Vec<(u8, f32)>, // (fan_speed, noise_db)
}

/// HVAC vent positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VentPosition {
    /// Dashboard vents
    Dashboard,
    /// Floor vents
    Floor,
    /// Rear passenger vents
    RearPassenger,
    /// Defroster vents
    Defroster,
}

/// Noise compensation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseCompensationConfig {
    /// Engine noise compensation
    pub engine_compensation: EngineNoiseCompensation,
    /// Road noise compensation
    pub road_compensation: RoadNoiseCompensation,
    /// Wind noise compensation
    pub wind_compensation: WindNoiseCompensation,
    /// Adaptive volume control
    pub adaptive_volume: AdaptiveVolumeConfig,
}

/// Engine noise compensation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineNoiseCompensation {
    /// Enable engine noise compensation
    pub enabled: bool,
    /// RPM-based compensation curve
    pub rpm_compensation: Vec<(u32, f32)>, // (rpm, compensation_db)
    /// Frequency bands to compensate
    pub frequency_bands: Vec<FrequencyBand>,
    /// Engine order tracking
    pub engine_order_tracking: bool,
}

/// Road noise compensation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadNoiseCompensation {
    /// Enable road noise compensation
    pub enabled: bool,
    /// Speed-based compensation curve
    pub speed_compensation: Vec<(f32, f32)>, // (speed_kmh, compensation_db)
    /// Road surface type detection
    pub surface_detection: bool,
    /// Tire noise modeling
    pub tire_noise_model: TireNoiseModel,
}

/// Wind noise compensation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindNoiseCompensation {
    /// Enable wind noise compensation
    pub enabled: bool,
    /// Speed-based wind noise curve
    pub wind_curve: Vec<(f32, f32)>, // (speed_kmh, wind_noise_db)
    /// Window state compensation
    pub window_compensation: bool,
}

/// Adaptive volume control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveVolumeConfig {
    /// Enable adaptive volume
    pub enabled: bool,
    /// Response speed (0.0-1.0, 1.0 = instant)
    pub response_speed: f32,
    /// Maximum volume adjustment (dB)
    pub max_adjustment_db: f32,
    /// Frequency weighting for noise detection
    pub frequency_weighting: FrequencyWeighting,
}

/// Frequency weighting for measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrequencyWeighting {
    /// A-weighting (human hearing curve)
    AWeighting,
    /// C-weighting (flatter response)
    CWeighting,
    /// Z-weighting (no weighting)
    ZWeighting,
}

/// Frequency band for compensation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyBand {
    /// Center frequency (Hz)
    pub frequency_hz: f32,
    /// Bandwidth (Hz)
    pub bandwidth_hz: f32,
    /// Compensation gain (dB)
    pub compensation_db: f32,
}

/// Tire noise modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TireNoiseModel {
    /// Tire type
    pub tire_type: TireType,
    /// Tire size
    pub tire_size: String,
    /// Road surface interaction
    pub surface_interaction: Vec<(SurfaceType, f32)>, // (surface, noise_multiplier)
}

/// Tire types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TireType {
    /// Summer tires
    Summer,
    /// Winter tires
    Winter,
    /// All-season tires
    AllSeason,
    /// Performance tires
    Performance,
    /// Off-road tires
    OffRoad,
}

/// Road surface types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurfaceType {
    /// Smooth asphalt
    SmoothAsphalt,
    /// Rough asphalt
    RoughAsphalt,
    /// Concrete
    Concrete,
    /// Gravel
    Gravel,
    /// Cobblestone
    Cobblestone,
    /// Wet surface
    Wet,
    /// Snow/ice
    Snow,
}

/// Passenger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassengerConfig {
    /// Active passengers and their positions
    pub passengers: Vec<PassengerInfo>,
    /// Zone-based audio settings
    pub zone_settings: Vec<AudioZone>,
    /// Individual passenger preferences
    pub preferences: HashMap<String, PassengerPreferences>,
}

/// Individual passenger information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassengerInfo {
    /// Passenger ID
    pub id: String,
    /// Seat position
    pub seat: SeatPosition,
    /// Height (affects head position)
    pub height_cm: f32,
    /// Hearing characteristics
    pub hearing: HearingCharacteristics,
    /// Activity (affects audio needs)
    pub activity: PassengerActivity,
}

/// Hearing characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearingCharacteristics {
    /// Hearing ability level
    pub ability: HearingAbility,
    /// Preferred volume level (dB adjustment)
    pub preferred_volume_db: f32,
    /// Frequency sensitivity adjustments
    pub frequency_adjustments: Vec<FrequencyBand>,
}

/// Hearing ability levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HearingAbility {
    /// Normal hearing
    Normal,
    /// Mild hearing loss
    MildLoss,
    /// Moderate hearing loss
    ModerateLoss,
    /// Severe hearing loss
    SevereLoss,
    /// Profound hearing loss
    ProfoundLoss,
}

/// Passenger activities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PassengerActivity {
    /// Driving (requires minimal audio distraction)
    Driving,
    /// Listening to music
    Music,
    /// Phone call
    PhoneCall,
    /// Watching video
    Video,
    /// Sleeping (requires minimal disturbance)
    Sleeping,
    /// Reading/working
    Reading,
    /// Conversation
    Conversation,
}

/// Audio zones within vehicle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioZone {
    /// Zone name/identifier
    pub name: String,
    /// Seats included in this zone
    pub seats: Vec<SeatPosition>,
    /// Audio source for this zone
    pub audio_source: ZoneAudioSource,
    /// Volume level for this zone
    pub volume_level: f32,
    /// EQ settings for this zone
    pub eq_settings: Vec<FrequencyBand>,
    /// Privacy mode (sound isolation)
    pub privacy_mode: bool,
}

/// Audio sources for zones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZoneAudioSource {
    /// Radio/terrestrial broadcast
    Radio {
        /// Radio station identifier
        station: String,
    },
    /// Streaming music service
    Streaming {
        /// Streaming service name
        service: String,
        /// Content identifier
        content: String,
    },
    /// Bluetooth device
    Bluetooth {
        /// Connected device name
        device_name: String,
    },
    /// USB/auxiliary input
    AuxInput,
    /// Phone call audio
    PhoneCall,
    /// Navigation/GPS instructions
    Navigation,
    /// Vehicle system alerts
    VehicleAlerts,
    /// Entertainment system
    Entertainment {
        /// Type of entertainment content
        content_type: String,
    },
}

/// Passenger preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassengerPreferences {
    /// Preferred audio sources
    pub preferred_sources: Vec<ZoneAudioSource>,
    /// EQ preferences
    pub eq_preferences: Vec<FrequencyBand>,
    /// Volume preferences
    pub volume_preference: f32,
    /// Spatial audio preferences
    pub spatial_preferences: SpatialPreferences,
}

/// Spatial audio preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialPreferences {
    /// Preferred soundstage width
    pub soundstage_width: f32,
    /// Preferred center image position
    pub center_position: Position3D,
    /// Bass preference
    pub bass_preference: f32,
    /// Surround effect strength
    pub surround_strength: f32,
}

/// Safety configuration for automotive audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Driver distraction prevention
    pub driver_protection: DriverProtectionConfig,
    /// Emergency audio handling
    pub emergency_config: EmergencyAudioConfig,
    /// External sound awareness
    pub external_awareness: ExternalAwarenessConfig,
    /// Legal compliance settings
    pub legal_compliance: LegalComplianceConfig,
}

/// Driver protection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverProtectionConfig {
    /// Limit driver zone volume while driving
    pub limit_driver_volume: bool,
    /// Maximum driver volume while driving (dB)
    pub max_driver_volume_db: f32,
    /// Disable complex audio menus while driving
    pub disable_complex_ui: bool,
    /// Voice control only mode
    pub voice_only_mode: bool,
}

/// Emergency audio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyAudioConfig {
    /// Emergency alert priority
    pub emergency_priority: bool,
    /// Emergency alert volume (dB)
    pub emergency_volume_db: f32,
    /// Automatic media pause for emergencies
    pub auto_pause_media: bool,
    /// Emergency alert types
    pub alert_types: Vec<EmergencyAlertType>,
}

/// Emergency alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyAlertType {
    /// Collision warning
    CollisionWarning,
    /// Lane departure warning
    LaneDeparture,
    /// Emergency vehicle approaching
    EmergencyVehicle,
    /// Low fuel/battery warning
    LowFuel,
    /// Engine/system malfunction
    SystemMalfunction,
    /// Navigation critical alerts
    NavigationCritical,
}

/// External sound awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalAwarenessConfig {
    /// Maintain awareness of external sounds
    pub enable_awareness: bool,
    /// Microphone-based external sound mixing
    pub external_mic_mixing: bool,
    /// Automatic volume reduction for external events
    pub auto_volume_reduction: bool,
    /// External sound priority
    pub external_priority: f32,
}

/// Legal compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalComplianceConfig {
    /// Maximum volume limits by jurisdiction
    pub volume_limits: Vec<(String, f32)>, // (region, max_db)
    /// Required warning messages
    pub warning_messages: bool,
    /// Hearing protection features
    pub hearing_protection: bool,
    /// Usage time limits
    pub time_limits: Option<TimeLimits>,
}

/// Usage time limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeLimits {
    /// Maximum continuous usage time (minutes)
    pub max_continuous_minutes: u32,
    /// Required break duration (minutes)
    pub required_break_minutes: u32,
    /// Daily usage limit (minutes)
    pub daily_limit_minutes: u32,
}

/// Vehicle spatial audio processor
#[derive(Debug)]
pub struct VehicleAudioProcessor {
    /// Configuration
    config: VehicleAudioConfig,
    /// Current vehicle state
    vehicle_state: VehicleState,
    /// Audio zones
    zones: Vec<AudioZone>,
    /// Noise compensation engine
    noise_compensator: NoiseCompensator,
    /// Passenger manager
    passenger_manager: PassengerManager,
    /// Safety monitor
    safety_monitor: SafetyMonitor,
    /// Performance metrics
    metrics: VehicleAudioMetrics,
}

/// Current vehicle state
#[derive(Debug, Clone)]
pub struct VehicleState {
    /// Current speed (km/h)
    pub speed_kmh: f32,
    /// Engine RPM
    pub engine_rpm: u32,
    /// Gear position
    pub gear: GearPosition,
    /// Window states
    pub windows: WindowState,
    /// HVAC state
    pub hvac: HvacState,
    /// External conditions
    pub external_conditions: ExternalConditions,
}

/// Gear positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GearPosition {
    /// Park
    Park,
    /// Reverse
    Reverse,
    /// Neutral
    Neutral,
    /// Drive/forward gears
    Drive(u8),
    /// Manual gear
    Manual(u8),
}

/// HVAC current state
#[derive(Debug, Clone)]
pub struct HvacState {
    /// Fan speed (0-10)
    pub fan_speed: u8,
    /// Temperature setting (Celsius)
    pub temperature_c: f32,
    /// AC compressor active
    pub ac_active: bool,
    /// Defrost active
    pub defrost_active: bool,
}

/// External environmental conditions
#[derive(Debug, Clone)]
pub struct ExternalConditions {
    /// Temperature (Celsius)
    pub temperature_c: f32,
    /// Rain/precipitation
    pub precipitation: PrecipitationType,
    /// Wind speed (km/h)
    pub wind_speed_kmh: f32,
    /// Road surface condition
    pub road_surface: SurfaceType,
}

/// Precipitation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecipitationType {
    /// No precipitation
    None,
    /// Light rain
    LightRain,
    /// Heavy rain
    HeavyRain,
    /// Snow
    Snow,
    /// Hail
    Hail,
}

/// Noise compensation engine
#[derive(Debug)]
pub struct NoiseCompensator {
    /// Engine noise model
    engine_model: EngineNoiseModel,
    /// Road noise model
    road_model: RoadNoiseModel,
    /// Wind noise model
    wind_model: WindNoiseModel,
    /// Active compensation filters
    compensation_filters: Vec<FrequencyBand>,
}

/// Engine noise model
#[derive(Debug)]
pub struct EngineNoiseModel {
    /// Engine type
    engine_type: EngineType,
    /// Noise characteristics per RPM
    rpm_noise_map: Vec<(u32, f32)>,
    /// Frequency content
    frequency_content: Vec<FrequencyBand>,
}

/// Road noise model
#[derive(Debug)]
pub struct RoadNoiseModel {
    /// Tire characteristics
    tire_model: TireNoiseModel,
    /// Speed-dependent noise
    speed_noise_map: Vec<(f32, f32)>,
    /// Surface-dependent multipliers
    surface_multipliers: HashMap<SurfaceType, f32>,
}

/// Wind noise model
#[derive(Debug)]
pub struct WindNoiseModel {
    /// Speed-dependent wind noise
    wind_curve: Vec<(f32, f32)>,
    /// Window state impact
    window_impact: f32,
    /// Vehicle aerodynamics factor
    aero_factor: f32,
}

/// Passenger management system
#[derive(Debug)]
pub struct PassengerManager {
    /// Active passengers
    passengers: Vec<PassengerInfo>,
    /// Passenger preferences
    preferences: HashMap<String, PassengerPreferences>,
    /// Seat occupancy detection
    occupancy_detection: SeatOccupancyDetection,
}

/// Seat occupancy detection
#[derive(Debug)]
pub struct SeatOccupancyDetection {
    /// Pressure sensors
    pressure_sensors: HashMap<SeatPosition, bool>,
    /// Weight detection
    weight_detection: HashMap<SeatPosition, f32>,
    /// Automatic passenger detection
    auto_detection: bool,
}

/// Safety monitoring system
#[derive(Debug)]
pub struct SafetyMonitor {
    /// Driver attention monitoring
    driver_attention: DriverAttentionMonitor,
    /// Emergency alert system
    emergency_alerts: EmergencyAlertSystem,
    /// Volume safety limits
    volume_limits: VolumeLimits,
}

/// Driver attention monitoring
#[derive(Debug)]
pub struct DriverAttentionMonitor {
    /// Current attention level (0.0-1.0)
    attention_level: f32,
    /// Audio complexity limits
    complexity_limits: bool,
    /// Voice-only fallback
    voice_only_active: bool,
}

/// Emergency alert system
#[derive(Debug)]
pub struct EmergencyAlertSystem {
    /// Active alerts
    active_alerts: Vec<EmergencyAlert>,
    /// Alert priority queue
    alert_queue: Vec<EmergencyAlert>,
    /// Alert audio overrides
    audio_overrides: bool,
}

/// Emergency alert
#[derive(Debug, Clone)]
pub struct EmergencyAlert {
    /// Alert type
    alert_type: EmergencyAlertType,
    /// Priority level (0-10)
    priority: u8,
    /// Alert message
    message: String,
    /// Audio file/tone
    audio_cue: String,
    /// Duration (seconds)
    duration_s: f32,
}

/// Volume safety limits
#[derive(Debug)]
pub struct VolumeLimits {
    /// Current limits per zone
    zone_limits: HashMap<String, f32>,
    /// Time-based limits
    time_limits: Option<TimeLimits>,
    /// Usage tracking
    usage_tracking: UsageTracking,
}

/// Usage time tracking
#[derive(Debug)]
pub struct UsageTracking {
    /// Session start time
    session_start: chrono::DateTime<chrono::Utc>,
    /// Total session time (minutes)
    session_time_minutes: u32,
    /// High volume exposure time
    high_volume_time: u32,
    /// Break time required
    break_required: bool,
}

/// Vehicle audio performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleAudioMetrics {
    /// Overall system latency (ms)
    pub system_latency_ms: f32,
    /// Noise compensation effectiveness (%)
    pub noise_compensation_effectiveness: f32,
    /// Passenger satisfaction scores
    pub passenger_satisfaction: HashMap<String, f32>,
    /// Safety compliance score
    pub safety_compliance_score: f32,
    /// Audio quality per zone
    pub zone_quality_scores: HashMap<String, f32>,
    /// System resource usage
    pub resource_usage: ResourceUsage,
}

/// System resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage (%)
    pub cpu_usage: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f32,
    /// DSP usage (%)
    pub dsp_usage: f32,
    /// Network bandwidth (if applicable)
    pub network_usage_mbps: f32,
}

impl VehicleAudioProcessor {
    /// Create a new vehicle audio processor
    pub fn new(config: VehicleAudioConfig) -> Result<Self> {
        let zones = config.passenger_config.zone_settings.clone();

        Ok(Self {
            config: config.clone(),
            vehicle_state: VehicleState::default(),
            zones,
            noise_compensator: NoiseCompensator::new(&config.noise_compensation)?,
            passenger_manager: PassengerManager::new(&config.passenger_config)?,
            safety_monitor: SafetyMonitor::new(&config.safety_config)?,
            metrics: VehicleAudioMetrics::default(),
        })
    }

    /// Update vehicle state
    pub fn update_vehicle_state(&mut self, state: VehicleState) -> Result<()> {
        info!(
            "Updating vehicle state: speed={}km/h, rpm={}",
            state.speed_kmh, state.engine_rpm
        );
        self.vehicle_state = state;

        // Update noise compensation based on new state
        self.noise_compensator
            .update_compensation(&self.vehicle_state)?;

        // Update safety monitoring
        self.safety_monitor
            .update_safety_state(&self.vehicle_state)?;

        Ok(())
    }

    /// Process spatial audio for all zones
    pub fn process_spatial_audio(
        &mut self,
        input_audio: &[f32],
    ) -> Result<HashMap<String, Vec<f32>>> {
        let mut zone_outputs = HashMap::new();

        for zone in &self.zones {
            let compensated_audio = self
                .noise_compensator
                .apply_compensation(input_audio, &zone.name)?;

            let spatial_audio = self.apply_zone_spatial_processing(&compensated_audio, zone)?;

            zone_outputs.insert(zone.name.clone(), spatial_audio);
        }

        self.update_metrics();
        Ok(zone_outputs)
    }

    /// Add passenger to system
    pub fn add_passenger(&mut self, passenger: PassengerInfo) -> Result<()> {
        info!(
            "Adding passenger: {} at seat {:?}",
            passenger.id, passenger.seat
        );
        self.passenger_manager.add_passenger(passenger)?;
        self.update_audio_zones()?;
        Ok(())
    }

    /// Remove passenger from system
    pub fn remove_passenger(&mut self, passenger_id: &str) -> Result<()> {
        info!("Removing passenger: {}", passenger_id);
        self.passenger_manager.remove_passenger(passenger_id)?;
        self.update_audio_zones()?;
        Ok(())
    }

    /// Update audio zone configuration
    pub fn update_zone(&mut self, zone: AudioZone) -> Result<()> {
        info!("Updating audio zone: {}", zone.name);

        // Find and update the zone
        if let Some(existing_zone) = self.zones.iter_mut().find(|z| z.name == zone.name) {
            *existing_zone = zone;
        } else {
            self.zones.push(zone);
        }

        Ok(())
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &VehicleAudioMetrics {
        &self.metrics
    }

    /// Emergency alert handling
    pub fn handle_emergency_alert(&mut self, alert: EmergencyAlert) -> Result<()> {
        warn!("Handling emergency alert: {:?}", alert.alert_type);
        self.safety_monitor.handle_emergency(alert)?;
        Ok(())
    }

    fn apply_zone_spatial_processing(&self, audio: &[f32], zone: &AudioZone) -> Result<Vec<f32>> {
        // Apply EQ settings for the zone
        let mut processed = audio.to_vec();

        for eq_band in &zone.eq_settings {
            // Apply frequency band adjustments
            // (Simplified implementation - would use proper DSP filters in practice)
            let gain_linear = 10.0_f32.powf(eq_band.compensation_db / 20.0);
            for sample in &mut processed {
                *sample *= gain_linear;
            }
        }

        // Apply volume level
        for sample in &mut processed {
            *sample *= zone.volume_level;
        }

        // Apply privacy mode isolation if enabled
        if zone.privacy_mode {
            // Apply spatial isolation processing
            // (Would involve more complex spatial audio algorithms)
        }

        Ok(processed)
    }

    fn update_audio_zones(&mut self) -> Result<()> {
        // Update zones based on current passengers and their preferences
        // This would involve complex logic to optimize audio for current occupants
        Ok(())
    }

    fn update_metrics(&mut self) {
        self.metrics = VehicleAudioMetrics {
            system_latency_ms: 25.0, // Placeholder
            noise_compensation_effectiveness: 85.0,
            passenger_satisfaction: HashMap::new(),
            safety_compliance_score: 95.0,
            zone_quality_scores: HashMap::new(),
            resource_usage: ResourceUsage {
                cpu_usage: 15.0,
                memory_usage_mb: 64.0,
                dsp_usage: 45.0,
                network_usage_mbps: 0.0,
            },
        };
    }
}

impl NoiseCompensator {
    fn new(config: &NoiseCompensationConfig) -> Result<Self> {
        Ok(Self {
            engine_model: EngineNoiseModel {
                engine_type: EngineType::Gasoline {
                    cylinders: 4,
                    displacement_l: 2.0,
                },
                rpm_noise_map: config.engine_compensation.rpm_compensation.clone(),
                frequency_content: config.engine_compensation.frequency_bands.clone(),
            },
            road_model: RoadNoiseModel {
                tire_model: TireNoiseModel {
                    tire_type: TireType::AllSeason,
                    tire_size: "215/60R16".to_string(),
                    surface_interaction: vec![
                        (SurfaceType::SmoothAsphalt, 1.0),
                        (SurfaceType::RoughAsphalt, 1.3),
                        (SurfaceType::Concrete, 1.1),
                    ],
                },
                speed_noise_map: config.road_compensation.speed_compensation.clone(),
                surface_multipliers: HashMap::new(),
            },
            wind_model: WindNoiseModel {
                wind_curve: config.wind_compensation.wind_curve.clone(),
                window_impact: 1.0,
                aero_factor: 1.0,
            },
            compensation_filters: Vec::new(),
        })
    }

    fn update_compensation(&mut self, vehicle_state: &VehicleState) -> Result<()> {
        // Update compensation filters based on current vehicle state
        self.compensation_filters.clear();

        // Add engine compensation
        if let Some((_, compensation_db)) = self
            .engine_model
            .rpm_noise_map
            .iter()
            .find(|(rpm, _)| *rpm >= vehicle_state.engine_rpm)
        {
            self.compensation_filters.push(FrequencyBand {
                frequency_hz: 100.0, // Engine fundamental frequency
                bandwidth_hz: 50.0,
                compensation_db: *compensation_db,
            });
        }

        // Add road noise compensation
        if let Some((_, compensation_db)) = self
            .road_model
            .speed_noise_map
            .iter()
            .find(|(speed, _)| *speed >= vehicle_state.speed_kmh)
        {
            self.compensation_filters.push(FrequencyBand {
                frequency_hz: 1000.0, // Road noise frequency
                bandwidth_hz: 500.0,
                compensation_db: *compensation_db,
            });
        }

        Ok(())
    }

    fn apply_compensation(&self, audio: &[f32], _zone_name: &str) -> Result<Vec<f32>> {
        let mut compensated = audio.to_vec();

        for filter in &self.compensation_filters {
            // Apply frequency band compensation
            let gain_linear = 10.0_f32.powf(filter.compensation_db / 20.0);
            for sample in &mut compensated {
                *sample *= gain_linear;
            }
        }

        Ok(compensated)
    }
}

impl PassengerManager {
    fn new(config: &PassengerConfig) -> Result<Self> {
        Ok(Self {
            passengers: config.passengers.clone(),
            preferences: config.preferences.clone(),
            occupancy_detection: SeatOccupancyDetection {
                pressure_sensors: HashMap::new(),
                weight_detection: HashMap::new(),
                auto_detection: true,
            },
        })
    }

    fn add_passenger(&mut self, passenger: PassengerInfo) -> Result<()> {
        self.passengers.push(passenger);
        Ok(())
    }

    fn remove_passenger(&mut self, passenger_id: &str) -> Result<()> {
        self.passengers.retain(|p| p.id != passenger_id);
        self.preferences.remove(passenger_id);
        Ok(())
    }
}

impl SafetyMonitor {
    fn new(config: &SafetyConfig) -> Result<Self> {
        Ok(Self {
            driver_attention: DriverAttentionMonitor {
                attention_level: 1.0,
                complexity_limits: config.driver_protection.disable_complex_ui,
                voice_only_active: config.driver_protection.voice_only_mode,
            },
            emergency_alerts: EmergencyAlertSystem {
                active_alerts: Vec::new(),
                alert_queue: Vec::new(),
                audio_overrides: config.emergency_config.auto_pause_media,
            },
            volume_limits: VolumeLimits {
                zone_limits: HashMap::new(),
                time_limits: config.legal_compliance.time_limits.clone(),
                usage_tracking: UsageTracking {
                    session_start: chrono::Utc::now(),
                    session_time_minutes: 0,
                    high_volume_time: 0,
                    break_required: false,
                },
            },
        })
    }

    fn update_safety_state(&mut self, _vehicle_state: &VehicleState) -> Result<()> {
        // Update safety monitoring based on vehicle state
        Ok(())
    }

    fn handle_emergency(&mut self, alert: EmergencyAlert) -> Result<()> {
        self.emergency_alerts.active_alerts.push(alert);
        Ok(())
    }
}

impl Default for VehicleState {
    fn default() -> Self {
        Self {
            speed_kmh: 0.0,
            engine_rpm: 800, // Idle RPM
            gear: GearPosition::Park,
            windows: WindowState::Closed,
            hvac: HvacState {
                fan_speed: 3,
                temperature_c: 22.0,
                ac_active: false,
                defrost_active: false,
            },
            external_conditions: ExternalConditions {
                temperature_c: 20.0,
                precipitation: PrecipitationType::None,
                wind_speed_kmh: 5.0,
                road_surface: SurfaceType::SmoothAsphalt,
            },
        }
    }
}

impl Default for VehicleAudioMetrics {
    fn default() -> Self {
        Self {
            system_latency_ms: 0.0,
            noise_compensation_effectiveness: 100.0,
            passenger_satisfaction: HashMap::new(),
            safety_compliance_score: 100.0,
            zone_quality_scores: HashMap::new(),
            resource_usage: ResourceUsage {
                cpu_usage: 0.0,
                memory_usage_mb: 0.0,
                dsp_usage: 0.0,
                network_usage_mbps: 0.0,
            },
        }
    }
}

/// Builder for vehicle audio configuration
#[derive(Debug, Default)]
pub struct VehicleAudioConfigBuilder {
    vehicle_type: Option<VehicleType>,
    speaker_config: Option<VehicleSpeakerConfig>,
    acoustic_config: Option<VehicleAcousticConfig>,
    noise_compensation: Option<NoiseCompensationConfig>,
    passenger_config: Option<PassengerConfig>,
    safety_config: Option<SafetyConfig>,
}

impl VehicleAudioConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set vehicle type
    pub fn vehicle_type(mut self, vehicle_type: VehicleType) -> Self {
        self.vehicle_type = Some(vehicle_type);
        self
    }

    /// Set speaker configuration
    pub fn speaker_config(mut self, config: VehicleSpeakerConfig) -> Self {
        self.speaker_config = Some(config);
        self
    }

    /// Set acoustic configuration
    pub fn acoustic_config(mut self, config: VehicleAcousticConfig) -> Self {
        self.acoustic_config = Some(config);
        self
    }

    /// Set noise compensation
    pub fn noise_compensation(mut self, config: NoiseCompensationConfig) -> Self {
        self.noise_compensation = Some(config);
        self
    }

    /// Set passenger configuration
    pub fn passenger_config(mut self, config: PassengerConfig) -> Self {
        self.passenger_config = Some(config);
        self
    }

    /// Set safety configuration
    pub fn safety_config(mut self, config: SafetyConfig) -> Self {
        self.safety_config = Some(config);
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<VehicleAudioConfig> {
        Ok(VehicleAudioConfig {
            vehicle_type: self.vehicle_type.unwrap_or(VehicleType::Sedan),
            speaker_config: self.speaker_config.unwrap_or_else(|| VehicleSpeakerConfig {
                front_speakers: Vec::new(),
                rear_speakers: Vec::new(),
                subwoofers: Vec::new(),
                tweeters: Vec::new(),
                headrest_speakers: Vec::new(),
                ceiling_speakers: Vec::new(),
            }),
            acoustic_config: self
                .acoustic_config
                .unwrap_or_else(|| VehicleAcousticConfig {
                    interior_materials: InteriorMaterials {
                        seats: MaterialType::Cloth,
                        dashboard: MaterialType::Plastic,
                        door_panels: MaterialType::Plastic,
                        ceiling: MaterialType::Cloth,
                        floor: MaterialType::Custom {
                            absorption_coefficient: 0.3,
                        },
                        carpet: Some(MaterialType::Cloth),
                    },
                    windows: WindowConfig {
                        tinting: WindowTinting::Light,
                        thickness_mm: 4.0,
                        window_state: WindowState::Closed,
                        acoustic_glass: false,
                    },
                    acoustic_treatment: AcousticTreatment {
                        sound_deadening: true,
                        insulation_level: 0.7,
                        active_noise_control: false,
                        vibration_damping: true,
                    },
                    hvac_config: HvacConfig {
                        fan_speed: 3,
                        recirculation: false,
                        open_vents: vec![VentPosition::Dashboard],
                        noise_levels: vec![(0, 30.0), (5, 45.0), (10, 60.0)],
                    },
                }),
            noise_compensation: self.noise_compensation.unwrap_or_else(|| {
                NoiseCompensationConfig {
                    engine_compensation: EngineNoiseCompensation {
                        enabled: true,
                        rpm_compensation: vec![(800, 0.0), (2000, 3.0), (4000, 6.0)],
                        frequency_bands: vec![FrequencyBand {
                            frequency_hz: 100.0,
                            bandwidth_hz: 50.0,
                            compensation_db: 3.0,
                        }],
                        engine_order_tracking: false,
                    },
                    road_compensation: RoadNoiseCompensation {
                        enabled: true,
                        speed_compensation: vec![(0.0, 0.0), (60.0, 3.0), (120.0, 6.0)],
                        surface_detection: false,
                        tire_noise_model: TireNoiseModel {
                            tire_type: TireType::AllSeason,
                            tire_size: "215/60R16".to_string(),
                            surface_interaction: vec![(SurfaceType::SmoothAsphalt, 1.0)],
                        },
                    },
                    wind_compensation: WindNoiseCompensation {
                        enabled: true,
                        wind_curve: vec![(0.0, 0.0), (80.0, 2.0), (120.0, 4.0)],
                        window_compensation: true,
                    },
                    adaptive_volume: AdaptiveVolumeConfig {
                        enabled: true,
                        response_speed: 0.5,
                        max_adjustment_db: 6.0,
                        frequency_weighting: FrequencyWeighting::AWeighting,
                    },
                }
            }),
            passenger_config: self.passenger_config.unwrap_or_else(|| PassengerConfig {
                passengers: Vec::new(),
                zone_settings: vec![AudioZone {
                    name: "driver".to_string(),
                    seats: vec![SeatPosition::Driver],
                    audio_source: ZoneAudioSource::Radio {
                        station: "FM1".to_string(),
                    },
                    volume_level: 0.7,
                    eq_settings: Vec::new(),
                    privacy_mode: false,
                }],
                preferences: HashMap::new(),
            }),
            safety_config: self.safety_config.unwrap_or_else(|| SafetyConfig {
                driver_protection: DriverProtectionConfig {
                    limit_driver_volume: true,
                    max_driver_volume_db: 85.0,
                    disable_complex_ui: true,
                    voice_only_mode: false,
                },
                emergency_config: EmergencyAudioConfig {
                    emergency_priority: true,
                    emergency_volume_db: 95.0,
                    auto_pause_media: true,
                    alert_types: vec![
                        EmergencyAlertType::CollisionWarning,
                        EmergencyAlertType::LaneDeparture,
                    ],
                },
                external_awareness: ExternalAwarenessConfig {
                    enable_awareness: true,
                    external_mic_mixing: false,
                    auto_volume_reduction: true,
                    external_priority: 0.8,
                },
                legal_compliance: LegalComplianceConfig {
                    volume_limits: vec![("US".to_string(), 100.0)],
                    warning_messages: true,
                    hearing_protection: true,
                    time_limits: None,
                },
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicle_audio_processor_creation() {
        let config = VehicleAudioConfigBuilder::new()
            .vehicle_type(VehicleType::Sedan)
            .build()
            .unwrap();

        let processor = VehicleAudioProcessor::new(config).unwrap();
        assert_eq!(processor.zones.len(), 1); // Default driver zone
    }

    #[test]
    fn test_vehicle_state_update() {
        let config = VehicleAudioConfigBuilder::new().build().unwrap();
        let mut processor = VehicleAudioProcessor::new(config).unwrap();

        let new_state = VehicleState {
            speed_kmh: 60.0,
            engine_rpm: 2000,
            gear: GearPosition::Drive(4),
            ..Default::default()
        };

        processor.update_vehicle_state(new_state).unwrap();
        assert_eq!(processor.vehicle_state.speed_kmh, 60.0);
        assert_eq!(processor.vehicle_state.engine_rpm, 2000);
    }

    #[test]
    fn test_passenger_management() {
        let config = VehicleAudioConfigBuilder::new().build().unwrap();
        let mut processor = VehicleAudioProcessor::new(config).unwrap();

        let passenger = PassengerInfo {
            id: "passenger1".to_string(),
            seat: SeatPosition::FrontPassenger,
            height_cm: 175.0,
            hearing: HearingCharacteristics {
                ability: HearingAbility::Normal,
                preferred_volume_db: 0.0,
                frequency_adjustments: Vec::new(),
            },
            activity: PassengerActivity::Music,
        };

        processor.add_passenger(passenger).unwrap();
        assert_eq!(processor.passenger_manager.passengers.len(), 1);

        processor.remove_passenger("passenger1").unwrap();
        assert_eq!(processor.passenger_manager.passengers.len(), 0);
    }

    #[test]
    fn test_noise_compensation() {
        let config = NoiseCompensationConfig {
            engine_compensation: EngineNoiseCompensation {
                enabled: true,
                rpm_compensation: vec![(1000, 2.0), (2000, 4.0)],
                frequency_bands: vec![FrequencyBand {
                    frequency_hz: 100.0,
                    bandwidth_hz: 50.0,
                    compensation_db: 3.0,
                }],
                engine_order_tracking: false,
            },
            road_compensation: RoadNoiseCompensation {
                enabled: true,
                speed_compensation: vec![(50.0, 2.0), (100.0, 4.0)],
                surface_detection: false,
                tire_noise_model: TireNoiseModel {
                    tire_type: TireType::AllSeason,
                    tire_size: "215/60R16".to_string(),
                    surface_interaction: Vec::new(),
                },
            },
            wind_compensation: WindNoiseCompensation {
                enabled: true,
                wind_curve: vec![(60.0, 1.0), (120.0, 3.0)],
                window_compensation: true,
            },
            adaptive_volume: AdaptiveVolumeConfig {
                enabled: true,
                response_speed: 0.5,
                max_adjustment_db: 6.0,
                frequency_weighting: FrequencyWeighting::AWeighting,
            },
        };

        let mut compensator = NoiseCompensator::new(&config).unwrap();

        // Initially no filters (they are created during update)
        assert!(compensator.compensation_filters.is_empty());

        // Create a vehicle state to trigger compensation filter creation
        let vehicle_state = VehicleState {
            speed_kmh: 60.0,
            engine_rpm: 1500, // This should trigger the compensation filter
            gear: GearPosition::Drive(3),
            windows: WindowState::Closed,
            hvac: HvacState {
                fan_speed: 2,
                temperature_c: 22.0,
                ac_active: true,
                defrost_active: false,
            },
            external_conditions: ExternalConditions {
                temperature_c: 20.0,
                wind_speed_kmh: 10.0,
                precipitation: PrecipitationType::None,
                road_surface: SurfaceType::SmoothAsphalt,
            },
        };

        // Update compensation filters with vehicle state
        compensator.update_compensation(&vehicle_state).unwrap();

        // Now filters should be populated
        assert!(!compensator.compensation_filters.is_empty());
    }

    #[test]
    fn test_vehicle_types() {
        let vehicle_types = vec![
            VehicleType::Sedan,
            VehicleType::SUV,
            VehicleType::Electric,
            VehicleType::Custom {
                dimensions: (4.5, 1.8, 1.5),
                seating_capacity: 5,
                engine_type: EngineType::Electric { motor_count: 2 },
            },
        ];

        assert_eq!(vehicle_types.len(), 4);
    }

    #[test]
    fn test_audio_zones() {
        let zone = AudioZone {
            name: "rear_passengers".to_string(),
            seats: vec![SeatPosition::RearLeft, SeatPosition::RearRight],
            audio_source: ZoneAudioSource::Entertainment {
                content_type: "movie".to_string(),
            },
            volume_level: 0.8,
            eq_settings: vec![FrequencyBand {
                frequency_hz: 1000.0,
                bandwidth_hz: 100.0,
                compensation_db: 2.0,
            }],
            privacy_mode: true,
        };

        assert_eq!(zone.seats.len(), 2);
        assert!(zone.privacy_mode);
    }

    #[test]
    fn test_emergency_alerts() {
        let alert = EmergencyAlert {
            alert_type: EmergencyAlertType::CollisionWarning,
            priority: 10,
            message: "Collision imminent!".to_string(),
            audio_cue: "collision_warning.wav".to_string(),
            duration_s: 3.0,
        };

        assert_eq!(alert.priority, 10);
        assert_eq!(alert.duration_s, 3.0);
    }

    #[test]
    fn test_material_acoustics() {
        let materials = InteriorMaterials {
            seats: MaterialType::Leather,
            dashboard: MaterialType::Plastic,
            door_panels: MaterialType::Cloth,
            ceiling: MaterialType::AcousticFoam,
            floor: MaterialType::Custom {
                absorption_coefficient: 0.4,
            },
            carpet: Some(MaterialType::Cloth),
        };

        match materials.floor {
            MaterialType::Custom {
                absorption_coefficient,
            } => {
                assert_eq!(absorption_coefficient, 0.4);
            }
            _ => panic!("Wrong material type"),
        }
    }

    #[test]
    fn test_safety_configuration() {
        let safety_config = SafetyConfig {
            driver_protection: DriverProtectionConfig {
                limit_driver_volume: true,
                max_driver_volume_db: 85.0,
                disable_complex_ui: true,
                voice_only_mode: false,
            },
            emergency_config: EmergencyAudioConfig {
                emergency_priority: true,
                emergency_volume_db: 95.0,
                auto_pause_media: true,
                alert_types: vec![EmergencyAlertType::CollisionWarning],
            },
            external_awareness: ExternalAwarenessConfig {
                enable_awareness: true,
                external_mic_mixing: true,
                auto_volume_reduction: true,
                external_priority: 0.9,
            },
            legal_compliance: LegalComplianceConfig {
                volume_limits: vec![("EU".to_string(), 95.0)],
                warning_messages: true,
                hearing_protection: true,
                time_limits: Some(TimeLimits {
                    max_continuous_minutes: 60,
                    required_break_minutes: 10,
                    daily_limit_minutes: 480,
                }),
            },
        };

        assert!(safety_config.driver_protection.limit_driver_volume);
        assert_eq!(safety_config.emergency_config.emergency_volume_db, 95.0);
    }
}
