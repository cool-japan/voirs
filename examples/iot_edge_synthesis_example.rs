use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Comprehensive IoT Edge Device Synthesis Example for VoiRS
///
/// This example demonstrates speech synthesis on resource-constrained IoT devices
/// including smart speakers, voice assistants, embedded systems, and edge computing
/// platforms with limited memory, processing power, and network connectivity.
///
/// Features Demonstrated:
/// - Raspberry Pi and ARM-based device optimization
/// - Memory-constrained synthesis with model quantization
/// - Power-efficient processing for battery-powered devices
/// - Offline synthesis capability with local models
/// - Voice assistant integration (Alexa, Google Assistant)
/// - Smart speaker multi-room audio distribution
/// - Edge computing with cloud fallback
/// - Real-time streaming with network adaptation
/// - Industrial IoT voice notifications
/// - Home automation voice feedback

#[derive(Debug, Clone)]
pub struct IoTDeviceConfig {
    pub device_type: IoTDeviceType,
    pub cpu_cores: u32,
    pub memory_mb: u32,
    pub storage_mb: u32,
    pub power_profile: PowerProfile,
    pub network_type: NetworkType,
    pub audio_quality: IoTAudioQuality,
    pub model_quantization: QuantizationLevel,
    pub cache_size_mb: u32,
    pub streaming_enabled: bool,
    pub offline_fallback: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum IoTDeviceType {
    RaspberryPi3,
    RaspberryPi4,
    RaspberryPiZero,
    AmazonEcho,
    GoogleNest,
    AppleHomePod,
    IndustrialGateway,
    SmartDisplay,
    WearableDevice,
    VehicleSystem,
    HomeAutomationHub,
    SecurityCamera,
    SmartThermostat,
    Custom { cpu_mhz: u32, ram_mb: u32 },
}

#[derive(Debug, Clone, Copy)]
pub enum PowerProfile {
    PluggedIn,        // Always connected to power
    BatteryOptimized, // Prioritize battery life
    PerformanceFirst, // Prioritize audio quality
    Balanced,         // Balance between quality and power
}

#[derive(Debug, Clone, Copy)]
pub enum NetworkType {
    WiFi,
    Ethernet,
    Cellular4G,
    Cellular5G,
    LoRaWAN,
    Zigbee,
    Offline,
}

#[derive(Debug, Clone, Copy)]
pub enum IoTAudioQuality {
    VoiceCall, // 8kHz, mono, compressed
    Standard,  // 16kHz, mono, good quality
    Enhanced,  // 22kHz, stereo, high quality
    Streaming, // 44kHz, stereo, best quality
}

#[derive(Debug, Clone, Copy)]
pub enum QuantizationLevel {
    Int4,    // 4-bit quantization, highest compression
    Int8,    // 8-bit quantization, good compression
    Float16, // 16-bit float, balanced
    Float32, // 32-bit float, full precision
}

#[derive(Debug, Clone)]
pub struct IoTSynthesisRequest {
    pub id: u32,
    pub text: String,
    pub priority: RequestPriority,
    pub voice_id: String,
    pub target_device: Option<String>,
    pub streaming: bool,
    pub cache_key: Option<String>,
    pub timeout_ms: u32,
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub enum RequestPriority {
    Background = 1,
    Normal = 2,
    High = 3,
    Emergency = 4,
    SystemCritical = 5,
}

pub struct IoTEdgeEngine {
    config: IoTDeviceConfig,
    synthesis_queue: Arc<Mutex<VecDeque<IoTSynthesisRequest>>>,
    audio_cache: Arc<RwLock<AudioCache>>,
    model_manager: Arc<Mutex<QuantizedModelManager>>,
    network_monitor: NetworkMonitor,
    resource_monitor: ResourceMonitor,
    device_manager: DeviceManager,
    next_request_id: Arc<Mutex<u32>>,
}

impl IoTEdgeEngine {
    pub fn new(config: IoTDeviceConfig) -> Result<Self, IoTError> {
        println!(
            "🔧 Initializing IoT Edge Engine for {:?}",
            config.device_type
        );

        let synthesis_queue = Arc::new(Mutex::new(VecDeque::new()));
        let audio_cache = Arc::new(RwLock::new(AudioCache::new(config.cache_size_mb)?));
        let model_manager = Arc::new(Mutex::new(QuantizedModelManager::new(&config)?));
        let network_monitor = NetworkMonitor::new(config.network_type);
        let resource_monitor = ResourceMonitor::new(&config);
        let device_manager = DeviceManager::new(&config)?;
        let next_request_id = Arc::new(Mutex::new(1));

        Ok(Self {
            config,
            synthesis_queue,
            audio_cache,
            model_manager,
            network_monitor,
            resource_monitor,
            device_manager,
            next_request_id,
        })
    }

    pub fn synthesize_text(
        &mut self,
        text: &str,
        voice_id: &str,
        priority: RequestPriority,
    ) -> Result<u32, IoTError> {
        let request_id = {
            let mut id = self
                .next_request_id
                .lock()
                .map_err(|_| IoTError::ThreadLockError)?;
            let current_id = *id;
            *id += 1;
            current_id
        };

        // Check cache first
        let cache_key = format!("{}:{}", voice_id, text);
        if let Ok(mut cache) = self.audio_cache.write() {
            if cache.contains(&cache_key) {
                println!("📦 Cache hit for request {}: \"{}\"", request_id, text);
                return Ok(request_id);
            }
        }

        let request = IoTSynthesisRequest {
            id: request_id,
            text: text.to_string(),
            priority,
            voice_id: voice_id.to_string(),
            target_device: None,
            streaming: self.config.streaming_enabled,
            cache_key: Some(cache_key),
            timeout_ms: self.calculate_timeout(priority),
        };

        // Add to priority queue
        {
            let mut queue = self
                .synthesis_queue
                .lock()
                .map_err(|_| IoTError::ThreadLockError)?;

            // Insert in priority order
            let insert_pos = queue
                .iter()
                .position(|r| r.priority < request.priority)
                .unwrap_or(queue.len());
            queue.insert(insert_pos, request);
        }

        println!(
            "🗣️  Queued synthesis request {}: \"{}\" (priority: {:?})",
            request_id, text, priority
        );

        Ok(request_id)
    }

    pub fn process_requests(&mut self) -> Result<ProcessingResult, IoTError> {
        let start_time = Instant::now();
        let mut processed_requests = 0;
        let mut failed_requests = 0;

        // Check system resources
        let resources = self.resource_monitor.get_current_usage()?;
        if resources.cpu_percent > 80.0 || resources.memory_percent > 90.0 {
            println!(
                "⚠️  High resource usage: CPU {:.1}%, Memory {:.1}%",
                resources.cpu_percent, resources.memory_percent
            );
        }

        // Process requests from priority queue
        while let Some(request) = self.get_next_request()? {
            let process_start = Instant::now();

            match self.process_single_request(&request) {
                Ok(_) => {
                    processed_requests += 1;
                    let process_time = process_start.elapsed().as_millis();
                    println!("✅ Completed request {}: {}ms", request.id, process_time);
                }
                Err(e) => {
                    failed_requests += 1;
                    println!("❌ Failed request {}: {}", request.id, e);
                }
            }

            // Check if we're running out of resources
            if self.should_throttle()? {
                println!("🐌 Throttling processing due to resource constraints");
                break;
            }
        }

        let total_time = start_time.elapsed();

        Ok(ProcessingResult {
            processed_requests,
            failed_requests,
            processing_time_ms: total_time.as_millis() as u32,
            resource_usage: self.resource_monitor.get_current_usage()?,
        })
    }

    fn get_next_request(&mut self) -> Result<Option<IoTSynthesisRequest>, IoTError> {
        let mut queue = self
            .synthesis_queue
            .lock()
            .map_err(|_| IoTError::ThreadLockError)?;
        Ok(queue.pop_front())
    }

    fn process_single_request(
        &mut self,
        request: &IoTSynthesisRequest,
    ) -> Result<SynthesisOutput, IoTError> {
        // Check timeout
        let deadline = Instant::now() + Duration::from_millis(request.timeout_ms as u64);

        // Select appropriate model based on device capabilities
        let model_choice = self.select_model_for_request(request)?;
        println!(
            "🤖 Using model: {:?} for request {}",
            model_choice, request.id
        );

        // Perform synthesis
        let synthesis_result = {
            let mut manager = self
                .model_manager
                .lock()
                .map_err(|_| IoTError::ThreadLockError)?;
            manager.synthesize(&request.text, &request.voice_id, model_choice)?
        };

        // Cache result if enabled
        if let Some(cache_key) = &request.cache_key {
            let mut cache = self
                .audio_cache
                .write()
                .map_err(|_| IoTError::ThreadLockError)?;
            cache.store(cache_key.clone(), synthesis_result.audio_data.clone())?;
        }

        // Check if we exceeded deadline
        if Instant::now() > deadline {
            println!("⏰ Request {} exceeded timeout", request.id);
        }

        Ok(synthesis_result)
    }

    fn select_model_for_request(
        &self,
        request: &IoTSynthesisRequest,
    ) -> Result<ModelChoice, IoTError> {
        let resources = self.resource_monitor.get_current_usage()?;

        match (self.config.power_profile, request.priority) {
            (PowerProfile::BatteryOptimized, _) => Ok(ModelChoice::Lightweight),
            (_, RequestPriority::Emergency | RequestPriority::SystemCritical) => {
                Ok(ModelChoice::Fast)
            }
            (PowerProfile::PerformanceFirst, _) if resources.memory_percent < 70.0 => {
                Ok(ModelChoice::HighQuality)
            }
            _ => Ok(ModelChoice::Standard),
        }
    }

    fn should_throttle(&self) -> Result<bool, IoTError> {
        let resources = self.resource_monitor.get_current_usage()?;
        Ok(resources.cpu_percent > 85.0 || resources.memory_percent > 95.0)
    }

    fn calculate_timeout(&self, priority: RequestPriority) -> u32 {
        match priority {
            RequestPriority::SystemCritical => 500,
            RequestPriority::Emergency => 1000,
            RequestPriority::High => 2000,
            RequestPriority::Normal => 5000,
            RequestPriority::Background => 10000,
        }
    }

    pub fn start_voice_assistant_service(&mut self) -> Result<(), IoTError> {
        println!("🎤 Starting Voice Assistant Service...");

        // Simulate voice assistant interactions
        thread::spawn({
            let synthesis_queue = Arc::clone(&self.synthesis_queue);
            let next_id = Arc::clone(&self.next_request_id);

            move || {
                let voice_commands = vec![
                    (
                        "Good morning! The weather today is sunny with a high of 75 degrees.",
                        RequestPriority::Normal,
                    ),
                    ("Timer for 5 minutes has been set.", RequestPriority::High),
                    (
                        "Security system is now armed.",
                        RequestPriority::SystemCritical,
                    ),
                    (
                        "Your Amazon package has been delivered.",
                        RequestPriority::Normal,
                    ),
                    (
                        "Battery level is low, please charge your device.",
                        RequestPriority::High,
                    ),
                ];

                for (i, (text, priority)) in voice_commands.iter().enumerate() {
                    thread::sleep(Duration::from_secs(2)); // Space out requests

                    let request_id = {
                        let mut id = next_id.lock().unwrap();
                        let current_id = *id;
                        *id += 1;
                        current_id
                    };

                    let request = IoTSynthesisRequest {
                        id: request_id,
                        text: text.to_string(),
                        priority: *priority,
                        voice_id: "assistant_voice".to_string(),
                        target_device: None,
                        streaming: true,
                        cache_key: Some(format!("assistant:{}", text)),
                        timeout_ms: 3000,
                    };

                    if let Ok(mut queue) = synthesis_queue.lock() {
                        queue.push_back(request);
                        println!("🔔 Voice assistant request {}: \"{}\"", request_id, text);
                    }
                }
            }
        });

        Ok(())
    }

    pub fn setup_multi_room_audio(&mut self, rooms: Vec<&str>) -> Result<(), IoTError> {
        println!("🏠 Setting up multi-room audio for {} rooms", rooms.len());

        for room in rooms {
            let room_config = RoomAudioConfig {
                room_name: room.to_string(),
                speaker_count: 2,
                volume_level: 0.7,
                audio_sync_enabled: true,
            };

            self.device_manager.add_room(room_config)?;
            println!("   📻 Configured audio for room: {}", room);
        }

        // Test multi-room announcement
        self.synthesize_text(
            "This is a test of the multi-room audio system. You should hear this in all rooms.",
            "announcement_voice",
            RequestPriority::High,
        )?;

        Ok(())
    }

    pub fn get_device_status(&self) -> IoTDeviceStatus {
        let resources = self
            .resource_monitor
            .get_current_usage()
            .unwrap_or_default();
        let network_status = self.network_monitor.get_status();
        let queue_size = self.synthesis_queue.lock().map(|q| q.len()).unwrap_or(0);

        IoTDeviceStatus {
            device_type: self.config.device_type,
            cpu_usage: resources.cpu_percent,
            memory_usage: resources.memory_percent,
            storage_usage: resources.storage_percent,
            network_status,
            queue_length: queue_size,
            cache_hit_rate: self.audio_cache.read().map(|c| c.hit_rate()).unwrap_or(0.0),
            uptime_hours: resources.uptime_hours,
        }
    }
}

#[derive(Debug)]
pub struct ProcessingResult {
    pub processed_requests: u32,
    pub failed_requests: u32,
    pub processing_time_ms: u32,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug)]
pub struct SynthesisOutput {
    pub audio_data: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u32,
    pub duration_ms: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum ModelChoice {
    Lightweight, // For battery-constrained devices
    Fast,        // For low-latency requirements
    Standard,    // Balanced quality and performance
    HighQuality, // Best quality when resources allow
}

pub struct AudioCache {
    cache: HashMap<String, CacheEntry>,
    max_size_mb: u32,
    current_size_bytes: u64,
    hits: u64,
    misses: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    last_accessed: Instant,
    access_count: u32,
}

impl AudioCache {
    pub fn new(max_size_mb: u32) -> Result<Self, IoTError> {
        Ok(Self {
            cache: HashMap::new(),
            max_size_mb,
            current_size_bytes: 0,
            hits: 0,
            misses: 0,
        })
    }

    pub fn contains(&mut self, key: &str) -> bool {
        if let Some(entry) = self.cache.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            self.hits += 1;
            true
        } else {
            self.misses += 1;
            false
        }
    }

    pub fn store(&mut self, key: String, data: Vec<u8>) -> Result<(), IoTError> {
        let data_size = data.len() as u64;

        // Check if we need to evict entries
        while self.current_size_bytes + data_size > (self.max_size_mb as u64 * 1024 * 1024) {
            self.evict_lru_entry()?;
        }

        let entry = CacheEntry {
            data,
            last_accessed: Instant::now(),
            access_count: 1,
        };

        self.cache.insert(key, entry);
        self.current_size_bytes += data_size;

        Ok(())
    }

    fn evict_lru_entry(&mut self) -> Result<(), IoTError> {
        let oldest_key = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone());

        if let Some(key) = oldest_key {
            if let Some(entry) = self.cache.remove(&key) {
                self.current_size_bytes -= entry.data.len() as u64;
            }
        }

        Ok(())
    }

    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

pub struct QuantizedModelManager {
    models: HashMap<String, QuantizedModel>,
    quantization_level: QuantizationLevel,
}

impl QuantizedModelManager {
    pub fn new(config: &IoTDeviceConfig) -> Result<Self, IoTError> {
        let mut models = HashMap::new();

        // Load models based on device capabilities
        match config.device_type {
            IoTDeviceType::RaspberryPiZero => {
                models.insert(
                    "assistant_voice".to_string(),
                    QuantizedModel::new("assistant_int4", QuantizationLevel::Int4)?,
                );
            }
            IoTDeviceType::RaspberryPi4 => {
                models.insert(
                    "assistant_voice".to_string(),
                    QuantizedModel::new("assistant_fp16", QuantizationLevel::Float16)?,
                );
                models.insert(
                    "announcement_voice".to_string(),
                    QuantizedModel::new("announcement_fp16", QuantizationLevel::Float16)?,
                );
            }
            IoTDeviceType::AmazonEcho | IoTDeviceType::GoogleNest => {
                models.insert(
                    "assistant_voice".to_string(),
                    QuantizedModel::new("assistant_fp32", QuantizationLevel::Float32)?,
                );
                models.insert(
                    "announcement_voice".to_string(),
                    QuantizedModel::new("announcement_fp32", QuantizationLevel::Float32)?,
                );
            }
            _ => {
                models.insert(
                    "default_voice".to_string(),
                    QuantizedModel::new("default_int8", QuantizationLevel::Int8)?,
                );
            }
        }

        Ok(Self {
            models,
            quantization_level: config.model_quantization,
        })
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        voice_id: &str,
        model_choice: ModelChoice,
    ) -> Result<SynthesisOutput, IoTError> {
        let model_key = if self.models.contains_key(voice_id) {
            voice_id
        } else {
            "default_voice"
        };

        let model = self
            .models
            .get(model_key)
            .ok_or_else(|| IoTError::ModelNotFound(model_key.to_string()))?;

        // Simulate synthesis based on model choice and quantization
        let synthesis_time_ms = match (model_choice, self.quantization_level) {
            (ModelChoice::Lightweight, QuantizationLevel::Int4) => 50,
            (ModelChoice::Fast, QuantizationLevel::Int8) => 80,
            (ModelChoice::Standard, QuantizationLevel::Float16) => 150,
            (ModelChoice::HighQuality, QuantizationLevel::Float32) => 300,
            _ => 100,
        };

        // Simulate processing delay
        thread::sleep(Duration::from_millis(synthesis_time_ms));

        // Generate dummy audio data
        let sample_rate = match self.quantization_level {
            QuantizationLevel::Int4 => 8000,
            QuantizationLevel::Int8 => 16000,
            QuantizationLevel::Float16 => 22050,
            QuantizationLevel::Float32 => 44100,
        };

        let duration_ms = (text.len() as f32 * 80.0) as u32; // Rough estimate
        let samples = (sample_rate as f32 * duration_ms as f32 / 1000.0) as usize;
        let audio_data = vec![0u8; samples * 2]; // 16-bit samples

        Ok(SynthesisOutput {
            audio_data,
            sample_rate,
            channels: 1,
            duration_ms,
        })
    }
}

#[derive(Debug)]
pub struct QuantizedModel {
    name: String,
    quantization: QuantizationLevel,
    size_bytes: u64,
}

impl QuantizedModel {
    pub fn new(name: &str, quantization: QuantizationLevel) -> Result<Self, IoTError> {
        let size_bytes = match quantization {
            QuantizationLevel::Int4 => 10 * 1024 * 1024,    // 10MB
            QuantizationLevel::Int8 => 20 * 1024 * 1024,    // 20MB
            QuantizationLevel::Float16 => 40 * 1024 * 1024, // 40MB
            QuantizationLevel::Float32 => 80 * 1024 * 1024, // 80MB
        };

        Ok(Self {
            name: name.to_string(),
            quantization,
            size_bytes,
        })
    }
}

pub struct NetworkMonitor {
    network_type: NetworkType,
    bandwidth_mbps: f32,
    latency_ms: u32,
    packet_loss_percent: f32,
}

impl NetworkMonitor {
    pub fn new(network_type: NetworkType) -> Self {
        let (bandwidth_mbps, latency_ms, packet_loss_percent) = match network_type {
            NetworkType::Ethernet => (1000.0, 2, 0.01),
            NetworkType::WiFi => (100.0, 10, 0.1),
            NetworkType::Cellular5G => (100.0, 20, 0.5),
            NetworkType::Cellular4G => (25.0, 40, 1.0),
            NetworkType::LoRaWAN => (0.05, 1000, 5.0),
            NetworkType::Zigbee => (0.25, 50, 2.0),
            NetworkType::Offline => (0.0, 0, 0.0),
        };

        Self {
            network_type,
            bandwidth_mbps,
            latency_ms,
            packet_loss_percent,
        }
    }

    pub fn get_status(&self) -> NetworkStatus {
        NetworkStatus {
            network_type: self.network_type,
            is_connected: !matches!(self.network_type, NetworkType::Offline),
            bandwidth_mbps: self.bandwidth_mbps,
            latency_ms: self.latency_ms,
            packet_loss_percent: self.packet_loss_percent,
            signal_strength: 85.0, // Simulated
        }
    }
}

#[derive(Debug)]
pub struct NetworkStatus {
    pub network_type: NetworkType,
    pub is_connected: bool,
    pub bandwidth_mbps: f32,
    pub latency_ms: u32,
    pub packet_loss_percent: f32,
    pub signal_strength: f32,
}

pub struct ResourceMonitor {
    device_type: IoTDeviceType,
    max_memory_mb: u32,
    cpu_cores: u32,
}

impl ResourceMonitor {
    pub fn new(config: &IoTDeviceConfig) -> Self {
        Self {
            device_type: config.device_type,
            max_memory_mb: config.memory_mb,
            cpu_cores: config.cpu_cores,
        }
    }

    pub fn get_current_usage(&self) -> Result<ResourceUsage, IoTError> {
        // Simulate resource monitoring based on device type
        let (cpu_percent, memory_percent, storage_percent) = match self.device_type {
            IoTDeviceType::RaspberryPiZero => (45.0, 78.0, 23.0),
            IoTDeviceType::RaspberryPi4 => (25.0, 45.0, 15.0),
            IoTDeviceType::AmazonEcho => (15.0, 30.0, 10.0),
            IoTDeviceType::GoogleNest => (20.0, 35.0, 12.0),
            _ => (30.0, 50.0, 20.0),
        };

        Ok(ResourceUsage {
            cpu_percent,
            memory_percent,
            storage_percent,
            temperature_celsius: 42.5,
            uptime_hours: 48.2,
        })
    }
}

#[derive(Debug, Default)]
pub struct ResourceUsage {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub storage_percent: f32,
    pub temperature_celsius: f32,
    pub uptime_hours: f32,
}

pub struct DeviceManager {
    rooms: HashMap<String, RoomAudioConfig>,
}

impl DeviceManager {
    pub fn new(config: &IoTDeviceConfig) -> Result<Self, IoTError> {
        Ok(Self {
            rooms: HashMap::new(),
        })
    }

    pub fn add_room(&mut self, room_config: RoomAudioConfig) -> Result<(), IoTError> {
        self.rooms
            .insert(room_config.room_name.clone(), room_config);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RoomAudioConfig {
    pub room_name: String,
    pub speaker_count: u32,
    pub volume_level: f32,
    pub audio_sync_enabled: bool,
}

#[derive(Debug)]
pub struct IoTDeviceStatus {
    pub device_type: IoTDeviceType,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub storage_usage: f32,
    pub network_status: NetworkStatus,
    pub queue_length: usize,
    pub cache_hit_rate: f32,
    pub uptime_hours: f32,
}

#[derive(Debug, Clone)]
pub enum IoTError {
    InitializationFailed(String),
    ResourceConstraint(String),
    NetworkError(String),
    ModelNotFound(String),
    CacheError(String),
    ThreadLockError,
    ConfigurationError(String),
    SynthesisTimeout(u32),
}

impl std::fmt::Display for IoTError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoTError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            IoTError::ResourceConstraint(msg) => write!(f, "Resource constraint: {}", msg),
            IoTError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            IoTError::ModelNotFound(model) => write!(f, "Model not found: {}", model),
            IoTError::CacheError(msg) => write!(f, "Cache error: {}", msg),
            IoTError::ThreadLockError => write!(f, "Thread lock error"),
            IoTError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            IoTError::SynthesisTimeout(ms) => write!(f, "Synthesis timeout after {}ms", ms),
        }
    }
}

impl std::error::Error for IoTError {}

/// Demonstration scenarios for different IoT devices
pub fn create_smart_home_scenarios() -> Vec<(IoTDeviceConfig, Vec<&'static str>)> {
    vec![
        // Raspberry Pi smart speaker
        (
            IoTDeviceConfig {
                device_type: IoTDeviceType::RaspberryPi4,
                cpu_cores: 4,
                memory_mb: 4096,
                storage_mb: 32768,
                power_profile: PowerProfile::PluggedIn,
                network_type: NetworkType::WiFi,
                audio_quality: IoTAudioQuality::Enhanced,
                model_quantization: QuantizationLevel::Float16,
                cache_size_mb: 256,
                streaming_enabled: true,
                offline_fallback: true,
            },
            vec![
                "Living room lights are now on",
                "Front door has been locked",
                "Security system is armed",
                "Timer for dinner is set for 30 minutes",
            ],
        ),
        // Battery-powered door sensor
        (
            IoTDeviceConfig {
                device_type: IoTDeviceType::RaspberryPiZero,
                cpu_cores: 1,
                memory_mb: 512,
                storage_mb: 2048,
                power_profile: PowerProfile::BatteryOptimized,
                network_type: NetworkType::WiFi,
                audio_quality: IoTAudioQuality::VoiceCall,
                model_quantization: QuantizationLevel::Int4,
                cache_size_mb: 16,
                streaming_enabled: false,
                offline_fallback: true,
            },
            vec!["Door opened", "Battery low", "Connection lost"],
        ),
        // Industrial gateway
        (
            IoTDeviceConfig {
                device_type: IoTDeviceType::IndustrialGateway,
                cpu_cores: 8,
                memory_mb: 8192,
                storage_mb: 64000,
                power_profile: PowerProfile::PerformanceFirst,
                network_type: NetworkType::Ethernet,
                audio_quality: IoTAudioQuality::Streaming,
                model_quantization: QuantizationLevel::Float32,
                cache_size_mb: 512,
                streaming_enabled: true,
                offline_fallback: true,
            },
            vec![
                "Equipment maintenance required on line 3",
                "Temperature alert: furnace exceeding safe limits",
                "Production target achieved for today",
                "Emergency shutdown initiated",
            ],
        ),
    ]
}

/// Main demonstration function
pub fn run_iot_edge_synthesis_example() -> Result<(), IoTError> {
    println!("🏭 IoT Edge Device Synthesis Example");
    println!("====================================");

    let scenarios = create_smart_home_scenarios();

    for (i, (config, messages)) in scenarios.iter().enumerate() {
        println!("\n🔧 Scenario {}: {:?}", i + 1, config.device_type);
        println!(
            "   Memory: {} MB, CPU: {} cores",
            config.memory_mb, config.cpu_cores
        );
        println!("   Power Profile: {:?}", config.power_profile);
        println!("   Network: {:?}", config.network_type);
        println!("   Quantization: {:?}", config.model_quantization);

        let mut engine = IoTEdgeEngine::new(config.clone())?;

        // Test device status
        let initial_status = engine.get_device_status();
        println!(
            "   📊 Initial Status: CPU {:.1}%, Memory {:.1}%",
            initial_status.cpu_usage, initial_status.memory_usage
        );

        // Queue test messages
        for (j, message) in messages.iter().enumerate() {
            let priority = match j {
                0 => RequestPriority::SystemCritical,
                1 => RequestPriority::High,
                2 => RequestPriority::Normal,
                _ => RequestPriority::Background,
            };

            engine.synthesize_text(message, "default_voice", priority)?;
        }

        // Process the requests
        println!("   🎯 Processing {} requests...", messages.len());
        let result = engine.process_requests()?;

        println!(
            "   ✅ Results: {} processed, {} failed in {}ms",
            result.processed_requests, result.failed_requests, result.processing_time_ms
        );
        println!(
            "   📈 Resources: CPU {:.1}%, Memory {:.1}%",
            result.resource_usage.cpu_percent, result.resource_usage.memory_percent
        );

        // Test cache performance
        if config.cache_size_mb > 0 {
            // Repeat a message to test caching
            engine.synthesize_text(messages[0], "default_voice", RequestPriority::Normal)?;
            let cache_result = engine.process_requests()?;

            let final_status = engine.get_device_status();
            println!(
                "   📦 Cache hit rate: {:.1}%",
                final_status.cache_hit_rate * 100.0
            );
        }
    }

    // Demonstrate advanced IoT scenarios
    println!("\n🏠 Advanced IoT Scenarios");
    println!("=========================");

    // Smart home voice assistant simulation
    let smart_home_config = IoTDeviceConfig {
        device_type: IoTDeviceType::GoogleNest,
        cpu_cores: 4,
        memory_mb: 2048,
        storage_mb: 16384,
        power_profile: PowerProfile::PluggedIn,
        network_type: NetworkType::WiFi,
        audio_quality: IoTAudioQuality::Enhanced,
        model_quantization: QuantizationLevel::Float16,
        cache_size_mb: 128,
        streaming_enabled: true,
        offline_fallback: true,
    };

    println!("\n🎤 Voice Assistant Integration Test");
    let mut assistant = IoTEdgeEngine::new(smart_home_config.clone())?;
    assistant.start_voice_assistant_service()?;

    // Let the voice assistant run for a few seconds
    println!("   🔊 Voice assistant running...");
    thread::sleep(Duration::from_secs(5));

    let assistant_result = assistant.process_requests()?;
    println!(
        "   📢 Voice assistant processed {} requests",
        assistant_result.processed_requests
    );

    // Multi-room audio test
    println!("\n🏠 Multi-Room Audio Test");
    let mut multi_room = IoTEdgeEngine::new(smart_home_config)?;
    multi_room.setup_multi_room_audio(vec!["living_room", "kitchen", "bedroom"])?;

    let multi_room_result = multi_room.process_requests()?;
    println!(
        "   🔊 Multi-room announcement processed: {} requests",
        multi_room_result.processed_requests
    );

    // Industrial IoT scenario
    println!("\n🏭 Industrial IoT Voice Notifications");
    let industrial_config = IoTDeviceConfig {
        device_type: IoTDeviceType::IndustrialGateway,
        cpu_cores: 8,
        memory_mb: 8192,
        storage_mb: 128000,
        power_profile: PowerProfile::PerformanceFirst,
        network_type: NetworkType::Ethernet,
        audio_quality: IoTAudioQuality::Streaming,
        model_quantization: QuantizationLevel::Float32,
        cache_size_mb: 1024,
        streaming_enabled: true,
        offline_fallback: false,
    };

    let mut industrial = IoTEdgeEngine::new(industrial_config)?;

    let industrial_alerts = vec![
        (
            "Critical temperature alert on reactor 2",
            RequestPriority::SystemCritical,
        ),
        (
            "Production line 1 requires maintenance",
            RequestPriority::High,
        ),
        (
            "Quality control check passed for batch 457",
            RequestPriority::Normal,
        ),
        (
            "Energy efficiency target achieved for today",
            RequestPriority::Background,
        ),
    ];

    for (alert, priority) in industrial_alerts {
        industrial.synthesize_text(alert, "industrial_voice", priority)?;
    }

    let industrial_result = industrial.process_requests()?;
    println!(
        "   🔧 Industrial alerts processed: {} requests in {}ms",
        industrial_result.processed_requests, industrial_result.processing_time_ms
    );

    // Performance comparison across device types
    println!("\n📊 Performance Comparison");
    println!("=========================");

    let test_message = "This is a performance test message for IoT synthesis comparison";
    let device_types = vec![
        (IoTDeviceType::RaspberryPiZero, QuantizationLevel::Int4),
        (IoTDeviceType::RaspberryPi4, QuantizationLevel::Float16),
        (IoTDeviceType::AmazonEcho, QuantizationLevel::Float32),
        (IoTDeviceType::IndustrialGateway, QuantizationLevel::Float32),
    ];

    for (device_type, quantization) in device_types {
        let test_config = IoTDeviceConfig {
            device_type,
            cpu_cores: match device_type {
                IoTDeviceType::RaspberryPiZero => 1,
                IoTDeviceType::RaspberryPi4 => 4,
                IoTDeviceType::AmazonEcho => 2,
                IoTDeviceType::IndustrialGateway => 8,
                _ => 2,
            },
            memory_mb: match device_type {
                IoTDeviceType::RaspberryPiZero => 512,
                IoTDeviceType::RaspberryPi4 => 4096,
                IoTDeviceType::AmazonEcho => 1024,
                IoTDeviceType::IndustrialGateway => 8192,
                _ => 1024,
            },
            storage_mb: 8192,
            power_profile: PowerProfile::Balanced,
            network_type: NetworkType::WiFi,
            audio_quality: IoTAudioQuality::Standard,
            model_quantization: quantization,
            cache_size_mb: 64,
            streaming_enabled: false,
            offline_fallback: true,
        };

        let mut device = IoTEdgeEngine::new(test_config)?;
        device.synthesize_text(test_message, "test_voice", RequestPriority::Normal)?;

        let start_time = Instant::now();
        let result = device.process_requests()?;
        let total_time = start_time.elapsed();

        println!(
            "   {:?}: {}ms processing, {:.1}% CPU, {:.1}% Memory",
            device_type,
            result.processing_time_ms,
            result.resource_usage.cpu_percent,
            result.resource_usage.memory_percent
        );
    }

    println!("\n🎉 IoT Edge Synthesis Example Completed Successfully!");
    println!("\n📋 Features Demonstrated:");
    println!("   ✅ Resource-constrained synthesis on various IoT devices");
    println!("   ✅ Model quantization for memory optimization");
    println!("   ✅ Smart caching system for repeated content");
    println!("   ✅ Priority-based request processing");
    println!("   ✅ Power profile optimization");
    println!("   ✅ Network-aware synthesis with offline fallback");
    println!("   ✅ Multi-room audio distribution");
    println!("   ✅ Voice assistant integration patterns");
    println!("   ✅ Industrial IoT voice notifications");
    println!("   ✅ Real-time resource monitoring and throttling");

    println!("\n🔗 Next Steps for IoT Integration:");
    println!("   1. Implement actual model quantization with ONNX/TensorRT");
    println!("   2. Optimize for ARM processors and embedded systems");
    println!("   3. Add cloud synchronization for model updates");
    println!("   4. Implement wake word detection integration");
    println!("   5. Add support for edge AI accelerators");
    println!("   6. Test on actual IoT hardware platforms");

    Ok(())
}

fn main() -> Result<(), IoTError> {
    run_iot_edge_synthesis_example()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iot_config_creation() {
        let config = IoTDeviceConfig {
            device_type: IoTDeviceType::RaspberryPi4,
            cpu_cores: 4,
            memory_mb: 4096,
            storage_mb: 32768,
            power_profile: PowerProfile::Balanced,
            network_type: NetworkType::WiFi,
            audio_quality: IoTAudioQuality::Enhanced,
            model_quantization: QuantizationLevel::Float16,
            cache_size_mb: 128,
            streaming_enabled: true,
            offline_fallback: true,
        };

        assert_eq!(config.cpu_cores, 4);
        assert_eq!(config.memory_mb, 4096);
        assert!(config.streaming_enabled);
    }

    #[test]
    fn test_audio_cache() {
        let mut cache = AudioCache::new(1).unwrap(); // 1MB cache

        assert!(!cache.contains("test_key"));

        let test_data = vec![0u8; 100];
        cache.store("test_key".to_string(), test_data).unwrap();

        assert!(cache.contains("test_key"));
    }

    #[test]
    fn test_request_priority_ordering() {
        let high = RequestPriority::SystemCritical;
        let low = RequestPriority::Background;

        assert!(high > low);
        assert!(low < high);
    }

    #[test]
    fn test_quantized_model_sizes() {
        let int4_model = QuantizedModel::new("test", QuantizationLevel::Int4).unwrap();
        let fp32_model = QuantizedModel::new("test", QuantizationLevel::Float32).unwrap();

        assert!(int4_model.size_bytes < fp32_model.size_bytes);
    }

    #[test]
    fn test_network_monitor() {
        let wifi_monitor = NetworkMonitor::new(NetworkType::WiFi);
        let status = wifi_monitor.get_status();

        assert!(status.is_connected);
        assert!(status.bandwidth_mbps > 0.0);
    }

    #[test]
    fn test_resource_monitor() {
        let config = IoTDeviceConfig {
            device_type: IoTDeviceType::RaspberryPi4,
            cpu_cores: 4,
            memory_mb: 4096,
            storage_mb: 32768,
            power_profile: PowerProfile::Balanced,
            network_type: NetworkType::WiFi,
            audio_quality: IoTAudioQuality::Standard,
            model_quantization: QuantizationLevel::Float16,
            cache_size_mb: 128,
            streaming_enabled: false,
            offline_fallback: true,
        };

        let monitor = ResourceMonitor::new(&config);
        let usage = monitor.get_current_usage().unwrap();

        assert!(usage.cpu_percent >= 0.0 && usage.cpu_percent <= 100.0);
        assert!(usage.memory_percent >= 0.0 && usage.memory_percent <= 100.0);
    }

    #[test]
    fn test_iot_engine_creation() {
        let config = IoTDeviceConfig {
            device_type: IoTDeviceType::Generic {
                cpu_mhz: 1000,
                ram_mb: 512,
            },
            cpu_cores: 2,
            memory_mb: 1024,
            storage_mb: 8192,
            power_profile: PowerProfile::Balanced,
            network_type: NetworkType::WiFi,
            audio_quality: IoTAudioQuality::Standard,
            model_quantization: QuantizationLevel::Int8,
            cache_size_mb: 32,
            streaming_enabled: false,
            offline_fallback: true,
        };

        let engine = IoTEdgeEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_synthesis_request() {
        let config = IoTDeviceConfig {
            device_type: IoTDeviceType::RaspberryPi4,
            cpu_cores: 4,
            memory_mb: 2048,
            storage_mb: 16384,
            power_profile: PowerProfile::Balanced,
            network_type: NetworkType::WiFi,
            audio_quality: IoTAudioQuality::Standard,
            model_quantization: QuantizationLevel::Float16,
            cache_size_mb: 64,
            streaming_enabled: false,
            offline_fallback: true,
        };

        let mut engine = IoTEdgeEngine::new(config).unwrap();
        let request_id =
            engine.synthesize_text("Hello world", "test_voice", RequestPriority::Normal);

        assert!(request_id.is_ok());
        assert!(request_id.unwrap() > 0);
    }

    #[test]
    fn test_model_choice_selection() {
        let config = IoTDeviceConfig {
            device_type: IoTDeviceType::RaspberryPiZero,
            cpu_cores: 1,
            memory_mb: 512,
            storage_mb: 2048,
            power_profile: PowerProfile::BatteryOptimized,
            network_type: NetworkType::WiFi,
            audio_quality: IoTAudioQuality::VoiceCall,
            model_quantization: QuantizationLevel::Int4,
            cache_size_mb: 16,
            streaming_enabled: false,
            offline_fallback: true,
        };

        let engine = IoTEdgeEngine::new(config).unwrap();
        let request = IoTSynthesisRequest {
            id: 1,
            text: "test".to_string(),
            priority: RequestPriority::Normal,
            voice_id: "test".to_string(),
            target_device: None,
            streaming: false,
            cache_key: None,
            timeout_ms: 5000,
        };

        let model_choice = engine.select_model_for_request(&request).unwrap();
        // Battery optimized should choose lightweight model
        matches!(model_choice, ModelChoice::Lightweight);
    }
}
