//! Room simulation augmentation
//!
//! This module provides room acoustics simulation for audio augmentation
//! using impulse response convolution and parametric reverberation.

use crate::{AudioData, Result};

/// Room type enumeration
#[derive(Debug, Clone, Copy)]
pub enum RoomType {
    /// Small room (bedroom, office)
    SmallRoom,
    /// Medium room (living room, classroom)
    MediumRoom,
    /// Large room (auditorium, hall)
    LargeRoom,
    /// Concert hall
    ConcertHall,
    /// Cathedral
    Cathedral,
    /// Studio (dead room)
    Studio,
    /// Bathroom (highly reflective)
    Bathroom,
    /// Outdoor space
    Outdoor,
}

/// Room simulation configuration
#[derive(Debug, Clone)]
pub struct RoomConfig {
    /// Types of rooms to simulate
    pub room_types: Vec<RoomType>,
    /// Reverberation time (RT60) in seconds
    pub reverb_time: f32,
    /// Early reflection delay in milliseconds
    pub early_delay: f32,
    /// Late reverberation level
    pub reverb_level: f32,
    /// Damping factor for high frequencies
    pub damping: f32,
    /// Room size factor
    pub room_size: f32,
    /// Use parametric reverberation
    pub use_parametric: bool,
    /// Diffusion factor
    pub diffusion: f32,
}

impl Default for RoomConfig {
    fn default() -> Self {
        Self {
            room_types: vec![
                RoomType::SmallRoom,
                RoomType::MediumRoom,
                RoomType::LargeRoom,
            ],
            reverb_time: 1.2,
            early_delay: 20.0,
            reverb_level: 0.3,
            damping: 0.5,
            room_size: 1.0,
            use_parametric: true,
            diffusion: 0.7,
        }
    }
}

/// Room acoustics augmentor
pub struct RoomAugmentor {
    config: RoomConfig,
    sample_rate: u32,
}

impl RoomAugmentor {
    /// Create new room augmentor with configuration
    pub fn new(config: RoomConfig, sample_rate: u32) -> Self {
        Self {
            config,
            sample_rate,
        }
    }

    /// Create room augmentor with default configuration
    pub fn default(sample_rate: u32) -> Self {
        Self::new(RoomConfig::default(), sample_rate)
    }

    /// Apply room simulation to audio
    pub fn apply_room_simulation(
        &self,
        audio: &AudioData,
        room_type: RoomType,
    ) -> Result<AudioData> {
        let _samples = audio.samples();
        let _sample_rate = audio.sample_rate();
        let _channels = audio.channels();

        if self.config.use_parametric {
            self.apply_parametric_reverb(audio, room_type)
        } else {
            self.apply_impulse_response_convolution(audio, room_type)
        }
    }

    /// Generate all room variants for given audio
    pub fn generate_variants(&self, audio: &AudioData) -> Result<Vec<AudioData>> {
        let mut variants = Vec::new();

        for &room_type in &self.config.room_types {
            let reverbed = self.apply_room_simulation(audio, room_type)?;
            variants.push(reverbed);
        }

        Ok(variants)
    }

    /// Apply parametric reverberation
    fn apply_parametric_reverb(&self, audio: &AudioData, room_type: RoomType) -> Result<AudioData> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels();

        // Get room-specific parameters
        let (rt60, early_delay, level, damping) = self.get_room_parameters(room_type);

        // Create reverb processor
        let mut reverb = ParametricReverb::new(sample_rate, rt60, early_delay, level, damping);

        // Process audio
        let mut output_samples = Vec::with_capacity(samples.len());

        for &sample in samples {
            let reverbed = reverb.process_sample(sample);
            output_samples.push(reverbed);
        }

        Ok(AudioData::new(output_samples, sample_rate, channels))
    }

    /// Apply impulse response convolution
    fn apply_impulse_response_convolution(
        &self,
        audio: &AudioData,
        room_type: RoomType,
    ) -> Result<AudioData> {
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let channels = audio.channels();

        // Generate impulse response for room type
        let impulse_response = self.generate_impulse_response(room_type, sample_rate)?;

        // Convolve with impulse response
        let convolved = self.convolve(samples, &impulse_response)?;

        Ok(AudioData::new(convolved, sample_rate, channels))
    }

    /// Get room-specific parameters
    fn get_room_parameters(&self, room_type: RoomType) -> (f32, f32, f32, f32) {
        match room_type {
            RoomType::SmallRoom => (0.4, 10.0, 0.2, 0.6),
            RoomType::MediumRoom => (0.8, 15.0, 0.3, 0.5),
            RoomType::LargeRoom => (1.5, 25.0, 0.4, 0.4),
            RoomType::ConcertHall => (2.5, 40.0, 0.5, 0.3),
            RoomType::Cathedral => (4.0, 60.0, 0.6, 0.2),
            RoomType::Studio => (0.2, 5.0, 0.1, 0.8),
            RoomType::Bathroom => (1.0, 8.0, 0.4, 0.3),
            RoomType::Outdoor => (0.0, 50.0, 0.05, 0.9),
        }
    }

    /// Generate impulse response for room type
    fn generate_impulse_response(&self, room_type: RoomType, sample_rate: u32) -> Result<Vec<f32>> {
        let (rt60, early_delay, level, damping) = self.get_room_parameters(room_type);

        // Calculate impulse response length
        let ir_length = (rt60 * sample_rate as f32) as usize;
        let mut impulse_response = vec![0.0; ir_length];

        // Add direct path (delta function at start)
        if !impulse_response.is_empty() {
            impulse_response[0] = 1.0;
        }

        // Add early reflections
        let early_samples = (early_delay * sample_rate as f32 / 1000.0) as usize;
        self.add_early_reflections(&mut impulse_response, early_samples, level)?;

        // Add late reverberation
        self.add_late_reverberation(&mut impulse_response, early_samples, rt60, damping)?;

        Ok(impulse_response)
    }

    /// Add early reflections to impulse response
    fn add_early_reflections(&self, ir: &mut [f32], start_idx: usize, level: f32) -> Result<()> {
        // Simple early reflection model with exponential decay
        let num_reflections = 8;
        let reflection_spacing = start_idx / num_reflections;

        for i in 0..num_reflections {
            let idx = start_idx + i * reflection_spacing;
            if idx < ir.len() {
                let amplitude = level * 0.5_f32.powf(i as f32);
                ir[idx] += amplitude;
            }
        }

        Ok(())
    }

    /// Add late reverberation to impulse response
    fn add_late_reverberation(
        &self,
        ir: &mut [f32],
        start_idx: usize,
        rt60: f32,
        damping: f32,
    ) -> Result<()> {
        let decay_rate = -60.0 / (rt60 * self.sample_rate as f32);

        #[allow(clippy::needless_range_loop)]
        // Index needed for time calculation and noise generation
        for i in start_idx..ir.len() {
            let time = i as f32 / self.sample_rate as f32;

            // Exponential decay
            let amplitude = 10.0_f32.powf(decay_rate * time);

            // Add noise with decay envelope
            let noise = self.generate_noise_sample(i);
            let damped_noise = noise * amplitude * (1.0 - damping + damping * (-time * 5.0).exp());

            ir[i] += damped_noise * 0.1;
        }

        Ok(())
    }

    /// Generate a single noise sample (simple LCG)
    fn generate_noise_sample(&self, seed: usize) -> f32 {
        let state = (seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
        (state as f32 / 0x7fffffff as f32) * 2.0 - 1.0
    }

    /// Convolve audio with impulse response
    fn convolve(&self, signal: &[f32], impulse: &[f32]) -> Result<Vec<f32>> {
        let output_length = signal.len() + impulse.len() - 1;
        let mut output = vec![0.0; output_length];

        // Direct convolution (can be optimized with FFT for large impulse responses)
        for i in 0..signal.len() {
            for j in 0..impulse.len() {
                if i + j < output.len() {
                    output[i + j] += signal[i] * impulse[j];
                }
            }
        }

        // Truncate to original length and normalize
        output.truncate(signal.len());
        self.normalize_output(&mut output);

        Ok(output)
    }

    /// Normalize output to prevent clipping
    fn normalize_output(&self, output: &mut [f32]) {
        if output.is_empty() {
            return;
        }

        let max_amplitude = output.iter().map(|&x| x.abs()).fold(0.0, f32::max);

        if max_amplitude > 1.0 {
            let scale = 0.95 / max_amplitude;
            for sample in output.iter_mut() {
                *sample *= scale;
            }
        }
    }
}

/// Parametric reverberation processor
struct ParametricReverb {
    sample_rate: u32,
    // All-pass filters for diffusion
    allpass_filters: Vec<AllpassFilter>,
    // Comb filters for reverberation
    comb_filters: Vec<CombFilter>,
    // Mix parameters
    dry_level: f32,
    wet_level: f32,
}

impl ParametricReverb {
    /// Create new parametric reverb
    fn new(sample_rate: u32, rt60: f32, early_delay: f32, level: f32, damping: f32) -> Self {
        let mut reverb = Self {
            sample_rate,
            allpass_filters: Vec::new(),
            comb_filters: Vec::new(),
            dry_level: 1.0 - level,
            wet_level: level,
        };

        // Initialize filters
        reverb.initialize_filters(rt60, early_delay, damping);

        reverb
    }

    /// Initialize filters for reverberation
    fn initialize_filters(&mut self, rt60: f32, early_delay: f32, damping: f32) {
        let early_samples = (early_delay * self.sample_rate as f32 / 1000.0) as usize;

        // Create allpass filters for diffusion
        let allpass_delays = [89, 127, 179, 241]; // Prime numbers for decorrelation
        for &delay in &allpass_delays {
            self.allpass_filters
                .push(AllpassFilter::new(delay + early_samples, 0.7));
        }

        // Create comb filters for reverberation
        let comb_delays = [1051, 1123, 1201, 1277, 1361, 1439, 1531, 1607]; // Prime numbers
        let feedback = (-3.0 * comb_delays[0] as f32 / (rt60 * self.sample_rate as f32)).exp();

        for &delay in &comb_delays {
            self.comb_filters.push(CombFilter::new(
                delay + early_samples * 2,
                feedback,
                damping,
            ));
        }
    }

    /// Process a single sample
    fn process_sample(&mut self, input: f32) -> f32 {
        let mut wet_signal = input;

        // Process through allpass filters
        for filter in &mut self.allpass_filters {
            wet_signal = filter.process(wet_signal);
        }

        // Process through comb filters and sum
        let mut comb_sum = 0.0;
        for filter in &mut self.comb_filters {
            comb_sum += filter.process(wet_signal);
        }

        // Mix dry and wet signals
        self.dry_level * input + self.wet_level * comb_sum
    }
}

/// All-pass filter for diffusion
struct AllpassFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    delay: usize,
    gain: f32,
}

impl AllpassFilter {
    fn new(delay: usize, gain: f32) -> Self {
        Self {
            buffer: vec![0.0; delay],
            write_pos: 0,
            delay,
            gain,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let read_pos = (self.write_pos + self.buffer.len() - self.delay) % self.buffer.len();
        let delayed = self.buffer[read_pos];

        let output = -self.gain * input + delayed;
        self.buffer[self.write_pos] = input + self.gain * delayed;

        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        output
    }
}

/// Comb filter for reverberation
struct CombFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    delay: usize,
    feedback: f32,
    damping: f32,
    filter_state: f32,
}

impl CombFilter {
    fn new(delay: usize, feedback: f32, damping: f32) -> Self {
        Self {
            buffer: vec![0.0; delay],
            write_pos: 0,
            delay,
            feedback,
            damping,
            filter_state: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let read_pos = (self.write_pos + self.buffer.len() - self.delay) % self.buffer.len();
        let delayed = self.buffer[read_pos];

        // Apply damping filter (simple lowpass)
        self.filter_state = self.filter_state * (1.0 - self.damping) + delayed * self.damping;

        let output = delayed;
        self.buffer[self.write_pos] = input + self.filter_state * self.feedback;

        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        output
    }
}

/// Room simulation statistics
#[derive(Debug, Clone)]
pub struct RoomStats {
    /// Number of variants generated
    pub variants_generated: usize,
    /// Room types applied
    pub room_types: Vec<RoomType>,
    /// Processing time
    pub processing_time: std::time::Duration,
    /// Quality metrics
    pub quality_metrics: Vec<f32>,
    /// Reverberation time measurements
    pub measured_rt60: Vec<f32>,
}

impl Default for RoomStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RoomStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            variants_generated: 0,
            room_types: Vec::new(),
            processing_time: std::time::Duration::from_secs(0),
            quality_metrics: Vec::new(),
            measured_rt60: Vec::new(),
        }
    }

    /// Add variant statistics
    pub fn add_variant(&mut self, room_type: RoomType, quality: f32, rt60: f32) {
        self.variants_generated += 1;
        self.room_types.push(room_type);
        self.quality_metrics.push(quality);
        self.measured_rt60.push(rt60);
    }

    /// Set processing time
    pub fn set_processing_time(&mut self, duration: std::time::Duration) {
        self.processing_time = duration;
    }

    /// Get average quality
    pub fn average_quality(&self) -> f32 {
        if self.quality_metrics.is_empty() {
            0.0
        } else {
            self.quality_metrics.iter().sum::<f32>() / self.quality_metrics.len() as f32
        }
    }

    /// Get average measured RT60
    pub fn average_rt60(&self) -> f32 {
        if self.measured_rt60.is_empty() {
            0.0
        } else {
            self.measured_rt60.iter().sum::<f32>() / self.measured_rt60.len() as f32
        }
    }
}

/// Batch room simulation processor
pub struct BatchRoomProcessor {
    augmentor: RoomAugmentor,
}

impl BatchRoomProcessor {
    /// Create new batch processor
    pub fn new(config: RoomConfig, sample_rate: u32) -> Self {
        Self {
            augmentor: RoomAugmentor::new(config, sample_rate),
        }
    }

    /// Process multiple audio files with room simulation
    pub fn process_batch(
        &self,
        audio_files: &[AudioData],
    ) -> Result<(Vec<Vec<AudioData>>, RoomStats)> {
        let start_time = std::time::Instant::now();
        let mut all_variants = Vec::new();
        let mut stats = RoomStats::new();

        for audio in audio_files {
            let variants = self.augmentor.generate_variants(audio)?;

            // Calculate quality metrics for each variant
            for (i, variant) in variants.iter().enumerate() {
                let room_type = self.augmentor.config.room_types[i];
                let quality = calculate_audio_quality(variant);
                let rt60 = measure_rt60(variant);
                stats.add_variant(room_type, quality, rt60);
            }

            all_variants.push(variants);
        }

        let processing_time = start_time.elapsed();
        stats.set_processing_time(processing_time);

        Ok((all_variants, stats))
    }
}

/// Calculate basic audio quality metric
fn calculate_audio_quality(audio: &AudioData) -> f32 {
    let samples = audio.samples();
    if samples.is_empty() {
        return 0.0;
    }

    // Calculate signal-to-noise ratio approximation
    let energy = samples.iter().map(|&x| x * x).sum::<f32>();
    let rms = (energy / samples.len() as f32).sqrt();

    // Simple quality metric based on RMS
    (rms * 100.0).min(100.0)
}

/// Measure RT60 (reverberation time)
fn measure_rt60(audio: &AudioData) -> f32 {
    let samples = audio.samples();
    if samples.is_empty() {
        return 0.0;
    }

    // Calculate energy decay curve (simplified)
    let window_size = 1024;
    let hop_size = 512;
    let mut energy_curve = Vec::new();

    for i in (0..samples.len()).step_by(hop_size) {
        let end = (i + window_size).min(samples.len());
        let window_energy: f32 = samples[i..end].iter().map(|&x| x * x).sum();
        energy_curve.push(window_energy / (end - i) as f32);
    }

    // Find RT60 (time for 60dB decay)
    if energy_curve.is_empty() {
        return 0.0;
    }

    let max_energy = energy_curve.iter().fold(0.0f32, |a, &b| a.max(b));
    let target_energy = max_energy * 0.001; // -60dB = 10^(-60/20) ≈ 0.001

    for (i, &energy) in energy_curve.iter().enumerate() {
        if energy <= target_energy {
            return i as f32 * hop_size as f32 / audio.sample_rate() as f32;
        }
    }

    // If no decay found, return length of audio
    samples.len() as f32 / audio.sample_rate() as f32
}
