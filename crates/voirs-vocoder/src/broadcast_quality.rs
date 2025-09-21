//! Professional broadcast quality enhancement for VoiRS
//!
//! This module implements broadcast-standard audio processing and quality enhancement
//! features to meet professional audio production requirements.

use crate::AudioBuffer;
use std::collections::VecDeque;

/// Professional broadcast quality enhancement processor
pub struct BroadcastQualityEnhancer {
    /// Sample rate for processing
    #[allow(dead_code)]
    sample_rate: f32,
    /// Loudness normalizer
    loudness_processor: LoudnessProcessor,
    /// Dynamic range processor (compressor/limiter)
    dynamics_processor: DynamicsProcessor,
    /// Spectral enhancer for clarity
    spectral_enhancer: SpectralEnhancer,
    /// Noise gate for clean audio
    noise_gate: NoiseGate,
    /// De-esser for sibilance control
    de_esser: DeEsser,
    /// Broadcast-standard EQ
    broadcast_eq: BroadcastEqualizer,
}

impl BroadcastQualityEnhancer {
    /// Create new broadcast quality enhancer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            loudness_processor: LoudnessProcessor::new(sample_rate),
            dynamics_processor: DynamicsProcessor::new(sample_rate),
            spectral_enhancer: SpectralEnhancer::new(sample_rate),
            noise_gate: NoiseGate::new(sample_rate),
            de_esser: DeEsser::new(sample_rate),
            broadcast_eq: BroadcastEqualizer::new(sample_rate),
        }
    }

    /// Process audio with broadcast quality enhancement
    pub fn enhance(&mut self, audio: &AudioBuffer) -> Result<AudioBuffer, BroadcastError> {
        let mut enhanced_data = audio.samples().to_vec();

        // Stage 1: Noise gating to remove unwanted noise
        enhanced_data = self.noise_gate.process(&enhanced_data)?;

        // Stage 2: Broadcast EQ for spectral balance
        enhanced_data = self.broadcast_eq.process(&enhanced_data)?;

        // Stage 3: De-essing to control sibilance
        enhanced_data = self.de_esser.process(&enhanced_data)?;

        // Stage 4: Spectral enhancement for clarity
        enhanced_data = self.spectral_enhancer.process(&enhanced_data)?;

        // Stage 5: Dynamic range processing
        enhanced_data = self.dynamics_processor.process(&enhanced_data)?;

        // Stage 6: Loudness normalization to broadcast standards
        enhanced_data = self.loudness_processor.process(&enhanced_data)?;

        Ok(AudioBuffer::new(
            enhanced_data,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Configure for specific broadcast standard
    pub fn configure_for_standard(&mut self, standard: BroadcastStandard) {
        match standard {
            BroadcastStandard::EBU128 => {
                self.loudness_processor.set_target_lufs(-23.0);
                self.loudness_processor.set_max_true_peak(-1.0);
                self.dynamics_processor.set_limiter_threshold(-3.0);
            }
            BroadcastStandard::ATSC => {
                self.loudness_processor.set_target_lufs(-24.0);
                self.loudness_processor.set_max_true_peak(-2.0);
                self.dynamics_processor.set_limiter_threshold(-4.0);
            }
            BroadcastStandard::Radio => {
                self.loudness_processor.set_target_lufs(-16.0);
                self.loudness_processor.set_max_true_peak(-1.0);
                self.dynamics_processor.set_limiter_threshold(-1.0);
                self.dynamics_processor.set_compression_ratio(4.0);
            }
            BroadcastStandard::Podcast => {
                self.loudness_processor.set_target_lufs(-16.0);
                self.loudness_processor.set_max_true_peak(-1.0);
                self.dynamics_processor.set_compression_ratio(3.0);
            }
        }
    }

    /// Get quality metrics for broadcast compliance
    pub fn get_quality_metrics(&self, audio: &AudioBuffer) -> BroadcastQualityMetrics {
        BroadcastQualityMetrics {
            integrated_loudness: self
                .loudness_processor
                .measure_integrated_loudness(audio.samples()),
            loudness_range: self
                .loudness_processor
                .measure_loudness_range(audio.samples()),
            true_peak: self.loudness_processor.measure_true_peak(audio.samples()),
            dynamic_range: self
                .dynamics_processor
                .measure_dynamic_range(audio.samples()),
            spectral_balance: self
                .spectral_enhancer
                .analyze_spectral_balance(audio.samples()),
            noise_floor: self.noise_gate.measure_noise_floor(audio.samples()),
            sibilance_level: self.de_esser.measure_sibilance(audio.samples()),
        }
    }
}

/// Broadcast standards for quality enhancement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastStandard {
    /// EBU R128 standard (European Broadcasting Union)
    EBU128,
    /// ATSC A/85 standard (Advanced Television Systems Committee)
    ATSC,
    /// Radio broadcasting standard
    Radio,
    /// Podcast/streaming standard
    Podcast,
}

/// Loudness processor for broadcast compliance
pub struct LoudnessProcessor {
    sample_rate: f32,
    target_lufs: f32,
    max_true_peak: f32,
    integration_buffer: VecDeque<f32>,
    integration_time: f32, // seconds
}

impl LoudnessProcessor {
    pub fn new(sample_rate: f32) -> Self {
        let integration_time = 0.4; // 400ms integration window
        let buffer_size = (sample_rate * integration_time) as usize;

        Self {
            sample_rate,
            target_lufs: -23.0, // EBU R128 default
            max_true_peak: -1.0,
            integration_buffer: VecDeque::with_capacity(buffer_size),
            integration_time,
        }
    }

    pub fn set_target_lufs(&mut self, lufs: f32) {
        self.target_lufs = lufs;
    }

    pub fn set_max_true_peak(&mut self, peak_db: f32) {
        self.max_true_peak = peak_db;
    }

    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<f32>, BroadcastError> {
        let mut processed = Vec::with_capacity(audio.len());

        for &sample in audio {
            // Update integration buffer
            self.integration_buffer.push_back(sample);
            if self.integration_buffer.len() > (self.sample_rate * self.integration_time) as usize {
                self.integration_buffer.pop_front();
            }

            // Calculate current loudness
            let current_lufs = self.calculate_momentary_loudness();

            // Apply loudness compensation
            let gain_db = self.target_lufs - current_lufs;
            let gain_linear = self.db_to_linear(gain_db.clamp(-12.0, 12.0)); // Limit gain range

            let mut processed_sample = sample * gain_linear;

            // Apply true peak limiting
            let peak_threshold = self.db_to_linear(self.max_true_peak);
            if processed_sample.abs() > peak_threshold {
                processed_sample = processed_sample.signum() * peak_threshold;
            }

            processed.push(processed_sample);
        }

        Ok(processed)
    }

    pub fn measure_integrated_loudness(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return -60.0; // Very quiet
        }

        // Simplified integrated loudness measurement
        let rms = self.calculate_rms(audio);
        self.linear_to_db(rms) - 0.691 // K-weighting approximation
    }

    pub fn measure_loudness_range(&self, audio: &[f32]) -> f32 {
        // Simplified loudness range calculation
        let chunk_size = (self.sample_rate * 0.4) as usize; // 400ms chunks
        let mut chunk_loudnesses = Vec::new();

        for chunk in audio.chunks(chunk_size) {
            if chunk.len() >= chunk_size / 2 {
                // Only process reasonably sized chunks
                let loudness = self.measure_integrated_loudness(chunk);
                chunk_loudnesses.push(loudness);
            }
        }

        if chunk_loudnesses.len() < 2 {
            return 0.0;
        }

        chunk_loudnesses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = chunk_loudnesses.len();
        let p95 = chunk_loudnesses[(len as f32 * 0.95) as usize];
        let p10 = chunk_loudnesses[(len as f32 * 0.10) as usize];

        p95 - p10
    }

    pub fn measure_true_peak(&self, audio: &[f32]) -> f32 {
        // Simplified true peak measurement (should use oversampling)
        let peak = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        self.linear_to_db(peak)
    }

    fn calculate_momentary_loudness(&self) -> f32 {
        if self.integration_buffer.is_empty() {
            return -60.0;
        }

        let rms = self.calculate_rms(&self.integration_buffer.iter().cloned().collect::<Vec<_>>());
        self.linear_to_db(rms) - 0.691 // K-weighting approximation
    }

    fn calculate_rms(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = audio.iter().map(|&x| x * x).sum();
        (sum_squares / audio.len() as f32).sqrt()
    }

    fn db_to_linear(&self, db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    fn linear_to_db(&self, linear: f32) -> f32 {
        if linear <= 0.0 {
            -60.0
        } else {
            20.0 * linear.log10()
        }
    }
}

/// Dynamic range processor (compressor/limiter)
pub struct DynamicsProcessor {
    sample_rate: f32,
    threshold: f32,
    ratio: f32,
    attack: f32,
    release: f32,
    limiter_threshold: f32,
    envelope_follower: f32,
}

impl DynamicsProcessor {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            threshold: -20.0, // dB
            ratio: 3.0,
            attack: 0.005,           // 5ms
            release: 0.1,            // 100ms
            limiter_threshold: -3.0, // dB
            envelope_follower: 0.0,
        }
    }

    pub fn set_compression_ratio(&mut self, ratio: f32) {
        self.ratio = ratio.max(1.0);
    }

    pub fn set_limiter_threshold(&mut self, threshold_db: f32) {
        self.limiter_threshold = threshold_db;
    }

    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<f32>, BroadcastError> {
        let mut processed = Vec::with_capacity(audio.len());

        let attack_coeff = self.calculate_time_constant(self.attack);
        let release_coeff = self.calculate_time_constant(self.release);
        let threshold_linear = self.db_to_linear(self.threshold);
        let limiter_threshold_linear = self.db_to_linear(self.limiter_threshold);

        for &sample in audio {
            let sample_abs = sample.abs();

            // Envelope follower
            let target = sample_abs;
            let coeff = if target > self.envelope_follower {
                attack_coeff
            } else {
                release_coeff
            };
            self.envelope_follower = target * coeff + self.envelope_follower * (1.0 - coeff);

            // Compression
            let mut gain = 1.0;
            if self.envelope_follower > threshold_linear {
                let over_threshold = self.linear_to_db(self.envelope_follower) - self.threshold;
                let compressed_over = over_threshold / self.ratio;
                let target_db = self.threshold + compressed_over;
                let current_db = self.linear_to_db(self.envelope_follower);
                gain = self.db_to_linear(target_db - current_db);
            }

            let mut processed_sample = sample * gain;

            // Limiting
            if processed_sample.abs() > limiter_threshold_linear {
                processed_sample = processed_sample.signum() * limiter_threshold_linear;
            }

            processed.push(processed_sample);
        }

        Ok(processed)
    }

    pub fn measure_dynamic_range(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Calculate RMS over 3-second windows
        let window_size = (self.sample_rate * 3.0) as usize;
        let mut rms_values = Vec::new();

        for chunk in audio.chunks(window_size) {
            if chunk.len() >= window_size / 2 {
                let rms = self.calculate_rms(chunk);
                if rms > 0.0 {
                    rms_values.push(self.linear_to_db(rms));
                }
            }
        }

        if rms_values.len() < 2 {
            return 0.0;
        }

        rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = rms_values.len();
        let p95 = rms_values[(len as f32 * 0.95) as usize];
        let p5 = rms_values[(len as f32 * 0.05) as usize];

        p95 - p5
    }

    fn calculate_time_constant(&self, time_sec: f32) -> f32 {
        1.0 - (-1.0 / (time_sec * self.sample_rate)).exp()
    }

    fn calculate_rms(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = audio.iter().map(|&x| x * x).sum();
        (sum_squares / audio.len() as f32).sqrt()
    }

    fn db_to_linear(&self, db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    fn linear_to_db(&self, linear: f32) -> f32 {
        if linear <= 0.0 {
            -60.0
        } else {
            20.0 * linear.log10()
        }
    }
}

/// Spectral enhancer for broadcast clarity
pub struct SpectralEnhancer {
    #[allow(dead_code)]
    sample_rate: f32,
    presence_boost: f32, // 3-5 kHz boost for speech clarity
    air_band_boost: f32, // 10-15 kHz boost for "air"
}

impl SpectralEnhancer {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            presence_boost: 2.0, // dB
            air_band_boost: 1.5, // dB
        }
    }

    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<f32>, BroadcastError> {
        // Simplified spectral enhancement using a basic filter approach
        // In a real implementation, this would use FFT for more precise frequency control

        let mut enhanced = Vec::with_capacity(audio.len());
        let mut prev_sample = 0.0f32;

        for &sample in audio {
            // High-pass filtering for presence boost (simplified)
            let high_freq = sample - prev_sample * 0.95;
            let presence_gain = 10.0_f32.powf(self.presence_boost / 20.0) - 1.0;
            let presence_enhanced = sample + high_freq * (presence_gain * 0.1);

            // Add gentle high-frequency enhancement using air_band_boost
            let air_gain = 10.0_f32.powf(self.air_band_boost / 20.0) - 1.0;
            let air_enhanced = presence_enhanced + (sample - prev_sample) * (air_gain * 0.05);

            enhanced.push(air_enhanced);
            prev_sample = sample;
        }

        Ok(enhanced)
    }

    pub fn analyze_spectral_balance(&self, audio: &[f32]) -> SpectralBalance {
        // Simplified spectral analysis
        let mut low_energy = 0.0f32;
        let mut mid_energy = 0.0f32;
        let mut high_energy = 0.0f32;

        // Use simple filtering to approximate frequency bands
        for i in 1..audio.len() {
            let sample = audio[i];
            let prev = audio[i - 1];

            low_energy += sample * sample;
            mid_energy += (sample - prev * 0.5).powi(2);
            high_energy += (sample - prev * 0.9).powi(2);
        }

        let total_energy = low_energy + mid_energy + high_energy;

        if total_energy > 0.0 {
            SpectralBalance {
                low_ratio: low_energy / total_energy,
                mid_ratio: mid_energy / total_energy,
                high_ratio: high_energy / total_energy,
                balance_score: self.calculate_balance_score(low_energy, mid_energy, high_energy),
            }
        } else {
            SpectralBalance {
                low_ratio: 0.0,
                mid_ratio: 0.0,
                high_ratio: 0.0,
                balance_score: 0.0,
            }
        }
    }

    fn calculate_balance_score(&self, low: f32, mid: f32, high: f32) -> f32 {
        let total = low + mid + high;
        if total == 0.0 {
            return 0.0;
        }

        // Ideal balance for speech: more mid, moderate low and high
        let low_ratio = low / total;
        let mid_ratio = mid / total;
        let high_ratio = high / total;

        // Score based on how close to ideal balance
        let ideal_low = 0.3;
        let ideal_mid = 0.5;
        let ideal_high = 0.2;

        let deviation = (low_ratio - ideal_low).abs()
            + (mid_ratio - ideal_mid).abs()
            + (high_ratio - ideal_high).abs();

        (1.0 - deviation).max(0.0)
    }
}

/// Noise gate for clean audio
pub struct NoiseGate {
    sample_rate: f32,
    threshold: f32,
    ratio: f32,
    attack: f32,
    release: f32,
    envelope: f32,
}

impl NoiseGate {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            threshold: -50.0, // dB
            ratio: 10.0,
            attack: 0.001, // 1ms
            release: 0.5,  // 500ms
            envelope: 0.0,
        }
    }

    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<f32>, BroadcastError> {
        let mut processed = Vec::with_capacity(audio.len());

        let attack_coeff = 1.0 - (-1.0 / (self.attack * self.sample_rate)).exp();
        let release_coeff = 1.0 - (-1.0 / (self.release * self.sample_rate)).exp();
        let threshold_linear = 10.0_f32.powf(self.threshold / 20.0);

        for &sample in audio {
            let sample_abs = sample.abs();

            // Envelope follower
            let coeff = if sample_abs > self.envelope {
                attack_coeff
            } else {
                release_coeff
            };
            self.envelope = sample_abs * coeff + self.envelope * (1.0 - coeff);

            // Gate calculation
            let gate_gain = if self.envelope < threshold_linear {
                let reduction = (self.envelope / threshold_linear).powf(1.0 / self.ratio - 1.0);
                reduction.min(1.0)
            } else {
                1.0
            };

            processed.push(sample * gate_gain);
        }

        Ok(processed)
    }

    pub fn measure_noise_floor(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return -60.0;
        }

        // Find the quietest 10% of the signal
        let mut rms_values: Vec<f32> = Vec::new();
        let window_size = (self.sample_rate * 0.1) as usize; // 100ms windows

        for chunk in audio.chunks(window_size) {
            if chunk.len() >= window_size / 2 {
                let rms = self.calculate_rms(chunk);
                if rms > 0.0 {
                    rms_values.push(20.0 * rms.log10());
                }
            }
        }

        if rms_values.is_empty() {
            return -60.0;
        }

        rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rms_values[(rms_values.len() as f32 * 0.1) as usize]
    }

    fn calculate_rms(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = audio.iter().map(|&x| x * x).sum();
        (sum_squares / audio.len() as f32).sqrt()
    }
}

/// De-esser for sibilance control
pub struct DeEsser {
    sample_rate: f32,
    threshold: f32,
    frequency: f32, // Center frequency for de-essing
    bandwidth: f32, // Q factor
    reduction: f32, // Maximum reduction in dB
}

impl DeEsser {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            threshold: -20.0,  // dB
            frequency: 6000.0, // Hz - typical sibilance frequency
            bandwidth: 2.0,
            reduction: 6.0, // dB
        }
    }

    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<f32>, BroadcastError> {
        // Simplified de-esser using basic high-frequency detection
        let mut processed = Vec::with_capacity(audio.len());
        let mut prev_sample = 0.0f32;

        for &sample in audio {
            // Detect high-frequency content (simplified sibilance detection)
            // Use frequency parameter to adjust high-frequency detection sensitivity
            let freq_factor = (self.frequency / self.sample_rate).min(0.5);
            let high_freq = sample - prev_sample * (1.0 - freq_factor);
            let sibilance_detector = high_freq.abs() * self.bandwidth;

            // Apply reduction if sibilance is detected above threshold
            let threshold_linear = 10.0_f32.powf(self.threshold / 20.0);
            let reduction_factor = if sibilance_detector > threshold_linear {
                let reduction_linear = 10.0_f32.powf(-self.reduction / 20.0);
                let blend = ((sibilance_detector - threshold_linear) / threshold_linear).min(1.0);
                1.0 - blend * (1.0 - reduction_linear)
            } else {
                1.0
            };

            // Apply frequency-selective reduction
            let de_essed = sample - high_freq * (1.0 - reduction_factor);

            processed.push(de_essed);
            prev_sample = sample;
        }

        Ok(processed)
    }

    pub fn measure_sibilance(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let mut sibilance_energy = 0.0f32;
        let mut total_energy = 0.0f32;

        for i in 1..audio.len() {
            let sample = audio[i];
            let prev = audio[i - 1];

            let high_freq = sample - prev * 0.8;
            sibilance_energy += high_freq * high_freq;
            total_energy += sample * sample;
        }

        if total_energy > 0.0 {
            sibilance_energy / total_energy
        } else {
            0.0
        }
    }
}

/// Broadcast-standard equalizer
pub struct BroadcastEqualizer {
    #[allow(dead_code)]
    sample_rate: f32,
    low_shelf_gain: f32,
    mid_peak_gain: f32,
    high_shelf_gain: f32,
}

impl BroadcastEqualizer {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            low_shelf_gain: 0.0,  // dB at 100Hz
            mid_peak_gain: 1.0,   // dB at 1kHz
            high_shelf_gain: 0.5, // dB at 10kHz
        }
    }

    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<f32>, BroadcastError> {
        // Simplified EQ implementation
        let mut equalized = Vec::with_capacity(audio.len());
        let mut low_history = [0.0f32; 2];
        let mut mid_history = [0.0f32; 2];
        let mut high_history = [0.0f32; 2];

        for &sample in audio {
            // Low shelf (simplified) - use low_shelf_gain
            let low_gain = 10.0_f32.powf(self.low_shelf_gain / 20.0) - 1.0;
            let low_enhanced = sample + low_history[0] * (low_gain * 0.1);
            low_history[1] = low_history[0];
            low_history[0] = sample;

            // Mid peak (simplified) - use mid_peak_gain
            let mid_gain = 10.0_f32.powf(self.mid_peak_gain / 20.0) - 1.0;
            let mid_enhanced = low_enhanced + (sample - mid_history[0] * 0.5) * (mid_gain * 0.1);
            mid_history[1] = mid_history[0];
            mid_history[0] = sample;

            // High shelf (simplified) - use high_shelf_gain
            let high_gain = 10.0_f32.powf(self.high_shelf_gain / 20.0) - 1.0;
            let high_enhanced =
                mid_enhanced + (sample - high_history[0] * 0.9) * (high_gain * 0.05);
            high_history[1] = high_history[0];
            high_history[0] = sample;

            equalized.push(high_enhanced);
        }

        Ok(equalized)
    }
}

/// Quality metrics for broadcast compliance
#[derive(Debug, Clone)]
pub struct BroadcastQualityMetrics {
    pub integrated_loudness: f32, // LUFS
    pub loudness_range: f32,      // LU
    pub true_peak: f32,           // dBTP
    pub dynamic_range: f32,       // dB
    pub spectral_balance: SpectralBalance,
    pub noise_floor: f32,     // dB
    pub sibilance_level: f32, // 0.0-1.0
}

impl BroadcastQualityMetrics {
    /// Check compliance with broadcast standards
    pub fn check_compliance(&self, standard: BroadcastStandard) -> ComplianceReport {
        let mut report = ComplianceReport {
            compliant: true,
            issues: Vec::new(),
            warnings: Vec::new(),
        };

        let (target_lufs, max_true_peak) = match standard {
            BroadcastStandard::EBU128 => (-23.0, -1.0),
            BroadcastStandard::ATSC => (-24.0, -2.0),
            BroadcastStandard::Radio => (-16.0, -1.0),
            BroadcastStandard::Podcast => (-16.0, -1.0),
        };

        // Check loudness compliance
        if (self.integrated_loudness - target_lufs).abs() > 2.0 {
            report.compliant = false;
            report.issues.push(format!(
                "Integrated loudness {:.1} LUFS is outside tolerance of target {:.1} LUFS",
                self.integrated_loudness, target_lufs
            ));
        } else if (self.integrated_loudness - target_lufs).abs() > 1.0 {
            report.warnings.push(format!(
                "Integrated loudness {:.1} LUFS is close to tolerance limit",
                self.integrated_loudness
            ));
        }

        // Check true peak compliance
        if self.true_peak > max_true_peak {
            report.compliant = false;
            report.issues.push(format!(
                "True peak {:.1} dBTP exceeds limit of {:.1} dBTP",
                self.true_peak, max_true_peak
            ));
        }

        // Check dynamic range
        if self.dynamic_range < 5.0 {
            report.warnings.push(format!(
                "Low dynamic range {:.1} dB may indicate over-compression",
                self.dynamic_range
            ));
        }

        // Check noise floor
        if self.noise_floor > -50.0 {
            report.warnings.push(format!(
                "High noise floor {:.1} dB may affect broadcast quality",
                self.noise_floor
            ));
        }

        // Check spectral balance
        if self.spectral_balance.balance_score < 0.7 {
            report
                .warnings
                .push("Poor spectral balance detected".to_string());
        }

        report
    }
}

/// Spectral balance analysis
#[derive(Debug, Clone)]
pub struct SpectralBalance {
    pub low_ratio: f32,     // 0.0-1.0
    pub mid_ratio: f32,     // 0.0-1.0
    pub high_ratio: f32,    // 0.0-1.0
    pub balance_score: f32, // 0.0-1.0 (1.0 = perfect balance)
}

/// Compliance report for broadcast standards
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    pub compliant: bool,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
}

/// Errors that can occur during broadcast processing
#[derive(Debug, thiserror::Error)]
pub enum BroadcastError {
    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_enhancer_creation() {
        let enhancer = BroadcastQualityEnhancer::new(44100.0);
        assert_eq!(enhancer.sample_rate, 44100.0);
    }

    #[test]
    fn test_standard_configuration() {
        let mut enhancer = BroadcastQualityEnhancer::new(44100.0);
        enhancer.configure_for_standard(BroadcastStandard::EBU128);
        // Configuration should succeed without panic
    }

    #[test]
    fn test_audio_enhancement() {
        let mut enhancer = BroadcastQualityEnhancer::new(44100.0);
        let test_audio = AudioBuffer::new(vec![0.1, 0.2, -0.1, -0.3, 0.4, -0.2], 44100, 1);

        let result = enhancer.enhance(&test_audio);
        assert!(result.is_ok());

        let enhanced = result.unwrap();
        assert_eq!(enhanced.samples().len(), test_audio.samples().len());
        assert_eq!(enhanced.sample_rate(), test_audio.sample_rate());
    }

    #[test]
    fn test_quality_metrics() {
        let enhancer = BroadcastQualityEnhancer::new(44100.0);
        let test_audio = AudioBuffer::new(
            vec![0.1; 44100], // 1 second of constant signal
            44100,
            1,
        );

        let metrics = enhancer.get_quality_metrics(&test_audio);
        assert!(metrics.integrated_loudness < 0.0); // Should be negative dB
        assert!(metrics.true_peak <= 0.0); // Should not exceed 0 dBFS
        assert!(metrics.spectral_balance.balance_score >= 0.0);
        assert!(metrics.spectral_balance.balance_score <= 1.0);
    }

    #[test]
    fn test_compliance_check() {
        let metrics = BroadcastQualityMetrics {
            integrated_loudness: -23.0,
            loudness_range: 5.0,
            true_peak: -1.5,
            dynamic_range: 15.0,
            spectral_balance: SpectralBalance {
                low_ratio: 0.3,
                mid_ratio: 0.5,
                high_ratio: 0.2,
                balance_score: 0.8,
            },
            noise_floor: -55.0,
            sibilance_level: 0.1,
        };

        let report = metrics.check_compliance(BroadcastStandard::EBU128);
        assert!(report.compliant);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn test_loudness_processor() {
        let mut processor = LoudnessProcessor::new(44100.0);
        let test_audio = vec![0.1, 0.2, -0.1, -0.3, 0.4, -0.2];

        let result = processor.process(&test_audio);
        assert!(result.is_ok());

        let processed = result.unwrap();
        assert_eq!(processed.len(), test_audio.len());
    }
}
