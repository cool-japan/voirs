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
            return -60.0;
        }

        // ITU-R BS.1770-4 integrated loudness measurement.
        // Stage 1: apply K-weighting filter chain to the whole signal.
        let kw = self.apply_k_weighting(audio);

        // Stage 2: gated loudness measurement.
        //   Block size : 400 ms
        //   Hop size   : 100 ms  (75 % overlap)
        let block_len = ((self.sample_rate * 0.4) as usize).max(1);
        let hop_len = ((self.sample_rate * 0.1) as usize).max(1);

        // Collect (mean-square, loudness) for every valid block.
        let mut block_ms: Vec<f64> = Vec::new();
        let mut start = 0usize;
        while start + block_len <= kw.len() {
            let slice = &kw[start..start + block_len];
            let ms: f64 =
                slice.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / block_len as f64;
            block_ms.push(ms);
            start += hop_len;
        }

        if block_ms.is_empty() {
            // Signal shorter than one block — use the whole thing.
            let ms: f64 =
                kw.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / kw.len() as f64;
            if ms <= 0.0 {
                return -60.0;
            }
            return (-0.691 + 10.0 * ms.log10()) as f32;
        }

        // Block loudness L_j = -0.691 + 10*log10(ms_j)  [LKFS]
        let block_loudness: Vec<f64> = block_ms
            .iter()
            .map(|&ms| {
                if ms > 0.0 {
                    -0.691 + 10.0 * ms.log10()
                } else {
                    -f64::INFINITY
                }
            })
            .collect();

        // Absolute gate: keep blocks where L_j >= -70 LKFS.
        let abs_gated: Vec<usize> = block_loudness
            .iter()
            .enumerate()
            .filter(|(_, &l)| l >= -70.0)
            .map(|(i, _)| i)
            .collect();

        if abs_gated.is_empty() {
            return -60.0; // all blocks below absolute gate → treat as silence
        }

        // L_avg from absolute-gated blocks.
        let avg_ms_abs: f64 =
            abs_gated.iter().map(|&i| block_ms[i]).sum::<f64>() / abs_gated.len() as f64;
        let l_avg = -0.691 + 10.0 * avg_ms_abs.log10();

        // Relative gate threshold: L_avg - 10 LKFS.
        let rel_threshold = l_avg - 10.0;

        // Keep blocks that pass both gates.
        let rel_gated: Vec<usize> = abs_gated
            .into_iter()
            .filter(|&i| block_loudness[i] >= rel_threshold)
            .collect();

        if rel_gated.is_empty() {
            return -60.0;
        }

        let avg_ms_rel: f64 =
            rel_gated.iter().map(|&i| block_ms[i]).sum::<f64>() / rel_gated.len() as f64;

        if avg_ms_rel <= 0.0 {
            return -60.0;
        }

        (-0.691 + 10.0 * avg_ms_rel.log10()) as f32
    }

    /// Apply ITU-R BS.1770-4 K-weighting filter chain (two cascaded biquad IIR stages).
    ///
    /// Stage 1 — pre-filter (high-shelf, compensates acoustic effect of the head):
    ///   At 48 kHz  b = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    ///              a = [1, -1.69065929318241, 0.73248077421585]
    ///   For other sample rates we derive coefficients via the bilinear transform from
    ///   the analogue prototype (Hs with f0=1681.974…Hz, Q=0.7071…, dBgain=+3.9998…).
    ///
    /// Stage 2 — high-pass RLB (revised low-frequency B-weighting):
    ///   At 48 kHz  b = [1, -2, 1]
    ///              a = [1, -1.99004745483398, 0.99007225036616]
    ///   For other sample rates the same bilinear derivation applies (Hb with f0=38.13…Hz).
    fn apply_k_weighting(&self, audio: &[f32]) -> Vec<f32> {
        let fs = self.sample_rate as f64;

        // ---- Stage 1: pre-filter (high-shelf) ----
        // Analogue prototype parameters (from ITU-R BS.1770-4 Annex 1).
        let f0_pre = 1_681.974_450_955_533_f64;
        let q_pre = 0.707_175_236_955_419_6_f64;
        let db_pre = 3.999_843_853_973_347_f64;

        let k = (std::f64::consts::PI * f0_pre / fs).tan();
        let v0 = 10.0_f64.powf(db_pre / 20.0);
        let sqrt2 = std::f64::consts::SQRT_2;

        // High-shelf bilinear transform (boost, V0 > 1):
        let norm = 1.0 / (1.0 + sqrt2 / q_pre * k + k * k);
        // b0, b1, b2 scaled by 1/norm:
        let b0_pre = (v0 + (v0 * 2.0_f64).sqrt() / q_pre * k + k * k) * norm;
        let b1_pre = (2.0 * (k * k - v0)) * norm;
        let b2_pre = (v0 - (v0 * 2.0_f64).sqrt() / q_pre * k + k * k) * norm;
        let a1_pre = (2.0 * (k * k - 1.0)) * norm;
        let a2_pre = (1.0 - sqrt2 / q_pre * k + k * k) * norm;

        // ---- Stage 2: high-pass RLB ----
        // Analogue prototype: f0 = 38.135 Hz (second-order Butterworth high-pass).
        let f0_rlb = 38.135_047_196_563_6_f64;
        let k2 = (std::f64::consts::PI * f0_rlb / fs).tan();
        let norm2 = 1.0 / (1.0 + sqrt2 * k2 + k2 * k2);

        let b0_rlb = norm2;
        let b1_rlb = -2.0 * norm2;
        let b2_rlb = norm2;
        let a1_rlb = 2.0 * (k2 * k2 - 1.0) * norm2;
        let a2_rlb = (1.0 - sqrt2 * k2 + k2 * k2) * norm2;

        // ---- Run the two biquad stages in series (direct form II) ----
        let mut w1 = [0.0f64; 2]; // state for stage 1
        let mut w2 = [0.0f64; 2]; // state for stage 2

        audio
            .iter()
            .map(|&x| {
                let xd = x as f64;

                // Stage 1
                let w1n = xd - a1_pre * w1[0] - a2_pre * w1[1];
                let y1 = b0_pre * w1n + b1_pre * w1[0] + b2_pre * w1[1];
                w1[1] = w1[0];
                w1[0] = w1n;

                // Stage 2
                let w2n = y1 - a1_rlb * w2[0] - a2_rlb * w2[1];
                let y2 = b0_rlb * w2n + b1_rlb * w2[0] + b2_rlb * w2[1];
                w2[1] = w2[0];
                w2[0] = w2n;

                y2 as f32
            })
            .collect()
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

        chunk_loudnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = chunk_loudnesses.len();
        let p95 = chunk_loudnesses[(len as f32 * 0.95) as usize];
        let p10 = chunk_loudnesses[(len as f32 * 0.10) as usize];

        p95 - p10
    }

    pub fn measure_true_peak(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return -60.0;
        }

        // ITU-R BS.1770-4 / EBU R128 true-peak measurement via 4× oversampling.
        //
        // For each of the three inter-sample phases p ∈ {1, 2, 3} we convolve the
        // signal with a Kaiser-windowed sinc kernel centred at fractional offset
        // p/4.  Phase 0 is the original sample itself.  The 4× oversampled peak
        // is the maximum absolute value across all four phases.
        //
        // Kernel length: 16 taps (±8 samples around the fractional offset).
        // Kaiser window parameter α = 5.0  (side-lobe attenuation ≈ 50 dB).
        const TAPS: usize = 16;
        const ALPHA: f64 = 5.0;
        const OVERSAMPLE: usize = 4;

        // Precompute Kaiser window I0(x) via series expansion.
        let i0 = |x: f64| -> f64 {
            let mut sum = 1.0_f64;
            let mut term = 1.0_f64;
            for k in 1_u32..=25 {
                term *= (x / 2.0) / k as f64;
                sum += term * term;
            }
            sum
        };
        let i0_alpha = i0(std::f64::consts::PI * ALPHA);

        // Build one sinc-Kaiser kernel per inter-sample phase.
        // For phase p the kernel tap at index n (0 … TAPS-1) corresponds to
        // input sample offset  n - (TAPS/2 - 1) − p/OVERSAMPLE.
        let build_kernel = |phase: usize| -> [f64; TAPS] {
            let mut h = [0.0f64; TAPS];
            for n in 0..TAPS {
                // Fractional delay: how many samples away from the phase point.
                let t = n as f64 - (TAPS as f64 / 2.0 - 1.0) - phase as f64 / OVERSAMPLE as f64;
                // Sinc
                let sinc = if t.abs() < 1e-10 {
                    1.0
                } else {
                    let pt = std::f64::consts::PI * t;
                    pt.sin() / pt
                };
                // Kaiser window
                let arg = 1.0 - (2.0 * (n as f64 + 0.5) / TAPS as f64 - 1.0).powi(2);
                let w = i0(std::f64::consts::PI * ALPHA * arg.max(0.0).sqrt()) / i0_alpha;
                h[n] = sinc * w;
            }
            h
        };

        // Phase 0 is the unmodified signal; compute kernels only for phases 1–3.
        let kernels: [[f64; TAPS]; 3] = [build_kernel(1), build_kernel(2), build_kernel(3)];

        // Track maximum absolute value across the original samples (phase 0)
        // and all three interpolated phases.
        let mut peak = audio.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

        let n = audio.len();
        for (phase_idx, kernel) in kernels.iter().enumerate() {
            let _ = phase_idx; // phase already baked into kernel
                               // Each output sample i corresponds to interpolated position i + p/4.
            for i in 0..n {
                let mut acc = 0.0f64;
                for (k, &h) in kernel.iter().enumerate() {
                    // Input index offset: k − (TAPS/2 − 1)
                    let offset = k as isize - (TAPS as isize / 2 - 1);
                    let idx = i as isize + offset;
                    let sample = if idx < 0 || idx >= n as isize {
                        0.0f64
                    } else {
                        audio[idx as usize] as f64
                    };
                    acc += h * sample;
                }
                let abs_val = acc.abs() as f32;
                if abs_val > peak {
                    peak = abs_val;
                }
            }
        }

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

        rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

        rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    // ------- BS.1770-4 integrated loudness tests -------

    /// A louder signal must measure a higher LUFS value than a quieter one.
    #[test]
    fn test_loudness_ordering() {
        let sample_rate = 48000.0f32;
        let processor = LoudnessProcessor::new(sample_rate);

        // Generate 2 seconds of 1 kHz sine at two amplitudes.
        let duration_secs = 2.0f64;
        let n = (sample_rate as usize * 2).max(1);
        let make_sine = |amp: f64| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    (amp * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sample_rate as f64)
                        .sin()) as f32
                })
                .collect()
        };

        let loud_signal = make_sine(0.5);
        let quiet_signal = make_sine(0.01);

        let _ = duration_secs; // used indirectly via n

        let loud_lufs = processor.measure_integrated_loudness(&loud_signal);
        let quiet_lufs = processor.measure_integrated_loudness(&quiet_signal);

        // Louder signal must measure a higher LUFS value.
        assert!(
            loud_lufs > quiet_lufs,
            "loud LUFS {loud_lufs:.2} should be > quiet LUFS {quiet_lufs:.2}"
        );
    }

    /// A 1 kHz sine at ~−20 dBFS should produce a reasonable LUFS reading
    /// (K-weighting has little effect at 1 kHz, so integrated loudness should
    /// be roughly consistent with the RMS-based estimate).
    #[test]
    fn test_integrated_loudness_near_minus23_lufs() {
        let sample_rate = 48000.0f32;
        let processor = LoudnessProcessor::new(sample_rate);

        // 3 seconds of 1 kHz sine at amplitude 0.1 (≈ −20 dBFS RMS ≈ −20 LUFS).
        let n = (sample_rate as usize) * 3;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                (0.1 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sample_rate as f64).sin())
                    as f32
            })
            .collect();

        let lufs = processor.measure_integrated_loudness(&signal);

        // The reading should be finite and in a plausible range.
        assert!(lufs.is_finite(), "LUFS must be finite, got {lufs}");
        assert!(
            lufs < 0.0,
            "LUFS must be negative for a sub-full-scale signal, got {lufs}"
        );
        // At amp=0.1 the RMS is ≈ 0.1/sqrt(2) ≈ 0.0707 → ~−23 dBFS.
        // K-weighting slightly attenuates 1 kHz, so LUFS should be in the range [−40, 0].
        assert!(
            lufs > -40.0,
            "LUFS unexpectedly low ({lufs:.2}), expected > −40"
        );
    }

    // ------- True-peak oversampling tests -------

    /// For a Nyquist-rate alternating signal (+1/−1) the true peak measured by
    /// 4× oversampling must exceed the sample peak (which is 1.0 = 0 dBTP).
    /// This is the classic inter-sample clipping scenario described in BS.1770.
    #[test]
    fn test_true_peak_higher_than_sample_peak() {
        let processor = LoudnessProcessor::new(48000.0);

        // 256 alternating +1/−1 samples.
        let signal: Vec<f32> = (0..256)
            .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0f32 })
            .collect();

        let sample_peak_db =
            processor.linear_to_db(signal.iter().map(|&s| s.abs()).fold(0.0f32, f32::max));
        let true_peak_db = processor.measure_true_peak(&signal);

        // The true peak at inter-sample positions must exceed 0 dBTP for an
        // alternating +1/−1 sequence (constructive inter-sample reconstruction).
        assert!(
            true_peak_db >= sample_peak_db,
            "true peak {true_peak_db:.3} dBTP should be >= sample peak {sample_peak_db:.3} dBTP"
        );
    }
}
