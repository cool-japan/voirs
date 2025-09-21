//! FFI utility functions and audio processing utilities.

use crate::{VoirsAudioBuffer, VoirsErrorCode};
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_float, c_uint},
    ptr,
};

/// Convert Rust string to owned C string
pub fn string_to_owned_c_str(s: &str) -> Result<CString, std::ffi::NulError> {
    CString::new(s)
}

/// Convert C string to Rust string slice (unsafe)
pub unsafe fn c_str_to_str<'a>(c_str: *const c_char) -> Result<&'a str, std::str::Utf8Error> {
    if c_str.is_null() {
        return Ok("");
    }
    CStr::from_ptr(c_str).to_str()
}

/// Create a null-terminated string array for C (optimized)
pub fn create_string_array(strings: &[String]) -> (*mut *mut c_char, usize) {
    let len = strings.len();
    // Pre-allocate with exact capacity + 1 for null terminator
    let mut ptrs = Vec::with_capacity(len + 1);

    // Convert strings directly without intermediate collection
    for s in strings {
        let ptr = match CString::new(s.as_str()) {
            Ok(cs) => cs.into_raw(),
            Err(_) => ptr::null_mut(),
        };
        ptrs.push(ptr);
    }

    ptrs.push(ptr::null_mut()); // Null terminator

    let array = ptrs.into_boxed_slice();
    let ptr = Box::into_raw(array) as *mut *mut c_char;

    (ptr, len)
}

/// Free a string array created by create_string_array
pub unsafe fn free_string_array(array: *mut *mut c_char, len: usize) {
    if array.is_null() {
        return;
    }

    for i in 0..len {
        let str_ptr = *array.add(i);
        if !str_ptr.is_null() {
            let _ = CString::from_raw(str_ptr);
        }
    }

    let _ = Box::from_raw(std::slice::from_raw_parts_mut(array, len + 1));
}

/// Audio processing utilities module
pub mod audio {
    use super::*;

    /// Audio analysis structure
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct VoirsAudioAnalysis {
        pub rms_level: c_float,          // RMS level (0.0 to 1.0)
        pub peak_level: c_float,         // Peak level (0.0 to 1.0)
        pub zero_crossing_rate: c_float, // Zero crossing rate
        pub spectral_centroid: c_float,  // Spectral centroid in Hz
        pub silence_ratio: c_float,      // Ratio of silence (0.0 to 1.0)
        pub dynamic_range: c_float,      // Dynamic range in dB
    }

    impl Default for VoirsAudioAnalysis {
        fn default() -> Self {
            Self {
                rms_level: 0.0,
                peak_level: 0.0,
                zero_crossing_rate: 0.0,
                spectral_centroid: 0.0,
                silence_ratio: 0.0,
                dynamic_range: 0.0,
            }
        }
    }

    /// Calculate RMS (Root Mean Square) level of audio (SIMD-optimized)
    pub fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // SIMD-optimized RMS calculation for large buffers
        if samples.len() >= 16 {
            calculate_rms_simd(samples)
        } else {
            // Fallback for small buffers
            let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
            (sum_squares / samples.len() as f32).sqrt()
        }
    }

    /// SIMD-optimized RMS calculation
    #[inline]
    fn calculate_rms_simd(samples: &[f32]) -> f32 {
        let mut sum_squares = 0.0f32;
        let chunks = samples.chunks_exact(4);
        let remainder = chunks.remainder();

        // Process 4 samples at a time for better vectorization
        for chunk in chunks {
            let sq0 = chunk[0] * chunk[0];
            let sq1 = chunk[1] * chunk[1];
            let sq2 = chunk[2] * chunk[2];
            let sq3 = chunk[3] * chunk[3];
            sum_squares += sq0 + sq1 + sq2 + sq3;
        }

        // Handle remaining samples
        for &sample in remainder {
            sum_squares += sample * sample;
        }

        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Calculate peak level of audio (SIMD-optimized)
    pub fn calculate_peak(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // SIMD-optimized peak detection for large buffers
        if samples.len() >= 16 {
            calculate_peak_simd(samples)
        } else {
            // Fallback for small buffers
            samples.iter().map(|&x| x.abs()).fold(0.0, f32::max)
        }
    }

    /// SIMD-optimized peak detection
    #[inline]
    fn calculate_peak_simd(samples: &[f32]) -> f32 {
        let mut max_peak = 0.0f32;
        let chunks = samples.chunks_exact(4);
        let remainder = chunks.remainder();

        // Process 4 samples at a time for better vectorization
        for chunk in chunks {
            let abs0 = chunk[0].abs();
            let abs1 = chunk[1].abs();
            let abs2 = chunk[2].abs();
            let abs3 = chunk[3].abs();
            max_peak = max_peak.max(abs0).max(abs1).max(abs2).max(abs3);
        }

        // Handle remaining samples
        for &sample in remainder {
            max_peak = max_peak.max(sample.abs());
        }

        max_peak
    }

    /// Calculate zero crossing rate
    pub fn calculate_zero_crossing_rate(samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (samples.len() - 1) as f32
    }

    /// Detect silence periods in audio
    pub fn detect_silence(samples: &[f32], threshold: f32) -> f32 {
        if samples.is_empty() {
            return 1.0;
        }

        let silent_samples = samples.iter().filter(|&&x| x.abs() < threshold).count();
        silent_samples as f32 / samples.len() as f32
    }

    /// Apply fade-in effect to audio buffer
    pub fn apply_fade_in(samples: &mut [f32], fade_samples: usize) {
        let fade_length = fade_samples.min(samples.len());
        for (i, sample) in samples.iter_mut().take(fade_length).enumerate() {
            let fade_factor = i as f32 / fade_length as f32;
            *sample *= fade_factor;
        }
    }

    /// Apply fade-out effect to audio buffer
    pub fn apply_fade_out(samples: &mut [f32], fade_samples: usize) {
        let fade_length = fade_samples.min(samples.len());
        let start_pos = samples.len().saturating_sub(fade_length);

        for (i, sample) in samples[start_pos..].iter_mut().enumerate() {
            let fade_factor = 1.0 - (i as f32 / fade_length as f32);
            *sample *= fade_factor;
        }
    }

    /// Normalize audio to specified peak level
    pub fn normalize_audio(samples: &mut [f32], target_peak: f32) {
        let current_peak = calculate_peak(samples);
        if current_peak > 0.0 && current_peak != target_peak {
            let scale_factor = target_peak / current_peak;
            for sample in samples {
                *sample *= scale_factor;
            }
        }
    }

    /// Apply simple high-pass filter (removes DC bias)
    pub fn apply_dc_filter(samples: &mut [f32], alpha: f32) {
        if samples.is_empty() {
            return;
        }

        let mut y_prev = 0.0;
        let mut x_prev = samples[0];

        for sample in samples.iter_mut() {
            let x_curr = *sample;
            let y_curr = alpha * (y_prev + x_curr - x_prev);
            *sample = y_curr;

            y_prev = y_curr;
            x_prev = x_curr;
        }
    }

    /// Apply simple low-pass filter for smoothing
    pub fn apply_low_pass_filter(samples: &mut [f32], alpha: f32) {
        if samples.is_empty() {
            return;
        }

        let mut y_prev = samples[0];

        for sample in samples.iter_mut() {
            let y_curr = alpha * *sample + (1.0 - alpha) * y_prev;
            *sample = y_curr;
            y_prev = y_curr;
        }
    }

    /// Calculate spectral envelope using simple DFT approximation (optimized)
    pub fn calculate_spectral_envelope(samples: &[f32], bins: usize) -> Vec<f32> {
        if samples.is_empty() || bins == 0 {
            return vec![0.0; bins];
        }

        // Pre-allocate with exact capacity and initialize with zeros
        let mut envelope = Vec::with_capacity(bins);
        envelope.resize(bins, 0.0);

        let samples_per_bin = samples.len() / bins;

        if samples_per_bin == 0 {
            return envelope;
        }

        // Optimize by avoiding repeated bounds checks and memory allocations
        let mut bin_idx = 0;
        while bin_idx < bins {
            let start_idx = bin_idx * samples_per_bin;
            let end_idx = ((bin_idx + 1) * samples_per_bin).min(samples.len());

            if start_idx < end_idx {
                // Direct slice access without additional bounds checks
                let bin_samples = unsafe { samples.get_unchecked(start_idx..end_idx) };
                envelope[bin_idx] = calculate_rms(bin_samples);
            }
            bin_idx += 1;
        }

        envelope
    }

    /// Optimized buffer pool for frequent audio buffer allocations
    pub struct AudioBufferPool {
        pools: std::collections::HashMap<usize, Vec<Vec<f32>>>,
        max_pool_size: usize,
    }

    impl AudioBufferPool {
        pub fn new(max_pool_size: usize) -> Self {
            Self {
                pools: std::collections::HashMap::new(),
                max_pool_size,
            }
        }

        pub fn get_buffer(&mut self, size: usize) -> Vec<f32> {
            if let Some(pool) = self.pools.get_mut(&size) {
                if let Some(mut buffer) = pool.pop() {
                    buffer.clear();
                    buffer.resize(size, 0.0);
                    return buffer;
                }
            }

            // Create new buffer if pool is empty
            vec![0.0; size]
        }

        pub fn return_buffer(&mut self, buffer: Vec<f32>) {
            let size = buffer.capacity();
            let pool = self.pools.entry(size).or_insert_with(Vec::new);

            if pool.len() < self.max_pool_size {
                pool.push(buffer);
            }
            // Drop buffer if pool is full
        }
    }

    /// Apply soft limiting to prevent harsh clipping
    pub fn apply_soft_limiter(samples: &mut [f32], threshold: f32) {
        for sample in samples.iter_mut() {
            if sample.abs() > threshold {
                let sign = sample.signum();
                let excess = sample.abs() - threshold;
                // Soft saturation curve: tanh for smooth limiting
                let limited = threshold + excess.tanh() * (1.0 - threshold);
                *sample = sign * limited;
            }
        }
    }

    /// Calculate harmonic-to-noise ratio (simplified version)
    pub fn calculate_hnr(samples: &[f32], sample_rate: u32) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Simple autocorrelation-based pitch detection
        let max_lag = (sample_rate / 80) as usize; // ~80 Hz minimum
        let min_lag = (sample_rate / 400) as usize; // ~400 Hz maximum

        if samples.len() < max_lag || max_lag <= min_lag {
            return 0.0;
        }

        let mut max_correlation = 0.0;
        let mut _best_lag = min_lag;

        for lag in min_lag..max_lag.min(samples.len()) {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in 0..(samples.len() - lag) {
                correlation += samples[i] * samples[i + lag];
                norm1 += samples[i] * samples[i];
                norm2 += samples[i + lag] * samples[i + lag];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                let normalized_correlation = correlation / (norm1 * norm2).sqrt();
                if normalized_correlation > max_correlation {
                    max_correlation = normalized_correlation;
                    _best_lag = lag;
                }
            }
        }

        // Convert correlation to approximate HNR in dB
        if max_correlation > 0.01 {
            20.0 * (max_correlation / (1.0 - max_correlation)).log10()
        } else {
            -20.0 // Very noisy signal
        }
    }

    /// Advanced audio enhancement combining multiple techniques
    pub fn enhance_audio_quality(samples: &mut [f32], sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        // 1. Remove DC bias
        apply_dc_filter(samples, 0.995);

        // 2. Apply gentle low-pass filtering to reduce high-frequency noise
        apply_low_pass_filter(samples, 0.95);

        // 3. Normalize to prevent clipping
        normalize_audio(samples, 0.95);

        // 4. Apply soft limiting for safety
        apply_soft_limiter(samples, 0.98);
    }

    /// High-performance audio enhancement using single-pass optimization
    pub fn enhance_audio_quality_optimized(samples: &mut [f32], _sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        // Calculate DC offset and peak in first pass
        let mut dc_sum = 0.0f32;
        let mut peak = 0.0f32;
        for &sample in samples.iter() {
            dc_sum += sample;
            peak = peak.max(sample.abs());
        }

        let dc_offset = dc_sum / samples.len() as f32;
        let normalization_factor = if peak > 0.0 { 0.95 / peak } else { 1.0 };

        // Apply DC removal, normalization, and soft limiting in single pass
        let dc_filter_coeff = 0.995;
        let mut dc_filtered_prev = 0.0;
        let limiter_threshold = 0.98;

        for sample in samples.iter_mut() {
            // DC removal with high-pass filter
            let dc_removed = *sample - dc_offset;
            let dc_filtered = dc_removed - dc_filter_coeff * dc_filtered_prev;
            dc_filtered_prev = dc_removed;

            // Normalize and apply soft limiting
            let normalized = dc_filtered * normalization_factor;
            *sample = if normalized.abs() > limiter_threshold {
                normalized.signum() * limiter_threshold
            } else {
                normalized
            };
        }
    }

    /// Calculate spectral rolloff frequency (frequency below which 85% of energy is concentrated)
    pub fn calculate_spectral_rolloff(samples: &[f32], sample_rate: u32) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Simple FFT-like analysis using autocorrelation
        let window_size = 512.min(samples.len());
        let mut energy_distribution = vec![0.0; window_size / 2];

        for i in 0..window_size / 2 {
            let _freq = (i as f32 * sample_rate as f32) / window_size as f32;
            let mut energy = 0.0;

            // Calculate energy at this frequency using windowed autocorrelation
            for j in 0..(window_size - i) {
                if j + i < samples.len() {
                    energy += samples[j] * samples[j + i];
                }
            }

            energy_distribution[i] = energy.abs();
        }

        // Find 85% rolloff point
        let total_energy: f32 = energy_distribution.iter().sum();
        let mut cumulative_energy = 0.0;
        let threshold = total_energy * 0.85;

        for (i, &energy) in energy_distribution.iter().enumerate() {
            cumulative_energy += energy;
            if cumulative_energy >= threshold {
                return (i as f32 * sample_rate as f32) / window_size as f32;
            }
        }

        // Default to half the Nyquist frequency
        sample_rate as f32 / 4.0
    }

    /// Calculate spectral flux (measure of how quickly the spectrum changes)
    pub fn calculate_spectral_flux(samples: &[f32], sample_rate: u32) -> f32 {
        if samples.len() < 1024 {
            return 0.0;
        }

        let frame_size = 512;
        let hop_size = 256;
        let mut flux_values = Vec::new();

        for i in (0..samples.len().saturating_sub(frame_size)).step_by(hop_size) {
            let frame = &samples[i..i + frame_size];
            let next_frame = if i + frame_size + hop_size < samples.len() {
                &samples[i + hop_size..i + frame_size + hop_size]
            } else {
                continue;
            };

            // Calculate spectral difference between frames
            let mut flux = 0.0;
            for (j, (&current, &next)) in frame.iter().zip(next_frame.iter()).enumerate() {
                let diff = next.abs() - current.abs();
                if diff > 0.0 {
                    flux += diff;
                }
            }

            flux_values.push(flux / frame_size as f32);
        }

        // Return average flux
        if flux_values.is_empty() {
            0.0
        } else {
            flux_values.iter().sum::<f32>() / flux_values.len() as f32
        }
    }

    /// Calculate audio brightness (ratio of high-frequency to low-frequency energy)
    pub fn calculate_brightness(samples: &[f32], sample_rate: u32) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let cutoff_freq = 1000.0; // 1kHz cutoff
        let window_size = 512.min(samples.len());
        let mut low_energy = 0.0;
        let mut high_energy = 0.0;

        for i in 0..window_size / 2 {
            let freq = (i as f32 * sample_rate as f32) / window_size as f32;
            let mut energy = 0.0;

            // Calculate energy at this frequency
            for j in 0..(window_size - i) {
                if j + i < samples.len() {
                    energy += samples[j] * samples[j + i];
                }
            }

            energy = energy.abs();

            if freq < cutoff_freq {
                low_energy += energy;
            } else {
                high_energy += energy;
            }
        }

        if low_energy > 0.0 {
            high_energy / low_energy
        } else {
            0.0
        }
    }

    /// Apply dynamic range compression to audio samples
    pub fn apply_dynamic_compression(
        samples: &mut [f32],
        threshold: f32,
        ratio: f32,
        attack: f32,
        release: f32,
    ) {
        if samples.is_empty() || ratio <= 1.0 {
            return;
        }

        let mut envelope = 0.0;
        let attack_coeff = (-1.0 / attack).exp();
        let release_coeff = (-1.0 / release).exp();

        for sample in samples.iter_mut() {
            let input_level = sample.abs();

            // Update envelope
            if input_level > envelope {
                envelope = input_level + (envelope - input_level) * attack_coeff;
            } else {
                envelope = input_level + (envelope - input_level) * release_coeff;
            }

            // Apply compression if above threshold
            if envelope > threshold {
                let excess = envelope - threshold;
                let compressed_excess = excess / ratio;
                let gain = (threshold + compressed_excess) / envelope;
                *sample *= gain;
            }
        }
    }

    /// Apply multiband EQ with simple 3-band filter
    pub fn apply_multiband_eq(samples: &mut [f32], low_gain: f32, mid_gain: f32, high_gain: f32) {
        if samples.is_empty() {
            return;
        }

        // Simple 3-band EQ using cascaded filters
        let mut low_state = 0.0;
        let mut high_state = 0.0;

        for sample in samples.iter_mut() {
            let input = *sample;

            // Low-pass filter (approximate 300Hz cutoff)
            let low_coeff = 0.1;
            low_state = low_state * (1.0 - low_coeff) + input * low_coeff;
            let low_band = low_state * low_gain;

            // High-pass filter (approximate 3kHz cutoff)
            let high_coeff = 0.9;
            high_state = high_state * (1.0 - high_coeff) + input * high_coeff;
            let high_band = (input - high_state) * high_gain;

            // Mid band (what's left)
            let mid_band = (input - low_state - (input - high_state)) * mid_gain;

            *sample = low_band + mid_band + high_band;
        }
    }

    /// Calculate comprehensive audio analysis
    pub fn analyze_audio(samples: &[f32], sample_rate: u32) -> VoirsAudioAnalysis {
        let mut analysis = VoirsAudioAnalysis::default();

        if samples.is_empty() {
            return analysis;
        }

        analysis.rms_level = calculate_rms(samples);
        analysis.peak_level = calculate_peak(samples);
        analysis.zero_crossing_rate = calculate_zero_crossing_rate(samples);
        analysis.silence_ratio = detect_silence(samples, 0.01); // 1% threshold

        // Calculate dynamic range (difference between peak and RMS in dB)
        if analysis.rms_level > 0.0 {
            analysis.dynamic_range = 20.0 * (analysis.peak_level / analysis.rms_level).log10();
        }

        // Simple spectral centroid estimation (not FFT-based)
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let magnitude = sample.abs();
            let frequency = (i as f32 / samples.len() as f32) * (sample_rate as f32 / 2.0);
            weighted_sum += magnitude * frequency;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            analysis.spectral_centroid = weighted_sum / magnitude_sum;
        }

        analysis
    }
}

/// Analyze audio buffer and return analysis structure
///
/// # Safety
/// The caller must ensure that:
/// - `buffer` points to a valid VoirsAudioBuffer
/// - `analysis` points to a valid VoirsAudioAnalysis structure for writing
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_analyze(
    buffer: *const VoirsAudioBuffer,
    analysis: *mut audio::VoirsAudioAnalysis,
) -> VoirsErrorCode {
    if buffer.is_null() || analysis.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &*buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples = std::slice::from_raw_parts(buffer_ref.samples, buffer_ref.length as usize);
        let result = audio::analyze_audio(samples, buffer_ref.sample_rate);

        *analysis = result;
    }

    VoirsErrorCode::Success
}

/// Apply fade-in effect to audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_fade_in(
    buffer: *mut VoirsAudioBuffer,
    fade_duration_ms: c_uint,
) -> VoirsErrorCode {
    if buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &mut *buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples =
            std::slice::from_raw_parts_mut(buffer_ref.samples, buffer_ref.length as usize);
        let fade_samples =
            ((fade_duration_ms as f32 / 1000.0) * buffer_ref.sample_rate as f32) as usize;

        audio::apply_fade_in(samples, fade_samples);
    }

    VoirsErrorCode::Success
}

/// Apply fade-out effect to audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_fade_out(
    buffer: *mut VoirsAudioBuffer,
    fade_duration_ms: c_uint,
) -> VoirsErrorCode {
    if buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &mut *buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples =
            std::slice::from_raw_parts_mut(buffer_ref.samples, buffer_ref.length as usize);
        let fade_samples =
            ((fade_duration_ms as f32 / 1000.0) * buffer_ref.sample_rate as f32) as usize;

        audio::apply_fade_out(samples, fade_samples);
    }

    VoirsErrorCode::Success
}

/// Apply low-pass filter to audio buffer for smoothing
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_low_pass_filter(
    buffer: *mut VoirsAudioBuffer,
    alpha: c_float,
) -> VoirsErrorCode {
    if buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &mut *buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples =
            std::slice::from_raw_parts_mut(buffer_ref.samples, buffer_ref.length as usize);

        audio::apply_low_pass_filter(samples, alpha);
    }

    VoirsErrorCode::Success
}

/// Apply soft limiter to audio buffer to prevent harsh clipping
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_soft_limiter(
    buffer: *mut VoirsAudioBuffer,
    threshold: c_float,
) -> VoirsErrorCode {
    if buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &mut *buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples =
            std::slice::from_raw_parts_mut(buffer_ref.samples, buffer_ref.length as usize);

        audio::apply_soft_limiter(samples, threshold);
    }

    VoirsErrorCode::Success
}

/// Calculate harmonic-to-noise ratio for audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_calculate_hnr(buffer: *const VoirsAudioBuffer) -> c_float {
    if buffer.is_null() {
        return 0.0;
    }

    unsafe {
        let buffer_ref = &*buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return 0.0;
        }

        let samples = std::slice::from_raw_parts(buffer_ref.samples, buffer_ref.length as usize);
        audio::calculate_hnr(samples, buffer_ref.sample_rate)
    }
}

/// Apply comprehensive audio enhancement to buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_enhance_quality(
    buffer: *mut VoirsAudioBuffer,
) -> VoirsErrorCode {
    if buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &mut *buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples =
            std::slice::from_raw_parts_mut(buffer_ref.samples, buffer_ref.length as usize);

        audio::enhance_audio_quality(samples, buffer_ref.sample_rate);
    }

    VoirsErrorCode::Success
}

/// Normalize audio buffer to specified peak level
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_normalize(
    buffer: *mut VoirsAudioBuffer,
    target_peak: c_float,
) -> VoirsErrorCode {
    if buffer.is_null() || target_peak <= 0.0 || target_peak > 1.0 {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &mut *buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples =
            std::slice::from_raw_parts_mut(buffer_ref.samples, buffer_ref.length as usize);
        audio::normalize_audio(samples, target_peak);
    }

    VoirsErrorCode::Success
}

/// Calculate spectral rolloff frequency for audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_calculate_spectral_rolloff(
    buffer: *const VoirsAudioBuffer,
) -> c_float {
    if buffer.is_null() {
        return 0.0;
    }

    unsafe {
        let buffer_ref = &*buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return 0.0;
        }

        let samples = std::slice::from_raw_parts(buffer_ref.samples, buffer_ref.length as usize);
        audio::calculate_spectral_rolloff(samples, buffer_ref.sample_rate)
    }
}

/// Calculate spectral flux for audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_calculate_spectral_flux(
    buffer: *const VoirsAudioBuffer,
) -> c_float {
    if buffer.is_null() {
        return 0.0;
    }

    unsafe {
        let buffer_ref = &*buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return 0.0;
        }

        let samples = std::slice::from_raw_parts(buffer_ref.samples, buffer_ref.length as usize);
        audio::calculate_spectral_flux(samples, buffer_ref.sample_rate)
    }
}

/// Calculate audio brightness for audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_calculate_brightness(
    buffer: *const VoirsAudioBuffer,
) -> c_float {
    if buffer.is_null() {
        return 0.0;
    }

    unsafe {
        let buffer_ref = &*buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return 0.0;
        }

        let samples = std::slice::from_raw_parts(buffer_ref.samples, buffer_ref.length as usize);
        audio::calculate_brightness(samples, buffer_ref.sample_rate)
    }
}

/// Apply dynamic range compression to audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_apply_compression(
    buffer: *mut VoirsAudioBuffer,
    threshold: c_float,
    ratio: c_float,
    attack: c_float,
    release: c_float,
) -> VoirsErrorCode {
    if buffer.is_null()
        || threshold <= 0.0
        || threshold > 1.0
        || ratio <= 1.0
        || attack <= 0.0
        || release <= 0.0
    {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &mut *buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples =
            std::slice::from_raw_parts_mut(buffer_ref.samples, buffer_ref.length as usize);
        audio::apply_dynamic_compression(samples, threshold, ratio, attack, release);
    }

    VoirsErrorCode::Success
}

/// Apply multiband EQ to audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_apply_multiband_eq(
    buffer: *mut VoirsAudioBuffer,
    low_gain: c_float,
    mid_gain: c_float,
    high_gain: c_float,
) -> VoirsErrorCode {
    if buffer.is_null() || low_gain < 0.0 || mid_gain < 0.0 || high_gain < 0.0 {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &mut *buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples =
            std::slice::from_raw_parts_mut(buffer_ref.samples, buffer_ref.length as usize);
        audio::apply_multiband_eq(samples, low_gain, mid_gain, high_gain);
    }

    VoirsErrorCode::Success
}

/// Apply DC removal filter to audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_remove_dc(
    buffer: *mut VoirsAudioBuffer,
    filter_strength: c_float,
) -> VoirsErrorCode {
    if buffer.is_null() || !(0.0..=1.0).contains(&filter_strength) {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let buffer_ref = &mut *buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let samples =
            std::slice::from_raw_parts_mut(buffer_ref.samples, buffer_ref.length as usize);
        audio::apply_dc_filter(samples, filter_strength);
    }

    VoirsErrorCode::Success
}

/// Calculate RMS level of audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_get_rms(buffer: *const VoirsAudioBuffer) -> c_float {
    if buffer.is_null() {
        return 0.0;
    }

    unsafe {
        let buffer_ref = &*buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return 0.0;
        }

        let samples = std::slice::from_raw_parts(buffer_ref.samples, buffer_ref.length as usize);
        audio::calculate_rms(samples)
    }
}

/// Calculate peak level of audio buffer
///
/// # Safety
/// The caller must ensure that `buffer` points to a valid VoirsAudioBuffer with valid samples.
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_get_peak(buffer: *const VoirsAudioBuffer) -> c_float {
    if buffer.is_null() {
        return 0.0;
    }

    unsafe {
        let buffer_ref = &*buffer;
        if buffer_ref.samples.is_null() || buffer_ref.length == 0 {
            return 0.0;
        }

        let samples = std::slice::from_raw_parts(buffer_ref.samples, buffer_ref.length as usize);
        audio::calculate_peak(samples)
    }
}

/// Create a new real-time performance monitor
///
/// # Safety
/// The caller must ensure that the returned handle is properly freed using `voirs_performance_monitor_free()`.
#[no_mangle]
pub extern "C" fn voirs_performance_monitor_create(
    sample_interval_ms: c_uint,
    max_samples: c_uint,
) -> *mut performance::RealTimePerformanceMonitor {
    let monitor = performance::RealTimePerformanceMonitor::new(
        sample_interval_ms as u64,
        max_samples as usize,
    );
    Box::into_raw(Box::new(monitor))
}

/// Record audio processing time measurement
///
/// # Safety
/// The caller must ensure that `monitor` points to a valid RealTimePerformanceMonitor.
#[no_mangle]
pub unsafe extern "C" fn voirs_performance_monitor_record_audio_time(
    monitor: *mut performance::RealTimePerformanceMonitor,
    duration_ms: c_float,
) -> VoirsErrorCode {
    if monitor.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        (*monitor).record_audio_processing_time(duration_ms as f64);
    }
    VoirsErrorCode::Success
}

/// Record memory usage measurement
///
/// # Safety
/// The caller must ensure that `monitor` points to a valid RealTimePerformanceMonitor.
#[no_mangle]
pub unsafe extern "C" fn voirs_performance_monitor_record_memory(
    monitor: *mut performance::RealTimePerformanceMonitor,
    bytes: c_uint,
) -> VoirsErrorCode {
    if monitor.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        (*monitor).record_memory_usage(bytes as usize);
    }
    VoirsErrorCode::Success
}

/// Record CPU usage measurement
///
/// # Safety
/// The caller must ensure that `monitor` points to a valid RealTimePerformanceMonitor.
#[no_mangle]
pub unsafe extern "C" fn voirs_performance_monitor_record_cpu(
    monitor: *mut performance::RealTimePerformanceMonitor,
    cpu_percent: c_float,
) -> VoirsErrorCode {
    if monitor.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        (*monitor).record_cpu_usage(cpu_percent as f64);
    }
    VoirsErrorCode::Success
}

/// Get performance summary from monitor
///
/// # Safety
/// The caller must ensure that `monitor` points to a valid RealTimePerformanceMonitor
/// and `summary` points to a valid PerformanceSummary structure.
#[no_mangle]
pub unsafe extern "C" fn voirs_performance_monitor_get_summary(
    monitor: *const performance::RealTimePerformanceMonitor,
    summary: *mut performance::PerformanceSummary,
) -> VoirsErrorCode {
    if monitor.is_null() || summary.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        *summary = (*monitor).get_performance_summary();
    }
    VoirsErrorCode::Success
}

/// Free a performance monitor
///
/// # Safety
/// The caller must ensure that `monitor` was created by `voirs_performance_monitor_create()`
/// and is called at most once per monitor.
#[no_mangle]
pub unsafe extern "C" fn voirs_performance_monitor_free(
    monitor: *mut performance::RealTimePerformanceMonitor,
) {
    if !monitor.is_null() {
        let _ = Box::from_raw(monitor);
    }
}

/// Create a new performance regression detector
///
/// # Safety
/// The caller must ensure that the returned handle is properly freed using `voirs_regression_detector_free()`.
#[no_mangle]
pub extern "C" fn voirs_regression_detector_create(
    window_size: c_uint,
    regression_threshold_percent: c_float,
) -> *mut performance::PerformanceRegressionDetector {
    let detector = performance::PerformanceRegressionDetector::new(
        window_size as usize,
        regression_threshold_percent as f64,
    );
    Box::into_raw(Box::new(detector))
}

/// Set baseline measurements for regression detection
///
/// # Safety
/// The caller must ensure that `detector` points to a valid PerformanceRegressionDetector
/// and `baseline_values` points to an array of at least `count` f32 values.
#[no_mangle]
pub unsafe extern "C" fn voirs_regression_detector_set_baseline(
    detector: *mut performance::PerformanceRegressionDetector,
    baseline_values: *const c_float,
    count: c_uint,
) -> VoirsErrorCode {
    if detector.is_null() || baseline_values.is_null() || count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let values_slice = std::slice::from_raw_parts(baseline_values, count as usize);
        let values_vec: Vec<f64> = values_slice.iter().map(|&x| x as f64).collect();
        (*detector).set_baseline(values_vec);
    }
    VoirsErrorCode::Success
}

/// Add a measurement to the regression detector
///
/// # Safety
/// The caller must ensure that `detector` points to a valid PerformanceRegressionDetector.
#[no_mangle]
pub unsafe extern "C" fn voirs_regression_detector_add_measurement(
    detector: *mut performance::PerformanceRegressionDetector,
    value: c_float,
) -> VoirsErrorCode {
    if detector.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        (*detector).add_measurement(value as f64);
    }
    VoirsErrorCode::Success
}

/// Check for performance regression
///
/// # Safety
/// The caller must ensure that `detector` points to a valid PerformanceRegressionDetector
/// and `result` points to a valid RegressionResult structure.
#[no_mangle]
pub unsafe extern "C" fn voirs_regression_detector_check(
    detector: *const performance::PerformanceRegressionDetector,
    result: *mut performance::RegressionResult,
) -> VoirsErrorCode {
    if detector.is_null() || result.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        *result = (*detector).check_for_regression();
    }
    VoirsErrorCode::Success
}

/// Free a regression detector
///
/// # Safety
/// The caller must ensure that `detector` was created by `voirs_regression_detector_create()`
/// and is called at most once per detector.
#[no_mangle]
pub unsafe extern "C" fn voirs_regression_detector_free(
    detector: *mut performance::PerformanceRegressionDetector,
) {
    if !detector.is_null() {
        let _ = Box::from_raw(detector);
    }
}

#[cfg(test)]
mod tests {
    use super::audio::*;
    use super::*;

    #[test]
    fn test_audio_analysis() {
        let samples = vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.2, -0.7, 0.4];
        let analysis = analyze_audio(&samples, 44100);

        assert!(analysis.rms_level > 0.0);
        assert!(analysis.peak_level > 0.0);
        assert!(analysis.zero_crossing_rate >= 0.0);
        assert!(analysis.silence_ratio >= 0.0 && analysis.silence_ratio <= 1.0);
    }

    #[test]
    fn test_rms_calculation() {
        let samples = vec![1.0, 0.0, -1.0, 0.0];
        let rms = calculate_rms(&samples);
        let expected = (2.0 / 4.0_f32).sqrt(); // sqrt(0.5)
        assert!((rms - expected).abs() < 0.001);

        // Test empty array
        assert_eq!(calculate_rms(&[]), 0.0);
    }

    #[test]
    fn test_peak_calculation() {
        let samples = vec![0.5, -0.8, 0.3, -0.2];
        let peak = calculate_peak(&samples);
        assert_eq!(peak, 0.8);

        // Test empty array
        assert_eq!(calculate_peak(&[]), 0.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let samples = vec![1.0, -1.0, 1.0, -1.0]; // 3 crossings in 3 intervals
        let zcr = calculate_zero_crossing_rate(&samples);
        assert_eq!(zcr, 1.0); // 3/3 = 1.0

        // Test constant signal (no crossings)
        let constant = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(calculate_zero_crossing_rate(&constant), 0.0);

        // Test empty/single sample
        assert_eq!(calculate_zero_crossing_rate(&[]), 0.0);
        assert_eq!(calculate_zero_crossing_rate(&[1.0]), 0.0);
    }

    #[test]
    fn test_silence_detection() {
        let samples = vec![0.005, 0.0, 0.5, 0.001]; // 3 samples below 0.01 threshold
        let silence_ratio = detect_silence(&samples, 0.01);
        assert_eq!(silence_ratio, 0.75); // 3/4 = 0.75

        // Test empty array
        assert_eq!(detect_silence(&[], 0.01), 1.0);
    }

    #[test]
    fn test_fade_in() {
        let mut samples = vec![1.0, 1.0, 1.0, 1.0];
        apply_fade_in(&mut samples, 2);

        assert_eq!(samples[0], 0.0); // 0/2 * 1.0
        assert_eq!(samples[1], 0.5); // 1/2 * 1.0
        assert_eq!(samples[2], 1.0); // Unchanged
        assert_eq!(samples[3], 1.0); // Unchanged
    }

    #[test]
    fn test_fade_out() {
        let mut samples = vec![1.0, 1.0, 1.0, 1.0];
        apply_fade_out(&mut samples, 2);

        assert_eq!(samples[0], 1.0); // Unchanged
        assert_eq!(samples[1], 1.0); // Unchanged
        assert_eq!(samples[2], 1.0); // 1 - 0/2 = 1.0
        assert_eq!(samples[3], 0.5); // 1 - 1/2 = 0.5
    }

    #[test]
    fn test_normalization() {
        let mut samples = vec![0.5, -0.8, 0.3, -0.2];
        let original_peak = calculate_peak(&samples);

        normalize_audio(&mut samples, 1.0);
        let new_peak = calculate_peak(&samples);

        assert!((new_peak - 1.0).abs() < 0.001);

        // Check that relative amplitudes are preserved
        let scale_factor = 1.0 / original_peak;
        assert!((samples[0] - 0.5 * scale_factor).abs() < 0.001);
    }

    #[test]
    fn test_dc_filter() {
        let mut samples = vec![1.0, 1.0, 1.0, 1.0]; // DC signal
        let original = samples.clone();

        apply_dc_filter(&mut samples, 0.95);

        // After DC filtering, the steady DC should be reduced
        assert!(samples.iter().sum::<f32>() < original.iter().sum::<f32>());
    }

    #[test]
    fn test_ffi_audio_analysis() {
        use crate::VoirsAudioBuffer;

        let samples = [0.5f32, -0.3, 0.8, -0.1];
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let mut analysis = VoirsAudioAnalysis::default();
        let result = unsafe { voirs_audio_analyze(&buffer, &mut analysis) };

        assert_eq!(result, crate::VoirsErrorCode::Success);
        assert!(analysis.rms_level > 0.0);
        assert!(analysis.peak_level > 0.0);
    }

    #[test]
    fn test_ffi_rms_calculation() {
        use crate::VoirsAudioBuffer;

        let samples = [1.0f32, 0.0, -1.0, 0.0];
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let rms = unsafe { voirs_audio_get_rms(&buffer) };
        let expected = (2.0 / 4.0_f32).sqrt();
        assert!((rms - expected).abs() < 0.001);
    }

    #[test]
    fn test_ffi_peak_calculation() {
        use crate::VoirsAudioBuffer;

        let samples = [0.5f32, -0.8, 0.3, -0.2];
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let peak = unsafe { voirs_audio_get_peak(&buffer) };
        assert_eq!(peak, 0.8);
    }

    #[test]
    fn test_ffi_fade_operations() {
        use crate::VoirsAudioBuffer;

        let mut samples = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut buffer = VoirsAudioBuffer {
            samples: samples.as_mut_ptr(),
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        // Test fade in (roughly 45ms at 44100 Hz = ~2 samples)
        let result = unsafe { voirs_audio_fade_in(&mut buffer, 45) };
        assert_eq!(result, crate::VoirsErrorCode::Success);

        // Reset samples for fade out test
        samples = vec![1.0f32, 1.0, 1.0, 1.0];
        buffer.samples = samples.as_mut_ptr();

        // Test fade out
        let result = unsafe { voirs_audio_fade_out(&mut buffer, 45) };
        assert_eq!(result, crate::VoirsErrorCode::Success);
    }

    #[test]
    fn test_ffi_normalization() {
        use crate::VoirsAudioBuffer;

        let mut samples = vec![0.5f32, -0.8, 0.3, -0.2];
        let mut buffer = VoirsAudioBuffer {
            samples: samples.as_mut_ptr(),
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let result = unsafe { voirs_audio_normalize(&mut buffer, 1.0) };
        assert_eq!(result, crate::VoirsErrorCode::Success);

        let new_peak = calculate_peak(&samples);
        assert!((new_peak - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ffi_dc_removal() {
        use crate::VoirsAudioBuffer;

        let mut samples = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut buffer = VoirsAudioBuffer {
            samples: samples.as_mut_ptr(),
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let result = unsafe { voirs_audio_remove_dc(&mut buffer, 0.95) };
        assert_eq!(result, crate::VoirsErrorCode::Success);
    }

    #[test]
    fn test_ffi_error_handling() {
        // Test null buffer
        assert_eq!(unsafe { voirs_audio_get_rms(std::ptr::null()) }, 0.0);
        assert_eq!(unsafe { voirs_audio_get_peak(std::ptr::null()) }, 0.0);

        // Test null analysis pointer
        use crate::VoirsAudioBuffer;
        let samples = [1.0f32];
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: 1,
            sample_rate: 44100,
            channels: 1,
            duration: 1.0 / 44100.0,
        };

        let result = unsafe { voirs_audio_analyze(&buffer, std::ptr::null_mut()) };
        assert_eq!(result, crate::VoirsErrorCode::InvalidParameter);
    }

    #[test]
    fn test_low_pass_filter() {
        let mut samples = vec![1.0, 0.0, 1.0, 0.0, 1.0]; // Alternating signal
        let original = samples.clone();

        apply_low_pass_filter(&mut samples, 0.5);

        // Low-pass should smooth the signal
        assert_ne!(samples, original);
        // Check that filtering has smoothed the signal
        assert!(samples[1] > 0.0); // Should be smoothed from 0.0
    }

    #[test]
    fn test_spectral_envelope() {
        let samples = vec![0.5, 0.8, 0.3, 0.7, 0.2, 0.9, 0.1, 0.6];
        let envelope = calculate_spectral_envelope(&samples, 4);

        assert_eq!(envelope.len(), 4);
        for &val in &envelope {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_soft_limiter() {
        let mut samples = vec![0.5, 1.5, -1.8, 0.3]; // Some samples exceed threshold
        apply_soft_limiter(&mut samples, 1.0);

        // Check that all samples are within reasonable bounds
        for &sample in &samples {
            assert!(sample.abs() <= 1.5); // Soft limiting, not hard clipping
        }

        // Original values within threshold should be unchanged
        assert_eq!(samples[0], 0.5);
        assert_eq!(samples[3], 0.3);
    }

    #[test]
    fn test_hnr_calculation() {
        // Create a simple periodic signal
        let mut samples = Vec::new();
        let period = 100;
        for i in 0..1000 {
            let phase = 2.0 * std::f32::consts::PI * (i % period) as f32 / period as f32;
            samples.push(phase.sin());
        }

        let hnr = calculate_hnr(&samples, 44100);
        // Periodic signal should have high HNR
        assert!(hnr > 5.0);

        // Test with noise
        let noise: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();
        let noise_hnr = calculate_hnr(&noise, 44100);
        assert!(noise_hnr < hnr); // Noise should have lower HNR
    }

    #[test]
    fn test_audio_enhancement() {
        let mut samples = vec![1.5, -1.8, 0.0, 0.5, 2.0, -2.5]; // Various levels including clipping
        let original = samples.clone();

        enhance_audio_quality(&mut samples, 44100);

        // Enhanced audio should be different from original
        assert_ne!(samples, original);

        // Check that peak levels are reasonable
        let peak = calculate_peak(&samples);
        assert!(peak <= 1.0); // Should be normalized/limited
    }

    #[test]
    fn test_ffi_low_pass_filter() {
        use crate::VoirsAudioBuffer;

        let mut samples = vec![1.0f32, 0.0, 1.0, 0.0];
        let mut buffer = VoirsAudioBuffer {
            samples: samples.as_mut_ptr(),
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let result = unsafe { voirs_audio_low_pass_filter(&mut buffer, 0.5) };
        assert_eq!(result, crate::VoirsErrorCode::Success);
    }

    #[test]
    fn test_ffi_soft_limiter() {
        use crate::VoirsAudioBuffer;

        let mut samples = vec![1.5f32, -1.8, 0.3, 2.0];
        let mut buffer = VoirsAudioBuffer {
            samples: samples.as_mut_ptr(),
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let result = unsafe { voirs_audio_soft_limiter(&mut buffer, 1.0) };
        assert_eq!(result, crate::VoirsErrorCode::Success);
    }

    #[test]
    fn test_ffi_hnr_calculation() {
        use crate::VoirsAudioBuffer;

        // Create a simple periodic signal
        let mut samples = Vec::new();
        for i in 0..1000 {
            let phase = 2.0 * std::f32::consts::PI * (i % 100) as f32 / 100.0;
            samples.push(phase.sin());
        }

        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let hnr = unsafe { voirs_audio_calculate_hnr(&buffer) };
        assert!(hnr > 0.0); // Should detect harmonicity in periodic signal
    }

    #[test]
    fn test_ffi_audio_enhancement() {
        use crate::VoirsAudioBuffer;

        let mut samples = vec![1.5f32, -1.8, 0.0, 0.5];
        let mut buffer = VoirsAudioBuffer {
            samples: samples.as_mut_ptr(),
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let result = unsafe { voirs_audio_enhance_quality(&mut buffer) };
        assert_eq!(result, crate::VoirsErrorCode::Success);
    }

    #[test]
    fn test_optimized_audio_enhancement() {
        let mut samples = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let mut expected_samples = samples.clone();

        // Apply regular enhancement
        audio::enhance_audio_quality(&mut expected_samples, 44100);

        // Apply optimized enhancement
        audio::enhance_audio_quality_optimized(&mut samples, 44100);

        // Both should produce valid audio output
        for &sample in &samples {
            assert!(sample.abs() <= 1.0, "Sample should be within valid range");
        }

        for &sample in &expected_samples {
            assert!(
                sample.abs() <= 1.0,
                "Expected sample should be within valid range"
            );
        }
    }
}

/// Advanced performance analysis and benchmarking utilities
pub mod performance {
    // Note: super::* import removed as it causes unused import warning
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    /// Performance benchmark result
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct BenchmarkResult {
        pub mean_duration_us: f64,       // Mean execution time in microseconds
        pub min_duration_us: f64,        // Minimum execution time in microseconds
        pub max_duration_us: f64,        // Maximum execution time in microseconds
        pub std_deviation_us: f64,       // Standard deviation in microseconds
        pub throughput_ops_per_sec: f64, // Operations per second
        pub samples_processed: u64,      // Total samples processed
        pub total_iterations: u32,       // Number of benchmark iterations
    }

    impl Default for BenchmarkResult {
        fn default() -> Self {
            Self {
                mean_duration_us: 0.0,
                min_duration_us: 0.0,
                max_duration_us: 0.0,
                std_deviation_us: 0.0,
                throughput_ops_per_sec: 0.0,
                samples_processed: 0,
                total_iterations: 0,
            }
        }
    }

    /// Audio processing benchmark suite
    pub struct AudioBenchmark {
        durations: Vec<Duration>,
        samples_per_iteration: usize,
    }

    impl Default for AudioBenchmark {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AudioBenchmark {
        pub fn new() -> Self {
            Self {
                durations: Vec::new(),
                samples_per_iteration: 0,
            }
        }

        /// Benchmark audio processing function
        pub fn benchmark_audio_function<F>(
            &mut self,
            samples: &[f32],
            iterations: u32,
            mut function: F,
        ) -> BenchmarkResult
        where
            F: FnMut(&[f32]),
        {
            self.durations.clear();
            self.samples_per_iteration = samples.len();

            // Warmup iterations
            for _ in 0..5 {
                function(samples);
            }

            // Actual benchmark iterations
            for _ in 0..iterations {
                let start = Instant::now();
                function(samples);
                let duration = start.elapsed();
                self.durations.push(duration);
            }

            self.calculate_results(iterations)
        }

        /// Benchmark mutable audio processing function
        pub fn benchmark_audio_function_mut<F>(
            &mut self,
            samples: &[f32],
            iterations: u32,
            mut function: F,
        ) -> BenchmarkResult
        where
            F: FnMut(&mut [f32]),
        {
            self.durations.clear();
            self.samples_per_iteration = samples.len();

            // Warmup iterations
            for _ in 0..5 {
                let mut test_samples = samples.to_vec();
                function(&mut test_samples);
            }

            // Actual benchmark iterations
            for _ in 0..iterations {
                let mut test_samples = samples.to_vec();
                let start = Instant::now();
                function(&mut test_samples);
                let duration = start.elapsed();
                self.durations.push(duration);
            }

            self.calculate_results(iterations)
        }

        fn calculate_results(&self, iterations: u32) -> BenchmarkResult {
            if self.durations.is_empty() {
                return BenchmarkResult::default();
            }

            let durations_us: Vec<f64> = self
                .durations
                .iter()
                .map(|d| d.as_secs_f64() * 1_000_000.0)
                .collect();

            let min_duration = durations_us.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_duration = durations_us
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mean_duration = durations_us.iter().sum::<f64>() / durations_us.len() as f64;

            let variance = durations_us
                .iter()
                .map(|x| (x - mean_duration).powi(2))
                .sum::<f64>()
                / durations_us.len() as f64;
            let std_deviation = variance.sqrt();

            let throughput = if mean_duration > 0.0 {
                1_000_000.0 / mean_duration // operations per second
            } else {
                0.0
            };

            BenchmarkResult {
                mean_duration_us: mean_duration,
                min_duration_us: min_duration,
                max_duration_us: max_duration,
                std_deviation_us: std_deviation,
                throughput_ops_per_sec: throughput,
                samples_processed: (self.samples_per_iteration * iterations as usize) as u64,
                total_iterations: iterations,
            }
        }
    }

    /// Memory allocation pattern analyzer
    pub struct MemoryPatternAnalyzer {
        allocation_sizes: VecDeque<usize>,
        allocation_times: VecDeque<Instant>,
        max_history: usize,
    }

    impl MemoryPatternAnalyzer {
        pub fn new(max_history: usize) -> Self {
            Self {
                allocation_sizes: VecDeque::new(),
                allocation_times: VecDeque::new(),
                max_history,
            }
        }

        pub fn record_allocation(&mut self, size: usize) {
            let now = Instant::now();

            self.allocation_sizes.push_back(size);
            self.allocation_times.push_back(now);

            // Maintain history limit
            while self.allocation_sizes.len() > self.max_history {
                self.allocation_sizes.pop_front();
                self.allocation_times.pop_front();
            }
        }

        pub fn analyze_patterns(&self) -> MemoryPatternStats {
            if self.allocation_sizes.is_empty() {
                return MemoryPatternStats::default();
            }

            let sizes: Vec<usize> = self.allocation_sizes.iter().cloned().collect();
            let total_allocations = sizes.len();
            let total_bytes: usize = sizes.iter().sum();
            let average_size = total_bytes as f64 / total_allocations as f64;

            let mut size_histogram = std::collections::HashMap::new();
            for &size in &sizes {
                let bucket = Self::size_to_bucket(size);
                *size_histogram.entry(bucket).or_insert(0) += 1;
            }

            let allocation_rate = if self.allocation_times.len() >= 2 {
                if let (Some(last_time), Some(first_time)) =
                    (self.allocation_times.back(), self.allocation_times.front())
                {
                    let time_span = last_time.duration_since(*first_time);
                    if time_span.as_secs_f64() > 0.0 {
                        total_allocations as f64 / time_span.as_secs_f64()
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };

            MemoryPatternStats {
                total_allocations,
                total_bytes,
                average_allocation_size: average_size,
                allocation_rate_per_sec: allocation_rate,
                size_distribution: size_histogram,
            }
        }

        fn size_to_bucket(size: usize) -> String {
            match size {
                0..=1024 => "0-1KB".to_string(),
                1025..=4096 => "1-4KB".to_string(),
                4097..=16384 => "4-16KB".to_string(),
                16385..=65536 => "16-64KB".to_string(),
                65537..=262144 => "64-256KB".to_string(),
                262145..=1048576 => "256KB-1MB".to_string(),
                _ => "1MB+".to_string(),
            }
        }
    }

    /// Memory allocation pattern statistics
    #[derive(Debug, Clone)]
    pub struct MemoryPatternStats {
        pub total_allocations: usize,
        pub total_bytes: usize,
        pub average_allocation_size: f64,
        pub allocation_rate_per_sec: f64,
        pub size_distribution: std::collections::HashMap<String, usize>,
    }

    impl Default for MemoryPatternStats {
        fn default() -> Self {
            Self {
                total_allocations: 0,
                total_bytes: 0,
                average_allocation_size: 0.0,
                allocation_rate_per_sec: 0.0,
                size_distribution: std::collections::HashMap::new(),
            }
        }
    }

    /// FFI call overhead measurement utility
    pub struct FFIOverheadAnalyzer {
        call_durations: Vec<Duration>,
        data_sizes: Vec<usize>,
    }

    /// Real-time performance monitoring system
    pub struct RealTimePerformanceMonitor {
        audio_processing_times: VecDeque<f64>,
        memory_usage_samples: VecDeque<usize>,
        cpu_usage_samples: VecDeque<f64>,
        start_time: Instant,
        sample_interval: Duration,
        max_samples: usize,
    }

    impl RealTimePerformanceMonitor {
        pub fn new(sample_interval_ms: u64, max_samples: usize) -> Self {
            Self {
                audio_processing_times: VecDeque::new(),
                memory_usage_samples: VecDeque::new(),
                cpu_usage_samples: VecDeque::new(),
                start_time: Instant::now(),
                sample_interval: Duration::from_millis(sample_interval_ms),
                max_samples,
            }
        }

        pub fn record_audio_processing_time(&mut self, duration_ms: f64) {
            self.audio_processing_times.push_back(duration_ms);
            let max_samples = self.max_samples;
            Self::maintain_sample_limit(&mut self.audio_processing_times, max_samples);
        }

        pub fn record_memory_usage(&mut self, bytes: usize) {
            self.memory_usage_samples.push_back(bytes);
            let max_samples = self.max_samples;
            Self::maintain_sample_limit(&mut self.memory_usage_samples, max_samples);
        }

        pub fn record_cpu_usage(&mut self, cpu_percent: f64) {
            self.cpu_usage_samples.push_back(cpu_percent);
            let max_samples = self.max_samples;
            Self::maintain_sample_limit(&mut self.cpu_usage_samples, max_samples);
        }

        fn maintain_sample_limit<T>(samples: &mut VecDeque<T>, max_samples: usize) {
            while samples.len() > max_samples {
                samples.pop_front();
            }
        }

        pub fn get_performance_summary(&self) -> PerformanceSummary {
            let audio_times: Vec<f64> = self.audio_processing_times.iter().cloned().collect();
            let memory_samples: Vec<usize> = self.memory_usage_samples.iter().cloned().collect();
            let cpu_samples: Vec<f64> = self.cpu_usage_samples.iter().cloned().collect();

            let audio_stats = self.calculate_stats(&audio_times);
            let memory_stats = self.calculate_memory_stats(&memory_samples);
            let cpu_stats = self.calculate_stats(&cpu_samples);

            PerformanceSummary {
                uptime_seconds: self.start_time.elapsed().as_secs(),
                audio_processing_avg_ms: audio_stats.mean,
                audio_processing_max_ms: audio_stats.max,
                memory_usage_avg_mb: memory_stats.mean / (1024.0 * 1024.0),
                memory_usage_peak_mb: memory_stats.max / (1024.0 * 1024.0),
                cpu_usage_avg_percent: cpu_stats.mean,
                cpu_usage_peak_percent: cpu_stats.max,
                total_samples_collected: self.audio_processing_times.len()
                    + self.memory_usage_samples.len()
                    + self.cpu_usage_samples.len(),
            }
        }

        fn calculate_stats(&self, values: &[f64]) -> StatsSummary {
            if values.is_empty() {
                return StatsSummary {
                    mean: 0.0,
                    max: 0.0,
                    min: 0.0,
                    std_dev: 0.0,
                };
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);

            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            StatsSummary {
                mean,
                max,
                min,
                std_dev,
            }
        }

        fn calculate_memory_stats(&self, values: &[usize]) -> StatsSummary {
            if values.is_empty() {
                return StatsSummary {
                    mean: 0.0,
                    max: 0.0,
                    min: 0.0,
                    std_dev: 0.0,
                };
            }

            let float_values: Vec<f64> = values.iter().map(|&x| x as f64).collect();
            self.calculate_stats(&float_values)
        }
    }

    /// Statistics summary structure
    #[derive(Debug, Clone, Copy)]
    pub struct StatsSummary {
        pub mean: f64,
        pub max: f64,
        pub min: f64,
        pub std_dev: f64,
    }

    /// Real-time performance summary
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct PerformanceSummary {
        pub uptime_seconds: u64,
        pub audio_processing_avg_ms: f64,
        pub audio_processing_max_ms: f64,
        pub memory_usage_avg_mb: f64,
        pub memory_usage_peak_mb: f64,
        pub cpu_usage_avg_percent: f64,
        pub cpu_usage_peak_percent: f64,
        pub total_samples_collected: usize,
    }

    impl Default for PerformanceSummary {
        fn default() -> Self {
            Self {
                uptime_seconds: 0,
                audio_processing_avg_ms: 0.0,
                audio_processing_max_ms: 0.0,
                memory_usage_avg_mb: 0.0,
                memory_usage_peak_mb: 0.0,
                cpu_usage_avg_percent: 0.0,
                cpu_usage_peak_percent: 0.0,
                total_samples_collected: 0,
            }
        }
    }

    /// Performance regression detector
    pub struct PerformanceRegressionDetector {
        baseline_metrics: Vec<f64>,
        current_window: VecDeque<f64>,
        window_size: usize,
        regression_threshold: f64, // Percentage increase considered regression
    }

    impl PerformanceRegressionDetector {
        pub fn new(window_size: usize, regression_threshold_percent: f64) -> Self {
            Self {
                baseline_metrics: Vec::new(),
                current_window: VecDeque::new(),
                window_size,
                regression_threshold: regression_threshold_percent / 100.0,
            }
        }

        pub fn set_baseline(&mut self, baseline_values: Vec<f64>) {
            self.baseline_metrics = baseline_values;
        }

        pub fn add_measurement(&mut self, value: f64) {
            self.current_window.push_back(value);
            if self.current_window.len() > self.window_size {
                self.current_window.pop_front();
            }
        }

        pub fn check_for_regression(&self) -> RegressionResult {
            if self.baseline_metrics.is_empty() || self.current_window.is_empty() {
                return RegressionResult {
                    has_regression: false,
                    baseline_avg: 0.0,
                    current_avg: 0.0,
                    regression_percent: 0.0,
                };
            }

            let baseline_avg =
                self.baseline_metrics.iter().sum::<f64>() / self.baseline_metrics.len() as f64;
            let current_avg =
                self.current_window.iter().sum::<f64>() / self.current_window.len() as f64;

            let regression_percent = if baseline_avg > 0.0 {
                (current_avg - baseline_avg) / baseline_avg
            } else {
                0.0
            };

            let has_regression = regression_percent > self.regression_threshold;

            RegressionResult {
                has_regression,
                baseline_avg,
                current_avg,
                regression_percent,
            }
        }
    }

    /// Performance regression detection result
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct RegressionResult {
        pub has_regression: bool,
        pub baseline_avg: f64,
        pub current_avg: f64,
        pub regression_percent: f64,
    }

    impl Default for FFIOverheadAnalyzer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FFIOverheadAnalyzer {
        pub fn new() -> Self {
            Self {
                call_durations: Vec::new(),
                data_sizes: Vec::new(),
            }
        }

        /// Measure FFI call overhead for a given function
        pub fn measure_ffi_call<F, T>(
            &mut self,
            data_size: usize,
            iterations: u32,
            mut ffi_function: F,
        ) -> FFIOverheadStats
        where
            F: FnMut() -> T,
        {
            self.call_durations.clear();
            self.data_sizes.clear();

            // Warmup
            for _ in 0..5 {
                ffi_function();
            }

            // Measure actual calls
            for _ in 0..iterations {
                let start = Instant::now();
                ffi_function();
                let duration = start.elapsed();

                self.call_durations.push(duration);
                self.data_sizes.push(data_size);
            }

            self.calculate_overhead_stats(iterations)
        }

        fn calculate_overhead_stats(&self, iterations: u32) -> FFIOverheadStats {
            if self.call_durations.is_empty() {
                return FFIOverheadStats::default();
            }

            let durations_ns: Vec<u64> = self
                .call_durations
                .iter()
                .map(|d| d.as_nanos() as u64)
                .collect();

            let min_overhead = durations_ns.iter().min().copied().unwrap_or(0);
            let max_overhead = durations_ns.iter().max().copied().unwrap_or(0);
            let mean_overhead = durations_ns.iter().sum::<u64>() / durations_ns.len() as u64;

            let total_data_processed: usize = self.data_sizes.iter().sum();
            let throughput_mbps = if mean_overhead > 0 {
                (total_data_processed as f64 * 1000.0) / (mean_overhead as f64 * iterations as f64)
            } else {
                0.0
            };

            FFIOverheadStats {
                mean_overhead_ns: mean_overhead,
                min_overhead_ns: min_overhead,
                max_overhead_ns: max_overhead,
                throughput_mb_per_sec: throughput_mbps,
                total_iterations: iterations,
                avg_data_size_bytes: total_data_processed / iterations as usize,
            }
        }
    }

    /// FFI call overhead statistics
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct FFIOverheadStats {
        pub mean_overhead_ns: u64,      // Mean overhead in nanoseconds
        pub min_overhead_ns: u64,       // Minimum overhead in nanoseconds
        pub max_overhead_ns: u64,       // Maximum overhead in nanoseconds
        pub throughput_mb_per_sec: f64, // Data throughput in MB/s
        pub total_iterations: u32,      // Number of iterations tested
        pub avg_data_size_bytes: usize, // Average data size per call
    }

    impl Default for FFIOverheadStats {
        fn default() -> Self {
            Self {
                mean_overhead_ns: 0,
                min_overhead_ns: 0,
                max_overhead_ns: 0,
                throughput_mb_per_sec: 0.0,
                total_iterations: 0,
                avg_data_size_bytes: 0,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::utils::audio::calculate_rms;

        #[test]
        fn test_audio_benchmark() {
            let samples = vec![0.5f32; 1000];
            let mut benchmark = AudioBenchmark::new();

            let result = benchmark.benchmark_audio_function(&samples, 100, |samples| {
                // Simulate some audio processing work with more computation
                let _rms = calculate_rms(samples);
                // Add more work to ensure measurable timing
                let mut sum = 0.0;
                for &sample in samples {
                    sum += sample * sample * sample; // Cubic operation for more work
                }
                std::hint::black_box(sum); // Prevent optimization
            });

            assert!(result.total_iterations == 100);
            assert!(result.samples_processed == 100000);
            assert!(result.mean_duration_us > 0.0);
            assert!(result.throughput_ops_per_sec > 0.0);
        }

        #[test]
        fn test_memory_pattern_analyzer() {
            let mut analyzer = MemoryPatternAnalyzer::new(100);

            // Simulate allocation pattern
            analyzer.record_allocation(1024);
            analyzer.record_allocation(2048);
            analyzer.record_allocation(512);

            let stats = analyzer.analyze_patterns();
            assert_eq!(stats.total_allocations, 3);
            assert_eq!(stats.total_bytes, 3584);
            assert!((stats.average_allocation_size - 1194.67).abs() < 0.1);
        }

        #[test]
        fn test_ffi_overhead_analyzer() {
            let mut analyzer = FFIOverheadAnalyzer::new();

            let stats = analyzer.measure_ffi_call(1000, 5, || {
                // Simulate FFI call overhead
                std::thread::sleep(Duration::from_nanos(100));
                42
            });

            assert_eq!(stats.total_iterations, 5);
            assert_eq!(stats.avg_data_size_bytes, 1000);
            assert!(stats.mean_overhead_ns > 0);
        }

        #[test]
        fn test_real_time_performance_monitor() {
            let mut monitor = RealTimePerformanceMonitor::new(100, 10);

            // Record some sample data
            monitor.record_audio_processing_time(15.5);
            monitor.record_audio_processing_time(12.3);
            monitor.record_memory_usage(1024 * 1024); // 1MB
            monitor.record_cpu_usage(45.0);

            let summary = monitor.get_performance_summary();
            assert!(summary.audio_processing_avg_ms > 0.0);
            assert!(summary.memory_usage_avg_mb > 0.0);
            assert!(summary.cpu_usage_avg_percent > 0.0);
            assert_eq!(summary.total_samples_collected, 4);
        }

        #[test]
        fn test_performance_regression_detector() {
            let mut detector = PerformanceRegressionDetector::new(5, 20.0); // 20% threshold

            // Set baseline (good performance)
            detector.set_baseline(vec![10.0, 11.0, 9.5, 10.5, 10.2]);

            // Add measurements that show no regression
            detector.add_measurement(10.1);
            detector.add_measurement(10.8);
            let result = detector.check_for_regression();
            assert!(!result.has_regression);

            // Add measurements that show regression
            detector.add_measurement(15.0); // 50% increase
            detector.add_measurement(14.5);
            detector.add_measurement(16.0);
            let result = detector.check_for_regression();
            assert!(result.has_regression);
            assert!(result.regression_percent > 0.2); // More than 20% regression
        }

        #[test]
        fn test_performance_monitor_sample_limit() {
            let mut monitor = RealTimePerformanceMonitor::new(100, 3); // Max 3 samples

            // Add more samples than the limit
            monitor.record_audio_processing_time(10.0);
            monitor.record_audio_processing_time(20.0);
            monitor.record_audio_processing_time(30.0);
            monitor.record_audio_processing_time(40.0); // Should evict first sample

            let summary = monitor.get_performance_summary();
            // Should average the last 3 samples: (20 + 30 + 40) / 3 = 30
            assert!((summary.audio_processing_avg_ms - 30.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_spectral_rolloff() {
        use crate::utils::audio;

        // Test with a simple signal
        let samples = vec![1.0, 0.5, 0.2, 0.1, 0.05, 0.01];
        let rolloff = audio::calculate_spectral_rolloff(&samples, 44100);
        assert!(rolloff > 0.0);

        // Test with empty input
        assert_eq!(audio::calculate_spectral_rolloff(&[], 44100), 0.0);
    }

    #[test]
    fn test_spectral_flux() {
        use crate::utils::audio;

        // Test with a changing signal
        let samples = vec![1.0; 1024]; // Constant signal should have low flux
        let flux = audio::calculate_spectral_flux(&samples, 44100);
        assert!(flux >= 0.0);

        // Test with insufficient samples
        let short_samples = vec![1.0; 100];
        assert_eq!(audio::calculate_spectral_flux(&short_samples, 44100), 0.0);
    }

    #[test]
    fn test_brightness() {
        use crate::utils::audio;

        // Test with a signal that has both low and high frequencies
        let samples = vec![1.0, 0.5, 0.2, 0.1, 0.05, 0.01];
        let brightness = audio::calculate_brightness(&samples, 44100);
        assert!(brightness >= 0.0);

        // Test with empty input
        assert_eq!(audio::calculate_brightness(&[], 44100), 0.0);
    }

    #[test]
    fn test_dynamic_compression() {
        use crate::utils::audio;

        let mut samples = vec![0.1, 0.8, 0.9, 0.2, 0.7]; // Mix of low and high amplitude
        let original_samples = samples.clone();

        audio::apply_dynamic_compression(&mut samples, 0.5, 2.0, 0.1, 0.1);

        // The high amplitude samples should be compressed more than low amplitude ones
        assert!(samples[1] <= original_samples[1]); // 0.8 should be compressed
        assert!(samples[2] <= original_samples[2]); // 0.9 should be compressed
        assert!(samples[0] == original_samples[0]); // 0.1 should be unchanged (below threshold)
    }

    #[test]
    fn test_multiband_eq() {
        use crate::utils::audio;

        let mut samples = vec![1.0f32, 0.5, -0.5, -1.0, 0.2, -0.2];
        let original_samples = samples.clone();

        audio::apply_multiband_eq(&mut samples, 1.0, 1.0, 1.0);

        // With gains of 1.0, the output should be similar to input
        for (i, &sample) in samples.iter().enumerate() {
            assert!((sample - original_samples[i]).abs() < 0.5); // Allow some filtering artifacts
        }
    }

    #[test]
    fn test_ffi_spectral_rolloff() {
        use crate::VoirsAudioBuffer;

        let samples = [1.0f32, 0.5, 0.2, 0.1, 0.05, 0.01];
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let rolloff = unsafe { super::voirs_audio_calculate_spectral_rolloff(&buffer) };
        assert!(rolloff > 0.0);
    }

    #[test]
    fn test_ffi_spectral_flux() {
        use crate::VoirsAudioBuffer;

        let samples = vec![1.0f32; 1024]; // Constant signal
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let flux = unsafe { super::voirs_audio_calculate_spectral_flux(&buffer) };
        assert!(flux >= 0.0);
    }

    #[test]
    fn test_ffi_brightness() {
        use crate::VoirsAudioBuffer;

        let samples = [1.0f32, 0.5, 0.2, 0.1, 0.05, 0.01];
        let buffer = VoirsAudioBuffer {
            samples: samples.as_ptr() as *mut f32,
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let brightness = unsafe { super::voirs_audio_calculate_brightness(&buffer) };
        assert!(brightness >= 0.0);
    }

    #[test]
    fn test_ffi_compression() {
        use crate::VoirsAudioBuffer;

        let mut samples = vec![0.1f32, 0.8, 0.9, 0.2, 0.7];
        let mut buffer = VoirsAudioBuffer {
            samples: samples.as_mut_ptr(),
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let result =
            unsafe { super::voirs_audio_apply_compression(&mut buffer, 0.5, 2.0, 0.1, 0.1) };
        assert_eq!(result, crate::VoirsErrorCode::Success);

        // High amplitude samples should be compressed
        assert!(samples[1] <= 0.8); // 0.8 should be compressed
        assert!(samples[2] <= 0.9); // 0.9 should be compressed
    }

    #[test]
    fn test_ffi_multiband_eq() {
        use crate::VoirsAudioBuffer;

        let mut samples = vec![1.0f32, 0.5, -0.5, -1.0, 0.2, -0.2];
        let mut buffer = VoirsAudioBuffer {
            samples: samples.as_mut_ptr(),
            length: samples.len() as u32,
            sample_rate: 44100,
            channels: 1,
            duration: samples.len() as f32 / 44100.0,
        };

        let result = unsafe { super::voirs_audio_apply_multiband_eq(&mut buffer, 1.0, 1.0, 1.0) };
        assert_eq!(result, crate::VoirsErrorCode::Success);

        // With gains of 1.0, the samples should be modified but not drastically
        for &sample in &samples {
            assert!(sample.abs() <= 2.0); // Should be reasonably bounded
        }
    }
}
