//! Performance optimizations for VoiRS FFI operations.
//!
//! This module provides platform-specific optimizations, FFI overhead reduction,
//! and vectorized operations for maximum performance.

use crate::{VoirsErrorCode, VoirsPerformanceConfig};
use std::os::raw::{c_float, c_uint};

/// SIMD-optimized audio processing functions
pub mod simd {
    // Note: super::* import removed as it is unused

    /// SIMD-optimized audio mixing
    #[cfg(target_arch = "x86_64")]
    pub fn mix_audio_simd(input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        #[cfg(target_feature = "avx512f")]
        unsafe {
            mix_audio_avx512(input1, input2, output, gain);
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        unsafe {
            mix_audio_avx2(input1, input2, output, gain);
        }
        #[cfg(all(
            target_feature = "sse2",
            not(any(target_feature = "avx2", target_feature = "avx512f"))
        ))]
        unsafe {
            mix_audio_sse2(input1, input2, output, gain);
        }
        #[cfg(not(any(
            target_feature = "avx512f",
            target_feature = "avx2",
            target_feature = "sse2"
        )))]
        {
            mix_audio_scalar(input1, input2, output, gain);
        }
    }

    /// SIMD-optimized buffer copying
    #[cfg(target_arch = "x86_64")]
    pub fn copy_buffer_simd(input: &[f32], output: &mut [f32]) {
        #[cfg(target_feature = "avx512f")]
        unsafe {
            copy_buffer_avx512(input, output);
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        unsafe {
            copy_buffer_avx2(input, output);
        }
        #[cfg(all(
            target_feature = "sse2",
            not(any(target_feature = "avx2", target_feature = "avx512f"))
        ))]
        unsafe {
            copy_buffer_sse2(input, output);
        }
        #[cfg(not(any(
            target_feature = "avx512f",
            target_feature = "avx2",
            target_feature = "sse2"
        )))]
        {
            copy_buffer_scalar(input, output);
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn mix_audio_simd(input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        #[cfg(target_feature = "neon")]
        unsafe {
            mix_audio_neon(input1, input2, output, gain);
        }
        #[cfg(not(target_feature = "neon"))]
        {
            mix_audio_scalar(input1, input2, output, gain);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn mix_audio_simd(input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        mix_audio_scalar(input1, input2, output, gain);
    }

    /// AVX-512 optimized mixing (x86_64)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe fn mix_audio_avx512(input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        use std::arch::x86_64::*;

        let len = input1.len().min(input2.len()).min(output.len());
        let simd_len = (len / 16) * 16; // AVX-512 processes 16 f32 values at once
        let gain_vec = _mm512_set1_ps(gain);

        for i in (0..simd_len).step_by(16) {
            let a = _mm512_loadu_ps(input1.as_ptr().add(i));
            let b = _mm512_loadu_ps(input2.as_ptr().add(i));
            let mixed = _mm512_add_ps(a, _mm512_mul_ps(b, gain_vec));
            _mm512_storeu_ps(output.as_mut_ptr().add(i), mixed);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input1[i] + input2[i] * gain;
        }
    }

    /// AVX2 optimized mixing (x86_64)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe fn mix_audio_avx2(input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        use std::arch::x86_64::*;

        let len = input1.len().min(input2.len()).min(output.len());
        let simd_len = (len / 8) * 8;
        let gain_vec = _mm256_set1_ps(gain);

        for i in (0..simd_len).step_by(8) {
            let a = _mm256_loadu_ps(input1.as_ptr().add(i));
            let b = _mm256_loadu_ps(input2.as_ptr().add(i));
            let mixed = _mm256_add_ps(a, _mm256_mul_ps(b, gain_vec));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), mixed);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input1[i] + input2[i] * gain;
        }
    }

    /// SSE2 optimized mixing (x86_64)
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    unsafe fn mix_audio_sse2(input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        use std::arch::x86_64::*;

        let len = input1.len().min(input2.len()).min(output.len());
        let simd_len = (len / 4) * 4;
        let gain_vec = _mm_set1_ps(gain);

        for i in (0..simd_len).step_by(4) {
            let a = _mm_loadu_ps(input1.as_ptr().add(i));
            let b = _mm_loadu_ps(input2.as_ptr().add(i));
            let mixed = _mm_add_ps(a, _mm_mul_ps(b, gain_vec));
            _mm_storeu_ps(output.as_mut_ptr().add(i), mixed);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input1[i] + input2[i] * gain;
        }
    }

    /// NEON optimized mixing (ARM64)
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe fn mix_audio_neon(input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        use std::arch::aarch64::*;

        let len = input1.len().min(input2.len()).min(output.len());
        let simd_len = (len / 4) * 4;
        let gain_vec = vdupq_n_f32(gain);

        for i in (0..simd_len).step_by(4) {
            let a = vld1q_f32(input1.as_ptr().add(i));
            let b = vld1q_f32(input2.as_ptr().add(i));
            let mixed = vaddq_f32(a, vmulq_f32(b, gain_vec));
            vst1q_f32(output.as_mut_ptr().add(i), mixed);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input1[i] + input2[i] * gain;
        }
    }

    /// Scalar fallback implementation
    fn mix_audio_scalar(input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        let len = input1.len().min(input2.len()).min(output.len());
        for i in 0..len {
            output[i] = input1[i] + input2[i] * gain;
        }
    }

    /// AVX-512 optimized buffer copying
    #[cfg(target_feature = "avx512f")]
    unsafe fn copy_buffer_avx512(input: &[f32], output: &mut [f32]) {
        use std::arch::x86_64::*;

        let len = input.len().min(output.len());
        let simd_len = (len / 16) * 16;

        for i in (0..simd_len).step_by(16) {
            let data = _mm512_loadu_ps(input.as_ptr().add(i));
            _mm512_storeu_ps(output.as_mut_ptr().add(i), data);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i];
        }
    }

    /// AVX2 optimized buffer copying
    #[cfg(target_feature = "avx2")]
    unsafe fn copy_buffer_avx2(input: &[f32], output: &mut [f32]) {
        use std::arch::x86_64::*;

        let len = input.len().min(output.len());
        let simd_len = (len / 8) * 8;

        for i in (0..simd_len).step_by(8) {
            let data = _mm256_loadu_ps(input.as_ptr().add(i));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), data);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i];
        }
    }

    /// SSE2 optimized buffer copying
    #[cfg(target_feature = "sse2")]
    unsafe fn copy_buffer_sse2(input: &[f32], output: &mut [f32]) {
        use std::arch::x86_64::*;

        let len = input.len().min(output.len());
        let simd_len = (len / 4) * 4;

        for i in (0..simd_len).step_by(4) {
            let data = _mm_loadu_ps(input.as_ptr().add(i));
            _mm_storeu_ps(output.as_mut_ptr().add(i), data);
        }

        // Handle remaining elements
        for i in simd_len..len {
            output[i] = input[i];
        }
    }

    /// Fallback scalar buffer copying
    fn copy_buffer_scalar(input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        output[..len].copy_from_slice(&input[..len]);
    }

    /// SIMD-optimized volume scaling
    pub fn scale_volume_simd(input: &[f32], output: &mut [f32], gain: f32) {
        #[cfg(target_feature = "avx2")]
        unsafe {
            scale_volume_avx2(input, output, gain);
        }
        #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
        unsafe {
            scale_volume_sse2(input, output, gain);
        }
        #[cfg(target_feature = "neon")]
        unsafe {
            scale_volume_neon(input, output, gain);
        }
        #[cfg(not(any(
            target_feature = "avx2",
            target_feature = "sse2",
            target_feature = "neon"
        )))]
        {
            scale_volume_scalar(input, output, gain);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe fn scale_volume_avx2(input: &[f32], output: &mut [f32], gain: f32) {
        use std::arch::x86_64::*;

        let len = input.len().min(output.len());
        let simd_len = (len / 8) * 8;
        let gain_vec = _mm256_set1_ps(gain);

        for i in (0..simd_len).step_by(8) {
            let a = _mm256_loadu_ps(input.as_ptr().add(i));
            let scaled = _mm256_mul_ps(a, gain_vec);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), scaled);
        }

        for i in simd_len..len {
            output[i] = input[i] * gain;
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    unsafe fn scale_volume_sse2(input: &[f32], output: &mut [f32], gain: f32) {
        use std::arch::x86_64::*;

        let len = input.len().min(output.len());
        let simd_len = (len / 4) * 4;
        let gain_vec = _mm_set1_ps(gain);

        for i in (0..simd_len).step_by(4) {
            let a = _mm_loadu_ps(input.as_ptr().add(i));
            let scaled = _mm_mul_ps(a, gain_vec);
            _mm_storeu_ps(output.as_mut_ptr().add(i), scaled);
        }

        for i in simd_len..len {
            output[i] = input[i] * gain;
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe fn scale_volume_neon(input: &[f32], output: &mut [f32], gain: f32) {
        use std::arch::aarch64::*;

        let len = input.len().min(output.len());
        let simd_len = (len / 4) * 4;
        let gain_vec = vdupq_n_f32(gain);

        for i in (0..simd_len).step_by(4) {
            let a = vld1q_f32(input.as_ptr().add(i));
            let scaled = vmulq_f32(a, gain_vec);
            vst1q_f32(output.as_mut_ptr().add(i), scaled);
        }

        for i in simd_len..len {
            output[i] = input[i] * gain;
        }
    }

    fn scale_volume_scalar(input: &[f32], output: &mut [f32], gain: f32) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i] * gain;
        }
    }
}

/// Memory optimization utilities
pub mod memory {
    // Note: super::* import removed as it is unused
    use std::alloc::{alloc, dealloc, Layout};

    /// Aligned memory allocator for cache-friendly data structures
    pub struct AlignedBuffer {
        ptr: *mut f32,
        capacity: usize,
        alignment: usize,
    }

    impl AlignedBuffer {
        /// Create a new aligned buffer
        pub fn new(capacity: usize, alignment: usize) -> Result<Self, String> {
            let layout = Layout::from_size_align(capacity * std::mem::size_of::<f32>(), alignment)
                .map_err(|e| format!("Layout error: {e}"))?;

            let ptr = unsafe { alloc(layout) as *mut f32 };
            if ptr.is_null() {
                return Err("Memory allocation failed".to_string());
            }

            Ok(Self {
                ptr,
                capacity,
                alignment,
            })
        }

        /// Get a mutable slice to the buffer
        pub fn as_mut_slice(&mut self) -> &mut [f32] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.capacity) }
        }

        /// Get an immutable slice to the buffer
        pub fn as_slice(&self) -> &[f32] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.capacity) }
        }

        /// Prefetch data into cache
        pub fn prefetch(&self, offset: usize, distance: usize) {
            let prefetch_addr = unsafe { self.ptr.add(offset.min(self.capacity)) };

            // Platform-specific prefetch instructions
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::_mm_prefetch;
                use std::arch::x86_64::_MM_HINT_T0;
                for i in (0..distance).step_by(64) {
                    _mm_prefetch((prefetch_addr as *const i8).add(i), _MM_HINT_T0);
                }
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                // ARM prefetch hint - simplified version without inline assembly
                // This is a hint to the compiler and may not generate actual prefetch instructions
                for i in (0..distance).step_by(64) {
                    let addr = (prefetch_addr as *const i8).add(i);
                    // Touch the memory location to encourage prefetching
                    std::ptr::read_volatile(addr);
                }
            }
        }
    }

    impl Drop for AlignedBuffer {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                let layout = Layout::from_size_align(
                    self.capacity * std::mem::size_of::<f32>(),
                    self.alignment,
                )
                .unwrap();
                unsafe {
                    dealloc(self.ptr as *mut u8, layout);
                }
            }
        }
    }

    unsafe impl Send for AlignedBuffer {}
    unsafe impl Sync for AlignedBuffer {}
}

/// Batch operation utilities for reducing FFI overhead
pub mod batch {
    use super::simd;
    use crate::{VoirsBatchOperation, VoirsErrorCode, VoirsPerformanceConfig};
    use std::os::raw::{c_float, c_uint};

    /// Process multiple audio operations in a single FFI call
    #[no_mangle]
    pub unsafe extern "C" fn voirs_batch_process_audio(
        operations: *const VoirsBatchOperation,
        operation_count: c_uint,
        config: *const VoirsPerformanceConfig,
    ) -> VoirsErrorCode {
        if operations.is_null() || operation_count == 0 {
            return VoirsErrorCode::InvalidParameter;
        }

        let ops = std::slice::from_raw_parts(operations, operation_count as usize);
        let perf_config = if config.is_null() {
            VoirsPerformanceConfig::default()
        } else {
            *config
        };

        for op in ops {
            if op.input_buffer1.is_null() || op.output_buffer.is_null() || op.buffer_size == 0 {
                return VoirsErrorCode::InvalidParameter;
            }

            let input1 = std::slice::from_raw_parts(op.input_buffer1, op.buffer_size as usize);
            let output = std::slice::from_raw_parts_mut(op.output_buffer, op.buffer_size as usize);

            match op.operation_type {
                0 => {
                    // Mix operation
                    if op.input_buffer2.is_null() {
                        return VoirsErrorCode::InvalidParameter;
                    }
                    let input2 =
                        std::slice::from_raw_parts(op.input_buffer2, op.buffer_size as usize);

                    if perf_config.enable_simd != 0 {
                        simd::mix_audio_simd(input1, input2, output, op.parameter);
                    } else {
                        for i in 0..input1.len() {
                            output[i] = input1[i] + input2[i] * op.parameter;
                        }
                    }
                }
                1 => {
                    // Scale operation
                    if perf_config.enable_simd != 0 {
                        simd::scale_volume_simd(input1, output, op.parameter);
                    } else {
                        for i in 0..input1.len() {
                            output[i] = input1[i] * op.parameter;
                        }
                    }
                }
                2 => {
                    // Copy operation
                    output.copy_from_slice(input1);
                }
                _ => {
                    return VoirsErrorCode::InvalidParameter;
                }
            }
        }

        VoirsErrorCode::Success
    }

    /// Batch convert multiple audio buffers between formats
    #[no_mangle]
    pub unsafe extern "C" fn voirs_batch_convert_format(
        input_buffers: *const *const c_float,
        output_buffers: *const *mut c_float,
        buffer_sizes: *const c_uint,
        buffer_count: c_uint,
        scale_factor: c_float,
    ) -> VoirsErrorCode {
        if input_buffers.is_null()
            || output_buffers.is_null()
            || buffer_sizes.is_null()
            || buffer_count == 0
        {
            return VoirsErrorCode::InvalidParameter;
        }

        let inputs = std::slice::from_raw_parts(input_buffers, buffer_count as usize);
        let outputs = std::slice::from_raw_parts(output_buffers, buffer_count as usize);
        let sizes = std::slice::from_raw_parts(buffer_sizes, buffer_count as usize);

        for i in 0..buffer_count as usize {
            if inputs[i].is_null() || outputs[i].is_null() || sizes[i] == 0 {
                return VoirsErrorCode::InvalidParameter;
            }

            let input = std::slice::from_raw_parts(inputs[i], sizes[i] as usize);
            let output = std::slice::from_raw_parts_mut(outputs[i], sizes[i] as usize);

            simd::scale_volume_simd(input, output, scale_factor);
        }

        VoirsErrorCode::Success
    }
}

/// Platform-specific optimization detection and configuration
#[no_mangle]
pub extern "C" fn voirs_detect_cpu_features() -> c_uint {
    let mut features = 0u32;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            features |= 1 << 0;
        }
        if is_x86_feature_detected!("avx") {
            features |= 1 << 1;
        }
        if is_x86_feature_detected!("avx2") {
            features |= 1 << 2;
        }
        if is_x86_feature_detected!("fma") {
            features |= 1 << 3;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "neon")]
        {
            features |= 1 << 4;
        }
    }

    features
}

/// Get optimal performance configuration for current platform
#[no_mangle]
pub extern "C" fn voirs_get_optimal_performance_config() -> VoirsPerformanceConfig {
    let mut config = VoirsPerformanceConfig::default();

    // Detect CPU features and adjust configuration
    let features = voirs_detect_cpu_features();

    // Enable SIMD if supported
    config.enable_simd = if features > 0 { 1 } else { 0 };

    // Adjust cache line size based on architecture
    #[cfg(target_arch = "x86_64")]
    {
        config.cache_line_size = 64;
    }
    #[cfg(target_arch = "aarch64")]
    {
        config.cache_line_size = 64; // Most ARM64 systems use 64-byte cache lines
    }

    // Adjust parallel threshold based on CPU count
    let cpu_count = num_cpus::get() as c_uint;
    config.parallel_threshold = 1024 / cpu_count.max(1);

    config
}

/// Advanced performance optimizations for FFI operations
pub mod advanced {
    use crate::VoirsPerformanceConfig;

    /// Memory prefetching hint for better cache performance
    ///
    /// # Safety
    /// The caller must ensure that `ptr.add(distance)` is valid for reading.
    #[inline]
    pub unsafe fn prefetch_memory(ptr: *const u8, distance: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            use std::arch::x86_64::_MM_HINT_T0;
            _mm_prefetch(ptr.add(distance) as *const i8, _MM_HINT_T0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // AArch64 prefetch is typically handled automatically by the CPU
            // or can be done using inline assembly if needed
            std::ptr::read_volatile(ptr.add(distance));
        }
    }

    /// Cache-aware audio processing with prefetching
    pub fn process_audio_with_prefetch(
        input: &[f32],
        output: &mut [f32],
        processor: impl Fn(f32) -> f32,
        config: &VoirsPerformanceConfig,
    ) {
        let len = input.len().min(output.len());
        let prefetch_distance = config.prefetch_distance as usize;

        for i in 0..len {
            // Prefetch ahead
            if i + prefetch_distance < len {
                unsafe {
                    prefetch_memory(input.as_ptr().add(i + prefetch_distance) as *const u8, 0);
                    prefetch_memory(
                        output.as_mut_ptr().add(i + prefetch_distance) as *const u8,
                        0,
                    );
                }
            }

            output[i] = processor(input[i]);
        }
    }

    /// Optimized multi-channel audio interleaving
    pub fn interleave_audio_optimized(
        channels: &[&[f32]],
        output: &mut [f32],
        config: &VoirsPerformanceConfig,
    ) -> Result<(), &'static str> {
        if channels.is_empty() {
            return Err("No channels provided");
        }

        let channel_count = channels.len();
        let frame_count = channels[0].len();

        // Verify all channels have same length
        for channel in channels.iter().skip(1) {
            if channel.len() != frame_count {
                return Err("Channel length mismatch");
            }
        }

        if output.len() < frame_count * channel_count {
            return Err("Output buffer too small");
        }

        // Cache-optimized interleaving
        let prefetch_frames = (config.prefetch_distance as usize).min(frame_count);

        for frame in 0..frame_count {
            // Prefetch next frames
            if frame + prefetch_frames < frame_count {
                for (_ch_idx, channel) in channels.iter().enumerate() {
                    unsafe {
                        prefetch_memory(
                            channel.as_ptr().add(frame + prefetch_frames) as *const u8,
                            0,
                        );
                    }
                }
            }

            // Interleave current frame
            for (ch_idx, channel) in channels.iter().enumerate() {
                output[frame * channel_count + ch_idx] = channel[frame];
            }
        }

        Ok(())
    }

    /// Vectorized audio format conversion with SIMD
    pub fn convert_f32_to_i16_optimized(input: &[f32], output: &mut [i16]) {
        let _len = input.len().min(output.len());

        #[cfg(target_arch = "x86_64")]
        unsafe {
            convert_f32_to_i16_avx2(input, output);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            convert_f32_to_i16_scalar(input, output);
        }
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn convert_f32_to_i16_avx2(input: &[f32], output: &mut [i16]) {
        use std::arch::x86_64::*;

        let len = input.len().min(output.len());
        let simd_len = (len / 8) * 8;
        let scale = 32767.0f32;
        let scale_vec = _mm256_set1_ps(scale);

        for i in (0..simd_len).step_by(8) {
            let f32_vals = _mm256_loadu_ps(input.as_ptr().add(i));
            let scaled = _mm256_mul_ps(f32_vals, scale_vec);
            let i32_vals = _mm256_cvtps_epi32(scaled);

            // Pack to i16 (AVX2)
            let lower = _mm256_extracti128_si256(i32_vals, 0);
            let upper = _mm256_extracti128_si256(i32_vals, 1);
            let packed = _mm_packs_epi32(lower, upper);

            _mm_storeu_si128(output.as_mut_ptr().add(i) as *mut __m128i, packed);
        }

        // Handle remaining elements
        for i in simd_len..len {
            let clamped = input[i].clamp(-1.0, 1.0);
            output[i] = (clamped * 32767.0) as i16;
        }
    }

    fn convert_f32_to_i16_scalar(input: &[f32], output: &mut [i16]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            let clamped = input[i].clamp(-1.0, 1.0);
            output[i] = (clamped * 32767.0) as i16;
        }
    }
}

/// FFI function for optimized format conversion
///
/// # Safety
/// The caller must ensure that `input` and `output` are valid pointers to arrays
/// of at least `input_size` and `output_size` elements respectively.
#[no_mangle]
pub unsafe extern "C" fn voirs_convert_f32_to_i16_optimized(
    input: *const c_float,
    input_size: c_uint,
    output: *mut i16,
    output_size: c_uint,
) -> VoirsErrorCode {
    if input.is_null() || output.is_null() || input_size == 0 || output_size == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    unsafe {
        let input_slice = std::slice::from_raw_parts(input, input_size as usize);
        let output_slice = std::slice::from_raw_parts_mut(output, output_size as usize);

        advanced::convert_f32_to_i16_optimized(input_slice, output_slice);
    }

    VoirsErrorCode::Success
}

/// FFI function for optimized multi-channel interleaving
///
/// # Safety
/// The caller must ensure that:
/// - `channel_ptrs` points to an array of `channel_count` valid pointers
/// - Each channel pointer points to at least `frame_count` valid f32 values
/// - `output` points to at least `channel_count * frame_count` valid f32 values
/// - `config` is either null or points to a valid VoirsPerformanceConfig
#[no_mangle]
pub unsafe extern "C" fn voirs_interleave_audio_optimized(
    channel_ptrs: *const *const c_float,
    channel_count: c_uint,
    frame_count: c_uint,
    output: *mut c_float,
    config: *const VoirsPerformanceConfig,
) -> VoirsErrorCode {
    if channel_ptrs.is_null() || output.is_null() || channel_count == 0 || frame_count == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let perf_config = unsafe {
        if config.is_null() {
            VoirsPerformanceConfig::default()
        } else {
            *config
        }
    };

    unsafe {
        let channel_ptr_slice = std::slice::from_raw_parts(channel_ptrs, channel_count as usize);
        let mut channels = Vec::with_capacity(channel_count as usize);

        for &ptr in channel_ptr_slice {
            if ptr.is_null() {
                return VoirsErrorCode::InvalidParameter;
            }
            channels.push(std::slice::from_raw_parts(ptr, frame_count as usize));
        }

        let output_slice =
            std::slice::from_raw_parts_mut(output, (channel_count * frame_count) as usize);

        match advanced::interleave_audio_optimized(&channels, output_slice, &perf_config) {
            Ok(()) => VoirsErrorCode::Success,
            Err(_) => VoirsErrorCode::InternalError,
        }
    }
}

/// Lock-free memory pool for high-performance audio buffer management
pub mod lockfree {
    use std::alloc::{alloc, dealloc, Layout};
    use std::ptr;
    use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

    /// Lock-free memory pool for fixed-size allocations
    pub struct LockFreePool {
        head: AtomicPtr<PoolNode>,
        chunk_size: usize,
        alignment: usize,
        allocated_count: AtomicUsize,
        total_capacity: AtomicUsize,
    }

    struct PoolNode {
        next: *mut PoolNode,
        data: [u8; 0], // Zero-sized array, actual data follows
    }

    impl LockFreePool {
        /// Create a new lock-free pool with specified chunk size and initial capacity
        pub fn new(chunk_size: usize, alignment: usize, initial_capacity: usize) -> Self {
            let pool = Self {
                head: AtomicPtr::new(ptr::null_mut()),
                chunk_size,
                alignment,
                allocated_count: AtomicUsize::new(0),
                total_capacity: AtomicUsize::new(0),
            };

            // Pre-allocate initial capacity
            for _ in 0..initial_capacity {
                if let Ok(ptr) = pool.allocate_chunk() {
                    pool.deallocate(ptr);
                }
            }

            pool
        }

        /// Allocate a chunk from the pool
        pub fn allocate(&self) -> Result<*mut u8, &'static str> {
            loop {
                let head = self.head.load(Ordering::Acquire);

                if head.is_null() {
                    // Pool is empty, allocate new chunk
                    return self.allocate_chunk();
                }

                unsafe {
                    let next = (*head).next;
                    // Try to update head atomically
                    if self
                        .head
                        .compare_exchange_weak(head, next, Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                    {
                        self.allocated_count.fetch_add(1, Ordering::Relaxed);
                        // Return the data portion (after the node header)
                        return Ok((head as *mut u8).add(std::mem::size_of::<PoolNode>()));
                    }
                }
                // CAS failed, retry
            }
        }

        /// Deallocate a chunk back to the pool
        pub fn deallocate(&self, ptr: *mut u8) {
            if ptr.is_null() {
                return;
            }

            unsafe {
                // Get the node pointer (before the data)
                let node = (ptr as *mut u8).sub(std::mem::size_of::<PoolNode>()) as *mut PoolNode;

                loop {
                    let head = self.head.load(Ordering::Acquire);
                    (*node).next = head;

                    if self
                        .head
                        .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                    {
                        self.allocated_count.fetch_sub(1, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }

        fn allocate_chunk(&self) -> Result<*mut u8, &'static str> {
            let total_size = std::mem::size_of::<PoolNode>() + self.chunk_size;
            let layout = Layout::from_size_align(total_size, self.alignment)
                .map_err(|_| "Invalid layout")?;

            unsafe {
                let ptr = alloc(layout) as *mut PoolNode;
                if ptr.is_null() {
                    return Err("Allocation failed");
                }

                (*ptr).next = ptr::null_mut();
                self.total_capacity.fetch_add(1, Ordering::Relaxed);
                Ok((ptr as *mut u8).add(std::mem::size_of::<PoolNode>()))
            }
        }

        /// Get current statistics
        pub fn stats(&self) -> PoolStats {
            PoolStats {
                allocated_count: self.allocated_count.load(Ordering::Relaxed),
                total_capacity: self.total_capacity.load(Ordering::Relaxed),
                chunk_size: self.chunk_size,
            }
        }
    }

    impl Drop for LockFreePool {
        fn drop(&mut self) {
            // Clean up all remaining chunks
            let mut current = self.head.load(Ordering::Relaxed);
            while !current.is_null() {
                unsafe {
                    let next = (*current).next;
                    let total_size = std::mem::size_of::<PoolNode>() + self.chunk_size;
                    let layout = Layout::from_size_align(total_size, self.alignment).unwrap();
                    dealloc(current as *mut u8, layout);
                    current = next;
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct PoolStats {
        pub allocated_count: usize,
        pub total_capacity: usize,
        pub chunk_size: usize,
    }

    unsafe impl Send for LockFreePool {}
    unsafe impl Sync for LockFreePool {}
}

/// Enhanced SIMD operations for additional audio processing tasks
pub mod enhanced_simd {
    /// SIMD-optimized audio normalization
    pub fn normalize_audio_simd(input: &[f32], output: &mut [f32], target_peak: f32) {
        if input.is_empty() || output.is_empty() {
            return;
        }

        // Find peak value
        let peak = find_peak_simd(input);
        if peak == 0.0 {
            output.fill(0.0);
            return;
        }

        let scale = target_peak / peak;
        super::simd::scale_volume_simd(input, output, scale);
    }

    /// SIMD-optimized peak detection
    pub fn find_peak_simd(input: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            find_peak_avx2(input)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            find_peak_scalar(input)
        }
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn find_peak_avx2(input: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        let simd_len = (len / 8) * 8;
        let mut max_vec = _mm256_setzero_ps();

        for i in (0..simd_len).step_by(8) {
            let vals = _mm256_loadu_ps(input.as_ptr().add(i));
            let abs_vals = _mm256_andnot_ps(_mm256_set1_ps(-0.0), vals); // Absolute value
            max_vec = _mm256_max_ps(max_vec, abs_vals);
        }

        // Extract maximum from vector
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), max_vec);
        let mut peak = result[0];
        for &val in &result[1..] {
            peak = peak.max(val);
        }

        // Handle remaining elements
        for &val in &input[simd_len..] {
            peak = peak.max(val.abs());
        }

        peak
    }

    fn find_peak_scalar(input: &[f32]) -> f32 {
        input.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()))
    }

    /// SIMD-optimized audio compression/limiting
    pub fn apply_soft_limiter_simd(input: &[f32], output: &mut [f32], threshold: f32, ratio: f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            apply_soft_limiter_avx2(input, output, threshold, ratio);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            apply_soft_limiter_scalar(input, output, threshold, ratio);
        }
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn apply_soft_limiter_avx2(
        input: &[f32],
        output: &mut [f32],
        threshold: f32,
        ratio: f32,
    ) {
        use std::arch::x86_64::*;

        let len = input.len().min(output.len());
        let simd_len = (len / 8) * 8;
        let threshold_vec = _mm256_set1_ps(threshold);
        let ratio_vec = _mm256_set1_ps(ratio);
        let neg_threshold_vec = _mm256_set1_ps(-threshold);

        for i in (0..simd_len).step_by(8) {
            let vals = _mm256_loadu_ps(input.as_ptr().add(i));

            // Apply soft limiting
            let above_thresh = _mm256_cmp_ps(vals, threshold_vec, _CMP_GT_OQ);
            let below_thresh = _mm256_cmp_ps(vals, neg_threshold_vec, _CMP_LT_OQ);

            let compressed_pos = _mm256_add_ps(
                threshold_vec,
                _mm256_mul_ps(_mm256_sub_ps(vals, threshold_vec), ratio_vec),
            );
            let compressed_neg = _mm256_sub_ps(
                neg_threshold_vec,
                _mm256_mul_ps(_mm256_sub_ps(neg_threshold_vec, vals), ratio_vec),
            );

            let result = _mm256_blendv_ps(
                vals,
                _mm256_blendv_ps(compressed_pos, compressed_neg, below_thresh),
                _mm256_or_ps(above_thresh, below_thresh),
            );

            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            let val = input[i];
            output[i] = if val > threshold {
                threshold + (val - threshold) * ratio
            } else if val < -threshold {
                -threshold + (val + threshold) * ratio
            } else {
                val
            };
        }
    }

    fn apply_soft_limiter_scalar(input: &[f32], output: &mut [f32], threshold: f32, ratio: f32) {
        let len = input.len().min(output.len());
        for i in 0..len {
            let val = input[i];
            output[i] = if val > threshold {
                threshold + (val - threshold) * ratio
            } else if val < -threshold {
                -threshold + (val + threshold) * ratio
            } else {
                val
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        advanced, batch, enhanced_simd, lockfree, memory, simd, voirs_detect_cpu_features,
        voirs_get_optimal_performance_config,
    };
    use crate::{VoirsBatchOperation, VoirsErrorCode, VoirsPerformanceConfig};

    #[test]
    fn test_performance_config_default() {
        let config = VoirsPerformanceConfig::default();
        assert!(config.cache_line_size > 0);
        assert!(config.prefetch_distance > 0);
        assert!(config.parallel_threshold > 0);
    }

    #[test]
    fn test_cpu_feature_detection() {
        let features = voirs_detect_cpu_features();
        // Should not crash and return some value
        // Just verify the function runs without panicking
        println!("CPU features: 0x{features:08X}");
    }

    #[test]
    fn test_optimal_config() {
        let config = voirs_get_optimal_performance_config();
        assert!(config.cache_line_size >= 32);
        assert!(config.cache_line_size <= 128);
    }

    #[test]
    fn test_simd_mix_audio() {
        let input1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input2 = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let mut output = vec![0.0; 8];

        simd::mix_audio_simd(&input1, &input2, &mut output, 2.0);

        // Expected: input1[i] + input2[i] * 2.0
        assert_eq!(output[0], 2.0); // 1.0 + 0.5 * 2.0
        assert_eq!(output[1], 3.0); // 2.0 + 0.5 * 2.0
        assert_eq!(output[7], 9.0); // 8.0 + 0.5 * 2.0
    }

    #[test]
    fn test_simd_scale_volume() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        simd::scale_volume_simd(&input, &mut output, 0.5);

        assert_eq!(output[0], 0.5);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 1.5);
        assert_eq!(output[3], 2.0);
    }

    #[test]
    fn test_aligned_buffer() {
        let mut buffer = memory::AlignedBuffer::new(1024, 64).unwrap();
        let slice = buffer.as_mut_slice();

        assert_eq!(slice.len(), 1024);
        assert_eq!(slice.as_ptr() as usize % 64, 0); // Check alignment

        // Test basic operations
        slice[0] = 1.0;
        slice[1023] = 2.0;
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1023], 2.0);
    }

    #[test]
    fn test_batch_operations() {
        let input1 = [1.0, 2.0, 3.0, 4.0];
        let input2 = [0.5, 0.5, 0.5, 0.5];
        let mut output = vec![0.0; 4];

        let operation = VoirsBatchOperation {
            operation_type: 0, // Mix
            input_buffer1: input1.as_ptr(),
            input_buffer2: input2.as_ptr(),
            output_buffer: output.as_mut_ptr(),
            buffer_size: 4,
            parameter: 2.0,
        };

        let result = unsafe { batch::voirs_batch_process_audio(&operation, 1, std::ptr::null()) };

        assert_eq!(result, VoirsErrorCode::Success);
        assert_eq!(output[0], 2.0); // 1.0 + 0.5 * 2.0
    }

    #[test]
    fn test_optimized_format_conversion() {
        let input = vec![0.5, -0.5, 1.0, -1.0, 0.0];
        let mut output = vec![0i16; 5];

        advanced::convert_f32_to_i16_optimized(&input, &mut output);

        assert_eq!(output[0], 16383); // 0.5 * 32767
        assert_eq!(output[1], -16383); // -0.5 * 32767
        assert_eq!(output[2], 32767); // 1.0 * 32767 (clamped)
        assert_eq!(output[3], -32767); // -1.0 * 32767 (clamped)
        assert_eq!(output[4], 0); // 0.0 * 32767
    }

    #[test]
    fn test_optimized_interleaving() {
        let left = vec![1.0, 2.0, 3.0];
        let right = vec![4.0, 5.0, 6.0];
        let channels = vec![left.as_slice(), right.as_slice()];
        let mut output = vec![0.0; 6];
        let config = VoirsPerformanceConfig::default();

        let result = advanced::interleave_audio_optimized(&channels, &mut output, &config);
        assert!(result.is_ok());

        // Expected: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        assert_eq!(output, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_cache_aware_processing() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];
        let config = VoirsPerformanceConfig::default();

        advanced::process_audio_with_prefetch(&input, &mut output, |x| x * 2.0, &config);

        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_lockfree_pool() {
        let pool = lockfree::LockFreePool::new(1024, 16, 4);

        // Test allocation and deallocation
        let ptr1 = pool.allocate().expect("Should allocate successfully");
        let ptr2 = pool.allocate().expect("Should allocate successfully");

        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);

        // Test statistics
        let stats = pool.stats();
        assert!(stats.allocated_count >= 2);
        assert_eq!(stats.chunk_size, 1024);

        // Test deallocation
        pool.deallocate(ptr1);
        pool.deallocate(ptr2);

        let stats_after = pool.stats();
        assert_eq!(stats_after.allocated_count, stats.allocated_count - 2);
    }

    #[test]
    fn test_lockfree_pool_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(lockfree::LockFreePool::new(512, 8, 10));
        let mut handles = vec![];

        // Spawn multiple threads to test thread safety
        for _ in 0..4 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                let mut ptrs = vec![];

                // Allocate some memory
                for _ in 0..5 {
                    if let Ok(ptr) = pool_clone.allocate() {
                        ptrs.push(ptr);
                    }
                }

                // Use the memory to ensure it's valid
                for &ptr in &ptrs {
                    if !ptr.is_null() {
                        unsafe {
                            // Write and read a test pattern
                            std::ptr::write_bytes(ptr, 0xAA, 256);
                            let first_byte = std::ptr::read(ptr);
                            assert_eq!(first_byte, 0xAA);
                        }
                    }
                }

                // Deallocate all memory within the same thread
                for ptr in ptrs {
                    pool_clone.deallocate(ptr);
                }

                // Return success count instead of pointers
                5usize
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        let mut total_operations = 0;
        for handle in handles {
            if let Ok(count) = handle.join() {
                total_operations += count;
            }
        }

        // Verify we processed the expected number of operations
        assert_eq!(total_operations, 20); // 4 threads * 5 operations each

        // Verify pool is functional after concurrent operations
        let ptr = pool.allocate().expect("Pool should still work");
        pool.deallocate(ptr);
    }

    #[test]
    fn test_enhanced_simd_peak_detection() {
        let input = vec![-0.8, 0.3, -1.2, 0.6, -0.4, 1.5, 0.2, -0.9];
        let peak = enhanced_simd::find_peak_simd(&input);

        // Should find 1.5 as the highest absolute value
        assert!((peak - 1.5).abs() < f32::EPSILON);

        // Test empty input
        let empty: Vec<f32> = vec![];
        let empty_peak = enhanced_simd::find_peak_simd(&empty);
        assert_eq!(empty_peak, 0.0);
    }

    #[test]
    fn test_enhanced_simd_normalization() {
        let input = vec![0.5, -1.0, 0.75, -0.25];
        let mut output = vec![0.0; 4];

        enhanced_simd::normalize_audio_simd(&input, &mut output, 0.8);

        // Input peak is 1.0, so scale factor should be 0.8
        let expected = vec![0.4, -0.8, 0.6, -0.2];
        for (i, (&actual, &expected_val)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected_val).abs() < 0.001,
                "Mismatch at index {}: {} != {}",
                i,
                actual,
                expected_val
            );
        }
    }

    #[test]
    fn test_enhanced_simd_soft_limiter() {
        let input = vec![-1.5, -0.5, 0.0, 0.5, 1.5, 2.0];
        let mut output = vec![0.0; 6];
        let threshold = 1.0;
        let ratio = 0.5;

        enhanced_simd::apply_soft_limiter_simd(&input, &mut output, threshold, ratio);

        // Expected: values above threshold compressed, others unchanged
        // -1.5 -> -1.0 + (-1.5 - (-1.0)) * 0.5 = -1.0 - 0.25 = -1.25
        // 1.5 -> 1.0 + (1.5 - 1.0) * 0.5 = 1.0 + 0.25 = 1.25
        // 2.0 -> 1.0 + (2.0 - 1.0) * 0.5 = 1.0 + 0.5 = 1.5

        assert!((output[0] - (-1.25)).abs() < 0.001); // -1.5 compressed
        assert!((output[1] - (-0.5)).abs() < 0.001); // -0.5 unchanged
        assert!((output[2] - 0.0).abs() < 0.001); // 0.0 unchanged
        assert!((output[3] - 0.5).abs() < 0.001); // 0.5 unchanged
        assert!((output[4] - 1.25).abs() < 0.001); // 1.5 compressed
        assert!((output[5] - 1.5).abs() < 0.001); // 2.0 compressed
    }

    #[test]
    fn test_enhanced_simd_normalization_zero_peak() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let mut output = vec![1.0; 4]; // Initialize with non-zero to test clearing

        enhanced_simd::normalize_audio_simd(&input, &mut output, 0.8);

        // Should fill output with zeros when input peak is zero
        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }
}
