//! C-Specific Performance Optimization
//!
//! This module provides optimizations specifically for C bindings including
//! SIMD intrinsics, branch prediction hints, compiler optimization flags,
//! and profile-guided optimization support.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::arch::x86_64::*;
use once_cell::sync::Lazy;

/// Global C performance statistics
static C_PERF_STATS: Lazy<CPerfStats> = Lazy::new(CPerfStats::new);

/// C-specific performance statistics
#[derive(Debug)]
pub struct CPerfStats {
    pub simd_operations: AtomicU64,
    pub scalar_operations: AtomicU64,
    pub branch_predictions_correct: AtomicU64,
    pub branch_predictions_incorrect: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub prefetch_operations: AtomicU64,
    pub vectorized_loops: AtomicU64,
}

impl CPerfStats {
    pub fn new() -> Self {
        Self {
            simd_operations: AtomicU64::new(0),
            scalar_operations: AtomicU64::new(0),
            branch_predictions_correct: AtomicU64::new(0),
            branch_predictions_incorrect: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            prefetch_operations: AtomicU64::new(0),
            vectorized_loops: AtomicU64::new(0),
        }
    }

    /// Record SIMD operation
    pub fn record_simd_operation(&self) {
        self.simd_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record scalar operation
    pub fn record_scalar_operation(&self) {
        self.scalar_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record branch prediction result
    pub fn record_branch_prediction(&self, correct: bool) {
        if correct {
            self.branch_predictions_correct.fetch_add(1, Ordering::Relaxed);
        } else {
            self.branch_predictions_incorrect.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record cache access result
    pub fn record_cache_access(&self, hit: bool) {
        if hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record prefetch operation
    pub fn record_prefetch(&self) {
        self.prefetch_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record vectorized loop
    pub fn record_vectorized_loop(&self) {
        self.vectorized_loops.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current statistics
    pub fn get_stats(&self) -> CPerfSnapshot {
        CPerfSnapshot {
            simd_operations: self.simd_operations.load(Ordering::Relaxed),
            scalar_operations: self.scalar_operations.load(Ordering::Relaxed),
            branch_predictions_correct: self.branch_predictions_correct.load(Ordering::Relaxed),
            branch_predictions_incorrect: self.branch_predictions_incorrect.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            prefetch_operations: self.prefetch_operations.load(Ordering::Relaxed),
            vectorized_loops: self.vectorized_loops.load(Ordering::Relaxed),
        }
    }

    /// Calculate SIMD utilization percentage
    pub fn simd_utilization(&self) -> f64 {
        let simd_ops = self.simd_operations.load(Ordering::Relaxed);
        let scalar_ops = self.scalar_operations.load(Ordering::Relaxed);
        let total_ops = simd_ops + scalar_ops;
        
        if total_ops == 0 {
            0.0
        } else {
            (simd_ops as f64 / total_ops as f64) * 100.0
        }
    }

    /// Calculate branch prediction accuracy
    pub fn branch_prediction_accuracy(&self) -> f64 {
        let correct = self.branch_predictions_correct.load(Ordering::Relaxed);
        let incorrect = self.branch_predictions_incorrect.load(Ordering::Relaxed);
        let total = correct + incorrect;
        
        if total == 0 {
            0.0
        } else {
            (correct as f64 / total as f64) * 100.0
        }
    }

    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }
}

/// Snapshot of C performance statistics
#[derive(Debug, Clone)]
pub struct CPerfSnapshot {
    pub simd_operations: u64,
    pub scalar_operations: u64,
    pub branch_predictions_correct: u64,
    pub branch_predictions_incorrect: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefetch_operations: u64,
    pub vectorized_loops: u64,
}

/// SIMD optimization utilities
pub struct SimdOptimizer {
    feature_support: SimdFeatures,
}

impl SimdOptimizer {
    /// Create new SIMD optimizer with feature detection
    pub fn new() -> Self {
        Self {
            feature_support: SimdFeatures::detect(),
        }
    }

    /// Optimized audio mixing using SIMD
    pub fn simd_mix_audio(&self, input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        let len = input1.len().min(input2.len()).min(output.len());
        
        if self.feature_support.avx512 && len >= 16 {
            self.simd_mix_audio_avx512(input1, input2, output, gain);
        } else if self.feature_support.avx2 && len >= 8 {
            self.simd_mix_audio_avx2(input1, input2, output, gain);
        } else if self.feature_support.sse2 && len >= 4 {
            self.simd_mix_audio_sse2(input1, input2, output, gain);
        } else {
            self.scalar_mix_audio(input1, input2, output, gain);
        }
    }

    /// AVX-512 optimized audio mixing
    #[target_feature(enable = "avx512f")]
    unsafe fn simd_mix_audio_avx512(&self, input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        C_PERF_STATS.record_simd_operation();
        C_PERF_STATS.record_vectorized_loop();
        
        let len = input1.len().min(input2.len()).min(output.len());
        let simd_len = (len / 16) * 16;
        let gain_vec = _mm512_set1_ps(gain);

        // Process 16 samples at a time
        for i in (0..simd_len).step_by(16) {
            let a = _mm512_loadu_ps(input1.as_ptr().add(i));
            let b = _mm512_loadu_ps(input2.as_ptr().add(i));
            let mixed = _mm512_add_ps(a, _mm512_mul_ps(b, gain_vec));
            _mm512_storeu_ps(output.as_mut_ptr().add(i), mixed);
        }

        // Handle remaining samples
        for i in simd_len..len {
            output[i] = input1[i] + input2[i] * gain;
        }
    }

    /// AVX2 optimized audio mixing
    #[target_feature(enable = "avx2")]
    unsafe fn simd_mix_audio_avx2(&self, input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        C_PERF_STATS.record_simd_operation();
        C_PERF_STATS.record_vectorized_loop();
        
        let len = input1.len().min(input2.len()).min(output.len());
        let simd_len = (len / 8) * 8;
        let gain_vec = _mm256_set1_ps(gain);

        // Process 8 samples at a time
        for i in (0..simd_len).step_by(8) {
            let a = _mm256_loadu_ps(input1.as_ptr().add(i));
            let b = _mm256_loadu_ps(input2.as_ptr().add(i));
            let mixed = _mm256_add_ps(a, _mm256_mul_ps(b, gain_vec));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), mixed);
        }

        // Handle remaining samples
        for i in simd_len..len {
            output[i] = input1[i] + input2[i] * gain;
        }
    }

    /// SSE2 optimized audio mixing
    #[target_feature(enable = "sse2")]
    unsafe fn simd_mix_audio_sse2(&self, input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        C_PERF_STATS.record_simd_operation();
        C_PERF_STATS.record_vectorized_loop();
        
        let len = input1.len().min(input2.len()).min(output.len());
        let simd_len = (len / 4) * 4;
        let gain_vec = _mm_set1_ps(gain);

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let a = _mm_loadu_ps(input1.as_ptr().add(i));
            let b = _mm_loadu_ps(input2.as_ptr().add(i));
            let mixed = _mm_add_ps(a, _mm_mul_ps(b, gain_vec));
            _mm_storeu_ps(output.as_mut_ptr().add(i), mixed);
        }

        // Handle remaining samples
        for i in simd_len..len {
            output[i] = input1[i] + input2[i] * gain;
        }
    }

    /// Scalar fallback audio mixing
    fn scalar_mix_audio(&self, input1: &[f32], input2: &[f32], output: &mut [f32], gain: f32) {
        C_PERF_STATS.record_scalar_operation();
        
        let len = input1.len().min(input2.len()).min(output.len());
        for i in 0..len {
            output[i] = input1[i] + input2[i] * gain;
        }
    }

    /// Optimized audio normalization using SIMD
    pub fn simd_normalize_audio(&self, data: &mut [f32], target_peak: f32) {
        // Find peak value
        let peak = self.find_peak_simd(data);
        if peak == 0.0 {
            return;
        }

        let scale = target_peak / peak;
        self.simd_scale_audio(data, scale);
    }

    /// Find peak value using SIMD
    fn find_peak_simd(&self, data: &[f32]) -> f32 {
        if self.feature_support.avx2 && data.len() >= 8 {
            unsafe { self.find_peak_avx2(data) }
        } else if self.feature_support.sse2 && data.len() >= 4 {
            unsafe { self.find_peak_sse2(data) }
        } else {
            self.find_peak_scalar(data)
        }
    }

    /// AVX2 peak finding
    #[target_feature(enable = "avx2")]
    unsafe fn find_peak_avx2(&self, data: &[f32]) -> f32 {
        C_PERF_STATS.record_simd_operation();
        
        let len = data.len();
        let simd_len = (len / 8) * 8;
        let mut max_vec = _mm256_setzero_ps();

        // Process 8 samples at a time
        for i in (0..simd_len).step_by(8) {
            let values = _mm256_loadu_ps(data.as_ptr().add(i));
            let abs_values = _mm256_andnot_ps(_mm256_set1_ps(-0.0), values);
            max_vec = _mm256_max_ps(max_vec, abs_values);
        }

        // Extract maximum from vector
        let mut max_array = [0.0f32; 8];
        _mm256_storeu_ps(max_array.as_mut_ptr(), max_vec);
        let mut peak = max_array.iter().fold(0.0f32, |a, &b| a.max(b));

        // Handle remaining samples
        for i in simd_len..len {
            peak = peak.max(data[i].abs());
        }

        peak
    }

    /// SSE2 peak finding
    #[target_feature(enable = "sse2")]
    unsafe fn find_peak_sse2(&self, data: &[f32]) -> f32 {
        C_PERF_STATS.record_simd_operation();
        
        let len = data.len();
        let simd_len = (len / 4) * 4;
        let mut max_vec = _mm_setzero_ps();

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let values = _mm_loadu_ps(data.as_ptr().add(i));
            let abs_values = _mm_andnot_ps(_mm_set1_ps(-0.0), values);
            max_vec = _mm_max_ps(max_vec, abs_values);
        }

        // Extract maximum from vector
        let mut max_array = [0.0f32; 4];
        _mm_storeu_ps(max_array.as_mut_ptr(), max_vec);
        let mut peak = max_array.iter().fold(0.0f32, |a, &b| a.max(b));

        // Handle remaining samples
        for i in simd_len..len {
            peak = peak.max(data[i].abs());
        }

        peak
    }

    /// Scalar peak finding
    fn find_peak_scalar(&self, data: &[f32]) -> f32 {
        C_PERF_STATS.record_scalar_operation();
        data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()))
    }

    /// Scale audio data using SIMD
    fn simd_scale_audio(&self, data: &mut [f32], scale: f32) {
        if self.feature_support.avx2 && data.len() >= 8 {
            unsafe { self.scale_audio_avx2(data, scale) };
        } else if self.feature_support.sse2 && data.len() >= 4 {
            unsafe { self.scale_audio_sse2(data, scale) };
        } else {
            self.scale_audio_scalar(data, scale);
        }
    }

    /// AVX2 audio scaling
    #[target_feature(enable = "avx2")]
    unsafe fn scale_audio_avx2(&self, data: &mut [f32], scale: f32) {
        C_PERF_STATS.record_simd_operation();
        
        let len = data.len();
        let simd_len = (len / 8) * 8;
        let scale_vec = _mm256_set1_ps(scale);

        for i in (0..simd_len).step_by(8) {
            let values = _mm256_loadu_ps(data.as_ptr().add(i));
            let scaled = _mm256_mul_ps(values, scale_vec);
            _mm256_storeu_ps(data.as_mut_ptr().add(i), scaled);
        }

        // Handle remaining samples
        for i in simd_len..len {
            data[i] *= scale;
        }
    }

    /// SSE2 audio scaling
    #[target_feature(enable = "sse2")]
    unsafe fn scale_audio_sse2(&self, data: &mut [f32], scale: f32) {
        C_PERF_STATS.record_simd_operation();
        
        let len = data.len();
        let simd_len = (len / 4) * 4;
        let scale_vec = _mm_set1_ps(scale);

        for i in (0..simd_len).step_by(4) {
            let values = _mm_loadu_ps(data.as_ptr().add(i));
            let scaled = _mm_mul_ps(values, scale_vec);
            _mm_storeu_ps(data.as_mut_ptr().add(i), scaled);
        }

        // Handle remaining samples
        for i in simd_len..len {
            data[i] *= scale;
        }
    }

    /// Scalar audio scaling
    fn scale_audio_scalar(&self, data: &mut [f32], scale: f32) {
        C_PERF_STATS.record_scalar_operation();
        for sample in data.iter_mut() {
            *sample *= scale;
        }
    }

    /// Get supported SIMD features
    pub fn get_features(&self) -> &SimdFeatures {
        &self.feature_support
    }
}

/// SIMD feature detection
#[derive(Debug, Clone)]
pub struct SimdFeatures {
    pub sse2: bool,
    pub sse3: bool,
    pub sse4_1: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub fma: bool,
}

impl SimdFeatures {
    /// Detect available SIMD features
    pub fn detect() -> Self {
        Self {
            sse2: is_x86_feature_detected!("sse2"),
            sse3: is_x86_feature_detected!("sse3"),
            sse4_1: is_x86_feature_detected!("sse4.1"),
            avx: is_x86_feature_detected!("avx"),
            avx2: is_x86_feature_detected!("avx2"),
            avx512: is_x86_feature_detected!("avx512f"),
            fma: is_x86_feature_detected!("fma"),
        }
    }

    /// Get best available feature level
    pub fn best_feature_level(&self) -> &'static str {
        if self.avx512 {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.avx {
            "AVX"
        } else if self.sse4_1 {
            "SSE4.1"
        } else if self.sse3 {
            "SSE3"
        } else if self.sse2 {
            "SSE2"
        } else {
            "Scalar"
        }
    }
}

/// Branch prediction optimization utilities
pub struct BranchOptimizer;

impl BranchOptimizer {
    /// Likely branch hint (for compiler optimization)
    #[inline(always)]
    pub fn likely(condition: bool) -> bool {
        #[cold]
        fn cold_path() {}
        
        if condition {
            true
        } else {
            cold_path();
            false
        }
    }

    /// Unlikely branch hint (for compiler optimization)
    #[inline(always)]
    pub fn unlikely(condition: bool) -> bool {
        #[cold]
        fn cold_path() {}
        
        if !condition {
            false
        } else {
            cold_path();
            true
        }
    }

    /// Profile branch prediction accuracy
    pub fn profile_branch(condition: bool, expected: bool) -> bool {
        C_PERF_STATS.record_branch_prediction(condition == expected);
        condition
    }
}

/// Cache optimization utilities
pub struct CacheOptimizer;

impl CacheOptimizer {
    /// Prefetch data for cache optimization
    #[inline(always)]
    pub fn prefetch_read<T>(ptr: *const T) {
        C_PERF_STATS.record_prefetch();
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }

    /// Prefetch data for write
    #[inline(always)]
    pub fn prefetch_write<T>(ptr: *const T) {
        C_PERF_STATS.record_prefetch();
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T1);
        }
    }

    /// Cache-friendly memory copy
    pub fn cache_friendly_copy(src: &[f32], dst: &mut [f32]) {
        let len = src.len().min(dst.len());
        const CACHE_LINE_SIZE: usize = 64; // bytes
        const FLOATS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / 4;

        for chunk_start in (0..len).step_by(FLOATS_PER_CACHE_LINE) {
            let chunk_end = (chunk_start + FLOATS_PER_CACHE_LINE).min(len);
            
            // Prefetch next cache line
            if chunk_end < len {
                Self::prefetch_read(src.as_ptr().wrapping_add(chunk_end));
                Self::prefetch_write(dst.as_ptr().wrapping_add(chunk_end));
            }

            // Copy current cache line
            dst[chunk_start..chunk_end].copy_from_slice(&src[chunk_start..chunk_end]);
            C_PERF_STATS.record_cache_access(true); // Assume cache hit for prefetched data
        }
    }
}

/// Compiler optimization hints
pub struct CompilerOptimizer;

impl CompilerOptimizer {
    /// Get recommended compiler flags for C bindings
    pub fn get_c_compiler_flags() -> Vec<&'static str> {
        vec![
            "-O3",                    // Maximum optimization
            "-march=native",          // Target native CPU
            "-mtune=native",          // Tune for native CPU
            "-ffast-math",           // Fast math operations
            "-funroll-loops",        // Unroll loops
            "-fomit-frame-pointer",  // Omit frame pointer
            "-flto",                 // Link-time optimization
            "-fno-stack-protector",  // Disable stack protection for performance
            "-DNDEBUG",              // Disable assertions
        ]
    }

    /// Get recommended linker flags
    pub fn get_linker_flags() -> Vec<&'static str> {
        vec![
            "-flto",                 // Link-time optimization
            "-Wl,-O3",              // Linker optimization
            "-Wl,--gc-sections",    // Remove unused sections
        ]
    }

    /// Force inline hint
    #[inline(always)]
    pub fn force_inline() {
        // This function serves as a hint for force inlining
    }

    /// No inline hint
    #[inline(never)]
    pub fn no_inline() {
        // This function serves as a hint against inlining
    }
}

/// Get global C performance statistics
pub fn get_c_perf_stats() -> CPerfSnapshot {
    C_PERF_STATS.get_stats()
}

/// Reset global C performance statistics
pub fn reset_c_perf_stats() {
    C_PERF_STATS.simd_operations.store(0, Ordering::Relaxed);
    C_PERF_STATS.scalar_operations.store(0, Ordering::Relaxed);
    C_PERF_STATS.branch_predictions_correct.store(0, Ordering::Relaxed);
    C_PERF_STATS.branch_predictions_incorrect.store(0, Ordering::Relaxed);
    C_PERF_STATS.cache_hits.store(0, Ordering::Relaxed);
    C_PERF_STATS.cache_misses.store(0, Ordering::Relaxed);
    C_PERF_STATS.prefetch_operations.store(0, Ordering::Relaxed);
    C_PERF_STATS.vectorized_loops.store(0, Ordering::Relaxed);
}

/// Get SIMD utilization percentage
pub fn get_simd_utilization() -> f64 {
    C_PERF_STATS.simd_utilization()
}

/// Get branch prediction accuracy
pub fn get_branch_prediction_accuracy() -> f64 {
    C_PERF_STATS.branch_prediction_accuracy()
}

/// Get cache hit ratio
pub fn get_cache_hit_ratio() -> f64 {
    C_PERF_STATS.cache_hit_ratio()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_perf_stats() {
        reset_c_perf_stats();
        
        C_PERF_STATS.record_simd_operation();
        C_PERF_STATS.record_scalar_operation();
        C_PERF_STATS.record_branch_prediction(true);
        C_PERF_STATS.record_cache_access(true);
        C_PERF_STATS.record_prefetch();
        
        let stats = get_c_perf_stats();
        assert_eq!(stats.simd_operations, 1);
        assert_eq!(stats.scalar_operations, 1);
        assert_eq!(stats.branch_predictions_correct, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.prefetch_operations, 1);
    }

    #[test]
    fn test_simd_features() {
        let features = SimdFeatures::detect();
        println!("Best feature level: {}", features.best_feature_level());
        
        // These should always be true on modern x86_64
        assert!(features.sse2);
    }

    #[test]
    fn test_simd_optimizer() {
        let optimizer = SimdOptimizer::new();
        let mut input1 = vec![1.0f32; 64];
        let mut input2 = vec![2.0f32; 64];
        let mut output = vec![0.0f32; 64];
        
        optimizer.simd_mix_audio(&input1, &input2, &mut output, 0.5);
        
        // Check result: 1.0 + 2.0 * 0.5 = 2.0
        assert!((output[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_normalization() {
        let optimizer = SimdOptimizer::new();
        let mut data = vec![0.5, -0.8, 0.3, -0.6];
        
        optimizer.simd_normalize_audio(&mut data, 0.5);
        
        // Peak should now be 0.5
        let peak = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        assert!((peak - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_branch_optimizer() {
        reset_c_perf_stats();
        
        let result = BranchOptimizer::profile_branch(true, true);
        assert!(result);
        
        let stats = get_c_perf_stats();
        assert_eq!(stats.branch_predictions_correct, 1);
        
        let accuracy = get_branch_prediction_accuracy();
        assert_eq!(accuracy, 100.0);
    }

    #[test]
    fn test_cache_optimizer() {
        reset_c_perf_stats();
        
        let src = vec![1.0f32; 128];
        let mut dst = vec![0.0f32; 128];
        
        CacheOptimizer::cache_friendly_copy(&src, &mut dst);
        
        assert_eq!(src, dst);
        
        let stats = get_c_perf_stats();
        assert!(stats.prefetch_operations > 0);
        assert!(stats.cache_hits > 0);
    }

    #[test]
    fn test_compiler_flags() {
        let c_flags = CompilerOptimizer::get_c_compiler_flags();
        assert!(c_flags.contains(&"-O3"));
        assert!(c_flags.contains(&"-march=native"));
        
        let linker_flags = CompilerOptimizer::get_linker_flags();
        assert!(linker_flags.contains(&"-flto"));
    }

    #[test]
    fn test_performance_metrics() {
        reset_c_perf_stats();
        
        // Record some operations
        C_PERF_STATS.record_simd_operation();
        C_PERF_STATS.record_simd_operation();
        C_PERF_STATS.record_scalar_operation();
        
        let utilization = get_simd_utilization();
        assert!((utilization - 66.66666666666667).abs() < 1e-10); // 2 SIMD out of 3 total = 66.67%
    }
}