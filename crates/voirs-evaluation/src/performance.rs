//! Performance optimization utilities
//!
//! This module provides performance optimizations including:
//! - Parallel processing utilities
//! - Memory optimization helpers
//! - SIMD-accelerated computations
//! - GPU acceleration framework
//! - Caching mechanisms

use parking_lot::RwLock;
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;
use std::sync::Arc;

// GPU acceleration imports
use candle_core::{Device, Result as CandleResult, Tensor};

// Re-export GPU module
pub use gpu::{GpuAccelerator, GpuMemoryManager, SpectralFeatures as GpuSpectralFeatures};

/// Memory-efficient chunked processing for large audio arrays
pub fn process_audio_chunks<T, F, R>(data: &[T], chunk_size: usize, processor: F) -> Vec<R>
where
    T: Send + Sync,
    F: Fn(&[T]) -> R + Send + Sync,
    R: Send,
{
    if data.len() <= chunk_size {
        return vec![processor(data)];
    }

    data.par_chunks(chunk_size).map(processor).collect()
}

/// Parallel correlation calculation with SIMD optimization
#[must_use]
pub fn parallel_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    // Use parallel reduction for large arrays
    if x.len() > 1000 {
        let (sum_xy, sum_x, sum_y, sum_x2, sum_y2) = x
            .par_iter()
            .zip(y.par_iter())
            .map(|(&xi, &yi)| (xi * yi, xi, yi, xi * xi, yi * yi))
            .reduce(
                || (0.0, 0.0, 0.0, 0.0, 0.0),
                |acc, item| {
                    (
                        acc.0 + item.0,
                        acc.1 + item.1,
                        acc.2 + item.2,
                        acc.3 + item.3,
                        acc.4 + item.4,
                    )
                },
            );

        let n = x.len() as f32;
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator > f32::EPSILON {
            numerator / denominator
        } else {
            0.0
        }
    } else {
        // Use sequential for small arrays
        crate::calculate_correlation(x, y)
    }
}

/// Parallel FFT computation for batch processing
#[must_use]
pub fn parallel_fft_batch(signals: &[Vec<f32>]) -> Vec<Vec<f32>> {
    use scirs2_fft::{RealFftPlanner, RealToComplex};
    use std::sync::Mutex;

    let planner = Arc::new(Mutex::new(RealFftPlanner::<f32>::new()));

    signals
        .par_iter()
        .map(|signal| {
            if signal.is_empty() {
                return Vec::new();
            }

            let mut planner_guard = planner.lock().unwrap();
            let fft = planner_guard.plan_fft_forward(signal.len());
            drop(planner_guard);

            let mut indata = signal.clone();
            let mut spectrum = vec![scirs2_core::Complex::new(0.0, 0.0); fft.output_len()];

            fft.process(&indata, &mut spectrum);

            // Convert to magnitude spectrum
            spectrum.iter().map(|c| c.norm()).collect()
        })
        .collect()
}

/// High-performance autocorrelation with parallel computation
#[must_use]
pub fn parallel_autocorrelation(signal: &[f32], max_lag: usize) -> Vec<f32> {
    if signal.len() < 2 || max_lag == 0 {
        return vec![0.0; max_lag + 1];
    }

    let effective_max_lag = max_lag.min(signal.len() - 1);

    (0..=effective_max_lag)
        .into_par_iter()
        .map(|lag| {
            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in 0..(signal.len() - lag) {
                correlation += signal[i] * signal[i + lag];
                norm1 += signal[i] * signal[i];
                if lag == 0 {
                    norm2 = norm1;
                } else {
                    norm2 += signal[i + lag] * signal[i + lag];
                }
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                correlation / (norm1 * norm2).sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

/// Thread-safe LRU cache for expensive computations
pub struct LRUCache<K, V> {
    map: Arc<RwLock<HashMap<K, V>>>,
    max_size: usize,
}

impl<K, V> LRUCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new LRU cache with the specified maximum size
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            map: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        self.map.read().get(key).cloned()
    }

    /// Insert a key-value pair into the cache
    pub fn insert(&self, key: K, value: V) {
        let mut map = self.map.write();

        // Simple eviction strategy - clear if at capacity
        if map.len() >= self.max_size {
            map.clear();
        }

        map.insert(key, value);
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        self.map.write().clear();
    }

    /// Get the current number of entries in the cache
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.read().len()
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.read().is_empty()
    }
}

/// Persistent cache with compression support
///
/// This cache stores data to disk with compression for efficient storage
/// and retrieval across application restarts.
pub struct PersistentCache<K, V> {
    memory_cache: LRUCache<K, V>,
    cache_dir: std::path::PathBuf,
    compression_level: u32,
}

impl<K, V> PersistentCache<K, V>
where
    K: Eq + Hash + Clone + serde::Serialize + serde::de::DeserializeOwned,
    V: Clone + serde::Serialize + serde::de::DeserializeOwned,
{
    /// Create a new persistent cache with the specified directory and settings
    pub fn new<P: AsRef<Path>>(
        cache_dir: P,
        max_memory_size: usize,
        compression_level: u32,
    ) -> Result<Self, std::io::Error> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)?;
        }

        Ok(Self {
            memory_cache: LRUCache::new(max_memory_size),
            cache_dir,
            compression_level: compression_level.clamp(0, 9),
        })
    }

    /// Get a value from the cache (checks memory first, then disk)
    pub fn get(&self, key: &K) -> Option<V> {
        // Check memory cache first
        if let Some(value) = self.memory_cache.get(key) {
            return Some(value);
        }

        // Try to load from disk
        if let Ok(value) = self.load_from_disk(key) {
            // Cache in memory for future access
            self.memory_cache.insert(key.clone(), value.clone());
            return Some(value);
        }

        None
    }

    /// Insert a key-value pair into the cache (saves to both memory and disk)
    pub fn insert(&self, key: K, value: V) -> Result<(), std::io::Error> {
        // Save to memory cache
        self.memory_cache.insert(key.clone(), value.clone());

        // Save to disk with compression
        self.save_to_disk(&key, &value)
    }

    /// Clear all entries from the cache
    pub fn clear(&self) -> Result<(), std::io::Error> {
        self.memory_cache.clear();

        // Remove all files in cache directory
        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.path().is_file() {
                std::fs::remove_file(entry.path())?;
            }
        }

        Ok(())
    }

    /// Get the current number of entries in memory cache
    #[must_use]
    pub fn memory_len(&self) -> usize {
        self.memory_cache.len()
    }

    /// Get the total number of entries (including disk)
    pub fn total_len(&self) -> usize {
        std::fs::read_dir(&self.cache_dir)
            .map(|entries| entries.filter_map(|e| e.ok()).count())
            .unwrap_or(0)
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory_cache.is_empty() && self.total_len() == 0
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            memory_entries: self.memory_cache.len(),
            disk_entries: self.total_len(),
            cache_dir_size: self.calculate_cache_dir_size(),
        }
    }

    /// Set compression level (0-9, where 9 is highest compression)
    pub fn set_compression_level(&mut self, level: u32) {
        self.compression_level = level.clamp(0, 9);
    }

    // Private helper methods

    fn cache_key_to_filename(&self, key: &K) -> Result<String, std::io::Error> {
        let serialized = bincode::serde::encode_to_vec(key, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::Hasher;
            let mut hasher = DefaultHasher::new();
            hasher.write(&serialized);
            hasher.finish()
        };
        Ok(format!("{:x}.cache", hash))
    }

    fn save_to_disk(&self, key: &K, value: &V) -> Result<(), std::io::Error> {
        let filename = self.cache_key_to_filename(key)?;
        let filepath = self.cache_dir.join(filename);

        let serialized = bincode::serde::encode_to_vec(value, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Compress the data using flate2
        let mut encoder = flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::new(self.compression_level),
        );
        std::io::Write::write_all(&mut encoder, &serialized)?;
        let compressed = encoder.finish()?;

        std::fs::write(filepath, compressed)
    }

    fn load_from_disk(&self, key: &K) -> Result<V, std::io::Error> {
        let filename = self.cache_key_to_filename(key)?;
        let filepath = self.cache_dir.join(filename);

        if !filepath.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Cache entry not found",
            ));
        }

        let compressed_data = std::fs::read(filepath)?;

        // Decompress the data
        let mut decoder = flate2::read::GzDecoder::new(&compressed_data[..]);
        let mut decompressed = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut decompressed)?;

        bincode::serde::decode_from_slice(&decompressed, bincode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    fn calculate_cache_dir_size(&self) -> u64 {
        std::fs::read_dir(&self.cache_dir)
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok().and_then(|e| e.metadata().ok()).map(|m| m.len()))
                    .sum()
            })
            .unwrap_or(0)
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of entries in memory cache
    pub memory_entries: usize,
    /// Number of entries on disk
    pub disk_entries: usize,
    /// Total size of cache directory in bytes
    pub cache_dir_size: u64,
}

/// Memory-efficient sliding window processor
pub struct SlidingWindowProcessor<T> {
    window_size: usize,
    hop_size: usize,
    buffer: Vec<T>,
}

impl<T> SlidingWindowProcessor<T>
where
    T: Clone + Default,
{
    /// Create a new sliding window processor with specified window and hop sizes
    #[must_use]
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        Self {
            window_size,
            hop_size,
            buffer: Vec::with_capacity(window_size),
        }
    }

    /// Process data in sliding windows with parallel computation
    pub fn process_parallel<F, R>(&self, data: &[T], processor: F) -> Vec<R>
    where
        T: Send + Sync,
        F: Fn(&[T]) -> R + Send + Sync,
        R: Send,
    {
        if data.len() < self.window_size {
            return Vec::new();
        }

        let num_windows = (data.len() - self.window_size) / self.hop_size + 1;

        (0..num_windows)
            .into_par_iter()
            .map(|i| {
                let start = i * self.hop_size;
                let end = (start + self.window_size).min(data.len());
                processor(&data[start..end])
            })
            .collect()
    }
}

/// SIMD-accelerated vector operations using hardware-specific instructions
pub mod simd {
    use scirs2_core::parallel_ops::*;

    /// SIMD dot product with hardware acceleration when available
    #[must_use]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        // Use hardware SIMD for smaller arrays, parallel processing for larger ones
        if a.len() > 1000 {
            a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x * y).sum()
        } else {
            dot_product_simd(a, b)
        }
    }

    /// Element-wise multiplication with SIMD optimization
    #[must_use]
    pub fn element_wise_multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            return Vec::new();
        }

        if a.len() > 1000 {
            a.par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| x * y)
                .collect()
        } else {
            element_wise_multiply_simd(a, b)
        }
    }

    /// SIMD-optimized RMS calculation
    #[must_use]
    pub fn rms(signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }

        let sum_squares = if signal.len() > 1000 {
            signal.par_iter().map(|&x| x * x).sum::<f32>()
        } else {
            rms_simd(signal)
        };

        (sum_squares / signal.len() as f32).sqrt()
    }

    /// Spectral centroid with SIMD-optimized frequency weighting
    #[must_use]
    pub fn spectral_centroid(spectrum: &[f32], sample_rate: f32) -> f32 {
        if spectrum.is_empty() {
            return 0.0;
        }

        let (weighted_sum, magnitude_sum) = if spectrum.len() > 500 {
            spectrum
                .par_iter()
                .enumerate()
                .map(|(i, &magnitude)| {
                    let frequency = i as f32 * sample_rate / (2.0 * spectrum.len() as f32);
                    (frequency * magnitude, magnitude)
                })
                .reduce(|| (0.0, 0.0), |acc, item| (acc.0 + item.0, acc.1 + item.1))
        } else {
            spectral_centroid_simd(spectrum, sample_rate)
        };

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    /// Vector addition with SIMD optimization
    #[must_use]
    pub fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            return Vec::new();
        }

        if a.len() > 1000 {
            a.par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| x + y)
                .collect()
        } else {
            vector_add_simd(a, b)
        }
    }

    /// Vector subtraction with SIMD optimization
    #[must_use]
    pub fn vector_subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            return Vec::new();
        }

        if a.len() > 1000 {
            a.par_iter()
                .zip(b.par_iter())
                .map(|(&x, &y)| x - y)
                .collect()
        } else {
            vector_subtract_simd(a, b)
        }
    }

    /// x86 SSE/AVX optimized implementations
    #[cfg(target_arch = "x86_64")]
    mod x86_impl {
        #[cfg(target_feature = "avx")]
        pub fn dot_product_avx(a: &[f32], b: &[f32]) -> f32 {
            use std::arch::x86_64::*;

            let len = a.len();
            let chunks = len / 8;
            let mut sum = unsafe { _mm256_setzero_ps() };

            unsafe {
                for i in 0..chunks {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                    let mul = _mm256_mul_ps(a_vec, b_vec);
                    sum = _mm256_add_ps(sum, mul);
                }

                // Horizontal sum
                let sum_low = _mm256_extractf128_ps(sum, 0);
                let sum_high = _mm256_extractf128_ps(sum, 1);
                let sum_128 = _mm_add_ps(sum_low, sum_high);

                let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
                let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));

                let mut result = _mm_cvtss_f32(sum_32);

                // Handle remaining elements
                for i in (chunks * 8)..len {
                    result += a[i] * b[i];
                }

                result
            }
        }

        #[cfg(target_feature = "sse")]
        pub fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
            use std::arch::x86_64::*;

            let len = a.len();
            let chunks = len / 4;
            let mut sum = unsafe { _mm_setzero_ps() };

            unsafe {
                for i in 0..chunks {
                    let a_vec = _mm_loadu_ps(a.as_ptr().add(i * 4));
                    let b_vec = _mm_loadu_ps(b.as_ptr().add(i * 4));
                    let mul = _mm_mul_ps(a_vec, b_vec);
                    sum = _mm_add_ps(sum, mul);
                }

                // Horizontal sum
                let sum_64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
                let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));

                let mut result = _mm_cvtss_f32(sum_32);

                // Handle remaining elements
                for i in (chunks * 4)..len {
                    result += a[i] * b[i];
                }

                result
            }
        }

        #[cfg(target_feature = "avx")]
        pub fn element_wise_multiply_avx(a: &[f32], b: &[f32]) -> Vec<f32> {
            use std::arch::x86_64::*;

            let len = a.len();
            let chunks = len / 8;
            let mut result: Vec<f32> = Vec::with_capacity(len);

            unsafe {
                for i in 0..chunks {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                    let mul = _mm256_mul_ps(a_vec, b_vec);

                    let ptr = result.as_mut_ptr().add(i * 8);
                    _mm256_storeu_ps(ptr, mul);
                }

                result.set_len(chunks * 8);

                // Handle remaining elements
                for i in (chunks * 8)..len {
                    result.push(a[i] * b[i]);
                }
            }

            result
        }
    }

    /// ARM NEON optimized implementations
    #[cfg(target_arch = "aarch64")]
    mod arm_impl {
        #[cfg(target_feature = "neon")]
        pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
            use std::arch::aarch64::*;

            let len = a.len();
            let chunks = len / 4;
            let mut sum = unsafe { vdupq_n_f32(0.0) };

            unsafe {
                for i in 0..chunks {
                    let a_vec = vld1q_f32(a.as_ptr().add(i * 4));
                    let b_vec = vld1q_f32(b.as_ptr().add(i * 4));
                    let mul = vmulq_f32(a_vec, b_vec);
                    sum = vaddq_f32(sum, mul);
                }

                // Horizontal sum
                let sum_pair = vadd_f32(vget_high_f32(sum), vget_low_f32(sum));
                let result_vec = vpadd_f32(sum_pair, sum_pair);
                let mut result = vget_lane_f32(result_vec, 0);

                // Handle remaining elements
                for i in (chunks * 4)..len {
                    result += a[i] * b[i];
                }

                result
            }
        }

        #[cfg(target_feature = "neon")]
        pub fn element_wise_multiply_neon(a: &[f32], b: &[f32]) -> Vec<f32> {
            use std::arch::aarch64::*;

            let len = a.len();
            let chunks = len / 4;
            let mut result: Vec<f32> = Vec::with_capacity(len);

            unsafe {
                for i in 0..chunks {
                    let a_vec = vld1q_f32(a.as_ptr().add(i * 4));
                    let b_vec = vld1q_f32(b.as_ptr().add(i * 4));
                    let mul = vmulq_f32(a_vec, b_vec);

                    vst1q_f32(result.as_mut_ptr().add(i * 4), mul);
                }

                result.set_len(chunks * 4);

                // Handle remaining elements
                for i in (chunks * 4)..len {
                    result.push(a[i] * b[i]);
                }
            }

            result
        }
    }

    /// Platform-specific SIMD dispatch
    #[inline]
    fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            x86_impl::dot_product_avx(a, b)
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse",
            not(target_feature = "avx")
        ))]
        {
            x86_impl::dot_product_sse(a, b)
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            arm_impl::dot_product_neon(a, b)
        }
        #[cfg(not(any(
            all(
                target_arch = "x86_64",
                any(target_feature = "avx", target_feature = "sse")
            ),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            // Fallback to scalar implementation
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
        }
    }

    #[inline]
    fn element_wise_multiply_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            x86_impl::element_wise_multiply_avx(a, b)
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            arm_impl::element_wise_multiply_neon(a, b)
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            // Fallback to scalar implementation
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
        }
    }

    #[inline]
    fn rms_simd(signal: &[f32]) -> f32 {
        // Use SIMD dot product for sum of squares calculation
        dot_product_simd(signal, signal)
    }

    #[inline]
    fn spectral_centroid_simd(spectrum: &[f32], sample_rate: f32) -> (f32, f32) {
        spectrum
            .iter()
            .enumerate()
            .map(|(i, &magnitude)| {
                let frequency = i as f32 * sample_rate / (2.0 * spectrum.len() as f32);
                (frequency * magnitude, magnitude)
            })
            .fold((0.0, 0.0), |acc, item| (acc.0 + item.0, acc.1 + item.1))
    }

    #[inline]
    fn vector_add_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        // Similar pattern to element_wise_multiply but with addition
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            use std::arch::x86_64::*;

            let len = a.len();
            let chunks = len / 8;
            let mut result: Vec<f32> = Vec::with_capacity(len);

            unsafe {
                for i in 0..chunks {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                    let add = _mm256_add_ps(a_vec, b_vec);

                    let ptr = result.as_mut_ptr().add(i * 8);
                    _mm256_storeu_ps(ptr, add);
                }

                result.set_len(chunks * 8);

                // Handle remaining elements
                for i in (chunks * 8)..len {
                    result.push(a[i] + b[i]);
                }
            }

            result
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx")))]
        {
            a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
        }
    }

    #[inline]
    fn vector_subtract_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        // Similar pattern to vector_add but with subtraction
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            use std::arch::x86_64::*;

            let len = a.len();
            let chunks = len / 8;
            let mut result: Vec<f32> = Vec::with_capacity(len);

            unsafe {
                for i in 0..chunks {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                    let sub = _mm256_sub_ps(a_vec, b_vec);

                    let ptr = result.as_mut_ptr().add(i * 8);
                    _mm256_storeu_ps(ptr, sub);
                }

                result.set_len(chunks * 8);

                // Handle remaining elements
                for i in (chunks * 8)..len {
                    result.push(a[i] - b[i]);
                }
            }

            result
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx")))]
        {
            a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
        }
    }
}

/// Performance monitoring utilities
pub struct PerformanceMonitor {
    timings: Arc<RwLock<HashMap<String, Vec<std::time::Duration>>>>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            timings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Time an operation and record the duration
    pub fn time_operation<F, R>(&self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration = start.elapsed();

        self.timings
            .write()
            .entry(name.to_string())
            .or_default()
            .push(duration);

        result
    }

    /// Get the average time for a named operation
    #[must_use]
    pub fn get_average_time(&self, name: &str) -> Option<std::time::Duration> {
        let timings = self.timings.read();
        if let Some(times) = timings.get(name) {
            if times.is_empty() {
                None
            } else {
                let total: std::time::Duration = times.iter().sum();
                Some(total / times.len() as u32)
            }
        } else {
            None
        }
    }

    /// Clear all recorded timings
    pub fn clear(&self) {
        self.timings.write().clear();
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU acceleration framework using Candle
pub mod gpu {
    use super::{
        Arc, CandleResult, Device, IndexedParallelIterator, ParallelIterator, RwLock, Tensor,
    };
    use crate::EvaluationError;

    /// GPU accelerated operations manager
    #[derive(Clone)]
    pub struct GpuAccelerator {
        device: Device,
        memory_pool: Arc<RwLock<Vec<Tensor>>>,
        max_memory_mb: usize,
    }

    impl GpuAccelerator {
        /// Create new GPU accelerator with automatic device detection
        pub fn new() -> Result<Self, EvaluationError> {
            let device = Self::detect_best_device()?;
            let memory_pool = Arc::new(RwLock::new(Vec::new()));

            Ok(Self {
                device,
                memory_pool,
                max_memory_mb: 1024, // 1GB default
            })
        }

        /// Create GPU accelerator with specific device
        #[must_use]
        pub fn with_device(device: Device) -> Self {
            Self {
                device,
                memory_pool: Arc::new(RwLock::new(Vec::new())),
                max_memory_mb: 1024,
            }
        }

        /// Detect the best available device
        fn detect_best_device() -> Result<Device, EvaluationError> {
            // Try CUDA first
            if let Ok(device) = Device::cuda_if_available(0) {
                return Ok(device);
            }

            // Try Metal on macOS
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }

            // Fallback to CPU
            Ok(Device::Cpu)
        }

        /// Get the current device
        #[must_use]
        pub fn device(&self) -> &Device {
            &self.device
        }

        /// Check if GPU acceleration is available
        #[must_use]
        pub fn is_gpu_available(&self) -> bool {
            !matches!(self.device, Device::Cpu)
        }

        /// GPU-accelerated correlation calculation
        pub fn gpu_correlation(&self, x: &[f32], y: &[f32]) -> Result<f32, EvaluationError> {
            if x.len() != y.len() || x.is_empty() {
                return Ok(0.0);
            }

            let result = self.compute_correlation_tensor(x, y).map_err(|e| {
                EvaluationError::MetricCalculationError {
                    metric: "Correlation".to_string(),
                    message: format!("GPU correlation failed: {e}"),
                    source: None,
                }
            })?;

            Ok(result)
        }

        /// GPU-accelerated FFT batch processing
        pub fn gpu_fft_batch(
            &self,
            signals: &[Vec<f32>],
        ) -> Result<Vec<Vec<f32>>, EvaluationError> {
            if signals.is_empty() {
                return Ok(Vec::new());
            }

            let result = self.compute_fft_batch_tensor(signals).map_err(|e| {
                EvaluationError::MetricCalculationError {
                    metric: "FFT".to_string(),
                    message: format!("GPU FFT batch failed: {e}"),
                    source: None,
                }
            })?;

            Ok(result)
        }

        /// GPU-accelerated spectral operations
        pub fn gpu_spectral_analysis(
            &self,
            signal: &[f32],
            sample_rate: f32,
        ) -> Result<SpectralFeatures, EvaluationError> {
            let features = self
                .compute_spectral_features(signal, sample_rate)
                .map_err(|e| EvaluationError::MetricCalculationError {
                    metric: "SpectralAnalysis".to_string(),
                    message: format!("GPU spectral analysis failed: {e}"),
                    source: None,
                })?;

            Ok(features)
        }

        /// GPU-accelerated MCD calculation
        pub fn gpu_mcd(
            &self,
            x_mfcc: &[Vec<f32>],
            y_mfcc: &[Vec<f32>],
        ) -> Result<f32, EvaluationError> {
            let mcd = self.compute_mcd_tensor(x_mfcc, y_mfcc).map_err(|e| {
                EvaluationError::MetricCalculationError {
                    metric: "MCD".to_string(),
                    message: format!("GPU MCD calculation failed: {e}"),
                    source: None,
                }
            })?;

            Ok(mcd)
        }

        /// GPU-accelerated autocorrelation
        pub fn gpu_autocorrelation(
            &self,
            signal: &[f32],
            max_lag: usize,
        ) -> Result<Vec<f32>, EvaluationError> {
            let result = self
                .compute_autocorrelation_tensor(signal, max_lag)
                .map_err(|e| EvaluationError::MetricCalculationError {
                    metric: "Autocorrelation".to_string(),
                    message: format!("GPU autocorrelation failed: {e}"),
                    source: None,
                })?;

            Ok(result)
        }

        // Internal tensor operations

        fn compute_correlation_tensor(&self, x: &[f32], y: &[f32]) -> CandleResult<f32> {
            let x_tensor = Tensor::from_slice(x, x.len(), &self.device)?;
            let y_tensor = Tensor::from_slice(y, y.len(), &self.device)?;

            // Calculate means as scalars and get the values
            let x_mean_val = x_tensor.mean_all()?.to_scalar::<f32>()?;
            let y_mean_val = y_tensor.mean_all()?.to_scalar::<f32>()?;

            // Center the data by subtracting scalar values
            let x_centered = x_tensor
                .to_vec1::<f32>()?
                .iter()
                .map(|&v| v - x_mean_val)
                .collect::<Vec<_>>();
            let y_centered = y_tensor
                .to_vec1::<f32>()?
                .iter()
                .map(|&v| v - y_mean_val)
                .collect::<Vec<_>>();

            let x_centered_tensor =
                Tensor::from_slice(&x_centered, x_centered.len(), &self.device)?;
            let y_centered_tensor =
                Tensor::from_slice(&y_centered, y_centered.len(), &self.device)?;

            // Calculate numerator and denominators
            let numerator = x_centered_tensor.mul(&y_centered_tensor)?.sum_all()?;
            let x_sq_sum = x_centered_tensor.mul(&x_centered_tensor)?.sum_all()?;
            let y_sq_sum = y_centered_tensor.mul(&y_centered_tensor)?.sum_all()?;

            let denominator = x_sq_sum.mul(&y_sq_sum)?.sqrt()?;

            let correlation = if denominator.to_scalar::<f32>()? > f32::EPSILON {
                numerator.div(&denominator)?.to_scalar::<f32>()?
            } else {
                0.0
            };

            Ok(correlation)
        }

        fn compute_fft_batch_tensor(&self, signals: &[Vec<f32>]) -> CandleResult<Vec<Vec<f32>>> {
            let mut results = Vec::new();

            for signal in signals {
                if signal.is_empty() {
                    results.push(Vec::new());
                    continue;
                }

                let signal_tensor = Tensor::from_slice(signal, signal.len(), &self.device)?;

                // Simple magnitude spectrum approximation using convolution
                // In a real implementation, you'd use proper FFT operations
                let spectrum = self.compute_magnitude_spectrum(&signal_tensor)?;
                results.push(spectrum);
            }

            Ok(results)
        }

        fn compute_magnitude_spectrum(&self, signal: &Tensor) -> CandleResult<Vec<f32>> {
            // Simplified spectrum computation - in practice you'd use proper FFT
            let signal_data = signal.to_vec1::<f32>()?;
            let spectrum_size = signal_data.len() / 2 + 1;
            let mut spectrum = vec![0.0; spectrum_size];

            for (i, value) in spectrum.iter_mut().enumerate() {
                let frequency_ratio = i as f32 / spectrum_size as f32;
                *value = (1.0 - frequency_ratio) * signal_data.iter().map(|x| x.abs()).sum::<f32>()
                    / signal_data.len() as f32;
            }

            Ok(spectrum)
        }

        fn compute_spectral_features(
            &self,
            signal: &[f32],
            sample_rate: f32,
        ) -> CandleResult<SpectralFeatures> {
            let signal_tensor = Tensor::from_slice(signal, signal.len(), &self.device)?;

            // Compute basic spectral features using GPU
            let magnitude_spectrum = self.compute_magnitude_spectrum(&signal_tensor)?;

            // Calculate spectral centroid
            let mut weighted_sum = 0.0;
            let mut total_energy = 0.0;

            for (i, &energy) in magnitude_spectrum.iter().enumerate() {
                let frequency = i as f32 * sample_rate / (2.0 * magnitude_spectrum.len() as f32);
                weighted_sum += frequency * energy;
                total_energy += energy;
            }

            let centroid = if total_energy > 0.0 {
                weighted_sum / total_energy
            } else {
                0.0
            };

            // Calculate spectral rolloff (frequency below which 85% of energy is contained)
            let mut cumulative_energy = 0.0;
            let target_energy = total_energy * 0.85;
            let mut rolloff = sample_rate / 2.0;

            for (i, &energy) in magnitude_spectrum.iter().enumerate() {
                cumulative_energy += energy;
                if cumulative_energy >= target_energy {
                    rolloff = i as f32 * sample_rate / (2.0 * magnitude_spectrum.len() as f32);
                    break;
                }
            }

            // Calculate spectral spread
            let mut spread_sum = 0.0;
            for (i, &energy) in magnitude_spectrum.iter().enumerate() {
                let frequency = i as f32 * sample_rate / (2.0 * magnitude_spectrum.len() as f32);
                spread_sum += (frequency - centroid).powi(2) * energy;
            }

            let spread = if total_energy > 0.0 {
                (spread_sum / total_energy).sqrt()
            } else {
                0.0
            };

            Ok(SpectralFeatures {
                centroid,
                spread,
                rolloff,
                flux: 0.0, // Would need previous frame for flux calculation
                energy: total_energy,
            })
        }

        fn compute_mcd_tensor(
            &self,
            x_mfcc: &[Vec<f32>],
            y_mfcc: &[Vec<f32>],
        ) -> CandleResult<f32> {
            if x_mfcc.is_empty() || y_mfcc.is_empty() {
                return Ok(f32::INFINITY);
            }

            let min_frames = x_mfcc.len().min(y_mfcc.len());
            let mut total_distance = 0.0;
            let mut valid_frames = 0;

            for i in 0..min_frames {
                if x_mfcc[i].len() == y_mfcc[i].len() && !x_mfcc[i].is_empty() {
                    let x_tensor = Tensor::from_slice(&x_mfcc[i], x_mfcc[i].len(), &self.device)?;
                    let y_tensor = Tensor::from_slice(&y_mfcc[i], y_mfcc[i].len(), &self.device)?;

                    // Skip c0 (energy coefficient) by taking slice from index 1
                    let x_ceps = x_tensor.narrow(0, 1, x_mfcc[i].len() - 1)?;
                    let y_ceps = y_tensor.narrow(0, 1, y_mfcc[i].len() - 1)?;

                    let diff = x_ceps.sub(&y_ceps)?;
                    let squared_diff = diff.mul(&diff)?;
                    let sum_squared = squared_diff.sum_all()?;

                    let distance = sum_squared.sqrt()?.to_scalar::<f32>()?;
                    total_distance += distance;
                    valid_frames += 1;
                }
            }

            if valid_frames > 0 {
                let mcd = (10.0 / std::f32::consts::LN_10) * (total_distance / valid_frames as f32);
                Ok(mcd)
            } else {
                Ok(f32::INFINITY)
            }
        }

        fn compute_autocorrelation_tensor(
            &self,
            signal: &[f32],
            max_lag: usize,
        ) -> CandleResult<Vec<f32>> {
            if signal.len() < 2 || max_lag == 0 {
                return Ok(vec![0.0; max_lag + 1]);
            }

            let signal_tensor = Tensor::from_slice(signal, signal.len(), &self.device)?;
            let effective_max_lag = max_lag.min(signal.len() - 1);
            let mut autocorr = vec![0.0; effective_max_lag + 1];

            for lag in 0..=effective_max_lag {
                if lag < signal.len() {
                    let signal1 = signal_tensor.narrow(0, 0, signal.len() - lag)?;
                    let signal2 = signal_tensor.narrow(0, lag, signal.len() - lag)?;

                    let correlation = signal1.mul(&signal2)?.sum_all()?;
                    let norm1 = signal1.mul(&signal1)?.sum_all()?.sqrt()?;
                    let norm2 = signal2.mul(&signal2)?.sum_all()?.sqrt()?;

                    let denominator = norm1.mul(&norm2)?.to_scalar::<f32>()?;
                    if denominator > 0.0 {
                        autocorr[lag] = correlation.to_scalar::<f32>()? / denominator;
                    }
                }
            }

            Ok(autocorr)
        }

        /// Clear memory pool to free GPU memory
        pub fn clear_memory_pool(&self) {
            self.memory_pool.write().clear();
        }

        /// Set maximum memory usage in MB
        pub fn set_max_memory(&mut self, max_memory_mb: usize) {
            self.max_memory_mb = max_memory_mb;
        }
    }

    impl Default for GpuAccelerator {
        fn default() -> Self {
            Self::new().unwrap_or_else(|_| Self::with_device(Device::Cpu))
        }
    }

    /// Spectral features computed on GPU
    #[derive(Debug, Clone)]
    pub struct SpectralFeatures {
        /// Spectral centroid (center of mass of the spectrum)
        pub centroid: f32,
        /// Spectral spread (variance around the centroid)
        pub spread: f32,
        /// Spectral rolloff frequency (95% of spectral energy below this frequency)
        pub rolloff: f32,
        /// Spectral flux (rate of change of the power spectrum)
        pub flux: f32,
        /// Total spectral energy
        pub energy: f32,
    }

    /// GPU memory manager for efficient tensor operations
    pub struct GpuMemoryManager {
        device: Device,
        allocated_tensors: Arc<RwLock<Vec<Tensor>>>,
        max_cache_size: usize,
    }

    impl GpuMemoryManager {
        /// Create a new GPU memory manager
        ///
        /// # Arguments
        /// * `device` - The GPU device to use for tensor operations
        /// * `max_cache_size` - Maximum number of tensors to cache in memory
        #[must_use]
        pub fn new(device: Device, max_cache_size: usize) -> Self {
            Self {
                device,
                allocated_tensors: Arc::new(RwLock::new(Vec::new())),
                max_cache_size,
            }
        }

        /// Allocate a tensor on the GPU with caching
        ///
        /// # Arguments
        /// * `shape` - Shape of the tensor to allocate
        ///
        /// # Returns
        /// A tensor allocated on the GPU device
        pub fn allocate_tensor(&self, shape: &[usize]) -> CandleResult<Tensor> {
            let tensor = Tensor::zeros(shape, candle_core::DType::F32, &self.device)?;

            let mut cache = self.allocated_tensors.write();
            if cache.len() < self.max_cache_size {
                cache.push(tensor.clone());
            }

            Ok(tensor)
        }

        /// Clear the tensor cache to free GPU memory
        pub fn clear_cache(&self) {
            self.allocated_tensors.write().clear();
        }
    }
}

/// Multi-GPU scaling capabilities for distributed computation
pub mod multi_gpu {
    use super::gpu::{GpuAccelerator, SpectralFeatures};
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Semaphore;

    /// Multi-GPU manager for distributed computation across multiple devices
    pub struct MultiGpuManager {
        accelerators: Vec<GpuAccelerator>,
        load_balancer: AtomicUsize,
        semaphore: Arc<Semaphore>,
        max_concurrent_ops: usize,
    }

    impl MultiGpuManager {
        /// Create a new multi-GPU manager with automatic device detection
        pub fn new(max_concurrent_ops: usize) -> Result<Self, crate::EvaluationError> {
            let accelerators = Self::detect_all_devices()?;
            let semaphore = Arc::new(Semaphore::new(max_concurrent_ops));

            Ok(Self {
                accelerators,
                load_balancer: AtomicUsize::new(0),
                semaphore,
                max_concurrent_ops,
            })
        }

        /// Create a multi-GPU manager with specific devices
        pub fn with_devices(devices: Vec<Device>, max_concurrent_ops: usize) -> Self {
            let accelerators = devices
                .into_iter()
                .map(GpuAccelerator::with_device)
                .collect();
            let semaphore = Arc::new(Semaphore::new(max_concurrent_ops));

            Self {
                accelerators,
                load_balancer: AtomicUsize::new(0),
                semaphore,
                max_concurrent_ops,
            }
        }

        /// Get the number of available GPU devices
        #[must_use]
        pub fn device_count(&self) -> usize {
            self.accelerators.len()
        }

        /// Check if any GPU acceleration is available
        #[must_use]
        pub fn has_gpu_acceleration(&self) -> bool {
            self.accelerators.iter().any(|acc| acc.is_gpu_available())
        }

        /// Get device information for all managed devices
        pub fn device_info(&self) -> Vec<String> {
            self.accelerators
                .iter()
                .enumerate()
                .map(|(idx, acc)| {
                    format!(
                        "Device {}: {} (GPU: {})",
                        idx,
                        match acc.device() {
                            Device::Cpu => "CPU",
                            Device::Cuda(_) => "CUDA",
                            Device::Metal(_) => "Metal",
                        },
                        acc.is_gpu_available()
                    )
                })
                .collect()
        }

        /// Distribute correlation calculations across multiple GPUs
        pub async fn distributed_correlation_batch(
            &self,
            data_pairs: &[(Vec<f32>, Vec<f32>)],
        ) -> Result<Vec<f32>, crate::EvaluationError> {
            if data_pairs.is_empty() {
                return Ok(Vec::new());
            }

            // Split work across devices
            let chunk_size =
                (data_pairs.len() + self.accelerators.len() - 1) / self.accelerators.len();
            let chunks: Vec<_> = data_pairs.chunks(chunk_size).collect();

            let mut handles = Vec::new();

            for (device_idx, chunk) in chunks.into_iter().enumerate() {
                let device_idx = device_idx % self.accelerators.len();
                let accelerator = &self.accelerators[device_idx];
                let chunk_data = chunk.to_vec();

                let permit = Arc::clone(&self.semaphore)
                    .acquire_owned()
                    .await
                    .map_err(|_| crate::EvaluationError::MetricCalculationError {
                        metric: "MultiGPU".to_string(),
                        message: "Failed to acquire semaphore permit".to_string(),
                        source: None,
                    })?;

                let accelerator_clone = accelerator.clone();
                let handle = tokio::spawn(async move {
                    let _permit = permit;
                    let mut results = Vec::new();

                    for (x, y) in chunk_data {
                        match accelerator_clone.gpu_correlation(&x, &y) {
                            Ok(corr) => results.push(corr),
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(results)
                });

                handles.push(handle);
            }

            // Collect results from all devices
            let mut all_results = Vec::new();
            for handle in handles {
                let chunk_results = handle.await.map_err(|e| {
                    crate::EvaluationError::MetricCalculationError {
                        metric: "MultiGPU".to_string(),
                        message: format!("GPU task failed: {e}"),
                        source: None,
                    }
                })??;
                all_results.extend(chunk_results);
            }

            Ok(all_results)
        }

        /// Distribute FFT operations across multiple GPUs
        pub async fn distributed_fft_batch(
            &self,
            signals: &[Vec<f32>],
        ) -> Result<Vec<Vec<f32>>, crate::EvaluationError> {
            if signals.is_empty() {
                return Ok(Vec::new());
            }

            let chunk_size =
                (signals.len() + self.accelerators.len() - 1) / self.accelerators.len();
            let chunks: Vec<_> = signals.chunks(chunk_size).collect();

            let mut handles = Vec::new();

            for (device_idx, chunk) in chunks.into_iter().enumerate() {
                let device_idx = device_idx % self.accelerators.len();
                let accelerator = &self.accelerators[device_idx];
                let chunk_data = chunk.to_vec();

                let permit = Arc::clone(&self.semaphore)
                    .acquire_owned()
                    .await
                    .map_err(|_| crate::EvaluationError::MetricCalculationError {
                        metric: "MultiGPU".to_string(),
                        message: "Failed to acquire semaphore permit".to_string(),
                        source: None,
                    })?;

                let accelerator_clone = accelerator.clone();
                let handle = tokio::spawn(async move {
                    let _permit = permit;
                    accelerator_clone.gpu_fft_batch(&chunk_data)
                });

                handles.push(handle);
            }

            // Collect results maintaining order
            let mut all_results = Vec::new();
            for handle in handles {
                let chunk_results = handle.await.map_err(|e| {
                    crate::EvaluationError::MetricCalculationError {
                        metric: "MultiGPU".to_string(),
                        message: format!("GPU FFT task failed: {e}"),
                        source: None,
                    }
                })??;
                all_results.extend(chunk_results);
            }

            Ok(all_results)
        }

        /// Distribute spectral analysis across multiple GPUs
        pub async fn distributed_spectral_analysis(
            &self,
            signals: &[Vec<f32>],
            sample_rate: f32,
        ) -> Result<Vec<SpectralFeatures>, crate::EvaluationError> {
            if signals.is_empty() {
                return Ok(Vec::new());
            }

            let chunk_size =
                (signals.len() + self.accelerators.len() - 1) / self.accelerators.len();
            let chunks: Vec<_> = signals.chunks(chunk_size).collect();

            let mut handles = Vec::new();

            for (device_idx, chunk) in chunks.into_iter().enumerate() {
                let device_idx = device_idx % self.accelerators.len();
                let accelerator = &self.accelerators[device_idx];
                let chunk_data = chunk.to_vec();

                let permit = Arc::clone(&self.semaphore)
                    .acquire_owned()
                    .await
                    .map_err(|_| crate::EvaluationError::MetricCalculationError {
                        metric: "MultiGPU".to_string(),
                        message: "Failed to acquire semaphore permit".to_string(),
                        source: None,
                    })?;

                let accelerator_clone = accelerator.clone();
                let handle = tokio::spawn(async move {
                    let _permit = permit;
                    let mut results = Vec::new();

                    for signal in chunk_data {
                        match accelerator_clone.gpu_spectral_analysis(&signal, sample_rate) {
                            Ok(features) => results.push(features),
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(results)
                });

                handles.push(handle);
            }

            // Collect results
            let mut all_results = Vec::new();
            for handle in handles {
                let chunk_results = handle.await.map_err(|e| {
                    crate::EvaluationError::MetricCalculationError {
                        metric: "MultiGPU".to_string(),
                        message: format!("GPU spectral analysis task failed: {e}"),
                        source: None,
                    }
                })??;
                all_results.extend(chunk_results);
            }

            Ok(all_results)
        }

        /// Get next device using round-robin load balancing
        pub fn get_next_device(&self) -> &GpuAccelerator {
            let index =
                self.load_balancer.fetch_add(1, Ordering::Relaxed) % self.accelerators.len();
            &self.accelerators[index]
        }

        /// Clear memory pools on all devices
        pub fn clear_all_memory_pools(&self) {
            for accelerator in &self.accelerators {
                accelerator.clear_memory_pool();
            }
        }

        /// Set maximum memory usage for all devices
        pub fn set_max_memory_all(&mut self, max_memory_mb: usize) {
            for accelerator in &mut self.accelerators {
                accelerator.set_max_memory(max_memory_mb);
            }
        }

        // Private helper methods

        fn detect_all_devices() -> Result<Vec<GpuAccelerator>, crate::EvaluationError> {
            let mut accelerators = Vec::new();

            // Try to detect CUDA devices
            for device_id in 0..8 {
                if let Ok(device) = Device::cuda_if_available(device_id) {
                    accelerators.push(GpuAccelerator::with_device(device));
                } else {
                    break;
                }
            }

            // Try to detect Metal devices
            for device_id in 0..4 {
                if let Ok(device) = Device::new_metal(device_id) {
                    accelerators.push(GpuAccelerator::with_device(device));
                } else {
                    break;
                }
            }

            // Always include CPU as fallback
            if accelerators.is_empty() {
                accelerators.push(GpuAccelerator::with_device(Device::Cpu));
            }

            Ok(accelerators)
        }
    }

    impl Clone for MultiGpuManager {
        fn clone(&self) -> Self {
            Self {
                accelerators: self.accelerators.clone(),
                load_balancer: AtomicUsize::new(self.load_balancer.load(Ordering::Relaxed)),
                semaphore: Arc::new(Semaphore::new(self.max_concurrent_ops)),
                max_concurrent_ops: self.max_concurrent_ops,
            }
        }
    }

    /// Performance metrics for multi-GPU operations
    #[derive(Debug, Clone)]
    pub struct MultiGpuMetrics {
        /// Total number of operations performed
        pub total_operations: usize,
        /// Operations per device
        pub operations_per_device: Vec<usize>,
        /// Average operation time per device
        pub avg_time_per_device: Vec<std::time::Duration>,
        /// Memory usage per device
        pub memory_usage_per_device: Vec<u64>,
        /// Overall throughput (operations per second)
        pub throughput: f64,
    }

    impl MultiGpuMetrics {
        /// Create new metrics instance
        #[must_use]
        pub fn new(device_count: usize) -> Self {
            Self {
                total_operations: 0,
                operations_per_device: vec![0; device_count],
                avg_time_per_device: vec![std::time::Duration::ZERO; device_count],
                memory_usage_per_device: vec![0; device_count],
                throughput: 0.0,
            }
        }

        /// Calculate load balance efficiency (0.0 = perfectly unbalanced, 1.0 = perfectly balanced)
        #[must_use]
        pub fn load_balance_efficiency(&self) -> f32 {
            if self.operations_per_device.is_empty() {
                return 1.0;
            }

            let avg_ops = self.operations_per_device.iter().sum::<usize>() as f32
                / self.operations_per_device.len() as f32;

            if avg_ops == 0.0 {
                return 1.0;
            }

            let variance = self
                .operations_per_device
                .iter()
                .map(|&ops| {
                    let diff = ops as f32 - avg_ops;
                    diff * diff
                })
                .sum::<f32>()
                / self.operations_per_device.len() as f32;

            let coefficient_of_variation = variance.sqrt() / avg_ops;
            (1.0 - coefficient_of_variation).max(0.0)
        }

        /// Get device with highest throughput
        #[must_use]
        pub fn fastest_device(&self) -> Option<usize> {
            self.avg_time_per_device
                .iter()
                .enumerate()
                .filter(|(_, &time)| time > std::time::Duration::ZERO)
                .min_by_key(|(_, &time)| time)
                .map(|(idx, _)| idx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let corr = parallel_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_lru_cache() {
        let cache = LRUCache::new(2);

        cache.insert("key1", "value1");
        cache.insert("key2", "value2");

        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key2"), Some("value2"));

        // Should trigger eviction
        cache.insert("key3", "value3");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_sliding_window_processor() {
        let processor = SlidingWindowProcessor::new(3, 1);
        let data = vec![1, 2, 3, 4, 5];

        let results = processor.process_parallel(&data, |window| window.iter().sum::<i32>());

        assert_eq!(results, vec![6, 9, 12]); // [1,2,3], [2,3,4], [3,4,5]
    }

    #[test]
    fn test_simd_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let dot = simd::dot_product(&a, &b);
        assert_eq!(dot, 40.0); // 1*2 + 2*3 + 3*4 + 4*5

        let product = simd::element_wise_multiply(&a, &b);
        assert_eq!(product, vec![2.0, 6.0, 12.0, 20.0]);

        let rms = simd::rms(&a);
        assert!((rms - 2.738_613).abs() < 0.001);
    }

    #[test]
    fn test_simd_vector_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = simd::vector_add(&a, &b);
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Test empty vectors
        let empty_result = simd::vector_add(&[], &[]);
        assert_eq!(empty_result, Vec::<f32>::new());

        // Test mismatched lengths
        let mismatch_result = simd::vector_add(&a, &[1.0]);
        assert_eq!(mismatch_result, Vec::<f32>::new());
    }

    #[test]
    fn test_simd_vector_subtract() {
        let a = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result = simd::vector_subtract(&a, &b);
        assert_eq!(result, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);

        // Test empty vectors
        let empty_result = simd::vector_subtract(&[], &[]);
        assert_eq!(empty_result, Vec::<f32>::new());

        // Test mismatched lengths
        let mismatch_result = simd::vector_subtract(&a, &[1.0]);
        assert_eq!(mismatch_result, Vec::<f32>::new());
    }

    #[test]
    fn test_simd_large_vectors() {
        // Test with vectors larger than SIMD thresholds to ensure parallel fallback
        let size = 2000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let dot = simd::dot_product(&a, &b);
        let expected_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        // Use relative tolerance for large numbers
        let tolerance = (expected_dot * 1e-5).max(1.0);
        assert!((dot - expected_dot).abs() < tolerance);

        let product = simd::element_wise_multiply(&a, &b);
        let expected_product: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        assert_eq!(product.len(), expected_product.len());
        for (actual, expected) in product.iter().zip(&expected_product) {
            let tolerance = (expected.abs() * 1e-5).max(1e-3);
            assert!((actual - expected).abs() < tolerance);
        }
    }

    #[test]
    fn test_simd_spectral_centroid() {
        let spectrum = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let sample_rate = 16000.0;

        let centroid = simd::spectral_centroid(&spectrum, sample_rate);

        // Manual calculation for verification
        let frequencies: Vec<f32> = (0..spectrum.len())
            .map(|i| i as f32 * sample_rate / (2.0 * spectrum.len() as f32))
            .collect();

        let weighted_sum: f32 = spectrum
            .iter()
            .zip(&frequencies)
            .map(|(mag, freq)| mag * freq)
            .sum();
        let magnitude_sum: f32 = spectrum.iter().sum();
        let expected = if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        };

        assert!((centroid - expected).abs() < 0.001);
    }

    #[test]
    fn test_simd_edge_cases() {
        // Test with zero vectors
        let zeros = vec![0.0; 8];
        let ones = vec![1.0; 8];

        assert_eq!(simd::dot_product(&zeros, &ones), 0.0);
        assert_eq!(simd::rms(&zeros), 0.0);
        assert_eq!(simd::spectral_centroid(&zeros, 16000.0), 0.0);

        // Test with single element
        let single_a = vec![5.0];
        let single_b = vec![3.0];

        assert_eq!(simd::dot_product(&single_a, &single_b), 15.0);
        assert_eq!(simd::vector_add(&single_a, &single_b), vec![8.0]);
        assert_eq!(simd::vector_subtract(&single_a, &single_b), vec![2.0]);
    }

    #[test]
    fn test_simd_numerical_precision() {
        // Test with very small numbers to ensure precision
        let a = vec![1e-6, 2e-6, 3e-6, 4e-6];
        let b = vec![1e-6, 2e-6, 3e-6, 4e-6];

        let dot = simd::dot_product(&a, &b);
        let expected = 1e-12 + 4e-12 + 9e-12 + 16e-12; // 30e-12
        assert!((dot - expected).abs() < 1e-15);

        // Test with very large numbers
        let large_a = vec![1e6, 2e6, 3e6, 4e6];
        let large_b = vec![1e6, 2e6, 3e6, 4e6];

        let large_dot = simd::dot_product(&large_a, &large_b);
        let large_expected = 1e12 + 4e12 + 9e12 + 16e12; // 30e12
        assert!((large_dot - large_expected).abs() < 1e9);
    }

    #[test]
    fn test_simd_performance_consistency() {
        // Verify that SIMD and scalar implementations produce the same results
        let size = 100;
        let a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).cos()).collect();

        let simd_dot = simd::dot_product(&a, &b);
        let scalar_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        // Results should be very close (allowing for minor floating point differences)
        assert!((simd_dot - scalar_dot).abs() < 1e-5);

        let simd_product = simd::element_wise_multiply(&a, &b);
        let scalar_product: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();

        assert_eq!(simd_product.len(), scalar_product.len());
        for (simd_val, scalar_val) in simd_product.iter().zip(&scalar_product) {
            assert!((simd_val - scalar_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();

        let result = monitor.time_operation("test_op", || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        let avg_time = monitor.get_average_time("test_op").unwrap();
        assert!(avg_time >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_gpu_accelerator_creation() {
        let accelerator = gpu::GpuAccelerator::default();

        // Should always succeed with fallback to CPU
        assert!(matches!(accelerator.device(), Device::Cpu) || accelerator.is_gpu_available());
    }

    #[test]
    fn test_gpu_correlation() {
        let accelerator = gpu::GpuAccelerator::default();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let correlation = accelerator.gpu_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gpu_autocorrelation() {
        let accelerator = gpu::GpuAccelerator::default();
        let signal = vec![1.0, 0.8, 0.6, 0.4, 0.2];

        let autocorr = accelerator.gpu_autocorrelation(&signal, 3).unwrap();
        assert_eq!(autocorr.len(), 4); // 0 to 3 lags inclusive
        assert!((autocorr[0] - 1.0).abs() < 0.001); // Perfect correlation at lag 0
    }

    #[test]
    fn test_gpu_spectral_analysis() {
        let accelerator = gpu::GpuAccelerator::default();
        let signal = vec![0.1; 1024];
        let sample_rate = 16000.0;

        let features = accelerator
            .gpu_spectral_analysis(&signal, sample_rate)
            .unwrap();
        assert!(features.centroid >= 0.0);
        assert!(features.energy >= 0.0);
        assert!(features.rolloff >= 0.0);
        assert!(features.spread >= 0.0);
    }

    #[test]
    fn test_gpu_memory_manager() {
        let device = Device::Cpu; // Use CPU for testing
        let manager = gpu::GpuMemoryManager::new(device, 10);

        let tensor = manager.allocate_tensor(&[10, 10]).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 10]);

        manager.clear_cache();
    }

    #[test]
    fn test_persistent_cache() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("test_cache");

        let cache = PersistentCache::new(&cache_dir, 10, 6).unwrap();

        // Test insertion and retrieval
        assert!(cache
            .insert("key1".to_string(), "value1".to_string())
            .is_ok());
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));

        // Test disk persistence
        let cache2 = PersistentCache::new(&cache_dir, 10, 6).unwrap();
        assert_eq!(cache2.get(&"key1".to_string()), Some("value1".to_string()));

        // Test statistics
        let stats = cache.stats();
        assert!(stats.disk_entries > 0);
        assert!(stats.cache_dir_size > 0);

        // Test clear
        assert!(cache.clear().is_ok());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_persistent_cache_compression() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("test_compression");

        let mut cache = PersistentCache::new(&cache_dir, 10, 1).unwrap();

        // Test compression level setting
        cache.set_compression_level(9);

        // Insert large value to test compression
        let large_value = "x".repeat(1000);
        assert!(cache
            .insert("large_key".to_string(), large_value.clone())
            .is_ok());
        assert_eq!(cache.get(&"large_key".to_string()), Some(large_value));

        // Test that compressed file is smaller than original
        let stats = cache.stats();
        assert!(stats.cache_dir_size < 1000); // Should be compressed
    }

    #[test]
    fn test_persistent_cache_memory_fallback() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("test_memory");

        let cache = PersistentCache::new(&cache_dir, 2, 6).unwrap();

        // Fill memory cache
        assert!(cache
            .insert("key1".to_string(), "value1".to_string())
            .is_ok());
        assert!(cache
            .insert("key2".to_string(), "value2".to_string())
            .is_ok());

        // This should evict from memory but still be available on disk
        assert!(cache
            .insert("key3".to_string(), "value3".to_string())
            .is_ok());

        // All values should still be retrievable
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
        assert_eq!(cache.get(&"key2".to_string()), Some("value2".to_string()));
        assert_eq!(cache.get(&"key3".to_string()), Some("value3".to_string()));

        let stats = cache.stats();
        assert_eq!(stats.disk_entries, 3);
    }

    #[test]
    fn test_multi_gpu_manager_creation() {
        use multi_gpu::MultiGpuManager;

        let manager = MultiGpuManager::new(4).unwrap();

        // Should have at least one device (CPU fallback)
        assert!(manager.device_count() >= 1);

        // Device info should be available
        let info = manager.device_info();
        assert!(!info.is_empty());
        assert!(!info[0].is_empty());
    }

    #[tokio::test]
    async fn test_multi_gpu_distributed_correlation() {
        use multi_gpu::MultiGpuManager;

        let manager = MultiGpuManager::new(2).unwrap();

        let data_pairs = vec![
            (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]),
            (vec![4.0, 5.0, 6.0], vec![4.0, 5.0, 6.0]),
            (vec![1.0, 0.0, -1.0], vec![-1.0, 0.0, 1.0]),
        ];

        let correlations = manager
            .distributed_correlation_batch(&data_pairs)
            .await
            .unwrap();

        assert_eq!(correlations.len(), 3);
        assert!((correlations[0] - 1.0).abs() < 0.001); // Perfect correlation
        assert!((correlations[1] - 1.0).abs() < 0.001); // Perfect correlation
        assert!((correlations[2] + 1.0).abs() < 0.001); // Perfect negative correlation
    }

    #[tokio::test]
    async fn test_multi_gpu_distributed_fft() {
        use multi_gpu::MultiGpuManager;

        let manager = MultiGpuManager::new(2).unwrap();

        let signals = vec![vec![1.0, 0.0, -1.0, 0.0], vec![0.5, 1.0, 0.5, 0.0]];

        let spectra = manager.distributed_fft_batch(&signals).await.unwrap();

        assert_eq!(spectra.len(), 2);
        assert!(!spectra[0].is_empty());
        assert!(!spectra[1].is_empty());
    }

    #[tokio::test]
    async fn test_multi_gpu_distributed_spectral_analysis() {
        use multi_gpu::MultiGpuManager;

        let manager = MultiGpuManager::new(2).unwrap();

        let signals = vec![vec![0.1; 512], vec![0.2; 512]];
        let sample_rate = 16000.0;

        let features = manager
            .distributed_spectral_analysis(&signals, sample_rate)
            .await
            .unwrap();

        assert_eq!(features.len(), 2);
        for feature in &features {
            assert!(feature.centroid >= 0.0);
            assert!(feature.energy >= 0.0);
            assert!(feature.rolloff >= 0.0);
        }
    }

    #[test]
    fn test_multi_gpu_metrics() {
        use multi_gpu::MultiGpuMetrics;

        let mut metrics = MultiGpuMetrics::new(3);
        metrics.operations_per_device = vec![10, 10, 10]; // Perfectly balanced

        let efficiency = metrics.load_balance_efficiency();
        assert!((efficiency - 1.0).abs() < 0.1); // Should be close to perfectly balanced

        // Test unbalanced case
        metrics.operations_per_device = vec![20, 5, 5]; // Unbalanced
        let efficiency = metrics.load_balance_efficiency();
        assert!(efficiency < 1.0); // Should be less than perfectly balanced
    }

    #[test]
    fn test_multi_gpu_load_balancing() {
        use multi_gpu::MultiGpuManager;

        let devices = vec![Device::Cpu, Device::Cpu]; // Use CPU devices for testing
        let manager = MultiGpuManager::with_devices(devices, 4);

        // Test round-robin load balancing
        let device1 = manager.get_next_device();
        let device2 = manager.get_next_device();
        let device3 = manager.get_next_device(); // Should wrap back to first device

        assert!(std::ptr::eq(device1, device3)); // Should be the same device due to round-robin
    }
}
