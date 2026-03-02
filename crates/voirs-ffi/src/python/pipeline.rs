//! # VoiRS Pipeline Python Bindings
//!
//! This module provides Python bindings for the VoiRS speech synthesis pipeline,
//! enabling high-quality text-to-speech synthesis from Python applications.
//!
//! ## Features
//!
//! - **Synchronous and Asynchronous Synthesis**: Support for both blocking and async operations
//! - **SSML Support**: Full SSML (Speech Synthesis Markup Language) support for advanced control
//! - **Voice Management**: Dynamic voice switching and enumeration
//! - **Performance Metrics**: Built-in performance tracking and cache statistics
//! - **Streaming Synthesis**: Real-time audio generation with streaming support
//! - **GPU Acceleration**: Optional GPU support for faster synthesis
//!
//! ## Usage Example
//!
//! ```python
//! from voirs_ffi import VoirsPipeline
//!
//! # Create pipeline with default settings
//! pipeline = VoirsPipeline(use_gpu=False, num_threads=4)
//!
//! # Simple synthesis
//! result = pipeline.synthesize("Hello, world!")
//! audio_data = result.audio_data  # NumPy array of float32 samples
//!
//! # SSML synthesis with advanced control
//! ssml = '''<speak>
//!     <prosody rate="slow" pitch="+2st">
//!         This is synthesized with custom prosody.
//!     </prosody>
//! </speak>'''
//! result = pipeline.synthesize_ssml(ssml)
//!
//! # Switch voices
//! voices = pipeline.list_voices()
//! pipeline.set_voice(voices[0].id)
//!
//! # Get performance metrics
//! metrics = pipeline.get_performance_metrics()
//! print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
//! ```
//!
//! ## Thread Safety
//!
//! The `VoirsPipeline` class is thread-safe and can be safely shared across Python threads
//! using appropriate synchronization mechanisms (e.g., locks).

use super::audio_buffer::PyAudioBuffer;
use super::common::*;
use super::error::VoirsErrorInfo;
use super::metrics::{SynthesisMetrics, SynthesisResult};
use super::voice::PyVoiceInfo;

/// Internal performance metrics tracker for the VoiRS Pipeline.
///
/// Tracks synthesis operations, cache hits/misses, and computes cache hit rates.
/// This is used internally by [`VoirsPipeline`] to provide performance statistics
/// to Python applications.
#[derive(Debug, Clone, Default)]
struct PerformanceTracker {
    total_syntheses: u64,
    cache_hits: u64,
    cache_misses: u64,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            total_syntheses: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    fn record_synthesis(&mut self, cache_hit: bool) {
        self.total_syntheses += 1;
        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
    }

    fn get_cache_hit_rate(&self) -> f64 {
        if self.total_syntheses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_syntheses as f64
        }
    }

    fn reset(&mut self) {
        self.total_syntheses = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
}

/// Main VoiRS speech synthesis pipeline for Python.
///
/// This class provides the primary interface for text-to-speech synthesis in Python,
/// wrapping the underlying Rust VoiRS SDK with a Pythonic API.
///
/// # Features
///
/// - **Text Synthesis**: Convert plain text to speech with `synthesize()`
/// - **SSML Synthesis**: Use SSML markup for advanced control with `synthesize_ssml()`
/// - **Voice Management**: List, select, and switch between voices
/// - **Performance Tracking**: Monitor cache hits, synthesis count, and performance metrics
/// - **Callback Support**: Register progress and error callbacks for async operations
/// - **Thread-Safe**: Safe to use across multiple Python threads with proper locking
///
/// # Example
///
/// ```python
/// # Create a new pipeline
/// pipeline = VoirsPipeline()
///
/// # Basic synthesis
/// result = pipeline.synthesize("Hello, world!")
///
/// # Access audio data as NumPy array
/// audio = result.audio_data
/// sample_rate = result.sample_rate
///
/// # SSML synthesis
/// ssml = '<speak><prosody rate="slow">Slow speech</prosody></speak>'
/// result = pipeline.synthesize_ssml(ssml)
/// ```
///
/// # Thread Safety
///
/// While the pipeline is internally thread-safe, Python's GIL (Global Interpreter Lock)
/// ensures that most operations are automatically synchronized. For concurrent access
/// from multiple threads, use appropriate Python synchronization primitives.
#[pyclass]
pub struct VoirsPipeline {
    inner: Arc<SdkPipeline>,
    rt: Runtime,
    progress_callback: Arc<parking_lot::Mutex<Option<PyObject>>>,
    error_callback: Arc<parking_lot::Mutex<Option<PyObject>>>,
    performance_tracker: Arc<parking_lot::Mutex<PerformanceTracker>>,
}

#[pymethods]
impl VoirsPipeline {
    #[new]
    fn new() -> PyResult<Self> {
        let rt = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let inner = rt
            .block_on(SdkPipeline::builder().build())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create pipeline: {}", e)))?;

        Ok(Self {
            inner: Arc::new(inner),
            rt,
            progress_callback: Arc::new(parking_lot::Mutex::new(None)),
            error_callback: Arc::new(parking_lot::Mutex::new(None)),
            performance_tracker: Arc::new(parking_lot::Mutex::new(PerformanceTracker::new())),
        })
    }

    #[staticmethod]
    fn with_config(
        use_gpu: Option<bool>,
        num_threads: Option<usize>,
        cache_dir: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let rt = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let mut builder = SdkPipeline::builder();

        if let Some(gpu) = use_gpu {
            builder = builder.with_gpu(gpu);
        }

        if let Some(threads) = num_threads {
            builder = builder.with_threads(threads);
        }

        if let Some(cache) = cache_dir {
            builder = builder.with_cache_dir(cache);
        }

        if let Some(dev) = device {
            builder = builder.with_device(dev.to_string());
        }

        let inner = rt
            .block_on(builder.build())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create pipeline: {}", e)))?;

        Ok(Self {
            inner: Arc::new(inner),
            rt,
            progress_callback: Arc::new(parking_lot::Mutex::new(None)),
            error_callback: Arc::new(parking_lot::Mutex::new(None)),
            performance_tracker: Arc::new(parking_lot::Mutex::new(PerformanceTracker::new())),
        })
    }

    /// Synthesize text to audio
    fn synthesize(&self, text: &str) -> PyResult<PyAudioBuffer> {
        let audio = self
            .rt
            .block_on(self.inner.synthesize(text))
            .map_err(|e| PyRuntimeError::new_err(format!("Synthesis failed: {}", e)))?;

        Ok(PyAudioBuffer::new(audio))
    }

    /// Synthesize SSML to audio
    fn synthesize_ssml(&self, ssml: &str) -> PyResult<PyAudioBuffer> {
        let audio = self
            .rt
            .block_on(self.inner.synthesize_ssml(ssml))
            .map_err(|e| PyRuntimeError::new_err(format!("SSML synthesis failed: {}", e)))?;

        Ok(PyAudioBuffer::new(audio))
    }

    /// Set the voice for synthesis
    fn set_voice(&self, voice_id: &str) -> PyResult<()> {
        self.rt
            .block_on(self.inner.set_voice(voice_id))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set voice: {}", e)))?;

        Ok(())
    }

    /// Get the current voice
    fn get_voice(&self) -> Option<String> {
        self.rt
            .block_on(self.inner.current_voice())
            .map(|v| v.id.clone())
    }

    /// List available voices
    fn list_voices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let voices = self
            .rt
            .block_on(self.inner.list_voices())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to list voices: {}", e)))?;

        let voice_list: Vec<Py<PyVoiceInfo>> = voices
            .into_iter()
            .map(|voice| {
                Py::new(
                    py,
                    PyVoiceInfo {
                        id: voice.id,
                        name: voice.name,
                        language: voice.language.to_string(),
                        quality: format!("{:?}", voice.characteristics.quality),
                        is_available: true, // Always available for now
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()?;

        PyList::new(py, voice_list)
    }

    /// Enhanced synthesis with metrics and better error handling
    fn synthesize_with_metrics(&self, text: &str) -> PyResult<SynthesisResult> {
        use std::time::Instant;

        let start_time = Instant::now();
        let start_memory = self.get_memory_usage_mb();

        let audio = self.rt.block_on(self.inner.synthesize(text)).map_err(|e| {
            self.create_structured_error(
                "synthesis_failed",
                &e.to_string(),
                Some("Check text content and voice configuration".to_string()),
            )
        })?;

        let processing_time_ms = start_time.elapsed().as_millis() as f64;
        let audio_duration_ms =
            (audio.samples().len() as f64) / (audio.sample_rate() as f64) * 1000.0;
        let real_time_factor = processing_time_ms / audio_duration_ms;
        let memory_usage_mb = self.get_memory_usage_mb() - start_memory;

        // Estimate cache hit based on processing speed
        // Real-time factor < 0.1 typically indicates cached/fast synthesis
        let is_cache_hit = real_time_factor < 0.1;

        // Record synthesis in performance tracker
        self.performance_tracker
            .lock()
            .record_synthesis(is_cache_hit);

        // Get current cache hit rate from performance tracker
        let cache_hit_rate = self.performance_tracker.lock().get_cache_hit_rate();

        let metrics = SynthesisMetrics::new(
            processing_time_ms,
            audio_duration_ms,
            real_time_factor,
            memory_usage_mb,
            cache_hit_rate,
        );

        Ok(SynthesisResult::new(PyAudioBuffer::new(audio), metrics))
    }

    /// Enhanced SSML synthesis with metrics
    fn synthesize_ssml_with_metrics(&self, ssml: &str) -> PyResult<SynthesisResult> {
        use std::time::Instant;

        let start_time = Instant::now();
        let start_memory = self.get_memory_usage_mb();

        let audio = self
            .rt
            .block_on(self.inner.synthesize_ssml(ssml))
            .map_err(|e| {
                self.create_structured_error(
                    "ssml_synthesis_failed",
                    &e.to_string(),
                    Some("Check SSML syntax and voice configuration".to_string()),
                )
            })?;

        let processing_time_ms = start_time.elapsed().as_millis() as f64;
        let audio_duration_ms =
            (audio.samples().len() as f64) / (audio.sample_rate() as f64) * 1000.0;
        let real_time_factor = processing_time_ms / audio_duration_ms;
        let memory_usage_mb = self.get_memory_usage_mb() - start_memory;

        // Estimate cache hit based on processing speed
        // Real-time factor < 0.1 typically indicates cached/fast synthesis
        let is_cache_hit = real_time_factor < 0.1;

        // Record synthesis in performance tracker
        self.performance_tracker
            .lock()
            .record_synthesis(is_cache_hit);

        // Get current cache hit rate from performance tracker
        let cache_hit_rate = self.performance_tracker.lock().get_cache_hit_rate();

        let metrics = SynthesisMetrics::new(
            processing_time_ms,
            audio_duration_ms,
            real_time_factor,
            memory_usage_mb,
            cache_hit_rate,
        );

        Ok(SynthesisResult::new(PyAudioBuffer::new(audio), metrics))
    }

    /// Batch synthesis with progress tracking
    fn batch_synthesize(&self, texts: Vec<String>) -> PyResult<Vec<SynthesisResult>> {
        let mut results = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            // Add progress tracking in the future
            let result = self.synthesize_with_metrics(text)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Batch synthesis with progress callback
    fn batch_synthesize_with_progress(
        &self,
        texts: Vec<String>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Vec<SynthesisResult>> {
        Python::attach(|py| {
            let mut results = Vec::new();
            let total_count = texts.len();

            for (i, text) in texts.iter().enumerate() {
                // Call progress callback if provided
                if let Some(ref callback) = progress_callback {
                    let progress = (i as f64) / (total_count as f64);
                    let args = (i, total_count, progress, text.as_str());

                    if let Err(e) = callback.call1(py, args) {
                        return Err(PyRuntimeError::new_err(format!(
                            "Progress callback failed: {}",
                            e
                        )));
                    }
                }

                let result = self.synthesize_with_metrics(text)?;
                results.push(result);
            }

            // Call progress callback one final time at 100%
            if let Some(ref callback) = progress_callback {
                let args = (total_count, total_count, 1.0, "");
                let _ = callback.call1(py, args); // Ignore errors on final call
            }

            Ok(results)
        })
    }

    /// Synthesize with streaming callback for real-time audio chunks
    fn synthesize_streaming(
        &self,
        text: &str,
        chunk_callback: PyObject,
        chunk_size: Option<usize>,
    ) -> PyResult<PyAudioBuffer> {
        Python::attach(|py| {
            // Use actual streaming synthesis from SDK
            let pipeline = Arc::new(self.inner.clone());
            let mut stream = self
                .rt
                .block_on(
                    <Arc<voirs_sdk::VoirsPipeline> as Clone>::clone(&pipeline)
                        .synthesize_stream(text),
                )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Streaming synthesis failed: {}", e))
                })?;

            let mut full_audio: Option<AudioBuffer> = None;
            let mut chunk_index = 0;

            // Process each chunk from the stream
            while let Some(chunk_result) = self.rt.block_on(futures::StreamExt::next(&mut stream)) {
                let chunk = chunk_result.map_err(|e| {
                    PyRuntimeError::new_err(format!("Chunk synthesis failed: {}", e))
                })?;

                // If chunk_size is specified, further split the chunk
                if let Some(size) = chunk_size {
                    let samples = chunk.samples();
                    let sub_chunks: Vec<&[f32]> = samples.chunks(size).collect();

                    for (sub_i, sub_chunk) in sub_chunks.iter().enumerate() {
                        let sub_chunk_audio = AudioBuffer::new(
                            sub_chunk.to_vec(),
                            chunk.sample_rate(),
                            chunk.channels(),
                        );
                        let py_chunk = PyAudioBuffer::new(sub_chunk_audio);

                        let args = (chunk_index * 1000 + sub_i, 0, py_chunk); // total_chunks unknown for streaming
                        if let Err(e) = chunk_callback.call1(py, args) {
                            return Err(PyRuntimeError::new_err(format!(
                                "Chunk callback failed: {}",
                                e
                            )));
                        }
                    }
                } else {
                    // Use the chunk as-is
                    let py_chunk = PyAudioBuffer::new(chunk.clone());
                    let args = (chunk_index, 0, py_chunk); // total_chunks unknown for streaming
                    if let Err(e) = chunk_callback.call1(py, args) {
                        return Err(PyRuntimeError::new_err(format!(
                            "Chunk callback failed: {}",
                            e
                        )));
                    }
                }

                // Accumulate chunks for final result
                if let Some(ref mut audio) = full_audio {
                    audio.append(&chunk).map_err(|e| {
                        PyRuntimeError::new_err(format!("Audio append failed: {}", e))
                    })?;
                } else {
                    full_audio = Some(chunk);
                }

                chunk_index += 1;
            }

            // Return the full audio or empty buffer if no chunks
            let final_audio = full_audio.unwrap_or_else(|| AudioBuffer::new(vec![], 22050, 1));
            Ok(PyAudioBuffer::new(final_audio))
        })
    }

    /// Synthesize with error callback for enhanced error handling
    fn synthesize_with_error_callback(
        &self,
        text: &str,
        error_callback: Option<PyObject>,
    ) -> PyResult<PyAudioBuffer> {
        Python::attach(|py| {
            match self.rt.block_on(self.inner.synthesize(text)) {
                Ok(audio) => Ok(PyAudioBuffer::new(audio)),
                Err(e) => {
                    // Call error callback if provided
                    if let Some(ref callback) = error_callback {
                        let error_info = VoirsErrorInfo::create(
                            "synthesis_failed".to_string(),
                            e.to_string(),
                            Some("Check text content and voice configuration".to_string()),
                            Some("Try different voice or simplify text".to_string()),
                        );

                        let args = (error_info,);
                        let _ = callback.call1(py, args); // Ignore callback errors
                    }

                    Err(PyRuntimeError::new_err(format!("Synthesis failed: {}", e)))
                }
            }
        })
    }

    /// Set progress callback for long-running operations
    fn set_progress_callback(&self, callback: Option<PyObject>) -> PyResult<()> {
        // Validate callback is callable if provided
        if let Some(ref cb) = callback {
            Python::attach(|py| {
                if !cb.bind(py).is_callable() {
                    return Err(PyValueError::new_err("Progress callback must be callable"));
                }
                Ok(())
            })?;
        }

        // Store callback in pipeline state for use across operations
        *self.progress_callback.lock() = callback;
        println!("Progress callback configured and stored");
        Ok(())
    }

    /// Set error callback for error handling
    fn set_error_callback(&self, callback: Option<PyObject>) -> PyResult<()> {
        // Validate callback is callable if provided
        if let Some(ref cb) = callback {
            Python::attach(|py| {
                if !cb.bind(py).is_callable() {
                    return Err(PyValueError::new_err("Error callback must be callable"));
                }
                Ok(())
            })?;
        }

        // Store callback in pipeline state for use across operations
        *self.error_callback.lock() = callback;
        println!("Error callback configured and stored");
        Ok(())
    }

    /// Synthesize with comprehensive callback support
    fn synthesize_with_callbacks(
        &self,
        text: &str,
        progress_callback: Option<PyObject>,
        chunk_callback: Option<PyObject>,
        error_callback: Option<PyObject>,
        chunk_size: Option<usize>,
    ) -> PyResult<PyAudioBuffer> {
        Python::attach(|py| {
            // Validate callbacks
            for (name, callback) in [
                ("progress", &progress_callback),
                ("chunk", &chunk_callback),
                ("error", &error_callback),
            ] {
                if let Some(ref cb) = callback {
                    if !cb.bind(py).is_callable() {
                        return Err(PyValueError::new_err(format!(
                            "{} callback must be callable",
                            name
                        )));
                    }
                }
            }

            // Call progress callback at start
            if let Some(ref callback) = progress_callback {
                let args = (0, 100, 0.0, "Starting synthesis");
                let _ = callback.call1(py, args);
            }

            // Perform synthesis with error handling
            let audio = match self.rt.block_on(self.inner.synthesize(text)) {
                Ok(audio) => audio,
                Err(e) => {
                    // Call error callback
                    if let Some(ref callback) = error_callback {
                        let error_info = VoirsErrorInfo::create(
                            "synthesis_failed".to_string(),
                            e.to_string(),
                            Some("Synthesis operation failed".to_string()),
                            Some("Check text content and voice settings".to_string()),
                        );
                        let _ = callback.call1(py, (error_info,));
                    }
                    return Err(PyRuntimeError::new_err(format!("Synthesis failed: {}", e)));
                }
            };

            // Call progress callback at 50%
            if let Some(ref callback) = progress_callback {
                let args = (50, 100, 0.5, "Synthesis complete, processing audio");
                let _ = callback.call1(py, args);
            }

            // Stream audio chunks if chunk callback provided
            if let Some(ref callback) = chunk_callback {
                let samples = audio.samples();
                let chunk_size = chunk_size.unwrap_or(1024);
                let chunks: Vec<&[f32]> = samples.chunks(chunk_size).collect();

                for (i, chunk) in chunks.iter().enumerate() {
                    let chunk_audio =
                        AudioBuffer::new(chunk.to_vec(), audio.sample_rate(), audio.channels());
                    let py_chunk = PyAudioBuffer::new(chunk_audio);

                    let args = (i, chunks.len(), py_chunk);
                    if let Err(e) = callback.call1(py, args) {
                        // Call error callback for chunk processing errors
                        if let Some(ref error_cb) = error_callback {
                            let error_info = VoirsErrorInfo::create(
                                "chunk_callback_failed".to_string(),
                                e.to_string(),
                                Some("Chunk callback execution failed".to_string()),
                                Some("Check callback implementation".to_string()),
                            );
                            let _ = error_cb.call1(py, (error_info,));
                        }
                        return Err(PyRuntimeError::new_err(format!(
                            "Chunk callback failed: {}",
                            e
                        )));
                    }
                }
            }

            // Call progress callback at completion
            if let Some(ref callback) = progress_callback {
                let args = (100, 100, 1.0, "Complete");
                let _ = callback.call1(py, args);
            }

            Ok(PyAudioBuffer::new(audio))
        })
    }

    /// Get system performance information
    fn get_performance_info(&self) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("cpu_cores", num_cpus::get())?;
            info.set_item("memory_usage_mb", self.get_memory_usage_mb())?;
            info.set_item("gpu_available", self.is_gpu_available())?;

            Ok(info.unbind())
        })
    }

    /// Get the library version
    #[staticmethod]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// Get current cache hit rate
    fn get_cache_hit_rate(&self) -> f64 {
        self.performance_tracker.lock().get_cache_hit_rate()
    }

    /// Get performance statistics
    fn get_performance_stats(&self) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let tracker = self.performance_tracker.lock();
            let stats = PyDict::new(py);
            stats.set_item("total_syntheses", tracker.total_syntheses)?;
            stats.set_item("cache_hits", tracker.cache_hits)?;
            stats.set_item("cache_misses", tracker.cache_misses)?;
            stats.set_item("cache_hit_rate", tracker.get_cache_hit_rate())?;
            Ok(stats.unbind())
        })
    }

    /// Reset performance statistics
    fn reset_performance_stats(&self) -> PyResult<()> {
        self.performance_tracker.lock().reset();
        Ok(())
    }
}

impl VoirsPipeline {
    /// Helper method to create structured errors
    fn create_structured_error(
        &self,
        code: &str,
        message: &str,
        suggestion: Option<String>,
    ) -> PyErr {
        let error_info =
            VoirsErrorInfo::new(code.to_string(), message.to_string(), None, suggestion);
        PyRuntimeError::new_err(error_info.message.clone())
    }

    /// Get current memory usage in MB
    fn get_memory_usage_mb(&self) -> f64 {
        // Simplified memory usage calculation
        // In a real implementation, this would use process memory stats
        0.0
    }

    /// Check if GPU is available
    fn is_gpu_available(&self) -> bool {
        // Simplified GPU check
        // In a real implementation, this would check actual GPU availability
        false
    }
}
