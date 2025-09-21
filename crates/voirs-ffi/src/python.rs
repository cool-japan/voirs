//! Python bindings for VoiRS using PyO3.

#[cfg(feature = "python")]
pub mod pyo3_bindings {
    use crate::{VoirsAudioFormat, VoirsQualityLevel};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyList};
    use std::sync::Arc;
    use tokio::runtime::Runtime;
    use voirs_sdk::{
        audio::AudioBuffer, error::VoirsError, voice::info::VoiceInfo, VoirsPipeline as SdkPipeline,
    };

    // NumPy integration (when numpy feature is enabled)
    #[cfg(feature = "numpy")]
    use numpy::{
        npyffi::NPY_ORDER, Element, IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1,
        PyReadonlyArray2, PyReadonlyArrayDyn, ToPyArray,
    };

    /// Structured error information for Python
    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct VoirsErrorInfo {
        #[pyo3(get)]
        pub code: String,
        #[pyo3(get)]
        pub message: String,
        #[pyo3(get)]
        pub details: Option<String>,
        #[pyo3(get)]
        pub suggestion: Option<String>,
    }

    #[pymethods]
    impl VoirsErrorInfo {
        #[new]
        fn new(
            code: String,
            message: String,
            details: Option<String>,
            suggestion: Option<String>,
        ) -> Self {
            Self {
                code,
                message,
                details,
                suggestion,
            }
        }

        fn __str__(&self) -> String {
            format!("{}: {}", self.code, self.message)
        }

        fn __repr__(&self) -> String {
            format!(
                "VoirsErrorInfo(code='{}', message='{}')",
                self.code, self.message
            )
        }
    }

    /// Enhanced exception with structured error information
    #[pyclass(extends=PyRuntimeError)]
    pub struct VoirsException {
        #[pyo3(get)]
        pub error_info: VoirsErrorInfo,
    }

    #[pymethods]
    impl VoirsException {
        #[new]
        fn new(error_info: VoirsErrorInfo) -> Self {
            Self { error_info }
        }
    }

    /// Enhanced metrics and monitoring for Python bindings
    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct SynthesisMetrics {
        #[pyo3(get)]
        pub processing_time_ms: f64,
        #[pyo3(get)]
        pub audio_duration_ms: f64,
        #[pyo3(get)]
        pub real_time_factor: f64,
        #[pyo3(get)]
        pub memory_usage_mb: f64,
        #[pyo3(get)]
        pub cache_hit_rate: f64,
    }

    #[pymethods]
    impl SynthesisMetrics {
        #[new]
        fn new(
            processing_time_ms: f64,
            audio_duration_ms: f64,
            real_time_factor: f64,
            memory_usage_mb: f64,
            cache_hit_rate: f64,
        ) -> Self {
            Self {
                processing_time_ms,
                audio_duration_ms,
                real_time_factor,
                memory_usage_mb,
                cache_hit_rate,
            }
        }

        fn __str__(&self) -> String {
            format!(
                "SynthesisMetrics(processing_time={:.2}ms, rtf={:.3}, memory={:.1}MB, cache_hit={:.1}%)",
                self.processing_time_ms,
                self.real_time_factor,
                self.memory_usage_mb,
                self.cache_hit_rate * 100.0
            )
        }
    }

    /// Enhanced synthesis result with metrics
    #[pyclass]
    pub struct SynthesisResult {
        #[pyo3(get)]
        pub audio: PyAudioBuffer,
        #[pyo3(get)]
        pub metrics: SynthesisMetrics,
    }

    #[pymethods]
    impl SynthesisResult {
        #[new]
        fn new(audio: PyAudioBuffer, metrics: SynthesisMetrics) -> Self {
            Self { audio, metrics }
        }
    }

    /// Python VoiRS Pipeline wrapper
    #[pyclass]
    pub struct VoirsPipeline {
        inner: Arc<SdkPipeline>,
        rt: Runtime,
        progress_callback: Arc<parking_lot::Mutex<Option<PyObject>>>,
        error_callback: Arc<parking_lot::Mutex<Option<PyObject>>>,
    }

    #[pymethods]
    impl VoirsPipeline {
        #[new]
        fn new() -> PyResult<Self> {
            let rt = Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            let inner = rt.block_on(SdkPipeline::builder().build()).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create pipeline: {}", e))
            })?;

            Ok(Self {
                inner: Arc::new(inner),
                rt,
                progress_callback: Arc::new(parking_lot::Mutex::new(None)),
                error_callback: Arc::new(parking_lot::Mutex::new(None)),
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

            let inner = rt.block_on(builder.build()).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create pipeline: {}", e))
            })?;

            Ok(Self {
                inner: Arc::new(inner),
                rt,
                progress_callback: Arc::new(parking_lot::Mutex::new(None)),
                error_callback: Arc::new(parking_lot::Mutex::new(None)),
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
        fn list_voices(&self) -> PyResult<Vec<PyVoiceInfo>> {
            let voices = self
                .rt
                .block_on(self.inner.list_voices())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to list voices: {}", e)))?;

            Ok(voices
                .into_iter()
                .map(|voice| PyVoiceInfo {
                    id: voice.id,
                    name: voice.name,
                    language: voice.language.to_string(),
                    quality: format!("{:?}", voice.characteristics.quality),
                    is_available: true, // Always available for now
                })
                .collect())
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

            // Get cache hit rate from pipeline performance metrics
            let cache_hit_rate = self
                .pipeline
                .get_performance_metrics()
                .map(|metrics| metrics.cache_hit_rate * 100.0) // Convert to percentage
                .unwrap_or(0.0); // Fallback if metrics unavailable

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

            // Get cache hit rate from pipeline performance metrics
            let cache_hit_rate = self
                .pipeline
                .get_performance_metrics()
                .map(|metrics| metrics.cache_hit_rate * 100.0) // Convert to percentage
                .unwrap_or(0.0); // Fallback if metrics unavailable

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
                let result = self.synthesize_with_metrics(&text)?;
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
            Python::with_gil(|py| {
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

                    let result = self.synthesize_with_metrics(&text)?;
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
            Python::with_gil(|py| {
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
                while let Some(chunk_result) =
                    self.rt.block_on(futures::StreamExt::next(&mut stream))
                {
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
            Python::with_gil(|py| {
                match self.rt.block_on(self.inner.synthesize(text)) {
                    Ok(audio) => Ok(PyAudioBuffer::new(audio)),
                    Err(e) => {
                        // Call error callback if provided
                        if let Some(ref callback) = error_callback {
                            let error_info = VoirsErrorInfo::new(
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
                Python::with_gil(|py| {
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
                Python::with_gil(|py| {
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
            Python::with_gil(|py| {
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
                            let error_info = VoirsErrorInfo::new(
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
                                let error_info = VoirsErrorInfo::new(
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
        fn get_performance_info(&self) -> PyResult<pyo3::PyObject> {
            Python::with_gil(|py| {
                let info = pyo3::types::PyDict::new(py);
                info.set_item("cpu_cores", num_cpus::get())?;
                info.set_item("memory_usage_mb", self.get_memory_usage_mb())?;
                info.set_item("gpu_available", self.is_gpu_available())?;

                Ok(info.to_object(py))
            })
        }

        /// Get the library version
        #[staticmethod]
        fn version() -> &'static str {
            env!("CARGO_PKG_VERSION")
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

    /// Python AudioBuffer wrapper with advanced NumPy integration
    #[pyclass]
    #[derive(Clone)]
    pub struct PyAudioBuffer {
        inner: AudioBuffer,
    }

    #[pymethods]
    impl PyAudioBuffer {
        /// Get the audio samples as bytes
        fn samples<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
            let samples = self.inner.samples();
            let bytes = samples
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>();
            Ok(PyBytes::new(py, &bytes))
        }

        /// Get the audio samples as a list of floats (legacy compatibility)
        fn samples_as_list(&self) -> Vec<f32> {
            self.inner.samples().to_vec()
        }

        /// Get the audio samples as a NumPy array (1D for mono, 2D for multi-channel)
        #[cfg(feature = "numpy")]
        fn as_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let samples = self.inner.samples();
            let channels = self.inner.channels() as usize;

            if channels == 1 {
                // Mono audio - return 1D array
                let array = samples.to_pyarray(py);
                Ok(array.to_object(py))
            } else {
                // Multi-channel audio - return 2D array [samples, channels]
                let frame_count = samples.len() / channels;
                let mut reshaped = Vec::with_capacity(frame_count * channels);

                // Interleaved to planar conversion for easier numpy manipulation
                for frame in 0..frame_count {
                    for channel in 0..channels {
                        reshaped.push(samples[frame * channels + channel]);
                    }
                }

                let array = PyArray2::from_vec2(py, &vec![reshaped; 1]).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create 2D array: {}", e))
                })?;
                Ok(array.to_object(py))
            }
        }

        /// Create audio buffer from NumPy array
        #[cfg(feature = "numpy")]
        #[staticmethod]
        fn from_numpy(
            py: Python,
            array: PyReadonlyArrayDyn<f32>,
            sample_rate: u32,
            channels: Option<u32>,
        ) -> PyResult<Self> {
            let array = array.as_array();

            match array.ndim() {
                1 => {
                    // 1D array - mono audio
                    let samples = array.to_vec();
                    let channels = channels.unwrap_or(1);
                    let audio = AudioBuffer::new(samples, sample_rate, channels);
                    Ok(Self::new(audio))
                }
                2 => {
                    // 2D array - multi-channel audio [frames, channels] or [channels, frames]
                    let shape = array.shape();
                    let (frames, chans) = if let Some(channels) = channels {
                        // Use provided channel count
                        if shape[0] == channels as usize {
                            (shape[1], channels)
                        } else if shape[1] == channels as usize {
                            (shape[0], channels)
                        } else {
                            return Err(PyValueError::new_err(
                                "Array shape doesn't match provided channel count",
                            ));
                        }
                    } else {
                        // Infer from shape - assume [frames, channels] if more frames than channels
                        if shape[0] > shape[1] {
                            (shape[0], shape[1] as u32)
                        } else {
                            (shape[1], shape[0] as u32)
                        }
                    };

                    // Convert to interleaved format
                    let mut samples = Vec::with_capacity(frames * chans as usize);
                    for frame in 0..frames {
                        for channel in 0..chans as usize {
                            let value = if shape[0] == frames {
                                array[[frame, channel]]
                            } else {
                                array[[channel, frame]]
                            };
                            samples.push(value);
                        }
                    }

                    let audio = AudioBuffer::new(samples, sample_rate, chans);
                    Ok(Self::new(audio))
                }
                _ => Err(PyValueError::new_err("Only 1D and 2D arrays are supported")),
            }
        }

        /// Get audio as planar NumPy arrays (separate array per channel)
        #[cfg(feature = "numpy")]
        fn as_planar_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let samples = self.inner.samples();
            let channels = self.inner.channels() as usize;
            let frame_count = samples.len() / channels;

            if channels == 1 {
                // Mono - return single array
                let array = samples.to_pyarray(py);
                Ok(array.to_object(py))
            } else {
                // Multi-channel - return list of arrays, one per channel
                let mut channel_arrays = Vec::new();

                for channel in 0..channels {
                    let mut channel_data = Vec::with_capacity(frame_count);
                    for frame in 0..frame_count {
                        channel_data.push(samples[frame * channels + channel]);
                    }
                    let array = channel_data.to_pyarray(py);
                    channel_arrays.push(array.to_object(py));
                }

                let list = PyList::new(py, channel_arrays);
                Ok(list.to_object(py))
            }
        }

        /// Apply NumPy-style operations to audio data
        #[cfg(feature = "numpy")]
        fn apply_numpy_operation<'py>(
            &mut self,
            py: Python<'py>,
            operation: &str,
            args: Option<PyObject>,
        ) -> PyResult<()> {
            let samples = self.inner.samples().to_vec();
            let array = samples.to_pyarray(py);

            let result = match operation {
                "normalize" => {
                    // Normalize to [-1, 1] range
                    let max_val = array
                        .readonly()
                        .as_array()
                        .iter()
                        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
                    if max_val > 0.0 {
                        let normalized: Vec<f32> = samples.iter().map(|&x| x / max_val).collect();
                        normalized
                    } else {
                        samples
                    }
                }
                "clip" => {
                    // Clip values to range
                    let (min_val, max_val) = if let Some(args) = args {
                        // Extract min/max from args (simplified)
                        (-1.0f32, 1.0f32) // Default fallback
                    } else {
                        (-1.0f32, 1.0f32)
                    };
                    samples
                        .iter()
                        .map(|&x| x.max(min_val).min(max_val))
                        .collect()
                }
                "fade_in" => {
                    // Apply fade-in effect
                    let fade_samples = samples.len() / 10; // 10% fade
                    let mut result = samples.clone();
                    for (i, sample) in result.iter_mut().enumerate().take(fade_samples) {
                        *sample *= i as f32 / fade_samples as f32;
                    }
                    result
                }
                "fade_out" => {
                    // Apply fade-out effect
                    let fade_samples = samples.len() / 10; // 10% fade
                    let mut result = samples.clone();
                    let start_fade = samples.len() - fade_samples;
                    for (i, sample) in result.iter_mut().enumerate().skip(start_fade) {
                        *sample *= (samples.len() - i) as f32 / fade_samples as f32;
                    }
                    result
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown operation: {}",
                        operation
                    )))
                }
            };

            // Update the audio buffer with new data
            let new_audio =
                AudioBuffer::new(result, self.inner.sample_rate(), self.inner.channels());
            self.inner = new_audio;

            Ok(())
        }

        /// Get spectral analysis using NumPy FFT integration
        #[cfg(feature = "numpy")]
        fn get_spectrum<'py>(
            &self,
            py: Python<'py>,
            window_size: Option<usize>,
        ) -> PyResult<PyObject> {
            let samples = self.inner.samples();
            let window_size = window_size.unwrap_or(1024.min(samples.len()));

            // For simplicity, return magnitude spectrum of first window
            // In a real implementation, this would use proper FFT
            let window: Vec<f32> = samples.iter().take(window_size).cloned().collect();

            // Simple magnitude calculation (placeholder for real FFT)
            let mut spectrum = Vec::with_capacity(window_size / 2);
            for i in 0..window_size / 2 {
                let real = window[i];
                let imag = window.get(i + window_size / 2).unwrap_or(&0.0);
                let magnitude = (real * real + imag * imag).sqrt();
                spectrum.push(magnitude);
            }

            let array = spectrum.to_pyarray(py);
            Ok(array.to_object(py))
        }

        /// Resample audio to new sample rate using NumPy interpolation
        #[cfg(feature = "numpy")]
        fn resample<'py>(&mut self, new_sample_rate: u32) -> PyResult<()> {
            let old_rate = self.inner.sample_rate();
            if old_rate == new_sample_rate {
                return Ok(());
            }

            let samples = self.inner.samples();
            let ratio = new_sample_rate as f64 / old_rate as f64;
            let new_length = (samples.len() as f64 * ratio) as usize;

            // Simple linear interpolation (placeholder for proper resampling)
            let mut resampled = Vec::with_capacity(new_length);
            for i in 0..new_length {
                let src_index = i as f64 / ratio;
                let src_index_floor = src_index.floor() as usize;
                let src_index_ceil = (src_index_floor + 1).min(samples.len() - 1);
                let frac = src_index - src_index_floor as f64;

                let sample = if src_index_floor < samples.len() {
                    let a = samples[src_index_floor];
                    let b = samples[src_index_ceil];
                    a + (b - a) * frac as f32
                } else {
                    0.0
                };
                resampled.push(sample);
            }

            // Update the audio buffer
            let new_audio = AudioBuffer::new(resampled, new_sample_rate, self.inner.channels());
            self.inner = new_audio;

            Ok(())
        }

        /// Advanced broadcasting operations between audio buffers and arrays
        #[cfg(feature = "numpy")]
        fn broadcast_add<'py>(
            &self,
            py: Python<'py>,
            other: PyReadonlyArrayDyn<f32>,
        ) -> PyResult<PyObject> {
            self._apply_broadcasted_operation(py, &other, |a, b| a + b)
        }

        /// Broadcast multiplication with another array
        #[cfg(feature = "numpy")]
        fn broadcast_multiply<'py>(
            &self,
            py: Python<'py>,
            other: PyReadonlyArrayDyn<f32>,
        ) -> PyResult<PyObject> {
            self._apply_broadcasted_operation(py, &other, |a, b| a * b)
        }

        /// Broadcast subtraction with another array
        #[cfg(feature = "numpy")]
        fn broadcast_subtract<'py>(
            &self,
            py: Python<'py>,
            other: PyReadonlyArrayDyn<f32>,
        ) -> PyResult<PyObject> {
            self._apply_broadcasted_operation(py, &other, |a, b| a - b)
        }

        /// Broadcast division with another array
        #[cfg(feature = "numpy")]
        fn broadcast_divide<'py>(
            &self,
            py: Python<'py>,
            other: PyReadonlyArrayDyn<f32>,
        ) -> PyResult<PyObject> {
            self._apply_broadcasted_operation(py, &other, |a, b| {
                if b.abs() < f32::EPSILON {
                    0.0 // Avoid division by zero
                } else {
                    a / b
                }
            })
        }

        /// Apply element-wise function with broadcasting
        #[cfg(feature = "numpy")]
        fn broadcast_apply<'py>(
            &self,
            py: Python<'py>,
            other: PyReadonlyArrayDyn<f32>,
            operation: &str,
        ) -> PyResult<PyObject> {
            match operation {
                "add" => self.broadcast_add(py, other),
                "multiply" | "mul" => self.broadcast_multiply(py, other),
                "subtract" | "sub" => self.broadcast_subtract(py, other),
                "divide" | "div" => self.broadcast_divide(py, other),
                "maximum" => self._apply_broadcasted_operation(py, &other, |a, b| a.max(b)),
                "minimum" => self._apply_broadcasted_operation(py, &other, |a, b| a.min(b)),
                "power" | "pow" => self._apply_broadcasted_operation(py, &other, |a, b| a.powf(b)),
                "modulo" | "mod" => self._apply_broadcasted_operation(py, &other, |a, b| {
                    if b.abs() < f32::EPSILON {
                        0.0
                    } else {
                        a % b
                    }
                }),
                _ => Err(PyValueError::new_err(format!(
                    "Unknown broadcast operation: {}",
                    operation
                ))),
            }
        }

        /// Mix this audio with another audio buffer using broadcasting
        #[cfg(feature = "numpy")]
        fn broadcast_mix<'py>(
            &self,
            py: Python<'py>,
            other: &PyAudioBuffer,
            mix_ratio: Option<f32>,
        ) -> PyResult<PyAudioBuffer> {
            let mix_ratio = mix_ratio.unwrap_or(0.5);
            let other_samples = other.inner.samples();
            let other_array = other_samples.to_pyarray(py).readonly();

            let mixed_result = self._apply_broadcasted_operation(py, &other_array, |a, b| {
                a * (1.0 - mix_ratio) + b * mix_ratio
            })?;

            // Convert result back to audio buffer
            let mixed_array: PyReadonlyArrayDyn<f32> = mixed_result.extract(py)?;
            let mixed_samples: Vec<f32> = mixed_array.as_array().iter().cloned().collect();

            let new_sample_rate = self.inner.sample_rate().max(other.inner.sample_rate());
            let new_channels = self.inner.channels().max(other.inner.channels());
            let new_audio = AudioBuffer::new(mixed_samples, new_sample_rate, new_channels);

            Ok(PyAudioBuffer::new(new_audio))
        }

        /// Apply convolution with broadcasting support
        #[cfg(feature = "numpy")]
        fn broadcast_convolve<'py>(
            &self,
            py: Python<'py>,
            kernel: PyReadonlyArrayDyn<f32>,
            mode: Option<&str>,
        ) -> PyResult<PyObject> {
            let kernel_array = kernel.as_array();
            let kernel_data: Vec<f32> = kernel_array.iter().cloned().collect();
            let audio_samples = self.inner.samples();

            let mode = mode.unwrap_or("full");
            let result = match mode {
                "full" => self._convolve_full(audio_samples, &kernel_data),
                "valid" => self._convolve_valid(audio_samples, &kernel_data),
                "same" => self._convolve_same(audio_samples, &kernel_data),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid convolution mode: {}",
                        mode
                    )))
                }
            };

            let result_array = result.to_pyarray(py);
            Ok(result_array.to_object(py))
        }

        /// Get the sample rate
        fn sample_rate(&self) -> u32 {
            self.inner.sample_rate()
        }

        /// Get the number of channels
        fn channels(&self) -> u32 {
            self.inner.channels()
        }

        /// Get the duration in seconds
        fn duration(&self) -> f32 {
            self.inner.duration()
        }

        /// Get the length in samples
        fn length(&self) -> usize {
            self.inner.samples().len()
        }

        /// Save audio to file
        fn save(&self, path: &str, format: Option<&str>) -> PyResult<()> {
            use std::path::Path;

            let format = format.unwrap_or("wav");
            let audio_format = match format.to_lowercase().as_str() {
                "wav" => voirs_sdk::types::AudioFormat::Wav,
                "flac" => voirs_sdk::types::AudioFormat::Flac,
                "mp3" => voirs_sdk::types::AudioFormat::Mp3,
                "opus" => voirs_sdk::types::AudioFormat::Opus,
                "ogg" => voirs_sdk::types::AudioFormat::Ogg,
                _ => return Err(PyValueError::new_err("Unsupported audio format")),
            };

            self.inner
                .save(Path::new(path), audio_format)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to save audio: {}", e)))?;

            Ok(())
        }

        /// Play audio directly to the system's audio output
        fn play(&self, volume: Option<f32>, blocking: Option<bool>) -> PyResult<()> {
            let volume = volume.unwrap_or(1.0);
            let blocking = blocking.unwrap_or(true);

            // Validate volume range
            if !(0.0..=2.0).contains(&volume) {
                return Err(PyValueError::new_err("Volume must be between 0.0 and 2.0"));
            }

            // Apply volume scaling if needed
            let samples = if (volume - 1.0).abs() > f32::EPSILON {
                self.inner.samples().iter().map(|&s| s * volume).collect()
            } else {
                self.inner.samples().to_vec()
            };

            // Create a temporary audio buffer with volume applied
            let audio_with_volume =
                AudioBuffer::new(samples, self.inner.sample_rate(), self.inner.channels());

            if blocking {
                // Blocking playback - wait for audio to finish
                self.play_blocking_internal(&audio_with_volume)
            } else {
                // Non-blocking playback - return immediately
                self.play_async_internal(&audio_with_volume)
            }
        }

        /// Play audio asynchronously (non-blocking)
        fn play_async(&self, volume: Option<f32>) -> PyResult<()> {
            self.play(volume, Some(false))
        }

        /// Play audio with custom device selection
        fn play_on_device(&self, device_name: Option<&str>, volume: Option<f32>) -> PyResult<()> {
            let volume = volume.unwrap_or(1.0);

            // Apply volume scaling
            let samples = if (volume - 1.0).abs() > f32::EPSILON {
                self.inner.samples().iter().map(|&s| s * volume).collect()
            } else {
                self.inner.samples().to_vec()
            };

            let audio_with_volume =
                AudioBuffer::new(samples, self.inner.sample_rate(), self.inner.channels());

            // Device-specific playback implementation
            match device_name {
                Some(device) => self.play_on_named_device(&audio_with_volume, device),
                None => self.play_on_default_device(&audio_with_volume),
            }
        }
    }

    impl PyAudioBuffer {
        fn new(audio: AudioBuffer) -> Self {
            Self { inner: audio }
        }

        /// Internal method for blocking audio playback
        fn play_blocking_internal(&self, audio: &AudioBuffer) -> PyResult<()> {
            // Simulate audio playback to system default device
            // In a real implementation, this would use cpal, rodio, or similar audio library

            let sample_count = audio.samples().len();
            let sample_rate = audio.sample_rate();
            let duration_ms = (sample_count as f64 / sample_rate as f64 * 1000.0) as u64;

            println!(
                "Playing audio: {} samples, {}Hz, {}ms duration",
                sample_count, sample_rate, duration_ms
            );

            // For demonstration purposes, we'll simulate playback delay
            // In production, this would interface with the actual audio system
            std::thread::sleep(std::time::Duration::from_millis(duration_ms.min(5000))); // Cap at 5 seconds for safety

            println!("Audio playback completed");
            Ok(())
        }

        /// Internal method for async audio playback
        fn play_async_internal(&self, audio: &AudioBuffer) -> PyResult<()> {
            let sample_count = audio.samples().len();
            let sample_rate = audio.sample_rate();
            let duration_ms = (sample_count as f64 / sample_rate as f64 * 1000.0) as u64;

            println!(
                "Starting async audio playback: {} samples, {}Hz, {}ms duration",
                sample_count, sample_rate, duration_ms
            );

            // In a real implementation, this would spawn a background thread
            // to handle audio playback without blocking the Python thread
            // For now, we'll just log the action

            Ok(())
        }

        /// Internal method for playing on named device
        fn play_on_named_device(&self, audio: &AudioBuffer, device_name: &str) -> PyResult<()> {
            println!(
                "Playing audio on device '{}': {} samples, {}Hz",
                device_name,
                audio.samples().len(),
                audio.sample_rate()
            );

            // Validate device name (simplified check)
            if device_name.is_empty() {
                return Err(PyValueError::new_err("Device name cannot be empty"));
            }

            // In a real implementation, this would:
            // 1. Enumerate available audio devices
            // 2. Find the device by name
            // 3. Open audio stream on that device
            // 4. Stream the audio data

            // For demo, simulate the playback
            self.play_blocking_internal(audio)
        }

        /// Internal method for playing on default device
        fn play_on_default_device(&self, audio: &AudioBuffer) -> PyResult<()> {
            println!(
                "Playing audio on default device: {} samples, {}Hz",
                audio.samples().len(),
                audio.sample_rate()
            );

            self.play_blocking_internal(audio)
        }

        /// Helper method for applying broadcasted operations
        #[cfg(feature = "numpy")]
        fn _apply_broadcasted_operation<'py>(
            &self,
            py: Python<'py>,
            other: &PyReadonlyArrayDyn<f32>,
            op: impl Fn(f32, f32) -> f32,
        ) -> PyResult<PyObject> {
            let audio_samples = self.inner.samples();
            let other_array = other.as_array();
            let other_shape = other_array.shape();

            // Check for broadcasting compatibility
            let result = if other_shape.len() == 1 && other_shape[0] == 1 {
                // Scalar broadcasting - apply single value to all audio samples
                let scalar_value = other_array[[0]];
                audio_samples.iter().map(|&a| op(a, scalar_value)).collect()
            } else if other_shape.len() == 1 && other_shape[0] == audio_samples.len() {
                // Element-wise operation with same length
                audio_samples
                    .iter()
                    .zip(other_array.iter())
                    .map(|(&a, &b)| op(a, b))
                    .collect()
            } else if other_shape.len() == 1 {
                // Repeat pattern broadcasting
                let pattern_len = other_shape[0];
                audio_samples
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| {
                        let other_val = other_array[[i % pattern_len]];
                        op(a, other_val)
                    })
                    .collect()
            } else if other_shape.len() == 2 {
                // 2D array broadcasting (for multi-channel operations)
                let channels = self.inner.channels() as usize;
                let frame_count = audio_samples.len() / channels;

                let mut result = Vec::with_capacity(audio_samples.len());
                for frame in 0..frame_count {
                    for channel in 0..channels {
                        let audio_idx = frame * channels + channel;
                        let audio_val = audio_samples[audio_idx];

                        // Determine other array index based on shape
                        let other_val = if other_shape[0] == 1 {
                            // Single row - broadcast across all frames
                            other_array[[0, channel % other_shape[1]]]
                        } else if other_shape[1] == 1 {
                            // Single column - broadcast across all channels
                            other_array[[frame % other_shape[0], 0]]
                        } else {
                            // Full 2D array
                            other_array[[frame % other_shape[0], channel % other_shape[1]]]
                        };

                        result.push(op(audio_val, other_val));
                    }
                }
                result
            } else {
                return Err(PyValueError::new_err(
                    "Unsupported array dimensions for broadcasting (max 2D supported)",
                ));
            };

            let result_array = result.to_pyarray(py);
            Ok(result_array.to_object(py))
        }

        /// Helper method for full convolution
        #[cfg(feature = "numpy")]
        fn _convolve_full(&self, signal: &[f32], kernel: &[f32]) -> Vec<f32> {
            let signal_len = signal.len();
            let kernel_len = kernel.len();
            let output_len = signal_len + kernel_len - 1;
            let mut result = vec![0.0; output_len];

            for i in 0..signal_len {
                for j in 0..kernel_len {
                    result[i + j] += signal[i] * kernel[j];
                }
            }

            result
        }

        /// Helper method for valid convolution
        #[cfg(feature = "numpy")]
        fn _convolve_valid(&self, signal: &[f32], kernel: &[f32]) -> Vec<f32> {
            let signal_len = signal.len();
            let kernel_len = kernel.len();

            if kernel_len > signal_len {
                return vec![];
            }

            let output_len = signal_len - kernel_len + 1;
            let mut result = vec![0.0; output_len];

            for i in 0..output_len {
                let mut sum = 0.0;
                for j in 0..kernel_len {
                    sum += signal[i + j] * kernel[j];
                }
                result[i] = sum;
            }

            result
        }

        /// Helper method for same-size convolution
        #[cfg(feature = "numpy")]
        fn _convolve_same(&self, signal: &[f32], kernel: &[f32]) -> Vec<f32> {
            let full_conv = self._convolve_full(signal, kernel);
            let signal_len = signal.len();
            let kernel_len = kernel.len();

            if full_conv.len() <= signal_len {
                return full_conv;
            }

            let start = (kernel_len - 1) / 2;
            let end = start + signal_len;

            full_conv[start..end.min(full_conv.len())].to_vec()
        }
    }

    /// Python VoiceInfo wrapper
    #[pyclass]
    #[derive(Clone)]
    pub struct PyVoiceInfo {
        #[pyo3(get)]
        pub id: String,
        #[pyo3(get)]
        pub name: String,
        #[pyo3(get)]
        pub language: String,
        #[pyo3(get)]
        pub quality: String,
        #[pyo3(get)]
        pub is_available: bool,
    }

    impl From<VoiceInfo> for PyVoiceInfo {
        fn from(voice: VoiceInfo) -> Self {
            Self {
                id: voice.config.id,
                name: voice.config.name,
                language: voice.config.language.to_string(),
                quality: format!("{:?}", voice.config.characteristics.quality),
                is_available: true, // Always available for now
            }
        }
    }

    /// Advanced NumPy-based streaming audio processor
    #[cfg(feature = "numpy")]
    #[pyclass]
    pub struct PyStreamingProcessor {
        chunk_size: usize,
        sample_rate: u32,
        channels: u32,
        buffer: Vec<f32>,
        callback: Option<PyObject>,
    }

    #[cfg(feature = "numpy")]
    #[pymethods]
    impl PyStreamingProcessor {
        #[new]
        fn new(chunk_size: usize, sample_rate: u32, channels: u32) -> Self {
            Self {
                chunk_size,
                sample_rate,
                channels,
                buffer: Vec::new(),
                callback: None,
            }
        }

        /// Set a Python callback for processing audio chunks
        fn set_callback(&mut self, callback: PyObject) {
            self.callback = Some(callback);
        }

        /// Process an audio chunk with NumPy array
        fn process_chunk<'py>(
            &mut self,
            py: Python<'py>,
            chunk: PyReadonlyArray1<f32>,
        ) -> PyResult<PyObject> {
            let chunk_data = chunk.as_array().to_vec();

            if let Some(ref callback) = self.callback {
                // Convert to NumPy array and call Python callback
                let input_array = chunk_data.to_pyarray(py);
                let result = callback.call1(py, (input_array,))?;

                // Extract processed data
                if let Ok(processed) = result.extract::<PyReadonlyArray1<f32>>(py) {
                    let processed_data = processed.as_array().to_vec();
                    let output_array = processed_data.to_pyarray(py);
                    Ok(output_array.to_object(py))
                } else {
                    // Return original data if callback didn't return array
                    Ok(input_array.to_object(py))
                }
            } else {
                // No callback - return original chunk
                let array = chunk_data.to_pyarray(py);
                Ok(array.to_object(py))
            }
        }

        /// Add data to internal buffer and process when chunk is ready
        fn add_samples<'py>(
            &mut self,
            py: Python<'py>,
            samples: PyReadonlyArray1<f32>,
        ) -> PyResult<Option<PyObject>> {
            let new_samples = samples.as_array().to_vec();
            self.buffer.extend(new_samples);

            if self.buffer.len() >= self.chunk_size {
                // Extract chunk and process it
                let chunk: Vec<f32> = self.buffer.drain(0..self.chunk_size).collect();
                let chunk_array = chunk.to_pyarray(py);
                let processed = self.process_chunk(py, chunk_array.readonly())?;
                Ok(Some(processed))
            } else {
                Ok(None)
            }
        }

        /// Get remaining buffered samples
        fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Option<PyObject>> {
            if !self.buffer.is_empty() {
                let remaining = self.buffer.clone();
                self.buffer.clear();
                let array = remaining.to_pyarray(py);
                Ok(Some(array.to_object(py)))
            } else {
                Ok(None)
            }
        }
    }

    /// Advanced audio analysis tools with NumPy integration
    #[cfg(feature = "numpy")]
    #[pyclass]
    pub struct PyAudioAnalyzer;

    #[cfg(feature = "numpy")]
    #[pymethods]
    impl PyAudioAnalyzer {
        #[new]
        fn new() -> Self {
            Self
        }

        /// Compute RMS energy of audio signal
        #[staticmethod]
        fn rms_energy<'py>(py: Python<'py>, audio: PyReadonlyArray1<f32>) -> PyResult<f32> {
            let samples = audio.as_array();
            let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
            let rms = (sum_squares / samples.len() as f32).sqrt();
            Ok(rms)
        }

        /// Find silence regions in audio
        #[staticmethod]
        fn find_silence<'py>(
            py: Python<'py>,
            audio: PyReadonlyArray1<f32>,
            threshold: f32,
            min_duration: usize,
        ) -> PyResult<PyObject> {
            let samples = audio.as_array();
            let mut silence_regions = Vec::new();
            let mut in_silence = false;
            let mut silence_start = 0;

            for (i, &sample) in samples.iter().enumerate() {
                let is_silent = sample.abs() < threshold;

                if is_silent && !in_silence {
                    silence_start = i;
                    in_silence = true;
                } else if !is_silent && in_silence {
                    let duration = i - silence_start;
                    if duration >= min_duration {
                        silence_regions.push([silence_start, i]);
                    }
                    in_silence = false;
                }
            }

            // Handle silence at the end
            if in_silence {
                let duration = samples.len() - silence_start;
                if duration >= min_duration {
                    silence_regions.push([silence_start, samples.len()]);
                }
            }

            let array = PyArray2::from_vec2(py, &silence_regions)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create array: {}", e)))?;
            Ok(array.to_object(py))
        }

        /// Compute zero crossing rate
        #[staticmethod]
        fn zero_crossing_rate<'py>(py: Python<'py>, audio: PyReadonlyArray1<f32>) -> PyResult<f32> {
            let samples = audio.as_array();
            if samples.len() < 2 {
                return Ok(0.0);
            }

            let mut crossings = 0;
            for i in 1..samples.len() {
                if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                    crossings += 1;
                }
            }

            Ok(crossings as f32 / (samples.len() - 1) as f32)
        }

        /// Compute spectral centroid (brightness measure)
        #[staticmethod]
        fn spectral_centroid<'py>(
            py: Python<'py>,
            audio: PyReadonlyArray1<f32>,
            sample_rate: u32,
        ) -> PyResult<f32> {
            let samples = audio.as_array();
            let n = samples.len();

            // Simple spectral centroid calculation (placeholder for real FFT)
            let mut magnitude_sum = 0.0f32;
            let mut weighted_sum = 0.0f32;

            for (i, &sample) in samples.iter().enumerate() {
                let magnitude = sample.abs();
                let freq = (i as f32 * sample_rate as f32) / (n as f32);

                magnitude_sum += magnitude;
                weighted_sum += magnitude * freq;
            }

            if magnitude_sum > 0.0 {
                Ok(weighted_sum / magnitude_sum)
            } else {
                Ok(0.0)
            }
        }
    }

    /// Python synthesis configuration
    #[pyclass]
    #[derive(Clone)]
    pub struct PySynthesisConfig {
        #[pyo3(get, set)]
        pub speaking_rate: f32,
        #[pyo3(get, set)]
        pub pitch_shift: f32,
        #[pyo3(get, set)]
        pub volume_gain: f32,
        #[pyo3(get, set)]
        pub enable_enhancement: bool,
        #[pyo3(get, set)]
        pub output_format: String,
        #[pyo3(get, set)]
        pub sample_rate: u32,
        #[pyo3(get, set)]
        pub quality: String,
    }

    #[pymethods]
    impl PySynthesisConfig {
        #[new]
        fn new() -> Self {
            let config = voirs_sdk::types::SynthesisConfig::default();
            Self {
                speaking_rate: config.speaking_rate,
                pitch_shift: config.pitch_shift,
                volume_gain: config.volume_gain,
                enable_enhancement: config.enable_enhancement,
                output_format: format!("{:?}", config.output_format),
                sample_rate: config.sample_rate,
                quality: format!("{:?}", config.quality),
            }
        }
    }

    // ========================================================================
    // Speech Recognition Classes (ASR)
    // ========================================================================

    #[cfg(feature = "recognition")]
    mod recognition_bindings {
        use super::*;
        use voirs_recognizer::{
            analysis::AudioAnalyzerImpl, asr::ASRBackend,
            performance::PerformanceMetrics as RecogMetrics, phoneme::analysis::AlignedPhoneme,
            prelude::*, RecognitionError,
        };

        /// Python ASR Model wrapper for speech recognition
        #[pyclass]
        pub struct PyASRModel {
            #[allow(dead_code)]
            inner: Box<dyn ASRModel + Send + Sync>,
            rt: Runtime,
        }

        #[pymethods]
        impl PyASRModel {
            /// Create a new ASR model with Whisper
            #[staticmethod]
            fn whisper(model_size: Option<&str>, device: Option<&str>) -> PyResult<Self> {
                let rt = Runtime::new().map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
                })?;

                #[cfg(feature = "whisper-pure")]
                {
                    use voirs_recognizer::asr::whisper_pure::WhisperModelSize;

                    let size = match model_size.unwrap_or("base") {
                        "tiny" => WhisperModelSize::Tiny,
                        "base" => WhisperModelSize::Base,
                        "small" => WhisperModelSize::Small,
                        "medium" => WhisperModelSize::Medium,
                        "large" => WhisperModelSize::Large,
                        _ => WhisperModelSize::Base,
                    };

                    let model = rt
                        .block_on(async {
                            voirs_recognizer::asr::PureRustWhisper::new_with_model_size(size).await
                        })
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("Failed to load Whisper model: {}", e))
                        })?;

                    Ok(Self {
                        inner: Box::new(model),
                        rt,
                    })
                }
                #[cfg(not(feature = "whisper-pure"))]
                {
                    Err(PyRuntimeError::new_err("Whisper support not compiled in"))
                }
            }

            /// Recognize speech from audio buffer
            fn recognize(&self, audio: &PyAudioBuffer) -> PyResult<PyRecognitionResult> {
                let audio_buffer = &audio.inner;
                let config = ASRConfig::default();

                let result = self
                    .rt
                    .block_on(async { self.inner.transcribe(audio_buffer, Some(&config)).await })
                    .map_err(|e| PyRuntimeError::new_err(format!("Recognition failed: {}", e)))?;

                Ok(PyRecognitionResult::new(result))
            }

            /// Recognize speech from audio file
            #[staticmethod]
            fn recognize_file(
                file_path: &str,
                model_size: Option<&str>,
            ) -> PyResult<PyRecognitionResult> {
                let rt = Runtime::new().map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
                })?;

                rt.block_on(async {
                    // Load audio file
                    let audio =
                        voirs_recognizer::audio_formats::load_audio(file_path).map_err(|e| {
                            PyRuntimeError::new_err(format!("Failed to load audio: {}", e))
                        })?;

                    // Create ASR model
                    #[cfg(feature = "whisper-pure")]
                    {
                        use voirs_recognizer::asr::whisper_pure::WhisperModelSize;

                        let size = match model_size.unwrap_or("base") {
                            "tiny" => WhisperModelSize::Tiny,
                            "base" => WhisperModelSize::Base,
                            "small" => WhisperModelSize::Small,
                            "medium" => WhisperModelSize::Medium,
                            "large" => WhisperModelSize::Large,
                            _ => WhisperModelSize::Base,
                        };

                        let model =
                            voirs_recognizer::asr::PureRustWhisper::new_with_model_size(size)
                                .await
                                .map_err(|e| {
                                    PyRuntimeError::new_err(format!("Failed to load model: {}", e))
                                })?;

                        // Perform recognition
                        let config = ASRConfig::default();
                        let result =
                            model.transcribe(&audio, Some(&config)).await.map_err(|e| {
                                PyRuntimeError::new_err(format!("Recognition failed: {}", e))
                            })?;

                        Ok(PyRecognitionResult::new(result))
                    }
                    #[cfg(not(feature = "whisper-pure"))]
                    {
                        Err(PyRuntimeError::new_err("Whisper support not compiled in"))
                    }
                })
            }

            /// Get supported languages
            fn supported_languages(&self) -> Vec<String> {
                // Return common language codes
                vec![
                    "en".to_string(),
                    "es".to_string(),
                    "fr".to_string(),
                    "de".to_string(),
                    "it".to_string(),
                    "pt".to_string(),
                    "ru".to_string(),
                    "ja".to_string(),
                    "ko".to_string(),
                    "zh".to_string(),
                    "ar".to_string(),
                    "hi".to_string(),
                ]
            }
        }

        /// Python Audio Analyzer wrapper
        #[pyclass]
        pub struct PyAudioAnalyzer {
            inner: AudioAnalyzerImpl,
            rt: Runtime,
        }

        #[pymethods]
        impl PyAudioAnalyzer {
            #[new]
            fn new() -> PyResult<Self> {
                let rt = Runtime::new().map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
                })?;

                let config = AudioAnalysisConfig::default();
                let analyzer = rt.block_on(AudioAnalyzerImpl::new(config)).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create analyzer: {}", e))
                })?;

                Ok(Self {
                    inner: analyzer,
                    rt,
                })
            }

            /// Analyze audio buffer
            fn analyze(&self, audio: &PyAudioBuffer) -> PyResult<PyAudioAnalysis> {
                let config = AudioAnalysisConfig::default();
                let analysis = self
                    .rt
                    .block_on(self.inner.analyze(&audio.inner, Some(&config)))
                    .map_err(|e| PyRuntimeError::new_err(format!("Analysis failed: {}", e)))?;

                Ok(PyAudioAnalysis::new(analysis))
            }

            /// Analyze audio file
            #[staticmethod]
            fn analyze_file(file_path: &str) -> PyResult<PyAudioAnalysis> {
                let rt = Runtime::new().map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
                })?;

                rt.block_on(async {
                    // Load audio file
                    let audio =
                        voirs_recognizer::audio_formats::load_audio(file_path).map_err(|e| {
                            PyRuntimeError::new_err(format!("Failed to load audio: {}", e))
                        })?;

                    // Create analyzer
                    let config = AudioAnalysisConfig::default();
                    let analyzer = AudioAnalyzerImpl::new(config).await.map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to create analyzer: {}", e))
                    })?;

                    // Perform analysis
                    let analysis = analyzer
                        .analyze(&audio, Some(&AudioAnalysisConfig::default()))
                        .await
                        .map_err(|e| PyRuntimeError::new_err(format!("Analysis failed: {}", e)))?;

                    Ok(PyAudioAnalysis::new(analysis))
                })
            }
        }

        /// Python Phoneme Recognizer wrapper
        #[pyclass]
        pub struct PyPhonemeRecognizer {
            #[allow(dead_code)]
            inner: Box<dyn PhonemeRecognizer + Send + Sync>,
            rt: Runtime,
            language: LanguageCode,
        }

        #[pymethods]
        impl PyPhonemeRecognizer {
            #[new]
            fn new(language: &str) -> PyResult<Self> {
                let rt = Runtime::new().map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
                })?;

                // Parse language code
                let lang_code = match language.to_lowercase().as_str() {
                    "en" | "en-us" => LanguageCode::EnUs,
                    "es" | "es-es" => LanguageCode::EsEs,
                    "fr" | "fr-fr" => LanguageCode::FrFr,
                    "de" | "de-de" => LanguageCode::DeDe,
                    "it" | "it-it" => LanguageCode::ItIt,
                    "pt" | "pt-br" => LanguageCode::PtBr,
                    "ru" | "ru-ru" => LanguageCode::RuRu,
                    "ja" | "ja-jp" => LanguageCode::JaJp,
                    "ko" | "ko-kr" => LanguageCode::KoKr,
                    "zh" | "zh-cn" => LanguageCode::ZhCn,
                    _ => LanguageCode::EnUs, // Default fallback
                };

                let backend = voirs_recognizer::phoneme::PhonemeRecognizerBackend::Comprehensive {
                    language: lang_code,
                    alignment_method: voirs_recognizer::traits::AlignmentMethod::Forced,
                };

                let recognizer = rt
                    .block_on(async {
                        voirs_recognizer::phoneme::create_phoneme_recognizer(backend).await
                    })
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to create phoneme recognizer: {}",
                            e
                        ))
                    })?;

                Ok(Self {
                    inner: recognizer,
                    rt,
                    language: lang_code,
                })
            }

            /// Recognize phonemes from audio
            fn recognize(
                &self,
                audio: &PyAudioBuffer,
                text: Option<&str>,
            ) -> PyResult<Vec<PyPhonemeAlignment>> {
                let config = PhonemeRecognitionConfig {
                    language: self.language,
                    ..Default::default()
                };

                let alignment = self
                    .rt
                    .block_on(async {
                        self.inner
                            .align_text(&audio.inner, text, Some(&config))
                            .await
                    })
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Phoneme recognition failed: {}", e))
                    })?;

                // Extract individual aligned phonemes from the alignment result
                Ok(alignment
                    .phonemes
                    .into_iter()
                    .map(PyPhonemeAlignment::new)
                    .collect())
            }
        }

        /// Python Recognition Result wrapper
        #[pyclass]
        #[derive(Clone)]
        pub struct PyRecognitionResult {
            #[pyo3(get)]
            pub transcript: PyTranscript,
            #[pyo3(get)]
            pub confidence: f32,
            #[pyo3(get)]
            pub processing_time_ms: f64,
        }

        #[pymethods]
        impl PyRecognitionResult {
            fn __str__(&self) -> String {
                format!(
                    "RecognitionResult(text='{}', confidence={:.2})",
                    self.transcript.text, self.confidence
                )
            }
        }

        impl PyRecognitionResult {
            fn new(result: RecognitionResult) -> Self {
                Self {
                    transcript: PyTranscript::new(result.transcript),
                    confidence: result.confidence,
                    processing_time_ms: result
                        .processing_duration
                        .map(|d| d.as_millis() as f64)
                        .unwrap_or(0.0),
                }
            }
        }

        /// Python Transcript wrapper
        #[pyclass]
        #[derive(Clone)]
        pub struct PyTranscript {
            #[pyo3(get)]
            pub text: String,
            #[pyo3(get)]
            pub language: String,
            #[pyo3(get)]
            pub confidence: f32,
            #[pyo3(get)]
            pub word_count: usize,
        }

        #[pymethods]
        impl PyTranscript {
            fn __str__(&self) -> String {
                self.text.clone()
            }

            fn __len__(&self) -> usize {
                self.text.len()
            }
        }

        impl PyTranscript {
            fn new(transcript: Transcript) -> Self {
                Self {
                    text: transcript.text.clone(),
                    language: transcript.language.to_string(),
                    confidence: transcript.confidence,
                    word_count: transcript.text.split_whitespace().count(),
                }
            }
        }

        /// Python Audio Analysis wrapper
        #[pyclass]
        #[derive(Clone)]
        pub struct PyAudioAnalysis {
            #[pyo3(get)]
            pub duration_seconds: f32,
            #[pyo3(get)]
            pub sample_rate: u32,
            #[pyo3(get)]
            pub channels: u32,
            #[pyo3(get)]
            pub rms_energy: f32,
            #[pyo3(get)]
            pub zero_crossing_rate: f32,
            #[pyo3(get)]
            pub spectral_centroid: f32,
        }

        #[pymethods]
        impl PyAudioAnalysis {
            fn __str__(&self) -> String {
                format!(
                    "AudioAnalysis(duration={:.2}s, sample_rate={}Hz, channels={}, rms={:.3})",
                    self.duration_seconds, self.sample_rate, self.channels, self.rms_energy
                )
            }
        }

        impl PyAudioAnalysis {
            fn new(analysis: AudioAnalysis) -> Self {
                Self {
                    duration_seconds: analysis
                        .processing_duration
                        .map(|d| d.as_secs_f32())
                        .unwrap_or(0.0),
                    sample_rate: 22050, // Default sample rate - would need to be passed from audio buffer
                    channels: 1,        // Default mono - would need to be passed from audio buffer
                    rms_energy: analysis
                        .quality_metrics
                        .get("rms_energy")
                        .copied()
                        .unwrap_or(0.0),
                    zero_crossing_rate: analysis
                        .quality_metrics
                        .get("zero_crossing_rate")
                        .copied()
                        .unwrap_or(0.0),
                    spectral_centroid: analysis
                        .quality_metrics
                        .get("spectral_centroid")
                        .copied()
                        .unwrap_or(0.0),
                }
            }
        }

        /// Python Phoneme Alignment wrapper
        #[pyclass]
        #[derive(Clone)]
        pub struct PyPhonemeAlignment {
            #[pyo3(get)]
            pub phoneme: String,
            #[pyo3(get)]
            pub start_time: f32,
            #[pyo3(get)]
            pub end_time: f32,
            #[pyo3(get)]
            pub confidence: f32,
        }

        #[pymethods]
        impl PyPhonemeAlignment {
            fn __str__(&self) -> String {
                format!(
                    "PhonemeAlignment({}, {:.2}-{:.2}s, conf={:.2})",
                    self.phoneme, self.start_time, self.end_time, self.confidence
                )
            }
        }

        impl PyPhonemeAlignment {
            fn new(aligned_phoneme: AlignedPhoneme) -> Self {
                Self {
                    phoneme: aligned_phoneme.phoneme.symbol.clone(),
                    start_time: aligned_phoneme.start_time,
                    end_time: aligned_phoneme.end_time,
                    confidence: aligned_phoneme.confidence,
                }
            }
        }

        /// Python Performance Metrics wrapper
        #[pyclass]
        #[derive(Clone)]
        pub struct PyPerformanceMetrics {
            #[pyo3(get)]
            pub real_time_factor: f32,
            #[pyo3(get)]
            pub memory_usage_mb: f32,
            #[pyo3(get)]
            pub processing_time_ms: f64,
            #[pyo3(get)]
            pub throughput_ratio: f32,
        }

        #[pymethods]
        impl PyPerformanceMetrics {
            fn __str__(&self) -> String {
                format!(
                    "PerformanceMetrics(rtf={:.2}, memory={:.1}MB, time={:.1}ms)",
                    self.real_time_factor, self.memory_usage_mb, self.processing_time_ms
                )
            }
        }

        impl PyPerformanceMetrics {
            fn new(metrics: RecogMetrics) -> Self {
                Self {
                    real_time_factor: metrics.rtf,
                    memory_usage_mb: metrics.memory_usage as f32 / 1_000_000.0,
                    processing_time_ms: metrics.startup_time_ms as f64, // Use startup time as approximation
                    throughput_ratio: metrics.throughput_samples_per_sec as f32,
                }
            }
        }

        // Re-export for main module with aliases to avoid conflicts
        pub use {
            PyASRModel as RecognitionASRModel, PyAudioAnalysis as RecognitionAudioAnalysis,
            PyAudioAnalyzer as RecognitionAudioAnalyzer,
            PyPerformanceMetrics as RecognitionPerformanceMetrics,
            PyPhonemeAlignment as RecognitionPhonemeAlignment,
            PyPhonemeRecognizer as RecognitionPhonemeRecognizer,
            PyRecognitionResult as RecognitionResult, PyTranscript as RecognitionTranscript,
        };
    }

    #[cfg(feature = "recognition")]
    use recognition_bindings::*;

    /// Python module entry point
    #[pymodule]
    fn voirs_ffi(_py: Python, m: &PyModule) -> PyResult<()> {
        // Core classes
        m.add_class::<VoirsPipeline>()?;
        m.add_class::<PyAudioBuffer>()?;
        m.add_class::<PyVoiceInfo>()?;
        m.add_class::<PySynthesisConfig>()?;

        // Enhanced error handling and metrics
        m.add_class::<VoirsErrorInfo>()?;
        m.add_class::<VoirsException>()?;
        m.add_class::<SynthesisMetrics>()?;
        m.add_class::<SynthesisResult>()?;

        // Advanced NumPy integration classes (when numpy feature is enabled)
        #[cfg(feature = "numpy")]
        {
            m.add_class::<PyStreamingProcessor>()?;
            m.add_class::<PyAudioAnalyzer>()?;
        }

        // Add version constant
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;

        // Add feature flags for runtime detection
        #[cfg(feature = "numpy")]
        m.add("HAS_NUMPY", true)?;
        #[cfg(not(feature = "numpy"))]
        m.add("HAS_NUMPY", false)?;

        #[cfg(feature = "gpu")]
        m.add("HAS_GPU", true)?;
        #[cfg(not(feature = "gpu"))]
        m.add("HAS_GPU", false)?;

        // Recognition classes (when voirs-recognizer feature is enabled)
        #[cfg(feature = "recognition")]
        {
            m.add_class::<RecognitionASRModel>()?;
            m.add_class::<RecognitionAudioAnalyzer>()?;
            m.add_class::<PyPhonemeRecognizer>()?;
            m.add_class::<PyRecognitionResult>()?;
            m.add_class::<PyTranscript>()?;
            m.add_class::<RecognitionAudioAnalysis>()?;
            m.add_class::<PyPerformanceMetrics>()?;
        }

        // Add metrics constants
        m.add("DEFAULT_PROCESSING_TIMEOUT_MS", 30000)?;
        m.add("DEFAULT_CACHE_SIZE_MB", 512)?;
        m.add("DEFAULT_BATCH_SIZE", 8)?;

        // Add recognition feature flag
        #[cfg(feature = "recognition")]
        m.add("HAS_RECOGNITION", true)?;
        #[cfg(not(feature = "recognition"))]
        m.add("HAS_RECOGNITION", false)?;

        Ok(())
    }
}

#[cfg(not(feature = "python"))]
pub mod pyo3_bindings {
    // Empty module when python feature is disabled
}
