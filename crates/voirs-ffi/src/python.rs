//! Python bindings for VoiRS using PyO3.

#[cfg(feature = "python")]
pub mod pyo3_bindings {
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use std::sync::Arc;
    use tokio::runtime::Runtime;
    use voirs::{VoirsPipeline as SdkPipeline, AudioBuffer, error::VoirsError};
    use crate::{VoirsQualityLevel, VoirsAudioFormat};
    
    /// Python VoiRS Pipeline wrapper
    #[pyclass]
    pub struct VoirsPipeline {
        inner: Arc<SdkPipeline>,
        rt: Runtime,
    }
    
    #[pymethods]
    impl VoirsPipeline {
        #[new]
        fn new() -> PyResult<Self> {
            let rt = Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;
            
            let inner = rt.block_on(SdkPipeline::builder().build())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create pipeline: {}", e)))?;
            
            Ok(Self {
                inner: Arc::new(inner),
                rt,
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
                builder = builder.with_device(dev);
            }
            
            let inner = rt.block_on(builder.build())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create pipeline: {}", e)))?;
            
            Ok(Self {
                inner: Arc::new(inner),
                rt,
            })
        }
        
        /// Synthesize text to audio
        fn synthesize(&self, text: &str) -> PyResult<PyAudioBuffer> {
            let audio = self.rt.block_on(self.inner.synthesize(text))
                .map_err(|e| PyRuntimeError::new_err(format!("Synthesis failed: {}", e)))?;
            
            Ok(PyAudioBuffer::new(audio))
        }
        
        /// Synthesize SSML to audio
        fn synthesize_ssml(&self, ssml: &str) -> PyResult<PyAudioBuffer> {
            let audio = self.rt.block_on(self.inner.synthesize_ssml(ssml))
                .map_err(|e| PyRuntimeError::new_err(format!("SSML synthesis failed: {}", e)))?;
            
            Ok(PyAudioBuffer::new(audio))
        }
        
        /// Set the voice for synthesis
        fn set_voice(&self, voice_id: &str) -> PyResult<()> {
            self.rt.block_on(self.inner.set_voice(voice_id))
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to set voice: {}", e)))?;
            
            Ok(())
        }
        
        /// Get the current voice
        fn get_voice(&self) -> Option<String> {
            self.inner.current_voice().map(|v| v.id.clone())
        }
        
        /// List available voices
        fn list_voices(&self) -> PyResult<Vec<PyVoiceInfo>> {
            let voices = self.rt.block_on(self.inner.list_voices())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to list voices: {}", e)))?;
            
            Ok(voices.into_iter().map(PyVoiceInfo::from).collect())
        }
        
        /// Get the library version
        #[staticmethod]
        fn version() -> &'static str {
            env!("CARGO_PKG_VERSION")
        }
    }
    
    /// Python AudioBuffer wrapper
    #[pyclass]
    pub struct PyAudioBuffer {
        inner: AudioBuffer,
    }
    
    #[pymethods]
    impl PyAudioBuffer {
        /// Get the audio samples as bytes
        fn samples<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
            let samples = self.inner.samples();
            let bytes = samples.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>();
            Ok(PyBytes::new(py, &bytes))
        }
        
        /// Get the audio samples as a list of floats
        fn samples_as_list(&self) -> Vec<f32> {
            self.inner.samples().to_vec()
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
                "wav" => voirs::types::AudioFormat::Wav,
                "flac" => voirs::types::AudioFormat::Flac,
                "mp3" => voirs::types::AudioFormat::Mp3,
                "opus" => voirs::types::AudioFormat::Opus,
                "ogg" => voirs::types::AudioFormat::Ogg,
                _ => return Err(PyValueError::new_err("Unsupported audio format")),
            };
            
            self.inner.save(Path::new(path), audio_format)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to save audio: {}", e)))?;
            
            Ok(())
        }
    }
    
    impl PyAudioBuffer {
        fn new(audio: AudioBuffer) -> Self {
            Self { inner: audio }
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
    
    impl From<voirs::VoiceInfo> for PyVoiceInfo {
        fn from(voice: voirs::VoiceInfo) -> Self {
            Self {
                id: voice.id,
                name: voice.name,
                language: voice.language.to_string(),
                quality: format!("{:?}", voice.quality),
                is_available: voice.is_available,
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
            let config = voirs::types::SynthesisConfig::default();
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
    
    /// Python module entry point
    #[pymodule]
    fn voirs_ffi(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<VoirsPipeline>()?;
        m.add_class::<PyAudioBuffer>()?;
        m.add_class::<PyVoiceInfo>()?;
        m.add_class::<PySynthesisConfig>()?;
        
        // Add version constant
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        
        Ok(())
    }
}

#[cfg(not(feature = "python"))]
pub mod pyo3_bindings {
    // Empty module when python feature is disabled
}