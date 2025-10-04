//! Core C API functions for VoiRS speech recognition.

use super::error::{handle_error, VoirsErrorHandler};
use super::memory::{c_string_to_string, string_to_c_string, VoirsMemoryManager};
use super::types::{
    VoirsAudioFormat, VoirsAudioFormatType, VoirsCapabilities, VoirsError, VoirsPerformanceMetrics,
    VoirsProgressCallback, VoirsRecognitionConfig, VoirsRecognitionResult, VoirsRecognizer,
    VoirsSegment, VoirsStreamingCallback, VoirsStreamingConfig, VoirsVersion,
};
use crate::integration::PipelineBuilder;
use crate::integration::UnifiedVoirsPipeline;
use crate::{ASRConfig, AudioBuffer, LanguageCode, RecognitionResult};
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;

/// Internal representation of the recognizer
pub struct VoirsRecognizerInternal {
    /// pipeline
    pub pipeline: UnifiedVoirsPipeline,
    /// runtime
    pub runtime: Runtime,
    /// memory manager
    pub memory_manager: VoirsMemoryManager,
    /// error handler
    pub error_handler: VoirsErrorHandler,
    /// config
    pub config: VoirsRecognitionConfig,
    /// Streaming context for real-time processing
    pub streaming_context: Option<Arc<Mutex<super::streaming::StreamingContext>>>,
    /// Performance metrics tracking
    pub metrics: PerformanceMetrics,
}

/// Performance metrics tracking structure
#[derive(Debug, Default)]
/// Performance Metrics
pub struct PerformanceMetrics {
    /// total processing time ms
    pub total_processing_time_ms: f64,
    /// total audio duration s
    pub total_audio_duration_s: f64,
    /// processed chunks
    pub processed_chunks: usize,
    /// failed recognitions
    pub failed_recognitions: usize,
    /// peak processing time ms
    pub peak_processing_time_ms: f64,
    /// memory usage bytes
    pub memory_usage_bytes: usize,
}

/// Initialize the VoiRS recognizer library
///
/// This function must be called before any other VoiRS functions.
/// Returns VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_init() -> VoirsError {
    // Initialize logging if not already done
    let _ = env_logger::try_init();

    VoirsError::Success
}

/// Get the version information of the VoiRS library
///
/// Returns a pointer to a VoirsVersion structure that contains version information.
/// The caller should not free this pointer.
#[no_mangle]
pub extern "C" fn voirs_get_version() -> *const VoirsVersion {
    use std::sync::OnceLock;

    static VERSION: OnceLock<VoirsVersion> = OnceLock::new();

    let version = VERSION.get_or_init(|| {
        let version_str = std::ffi::CString::new(env!("CARGO_PKG_VERSION")).unwrap();
        let timestamp_str = std::ffi::CString::new("2025-07-20T00:00:00Z").unwrap();

        // Leak the strings to get stable pointers for C API
        VoirsVersion {
            major: 0,
            minor: 1,
            patch: 0,
            version_string: version_str.into_raw(),
            build_timestamp: timestamp_str.into_raw(),
        }
    });

    version
}

/// Create a new VoiRS recognizer instance
///
/// # Arguments
/// * `config` - Configuration for the recognizer. If null, default configuration will be used.
/// * `recognizer` - Output pointer to store the created recognizer instance
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_recognizer_create(
    config: *const VoirsRecognitionConfig,
    recognizer: *mut *mut VoirsRecognizer,
) -> VoirsError {
    if recognizer.is_null() {
        return VoirsError::NullPointer;
    }

    let config = if config.is_null() {
        VoirsRecognitionConfig::default()
    } else {
        unsafe { *config }
    };

    let result = std::panic::catch_unwind(|| {
        // Create Tokio runtime
        let runtime = match Runtime::new() {
            Ok(rt) => rt,
            Err(_) => return VoirsError::InitializationFailed,
        };

        // Build ASR pipeline
        let pipeline_result = runtime.block_on(async {
            let mut builder = PipelineBuilder::new();

            // Configure model
            if !config.model_name.is_null() {
                if let Ok(model_name) = c_string_to_string(config.model_name) {
                    builder = builder.with_model(&model_name);
                }
            }

            // Configure language
            if !config.language.is_null() {
                if let Ok(language_str) = c_string_to_string(config.language) {
                    if let Some(language) = parse_language_code(&language_str) {
                        builder = builder.with_language(language);
                    }
                }
            }

            // Configure sample rate
            builder = builder.with_sample_rate(config.sample_rate);

            // Configure ASR settings
            let mut asr_config = ASRConfig::default();
            asr_config.enable_vad = config.enable_vad;
            asr_config.confidence_threshold = config.confidence_threshold;
            asr_config.beam_size = config.beam_size;
            asr_config.temperature = config.temperature;

            builder = builder.with_config(asr_config);

            builder.build().await
        });

        let pipeline = match pipeline_result {
            Ok(p) => p,
            Err(_) => return VoirsError::InitializationFailed,
        };

        // Create internal recognizer
        let internal = Box::new(VoirsRecognizerInternal {
            pipeline,
            runtime,
            memory_manager: VoirsMemoryManager::new(),
            error_handler: VoirsErrorHandler::new(),
            config,
            streaming_context: None,
            metrics: PerformanceMetrics::default(),
        });

        unsafe {
            *recognizer = Box::into_raw(internal) as *mut VoirsRecognizer;
        }

        VoirsError::Success
    });

    match result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Destroy a VoiRS recognizer instance
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance to destroy
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
/// Item
pub extern "C" fn voirs_recognizer_destroy(recognizer: *mut VoirsRecognizer) -> VoirsError {
    if recognizer.is_null() {
        return VoirsError::NullPointer;
    }

    let result = std::panic::catch_unwind(|| {
        unsafe {
            let internal = Box::from_raw(recognizer as *mut VoirsRecognizerInternal);
            drop(internal);
        }
        VoirsError::Success
    });

    match result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Get the capabilities of the recognizer
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `capabilities` - Output pointer to store the capabilities
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_recognizer_get_capabilities(
    recognizer: *mut VoirsRecognizer,
    capabilities: *mut *const VoirsCapabilities,
) -> VoirsError {
    if recognizer.is_null() || capabilities.is_null() {
        return VoirsError::NullPointer;
    }

    let result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        // Get supported models and languages
        let models = vec![
            "whisper-tiny",
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large-v3",
        ];
        let languages = vec!["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"];

        let model_ptrs: Vec<*const c_char> = models
            .iter()
            .map(|s| internal.memory_manager.store_string(s))
            .collect();

        let language_ptrs: Vec<*const c_char> = languages
            .iter()
            .map(|s| internal.memory_manager.store_string(s))
            .collect();

        let caps = VoirsCapabilities {
            streaming: true,
            multilingual: true,
            vad: true,
            confidence_scoring: true,
            segment_timestamps: true,
            language_detection: true,
            supported_models_count: model_ptrs.len(),
            supported_models: internal.memory_manager.store_ptr_array(&model_ptrs),
            supported_languages_count: language_ptrs.len(),
            supported_languages: internal.memory_manager.store_ptr_array(&language_ptrs),
        };

        unsafe {
            *capabilities = internal.memory_manager.store_struct(caps);
        }

        VoirsError::Success
    });

    match result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Get performance metrics from the recognizer
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `metrics` - Output pointer to store the metrics
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_recognizer_get_metrics(
    recognizer: *mut VoirsRecognizer,
    metrics: *mut *const VoirsPerformanceMetrics,
) -> VoirsError {
    if recognizer.is_null() || metrics.is_null() {
        return VoirsError::NullPointer;
    }

    let result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        // Calculate real metrics from collected data
        let real_time_factor = if internal.metrics.total_audio_duration_s > 0.0 {
            (internal.metrics.total_processing_time_ms / 1000.0)
                / internal.metrics.total_audio_duration_s
        } else {
            0.0
        };

        let avg_processing_time_ms = if internal.metrics.processed_chunks > 0 {
            internal.metrics.total_processing_time_ms / internal.metrics.processed_chunks as f64
        } else {
            0.0
        };

        // Estimate memory usage (this could be more sophisticated)
        let estimated_memory = estimate_memory_usage();

        let perf_metrics = VoirsPerformanceMetrics {
            real_time_factor: real_time_factor as f32,
            avg_processing_time_ms: avg_processing_time_ms as f32,
            peak_processing_time_ms: internal.metrics.peak_processing_time_ms as f32,
            memory_usage_bytes: estimated_memory,
            processed_chunks: internal.metrics.processed_chunks,
            failed_recognitions: internal.metrics.failed_recognitions,
        };

        unsafe {
            *metrics = internal.memory_manager.store_struct(perf_metrics);
        }

        VoirsError::Success
    });

    match result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

/// Switch to a different model
///
/// # Arguments
/// * `recognizer` - Pointer to the recognizer instance
/// * `model_name` - Name of the model to switch to
///
/// # Returns
/// VoirsError::Success on success, or an error code on failure.
#[no_mangle]
pub extern "C" fn voirs_recognizer_switch_model(
    recognizer: *mut VoirsRecognizer,
    model_name: *const c_char,
) -> VoirsError {
    if recognizer.is_null() || model_name.is_null() {
        return VoirsError::NullPointer;
    }

    let result = std::panic::catch_unwind(|| {
        let internal = unsafe { &mut *(recognizer as *mut VoirsRecognizerInternal) };

        let model_name_str = match c_string_to_string(model_name) {
            Ok(s) => s,
            Err(_) => return VoirsError::InvalidArgument,
        };

        // Implement model switching by rebuilding the pipeline
        let switch_result = internal.runtime.block_on(async {
            let mut builder = PipelineBuilder::new();
            builder = builder.with_model(&model_name_str);

            // Preserve other configuration
            if !internal.config.language.is_null() {
                if let Ok(language_str) = c_string_to_string(internal.config.language) {
                    if let Some(language) = parse_language_code(&language_str) {
                        builder = builder.with_language(language);
                    }
                }
            }

            builder = builder.with_sample_rate(internal.config.sample_rate);

            let mut asr_config = ASRConfig::default();
            asr_config.enable_vad = internal.config.enable_vad;
            asr_config.confidence_threshold = internal.config.confidence_threshold;
            asr_config.beam_size = internal.config.beam_size;
            asr_config.temperature = internal.config.temperature;
            builder = builder.with_config(asr_config);

            builder.build().await
        });

        match switch_result {
            Ok(new_pipeline) => {
                internal.pipeline = new_pipeline;
                internal.config.model_name = string_to_c_string(&model_name_str);
                VoirsError::Success
            }
            Err(_) => VoirsError::ModelLoadFailed,
        }
    });

    match result {
        Ok(error) => error,
        Err(_) => VoirsError::InternalError,
    }
}

fn parse_language_code(language_str: &str) -> Option<LanguageCode> {
    match language_str.to_lowercase().as_str() {
        "en" | "en-us" | "en_us" => Some(LanguageCode::EnUs),
        "en-gb" | "en_gb" => Some(LanguageCode::EnGb),
        "ja" | "ja-jp" | "ja_jp" => Some(LanguageCode::JaJp),
        "es" | "es-es" | "es_es" => Some(LanguageCode::EsEs),
        "es-mx" | "es_mx" => Some(LanguageCode::EsMx),
        "fr" | "fr-fr" | "fr_fr" => Some(LanguageCode::FrFr),
        "de" | "de-de" | "de_de" => Some(LanguageCode::DeDe),
        "zh" | "zh-cn" | "zh_cn" => Some(LanguageCode::ZhCn),
        "pt" | "pt-br" | "pt_br" => Some(LanguageCode::PtBr),
        "ru" | "ru-ru" | "ru_ru" => Some(LanguageCode::RuRu),
        "it" | "it-it" | "it_it" => Some(LanguageCode::ItIt),
        "ko" | "ko-kr" | "ko_kr" => Some(LanguageCode::KoKr),
        "nl" | "nl-nl" | "nl_nl" => Some(LanguageCode::NlNl),
        "sv" | "sv-se" | "sv_se" => Some(LanguageCode::SvSe),
        "no" | "no-no" | "no_no" => Some(LanguageCode::NoNo),
        "da" | "da-dk" | "da_dk" => Some(LanguageCode::DaDk),
        _ => None,
    }
}

/// Estimate current memory usage
fn estimate_memory_usage() -> usize {
    #[cfg(target_os = "linux")]
    {
        // Read from /proc/self/status on Linux
        if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // macOS memory estimation
        return 128 * 1024 * 1024; // 128MB typical estimate
    }

    #[cfg(target_os = "windows")]
    {
        // Windows memory estimation
        return 128 * 1024 * 1024; // 128MB typical estimate
    }

    // Fallback for other platforms
    64 * 1024 * 1024 // 64MB conservative estimate
}
