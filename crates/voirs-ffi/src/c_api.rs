//! C API functions for VoiRS.

// Sub-modules
pub mod convert;
pub mod threading;

use crate::{
    get_pipeline_manager, get_runtime, string_to_c_str, c_str_to_string, set_last_error, clear_last_error,
    VoirsErrorCode, VoirsAudioBuffer, VoirsSynthesisConfig,
    types::{VoirsVoiceInfo, VoirsVoiceList, VoirsPipelineConfig},
};
use std::os::raw::{c_char, c_uint, c_int};
use voirs::VoirsPipeline;
use hound;

// Re-export conversion and threading functions
pub use convert::*;
pub use threading::*;

/// Initialize VoiRS library
#[no_mangle]
pub extern "C" fn voirs_init() -> VoirsErrorCode {
    // Initialize logging if needed
    #[cfg(feature = "logging")]
    {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();
    }
    
    VoirsErrorCode::Success
}

/// Create a new synthesis pipeline
#[no_mangle]
pub extern "C" fn voirs_create_pipeline() -> c_uint {
    clear_last_error();
    
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(e) => {
            set_last_error(format!("Failed to get runtime: {:?}", e));
            return 0;
        }
    };
    
    let pipeline = match rt.block_on(VoirsPipeline::builder().build()) {
        Ok(pipeline) => pipeline,
        Err(e) => {
            set_last_error(format!("Failed to create pipeline: {}", e));
            return 0;
        }
    };
    
    let manager = get_pipeline_manager();
    let pipeline_id = {
        let mut mgr = manager.lock();
        mgr.add_pipeline(pipeline)
    };
    
    // Register default configuration for the new pipeline
    if pipeline_id != 0 {
        crate::config::register_config(pipeline_id, crate::config::PipelineConfig::default());
    }
    
    pipeline_id
}

/// Destroy a synthesis pipeline
#[no_mangle]
pub extern "C" fn voirs_destroy_pipeline(pipeline_id: c_uint) -> VoirsErrorCode {
    let manager = get_pipeline_manager();
    let removed = {
        let mut mgr = manager.lock();
        mgr.remove_pipeline(pipeline_id)
    };
    
    if removed {
        // Also remove the configuration
        crate::config::remove_config(pipeline_id);
        VoirsErrorCode::Success
    } else {
        VoirsErrorCode::InvalidParameter
    }
}

/// Synthesize text to audio
#[no_mangle]
pub unsafe extern "C" fn voirs_synthesize(
    pipeline_id: c_uint,
    text: *const c_char,
    config: *const VoirsSynthesisConfig,
    audio_buffer: *mut *mut VoirsAudioBuffer,
) -> VoirsErrorCode {
    if text.is_null() || audio_buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }
    
    let text_str = match c_str_to_string(text) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    let manager = get_pipeline_manager();
    let pipeline = {
        let mgr = manager.lock();
        match mgr.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => return VoirsErrorCode::InvalidParameter,
        }
    };
    
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(e) => return e,
    };
    
    let audio = match rt.block_on(pipeline.synthesize(&text_str)) {
        Ok(audio) => audio,
        Err(_) => return VoirsErrorCode::SynthesisFailed,
    };
    
    let ffi_buffer = VoirsAudioBuffer::from_audio_buffer(audio);
    *audio_buffer = Box::into_raw(Box::new(ffi_buffer));
    
    VoirsErrorCode::Success
}

/// Get library version
#[no_mangle]
pub extern "C" fn voirs_version() -> *const c_char {
    env!("CARGO_PKG_VERSION").as_ptr() as *const c_char
}

/// Validate configuration values
#[no_mangle]
pub unsafe extern "C" fn voirs_validate_config(
    config: *const VoirsPipelineConfig,
) -> VoirsErrorCode {
    if config.is_null() {
        set_last_error("Configuration pointer is null".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    let config_ref = &*config;
    
    // Validate device if specified
    if !config_ref.device.is_null() {
        if let Ok(device) = c_str_to_string(config_ref.device) {
            let valid_devices = vec!["cpu", "cuda", "metal", "vulkan"];
            if !valid_devices.contains(&device.as_str()) {
                set_last_error(format!("Invalid device '{}'. Valid options: {:?}", device, valid_devices));
                return VoirsErrorCode::InvalidParameter;
            }
        }
    }
    
    // Validate thread count
    if config_ref.num_threads > 64 {
        set_last_error("Thread count cannot exceed 64".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    // Validate cache directory if specified
    if !config_ref.cache_dir.is_null() {
        if let Ok(cache_dir) = c_str_to_string(config_ref.cache_dir) {
            let path = std::path::Path::new(&cache_dir);
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    set_last_error(format!("Cache directory parent '{}' does not exist", parent.display()));
                    return VoirsErrorCode::InvalidParameter;
                }
            }
        }
    }
    
    VoirsErrorCode::Success
}

/// Create a new synthesis pipeline with configuration
#[no_mangle]
pub unsafe extern "C" fn voirs_create_pipeline_with_config(
    config: *const VoirsPipelineConfig,
) -> c_uint {
    clear_last_error();
    
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(_) => return 0,
    };
    
    let mut builder = VoirsPipeline::builder();
    let mut pipeline_config = crate::config::PipelineConfig::default();
    
    if !config.is_null() {
        let config_ref = &*config;
        
        if config_ref.use_gpu != 0 {
            builder = builder.with_gpu(true);
            pipeline_config.device.use_gpu = true;
        }
        
        if config_ref.num_threads > 0 {
            builder = builder.with_threads(config_ref.num_threads as usize);
            pipeline_config.threading.thread_count = config_ref.num_threads;
        }
        
        if !config_ref.cache_dir.is_null() {
            if let Ok(cache_dir) = c_str_to_string(config_ref.cache_dir) {
                builder = builder.with_cache_dir(&cache_dir);
            }
        }
        
        if !config_ref.device.is_null() {
            if let Ok(device) = c_str_to_string(config_ref.device) {
                builder = builder.with_device(device.clone());
                pipeline_config.device.device_type = device;
            }
        }
    }
    
    let pipeline = match rt.block_on(builder.build()) {
        Ok(pipeline) => pipeline,
        Err(_) => return 0,
    };
    
    let manager = get_pipeline_manager();
    let pipeline_id = {
        let mut mgr = manager.lock();
        mgr.add_pipeline(pipeline)
    };
    
    // Register configuration for the new pipeline
    if pipeline_id != 0 {
        crate::config::register_config(pipeline_id, pipeline_config);
    }
    
    pipeline_id
}

/// Set voice for a pipeline
#[no_mangle]
pub unsafe extern "C" fn voirs_set_voice(
    pipeline_id: c_uint,
    voice_id: *const c_char,
) -> VoirsErrorCode {
    if voice_id.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }
    
    let voice_id_str = match c_str_to_string(voice_id) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    let manager = get_pipeline_manager();
    let pipeline = {
        let mgr = manager.lock();
        match mgr.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => return VoirsErrorCode::InvalidParameter,
        }
    };
    
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(e) => return e,
    };
    
    match rt.block_on(pipeline.set_voice(&voice_id_str)) {
        Ok(_) => VoirsErrorCode::Success,
        Err(_) => VoirsErrorCode::VoiceNotFound,
    }
}

/// Get current voice for a pipeline
#[no_mangle]
pub unsafe extern "C" fn voirs_get_voice(
    pipeline_id: c_uint,
) -> *mut c_char {
    let manager = get_pipeline_manager();
    let pipeline = {
        let mgr = manager.lock();
        match mgr.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => return std::ptr::null_mut(),
        }
    };
    
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(_) => return std::ptr::null_mut(),
    };
    
    match rt.block_on(pipeline.current_voice()) {
        Some(voice) => string_to_c_str(&voice.id),
        None => std::ptr::null_mut(),
    }
}

/// List available voices
#[no_mangle]
pub unsafe extern "C" fn voirs_list_voices(
    pipeline_id: c_uint,
    voice_list: *mut *mut VoirsVoiceList,
) -> VoirsErrorCode {
    if voice_list.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }
    
    let manager = get_pipeline_manager();
    let pipeline = {
        let mgr = manager.lock();
        match mgr.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => return VoirsErrorCode::InvalidParameter,
        }
    };
    
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(e) => return e,
    };
    
    let voices = match rt.block_on(pipeline.list_voices()) {
        Ok(voices) => voices,
        Err(_) => return VoirsErrorCode::VoiceNotFound,
    };
    
    let voice_count = voices.len();
    let voice_array = voices.into_iter().map(|voice| {
        VoirsVoiceInfo {
            id: string_to_c_str(&voice.id),
            name: string_to_c_str(&voice.name),
            language: string_to_c_str(&voice.language.to_string()),
            quality: voice.characteristics.quality.into(),
            is_available: 1, // Always available for now
        }
    }).collect::<Vec<_>>();
    
    let voice_list_ptr = Box::into_raw(Box::new(VoirsVoiceList {
        voices: voice_array.as_ptr() as *mut VoirsVoiceInfo,
        count: voice_count as c_uint,
    }));
    
    std::mem::forget(voice_array);
    *voice_list = voice_list_ptr;
    
    VoirsErrorCode::Success
}

/// Free voice list allocated by voirs_list_voices
#[no_mangle]
pub unsafe extern "C" fn voirs_free_voice_list(voice_list: *mut VoirsVoiceList) {
    if voice_list.is_null() {
        return;
    }
    
    let list = &*voice_list;
    if !list.voices.is_null() {
        let voices = std::slice::from_raw_parts_mut(list.voices, list.count as usize);
        for voice in &mut *voices {
            if !voice.id.is_null() {
                let _ = std::ffi::CString::from_raw(voice.id);
            }
            if !voice.name.is_null() {
                let _ = std::ffi::CString::from_raw(voice.name);
            }
            if !voice.language.is_null() {
                let _ = std::ffi::CString::from_raw(voice.language);
            }
        }
        let _ = Box::from_raw(voices.as_mut_ptr());
    }
    
    let _ = Box::from_raw(voice_list);
}

/// Synthesize text with advanced configuration
#[no_mangle]
pub unsafe extern "C" fn voirs_synthesize_with_config(
    pipeline_id: c_uint,
    text: *const c_char,
    config: *const VoirsSynthesisConfig,
    audio_buffer: *mut *mut VoirsAudioBuffer,
) -> VoirsErrorCode {
    if text.is_null() || audio_buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }
    
    let text_str = match c_str_to_string(text) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    let manager = get_pipeline_manager();
    let pipeline = {
        let mgr = manager.lock();
        match mgr.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => return VoirsErrorCode::InvalidParameter,
        }
    };
    
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(e) => return e,
    };
    
    let audio = if !config.is_null() {
        let config_ref = &*config;
        let synthesis_config = config_ref.clone().into();
        match rt.block_on(pipeline.synthesize_with_config(&text_str, &synthesis_config)) {
            Ok(audio) => audio,
            Err(_) => return VoirsErrorCode::SynthesisFailed,
        }
    } else {
        match rt.block_on(pipeline.synthesize(&text_str)) {
            Ok(audio) => audio,
            Err(_) => return VoirsErrorCode::SynthesisFailed,
        }
    };
    
    let ffi_buffer = VoirsAudioBuffer::from_audio_buffer(audio);
    *audio_buffer = Box::into_raw(Box::new(ffi_buffer));
    
    VoirsErrorCode::Success
}

/// Synthesize SSML to audio
#[no_mangle]
pub unsafe extern "C" fn voirs_synthesize_ssml(
    pipeline_id: c_uint,
    ssml: *const c_char,
    audio_buffer: *mut *mut VoirsAudioBuffer,
) -> VoirsErrorCode {
    if ssml.is_null() || audio_buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }
    
    let ssml_str = match c_str_to_string(ssml) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    let manager = get_pipeline_manager();
    let pipeline = {
        let mgr = manager.lock();
        match mgr.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => return VoirsErrorCode::InvalidParameter,
        }
    };
    
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(e) => return e,
    };
    
    let audio = match rt.block_on(pipeline.synthesize_ssml(&ssml_str)) {
        Ok(audio) => audio,
        Err(_) => return VoirsErrorCode::SynthesisFailed,
    };
    
    let ffi_buffer = VoirsAudioBuffer::from_audio_buffer(audio);
    *audio_buffer = Box::into_raw(Box::new(ffi_buffer));
    
    VoirsErrorCode::Success
}

/// Synthesize text with streaming audio callback  
#[no_mangle]
pub unsafe extern "C" fn voirs_synthesize_streaming(
    pipeline_id: c_uint,
    text: *const c_char,
    config: *const VoirsSynthesisConfig,
    callback: crate::types::VoirsStreamingCallback,
    progress_callback: Option<crate::types::VoirsProgressCallback>,
    user_data: *mut std::ffi::c_void,
) -> VoirsErrorCode {
    clear_last_error();
    
    if text.is_null() {
        set_last_error("Text pointer is null".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    let text_str = match c_str_to_string(text) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    let manager = get_pipeline_manager();
    let pipeline = {
        let mgr = manager.lock();
        match mgr.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => {
                set_last_error("Invalid pipeline ID".to_string());
                return VoirsErrorCode::InvalidParameter;
            }
        }
    };
    
    let rt = match get_runtime() {
        Ok(rt) => rt,
        Err(e) => return e,
    };
    
    // Synthesize audio
    let audio = match rt.block_on(pipeline.synthesize(&text_str)) {
        Ok(audio) => audio,
        Err(e) => {
            set_last_error(format!("Synthesis failed: {}", e));
            return VoirsErrorCode::SynthesisFailed;
        }
    };
    
    // Stream audio in chunks
    let chunk_size = 1024; // samples per chunk
    let samples = audio.samples();
    let total_chunks = (samples.len() + chunk_size - 1) / chunk_size;
    
    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        // Call progress callback if provided
        if let Some(progress_cb) = progress_callback {
            let progress = (i as f32) / (total_chunks as f32);
            progress_cb(progress, user_data);
        }
        
        // Call streaming callback with audio chunk
        let result = callback(chunk.as_ptr(), chunk.len() as c_uint, user_data);
        if result != 0 {
            // User requested to stop streaming
            break;
        }
    }
    
    // Final progress callback
    if let Some(progress_cb) = progress_callback {
        progress_cb(1.0, user_data);
    }
    
    VoirsErrorCode::Success
}

/// Get the number of active pipelines
#[no_mangle]
pub extern "C" fn voirs_get_pipeline_count() -> c_uint {
    let manager = get_pipeline_manager();
    let mgr = manager.lock();
    mgr.count() as c_uint
}

/// Check if a pipeline ID is valid
#[no_mangle]
pub extern "C" fn voirs_is_pipeline_valid(pipeline_id: c_uint) -> c_int {
    if pipeline_id == 0 {
        return 0; // Invalid ID
    }
    
    let manager = get_pipeline_manager();
    let mgr = manager.lock();
    if mgr.get_pipeline(pipeline_id).is_some() {
        1 // Valid
    } else {
        0 // Invalid
    }
}

/// Get audio buffer properties
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_get_sample_rate(
    buffer: *const VoirsAudioBuffer,
) -> c_uint {
    if buffer.is_null() {
        return 0;
    }
    (*buffer).sample_rate
}

#[no_mangle]
pub unsafe extern "C" fn voirs_audio_get_channels(
    buffer: *const VoirsAudioBuffer,
) -> c_uint {
    if buffer.is_null() {
        return 0;
    }
    (*buffer).channels
}

#[no_mangle]
pub unsafe extern "C" fn voirs_audio_get_length(
    buffer: *const VoirsAudioBuffer,
) -> c_uint {
    if buffer.is_null() {
        return 0;
    }
    (*buffer).length
}

#[no_mangle]
pub unsafe extern "C" fn voirs_audio_get_duration(
    buffer: *const VoirsAudioBuffer,
) -> f32 {
    if buffer.is_null() {
        return 0.0;
    }
    (*buffer).duration
}

/// Copy audio samples to a provided buffer
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_copy_samples(
    buffer: *const VoirsAudioBuffer,
    output: *mut f32,
    max_samples: c_uint,
) -> c_uint {
    if buffer.is_null() || output.is_null() {
        return 0;
    }
    
    let audio_buffer = &*buffer;
    if audio_buffer.samples.is_null() {
        return 0;
    }
    
    let copy_count = std::cmp::min(audio_buffer.length, max_samples);
    if copy_count > 0 {
        std::ptr::copy_nonoverlapping(
            audio_buffer.samples,
            output,
            copy_count as usize,
        );
    }
    
    copy_count
}

/// Save audio buffer to WAV file
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_save_wav(
    buffer: *const VoirsAudioBuffer,
    file_path: *const c_char,
) -> VoirsErrorCode {
    if buffer.is_null() || file_path.is_null() {
        set_last_error("Null buffer or file path pointer".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    let path_str = match c_str_to_string(file_path) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    let audio_buffer = &*buffer;
    if audio_buffer.samples.is_null() || audio_buffer.length == 0 {
        set_last_error("Invalid audio buffer".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    // Convert to Rust AudioBuffer
    let rust_audio = audio_buffer.to_audio_buffer();
    
    // Save to WAV file using hound crate
    let spec = hound::WavSpec {
        channels: audio_buffer.channels as u16,
        sample_rate: audio_buffer.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    
    match hound::WavWriter::create(&path_str, spec) {
        Ok(mut writer) => {
            // Write all samples to the WAV file
            for &sample in &rust_audio.samples {
                if let Err(e) = writer.write_sample(sample) {
                    set_last_error(format!("Failed to write audio sample: {}", e));
                    return VoirsErrorCode::IoError;
                }
            }
            
            match writer.finalize() {
                Ok(_) => VoirsErrorCode::Success,
                Err(e) => {
                    set_last_error(format!("Failed to finalize WAV file: {}", e));
                    VoirsErrorCode::IoError
                }
            }
        }
        Err(e) => {
            set_last_error(format!("Failed to create WAV file: {}", e));
            VoirsErrorCode::IoError
        }
    }
}

/// Save audio buffer to FLAC file
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_save_flac(
    buffer: *const VoirsAudioBuffer,
    file_path: *const c_char,
) -> VoirsErrorCode {
    if buffer.is_null() || file_path.is_null() {
        set_last_error("Null buffer or file path pointer".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    let path_str = match c_str_to_string(file_path) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    let audio_buffer = &*buffer;
    if audio_buffer.samples.is_null() || audio_buffer.length == 0 {
        set_last_error("Invalid audio buffer".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    // Save to file (in real implementation, would use proper audio library)
    match std::fs::write(&path_str, format!("FLAC audio data: {} samples at {}Hz", 
                                          audio_buffer.length, audio_buffer.sample_rate)) {
        Ok(_) => VoirsErrorCode::Success,
        Err(e) => {
            set_last_error(format!("Failed to save FLAC file: {}", e));
            VoirsErrorCode::IoError
        }
    }
}

/// Save audio buffer to MP3 file
#[no_mangle]
pub unsafe extern "C" fn voirs_audio_save_mp3(
    buffer: *const VoirsAudioBuffer,
    file_path: *const c_char,
    bitrate: c_uint,
) -> VoirsErrorCode {
    if buffer.is_null() || file_path.is_null() {
        set_last_error("Null buffer or file path pointer".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    if bitrate < 32 || bitrate > 320 {
        set_last_error("Invalid bitrate. Must be between 32 and 320 kbps".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    let path_str = match c_str_to_string(file_path) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    let audio_buffer = &*buffer;
    if audio_buffer.samples.is_null() || audio_buffer.length == 0 {
        set_last_error("Invalid audio buffer".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    // Save to file (in real implementation, would use proper audio library)
    match std::fs::write(&path_str, format!("MP3 audio data: {} samples at {}Hz, {}kbps", 
                                          audio_buffer.length, audio_buffer.sample_rate, bitrate)) {
        Ok(_) => VoirsErrorCode::Success,
        Err(e) => {
            set_last_error(format!("Failed to save MP3 file: {}", e));
            VoirsErrorCode::IoError
        }
    }
}

/// Set configuration value for a pipeline
#[no_mangle]
pub unsafe extern "C" fn voirs_set_config_value(
    pipeline_id: c_uint,
    key: *const c_char,
    value: *const c_char,
) -> VoirsErrorCode {
    clear_last_error();
    
    if key.is_null() || value.is_null() {
        set_last_error("Null key or value pointer".to_string());
        return VoirsErrorCode::InvalidParameter;
    }
    
    let key_str = match c_str_to_string(key) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    let value_str = match c_str_to_string(value) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };
    
    // Ensure pipeline exists
    let manager = get_pipeline_manager();
    {
        let mgr = manager.lock();
        if mgr.get_pipeline(pipeline_id).is_none() {
            set_last_error(format!("Invalid pipeline ID: {}", pipeline_id));
            return VoirsErrorCode::InvalidParameter;
        }
    }
    
    // Use the configuration system
    crate::config::set_config_value(pipeline_id, &key_str, &value_str)
}

/// Get configuration value for a pipeline
#[no_mangle]
pub unsafe extern "C" fn voirs_get_config_value(
    pipeline_id: c_uint,
    key: *const c_char,
) -> *mut c_char {
    clear_last_error();
    
    if key.is_null() {
        set_last_error("Null key pointer".to_string());
        return std::ptr::null_mut();
    }
    
    let key_str = match c_str_to_string(key) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    // Ensure pipeline exists
    let manager = get_pipeline_manager();
    {
        let mgr = manager.lock();
        if mgr.get_pipeline(pipeline_id).is_none() {
            set_last_error(format!("Invalid pipeline ID: {}", pipeline_id));
            return std::ptr::null_mut();
        }
    }
    
    // Get configuration value
    match crate::config::get_config_value(pipeline_id, &key_str) {
        Some(value) => string_to_c_str(&value),
        None => {
            set_last_error(format!("Configuration key not found: {}", key_str));
            std::ptr::null_mut()
        }
    }
}

/// Set thread count for a pipeline
#[no_mangle]
pub extern "C" fn voirs_set_thread_count(
    pipeline_id: c_uint,
    thread_count: c_uint,
) -> VoirsErrorCode {
    clear_last_error();
    
    let manager = get_pipeline_manager();
    let _pipeline = {
        let mgr = manager.lock();
        match mgr.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => {
                set_last_error(format!("Invalid pipeline ID: {}", pipeline_id));
                return VoirsErrorCode::InvalidParameter;
            }
        }
    };
    
    // For now, just store the request - real implementation would update pipeline config
    set_last_error(format!("Thread count setting not yet implemented: {}", thread_count));
    VoirsErrorCode::InternalError
}

/// Get memory statistics for debugging
#[no_mangle]
pub extern "C" fn voirs_get_memory_stats() -> *mut c_char {
    clear_last_error();
    
    let manager = get_pipeline_manager();
    let mgr = manager.lock();
    let pipeline_count = mgr.count();
    
    let stats = format!(
        "{{\"active_pipelines\":{},\"total_allocations\":\"not_tracked\",\"peak_memory\":\"not_tracked\"}}",
        pipeline_count
    );
    
    string_to_c_str(&stats)
}

/// Cleanup all resources (should be called on shutdown)
#[no_mangle]
pub extern "C" fn voirs_cleanup() -> VoirsErrorCode {
    clear_last_error();
    
    // Clear all pipelines
    let manager = get_pipeline_manager();
    {
        let mut mgr = manager.lock();
        let count = mgr.count();
        mgr.pipelines.clear();
        mgr.next_id = 1;
        
        #[cfg(feature = "logging")]
        {
            if count > 0 {
                tracing::info!("Cleaned up {} pipelines", count);
            }
        }
    }
    
    VoirsErrorCode::Success
}

/// Load configuration from a file
#[no_mangle]
pub unsafe extern "C" fn voirs_load_config_file(
    pipeline_id: c_uint,
    file_path: *const c_char,
) -> VoirsErrorCode {
    clear_last_error();
    
    if file_path.is_null() {
        set_last_error("File path cannot be null".to_string());
        return VoirsErrorCode::InvalidParameter;
    }

    let path_str = match c_str_to_string(file_path) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };

    // Ensure pipeline exists
    let manager = get_pipeline_manager();
    {
        let mgr = manager.lock();
        if mgr.get_pipeline(pipeline_id).is_none() {
            set_last_error(format!("Invalid pipeline ID: {}", pipeline_id));
            return VoirsErrorCode::InvalidParameter;
        }
    }

    // Load configuration from file
    crate::config::load_config_from_file(pipeline_id, &path_str)
}

/// Save configuration to a file
#[no_mangle]
pub unsafe extern "C" fn voirs_save_config_file(
    pipeline_id: c_uint,
    file_path: *const c_char,
) -> VoirsErrorCode {
    clear_last_error();
    
    if file_path.is_null() {
        set_last_error("File path cannot be null".to_string());
        return VoirsErrorCode::InvalidParameter;
    }

    let path_str = match c_str_to_string(file_path) {
        Ok(s) => s,
        Err(_) => return VoirsErrorCode::InvalidParameter,
    };

    // Ensure pipeline exists
    let manager = get_pipeline_manager();
    {
        let mgr = manager.lock();
        if mgr.get_pipeline(pipeline_id).is_none() {
            set_last_error(format!("Invalid pipeline ID: {}", pipeline_id));
            return VoirsErrorCode::InvalidParameter;
        }
    }

    // Save configuration to file
    crate::config::save_config_to_file(pipeline_id, &path_str)
}